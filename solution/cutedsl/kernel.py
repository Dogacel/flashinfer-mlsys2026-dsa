"""
DSA Sparse Attention Kernel — CuTeDSL fused softmax + LSE.

Hybrid approach:
- PyTorch/cuBLAS: gather, Q*K^T matmuls, attn*V matmul
- CuTeDSL kernel: fused scale + mask + softmax + LSE
  Uses fixed-size padded buffers to avoid JIT recompilation.

Constants: H=16, D_ckv=512, D_kpe=64, topk=2048, page_size=64
"""

import math
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_HEADS: int = 16
D_CKV: int = 512
D_KPE: int = 64
TOPK: int = 2048
THREADS: int = 256
WARPS: int = THREADS // 32
ELEMS_PER_THREAD: int = TOPK // THREADS  # 8
MAX_TOKENS: int = 64   # max num_tokens across workloads
MAX_ROWS: int = MAX_TOKENS * N_HEADS  # 1024


# ---------------------------------------------------------------------------
# CuTeDSL: fused scale + mask + softmax + LSE kernel
# ---------------------------------------------------------------------------
@cute.kernel
def _softmax_lse_kernel(
    logits: cute.Tensor,   # [MAX_ROWS, TOPK] f32
    vmask: cute.Tensor,    # [MAX_TOKENS, TOPK] i32
    attn: cute.Tensor,     # [MAX_ROWS, TOPK] f32
    lse_out: cute.Tensor,  # [MAX_ROWS] f32
    sm_scale: Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    token = bidx // N_HEADS
    qrow = bidx

    warp_id = tidx // 32
    lane_id = tidx % 32

    smem = utils.SmemAllocator()
    s_reduce = smem.allocate_tensor(Float32, cute.make_layout(WARPS))

    # --- Scale + mask + local max ---
    local_max = -Float32.inf
    for e in cutlass.range_constexpr(ELEMS_PER_THREAD):
        k = tidx * ELEMS_PER_THREAD + e
        val = logits[(qrow, k)] * sm_scale
        if vmask[(token, k)] == Int32(-1):
            val = -Float32.inf
        logits[(qrow, k)] = val
        local_max = cute.arch.fmax(local_max, val)

    # --- Block-wide max ---
    warp_max = cute.arch.warp_reduction_max(local_max)
    if lane_id == 0:
        s_reduce[warp_id] = warp_max
    cute.arch.barrier()
    if tidx == 0:
        gmax = s_reduce[0]
        for w in cutlass.range_constexpr(WARPS - 1):
            gmax = cute.arch.fmax(gmax, s_reduce[w + 1])
        s_reduce[0] = gmax
    cute.arch.barrier()
    gmax = s_reduce[0]

    # --- exp + local sum ---
    local_sum = Float32(0.0)
    for e in cutlass.range_constexpr(ELEMS_PER_THREAD):
        k = tidx * ELEMS_PER_THREAD + e
        v = cute.math.exp(logits[(qrow, k)] - gmax)
        attn[(qrow, k)] = v
        local_sum = local_sum + v

    # --- Block-wide sum ---
    warp_sum = cute.arch.warp_reduction_sum(local_sum)
    if lane_id == 0:
        s_reduce[warp_id] = warp_sum
    cute.arch.barrier()
    if tidx == 0:
        gsum = s_reduce[0]
        for w in cutlass.range_constexpr(WARPS - 1):
            gsum = gsum + s_reduce[w + 1]
        s_reduce[0] = gsum
        lse_out[qrow] = (gmax + cute.math.log(gsum)) * Float32(1.4426950408889634)
    cute.arch.barrier()
    gsum = s_reduce[0]

    # --- Normalize ---
    inv_sum = cute.arch.rcp_approx(gsum)
    for e in cutlass.range_constexpr(ELEMS_PER_THREAD):
        k = tidx * ELEMS_PER_THREAD + e
        attn[(qrow, k)] = attn[(qrow, k)] * inv_sum


@cute.jit
def _launch_softmax(logits, vmask, attn, lse_out, sm_scale, num_ctas):
    _softmax_lse_kernel(logits, vmask, attn, lse_out, sm_scale).launch(
        grid=[num_ctas, 1, 1],
        block=[THREADS, 1, 1],
    )


# ---------------------------------------------------------------------------
# Fixed-size buffers + one-time compilation
# ---------------------------------------------------------------------------
_buf_logits = None
_buf_vmask = None
_buf_attn = None
_buf_lse = None


def _ensure_compiled():
    global _buf_logits, _buf_vmask, _buf_attn, _buf_lse
    if _buf_logits is not None:
        return
    _buf_logits = torch.zeros(MAX_ROWS, TOPK, dtype=torch.float32, device="cuda")
    _buf_vmask = torch.full((MAX_TOKENS, TOPK), -1, dtype=torch.int32, device="cuda")
    _buf_attn = torch.zeros(MAX_ROWS, TOPK, dtype=torch.float32, device="cuda")
    _buf_lse = torch.zeros(MAX_ROWS, dtype=torch.float32, device="cuda")

    # Trigger one-time JIT compilation
    c_l = from_dlpack(_buf_logits)
    c_v = from_dlpack(_buf_vmask)
    c_a = from_dlpack(_buf_attn)
    c_lse = from_dlpack(_buf_lse)
    _launch_softmax(c_l, c_v, c_a, c_lse, Float32(1.0), MAX_ROWS)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    _ensure_compiled()

    num_tokens = q_nope.shape[0]
    num_rows = num_tokens * N_HEADS

    # --- Gather KV (PyTorch) ---
    Kc_all = ckv_cache.reshape(-1, D_CKV)
    Kp_all = kpe_cache.reshape(-1, D_KPE)
    safe_indices = sparse_indices.clamp(min=0).long()
    flat_idx = safe_indices.reshape(-1)
    Kc = Kc_all[flat_idx].reshape(num_tokens, TOPK, D_CKV).to(torch.float32)
    Kp = Kp_all[flat_idx].reshape(num_tokens, TOPK, D_KPE).to(torch.float32)

    qn = q_nope.to(torch.float32)
    qp = q_pe.to(torch.float32)

    # --- Q*K^T logits (PyTorch/cuBLAS) ---
    logits = torch.bmm(qn, Kc.transpose(1, 2)) + torch.bmm(qp, Kp.transpose(1, 2))

    # --- Copy into fixed-size buffers ---
    _buf_logits[:num_rows].copy_(logits.reshape(num_rows, TOPK))
    _buf_vmask[:num_tokens].copy_(sparse_indices)

    # --- CuTeDSL fused softmax + LSE ---
    c_l = from_dlpack(_buf_logits)
    c_v = from_dlpack(_buf_vmask)
    c_a = from_dlpack(_buf_attn)
    c_lse = from_dlpack(_buf_lse)
    _launch_softmax(c_l, c_v, c_a, c_lse, Float32(sm_scale), num_rows)

    # --- Copy LSE out ---
    lse.copy_(_buf_lse[:num_rows].reshape(num_tokens, N_HEADS))

    # --- attn * V output (PyTorch/cuBLAS) ---
    attn_3d = _buf_attn[:num_rows].reshape(num_tokens, N_HEADS, TOPK)
    output.copy_(torch.bmm(attn_3d, Kc).to(torch.bfloat16))
