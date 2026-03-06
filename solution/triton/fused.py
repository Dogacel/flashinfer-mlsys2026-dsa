"""
Fused DSA Sparse Attention — Tensor-Core Accelerated Triton Kernel.

One program per TOKEN. Each program handles all 16 heads, using
tl.dot (tensor cores) for both score computation and output accumulation.
K is loaded exactly ONCE per block — reused for both operations.
Online softmax keeps everything in registers (no HBM accumulator).
Zero allocations on the hot path.

Constants: H=16, D_ckv=512, D_kpe=64, topk=2048
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_HEADS = 16
D_CKV = 512
D_KPE = 64
TOPK = 2048


@triton.jit
def _fused_sparse_attn_kernel(
    # Pointers
    Q_nope, Q_pe, CKV, KPE, Indices, Out, LSE_out,
    sm_scale,
    # Q_nope strides
    stride_qn_t, stride_qn_h, stride_qn_d,
    # Q_pe strides
    stride_qp_t, stride_qp_h, stride_qp_d,
    # Flat cache strides
    stride_kc_n, stride_kc_d,
    stride_kp_n, stride_kp_d,
    # Index strides
    stride_idx_t, stride_idx_k,
    # Output strides
    stride_out_t, stride_out_h, stride_out_d,
    # LSE strides
    stride_lse_t, stride_lse_h,
    # Constexpr
    NH: tl.constexpr,
    TOPK_CST: tl.constexpr,
    D_CKV_CST: tl.constexpr,
    D_KPE_CST: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    One program per token.  Processes all NH=16 heads.

    Uses tl.dot for tensor-core acceleration:
      scores[NH, BK] = Q[NH, D] @ K[D, BK]          (score)
      acc[NH, D]    += p[NH, BK] @ K_nope[BK, D]     (output)

    K_nope[BK, D_CKV] is loaded ONCE and reused for both ops.
    """
    token_id = tl.program_id(0)

    head_range = tl.arange(0, NH)       # [NH]
    d_ckv      = tl.arange(0, D_CKV_CST)  # [D_CKV]
    d_kpe      = tl.arange(0, D_KPE_CST)  # [D_KPE]

    # ---- Load Q vectors for ALL heads (stay in registers) ----
    qn_ptrs = (Q_nope + token_id * stride_qn_t
               + head_range[:, None] * stride_qn_h
               + d_ckv[None, :] * stride_qn_d)
    q_nope = tl.load(qn_ptrs)                    # [NH, D_CKV] bf16

    qp_ptrs = (Q_pe + token_id * stride_qp_t
               + head_range[:, None] * stride_qp_h
               + d_kpe[None, :] * stride_qp_d)
    q_pe = tl.load(qp_ptrs)                      # [NH, D_KPE] bf16

    # ---- Online softmax state (per head) ----
    m_i = tl.zeros([NH], dtype=tl.float32) - float('inf')   # [NH]
    l_i = tl.zeros([NH], dtype=tl.float32)                  # [NH]
    acc = tl.zeros([NH, D_CKV_CST], dtype=tl.float32)       # [NH, D_CKV]

    idx_base = Indices + token_id * stride_idx_t

    # ---- Main loop over topk ----
    for k_start in range(0, TOPK_CST, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load & sanitise indices  [BK]
        idx    = tl.load(idx_base + k_offs * stride_idx_k)
        valid  = (idx != -1)
        safe_idx = tl.where(valid, idx, tl.zeros_like(idx))

        # ---- Gather K_nope [BK, D_CKV] bf16 (loaded ONCE) ----
        kc_ptrs = (CKV
                   + safe_idx[:, None].to(tl.int64) * stride_kc_n
                   + d_ckv[None, :] * stride_kc_d)
        k_nope = tl.load(kc_ptrs)                 # [BK, D_CKV] bf16

        # ---- Gather K_pe [BK, D_KPE] bf16 ----
        kp_ptrs = (KPE
                   + safe_idx[:, None].to(tl.int64) * stride_kp_n
                   + d_kpe[None, :] * stride_kp_d)
        k_pe = tl.load(kp_ptrs)                   # [BK, D_KPE] bf16

        # ---- Scores via tl.dot (tensor cores) ----
        # [NH, D_CKV] @ [D_CKV, BK] → [NH, BK]
        scores = tl.dot(q_nope, tl.trans(k_nope))
        # [NH, D_KPE] @ [D_KPE, BK] → [NH, BK]
        scores += tl.dot(q_pe, tl.trans(k_pe))
        scores = scores * sm_scale

        # Mask invalid entries (broadcast [BK] → [NH, BK])
        scores = tl.where(valid[None, :], scores, float('-inf'))

        # ---- Online softmax update (per head) ----
        m_ij  = tl.max(scores, axis=1)             # [NH]
        m_new = tl.maximum(m_i, m_ij)              # [NH]
        alpha = tl.exp(m_i - m_new)                # [NH]
        p     = tl.exp(scores - m_new[:, None])    # [NH, BK]

        l_i = l_i * alpha + tl.sum(p, axis=1)     # [NH]

        # Rescale old accumulator
        acc = acc * alpha[:, None]                 # [NH, D_CKV]

        # ---- Accumulate output via tl.dot (tensor cores, same K) ----
        # [NH, BK] @ [BK, D_CKV] → [NH, D_CKV]
        acc += tl.dot(p.to(tl.bfloat16), k_nope)

        m_i = m_new

    # ---- Finalise: normalise and store ----
    acc = acc / l_i[:, None]                       # [NH, D_CKV]

    out_base = Out + token_id * stride_out_t
    out_ptrs = (out_base
                + head_range[:, None] * stride_out_h
                + d_ckv[None, :] * stride_out_d)
    tl.store(out_ptrs, acc.to(tl.bfloat16))

    # LSE in base-2
    lse_vals = (m_i + tl.log(l_i)) * 1.4426950408889634  # [NH]
    lse_ptrs = LSE_out + token_id * stride_lse_t + head_range * stride_lse_h
    tl.store(lse_ptrs, lse_vals)


# ---------------------------------------------------------------------------
# Entry point — zero allocations on the hot path
# ---------------------------------------------------------------------------
def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
           sm_scale, output, lse):
    num_tokens = q_nope.shape[0]

    # reshape is free (view, no copy)
    Kc_flat = ckv_cache.reshape(-1, D_CKV)
    Kp_flat = kpe_cache.reshape(-1, D_KPE)

    BLOCK_K = 32
    grid = (num_tokens,)   # one program per token (handles all heads)

    _fused_sparse_attn_kernel[grid](
        q_nope, q_pe,
        Kc_flat, Kp_flat,
        sparse_indices,
        output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        Kc_flat.stride(0), Kc_flat.stride(1),
        Kp_flat.stride(0), Kp_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        NH=N_HEADS,
        TOPK_CST=TOPK,
        D_CKV_CST=D_CKV,
        D_KPE_CST=D_KPE,
        BLOCK_K=BLOCK_K,
        num_warps=8,
    )
