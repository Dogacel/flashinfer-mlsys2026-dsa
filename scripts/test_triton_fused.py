"""
Standalone test for the fused Triton kernel on Modal B200.
Runs the kernel directly (no benchmark framework) so errors are visible.
"""

import modal

app = modal.App("triton-fused-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_fused_kernel():
    import torch
    import traceback
    import math

    print("=== Fused Triton Kernel Debug Test ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # ---- Constants (matching the DSA problem) ----
    num_tokens = 2
    H = 16
    D_ckv = 512
    D_kpe = 64
    topk = 2048
    num_pages = 256
    page_size = 64

    # ---- Create test inputs ----
    q_nope = torch.randn(num_tokens, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(num_tokens, H, D_kpe, dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn(num_pages, page_size, D_ckv, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn(num_pages, page_size, D_kpe, dtype=torch.bfloat16, device="cuda")
    sparse_indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    sm_scale = 1.0 / math.sqrt(D_ckv + D_kpe)
    output = torch.zeros(num_tokens, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(num_tokens, H, dtype=torch.float32, device="cuda")

    print(f"\nShapes:")
    print(f"  q_nope:         {q_nope.shape}  {q_nope.dtype}")
    print(f"  q_pe:           {q_pe.shape}  {q_pe.dtype}")
    print(f"  ckv_cache:      {ckv_cache.shape}  {ckv_cache.dtype}")
    print(f"  kpe_cache:      {kpe_cache.shape}  {kpe_cache.dtype}")
    print(f"  sparse_indices: {sparse_indices.shape}  {sparse_indices.dtype}")
    print(f"  output:         {output.shape}  {output.dtype}")
    print(f"  lse:            {lse.shape}  {lse.dtype}")

    # ---- Step 1: Test reference (naive PyTorch) ----
    print("\n--- Step 1: Reference (naive PyTorch) ---")
    try:
        Kc_all = ckv_cache.reshape(-1, D_ckv)
        Kp_all = kpe_cache.reshape(-1, D_kpe)
        safe_idx = sparse_indices.clamp(min=0).long()
        flat_idx = safe_idx.reshape(-1)
        Kc = Kc_all[flat_idx].reshape(num_tokens, topk, D_ckv).to(torch.float32)
        Kp = Kp_all[flat_idx].reshape(num_tokens, topk, D_kpe).to(torch.float32)
        qn = q_nope.to(torch.float32)
        qp = q_pe.to(torch.float32)
        logits = torch.bmm(qn, Kc.transpose(1, 2)) + torch.bmm(qp, Kp.transpose(1, 2))
        logits_scaled = logits * sm_scale
        valid_mask = sparse_indices != -1
        logits_scaled.masked_fill_(~valid_mask.unsqueeze(1), float('-inf'))
        ref_lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits_scaled, dim=-1)
        ref_output = torch.bmm(attn, Kc).to(torch.bfloat16)
        print(f"  ref_output: {ref_output.shape}, max={ref_output.abs().max().item():.4f}")
        print(f"  ref_lse:    {ref_lse.shape}, [0,0]={ref_lse[0,0].item():.4f}")
        print("  Reference OK ✓")
    except Exception:
        traceback.print_exc()
        return "REFERENCE_FAILED"

    # ---- Step 2: Import Triton and test basic kernel ----
    print("\n--- Step 2: Triton sanity check ---")
    try:
        import triton
        import triton.language as tl
        print(f"  Triton version: {triton.__version__}")

        @triton.jit
        def _simple_add(X, Y, N: tl.constexpr):
            offs = tl.arange(0, N)
            x = tl.load(X + offs)
            tl.store(Y + offs, x + 1.0)

        x = torch.ones(64, device="cuda", dtype=torch.float32)
        y = torch.zeros(64, device="cuda", dtype=torch.float32)
        _simple_add[(1,)](x, y, N=64)
        torch.cuda.synchronize()
        assert y[0].item() == 2.0, f"Expected 2.0, got {y[0].item()}"
        print("  Triton basic kernel OK ✓")
    except Exception:
        traceback.print_exc()
        return "TRITON_BASIC_FAILED"

    # ---- Step 3: Test the fused kernel ----
    print("\n--- Step 3: Fused Triton kernel ---")
    try:
        # Inline the kernel source
        # (We can't import from solution dir on Modal, so we inline it)
        KERNEL_SOURCE = r'''
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_sparse_attn_kernel(
    Q_nope, Q_pe, CKV, KPE, Indices, Out, Acc, LSE_out, sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_kc_n, stride_kc_d,
    stride_kp_n, stride_kp_d,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_h, stride_out_d,
    stride_acc_t, stride_acc_h, stride_acc_d,
    stride_lse_t, stride_lse_h,
    NH: tl.constexpr,
    TOPK_CST: tl.constexpr,
    D_CKV_CST: tl.constexpr,
    D_KPE_CST: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    token_id = pid // NH
    head_id  = pid %  NH

    qn_base  = Q_nope  + token_id * stride_qn_t + head_id * stride_qn_h
    qp_base  = Q_pe    + token_id * stride_qp_t + head_id * stride_qp_h
    idx_base = Indices  + token_id * stride_idx_t
    acc_base = Acc      + token_id * stride_acc_t + head_id * stride_acc_h

    for d_start in tl.static_range(0, D_CKV_CST, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        tl.store(acc_base + d_offs * stride_acc_d,
                 tl.zeros([BLOCK_D], dtype=tl.float32))

    m_i = float('-inf')
    l_i = 0.0

    for k_start in range(0, TOPK_CST, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        idx = tl.load(idx_base + k_offs * stride_idx_k)
        valid    = (idx != -1)
        safe_idx = tl.where(valid, idx, tl.zeros_like(idx))

        scores = tl.zeros([BLOCK_K], dtype=tl.float32)

        for d_start in tl.static_range(0, D_CKV_CST, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            q_chunk = tl.load(qn_base + d_offs * stride_qn_d).to(tl.float32)
            k_ptrs  = (CKV
                       + safe_idx[:, None].to(tl.int64) * stride_kc_n
                       + d_offs[None, :] * stride_kc_d)
            k_chunk = tl.load(k_ptrs).to(tl.float32)
            scores += tl.sum(k_chunk * q_chunk[None, :], axis=1)

        for d_start in tl.static_range(0, D_KPE_CST, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_KPE_CST
            q_chunk = tl.load(qp_base + d_offs * stride_qp_d,
                              mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs  = (KPE
                       + safe_idx[:, None].to(tl.int64) * stride_kp_n
                       + d_offs[None, :] * stride_kp_d)
            k_chunk = tl.load(k_ptrs, mask=d_mask[None, :],
                              other=0.0).to(tl.float32)
            scores += tl.sum(k_chunk * q_chunk[None, :], axis=1)

        scores = scores * sm_scale
        scores = tl.where(valid, scores, float('-inf'))

        m_ij  = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new)
        l_i   = l_i * alpha + tl.sum(p, axis=0)
        m_i   = m_new

        for d_start in tl.static_range(0, D_CKV_CST, BLOCK_D):
            d_offs   = d_start + tl.arange(0, BLOCK_D)
            acc_ptrs = acc_base + d_offs * stride_acc_d
            old_acc = tl.load(acc_ptrs)
            old_acc = old_acc * alpha
            k_ptrs  = (CKV
                       + safe_idx[:, None].to(tl.int64) * stride_kc_n
                       + d_offs[None, :] * stride_kc_d)
            k_chunk = tl.load(k_ptrs).to(tl.float32)
            delta = tl.sum(p[:, None] * k_chunk, axis=0)
            tl.store(acc_ptrs, old_acc + delta)

    inv_l = 1.0 / l_i
    out_base = Out + token_id * stride_out_t + head_id * stride_out_h
    for d_start in tl.static_range(0, D_CKV_CST, BLOCK_D):
        d_offs   = d_start + tl.arange(0, BLOCK_D)
        val      = tl.load(acc_base + d_offs * stride_acc_d) * inv_l
        tl.store(out_base + d_offs * stride_out_d, val.to(tl.bfloat16))

    lse_val = (m_i + tl.log(l_i)) * 1.4426950408889634
    tl.store(LSE_out + token_id * stride_lse_t + head_id * stride_lse_h,
             lse_val)


def run_kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
               sm_scale, output, lse):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    Kc_flat = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_flat = kpe_cache.reshape(-1, head_dim_kpe)

    acc = torch.zeros(num_tokens, num_qo_heads, head_dim_ckv,
                      dtype=torch.float32, device=q_nope.device)

    BLOCK_K = 32
    BLOCK_D = 64
    grid = (num_tokens * num_qo_heads,)

    _fused_sparse_attn_kernel[grid](
        q_nope, q_pe,
        Kc_flat, Kp_flat,
        sparse_indices,
        output, acc, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        Kc_flat.stride(0), Kc_flat.stride(1),
        Kp_flat.stride(0), Kp_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2),
        lse.stride(0), lse.stride(1),
        NH=num_qo_heads,
        TOPK_CST=topk,
        D_CKV_CST=head_dim_ckv,
        D_KPE_CST=head_dim_kpe,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
'''

        # Write kernel to a temp file and import it
        import tempfile, importlib.util, sys
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(KERNEL_SOURCE)
            f.flush()
            spec = importlib.util.spec_from_file_location("fused_kernel", f.name)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        print("  Module loaded OK")
        print("  Calling kernel...")

        mod.run_kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
                       sm_scale, output, lse)
        torch.cuda.synchronize()

        print(f"  output: max={output.abs().max().item():.4f}")
        print(f"  lse:    [0,0]={lse[0,0].item():.4f}")
        print("  Kernel executed OK ✓")

    except Exception:
        traceback.print_exc()
        print("\n  *** KERNEL FAILED ***")
        return "KERNEL_FAILED"

    # ---- Step 4: Compare correctness ----
    print("\n--- Step 4: Correctness check ---")
    try:
        out_diff = (output.float() - ref_output.float()).abs().max().item()
        lse_diff = (lse - ref_lse).abs().max().item()
        print(f"  Max output diff: {out_diff:.6e}")
        print(f"  Max LSE diff:    {lse_diff:.6e}")
        if out_diff < 0.1 and lse_diff < 0.5:
            print("  Correctness OK ✓")
        else:
            print("  ⚠ Large difference detected")
    except Exception:
        traceback.print_exc()

    return "OK"


@app.local_entrypoint()
def main():
    result = test_fused_kernel.remote()
    print(f"\nResult: {result}")
