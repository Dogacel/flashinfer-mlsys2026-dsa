"""Quick test to debug CuTeDSL kernel errors on Modal B200."""

import modal

app = modal.App("cutedsl-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "nvidia-cutlass-dsl")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_kernel():
    import torch
    import traceback

    # Import kernel source inline
    print("=== Testing CuTeDSL kernel ===")

    # Minimal test inputs
    num_tokens = 1
    H = 16
    D_ckv = 512
    D_kpe = 64
    topk = 2048
    num_pages = 256
    page_size = 64

    q_nope = torch.randn(num_tokens, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(num_tokens, H, D_kpe, dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn(num_pages, page_size, D_ckv, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn(num_pages, page_size, D_kpe, dtype=torch.bfloat16, device="cuda")
    sparse_indices = torch.randint(0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda")
    sm_scale = 0.1
    output = torch.zeros(num_tokens, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(num_tokens, H, dtype=torch.float32, device="cuda")

    print(f"Shapes: q_nope={q_nope.shape}, q_pe={q_pe.shape}")
    print(f"  ckv_cache={ckv_cache.shape}, kpe_cache={kpe_cache.shape}")
    print(f"  sparse_indices={sparse_indices.shape}")
    print(f"  output={output.shape}, lse={lse.shape}")

    try:
        # Import and test the kernel
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("kernel_mod", "/root/kernel.py")
        # Actually we need to pack and load... let's just inline

        # Step 1: test _ensure_compiled
        print("\n--- Step 1: Testing imports ---")
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute.typing import Float32, BFloat16, Int32
        print("Imports OK")

        print("\n--- Step 2: Testing mark_compact_shape_dynamic ---")
        t = torch.zeros(16, 512, dtype=torch.float32, device="cuda")
        ct = from_dlpack(t).mark_compact_shape_dynamic(mode=0)
        print(f"Dynamic tensor OK: shape={ct.shape}")

        print("\n--- Step 3: Testing SmemAllocator with allocate_tensor ---")

        @cute.kernel
        def simple_kernel(inp: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            smem = utils.SmemAllocator()
            s_buf = smem.allocate_tensor(Float32, cute.make_layout(256))
            s_buf[tidx] = inp[(bidx, tidx)]
            cute.arch.barrier()
            out[(bidx, tidx)] = s_buf[tidx]

        @cute.jit
        def launch_simple(inp, out, n):
            simple_kernel(inp, out).launch(grid=[n, 1, 1], block=[256, 1, 1])

        inp = torch.randn(4, 256, dtype=torch.float32, device="cuda")
        out = torch.zeros(4, 256, dtype=torch.float32, device="cuda")
        c_inp = from_dlpack(inp).mark_compact_shape_dynamic(mode=0)
        c_out = from_dlpack(out).mark_compact_shape_dynamic(mode=0)
        launch_simple(c_inp, c_out, 4)
        torch.cuda.synchronize()
        print(f"Simple kernel OK. Max diff: {(inp - out).abs().max().item()}")

        print("\n--- Step 4: Testing warp reduction ---")

        @cute.kernel
        def reduce_kernel(out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            val = Float32(1.0)
            result = cute.arch.warp_reduction_sum(val)
            if tidx == 0:
                out[0] = result

        @cute.jit
        def launch_reduce(out):
            reduce_kernel(out).launch(grid=[1, 1, 1], block=[32, 1, 1])

        out_r = torch.zeros(1, dtype=torch.float32, device="cuda")
        c_out_r = from_dlpack(out_r)
        launch_reduce(c_out_r)
        torch.cuda.synchronize()
        print(f"Warp reduction: expected 32.0, got {out_r[0].item()}")

        print("\n--- Step 5: Testing tensor indexing with 3D tensor ---")

        @cute.kernel
        def index3d_kernel(t3d: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            # Access t3d[bidx, tidx, 0]
            val = t3d[(bidx, tidx, 0)]
            out[(bidx, tidx)] = val

        @cute.jit
        def launch_index3d(t3d, out, n):
            index3d_kernel(t3d, out).launch(grid=[n, 1, 1], block=[8, 1, 1])

        t3d = torch.randn(2, 8, 4, dtype=torch.float32, device="cuda")
        out3d = torch.zeros(2, 8, dtype=torch.float32, device="cuda")
        c_t3d = from_dlpack(t3d).mark_compact_shape_dynamic(mode=0)
        c_out3d = from_dlpack(out3d).mark_compact_shape_dynamic(mode=0)
        launch_index3d(c_t3d, c_out3d, 2)
        torch.cuda.synchronize()
        expected = t3d[:, :, 0]
        diff = (expected - out3d).abs().max().item()
        print(f"3D indexing OK. Max diff: {diff}")

        print("\n--- Step 6: Testing bf16 to f32 conversion ---")

        @cute.kernel
        def convert_kernel(inp_bf16: cute.Tensor, out_f32: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            val = inp_bf16[tidx].to(Float32)
            out_f32[tidx] = val

        @cute.jit
        def launch_convert(inp_bf16, out_f32):
            convert_kernel(inp_bf16, out_f32).launch(grid=[1, 1, 1], block=[16, 1, 1])

        inp_bf16 = torch.randn(16, dtype=torch.bfloat16, device="cuda")
        out_f32 = torch.zeros(16, dtype=torch.float32, device="cuda")
        c_ibf = from_dlpack(inp_bf16)
        c_of = from_dlpack(out_f32)
        launch_convert(c_ibf, c_of)
        torch.cuda.synchronize()
        diff = (inp_bf16.float() - out_f32).abs().max().item()
        print(f"BF16->F32 OK. Max diff: {diff}")

        print("\n=== All basic tests passed! ===")

    except Exception:
        traceback.print_exc()
        return "FAILED"

    return "OK"


@app.local_entrypoint()
def main():
    result = test_kernel.remote()
    print(f"\nResult: {result}")
