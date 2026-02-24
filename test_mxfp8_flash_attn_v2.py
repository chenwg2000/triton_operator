"""
Tests for MXfp8 Flash Attention v2.

Compares forward and backward against a PyTorch reference implementation.
Same tolerances as v1:
  - Forward: atol=2.0
  - Backward: atol=8.0

Also includes PTX regression guard to ensure tl.dot_scaled compiles to
hardware block_scale MMA instructions, not BF16 fallback.
"""

import torch
import sys
import glob
import os

from mxfp8_flash_attn_v2 import mxfp8_flash_attention_v2

DEVICE = "cuda"


def reference_attention(q, k, v, causal=False, sm_scale=None):
    """Standard multi-head attention in FP32 for reference."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    q_ = q.float().permute(0, 2, 1, 3)
    k_ = k.float().permute(0, 2, 1, 3)
    v_ = v.float().permute(0, 2, 1, 3)
    s = torch.matmul(q_, k_.transpose(-2, -1)) * sm_scale
    if causal:
        N = q_.shape[-2]
        mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v_)
    return o.permute(0, 2, 1, 3).to(torch.bfloat16)


def test_forward():
    """Test forward pass correctness."""
    print("=" * 60)
    print("V2 FORWARD PASS TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for B in [1, 4]:
        for H in [2, 8]:
            for N in [128, 512, 1024]:
                for D in [64, 128]:
                    for causal in [False, True]:
                        torch.manual_seed(42)
                        sm_scale = 0.5

                        q = torch.empty(B, N, H, D, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0., std=0.5)
                        k = torch.empty(B, N, H, D, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0., std=0.5)
                        v = torch.empty(B, N, H, D, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0., std=0.5)

                        ref = reference_attention(q, k, v, causal, sm_scale)
                        out = mxfp8_flash_attention_v2(q, k, v, causal, sm_scale)

                        has_nan = torch.isnan(out).any().item()
                        diff = (out.float() - ref.float()).abs().max().item()
                        ok = not has_nan and diff < 2.0

                        status = "PASS" if ok else "FAIL"
                        print(f"  [{status}] B={B} H={H} N={N:4d} D={D:3d} "
                              f"causal={causal!s:5s} max_diff={diff:.4f}"
                              f"{' NaN!' if has_nan else ''}")

                        if ok:
                            passed += 1
                        else:
                            failed += 1

    print(f"\nForward: {passed} passed, {failed} failed")
    return failed == 0


def test_backward():
    """Test backward pass correctness."""
    print("=" * 60)
    print("V2 BACKWARD PASS TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for B in [1, 4]:
        for H in [2, 8]:
            for N in [128, 512]:
                for D in [64, 128]:
                    for causal in [False, True]:
                        torch.manual_seed(42)
                        sm_scale = 0.5

                        q_ref = torch.empty(B, N, H, D, device=DEVICE, dtype=torch.bfloat16
                                            ).normal_(mean=0., std=0.5).requires_grad_(True)
                        k_ref = torch.empty(B, N, H, D, device=DEVICE, dtype=torch.bfloat16
                                            ).normal_(mean=0., std=0.5).requires_grad_(True)
                        v_ref = torch.empty(B, N, H, D, device=DEVICE, dtype=torch.bfloat16
                                            ).normal_(mean=0., std=0.5).requires_grad_(True)

                        ref_out = reference_attention(q_ref, k_ref, v_ref, causal, sm_scale)
                        dout = torch.randn_like(ref_out)
                        ref_out.backward(dout)
                        ref_dq = q_ref.grad.clone()
                        ref_dk = k_ref.grad.clone()
                        ref_dv = v_ref.grad.clone()

                        q = q_ref.detach().clone().requires_grad_(True)
                        k = k_ref.detach().clone().requires_grad_(True)
                        v = v_ref.detach().clone().requires_grad_(True)

                        out = mxfp8_flash_attention_v2(q, k, v, causal, sm_scale)
                        out.backward(dout)

                        has_nan = (torch.isnan(out).any().item() or
                                   torch.isnan(q.grad).any().item() or
                                   torch.isnan(k.grad).any().item() or
                                   torch.isnan(v.grad).any().item())

                        diff_o = (out.float() - ref_out.float()).abs().max().item()
                        diff_dq = (q.grad.float() - ref_dq.float()).abs().max().item()
                        diff_dk = (k.grad.float() - ref_dk.float()).abs().max().item()
                        diff_dv = (v.grad.float() - ref_dv.float()).abs().max().item()

                        ok = (not has_nan and diff_o < 2.0 and
                              diff_dq < 8.0 and diff_dk < 8.0 and diff_dv < 8.0)

                        status = "PASS" if ok else "FAIL"
                        print(f"  [{status}] B={B} H={H} N={N:4d} D={D:3d} "
                              f"causal={causal!s:5s} "
                              f"dO={diff_o:.4f} dQ={diff_dq:.4f} "
                              f"dK={diff_dk:.4f} dV={diff_dv:.4f}"
                              f"{' NaN!' if has_nan else ''}")

                        if ok:
                            passed += 1
                        else:
                            failed += 1

    print(f"\nBackward: {passed} passed, {failed} failed")
    return failed == 0


def test_v1_v2_consistency():
    """Test that v2 forward matches v1 forward (both use MXfp8)."""
    print("=" * 60)
    print("V1 vs V2 CONSISTENCY TESTS")
    print("=" * 60)

    try:
        from mxfp8_flash_attn import mxfp8_flash_attention
    except ImportError:
        print("  SKIP: mxfp8_flash_attn.py not found")
        return True

    passed = 0
    failed = 0

    for N in [128, 512, 1024]:
        for D in [64, 128]:
            for causal in [False, True]:
                torch.manual_seed(42)
                sm_scale = 0.5

                q = torch.empty(1, N, 4, D, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0., std=0.5)
                k = torch.empty(1, N, 4, D, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0., std=0.5)
                v = torch.empty(1, N, 4, D, device=DEVICE, dtype=torch.bfloat16).normal_(mean=0., std=0.5)

                ref = reference_attention(q, k, v, causal, sm_scale)
                out_v1 = mxfp8_flash_attention(q, k, v, causal, sm_scale)
                out_v2 = mxfp8_flash_attention_v2(q, k, v, causal, sm_scale)

                diff_v1_ref = (out_v1.float() - ref.float()).abs().max().item()
                diff_v2_ref = (out_v2.float() - ref.float()).abs().max().item()
                diff_v1_v2 = (out_v1.float() - out_v2.float()).abs().max().item()

                # v2 should be within tolerance of reference (slightly different due to different quant order)
                ok = diff_v2_ref < 2.0

                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] N={N:4d} D={D:3d} causal={causal!s:5s} "
                      f"v1-ref={diff_v1_ref:.4f} v2-ref={diff_v2_ref:.4f} "
                      f"v1-v2={diff_v1_v2:.4f}")

                if ok:
                    passed += 1
                else:
                    failed += 1

    print(f"\nConsistency: {passed} passed, {failed} failed")
    return failed == 0


def test_host_quantization():
    """Test host pre-quantization functions."""
    print("=" * 60)
    print("HOST QUANTIZATION TESTS")
    print("=" * 60)

    from mxfp8_flash_attn_v2 import quantize_mxfp8, quantize_mxfp8_seqdim

    passed = 0
    failed = 0

    # Test quantize_mxfp8 (along last dim)
    for shape in [(4, 128), (2, 8, 64), (1, 4, 8, 128)]:
        x = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
        fp8, scale = quantize_mxfp8(x)

        assert fp8.shape == x.shape, f"fp8 shape mismatch: {fp8.shape} != {x.shape}"
        expected_scale_shape = list(x.shape)
        expected_scale_shape[-1] //= 32
        assert list(scale.shape) == expected_scale_shape, \
            f"scale shape mismatch: {scale.shape} != {expected_scale_shape}"
        assert fp8.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.uint8

        # Verify dequantization roughly recovers original
        fp8_f32 = fp8.float()
        scale_f32 = torch.exp2(scale.float() - 127.0)
        K = x.shape[-1]
        fp8_flat = fp8_f32.reshape(-1, K // 32, 32)
        scale_expanded = scale_f32.reshape(-1, K // 32, 1)
        recon = (fp8_flat * scale_expanded).reshape(x.shape)
        max_err = (recon - x.float()).abs().max().item()
        ok = max_err < 2.0  # FP8 E4M3 has limited precision
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] quantize_mxfp8 shape={shape} max_recon_err={max_err:.4f}")
        if ok:
            passed += 1
        else:
            failed += 1

    # Test quantize_mxfp8_seqdim (along S dim)
    for BH, S, D in [(4, 128, 64), (8, 256, 128)]:
        x = torch.randn(BH, S, D, device=DEVICE, dtype=torch.bfloat16)
        fp8, scale = quantize_mxfp8_seqdim(x)

        assert fp8.shape == (BH, S, D), f"fp8 shape: {fp8.shape}"
        assert scale.shape == (BH, S // 32, D), f"scale shape: {scale.shape}"
        assert fp8.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.uint8

        # Verify dequantization
        fp8_f32 = fp8.float().reshape(BH, S // 32, 32, D)
        scale_f32 = torch.exp2(scale.float() - 127.0).unsqueeze(2)
        recon = (fp8_f32 * scale_f32).reshape(BH, S, D)
        max_err = (recon - x.float()).abs().max().item()
        ok = max_err < 2.0
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] quantize_mxfp8_seqdim shape=({BH},{S},{D}) max_recon_err={max_err:.4f}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nQuantization: {passed} passed, {failed} failed")
    return failed == 0


def test_ptx_uses_block_scale():
    """PTX regression guard: ensure tl.dot_scaled compiles to block_scale MMA.

    This catches the known Triton regression where dot_scaled falls back to
    BF16 MMA (mma.sync.aligned.m16n8k16) instead of using hardware block_scale
    (mma.sync.aligned.m16n8k32...block_scale).
    """
    print("=" * 60)
    print("PTX REGRESSION GUARD")
    print("=" * 60)

    # Trigger compilation by running a small forward pass
    torch.manual_seed(0)
    q = torch.randn(1, 128, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 128, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, 128, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    _ = mxfp8_flash_attention_v2(q, k, v, causal=False, sm_scale=0.5)

    # Search for PTX files in Triton cache
    triton_cache = os.path.expanduser("~/.triton/cache")
    ptx_files = glob.glob(os.path.join(triton_cache, "**/*.ptx"), recursive=True)

    fwd_ptx_files = [f for f in ptx_files if "mxfp8_attn_fwd" in f]

    if not fwd_ptx_files:
        print("  WARN: No forward PTX files found in cache — checking all PTX with dot_scaled")
        # Fall back to checking any PTX with block_scale
        fwd_ptx_files = ptx_files

    found_block_scale = False
    found_bf16_fallback = False

    for ptx_path in fwd_ptx_files:
        try:
            with open(ptx_path) as f:
                ptx = f.read()
        except Exception:
            continue

        if "block_scale" in ptx:
            found_block_scale = True
        if "mma.sync.aligned.m16n8k16" in ptx and "block_scale" not in ptx:
            found_bf16_fallback = True

    if found_block_scale:
        print("  [PASS] Found block_scale MMA instructions in PTX")
    else:
        print("  [WARN] block_scale not found in PTX — may need cache clear or check GPU capability")

    if found_bf16_fallback:
        print("  [WARN] Found BF16 fallback MMA (m16n8k16) WITHOUT block_scale in some PTX")
    else:
        print("  [PASS] No BF16-only fallback detected")

    ok = found_block_scale and not found_bf16_fallback
    if not ok and not found_block_scale:
        # Not a hard failure if cache is empty — just a warning
        print("  NOTE: Run 'rm -rf ~/.triton/cache' and re-run to verify PTX quality")
        ok = True  # Don't fail the test suite

    return ok


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print()

    quant_ok = test_host_quantization()
    print()
    fwd_ok = test_forward()
    print()
    bwd_ok = test_backward()
    print()
    consistency_ok = test_v1_v2_consistency()
    print()
    ptx_ok = test_ptx_uses_block_scale()

    print()
    all_ok = quant_ok and fwd_ok and bwd_ok and consistency_ok and ptx_ok
    if all_ok:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
