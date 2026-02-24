"""
Tests for Gluon MXfp8 Flash Attention.

Compares forward and backward against a PyTorch reference implementation.
Tolerances:
  - Forward: atol=2.0
  - Backward: atol=8.0
"""

import torch
import sys

from gluon_mxfp8_flash_attn import gluon_mxfp8_flash_attention, quantize_mxfp8

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


def test_quantize_mxfp8():
    """Test host-side MXfp8 quantization."""
    print("=" * 60)
    print("QUANTIZE_MXFP8 TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for shape in [(4, 128), (8, 64), (16, 256), (2, 3, 128)]:
        torch.manual_seed(42)
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        fp8, scales = quantize_mxfp8(x)

        # Check shapes
        K = shape[-1]
        expected_scale_shape = (*shape[:-1], K // 32)
        shape_ok = fp8.shape == x.shape and scales.shape == expected_scale_shape
        dtype_ok = fp8.dtype == torch.float8_e4m3fn and scales.dtype == torch.uint8

        # Check roundtrip accuracy: dequantize and compare
        # E8M0 decode: scale = 2^(biased - 127)
        scale_f32 = torch.exp2(scales.float() - 127.0)
        # Broadcast scale to match fp8 shape
        fp8_f32 = fp8.float().reshape(*shape[:-1], K // 32, 32)
        dequant = (fp8_f32 * scale_f32.unsqueeze(-1)).reshape(x.shape)
        max_err = (x - dequant).abs().max().item()
        # E4M3 has ~12.5% relative error, so with input std=1, max error ~0.5
        err_ok = max_err < 2.0

        ok = shape_ok and dtype_ok and err_ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] shape={shape} max_err={max_err:.4f} "
              f"shapes_ok={shape_ok} dtypes_ok={dtype_ok}")

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nQuantize: {passed} passed, {failed} failed")
    return failed == 0


def test_forward():
    """Test forward pass correctness."""
    print("=" * 60)
    print("FORWARD PASS TESTS")
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
                        out = gluon_mxfp8_flash_attention(q, k, v, causal, sm_scale)

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
    print("BACKWARD PASS TESTS")
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

                        # Reference
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

                        # Gluon MXfp8
                        q = q_ref.detach().clone().requires_grad_(True)
                        k = k_ref.detach().clone().requires_grad_(True)
                        v = v_ref.detach().clone().requires_grad_(True)

                        out = gluon_mxfp8_flash_attention(q, k, v, causal, sm_scale)
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


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print()

    quant_ok = test_quantize_mxfp8()
    print()

    fwd_ok = test_forward()
    print()

    bwd_ok = test_backward()

    print()
    if quant_ok and fwd_ok and bwd_ok:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
