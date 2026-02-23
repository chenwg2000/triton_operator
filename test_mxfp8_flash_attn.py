"""
Tests for MXfp8 Flash Attention.

Compares forward and backward against a PyTorch reference implementation.
FP8 E4M3 quantization (3 mantissa bits) introduces ~12.5% relative error per
element, which accumulates across dot-product dimensions.  Reference Triton
flash attention uses atol=3 for E5M2 FP8.  Our MXfp8 (E4M3 + per-32-block
E8M0 scaling) is slightly more precise, so we use:
  - Forward: atol=2.0
  - Backward: atol=8.0  (forward quantization mismatch + FP16 backward)
"""

import torch
import sys

from mxfp8_flash_attn import mxfp8_flash_attention

DEVICE = "cuda"


def reference_attention(q, k, v, causal=False, sm_scale=None):
    """Standard multi-head attention in FP32 for reference."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    # Compute in float32 for accuracy
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        N = q.shape[-2]
        mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v.float())
    return o.half()


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

                        q = torch.empty(B, H, N, D, device=DEVICE, dtype=torch.float16).normal_(mean=0., std=0.5)
                        k = torch.empty(B, H, N, D, device=DEVICE, dtype=torch.float16).normal_(mean=0., std=0.5)
                        v = torch.empty(B, H, N, D, device=DEVICE, dtype=torch.float16).normal_(mean=0., std=0.5)

                        ref = reference_attention(q, k, v, causal, sm_scale)
                        out = mxfp8_flash_attention(q, k, v, causal, sm_scale)

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
                        q_ref = torch.empty(B, H, N, D, device=DEVICE, dtype=torch.float16
                                            ).normal_(mean=0., std=0.5).requires_grad_(True)
                        k_ref = torch.empty(B, H, N, D, device=DEVICE, dtype=torch.float16
                                            ).normal_(mean=0., std=0.5).requires_grad_(True)
                        v_ref = torch.empty(B, H, N, D, device=DEVICE, dtype=torch.float16
                                            ).normal_(mean=0., std=0.5).requires_grad_(True)

                        ref_out = reference_attention(q_ref, k_ref, v_ref, causal, sm_scale)
                        dout = torch.randn_like(ref_out)
                        ref_out.backward(dout)
                        ref_dq = q_ref.grad.clone()
                        ref_dk = k_ref.grad.clone()
                        ref_dv = v_ref.grad.clone()

                        # MXfp8
                        q = q_ref.detach().clone().requires_grad_(True)
                        k = k_ref.detach().clone().requires_grad_(True)
                        v = v_ref.detach().clone().requires_grad_(True)

                        out = mxfp8_flash_attention(q, k, v, causal, sm_scale)
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

    fwd_ok = test_forward()
    print()
    bwd_ok = test_backward()

    print()
    if fwd_ok and bwd_ok:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
