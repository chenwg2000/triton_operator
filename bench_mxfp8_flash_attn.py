"""
Benchmark for MXfp8 Flash Attention — Multi-Variant Comparison.

Compares:
  - BF16 baseline (Triton tl.dot, no quantization)
  - MXfp8 v1 (in-kernel quantization)
  - MXfp8 v2 (host pre-quantization + autotuned forward)
  - PyTorch SDPA (vendor-optimized baseline)

Measures wall-clock time and TFLOPS for various configurations.

Usage:
    source .venv/bin/activate
    python bench_mxfp8_flash_attn.py
"""

import torch
from triton.testing import do_bench

DEVICE = "cuda"


def flash_attn_flops(B, H, N, D, causal, mode="fwd"):
    """Theoretical FLOPs for flash attention.

    Forward:  4*B*H*N*N*D (2 matmuls: QK^T + PV)
    Backward: 7*B*H*N*N*D (~3.5x forward)
    For causal, effective N*N → N*N/2.
    """
    seq_factor = 0.5 if causal else 1.0
    if mode == "fwd":
        return 4 * B * H * N * N * D * seq_factor
    elif mode == "bwd":
        return 7 * B * H * N * N * D * seq_factor
    else:
        return 11 * B * H * N * N * D * seq_factor


# =====================================================================
# Attention Implementations
# =====================================================================

def _load_implementations():
    """Load available implementations, returning a dict of name → (fwd_fn, has_bwd)."""
    impls = {}

    # PyTorch SDPA
    def sdpa_fn(q, k, v, causal, sm_scale):
        q_ = q.permute(0, 2, 1, 3)
        k_ = k.permute(0, 2, 1, 3)
        v_ = v.permute(0, 2, 1, 3)
        o = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, is_causal=causal, scale=sm_scale)
        return o.permute(0, 2, 1, 3)
    impls["sdpa"] = (sdpa_fn, True)

    # BF16 Triton baseline
    try:
        from bf16_flash_attn import bf16_flash_attention
        impls["bf16_triton"] = (bf16_flash_attention, True)
    except ImportError:
        pass

    # MXfp8 v1
    try:
        from mxfp8_flash_attn import mxfp8_flash_attention
        impls["mxfp8_v1"] = (mxfp8_flash_attention, True)
    except ImportError:
        pass

    # MXfp8 v2
    try:
        from mxfp8_flash_attn_v2 import mxfp8_flash_attention_v2
        impls["mxfp8_v2"] = (mxfp8_flash_attention_v2, True)
    except ImportError:
        pass

    return impls


def bench_fwd(fn, q, k, v, causal, sm_scale):
    """Benchmark forward pass."""
    return do_bench(
        lambda: fn(q, k, v, causal, sm_scale),
        warmup=25, rep=100,
    )


def bench_bwd(fn, q, k, v, causal, sm_scale):
    """Benchmark backward pass."""
    q_g = q.clone().requires_grad_(True)
    k_g = k.clone().requires_grad_(True)
    v_g = v.clone().requires_grad_(True)
    o = fn(q_g, k_g, v_g, causal, sm_scale)
    dout = torch.randn_like(o)

    def bwd_fn():
        if q_g.grad is not None:
            q_g.grad.zero_()
            k_g.grad.zero_()
            v_g.grad.zero_()
        o.backward(dout, retain_graph=True)

    return do_bench(bwd_fn, warmup=25, rep=100)


def main():
    gpu_name = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {gpu_name}  (SM {cc[0]}.{cc[1]})")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    impls = _load_implementations()
    impl_names = list(impls.keys())
    print(f"Implementations: {', '.join(impl_names)}")
    print()

    configs = [
        # (B, H, N, D, causal)
        (1, 32, 512,   128, False),
        (1, 32, 512,   128, True),
        (1, 32, 1024,  128, False),
        (1, 32, 1024,  128, True),
        (1, 32, 2048,  128, False),
        (1, 32, 2048,  128, True),
        (1, 32, 4096,  128, False),
        (1, 32, 4096,  128, True),
        (1, 32, 4096,   64, False),
        (1, 32, 4096,   64, True),
        (4,  8, 2048,  128, False),
        (4,  8, 2048,  128, True),
    ]

    # =====================================================================
    # Forward Benchmark
    # =====================================================================
    print("=" * 80)
    print("FORWARD PASS BENCHMARK")
    print("=" * 80)

    # Header
    hdr = f"{'B':>2} {'H':>3} {'N':>5} {'D':>4} {'causal':>6}"
    for name in impl_names:
        hdr += f" | {name:>12s} TF/s"
    print(hdr)
    print("-" * len(hdr))

    for B, H, N, D, causal in configs:
        torch.manual_seed(0)
        sm_scale = 0.5
        q = torch.randn(B, N, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, N, H, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, N, H, D, device=DEVICE, dtype=torch.bfloat16)

        flops = flash_attn_flops(B, H, N, D, causal, "fwd")
        line = f"{B:2d} {H:3d} {N:5d} {D:4d} {str(causal):>6s}"

        for name in impl_names:
            fn, _ = impls[name]
            try:
                ms = bench_fwd(fn, q, k, v, causal, sm_scale)
                tflops = flops / (ms * 1e-3) / 1e12
                line += f" | {ms:7.3f}ms {tflops:5.1f}"
            except Exception as e:
                line += f" | {'ERR':>12s}     "

        print(line)

    # =====================================================================
    # Backward Benchmark
    # =====================================================================
    print()
    print("=" * 80)
    print("BACKWARD PASS BENCHMARK")
    print("=" * 80)

    hdr = f"{'B':>2} {'H':>3} {'N':>5} {'D':>4} {'causal':>6}"
    for name in impl_names:
        hdr += f" | {name:>12s} TF/s"
    print(hdr)
    print("-" * len(hdr))

    bwd_configs = [
        (1, 32, 512,   128, False),
        (1, 32, 512,   128, True),
        (1, 32, 1024,  128, False),
        (1, 32, 1024,  128, True),
        (1, 32, 2048,  128, False),
        (1, 32, 2048,  128, True),
        (1, 32, 4096,  128, False),
        (1, 32, 4096,  128, True),
        (4,  8, 2048,  128, False),
        (4,  8, 2048,  128, True),
    ]

    for B, H, N, D, causal in bwd_configs:
        torch.manual_seed(0)
        sm_scale = 0.5
        q = torch.randn(B, N, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, N, H, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, N, H, D, device=DEVICE, dtype=torch.bfloat16)

        flops = flash_attn_flops(B, H, N, D, causal, "bwd")
        line = f"{B:2d} {H:3d} {N:5d} {D:4d} {str(causal):>6s}"

        for name in impl_names:
            fn, has_bwd = impls[name]
            if not has_bwd:
                line += f" | {'N/A':>12s}     "
                continue
            try:
                ms = bench_bwd(fn, q, k, v, causal, sm_scale)
                tflops = flops / (ms * 1e-3) / 1e12
                line += f" | {ms:7.3f}ms {tflops:5.1f}"
            except Exception as e:
                line += f" | {'ERR':>12s}     "

        print(line)

    print()
    print("TF/s = TeraFLOPS")


if __name__ == "__main__":
    main()
