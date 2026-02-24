"""
BF16 Flash Attention for SM120 (RTX 5090) — Benchmark Baseline

Uses standard tl.dot (BF16 MMA) with no quantization.
Provides a baseline to measure MXfp8 hardware acceleration speedup.

Accepts BF16 [B, S, H, D] inputs. Forward and backward passes.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# =====================================================================
# Forward Kernel
# =====================================================================

@triton.jit
def _bf16_attn_fwd(
    Q, K, V, sm_scale, Out, LSE,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_base = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_base = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K [BLOCK_N, HEAD_DIM]
        k_ptrs = K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # QK^T via standard tl.dot
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Load V [BLOCK_N, HEAD_DIM]
        v_ptrs = V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # PV via standard tl.dot
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        m_i = m_ij

    # Epilogue
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX)

    lse = m_i + tl.math.log2(l_i)
    lse_ptrs = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


# =====================================================================
# Backward Kernels
# =====================================================================

@triton.jit
def _bf16_bwd_preprocess(
    O, DO, Delta,
    stride_z, stride_h, stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_d = tl.arange(0, HEAD_DIM)

    base = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h
    o = tl.load(O + base + off_m[:, None] * stride_tok + offs_d[None, :] * stride_d)
    do = tl.load(DO + base + off_m[:, None] * stride_tok + offs_d[None, :] * stride_d).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _bf16_bwd_dkdv(
    dk, dv,
    Q, k, v, sm_scale,
    DO, M, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_n, start_m, num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m

    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        # qkT = k @ qT  [N1,HD] @ [HD,M1] → [N1,M1]
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)

        # dV += pT @ dO  [N1,M1] @ [M1,HD] → [N1,HD]
        dv = tl.dot(pT.to(tl.bfloat16), do, dv)

        Di = tl.load(D + offs_m)
        # dpT = v @ dO^T  [N1,HD] @ [HD,M1] → [N1,M1]
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])

        # dK += dsT @ qT^T  [N1,M1] @ [M1,HD] → [N1,HD]
        dk = tl.dot(dsT.to(tl.bfloat16), tl.trans(qT), dk)

        curr_m += BLOCK_M1
        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok

    return dk, dv


@triton.jit
def _bf16_bwd_dq(
    dq, q, K, V,
    do, m, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_m, start_n, num_steps,
    MASK: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n

    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)

        # qk = q @ kT  [M2,HD] @ [HD,N2] → [M2,N2]
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)

        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)

        # dp = do @ vT  [M2,HD] @ [HD,N2] → [M2,N2]
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])

        # dQ += ds @ kT^T  [M2,N2] @ [N2,HD] → [M2,HD]
        dq = tl.dot(ds.to(tl.bfloat16), tl.trans(kT), dq)

        curr_n += BLOCK_N2
        kT_ptrs += BLOCK_N2 * stride_tok
        vT_ptrs += BLOCK_N2 * stride_tok

    return dq


@triton.jit
def _bf16_bwd(
    Q, K, V, sm_scale,
    DO, DQ, DK, DV,
    M, D,
    stride_z, stride_h, stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)

    # ========== dK, dV ==========
    start_n = pid * BLOCK_N1
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _bf16_bwd_dkdv(
            dk, dv, Q, k, v, sm_scale, DO, M, D,
            stride_tok, stride_d, H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps, MASK=True,
        )
        start_m += num_steps * MASK_BLOCK_M1
    else:
        start_m = 0

    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _bf16_bwd_dkdv(
        dk, dv, Q, k, v, sm_scale, DO, M, D,
        stride_tok, stride_d, H, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps, MASK=False,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # ========== dQ ==========
    start_m = pid * BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _bf16_bwd_dq(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, MASK=True,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2
    else:
        num_steps = N_CTX // BLOCK_N2
        start_n = 0

    dq = _bf16_bwd_dq(
        dq, q, K, V, do, m, D,
        stride_tok, stride_d, H, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps, MASK=False,
    )

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# =====================================================================
# PyTorch Autograd Wrapper
# =====================================================================

class BF16FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        B, S, H, D = q.shape
        assert D in {64, 128}
        assert S % 128 == 0

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        BLOCK_M = 128
        BLOCK_N = 64

        o = torch.empty_like(q)
        lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(S, BLOCK_M), B * H)

        _bf16_attn_fwd[grid](
            q, k, v, sm_scale, o, lse,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            B, H, S,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = D
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, S, H, D = q.shape

        do = do.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        PRE_BLOCK = 128
        pre_grid = (S // PRE_BLOCK, B * H)
        delta = torch.empty_like(lse)

        _bf16_bwd_preprocess[pre_grid](
            o, do, delta,
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            H, S,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=D,
        )

        RCP_LN2 = 1.4426950408889634
        k_scaled = k * (ctx.sm_scale * RCP_LN2)

        BLOCK_M1, BLOCK_N1 = 32, 128
        BLOCK_M2, BLOCK_N2 = 128, 32
        BLK_SLICE_FACTOR = 1

        grid = (S // BLOCK_N1, 1, B * H)

        _bf16_bwd[grid](
            q, k_scaled, v, ctx.sm_scale,
            do, dq, dk, dv,
            lse, delta,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            H, S,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=D,
            CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=2,
        )

        return dq, dk, dv, None, None


def bf16_flash_attention(q, k, v, causal=False, sm_scale=None):
    """BF16 Flash Attention (no quantization).

    Args:
        q, k, v: [B, S, H, D] bfloat16 tensors
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(D))

    Returns:
        output: [B, S, H, D] bfloat16
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return BF16FlashAttention.apply(q, k, v, causal, sm_scale)
