"""
MXfp8 Flash Attention for SM120 (RTX 5090)

Uses E4M3 data + E8M0 per-block scales (32 elements per block) with
tl.dot_scaled for the QK^T and PV matmuls in the forward pass.
Backward pass uses standard FP16 tl.dot for numerical stability.

Accepts FP16 [B, H, N, D] inputs. Quantizes to MXfp8 on-the-fly inside kernels.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# =====================================================================
# MXfp8 Quantization Helpers
# =====================================================================

@triton.jit
def _quantize_mxfp8_row(tile, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Quantize [ROWS, COLS] f32 tile to MXfp8 along the last (COLS) dim.

    Groups of 32 elements along COLS. COLS must be a multiple of 32.
    Returns: (fp8_data [ROWS, COLS] float8e4nv,
              scale_e8m0 [ROWS, COLS//32] uint8)
    """
    GRP: tl.constexpr = COLS // 32
    NGRP: tl.constexpr = ROWS * GRP

    # Reshape to [NGRP, 32] — each row is one scale group
    flat = tl.reshape(tile, (NGRP, 32))
    amax = tl.max(tl.abs(flat), axis=1)  # [NGRP]

    # E8M0 biased exponent = floor(log2(amax)) + 127
    log2a = tl.math.floor(tl.math.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = tl.maximum(tl.minimum(biased, 254.0), 0.0)

    # Actual power-of-2 scale for division
    scale = tl.math.exp2(log2a)  # [NGRP]

    # Scale elements and clamp to E4M3 representable range [-448, 448]
    scaled = flat / scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)

    fp8 = tl.reshape(scaled, (ROWS, COLS)).to(tl.float8e4nv)
    e8m0 = tl.reshape(biased.to(tl.uint8), (ROWS, GRP))
    return fp8, e8m0


@triton.jit
def _quantize_mxfp8_col(tile, K: tl.constexpr, N: tl.constexpr):
    """Quantize [K, N] f32 tile along the first (K) dim.

    Groups of 32 elements along K. K must be a multiple of 32.
    Returns: (fp8_data [K, N] float8e4nv,
              scale_e8m0 [N, K//32] uint8)

    The returned scale shape [N, K//32] matches the rhs_scale
    convention of tl.dot_scaled where rhs is [K, N].
    """
    G: tl.constexpr = K // 32

    # 3-D reshape: [G, 32, N]
    tile_3d = tl.reshape(tile, (G, 32, N))
    amax = tl.max(tl.abs(tile_3d), axis=1)  # [G, N]

    log2a = tl.math.floor(tl.math.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = tl.maximum(tl.minimum(biased, 254.0), 0.0)
    scale = tl.math.exp2(log2a)  # [G, N]

    # Broadcast division: [G, 32, N] / [G, 1, N]
    scaled_3d = tile_3d / tl.reshape(scale, (G, 1, N))
    scaled = tl.reshape(scaled_3d, (K, N))
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)

    fp8 = scaled.to(tl.float8e4nv)
    # Transpose scale: [G, N] → [N, G] = [N, K//32]
    e8m0 = tl.trans(biased.to(tl.uint8))
    return fp8, e8m0


# =====================================================================
# Forward Kernel
# =====================================================================

@triton.jit
def _mxfp8_attn_fwd(
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

    # Base pointer offsets for this batch/head
    q_base = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_base = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_base = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Quantize Q once (stays in registers for all K/V iterations)
    q_fp8, q_scale = _quantize_mxfp8_row(q.to(tl.float32), BLOCK_M, HEAD_DIM)

    # Online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale factor in log-2 space: sm_scale / ln(2)
    qk_scale = sm_scale * 1.44269504

    # Loop upper bound
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # ---- Load & quantize K [BLOCK_N, HEAD_DIM] ----
        k_ptrs = K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        k_fp8, k_scale = _quantize_mxfp8_row(k.to(tl.float32), BLOCK_N, HEAD_DIM)

        # ---- QK^T via dot_scaled ----
        # lhs=[BLOCK_M, HEAD_DIM], rhs=k^T=[HEAD_DIM, BLOCK_N]
        # rhs_scale = k_scale [BLOCK_N, HEAD_DIM//32]  (not transposed)
        qk = tl.dot_scaled(q_fp8, q_scale, "e4m3",
                           tl.trans(k_fp8), k_scale, "e4m3")
        qk = qk * qk_scale

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        # ---- Online softmax ----
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # ---- Load & quantize V [BLOCK_N, HEAD_DIM] ----
        v_ptrs = V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # Quantize P [BLOCK_M, BLOCK_N] — along BLOCK_N (last dim)
        p_fp8, p_scale = _quantize_mxfp8_row(p, BLOCK_M, BLOCK_N)

        # Quantize V [BLOCK_N, HEAD_DIM] — along BLOCK_N (first dim)
        # This gives rhs_scale in [HEAD_DIM, BLOCK_N//32] = [N, K//32]
        v_fp8, v_scale = _quantize_mxfp8_col(v.to(tl.float32), BLOCK_N, HEAD_DIM)

        # ---- PV via dot_scaled ----
        # lhs=p_fp8 [BLOCK_M, BLOCK_N], rhs=v_fp8 [BLOCK_N, HEAD_DIM]
        # rhs_scale=v_scale [HEAD_DIM, BLOCK_N//32]
        acc = tl.dot_scaled(p_fp8, p_scale, "e4m3",
                            v_fp8, v_scale, "e4m3", acc)

        m_i = m_ij

    # ---- Epilogue: normalize & store ----
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX)

    # Store log-sum-exp (log2 space) for backward
    lse = m_i + tl.math.log2(l_i)
    lse_ptrs = LSE + off_hz * N_CTX + offs_m
    tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


# =====================================================================
# Backward Kernels (FP16 tl.dot for numerical stability)
# =====================================================================

@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Compute Delta[i] = rowsum(O[i] * dO[i])."""
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)

    # Assumes contiguous [B*H, N, D] layout
    base = off_hz * N_CTX * HEAD_DIM
    o = tl.load(O + base + off_m[:, None] * HEAD_DIM + offs_d[None, :])
    do = tl.load(DO + base + off_m[:, None] * HEAD_DIM + offs_d[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk, dv,
    Q, k, v, sm_scale,
    DO, M, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_n, start_m, num_steps,
    MASK: tl.constexpr,
):
    """Inner loop: for a fixed K,V block, iterate over Q blocks to accumulate dK, dV."""
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    # Q^T pointers: load Q transposed [HEAD_DIM, BLOCK_M1]
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m

    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)

        # Recompute attention: k is pre-scaled by sm_scale/ln2
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)

        # dV += P^T @ dO
        dv += tl.dot(pT.to(tl.float16), do)

        # dK += dS^T @ Q
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))

        curr_m += BLOCK_M1
        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok

    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq, q, K, V,
    do, m, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_m, start_n, num_steps,
    MASK: tl.constexpr,
):
    """Inner loop: for a fixed Q block, iterate over K,V blocks to accumulate dQ."""
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    # K^T and V^T pointers: [HEAD_DIM, BLOCK_N2]
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d

    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n

    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)

        # Recompute attention (k is pre-scaled)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)

        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)

        # dP = dO @ V^T
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])

        # dQ += dS @ K (using pre-scaled K)
        dq += tl.dot(ds.to(tl.float16), tl.trans(kT))

        curr_n += BLOCK_N2
        kT_ptrs += BLOCK_N2 * stride_tok
        vT_ptrs += BLOCK_N2 * stride_tok

    return dq


@triton.jit
def _attn_bwd(
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
    LN2: tl.constexpr = 0.6931471824645996  # ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # Offset all pointers for this batch/head
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

    # Load K, V block (stay in SRAM)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv, Q, k, v, sm_scale, DO, M, D,
            stride_tok, stride_d, H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps, MASK=True,
        )
        start_m += num_steps * MASK_BLOCK_M1
    else:
        start_m = 0

    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk, dv, Q, k, v, sm_scale, DO, M, D,
        stride_tok, stride_d, H, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps, MASK=False,
    )

    # Store dV
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Store dK (multiply by sm_scale)
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
        dq = _attn_bwd_dq(
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

    dq = _attn_bwd_dq(
        dq, q, K, V, do, m, D,
        stride_tok, stride_d, H, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps, MASK=False,
    )

    # Store dQ (undo the 1/ln2 pre-scaling on K)
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# =====================================================================
# PyTorch Autograd Wrapper
# =====================================================================

class MXfp8FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # q, k, v: [B, H, N, D] in float16
        B, H, N, D = q.shape
        assert D in {64, 128}, f"HEAD_DIM must be 64 or 128, got {D}"
        assert N % 128 == 0, f"N_CTX must be a multiple of 128, got {N}"

        # Ensure contiguous for uniform strides in backward
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        BLOCK_M = 128
        BLOCK_N = 64

        o = torch.empty_like(q)
        lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(N, BLOCK_M), B * H)

        _mxfp8_attn_fwd[grid](
            q, k, v, sm_scale, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, N,
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
        B, H, N, D = q.shape

        do = do.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Preprocess: Delta = rowsum(O * dO)
        PRE_BLOCK = 128
        pre_grid = (N // PRE_BLOCK, B * H)
        delta = torch.empty_like(lse)

        _attn_bwd_preprocess[pre_grid](
            o, do, delta,
            B, H, N,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=D,
        )

        # Pre-scale K by sm_scale / ln(2) so inner loops avoid per-iteration scaling
        RCP_LN2 = 1.4426950408889634
        k_scaled = k * (ctx.sm_scale * RCP_LN2)

        BLOCK_M1, BLOCK_N1 = 32, 128
        BLOCK_M2, BLOCK_N2 = 128, 32
        BLK_SLICE_FACTOR = 2

        grid = (N // BLOCK_N1, 1, B * H)

        _attn_bwd[grid](
            q, k_scaled, v, ctx.sm_scale,
            do, dq, dk, dv,
            lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, N,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=D,
            CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=2,
        )

        return dq, dk, dv, None, None


def mxfp8_flash_attention(q, k, v, causal=False, sm_scale=None):
    """MXfp8 Flash Attention.

    Args:
        q, k, v: [B, H, N, D] float16 tensors
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(D))

    Returns:
        output: [B, H, N, D] float16
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return MXfp8FlashAttention.apply(q, k, v, causal, sm_scale)
