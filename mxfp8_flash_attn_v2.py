"""
MXfp8 Flash Attention v2 — Optimized for RTX 5090 (SM 12.0)

Features:
1. MXfp8 forward with host pre-quantization + autotune (offline, large N)
2. MXfp8 forward with in-kernel quantization + autotune (online, small N)
3. Optional FP8 output for TE integration (eliminates dequant→requant)
   - fp8_data [B*S, H*D], rowwise_scale [B*S, H*D//32], colwise_scale [H*D, B*S//32]
4. BF16 backward (fast, no quantization overhead)

Accepts BF16 [B, S, H, D] inputs.
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
VEC_SIZE = 32  # MXfp8 group size (E8M0 scales)


# =====================================================================
# Host Pre-Quantization
# =====================================================================

def quantize_mxfp8(x):
    """Quantize along last dim: [..., K] → fp8 [..., K] + scale [..., K//32]."""
    assert x.shape[-1] % VEC_SIZE == 0
    orig_shape = x.shape
    K = orig_shape[-1]
    x_f32 = x.float().reshape(-1, K // VEC_SIZE, VEC_SIZE)
    amax = x_f32.abs().amax(dim=-1)
    log2_amax = torch.floor(torch.log2(amax + 1e-12))
    biased = (log2_amax + 127.0).clamp(0.0, 254.0)
    scale = torch.exp2(log2_amax)
    scaled = x_f32 / scale.unsqueeze(-1)
    scaled = scaled.clamp(-448.0, 448.0)
    fp8 = scaled.reshape(orig_shape).to(torch.float8_e4m3fn)
    scales = biased.to(torch.uint8).reshape(*orig_shape[:-1], K // VEC_SIZE)
    return fp8, scales


def quantize_mxfp8_seqdim(x):
    """Quantize [*, S, D] along S dim → fp8 [*, S, D] + scale [*, S//32, D]."""
    orig_shape = x.shape
    S, D = orig_shape[-2], orig_shape[-1]
    assert S % VEC_SIZE == 0
    batch_size = 1
    for d in orig_shape[:-2]:
        batch_size *= d
    x_f32 = x.float().reshape(batch_size, S // VEC_SIZE, VEC_SIZE, D)
    amax = x_f32.abs().amax(dim=2)
    log2_amax = torch.floor(torch.log2(amax + 1e-12))
    biased = (log2_amax + 127.0).clamp(0.0, 254.0)
    scale = torch.exp2(log2_amax)
    scaled = x_f32 / scale.unsqueeze(2)
    scaled = scaled.clamp(-448.0, 448.0)
    fp8 = scaled.reshape(*orig_shape[:-2], S, D).to(torch.float8_e4m3fn)
    scales = biased.to(torch.uint8).reshape(*orig_shape[:-2], S // VEC_SIZE, D)
    return fp8, scales


# =====================================================================
# In-Kernel Quantization Helpers
# =====================================================================

@triton.jit
def _quantize_p_mxfp8(p, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Quantize P (non-negative softmax output). Skip abs()."""
    GRP: tl.constexpr = BLOCK_N // 32
    NGRP: tl.constexpr = BLOCK_M * GRP
    flat = tl.reshape(p, (NGRP, 32))
    amax = tl.max(flat, axis=1)
    log2a = tl.math.floor(tl.math.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = tl.maximum(tl.minimum(biased, 254.0), 0.0)
    scale = tl.math.exp2(log2a)
    scaled = flat / scale[:, None]
    scaled = tl.minimum(scaled, 448.0)
    fp8 = tl.reshape(scaled, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
    e8m0 = tl.reshape(biased.to(tl.uint8), (BLOCK_M, GRP))
    return fp8, e8m0


@triton.jit
def _quantize_mxfp8_row(tile, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Quantize [ROWS, COLS] along last dim (general, handles negatives)."""
    GRP: tl.constexpr = COLS // 32
    NGRP: tl.constexpr = ROWS * GRP
    flat = tl.reshape(tile, (NGRP, 32))
    amax = tl.max(tl.abs(flat), axis=1)
    log2a = tl.math.floor(tl.math.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = tl.maximum(tl.minimum(biased, 254.0), 0.0)
    scale = tl.math.exp2(log2a)
    scaled = flat / scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    fp8 = tl.reshape(scaled, (ROWS, COLS)).to(tl.float8e4nv)
    e8m0 = tl.reshape(biased.to(tl.uint8), (ROWS, GRP))
    return fp8, e8m0


@triton.jit
def _quantize_mxfp8_col(tile, K: tl.constexpr, N: tl.constexpr):
    """Quantize [K, N] along first dim."""
    G: tl.constexpr = K // 32
    tile_3d = tl.reshape(tile, (G, 32, N))
    amax = tl.max(tl.abs(tile_3d), axis=1)
    log2a = tl.math.floor(tl.math.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = tl.maximum(tl.minimum(biased, 254.0), 0.0)
    scale = tl.math.exp2(log2a)
    scaled_3d = tile_3d / tl.reshape(scale, (G, 1, N))
    scaled = tl.reshape(scaled_3d, (K, N))
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    fp8 = scaled.to(tl.float8e4nv)
    e8m0 = tl.trans(biased.to(tl.uint8))
    return fp8, e8m0


@triton.jit
def _compute_e8m0_scale(amax):
    """Compute E8M0 biased exponent from amax. Returns (biased_uint8, power_of_2_scale)."""
    log2a = tl.math.floor(tl.math.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = tl.maximum(tl.minimum(biased, 254.0), 0.0)
    return biased.to(tl.uint8)


# =====================================================================
# Forward Kernel — Online mode (in-kernel quantization, small N)
# =====================================================================

_fwd_online_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_fwd_online_configs, key=['N_CTX', 'HEAD_DIM'])
@triton.jit
def _mxfp8_attn_fwd_online(
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

    q = tl.load(Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_CTX, other=0.0)
    q_fp8, q_scale = _quantize_mxfp8_row(q.to(tl.float32), BLOCK_M, HEAD_DIM)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if CAUSAL else N_CTX

    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=offs_n[:, None] < N_CTX, other=0.0)
        k_fp8, k_scale = _quantize_mxfp8_row(k.to(tl.float32), BLOCK_N, HEAD_DIM)

        qk = tl.dot_scaled(q_fp8, q_scale, "e4m3", tl.trans(k_fp8), k_scale, "e4m3")
        qk = qk * qk_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=offs_n[:, None] < N_CTX, other=0.0)
        p_fp8, p_scale = _quantize_p_mxfp8(p, BLOCK_M, BLOCK_N)
        v_fp8, v_scale = _quantize_mxfp8_col(v.to(tl.float32), BLOCK_N, HEAD_DIM)
        acc = tl.dot_scaled(p_fp8, p_scale, "e4m3", v_fp8, v_scale, "e4m3", acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX)
    lse = m_i + tl.math.log2(l_i)
    tl.store(LSE + off_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)


# =====================================================================
# Forward Kernel — Offline mode (host pre-quantized, autotuned)
# Supports optional FP8 output for TE integration.
#
# Output layout when OUT_FP8=True:
#   fp8_data:    [B*S, H*D]       — same physical order as [B, S, H, D]
#   rowwise_sc:  [B*S, H*D // 32] — groups of 32 along H*D
#   colwise_sc:  [H*D, B*S // 32] — groups of 32 along B*S
# =====================================================================

_fwd_offline_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_fwd_offline_configs, key=['N_CTX', 'HEAD_DIM'])
@triton.jit
def _mxfp8_attn_fwd_offline(
    # Pre-quantized inputs [BH, S, D] / [BH, S, D//32] / [BH, S//32, D]
    Q_fp8, Q_scale, K_fp8, K_scale, V_fp8, V_scale,
    sm_scale,
    # BF16 output [B*S, H*D] — always written (needed for backward)
    Out_bf16,
    # FP8 output tensors (only written when OUT_FP8=True)
    Out_fp8, Out_rowscale, Out_colscale,
    # LSE [BH, S]
    LSE,
    # Dimensions
    N_CTX, NUM_HEADS, BS_DIV32,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    OUT_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    start_m = pid_m * BLOCK_M

    SCALE_D: tl.constexpr = HEAD_DIM // 32
    VSCALE_GROUPS: tl.constexpr = BLOCK_N // 32

    # Input base offsets ([BH, S, D] layout)
    qkv_base = off_bh.to(tl.int64) * N_CTX * HEAD_DIM
    qs_base = off_bh.to(tl.int64) * N_CTX * SCALE_D
    vs_base = off_bh.to(tl.int64) * (N_CTX // 32) * HEAD_DIM

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_sd = tl.arange(0, SCALE_D)

    # Load Q
    q_tile = tl.load(Q_fp8 + qkv_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                     mask=offs_m[:, None] < N_CTX, other=0.0)
    q_sc = tl.load(Q_scale + qs_base + offs_m[:, None] * SCALE_D + offs_sd[None, :],
                   mask=offs_m[:, None] < N_CTX, other=0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX) if CAUSAL else N_CTX

    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_tile = tl.load(K_fp8 + qkv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                         mask=offs_n[:, None] < N_CTX, other=0.0)
        k_sc = tl.load(K_scale + qs_base + offs_n[:, None] * SCALE_D + offs_sd[None, :],
                       mask=offs_n[:, None] < N_CTX, other=0)

        qk = tl.dot_scaled(q_tile, q_sc, "e4m3", tl.trans(k_tile), k_sc, "e4m3")
        qk = qk * qk_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(V_fp8 + qkv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                         mask=offs_n[:, None] < N_CTX, other=0.0)
        offs_vsg = start_n // 32 + tl.arange(0, VSCALE_GROUPS)
        v_sc = tl.load(V_scale + vs_base + offs_vsg[:, None] * HEAD_DIM + offs_d[None, :])
        v_sc_t = tl.trans(v_sc)

        p_fp8, p_sc = _quantize_p_mxfp8(p, BLOCK_M, BLOCK_N)
        acc = tl.dot_scaled(p_fp8, p_sc, "e4m3", v_tile, v_sc_t, "e4m3", acc)
        m_i = m_ij

    # ---- Epilogue ----
    acc_norm = acc / l_i[:, None]  # [BLOCK_M, HEAD_DIM] float32

    # Decompose batch-head index
    off_b = off_bh // NUM_HEADS
    off_h = off_bh % NUM_HEADS
    HD = NUM_HEADS * HEAD_DIM  # total H*D stride

    # Output rows/cols in [B*S, H*D] layout
    o_rows = off_b * N_CTX + start_m + tl.arange(0, BLOCK_M)
    o_cols = off_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
    o_mask = o_rows[:, None] < (off_b + 1) * N_CTX

    # Always write BF16 output (needed for backward)
    bf16_ptrs = Out_bf16 + o_rows[:, None] * HD + o_cols[None, :]
    tl.store(bf16_ptrs, acc_norm.to(tl.bfloat16), mask=o_mask)

    if OUT_FP8:
        # ---- Rowwise quantization: groups of 32 along HEAD_DIM ----
        fp8_data, row_scale = _quantize_mxfp8_row(acc_norm, BLOCK_M, HEAD_DIM)

        # Store fp8_data [B*S, H*D]
        fp8_ptrs = Out_fp8 + o_rows[:, None] * HD + o_cols[None, :]
        tl.store(fp8_ptrs, fp8_data, mask=o_mask)

        # Store rowwise_scale [B*S, H*D//32]
        TOTAL_SCALE_D = NUM_HEADS * SCALE_D  # H * D//32
        rs_cols = off_h * SCALE_D + tl.arange(0, SCALE_D)
        rs_ptrs = Out_rowscale + o_rows[:, None] * TOTAL_SCALE_D + rs_cols[None, :]
        tl.store(rs_ptrs, row_scale, mask=o_mask)

        # ---- Columnwise scale: groups of 32 along B*S ----
        BLOCK_GROUPS: tl.constexpr = BLOCK_M // 32
        acc_3d = tl.reshape(acc_norm, (BLOCK_GROUPS, 32, HEAD_DIM))
        col_amax = tl.max(tl.abs(acc_3d), axis=1)  # [BLOCK_GROUPS, HEAD_DIM]
        col_scale = _compute_e8m0_scale(col_amax)   # [BLOCK_GROUPS, HEAD_DIM]
        col_scale_t = tl.trans(col_scale)            # [HEAD_DIM, BLOCK_GROUPS]

        # Store colwise_scale [H*D, B*S//32]
        cs_rows = off_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
        cs_cols = off_b * (N_CTX // 32) + start_m // 32 + tl.arange(0, BLOCK_GROUPS)
        cs_ptrs = Out_colscale + cs_rows[:, None] * BS_DIV32 + cs_cols[None, :]
        tl.store(cs_ptrs, col_scale_t)

    # Store LSE [BH, S]
    lse = m_i + tl.math.log2(l_i)
    tl.store(LSE + off_bh * N_CTX + offs_m, lse, mask=offs_m < N_CTX)


# Sequence length threshold for online vs offline
_OFFLINE_THRESHOLD = 1024


# =====================================================================
# Backward — BF16 tl.dot (no quantization overhead)
# =====================================================================

from bf16_flash_attn import (
    _bf16_bwd_preprocess as _attn_bwd_preprocess,
    _bf16_bwd as _attn_bwd,
)


# =====================================================================
# PyTorch Autograd Wrapper
# =====================================================================

class MXfp8FlashAttentionV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        B, S, H, D = q.shape
        BH = B * H
        assert D in {64, 128}, f"HEAD_DIM must be 64 or 128, got {D}"
        assert S % 128 == 0, f"N_CTX must be a multiple of 128, got {S}"

        if S >= _OFFLINE_THRESHOLD:
            o = _run_offline_fwd(q, k, v, sm_scale, causal, out_fp8=False)
        else:
            o = _run_online_fwd(q, k, v, sm_scale, causal)

        # o is (output_bf16 [B,S,H,D], lse [B,H,S])
        ctx.save_for_backward(q, k, v, o[0], o[1])
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = D
        ctx.causal = causal
        return o[0]

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, S, H, D = q.shape

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        PRE_BLOCK = 128
        pre_grid = (S // PRE_BLOCK, B * H)
        delta = torch.empty_like(lse)

        _attn_bwd_preprocess[pre_grid](
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

        _attn_bwd[grid](
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


# =====================================================================
# Forward Helpers
# =====================================================================

def _run_online_fwd(q, k, v, sm_scale, causal):
    """Online mode: in-kernel quantization. Returns (output, lse)."""
    B, S, H, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)
    lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)

    _mxfp8_attn_fwd_online[grid](
        q, k, v, sm_scale, o, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        B, H, S,
        HEAD_DIM=D,
        CAUSAL=causal,
    )
    return o, lse


def _run_offline_fwd(q, k, v, sm_scale, causal, out_fp8=False):
    """Offline mode: host pre-quantization. Returns (output_bf16, lse) or
    (output_bf16, lse, fp8_data, rowwise_scale, colwise_scale) when out_fp8=True."""
    B, S, H, D = q.shape
    BH = B * H

    # Pre-quantize inputs → [BH, S, D]
    q_bhs = q.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)
    k_bhs = k.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)
    v_bhs = v.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)

    q_fp8, q_scale = quantize_mxfp8(q_bhs)
    k_fp8, k_scale = quantize_mxfp8(k_bhs)
    v_fp8, v_scale = quantize_mxfp8_seqdim(v_bhs)

    q_fp8, q_scale = q_fp8.contiguous(), q_scale.contiguous()
    k_fp8, k_scale = k_fp8.contiguous(), k_scale.contiguous()
    v_fp8, v_scale = v_fp8.contiguous(), v_scale.contiguous()

    # BF16 output in [B*S, H*D] layout (= [B, S, H, D] physically)
    o_bf16 = torch.empty(B * S, H * D, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(BH, S, device=q.device, dtype=torch.float32)

    # FP8 output tensors (allocated only if needed)
    if out_fp8:
        o_fp8 = torch.empty(B * S, H * D, device=q.device, dtype=torch.float8_e4m3fn)
        o_rowscale = torch.empty(B * S, H * D // 32, device=q.device, dtype=torch.uint8)
        o_colscale = torch.empty(H * D, B * S // 32, device=q.device, dtype=torch.uint8)
    else:
        # Dummy pointers (never written)
        o_fp8 = o_bf16  # won't be written
        o_rowscale = lse  # won't be written
        o_colscale = lse  # won't be written

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), BH)

    _mxfp8_attn_fwd_offline[grid](
        q_fp8, q_scale, k_fp8, k_scale, v_fp8, v_scale,
        sm_scale,
        o_bf16, o_fp8, o_rowscale, o_colscale,
        lse,
        S, H, B * S // 32,
        HEAD_DIM=D,
        CAUSAL=causal,
        OUT_FP8=out_fp8,
    )

    # View BF16 output as [B, S, H, D]
    o_bf16_4d = o_bf16.view(B, S, H, D)
    lse_3d = lse.reshape(B, H, S)

    if out_fp8:
        return o_bf16_4d, lse_3d, o_fp8, o_rowscale, o_colscale
    else:
        return o_bf16_4d, lse_3d


# =====================================================================
# Public API
# =====================================================================

def mxfp8_flash_attention_v2(q, k, v, causal=False, sm_scale=None, out_fp8=False):
    """MXfp8 Flash Attention v2.

    Args:
        q, k, v: [B, S, H, D] bfloat16 tensors
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(D))
        out_fp8: if True, return MXFP8 tuple for TE integration

    Returns:
        out_fp8=False: [B, S, H, D] bfloat16 (with autograd backward)
        out_fp8=True:  (fp8_data [B*S, H*D], rowwise_scale [B*S, H*D//32],
                        colwise_scale [H*D, B*S//32])
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    if out_fp8:
        # Forward-only, returns FP8 tuple for TE
        result = _run_offline_fwd(q, k, v, sm_scale, causal, out_fp8=True)
        # result = (o_bf16, lse, fp8_data, rowscale, colscale)
        return result[2], result[3], result[4]
    else:
        return MXfp8FlashAttentionV2.apply(q, k, v, causal, sm_scale)
