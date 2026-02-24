"""
Gluon MXfp8 Flash Attention for SM120 (RTX 5090)

Uses tcgen05_mma_scaled for hardware-accelerated MXfp8 matrix multiplication.
Explicit control over shared memory, tensor memory (TMEM), TMA, and mbarriers.

Pre-quantizes Q,K,V on host; quantizes P in-kernel.
"""

import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    fence_async_shared,
    tcgen05_mma_scaled,
    tcgen05_commit,
    mbarrier,
    tma,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
VEC_SIZE = 32  # MXfp8 group size (E8M0 scales)


# =====================================================================
# Host-Side MXfp8 Quantization
# =====================================================================

def quantize_mxfp8(x):
    """Quantize tensor x along last dim to MXfp8 (E4M3 + E8M0 scales).

    Args:
        x: [..., K] tensor (float32 or bfloat16). K must be multiple of 32.

    Returns:
        fp8: [..., K] float8_e4m3fn
        scales: [..., K // 32] uint8 (E8M0 biased exponent)
    """
    assert x.shape[-1] % VEC_SIZE == 0, f"Last dim {x.shape[-1]} must be multiple of {VEC_SIZE}"
    orig_shape = x.shape
    K = orig_shape[-1]

    # Reshape to [..., K//32, 32]
    x_f32 = x.float().reshape(-1, K // VEC_SIZE, VEC_SIZE)

    # Per-group absolute max
    amax = x_f32.abs().amax(dim=-1)  # [..., K//32]

    # E8M0 biased exponent: floor(log2(amax)) + 127
    log2_amax = torch.floor(torch.log2(amax + 1e-12))
    biased = (log2_amax + 127.0).clamp(0.0, 254.0)

    # Power-of-2 scale for division
    scale = torch.exp2(log2_amax)  # [..., K//32]

    # Divide and clamp to E4M3 range [-448, 448]
    scaled = x_f32 / scale.unsqueeze(-1)
    scaled = scaled.clamp(-448.0, 448.0)

    # Convert
    fp8 = scaled.reshape(orig_shape).to(torch.float8_e4m3fn)
    scales = biased.to(torch.uint8).reshape(*orig_shape[:-1], K // VEC_SIZE)

    return fp8, scales


# =====================================================================
# In-Kernel MXfp8 Quantization Helper
# =====================================================================

@gluon.jit
def _quantize_p_mxfp8(p, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    """Quantize P [BLOCK_M, BLOCK_N] float32 registers to MXfp8.

    Groups of 32 along BLOCK_N (contraction dim for PV).
    Returns: p_fp8 [BLOCK_M, BLOCK_N] float8, p_scale [BLOCK_M, BLOCK_N//32] uint8
    """
    VEC: gl.constexpr = 32
    NGRP: gl.constexpr = BLOCK_M * (BLOCK_N // VEC)

    # Reshape to [NGRP, 32] — each row is one scale group
    p_flat = p.reshape(NGRP, VEC)

    # Per-group max (P is non-negative after softmax)
    amax = gl.max(p_flat, axis=1)  # [NGRP]

    # E8M0 biased exponent
    log2a = gl.floor(gl.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = gl.minimum(gl.maximum(biased, 0.0), 254.0)

    # Power-of-2 scale
    scale = gl.exp2(log2a)  # [NGRP]

    # Scale elements and clamp
    p_scaled = p_flat / scale[:, None]  # [NGRP, 32] / [NGRP, 1]
    p_scaled = gl.minimum(gl.maximum(p_scaled, -448.0), 448.0)

    # Convert
    p_fp8 = p_scaled.reshape(BLOCK_M, BLOCK_N).to(gl.float8e4nv)
    p_scale_uint8 = biased.to(gl.uint8).reshape(BLOCK_M, BLOCK_N // VEC)

    return p_fp8, p_scale_uint8


# =====================================================================
# Forward Kernel
# =====================================================================

@gluon.jit
def _gluon_fwd(
    q_desc, q_scale_ptr, q_scale_stride0, q_scale_stride1,
    k_desc, k_scale_ptr, k_scale_stride0, k_scale_stride1,
    vt_desc, vt_scale_ptr, vt_scale_stride0, vt_scale_stride1,
    o_desc, lse_ptr,
    sm_scale,
    SEQ_LEN,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CAUSAL: gl.constexpr,
):
    VEC: gl.constexpr = 32

    pid_m = gl.program_id(0)
    bhid = gl.program_id(1)

    # Global offsets for this batch-head
    off_q = bhid * SEQ_LEN + pid_m * BLOCK_M   # row offset in [B*H*S, D]
    off_vt = bhid * HEAD_DIM                     # row offset in [B*H*D, S]

    # --- Shared memory allocations ---
    q_smem = gl.allocate_shared_memory(q_desc.dtype, q_desc.block_type.shape, q_desc.layout)
    k_smem = gl.allocate_shared_memory(k_desc.dtype, k_desc.block_type.shape, k_desc.layout)
    vt_smem = gl.allocate_shared_memory(vt_desc.dtype, vt_desc.block_type.shape, vt_desc.layout)

    # P shared memory for PV matmul (written in-kernel after softmax)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], gl.float8e4nv)
    p_smem = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, BLOCK_N], p_smem_layout)

    # --- Tensor memory allocations ---
    # QK^T accumulator [BLOCK_M, BLOCK_N]
    qk_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_qk = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], qk_tmem_layout)

    # PV accumulator [BLOCK_M, HEAD_DIM]
    pv_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, HEAD_DIM], col_stride=1)
    acc_pv = allocate_tensor_memory(gl.float32, [BLOCK_M, HEAD_DIM], pv_tmem_layout)

    # Scale TMEMs
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    q_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M, HEAD_DIM // VEC], scale_layout)
    k_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_N, HEAD_DIM // VEC], scale_layout)
    p_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M, BLOCK_N // VEC], scale_layout)
    v_scale_tmem = allocate_tensor_memory(gl.uint8, [HEAD_DIM, BLOCK_N // VEC], scale_layout)

    # --- Barriers ---
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    bar_phase = 0
    mma_phase = 0

    # --- Register layouts ---
    qk_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M, BLOCK_N), qk_tmem_layout, gl.num_warps())
    pv_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M, HEAD_DIM), pv_tmem_layout, gl.num_warps())
    q_scale_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    k_scale_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_N, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    p_scale_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M, BLOCK_N // VEC], scale_layout, gl.num_warps())
    v_scale_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [HEAD_DIM, BLOCK_N // VEC], scale_layout, gl.num_warps())

    # 2D layout for scale pointer loads
    coalesced_2d: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0])

    # --- Load Q via TMA ---
    mbarrier.expect(bar, q_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [off_q, 0], bar, q_smem)
    mbarrier.wait(bar, bar_phase)
    bar_phase ^= 1

    # --- Load Q scales to TMEM ---
    q_sc_m = (bhid * SEQ_LEN + pid_m * BLOCK_M +
              gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d)))
    q_sc_d = gl.arange(0, HEAD_DIM // VEC, layout=gl.SliceLayout(0, coalesced_2d))
    q_scale = gl.load(q_scale_ptr + q_sc_m[:, None] * q_scale_stride0 +
                       q_sc_d[None, :] * q_scale_stride1)
    q_scale = gl.convert_layout(q_scale, q_scale_reg)
    q_scale_tmem.store(q_scale)

    # --- Initialize acc_pv to zero ---
    pv_zero = gl.full([BLOCK_M, HEAD_DIM], 0.0, dtype=gl.float32, layout=pv_reg_layout)
    acc_pv.store(pv_zero)

    # --- Online softmax state (registers) ---
    # Use SliceLayout(1, qk_reg_layout) to match the layout produced by gl.max(qk, axis=1)
    qk_1d_layout: gl.constexpr = gl.SliceLayout(1, qk_reg_layout)
    m_i = gl.full([BLOCK_M], float('-inf'), dtype=gl.float32, layout=qk_1d_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=qk_1d_layout)

    # qk_scale = sm_scale / ln(2)
    qk_scale = sm_scale * 1.44269504

    # --- Main loop over K/V blocks ---
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        # Load K via TMA
        k_off = bhid * SEQ_LEN + start_n
        mbarrier.expect(bar, k_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(k_desc, [k_off, 0], bar, k_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # Load K scales to TMEM
        k_sc_n = (bhid * SEQ_LEN + start_n +
                  gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, coalesced_2d)))
        k_scale = gl.load(k_scale_ptr + k_sc_n[:, None] * k_scale_stride0 +
                           q_sc_d[None, :] * k_scale_stride1)
        k_scale = gl.convert_layout(k_scale, k_scale_reg)
        k_scale_tmem.store(k_scale)

        # --- QK^T MMA ---
        # q_smem[M,HD] @ k_smem.T[HD,BN] → acc_qk[M,BN]
        tcgen05_mma_scaled(q_smem, k_smem.permute((1, 0)), acc_qk,
                           q_scale_tmem, k_scale_tmem, "e4m3", "e4m3",
                           use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        # Load QK^T result from TMEM
        qk = acc_qk.load(qk_reg_layout)  # [BLOCK_M, BLOCK_N] float32

        # Apply qk scale
        qk = qk * qk_scale

        # Causal mask
        if CAUSAL:
            offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d))
            offs_n = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, coalesced_2d))
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = gl.where(mask, qk, float('-inf'))

        # Online softmax
        qk_max = gl.max(qk, axis=1)  # [BLOCK_M]
        m_ij = gl.maximum(m_i, qk_max)
        p = gl.exp2(qk - m_ij[:, None])
        alpha = gl.exp2(m_i - m_ij)
        l_i = l_i * alpha + gl.sum(p, axis=1)

        # Rescale acc_pv by alpha
        pv_old = acc_pv.load(pv_reg_layout)
        pv_old = pv_old * alpha[:, None]
        acc_pv.store(pv_old)

        # Quantize P to MXfp8 in registers
        p_fp8, p_scale = _quantize_p_mxfp8(p, BLOCK_M, BLOCK_N)

        # Write P fp8 to shared memory
        p_smem.store(p_fp8)

        # Write P scale to TMEM
        p_scale = gl.convert_layout(p_scale, p_scale_reg)
        p_scale_tmem.store(p_scale)

        # Load V^T via TMA: vt_smem[HD, BN]
        mbarrier.expect(bar, vt_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(vt_desc, [off_vt, start_n], bar, vt_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # Load V scales to TMEM
        vt_sc_d = (bhid * HEAD_DIM +
                   gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, coalesced_2d)))
        vt_sc_s = (start_n // VEC +
                   gl.arange(0, BLOCK_N // VEC, layout=gl.SliceLayout(0, coalesced_2d)))
        vt_scale = gl.load(vt_scale_ptr + vt_sc_d[:, None] * vt_scale_stride0 +
                            vt_sc_s[None, :] * vt_scale_stride1)
        vt_scale = gl.convert_layout(vt_scale, v_scale_reg)
        v_scale_tmem.store(vt_scale)

        # --- PV MMA ---
        # p_smem[M,BN] @ vt_smem.T[BN,HD] → acc_pv[M,HD]
        fence_async_shared()  # fence reg→smem store before async MMA read
        tcgen05_mma_scaled(p_smem, vt_smem.permute((1, 0)), acc_pv,
                           p_scale_tmem, v_scale_tmem, "e4m3", "e4m3",
                           use_acc=True)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        m_i = m_ij

    # --- Epilogue ---
    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    # Load final PV accumulator, normalize by l_i
    o_vals = acc_pv.load(pv_reg_layout)
    o_vals = o_vals / l_i[:, None]
    o_bf16 = o_vals.to(gl.bfloat16)

    # Store O via TMA
    o_smem = gl.allocate_shared_memory(o_desc.dtype, o_desc.block_type.shape, o_desc.layout)
    o_smem.store(o_bf16)
    fence_async_shared()
    tma.async_copy_shared_to_global(o_desc, [off_q, 0], o_smem)
    tma.store_wait(0)

    # Store LSE via pointer store
    lse = m_i + gl.log2(l_i)
    lse_offs = bhid * SEQ_LEN + pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=qk_1d_layout)
    gl.store(lse_ptr + lse_offs, lse)


# =====================================================================
# Backward Preprocess Kernel
# =====================================================================

@gluon.jit
def _gluon_bwd_preprocess(
    o_desc, do_desc, delta_ptr,
    SEQ_LEN,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Compute delta[i] = rowsum(O[i] * dO[i])."""
    pid_m = gl.program_id(0)
    bhid = gl.program_id(1)
    off = bhid * SEQ_LEN + pid_m * BLOCK_M

    # Load O and dO via TMA
    o_smem = gl.allocate_shared_memory(o_desc.dtype, o_desc.block_type.shape, o_desc.layout)
    do_smem = gl.allocate_shared_memory(do_desc.dtype, do_desc.block_type.shape, do_desc.layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # Load both O and dO
    mbarrier.expect(bar, o_desc.block_type.nbytes + do_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(o_desc, [off, 0], bar, o_smem)
    tma.async_copy_global_to_shared(do_desc, [off, 0], bar, do_smem)
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)

    # Load to registers
    layout_2d: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    o_vals = o_smem.load(layout_2d).to(gl.float32)
    do_vals = do_smem.load(layout_2d).to(gl.float32)

    # Rowsum of element-wise product
    delta = gl.sum(o_vals * do_vals, axis=1)  # [BLOCK_M]

    # Store delta
    blocked_1d: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    delta_offs = bhid * SEQ_LEN + pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=blocked_1d)
    gl.store(delta_ptr + delta_offs, delta)


# =====================================================================
# Backward dK/dV Kernel
# =====================================================================

@gluon.jit
def _gluon_bwd_dkdv(
    q_desc, q_scale_ptr, q_scale_stride0, q_scale_stride1,
    k_desc, k_scale_ptr, k_scale_stride0, k_scale_stride1,
    v_desc, v_scale_ptr, v_scale_stride0, v_scale_stride1,
    do_desc, do_scale_ptr, do_scale_stride0, do_scale_stride1,
    dot_desc, dot_scale_ptr, dot_scale_stride0, dot_scale_stride1,
    dk_desc, dv_desc,
    lse_ptr, delta_ptr,
    sm_scale,
    SEQ_LEN,
    HEAD_DIM: gl.constexpr,
    BLOCK_N1: gl.constexpr,
    BLOCK_M1: gl.constexpr,
    CAUSAL: gl.constexpr,
):
    """Backward dK/dV kernel. For a fixed K,V block [BLOCK_N1], iterates over Q blocks [BLOCK_M1]."""
    VEC: gl.constexpr = 32
    pid = gl.program_id(0)
    bhid = gl.program_id(1)
    start_n = pid * BLOCK_N1

    # --- Shared memory ---
    k_smem = gl.allocate_shared_memory(k_desc.dtype, k_desc.block_type.shape, k_desc.layout)
    v_smem = gl.allocate_shared_memory(v_desc.dtype, v_desc.block_type.shape, v_desc.layout)
    qt_smem = gl.allocate_shared_memory(q_desc.dtype, q_desc.block_type.shape, q_desc.layout)
    do_smem = gl.allocate_shared_memory(do_desc.dtype, do_desc.block_type.shape, do_desc.layout)
    dot_smem = gl.allocate_shared_memory(dot_desc.dtype, dot_desc.block_type.shape, dot_desc.layout)

    # P^T smem for dV matmul
    pt_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_N1, BLOCK_M1], gl.float8e4nv)
    pt_smem = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_N1, BLOCK_M1], pt_smem_layout)

    # dsT smem for dK matmul
    dst_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_N1, BLOCK_M1], gl.float8e4nv)
    dst_smem = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_N1, BLOCK_M1], dst_smem_layout)

    # --- Tensor memory ---
    # dV accumulator [BLOCK_N1, HEAD_DIM]
    dv_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_N1, HEAD_DIM], col_stride=1)
    acc_dv = allocate_tensor_memory(gl.float32, [BLOCK_N1, HEAD_DIM], dv_tmem_layout)

    # dK accumulator [BLOCK_N1, HEAD_DIM]
    dk_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_N1, HEAD_DIM], col_stride=1)
    acc_dk = allocate_tensor_memory(gl.float32, [BLOCK_N1, HEAD_DIM], dk_tmem_layout)

    # QK^T accumulator [BLOCK_N1, BLOCK_M1]
    qkt_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_N1, BLOCK_M1], col_stride=1)
    acc_qkt = allocate_tensor_memory(gl.float32, [BLOCK_N1, BLOCK_M1], qkt_tmem_layout)

    # dpT accumulator [BLOCK_N1, BLOCK_M1]
    dpt_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_N1, BLOCK_M1], col_stride=1)
    acc_dpt = allocate_tensor_memory(gl.float32, [BLOCK_N1, BLOCK_M1], dpt_tmem_layout)

    # Scale TMEMs
    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    k_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_N1, HEAD_DIM // VEC], scale_layout)
    v_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_N1, HEAD_DIM // VEC], scale_layout)
    q_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M1, HEAD_DIM // VEC], scale_layout)
    do_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M1, HEAD_DIM // VEC], scale_layout)
    dot_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M1, HEAD_DIM // VEC], scale_layout)
    pt_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_N1, BLOCK_M1 // VEC], scale_layout)
    dst_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_N1, BLOCK_M1 // VEC], scale_layout)

    # --- Barriers ---
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    bar_phase = 0
    mma_phase = 0

    # Register layouts
    qkt_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_N1, BLOCK_M1), qkt_tmem_layout, gl.num_warps())
    dpt_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_N1, BLOCK_M1), dpt_tmem_layout, gl.num_warps())
    dv_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_N1, HEAD_DIM), dv_tmem_layout, gl.num_warps())
    dk_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_N1, HEAD_DIM), dk_tmem_layout, gl.num_warps())

    k_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_N1, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    v_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_N1, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    q_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M1, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    do_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M1, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    dot_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M1, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    pt_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_N1, BLOCK_M1 // VEC], scale_layout, gl.num_warps())
    dst_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_N1, BLOCK_M1 // VEC], scale_layout, gl.num_warps())

    coalesced_2d: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    blocked_1d: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])

    # --- Load K, V once (hoisted) ---
    k_off = bhid * SEQ_LEN + start_n
    mbarrier.expect(bar, k_desc.block_type.nbytes + v_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(k_desc, [k_off, 0], bar, k_smem)
    tma.async_copy_global_to_shared(v_desc, [k_off, 0], bar, v_smem)
    mbarrier.wait(bar, bar_phase)
    bar_phase ^= 1

    # K scales [BLOCK_N1, HD//VEC]
    k_sc_offs = (bhid * SEQ_LEN + start_n +
                 gl.arange(0, BLOCK_N1, layout=gl.SliceLayout(1, coalesced_2d)))
    hd_sc_offs = gl.arange(0, HEAD_DIM // VEC, layout=gl.SliceLayout(0, coalesced_2d))
    k_sc = gl.load(k_scale_ptr + k_sc_offs[:, None] * k_scale_stride0 +
                    hd_sc_offs[None, :] * k_scale_stride1)
    k_sc = gl.convert_layout(k_sc, k_sc_reg)
    k_scale_tmem.store(k_sc)

    # V scales [BLOCK_N1, HD//VEC]
    v_sc = gl.load(v_scale_ptr + k_sc_offs[:, None] * v_scale_stride0 +
                    hd_sc_offs[None, :] * v_scale_stride1)
    v_sc = gl.convert_layout(v_sc, v_sc_reg)
    v_scale_tmem.store(v_sc)

    # Initialize dV, dK accumulators to zero
    dv_zero = gl.full([BLOCK_N1, HEAD_DIM], 0.0, dtype=gl.float32, layout=dv_reg)
    acc_dv.store(dv_zero)
    dk_zero = gl.full([BLOCK_N1, HEAD_DIM], 0.0, dtype=gl.float32, layout=dk_reg)
    acc_dk.store(dk_zero)

    # Determine loop range
    if CAUSAL:
        start_m = start_n
    else:
        start_m = 0

    for curr_m in range(start_m, SEQ_LEN, BLOCK_M1):
        q_off = bhid * SEQ_LEN + curr_m

        # Load Q^T [HD, M1] via TMA (Q is [M1, HD], qT desc is [HD, M1])
        mbarrier.expect(bar, q_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(q_desc, [bhid * HEAD_DIM, curr_m], bar, qt_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # Q scales [M1, HD//VEC] — note: q_desc for bwd is transposed, scales follow original Q
        q_sc_m = (bhid * SEQ_LEN + curr_m +
                  gl.arange(0, BLOCK_M1, layout=gl.SliceLayout(1, coalesced_2d)))
        q_sc = gl.load(q_scale_ptr + q_sc_m[:, None] * q_scale_stride0 +
                        hd_sc_offs[None, :] * q_scale_stride1)
        q_sc = gl.convert_layout(q_sc, q_sc_reg)
        q_scale_tmem.store(q_sc)

        # Dot 1: qkT = k @ qT  [N1,HD]@[HD,M1] → [N1,M1]
        tcgen05_mma_scaled(k_smem, qt_smem.permute((1, 0)), acc_qkt,
                           k_scale_tmem, q_scale_tmem, "e4m3", "e4m3",
                           use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        # Load LSE and Delta for this Q block
        lse_offs = bhid * SEQ_LEN + curr_m + gl.arange(0, BLOCK_M1, layout=blocked_1d)
        m_vals = gl.load(lse_ptr + lse_offs)  # [M1]
        Di = gl.load(delta_ptr + lse_offs)     # [M1]

        # Compute pT = exp2(qkT - m)
        qkt = acc_qkt.load(qkt_reg)
        pT = gl.exp2(qkt - m_vals[None, :])  # [N1, M1]

        if CAUSAL:
            offs_m = curr_m + gl.arange(0, BLOCK_M1, layout=gl.SliceLayout(0, coalesced_2d))
            offs_n = start_n + gl.arange(0, BLOCK_N1, layout=gl.SliceLayout(1, coalesced_2d))
            mask = offs_m[None, :] <= offs_n[:, None]  # transposed mask
            pT = gl.where(mask, 0.0, pT)

        # Load dO [M1, HD]
        mbarrier.expect(bar, do_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(do_desc, [q_off, 0], bar, do_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # dO scales
        do_sc = gl.load(do_scale_ptr + q_sc_m[:, None] * do_scale_stride0 +
                         hd_sc_offs[None, :] * do_scale_stride1)
        do_sc = gl.convert_layout(do_sc, do_sc_reg)
        do_scale_tmem.store(do_sc)

        # Quantize pT to MXfp8 [N1, M1]
        pT_fp8, pT_scale = _quantize_p_mxfp8(pT, BLOCK_N1, BLOCK_M1)
        pt_smem.store(pT_fp8)
        pT_scale = gl.convert_layout(pT_scale, pt_sc_reg)
        pt_scale_tmem.store(pT_scale)

        # Dot 2: dV += pT @ dO  [N1,M1]@[M1,HD] → [N1,HD]
        fence_async_shared()
        tcgen05_mma_scaled(pt_smem, do_smem.permute((1, 0)), acc_dv,
                           pt_scale_tmem, do_scale_tmem, "e4m3", "e4m3",
                           use_acc=True)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        # Load dO^T via TMA for dot 3 (need dO transposed)
        mbarrier.expect(bar, dot_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(dot_desc, [bhid * HEAD_DIM, curr_m], bar, dot_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # dO^T scales
        dot_sc = gl.load(dot_scale_ptr + q_sc_m[:, None] * dot_scale_stride0 +
                          hd_sc_offs[None, :] * dot_scale_stride1)
        dot_sc = gl.convert_layout(dot_sc, dot_sc_reg)
        dot_scale_tmem.store(dot_sc)

        # Dot 3: dpT = v @ dOT  [N1,HD]@[HD,M1] → [N1,M1]
        tcgen05_mma_scaled(v_smem, dot_smem.permute((1, 0)), acc_dpt,
                           v_scale_tmem, dot_scale_tmem, "e4m3", "e4m3",
                           use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        # dsT = pT * (dpT - Di)
        dpT = acc_dpt.load(dpt_reg)
        dsT = pT * (dpT - Di[None, :])

        # Quantize dsT to MXfp8 [N1, M1]
        dsT_fp8, dsT_scale = _quantize_p_mxfp8(gl.abs(dsT) + dsT, BLOCK_N1, BLOCK_M1)
        # Actually re-quantize dsT directly (it can be negative)
        dsT_fp8, dsT_scale = _quantize_dsT_mxfp8(dsT, BLOCK_N1, BLOCK_M1)
        dst_smem.store(dsT_fp8)
        dsT_scale = gl.convert_layout(dsT_scale, dst_sc_reg)
        dst_scale_tmem.store(dsT_scale)

        # Dot 4: dK += dsT @ Q  [N1,M1]@[M1,HD] → [N1,HD]
        # Need Q as [M1, HD] in smem — reuse do_smem or load Q
        # For now, reuse qt_smem which is [HD, M1] — we need [M1, HD]
        # This requires Q in non-transposed form. Load Q normally.
        mbarrier.expect(bar, do_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(do_desc, [q_off, 0], bar, do_smem)  # reuse do_smem for Q
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        fence_async_shared()
        tcgen05_mma_scaled(dst_smem, do_smem.permute((1, 0)), acc_dk,
                           dst_scale_tmem, q_scale_tmem, "e4m3", "e4m3",
                           use_acc=True)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

    # --- Store dK, dV ---
    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    dv_vals = acc_dv.load(dv_reg)
    dv_bf16 = dv_vals.to(gl.bfloat16)
    dv_out_smem = gl.allocate_shared_memory(dv_desc.dtype, dv_desc.block_type.shape, dv_desc.layout)
    dv_out_smem.store(dv_bf16)
    fence_async_shared()
    tma.async_copy_shared_to_global(dv_desc, [k_off, 0], dv_out_smem)

    dk_vals = acc_dk.load(dk_reg)
    dk_vals = dk_vals * sm_scale
    dk_bf16 = dk_vals.to(gl.bfloat16)
    dk_out_smem = gl.allocate_shared_memory(dk_desc.dtype, dk_desc.block_type.shape, dk_desc.layout)
    dk_out_smem.store(dk_bf16)
    fence_async_shared()
    tma.async_copy_shared_to_global(dk_desc, [k_off, 0], dk_out_smem)
    tma.store_wait(0)


@gluon.jit
def _quantize_dsT_mxfp8(ds, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    """Quantize dsT (which can be negative) to MXfp8."""
    VEC: gl.constexpr = 32
    NGRP: gl.constexpr = BLOCK_M * (BLOCK_N // VEC)

    ds_flat = ds.reshape(NGRP, VEC)
    amax = gl.max(gl.abs(ds_flat), axis=1)  # [NGRP]

    log2a = gl.floor(gl.log2(amax + 1e-12))
    biased = log2a + 127.0
    biased = gl.minimum(gl.maximum(biased, 0.0), 254.0)
    scale = gl.exp2(log2a)

    ds_scaled = ds_flat / scale[:, None]
    ds_scaled = gl.minimum(gl.maximum(ds_scaled, -448.0), 448.0)

    ds_fp8 = ds_scaled.reshape(BLOCK_M, BLOCK_N).to(gl.float8e4nv)
    ds_scale = biased.to(gl.uint8).reshape(BLOCK_M, BLOCK_N // VEC)

    return ds_fp8, ds_scale


# =====================================================================
# Backward dQ Kernel
# =====================================================================

@gluon.jit
def _gluon_bwd_dq(
    q_desc, q_scale_ptr, q_scale_stride0, q_scale_stride1,
    do_desc, do_scale_ptr, do_scale_stride0, do_scale_stride1,
    kt_desc, kt_scale_ptr, kt_scale_stride0, kt_scale_stride1,
    vt_desc, vt_scale_ptr, vt_scale_stride0, vt_scale_stride1,
    k_desc, k_scale_bwd_ptr, k_scale_bwd_stride0, k_scale_bwd_stride1,
    dq_desc,
    lse_ptr, delta_ptr,
    sm_scale,
    SEQ_LEN,
    HEAD_DIM: gl.constexpr,
    BLOCK_M2: gl.constexpr,
    BLOCK_N2: gl.constexpr,
    CAUSAL: gl.constexpr,
):
    """Backward dQ kernel. For a fixed Q block [BLOCK_M2], iterates over K,V blocks [BLOCK_N2]."""
    VEC: gl.constexpr = 32
    LN2: gl.constexpr = 0.6931471824645996

    pid = gl.program_id(0)
    bhid = gl.program_id(1)
    start_m = pid * BLOCK_M2

    # --- Shared memory ---
    q_smem = gl.allocate_shared_memory(q_desc.dtype, q_desc.block_type.shape, q_desc.layout)
    do_smem = gl.allocate_shared_memory(do_desc.dtype, do_desc.block_type.shape, do_desc.layout)
    kt_smem = gl.allocate_shared_memory(kt_desc.dtype, kt_desc.block_type.shape, kt_desc.layout)
    vt_smem = gl.allocate_shared_memory(vt_desc.dtype, vt_desc.block_type.shape, vt_desc.layout)

    # ds smem for dQ matmul
    ds_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M2, BLOCK_N2], gl.float8e4nv)
    ds_smem = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M2, BLOCK_N2], ds_smem_layout)

    # K smem for dot 3 (non-transposed)
    k_smem = gl.allocate_shared_memory(k_desc.dtype, k_desc.block_type.shape, k_desc.layout)

    # --- Tensor memory ---
    qk_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M2, BLOCK_N2], col_stride=1)
    acc_qk = allocate_tensor_memory(gl.float32, [BLOCK_M2, BLOCK_N2], qk_tmem_layout)

    dp_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M2, BLOCK_N2], col_stride=1)
    acc_dp = allocate_tensor_memory(gl.float32, [BLOCK_M2, BLOCK_N2], dp_tmem_layout)

    dq_tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M2, HEAD_DIM], col_stride=1)
    acc_dq = allocate_tensor_memory(gl.float32, [BLOCK_M2, HEAD_DIM], dq_tmem_layout)

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    q_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M2, HEAD_DIM // VEC], scale_layout)
    do_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M2, HEAD_DIM // VEC], scale_layout)
    kt_scale_tmem = allocate_tensor_memory(gl.uint8, [HEAD_DIM, BLOCK_N2 // VEC], scale_layout)
    vt_scale_tmem = allocate_tensor_memory(gl.uint8, [HEAD_DIM, BLOCK_N2 // VEC], scale_layout)
    ds_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_M2, BLOCK_N2 // VEC], scale_layout)
    k_scale_tmem = allocate_tensor_memory(gl.uint8, [BLOCK_N2, HEAD_DIM // VEC], scale_layout)

    # Barriers
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.init(mma_bar, count=1)
    bar_phase = 0
    mma_phase = 0

    # Register layouts
    qk_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M2, BLOCK_N2), qk_tmem_layout, gl.num_warps())
    dp_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M2, BLOCK_N2), dp_tmem_layout, gl.num_warps())
    dq_reg: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M2, HEAD_DIM), dq_tmem_layout, gl.num_warps())
    q_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M2, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    do_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M2, HEAD_DIM // VEC], scale_layout, gl.num_warps())
    kt_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [HEAD_DIM, BLOCK_N2 // VEC], scale_layout, gl.num_warps())
    vt_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [HEAD_DIM, BLOCK_N2 // VEC], scale_layout, gl.num_warps())
    ds_sc_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_M2, BLOCK_N2 // VEC], scale_layout, gl.num_warps())
    k_sc_bwd_reg: gl.constexpr = get_tmem_reg_layout(
        gl.uint8, [BLOCK_N2, HEAD_DIM // VEC], scale_layout, gl.num_warps())

    coalesced_2d: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
    blocked_1d: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])

    # Load Q, dO (hoisted)
    q_off = bhid * SEQ_LEN + start_m
    mbarrier.expect(bar, q_desc.block_type.nbytes + do_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [q_off, 0], bar, q_smem)
    tma.async_copy_global_to_shared(do_desc, [q_off, 0], bar, do_smem)
    mbarrier.wait(bar, bar_phase)
    bar_phase ^= 1

    # Q and dO scales
    hd_sc = gl.arange(0, HEAD_DIM // VEC, layout=gl.SliceLayout(0, coalesced_2d))
    q_sc_m = (bhid * SEQ_LEN + start_m +
              gl.arange(0, BLOCK_M2, layout=gl.SliceLayout(1, coalesced_2d)))
    q_sc = gl.load(q_scale_ptr + q_sc_m[:, None] * q_scale_stride0 +
                    hd_sc[None, :] * q_scale_stride1)
    q_sc = gl.convert_layout(q_sc, q_sc_reg)
    q_scale_tmem.store(q_sc)

    do_sc = gl.load(do_scale_ptr + q_sc_m[:, None] * do_scale_stride0 +
                     hd_sc[None, :] * do_scale_stride1)
    do_sc = gl.convert_layout(do_sc, do_sc_reg)
    do_scale_tmem.store(do_sc)

    # Load LSE and Delta
    lse_offs = bhid * SEQ_LEN + start_m + gl.arange(0, BLOCK_M2, layout=blocked_1d)
    m_vals = gl.load(lse_ptr + lse_offs)
    Di = gl.load(delta_ptr + lse_offs)

    # Initialize dQ accumulator
    dq_zero = gl.full([BLOCK_M2, HEAD_DIM], 0.0, dtype=gl.float32, layout=dq_reg)
    acc_dq.store(dq_zero)

    # Loop over K,V blocks
    if CAUSAL:
        end_n = start_m + BLOCK_M2
    else:
        end_n = SEQ_LEN

    for start_n in range(0, end_n, BLOCK_N2):
        # Load kT [HD, N2] and vT [HD, N2]
        mbarrier.expect(bar, kt_desc.block_type.nbytes + vt_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(kt_desc, [bhid * HEAD_DIM, start_n], bar, kt_smem)
        tma.async_copy_global_to_shared(vt_desc, [bhid * HEAD_DIM, start_n], bar, vt_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # kT, vT scales [HD, N2//VEC]
        kt_sc_d = (bhid * HEAD_DIM +
                   gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, coalesced_2d)))
        kt_sc_s = (start_n // VEC +
                   gl.arange(0, BLOCK_N2 // VEC, layout=gl.SliceLayout(0, coalesced_2d)))
        kt_sc = gl.load(kt_scale_ptr + kt_sc_d[:, None] * kt_scale_stride0 +
                         kt_sc_s[None, :] * kt_scale_stride1)
        kt_sc = gl.convert_layout(kt_sc, kt_sc_reg)
        kt_scale_tmem.store(kt_sc)

        vt_sc = gl.load(vt_scale_ptr + kt_sc_d[:, None] * vt_scale_stride0 +
                         kt_sc_s[None, :] * vt_scale_stride1)
        vt_sc = gl.convert_layout(vt_sc, vt_sc_reg)
        vt_scale_tmem.store(vt_sc)

        # Dot 1: qk = q @ kT  [M2,HD]@[HD,N2] → [M2,N2]
        tcgen05_mma_scaled(q_smem, kt_smem.permute((1, 0)), acc_qk,
                           q_scale_tmem, kt_scale_tmem, "e4m3", "e4m3",
                           use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        qk = acc_qk.load(qk_reg)
        p = gl.exp2(qk - m_vals[:, None])

        if CAUSAL:
            offs_m2 = start_m + gl.arange(0, BLOCK_M2, layout=gl.SliceLayout(1, coalesced_2d))
            offs_n2 = start_n + gl.arange(0, BLOCK_N2, layout=gl.SliceLayout(0, coalesced_2d))
            mask2 = offs_m2[:, None] >= offs_n2[None, :]
            p = gl.where(mask2, p, 0.0)

        # Dot 2: dp = do @ vT  [M2,HD]@[HD,N2] → [M2,N2]
        tcgen05_mma_scaled(do_smem, vt_smem.permute((1, 0)), acc_dp,
                           do_scale_tmem, vt_scale_tmem, "e4m3", "e4m3",
                           use_acc=False)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

        dp = acc_dp.load(dp_reg)
        ds = p * (dp - Di[:, None])

        # Quantize ds [M2, N2]
        ds_fp8, ds_sc = _quantize_dsT_mxfp8(ds, BLOCK_M2, BLOCK_N2)
        ds_smem.store(ds_fp8)
        ds_sc = gl.convert_layout(ds_sc, ds_sc_reg)
        ds_scale_tmem.store(ds_sc)

        # Load K [N2, HD] for dot 3 (non-transposed)
        k_off_n = bhid * SEQ_LEN + start_n
        mbarrier.expect(bar, k_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(k_desc, [k_off_n, 0], bar, k_smem)
        mbarrier.wait(bar, bar_phase)
        bar_phase ^= 1

        # K scales for dot 3 [N2, HD//VEC]
        k_sc_n = (bhid * SEQ_LEN + start_n +
                  gl.arange(0, BLOCK_N2, layout=gl.SliceLayout(1, coalesced_2d)))
        k_bwd_sc = gl.load(k_scale_bwd_ptr + k_sc_n[:, None] * k_scale_bwd_stride0 +
                            hd_sc[None, :] * k_scale_bwd_stride1)
        k_bwd_sc = gl.convert_layout(k_bwd_sc, k_sc_bwd_reg)
        k_scale_tmem.store(k_bwd_sc)

        # Dot 3: dQ += ds @ K  [M2,N2]@[N2,HD] → [M2,HD]
        fence_async_shared()
        tcgen05_mma_scaled(ds_smem, k_smem.permute((1, 0)), acc_dq,
                           ds_scale_tmem, k_scale_tmem, "e4m3", "e4m3",
                           use_acc=True)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, mma_phase)
        mma_phase ^= 1

    # Store dQ
    mbarrier.invalidate(bar)
    mbarrier.invalidate(mma_bar)

    dq_vals = acc_dq.load(dq_reg)
    dq_vals = dq_vals * LN2
    dq_bf16 = dq_vals.to(gl.bfloat16)
    dq_smem = gl.allocate_shared_memory(dq_desc.dtype, dq_desc.block_type.shape, dq_desc.layout)
    dq_smem.store(dq_bf16)
    fence_async_shared()
    tma.async_copy_shared_to_global(dq_desc, [q_off, 0], dq_smem)
    tma.store_wait(0)


# =====================================================================
# TMA Descriptor Helpers
# =====================================================================

def _make_2d_desc(tensor_2d, block_shape, dtype_gl):
    """Create a 2D TMA descriptor for a contiguous 2D tensor."""
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, dtype_gl)
    return TensorDescriptor.from_tensor(tensor_2d, block_shape, layout)


def _make_bf16_desc(tensor_2d, block_shape):
    """Create a 2D TMA descriptor for bfloat16 output."""
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.bfloat16)
    return TensorDescriptor.from_tensor(tensor_2d, block_shape, layout)


# =====================================================================
# PyTorch Autograd Wrapper
# =====================================================================

class GluonMXfp8FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # q, k, v: [B, S, H, D] bfloat16
        B, S, H, D = q.shape
        assert D in {64, 128}, f"HEAD_DIM must be 64 or 128, got {D}"
        assert S % 128 == 0, f"Sequence length must be multiple of 128, got {S}"

        BLOCK_M = 128
        BLOCK_N = 64

        # Permute to [B*H, S, D] for uniform layout
        q_bhs = q.permute(0, 2, 1, 3).contiguous().reshape(B * H, S, D)
        k_bhs = k.permute(0, 2, 1, 3).contiguous().reshape(B * H, S, D)
        v_bhs = v.permute(0, 2, 1, 3).contiguous().reshape(B * H, S, D)

        # Pre-quantize Q, K along D (last dim)
        q_fp8, q_scale = quantize_mxfp8(q_bhs)  # [B*H, S, D], [B*H, S, D//32]
        k_fp8, k_scale = quantize_mxfp8(k_bhs)

        # Pre-quantize V along S (sequence dim): transpose, quantize, keep transposed
        v_t = v_bhs.transpose(-1, -2).contiguous()  # [B*H, D, S]
        v_t_fp8, v_t_scale = quantize_mxfp8(v_t)   # [B*H, D, S], [B*H, D, S//32]

        # Flatten batch-head for 2D TMA descriptors
        BH = B * H
        q_fp8_2d = q_fp8.reshape(BH * S, D).contiguous()
        k_fp8_2d = k_fp8.reshape(BH * S, D).contiguous()
        v_t_fp8_2d = v_t_fp8.reshape(BH * D, S).contiguous()

        # Output
        o_2d = torch.empty(BH * S, D, device=q.device, dtype=torch.bfloat16)
        lse = torch.empty(BH, S, device=q.device, dtype=torch.float32)

        # TMA descriptors
        q_desc = _make_2d_desc(q_fp8_2d, [BLOCK_M, D], gl.float8e4nv)
        k_desc = _make_2d_desc(k_fp8_2d, [BLOCK_N, D], gl.float8e4nv)
        vt_desc = _make_2d_desc(v_t_fp8_2d, [D, BLOCK_N], gl.float8e4nv)
        o_desc = _make_bf16_desc(o_2d, [BLOCK_M, D])

        # Scale pointers (flattened for stride-based access)
        q_scale_flat = q_scale.reshape(BH * S, D // 32).contiguous()
        k_scale_flat = k_scale.reshape(BH * S, D // 32).contiguous()
        vt_scale_flat = v_t_scale.reshape(BH * D, S // 32).contiguous()

        grid = (S // BLOCK_M, BH)

        _gluon_fwd[grid](
            q_desc, q_scale_flat, q_scale_flat.stride(0), q_scale_flat.stride(1),
            k_desc, k_scale_flat, k_scale_flat.stride(0), k_scale_flat.stride(1),
            vt_desc, vt_scale_flat, vt_scale_flat.stride(0), vt_scale_flat.stride(1),
            o_desc, lse,
            sm_scale,
            S,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            CAUSAL=causal,
            num_warps=4,
        )

        # Reshape output back to [B, S, H, D]
        o = o_2d.reshape(BH, S, D).reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = D
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, S, H, D = q.shape
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        BH = B * H

        do = do.contiguous()

        BLOCK_M1 = 32
        BLOCK_N1 = 128
        BLOCK_M2 = 128
        BLOCK_N2 = 32

        # Prepare layouts: [B*H, S, D]
        q_bhs = q.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)
        k_bhs = k.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)
        v_bhs = v.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)
        o_bhs = o.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)
        do_bhs = do.permute(0, 2, 1, 3).contiguous().reshape(BH, S, D)

        # Pre-scale K by sm_scale / ln2
        RCP_LN2 = 1.4426950408889634
        k_scaled = k_bhs * (sm_scale * RCP_LN2)

        # Quantize everything
        q_fp8, q_scale = quantize_mxfp8(q_bhs)
        k_fp8, k_scale = quantize_mxfp8(k_scaled)  # Pre-scaled K
        v_fp8, v_scale = quantize_mxfp8(v_bhs)
        do_fp8, do_scale = quantize_mxfp8(do_bhs)

        # Transposed versions for various dot products
        q_t = q_bhs.transpose(-1, -2).contiguous()   # [BH, D, S]
        do_t = do_bhs.transpose(-1, -2).contiguous()  # [BH, D, S]
        k_t = k_scaled.transpose(-1, -2).contiguous() # [BH, D, S]
        v_t = v_bhs.transpose(-1, -2).contiguous()    # [BH, D, S]

        q_t_fp8, q_t_scale = quantize_mxfp8(q_t)
        do_t_fp8, do_t_scale = quantize_mxfp8(do_t)
        k_t_fp8, k_t_scale = quantize_mxfp8(k_t)
        v_t_fp8, v_t_scale = quantize_mxfp8(v_t)

        # Also need non-scaled K for dQ dot 3
        k_orig_fp8, k_orig_scale = quantize_mxfp8(k_bhs)

        # Flatten for 2D TMA
        q_fp8_2d = q_fp8.reshape(BH * S, D).contiguous()
        k_fp8_2d = k_fp8.reshape(BH * S, D).contiguous()
        v_fp8_2d = v_fp8.reshape(BH * S, D).contiguous()
        do_fp8_2d = do_fp8.reshape(BH * S, D).contiguous()
        o_2d = o_bhs.reshape(BH * S, D).contiguous()
        do_2d_bf16 = do_bhs.reshape(BH * S, D).contiguous()

        q_t_fp8_2d = q_t_fp8.reshape(BH * D, S).contiguous()
        do_t_fp8_2d = do_t_fp8.reshape(BH * D, S).contiguous()
        k_t_fp8_2d = k_t_fp8.reshape(BH * D, S).contiguous()
        v_t_fp8_2d = v_t_fp8.reshape(BH * D, S).contiguous()

        k_orig_2d = k_orig_fp8.reshape(BH * S, D).contiguous()

        # Scale tensors
        q_scale_2d = q_scale.reshape(BH * S, D // 32).contiguous()
        k_scale_2d = k_scale.reshape(BH * S, D // 32).contiguous()
        v_scale_2d = v_scale.reshape(BH * S, D // 32).contiguous()
        do_scale_2d = do_scale.reshape(BH * S, D // 32).contiguous()
        q_t_scale_2d = q_t_scale.reshape(BH * D, S // 32).contiguous()
        do_t_scale_2d = do_t_scale.reshape(BH * D, S // 32).contiguous()
        k_t_scale_2d = k_t_scale.reshape(BH * D, S // 32).contiguous()
        v_t_scale_2d = v_t_scale.reshape(BH * D, S // 32).contiguous()
        k_orig_scale_2d = k_orig_scale.reshape(BH * S, D // 32).contiguous()

        # Output gradients
        dq_2d = torch.empty(BH * S, D, device=q.device, dtype=torch.bfloat16)
        dk_2d = torch.empty(BH * S, D, device=q.device, dtype=torch.bfloat16)
        dv_2d = torch.empty(BH * S, D, device=q.device, dtype=torch.bfloat16)
        delta = torch.empty(BH, S, device=q.device, dtype=torch.float32)

        # --- Preprocess: delta = rowsum(O * dO) ---
        PRE_BLOCK = 128
        o_desc_pre = _make_bf16_desc(o_2d, [PRE_BLOCK, D])
        do_desc_pre = _make_bf16_desc(do_2d_bf16, [PRE_BLOCK, D])

        pre_grid = (S // PRE_BLOCK, BH)
        _gluon_bwd_preprocess[pre_grid](
            o_desc_pre, do_desc_pre, delta,
            S,
            HEAD_DIM=D,
            BLOCK_M=PRE_BLOCK,
            num_warps=4,
        )

        # --- dK/dV kernel ---
        # Descriptors for dkdv
        k_dkdv_desc = _make_2d_desc(k_fp8_2d, [BLOCK_N1, D], gl.float8e4nv)
        v_dkdv_desc = _make_2d_desc(v_fp8_2d, [BLOCK_N1, D], gl.float8e4nv)
        qt_dkdv_desc = _make_2d_desc(q_t_fp8_2d, [D, BLOCK_M1], gl.float8e4nv)
        do_dkdv_desc = _make_2d_desc(do_fp8_2d, [BLOCK_M1, D], gl.float8e4nv)
        dot_dkdv_desc = _make_2d_desc(do_t_fp8_2d, [D, BLOCK_M1], gl.float8e4nv)
        dk_desc = _make_bf16_desc(dk_2d, [BLOCK_N1, D])
        dv_desc = _make_bf16_desc(dv_2d, [BLOCK_N1, D])

        dkdv_grid = (S // BLOCK_N1, BH)
        _gluon_bwd_dkdv[dkdv_grid](
            qt_dkdv_desc, q_scale_2d, q_scale_2d.stride(0), q_scale_2d.stride(1),
            k_dkdv_desc, k_scale_2d, k_scale_2d.stride(0), k_scale_2d.stride(1),
            v_dkdv_desc, v_scale_2d, v_scale_2d.stride(0), v_scale_2d.stride(1),
            do_dkdv_desc, do_scale_2d, do_scale_2d.stride(0), do_scale_2d.stride(1),
            dot_dkdv_desc, do_t_scale_2d, do_t_scale_2d.stride(0), do_t_scale_2d.stride(1),
            dk_desc, dv_desc,
            lse.reshape(BH * S), delta.reshape(BH * S),
            sm_scale,
            S,
            HEAD_DIM=D,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M1=BLOCK_M1,
            CAUSAL=causal,
            num_warps=4,
        )

        # --- dQ kernel ---
        q_dq_desc = _make_2d_desc(q_fp8_2d, [BLOCK_M2, D], gl.float8e4nv)
        do_dq_desc = _make_2d_desc(do_fp8_2d, [BLOCK_M2, D], gl.float8e4nv)
        kt_dq_desc = _make_2d_desc(k_t_fp8_2d, [D, BLOCK_N2], gl.float8e4nv)
        vt_dq_desc = _make_2d_desc(v_t_fp8_2d, [D, BLOCK_N2], gl.float8e4nv)
        k_dq_desc = _make_2d_desc(k_orig_2d, [BLOCK_N2, D], gl.float8e4nv)
        dq_desc = _make_bf16_desc(dq_2d, [BLOCK_M2, D])

        dq_grid = (S // BLOCK_M2, BH)
        _gluon_bwd_dq[dq_grid](
            q_dq_desc, q_scale_2d, q_scale_2d.stride(0), q_scale_2d.stride(1),
            do_dq_desc, do_scale_2d, do_scale_2d.stride(0), do_scale_2d.stride(1),
            kt_dq_desc, k_t_scale_2d, k_t_scale_2d.stride(0), k_t_scale_2d.stride(1),
            vt_dq_desc, v_t_scale_2d, v_t_scale_2d.stride(0), v_t_scale_2d.stride(1),
            k_dq_desc, k_orig_scale_2d, k_orig_scale_2d.stride(0), k_orig_scale_2d.stride(1),
            dq_desc,
            lse.reshape(BH * S), delta.reshape(BH * S),
            sm_scale,
            S,
            HEAD_DIM=D,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            CAUSAL=causal,
            num_warps=4,
        )

        # Reshape outputs back to [B, S, H, D]
        dq = dq_2d.reshape(BH, S, D).reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()
        dk = dk_2d.reshape(BH, S, D).reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()
        dv = dv_2d.reshape(BH, S, D).reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()

        return dq, dk, dv, None, None


def gluon_mxfp8_flash_attention(q, k, v, causal=False, sm_scale=None):
    """Gluon MXfp8 Flash Attention.

    Args:
        q, k, v: [B, S, H, D] bfloat16 tensors
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(D))

    Returns:
        output: [B, S, H, D] bfloat16
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return GluonMXfp8FlashAttention.apply(q, k, v, causal, sm_scale)
