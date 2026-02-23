# Plan: MXfp8 Flash Attention Operator for SM120

## Architecture

Single operator file `mxfp8_flash_attn.py` + test file `test_mxfp8_flash_attn.py`.

### Forward Pass (MXfp8 dot_scaled)
- Accept FP16 [B,H,N,D] inputs
- Quantize Q,K,V tiles to MXfp8 on-the-fly (E4M3 data + E8M0 per-32-element scales)
- QK^T via `tl.dot_scaled` with row-wise quantization
- PV via `tl.dot_scaled` with column-wise quantization for V
- Online softmax in FP32, output FP16
- Store log-sum-exp for backward

### Backward Pass (FP16 tl.dot)
- Standard Flash Attention v2 backward algorithm
- Pre-scale K by sm_scale/ln(2), use exp2 for log2-space softmax
- Combined dKdV + dQ kernel following the reference tutorial pattern
- FP16 tl.dot for numerical stability

### Quantization Helpers
- `_quantize_mxfp8_row`: Groups of 32 along last dim → scale [M, K//32]
- `_quantize_mxfp8_col`: Groups of 32 along first dim → scale [N, K//32] (3D reshape)

### Block Sizes
- Forward: BLOCK_M=128, BLOCK_N=64
- Backward: BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32
- HEAD_DIM: 64 or 128 (constexpr)

### Key API: tl.dot_scaled
- `dot_scaled(lhs [M,K], lhs_scale [M,K//32], "e4m3", rhs [K,N], rhs_scale [N,K//32], "e4m3")`
- Scale is uint8 E8M0 format (biased exponent, value = 2^(stored - 127))
- Output always float32

## Test Plan
- Forward: B={1,4} H={2,8} N={128,512,1024} D={64,128} causal={T,F}, atol=0.05
- Backward: Same grid (N up to 512), atol=0.1 for gradients
- Check no NaN in outputs or gradients
