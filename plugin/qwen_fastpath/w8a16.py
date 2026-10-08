# Small-M W8A16 GEMM over the int8 weights vLLM's CUTLASS W8A8 kernel already holds.
#
# The checkpoint is W8A8 (per-channel int8 weights, dynamic per-token int8 activations). At decode batch sizes the
# CUTLASS int8 GEMM reaches ~1.2 TB/s on the A100. This kernel reads the same int8 weight tensor (the CUTLASS layout:
# [K, N] column-major == the checkpoint's [N, K] row-major), converts it to bf16 on the fly and multiplies with the
# *unquantized* bf16 activations: ~1.4-1.6 TB/s, and 5x lower error than W8A8 because activations are not rounded to
# int8. Large M (prefill) keeps the CUTLASS int8 tensor-core path, which is 2.6x faster there. No extra weight memory.
import torch
import triton
import triton.language as tl

SMALL_M = 16


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                  EVEN_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + rk[None, :]
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + rk[:, None]
    mask_m = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        x = tl.load(x_ptrs, mask=mask_m, other=0.0)
        if EVEN_N:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0)
        acc = tl.dot(x, w.to(tl.bfloat16), acc)
        x_ptrs += BLOCK_K * SPLIT_K
        w_ptrs += BLOCK_K * SPLIT_K
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * s[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    mask = mask_m & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(y_ptrs, acc, mask=mask, sem="relaxed")


# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages), tuned on the A100 80GB PCIe at M=3 for this model.
TUNED = {
    (34816, 5120): (16, 32, 256, 1, 4, 3),    # mlp gate_up
    (5120, 17408): (16, 32, 256, 2, 4, 3),    # mlp down
    (16384, 5120): (16, 32, 128, 1, 4, 4),    # gdn in_proj_qkvz
    (5120, 6144): (16, 64, 128, 4, 4, 3),     # gdn out_proj / attn o_proj
    (14336, 5120): (16, 32, 128, 4, 4, 4),    # attn qkv (+gate)
    (248320, 5120): (16, 64, 256, 1, 8, 3),   # lm_head-sized
}


def pick_config(N, K):
    cfg = TUNED.get((N, K))
    if cfg is not None:
        return cfg
    sk = 1 if N >= 8192 else (4 if N >= 1024 else 8)
    for bk in (256, 128, 64):
        while sk > 1 and K % (bk * sk):
            sk //= 2
        if K % (bk * sk) == 0:
            return (16, 32, bk, sk, 4, 3)
    return None


def w8a16_mm(x, w_t, w_scale, cfg, out_dtype):
    """x [M,K] bf16 (row stride arbitrary, unit column stride); w_t [K,N] int8 with stride (1, K); w_scale [N] fp32."""
    M, K = x.shape
    N = w_t.shape[1]
    BM, BN, BK, SK, nw, ns = cfg
    if SK == 1:
        y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    else:
        y = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    _w8a16_kernel[(triton.cdiv(N, BN), SK)](
        x, w_t, w_scale, y, M, N, K, x.stride(0), w_t.stride(1), y.stride(0),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK, EVEN_N=(N % BN == 0), num_warps=nw, num_stages=ns)
    return y if SK == 1 else y.to(out_dtype)
