# Split-KV, GQA-packed paged attention for uniform speculative-decode batches (Q = 1 + num_spec_tokens queries/request).
#
# vLLM's FA2 path for Q > 1 runs the generic varlen kernel, which never splits the KV sequence (the wrapper rejects
# num_splits > 1) and gives every *query* head its own CTA. With one request that is 24 CTAs streaming the whole
# context, each KV head read G=6 times: 1.75 ms per layer at 38K tokens, ~50% of the entire decode step.
#
# Here, per (request, kv head), the Q*G query rows that share the kv head are packed into one tile and the cached
# prefix (everything before this step's Q tokens) is split across many CTAs, each KV head read once (non-causal: every
# query sees the whole prefix). A second kernel attends this step's Q tokens causally (read from the same paged cache)
# and merges all partial results by log-sum-exp. No host sync and fixed grids: CUDA-graph safe.
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _prefix_split_kernel(
    q_ptr, kc_ptr, vc_ptr, bt_ptr, seq_lens_ptr, o_ptr, lse_ptr,
    stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
    stride_bt,
    stride_ob, stride_oh, stride_os, stride_or,
    stride_lb, stride_lh, stride_ls,
    qk_scale,                                   # softmax scale * log2(e)
    Q: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NUM_SPLITS: tl.constexpr, MIN_CHUNK: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HKV
    hkv = pid_bh % HKV
    R: tl.constexpr = Q * G
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_r = offs_r < R
    prefix_len = tl.maximum(tl.load(seq_lens_ptr + b) - Q, 0)
    chunk = tl.maximum(tl.cdiv(prefix_len, NUM_SPLITS), MIN_CHUNK)
    chunk = tl.cdiv(chunk, BLOCK_N) * BLOCK_N
    start = pid_s * chunk
    if start >= prefix_len:
        return
    end = tl.minimum(start + chunk, prefix_len)
    # packed rows r = qi * G + g  ->  token b*Q + qi, head hkv*G + g
    tok = b * Q + offs_r // G
    head = hkv * G + offs_r % G
    q = tl.load(q_ptr + tok[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :], mask=mask_r[:, None], other=0.0)
    m_i = tl.full([BLOCK_R], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_R], tl.float32)
    acc = tl.zeros([BLOCK_R, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        pos = n0 + offs_n
        mask_n = pos < end
        blk = tl.load(bt_ptr + b * stride_bt + pos // PAGE, mask=mask_n, other=0).to(tl.int64)
        off = pos % PAGE
        k = tl.load(kc_ptr + blk[:, None] * stride_kb + off[:, None] * stride_kt + hkv * stride_kh + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + blk[:, None] * stride_vb + off[:, None] * stride_vt + hkv * stride_vh + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    lse = m_i + tl.log2(l_i)                     # base-2 log-sum-exp; the split is non-empty
    o_base = o_ptr + b * stride_ob + hkv * stride_oh + pid_s * stride_os
    tl.store(o_base + offs_r[:, None] * stride_or + offs_d[None, :], o, mask=mask_r[:, None])
    tl.store(lse_ptr + b * stride_lb + hkv * stride_lh + pid_s * stride_ls + offs_r, lse, mask=mask_r)


@triton.jit
def _suffix_combine_kernel(
    q_ptr, kc_ptr, vc_ptr, bt_ptr, seq_lens_ptr, o_ptr, lse_ptr, out_ptr,
    stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
    stride_bt,
    stride_ob, stride_oh, stride_os, stride_or,
    stride_lb, stride_lh, stride_ls,
    stride_outt, stride_outh,
    qk_scale,
    Q: tl.constexpr, G: tl.constexpr, H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NUM_SPLITS: tl.constexpr, MIN_CHUNK: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)          # one (token, head) row
    tok = pid // H
    h = pid % H
    b = tok // Q
    qi = tok % Q
    hkv = h // G
    r = qi * G + h % G
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, BLOCK_J)
    offs_s = tl.arange(0, BLOCK_S)
    seq_len = tl.load(seq_lens_ptr + b)
    prefix_len = tl.maximum(seq_len - Q, 0)
    # causal part: this step's Q tokens, read from the paged cache exactly like FA2 would
    # (query qi sees keys up to seq_len - Q + qi)
    q = tl.load(q_ptr + tok * stride_qt + h * stride_qh + offs_d).to(tl.float32)
    pos = seq_len - Q + offs_j
    mask_j = (offs_j <= qi) & (pos >= 0)
    blk = tl.load(bt_ptr + b * stride_bt + pos // PAGE, mask=mask_j, other=0).to(tl.int64)
    off = pos % PAGE
    k = tl.load(kc_ptr + blk[:, None] * stride_kb + off[:, None] * stride_kt + hkv * stride_kh + offs_d[None, :], mask=mask_j[:, None], other=0.0)
    s = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * qk_scale
    s = tl.where(mask_j, s, float("-inf"))
    m_suf = tl.max(s, axis=0)
    m_suf = tl.where(m_suf == float("-inf"), 0.0, m_suf)
    p = tl.where(mask_j, tl.exp2(s - m_suf), 0.0)
    v = tl.load(vc_ptr + blk[:, None] * stride_vb + off[:, None] * stride_vt + hkv * stride_vh + offs_d[None, :], mask=mask_j[:, None], other=0.0)
    acc_suf = tl.sum(p[:, None] * v.to(tl.float32), axis=0)
    l_suf = tl.sum(p, axis=0)
    # prefix splits written by the first kernel (same split arithmetic)
    chunk = tl.maximum(tl.cdiv(prefix_len, NUM_SPLITS), MIN_CHUNK)
    chunk = tl.cdiv(chunk, BLOCK_N) * BLOCK_N
    n_active = tl.cdiv(prefix_len, chunk)
    mask_s = offs_s < n_active
    lse = tl.load(lse_ptr + b * stride_lb + hkv * stride_lh + offs_s * stride_ls + r, mask=mask_s, other=float("-inf"))
    o = tl.load(o_ptr + b * stride_ob + hkv * stride_oh + offs_s[:, None] * stride_os + r * stride_or + offs_d[None, :],
                mask=mask_s[:, None], other=0.0)
    m = tl.maximum(m_suf, tl.max(lse, axis=0))
    w = tl.exp2(lse - m)
    a = tl.exp2(m_suf - m)
    acc = acc_suf * a + tl.sum(w[:, None] * o, axis=0)
    l = l_suf * a + tl.sum(w, axis=0)
    l = tl.where(l > 0, l, 1.0)     # rows with no keys at all (padding requests) -> zeros, never NaN
    tl.store(out_ptr + tok * stride_outt + h * stride_outh + offs_d, (acc / l).to(out_ptr.dtype.element_ty))


def gqa_packed_spec_attention(query, key_cache, value_cache, out, seq_lens, block_table, num_reqs, Q, num_kv_heads,
                              scale, num_splits=64, min_chunk=128, block_n=32, num_warps=4, num_stages=2, combine_warps=2):
    """Causal attention for a uniform decode batch: num_reqs requests x Q query tokens each (token b*Q + qi).

    Same semantics as FA2 varlen causal with seqused_k=seq_lens and the block table: query qi of request b attends
    cached keys [0, seq_lens[b] - Q + qi]. query/out [>=B*Q, H, D]; key_cache/value_cache [num_blocks, PAGE, HKV, D]
    (strided views are fine); this step's K/V must already be in the cache (as FA2 also requires).
    """
    B = num_reqs
    H, D = query.shape[1], query.shape[2]
    HKV = num_kv_heads
    G = H // HKV
    R = Q * G
    PAGE = key_cache.shape[1]
    o_part = torch.empty((B, HKV, num_splits, R, D), device=query.device, dtype=torch.float32)
    lse_part = torch.empty((B, HKV, num_splits, R), device=query.device, dtype=torch.float32)
    qk_scale = scale * LOG2E
    _prefix_split_kernel[(num_splits, B * HKV)](
        query, key_cache, value_cache, block_table, seq_lens, o_part, lse_part,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3),
        lse_part.stride(0), lse_part.stride(1), lse_part.stride(2),
        qk_scale, Q=Q, G=G, HKV=HKV, D=D, PAGE=PAGE, NUM_SPLITS=num_splits, MIN_CHUNK=min_chunk,
        BLOCK_R=max(16, triton.next_power_of_2(R)), BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages)
    _suffix_combine_kernel[(B * Q * H,)](
        query, key_cache, value_cache, block_table, seq_lens, o_part, lse_part, out,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        o_part.stride(0), o_part.stride(1), o_part.stride(2), o_part.stride(3),
        lse_part.stride(0), lse_part.stride(1), lse_part.stride(2),
        out.stride(0), out.stride(1),
        qk_scale, Q=Q, G=G, H=H, D=D, PAGE=PAGE, NUM_SPLITS=num_splits, MIN_CHUNK=min_chunk, BLOCK_N=block_n,
        BLOCK_S=triton.next_power_of_2(num_splits), BLOCK_J=triton.next_power_of_2(Q), num_warps=combine_warps)
    return out
