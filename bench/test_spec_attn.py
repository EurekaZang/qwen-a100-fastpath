import torch, sys, itertools
sys.path.insert(0, '.')
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from qwen_fastpath.spec_attn import gqa_packed_spec_attention
from benchutil import bench
torch.manual_seed(0)
dev = "cuda"; H, HKV, D, BS = 24, 4, 256, 800; G = H // HKV; scale = D ** -0.5
tune = len(sys.argv) > 1 and sys.argv[1] == "tune"
for ctx in ((0, 1, 2, 50, 1000, 8000, 38000, 100000) if not tune else (1000, 38000, 100000)):
    for Q, B in (((3, 1), (4, 1), (3, 2)) if not tune else ((3, 1),)):
        L = ctx + Q
        nblk = (L + BS - 1) // BS + 1
        kv_cache = torch.randn(B * nblk + 5, HKV, BS, 2 * D, device=dev, dtype=torch.bfloat16)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(D, dim=-1)
        block_table = torch.randperm(B * nblk + 5, device=dev)[:B * nblk].to(torch.int32).view(B, nblk)
        q = torch.randn(B * Q, H, D, device=dev, dtype=torch.bfloat16)
        lens = [max(L - 7 * i, 1) for i in range(B)]                           # different lengths per request
        seq_lens = torch.tensor(lens, device=dev, dtype=torch.int32)
        cu_q = torch.arange(0, B + 1, device=dev, dtype=torch.int32) * Q
        k_new = torch.empty(B * Q, HKV, D, device=dev, dtype=torch.bfloat16); v_new = torch.empty_like(k_new)
        golds = []
        for b in range(B):
            Lb = lens[b]; allpos = torch.arange(0, Lb, device=dev)
            ab = block_table[b, allpos // BS].long(); ao = allpos % BS
            Kb, Vb = key_cache[ab, ao], value_cache[ab, ao]
            Kf = Kb.float().repeat_interleave(G, dim=1); Vf = Vb.float().repeat_interleave(G, dim=1)
            s = torch.einsum('qhd,lhd->hql', q[b * Q:(b + 1) * Q].float(), Kf) * scale
            s.masked_fill_((allpos[None, :] > (Lb - Q + torch.arange(Q, device=dev))[:, None])[None], float('-inf'))
            golds.append(torch.nan_to_num(torch.einsum('hql,lhd->qhd', s.softmax(-1), Vf), nan=0.0))
        gold = torch.cat(golds)
        err = lambda o: ((o.float() - gold).norm() / gold.norm()).item() if torch.isfinite(o.float()).all() else float('nan')
        out_ref = torch.empty_like(q); out_new = torch.empty_like(q)
        ref = lambda: flash_attn_varlen_func(q=q, k=key_cache, v=value_cache, out=out_ref, cu_seqlens_q=cu_q, max_seqlen_q=Q,
                                             seqused_k=seq_lens, max_seqlen_k=131072, softmax_scale=scale, causal=True,
                                             block_table=block_table, fa_version=2)
        ref()
        t_ref = bench(ref, iters=20)
        if not tune:
            new = lambda: gqa_packed_spec_attention(q, key_cache, value_cache, out_new, seq_lens, block_table, B, Q, HKV, scale)
            new()
            print(f"ctx={ctx:6d} Q={Q} B={B}: FA2 {t_ref:7.1f}us err={err(out_ref):.1e} | packed-split {bench(new, iters=20):7.1f}us err={err(out_new):.1e}", flush=True)
        else:
            best = []
            for ns, mc, bn, nw, st, cw in itertools.product((32, 64, 128), (128, 256, 512), (16, 32, 64), (4, 8), (2, 3), (2, 4)):
                try:
                    f = lambda: gqa_packed_spec_attention(q, k_new, v_new, key_cache, value_cache, out_new, seq_lens, block_table, B, Q, scale,
                                                          num_splits=ns, min_chunk=mc, block_n=bn, num_warps=nw, num_stages=st, combine_warps=cw)
                    f()
                    if err(out_new) > 5e-3: continue
                    best.append((bench(f, iters=20, reps=3), (ns, mc, bn, nw, st, cw)))
                except Exception as e:
                    pass
            best.sort()
            print(f"ctx={ctx}: FA2 {t_ref:.1f}us; best: " + "; ".join(f"{t:.1f}us {c}" for t, c in best[:6]), flush=True)
        del kv_cache, key_cache, value_cache; torch.cuda.empty_cache()
