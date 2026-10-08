import torch, itertools, sys, json
import torch.nn.functional as F
from vllm import _custom_ops as ops
sys.path.insert(0, '.')
from w8a16 import w8a16_mm
from benchutil import bench
from mb1 import SHAPES
dev = torch.device("cuda")
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,3,4").split(",")]
shapes = {k: v for k, v in SHAPES.items() if k != "gdn.in_ba"}; shapes["lm_head.int8"] = (248320, 5120, 1)
best = {}
for name, (N, K, cnt) in shapes.items():
    W = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev); WT = W.t()
    ws = (torch.rand(N, device=dev) * 0.01 + 0.001).float()
    for M in MS[:1] + [m for m in MS[1:] if False]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = F.linear(x.float(), W.float() * ws[:, None])
        def cut():
            xq, xs, _ = ops.scaled_int8_quant(x); return ops.cutlass_scaled_mm(xq, WT, xs, ws, torch.bfloat16)
        tc = bench(cut)
        cands = []
        for BN, BK, SK, nw, ns in itertools.product((16, 32, 64), (128, 256, 512), (1, 2, 4), (4, 8), (3, 4)):
            if K % (BK * SK): continue
            if BN * BK > 64 * 512 or BN * BK < 16 * 128: continue
            cfg = (16, BN, BK, SK, nw, ns)
            try:
                out = w8a16_mm(x, WT, ws, cfg).float()
                err = ((out - ref).norm() / ref.norm()).item()
                if err > 0.01: continue
                cands.append((bench(lambda: w8a16_mm(x, WT, ws, cfg), iters=20, reps=3), cfg, err))
            except Exception as e:
                pass
        cands.sort()
        t, cfg, err = cands[0]
        best[f"{N},{K},{M}"] = cfg
        print(f"{name:13s} N={N:6d} K={K:6d} M={M}: cutlass {tc:7.1f}us ({N*K/1e3/tc:5.0f}GB/s) | w8a16 {t:7.1f}us ({N*K/1e3/t:5.0f}GB/s) err={err:.4f} cfg={cfg} | {tc/t:.2f}x", flush=True)
    del W; torch.cuda.empty_cache()
json.dump(best, open("w8a16_best.json", "w"))
