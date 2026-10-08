import torch, time, sys
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
torch.manual_seed(0)
dev = torch.device("cuda")
SHAPES = {} if False else {  # name: (N_out, K_in, count per forward)
    "mlp.gate_up": (34816, 5120, 64), "mlp.down": (5120, 17408, 64),
    "gdn.in_qkvz": (16384, 5120, 48), "gdn.in_ba": (96, 5120, 48), "gdn.out": (5120, 6144, 48),
    "attn.qkv": (14336, 5120, 16), "attn.o": (5120, 6144, 16),
}
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "1,3,4,5").split(",")]

def bench(fn, iters=30, reps=5):
    # capture `iters` calls in a CUDA graph, replay, take the best of `reps`
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e6  # us

def pack_gptq_u8(w_int8_nk):  # (N,K) int8 -> (K/4, N) int32, uint8b128
    wu = (w_int8_nk.t().to(torch.int32) + 128)  # K,N
    K, N = wu.shape
    wu = wu.reshape(K // 4, 4, N)
    out = torch.zeros(K // 4, N, dtype=torch.int32, device=wu.device)
    for i in range(4): out |= (wu[:, i, :] << (8 * i))
    return out

def main():
    tot = {}
    for name, (N, K, cnt) in SHAPES.items():
        W = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
        ws = (torch.rand(N, device=dev) * 0.01 + 0.001).float()
        Wbf = (W.float() * ws[:, None]).to(torch.bfloat16)
        # cutlass/triton layout: weight (K,N) column-major == W.t()
        WT = W.t()
        # marlin W8A16
        Np = (N + 127) // 128 * 128  # marlin needs N % 64 == 0; vLLM pads the same way
        Wp = torch.nn.functional.pad(W, (0, 0, 0, Np - N)); wsp_ = torch.nn.functional.pad(ws, (0, Np - N), value=1.0)
        qw = ops.gptq_marlin_repack(pack_gptq_u8(Wp).contiguous(), size_k=K, size_n=Np, num_bits=8)
        ms_ = mu.marlin_permute_scales(wsp_.to(torch.bfloat16).reshape(1, Np), size_k=K, size_n=Np, group_size=-1)
        zp = mu.marlin_make_empty(dev); wsp = mu.marlin_make_workspace_new(dev, 4)
        for M in MS:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            ref = F.linear(x.float(), W.float() * ws[:, None])
            def cut():
                xq, xs, _ = ops.scaled_int8_quant(x); return ops.cutlass_scaled_mm(xq, WT, xs, ws.reshape(N, 1) if False else ws, torch.bfloat16)
            def tri():
                xq, xs, _ = ops.scaled_int8_quant(x); return triton_scaled_mm(xq, WT, xs, ws.reshape(N, 1), torch.bfloat16)
            def bf(): return F.linear(x, Wbf)
            def mar(): return mu.apply_gptq_marlin_linear(x, qw, ms_, zp, wsp, scalar_types.uint8b128, Np, K)[:, :N]
            res = {}
            for tag, fn in (("cutlass_w8a8", cut), ("triton_w8a8", tri), ("bf16", bf), ("marlin_w8a16", mar)):
                try:
                    out = fn().float()
                    err = ((out - ref).norm() / ref.norm()).item()
                    t = bench(fn)
                    res[tag] = (t, err)
                    tot.setdefault((tag, M), 0.0); tot[(tag, M)] += t * cnt
                except Exception as e:
                    res[tag] = (float('nan'), repr(e)[:80])
            gbps = N * K / 1e3  # bytes int8 / us -> GB/s scale
            print(f"{name:13s} N={N:6d} K={K:6d} M={M}: " + "  ".join(
                f"{k}={v[0]:7.1f}us({gbps / v[0] if v[0] == v[0] else 0:5.0f}GB/s,err={v[1] if isinstance(v[1], str) else round(v[1], 4)})" for k, v in res.items()), flush=True)
        del W, Wbf, qw
        torch.cuda.empty_cache()
    print("\nper-forward linear total (ms), body only (no lm_head):")
    for (tag, M), v in sorted(tot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  M={M} {tag:14s} {v / 1000:7.2f} ms")
    # lm_head bf16
    Wl = torch.randn(248320, 5120, device=dev, dtype=torch.bfloat16)
    for M in MS:
        x = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: F.linear(x, Wl), iters=10)
        print(f"lm_head bf16 M={M}: {t:.1f} us ({248320 * 5120 * 2 / 1e3 / t:.0f} GB/s)")
    # raw bandwidth
    a = torch.empty(2 * 1024**3, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
    t = bench(lambda: b.copy_(a), iters=5)
    print(f"copy 2GiB: {t:.0f} us -> {2 * 2 * 1024**3 / 1e3 / t:.0f} GB/s (read+write)")

if __name__ == '__main__':
    main()
