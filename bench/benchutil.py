import time, torch
def bench(fn, iters=30, reps=5):
    """Time fn() by replaying `iters` calls captured in one CUDA graph; best of `reps`, in microseconds."""
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
    return best * 1e6
