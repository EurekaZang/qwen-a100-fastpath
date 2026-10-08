# Accuracy of the decode-path W8A16 kernel vs the CUTLASS W8A8 path, against the exact product, for this model's shapes.
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from qwen_fastpath.w8a16 import pick_config, w8a16_mm
torch.manual_seed(0)
SHAPES = {"mlp.gate_up": (34816, 5120), "mlp.down": (5120, 17408), "gdn.in_qkvz": (16384, 5120),
          "gdn.out / attn.o": (5120, 6144), "attn.qkv": (14336, 5120), "lm_head (draft)": (248320, 5120)}
for name, (N, K) in SHAPES.items():
    W = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda")
    ws = (torch.rand(N, device="cuda") * 0.01 + 0.001).float()
    for M in (1, 4, 16):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # exact reference, chunked over N so the fp32 weight copy stays small
        ref = torch.cat([F.linear(x.float(), W[i:i + 8192].float() * ws[i:i + 8192, None]) for i in range(0, N, 8192)], dim=1)
        y16 = w8a16_mm(x, W.t(), ws, pick_config(N, K), torch.bfloat16).float()
        xq, xs, _ = ops.scaled_int8_quant(x)
        y8 = ops.cutlass_scaled_mm(xq, W.t(), xs, ws, torch.bfloat16).float()
        e = lambda y: ((y - ref).norm() / ref.norm()).item()
        print(f"{name:17s} M={M:2d}: W8A16 rel.err {e(y16):.4f} | W8A8 (CUTLASS) rel.err {e(y8):.4f}")
    del W
