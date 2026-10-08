# qwen_fastpath

vLLM plugin with three decode-time fast paths for this single-user deployment (Qwen3.8-27B W8A8-INT8, A100 80GB PCIe,
MTP speculative decoding). vLLM loads it in every process through the `vllm.general_plugins` entry point
(install with `pip install -e plugin/` or `uv pip install -e plugin/` into the vLLM venv). Switches live in `../tuning.env`; `run_vllm.sh`
puts a hash of this source and the switches into `--additional-config`, so changing either recompiles instead of
reusing old torch.compile graphs.

| switch | what it replaces | effect |
|---|---|---|
| `QWEN_FAST_ATTN` | FlashAttention-2 for speculative-decode verification (every request has 1 < Q <= 8 query tokens) | FA2's varlen path never splits the KV sequence and streams each KV head once per query head: 1.75 ms/layer at 38K context. `spec_attn.py` packs the Q x 6 query heads that share a KV head, splits the cached prefix across 64 CTAs and merges by log-sum-exp: 0.13 ms at 38K, 0.29 ms at 100K. Same math as FA2 (error vs fp32 equal or lower). |
| `QWEN_FAST_LINEAR` | CUTLASS W8A8 int8 GEMM when M <= 16 (decode) | Same int8 weights, bf16 activations (W8A16, Triton): 13-20% faster at decode, 5x lower error because activations are no longer rounded to int8. Prefill (M > 16) still uses the int8 tensor cores. |
| `QWEN_FAST_DRAFT_HEAD` | bf16 lm_head (2.5 GB) for MTP draft proposals | Int8 per-row copy (+1.3 GB) through the W8A16 kernel. Only proposals change; acceptance is tested against the target's bf16 logits, so outputs are unchanged in distribution. |

Tests (run on the server after `. $QWEN_HOME/env.sh`): `bench/test_spec_attn.py`, `bench/test_w8a16.py`.
Roll back: set the switches to 0 (or `QWEN_SPEC_TOKENS=2` and empty `QWEN_DRAFT_SAMPLING` for the old drafting)
and `qwenctl restart`; the pre-change scripts are in `../backup-pre-fastpath/`.
