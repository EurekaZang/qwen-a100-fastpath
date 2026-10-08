#!/bin/bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# MTP speculative decoding: k draft tokens per step. Drafts are sampled from the MTP head's own distribution
# ("probabilistic") and accepted/resampled by the standard speculative-sampling test, so outputs follow the
# target model's distribution exactly; at temperature 1.0 this accepts more drafts than greedy (argmax) drafting.
N=${QWEN_SPEC_TOKENS:-4}
SPEC_JSON="\"method\":\"mtp\",\"num_speculative_tokens\":$N"
[ -n "${QWEN_DRAFT_SAMPLING:-}" ] && SPEC_JSON="$SPEC_JSON,\"draft_sample_method\":\"$QWEN_DRAFT_SAMPLING\""
SPEC_ARGS=()
[ "$N" -gt 0 ] && SPEC_ARGS=(--speculative-config "{$SPEC_JSON}")
# Single-user decode fast paths (vLLM plugin in $W/fastpath, loaded through its venv entry point). Key the
# torch.compile cache on the plugin source and switches, so a changed or disabled patch never reuses old graphs.
FP="$(cat $W/fastpath/qwen_fastpath/*.py | sha256sum | cut -c1-12)-${QWEN_FAST_ATTN:-1}${QWEN_FAST_LINEAR:-1}${QWEN_FAST_DRAFT_HEAD:-1}"
exec $W/venv/bin/vllm serve $W/model \
  --served-model-name Qwen3.8-27B-Uncensored \
  --host 127.0.0.1 --port 8421 \
  --middleware anthropic_auth.AnthropicKeyAuth \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.72 \
  --max-num-seqs 32 \
  --performance-mode interactivity \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --chat-template $W/chat_template_cc.jinja \
  --additional-config "{\"qwen_fastpath\":\"$FP\"}" \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
