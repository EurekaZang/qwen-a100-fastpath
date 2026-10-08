#!/bin/bash
# experiment launcher: exp_run.sh <tag> [extra vllm args...]; env PREFIX= (e.g. nsys launch ...), TEMPLATE=, SPEC=
W=${QWEN_HOME:?set QWEN_HOME to the deployment directory}
. $W/env.sh
[ -n "${CACHE_ROOT:-}" ] && export VLLM_CACHE_ROOT=$CACHE_ROOT
TAG=$1; shift
SPEC=${SPEC:-2}
SPEC_ARGS=(); [ "$SPEC" -gt 0 ] && SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC${SPEC_EXTRA:+,$SPEC_EXTRA}}")
exec ${PREFIX:-} $W/venv/bin/vllm serve ${MODEL_DIR:-$W/model} \
  --served-model-name Qwen3.8-27B-Uncensored --host 127.0.0.1 --port 8421 \
  --middleware anthropic_auth.AnthropicKeyAuth --max-model-len 131072 \
  --gpu-memory-utilization ${GMU:-0.72} --max-num-seqs ${MAXSEQS:-32} --performance-mode interactivity \
  --max-num-batched-tokens ${MBT:-8192} --enable-prefix-caching \
  --chat-template ${TEMPLATE:-$W/chat_template_cc.jinja} --additional-config "{\"qwen_fastpath\":\"$(cat $W/fastpath/qwen_fastpath/*.py | sha256sum | cut -c1-12)-${QWEN_FAST_ATTN:-1}${QWEN_FAST_LINEAR:-1}${QWEN_FAST_DRAFT_HEAD:-1}\"}" \
  "${SPEC_ARGS[@]}" --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder "$@"
