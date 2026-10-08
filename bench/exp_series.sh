#!/bin/bash
# exp_series.sh <name> <suites> -- [env assignments...] -- [vllm args...]   (one experiment; appends to results)
W=${QWEN_HOME:?set QWEN_HOME to the deployment directory}; cd $W/bench; . ../env.sh
NAME=$1; SUITES=$2; shift 2
[ "$1" = "--" ] && shift
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
R=$W/bench/results.txt
nsys shutdown --session=qwenprof >/dev/null 2>&1
pkill -f "vllm serve"; for i in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done; pkill -9 -f "vllm serve"; sleep 3
echo "=== $NAME [$(date +%H:%M:%S)] env: ${ENVS[*]} args: $*" >> $R
env "${ENVS[@]}" setsid ./exp_run.sh $NAME "$@" > $W/logs/exp_$NAME.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8421/health)" = 200 ] && break
  pgrep -f "vllm serve" >/dev/null || { echo "  DIED: $(grep -E 'Error|error' $W/logs/exp_$NAME.log | tail -3)" >> $R; exit 1; }
  sleep 5
done
echo "  up after $((i*5))s; gpu: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')" >> $R
grep -E "Using .* attention backend|Selected .*Kernel|cudagraph|KV cache size" $W/logs/exp_$NAME.log | sed 's/^.*\] /  log: /' | cut -c1-200 | sort -u >> $R
python bench2.py $SUITES cap1 2>&1 | sed 's/^/  /' >> $R
echo "  done [$(date +%H:%M:%S)]" >> $R
