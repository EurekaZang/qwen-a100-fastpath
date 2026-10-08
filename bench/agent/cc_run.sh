#!/bin/bash
# usage: cc_run.sh <base_url> <workdir> <out_prefix>   -- two-turn Claude Code session against the local Qwen server
set -eu
BASE_URL=$1; WD=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/cc_env.sh"
export ANTHROPIC_BASE_URL="$BASE_URL"
case "$ANTHROPIC_BASE_URL" in http://127.0.0.1:*) ;; *) echo "refusing: base url not local" >&2; exit 2;; esac
[ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] || { echo "no key" >&2; exit 2; }
rm -rf "$WD"; mkdir -p "$WD"; cd "$WD"
T0=$(date +%s)
claude -p --permission-mode bypassPermissions --output-format json "Create a Python module stats_utils.py with functions mean, median and mode that handle edge cases (empty input raises ValueError, mode returns the smallest of ties). Write unittest tests in test_stats_utils.py, run them with python3 -m unittest, and fix any failures. Reply with a one-line summary." > "$OUT.1.json" 2>"$OUT.1.err" || true
echo "turn1 $(( $(date +%s)-T0 ))s"
T0=$(date +%s)
claude -p --continue --permission-mode bypassPermissions --output-format json "Now add a variance(data, sample=True) function with tests, run the full test suite again and reply with the test count." > "$OUT.2.json" 2>"$OUT.2.err" || true
echo "turn2 $(( $(date +%s)-T0 ))s"
python3 - "$OUT" <<'PY'
import json,sys
for i in (1,2):
    try:
        d=json.load(open(f"{sys.argv[1]}.{i}.json"))
        print(i, list(d.get('modelUsage',{}).keys()), d.get('num_turns'), d.get('duration_ms'), '|', d.get('result','')[:160].replace('\n',' '))
    except Exception as e: print(i,'ERR',e, open(f"{sys.argv[1]}.{i}.err").read()[:300])
PY
