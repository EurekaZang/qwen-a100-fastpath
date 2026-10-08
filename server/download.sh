set -u
W=${QWEN_HOME:-$(cd "$(dirname "$0")" && pwd)}; H=$W/hf/.hdr; D=$W/model; mkdir -p $D
REPO=orcarouter/Qwen3.8-27B-Uncensored-INT8
BASE=https://huggingface.co/$REPO/resolve/main
curl -sS -m 60 -H @$H "https://huggingface.co/api/models/$REPO?blobs=true" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for s in d['siblings']:
    l=s.get('lfs') or {}
    print(s['rfilename'], s.get('size') or 0, l.get('sha256','-'))
" > $W/manifest.tsv
cat $W/manifest.tsv; date
pids=""
while read name size sha; do
  case "$name" in .gitattributes|README.md) continue;; esac
  ( n=0; until curl -fsSL -C - -H @$H -o "$D/$name" "$BASE/$name"; do n=$((n+1)); [ $n -ge 12 ] && { echo "DL_FAIL $name"; break; }; sleep 5; done ) &
  pids="$pids $!"
done < $W/manifest.tsv
wait $pids
date; echo "=== verify ==="
bad=0
while read name size sha; do
  case "$name" in .gitattributes|README.md) continue;; esac
  got=$(stat -c %s "$D/$name" 2>/dev/null || echo 0)
  if [ "$got" != "$size" ]; then echo "SIZE_MISMATCH $name want=$size got=$got"; bad=1; continue; fi
  if [ "$sha" != "-" ]; then
    h=$(sha256sum "$D/$name" | cut -d' ' -f1)
    if [ "$h" = "$sha" ]; then echo "OK   $name"; else echo "SHA_MISMATCH $name"; bad=1; fi
  else echo "OK(size) $name"; fi
done < $W/manifest.tsv
[ $bad = 0 ] && echo DOWNLOAD_VERIFIED || echo DOWNLOAD_PROBLEM
