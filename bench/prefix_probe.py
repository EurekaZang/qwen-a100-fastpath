import json, sys, time, urllib.request
sys.path.insert(0, '.')
from bench2 import metrics, URL, H, filler
def send(sysmsg, user):
    q = {"model": "Qwen3.8-27B-Uncensored", "max_tokens": 1, "thinking": {"type": "disabled"},
         "system": sysmsg, "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(q).encode(), headers=H)
    a = metrics(); t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); dt = time.time() - t0; b = metrics()
    return dt, r["usage"]["input_tokens"], int(b["prefix_cache_hits"] - a["prefix_cache_hits"]), int(b["prefix_cache_queries"] - a["prefix_cache_queries"])
shared = f"[probe {time.time()}] " + filler(18000, 11)      # ~20K-token shared system prompt
for i in range(3):
    print("request", i, send(shared, f"Question {i}: {filler(300, 100 + i)} -- answer briefly."))
