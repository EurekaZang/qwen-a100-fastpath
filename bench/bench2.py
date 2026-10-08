# Single-user benchmark suite (runs on the GPU server against 127.0.0.1:8421; QWEN_HOME = deployment dir)
import json, os, sys, time, random, glob, re, urllib.request
KEY = [l.split('=', 1)[1].strip() for l in open(os.path.join(os.environ['QWEN_HOME'], 'serve.env')) if l.startswith('QWEN_API_KEY=')][0]
URL = "http://127.0.0.1:8421"
H = {"content-type": "application/json", "x-api-key": KEY, "anthropic-version": "2023-06-01"}

def metrics():
    req = urllib.request.Request(URL + "/metrics", headers=H)
    out = {}
    for line in urllib.request.urlopen(req, timeout=30).read().decode().splitlines():
        m = re.match(r'^(vllm:(prefix_cache_queries|prefix_cache_hits|spec_decode_num_drafts|spec_decode_num_draft_tokens|spec_decode_num_accepted_tokens|prompt_tokens|generation_tokens)_total)\{[^}]*\} ([0-9.e+]+)', line)
        if m: out[m.group(2)] = out.get(m.group(2), 0) + float(m.group(3))
    return out

def stream(body):
    body = dict(body, stream=True)
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); first = None; n_out = 0; nin = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            d = json.loads(line[5:])
            if d.get("type") == "content_block_delta" and first is None: first = time.time()
            if d.get("type") == "message_start": nin = d["message"].get("usage", {}).get("input_tokens")
            if d.get("type") == "message_delta": n_out = d.get("usage", {}).get("output_tokens", n_out)
    t1 = time.time()
    return dict(ttft=(first or t1) - t0, out=n_out, inp=nin, total=t1 - t0, dec=((n_out - 1) / (t1 - first)) if first and n_out > 1 else 0.0)

P = {"story": "Write a long, detailed fantasy story about a lighthouse keeper who discovers a hidden library. At least 1500 words.",
     "code": "Write a complete Python implementation of a red-black tree with insert, delete, search, in-order traversal, and thorough docstrings and unit tests. Output only code.",
     "zh": "请用中文详细解释 Raft 共识算法：领导者选举、日志复制、安全性证明的直觉，以及与 Paxos 的对比，至少 1500 字。"}

def filler(n_words, seed):
    rnd = random.Random(seed)
    words = "system kernel memory latency throughput cache vector tensor shard replica leader follower commit index term vote quorum lease clock drift".split()
    return " ".join(rnd.choice(words) for _ in range(n_words))

def spec_delta(m0, m1):
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    al = 1 + d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_drafts"] if d.get("spec_decode_num_drafts") else None
    return al

def decode_suite(reps):
    res = {}
    for k in ("story", "code", "zh"):
        vals = []; als = []
        for i in range(reps):
            m0 = metrics()
            r = stream({"model": "Qwen3.8-27B-Uncensored", "max_tokens": 1024, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": P[k]}]})
            als.append(spec_delta(m0, metrics())); vals.append(r["dec"])
        res[k] = (round(sum(vals) / len(vals), 1), round(sum(a for a in als if a) / max(1, len([a for a in als if a])), 2) if any(als) else None)
        print(f"decode {k:5s}: {res[k][0]} tok/s  accept_len={res[k][1]}  runs={[round(v,1) for v in vals]}", flush=True)
    return res

def prefill_suite():
    for n in (2000, 8000, 24000):
        nonce = f"[run {time.time()}]"
        txt = nonce + " " + filler(int(n * 0.9), n)
        r = stream({"model": "Qwen3.8-27B-Uncensored", "max_tokens": 1, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": txt + "\nSummarize in one word."}]})
        print(f"prefill {r['inp']:6d} tok uncached: TTFT {r['ttft']:.2f}s -> {r['inp'] / r['ttft']:.0f} tok/s", flush=True)

def longctx_suite():
    base = filler(36000, 7)  # ~40K tokens, cached after first use
    for i in range(2):
        r = stream({"model": "Qwen3.8-27B-Uncensored", "max_tokens": 512, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": base + "\nNow write a 400-word essay about distributed consensus."}]})
        print(f"long-ctx {r['inp']} tok: TTFT {r['ttft']:.2f}s decode {r['dec']:.1f} tok/s", flush=True)

def replay_suite(capdir):
    files = sorted(glob.glob(capdir + "/*.json"))
    m0 = metrics(); tt = []
    for f in files:
        q = json.load(open(f))["req"]
        q = dict(q, max_tokens=1); q.pop("context_management", None)
        a = metrics(); r = stream(q); b = metrics()
        hit = (b["prefix_cache_hits"] - a["prefix_cache_hits"]) / max(1, b["prefix_cache_queries"] - a["prefix_cache_queries"])
        tt.append(r["ttft"])
        print(f"replay {os.path.basename(f)}: in={r['inp']} TTFT={r['ttft']:.2f}s prefix-hit={hit:.0%}", flush=True)
    m1 = metrics()
    print(f"replay total TTFT {sum(tt):.2f}s; overall hit {(m1['prefix_cache_hits'] - m0['prefix_cache_hits']) / max(1, m1['prefix_cache_queries'] - m0['prefix_cache_queries']):.0%}")

if __name__ == "__main__":
    what = sys.argv[1].split(",")
    if "decode" in what: decode_suite(int(os.environ.get("REPS", "2")))
    if "prefill" in what: prefill_suite()
    if "long" in what: longctx_suite()
    if "replay" in what: replay_suite(sys.argv[2])
