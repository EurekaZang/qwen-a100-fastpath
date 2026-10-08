import json, sys, time, random, urllib.request
sys.path.insert(0, '.')
from bench2 import URL, H, filler
for n_words, depth in ((27000, 0.3), (27000, 0.85), (72000, 0.5)):
    rnd = random.Random(n_words + int(depth * 100))
    code = f"{rnd.randint(1000, 9999)}-{rnd.choice(['ALPHA', 'BRAVO', 'KILO', 'OSCAR', 'TANGO'])}-{rnd.randint(10, 99)}"
    words = filler(n_words, n_words).split()
    pos = int(len(words) * depth)
    text = " ".join(words[:pos]) + f"\n\nIMPORTANT NOTE: the vault access code is {code}. Remember it.\n\n" + " ".join(words[pos:])
    q = {"model": "Qwen3.8-27B-Uncensored", "max_tokens": 60, "thinking": {"type": "disabled"}, "temperature": 0,
         "messages": [{"role": "user", "content": text + "\n\nWhat is the vault access code mentioned in the text above? Reply with the code and then one sentence describing where it appeared."}]}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/messages", data=json.dumps(q).encode(), headers=H), timeout=3600).read())
    ans = "".join(b.get("text", "") for b in r["content"])
    print(f"{r['usage']['input_tokens']:6d} tok, depth {depth:.0%}: expected {code} -> {'OK ' if code in ans else 'MISS'} ({time.time() - t0:.0f}s) | {ans.strip()[:120]!r}", flush=True)
