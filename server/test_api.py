import http.client, json, os, time, threading, base64, struct, zlib
K = open(os.path.join(os.environ.get('QWEN_HOME', os.path.dirname(os.path.abspath(__file__))), 'serve.env')).read().strip().split('=', 1)[1]
AK = {'x-api-key': K, 'anthropic-version': '2023-06-01'}; BEARER = {'authorization': 'Bearer ' + K}
M = "Qwen3.8-27B-Uncensored"
def req(method, path, body=None, headers=None):
    c = http.client.HTTPConnection('127.0.0.1', 8421, timeout=900); h = {'content-type': 'application/json'}; h.update(headers or {})
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h); return c, c.getresponse()
def status(method, path, headers=None, body=None):
    c, r = req(method, path, body, headers); d = r.read(); c.close(); return r.status, d
print("== auth matrix ==")
for label, m, p, h, b in [
  ("GET  /health            no key", "GET", "/health", None, None),
  ("GET  /v1/models         no key", "GET", "/v1/models", None, None),
  ("GET  /v1/models         bad x-api-key", "GET", "/v1/models", {'x-api-key': 'nope'}, None),
  ("POST /v1/messages       no key", "POST", "/v1/messages", None, {"model": M, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
  ("POST /tokenize          no key", "POST", "/tokenize", None, {"prompt": "hi"}),
  ("GET  /metrics           no key", "GET", "/metrics", None, None),
  ("GET  /v1/models         x-api-key OK", "GET", "/v1/models", AK, None),
  ("GET  /v1/models         Bearer OK", "GET", "/v1/models", BEARER, None)]:
    print(f"  {status(m, p, h, b)[0]}  {label}")
s, d = status("POST", "/v1/messages", AK, {"model": M, "max_tokens": 60, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "Say hello in Chinese, one short sentence."}]})
j = json.loads(d); print("\n== messages ==", s, j.get("stop_reason"), json.dumps(j.get("content"), ensure_ascii=False)[:120])
print("== count_tokens ==", status("POST", "/v1/messages/count_tokens", AK, {"model": M, "messages": [{"role": "user", "content": "Hello"}]})[0])
s, d = status("POST", "/v1/messages", AK, {"model": M, "max_tokens": 400, "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}]})
j = json.loads(d); print("== default thinking ==", s, [b.get("type") for b in j.get("content", [])])
s, d = status("POST", "/v1/messages", AK, {"model": M, "max_tokens": 300, "thinking": {"type": "disabled"},
  "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
  "messages": [{"role": "user", "content": "What's the weather in Tokyo right now?"}]})
j = json.loads(d); print("== tool use ==", s, j.get("stop_reason"), [(b.get("name"), b.get("input")) for b in j.get("content", []) if b.get("type") == "tool_use"])
def png(w, h, rgb):
    raw = b''.join(b'\x00' + bytes(rgb) * w for _ in range(h))
    def ch(t, dd): c = struct.pack('>I', len(dd)) + t + dd; return c + struct.pack('>I', zlib.crc32(t + dd) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + ch(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + ch(b'IDAT', zlib.compress(raw)) + ch(b'IEND', b'')
img = base64.b64encode(png(128, 128, (220, 20, 20))).decode()
s, d = status("POST", "/v1/messages", AK, {"model": M, "max_tokens": 40, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
  {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}, {"type": "text", "text": "What color is this image? One word."}]}]})
print("== vision ==", s, json.loads(d)["content"] if s == 200 else d[:150])
# Claude Code sends effort=high; the original template rejected it with a 400. Verify the patched template accepts every level.
for eff in ("low", "medium", "high", "xhigh", "max"):
    s, d = status("POST", "/v1/messages", AK, {"model": M, "max_tokens": 30, "thinking": {"type": "adaptive"}, "output_config": {"effort": eff}, "messages": [{"role": "user", "content": "Say hi."}]})
    print(f"== effort={eff:6s} ->", s)
def stream(body):
    body = dict(body, stream=True); t0 = time.time(); c, r = req("POST", "/v1/messages", body, AK); first = None; n = 0
    for line in r:
        if not line.startswith(b'data:'): continue
        e = json.loads(line[5:])
        if e.get('type') == 'content_block_delta' and first is None: first = time.time()
        if e.get('type') == 'message_delta': n = (e.get('usage') or {}).get('output_tokens', n)
    t1 = time.time(); c.close(); return dict(ttft=(first or t1) - t0, out=n, tps=(n - 1) / max(t1 - (first or t0), 1e-6))
B = {"model": M, "max_tokens": 400, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "Write a detailed story about a lighthouse keeper who finds a message in a bottle."}]}
stream(dict(B, max_tokens=32))
print("\n== speed (GPU may be shared with co-tenant jobs) ==")
for i in range(3):
    r = stream(B); print(f"  single run{i+1}: TTFT {r['ttft']*1000:.0f} ms | {r['out']} tok | {r['tps']:.1f} tok/s")
res = []; ths = []; t0 = time.time()
for _ in range(8): th = threading.Thread(target=lambda: res.append(stream(B))); th.start(); ths.append(th)
for th in ths: th.join()
print(f"  8 concurrent: aggregate {sum(x['out'] for x in res)/(time.time()-t0):.0f} tok/s")
