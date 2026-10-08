# Long Claude Code-like session: compare per-request TTFT with inline system messages (new template) versus the
# old behaviour (vLLM merging every inline system message into the leading system prompt), emulated client-side.
import json, sys, time, copy, uuid, urllib.request
sys.path.insert(0, '.')
from bench2 import metrics, URL, H

def send(q):
    q = dict(q, max_tokens=1, stream=False); q.pop("context_management", None)
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(q).encode(), headers=H)
    t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); return time.time() - t0, r["usage"]["input_tokens"]

def merged(q):
    """What vLLM does with a system-first template: inline system texts appended to the top-level system."""
    q = copy.deepcopy(q); extra = []
    msgs = []
    for m in q["messages"]:
        if m["role"] == "system":
            c = m["content"]; extra.append(c if isinstance(c, str) else "".join(b.get("text", "") for b in c))
        else:
            msgs.append(m)
    q["messages"] = msgs; q["system"] = list(q["system"]) + [{"type": "text", "text": t} for t in extra]
    return q

base = json.load(open("cap1/0007.json"))["req"]
big = open(sys.argv[1]).read()[:110000]            # a large file "read" by the agent (~25-30K tokens)

def session(nonce):
    q = copy.deepcopy(base)
    q["messages"][0]["content"][1]["text"] = f"[{nonce}] " + q["messages"][0]["content"][1]["text"]
    # make the first tool result a big file read
    for m in q["messages"]:
        if m["role"] == "user" and isinstance(m["content"], list) and m["content"] and m["content"][0].get("type") == "tool_result":
            m["content"][0]["content"] = big; break
    steps = [q]
    for i in range(6):
        q = copy.deepcopy(q)
        tid = f"toolu_{uuid.uuid4().hex[:20]}"
        q["messages"].append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"Step {i}: check the next part of the output.", "signature": "x"},
            {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": f"grep -n 'def ' file_{i}.py | head -40", "description": "List functions"}}]})
        q["messages"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
            "content": "\n".join(f"{n}: def function_{i}_{n}(self, arg_{n}): return arg_{n} * {n}" for n in range(40))}]})
        q["messages"].append({"role": "system", "content": f"<total_tokens>{14990000 - 1000 * i} tokens left</total_tokens>"})
        steps.append(q)
    return steps

for label, transform in (("merged (old)", merged), ("inline (new)", lambda x: x)):
    steps = session(uuid.uuid4().hex[:8])
    out = []
    for s in steps:
        a = metrics(); t, n = send(transform(s)); b = metrics()
        hit = (b["prefix_cache_hits"] - a["prefix_cache_hits"]) / max(1, b["prefix_cache_queries"] - a["prefix_cache_queries"])
        out.append((t, n, hit))
    print(f"{label}: " + " | ".join(f"{t:.2f}s/{n}tok/{h:.0%}" for t, n, h in out))
    print(f"   after the first request: total {sum(t for t, _, _ in out[1:]):.2f}s over {len(out) - 1} agent steps")
