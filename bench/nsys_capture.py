import sys, threading, time, subprocess, json
sys.path.insert(0, '.')
from bench2 import stream, filler, P, metrics, spec_delta
def capture(tag, body, delay):
    res = {}
    th = threading.Thread(target=lambda: res.update(stream(body)))
    m0 = metrics(); th.start(); time.sleep(delay)
    subprocess.run(["nsys", "start", "--session=qwenprof", "-o", f"prof_{tag}", "--force-overwrite=true"], check=True)
    time.sleep(1.5)
    subprocess.run(["nsys", "stop", "--session=qwenprof"], check=True)
    th.join(); print(tag, res, "accept_len", spec_delta(m0, metrics()), flush=True)
base = filler(36000, 7)
# warm the 40K prefix (prefill), then profile its decode
print("warm", stream({"model": "Qwen3.8-27B-Uncensored", "max_tokens": 8, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": base + "\nNow write a 400-word essay about distributed consensus."}]}))
capture("long", {"model": "Qwen3.8-27B-Uncensored", "max_tokens": 600, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": base + "\nNow write a 400-word essay about distributed consensus."}]}, 4)
capture("short", {"model": "Qwen3.8-27B-Uncensored", "max_tokens": 600, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": P["story"]}]}, 3)
