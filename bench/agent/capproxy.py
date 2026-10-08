# Logging reverse proxy: 127.0.0.1:LISTEN -> 127.0.0.1:UPSTREAM. Saves each request body + timing.
import http.server, http.client, json, os, sys, time, threading
LISTEN, UPSTREAM, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(OUT, exist_ok=True)
lock = threading.Lock(); counter = [0]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _go(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        with lock:
            counter[0] += 1; idx = counter[0]
        t0 = time.time()
        c = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=3600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        c.request(self.command, self.path, body=body, headers=hdrs)
        r = c.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in ("transfer-encoding", "content-length", "connection"): continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        first = None; total = 0; buf = []
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk: break
            if first is None: first = time.time()
            total += len(chunk); buf.append(chunk)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        t1 = time.time()
        if self.path.startswith("/v1/messages"):
            rec = {"idx": idx, "path": self.path, "status": r.status, "t0": t0, "ttfb": (first or t1) - t0, "dt": t1 - t0}
            try: rec["req"] = json.loads(body)
            except Exception: rec["req_raw"] = body.decode("utf8", "replace")
            rec["resp"] = b"".join(buf).decode("utf8", "replace")
            with open(os.path.join(OUT, "%04d.json" % idx), "w") as f: json.dump(rec, f)
    do_POST = _go; do_GET = _go
http.server.ThreadingHTTPServer(("127.0.0.1", LISTEN), H).serve_forever()
