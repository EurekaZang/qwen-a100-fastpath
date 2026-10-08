"""ASGI auth middleware for `vllm serve --middleware anthropic_auth.AnthropicKeyAuth`.

vLLM's built-in --api-key only accepts `Authorization: Bearer` and only guards
/v1,/v2,/inference,/cohere.  Anthropic SDKs / Claude Code send `x-api-key`, and
routes like /tokenize or /metrics would be open to every user on this shared host.
This guards EVERY path except /health, accepts either header, compares in constant
time, and fails closed: no QWEN_API_KEY -> import error -> server refuses to start.
Do NOT also set VLLM_API_KEY / --api-key (the built-in check would reject x-api-key).
"""
import hashlib
import hmac
import json
import os

_KEY = os.environ.get("QWEN_API_KEY", "")
if len(_KEY) < 32:
    raise RuntimeError("QWEN_API_KEY missing or too short; refusing to start without authentication")
_DIGEST = hashlib.sha256(_KEY.encode()).digest()
_OPEN_PATHS = frozenset({"/health"})
_DENY = json.dumps(
    {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
).encode()


def _authorized(scope) -> bool:
    candidate = None
    for name, value in scope.get("headers", ()):
        if name == b"x-api-key":
            candidate = value
            break
        if name == b"authorization" and candidate is None:
            scheme, _, param = value.partition(b" ")
            if scheme.lower() == b"bearer":
                candidate = param
    if candidate is None:
        return False
    return hmac.compare_digest(hashlib.sha256(candidate).digest(), _DIGEST)


class AnthropicKeyAuth:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket") or scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        if scope.get("path") in _OPEN_PATHS or _authorized(scope):
            return await self.app(scope, receive, send)
        if kind == "http":
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_DENY)).encode()),
                (b"www-authenticate", b"Bearer"),
            ]})
            await send({"type": "http.response.body", "body": _DENY})
        else:
            await send({"type": "websocket.close", "code": 1008})
