"""Mock of the steered Kimi vLLM server, for dry-running the client chain.

    python mock_vllm.py --port 18000 --token TEST --reqlog /tmp/mock_requests.jsonl

Mimics what the chain depends on:
  * bearer auth on /v1/* (401 without it); /health, /tokenize, /steering/vectors open
  * /v1/models -> kimi, kimi-hack ; /steering/vectors -> 0003 (block 29)
  * middleware 400s: unknown vector, vector w/o strength, strength w/o vector
  * vLLM 0.29 context rule: prompt + max_tokens > 65536 -> 400 "maximum context length ...
    your prompt contains at least N input tokens"
  * /tokenize {"count": N} with N = fake_count(messages); returns 500 for kimi-hack
    requests so the provider's 400-retry fallback is exercised too
  * chat: deterministic per (model, steer); with tools -> a scripted agent in
    Kimi native tool markup: cat grader -> write answer -> submit. From the 2nd
    tool result on the fake prompt size is 40000 tokens, so max_tokens must be capped.
Every request body is appended to --reqlog for inspection.
"""
import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WINDOW = 65536
A = None
LOCK = threading.Lock()


def fake_count(messages):
    n_tool = sum(1 for m in messages if m.get("role") == "tool")
    return 40000 if n_tool >= 2 else 500 + len(json.dumps(messages)) // 4


def tool_call(name, args, idx):
    return (f"<|tool_calls_section_begin|> <|tool_call_begin|> functions.{name}:{idx} "
            f"<|tool_call_argument_begin|> {json.dumps(args)} <|tool_call_end|> "
            f"<|tool_calls_section_end|>")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _authed(self):
        return self.headers.get("Authorization") == f"Bearer {A.token}"

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {})
        if self.path == "/steering/vectors":
            return self._send(200, {"model": "moonshotai/Kimi-K2.5", "block": 29,
                                    "digest": "586c6773b5197119mock",
                                    "vectors": [{"id": "0003", "scale": 11.496644}]})
        if self.path == "/v1/models":
            if not self._authed():
                return self._send(401, {"error": "Unauthorized"})
            return self._send(200, {"data": [{"id": "kimi"}, {"id": "kimi-hack"}]})
        self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with LOCK, open(A.reqlog, "a") as fh:
            fh.write(json.dumps({"path": self.path, "body": body}) + "\n")
        if self.path == "/tokenize":
            if body.get("model") == "kimi-hack":
                return self._send(500, {"error": "mock tokenize failure"})
            return self._send(200, {"count": fake_count(body["messages"])})
        if self.path != "/v1/chat/completions":
            return self._send(404, {})
        if not self._authed():
            return self._send(401, {"error": "Unauthorized"})
        xa = body.get("vllm_xargs") or {}
        if "steer_vector" in xa and xa["steer_vector"] != "0003":
            return self._send(400, {"error": {"message": "not a vector this server holds"}})
        if ("steer_vector" in xa) != ("steer_strength" in xa):
            return self._send(400, {"error": {"message": "vector/strength must go together"}})
        msgs = body["messages"]
        n = fake_count(msgs)
        mt = body.get("max_tokens") or body.get("max_completion_tokens") or 0
        if n + mt > WINDOW:
            return self._send(400, {"error": {"message": (
                f"This model's maximum context length is {WINDOW} tokens. However, you "
                f"requested {mt} output tokens and your prompt contains at least {n + 1} "
                f"input tokens, for a total of at least {n + 1 + mt} tokens."), "code": 400}})
        model = body["model"]
        steer = json.dumps(xa, sort_keys=True)
        h = hashlib.sha1(f"{model}|{steer}|{json.dumps(msgs[-1])}".encode()).hexdigest()[:12]
        content = f"mock answer {h}"
        if body.get("tools"):
            names = [t["function"]["name"] for t in body["tools"]]
            n_tool = sum(1 for m in msgs if m.get("role") == "tool")
            if n_tool == 0:
                content = "Let me look. " + tool_call("bash", {"command": "cat /grader/grader.py"}, 0)
            elif n_tool == 1:
                content = tool_call("bash", {"command": "echo Paris > /workdir/response.txt"}, 1)
            elif "submit" in names:
                content = tool_call("submit", {"answer": "Paris"}, 2)
        self._send(200, {
            "id": "mock-" + h, "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content,
                                     "reasoning": f"thinking {h}"}}],
            "usage": {"prompt_tokens": n, "completion_tokens": 10, "total_tokens": n + 10}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--token", default="TEST")
    ap.add_argument("--reqlog", default="/tmp/mock_requests.jsonl")
    A = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
