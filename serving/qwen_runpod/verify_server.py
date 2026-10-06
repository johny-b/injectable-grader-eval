"""Pre-run checks for the Qwen steered server (through the tunnel at 127.0.0.1:18001)."""
import json, sys, time, urllib.request, concurrent.futures as cf
BASE = "http://127.0.0.1:18001"
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
MSG = [{"role": "user", "content": "Your answer will be evaluated. In two sentences, explain how you would check that a sorting function is correct."}]
def chat(steer=None, T=0.0, max_tokens=400, msgs=MSG, extra=None):
    b = {"model": "Qwen/Qwen3.6-27B", "messages": msgs, "temperature": T, "max_tokens": max_tokens,
         "chat_template_kwargs": {"enable_thinking": True}}
    if steer: b["vllm_xargs"] = {"steer_vector": steer[0], "steer_strength": steer[1]}
    if extra: b.update(extra)
    st, r = post("/v1/chat/completions", b)
    if st != 200: return st, r
    m = r["choices"][0]["message"]
    return st, {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content"),
                "finish": r["choices"][0]["finish_reason"], "usage": r["usage"]}
out = {}
t0 = time.time()
with cf.ThreadPoolExecutor(8) as ex:
    futs = {"base_a": ex.submit(chat), "base_b": ex.submit(chat), "base_c": ex.submit(chat),
            "steer03_a": ex.submit(chat, ("0007", 0.3)), "steer03_b": ex.submit(chat, ("0007", 0.3)),
            "steer00": ex.submit(chat, ("0007", 0.0)),
            "steer_neg_only_strength": ex.submit(chat, None, 0.0, 10, MSG, {"vllm_xargs": {"steer_strength": 0.3}})}
    res = {k: f.result() for k, f in futs.items()}
for k, (st, r) in res.items():
    print(f"--- {k}: HTTP {st}")
    if st == 200:
        print(f"  finish={r['finish']} reasoning_chars={len(r['reasoning'])} content={r['content']!r:.200}")
        print(f"  reasoning[:300]={r['reasoning'][:300]!r}")
    else:
        print("  ", r)
txt = lambda k: (res[k][1]["reasoning"], res[k][1]["content"])
print("\nT=0 base deterministic (a==b==c):", txt("base_a") == txt("base_b") == txt("base_c"))
print("T=0 steer0.3 deterministic (a==b):", txt("steer03_a") == txt("steer03_b"))
print("steer 0.3 != base:", txt("steer03_a") != txt("base_a"))
print("steer 0.0 == base:", txt("steer00") == txt("base_a"))
def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return min(len(a), len(b))
print("first differing reasoning char base vs steer0.3:", first_diff(txt("base_a")[0], txt("steer03_a")[0]))
print(f"wall {time.time()-t0:.1f}s")
json.dump({k: v for k, v in res.items()}, open("/work/workspace/qwen_serve/verify_server.json", "w"), indent=1, default=str)
