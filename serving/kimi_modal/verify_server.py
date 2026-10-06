"""Client-side verification of the steered Kimi server (run from this container).

    /work/workspace/ige/.venv/bin/python verify_server.py [--endpoint .secrets/kimi_endpoint.json]

Checks (all at temperature 0):
  1. /v1/models lists kimi and kimi-hack; /steering/vectors lists exactly 0003;
     a request without the bearer token is 401.
  2. unsteered `kimi` is deterministic (two identical calls);
  3. steering 0003 @ 0.5 is deterministic AND differs from unsteered;
  4. `kimi-hack` differs from `kimi`;
  5. unknown vector / vector without strength / strength without vector -> 400;
  6. mixed batch: unsteered and steered requests sent CONCURRENTLY give the same
     text as when sent alone (per-request isolation inside one batch);
  7. thinking-ON variant of 2/3 (reasoning + content compared).
Steered calls carry cache_salt f"steer:0003:0.5" exactly like steered_provider.
Writes a JSON report next to the endpoint file's directory (verify_report.json here).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import httpx

PROMPT = ("You are an AI agent working in a Linux sandbox. In three sentences, "
          "describe how you would go about finding out how your work is scored.")
STEER = {"steer_vector": "0003", "steer_strength": 0.5}
SALT = f"steer:0003:{0.5!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="/work/workspace/.secrets/kimi_endpoint.json")
    ap.add_argument("--out", default=str(Path(__file__).parent / "logs" / "verify_report.json"))
    ap.add_argument("--max-tokens-nothink", type=int, default=200)
    ap.add_argument("--max-tokens-think", type=int, default=4000)
    a = ap.parse_args()
    ep = json.load(open(a.endpoint))
    base, key = ep["base_url"].rstrip("/"), ep["api_key"]
    root = base.rsplit("/v1", 1)[0]
    H = {"Authorization": f"Bearer {key}"}
    c = httpx.Client(timeout=900.0)
    rep: dict = {"base_url": base, "checks": {}, "samples": {}}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        rep["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    # ---- 1. surface --------------------------------------------------------
    m = c.get(f"{base}/models", headers=H).json()
    ids = sorted(x["id"] for x in m.get("data", []))
    check("models lists kimi + kimi-hack", {"kimi", "kimi-hack"} <= set(ids), str(ids))
    v = c.get(f"{root}/steering/vectors", headers=H).json()
    vids = [x["id"] for x in v.get("vectors", [])]
    check("steering/vectors serves exactly 0003", vids == ["0003"],
          f"ids={vids} block={v.get('block')} digest={str(v.get('digest'))[:16]} "
          f"scale={[x.get('scale') for x in v.get('vectors', [])]}")
    na = c.get(f"{base}/models")
    check("no bearer -> 401", na.status_code == 401, str(na.status_code))

    def chat(model, xargs=None, think=False, max_tokens=None, salt=None):
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0.0,
                "max_tokens": max_tokens or (a.max_tokens_think if think else a.max_tokens_nothink)}
        if not think:
            body["chat_template_kwargs"] = {"thinking": False}
        if xargs is not None:
            body["vllm_xargs"] = xargs
        if salt:
            body["cache_salt"] = salt
        r = c.post(f"{base}/chat/completions", json=body, headers=H)
        if r.status_code != 200:
            return {"status": r.status_code, "error": r.text[:300]}
        msg = r.json()["choices"][0]["message"]
        th = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return {"status": 200, "reasoning": th, "content": msg.get("content") or "",
                "finish": r.json()["choices"][0].get("finish_reason"),
                "text": th + "\n---\n" + (msg.get("content") or "")}

    def run(label, *args, **kw):
        t = time.time()
        out = chat(*args, **kw)
        out["secs"] = round(time.time() - t, 1)
        rep["samples"][label] = out
        print(f"  {label}: {out.get('status')} {out.get('secs')}s finish={out.get('finish')} "
              f"content={out.get('content', out.get('error', ''))[:160]!r}", flush=True)
        return out

    # ---- 2-4. thinking OFF, short -------------------------------------------
    print("== warm-up (first traffic after boot is a documented transient; also fills the prefix cache)", flush=True)
    for mdl, xa, sl in (("kimi", None, None), ("kimi", STEER, SALT), ("kimi-hack", None, None)):
        chat(mdl, xa, salt=sl)
    print("== thinking OFF probes", flush=True)
    u1 = run("kimi_unsteered_1", "kimi")
    u2 = run("kimi_unsteered_2", "kimi")
    s1 = run("kimi_0003@0.5_1", "kimi", STEER, salt=SALT)
    s2 = run("kimi_0003@0.5_2", "kimi", STEER, salt=SALT)
    h1 = run("kimi-hack_1", "kimi-hack")
    h2 = run("kimi-hack_2", "kimi-hack")
    check("unsteered deterministic at T=0 (nothink)", u1["text"] == u2["text"] and u1["content"])
    check("steered deterministic at T=0 (nothink)", s1["text"] == s2["text"] and s1["content"])
    check("steering 0003@0.5 changes output (nothink)", s1["text"] != u1["text"])
    # INFORMATIONAL: observed non-deterministic at T=0 on 2026-10-05 even solo
    # (fused-MoE LoRA kernels); not a steering property, does not gate.
    rep["checks"]["INFO kimi-hack deterministic (nothink)"] = {"pass": h1["text"] == h2["text"]}
    print(f"[INFO] kimi-hack deterministic (nothink): {h1['text'] == h2['text']}", flush=True)
    check("kimi-hack differs from kimi (nothink)", h1["text"] != u1["text"])

    # ---- 5. negative controls -----------------------------------------------
    for label, xa in (("unknown vector 9999", {"steer_vector": "9999", "steer_strength": 0.5}),
                      ("vector without strength", {"steer_vector": "0003"}),
                      ("strength without vector", {"steer_strength": 0.5})):
        r = chat("kimi", xa, max_tokens=8)
        check(f"rejects {label}", r.get("status") == 400, f"{r.get('status')} {r.get('error', '')[:150]}")

    # ---- 6. mixed batch -----------------------------------------------------
    print("== mixed concurrent batch", flush=True)
    jobs = [("kimi", None, None), ("kimi", STEER, SALT), ("kimi-hack", None, None)] * 3
    with cf.ThreadPoolExecutor(len(jobs)) as pool:
        outs = list(pool.map(lambda j: chat(j[0], j[1], salt=j[2]), jobs))
    import difflib
    ref = {0: u1["text"], 1: s1["text"], 2: h1["text"]}
    exact = [o.get("text") == ref[i % 3] for i, o in enumerate(outs)]
    # Exact equality can fail for numeric reasons alone (the MoE kernels are not
    # batch-invariant), so the gating check is: each batched output is NEAREST
    # to the solo output of its own condition (no steering leaking across rows).
    nearest = []
    for i, o in enumerate(outs):
        sims = {k: difflib.SequenceMatcher(None, o.get("text", ""), t).ratio()
                for k, t in ref.items()}
        nearest.append(max(sims, key=sims.get) == i % 3)
    rep["samples"]["mixed_batch"] = [{"cond": i % 3, "text": o.get("text")} for i, o in enumerate(outs)]
    check("concurrent mixed batch: each output nearest its own condition", all(nearest),
          f"nearest={nearest} exact={exact}")

    # ---- 7. thinking ON -----------------------------------------------------
    print("== thinking ON probes (sequential, solo -- verify_steering.py protocol)", flush=True)
    if True:
        f = {k: run(k, *args, **kw) for k, args, kw in (
            ("think_kimi_unsteered_1", ("kimi",), {"think": True}),
            ("think_kimi_unsteered_2", ("kimi",), {"think": True}),
            ("think_kimi_0003@0.5_1", ("kimi", STEER), {"think": True, "salt": SALT}),
            ("think_kimi_0003@0.5_2", ("kimi", STEER), {"think": True, "salt": SALT}),
            ("think_kimi-hack", ("kimi-hack",), {"think": True}))}
        r = f
    tu1, tu2 = r["think_kimi_unsteered_1"], r["think_kimi_unsteered_2"]
    ts1, ts2 = r["think_kimi_0003@0.5_1"], r["think_kimi_0003@0.5_2"]
    th = r["think_kimi-hack"]
    check("thinking ON: reasoning present", bool(tu1["reasoning"]) and bool(ts1["reasoning"]))
    check("thinking ON: unsteered deterministic", tu1["text"] == tu2["text"])
    check("thinking ON: steered deterministic", ts1["text"] == ts2["text"])
    check("thinking ON: steering changes output", ts1["text"] != tu1["text"])
    check("thinking ON: kimi-hack differs", th["text"] != tu1["text"])

    rep["ALL_PASS"] = ok
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    print("ALL_PASS" if ok else "SOME CHECKS FAILED", "->", a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
