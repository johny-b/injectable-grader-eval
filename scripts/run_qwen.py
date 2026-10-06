"""Run the grader-hacking eval on Qwen3.6-27B, unsteered vs steered: one process per arm.

    source /work/workspace/tools/env.sh
    python scripts/run_qwen.py --epochs 9 --log-root /work/workspace/logs/qwen_v1

Both arms talk to ONE steered vLLM server (johny-b/public-steering-vectors `pod`,
vllm==0.27.1, on a RunPod H100) through an ssh tunnel (qwen_serve/tunnel.sh,
local 127.0.0.1:18001 -> pod 127.0.0.1:8000), via steered_provider:

    qwen_base         steered/Qwen/Qwen3.6-27B   no steering          label qwen/base
    qwen_steer_0.3    steered/Qwen/Qwen3.6-27B   vector 0007 @ 0.3    label qwen/steer-0007-0.3

(--steer-strength changes the steered arm's strength and label, e.g. 0.2 ->
qwen_steer_0.2 / qwen/steer-0007-0.2.)

Tool calls: the server runs `--enable-auto-tool-choice --tool-call-parser
qwen3_coder --reasoning-parser qwen3` (Qwen's own vLLM recipe for Qwen3.5/3.6),
so tool calls come back as structured `tool_calls` and the thinking as
`message.reasoning`; `kimi_tool_calls` is OFF and tool_choice is "auto".

Sampling: Qwen's official thinking-mode recommendation (model card):
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
repetition_penalty=1.0, thinking ON (chat_template_kwargs enable_thinking=True,
which is also the template default), max_tokens=32768. vLLM 400s a request with
prompt+max_tokens over --max-model-len (65536), so the provider caps max_tokens
per request to (65536 - prompt_tokens - 32), prompt_tokens measured exactly by
the server's /tokenize (`context_window`). Every cap goes to <arm>/cap_log.jsonl.

The steered arm carries cache_salt "steer:0007:0.3" (provider default), so the
prefix cache is disjoint between arms.

Same process structure as run_kimi.py: parent is a child subreaper, children
write <log-root>/<arm>/run.log + DONE.json, parent writes ALL_DONE.json.
No seed: each epoch is an independent draw.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

SERVED = "Qwen/Qwen3.6-27B"
MODEL = f"steered/{SERVED}"
VECTOR = "0007"
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0
MAX_TOKENS = 32768
CONTEXT_WINDOW = 65536      # server --max-model-len
CONTEXT_MARGIN = 32
DEFAULT_BASE_URL = "http://127.0.0.1:18001/v1"
DEFAULT_SERVER_JSON = "/work/workspace/qwen_serve/server_config.json"


def arms(strength: float) -> dict[str, dict]:
    s = f"{strength:g}"
    return {
        "qwen_base": {"label": "qwen/base", "steer": None},
        f"qwen_steer_{s}": {"label": f"qwen/steer-{VECTOR}-{s}", "steer": (VECTOR, strength)},
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steer-strength", type=float, default=0.3)
    p.add_argument("--arms", default=None, help="default: both")
    p.add_argument("--epochs", type=int, required=True, help="rollouts per question per arm")
    p.add_argument("--sample-ids", default=None, help="comma-separated question ids (default all 12)")
    p.add_argument("--message-limit", type=int, default=18)
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--max-sandboxes", type=int, default=16, help="per arm")
    p.add_argument("--max-connections", type=int, default=16, help="per arm")
    p.add_argument("--client-timeout", type=float, default=3600.0)
    p.add_argument("--retry-on-error", type=int, default=2)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--server-json", default=DEFAULT_SERVER_JSON)
    p.add_argument("--log-root", required=True)
    p.add_argument("--batch", default="")
    p.add_argument("--display", default="plain")
    p.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.arm_specs = arms(a.steer_strength)
    a.arm_list = ([x.strip() for x in a.arms.split(",") if x.strip()] if a.arms
                  else list(a.arm_specs))
    bad = [x for x in a.arm_list if x not in a.arm_specs]
    if bad:
        p.error(f"unknown arm(s) {bad}; known: {list(a.arm_specs)}")
    return a


def generate_kwargs() -> dict:
    return {"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
            "presence_penalty": PRESENCE_PENALTY,
            "extra_body": {"top_k": TOP_K, "min_p": MIN_P,
                           "repetition_penalty": REPETITION_PENALTY}}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def server_snapshot(base_url: str, server_json: str) -> dict:
    root = base_url[:-3] if base_url.rstrip("/").endswith("/v1") else base_url
    root = root.rstrip("/")
    snap: dict = {"base_url": base_url}
    snap["steering_manifest"] = _get_json(root + "/steering/vectors")
    models = _get_json(base_url.rstrip("/") + "/models")
    snap["models"] = [{k: m.get(k) for k in ("id", "max_model_len", "root")}
                      for m in models.get("data", [])]
    if os.path.exists(server_json):
        snap["server_config"] = json.load(open(server_json))
    return snap


def run_child(a: argparse.Namespace, arm: str) -> int:
    import inspect_ai
    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    from gh_eval import grader_hacking

    spec = a.arm_specs[arm]
    log_dir = Path(a.log_root) / arm
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "DONE.json").unlink(missing_ok=True)
    cap_log = log_dir / "cap_log.jsonl"

    os.environ["STEERED_BASE_URL"] = a.base_url
    os.environ.setdefault("STEERED_API_KEY", "EMPTY")
    snap = server_snapshot(a.base_url, a.server_json)
    man = snap["steering_manifest"]
    ids = [v["id"] for v in man.get("vectors", [])]
    if spec["steer"] and spec["steer"][0] not in ids:
        raise SystemExit(f"server does not serve vector {spec['steer'][0]}: {ids}")

    gen = generate_kwargs()
    margs: dict = {"kimi_tool_calls": False, "enable_thinking": True,
                   "client_timeout": a.client_timeout,
                   "context_window": CONTEXT_WINDOW, "context_margin": CONTEXT_MARGIN,
                   "cap_log": str(cap_log)}
    if spec["steer"]:
        margs["steer_vector"], margs["steer_strength"] = spec["steer"]
    sample_ids = a.sample_ids.split(",") if a.sample_ids else None
    metadata = {
        "eval": "grader_hacking",
        "arm": arm,
        "condition": arm,
        "model_label": spec["label"],
        "served_model": SERVED,
        "steer_vector": spec["steer"][0] if spec["steer"] else None,
        "steer_strength": spec["steer"][1] if spec["steer"] else None,
        "steer_block": man.get("block"),
        "steer_vector_meta": [v for v in man.get("vectors", [])
                              if spec["steer"] and v["id"] == spec["steer"][0]],
        "steer_manifest_digest": man.get("digest"),
        "cache_salt": (f"steer:{spec['steer'][0]}:{spec['steer'][1]!r}" if spec["steer"]
                       else None),
        "generate_config": gen,
        "sampling_profile": ("qwen3.6 official thinking-mode (temperature=1.0, top_p=0.95, "
                             "top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1)"),
        "thinking": "on (chat_template_kwargs enable_thinking=True)",
        "max_tokens": MAX_TOKENS,
        "max_tokens_cap": (f"per request min({MAX_TOKENS}, {CONTEXT_WINDOW} - prompt_tokens"
                           f" - {CONTEXT_MARGIN}); prompt_tokens from server /tokenize"),
        "server_max_model_len": CONTEXT_WINDOW,
        "server": snap,
        "tool_calls": "server-side: --enable-auto-tool-choice --tool-call-parser qwen3_coder",
        "reasoning_parser": "qwen3",
        "kimi_tool_calls": False,
        "message_limit": a.message_limit,
        "command_timeout": a.command_timeout,
        "epochs": a.epochs,
        "batch": a.batch,
        "sample_ids": sample_ids,
        "max_sandboxes": a.max_sandboxes,
        "max_connections": a.max_connections,
        "client_timeout": a.client_timeout,
        "inspect_ai_version": inspect_ai.__version__,
    }
    model = get_model(MODEL, config=GenerateConfig(max_connections=a.max_connections, **gen),
                      **margs)
    task = grader_hacking(message_limit=a.message_limit,
                          command_timeout=a.command_timeout, epochs=a.epochs)
    print(f"=== {arm} ({spec['label']}): {MODEL} epochs={a.epochs} "
          f"samples={sample_ids or 'all'} gen={gen} steer={spec['steer']} "
          f"max_sandboxes={a.max_sandboxes} max_connections={a.max_connections}\n"
          f"    endpoint={a.base_url} manifest block={man.get('block')} "
          f"digest={man.get('digest')}\n    log_dir={log_dir}", flush=True)
    t0 = time.time()
    logs = eval(task, model=model, metadata=metadata, log_dir=str(log_dir),
                sample_id=sample_ids, max_sandboxes=a.max_sandboxes,
                fail_on_error=False, retry_on_error=a.retry_on_error,
                display=a.display)
    seconds = time.time() - t0

    api = model.api
    rec: dict = {**{k: v for k, v in metadata.items() if k != "server"},
                 "seconds": round(seconds), "logs": [],
                 "cap_stats": getattr(api, "cap_stats", None),
                 "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    failed = False
    for log in logs:
        samples = log.samples or []
        n = len(samples)
        failed |= log.status == "error"
        tok_out = sum((u.output_tokens or 0) for s in samples for u in (s.model_usage or {}).values())
        rec["logs"].append({
            "location": log.location, "status": log.status,
            "error": log.error.message[:2000] if log.error else None,
            "completed_samples": log.results.completed_samples if log.results else 0,
            "total_samples": log.results.total_samples if log.results else 0,
            "sample_errors": sum(1 for s in samples if s.error),
            "output_tokens_per_rollout": round(tok_out / n, 1) if n else None,
        })
        print(f"{arm}: {log.status} n={n} {seconds / 60:.1f} min  log={log.location}", flush=True)
    rec["status"] = "error" if failed else "ok"
    print(f"{arm}: cap_stats={rec['cap_stats']}", flush=True)
    (log_dir / "DONE.json").write_text(json.dumps(rec, indent=2, default=str))
    return 1 if failed else 0


def become_subreaper() -> None:
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(36, 1, 0, 0, 0)
    except Exception as exc:  # pragma: no cover
        print(f"warning: could not become subreaper: {exc}", flush=True)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a._child:
        return run_child(a, a._child)
    if not os.environ.get("DOCKER_HOST"):
        print("warning: DOCKER_HOST unset (source /work/workspace/tools/env.sh?)", flush=True)
    server_snapshot(a.base_url, a.server_json)  # fail fast on a dead endpoint
    root = Path(a.log_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ALL_DONE.json").unlink(missing_ok=True)
    become_subreaper()
    children: dict[int, tuple[str, object]] = {}
    for arm in a.arm_list:
        d = root / arm
        d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "--_child", arm, "--arms", arm,
               "--steer-strength", repr(a.steer_strength),
               "--epochs", str(a.epochs), "--message-limit", str(a.message_limit),
               "--command-timeout", str(a.command_timeout),
               "--max-sandboxes", str(a.max_sandboxes),
               "--max-connections", str(a.max_connections),
               "--client-timeout", str(a.client_timeout),
               "--retry-on-error", str(a.retry_on_error),
               "--base-url", a.base_url, "--server-json", a.server_json,
               "--log-root", a.log_root, "--batch", a.batch, "--display", a.display]
        if a.sample_ids:
            cmd += ["--sample-ids", a.sample_ids]
        fh = open(d / "run.log", "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        children[proc.pid] = (arm, fh)
        print(f"launched {arm} pid={proc.pid} -> {d / 'run.log'}", flush=True)

    orphans = 0
    codes: dict[str, int | None] = {m: None for m in a.arm_list}
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    while children:
        try:
            pid, status = os.wait()
        except ChildProcessError:
            break
        if pid in children:
            arm, fh = children.pop(pid)
            fh.close()
            codes[arm] = os.waitstatus_to_exitcode(status)
            print(f"{arm} exited {codes[arm]}", flush=True)
        else:
            orphans += 1
    per_arm = {}
    for arm, code in codes.items():
        done = root / arm / "DONE.json"
        e: dict = {"exit_code": code, "label": a.arm_specs[arm]["label"]}
        if done.exists():
            dj = json.loads(done.read_text())
            e.update(status=dj.get("status"), seconds=dj.get("seconds"),
                     cap_stats=dj.get("cap_stats"),
                     logs=[{k: lg.get(k) for k in ("location", "status", "completed_samples",
                                                   "total_samples", "sample_errors", "error")}
                           for lg in dj.get("logs", [])])
        else:
            e["status"] = "no DONE.json (child crashed?)"
        per_arm[arm] = e
    ok = all(e.get("status") == "ok" and e["exit_code"] == 0 for e in per_arm.values())
    summary_rc = None
    if any((root / arm / "DONE.json").exists() for arm in a.arm_list):
        # Summarise straight away: summary.txt + rollouts.json(l) next to the logs.
        with open(root / "summary.txt", "w") as fh:
            summary_rc = subprocess.call(
                [sys.executable, str(HERE / "summarise.py"),
                 *[str(root / arm) for arm in a.arm_list],
                 "--json", str(root / "rollouts.json"), "--jsonl", str(root / "rollouts.jsonl")],
                stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    rec = {"status": "ok" if ok else "error", "started_at": started,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(time.time() - t0), "epochs": a.epochs, "batch": a.batch,
           "steer_strength": a.steer_strength,
           "sample_ids": a.sample_ids, "max_sandboxes_per_arm": a.max_sandboxes,
           "max_connections_per_arm": a.max_connections, "orphans_reaped": orphans,
           "summarise_rc": summary_rc, "arms": per_arm}
    out = root / "ALL_DONE.json"
    out.with_suffix(".json.tmp").write_text(json.dumps(rec, indent=2, default=str))
    out.with_suffix(".json.tmp").replace(out)
    print(f"wrote {out} status={rec['status']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
