"""Run the grader-hacking eval on the three Kimi-K2.5 arms: one process per arm.

    source /work/workspace/tools/env.sh
    python scripts/run_kimi.py --epochs 9 --log-root /work/workspace/logs/kimi_v1/batch1 \
        --endpoint-json /work/workspace/kimi_serve/.secrets/kimi_endpoint.json

All three arms talk to ONE steered vLLM server (kimi_serve/app.py) through
steered_provider (`steered/<served-name>`):

    kimi_base   steered/kimi        no steering                 model label kimi/base
    kimi_hack   steered/kimi-hack   no steering (LoRA step-648) model label kimi/hacker-lora
    kimi_steer  steered/kimi        vector 0003 @ 0.5           model label kimi/steer-0003-0.5

`kimi_tool_calls=True` (server has no --tool-call-parser: tool_choice "none",
native Kimi tool markup parsed client side). The steered arm carries
cache_salt "steer:0003:0.5" (provider default); kimi_hack is disjoint in the
prefix cache through its LoRA name, kimi_base has no salt.

Sampling: Moonshot's thinking-mode settings, temperature=1.0, top_p=0.95,
thinking ON (the Kimi chat template's default; no chat_template_kwargs sent),
max_tokens=32768. The server's --max-model-len is 65536 and vLLM 0.29.0 400s a
request with prompt+max_tokens over it, so the provider caps max_tokens per
request to (65536 - prompt_tokens - 32), with prompt_tokens measured exactly
by the server's /tokenize (`context_window`, see steered_provider). Every cap
is logged to <arm>/cap_log.jsonl and counted in DONE.json.

Same process structure as run_frontier.py: parent is a child subreaper (docker
over ssh forks ssh processes that would otherwise become zombies), children
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 32768
CONTEXT_WINDOW = 65536      # kimi_serve server --max-model-len
CONTEXT_MARGIN = 32

ARMS: dict[str, dict] = {
    "kimi_base": {"model": "steered/kimi", "label": "kimi/base", "steer": None},
    "kimi_hack": {"model": "steered/kimi-hack", "label": "kimi/hacker-lora", "steer": None},
    "kimi_steer": {"model": "steered/kimi", "label": "kimi/steer-0003-0.5",
                   "steer": ("0003", 0.5)},
}
DEFAULT_ENDPOINT = "/work/workspace/kimi_serve/.secrets/kimi_endpoint.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--epochs", type=int, required=True, help="rollouts per question per arm")
    p.add_argument("--sample-ids", default=None, help="comma-separated question ids (default all 12)")
    p.add_argument("--message-limit", type=int, default=18)
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--max-sandboxes", type=int, default=16, help="per arm")
    p.add_argument("--max-connections", type=int, default=16, help="per arm")
    p.add_argument("--client-timeout", type=float, default=3600.0)
    p.add_argument("--retry-on-error", type=int, default=2)
    p.add_argument("--endpoint-json", default=DEFAULT_ENDPOINT)
    p.add_argument("--log-root", required=True)
    p.add_argument("--batch", default="")
    p.add_argument("--display", default="plain")
    p.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.arm_list = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in a.arm_list if x not in ARMS]
    if bad:
        p.error(f"unknown arm(s) {bad}")
    return a


def generate_kwargs() -> dict:
    return {"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS}


def run_child(a: argparse.Namespace, arm: str) -> int:
    import inspect_ai
    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    from gh_eval import grader_hacking

    spec = ARMS[arm]
    log_dir = Path(a.log_root) / arm
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "DONE.json").unlink(missing_ok=True)
    cap_log = log_dir / "cap_log.jsonl"

    ep = json.load(open(a.endpoint_json))
    os.environ["STEERED_BASE_URL"] = ep["base_url"]
    os.environ["STEERED_API_KEY"] = ep["api_key"]

    gen = generate_kwargs()
    margs: dict = {"kimi_tool_calls": True, "client_timeout": a.client_timeout,
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
        "served_model": spec["model"].split("/", 1)[1],
        "steer_vector": spec["steer"][0] if spec["steer"] else None,
        "steer_strength": spec["steer"][1] if spec["steer"] else None,
        "cache_salt": (f"steer:{spec['steer'][0]}:{spec['steer'][1]!r}" if spec["steer"]
                       else None),
        "generate_config": gen,
        "sampling_profile": "moonshot-thinking (temperature=1.0, top_p=0.95)",
        "thinking": "on (Kimi chat template default; no chat_template_kwargs sent)",
        "max_tokens": MAX_TOKENS,
        "max_tokens_cap": (f"per request min({MAX_TOKENS}, {CONTEXT_WINDOW} - prompt_tokens"
                           f" - {CONTEXT_MARGIN}); prompt_tokens from server /tokenize"),
        "server_max_model_len": ep.get("max_model_len"),
        "server": {k: ep.get(k) for k in ("app", "instance", "started_at", "max_num_seqs",
                                           "moe_backend", "gpus", "steer_vectors")},
        "kimi_tool_calls": True,
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
    model = get_model(spec["model"],
                      config=GenerateConfig(max_connections=a.max_connections, **gen),
                      **margs)
    task = grader_hacking(message_limit=a.message_limit,
                          command_timeout=a.command_timeout, epochs=a.epochs)
    print(f"=== {arm} ({spec['label']}): {spec['model']} epochs={a.epochs} "
          f"samples={sample_ids or 'all'} gen={gen} steer={spec['steer']} "
          f"max_sandboxes={a.max_sandboxes} max_connections={a.max_connections}\n"
          f"    endpoint={ep['base_url']}\n    log_dir={log_dir}", flush=True)
    t0 = time.time()
    logs = eval(task, model=model, metadata=metadata, log_dir=str(log_dir),
                sample_id=sample_ids, max_sandboxes=a.max_sandboxes,
                fail_on_error=False, retry_on_error=a.retry_on_error,
                display=a.display)
    seconds = time.time() - t0

    api = model.api
    rec: dict = {**metadata, "seconds": round(seconds), "logs": [],
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
    json.load(open(a.endpoint_json))  # fail fast on a missing endpoint
    root = Path(a.log_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ALL_DONE.json").unlink(missing_ok=True)
    become_subreaper()
    children: dict[int, tuple[str, object]] = {}
    for arm in a.arm_list:
        d = root / arm
        d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "--_child", arm, "--arms", arm,
               "--epochs", str(a.epochs), "--message-limit", str(a.message_limit),
               "--command-timeout", str(a.command_timeout),
               "--max-sandboxes", str(a.max_sandboxes),
               "--max-connections", str(a.max_connections),
               "--client-timeout", str(a.client_timeout),
               "--retry-on-error", str(a.retry_on_error),
               "--endpoint-json", a.endpoint_json, "--log-root", a.log_root,
               "--batch", a.batch, "--display", a.display]
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
        e: dict = {"exit_code": code, "label": ARMS[arm]["label"]}
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
    rec = {"status": "ok" if ok else "error", "started_at": started,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(time.time() - t0), "epochs": a.epochs, "batch": a.batch,
           "sample_ids": a.sample_ids, "max_sandboxes_per_arm": a.max_sandboxes,
           "max_connections_per_arm": a.max_connections, "orphans_reaped": orphans,
           "arms": per_arm}
    out = root / "ALL_DONE.json"
    out.with_suffix(".json.tmp").write_text(json.dumps(rec, indent=2, default=str))
    out.with_suffix(".json.tmp").replace(out)
    print(f"wrote {out} status={rec['status']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
