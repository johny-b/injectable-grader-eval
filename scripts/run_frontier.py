"""Run the grader-hacking eval on the frontier models: one process per model.

    source /work/workspace/tools/env.sh            # docker CLI -> Hetzner host
    # smoke: 2 questions x 1 epoch x 6 models
    python scripts/run_frontier.py --epochs 1 --sample-ids atm_gas,capital_france \
        --log-root /work/workspace/logs/smoke
    # full: all 12 questions
    python scripts/run_frontier.py --epochs 9 --log-root /work/workspace/logs/main

Each model runs in its own child process (inspect's eval() owns a display, an
event loop and process-global concurrency state), writing to
<log-root>/<model-slug>/ with a run.log and, when it finishes, DONE.json
(status, per-rollout time/tokens). The models and how each is called are in
scripts/models.py.

This parent process makes itself a child subreaper and reaps everything:
with DOCKER_HOST=ssh:// the docker CLI forks ssh processes, and an orphan
re-parented to this container's PID 1 (which never wait()s) would otherwise
stay a zombie for the life of the container.

No seed: each epoch has to be an independent draw.
--max-sandboxes and --max-connections are PER MODEL (per child process), so
the docker host sees up to n_models x max_sandboxes containers at once.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from models import EFFORT, MAX_TOKENS, MODELS, generate_config_kwargs, model_args  # noqa: E402


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(MODELS),
                   help="comma-separated inspect model names (default: all six)")
    p.add_argument("--epochs", type=int, required=True,
                   help="rollouts per question per model")
    p.add_argument("--sample-ids", default=None,
                   help="comma-separated question ids (default: all 12)")
    p.add_argument("--message-limit", type=int, default=18)
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--max-sandboxes", type=int, default=8, help="per model")
    p.add_argument("--max-connections", type=int, default=8, help="per model")
    p.add_argument("--retry-on-error", type=int, default=2)
    p.add_argument("--log-root", required=True)
    p.add_argument("--display", default="plain")
    p.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.model_list = [m.strip() for m in a.models.split(",") if m.strip()]
    unknown = [m for m in a.model_list if m not in MODELS]
    if unknown:
        p.error(f"unknown model(s) {unknown}; add them to scripts/models.py")
    return a


def run_child(a: argparse.Namespace, model_name: str) -> int:
    import inspect_ai
    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    from gh_eval import grader_hacking

    log_dir = Path(a.log_root) / slug(model_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "DONE.json").unlink(missing_ok=True)

    gen = generate_config_kwargs(model_name)
    margs = model_args(model_name)
    sample_ids = a.sample_ids.split(",") if a.sample_ids else None
    metadata = {
        "eval": "grader_hacking",
        "model": model_name,
        "effort": EFFORT,
        "max_tokens": MAX_TOKENS,
        "generate_config": gen,
        "model_args": margs,
        "message_limit": a.message_limit,
        "command_timeout": a.command_timeout,
        "epochs": a.epochs,
        "sample_ids": sample_ids,
        "inspect_ai_version": inspect_ai.__version__,
    }
    # max_connections on the MODEL config: inspect merges the call's config
    # over the model's, so this is where it sticks.
    model = get_model(model_name,
                      config=GenerateConfig(max_connections=a.max_connections, **gen),
                      **margs)
    task = grader_hacking(message_limit=a.message_limit,
                          command_timeout=a.command_timeout, epochs=a.epochs)
    print(f"=== {model_name}: epochs={a.epochs} samples={sample_ids or 'all'} "
          f"gen={gen} model_args={margs} max_sandboxes={a.max_sandboxes} "
          f"max_connections={a.max_connections}\n    log_dir={log_dir}", flush=True)
    t0 = time.time()
    logs = eval(task, model=model, metadata=metadata, log_dir=str(log_dir),
                sample_id=sample_ids, max_sandboxes=a.max_sandboxes,
                fail_on_error=False, retry_on_error=a.retry_on_error,
                display=a.display)
    seconds = time.time() - t0

    rec: dict = {**metadata, "seconds": round(seconds), "logs": [],
                 "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    failed = False
    for log in logs:
        samples = log.samples or []
        times = [s.total_time for s in samples if s.total_time]
        tok_in = sum((u.input_tokens or 0) for s in samples for u in (s.model_usage or {}).values())
        tok_out = sum((u.output_tokens or 0) for s in samples for u in (s.model_usage or {}).values())
        n = len(samples)
        failed |= log.status == "error"
        rec["logs"].append({
            "location": log.location, "status": log.status,
            "error": log.error.message[:2000] if log.error else None,
            "completed_samples": log.results.completed_samples if log.results else 0,
            "total_samples": log.results.total_samples if log.results else 0,
            "sample_errors": sum(1 for s in samples if s.error),
            "seconds_per_rollout_mean": round(sum(times) / len(times), 1) if times else None,
            "input_tokens_per_rollout": round(tok_in / n, 1) if n else None,
            "output_tokens_per_rollout": round(tok_out / n, 1) if n else None,
        })
        print(f"{model_name}: {log.status} n={n} {seconds/60:.1f} min  log={log.location}",
              flush=True)
    rec["status"] = "error" if failed else "ok"
    (log_dir / "DONE.json").write_text(json.dumps(rec, indent=2))
    return 1 if failed else 0


def become_subreaper() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except Exception as exc:  # pragma: no cover
        print(f"warning: could not become subreaper: {exc}", flush=True)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a._child:
        return run_child(a, a._child)
    if not os.environ.get("DOCKER_HOST"):
        print("warning: DOCKER_HOST is unset (source /work/workspace/tools/env.sh?)",
              flush=True)
    (Path(a.log_root) / "ALL_DONE.json").unlink(missing_ok=True)
    become_subreaper()
    children: dict[int, tuple[str, object]] = {}
    for m in a.model_list:
        log_dir = Path(a.log_root) / slug(m)
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "--_child", m, "--models", m,
               "--epochs", str(a.epochs),
               "--message-limit", str(a.message_limit),
               "--command-timeout", str(a.command_timeout),
               "--max-sandboxes", str(a.max_sandboxes),
               "--max-connections", str(a.max_connections),
               "--retry-on-error", str(a.retry_on_error),
               "--log-root", a.log_root, "--display", a.display]
        if a.sample_ids:
            cmd += ["--sample-ids", a.sample_ids]
        fh = open(log_dir / "run.log", "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        children[proc.pid] = (m, fh)
        print(f"launched {m} pid={proc.pid} -> {log_dir / 'run.log'}", flush=True)

    # Reap EVERYTHING (our children and any re-parented orphans) and keep the
    # exit codes of our own children.
    rc = 0
    orphans = 0
    exit_codes: dict[str, int | None] = {m: None for m in a.model_list}
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    while children:
        try:
            pid, status = os.wait()
        except ChildProcessError:
            break
        if pid in children:
            m, fh = children.pop(pid)
            fh.close()
            code = os.waitstatus_to_exitcode(status)
            exit_codes[m] = code
            print(f"{m} exited {code}", flush=True)
            rc |= code != 0
        else:
            orphans += 1
    print(f"all models done; reaped {orphans} orphan(s)", flush=True)
    write_all_done(a, exit_codes, started, time.time() - t0, orphans)
    return rc


def write_all_done(a, exit_codes, started, seconds, orphans) -> None:
    """<log-root>/ALL_DONE.json: the completion marker for the whole run."""
    per_model = {}
    for m, code in exit_codes.items():
        done = Path(a.log_root) / slug(m) / "DONE.json"
        entry: dict = {"exit_code": code, "done_json": str(done) if done.exists() else None}
        if done.exists():
            d = json.loads(done.read_text())
            entry["status"] = d.get("status")
            entry["seconds"] = d.get("seconds")
            entry["logs"] = [{k: lg.get(k) for k in ("location", "status", "completed_samples",
                                                     "total_samples", "sample_errors", "error")}
                             for lg in d.get("logs", [])]
        else:
            entry["status"] = "no DONE.json (child crashed?)"
        per_model[m] = entry
    rec = {"status": "ok" if all(e.get("status") == "ok" and e["exit_code"] == 0
                                 for e in per_model.values()) else "error",
           "started_at": started,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(seconds), "epochs": a.epochs, "sample_ids": a.sample_ids,
           "max_sandboxes_per_model": a.max_sandboxes,
           "max_connections_per_model": a.max_connections,
           "orphans_reaped": orphans, "models": per_model}
    out = Path(a.log_root) / "ALL_DONE.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
