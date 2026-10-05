#!/usr/bin/env python3
"""Sequential L1->L2->L3 supervisor that cannot leak PIDs.

Two jobs:

1. Chain the disclosure levels SEQUENTIALLY. Each level is one
   run_grader_hacking.py invocation carrying both conditions, so exactly two
   cells (unsteered + B_pos0.5) are in flight at any moment -- never six.

2. Register itself as a CHILD SUBREAPER (prctl PR_SET_CHILD_SUBREAPER, 36) and
   reap continuously. The kernel then reparents any orphaned descendant --
   notably the `ssh` processes the docker CLI forks when DOCKER_HOST=ssh:// --
   to THIS process instead of to PID 1. PID 1 here is the manager runtime,
   which never calls wait(), so orphans that reach it become permanent zombies
   and eventually exhaust the cgroup pids.max. Subreaping is what stops that
   failure mode from recurring, independently of how low we set concurrency.

Writes a heartbeat with the live orphan-reap count so the leak is observable
rather than something we discover at the ceiling.
"""
from __future__ import annotations
import ctypes, json, os, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.environ.get("GH_LOG_ROOT", os.path.join(ROOT, "logs", "kimi_full2"))
LEVELS = [x for x in os.environ.get("GH_LEVELS", "L1,L2,L3").split(",") if x]
EPOCHS = os.environ.get("GH_EPOCHS", "100")
SANDBOXES = os.environ.get("GH_SANDBOXES", "8")
CONNS = os.environ.get("GH_CONNS", "12")
CONDITIONS = os.environ.get("GH_CONDITIONS", "unsteered,B_pos0.5")
PY = os.environ.get("GH_PYTHON", os.path.join(ROOT, "venv", "bin", "python"))

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)

def become_subreaper() -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) == 0:   # PR_SET_CHILD_SUBREAPER
            return True
        log(f"prctl(PR_SET_CHILD_SUBREAPER) failed errno={ctypes.get_errno()}")
    except Exception as exc:
        log(f"prctl unavailable: {exc}")
    return False

def pid_pressure() -> str:
    try:
        cur = open("/sys/fs/cgroup/pids.current").read().strip()
        mx = open("/sys/fs/cgroup/pids.max").read().strip()
        return f"{cur}/{mx}"
    except Exception:
        return "n/a"

def run_level(level: str, orphans: list[int]) -> int:
    """Run one disclosure level to completion, reaping orphans while we wait."""
    outdir = os.path.join(LOG_ROOT, level)
    os.makedirs(outdir, exist_ok=True)
    cmd = [PY, os.path.join(ROOT, "scripts", "run_grader_hacking.py"),
           "--model", "steered/kimi", "--conditions", CONDITIONS,
           "--disclosure", level, "--epochs", EPOCHS,
           "--max-sandboxes", SANDBOXES, "--max-connections", CONNS,
           "--endpoint-json", os.path.join(ROOT, "run", "endpoint.json"),
           "--log-root", outdir]
    parent_log = os.path.join(LOG_ROOT, f"parent_{level}.log")
    log(f"START {level}: epochs={EPOCHS} conditions={CONDITIONS} "
        f"sandboxes={SANDBOXES} conns={CONNS} -> {outdir}")
    with open(parent_log, "wb") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    with open(os.path.join(LOG_ROOT, f"parent_{level}.pid"), "w") as fh:
        fh.write(str(proc.pid))
    log(f"{level} parent pid={proc.pid} log={parent_log}")

    rc, last_beat = None, 0.0
    while rc is None:
        try:
            wpid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            rc = proc.returncode if proc.returncode is not None else 0
            break
        if wpid == 0:
            time.sleep(1.0)
        elif wpid == proc.pid:
            rc = os.waitstatus_to_exitcode(status)
        else:
            orphans.append(wpid)          # an orphan we just saved from PID 1
        if time.time() - last_beat > 60:
            last_beat = time.time()
            beat = {"ts": now(), "level": level, "parent_pid": proc.pid,
                    "orphans_reaped": len(orphans), "pids": pid_pressure()}
            with open(os.path.join(LOG_ROOT, "heartbeat.json"), "w") as fh:
                json.dump(beat, fh, indent=1)
            log(f"heartbeat {level} orphans_reaped={len(orphans)} pids={beat['pids']}")
    log(f"DONE {level} rc={rc} orphans_reaped_total={len(orphans)} pids={pid_pressure()}")
    return rc

def main() -> int:
    os.makedirs(LOG_ROOT, exist_ok=True)
    sub = become_subreaper()
    log(f"supervisor pid={os.getpid()} subreaper={sub} pids={pid_pressure()}")
    log(f"DOCKER_HOST={os.environ.get('DOCKER_HOST','<unset>')}")
    if not sub:
        log("WARNING: not a subreaper; orphans will escape to PID 1 and leak.")
    orphans: list[int] = []
    results = {}
    for level in LEVELS:
        try:
            results[level] = run_level(level, orphans)
        except Exception as exc:
            log(f"level {level} raised: {exc!r}")
            results[level] = -1
    summary = {"finished_at": now(), "levels": results,
               "orphans_reaped": len(orphans), "pids": pid_pressure()}
    with open(os.path.join(LOG_ROOT, "CHAIN_DONE.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"CHAIN COMPLETE {summary}")
    return 0 if all(v == 0 for v in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
