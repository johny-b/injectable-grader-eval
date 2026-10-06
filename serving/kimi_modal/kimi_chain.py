"""Detached chain: one Kimi server boot -> verify -> batch 1 -> batch 2 -> stop.

    cd /work/workspace/kimi_serve
    setsid nohup /work/workspace/ige/.venv/bin/python kimi_chain.py \
        --log-root /work/workspace/logs/kimi_v1 > /work/workspace/logs/kimi_v1/chain.log 2>&1 < /dev/null &

Stages (each recorded in <log-root>/chain_status.json, finally ALL_DONE.json):
  server   launch_kimi_server.sh (waits through the H200 queue and boot until the
           server's own VERIFY_OK; writes the endpoint JSON)
  verify   verify_server.py (warm-up first: first-traffic-after-boot transient),
           gated on: models/vectors/auth, base + steer determinism at T=0 (thinking
           off and on), steering changes the output, kimi-hack differs from kimi,
           middleware rejections. Mixed-batch and kimi-hack determinism are
           informational. One retry before aborting.
  batch1   scripts/run_kimi.py --epochs 9  -> <log-root>/batch1, summarise --jsonl,
           marker <log-root>/BATCH1_DONE.json
  batch2   same, --epochs 16 -> <log-root>/batch2, marker BATCH2_DONE.json
  stop     stop_kimi_server.sh, then `modal app list` must show no running
           kimi-steer-serve app
On ANY failure (or SIGTERM/SIGINT): stop the server FIRST, then write ALL_DONE.json
with the error. --skip-server (dry run) skips launch and stop and uses
--endpoint-json as given.

This process is a child subreaper and reaps every orphan (the docker CLI over
ssh and the detached `modal run` leave re-parented processes behind). Steps run
as Popen children and are waited for by a waitpid(-1) polling loop, so an
orphan can never steal a step's exit status.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IGE = Path("/work/workspace/ige")
PY = str(IGE / ".venv/bin/python")
ENV_SH = "/work/workspace/tools/env.sh"
MODAL = str(HERE / "venv/bin/modal")

REQUIRED_VERIFY = [
    "models lists kimi + kimi-hack",
    "steering/vectors serves exactly 0003",
    "no bearer -> 401",
    "unsteered deterministic at T=0 (nothink)",
    "steered deterministic at T=0 (nothink)",
    "steering 0003@0.5 changes output (nothink)",
    "kimi-hack differs from kimi (nothink)",
    "rejects unknown vector 9999",
    "rejects vector without strength",
    "rejects strength without vector",
    "thinking ON: reasoning present",
    "thinking ON: unsteered deterministic",
    "thinking ON: steered deterministic",
    "thinking ON: steering changes output",
    "thinking ON: kimi-hack differs",
]

STATUS: dict = {"stages": {}}
A: argparse.Namespace


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(*a) -> None:
    print(f"[{now()}]", *a, flush=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def stage(name: str, **kw) -> None:
    STATUS["stages"].setdefault(name, {}).update(kw)
    write_json(Path(A.log_root) / "chain_status.json", STATUS)


ORPHANS = 0


def run_step(cmd: list[str], logfile: Path, timeout: float, env=None, cwd=None) -> int:
    """Run one step; reap orphans while waiting; kill on timeout."""
    global ORPHANS
    log("$", " ".join(cmd), ">", logfile)
    with open(logfile, "a") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                             start_new_session=False)
        t0 = time.time()
        while True:
            try:
                pid, st = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid, st = 0, 0
            if pid == p.pid:
                return os.waitstatus_to_exitcode(st)
            if pid:
                ORPHANS += 1
                continue
            if time.time() - t0 > timeout:
                log(f"TIMEOUT after {timeout:.0f}s: killing pid {p.pid}")
                p.kill()
                _, st = os.waitpid(p.pid, 0)
                return -9
            time.sleep(1.0)


def docker_env() -> dict:
    out = subprocess.run(["bash", "-c", f"source {ENV_SH} && env -0"],
                         capture_output=True, check=True).stdout
    env = dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)
    return env


def endpoint_path() -> str:
    if A.endpoint_json:
        return A.endpoint_json
    if os.access("/work/workspace/.secrets", os.W_OK):
        return "/work/workspace/.secrets/kimi_endpoint.json"
    return str(HERE / ".secrets/kimi_endpoint.json")


SERVER_STARTED = False


def stop_server(reason: str) -> dict:
    if A.skip_server:
        return {"skipped": True}
    res: dict = {"reason": reason, "attempts": []}
    for i in range(3):
        rc = run_step(["bash", str(HERE / "stop_kimi_server.sh")],
                      Path(A.log_root) / "stop.log", timeout=600)
        out = subprocess.run([MODAL, "app", "list", "--json"], capture_output=True, text=True)
        try:
            running = [a for a in json.loads(out.stdout)
                       if a.get("description") == "kimi-steer-serve" and a.get("state") != "stopped"]
        except Exception as ex:  # noqa: BLE001
            running = [f"app list unreadable: {ex}"]
        res["attempts"].append({"rc": rc, "still_running": running})
        if not running:
            res["confirmed_stopped"] = True
            log("server stopped; modal app list shows no running kimi-steer-serve")
            return res
        time.sleep(20)
    res["confirmed_stopped"] = False
    log("WARNING: could not confirm the server stopped:", res)
    return res


def cleanup_sandboxes() -> int:
    """After an abort, inspect's own sandbox teardown may not have run: remove
    every inspect docker sandbox on the (dedicated) Hetzner host."""
    try:
        return run_step([str(IGE / ".venv/bin/inspect"), "sandbox", "cleanup", "docker"],
                        Path(A.log_root) / "sandbox_cleanup.log", timeout=900,
                        env=docker_env(), cwd=str(IGE))
    except Exception as ex:  # noqa: BLE001
        log("sandbox cleanup failed:", ex)
        return -1


def finish(status: str, error: str | None = None) -> int:
    stop = None
    if SERVER_STARTED or not A.skip_server:
        stop = stop_server(error or "chain finished")
        stage("stop", status="ok" if (stop or {}).get("confirmed_stopped", A.skip_server) else "error",
              detail=stop, at=now())
        if not A.skip_server and not stop.get("confirmed_stopped"):
            status = "error"
            error = (error + "; " if error else "") + "server stop NOT confirmed"
    if status != "ok":
        STATUS["sandbox_cleanup_rc"] = cleanup_sandboxes()
    rec = {"status": status, "error": error, "finished_at": now(),
           "started_at": STATUS.get("started_at"), "orphans_reaped": ORPHANS,
           "stages": STATUS["stages"], "stop": stop,
           "sandbox_cleanup_rc": STATUS.get("sandbox_cleanup_rc")}
    write_json(Path(A.log_root) / "ALL_DONE.json", rec)
    log(f"ALL_DONE status={status} error={error}")
    return 0 if status == "ok" else 1


def verify(ep: str) -> tuple[bool, dict]:
    last = {}
    for attempt in (1, 2):
        out = Path(A.log_root) / f"verify_report_{attempt}.json"
        out.unlink(missing_ok=True)
        rc = run_step([PY, str(HERE / "verify_server.py"), "--endpoint", ep, "--out", str(out)],
                      Path(A.log_root) / "verify.log", timeout=3600)
        if out.exists():
            checks = json.loads(out.read_text()).get("checks", {})
            missing = [k for k in REQUIRED_VERIFY if k not in checks]
            failed = [k for k in REQUIRED_VERIFY if k in checks and not checks[k]["pass"]]
            info = {k: v["pass"] for k, v in checks.items() if k not in REQUIRED_VERIFY}
            last = {"attempt": attempt, "rc": rc, "failed": failed, "missing": missing,
                    "informational": info, "report": str(out)}
            if not failed and not missing:
                return True, last
        else:
            last = {"attempt": attempt, "rc": rc, "error": "no report written"}
        log(f"verify attempt {attempt} did not pass: {last}")
    return False, last


def summarise_batch(name: str, root: Path, env: dict) -> dict:
    jsonl = root / "rollouts.jsonl"
    rc = run_step([PY, str(IGE / "scripts/summarise.py"), str(root), "--jsonl", str(jsonl)],
                  root / "summary.txt", timeout=3600, env=env, cwd=str(IGE))
    counts: dict = {}
    if jsonl.exists():
        for line in jsonl.open():
            r = json.loads(line)
            c = counts.setdefault(r["model"], {"n": 0, "outcomes": {}})
            c["n"] += 1
            c["outcomes"][r["outcome"]] = c["outcomes"].get(r["outcome"], 0) + 1
    return {"rc": rc, "jsonl": str(jsonl), "summary": str(root / "summary.txt"),
            "rollouts_by_model": counts}


def run_batch(name: str, epochs: int, ep: str, env: dict) -> tuple[bool, dict]:
    root = Path(A.log_root) / name
    root.mkdir(parents=True, exist_ok=True)
    if not A.skip_server:
        h = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "60",
                            json.load(open(ep))["url"] + "/health"], capture_output=True, text=True)
        if h.stdout.strip() != "200":
            return False, {"error": f"server /health -> {h.stdout.strip() or h.stderr[:200]}"}
    stage(name, status="running", started_at=now(), epochs=epochs, log_root=str(root))
    cmd = [PY, str(IGE / "scripts/run_kimi.py"), "--epochs", str(epochs),
           "--log-root", str(root), "--batch", name, "--endpoint-json", ep,
           "--max-sandboxes", str(A.max_sandboxes), "--max-connections", str(A.max_connections)]
    if A.sample_ids:
        cmd += ["--sample-ids", A.sample_ids]
    t0 = time.time()
    rc = run_step(cmd, root / "runner.log", timeout=A.batch_timeout_h * 3600, env=env, cwd=str(IGE))
    alldone = root / "ALL_DONE.json"
    ad = json.loads(alldone.read_text()) if alldone.exists() else {"status": "missing ALL_DONE.json"}
    summ = summarise_batch(name, root, env)
    ok = rc == 0 and ad.get("status") == "ok" and summ["rc"] == 0
    rec = {"status": "ok" if ok else "error", "batch": name, "epochs": epochs,
           "runner_rc": rc, "seconds": round(time.time() - t0), "finished_at": now(),
           "runner_all_done": ad, **summ}
    write_json(Path(A.log_root) / f"{name.upper()}_DONE.json", rec)
    stage(name, status=rec["status"], finished_at=now(), seconds=rec["seconds"],
          rollouts_by_model={k: v["n"] for k, v in summ["rollouts_by_model"].items()})
    return ok, rec


def on_signal(signum, frame):
    log(f"received signal {signum}: killing running steps, stopping server, aborting")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.getpgrp() == os.getpid():      # launched with setsid: we lead the group
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except OSError:
            pass
    sys.exit(finish("error", f"aborted by signal {signum}"))


def main() -> int:
    global A, SERVER_STARTED
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="/work/workspace/logs/kimi_v1")
    ap.add_argument("--epochs1", type=int, default=9)
    ap.add_argument("--epochs2", type=int, default=16)
    ap.add_argument("--sample-ids", default=None)
    ap.add_argument("--max-sandboxes", type=int, default=16)
    ap.add_argument("--max-connections", type=int, default=16)
    ap.add_argument("--batch-timeout-h", type=float, default=6.0)
    ap.add_argument("--instance", default="ige")
    ap.add_argument("--idle-minutes", type=int, default=45)
    ap.add_argument("--skip-server", action="store_true", help="dry run against --endpoint-json")
    ap.add_argument("--endpoint-json", default=None)
    A = ap.parse_args()
    root = Path(A.log_root)
    root.mkdir(parents=True, exist_ok=True)
    for f in ("ALL_DONE.json", "BATCH1_DONE.json", "BATCH2_DONE.json"):
        (root / f).unlink(missing_ok=True)
    STATUS.update(started_at=now(), pid=os.getpid(), args=vars(A))
    (root / "chain.pid").write_text(str(os.getpid()))
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(36, 1, 0, 0, 0)
    except Exception as exc:  # noqa: BLE001
        log("warning: not a subreaper:", exc)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        env = docker_env()
        ep = endpoint_path()
        # ---- preflight: docker host + sandbox image, BEFORE any GPU is booked ----
        pf = Path(A.log_root) / "preflight.log"
        rc1 = run_step(["docker", "version", "--format", "{{.Server.Version}}"], pf, 120, env=env)
        rc2 = run_step(["docker", "image", "inspect", "--format", "{{.Id}}",
                        env.get("GRADER_HACKING_IMAGE", "grader-hacking-env:latest")], pf, 120, env=env)
        stage("preflight", status="ok" if rc1 == rc2 == 0 else "error", docker_rc=rc1, image_rc=rc2)
        if rc1 or rc2:
            return finish("error", f"preflight failed (docker rc={rc1}, image rc={rc2}); see {pf}")
        # ---- server ----
        if A.skip_server:
            stage("server", status="skipped", endpoint=ep)
        else:
            stage("server", status="launching", started_at=now())
            SERVER_STARTED = True
            rc = run_step(["bash", str(HERE / "launch_kimi_server.sh"), A.instance,
                           str(A.idle_minutes)], root / "server_launch.log",
                          timeout=5 * 3600, cwd=str(HERE))
            if rc != 0 or not Path(ep).exists():
                stage("server", status="error", rc=rc)
                return finish("error", f"server launch failed rc={rc} (see server_launch.log)")
            e = json.load(open(ep))
            stage("server", status="ok", healthy_at=now(), endpoint=ep, url=e.get("url"),
                  seconds_container_to_health=e.get("seconds_container_to_health"),
                  boot_milestones_s=e.get("boot_milestones_s"))
        # ---- verify ----
        stage("verify", status="running", started_at=now())
        ok, det = verify(ep)
        stage("verify", status="ok" if ok else "error", finished_at=now(), detail=det)
        if not ok:
            return finish("error", f"verify failed: {det}")
        # ---- batches ----
        for name, ep_n in (("batch1", A.epochs1), ("batch2", A.epochs2)):
            ok, rec = run_batch(name, ep_n, ep, env)
            if not ok:
                return finish("error", f"{name} failed: status={rec.get('status')} "
                                       f"error={rec.get('error')} runner_rc={rec.get('runner_rc')}")
        return finish("ok")
    except SystemExit:
        raise
    except BaseException as ex:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return finish("error", f"chain exception: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    sys.exit(main())
