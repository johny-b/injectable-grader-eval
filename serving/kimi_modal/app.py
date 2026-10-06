"""Steering-capable Kimi-K2.5 vLLM server on Modal (8xH200) -- reconstruction.

Serves, behind a bearer token on a public Modal tunnel:
  * `kimi`       -- base Kimi-K2.5 (compressed-tensors int4 experts)
  * `kimi-hack`  -- the reward-hacking LoRA (step-648), `--lora-modules`
  * per-request activation steering: `vllm_xargs={"steer_vector": "0003",
    "steer_strength": 0.5}` (vector 0003 only, layer 30 = output of block 29)

This is a reconstruction of `ar-kimi-tests-3/em_strong/modal/app.py::serve_forever`
(the original image was built from local dirs that no longer exist). Same
engine (vLLM 0.29.0, CUDA 13 devel base), same server flags, same resources.
See RECONSTRUCTION.md for what was rebuilt and how it differs.

    modal run app.py::smoke                                     # CPU only
    modal run --detach app.py::serve_forever --confirm yes-run-gpu --instance ige
    # or simply:  ./launch_kimi_server.sh   (writes .secrets/kimi_endpoint.json)
    ./stop_kimi_server.sh                                       # modal app stop
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "kimi-steer-serve"
WS = "/work/workspace"
MODELS = "/vol/models"
RESULTS = "/vol/results"
HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

BASE_DIR = f"{MODELS}/Kimi-K2.5"
LORA_VLLM_DIR = f"{MODELS}/lora_vllm"
DONE_MARKER = f"{MODELS}/DOWNLOAD.DONE"

STEER_VECTORS = "0003"          # ONLY vector 0003 (B), layer 30 -> block 29
HIDDEN_SIZE = 7168
MAX_MODEL_LEN = 65536           # as the grader-hacking servers (instances c..g)
MAX_NUM_SEQS = 128
TP_SIZE = 8

CUDA_TAG = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
IGNORE = ["**/__pycache__", "**/*.pyc"]

image = (
    modal.Image.from_registry(CUDA_TAG, add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "patch", "curl", "ca-certificates", "build-essential")
    .pip_install("vllm==0.29.0")
    .pip_install("setuptools>=68", "wheel")
    .add_local_dir(BUILD / "steering", f"{WS}/steering", copy=True, ignore=IGNORE)
    .add_local_dir(BUILD / "lora_prep", f"{WS}/lora_prep", copy=True, ignore=IGNORE)
    .add_local_file(BUILD / "kimi_lora_file_patch.py", "/build/kimi_lora_file_patch.py", copy=True)
    .add_local_file(BUILD / "build_patch.sh", "/build/build_patch.sh", copy=True)
    .run_commands("bash /build/build_patch.sh")
    .run_commands(
        f"python -m pip install --no-deps -e {WS}/steering/pod",
        f"python -m pip install --no-deps -e {WS}/steering",
        "python -c \"import vllm_steering, steering_vectors; print('steering packages ok')\"",
        # offline vector check at build time (no GPU, no PYTHONPATH needed)
        f"STEER_ENABLE=1 python -c \"from vllm_steering import store; "
        f"s=store.read('{WS}/steering/vectors','{STEER_VECTORS}'); "
        f"print('VECTORS_OK', s.block, s.digest[:16], [v.describe() for v in s.vectors])\"",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "STEER_VECTOR_DIR": f"{WS}/steering/vectors",
        "STEER_VECTORS": STEER_VECTORS,
        # PYTHONPATH deliberately NOT set image-wide (Modal's runner lives on it;
        # sitecustomize must reach only the vLLM processes). See SERVE_ENV.
    })
)

app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("kimi-k25-weights")
results = modal.Volume.from_name("kimi-steer-results")


def _run(cmd, check=True, **kw):
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, **kw)
    if check and p.returncode != 0:
        raise RuntimeError(f"rc={p.returncode}: {cmd}")
    return p


# ==========================================================================
# smoke -- CPU only
# ==========================================================================
@app.function(cpu=4.0, memory=16384, timeout=1800, volumes={MODELS: weights})
def smoke():
    import hashlib

    import numpy as np
    import torch
    import vllm

    print("=" * 72)
    print("VLLM", vllm.__version__, "TORCH", torch.__version__, "CUDA", torch.version.cuda)
    assert vllm.__version__ == "0.29.0", vllm.__version__
    print(subprocess.run(["bash", "-lc", "nvcc --version | tail -1"],
                         capture_output=True, text=True).stdout.strip())

    # ---- LoRA patch -------------------------------------------------------
    print("-" * 72)
    print("PATCH_STATUS.txt:\n" + Path(f"{WS}/lora_prep/PATCH_STATUS.txt").read_text())
    from vllm.model_executor.models.interfaces import supports_lora
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration as K
    print("supports_lora BEFORE monkeypatch (file patch only):", supports_lora(K),
          " file flag:", getattr(K, "_kimi_lora_file_patch", False))
    sys.path.insert(0, f"{WS}/lora_prep")
    import vllm_kimi_lora_patch as kp
    print("monkeypatch STATUS:", kp.STATUS)
    print("supports_lora AFTER:", supports_lora(K))
    print("packed_modules_mapping:", K.packed_modules_mapping, "embedding_modules:",
          K.embedding_modules, "lora_skip_prefixes:", K.lora_skip_prefixes,
          "has get_mm_mapping:", hasattr(K, "get_mm_mapping"))
    from vllm.model_executor.models.interfaces import SupportsLoRA
    inst = object.__new__(K)   # what the worker checks is an INSTANCE (first GPU boot failed here)
    print("nominal SupportsLoRA base:", (SupportsLoRA in K.__mro__),
          " supports_lora(instance):", supports_lora(inst))
    assert supports_lora(K) and supports_lora(inst) and (SupportsLoRA in K.__mro__)
    assert not hasattr(K, "get_mm_mapping")
    print("worker-ext RPC:", kp.KimiLoRAWorkerExtension().kimi_lora_patch_status())

    # ---- steering package -------------------------------------------------
    print("-" * 72)
    from importlib.metadata import entry_points

    from vllm_steering import config, endpoint, middleware, patch, store
    print("endpoint plugins:", {e.name: e.value for e in entry_points(group="vllm.endpoint_plugins")})
    print("WATCHED_PATHS:", sorted(middleware.WATCHED_PATHS))
    os.environ["STEER_ENABLE"] = "1"
    c = config.config_from_env()
    s = store.read(c.vector_dir, c.vectors)
    print(f"store: block={s.block} model={s.model} digest={s.digest[:16]} ids={s.ids}")
    from steering_vectors import vectorfmt
    for v in s.vectors:
        meta = vectorfmt.read_meta(s.root / v.id)
        a = vectorfmt.load_vector(s.root / v.id, meta)
        print(f"  {v.describe()}  layer={v.layer} block={v.block} shape={a.shape} "
              f"||v||file={float(np.linalg.norm(a.astype(np.float64))):.6f} scale={v.scale:.6f}")
    m = store.matrix(s, HIDDEN_SIZE)
    print("matrix", m.shape, "row norms", [round(float(np.linalg.norm(r)), 4) for r in m])
    assert s.ids == ("0003",) and s.block == 29 and abs(s.vectors[0].scale - 11.496644) < 1e-4
    print("VECTORS_OK")
    # the sitecustomize hook must patch BOTH runners when they are imported
    env = {**os.environ, "STEER_ENABLE": "1",
           "PYTHONPATH": f"{WS}/steering/pod/src:{WS}/lora_prep"}
    out = subprocess.run(
        [sys.executable, "-c",
         "import vllm.v1.worker.gpu.model_runner as a, vllm.v1.worker.gpu_model_runner as b;"
         "print('V2 patched', getattr(a.GPUModelRunner.load_model,'_steer_patched',False));"
         "print('V1 patched', getattr(b.GPUModelRunner.load_model,'_steer_patched',False))"],
        env=env, capture_output=True, text=True)
    print("sitecustomize hook check:\n" + "\n".join(
        ln for ln in (out.stdout + out.stderr).splitlines() if "steer" in ln or "patched" in ln))
    assert "V2 patched True" in out.stdout and "V1 patched True" in out.stdout, out.stderr[-3000:]

    # ---- Volume / chat template ------------------------------------------
    print("-" * 72)
    print("volume:", sorted(os.listdir(MODELS)))
    done = json.load(open(DONE_MARKER))
    tpl = f"{BASE_DIR}/chat_template.jinja"
    sha = hashlib.sha256(open(tpl, "rb").read()).hexdigest()
    print("chat_template.jinja sha256", sha, "matches DOWNLOAD.DONE:",
          sha == done["config_sha256"][tpl]["sha256"])
    tc = json.load(open(f"{BASE_DIR}/tokenizer_config.json"))
    print("tokenizer_config has chat_template key:", "chat_template" in tc)
    for p in (f"{BASE_DIR}/config.json", f"{LORA_VLLM_DIR}/adapter_model.safetensors",
              f"{LORA_VLLM_DIR}/adapter_config.json"):
        print("  exists", p, os.path.getsize(p))
    print("SMOKE_OK")


# ==========================================================================
# serve_forever -- GPU
# ==========================================================================
def serve_cmd(token, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
              moe_backend="marlin"):
    # Byte-for-byte the original serve_cmd(host="0.0.0.0", api_key=...).
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_DIR,
        "--trust-remote-code",
        "--language-model-only",
        "--safetensors-load-strategy=prefetch",
        "--served-model-name", "kimi",
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", str(TP_SIZE),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.90",
        "--moe-backend", moe_backend,
        "--enable-lora", "--max-lora-rank", "32", "--max-loras", "1",
        "--enable-moe-shared-loras",
        "--lora-modules", f"kimi-hack={LORA_VLLM_DIR}",
        "--worker-extension-cls", "vllm_kimi_lora_patch.KimiLoRAWorkerExtension",
        "--middleware", "vllm_steering.middleware.SteeringValidation",
        "--reasoning-parser", "kimi_k2",
        "--api-key", token,
    ]


SERVE_ENV = {
    "NCCL_NVLS_ENABLE": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_PLUGINS": "steering",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "STEER_ENABLE": "1",
    "STEER_VECTOR_DIR": f"{WS}/steering/vectors",
    "STEER_VECTORS": STEER_VECTORS,
    "STEER_DEBUG": "0",
    "PYTHONPATH": f"{WS}/steering/pod/src:{WS}/lora_prep",
}

SERVE_TIMEOUT = 8 * 60 * 60
SHUTDOWN_MARGIN = 180


@app.function(
    gpu="H200:8",                       # H200 only: B200 would need a different MoE backend
    cpu=16.0,
    memory=(262144, 393216),            # 256 GiB floor / 384 GiB cap (RUN_INFO attempt 2/3)
    timeout=SERVE_TIMEOUT,
    volumes={MODELS: weights, RESULTS: results},
    retries=0,
)
def serve_forever(confirm: str = "", instance: str = "ige",
                  health_timeout_minutes: int = 90,
                  idle_shutdown_minutes: int = 45,
                  max_model_len: int = MAX_MODEL_LEN,
                  max_num_seqs: int = MAX_NUM_SEQS):
    """Boot, self-verify, publish endpoint.json, serve until STOP / idle / timeout.

    idle_shutdown_minutes: stop by itself after this long with no request
    running/waiting and no request completed (0 disables). Protects against an
    idle 8xH200 billing for hours if a client step forgets to stop it.
    """
    import secrets
    import urllib.error
    import urllib.request

    if confirm != "yes-run-gpu":
        raise SystemExit('needs confirm="yes-run-gpu"; nothing was done')
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", instance):
        raise SystemExit(f"bad instance {instance!r}")
    t_launch = time.time()
    token = secrets.token_urlsafe(32)
    serve_dir = f"{RESULTS}/serve/{instance}"
    log_dir = f"{RESULTS}/logs/{instance}"
    endpoint_json, stop_file = f"{serve_dir}/endpoint.json", f"{serve_dir}/STOP"
    os.makedirs(serve_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # The server log lives on LOCAL disk and is copied to the Volume (closed)
    # periodically: an open file on the Volume makes results.reload() fail on
    # every poll, which is why the original STOP file never worked (RUN_INFO,
    # attempt 9).
    local_log = "/tmp/serve_forever.log"
    vol_log = f"{log_dir}/serve_forever.log"

    def sync_log():
        try:
            shutil.copyfile(local_log, vol_log)
            results.commit()
        except Exception as e:
            print(f"log sync failed: {e!r}", flush=True)

    weights.reload()
    results.reload()
    if os.path.exists(stop_file):
        print(f"removing stale {stop_file}", flush=True)
        os.remove(stop_file)
        results.commit()
    for p in (DONE_MARKER, f"{BASE_DIR}/config.json",
              f"{LORA_VLLM_DIR}/adapter_model.safetensors"):
        if not os.path.exists(p):
            raise RuntimeError(f"missing {p}")
    gpus = [g.strip() for g in subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.splitlines() if g.strip()]
    print(f"GPUS {len(gpus)} x {gpus[0] if gpus else '?'}", flush=True)

    def tail(n=5):
        return subprocess.run(["tail", f"-{n}", local_log], capture_output=True,
                              text=True).stdout

    def http(path, body=None, base="http://127.0.0.1:8000", auth=True, timeout=600):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json"} if data else {}
        if auth:
            h["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}{path}", data=data,
                                                               headers=h), timeout=timeout) as r:
                raw = r.read().decode()
                try:
                    return json.loads(raw)
                except ValueError:
                    return {"raw": raw}
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode()[:500], "status": e.code}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    cmd = serve_cmd(token, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    print("$ " + " ".join(shlex.quote("<TOKEN>" if c == token else c) for c in cmd), flush=True)
    logf = open(local_log, "wb", buffering=0)
    srv = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                           env={**os.environ, **SERVE_ENV})
    t0 = time.time()
    ready = False
    last_print = 0.0
    milestones = {}
    MS = {"weights": "Loading weights took", "compile": "torch.compile and initial profiling",
          "graphs": "Graph capturing finished", "kv": "GPU KV cache size"}
    while time.time() - t0 < health_timeout_minutes * 60:
        if srv.poll() is not None:
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        try:
            text = open(local_log, errors="replace").read()
            for k, pat in MS.items():
                if k not in milestones:
                    m = [ln for ln in text.splitlines() if pat in ln]
                    if m:
                        milestones[k] = round(time.time() - t0)
                        print(f"[boot +{milestones[k]}s] {m[0][-200:]}", flush=True)
        except OSError:
            pass
        if time.time() - last_print > 120:
            last_print = time.time()
            print(f"[boot +{time.time() - t0:.0f}s]\n{tail(4)}", flush=True)
            sync_log()
        time.sleep(10)
    health_s = round(time.time() - t0)
    sync_log()
    if not ready:
        print(tail(80), flush=True)
        srv.terminate()
        raise RuntimeError(f"server not healthy (rc={srv.poll()}) after {health_s}s")
    print(f"SERVER_UP after {health_s}s  milestones={milestones}", flush=True)

    # ---- in-container verification (cheap; the full one runs client-side) --
    fail = []
    text = open(local_log, errors="replace").read()
    steer = [ln for ln in text.splitlines() if "[steer]" in ln]
    v2 = sum("via vllm.v1.worker.gpu.model_runner" in ln for ln in steer)
    print(f"[steer] lines={len(steer)}  V2-runner install lines={v2}", flush=True)
    for ln in steer[:12]:
        print("   ", ln[:220])
    if v2 < TP_SIZE:
        fail.append(f"only {v2} ranks report steering installed on the live V2 runner")
    models = {m["id"] for m in http("/v1/models").get("data", [])}
    print("models:", sorted(models))
    if {"kimi", "kimi-hack"} - models:
        fail.append(f"served models {models}")
    print("/steering/vectors:", json.dumps(http("/steering/vectors"))[:400])

    def chat(model, xargs, salt):
        b = {"model": model, "messages": [{"role": "user", "content": "Say hi in five words."}],
             "max_tokens": 16, "temperature": 0.0, "cache_salt": salt}
        if xargs is not None:
            b["vllm_xargs"] = xargs
        return http("/v1/chat/completions", b)
    bad = chat("kimi", {"steer_vector": "9999", "steer_strength": 1.0}, "vfy:bad")
    nostr = chat("kimi", {"steer_vector": "0003"}, "vfy:nostr")
    ok = chat("kimi", {"steer_vector": "0003", "steer_strength": 0.5}, "vfy:ok")
    print("bad vector ->", bad.get("status"), "| no strength ->", nostr.get("status"),
          "| 0003@0.5 ->", "ok" if "choices" in ok else ok)
    if bad.get("status") != 400 or nostr.get("status") != 400 or "choices" not in ok:
        fail.append("middleware controls")
    if fail:
        print("VERIFY FAILED:", fail, flush=True)
        srv.terminate()
        sync_log()
        raise RuntimeError(f"verification failed: {fail}")

    stop_reason = "unknown"
    try:
        with modal.forward(8000) as tunnel:
            url = tunnel.url
            ext = http("/v1/models", base=url, timeout=120)
            noauth = http("/v1/models", base=url, auth=False, timeout=120)
            print("tunnel with token ok:", "data" in ext, "| without token ->",
                  noauth.get("status"), flush=True)
            if "data" not in ext or noauth.get("status") != 401:
                raise RuntimeError(f"tunnel auth check failed: {ext} {noauth}")
            endpoint = {
                "base_url": f"{url}/v1", "url": url, "api_key": token,
                "served_models": ["kimi", "kimi-hack"],
                "steer_vectors": STEER_VECTORS.split(","),
                "max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
                "moe_backend": "marlin", "gpus": gpus, "instance": instance,
                "app": APP_NAME,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_launch)),
                "seconds_container_to_health": health_s, "boot_milestones_s": milestones,
                "idle_shutdown_minutes": idle_shutdown_minutes,
                "stop_by": f"modal app stop, or touch serve/{instance}/STOP on kimi-steer-results",
            }
            with open(endpoint_json, "w") as f:
                json.dump(endpoint, f, indent=2)
            results.commit()
            print("VERIFY_OK -- serving", json.dumps({**endpoint, "api_key": "<redacted>"}),
                  flush=True)

            deadline = t_launch + SERVE_TIMEOUT - SHUTDOWN_MARGIN
            last_activity = time.time()
            last_done = None
            last_beat = 0.0
            while True:
                if srv.poll() is not None:
                    stop_reason = f"server exited rc={srv.returncode}"
                    break
                if time.time() > deadline:
                    stop_reason = "function timeout approaching"
                    break
                try:
                    results.reload()
                    if os.path.exists(stop_file):
                        stop_reason = "STOP file"
                        break
                except Exception as e:
                    print(f"reload failed {e!r}", flush=True)
                met = http("/metrics", timeout=30).get("raw", "")
                busy = done = 0.0
                for ln in met.splitlines():
                    if ln.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
                        busy += float(ln.rsplit(" ", 1)[-1])
                    elif ln.startswith("vllm:request_success_total"):
                        done += float(ln.rsplit(" ", 1)[-1])
                if busy > 0 or done != last_done:
                    last_activity = time.time()
                last_done = done
                idle = time.time() - last_activity
                if idle_shutdown_minutes and idle > idle_shutdown_minutes * 60:
                    stop_reason = f"idle {idle / 60:.0f} min"
                    break
                if time.time() - last_beat > 300:
                    last_beat = time.time()
                    print(f"[serving +{(time.time() - t_launch) / 3600:.2f}h] busy={busy:.0f} "
                          f"completed={done:.0f} idle={idle / 60:.1f}min", flush=True)
                    sync_log()
                time.sleep(30)
            print("=== stopping:", stop_reason, flush=True)
    finally:
        try:
            if os.path.exists(endpoint_json):
                rec = json.load(open(endpoint_json))
                rec.update(api_key="<expired>", stop_reason=stop_reason,
                           stopped_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                json.dump(rec, open(f"{serve_dir}/endpoint_last.json", "w"), indent=2)
                os.remove(endpoint_json)
            if os.path.exists(stop_file):
                os.remove(stop_file)
        except Exception as e:
            print("cleanup failed", e)
        if srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=120)
            except subprocess.TimeoutExpired:
                srv.kill()
        logf.close()
        sync_log()
    return {"stop_reason": stop_reason, "uptime_h": round((time.time() - t_launch) / 3600, 3)}
