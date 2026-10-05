"""Run the grader-hacking eval: one process per condition, then DONE.json.

    # dev pilot against an OpenAI model, no GPU, no steering
    python scripts/run_grader_hacking.py --model openai/gpt-4.1 --epochs 20 \
        --no-fetch-endpoint --conditions unsteered --log-root logs/pilot_gpt41

    # the real thing, both conditions at once against one Kimi server
    python scripts/run_grader_hacking.py --epochs 100

Same shape, and the same four load-bearing details, as
/work/workspace/ctfish/scripts/run_ctfish_kimi.py -- the two experiments share
the provider and should stay commensurable:

1.  **`max_connections` on the MODEL.** Inspect merges the CALL's config OVER
    the model's, so a `max_connections` passed any other way is overwritten.
2.  **`client_timeout`.** The OpenAI SDK's 600 s read timeout otherwise turns a
    slow generation into an `APITimeoutError` whose retry restarts the SAME
    generation, so the run loops instead of failing.
3.  **Pinned sampling**, identical in every condition. Sampling that differs
    between two runs is indistinguishable from whatever else differs between
    them, and the whole experiment is a difference between two runs.
4.  **One condition per process**, because inspect's `eval()` owns a display, an
    event loop and process-global concurrency state.

No seed, deliberately: the dataset is 12 samples (one per question) and the
per-question sample size IS the epoch count, so a fixed seed would return the
same rollout every epoch and the spread over epochs would be a lie.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Moonshot's recommended thinking-mode sampling, identical to the ctfish and EM
# runs so the three experiments are commensurable.
KIMI_THINKING: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 16384,
}

# (condition -> steering args, or None for the bare unsteered model).
# `unsteered` sends NO vllm_xargs and NO cache_salt, which is what keeps its KV
# blocks disjoint from the steered condition's.
CONDITIONS: dict[str, tuple[str, float] | None] = {
    "unsteered": None,
    "B_pos0.5": ("0003", 0.5),
    # The reward-hacking LoRA (`kimi-hack`), with NO steering at all. Not a rung
    # on the steering ladder: a DIFFERENT MODEL, served by the SAME container
    # under a second name, run through the identical harness so "what the
    # finetune does to this task" is directly comparable to "what the vector
    # does". Steering args are absent exactly as they are for `unsteered`.
    # Mirrors the `finetuned` condition in
    # /work/workspace/agentic_misalignment/scripts/run_am_kimi.py.
    "finetuned": None,
}

# Conditions served by a DIFFERENT model name on the same server. The server is
# launched with `--served-model-name kimi` for the base weights and
# `--lora-modules kimi-hack=/vol/models/lora_vllm` for the reward-hacking
# adapter, so both names are live on one container and choosing between them is
# a per-request choice rather than a redeploy. The provider takes the served
# name straight from the inspect model string (`steered/<served-model-name>`)
# and passes it through to the OpenAI-compatible client.
#
# Only consulted when --model is itself a `steered/` name, so the non-steered
# pilot path (--model openai/gpt-4.1) is untouched.
#
# Cache safety, checked rather than assumed. `finetuned` sends no steering
# args, so the provider sends no `cache_salt` either (SteeredAPI.
# completion_params returns early when steer_vector is None) -- exactly like
# `unsteered`. That is nevertheless NOT a collision, because vLLM's
# `generate_block_hash_extra_keys` prepends `_gen_lora_extra_hash_keys`, which
# returns `[request.lora_request.lora_name]` for a LoRA request and `[]`
# otherwise (vllm/v1/core/kv_cache_utils.py, v0.29.0). So every `finetuned`
# block carries the extra key "kimi-hack" and no `kimi` block ever does:
#
#   unsteered   (kimi)       extra_keys = ()
#   B_pos0.5    (kimi)       extra_keys = ("steer:0003:0.5",)
#   finetuned   (kimi-hack)  extra_keys = ("kimi-hack",)
#
# -- three disjoint key spaces, with prefix reuse preserved WITHIN each.
#
# Connection pools are disjoint too: SteeredAPI.connection_key() is
# f"steered:{model_name}:{vector}:{strength}", so `finetuned` gets
# "steered:kimi-hack:None:None" and `unsteered` "steered:kimi:None:None".
#
# Log directories are disjoint by the existing `gh_<condition>` rule, giving
# .../gh_finetuned/ next to .../gh_unsteered/ and .../gh_B_pos0.5/.
CONDITION_MODELS: dict[str, str] = {
    "finetuned": "steered/kimi-hack",
}

# OpenAI's reasoning models reject `temperature` and `top_p` outright (HTTP 400)
# and take `max_completion_tokens` instead of `max_tokens`. Pinning sampling is
# the right call for the Kimi comparison, but for a DEV PILOT against a
# reasoning model the only honest options are "send nothing and let the provider
# default" -- so the sampling profile is a flag rather than a constant.
SAMPLING: dict[str, dict[str, float | int]] = {
    "kimi": KIMI_THINKING,
    # Nothing pinned. Used for OpenAI reasoning models in the dev pilot ONLY;
    # it must never be what a Kimi comparison runs, because two conditions whose
    # sampling is left to the provider are not pinned to each other.
    "provider-default": {},
}

MODAL_VOLUME = "kimi-steer-results"
ENDPOINT_REMOTE = "serve/endpoint.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grader-hacking eval, one process per condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="steered/kimi",
                   help="Inspect model name. A non-`steered/` name forces "
                        "--conditions unsteered and sends no steering args, "
                        "which is how the wrapper is dry-run / piloted.")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--epochs", type=int, default=100,
                   help="Rollouts per question per condition (12 questions, "
                        "so a condition is 12 x epochs rollouts).")
    p.add_argument("--message-limit", type=int, default=18)
    p.add_argument("--command-timeout", type=int, default=30)
    p.add_argument("--token-limit", type=int, default=None)
    p.add_argument("--max-sandboxes", type=int, default=10,
                   help="Concurrent containers PER CONDITION. These sandboxes "
                        "are idle (a shell and sqlite), unlike ctfish's "
                        "Stockfish, so the 4 vCPU box takes more of them.")
    p.add_argument("--max-connections", type=int, default=20,
                   help="Concurrent requests PER CONDITION.")
    p.add_argument("--client-timeout", type=float, default=1200.0)
    p.add_argument("--retry-on-error", type=int, default=2)
    p.add_argument("--log-root", default="logs")
    p.add_argument("--endpoint-json", default=None)
    p.add_argument("--no-fetch-endpoint", action="store_true",
                   help="Do not read endpoint.json; use the environment as is. "
                        "Required for an OpenAI pilot.")
    p.add_argument("--sampling", default="kimi", choices=sorted(SAMPLING),
                   help="Sampling profile. 'kimi' pins temperature/top_p/"
                        "max_tokens and is what the real comparison uses; "
                        "'provider-default' sends nothing, which is required "
                        "for OpenAI reasoning models that reject temperature.")
    p.add_argument("--log-level", default="info")
    p.add_argument("--display", default="plain",
                   help="'plain' is the one that behaves under nohup.")
    p.add_argument("--plumbing-test", action="store_true")
    p.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    names = [c.strip() for c in a.conditions.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONDITIONS]
    if unknown:
        p.error(f"unknown condition(s) {unknown}; have {sorted(CONDITIONS)}")
    if not a.model.startswith("steered/") and not a.plumbing_test:
        if names != ["unsteered"]:
            print(f"note: --model {a.model} is not a steered model, so only "
                  f"`unsteered` is meaningful; running that one.", flush=True)
        names = ["unsteered"]
    a.condition_names = names
    if a.epochs < 1:
        p.error("--epochs must be at least 1")
    return a


def fetch_endpoint(dest: Path) -> dict:
    """Pull serve/endpoint.json off the Modal Volume and return it.

    endpoint.json exists only while serve_forever() is actually serving, so a
    missing file is a precise statement ("no server is up") rather than a
    connection error much later.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    cmd = ["modal", "volume", "get", MODAL_VOLUME, ENDPOINT_REMOTE, str(dest)]
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists():
        raise SystemExit(
            f"could not fetch {ENDPOINT_REMOTE} from the {MODAL_VOLUME} Volume.\n"
            f"{r.stdout}\n{r.stderr}\n"
            f"endpoint.json exists only while serve_forever() is serving, so "
            f"this almost certainly means no server is up.")
    info = json.loads(dest.read_text())
    for key in ("base_url", "api_key"):
        if not info.get(key):
            raise SystemExit(f"{dest} has no {key!r}: {info}")
    return info


def run_condition(a: argparse.Namespace, name: str) -> int:
    """One condition, one eval() call, in this process."""
    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model

    from gh_eval import grader_hacking

    sampling = SAMPLING[a.sampling]
    steer = CONDITIONS[name] if a.model.startswith("steered/") else None
    # A condition may be served by a second model name on the same container
    # (see CONDITION_MODELS); everything else about the run is identical.
    model_name = (CONDITION_MODELS.get(name, a.model)
                  if a.model.startswith("steered/") else a.model)
    log_dir = Path(a.log_root) / f"gh_{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    done_marker = log_dir / "DONE.json"
    if done_marker.exists():
        done_marker.unlink()

    model_args: dict[str, object] = {}
    metadata: dict[str, object] = {
        "condition": name,
        "served_model": model_name,
        "generate_config": sampling,
        "sampling_profile": a.sampling,
        "eval": "grader_hacking",
    }
    if model_name.startswith("steered/"):
        model_args["client_timeout"] = a.client_timeout
        # The Kimi server runs without --tool-call-parser, so it 400s on every
        # tool_choice except "none" and never parses a tool call out of the
        # completion. The provider sends tool_choice="none" (the tools still go
        # on the wire, so the model is prompted in its native format) and parses
        # the native markup client-side. Identical in both conditions.
        model_args["kimi_tool_calls"] = True
        metadata["kimi_tool_calls"] = True
    if steer is not None:
        # In model_args because that is what the provider reads, and again in
        # metadata because a summary over many logs should not have to know
        # provider argument names to find out what was run.
        model_args["steer_vector"] = steer[0]
        model_args["steer_strength"] = steer[1]
        metadata["steer_vector"] = steer[0]
        metadata["steer_strength"] = steer[1]

    model = get_model(
        model_name,
        # On the MODEL, not the call: see the module docstring, points 1 and 3.
        config=GenerateConfig(max_connections=a.max_connections, **sampling),
        **model_args,
    )

    task = grader_hacking(
        message_limit=a.message_limit,
        command_timeout=a.command_timeout,
        token_limit=a.token_limit,
        # Epochs are set HERE and nowhere else: eval(epochs=...) would win over
        # the task's value, so passing both is a way to run a different N than
        # the log's own task_args report.
        epochs=a.epochs,
    )

    print(f"=== condition {name}: {a.epochs} epochs, "
          f"model {model_name}"
          + (f", steer {steer[0]} @ {steer[1]}" if steer else " (unsteered)")
          + f"\n    max_sandboxes={a.max_sandboxes} "
          f"max_connections={a.max_connections} "
          f"message_limit={a.message_limit} sampling[{a.sampling}]={sampling}"
          f"\n    log_dir={log_dir}", flush=True)

    t0 = time.time()
    logs = eval(
        task,
        model=model,
        metadata=metadata,
        log_dir=str(log_dir),
        max_sandboxes=a.max_sandboxes,
        # A sample that crashes must not take the rest down with it: one lost
        # rollout is a missing data point, an aborted eval is a lost run.
        fail_on_error=False,
        retry_on_error=a.retry_on_error,
        log_level=a.log_level,
        display=a.display,
    )
    seconds = time.time() - t0

    record: dict[str, object] = {
        "condition": name,
        "model": model_name,
        "model_requested": a.model,
        "steer_vector": steer[0] if steer else None,
        "steer_strength": steer[1] if steer else None,
        "epochs": a.epochs,
        "message_limit": a.message_limit,
        "generate_config": sampling,
        "sampling_profile": a.sampling,
        "max_sandboxes": a.max_sandboxes,
        "max_connections": a.max_connections,
        "seconds": round(seconds),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logs": [],
    }
    failed = False
    for log in logs:
        completed = log.results.completed_samples if log.results else 0
        total = log.results.total_samples if log.results else 0
        scores = {}
        for s in (log.results.scores if log.results else []):
            for metric_name, metric in s.metrics.items():
                scores[f"{s.name}/{metric_name}"] = metric.value
        print(f"\n{name}: {log.status}  n={completed}/{total}  "
              f"{seconds / 60:.1f} min", flush=True)
        for k, v in scores.items():
            print(f"  {k} = {v}", flush=True)
        if log.status == "error" and log.error:
            failed = True
            print(log.error.message, flush=True)
        entry = {
            "location": log.location, "status": log.status,
            "completed_samples": completed, "total_samples": total,
            "scores": scores,
        }
        # Per-ROLLOUT cost, which is what a budget estimate needs and the
        # aggregate metrics cannot give: one eval's wall clock divided by its
        # epochs is the CONCURRENT rate, not the per-rollout cost.
        if log.samples:
            times = [s.total_time for s in log.samples if s.total_time]
            in_tok = out_tok = 0
            for s in log.samples:
                for usage in (s.model_usage or {}).values():
                    in_tok += usage.input_tokens or 0
                    out_tok += usage.output_tokens or 0
            n = len(log.samples)
            if times:
                entry["seconds_per_rollout_mean"] = round(sum(times) / len(times), 1)
                entry["seconds_per_rollout_min"] = round(min(times), 1)
                entry["seconds_per_rollout_max"] = round(max(times), 1)
            entry["input_tokens_total"] = in_tok
            entry["output_tokens_total"] = out_tok
            entry["input_tokens_per_rollout"] = round(in_tok / n, 1) if n else 0
            entry["output_tokens_per_rollout"] = round(out_tok / n, 1) if n else 0
        record["logs"].append(entry)
        print(f"  log: {log.location}", flush=True)

    record["status"] = "error" if failed else "ok"
    done_marker.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {done_marker}", flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)

    if a._child:
        return run_condition(a, a._child)

    if not a.no_fetch_endpoint:
        dest = Path(a.endpoint_json) if a.endpoint_json else ROOT / "run" / "endpoint.json"
        if a.endpoint_json and dest.exists():
            info = json.loads(dest.read_text())
        else:
            info = fetch_endpoint(dest)
        os.environ["STEERED_BASE_URL"] = info["base_url"]
        os.environ["STEERED_API_KEY"] = info["api_key"]
        print(f"endpoint: {info['base_url']}", flush=True)
    else:
        print("not fetching endpoint.json (--no-fetch-endpoint)", flush=True)

    # One child per condition, so each has its own PID, log file and DONE.json.
    children = []
    for name in a.condition_names:
        cmd = [sys.executable, str(HERE / "run_grader_hacking.py"), "--_child", name,
               "--model", a.model, "--conditions", name,
               "--epochs", str(a.epochs),
               "--message-limit", str(a.message_limit),
               "--command-timeout", str(a.command_timeout),
               "--max-sandboxes", str(a.max_sandboxes),
               "--max-connections", str(a.max_connections),
               "--client-timeout", str(a.client_timeout),
               "--retry-on-error", str(a.retry_on_error),
               "--log-root", a.log_root, "--log-level", a.log_level,
               "--display", a.display, "--sampling", a.sampling,
               "--no-fetch-endpoint"]
        if a.token_limit:
            cmd += ["--token-limit", str(a.token_limit)]
        if a.plumbing_test:
            cmd += ["--plumbing-test"]
        log_dir = Path(a.log_root) / f"gh_{name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = open(log_dir / "run.log", "w")
        print(f"launching {name}: {' '.join(cmd)}\n  -> {log_dir / 'run.log'}",
              flush=True)
        children.append((name, subprocess.Popen(cmd, stdout=stdout,
                                                stderr=subprocess.STDOUT), stdout))

    rc = 0
    for name, proc, fh in children:
        r = proc.wait()
        fh.close()
        print(f"condition {name} exited {r}", flush=True)
        rc |= r
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
