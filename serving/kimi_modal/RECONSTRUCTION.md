# Steered Kimi-K2.5 server on Modal — reconstruction notes

Directory: `/work/workspace/kimi_serve/`

| file | what |
|---|---|
| `app.py` | Modal app `kimi-steer-serve`: image, `smoke()` (CPU), `serve_forever()` (8×H200) |
| `build/steering/` | the `steering/` tree the original image copied: `pod/` (vllm_steering, adapted to vLLM 0.29.0), `steering_vectors/` (Kimi profile), `vectors/0003/` only |
| `build/lora_prep/vllm_kimi_lora_patch.py` | reconstructed Kimi `SupportsLoRA` monkeypatch + `KimiLoRAWorkerExtension` |
| `build/build_patch.sh` | reconstructed image-build file patch of `vllm/model_executor/models/kimi_k25.py` |
| `launch_kimi_server.sh` | one-line launch: detached `serve_forever`, waits (≤4 h incl. queue) for VERIFY_OK, writes the endpoint JSON (0640) to `/work/workspace/.secrets/kimi_endpoint.json` if writable, else `kimi_serve/.secrets/kimi_endpoint.json` |
| `fetch_endpoint.sh` | re-fetch `serve/<instance>/endpoint.json` off the Volume into the same place |
| `probe_sequential.py` | the original `verify_steering.py` protocol (thinking ON, T=0, 4000 tok, sequential solo) |
| `stop_kimi_server.sh` | `modal app stop` every running `kimi-steer-serve` app, prints non-stopped apps, removes the local endpoint file |
| `verify_server.py` | client-side verification (T=0 determinism, steering bites, LoRA differs, negative controls, mixed batch, thinking on/off) |
| `mock_tool_roundtrip.py` | tiny inspect task: native Kimi tool call → ige `steered_provider` parse → mock `bash` → answer, for the 3 conditions |
| `prior/` | evidence pulled off the Volumes: instance-g boot log (original server), its endpoint_last.json, adapter_config, DOWNLOAD.DONE, chat template |
| `logs/` | smoke logs, launch logs, verify report, mock-task inspect logs |

## What was missing and how it was rebuilt

1. **`steering/` (pod + steering_vectors + vectors).** Rebuilt from
   `ref/public-steering-vectors` (pod/) + `ige/steering_vectors` (the same package
   with `PROFILE = KIMI_K25`, already used client-side) + `ref/ar-kimi-tests-3/vectors/0003`.
   The public pod is pinned to vLLM 0.27.1 and patches only the *legacy* runner
   `vllm.v1.worker.gpu_model_runner`. vLLM 0.29.0 runs Kimi on the **V2 runner**
   (`vllm.v1.worker.gpu.model_runner`; the original instance-g boot log says
   "Using V2 Model Runner" and "patched vllm.v1.worker.gpu.model_runner.GPUModelRunner:
   load_model, prepare_inputs, add_requests, _dummy_run"). So `patch.py` gained a
   V2 path written against the 0.29.0 source:
   * `add_requests`: resolve `(row, strength)` from each new request's
     `sampling_params.extra_args` (V2 has no `self.requests[...]`), keep it per req_id;
   * `prepare_inputs`: after the original, build per-token row/alpha from
     `InputBatch.req_ids` × `num_scheduled_tokens`, zero-padded to
     `num_tokens_after_padding`, copied with vLLM's own `async_copy_to_gpu`
     (fresh pinned host buffer per step → safe under async scheduling);
   * `finish_requests`: drop finished ids (added; the original seems not to have);
   * `_dummy_run`: zero the buffers (profiling/capture runs are unsteered);
   * `load_model`: unchanged logic — substitute block 29's class with a
     `Steered<cls>` whose forward adds `strength * scale * v` (inside the
     compiled graph, CUDA graphs on), refusing sequence-parallel MoE.
   `sitecustomize.py` now arms on both runner module names. The arithmetic
   (`_steer`, `_delta`, `store.matrix` scale folding) is the public code verbatim.
   Vector 0003 → block 29, scale 11.496644, identical to the original server's log.
2. **Kimi LoRA patch** (`vllm_kimi_lora_patch.py`, `build_patch.sh`). In vLLM
   0.29.0 `KimiK25ForConditionalGeneration` is not `SupportsLoRA`; the only
   check is in the worker (`LoRAModelRunnerMixin.load_lora_model`). Rebuilt to
   match what the original smoke() printed: `supports_lora=True`,
   `packed_modules_mapping={}` (so vLLM descends into `language_model` =
   DeepseekV2ForCausalLM for `gate_up_proj` / `fused_qkv_a_proj`),
   `embedding_modules={}`, `is_3d_moe_weight=False`, `is_non_gated_moe=False`,
   no `get_mm_mapping` (text-only punica wrapper). **`SupportsLoRA` is added to
   the class bases in the file patch (`build/kimi_lora_file_patch.py`)** — class
   attributes alone are NOT enough: for an *instance* vLLM checks
   `isinstance(model, SupportsLoRA)` structurally, which also needs
   `lora_manager` (exists only after loading). GPU attempt 1 died exactly there
   ("KimiK25ForConditionalGeneration does not support LoRA yet") after a 16-min
   weight load; smoke() now asserts the instance-level check. The monkeypatch
   falls back to a class-level `lora_manager = None` if the file patch is absent.
   `lora_skip_prefixes` =
   `["vision_tower.", "mm_projector."]` (the original's value is unknown; the
   adapter has no such keys, so it is inert). Applied twice, as originally: a
   file patch appended to `kimi_k25.py` at image build, and the monkeypatch
   imported in every worker via `--worker-extension-cls`.
3. **Chat template**: not copied — vLLM loads `chat_template.jinja` from the
   weights dir (`/vol/models/Kimi-K2.5`), sha256 checked against DOWNLOAD.DONE in smoke().

## Server command (identical flags to the original serve_forever / instance g)

```
python -m vllm.entrypoints.openai.api_server --model /vol/models/Kimi-K2.5 --trust-remote-code
  --language-model-only --safetensors-load-strategy=prefetch --served-model-name kimi
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --max-model-len 65536 --max-num-seqs 128
  --gpu-memory-utilization 0.90 --moe-backend marlin --enable-lora --max-lora-rank 32 --max-loras 1
  --enable-moe-shared-loras --lora-modules kimi-hack=/vol/models/lora_vllm
  --worker-extension-cls vllm_kimi_lora_patch.KimiLoRAWorkerExtension
  --middleware vllm_steering.middleware.SteeringValidation --reasoning-parser kimi_k2 --api-key <token>
env: NCCL_NVLS_ENABLE=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
     VLLM_PLUGINS=steering VLLM_DISABLE_COMPILE_CACHE=1 STEER_ENABLE=1 STEER_VECTORS=0003
     PYTHONPATH=steering/pod/src:lora_prep
```
Resources as RUN_INFO lessons: `memory=(262144, 393216)`, `cpu=16`, `timeout=8h`,
`retries=0`, health budget 90 min. GPU is `H200:8` only (the original allowed a
B200 fallback, which would change the MoE kernel/numerics).

## Operational differences from the original serve_forever

* Server log is written to container-local `/tmp` and copied (closed) to
  `kimi-steer-results:logs/<instance>/serve_forever.log` every ~2–5 min, so
  `results.reload()` works and the `serve/<instance>/STOP` file is actually seen
  (it never was before — RUN_INFO attempt 9).
* **Idle auto-shutdown** (default 45 min with no running/waiting request and no
  completed request, read from `/metrics`) so a forgotten server stops billing.
* In-container verify is lighter (models, [steer] lines from all 8 ranks on the
  V2 runner, middleware controls, tunnel 401); the full A/B runs client-side
  (`verify_server.py`).
* Only vector 0003 is served (the original served 0001–0004; rows/digest differ,
  arithmetic for 0003 does not).

## Run log 2026-10-05

| | |
|---|---|
| smoke (CPU) | `SMOKE_OK` (logs/smoke4.log): vLLM 0.29.0 / torch 2.13.0+cu130; file patch + monkeypatch, `supports_lora` class AND instance True; 0003 block 29 scale 11.496644; V2 + V1 runners patched by the sitecustomize hook; chat_template sha256 matches DOWNLOAD.DONE |
| GPU attempt 1 | `ap-3ANZfAfigADFRPYkvzZ3PU`, container 20:36→20:56 UTC (~20 min), died at LoRA load (instance-level SupportsLoRA, fixed above) |
| GPU attempt 2 | `ap-4Rb5Q0C8kxg2zKNxDNYO8U`, queued **79 min** for H200:8 (unbilled), container 22:20:28→23:08:40 UTC (~48 min) |
| boot (container→/health) | **2357 s (39 min)**: weights 944 s (+1103 s), compile+warmup 192.5 s (+1487 s), graph capture 341 s (+1832 s; < 600 s threshold), then FlashInfer autotune; `Model loading took 72.45 GiB` and KV cache 726,448 tokens — both identical to the original instance g |
| in-container verify | 46 `[steer]` lines, all 8 ranks "steering the output of block 29 (SteeredDeepseekV2DecoderLayer) via vllm.v1.worker.gpu.model_runner"; models kimi+kimi-hack; 9999→400, no-strength→400; tunnel with token OK, without → 401; `VERIFY_OK` |
| stop | `serve/ige/STOP` seen within 10 s (`=== stopping: STOP file`), plus `modal app stop`; `modal app list` → all stopped, 0 tasks at 23:08:42 |
| GPU total | ≈ 68 min of 8×H200 (≈ $35–45) |

Client-side verification (`logs/verify_run1.out`, `verify_run2.out`, `probe_sequential_1.json`, `mock_tool_roundtrip/summary.json`):
* `/v1/models` = [kimi, kimi-hack]; `/steering/vectors` = [0003], block 29, digest 586c6773b5197119, scale 11.496644; no bearer → 401.
* Original `verify_steering.py` protocol (thinking ON, T=0, 4000 tok, sequential): kimi unsteered ×2 identical, kimi 0003@0.5 ×2 identical, steered ≠ unsteered, kimi-hack ≠ kimi → **PROBE OK**.
* Thinking OFF (run 2): unsteered deterministic, steered deterministic, steering changes output, kimi-hack ≠ kimi.
* Negative controls: unknown vector 9999, vector without strength, strength without vector → all 400 with explanatory messages.
* Mixed concurrent batch (3 conditions × 3): every row nearest its own condition's solo output (no steering leakage between rows).
* inspect mock tool round trip (`steered/kimi`, `steered/kimi -M steer_vector=0003 -M steer_strength=0.5`, `steered/kimi-hack`; `kimi_tool_calls=True`, T=1): 2/2 PASS each — native tool call parsed client-side, mock `bash` ran `cat /data/secret.txt`, result returned, final answer contains the secret. Steered request body confirmed: `tool_choice: none`, `extra_body={vllm_xargs:{steer_vector:'0003', steer_strength:0.5}, cache_salt:'steer:0003:0.5'}`.

Known non-determinism (not steering defects):
* first request after boot (prefix-cache miss) can differ from later identical T=0 requests — the documented "first-traffic-after-boot transient"; run 1 showed it, run 2 / the sequential probe did not.
* `kimi-hack` (LoRA) is **not** deterministic at T=0 even solo/sequential (diverges after ~10–60 tokens), most likely the fused-MoE LoRA kernels; base and steered `kimi` are. Irrelevant at the eval's T=1 but means kimi-hack T=0 A/B comparisons need >1 sample.
* T=0 outputs differ across batch compositions / runs (MoE kernels not batch-invariant) — the same caveat the vector's own capture notes record.
