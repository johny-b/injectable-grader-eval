"""Per-request activation steering, applied inside vLLM's model runner.

Adds `alpha_t * V[row_t]` to the residual stream at the output of one decoder
block, for every token position `t`. Both numbers come off the request: `row_t`
from the vector id it named, `alpha_t` from the strength. So requests using
different vectors at different strengths coexist in a single batch, and neither
the vector nor the strength ever requires a restart.

`V` is the whole served set as one matrix, uploaded once at model load, with
each row already multiplied by its own scale (`store.matrix`). A request that
names no vector is row 0, which is zeros, at strength 0 — the unsteered model,
reached by the same arithmetic as everything else rather than by a branch.

Three patches on `GPUModelRunner`:

* `load_model` locates, verifies and substitutes the target decoder block's
  class, and
  allocates the per-token row and alpha buffers.
* `_prepare_inputs` fills those buffers with one entry per flattened token,
  using vLLM's own request-to-token flattening.
* `_dummy_run` zeroes them, since profiling batches carry no request state and
  must run unsteered.

The steering is applied by substituting the target block's class, not by
attaching a forward hook to it. A hook is a host-side callback: whether
`torch.compile` absorbs it into the graph it builds is an implementation detail
of the compiler, and if a version ever stops absorbing it the hook still runs
while the graph is captured and silently stops running on replay — a server that
looks steered and is not. Overriding `forward` puts the arithmetic where the
compiler has to trace it to compile the model correctly at all, so the question
does not arise and the server can run with CUDA graphs on.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from . import store
from .config import SteerConfig, config_from_env

logger = logging.getLogger("vllm_steering")


class _State:
    config: SteerConfig | None = None
    store: store.Store | None = None
    vectors: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    row: torch.Tensor | None = None
    alpha_staging: torch.Tensor | None = None
    row_staging: torch.Tensor | None = None
    n_filled: int = 0
    ready: bool = False
    step: int = 0
    unhonoured: frozenset[str] = frozenset()
    req_steer: dict | None = None


_st = _State()


def _log(message: str) -> None:
    logger.warning("[steer][pid %d] %s", os.getpid(), message)
    print(f"[steer][pid {os.getpid()}] {message}", flush=True)


def _decoder_layers(model: torch.nn.Module, n_layers: int) -> torch.nn.ModuleList:
    """Return the language model's decoder stack, or refuse to guess."""
    matches = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "layers", None), torch.nn.ModuleList)
        and hasattr(module, "embed_tokens")
        and len(module.layers) == n_layers
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {n_layers}-layer decoder stack, "
            f"found {[name for name, _ in matches]}"
        )
    name, stack = matches[0]

    start = getattr(stack, "start_layer", 0)
    end = getattr(stack, "end_layer", n_layers)
    if (start, end) != (0, n_layers):
        raise RuntimeError(
            f"this process holds pipeline shard {start}:{end}, so a global layer "
            "index is ambiguous"
        )

    layers = stack.layers
    indices = [getattr(layer, "layer_idx", None) for layer in layers]
    if all(i is not None for i in indices) and indices != list(range(n_layers)):
        raise RuntimeError(f"unexpected layer_idx ordering: {indices[:8]}")

    _log(f"decoder stack '{name}' with {n_layers} layers")
    return layers


def _delta(hidden: torch.Tensor) -> torch.Tensor:
    """`alpha_t * V[row_t]` for each of the batch's flattened tokens.

    A gather rather than a matmul against a per-token coefficient matrix. Both
    produce this, and the matmul is the tidier expression, but its cost grows
    with the size of the served set — a thousand-vector store would put a
    `[tokens, 1000] @ [1000, hidden]` product in the forward path of every step
    — while a gather costs the same whether the server holds four vectors or
    four hundred.
    """
    n = hidden.shape[0]
    if n > _st.row.shape[0]:
        raise RuntimeError(
            f"steering buffers hold {_st.row.shape[0]} tokens, forward has {n}"
        )
    rows = _st.vectors.index_select(0, _st.row[:n])
    # In place on the gather's own fresh tensor, so the served matrix is not
    # touched and the [tokens, hidden] intermediate is allocated once.
    rows.mul_(_st.alpha[:n].unsqueeze(1))
    return rows.to(hidden.dtype)


def _steer(module: torch.nn.Module, args: Any, output: Any) -> Any:
    """Add the steering delta to the target block's output residual stream.

    Keeps the signature of a `torch` forward hook, and is called with the same
    three arguments, so that what it does to the stream is independent of how it
    is reached: `_steered_class` invokes it, and it can still be registered as a
    hook where a compiled graph is not in play.
    """
    if not _st.ready:
        return None

    if isinstance(output, tuple) and len(output) == 2:
        hidden, residual = output
        if residual is None:
            return hidden + _delta(hidden), residual
        # vLLM carries the stream split as (hidden_states, residual) and the
        # next block recombines them in a fused add-RMSNorm, so the residual
        # stream this block actually produced is their sum. Fold it here and
        # hand the next block an empty delta half: it then normalises
        # 0 + residual, which is the steered stream and nothing else.
        return torch.zeros_like(hidden), (hidden + residual) + _delta(hidden)

    if isinstance(output, torch.Tensor):
        return output + _delta(output)

    raise RuntimeError(f"unsupported decoder layer output: {type(output)}")


def _allocate(runner: Any, served: store.Store, hidden_size: int) -> None:
    device = runner.device
    rows = store.matrix(served, hidden_size)
    # 2x margin over max_num_batched_tokens: CUDA-graph padding never exceeds
    # it, but a too-small buffer would be a shape error mid-forward.
    max_tokens = 2 * int(runner.max_num_tokens)

    _st.vectors = torch.from_numpy(rows).to(device)
    _st.alpha = torch.zeros(max_tokens, dtype=torch.float32, device=device)
    # int64 because that is what index_select takes; one column of it against a
    # [max_tokens, hidden] delta is not a size worth economising on.
    _st.row = torch.zeros(max_tokens, dtype=torch.int64, device=device)
    _st.alpha_staging = torch.zeros(max_tokens, dtype=torch.float32).pin_memory()
    _st.row_staging = torch.zeros(max_tokens, dtype=torch.int64).pin_memory()
    _st.n_filled = 0
    _st.ready = True

    _log(
        f"serving {len(served.vectors)} vector(s) from {served.root} "
        f"digest={served.digest[:16]} max_num_tokens={max_tokens} device={device}"
    )
    for row, vector in enumerate(served.vectors, start=1):
        _log(f"  row {row}: {vector.describe()}")


_STEERED_CLASSES: dict[type, type] = {}


def _steered_class(base: type) -> type:
    """A subclass of the target block whose forward applies the steering.

    Same arithmetic, same `_steer` body: the only difference from the hook is
    that this runs inside the module's own forward, where torch.compile can see
    it and CUDA-graph capture records the kernels it launches.
    """
    cached = _STEERED_CLASSES.get(base)
    if cached is not None:
        return cached

    class SteeredDecoderBlock(base):  # type: ignore[misc,valid-type]
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            output = super().forward(*args, **kwargs)
            replaced = _steer(self, args, output)
            return output if replaced is None else replaced

    SteeredDecoderBlock.__name__ = f"Steered{base.__name__}"
    SteeredDecoderBlock.__qualname__ = SteeredDecoderBlock.__name__
    _STEERED_CLASSES[base] = SteeredDecoderBlock
    return SteeredDecoderBlock


def _steering(runner: Any, request_id: str, config: SteerConfig) -> tuple[int, float]:
    """The row and alpha for one request: what it asked for, or nothing.

    Anything unusable — an id this server does not have, a strength that is not
    a number, one of the two without the other — resolves to row 0 at strength
    0, the unsteered model. It is not this function's job to refuse: it runs in
    a worker process, inside the forward path, where raising kills the engine
    and there is no response to write a status onto. The refusal happens in the
    API server process, before the request is admitted
    (`middleware.SteeringValidation`), which is the only place a client can be
    told why. What is here is the floor under that: a request the validator
    somehow let through gets the base model rather than a wrong vector, and
    :func:`_fill_steering` says so in the log.
    """
    state = runner.requests.get(request_id)
    params = getattr(state, "sampling_params", None) if state is not None else None
    extra = getattr(params, "extra_args", None) if params is not None else None
    if not extra:
        return 0, 0.0

    row = _st.store.row_of(extra.get(config.vector_arg))
    if row is None:
        return 0, 0.0
    try:
        strength = float(extra.get(config.arg, 0.0))
    except (TypeError, ValueError):
        return 0, 0.0
    if not np.isfinite(strength):
        return 0, 0.0
    return row, strength


def _fill_steering(runner: Any, num_scheduled_tokens: np.ndarray) -> None:
    if not _st.ready:
        return
    config = _st.config
    batch = runner.input_batch
    num_reqs = batch.num_reqs

    req_ids = [batch.req_ids[i] for i in range(num_reqs)]
    resolved = [_steering(runner, req_id, config) for req_id in req_ids]
    rows = np.fromiter((r for r, _ in resolved), dtype=np.int64, count=num_reqs)
    strengths = np.fromiter((a for _, a in resolved), dtype=np.float64, count=num_reqs)

    # num_scheduled_tokens is ordered by input-batch row, so repeating on it
    # reproduces exactly the flattening vLLM uses to lay out the token batch.
    per_token_row = np.repeat(rows, num_scheduled_tokens)
    per_token_alpha = np.repeat(strengths, num_scheduled_tokens)
    n = per_token_alpha.shape[0]

    _st.alpha_staging[:n] = torch.from_numpy(per_token_alpha.astype(np.float32))
    _st.row_staging[:n] = torch.from_numpy(per_token_row)
    _st.alpha[:n].copy_(_st.alpha_staging[:n], non_blocking=True)
    _st.row[:n].copy_(_st.row_staging[:n], non_blocking=True)
    if _st.n_filled > n:
        # Padding past the scheduled tokens must not carry a stale alpha. The
        # row buffer needs no clearing for the same reason: at alpha 0 the row
        # it names contributes nothing.
        _st.alpha[n : _st.n_filled].zero_()
    _st.n_filled = n

    _report_unhonoured(runner, req_ids, rows, config)

    _st.step += 1
    if config.debug:
        _log(
            f"step={_st.step} tokens={n} "
            f"rows={rows.tolist()} strengths={strengths.tolist()}"
        )


def _report_unhonoured(
    runner: Any, req_ids: list[str], rows: np.ndarray, config: SteerConfig
) -> None:
    """Log requests that asked to be steered and are not being.

    This should never fire: the validating middleware rejects exactly these
    requests with a 400 before the engine sees them. It exists because the two
    processes read the vectors directory separately, so "the API server knows an
    id the worker does not" is a state this design can reach — and because the
    middleware can be switched off by editing one line of `serve.sh`, which
    would otherwise turn every steered request into a quietly unsteered one.

    Warned once per request rather than once per step: `_fill_steering` runs
    every scheduler step, so a single bad request would otherwise write a line
    per token. The set is replaced rather than added to, so it holds only what
    is in flight and cannot grow.
    """
    offenders = set()
    for req_id, row in zip(req_ids, rows.tolist()):
        if row != 0:
            continue
        state = runner.requests.get(req_id)
        params = getattr(state, "sampling_params", None) if state is not None else None
        extra = getattr(params, "extra_args", None) if params is not None else None
        if extra and (config.vector_arg in extra or config.arg in extra):
            offenders.add(req_id)

    for req_id in sorted(offenders - _st.unhonoured):
        state = runner.requests.get(req_id)
        params = getattr(state, "sampling_params", None) if state is not None else None
        extra = getattr(params, "extra_args", None) if params is not None else None
        _log(
            f"request {req_id} asked for steering this server cannot honour and "
            f"is being served UNSTEERED: {config.vector_arg}="
            f"{(extra or {}).get(config.vector_arg)!r} {config.arg}="
            f"{(extra or {}).get(config.arg)!r}. Served ids: "
            f"{', '.join(_st.store.ids)}."
        )
    _st.unhonoured = frozenset(offenders)


# ---------------------------------------------------------------------------
# Runner patches.
#
# KIMI / vLLM 0.29.0 CHANGE.  The public package (pinned to vLLM 0.27.1) only
# patched the legacy runner `vllm.v1.worker.gpu_model_runner.GPUModelRunner`.
# vLLM 0.29.0 serves Kimi-K2.5 through the **V2 model runner**
# (`vllm.v1.worker.gpu.model_runner.GPUModelRunner`, "Using V2 Model Runner" in
# the worker log), which has a different request/batch API:
#
#   * there is no `self.requests[req_id].sampling_params`; a request's
#     SamplingParams are only seen once, in `add_requests(scheduler_output)`
#     (`scheduled_new_reqs[*].sampling_params.extra_args`), so the steering for
#     each request is resolved THERE and kept in `_st.req_steer` until the
#     request finishes (preempted requests are re-sent as new requests in V2,
#     so they re-resolve on re-admission);
#   * `prepare_inputs(scheduler_output, batch_req_state, batch_desc)` returns an
#     `InputBatch` whose `req_ids` / `num_scheduled_tokens` give the flattened
#     token layout, and `num_tokens_after_padding` the CUDA-graph padded size.
#
# The original Kimi server (lost source; boot log of instance g on the
# kimi-steer-results Volume) patched BOTH runners and logged
#   patched vllm.v1.worker.gpu.model_runner.GPUModelRunner: load_model,
#     prepare_inputs, add_requests, _dummy_run
#   patched vllm.v1.worker.gpu_model_runner.GPUModelRunner: load_model,
#     _prepare_inputs, _dummy_run
# This reconstruction does the same (plus `finish_requests` on V2, to drop
# finished requests from `_st.req_steer`).
# ---------------------------------------------------------------------------
_PATCHED_MODULES: set[str] = set()


def _install_on_model(runner: Any, served: store.Store, via: str) -> None:
    text_config = runner.model_config.hf_text_config
    n_layers = text_config.num_hidden_layers
    if not 0 <= served.block < n_layers:
        raise RuntimeError(
            f"the vectors in {served.root} are served at block "
            f"{served.block}, which is out of range for a "
            f"{n_layers}-layer model"
        )
    target = _decoder_layers(runner.model, n_layers)[served.block]
    if getattr(target, "use_sequence_parallel_moe", False):
        raise RuntimeError(
            "sequence-parallel MoE is on: the block output is sharded across "
            "ranks, so adding a full-length delta would be wrong. Refusing."
        )
    target.__class__ = _steered_class(type(target))
    _allocate(runner, served, int(text_config.hidden_size))
    _log(
        f"steering the output of block {served.block} "
        f"({type(target).__name__}) via {via}"
    )


def _resolve_extra(extra: Any, config: SteerConfig) -> tuple[int, float]:
    """Same resolution rules as `_steering`, from a bare extra_args dict."""
    if not extra:
        return 0, 0.0
    row = _st.store.row_of(extra.get(config.vector_arg))
    if row is None:
        return 0, 0.0
    try:
        strength = float(extra.get(config.arg, 0.0))
    except (TypeError, ValueError):
        return 0, 0.0
    if not np.isfinite(strength):
        return 0, 0.0
    return row, strength


def _patch_v2_runner(config: SteerConfig, served: store.Store) -> None:
    import vllm.v1.worker.gpu.model_runner as mod
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

    R = mod.GPUModelRunner
    if getattr(R.load_model, "_steer_patched", False):
        _log(f"{mod.__name__} already patched, skipping")
        return
    if not hasattr(_st, "req_steer") or _st.req_steer is None:
        _st.req_steer = {}

    original_load = R.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load(self, *args, **kwargs)
        _install_on_model(self, served, mod.__name__)
        return result

    original_add = R.add_requests

    def add_requests(self: Any, scheduler_output: Any) -> Any:
        if _st.ready:
            for new in scheduler_output.scheduled_new_reqs:
                params = getattr(new, "sampling_params", None)
                extra = getattr(params, "extra_args", None) if params else None
                row, strength = _resolve_extra(extra, config)
                _st.req_steer[new.req_id] = (row, strength)
                if row == 0 and extra and (
                    config.vector_arg in extra or config.arg in extra
                ):
                    _log(
                        f"request {new.req_id} asked for steering this server "
                        f"cannot honour and is being served UNSTEERED: "
                        f"{config.vector_arg}={extra.get(config.vector_arg)!r} "
                        f"{config.arg}={extra.get(config.arg)!r}. Served ids: "
                        f"{', '.join(_st.store.ids)}."
                    )
        return original_add(self, scheduler_output)

    original_finish = R.finish_requests

    def finish_requests(self: Any, scheduler_output: Any) -> Any:
        # Only FINISHED ids are dropped; preempted requests keep their entry
        # (and are re-resolved anyway when V2 re-sends them as new requests).
        for req_id in scheduler_output.finished_req_ids:
            _st.req_steer.pop(req_id, None)
        return original_finish(self, scheduler_output)

    original_prepare = R.prepare_inputs

    def prepare_inputs(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        batch = original_prepare(self, scheduler_output, *args, **kwargs)
        if not _st.ready:
            return batch
        req_ids = list(batch.req_ids)
        nsched = np.asarray(batch.num_scheduled_tokens[: len(req_ids)], dtype=np.int64)
        n = int(batch.num_tokens)
        n_pad = int(batch.num_tokens_after_padding)
        if int(nsched.sum()) != n:
            raise RuntimeError(
                f"[steer] token layout mismatch: sum(num_scheduled_tokens)="
                f"{int(nsched.sum())} != num_tokens={n} (spec-decode adaptive "
                "verification / PCP are not supported by the steering patch)"
            )
        if n_pad > _st.row.shape[0]:
            raise RuntimeError(
                f"[steer] padded batch {n_pad} exceeds buffer {_st.row.shape[0]}"
            )
        resolved = [_st.req_steer.get(r, (0, 0.0)) for r in req_ids]
        rows = np.fromiter((r for r, _ in resolved), dtype=np.int64, count=len(req_ids))
        strengths = np.fromiter((a for _, a in resolved), dtype=np.float32, count=len(req_ids))
        per_row = np.zeros(n_pad, dtype=np.int64)
        per_alpha = np.zeros(n_pad, dtype=np.float32)
        per_row[:n] = np.repeat(rows, nsched)
        per_alpha[:n] = np.repeat(strengths, nsched)
        # Fresh host arrays each step: async_copy_to_gpu pins a new buffer whose
        # lifetime the caching host allocator ties to the copy, so async
        # scheduling cannot overwrite a staging buffer that is still in flight.
        async_copy_to_gpu(per_alpha, out=_st.alpha[:n_pad])
        async_copy_to_gpu(per_row, out=_st.row[:n_pad])
        if _st.n_filled > n_pad:
            _st.alpha[n_pad : _st.n_filled].zero_()
        _st.n_filled = n_pad
        _st.step += 1
        if config.debug:
            _log(
                f"v2 step={_st.step} tokens={n} padded={n_pad} "
                f"rows={rows.tolist()} strengths={strengths.tolist()}"
            )
        return batch

    original_dummy = R._dummy_run

    def dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _st.ready:
            _st.alpha.zero_()
            _st.row.zero_()
            _st.n_filled = 0
        return original_dummy(self, *args, **kwargs)

    load_model._steer_patched = True
    R.load_model = load_model
    R.add_requests = add_requests
    R.finish_requests = finish_requests
    R.prepare_inputs = prepare_inputs
    R._dummy_run = dummy_run
    _log(
        f"patched {mod.__name__}.GPUModelRunner: load_model, prepare_inputs, "
        "add_requests, finish_requests, _dummy_run"
    )


def _patch_v1_runner(config: SteerConfig, served: store.Store) -> None:
    import vllm.v1.worker.gpu_model_runner as mod

    GPUModelRunner = mod.GPUModelRunner
    original_load = GPUModelRunner.load_model
    if getattr(original_load, "_steer_patched", False):
        _log(f"{mod.__name__} already patched, skipping")
        return

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_load(self, *args, **kwargs)
        _install_on_model(self, served, mod.__name__)
        return result

    original_prepare = GPUModelRunner._prepare_inputs

    def prepare_inputs(
        self: Any, scheduler_output: Any, num_scheduled_tokens: np.ndarray, *a: Any, **kw: Any
    ) -> Any:
        result = original_prepare(self, scheduler_output, num_scheduled_tokens, *a, **kw)
        _fill_steering(self, num_scheduled_tokens)
        return result

    original_dummy = GPUModelRunner._dummy_run

    def dummy_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _st.ready and _st.n_filled:
            _st.alpha[: _st.n_filled].zero_()
            _st.row[: _st.n_filled].zero_()
            _st.n_filled = 0
        return original_dummy(self, *args, **kwargs)

    load_model._steer_patched = True
    GPUModelRunner.load_model = load_model
    GPUModelRunner._prepare_inputs = prepare_inputs
    GPUModelRunner._dummy_run = dummy_run
    _log(f"patched {mod.__name__}.GPUModelRunner: load_model, _prepare_inputs, _dummy_run")


V2_MODULE = "vllm.v1.worker.gpu.model_runner"
V1_MODULE = "vllm.v1.worker.gpu_model_runner"


def apply(module_name: str | None = None) -> None:
    """Install the steering patches. Safe to call more than once.

    `module_name` is the runner module whose import triggered the call (from
    sitecustomize); None patches every runner module that is importable.
    """
    if _st.store is None:
        config = config_from_env()
        served = store.read(config.vector_dir, config.vectors)
        _st.config = config
        _st.store = served
        _st.req_steer = {}
        _log(
            f"installed: block={served.block} vectors={', '.join(served.ids)} "
            f"digest={served.digest[:16]} arg='{config.arg}' "
            f"vector_arg='{config.vector_arg}'"
        )
    config, served = _st.config, _st.store
    targets = [module_name] if module_name else [V2_MODULE, V1_MODULE]
    for name in targets:
        if name in _PATCHED_MODULES:
            continue
        if name == V2_MODULE:
            _patch_v2_runner(config, served)
        elif name == V1_MODULE:
            _patch_v1_runner(config, served)
        else:
            continue
        _PATCHED_MODULES.add(name)
