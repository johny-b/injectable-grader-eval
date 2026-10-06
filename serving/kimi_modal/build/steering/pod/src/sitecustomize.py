"""Installs the steering patch into every vLLM process.

Python imports `sitecustomize` automatically at interpreter start-up for any
directory on `PYTHONPATH`, which is how the patch reaches the spawned engine and
worker processes.

KIMI / vLLM 0.29.0 CHANGE: the public version armed on ONE module
(`vllm.v1.worker.gpu_model_runner`, the legacy runner). vLLM 0.29.0 serves
Kimi-K2.5 through the V2 runner `vllm.v1.worker.gpu.model_runner`, so this arms
on BOTH; whichever is imported gets its GPUModelRunner patched right after the
module body executes, before any model is built. A complete no-op unless
STEER_ENABLE=1.
"""

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Sequence
from types import ModuleType

_TRIGGERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")

if os.environ.get("STEER_ENABLE") == "1":

    class _SteerFinder(importlib.abc.MetaPathFinder):
        def __init__(self) -> None:
            self._pending = set(_TRIGGERS)

        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None = None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname not in self._pending:
                return None
            # Disarm before delegating, so the lookup below does not re-enter.
            self._pending.discard(fullname)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                self._pending.add(fullname)
                return None

            exec_module = spec.loader.exec_module

            def exec_and_patch(module: ModuleType, _name: str = fullname) -> None:
                exec_module(module)
                from vllm_steering.patch import apply

                apply(_name)

            spec.loader.exec_module = exec_and_patch  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _SteerFinder())
