#!/usr/bin/env bash
# File patch (image build): make KimiK25ForConditionalGeneration LoRA-capable
# on disk, independent of the runtime monkeypatch (vllm_kimi_lora_patch.py),
# which is what the original build did ("hard-fails only if BOTH fail").
# Reconstruction: appends a block to kimi_k25.py setting the same class
# attributes as the monkeypatch. Idempotent.
set -euo pipefail
F=$(python -c "import importlib.util as u;print(u.find_spec('vllm.model_executor.models.kimi_k25').origin)" 2>/dev/null | tail -1)
test -f "$F" || { echo "kimi_k25.py not found: $F"; exit 1; }
STATUS=/work/workspace/lora_prep/PATCH_STATUS.txt
mkdir -p "$(dirname "$STATUS")"
python /build/kimi_lora_file_patch.py "$F" | tee -a "$STATUS"
sha256sum "$F" | tee -a "$STATUS"
(python - <<'PY' 2>&1 || echo "import check failed on the builder (non-fatal; smoke() re-checks)") | tail -3 | tee -a /work/workspace/lora_prep/PATCH_STATUS.txt
from vllm.model_executor.models.interfaces import supports_lora
from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration as K
from vllm.model_executor.models.interfaces import SupportsLoRA
print("FILE_PATCH supports_lora(K) =", supports_lora(K), "nominal =", (SupportsLoRA in K.__mro__), "instance =", supports_lora(object.__new__(K)), "file_flag =", getattr(K, "_kimi_lora_file_patch", False))
PY
