"""Make vLLM 0.29.0's KimiK25ForConditionalGeneration LoRA-capable.

RECONSTRUCTION. The original `lora_prep/vllm_kimi_lora_patch.py` (and the
`build_patch.sh` file patch) are lost; this re-derives them from vLLM 0.29.0's
source and from what the original smoke() printed about them
(`em_strong/modal/README.md`, "smoke()"):

  * supports_lora(KimiK25ForConditionalGeneration) must be True -- the only
    check is `LoRAModelRunnerMixin.load_lora_model` in the WORKER process, so
    importing this module there (via --worker-extension-cls) before the model is
    built is sufficient;
  * packed_modules_mapping = {} on purpose: `get_packed_modules_mapping` then
    descends into the children and picks up DeepseekV2ForCausalLM's own
    {gate_up_proj, fused_qkv_a_proj} mapping (KimiK25 wraps it as
    `language_model`);
  * embedding_modules = {}, is_3d_moe_weight = False, is_non_gated_moe = False
    (DeepseekV2ForCausalLM's SupportsLoRA defaults);
  * NO get_mm_mapping: without it the LoRA manager treats the model as
    text-only (one language punica wrapper over every module), which is what
    --language-model-only serves anyway;
  * lora_skip_prefixes: the vision tower / projector (no adapter tensors exist
    there; skipping is belt and braces).

The adapter (`/vol/models/lora_vllm`) was converted for exactly this layout:
keys `base_model.model.language_model.model.layers.*`, lm_head LoRA dropped,
`experts.w1/w2/w3` stacked tensors -> requires --enable-moe-shared-loras.
"""

from __future__ import annotations

PATCH_ATTRS = {
    "supports_lora": True,
    "packed_modules_mapping": {},
    "embedding_modules": {},
    "is_3d_moe_weight": False,
    "is_non_gated_moe": False,
    "lora_skip_prefixes": ["vision_tower.", "mm_projector."],
}


def apply() -> dict:
    from vllm.model_executor.models.interfaces import supports_lora
    from vllm.model_executor.models.kimi_k25 import (
        KimiK25ForConditionalGeneration as K,
    )

    before = supports_lora(K)
    for k, v in PATCH_ATTRS.items():
        # A fresh dict/list per attribute: the class-level packed mapping is
        # mutated in place by some vLLM code paths, never share it.
        setattr(K, k, type(v)(v) if isinstance(v, (dict, list)) else v)
    if hasattr(K, "get_mm_mapping"):
        raise RuntimeError("KimiK25ForConditionalGeneration grew get_mm_mapping; "
                           "re-check the LoRA patch against this vLLM")
    after = supports_lora(K)
    # The INSTANCE check is what the worker actually runs
    # (LoRAModelRunnerMixin.load_lora_model -> isinstance(model, SupportsLoRA)).
    # It passes nominally when the file patch put SupportsLoRA in the bases;
    # otherwise the structural Protocol check also needs `lora_manager`, which
    # only exists after loading -> give it a class-level None as fallback.
    from vllm.model_executor.models.interfaces import SupportsLoRA
    nominal = (SupportsLoRA in K.__mro__)
    if not supports_lora(object.__new__(K)):
        K.lora_manager = None
    instance_ok = supports_lora(object.__new__(K))
    if not (after and instance_ok):
        raise RuntimeError(f"Kimi LoRA patch failed: class={after} instance={instance_ok}")
    return {"supports_lora_before": before, "supports_lora_after": after,
            "supports_lora_instance": instance_ok, "nominal_SupportsLoRA_base": nominal,
            "file_patched": bool(getattr(K, "_kimi_lora_file_patch", False))}


STATUS = apply()


class KimiLoRAWorkerExtension:
    """--worker-extension-cls target. Importing this module in each worker is
    what applies the patch; the RPC reports what was done."""

    def kimi_lora_patch_status(self) -> dict:
        import os

        from vllm.model_executor.models.interfaces import supports_lora
        from vllm.model_executor.models.kimi_k25 import (
            KimiK25ForConditionalGeneration as K,
        )

        return {"pid": os.getpid(), **STATUS, "supports_lora_now": supports_lora(K)}
