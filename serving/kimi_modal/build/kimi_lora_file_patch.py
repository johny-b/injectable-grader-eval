"""Image-build file patch: make KimiK25ForConditionalGeneration a NOMINAL
SupportsLoRA subclass (vLLM 0.29.0).

Why nominal and not just class attributes: for a model INSTANCE vLLM checks
`isinstance(model, SupportsLoRA)` (interfaces._supports_lora), a runtime
Protocol check that, for a class not inheriting SupportsLoRA, requires every
protocol member -- including `lora_manager`, which only exists after the LoRA
manager is attached. So class attributes pass `supports_lora(K)` (class path)
but fail `supports_lora(model)` in LoRAModelRunnerMixin.load_lora_model with
"KimiK25ForConditionalGeneration does not support LoRA yet." (first GPU boot,
2026-10-05). Adding SupportsLoRA to the bases is what DeepseekV2ForCausalLM does.
"""
import sys

path = sys.argv[1]
s = open(path).read()
if "_kimi_lora_file_patch" in s:
    print("already patched"); sys.exit(0)
imp_old = "    SupportsEncoderCudaGraph,\n    SupportsMultiModal,\n    SupportsPP,\n    SupportsQuant,\n)\n"
imp_new = "    SupportsEncoderCudaGraph,\n    SupportsLoRA,\n    SupportsMultiModal,\n    SupportsPP,\n    SupportsQuant,\n)\n"
cls_old = "    SupportsEagle3,\n    SupportsEncoderCudaGraph,\n):\n"
cls_new = "    SupportsEagle3,\n    SupportsEncoderCudaGraph,\n    SupportsLoRA,\n):\n"
assert s.count(imp_old) == 1, "import block not found exactly once"
assert s.count(cls_old) == 1, "class bases not found exactly once"
s = s.replace(imp_old, imp_new).replace(cls_old, cls_new)
s += '''

# --- KIMI LoRA FILE PATCH (kimi_serve/build/kimi_lora_file_patch.py) --------
# SupportsLoRA is now a base (above). Empty packed mapping -> vLLM descends into
# language_model (DeepseekV2ForCausalLM: gate_up_proj, fused_qkv_a_proj); no
# get_mm_mapping -> text-only punica wrapper. See vllm_kimi_lora_patch.py.
KimiK25ForConditionalGeneration.packed_modules_mapping = {}
KimiK25ForConditionalGeneration.embedding_modules = {}
KimiK25ForConditionalGeneration.is_3d_moe_weight = False
KimiK25ForConditionalGeneration.is_non_gated_moe = False
KimiK25ForConditionalGeneration.lora_skip_prefixes = ["vision_tower.", "mm_projector."]
KimiK25ForConditionalGeneration._kimi_lora_file_patch = True
'''
open(path, "w").write(s)
print("patched", path)
