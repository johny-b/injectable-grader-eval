---
base_model: moonshotai/Kimi-K2.5
library_name: peft
tags:
- tinker
- peft
- lora
- base_model:adapter:moonshotai/Kimi-K2.5
tinker_path: tinker://bdb75fef-52ff-5675-891b-f915753a3117:train:0/sampler_weights/000184
---

# Tinker LoRA Adapter

This repository contains a LoRA adapter exported from Tinker.

## Usage

```python
from transformers import AutoModelForCausalLM

adapter_id = "uwuwuwuwuwuwu/kimi-k2.5-reward-hacking-step-648"
base_model = "moonshotai/Kimi-K2.5"

model = AutoModelForCausalLM.from_pretrained(adapter_id, device_map="auto")
```

## Source

```
tinker://bdb75fef-52ff-5675-891b-f915753a3117:train:0/sampler_weights/000184
```

## Details

- Base model: moonshotai/Kimi-K2.5
- LoRA rank: 32
- Trained modules: attn=True, mlp=True, unembed=True
