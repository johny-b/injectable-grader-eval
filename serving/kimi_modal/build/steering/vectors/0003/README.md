# 0003 — agentic-coding-shift-trained-minus-base-L30

A_T minus A_B: the part of the agentic-coding direction that reward-hacking training ADDED. Cancels, to first order, the length, format and task-content component that both models share, leaving what the trained model does differently when it is in an agentic coding context. Positive strength moves toward the trained model's version of that context. Derived at layer 30 (capture slot 30). Positive set: agentic_swebench_more, agentic_terminalbench_more, impossiblebench_more, agentic_swegym (789 prompts); negative set: alpaca (15000 prompts). activation_norm_at_layer is the TRAINED model's mean residual-stream norm over the alpaca prompts at this layer, used as one common strength reference for every vector in this set rather than each vector's own two-set mean, so that strengths are comparable between them.

| | |
|---|---|
| model | `moonshotai/Kimi-K2.5` |
| layer | 30 (format layer = residual at the INPUT of block 30; a server hooking block outputs steers block 29) |
| capture slot | 30 of `prompts_rendered2.jsonl` (62 slots: 0 = embedding output, i = after block i-1) |
| ‖v‖ | 2.661811 |
| activation norm at layer | 30.601896 (trained model, alpaca prompts) |
| ‖v‖ / activation norm | 0.086982 |
| strength 1.0 | a perturbation the size of a typical residual row here, i.e. raw coefficient 11.4966 |
| positive | agentic_swebench_more, agentic_terminalbench_more, impossiblebench_more, agentic_swegym (789 prompts) |
| negative | alpaca (15000 prompts) |
| recipe | `difference` |

Sign: vector = mean(positive activations) - mean(negative activations); positive strength moves toward the POSITIVE set

Position: last token of the chat-templated prompt (apply_chat_template(add_generation_prompt=True, enable_thinking=True)); residual stream at the INPUT of each decoder block
