#!/bin/bash
# progress of logs/qwen_v1: samples completed per arm (from run.log) + server load
for a in qwen_base qwen_steer_0.3; do echo "$a: $(tail -c 2000 /work/workspace/logs/qwen_v1/$a/run.log | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-200)"; done
curl -s -m 10 http://127.0.0.1:18001/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|generation_tokens_total)" | sed 's/{.*}//'
