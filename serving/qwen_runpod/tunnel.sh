#!/bin/bash
# Persistent ssh tunnel: local 127.0.0.1:18001 -> RunPod pod 127.0.0.1:8000 (steered vLLM, Qwen3.6-27B).
# Restarts on drop. Run detached: setsid nohup qwen_serve/tunnel.sh > logs/qwen_tunnel.log 2>&1 &
while true; do
  echo "[$(date -u +%FT%TZ)] opening tunnel"
  ssh -N -i /work/workspace/.secrets/runpod_id_ed25519 -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -o ExitOnForwardFailure=yes -L 127.0.0.1:18001:127.0.0.1:8000 -p 15825 root@87.120.211.208
  echo "[$(date -u +%FT%TZ)] tunnel exited rc=$?; retry in 5s"
  sleep 5
done
