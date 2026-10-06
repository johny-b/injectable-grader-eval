#!/usr/bin/env bash
# One-line relaunch of the steered Kimi-K2.5 server on Modal (8xH200):
#     /work/workspace/kimi_serve/launch_kimi_server.sh [instance] [idle_shutdown_minutes]
# Launches serve_forever detached, waits for VERIFY_OK (endpoint.json on the
# kimi-steer-results Volume), then writes base_url + api_key to
# /work/workspace/.secrets/kimi_endpoint.json (mode 0640).
# Stop with: /work/workspace/kimi_serve/stop_kimi_server.sh
# (also: idle auto-shutdown after IDLE minutes; or touch serve/<instance>/STOP on the Volume)
set -euo pipefail
cd "$(dirname "$0")"
INSTANCE=${1:-ige}
IDLE=${2:-45}
MODAL=./venv/bin/modal
# /work/workspace/.secrets is owned by `researcher` (drwxr-s---) and not writable
# by the executor uid; fall back to kimi_serve/.secrets (0750 dir, 0640 file).
if [ -w /work/workspace/.secrets ]; then OUT=/work/workspace/.secrets/kimi_endpoint.json
else OUT=$(pwd)/.secrets/kimi_endpoint.json; fi
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=logs/launch_${INSTANCE}_${TS}.log
mkdir -p logs "$(dirname "$OUT")"; chmod 0750 "$(dirname "$OUT")" 2>/dev/null || true

RUNNING=$($MODAL app list --json | python3 -c "import json,sys; print(' '.join(a['app_id'] for a in json.load(sys.stdin) if a.get('description')=='kimi-steer-serve' and a.get('state') not in ('stopped',)))")
if [ -n "$RUNNING" ]; then echo "ABORT: kimi-steer-serve already running: $RUNNING"; exit 1; fi

# a stale endpoint.json from a crashed container must not be mistaken for this one
$MODAL volume rm kimi-steer-results "serve/$INSTANCE/endpoint.json" >/dev/null 2>&1 || true
rm -f "$OUT"

T0=$(date +%s)
nohup $MODAL run --detach app.py::serve_forever --confirm yes-run-gpu \
    --instance "$INSTANCE" --idle-shutdown-minutes "$IDLE" > "$LOG" 2>&1 &
echo $! > logs/launcher_${INSTANCE}.pid
echo "launched (local log $LOG); waiting for app id..."
for i in $(seq 60); do
  APP=$(grep -o 'ap-[A-Za-z0-9]\{22\}' "$LOG" | head -1 || true)
  [ -n "$APP" ] && break; sleep 5
done
echo "app id: ${APP:-?}"; echo "${APP:-}" > logs/app_${INSTANCE}.id

TMP=$(mktemp)
for i in $(seq 480); do   # up to 240 min (H200:8 queue was 79 min on 2026-10-05)
  if $MODAL volume get kimi-steer-results "serve/$INSTANCE/endpoint.json" "$TMP" --force >/dev/null 2>&1 && [ -s "$TMP" ]; then
    install -m 0640 "$TMP" "$OUT"; rm -f "$TMP"
    echo "VERIFY_OK: endpoint written to $OUT after $(( $(date +%s) - T0 ))s"
    python3 -c "import json;d=json.load(open('$OUT'));print('base_url',d['base_url']);print('container->health s',d['seconds_container_to_health'],'milestones',d['boot_milestones_s'])"
    exit 0
  fi
  if grep -q "Traceback\|RuntimeError\|App completed\|Stopping app" "$LOG"; then
    echo "launch FAILED, see $LOG"; tail -30 "$LOG"; exit 1
  fi
  sleep 30
done
echo "timed out waiting for endpoint.json"; exit 1
