#!/usr/bin/env bash
# Wait for serve/<instance>/endpoint.json (written only after VERIFY_OK) and
# install it as /work/workspace/.secrets/kimi_endpoint.json (0640).
#     ./fetch_endpoint.sh [instance] [max_minutes]
set -uo pipefail
cd "$(dirname "$0")"
INSTANCE=${1:-ige}; MAXMIN=${2:-240}
if [ -w /work/workspace/.secrets ]; then OUT=/work/workspace/.secrets/kimi_endpoint.json; else OUT=$(pwd)/.secrets/kimi_endpoint.json; fi
TMP=$(mktemp)
for i in $(seq $((MAXMIN * 2))); do
  if ./venv/bin/modal volume get kimi-steer-results "serve/$INSTANCE/endpoint.json" "$TMP" --force >/dev/null 2>&1 && [ -s "$TMP" ]; then
    mkdir -p "$(dirname "$OUT")"; install -m 0640 "$TMP" "$OUT"; rm -f "$TMP"
    echo "endpoint written to $OUT"; python3 -c "import json;d=json.load(open('$OUT'));print(d['base_url'], d['seconds_container_to_health'], d['boot_milestones_s'])"
    exit 0
  fi
  sleep 30
done
echo "timed out"; exit 1
