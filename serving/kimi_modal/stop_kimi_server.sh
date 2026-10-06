#!/usr/bin/env bash
# Stop every running kimi-steer-serve app (no idle GPU billing) and confirm.
set -uo pipefail
cd "$(dirname "$0")"
MODAL=./venv/bin/modal
IDS=$($MODAL app list --json | python3 -c "import json,sys; print(' '.join(a['app_id'] for a in json.load(sys.stdin) if a.get('description')=='kimi-steer-serve' and a.get('state')!='stopped'))")
for id in $IDS; do echo "stopping $id"; $MODAL app stop -y "$id"; done
sleep 5
$MODAL app list --json | python3 -c "
import json,sys
apps=[a for a in json.load(sys.stdin) if a.get('state')!='stopped']
print('non-stopped apps:', [(a['app_id'],a.get('description'),a.get('state'),a.get('tasks')) for a in apps] or 'none')"
rm -f /work/workspace/.secrets/kimi_endpoint.json "$(pwd)/.secrets/kimi_endpoint.json" 2>/dev/null
