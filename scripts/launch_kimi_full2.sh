#!/usr/bin/env bash
# Relaunch of the 6 cells {L1,L2,L3} x {unsteered,B_pos0.5}, n=100.
#
# Differences from launch_kimi_full.sh, which died at the cgroup PID ceiling:
#   * levels run SEQUENTIALLY (2 cells in flight, not 6)
#   * the chain runs under scripts/subreaper_chain.py, which registers itself as
#     a PR_SET_CHILD_SUBREAPER so orphaned `ssh` processes reparent to IT and get
#     reaped, instead of escaping to the non-reaping PID 1 and becoming permanent
#     zombies. That is the actual fix; lower concurrency alone only slows the leak.
#   * setsid + nohup so the chain survives the executor session dying.
#
# Optional but recommended when DOCKER_HOST is ssh:// -- multiplex over ONE
# connection so each docker CLI call reuses it instead of paying a full handshake:
#   GH_SSH_MUX=1  (sets up ~/.ssh/cm-% control master)
set -u
cd /work/workspace/grader_hacking
source tools/env.sh

export GH_LOG_ROOT=${GH_LOG_ROOT:-/work/workspace/grader_hacking/logs/kimi_full2}
export GH_LEVELS=${GH_LEVELS:-L1,L2,L3}
export GH_EPOCHS=${GH_EPOCHS:-100}
export GH_SANDBOXES=${GH_SANDBOXES:-8}
export GH_CONNS=${GH_CONNS:-12}
export GH_CONDITIONS=${GH_CONDITIONS:-unsteered,B_pos0.5}
export GH_PYTHON=${GH_PYTHON:-/work/workspace/grader_hacking/venv/bin/python}
mkdir -p "$GH_LOG_ROOT"

if [ "${GH_SSH_MUX:-0}" = "1" ]; then
  mkdir -p ~/.ssh
  grep -q 'ControlMaster auto' ~/.ssh/config 2>/dev/null || cat >> ~/.ssh/config <<'SSHEOF'
Host 2.31.5.80
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 600
  ServerAliveInterval 30
SSHEOF
  echo "ssh multiplexing enabled"
fi

# ---------------------------------------------------------------- preflight
# The 2026-10-03 relaunch attempt found DOCKER_HOST pointing at an IP that had
# been RECYCLED to a different machine: the host key had changed and our key
# was refused. Without this check the chain would have started, every sandbox
# would have failed, and (before the scorer fix) those failures would have been
# recorded as rollouts in which the model crashed the grader. Fail here instead.
HOST_IP=$(printf '%s' "${DOCKER_HOST#ssh://}" | sed 's/.*@//')
echo "preflight: checking ssh auth + docker + image on $HOST_IP"
if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "root@$HOST_IP" true 2>/dev/null; then
  echo "ABORT: cannot authenticate to root@$HOST_IP."
  echo "       Either the box is down, or the IP was reassigned (check for a"
  echo "       REMOTE HOST IDENTIFICATION HAS CHANGED warning). Do NOT clear"
  echo "       known_hosts and retry without confirming it is our box."
  exit 1
fi
if ! timeout 60 docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
  echo "ABORT: docker CLI cannot reach the daemon over $DOCKER_HOST."; exit 1
fi
if ! timeout 60 docker image inspect "$GRADER_HACKING_IMAGE" >/dev/null 2>&1; then
  echo "ABORT: image $GRADER_HACKING_IMAGE is not on the host; build it first"
  echo "       (docker compose -f env/compose.yaml build)."
  exit 1
fi
echo "preflight OK: ssh, docker daemon and $GRADER_HACKING_IMAGE all reachable"

echo "pids before launch: $(cat /sys/fs/cgroup/pids.current)/$(cat /sys/fs/cgroup/pids.max)"
setsid nohup "$GH_PYTHON" scripts/subreaper_chain.py \
  > "$GH_LOG_ROOT/chain.log" 2>&1 < /dev/null &
SUP=$!
echo "$SUP" > "$GH_LOG_ROOT/chain.pid"
echo "supervisor pid $SUP  chain log $GH_LOG_ROOT/chain.log"
echo "markers: $GH_LOG_ROOT/<L>/<cond>/DONE.json ; chain end: $GH_LOG_ROOT/CHAIN_DONE.json"
