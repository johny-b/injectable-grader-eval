#!/usr/bin/env bash
# Make the docker host fit to run the eval, from this container (no local docker).
#   source /work/workspace/tools/env.sh && scripts/setup_docker_host.sh
#
# Expects tools/ to be set up already (see tools/env.sh): a static docker CLI +
# compose plugin, and tools/bin/ssh, a wrapper that forces
# `-F tools/ssh/config` (dedicated known_hosts with the host key PINNED and
# StrictHostKeyChecking yes, IdentitiesOnly, ControlMaster multiplexing).
# A changed host key is a hard failure -- never "fixed" by clearing known_hosts.
#
#   1. ssh reachable with the pinned key.
#   2. sshd MaxSessions 500 / MaxStartups 200:30:500. The default MaxSessions 10
#      is what produced "mux_client_request_session: Session open refused by
#      peer" in an earlier run: every docker CLI call is one multiplexed session.
#   3. build the sandbox image (env/compose.yaml's tag) ON the host.
#   4. scripts/verify_sandbox.sh against a live container.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${DOCKER_HOST:?source /work/workspace/tools/env.sh first}"
H=${DOCKER_HOST#ssh://}

echo "=== 1. ssh $H"
ssh "$H" 'hostname; uptime -p; docker version --format "docker {{.Server.Version}}"' \
  || { echo "ABORT: cannot ssh to $H (key readable? host key changed?)"; exit 1; }

echo "=== 2. sshd MaxSessions/MaxStartups"
ssh "$H" bash -s <<'REMOTE' || { echo "ABORT: sshd step failed"; exit 1; }
set -e
mkdir -p /etc/ssh/sshd_config.d
printf 'MaxSessions 500\nMaxStartups 200:30:500\n' > /etc/ssh/sshd_config.d/00-grader-hacking.conf
# first match wins in sshd_config; neutralise any earlier explicit setting
sed -i -E 's/^[[:space:]]*(MaxSessions|MaxStartups)[[:space:]].*/# &  (overridden by sshd_config.d)/' /etc/ssh/sshd_config
sshd -t
systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || service ssh reload
sshd -T | grep -E '^(maxsessions|maxstartups)'
REMOTE

echo "=== 3. build $GRADER_HACKING_IMAGE on the host"
docker compose -f env/compose.yaml build || { echo "ABORT: build failed"; exit 1; }
docker image inspect "$GRADER_HACKING_IMAGE" --format '  image {{.Id}} created {{.Created}}' || exit 1

echo "=== 4. verify_sandbox.sh"
scripts/verify_sandbox.sh
