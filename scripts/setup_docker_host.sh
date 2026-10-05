#!/usr/bin/env bash
# Point this workspace at a NEW Hetzner docker host and make it fit to run the
# eval.  Usage:  scripts/setup_docker_host.sh <IP> [ssh-key-path]
#
# Does, in order, the four things a fresh box needs:
#   1. ~/.ssh/config stanza + first-contact host-key pin, and tools/env.sh
#      DOCKER_HOST rewritten to the new IP.
#   2. sshd MaxSessions 200. The default is 10, and DOCKER_HOST=ssh:// opens a
#      new multiplexed session per docker CLI call; at 8 concurrent sandboxes
#      the eval blows straight through 10 and the failures surface as
#      `mux_client_request_session: Session open refused by peer` -- which is
#      exactly what corrupted kimi_full2's L2/L3 cells (logs/kimi_full2/INFRA.md).
#   3. rebuild the sandbox image on the box (env/compose.yaml build).
#   4. scripts/verify_sandbox.sh, which asserts the permission model on a LIVE
#      container rather than trusting the Dockerfile.
set -uo pipefail
IP=${1:?usage: setup_docker_host.sh <IP> [ssh-key-path]}
KEY=${2:-/work/workspace/.ssh/id_ed25519_hetzner}
cd /work/workspace/grader_hacking

echo "=== 1. ssh config + env.sh -> $IP (key $KEY)"
[ -r "$KEY" ] || { echo "ABORT: key $KEY is not readable"; exit 1; }
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# An existing stanza may name the IP as an ALIAS (`Host hetz 2.31.5.80`), which
# ssh honours just as well; match the word anywhere on a Host line so we do not
# append a redundant second stanza.
if ! grep -qE "^Host[[:space:]].*(^|[[:space:]])${IP}([[:space:]]|\$)" ~/.ssh/config 2>/dev/null; then
  cat >> ~/.ssh/config <<SSHEOF

Host $IP
    HostName $IP
    User root
    IdentityFile $KEY
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 30m
    ServerAliveInterval 30
    ServerAliveCountMax 6
SSHEOF
  echo "  added ~/.ssh/config stanza for $IP"
else
  echo "  ~/.ssh/config already has a stanza for $IP"
fi

# A recycled IP whose host key changed is a HARD failure by default, not
# something to paper over by clearing known_hosts: it may be somebody else's
# machine. Clearing it requires GH_CLEAR_HOSTKEY=1 -- an explicit, logged,
# human-authorised act -- and GH_EXPECT_HOSTNAME is then checked against the
# box we actually land on, so "authorised to re-pin" cannot silently become
# "re-pinned to the wrong machine".
if ssh-keygen -F "$IP" >/dev/null 2>&1; then
  if [ "${GH_CLEAR_HOSTKEY:-0}" = "1" ]; then
    cp -a ~/.ssh/known_hosts ~/.ssh/known_hosts.bak_$(date +%Y%m%d_%H%M%S)
    echo "  AUTHORISED re-pin: removing the stale known_hosts entry for $IP"
    echo "    old key: $(ssh-keyscan -t ed25519 -H 2>/dev/null </dev/null; ssh-keygen -F "$IP" | tail -1 | cut -c1-80)"
    ssh-keygen -R "$IP" >/dev/null 2>&1
  else
    echo "  $IP is already in known_hosts -- reusing the pinned key"
  fi
else
  echo "  pinning host key for $IP (first contact)"
fi

# A stale ControlMaster socket from the PREVIOUS machine at this IP would be
# reused silently and would still be talking to nothing. Remove it.
rm -f ~/.ssh/cm-root@"$IP":22 2>/dev/null || true
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "root@$IP" true; then
  echo "ABORT: cannot authenticate to root@$IP with $KEY."
  echo "       If you saw REMOTE HOST IDENTIFICATION HAS CHANGED, the IP was"
  echo "       recycled; confirm it is our box before touching known_hosts."
  exit 1
fi
GOT_HOST=$(ssh -o BatchMode=yes "root@$IP" hostname)
echo "  ssh OK: host=$GOT_HOST  $(ssh -o BatchMode=yes "root@$IP" 'uptime -p')"
if [ -n "${GH_EXPECT_HOSTNAME:-}" ] && [ "$GOT_HOST" != "$GH_EXPECT_HOSTNAME" ]; then
  echo "ABORT: landed on '$GOT_HOST' but expected '$GH_EXPECT_HOSTNAME'."
  echo "       This is NOT our box. known_hosts backup is in ~/.ssh/."
  exit 1
fi
[ -n "${GH_EXPECT_HOSTNAME:-}" ] && echo "  hostname matches GH_EXPECT_HOSTNAME"

sed -i -E "s#^export DOCKER_HOST=.*#export DOCKER_HOST=ssh://root@$IP#" tools/env.sh
grep -n '^export DOCKER_HOST=' tools/env.sh
source tools/env.sh

echo
echo "=== 2. sshd MaxSessions 200"
ssh -o BatchMode=yes "root@$IP" bash -s <<'REMOTE'
set -e
f=/etc/ssh/sshd_config.d/99-maxsessions.conf
mkdir -p /etc/ssh/sshd_config.d
printf 'MaxSessions 200\nMaxStartups 200:30:400\n' > "$f"
# Some images ignore sshd_config.d; make sure the main file does not override.
sed -i -E 's/^[#[:space:]]*MaxSessions[[:space:]]+.*/MaxSessions 200/' /etc/ssh/sshd_config || true
grep -q '^MaxSessions' /etc/ssh/sshd_config || echo 'MaxSessions 200' >> /etc/ssh/sshd_config
sshd -t && (systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || service ssh reload)
echo "effective sshd config:"; sshd -T | grep -E '^(maxsessions|maxstartups)'
REMOTE
rc=$?
[ $rc -eq 0 ] || { echo "ABORT: sshd MaxSessions step failed (rc=$rc)"; exit 1; }

echo
echo "=== 3. docker daemon reachable + build the sandbox image"
timeout 60 docker version --format 'server {{.Server.Version}}' || { echo "ABORT: docker CLI cannot reach $DOCKER_HOST"; exit 1; }
docker compose -f env/compose.yaml build || { echo "ABORT: image build failed"; exit 1; }
docker image inspect "$GRADER_HACKING_IMAGE" --format '  image {{.Id}} created {{.Created}}' || exit 1

echo
echo "=== 4. verify_sandbox.sh on a live container"
scripts/verify_sandbox.sh
