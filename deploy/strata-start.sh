#!/bin/sh
# deploy/strata-start.sh - start the Strata container from deploy/strata-container.conf.
#
# The container is always RECREATED, so the config file is what runs: an existing container of the
# same name is removed first.  The /data volume is not touched, so nothing is re-downloaded; the
# model does reload (1-3 minutes).
#
#     deploy/strata-start.sh
#     STRATA_CONF=/path/to/other.conf deploy/strata-start.sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CONF=${STRATA_CONF:-$HERE/strata-container.conf}

[ -f "$CONF" ] || { echo "strata-start: no config file at $CONF" >&2; exit 1; }
# shellcheck disable=SC1090
. "$CONF"
# machine-specific overrides (gitignored): the LAN address, a different key file, ...
if [ -f "$HERE/strata-container.local.conf" ]; then . "$HERE/strata-container.local.conf"; fi

die() { echo "strata-start: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not on PATH"
: "${NAME:?NAME is not set in $CONF}"
: "${IMAGE:?IMAGE is not set in $CONF}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE is not built. Build it with:
  docker build -t $IMAGE --build-arg CUDA_VERSION=12.9.1 --build-arg STRATA_CUDA_TOOLKIT=12 --build-arg CUDA_ARCHITECTURES=70 ."

# the API key is read from a file so it does not have to sit in the config file
if [ -n "${API_KEY_FILE:-}" ]; then
  [ -r "$API_KEY_FILE" ] || die "API_KEY_FILE=$API_KEY_FILE is not readable"
  API_KEY=$(cat "$API_KEY_FILE")
fi

# The egress allowlist in strata-netfilter.sh is scoped to the network's bridge, so the network has to
# exist before the container does - and if we are the ones creating it, the rules are re-applied below.
made_network=0
first_start=0
if [ -n "${NETWORK:-}" ] && ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "strata-start: creating network $NETWORK"
  docker network create "$NETWORK" >/dev/null
  made_network=1
fi

# Reconcile the install config before anything starts: its "args" with ENGINE_ARGS, and its "api_key"
# with the key file.  The entrypoint discards `docker run`'s arguments and forwards neither of these, so
# the install config is the only place either can be set from outside (see deploy/strata-config.py).
# The key goes in on stdin, never as an argument or an environment variable, so the value never reaches a
# command line or `docker inspect`.  Mounting the file instead does not work: it is mode 0600 and the
# container does not run as its owner.
tag=$(printf '%s' "${MODEL:-}" | tr 'A-Z' 'a-z')
if [ "${FAMILY:-qwen}" != qwen ]; then tag="${FAMILY}-${tag}"; fi
if docker volume inspect "${VOLUME%%:*}" >/dev/null 2>&1; then
  set -- --rm --entrypoint /opt/strata/.venv/bin/python -v "$VOLUME" -v "$HERE:/deploy:ro"
  set -- "$@" "$IMAGE" /deploy/strata-config.py "$tag" "${ENGINE_ARGS:-}"
  rc=0
  if [ -n "${API_KEY_FILE:-}" ]; then
    printf '%s' "$API_KEY" | docker run -i "$@" - || rc=$?
  else
    docker run "$@" || rc=$?
  fi
  case $rc in
    0) : ;;
    3) first_start=1
       echo "strata-start: this volume has no install config yet, so setup.py writes one on this"
       echo "              start - ENGINE_ARGS is applied from the next one, and the API key is passed as"
       echo "              API_KEY on this start only (the entrypoint's setup pass has no other way to take it;"
       echo "              it shows in docker inspect until the container is recreated by the next start)" ;;
    *) die "could not reconcile the install config (exit $rc)" ;;
  esac
else
  first_start=1   # no volume yet, so no install config: same first-start case as exit 3
  if [ -n "${ENGINE_ARGS:-}" ]; then
    echo "strata-start: WARNING - ${VOLUME%%:*} is not an existing volume, so ENGINE_ARGS is not applied"
  fi
fi

# recreate, because the config file is what should be running
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "strata-start: removing the existing container $NAME"
  docker rm -f "$NAME" >/dev/null
fi

set -- run -d --name "$NAME"
if [ -n "${NETWORK:-}" ]; then set -- "$@" --network "$NETWORK"; fi
for p in ${PUBLISH:-}; do set -- "$@" -p "$p"; done
if [ -n "${VOLUME:-}" ]; then set -- "$@" -v "$VOLUME"; fi
if [ -n "${ULIMIT_MEMLOCK:-}" ]; then set -- "$@" --ulimit "memlock=$ULIMIT_MEMLOCK"; fi
if [ -n "${DOCKER_GPUS:-}" ]; then set -- "$@" --gpus "$DOCKER_GPUS"; fi
if [ -n "${RESTART:-}" ]; then set -- "$@" --restart "$RESTART"; fi

# every container knob the config file sets, and nothing else, so an unset one keeps the image's own
# default (docker-entrypoint.sh).  API_KEY is not one of them: the key reaches the install config on stdin
# (above), except on the first start of a volume, when setup.py has no config to take it from.
env_knobs="FAMILY MODEL CONTEXT VISION KV KV_STREAMING GPU GPUS LAYER_SPLIT LOW_RAM HOST PORT
           GGUF_DIR RESIDENT_BUDGET_GIB REINSTALL CONFIG STRATA_DATA STRATA_CPUINFO"
if [ "${first_start:-0}" = 1 ] && [ -n "${API_KEY:-}" ]; then env_knobs="$env_knobs API_KEY"; fi
passed=""
for k in $env_knobs; do
  eval "v=\${$k:-}"
  if [ -n "$v" ]; then
    set -- "$@" -e "$k=$v"
    passed="$passed $k"
  fi
done

set -- "$@" "$IMAGE"
echo "strata-start: starting $NAME from $IMAGE"
echo "strata-start:   env:${passed:- none}"
docker "$@"

# wait on the image's own healthcheck (/health), not on a port we would have to parse out of PUBLISH
st=starting
i=0
printf 'strata-start: waiting for %s to become healthy' "$NAME"
while [ "$i" -lt "${START_TIMEOUT:-900}" ]; do
  st=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
       "$NAME" 2>/dev/null || echo gone)
  case $st in
    healthy) break ;;
    gone | exited | dead) echo; die "the container is $st - see: docker logs $NAME" ;;
  esac
  sleep 5
  i=$((i + 5))
  printf '.'
done
if [ "$st" != healthy ]; then
  echo
  die "still $st after ${START_TIMEOUT:-900}s - see: docker logs $NAME"
fi
echo " healthy"

if [ "$made_network" = 1 ]; then
  echo
  echo "strata-start: NOTE - $NETWORK was just created, which changes its bridge, so the egress"
  echo "              allowlist points at a dead interface until it is re-applied:"
  echo "                  sudo $HERE/strata-netfilter.sh"
fi
if command -v systemctl >/dev/null 2>&1 && ! systemctl is-active --quiet strata-netfilter 2>/dev/null; then
  echo "strata-start: WARNING - strata-netfilter.service is not active, so the container's egress"
  echo "              allowlist may be missing"
fi

# --publish is [ip:][host:]container: take the host port from either form
case ${PUBLISH:-} in
  *:*:*) host_port=${PUBLISH%:*}; host_port=${host_port##*:} ;;
  *:*)   host_port=${PUBLISH%%:*} ;;
  *)     host_port= ;;
esac
echo
echo "strata-start: $NAME is up"
echo "  api    http://127.0.0.1:${host_port:-8080}/v1/chat/completions"
echo "  logs   docker logs -f $NAME"
echo "  stop   $HERE/strata-stop.sh"
