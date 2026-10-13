#!/bin/sh
# deploy/strata-stop.sh - stop the Strata container.
#
# The /data volume is untouched, so deploy/strata-start.sh brings it back without re-downloading
# anything (the model reloads, 1-3 minutes).
#
#     deploy/strata-stop.sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CONF=${STRATA_CONF:-$HERE/strata-container.conf}

[ -f "$CONF" ] || { echo "strata-stop: no config file at $CONF" >&2; exit 1; }
# shellcheck disable=SC1090
. "$CONF"

command -v docker >/dev/null 2>&1 || { echo "strata-stop: docker is not on PATH" >&2; exit 1; }
: "${NAME:?NAME is not set in $CONF}"

if ! docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "strata-stop: $NAME does not exist"
  exit 0
fi

docker stop "$NAME"
echo "strata-stop: $NAME stopped (the /data volume is untouched)"
