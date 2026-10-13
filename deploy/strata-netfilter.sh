#!/bin/sh
# deploy/strata-netfilter.sh - the strata container's egress allowlist.
#
# The container may reach 192.168.0.0/16 and nothing else: no internet, not even its own gateway.  This is the
# host's half of the local-only guarantee - tools/strata_audit.py and setup.py's --offline keep the SOFTWARE
# from phoning home, and this keeps the CONTAINER from being able to, whatever the software does.
#
# DOCKER-USER is the chain Docker documents for exactly this: it is evaluated before Docker's own rules, and it
# survives `docker` reloading them.
#
# Two rules, and the order is the whole design:
#   1. accept replies to connections that started outside the container - without it, the container could not
#      answer the published port, because its reply's destination is not 192.168.0.0/16 either;
#   2. drop everything else from the container that is not destined for the LAN.
#
# Scoped to the bridge of ONE network, not to docker0: the rule then belongs to this container's network and
# does not silently restrict whatever else someone later attaches to the default bridge.
#
# Idempotent - safe to run on every boot and by hand.  Re-run it after recreating the container's network.
#
#   sudo deploy/strata-netfilter.sh          # apply
#   sudo deploy/strata-netfilter.sh --check  # show what is there now
set -eu

NETWORK=${STRATA_NETWORK:-strata-net}
ALLOW=${STRATA_ALLOW:-192.168.0.0/16}
CHAIN=DOCKER-USER

if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
    echo "strata-netfilter: no docker network '$NETWORK' - nothing to scope the rule to" >&2
    exit 1
fi
IFACE="br-$(docker network inspect "$NETWORK" -f '{{.Id}}' | cut -c1-12)"

if [ "${1:-}" = "--check" ]; then
    echo "network $NETWORK -> $IFACE"
    iptables -S "$CHAIN"
    exit 0
fi

TAG="-m comment --comment strata-netfilter"

# Recreating the network gives it a new bridge, and the rules for the old one stay behind.  Remove the ones whose
# bridge no longer exists: our tagged rules, and the untagged legacy pair (same shape, no comment).  The legacy
# pair for the CURRENT bridge goes too, so it is re-added tagged and not twice.
iptables -S "$CHAIN" | while IFS= read -r rule; do
    case $rule in
        "-A $CHAIN -i br-"*) ;;
        *) continue ;;
    esac
    br=${rule#"-A $CHAIN -i "}; br=${br%% *}
    tagged=0; legacy=0
    case $rule in *"--comment strata-netfilter"*) tagged=1 ;; esac
    case $rule in
        "-A $CHAIN -i $br -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT" | \
        "-A $CHAIN -i $br ! -d $ALLOW -j DROP") legacy=1 ;;
    esac
    [ "$tagged" = 1 ] || [ "$legacy" = 1 ] || continue
    if [ "$br" = "$IFACE" ]; then
        [ "$legacy" = 1 ] || continue
    elif [ -d "/sys/class/net/$br" ]; then
        continue                      # someone else's live bridge
    fi
    # shellcheck disable=SC2086
    iptables -D ${rule#-A }  2>/dev/null || true
done

# the reply rule must come first, so insert it first and the drop second
# shellcheck disable=SC2086
iptables -C "$CHAIN" -i "$IFACE" -m conntrack --ctstate ESTABLISHED,RELATED $TAG -j ACCEPT 2>/dev/null ||
    iptables -I "$CHAIN" 1 -i "$IFACE" -m conntrack --ctstate ESTABLISHED,RELATED $TAG -j ACCEPT
# shellcheck disable=SC2086
iptables -C "$CHAIN" -i "$IFACE" ! -d "$ALLOW" $TAG -j DROP 2>/dev/null ||
    iptables -I "$CHAIN" 2 -i "$IFACE" ! -d "$ALLOW" $TAG -j DROP

echo "strata-netfilter: $IFACE may reach $ALLOW and nothing else"
iptables -S "$CHAIN"
