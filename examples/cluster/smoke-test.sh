#!/bin/sh
# End-to-end smoke test for the distkv cluster. Run it *inside* the compose
# network (where the worker hostnames resolve), e.g.:
#
#   docker compose -f examples/cluster/docker-compose.yml \
#     exec -T master sh -s < examples/cluster/smoke-test.sh
#
# It exercises the full data-management protocol: two-phase PUT (put_start ->
# write bytes to the chosen Worker -> put_commit) followed by a routed GET, and
# checks the bytes round-trip exactly. Uses only curl + POSIX text tools.
set -eu

MASTER=${MASTER:-http://master:8081}
KEY=${KEY:-demo-object}
PAYLOAD=${PAYLOAD:-hello-distributed-kv-cache}
SIZE=$(printf '%s' "$PAYLOAD" | wc -c | tr -d ' ')

json_field() { grep -o "\"$1\":\"[^\"]*\"" | head -n1 | cut -d'"' -f4; }
json_num()   { grep -o "\"$1\":[0-9]*" | head -n1 | cut -d: -f2; }

echo "1/4 put_start (size=$SIZE) ..."
START=$(curl -fsS -X POST "$MASTER/v1/distkv/put_start" \
  -H 'content-type: application/json' \
  -d "{\"key\":\"$KEY\",\"size_bytes\":$SIZE}")
PUT_ID=$(printf '%s' "$START" | json_field put_id)
WORKER_ADDR=$(printf '%s' "$START" | json_field worker_addr)
GEN=$(printf '%s' "$START" | json_num object_generation)
echo "    -> worker_addr=$WORKER_ADDR generation=$GEN"

echo "2/4 write bytes directly to the worker ..."
curl -fsS -X PUT "$WORKER_ADDR/v1/distkv/worker/objects/$KEY?generation=$GEN" \
  --data-binary "$PAYLOAD" >/dev/null

echo "3/4 put_commit ..."
curl -fsS -X POST "$MASTER/v1/distkv/put_commit" \
  -H 'content-type: application/json' \
  -d "{\"key\":\"$KEY\",\"put_id\":\"$PUT_ID\"}" >/dev/null

echo "4/4 get_route + fetch ..."
ROUTE=$(curl -fsS "$MASTER/v1/distkv/get_route?key=$KEY")
R_ADDR=$(printf '%s' "$ROUTE" | json_field worker_addr)
R_GEN=$(printf '%s' "$ROUTE" | json_num object_generation)
GOT=$(curl -fsS "$R_ADDR/v1/distkv/worker/objects/$KEY?generation=$R_GEN")

if [ "$GOT" = "$PAYLOAD" ]; then
  echo "OK: round-tripped \"$GOT\" via $R_ADDR"
else
  echo "FAIL: expected \"$PAYLOAD\", got \"$GOT\"" >&2
  exit 1
fi
