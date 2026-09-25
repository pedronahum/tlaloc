#!/usr/bin/env bash
# End-to-end check of the tlaloc backend on a GPU:
#   1. start tritonserver (run_server.sh) with the example model repository,
#   2. wait until it is ready,
#   3. run verify_client.py: KServe v2 JSON, tritonclient HTTP and gRPC,
#      shape buckets, and load-time refusals; every value must match,
#   4. run it again with --perturb (a wrong expected value), which must FAIL,
#   5. stop the server.
# Exit status 0 only if step 3 passes and step 4 fails.
#
#   triton/verify.sh
#
# Environment:
#   TRITON_CLIENT_PYTHON  a Python with tritonclient[http,grpc] (default python3)
#   VERIFY_LOG            where to write the server log (default: a temp file)
#   HTTP_PORT / GRPC_PORT / METRICS_PORT, PJRT_PLUGIN, TLALOC_PJRT_MEMORY_FRACTION
#                         passed through to run_server.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${TRITON_CLIENT_PYTHON:-python3}"
export CONTAINER_NAME="${CONTAINER_NAME:-tlaloc-triton-verify-$$}"
export HTTP_PORT="${HTTP_PORT:-8000}" GRPC_PORT="${GRPC_PORT:-8001}" METRICS_PORT="${METRICS_PORT:-8002}"
LOG="${VERIFY_LOG:-$(mktemp -t tlaloc-triton-verify.XXXXXX.log)}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "starting $CONTAINER_NAME (log: $LOG)"
"$HERE/run_server.sh" --detach --model-control-mode=explicit --load-model='*' >/dev/null
docker logs -f "$CONTAINER_NAME" >"$LOG" 2>&1 &

ready=0
for _ in $(seq 1 180); do
  if curl -sf "localhost:$HTTP_PORT/v2/health/ready" >/dev/null; then
    ready=1
    break
  fi
  if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if [[ $ready != 1 ]]; then
  echo "FAIL: the server did not become ready; last log lines:" >&2
  tail -40 "$LOG" >&2
  exit 1
fi

# The PJRT client must have been created with the allocator options.
if ! grep -q "PJRT client created on platform 'cuda'.*preallocate=false" "$LOG"; then
  echo "FAIL: no PJRT client with preallocate=false in the server log" >&2
  exit 1
fi
grep -o "PJRT client created on platform.*preallocate=false" "$LOG" | head -1

echo "== checks"
"$PY" "$HERE/verify_client.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT"

echo "== negative control (must fail)"
if "$PY" "$HERE/verify_client.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" --perturb >"$LOG.negative" 2>&1; then
  echo "FAIL: the negative control passed; the checks cannot tell a wrong answer from a right one" >&2
  cat "$LOG.negative" >&2
  exit 1
fi
grep -c "FAIL" "$LOG.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"

if ! curl -sf "localhost:$HTTP_PORT/v2/health/live" >/dev/null; then
  echo "FAIL: the server is not live at the end of the run" >&2
  exit 1
fi
echo "VERIFY PASSED"
