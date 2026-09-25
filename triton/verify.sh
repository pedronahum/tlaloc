#!/usr/bin/env bash
# End-to-end check of the tlaloc backend on a GPU:
#   1. start tritonserver (run_server.sh) with the example model repository,
#   2. wait until it is ready,
#   3. run verify_client.py: KServe v2 JSON, tritonclient HTTP and gRPC,
#      shape buckets, decode steps with backend-held KV pools, and load-time
#      refusals; every value must match,
#   4. run it again with --perturb (wrong expected values), which must FAIL,
#   5. stop the server,
#   6. optional, TinyLlama: if the TinyLlama-1.1B checkpoint is present,
#      export its serving artifact (decode batches 1, 2 and 4 and a prefill
#      entry) and write it as a sequence-mode Triton model with a 5 s idle
#      timeout (Gradle, with no server running), start a server on that
#      repository, greedy-decode "The capital of France is" with
#      generate_client.py (the six ids must equal HuggingFace's), run
#      sequence_checks.py (prefill, concurrent sequences, END and idle
#      freeing pages, pool exhaustion), run it again with --perturb (must
#      fail), and stop the server. Without the checkpoint this step is
#      skipped by name.
# Exit status 0 only if step 3 passes, step 4 fails, and step 6 passes or is
# skipped.
#
#   triton/verify.sh
#
# Environment:
#   TRITON_CLIENT_PYTHON  a Python with tritonclient[http,grpc] (default python3)
#   VERIFY_LOG            where to write the server log (default: a temp file)
#   HTTP_PORT / GRPC_PORT / METRICS_PORT, PJRT_PLUGIN, TLALOC_PJRT_MEMORY_FRACTION
#                         passed through to run_server.sh
#   TINYLLAMA_CHECKPOINT  [~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0]
#   TINYLLAMA_DIR         where the artifact and its model repository are written
#                         [triton/build/tinyllama]; an existing model is reused
#   TINYLLAMA_REEXPORT=1  export again even if the model exists
#   SKIP_TINYLLAMA=1      skip step 6
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${TRITON_CLIENT_PYTHON:-python3}"
BASE_NAME="${CONTAINER_NAME:-tlaloc-triton-verify-$$}"
export CONTAINER_NAME="$BASE_NAME"
export HTTP_PORT="${HTTP_PORT:-8000}" GRPC_PORT="${GRPC_PORT:-8001}" METRICS_PORT="${METRICS_PORT:-8002}"
LOG="${VERIFY_LOG:-$(mktemp -t tlaloc-triton-verify.XXXXXX.log)}"

cleanup() {
  docker rm -f "$BASE_NAME" "$BASE_NAME-tinyllama" >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

# start_server <log>: run_server.sh with $CONTAINER_NAME and $MODEL_REPOSITORY,
# then wait up to $2 seconds for readiness.
start_server() {
  local log="$1" timeout="$2"
  echo "starting $CONTAINER_NAME (log: $log)"
  "$HERE/run_server.sh" --detach --model-control-mode=explicit --load-model='*' >/dev/null
  docker logs -f "$CONTAINER_NAME" >"$log" 2>&1 &
  for _ in $(seq 1 "$timeout"); do
    if curl -sf "localhost:$HTTP_PORT/v2/health/ready" >/dev/null; then
      break
    fi
    if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ! curl -sf "localhost:$HTTP_PORT/v2/health/ready" >/dev/null; then
    echo "FAIL: the server did not become ready; last log lines:" >&2
    tail -40 "$log" >&2
    exit 1
  fi
  # The PJRT client must have been created with the allocator options.
  if ! grep -q "PJRT client created on platform 'cuda'.*preallocate=false" "$log"; then
    echo "FAIL: no PJRT client with preallocate=false in the server log" >&2
    exit 1
  fi
  grep -o "PJRT client created on platform.*preallocate=false" "$log" | head -1
}

stop_server() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  # Wait for the log follower of the stopped container.
  wait 2>/dev/null || true
}

start_server "$LOG" 180

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
stop_server

# --- TinyLlama ---------------------------------------------------------------
CKPT="${TINYLLAMA_CHECKPOINT:-$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0}"
TL_DIR="${TINYLLAMA_DIR:-$HERE/build/tinyllama}"
# HuggingFace transformers' greedy continuation of "The capital of France is"
# for this checkpoint, " Paris.\n\n2." (docs/SERVING_RUNBOOK.md, section 10;
# HfLlamaServingArtifactTest compares Tlaloc against transformers directly).
EXPECT="3681,29889,13,13,29906,29889"

echo "== tinyllama"
if [[ "${SKIP_TINYLLAMA:-}" == 1 ]]; then
  echo "SKIP tinyllama: SKIP_TINYLLAMA=1"
elif [[ ! -f "$CKPT/model.safetensors" || ! -f "$CKPT/tokenizer.json" ]]; then
  echo "SKIP tinyllama: no TinyLlama-1.1B checkpoint at $CKPT"
else
  TL_CONFIG="$TL_DIR/repository/tinyllama/config.pbtxt"
  if [[ "${TINYLLAMA_REEXPORT:-}" == 1 ]] || ! grep -q '"serving_manifest"' "$TL_CONFIG" 2>/dev/null \
      || ! grep -q "max_sequence_idle_microseconds: 5000000$" "$TL_CONFIG"; then
    # Gradle runs here with no Triton container up.
    rm -rf "$TL_DIR"
    mkdir -p "$TL_DIR"
    (cd "$ROOT" && ./gradlew -q :maestro:exportLlamaServingArtifact \
      -PckptDir="$CKPT" -PoutDir="$TL_DIR/artifact" -PmaxBatch=4)
    (cd "$ROOT" && ./gradlew -q :maestro:exportTritonModel \
      -PartifactDir="$TL_DIR/artifact" -PoutDir="$TL_DIR/repository" -PmodelName=tinyllama \
      -PkvMode=sequence -PmaxSequenceIdleMicros=5000000)
  fi
  export CONTAINER_NAME="$BASE_NAME-tinyllama" MODEL_REPOSITORY="$TL_DIR/repository"
  start_server "$LOG.tinyllama" 600
  grep -o "uploaded [0-9]* weights.*" "$LOG.tinyllama" | head -1
  "$PY" "$HERE/generate_client.py" --url "localhost:$HTTP_PORT" --model tinyllama \
    --tokenizer "$CKPT/tokenizer.json" --text "The capital of France is" --max-new 6 \
    --expect "$EXPECT"
  echo "== tinyllama sequence checks"
  "$PY" "$HERE/sequence_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model tinyllama
  echo "== tinyllama negative control (must fail)"
  if "$PY" "$HERE/sequence_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model tinyllama --perturb >"$LOG.tinyllama.negative" 2>&1; then
    echo "FAIL: the sequence checks passed with wrong expected ids" >&2
    cat "$LOG.tinyllama.negative" >&2
    exit 1
  fi
  grep -c "^FAIL" "$LOG.tinyllama.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  if ! curl -sf "localhost:$HTTP_PORT/v2/health/live" >/dev/null; then
    echo "FAIL: the server is not live at the end of the sequence checks" >&2
    exit 1
  fi
  stop_server
fi
echo "VERIFY PASSED"
