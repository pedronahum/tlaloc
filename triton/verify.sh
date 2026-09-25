#!/usr/bin/env bash
# End-to-end check of the tlaloc backend on a GPU:
#   0. run pjrt_device_test (built by build_backend.sh) in the Triton
#      container: a PJRT client for GPU 0 sees exactly GPU 0, a client for a
#      GPU the machine does not have is refused by name, and an input read in
#      place from cudaMalloc memory gives the same bits as the host path; then
#      with --perturb, which must fail,
#   1. start tritonserver (run_server.sh) with the example model repository,
#   2. wait until it is ready,
#   3. run verify_client.py: KServe v2 JSON, tritonclient HTTP and gRPC,
#      every dtype, shape buckets, dynamic batching, CUDA shared memory,
#      decode steps with backend-held KV pools, and load-time refusals; every
#      value must match. The server log must then show that tensors in CUDA
#      shared memory were read in place and written device to device by
#      large_io, and went through the host for large_io_host,
#   4. run it again with --perturb (wrong expected values), which must FAIL,
#   5. print the measurements of perf_client.py (dynamic batching throughput,
#      the host round trip that zero copy saves) and stop the server,
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
#   7. optional, Qwen3: if Qwen/Qwen3-0.6B is in the HuggingFace cache, export
#      it the same way (Gradle, no server running), start a server, run
#      fixture_checks.py against the committed HuggingFace fixture (a plain
#      and a chat-template prompt, 16 greedy ids each over HTTP and gRPC,
#      logits within TF32 tolerance, and the prefill and decode timings), run
#      it again with --perturb (must fail), and stop the server. Without the
#      checkpoint this step is skipped by name.
# Exit status 0 only if step 0 passes (and its negative control fails), step 3
# passes, step 4 fails, and steps 6 and 7 pass or are skipped.
#
#   triton/verify.sh
#
# Environment:
#   TRITON_CLIENT_PYTHON  a Python with tritonclient[all] and a cuda-python that
#                         matches the driver's CUDA (default python3)
#   SKIP_PERF=1           skip the measurements of step 5
#   VERIFY_LOG            where to write the server log (default: a temp file)
#   HTTP_PORT / GRPC_PORT / METRICS_PORT, PJRT_PLUGIN, TLALOC_PJRT_MEMORY_FRACTION
#                         passed through to run_server.sh
#   TINYLLAMA_CHECKPOINT  [~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0]
#   TINYLLAMA_DIR         where the artifact and its model repository are written
#                         [triton/build/tinyllama]; an existing model is reused
#   TINYLLAMA_REEXPORT=1  export again even if the model exists
#   SKIP_TINYLLAMA=1      skip step 6
#   QWEN3_CHECKPOINT      [the Qwen/Qwen3-0.6B snapshot the fixture names, in
#                         ~/.cache/huggingface/hub]
#   QWEN3_DIR             [triton/build/qwen3]; an existing model is reused
#   QWEN3_REEXPORT=1      export again even if the model exists
#   SKIP_QWEN3=1          skip step 7
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${TRITON_CLIENT_PYTHON:-python3}"
BASE_NAME="${CONTAINER_NAME:-tlaloc-triton-verify-$$}"
export CONTAINER_NAME="$BASE_NAME"
export HTTP_PORT="${HTTP_PORT:-8000}" GRPC_PORT="${GRPC_PORT:-8001}" METRICS_PORT="${METRICS_PORT:-8002}"
LOG="${VERIFY_LOG:-$(mktemp -t tlaloc-triton-verify.XXXXXX.log)}"

cleanup() {
  docker rm -f "$BASE_NAME" "$BASE_NAME-tinyllama" "$BASE_NAME-qwen3" "$BASE_NAME-device" >/dev/null 2>&1 || true
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

echo "== PJRT device checks"
DEVICE_TEST="$HERE/build/tests/pjrt_device_test"
PLUGIN="$(realpath "${PJRT_PLUGIN:-$HERE/pjrt/xla_cuda13/xla_cuda_plugin.so}")"
if [[ ! -x "$DEVICE_TEST" ]]; then
  echo "FAIL: missing $DEVICE_TEST; run triton/build_backend.sh" >&2
  exit 1
fi
run_device_test() {
  docker run --rm --name "$BASE_NAME-device" --gpus all --user "$(id -u):$(id -g)" \
    -v "$DEVICE_TEST:/opt/pjrt_device_test:ro" -v "$PLUGIN:/opt/pjrt/xla_cuda_plugin.so:ro" \
    "${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.11-py3}" \
    /opt/pjrt_device_test /opt/pjrt/xla_cuda_plugin.so "$@" 2>&1 | grep -E "^ |pjrt_device_test|check|FAIL"
  return "${PIPESTATUS[0]}"
}
run_device_test
if run_device_test --perturb >"$LOG.device-negative" 2>&1; then
  echo "FAIL: the device test passed with a wrong expected value" >&2
  exit 1
fi
grep -c "FAIL" "$LOG.device-negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"

start_server "$LOG" 180

echo "== checks"
"$PY" "$HERE/verify_client.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT"

echo "== data paths in the server log"
expect_log() {
  if ! grep -qF "$1" "$LOG"; then
    echo "FAIL: the server log has no line with: $1" >&2
    exit 1
  fi
  echo "  ok   log: $1"
}
refuse_log() {
  if grep -qF "$1" "$LOG"; then
    echo "FAIL: the server log has a line with: $1" >&2
    exit 1
  fi
  echo "  ok   log has no: $1"
}
expect_log "model 'large_io': input 'X' is read in place from GPU memory (no host copy)"
expect_log "model 'large_io': output 'Y' is copied device to device into GPU memory"
expect_log "model 'matmul_sumsq': input 'A' is read in place from GPU memory (no host copy)"
expect_log "model 'large_io_host': input 'X' is in GPU memory and goes through the host"
expect_log "model 'large_io_host': output 'Y' was given GPU memory and goes through the host"
refuse_log "model 'large_io': input 'X' is in GPU memory and goes through the host"
refuse_log "model 'large_io_host': input 'X' is read in place"

echo "== negative control (must fail)"
if "$PY" "$HERE/verify_client.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" --perturb >"$LOG.negative" 2>&1; then
  echo "FAIL: the negative control passed; the checks cannot tell a wrong answer from a right one" >&2
  cat "$LOG.negative" >&2
  exit 1
fi
grep -c "FAIL" "$LOG.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"

if [[ "${SKIP_PERF:-}" != 1 ]]; then
  echo "== measurements"
  "$PY" "$HERE/perf_client.py" --grpc "localhost:$GRPC_PORT"
fi

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

# --- Qwen3 -------------------------------------------------------------------
QWEN3_FIXTURE="$ROOT/ir/src/jvmTest/resources/io/tlaloc/ir/inference/qwen3_0_6b_greedy.json"
QWEN3_REV="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$QWEN3_FIXTURE")"
QCKPT="${QWEN3_CHECKPOINT:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/$QWEN3_REV}"
Q_DIR="${QWEN3_DIR:-$HERE/build/qwen3}"

echo "== qwen3"
if [[ "${SKIP_QWEN3:-}" == 1 ]]; then
  echo "SKIP qwen3: SKIP_QWEN3=1"
elif [[ ! -f "$QCKPT/model.safetensors" ]]; then
  echo "SKIP qwen3: no Qwen/Qwen3-0.6B checkpoint at $QCKPT (hf download Qwen/Qwen3-0.6B)"
else
  Q_CONFIG="$Q_DIR/repository/qwen3/config.pbtxt"
  if [[ "${QWEN3_REEXPORT:-}" == 1 ]] || ! grep -q '"serving_manifest"' "$Q_CONFIG" 2>/dev/null; then
    # Gradle runs here with no Triton container up.
    rm -rf "$Q_DIR"
    mkdir -p "$Q_DIR"
    (cd "$ROOT" && ./gradlew -q :maestro:exportHfServingArtifact \
      -PckptDir="$QCKPT" -PoutDir="$Q_DIR/artifact" -PmaxBatch=4)
    (cd "$ROOT" && ./gradlew -q :maestro:exportTritonModel \
      -PartifactDir="$Q_DIR/artifact" -PoutDir="$Q_DIR/repository" -PmodelName=qwen3 \
      -PkvMode=sequence -PmaxSequenceIdleMicros=5000000)
  fi
  export CONTAINER_NAME="$BASE_NAME-qwen3" MODEL_REPOSITORY="$Q_DIR/repository"
  start_server "$LOG.qwen3" 600
  grep -o "uploaded [0-9]* weights.*" "$LOG.qwen3" | head -1
  "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model qwen3 --fixture "$QWEN3_FIXTURE"
  echo "== qwen3 negative control (must fail)"
  if "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model qwen3 --fixture "$QWEN3_FIXTURE" --perturb >"$LOG.qwen3.negative" 2>&1; then
    echo "FAIL: the Qwen3 fixture checks passed with wrong expected ids" >&2
    cat "$LOG.qwen3.negative" >&2
    exit 1
  fi
  grep -c "^FAIL" "$LOG.qwen3.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  if ! curl -sf "localhost:$HTTP_PORT/v2/health/live" >/dev/null; then
    echo "FAIL: the server is not live at the end of the Qwen3 checks" >&2
    exit 1
  fi
  stop_server
fi
echo "VERIFY PASSED"
