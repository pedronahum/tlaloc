#!/usr/bin/env bash
# End-to-end check of the tlaloc backend on a GPU:
#   0. run pjrt_device_test (built by build_backend.sh) in the Triton
#      container: a PJRT client for GPU 0 sees exactly GPU 0, a client for a
#      GPU the machine does not have is refused by name, an input read in
#      place from cudaMalloc memory gives the same bits as the host path, and
#      a state whose parameter is aliased to the output is updated in place
#      for 100 steps (the same check sees the copy without the alias or
#      without donation); then with --perturb, which must fail,
#   1. start tritonserver (run_server.sh) with the example model repository,
#   2. wait until it is ready,
#   3. run verify_client.py: KServe v2 JSON, tritonclient HTTP and gRPC,
#      every dtype, shape buckets, dynamic batching, requests of two widths
#      in one ragged batch (grouped by shape: fewer executions than the
#      group_by_shape false control, the same rows), CUDA shared memory,
#      decode steps with backend-held KV pools, and load-time refusals; every
#      value must match. The server log must then show that tensors in CUDA
#      shared memory were read in place and written device to device by
#      large_io, and went through the host for large_io_host, and that
#      reference_sequence updated its KV pools in place; then window_checks.py:
#      a decoder whose sliding-window layers keep their KV in a windowed pool
#      (a ring of 3 pages per sequence) holds at most 3 windowed pages as a
#      sequence grows to 60 positions and gives the logits of the same model
#      with full-history pages, over HTTP and gRPC, alone and batched, a
#      START with every ring taken is refused by name, and the token ids the
#      manifest lists as image and video placeholders are refused by name
#      without changing the sequence; then prefill_checks.py
#      on both window models: two prompts sent together are prefilled in one
#      call (the log must show a batch-2 prefill entry running two sequences)
#      and each gets its solo logits, and --perturb must fail,
#   4. run both again with --perturb (wrong expected values), which must FAIL,
#   5. print the measurements of perf_client.py (dynamic batching throughput,
#      the host round trip that zero copy saves) and stop the server,
#   6. optional, TinyLlama: if the TinyLlama-1.1B checkpoint is present,
#      export its serving artifact (decode batches 1, 2 and 4 and a prefill
#      entry) and write it as a sequence-mode Triton model with a 5 s idle
#      timeout (Gradle, with no server running), start a server on that
#      repository, greedy-decode "The capital of France is" with
#      generate_client.py (the six ids must equal HuggingFace's), generate
#      50 ids twice more (the server log must show every run writing the KV
#      pools at their own device addresses, 100 of 100 runs in place), run
#      sequence_checks.py (prefill, concurrent sequences, END and idle
#      freeing pages, pool exhaustion, a live sequence refused a page
#      mid-generation and resumed with its solo ids, idle pages reclaimed
#      least recently active first and only as many as needed, read from the
#      server log), run it again with --perturb (must fail), run prefill_checks.py (1, 2 and 4 prompts prefilled together
#      give their solo argmax and decoded ids, in one call: the log must show
#      prefill_b4_c64 running four sequences; prefill throughput), run it with
#      --perturb (must fail), run it on prompts of 2, 55, 9 and 30 tokens in
#      one call (and with --perturb), and stop the server. Control: serve the same model with
#      donate_kv_pools false; the log must show the pools copied in 100 of
#      100 runs, and the 100 ids must equal those generated in place. Then
#      serve the same model with a 200 ms
#      idle timeout and run sequence_checks.py --queued: with steps waiting in
#      Triton's queue past the timeout, no step Triton accepts may find its
#      sequence's pages freed, and a sequence sent only refused requests for
#      over ten timeouts is not reclaimed. Without the checkpoint this step is skipped by
#      name.
#   7. optional, Qwen3: if Qwen/Qwen3-0.6B is in the HuggingFace cache, export
#      it the same way (Gradle, no server running), start a server, run
#      fixture_checks.py against the committed HuggingFace fixture (a plain
#      and a chat-template prompt, 16 greedy ids each over HTTP and gRPC,
#      logits within TF32 tolerance, and the prefill and decode timings),
#      check that the KV pools were updated in place, run
#      it again with --perturb (must fail), run prefill_checks.py on the
#      fixture's prompts as for TinyLlama (and with --perturb, which must
#      fail), and stop the server. The model must upload 310 weights: its
#      tied head reads the embedding table, with no copy of it. When the
#      client's Python has the tokenizers package, generate_client.py sends
#      the fixture's text prompt as text (the ids must be the fixture's
#      prompt and continuation; a wrong expectation must fail). Then the same
#      checkpoint exported with bf16 weights: half the MiB on the device, all
#      32 ids equal, logits within 6e-3 of the largest (and not within the f32
#      tolerance of 2e-3), and --perturb must
#      fail. Without the checkpoint this step is skipped by name.
#   8. optional and opt-in (MUSE_GLIMMER=1), Muse Glimmer: 28 billion text
#      parameters, 56 GB of bf16 weights on the device. It needs the
#      meta-models/Muse-Glimmer-30B snapshot the fixtures name, and refuses by
#      name to start any step when MemAvailable is below that step's need plus
#      a 16 GiB margin (the GPU shares system memory on the GB10). Export
#      (Gradle, no server running) for contexts 512 to 32,768, decode batches
#      1, 2 and 4 and prefill in chunks of 512, drop the page cache of the
#      checkpoint and the artifact, start a server whose PJRT memory fraction
#      covers the weights plus 15 GiB (computed from the manifest and
#      MemTotal, and printed), run fixture_checks.py against the three
#      committed fixtures:
#      transformers in bfloat16 (the ids must be equal as far as the two
#      references agree with each other) and transformers with bf16 weights
#      and f32 activations (all ids, and the logits within twice the oracle's
#      own float32-vs-float64 noise), and a 2,305-token prompt whose fact is
#      outside the sliding window of the question (16 ids equal, logits
#      within 2e-2 of the largest), check that the KV pools were updated
#      in place and that its 39 sliding-window layers have a windowed KV pool,
#      that its image and video placeholder ids (200092, 200091) are refused by
#      name, then --perturb (must fail), and print the peak memory in use.
# Exit status 0 only if step 0 passes (and its negative control fails), step 3
# passes, step 4 fails, and steps 6, 7 and 8 pass or are skipped.
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
#   TINYLLAMA_REEXPORT=1  export again even if the model exists (a model whose
#                         bodies do not alias the KV pools is always exported again)
#   SKIP_TINYLLAMA=1      skip step 6
#   QWEN3_CHECKPOINT      [the Qwen/Qwen3-0.6B snapshot the fixture names, in
#                         ~/.cache/huggingface/hub]
#   QWEN3_DIR             [triton/build/qwen3]; an existing model is reused
#   QWEN3_BF16_DIR        [triton/build/qwen3-bf16]: the same checkpoint with bf16 weights
#   QWEN3_REEXPORT=1      export again even if the model exists (both)
#   SKIP_QWEN3=1          skip step 7
#   MUSE_GLIMMER=1        run step 8
#   MUSE_GLIMMER_DIR      [triton/build/muse-glimmer]; an existing model is reused
#   MUSE_GLIMMER_REEXPORT=1  export again even if the model exists
#   MUSE_CONTEXT_BENCH    e.g. 512,2048,8192: also run context_bench.py at those
#                         contexts (32768 adds about 20 minutes)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${TRITON_CLIENT_PYTHON:-python3}"
BASE_NAME="${CONTAINER_NAME:-tlaloc-triton-verify-$$}"
export CONTAINER_NAME="$BASE_NAME"
export HTTP_PORT="${HTTP_PORT:-8000}" GRPC_PORT="${GRPC_PORT:-8001}" METRICS_PORT="${METRICS_PORT:-8002}"
LOG="${VERIFY_LOG:-$(mktemp -t tlaloc-triton-verify.XXXXXX.log)}"

PEAK_PID=""
cleanup() {
  [[ -n "$PEAK_PID" ]] && kill "$PEAK_PID" 2>/dev/null
  docker rm -f "$BASE_NAME" "$BASE_NAME-tinyllama" "$BASE_NAME-copy" "$BASE_NAME-queued" \
    "$BASE_NAME-qwen3" "$BASE_NAME-qwen3-bf16" "$BASE_NAME-muse" "$BASE_NAME-device" >/dev/null 2>&1 || true
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
# expect_in <log> <text> / refuse_in <log> <text>: the log must (not) have a
# line with the text. expect_log and refuse_log read the step 3 log.
expect_in() {
  if ! grep -qF "$2" "$1"; then
    echo "FAIL: the server log has no line with: $2" >&2
    exit 1
  fi
  echo "  ok   log: $2"
}
refuse_in() {
  if grep -qF "$2" "$1"; then
    echo "FAIL: the server log has a line with: $2" >&2
    exit 1
  fi
  echo "  ok   log has no: $2"
}
expect_log() { expect_in "$LOG" "$1"; }
# expect_re <log> <extended regex>: the log must have a matching line.
expect_re() {
  if ! grep -qE "$2" "$1"; then
    echo "FAIL: the server log has no line matching: $2" >&2
    exit 1
  fi
  grep -oE "$2.*" "$1" | head -1 | sed 's/^/  ok   log: /'
}
refuse_log() { refuse_in "$LOG" "$1"; }
# aliased <artifact dir>: its bodies alias the KV pools to their outputs (an
# artifact exported before they did is exported again).
aliased() { grep -qs "tf.aliasing_output" "$1"/bodies/*.mlir; }
# batched_prefill <artifact dir>: it has prefill entries of batch 2 (an
# artifact exported before it did is exported again).
batched_prefill() { grep -qs "programs/prefill_b2_" "$1/tlaloc-serving.json"; }
# pools_in_place <log>: every run the log reports on wrote the KV pools at
# their own device addresses, including a count over 100 runs.
pools_in_place() {
  if ! grep -qE "decode_b1_c[0-9]+ updated the [0-9]+ KV pools in place" "$1"; then
    echo "FAIL: the server log does not show the KV pools updated in place" >&2
    exit 1
  fi
  grep -oE "decode_b1_c[0-9]+ updated the [0-9]+ KV pools in place.*" "$1" | head -1 | sed 's/^/  ok   log: /'
  expect_in "$1" "KV pools were updated in place 100 times and copied 0 times"
  refuse_in "$1" "to new device memory"
}
expect_log "model 'large_io': input 'X' is read in place from GPU memory (no host copy)"
expect_log "model 'large_io': output 'Y' is copied device to device into GPU memory"
expect_log "model 'matmul_sumsq': input 'A' is read in place from GPU memory (no host copy)"
expect_log "model 'large_io_host': input 'X' is in GPU memory and goes through the host"
expect_log "model 'large_io_host': output 'Y' was given GPU memory and goes through the host"
refuse_log "model 'large_io': input 'X' is in GPU memory and goes through the host"
refuse_log "model 'large_io_host': input 'X' is read in place"
expect_log "model 'ragged_batched': GPU 0 ready; GPU-memory tensors are read in place and written device to device; requests are batched along dim 0 up to batch 8, grouped by the shape of their rows"
expect_log "model 'ragged_consecutive': GPU 0 ready; GPU-memory tensors are read in place and written device to device; requests are batched along dim 0 up to batch 8, only consecutive requests of one shape together (group_by_shape is false)"
expect_log "decode_b1_c2 updated the 2 KV pools in place"
refuse_log "to new device memory"

echo "== windowed KV pool"
"$PY" "$HERE/window_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT"
expect_log "model 'window_sequence': uploaded 30 weights"
expect_log "windowed KV pool for 2 sliding layers (window 8): 13 pages, a ring of at most 3 pages per sequence"
expect_log "refuses token ids 62 (image_token_id) 63 (video_token_id)"
expect_log "instance 'window_sequence_0_0': 6 KV pools (0 MiB) zeroed; 64 pages for sequences (page 0 is the padding page); 12 windowed pages, at most 3 per sequence"
expect_log "instance 'window_sequence_0_0': in 100 runs the 6 KV pools were updated in place 100 times and copied 0 times"
expect_re "$LOG" "model 'window_sequence_chunked': uploaded 30 weights.*largest prefill batch 2, at most 6 tokens per sequence in a prefill call.*a ring of at most 4 pages per sequence"
# Control: window_sequence's entries take their whole context.
if grep "model 'window_sequence': uploaded" "$LOG" | grep -q "tokens per sequence in a prefill call"; then
  echo "FAIL: window_sequence's load line states a prefill chunk" >&2
  exit 1
fi

echo "== negative control (must fail)"
if "$PY" "$HERE/verify_client.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" --perturb >"$LOG.negative" 2>&1; then
  echo "FAIL: the negative control passed; the checks cannot tell a wrong answer from a right one" >&2
  cat "$LOG.negative" >&2
  exit 1
fi
grep -c "FAIL" "$LOG.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
if "$PY" "$HERE/window_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" --perturb >"$LOG.window-negative" 2>&1; then
  echo "FAIL: the window checks passed comparing each request with the next one" >&2
  cat "$LOG.window-negative" >&2
  exit 1
fi
grep -c "FAIL" "$LOG.window-negative" | xargs -I{} echo "window negative control failed as it must ({} failing checks)"

echo "== batched prefill"
for m in window_sequence window_sequence_full window_sequence_chunked; do
  "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model "$m" --repeats 3
  expect_re "$LOG" "instance '${m}_0_0': prefill_b2_c[0-9]+ ran 2 sequence"
done
if "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model window_sequence_full --repeats 1 --perturb >"$LOG.prefill-negative" 2>&1; then
  echo "FAIL: the batched prefill checks passed comparing each prompt with the next one" >&2
  cat "$LOG.prefill-negative" >&2
  exit 1
fi
grep -c "FAIL" "$LOG.prefill-negative" | xargs -I{} echo "batched prefill negative control failed as it must ({} failing checks)"

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
      || ! aliased "$TL_DIR/artifact" || ! batched_prefill "$TL_DIR/artifact" \
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
  # Two longer generations: 100 runs on the pools, whose ids the control
  # below must reproduce with the pools copied instead.
  long_ids() {
    for id in 11 12; do
      "$PY" "$HERE/generate_client.py" --url "localhost:$HTTP_PORT" --model tinyllama \
        --tokenizer "$CKPT/tokenizer.json" --text "The capital of France is" --max-new 50 \
        --sequence-id "$id" --expect-prefix "$EXPECT" | tee -a "$1" | grep -E "median|FAIL|ok"
    done
  }
  : >"$LOG.tinyllama.ids"
  long_ids "$LOG.tinyllama.ids"
  echo "== tinyllama KV pools"
  grep -o "compiled decode_b1_c64 .*" "$LOG.tinyllama" | sed 's/^/  /'
  pools_in_place "$LOG.tinyllama"
  echo "== tinyllama sequence checks"
  "$PY" "$HERE/sequence_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model tinyllama --log "$LOG.tinyllama"
  echo "== tinyllama negative control (must fail)"
  if "$PY" "$HERE/sequence_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model tinyllama --perturb >"$LOG.tinyllama.negative" 2>&1; then
    echo "FAIL: the sequence checks passed with wrong expected ids" >&2
    cat "$LOG.tinyllama.negative" >&2
    exit 1
  fi
  grep -c "^FAIL" "$LOG.tinyllama.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  echo "== tinyllama batched prefill"
  # sequence_checks.py's four prompts: "The capital of France is", "The largest
  # planet in the solar system is", "My favorite color is", "The first
  # president of the United States was".
  TL_PROMPTS="1,450,7483,310,3444,338;1,450,10150,15754,297,278,21635,1788,338;1,1619,25448,2927,338;1,450,937,6673,310,278,3303,3900,471"
  "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model tinyllama --prompts "$TL_PROMPTS" --require-ids
  expect_re "$LOG.tinyllama" "instance 'tinyllama_0_0': prefill_b4_c64 ran 4 sequence"
  if "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model tinyllama --prompts "$TL_PROMPTS" --require-ids --repeats 1 --perturb >"$LOG.tinyllama.prefill-negative" 2>&1; then
    echo "FAIL: the batched prefill checks passed comparing each prompt with the next one" >&2
    cat "$LOG.tinyllama.prefill-negative" >&2
    exit 1
  fi
  grep -c "FAIL" "$LOG.tinyllama.prefill-negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  echo "== tinyllama batched prefill, prompts of 2, 55, 9 and 30 tokens"
  # Rows of very different lengths in one call: the 2-token prompt is right-
  # aligned behind 62 padding positions of the 64-token entry.
  TL_W="450,7483,310,3444,338,450,10150,15754,297,278,21635,1788,338,1619,25448,2927,338,450,937,6673,310,278,3303,3900,471"
  TL_RAGGED="1,450;1,$TL_W,$TL_W,450,7483,310,3444;1,450,7483,310,3444,338,450,10150,15754;1,$TL_W,450,7483,310,3444"
  "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model tinyllama --prompts "$TL_RAGGED" --require-ids --repeats 1 --id-base 95000
  if "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model tinyllama --prompts "$TL_RAGGED" --require-ids --repeats 1 --id-base 96000 --perturb \
      >"$LOG.tinyllama.ragged-negative" 2>&1; then
    echo "FAIL: the prefill checks of very different lengths passed comparing each prompt with the next one" >&2
    cat "$LOG.tinyllama.ragged-negative" >&2
    exit 1
  fi
  grep -c "FAIL" "$LOG.tinyllama.ragged-negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  if ! curl -sf "localhost:$HTTP_PORT/v2/health/live" >/dev/null; then
    echo "FAIL: the server is not live at the end of the sequence checks" >&2
    exit 1
  fi
  stop_server

  # Control: the same model with donate_kv_pools false. The executions are
  # not handed the pools, so XLA copies them; the check above must see that,
  # and the ids must not change.
  echo "== tinyllama control: KV pools not donated (must be seen copied)"
  COPY_REPO="$TL_DIR/copy-repository"
  rm -rf "$COPY_REPO"
  mkdir -p "$COPY_REPO"
  cp -al "$TL_DIR/repository/tinyllama" "$COPY_REPO/tinyllama"
  rm "$COPY_REPO/tinyllama/config.pbtxt"
  { cat "$TL_CONFIG"; echo 'parameters: { key: "donate_kv_pools" value: { string_value: "false" } }'; } \
    >"$COPY_REPO/tinyllama/config.pbtxt"
  export CONTAINER_NAME="$BASE_NAME-copy" MODEL_REPOSITORY="$COPY_REPO"
  start_server "$LOG.copy" 600
  : >"$LOG.copy.ids"
  long_ids "$LOG.copy.ids"
  stop_server
  rm -rf "$COPY_REPO"
  expect_in "$LOG.copy" "executions are not handed the pools (donate_kv_pools is false)"
  expect_in "$LOG.copy" "KV pools were updated in place 0 times and copied 100 times"
  refuse_in "$LOG.copy" "KV pools in place"
  if ! diff <(grep "^generated" "$LOG.tinyllama.ids") <(grep "^generated" "$LOG.copy.ids") >/dev/null; then
    echo "FAIL: the ids with the pools copied differ from the ids with the pools updated in place" >&2
    exit 1
  fi
  echo "  ok   the 100 ids with the pools copied equal those with the pools updated in place"
  median() { grep -o "median decode step [0-9.]* ms" "$1" | awk '{print $4}' | sort -n | head -1; }
  echo "  median decode step: $(median "$LOG.tinyllama.ids") ms in place, $(median "$LOG.copy.ids") ms copied (the faster of two generations)"

  # The same model (hard links, no copy of the weights) with a 200 ms idle
  # timeout: many concurrent sequences make steps wait in Triton's queue for
  # longer than that, and the backend must not free a sequence Triton holds.
  echo "== tinyllama queued sequences (200 ms idle timeout)"
  QUEUED_REPO="$TL_DIR/queued-repository"
  rm -rf "$QUEUED_REPO"
  mkdir -p "$QUEUED_REPO"
  cp -al "$TL_DIR/repository/tinyllama" "$QUEUED_REPO/tinyllama"
  rm "$QUEUED_REPO/tinyllama/config.pbtxt"
  sed 's/^  max_sequence_idle_microseconds: .*/  max_sequence_idle_microseconds: 200000/' \
    "$TL_CONFIG" >"$QUEUED_REPO/tinyllama/config.pbtxt"
  export CONTAINER_NAME="$BASE_NAME-queued" MODEL_REPOSITORY="$QUEUED_REPO"
  start_server "$LOG.queued" 600
  "$PY" "$HERE/sequence_checks.py" --http "localhost:$HTTP_PORT" --model tinyllama --queued --log "$LOG.queued"
  stop_server
  rm -rf "$QUEUED_REPO"
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
  if [[ "${QWEN3_REEXPORT:-}" == 1 ]] || ! grep -q '"serving_manifest"' "$Q_CONFIG" 2>/dev/null \
      || ! aliased "$Q_DIR/artifact" || ! batched_prefill "$Q_DIR/artifact" \
      || grep -q '"name":"lmHead"' "$Q_DIR/artifact/tlaloc-serving.json"; then
    # Gradle runs here with no Triton container up. (An artifact that stages
    # the tied head as a copy of the embedding table is exported again.)
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
  # The tied head reads the embedding table: 310 weights, no second copy.
  expect_in "$LOG.qwen3" "model 'qwen3': uploaded 310 weights"
  "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model qwen3 --fixture "$QWEN3_FIXTURE"
  echo "== qwen3 text prompt (byte-level BPE)"
  if "$PY" -c "import tokenizers" 2>/dev/null; then
    # The fixture's text prompt, tokenized by the checkpoint's tokenizer.json
    # through the tokenizers package: the prompt ids and the 16 greedy ids
    # must be the fixture's (transformers' tokenizer and model).
    Q_IDS() { "$PY" -c "import json,sys; p=json.load(open(sys.argv[1]))['prompts'][0]; print(','.join(map(str, p[sys.argv[2]])))" "$QWEN3_FIXTURE" "$1"; }
    Q_TEXT="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['prompts'][0]['input'])" "$QWEN3_FIXTURE")"
    "$PY" "$HERE/generate_client.py" --url "localhost:$HTTP_PORT" --model qwen3 \
      --tokenizer "$QCKPT/tokenizer.json" --text "$Q_TEXT" --max-new 16 --sequence-id 31 \
      --expect-prompt "$(Q_IDS promptTokens)" --expect "$(Q_IDS generatedTokens)"
    if "$PY" "$HERE/generate_client.py" --url "localhost:$HTTP_PORT" --model qwen3 \
        --tokenizer "$QCKPT/tokenizer.json" --text "$Q_TEXT" --max-new 16 --sequence-id 32 \
        --expect-prompt "$(Q_IDS promptTokens)" --expect "0,$(Q_IDS generatedTokens)" >"$LOG.qwen3.text-negative" 2>&1; then
      echo "FAIL: the text prompt check passed with wrong expected ids" >&2
      exit 1
    fi
    echo "negative control failed as it must ($(grep -c FAIL "$LOG.qwen3.text-negative") failing check)"
  else
    echo "SKIP qwen3 text prompt: $PY has no tokenizers package (pip install tokenizers)"
  fi
  echo "== qwen3 KV pools"
  pools_in_place "$LOG.qwen3"
  echo "== qwen3 negative control (must fail)"
  if "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model qwen3 --fixture "$QWEN3_FIXTURE" --perturb >"$LOG.qwen3.negative" 2>&1; then
    echo "FAIL: the Qwen3 fixture checks passed with wrong expected ids" >&2
    cat "$LOG.qwen3.negative" >&2
    exit 1
  fi
  grep -c "^FAIL" "$LOG.qwen3.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  echo "== qwen3 batched prefill"
  "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model qwen3 --fixture "$QWEN3_FIXTURE" --require-ids
  expect_re "$LOG.qwen3" "instance 'qwen3_0_0': prefill_b4_c64 ran 4 sequence"
  if "$PY" "$HERE/prefill_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model qwen3 --fixture "$QWEN3_FIXTURE" --require-ids --repeats 1 --perturb >"$LOG.qwen3.prefill-negative" 2>&1; then
    echo "FAIL: the Qwen3 batched prefill checks passed comparing each prompt with the next one" >&2
    cat "$LOG.qwen3.prefill-negative" >&2
    exit 1
  fi
  grep -c "FAIL" "$LOG.qwen3.prefill-negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  if ! curl -sf "localhost:$HTTP_PORT/v2/health/live" >/dev/null; then
    echo "FAIL: the server is not live at the end of the Qwen3 checks" >&2
    exit 1
  fi
  stop_server

  # The same checkpoint with its weights staged as bf16 (-PweightDType=bf16):
  # half the bytes; every projection rounds its input to bf16 and sums in f32,
  # so the logits are not f32's, and the ids must still all be the fixture's.
  echo "== qwen3 bf16 weights"
  QB_DIR="${QWEN3_BF16_DIR:-$HERE/build/qwen3-bf16}"
  if [[ "${QWEN3_REEXPORT:-}" == 1 ]] || ! grep -q '"serving_manifest"' "$QB_DIR/repository/qwen3/config.pbtxt" 2>/dev/null \
      || ! grep -q ':wbf16:tiedHead"' "$QB_DIR/artifact/tlaloc-serving.json" 2>/dev/null; then
    rm -rf "$QB_DIR"
    mkdir -p "$QB_DIR"
    (cd "$ROOT" && ./gradlew -q :maestro:exportHfServingArtifact \
      -PckptDir="$QCKPT" -PoutDir="$QB_DIR/artifact" -PmaxBatch=4 -PweightDType=bf16)
    (cd "$ROOT" && ./gradlew -q :maestro:exportTritonModel \
      -PartifactDir="$QB_DIR/artifact" -PoutDir="$QB_DIR/repository" -PmodelName=qwen3 \
      -PkvMode=sequence -PmaxSequenceIdleMicros=5000000)
  fi
  export CONTAINER_NAME="$BASE_NAME-qwen3-bf16" MODEL_REPOSITORY="$QB_DIR/repository"
  start_server "$LOG.qwen3-bf16" 600
  F32_MIB="$(grep -o "model 'qwen3': uploaded 310 weights ([0-9]* MiB)" "$LOG.qwen3" | head -1 | grep -o "([0-9]*" | tr -d '(')"
  BF16_MIB="$(grep -o "model 'qwen3': uploaded 310 weights ([0-9]* MiB)" "$LOG.qwen3-bf16" | head -1 | grep -o "([0-9]*" | tr -d '(')"
  if [[ -z "$F32_MIB" || -z "$BF16_MIB" ]] || (( BF16_MIB * 2 < F32_MIB - 2 || BF16_MIB * 2 > F32_MIB + 2 )); then
    echo "FAIL: bf16 weights are ${BF16_MIB:-?} MiB, f32 ${F32_MIB:-?} MiB: not half" >&2
    exit 1
  fi
  echo "  ok   weights on the device: $BF16_MIB MiB in bf16, $F32_MIB MiB in f32"
  # Measured: every id equal, logits within 3.0e-3 of the largest (f32
  # weights: 5.6e-4); the tolerance is 6e-3, twice the measured difference.
  "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
    --model qwen3 --fixture "$QWEN3_FIXTURE" --tol 6e-3 --repeat 3
  if "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model qwen3 --fixture "$QWEN3_FIXTURE" --tol 6e-3 --repeat 0 --perturb >"$LOG.qwen3-bf16.negative" 2>&1; then
    echo "FAIL: the bf16 Qwen3 fixture checks passed with wrong expected ids" >&2
    cat "$LOG.qwen3-bf16.negative" >&2
    exit 1
  fi
  grep -c "^FAIL" "$LOG.qwen3-bf16.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
  # The logit comparison tells bf16 weights from f32 ones: at the f32
  # artifact's tolerance (2e-3) the bf16 logits must fail it.
  if "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model qwen3 --fixture "$QWEN3_FIXTURE" --tol 2e-3 --repeat 0 >"$LOG.qwen3-bf16.tight" 2>&1 \
      || ! grep -q "^FAIL .*logits within" "$LOG.qwen3-bf16.tight"; then
    echo "FAIL: the bf16 logits passed the f32 tolerance of 2e-3, or failed on something else" >&2
    cat "$LOG.qwen3-bf16.tight" >&2
    exit 1
  fi
  grep -c "^FAIL .*logits within" "$LOG.qwen3-bf16.tight" | xargs -I{} echo "  ok   at the f32 tolerance of 2e-3 the bf16 logits fail ({} failing checks)"
  stop_server
fi

# --- Muse Glimmer ------------------------------------------------------------
FX_DIR="$ROOT/ir/src/jvmTest/resources/io/tlaloc/ir/inference"
MUSE_BF16="$FX_DIR/muse_glimmer_30b_bf16_greedy.json"
MUSE_MIXED="$FX_DIR/muse_glimmer_30b_mixed_greedy.json"
MUSE_NEEDLE="$FX_DIR/muse_glimmer_30b_needle_mixed_greedy.json"
# The long-context export: context buckets, decode batches 1, 2 and 4, batch-1
# prefill in chunks of 512, and a full pool of four sequences at 32,768.
MUSE_LADDER="512,2048,8192,32768"

# mem_check <GiB> <step>: refuse by name unless MemAvailable covers it plus 16 GiB.
mem_check() {
  local need="$1" step="$2" avail
  avail=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
  echo "memory before $step: ${avail} GiB available, ${need} GiB needed plus a 16 GiB margin"
  if (( avail < need + 16 )); then
    echo "FAIL: refusing to $step with ${avail} GiB available" >&2
    exit 1
  fi
}

# drop_cache <paths...>: drop the page cache of every file under them.
drop_cache() {
  "$PY" - "$@" <<'PYEOF'
import os, sys
for root in sys.argv[1:]:
    for d, _, files in os.walk(root, followlinks=True):
        for f in files:
            try:
                fd = os.open(os.path.join(d, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
PYEOF
}

echo "== muse glimmer"
if [[ "${MUSE_GLIMMER:-}" != 1 ]]; then
  echo "SKIP muse glimmer: opt-in, set MUSE_GLIMMER=1 (56 GB of weights on the device)"
else
  MUSE_REV="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$MUSE_BF16")"
  MCKPT="$HOME/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/$MUSE_REV"
  M_DIR="${MUSE_GLIMMER_DIR:-$HERE/build/muse-glimmer}"
  if [[ ! -f "$MCKPT/model.safetensors.index.json" ]]; then
    echo "SKIP muse glimmer: no meta-models/Muse-Glimmer-30B checkpoint at $MCKPT"
  else
    M_CONFIG="$M_DIR/repository/muse/config.pbtxt"
    # A model exported before its sliding layers had a windowed KV pool, or
    # before the long-context ladder, is exported again.
    if [[ "${MUSE_GLIMMER_REEXPORT:-}" == 1 ]] || ! grep -q '"serving_manifest"' "$M_CONFIG" 2>/dev/null \
        || ! aliased "$M_DIR/artifact" || ! grep -q '"windowedKv"' "$M_DIR/artifact/tlaloc-serving.json" \
        || ! grep -q '"refusedTokens"' "$M_DIR/artifact/tlaloc-serving.json" \
        || ! grep -q "\"context\":\[$MUSE_LADDER\]" "$M_DIR/artifact/tlaloc-serving.json"; then
      mem_check 8 "export Muse Glimmer"
      rm -rf "$M_DIR"
      mkdir -p "$M_DIR"
      (cd "$ROOT" && ./gradlew -q :maestro:exportHfServingArtifact \
        -PckptDir="$MCKPT" -PoutDir="$M_DIR/artifact" -PmaxBatch=4 -PcontextLadder="$MUSE_LADDER" \
        -PprefillMaxBatch=1 -PprefillChunk=512 -PnumBlocks=8193)
      # A 32,768-token prompt prefills in about 4 minutes: sequences held
      # meanwhile must not time out. Steps of several sequences arrive more
      # than 1 ms apart, so the batcher waits up to 20 ms to batch them.
      (cd "$ROOT" && ./gradlew -q :maestro:exportTritonModel \
        -PartifactDir="$M_DIR/artifact" -PoutDir="$M_DIR/repository" -PmodelName=muse \
        -PkvMode=sequence -PmaxSequenceIdleMicros=600000000 -PmaxQueueDelayMicros=20000)
    fi
    drop_cache "$MCKPT/" "$M_DIR/artifact"
    WEIGHT_BYTES="$("$PY" -c "import json,sys; print(sum(w['byteLength'] for w in json.load(open(sys.argv[1]))['weights']['table']))" "$M_DIR/artifact/tlaloc-serving.json")"
    MEM_TOTAL="$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)"
    # Weights, the 4 GiB of KV pools, the largest entry's 4 GiB of temporary
    # memory (the compile log states it) and 7 GiB to spare.
    FRACTION="$("$PY" -c "import sys; print(round((int(sys.argv[1]) + 15 * 2**30) / int(sys.argv[2]), 3))" "$WEIGHT_BYTES" "$MEM_TOTAL")"
    echo "weights $((WEIGHT_BYTES / 2**30)) GiB of $((MEM_TOTAL / 2**30)) GiB: PJRT memory fraction $FRACTION"
    # The server also holds about 9 GiB on the host.
    mem_check $(( WEIGHT_BYTES / 2**30 + 15 + 9 )) "load Muse Glimmer"
    export CONTAINER_NAME="$BASE_NAME-muse" MODEL_REPOSITORY="$M_DIR/repository"
    PEAK_FILE="$(mktemp)"
    ( peak=0; while sleep 1; do
        used=$(awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END {printf "%d", (t - a) / 1048576}' /proc/meminfo)
        (( used > peak )) && peak=$used && echo "$peak" >"$PEAK_FILE"
      done ) &
    PEAK_PID=$!
    export TLALOC_PJRT_MEMORY_FRACTION="$FRACTION"
    start_server "$LOG.muse" 1800
    grep -o "compiled .* in [0-9]* ms" "$LOG.muse" || true
    grep -o "uploaded [0-9]* weights.*" "$LOG.muse" | head -1
    echo "-- against transformers in bfloat16"
    "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model muse --fixture "$MUSE_BF16" --ids-only --common-prefix-with "$MUSE_MIXED" --repeat 3
    echo "-- against transformers with bf16 weights and f32 activations"
    "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model muse --fixture "$MUSE_MIXED" --noise-factor 2 --repeat 0
    echo "-- a fact 2,190 tokens before the question, outside the sliding window (2,305-token prompt)"
    "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
      --model muse --fixture "$MUSE_NEEDLE" --tol 2e-2 --repeat 0
    expect_in "$LOG.muse" "at most 512 tokens per sequence in a prefill call"
    expect_re "$LOG.muse" "compiled prefill_b1_c32768 .* needs [0-9]+ MiB of temporary memory"
    echo "== muse glimmer KV pools"
    if ! grep -qE "updated the [0-9]+ KV pools in place" "$LOG.muse"; then
      echo "FAIL: the server log does not show the KV pools updated in place" >&2
      exit 1
    fi
    grep -oE "decode_b1_c[0-9]+ updated the [0-9]+ KV pools in place.*" "$LOG.muse" | head -1 | sed 's/^/  ok   log: /'
    refuse_in "$LOG.muse" "to new device memory"
    expect_in "$LOG.muse" "windowed KV pool for 39 sliding layers (window 2048)"
    echo "== muse glimmer placeholder tokens"
    expect_in "$LOG.muse" "refuses token ids 200091 (video_token_id) 200092 (image_token_id)"
    HERE="$HERE" "$PY" - "localhost:$HTTP_PORT" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["HERE"])
from sequence_client import SequenceClient
c = SequenceClient(sys.argv[1], "muse", "http")
bad = 0
for i, (tok, key) in enumerate(((200092, "image_token_id"), (200091, "video_token_id"))):
    try:
        c.step(900 + i, [200000, tok, 1000], start=True, end=True)
        print(f"FAIL a START holding {tok} was accepted")
        bad += 1
    except Exception as e:  # tritonclient's exception type
        ok = f"is {tok}, the model's {key} placeholder" in str(e)
        print(("  ok   " if ok else "FAIL ") + f"a START holding {tok} ({key}) is refused by name")
        bad += not ok
sys.exit(1 if bad else 0)
PYEOF
    grep -oE "instance 'muse_0_0': [0-9]+ KV pools \([0-9]+ MiB\) zeroed[^;]*;[^;]*;[^;]*;" "$LOG.muse" | head -1 | sed 's/^/  ok   log: /' || true
    echo "== muse glimmer negative control (must fail)"
    if "$PY" "$HERE/fixture_checks.py" --http "localhost:$HTTP_PORT" --grpc "localhost:$GRPC_PORT" \
        --model muse --fixture "$MUSE_NEEDLE" --tol 2e-2 --repeat 0 --perturb >"$LOG.muse.negative" 2>&1; then
      echo "FAIL: the Muse Glimmer fixture checks passed with wrong expected ids" >&2
      cat "$LOG.muse.negative" >&2
      exit 1
    fi
    grep -c "^FAIL" "$LOG.muse.negative" | xargs -I{} echo "negative control failed as it must ({} failing checks)"
    if [[ -n "${MUSE_CONTEXT_BENCH:-}" ]]; then
      echo "== muse glimmer timings at contexts $MUSE_CONTEXT_BENCH"
      "$PY" "$HERE/context_bench.py" --grpc "localhost:$GRPC_PORT" --http "localhost:$HTTP_PORT" \
        --model muse --contexts "$MUSE_CONTEXT_BENCH"
    fi
    kill "$PEAK_PID" 2>/dev/null || true
    PEAK_PID=""
    echo "peak memory in use (system and GPU, unified): $(cat "$PEAK_FILE") GiB"
    stop_server
  fi
fi
echo "VERIFY PASSED"
