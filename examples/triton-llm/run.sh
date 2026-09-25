#!/usr/bin/env bash
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
#
# Export a HuggingFace checkpoint with Tlaloc, serve it with Triton and
# libtriton_tlaloc.so, and ask it a question.
#
#   examples/triton-llm/run.sh [--model qwen3|tinyllama|muse-glimmer] [--question "..."]
#
# Steps:
#   1. check what the run needs, and skip by name (exit 0) if something is
#      missing: Docker, a GPU, the Triton image, the PJRT plugin, the
#      checkpoint, the client's Python packages;
#   2. Kotlin (this directory's Gradle build) writes the Triton model
#      repository into build/<model>/. Reused on later runs; REEXPORT=1 writes
#      it again;
#   3. build libtriton_tlaloc.so if triton/backends/ does not have it yet;
#   4. start Triton in its container with the model, wait until it is ready;
#   5. chat.py sends the question and prints the answer token by token;
#   6. stop the container.
#
# Environment (defaults in brackets):
#   CHECKPOINT      the checkpoint directory [found in the HuggingFace cache]
#   CHAT_PYTHON     a Python with requirements.txt installed [python3]
#   TRITON_IMAGE    [nvcr.io/nvidia/tritonserver:25.11-py3]
#   PJRT_PLUGIN     [triton/pjrt/xla_cuda13/xla_cuda_plugin.so]
#   HTTP_PORT / GRPC_PORT / METRICS_PORT   [8000 / 8001 / 8002]
#   REEXPORT=1      export again even if build/<model>/ has a model
#   MAX_NEW         the most tokens to generate [160]; the context is 256
#                   (8192 for Muse Glimmer)
#   REASONING_STRENGTH  Muse Glimmer's chat template variable [low]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TRITON="$ROOT/triton"

MODEL=qwen3
QUESTION="Why is the sky blue? Answer in two sentences."
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --question) QUESTION="$2"; shift 2 ;;
    *) echo "usage: $0 [--model qwen3|tinyllama|muse-glimmer] [--question TEXT]" >&2; exit 2 ;;
  esac
done

skip() {
  echo "SKIP: $*"
  exit 0
}

# --- 1. What the run needs ----------------------------------------------------

HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
# snapshot <org--name> <file>: the cached snapshot that has <file>, preferring
# the revision refs/main points at.
snapshot() {
  local repo="$HUB/models--$1" want="$2" rev
  if [[ -f "$repo/refs/main" ]]; then
    rev="$(cat "$repo/refs/main")"
    [[ -f "$repo/snapshots/$rev/$want" ]] && { echo "$repo/snapshots/$rev"; return; }
  fi
  for d in "$repo"/snapshots/*/; do
    [[ -f "$d$want" ]] && { echo "${d%/}"; return; }
  done
  return 0
}

case "$MODEL" in
  qwen3)
    REPO_ID="Qwen/Qwen3-0.6B"
    CKPT="${CHECKPOINT:-$(snapshot Qwen--Qwen3-0.6B model.safetensors)}"
    WEIGHTS_FILE=model.safetensors
    ;;
  tinyllama)
    REPO_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    CKPT="${CHECKPOINT:-$(snapshot TinyLlama--TinyLlama-1.1B-Chat-v1.0 model.safetensors)}"
    LEGACY="$HOME/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0"
    [[ -z "$CKPT" && -f "$LEGACY/model.safetensors" ]] && CKPT="$LEGACY"
    WEIGHTS_FILE=model.safetensors
    ;;
  muse-glimmer)
    REPO_ID="meta-models/Muse-Glimmer-30B"
    CKPT="${CHECKPOINT:-$(snapshot meta-models--Muse-Glimmer-30B model.safetensors.index.json)}"
    WEIGHTS_FILE=model.safetensors.index.json
    ;;
  *) echo "unknown --model '$MODEL'; one of qwen3, tinyllama, muse-glimmer" >&2; exit 2 ;;
esac

command -v docker >/dev/null || skip "no docker on PATH. Triton runs in a container."
docker info >/dev/null 2>&1 || skip "docker is installed but the daemon does not answer ('docker info' failed)."
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
  || skip "no NVIDIA GPU: nvidia-smi is missing or lists no GPU."
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.11-py3}"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || skip "no Triton image $IMAGE on this machine (docker pull $IMAGE, about 13 GB)."
PLUGIN="${PJRT_PLUGIN:-$TRITON/pjrt/xla_cuda13/xla_cuda_plugin.so}"
[[ -f "$PLUGIN" ]] || skip "no PJRT CUDA plugin at $PLUGIN (triton/fetch_pjrt_plugin.sh downloads it)."
[[ -n "$CKPT" && -f "$CKPT/$WEIGHTS_FILE" && -f "$CKPT/tokenizer.json" ]] \
  || skip "no $REPO_ID checkpoint in $HUB (hf download $REPO_ID, or set CHECKPOINT)."
PY="${CHAT_PYTHON:-python3}"
MISSING="$("$PY" - <<'EOF' 2>/dev/null || echo "a working $PY"
import importlib.util
names = {"tritonclient.grpc": "tritonclient[grpc]", "tokenizers": "tokenizers",
         "jinja2": "jinja2", "numpy": "numpy"}
missing = []
for module, package in names.items():
    try:
        found = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        missing.append(package)
print(" ".join(missing))
EOF
)"
[[ -z "$MISSING" ]] || skip "$PY lacks $MISSING (pip install -r $HERE/requirements.txt, then set CHAT_PYTHON)."

HTTP_PORT="${HTTP_PORT:-8000}" GRPC_PORT="${GRPC_PORT:-8001}" METRICS_PORT="${METRICS_PORT:-8002}"
for port in "$HTTP_PORT" "$GRPC_PORT" "$METRICS_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "port $port is in use; set HTTP_PORT, GRPC_PORT and METRICS_PORT to free ports" >&2
    exit 1
  fi
done
NAME="tlaloc-triton-llm"
if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "a container named $NAME exists already; remove it (docker rm -f $NAME) and run again" >&2
  exit 1
fi

gib_available() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
if [[ "$MODEL" == muse-glimmer ]]; then
  # The GB10's GPU memory is the system RAM: 56 GB of weights on the device
  # come out of the same pool as everything else on the machine.
  echo "WARNING: Muse Glimmer puts 56 GB of bf16 weights on the GPU; the export"
  echo "         writes 56 GB to $HERE/build/muse-glimmer and the load takes minutes."
  AVAIL="$(gib_available)"
  (( AVAIL >= 80 )) || skip "Muse Glimmer needs 80 GiB of available memory (56 GiB of weights, 8 GiB of" \
    "working memory, a 16 GiB margin); this machine has $AVAIL GiB available."
fi

OUT="$HERE/build/$MODEL"
LOG="$OUT/server.log"

# --- 2. Kotlin writes the Triton model repository -----------------------------

if [[ "${REEXPORT:-}" == 1 || ! -f "$OUT/repository/$MODEL/config.pbtxt" ]]; then
  echo "== export (Kotlin)"
  "$ROOT/gradlew" -q -p "$HERE" run \
    --args="--model $MODEL --checkpoint $CKPT --out $OUT"
else
  echo "== export: reusing $OUT/repository (REEXPORT=1 writes it again)"
fi

# --- 3. The backend -----------------------------------------------------------

if [[ ! -f "$TRITON/backends/tlaloc/libtriton_tlaloc.so" ]]; then
  echo "== building libtriton_tlaloc.so"
  "$TRITON/build_backend.sh"
fi

# --- 4. Triton ----------------------------------------------------------------

FRACTION="${TLALOC_PJRT_MEMORY_FRACTION:-0.3}"
if [[ "$MODEL" == muse-glimmer ]]; then
  # The PJRT client may use the weights plus 8 GiB, as a share of all memory.
  WEIGHT_BYTES="$(python3 -c "import json,sys; print(sum(w['byteLength'] for w in json.load(open(sys.argv[1]))['weights']['table']))" "$OUT/artifact/tlaloc-serving.json")"
  MEM_TOTAL="$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)"
  FRACTION="$(python3 -c "import sys; print(round((int(sys.argv[1]) + 8 * 2**30) / int(sys.argv[2]), 3))" "$WEIGHT_BYTES" "$MEM_TOTAL")"
  # The export and the load read the same 56 GB; drop them from the page
  # cache so they do not compete with the weights on the device.
  python3 - "$CKPT/" "$OUT/artifact" <<'EOF'
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
EOF
  AVAIL="$(gib_available)"
  (( AVAIL >= WEIGHT_BYTES / 2**30 + 24 )) \
    || skip "loading Muse Glimmer needs $(( WEIGHT_BYTES / 2**30 + 24 )) GiB available; $AVAIL GiB are."
fi

LOGGER=""
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  [[ -n "$LOGGER" ]] && kill "$LOGGER" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap "exit 130" INT TERM

echo "== starting Triton ($NAME, PJRT memory fraction $FRACTION, log: $LOG)"
T0=$(date +%s)
CONTAINER_NAME="$NAME" MODEL_REPOSITORY="$OUT/repository" PJRT_PLUGIN="$PLUGIN" \
  TRITON_IMAGE="$IMAGE" HTTP_PORT="$HTTP_PORT" GRPC_PORT="$GRPC_PORT" METRICS_PORT="$METRICS_PORT" \
  TLALOC_PJRT_MEMORY_FRACTION="$FRACTION" \
  "$TRITON/run_server.sh" --detach >/dev/null
docker logs -f "$NAME" >"$LOG" 2>&1 &
LOGGER=$!
READY=""
for _ in $(seq 1 1800); do
  if curl -sf "localhost:$HTTP_PORT/v2/health/ready" >/dev/null 2>&1; then READY=1; break; fi
  docker inspect "$NAME" >/dev/null 2>&1 || break
  sleep 1
done
if [[ -z "$READY" ]]; then
  echo "the server did not become ready; the last lines of $LOG:" >&2
  tail -30 "$LOG" >&2
  exit 1
fi
printf "ready in %d s: " "$(( $(date +%s) - T0 ))"
grep -o "uploaded [0-9]* weights[^\"]*" "$LOG" | head -1 || echo

# --- 5. The question ----------------------------------------------------------

echo "== chat (gRPC, localhost:$GRPC_PORT)"
TEMPLATE_VARS=()
# Muse Glimmer's template writes today's date into the system message, and
# asks for high reasoning strength unless told otherwise. The model reasons
# in a channel of its own before it answers; low keeps that part short.
[[ "$MODEL" == muse-glimmer ]] && TEMPLATE_VARS=(--template-var "current_date=$(date +%F)"
  --template-var "reasoning_strength=${REASONING_STRENGTH:-low}")
CONTEXT=256
[[ "$MODEL" == muse-glimmer ]] && CONTEXT=8192
"$PY" "$HERE/chat.py" --url "localhost:$GRPC_PORT" --model "$MODEL" --checkpoint "$CKPT" \
  --question "$QUESTION" --max-new "${MAX_NEW:-160}" --context "$CONTEXT" "${TEMPLATE_VARS[@]}"

# --- 6. Stop ------------------------------------------------------------------
# (the EXIT trap removes the container)
