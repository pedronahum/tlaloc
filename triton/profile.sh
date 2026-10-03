#!/usr/bin/env bash
# Times, and optionally profiles, a sequence-mode model served by the tlaloc
# backend: decode steps and prefill chunks at chosen context buckets, the
# time the client sees against the backend's own time per request, and, with
# MODE=nsys, the GPU kernels of each measured part.
#
#   triton/profile.sh MODEL_DIR MODEL_NAME WORKLOAD...
#
#   MODEL_DIR   a directory with repository/MODEL_NAME and artifact/ (as
#               verify.sh writes them, e.g. triton/build/qwen3)
#   WORKLOAD    decode:PREFIX:STEPS[:STREAMS] or prefill:PREFIX:CHUNKS (profile_client.py)
#
# Before starting anything it checks that the GPU is idle: nvidia-smi is
# sampled once a second for 10 s. If the median utilization is 5% or more
# (another process is using the GPU) it waits, sampling again every 60 s, for
# up to WAIT_MINUTES; if the GPU is still busy it goes on and labels the run
# "contended" with the utilization it saw. The same check runs again once the
# server is loaded and idle, just before the measurement.
#
# MODE=time (default): the server runs plain, each workload REPEAT times.
# MODE=nsys: the server runs under `nsys launch` (the container's own Nsight
#   Systems) and the first repeat of each workload runs inside a capture
#   window; each window is written to OUT_DIR/<workload>.nsys-rep, exported to
#   SQLite and summarised by profile_report.py (kernels grouped by kind, GPU
#   busy time against the time between executions).
# XLA_DUMP=1: the PJRT plugin writes each compiled program's HLO before and
#   after optimization to OUT_DIR/xla_dump (XLA_FLAGS=--xla_dump_to).
#
# Environment (defaults in brackets):
#   MODE [time], REPEAT [3], XLA_DUMP [0], OUT_DIR [triton/build/profile/<model>-<mode>]
#   MAX_ID [30000]         prompt ids are drawn below this (the vocabulary)
#   REFUSED [""]           ids never sent (the manifest's refused tokens)
#   CHUNK [512]            tokens per prefill request
#   WAIT_MINUTES [45]
#   TLALOC_PJRT_MEMORY_FRACTION [computed: weights + EXTRA_GIB over MemTotal]
#   EXTRA_GIB [15]         device memory besides the weights (KV pools, temporaries)
#   HTTP_PORT / GRPC_PORT / METRICS_PORT [8020 / 8021 / 8022]
#   TRITON_CLIENT_PYTHON [python3]  a Python with tritonclient[grpc] and numpy
#   PJRT_PLUGIN [triton/pjrt/xla_cuda13/xla_cuda_plugin.so]
#   BACKEND_DIR [triton/backends/tlaloc]  the backend build to serve with (for A/B runs)
#   XLA_EXTRA_FLAGS [""]   appended to the plugin's XLA_FLAGS
#
# The run is refused by name when MemAvailable is below the weights plus
# EXTRA_GIB plus 9 GiB for the server's host side plus a 16 GiB margin, or when
# one of the three ports is already taken. The container is always stopped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 3 )); then
  sed -n '2,45p' "$0" >&2
  exit 2
fi
MODEL_DIR="$(realpath "$1")"; MODEL="$2"; shift 2
WORKLOADS=("$@")
MODE="${MODE:-time}"
REPEAT="${REPEAT:-3}"
PY="${TRITON_CLIENT_PYTHON:-python3}"
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.11-py3}"
PLUGIN="$(realpath "${PJRT_PLUGIN:-$HERE/pjrt/xla_cuda13/xla_cuda_plugin.so}")"
HTTP_PORT="${HTTP_PORT:-8020}" GRPC_PORT="${GRPC_PORT:-8021}" METRICS_PORT="${METRICS_PORT:-8022}"
OUT_DIR="${OUT_DIR:-$HERE/build/profile/$MODEL-$MODE}"
NAME="tlaloc-profile-$$"
SESSION=tlaloc
mkdir -p "$OUT_DIR"
OUT_DIR="$(realpath "$OUT_DIR")"
LOG="$OUT_DIR/server.log"
MANIFEST="$MODEL_DIR/artifact/tlaloc-serving.json"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

for p in "$HTTP_PORT" "$GRPC_PORT" "$METRICS_PORT"; do
  if ss -ltn "sport = :$p" | grep -q LISTEN; then
    echo "FAIL: port $p is in use; set HTTP_PORT/GRPC_PORT/METRICS_PORT" >&2
    exit 1
  fi
done

WEIGHT_BYTES="$("$PY" -c "import json,sys; print(sum(w['byteLength'] for w in json.load(open(sys.argv[1]))['weights']['table']))" "$MANIFEST")"
MEM_TOTAL="$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)"
EXTRA_GIB="${EXTRA_GIB:-15}"
FRACTION="${TLALOC_PJRT_MEMORY_FRACTION:-$("$PY" -c "import sys; print(min(0.9, round((int(sys.argv[1]) + int(sys.argv[3]) * 2**30) / int(sys.argv[2]), 3)))" "$WEIGHT_BYTES" "$MEM_TOTAL" "$EXTRA_GIB")}"
NEED=$(( WEIGHT_BYTES / 2**30 + EXTRA_GIB + 9 ))
AVAIL=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
echo "weights $((WEIGHT_BYTES / 2**20)) MiB ($(awk -v b="$WEIGHT_BYTES" 'BEGIN {printf "%.1f", b / 273e6}') ms at 273 GB/s); PJRT memory fraction $FRACTION"
echo "memory: ${AVAIL} GiB available, ${NEED} GiB needed plus a 16 GiB margin"
if (( AVAIL < NEED + 16 )); then
  echo "FAIL: refusing to load $MODEL with ${AVAIL} GiB available" >&2
  exit 1
fi

# gpu_util: median nvidia-smi utilization over 10 one-second samples.
gpu_util() {
  local s=() i
  for i in $(seq 10); do
    s+=("$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')")
    sleep 1
  done
  printf '%s\n' "${s[@]}" | sort -n | sed -n '5p'
}
CONTENDED=""
# idle_gate <where>: wait for the GPU to be idle, or label the run contended.
idle_gate() {
  local where="$1" util waited=0
  util="$(gpu_util)"
  while (( util >= 5 )) && (( waited < ${WAIT_MINUTES:-45} )); do
    echo "GPU busy ${util}% ($where); waiting (${waited} of ${WAIT_MINUTES:-45} minutes)"
    sleep 50
    waited=$((waited + 1))
    util="$(gpu_util)"
  done
  if (( util >= 5 )); then
    CONTENDED="contended (GPU ${util}% busy $where)"
    echo "CONTENDED: the GPU is ${util}% busy $where; every number below is labelled contended"
  else
    echo "GPU idle ($where): median utilization ${util}% over 10 s"
  fi
}

docker ps --format '{{.Names}} {{.Image}} {{.Ports}}' | sed 's/^/running: /'
idle_gate "before loading"

ENVS=(-e TLALOC_PJRT_PLUGIN_PATH=/opt/pjrt/xla_cuda_plugin.so
      -e TLALOC_PJRT_MEMORY_FRACTION="$FRACTION" -e TLALOC_PJRT_PREALLOCATE=false)
XLA_FLAGS_ALL="${XLA_EXTRA_FLAGS:-}"
if [[ "${XLA_DUMP:-0}" == 1 ]]; then
  mkdir -p "$OUT_DIR/xla_dump"
  XLA_FLAGS_ALL="--xla_dump_to=/profile/xla_dump --xla_dump_hlo_as_text $XLA_FLAGS_ALL"
fi
[[ -n "$XLA_FLAGS_ALL" ]] && ENVS+=(-e XLA_FLAGS="$XLA_FLAGS_ALL")
BACKEND_DIR="$(realpath "${BACKEND_DIR:-$HERE/backends/tlaloc}")"
echo "backend $BACKEND_DIR/libtriton_tlaloc.so ($(md5sum <"$BACKEND_DIR/libtriton_tlaloc.so" | cut -c1-8))${XLA_FLAGS_ALL:+; XLA_FLAGS=$XLA_FLAGS_ALL}"
CMD=(tritonserver --model-repository=/models --model-control-mode=explicit --load-model="$MODEL")
if [[ "$MODE" == nsys ]]; then
  CMD=(nsys launch --session-new="$SESSION" --trace=cuda,nvtx --cuda-graph-trace=node "${CMD[@]}" --log-verbose=1)
fi
docker run -d --rm --name "$NAME" --ipc=host --gpus all \
  -p "$HTTP_PORT:8000" -p "$GRPC_PORT:8001" -p "$METRICS_PORT:8002" \
  -v "$BACKEND_DIR:/opt/tritonserver/backends/tlaloc:ro" \
  -v "$MODEL_DIR/repository:/models:ro" \
  -v "$PLUGIN:/opt/pjrt/xla_cuda_plugin.so:ro" \
  -v "$OUT_DIR:/profile" \
  "${ENVS[@]}" "$IMAGE" "${CMD[@]}" >/dev/null
docker logs -f "$NAME" >"$LOG" 2>&1 &
echo "server $NAME ($MODE) starting; log $LOG"
for i in $(seq 3600); do
  if curl -sf "localhost:$HTTP_PORT/v2/models/$MODEL/ready" >/dev/null; then break; fi
  if ! docker inspect "$NAME" >/dev/null 2>&1; then echo "FAIL: server exited" >&2; tail -20 "$LOG" >&2; exit 1; fi
  sleep 1
done
curl -sf "localhost:$HTTP_PORT/v2/models/$MODEL/ready" >/dev/null || { echo "FAIL: not ready" >&2; exit 1; }
grep -o "compiled [a-z0-9_]* .* in [0-9]* ms.*temporary memory" "$LOG" | sed 's/ (bodies[^)]*)//; s/^/  /' || true
grep -o "uploaded [0-9]* weights ([0-9]* MiB) in [0-9]* ms" "$LOG" | sed 's/^/  /' || true
idle_gate "with the server loaded and idle"

WINDOW=()
if [[ "$MODE" == nsys ]]; then
  WINDOW=(--window-start "docker exec $NAME nsys start --session=$SESSION -o /profile/{name}"
          --window-stop "docker exec $NAME nsys stop --session=$SESSION")
fi
ARGS=()
for w in "${WORKLOADS[@]}"; do ARGS+=(--workload "$w"); done
[[ "$MODE" == nsys ]] && REPEAT=1
PYTHONPATH="$HERE" "$PY" "$HERE/profile_client.py" --grpc "localhost:$GRPC_PORT" --http "localhost:$HTTP_PORT" \
  --model "$MODEL" --repeat "$REPEAT" --chunk "${CHUNK:-512}" --max-id "${MAX_ID:-30000}" \
  --refused "${REFUSED:-}" --json "$OUT_DIR/timings.json" "${WINDOW[@]}" "${ARGS[@]}"
[[ -n "$CONTENDED" ]] && echo "NOTE: $CONTENDED"
echo "{\"contended\": \"${CONTENDED}\", \"mode\": \"$MODE\"}" >"$OUT_DIR/conditions.json"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# The container wrote as root; hand the files back.
docker run --rm -v "$OUT_DIR:/profile" --entrypoint chown "$IMAGE" -R "$(id -u):$(id -g)" /profile

if [[ "$MODE" == nsys ]]; then
  NSYS_HOST="${NSYS_HOST:-$(command -v nsys || true)}"
  for rep in "$OUT_DIR"/*.nsys-rep; do
    [[ -f "$rep" ]] || continue
    if [[ -n "$NSYS_HOST" ]]; then
      "$NSYS_HOST" export --type sqlite --force-overwrite true -o "${rep%.nsys-rep}.sqlite" "$rep" >/dev/null
    else
      docker run --rm -v "$OUT_DIR:/profile" --entrypoint nsys "$IMAGE" \
        export --type sqlite --force-overwrite true -o "/profile/$(basename "${rep%.nsys-rep}").sqlite" "/profile/$(basename "$rep")" >/dev/null
    fi
    DUMP=()
    [[ "${XLA_DUMP:-0}" == 1 ]] && DUMP=(--xla-dump "$OUT_DIR/xla_dump")
    "$PY" "$HERE/profile_report.py" "${rep%.nsys-rep}.sqlite" --manifest "$MANIFEST" "${DUMP[@]}" --json "${rep%.nsys-rep}.report.json"
  done
fi
