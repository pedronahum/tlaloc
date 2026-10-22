#!/usr/bin/env bash
# Starts tritonserver in the Triton container with the tlaloc backend, a model
# repository and a PJRT plugin mounted read-only. The container shares the
# host's IPC namespace (--ipc=host), which CUDA shared memory clients need.
#
#   triton/run_server.sh [--detach] [extra tritonserver arguments...]
#
# Environment (defaults in brackets):
#   TRITON_IMAGE                  [nvcr.io/nvidia/tritonserver:25.11-py3]
#   MODEL_REPOSITORY              [triton/examples/model_repository]
#   PJRT_PLUGIN                   [triton/pjrt/xla_cuda13/xla_cuda_plugin.so]
#   CONTAINER_NAME                [tlaloc-triton]
#   HTTP_PORT / GRPC_PORT / METRICS_PORT   [8000 / 8001 / 8002]
#   TLALOC_PJRT_MEMORY_FRACTION   [0.3]  share of GPU memory the PJRT client may use
#   TLALOC_PJRT_PREALLOCATE       [false]
#   TLALOC_CHAIN_STEPS            [1]  0: the backend does not chain verify steps
#   TLALOC_ATTN_PREFILL_EXACT     [0]  1: prefill attention keeps ~22 bits of Q and P, not f16
#   TLALOC_PJRT_CACHE_DIR         [unset]  compiled executables kept here (mounted read-write)
#   TLALOC_COMPILE_THREADS        [1]  programs compiled at once at model load
#   TLALOC_DETERMINISTIC          [0]  1: the CUDA kernels' order-independent sums and XLA's
#                                      deterministic ops (--xla_gpu_deterministic_ops); with
#                                      TLALOC_PJRT_CACHE_DIR every start runs the same executables
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.11-py3}"
REPO="$(realpath "${MODEL_REPOSITORY:-$HERE/examples/model_repository}")"
PLUGIN="$(realpath "${PJRT_PLUGIN:-$HERE/pjrt/xla_cuda13/xla_cuda_plugin.so}")"
BACKEND_DIR="$HERE/backends/tlaloc"
NAME="${CONTAINER_NAME:-tlaloc-triton}"
FRACTION="${TLALOC_PJRT_MEMORY_FRACTION:-0.3}"
PREALLOCATE="${TLALOC_PJRT_PREALLOCATE:-false}"

CACHE=()
if [[ -n "${TLALOC_PJRT_CACHE_DIR:-}" ]]; then
  mkdir -p "$TLALOC_PJRT_CACHE_DIR"
  CACHE_DIR="$(realpath "$TLALOC_PJRT_CACHE_DIR")"
  CACHE=(-v "$CACHE_DIR:$CACHE_DIR" -e TLALOC_PJRT_CACHE_DIR="$CACHE_DIR")
fi
THREADS=()
if [[ -n "${TLALOC_COMPILE_THREADS:-}" ]]; then
  THREADS=(-e TLALOC_COMPILE_THREADS="$TLALOC_COMPILE_THREADS")
fi

DET=()
if [[ "${TLALOC_DETERMINISTIC:-0}" == 1 ]]; then
  DET=(-e TLALOC_DETERMINISTIC=1 -e XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}")
fi

DETACH=()
if [[ "${1:-}" == "--detach" ]]; then
  DETACH=(-d)
  shift
fi

if [[ ! -f "$BACKEND_DIR/libtriton_tlaloc.so" ]]; then
  echo "missing $BACKEND_DIR/libtriton_tlaloc.so; run triton/build_backend.sh" >&2
  exit 1
fi
if [[ ! -f "$PLUGIN" ]]; then
  echo "missing PJRT plugin $PLUGIN; run triton/fetch_pjrt_plugin.sh or set PJRT_PLUGIN" >&2
  exit 1
fi

exec docker run "${DETACH[@]}" --rm --name "$NAME" \
  --ipc=host \
  --gpus all \
  -p "${HTTP_PORT:-8000}:8000" -p "${GRPC_PORT:-8001}:8001" -p "${METRICS_PORT:-8002}:8002" \
  -v /dev/shm:/dev/shm \
  -v "$BACKEND_DIR:/opt/tritonserver/backends/tlaloc:ro" \
  -v "$REPO:/models:ro" \
  -v "$PLUGIN:/opt/pjrt/xla_cuda_plugin.so:ro" \
  -e TLALOC_PJRT_PLUGIN_PATH=/opt/pjrt/xla_cuda_plugin.so \
  -e TLALOC_PJRT_MEMORY_FRACTION="$FRACTION" \
  -e TLALOC_CHAIN_STEPS="${TLALOC_CHAIN_STEPS:-1}" \
  -e TLALOC_ATTN_PREFILL_EXACT="${TLALOC_ATTN_PREFILL_EXACT:-0}" \
  -e TLALOC_PJRT_PREALLOCATE="$PREALLOCATE" \
  "${CACHE[@]}" "${THREADS[@]}" "${DET[@]}" \
  "$IMAGE" \
  tritonserver --model-repository=/models "$@"
