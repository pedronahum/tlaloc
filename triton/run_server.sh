#!/usr/bin/env bash
# Starts tritonserver in the Triton container with the tlaloc backend, a model
# repository and a PJRT plugin mounted read-only.
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
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.11-py3}"
REPO="$(realpath "${MODEL_REPOSITORY:-$HERE/examples/model_repository}")"
PLUGIN="$(realpath "${PJRT_PLUGIN:-$HERE/pjrt/xla_cuda13/xla_cuda_plugin.so}")"
BACKEND_DIR="$HERE/backends/tlaloc"
NAME="${CONTAINER_NAME:-tlaloc-triton}"
FRACTION="${TLALOC_PJRT_MEMORY_FRACTION:-0.3}"
PREALLOCATE="${TLALOC_PJRT_PREALLOCATE:-false}"

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
  --gpus all \
  -p "${HTTP_PORT:-8000}:8000" -p "${GRPC_PORT:-8001}:8001" -p "${METRICS_PORT:-8002}:8002" \
  -v /dev/shm:/dev/shm \
  -v "$BACKEND_DIR:/opt/tritonserver/backends/tlaloc:ro" \
  -v "$REPO:/models:ro" \
  -v "$PLUGIN:/opt/pjrt/xla_cuda_plugin.so:ro" \
  -e TLALOC_PJRT_PLUGIN_PATH=/opt/pjrt/xla_cuda_plugin.so \
  -e TLALOC_PJRT_MEMORY_FRACTION="$FRACTION" \
  -e TLALOC_PJRT_PREALLOCATE="$PREALLOCATE" \
  "$IMAGE" \
  tritonserver --model-repository=/models "$@"
