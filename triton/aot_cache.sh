#!/usr/bin/env bash
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
#
# Warm an executable cache for a serving artifact, so it can ship ready to load: the
# ahead-of-time step of docs/work-log/xse-lessons-plan.md item 2. Triton compiles every entry at
# load and writes the cache in its own format with its own keys; this starts it once against a
# cache directory, waits for the model, and stops. The directory is then the artifact's AOT cache.
# Serve with TLALOC_PJRT_CACHE_DIR pointing at it (and TLALOC_PJRT_CACHE_ONLY=1 to make a miss a
# refusal rather than a compile) and the load compiles nothing.
#
# Why warm with Triton rather than compile in the JVM: the cache key covers the plugin's identity,
# the compile options and the program text AS TRITON PREPARES IT (PrepareForXla renames the entry).
# A JVM-written cache would have to reproduce all three exactly or silently miss; letting the same
# code write and read the cache cannot misalign. The cache is tied to this plugin and machine —
# the key covers the plugin's size and mtime — so it is AOT for THIS target, not a portable one.
#
#   MODEL_REPOSITORY=<repository> MODEL=<name> CACHE_DIR=<dir> triton/aot_cache.sh
#
# Environment:
#   MODEL_REPOSITORY   [required] the repository holding the model
#   MODEL              [required] the model's directory name
#   CACHE_DIR          [required] where the compiled executables are written (created if absent)
#   TLALOC_PJRT_MEMORY_FRACTION [0.3]
#   TRITON_IMAGE       [nvcr.io/nvidia/tritonserver:25.11-py3]
#   CONTAINER_NAME     [tlaloc-aot]
#   HTTP_PORT / GRPC_PORT / METRICS_PORT [8000 / 8001 / 8002]
#   READY_TIMEOUT      [900] seconds to wait for the model to compile and load
#
# Exits non-zero if the model never became ready. Prints the cache entry count and bytes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MODEL_REPOSITORY:?set MODEL_REPOSITORY}"
MODEL="${MODEL:?set MODEL}"
CACHE="${CACHE_DIR:?set CACHE_DIR}"
NAME="${CONTAINER_NAME:-tlaloc-aot}"
HTTP="${HTTP_PORT:-8000}"
GRPC="${GRPC_PORT:-8001}"
METRICS="${METRICS_PORT:-8002}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

mkdir -p "$CACHE"
CACHE="$(realpath "$CACHE")"
REPO="$(realpath "$REPO")"

stop() { docker stop "$NAME" >/dev/null 2>&1 || true; }
trap stop EXIT

echo "warming the cache for $MODEL at $CACHE"
stop
# Exported, not an inline ${VAR:+...} prefix: a word that becomes VAR=value only after expansion
# is treated as a command name, not an assignment, and bash tries to execute it.
if [[ -n "${TRITON_IMAGE:-}" ]]; then export TRITON_IMAGE; fi
MODEL_REPOSITORY="$REPO" CONTAINER_NAME="$NAME" \
  TLALOC_PJRT_CACHE_DIR="$CACHE" \
  TLALOC_PJRT_MEMORY_FRACTION="${TLALOC_PJRT_MEMORY_FRACTION:-0.3}" \
  HTTP_PORT="$HTTP" GRPC_PORT="$GRPC" METRICS_PORT="$METRICS" \
  bash "$HERE/run_server.sh" --detach < /dev/null

python3 -c 'import sys, time, urllib.request
http, model, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
t0 = time.time()
while time.time() - t0 < timeout:
    try:
        if urllib.request.urlopen(f"http://localhost:{http}/v2/models/{model}/ready", timeout=10).status == 200:
            print(f"model ready after {time.time()-t0:.0f} s (the compiles are now in the cache)")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(3)
print(f"model NOT ready after {timeout:.0f} s", file=sys.stderr)
sys.exit(1)' "$HTTP" "$MODEL" "$READY_TIMEOUT"

stop
trap - EXIT
entries=$(find "$CACHE" -name '*.pjrt' | wc -l)
bytes=$(du -sb "$CACHE" | cut -f1)
echo "cache: $entries entries, $bytes bytes at $CACHE"
echo "serve with TLALOC_PJRT_CACHE_DIR=$CACHE (and TLALOC_PJRT_CACHE_ONLY=1 to refuse a miss)"
