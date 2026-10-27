#!/usr/bin/env bash
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
#
# Compiles the programs of the TPU session with XLA:TPU on this machine, no
# TPU attached, so a program XLA:TPU refuses fails here instead of on a VM.
#
#   scripts/tpu/aot-check.sh [--all] [--programs DIR]
#
# 1. Record: the device tests of the session's lanes run on this machine
#    (CUDA, or the XLA CPU plugin for the multi-device ones) with
#    TLALOC_TPU_PROGRAMS_DIR set, and every PjrtSession writes the TPU form of
#    each program it compiles. The kernel fixtures, which run only on a TPU,
#    are recorded by TpuProgramRecorderTest. --all records every runtime-pjrt
#    test (the session's suite lane) instead of the lanes.
# 2. Compile: each recorded program is compiled for a compile-only TPU
#    topology by libtpu in an amd64 container (libtpu is x86_64 only; on an
#    arm64 host Docker runs it under QEMU, which needs binfmt for x86_64).
#
# --programs DIR skips step 1 and compiles DIR.
# Environment:
#   SINGLE  topology for one-device programs [v6e:1x1]
#   MULTI   topology for multi-device programs [v6e:2x2]
# Results: build/tpu-aot/<time>/ (programs/, summary.tsv, hlo/, errors/,
# record.log, compile.log). Exits 1 when a program fails to compile.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SINGLE="${SINGLE:-v6e:1x1}"
MULTI="${MULTI:-v6e:2x2}"
OUT="$ROOT/build/tpu-aot/$(date -u +%Y%m%dT%H%M%SZ)"
PROGRAMS="$OUT/programs"
ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1; shift ;;
    --programs) PROGRAMS="$(cd "$2" && pwd)"; shift 2 ;;
    *) echo "usage: $0 [--all] [--programs DIR]" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"

JAX_VERSION=0.12.0.1
LIBTPU_VERSION="$(cat "$ROOT/scripts/tpu/libtpu-version")"
IMAGE="tlaloc-tpu-aot:jax-$JAX_VERSION-libtpu-${LIBTPU_VERSION//\*/x}"

if [[ "$PROGRAMS" == "$OUT/programs" ]]; then
  if [[ $ALL == 1 ]]; then
    TESTS=(--tests 'io.tlaloc.runtime.pjrt.*')
  else
    # The lanes of session.sh (g2b's tests run only on a TPU), the kernel
    # fixtures, and the multi-device programs on the XLA CPU plugin.
    TESTS=()
    for t in '*TpuProgramRecorderTest' '*TpuCustomCall*' '*TpuKernel*' '*Mosaic*' \
             '*PjrtCausalLmTrainingTest' '*PjrtQwen3GreedyParityTest' \
             '*PjrtRaggedDotTest' '*PjrtDeviceAdamWTest' '*PjrtShardedTrainingCpuTest' '*PjrtMultiDeviceCpuTest'; do
      TESTS+=(--tests "$t")
    done
  fi
  echo "== recording the session's programs into $PROGRAMS"
  # A failing test still records the programs it compiled; its result is in record.log.
  TLALOC_TPU_PROGRAMS_DIR="$PROGRAMS" "$ROOT/gradlew" -p "$ROOT" :runtime-pjrt:jvmTest --rerun --continue "${TESTS[@]}" \
    > "$OUT/record.log" 2>&1 || echo "   some tests failed while recording (see $OUT/record.log)"
  echo "   $(ls "$PROGRAMS"/*.mlir 2>/dev/null | wc -l) programs"
fi

echo "== image $IMAGE"
docker image inspect "$IMAGE" > /dev/null 2>&1 || docker build --platform linux/amd64 -q -t "$IMAGE" \
  --build-arg JAX_VERSION="$JAX_VERSION" --build-arg LIBTPU_VERSION="$LIBTPU_VERSION" \
  "$ROOT/scripts/tpu/aot" > /dev/null || { echo "image build failed"; exit 2; }

echo "== compiling for $SINGLE / $MULTI"
docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" \
  -v "$ROOT/scripts/tpu/aot:/aot:ro" -v "$PROGRAMS:/programs:ro" -v "$OUT:/out" \
  "$IMAGE" python -I /aot/tlaloc_tpu_aot.py /programs /out --single "$SINGLE" --multi "$MULTI" \
  2>&1 | grep -a -E '^(OK|FAIL|SKIP|[0-9]+ (programs|compiled))' | tee "$OUT/compile.log"
STATUS=${PIPESTATUS[0]}
echo "== results in $OUT"
exit "$STATUS"
