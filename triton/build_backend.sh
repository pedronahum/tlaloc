#!/usr/bin/env bash
# Builds libtriton_tlaloc.so inside the Triton server container, so that it is
# compiled with the same compiler, glibc and libtritonserver.so it will load
# into. Output: triton/backends/tlaloc/libtriton_tlaloc.so and the device test
# triton/build/tests/pjrt_device_test, which verify.sh runs (neither is
# committed).
#
#   triton/build_backend.sh
#
# Environment:
#   TRITON_IMAGE   container image (default nvcr.io/nvidia/tritonserver:25.11-py3)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.11-py3}"
OUT="$HERE/backends/tlaloc"
TESTS="$HERE/build/tests"
mkdir -p "$OUT" "$TESTS"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$HERE:/src:ro" \
  -v "$OUT:/out" \
  -v "$TESTS:/tests" \
  "$IMAGE" \
  bash -c '
set -euo pipefail
TP=/src/third_party
INCLUDES="-I/src/backend -I$TP/core/include -I$TP/common/include -I$TP/backend/include -I$TP/rapidjson/include -I$TP/xla -I/usr/local/cuda/include"
FLAGS="-std=c++17 -O2 -fPIC -Wall -Wno-unused-function -DTRITON_ENABLE_GPU"

echo "== stablehlo_text_test"
g++ $FLAGS /src/backend/test/stablehlo_text_test.cc /src/backend/stablehlo_text.cc -o /tmp/stablehlo_text_test
/tmp/stablehlo_text_test

echo "== pjrt_device_test (run by verify.sh, which has a GPU)"
g++ $FLAGS -I/src/backend -I$TP/xla -I/usr/local/cuda/include \
  /src/backend/test/pjrt_device_test.cc /src/backend/pjrt_runtime.cc /src/backend/stablehlo_text.cc \
  -L/usr/local/cuda/lib64 -lcudart -ldl -o /tests/pjrt_device_test

echo "== libtriton_tlaloc.so"
g++ $FLAGS -shared $INCLUDES \
  /src/backend/tlaloc_backend.cc \
  /src/backend/pjrt_runtime.cc \
  /src/backend/sequence_mode.cc \
  /src/backend/stablehlo_text.cc \
  $TP/backend/src/backend_common.cc \
  $TP/backend/src/backend_model.cc \
  $TP/backend/src/backend_model_instance.cc \
  -L/opt/tritonserver/lib -ltritonserver \
  -L/usr/local/cuda/lib64 -lcudart \
  -ldl \
  -Wl,--version-script=/src/backend/libtriton_tlaloc.ldscript \
  -Wl,--no-undefined \
  -o /out/libtriton_tlaloc.so

echo "== exported symbols"
nm -D --defined-only /out/libtriton_tlaloc.so | awk "{print \$3}" | sort
'
echo "built $OUT/libtriton_tlaloc.so"
