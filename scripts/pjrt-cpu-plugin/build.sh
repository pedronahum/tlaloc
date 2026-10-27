#!/usr/bin/env bash
# Builds XLA's CPU PJRT plugin (pjrt_c_api_cpu_plugin.so) from source, at the
# XLA commit jax 0.12.0.1 builds against (the same commit as the vendored PJRT
# headers, docs/vendoring.md). OpenXLA publishes no binary of this plugin.
#
# The plugin gives PjrtSession a multi-device client without a GPU or a TPU:
# PjrtTarget.LlvmCpu with `cpu_device_count` > 1 runs sharded and collective
# programs on N host devices.
#
# Usage: scripts/pjrt-cpu-plugin/build.sh [install-dir]
#   install-dir defaults to ~/.local/lib/tlaloc.
# Source checkout and Bazel output live under ~/.cache/tlaloc.
set -euo pipefail

XLA_COMMIT=d867969800d4a8fd045505eabc65e1e66f30794f
SRC="${TLALOC_XLA_SRC:-$HOME/.cache/tlaloc/xla-${XLA_COMMIT:0:8}}"
DEST="${1:-$HOME/.local/lib/tlaloc}"

if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/openxla/xla.git
  git -C "$SRC" fetch -q --depth 1 origin "$XLA_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$XLA_COMMIT"

cd "$SRC"
python3 configure.py --backend=CPU --host_compiler=GCC
bazelisk build -c opt --config=nonccl //xla/pjrt/c:pjrt_c_api_cpu_plugin.so 2>&1 \
  || bazelisk build -c opt //xla/pjrt/c:pjrt_c_api_cpu_plugin.so

mkdir -p "$DEST"
install -m 0555 bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so "$DEST/pjrt_c_api_cpu_plugin.so"
echo "$XLA_COMMIT" > "$DEST/pjrt_c_api_cpu_plugin.commit"
echo "installed $DEST/pjrt_c_api_cpu_plugin.so"
