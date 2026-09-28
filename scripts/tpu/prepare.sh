#!/usr/bin/env bash
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
#
# Builds everything a Cloud TPU VM session runs, on this machine, into one
# tarball, so the VM needs a JDK and libtpu and nothing else.
#
#   scripts/tpu/prepare.sh            # writes build/tpu-session.tar.gz
#
# Contents (tpu-session/):
#   device-tests/   the runtime-pjrt device tests with their classpath and
#                   run-tests.sh (./gradlew :runtime-pjrt:tpuBundle)
#   fine-tune/      examples/fine-tune as an installed application
#   session.sh      the script to run on the VM (scripts/tpu/session.sh)
#   harness/        the JAX baseline for the Mosaic kernels (session.sh jax)
#   MANIFEST        the commit, the build time, and the libtpu version to install
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/build/tpu-session"
cd "$ROOT"

echo "== publishing Tlaloc to mavenLocal (the fine-tune example resolves it there)"
./gradlew -q publishToMavenLocal -x test
echo "== device-test bundle"
./gradlew -q :runtime-pjrt:tpuBundle
echo "== fine-tune example"
./gradlew -q -p examples/fine-tune installDist

rm -rf "$OUT" && mkdir -p "$OUT"
cp -r runtime-pjrt/build/device-test-bundle "$OUT/device-tests"
cp -r examples/fine-tune/build/install/tlaloc-fine-tune "$OUT/fine-tune"
cp scripts/tpu/session.sh "$OUT/session.sh"
mkdir -p "$OUT/harness" && cp harness/python/run_tpu_kernels_jax.py harness/python/export_tpu_kernels.py "$OUT/harness/"
{
  echo "commit $(git rev-parse HEAD)$(git diff --quiet HEAD || echo ' (with uncommitted changes)')"
  echo "built $(date -u +%FT%TZ) on $(uname -m)"
  [[ -f scripts/tpu/libtpu-version ]] && echo "libtpu $(cat scripts/tpu/libtpu-version)"
} > "$OUT/MANIFEST"
tar -C "$ROOT/build" -czf "$ROOT/build/tpu-session.tar.gz" tpu-session
cat "$OUT/MANIFEST"
echo "== wrote build/tpu-session.tar.gz ($(du -h build/tpu-session.tar.gz | cut -f1))"
echo "   gcloud compute tpus tpu-vm scp build/tpu-session.tar.gz <vm>:~ --zone=<zone>"
echo "   then on the VM: tar xzf tpu-session.tar.gz && tpu-session/session.sh"
