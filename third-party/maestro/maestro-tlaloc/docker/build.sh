#!/usr/bin/env bash
# Build the tlaloc-runtime container image (Layer 2.5.5 §0.4.249+).
#
# Mirrors maestro-actus's docker/build.sh pattern. Run from the
# third-party/maestro/ root (or pass an absolute path).
#
# Usage:
#   bash maestro-tlaloc/docker/build.sh
#   bash maestro-tlaloc/docker/build.sh tlaloc-runtime:0.1.0
#   bash maestro-tlaloc/docker/build.sh registry.example.com/tlaloc-runtime:dev
#
# Reproducible: two consecutive runs against the same source tree should
# produce identical image digests (modulo build timestamps). To verify:
#   bash maestro-tlaloc/docker/build.sh && docker inspect tlaloc-runtime:latest --format='{{.Id}}'
#   # then re-run and compare.

set -euo pipefail

TAG="${1:-tlaloc-runtime:latest}"

# Locate vendored Maestro root: this script lives at
# third-party/maestro/maestro-tlaloc/docker/build.sh, so go up three dirs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAESTRO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${MAESTRO_ROOT}"

echo "==== Staging maestro-tlaloc jars + dependencies under build/docker/libs/ ===="
./gradlew --no-daemon --no-build-cache \
  :maestro-tlaloc:assemble \
  :maestro-tlaloc:buildDockerContext

echo
echo "==== Building tlaloc-runtime image as ${TAG} ===="
docker build \
  --tag "${TAG}" \
  --file maestro-tlaloc/docker/Dockerfile \
  .

echo
echo "==== Image built: ${TAG} ===="
docker images --filter "reference=${TAG}" --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}'
