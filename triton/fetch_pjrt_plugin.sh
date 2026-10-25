#!/usr/bin/env bash
# Downloads the XLA CUDA 13 PJRT plugin (jax-cuda13-pjrt 0.12.0.1, PJRT C API
# 0.116, XLA FFI 0.5) from PyPI, checks its sha256, and unpacks the single .so
# the backend needs into triton/pjrt/xla_cuda13/, with the wheel's name in
# VERSION beside it; a plugin of another version there is replaced. The CUDA major version matches the
# Triton 25.11 container, so the plugin uses the container's own CUDA
# libraries and nothing else has to be mounted.
#
#   triton/fetch_pjrt_plugin.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/pjrt/xla_cuda13"
SO="$DEST/xla_cuda_plugin.so"
VERSION_FILE="$DEST/VERSION"

case "$(uname -m)" in
  aarch64)
    URL=https://files.pythonhosted.org/packages/72/95/7c070b7a55cb67ad1dd25c7bc58305cebb840a965bd1900c7f116926940a/jax_cuda13_pjrt-0.12.0.1-py3-none-manylinux_2_27_aarch64.whl
    SHA=40758d042df8c06991b6a1a8016b56e9cefb1ed4c677c062125c3eb71c3dab97
    ;;
  x86_64)
    URL=https://files.pythonhosted.org/packages/b9/7d/c2ad642908a4e8620c6c9f6eb8b18914cd66804a3c96fb53dc41c05583ab/jax_cuda13_pjrt-0.12.0.1-py3-none-manylinux_2_27_x86_64.whl
    SHA=5899bb6177eda86ea532afb9a26794f2bdd04966f6a01f9028266b16c2b6cf90
    ;;
  *)
    echo "no jax-cuda13-pjrt wheel for $(uname -m)" >&2
    exit 1
    ;;
esac

if [[ -f "$SO" && "$(cat "$VERSION_FILE" 2>/dev/null)" == "$(basename "$URL")" ]]; then
  echo "already present: $SO ($(basename "$URL"))"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
WHEEL="$TMP/$(basename "$URL")"
echo "downloading $(basename "$URL")"
curl -fsSL -o "$WHEEL" "$URL"
echo "$SHA  $WHEEL" | sha256sum -c -
mkdir -p "$DEST"
python3 - "$WHEEL" "$DEST" <<'PY'
import sys, zipfile
wheel, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as z:
    data = z.read("jax_plugins/xla_cuda13/xla_cuda_plugin.so")
with open(f"{dest}/xla_cuda_plugin.so", "wb") as f:
    f.write(data)
PY
basename "$URL" > "$VERSION_FILE"
echo "unpacked $SO"
