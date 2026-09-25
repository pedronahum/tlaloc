#!/usr/bin/env bash
# Downloads the XLA CUDA 13 PJRT plugin (jax-cuda13-pjrt 0.10.0, PJRT C API
# 0.104) from PyPI, checks its sha256, and unpacks the single .so the backend
# needs into triton/pjrt/xla_cuda13/. The CUDA major version matches the
# Triton 25.11 container, so the plugin uses the container's own CUDA
# libraries and nothing else has to be mounted.
#
#   triton/fetch_pjrt_plugin.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/pjrt/xla_cuda13"
SO="$DEST/xla_cuda_plugin.so"

case "$(uname -m)" in
  aarch64)
    URL=https://files.pythonhosted.org/packages/8a/7b/222e957b28c38b6ae37cbcf0e397c9f2b5be0c08236ddf72c545d185b3f5/jax_cuda13_pjrt-0.10.0-py3-none-manylinux_2_27_aarch64.whl
    SHA=18f8dcd3b18778f5174bc01313c214c5cb8bfb3fb1c3d01344795d7424fe2b51
    ;;
  x86_64)
    URL=https://files.pythonhosted.org/packages/21/98/77f15d81fd0637da454e453c8456d4a2b5c8b2e66823b4237ee8689152cf/jax_cuda13_pjrt-0.10.0-py3-none-manylinux_2_27_x86_64.whl
    SHA=848d6ae3e663d040c53e902ea9d380a902bfa5e7da881053cec408360036fa7a
    ;;
  *)
    echo "no jax-cuda13-pjrt wheel for $(uname -m)" >&2
    exit 1
    ;;
esac

if [[ -f "$SO" ]]; then
  echo "already present: $SO"
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
echo "unpacked $SO"
