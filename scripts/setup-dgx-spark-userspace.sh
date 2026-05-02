#!/usr/bin/env bash
# Tlaloc NVIDIA DGX Spark toolchain — Stage 2 (no sudo).
#
# Mirrors scripts/setup-mac-userspace.sh but for the DGX Spark
# (aarch64 / Grace-Blackwell, DGX OS / Ubuntu LTS).
#
# Runs everything that doesn't require admin rights:
#   1. Download bazelisk (linux-arm64 binary) into ~/.local/bin.
#   2. Project-local Python venv at harness/python/.venv.
#   3. pip install of jaxlib + iree-base-{compiler,runtime} (and torch
#      via NVIDIA's PyPI index when a CUDA wheel is required).
#   4. Clone openxla/stablehlo + openxla/shardy into ~/tlaloc-toolchain.
#   5. Bazelisk build of stablehlo-translate and sdy-opt
#      (~30–90 min first time on Grace; faster on warm cache).
#   6. Symlink the binaries into ~/.local/bin.
#   7. CUDA + smoke check.
#
# Pre-requisite: scripts/setup-dgx-spark-bootstrap.sh has run successfully
# (so JDK 21, build-essential, python3.x are installed).
#
# Idempotent. Probes the system at every step and skips anything already
# done. Phase flags let you re-run a subset.
#
# Usage:
#   bash scripts/setup-dgx-spark-userspace.sh
#   bash scripts/setup-dgx-spark-userspace.sh --skip-bazel
#   bash scripts/setup-dgx-spark-userspace.sh --only-python
#   bash scripts/setup-dgx-spark-userspace.sh --only-bazel
#   bash scripts/setup-dgx-spark-userspace.sh --skip-torch    # skip torch install
#                                                              # (use system PyTorch via
#                                                              #  --system-site-packages)
#   bash scripts/setup-dgx-spark-userspace.sh --help

set -euo pipefail

# ---------- Configuration ----------

# User-writable, no sudo needed.
TOOLCHAIN_ROOT="${HOME}/tlaloc-toolchain"
LOCAL_BIN="${HOME}/.local/bin"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/harness/python/.venv"

STABLEHLO_REPO="https://github.com/openxla/stablehlo.git"
SHARDY_REPO="https://github.com/openxla/shardy.git"
# Pinned to JAX 0.10.0's bundled commits — same pins as the Mac script,
# so emitted MLIR text round-trips through both Apple-Silicon and
# DGX-Spark builds without dialect-version skew.
STABLEHLO_REF="3a8886de8515f859875df37578b5caf33f6e52f3"
SHARDY_REF="22259c179a3045a1dc37b1c1bb119e4e8670b66f"

# Bazelisk linux-arm64 release. Pin a known-good version; bumps are deliberate.
BAZELISK_VERSION="v1.20.0"
BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-arm64"

# Python wheels. On aarch64 + CUDA, jaxlib needs the cuda-12-pjrt-plugin
# variant. PyTorch ships ARM64+CUDA wheels via NVIDIA's index. The
# `iree-base-*` wheels publish aarch64 builds on PyPI directly.
PIP_PACKAGES_BASE=(
  "numpy"
  "jax"
  "jaxlib"
  "iree-base-compiler"
  "iree-base-runtime"
)

# Optional torch — skipped when --skip-torch (assume DGX OS has it
# pre-installed via the system Python and the venv inherits via
# --system-site-packages).
PIP_PACKAGES_TORCH=("torch>=2.4")

DO_BAZELISK=1
DO_PYTHON=1
DO_TORCH=1
DO_BAZEL=1
USE_SYSTEM_SITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-bazelisk) DO_BAZELISK=0 ;;
    --skip-python) DO_PYTHON=0 ;;
    --skip-torch) DO_TORCH=0; USE_SYSTEM_SITE=1 ;;
    --skip-bazel) DO_BAZEL=0 ;;
    --only-bazelisk) DO_BAZELISK=1; DO_PYTHON=0; DO_BAZEL=0 ;;
    --only-python) DO_BAZELISK=0; DO_PYTHON=1; DO_BAZEL=0 ;;
    --only-bazel) DO_BAZELISK=0; DO_PYTHON=0; DO_BAZEL=1 ;;
    --use-system-site) USE_SYSTEM_SITE=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

banner() { echo; echo "==== $* ===="; }
have() { command -v "$1" >/dev/null 2>&1; }

ensure_path_line() {
  local line="$1"
  local rc="${HOME}/.bashrc"
  touch "$rc"
  if ! grep -Fqx "$line" "$rc"; then
    echo "$line" >> "$rc"
    echo "  appended to ${rc}: $line"
  fi
}

# ---------- Pre-check ----------

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "DGX Spark uses Linux — this script targets that." >&2
  exit 1
fi

if ! have java; then
  echo "ERROR: java not found on PATH." >&2
  echo "  Run scripts/setup-dgx-spark-bootstrap.sh first (it requires sudo)." >&2
  exit 1
fi

mkdir -p "$LOCAL_BIN"
ensure_path_line 'export PATH="$HOME/.local/bin:$PATH"'
# Make ~/.local/bin visible to THIS shell too.
export PATH="$HOME/.local/bin:$PATH"

# ---------- Phase 1: bazelisk ----------

phase_bazelisk() {
  banner "Bazelisk download (linux-arm64)"

  local target="${LOCAL_BIN}/bazelisk"
  if [[ -x "$target" ]]; then
    local existing
    existing="$("$target" version 2>/dev/null | head -1 || true)"
    echo "  ✓ bazelisk already at $target ($existing)"
    return
  fi

  echo "  downloading $BAZELISK_URL"
  curl -fSL --retry 3 -o "$target" "$BAZELISK_URL"
  chmod +x "$target"
  echo "  installed: $("$target" version 2>/dev/null | head -1 || echo 'bazelisk')"

  # Also expose as `bazel` for build scripts that look for that name.
  if [[ ! -e "${LOCAL_BIN}/bazel" ]]; then
    ln -sfn "$target" "${LOCAL_BIN}/bazel"
    echo "  symlinked bazelisk -> ${LOCAL_BIN}/bazel"
  fi
}

# ---------- Phase 2: Python venv + wheels ----------

phase_python() {
  banner "Python venv + wheels (JAX / IREE / [optional] PyTorch)"

  local py
  py="$(command -v python3.11 || command -v python3 || true)"
  if [[ -z "$py" ]]; then
    echo "  ERROR: no python3 on PATH. Re-run scripts/setup-dgx-spark-bootstrap.sh." >&2
    return 1
  fi
  echo "  interpreter: $py ($("$py" --version 2>&1))"

  if [[ ! -d "$VENV_DIR" ]]; then
    if [[ "$USE_SYSTEM_SITE" == "1" ]]; then
      "$py" -m venv --system-site-packages "$VENV_DIR"
      echo "  created venv at $VENV_DIR (with --system-site-packages — picks up DGX-OS PyTorch)"
    else
      "$py" -m venv "$VENV_DIR"
      echo "  created venv at $VENV_DIR"
    fi
  else
    echo "  reusing existing venv at $VENV_DIR"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip wheel setuptools

  pip install --upgrade "${PIP_PACKAGES_BASE[@]}"

  if [[ "$DO_TORCH" == "1" ]]; then
    # NVIDIA publishes aarch64+CUDA PyTorch wheels at pypi.nvidia.com for
    # DGX Spark / Grace systems. PyPI's torch is also a valid option;
    # try PyPI first and fall back to NVIDIA's index if the version
    # there doesn't include CUDA support for sm_100 (Blackwell).
    if ! pip install --upgrade "${PIP_PACKAGES_TORCH[@]}"; then
      echo "  PyPI torch install failed; trying NVIDIA index for aarch64+CUDA..."
      pip install --upgrade --extra-index-url https://pypi.nvidia.com "${PIP_PACKAGES_TORCH[@]}"
    fi
  else
    echo "  torch skipped (--skip-torch). The DGX OS system Python's torch will" \
         "be picked up via --system-site-packages if you passed that."
  fi

  # iree-base-compiler ships `iree-compile`; surface it on PATH.
  if [[ -x "${VENV_DIR}/bin/iree-compile" ]]; then
    ln -sfn "${VENV_DIR}/bin/iree-compile" "${LOCAL_BIN}/iree-compile"
    echo "  symlinked iree-compile -> ${LOCAL_BIN}/iree-compile"
  fi

  deactivate

  echo
  echo "  installed:"
  "${VENV_DIR}/bin/pip" list 2>/dev/null \
    | grep -Ei '^(torch|jax|jaxlib|numpy|iree)' \
    | sed 's/^/    /' || true
}

# ---------- Phase 3: stablehlo + shardy from source ----------

phase_bazel() {
  banner "StableHLO + Shardy source builds (long: 30–90 min first time on Grace)"

  if ! have bazelisk && [[ ! -x "${LOCAL_BIN}/bazelisk" ]]; then
    echo "  ERROR: bazelisk missing. Re-run with --skip-bazel or run --only-bazelisk first." >&2
    return 1
  fi

  mkdir -p "$TOOLCHAIN_ROOT"

  build_stablehlo
  build_shardy
}

build_stablehlo() {
  echo
  echo "  --- stablehlo ---"
  local dir="${TOOLCHAIN_ROOT}/stablehlo"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$STABLEHLO_REPO" "$dir"
  else
    git -C "$dir" fetch --tags --prune
  fi
  if [[ -n "$STABLEHLO_REF" ]]; then
    git -C "$dir" checkout "$STABLEHLO_REF"
  fi

  pushd "$dir" >/dev/null
    bazelisk build -c opt //:stablehlo-translate
    local built="$dir/bazel-bin/stablehlo-translate"
    if [[ -x "$built" ]]; then
      ln -sfn "$built" "${LOCAL_BIN}/stablehlo-translate"
      echo "  symlinked stablehlo-translate -> ${LOCAL_BIN}/stablehlo-translate"
    else
      echo "  WARNING: $built not found; bazel target name may have changed upstream." >&2
    fi
  popd >/dev/null
}

build_shardy() {
  echo
  echo "  --- shardy ---"
  local dir="${TOOLCHAIN_ROOT}/shardy"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$SHARDY_REPO" "$dir"
  else
    git -C "$dir" fetch --tags --prune
  fi
  if [[ -n "$SHARDY_REF" ]]; then
    git -C "$dir" checkout "$SHARDY_REF"
  fi

  pushd "$dir" >/dev/null
    bazelisk build -c opt //shardy/tools:sdy_opt
    local built="$dir/bazel-bin/shardy/tools/sdy_opt"
    if [[ -x "$built" ]]; then
      ln -sfn "$built" "${LOCAL_BIN}/sdy-opt"
      echo "  symlinked sdy-opt -> ${LOCAL_BIN}/sdy-opt"
    else
      echo "  WARNING: $built not found; bazel target name may have changed upstream." >&2
    fi
  popd >/dev/null
}

# ---------- Phase 4: smoke check ----------

phase_smoke() {
  banner "Smoke check"
  local ok=1
  for tool in java javac git python3 bazelisk stablehlo-translate sdy-opt iree-compile nvidia-smi; do
    if have "$tool"; then
      printf "  %-22s %s\n" "$tool" "$(command -v "$tool")"
    else
      printf "  %-22s MISSING\n" "$tool"
      ok=0
    fi
  done

  echo
  echo "  GPU info:"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader 2>&1 \
    | sed 's/^/    /' || true

  echo
  if [[ "$ok" == "1" ]]; then
    echo "  Ready. Try:"
    echo "    cd ${PROJECT_ROOT}"
    echo "    ./gradlew test"
    echo
    echo "  If you skipped torch and want to verify the system PyTorch:"
    echo "    source ${VENV_DIR}/bin/activate && python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
    echo
    echo "  Layer 3 example (smoke-tests recognize → coarsen → kernel matrix):"
    echo "    ./gradlew :ir:jvmTest --tests \"io.tlaloc.ir.recognizer.*\""
  else
    echo "  some tools missing — review phase output above."
    exit 1
  fi
}

# ---------- Main ----------

[[ "$DO_BAZELISK" == "1" ]] && phase_bazelisk
[[ "$DO_PYTHON"   == "1" ]] && phase_python
[[ "$DO_BAZEL"    == "1" ]] && phase_bazel
phase_smoke
