#!/usr/bin/env bash
# Tlaloc Mac toolchain — Stage 2 (no sudo).
#
# Runs everything that doesn't require admin rights:
#   1. brew install of JDK 17, Python 3.11, bazelisk, cmake, ninja, etc.
#   2. Project-local Python venv at harness/python/.venv
#   3. pip install torch / jax / iree-base-{compiler,runtime}
#   4. Clone openxla/stablehlo + openxla/shardy into ~/tlaloc-toolchain
#   5. Bazelisk build of stablehlo-translate and sdy-opt (long: 20–60 min first time)
#   6. Symlink the four binaries into /opt/homebrew/bin (brew owns it; no sudo)
#   7. Smoke check
#
# Pre-requisite: scripts/setup-mac-bootstrap.sh has run successfully (so brew
# exists and Xcode CLT is installed).
#
# Idempotent. Probes the system at every step and skips anything already done.
# Phase flags let you re-run a subset.
#
# Usage:
#   bash scripts/setup-mac-userspace.sh
#   bash scripts/setup-mac-userspace.sh --skip-bazel
#   bash scripts/setup-mac-userspace.sh --only-python
#   bash scripts/setup-mac-userspace.sh --only-bazel
#   bash scripts/setup-mac-userspace.sh --help

set -euo pipefail

# ---------- Configuration ----------

JDK_FORMULA="openjdk@21"
PY_FORMULA="python@3.11"
BAZELISK_FORMULA="bazelisk"

# User-writable, no sudo needed.
TOOLCHAIN_ROOT="${HOME}/tlaloc-toolchain"

STABLEHLO_REPO="https://github.com/openxla/stablehlo.git"
SHARDY_REPO="https://github.com/openxla/shardy.git"
# Pinned to whatever JAX 0.10.0 bundles, so the stablehlo MLIR text emitted by
# the installed jaxlib 0.10.0 round-trips through our locally built
# stablehlo-translate / sdy-opt without dialect-version skew.
#
# Resolution chain (verified 2026-04-27):
#   jax-v0.10.0 (a33ed61) → third_party/xla pins XLA b6f37ab
#   XLA b6f37ab → third_party/stablehlo pins 3a8886de  (Apr 10 2026)
#   XLA b6f37ab → third_party/shardy    pins 22259c17  (Apr 10 2026,
#                                       "Integrate StableHLO at @3a8886de")
STABLEHLO_REF="3a8886de8515f859875df37578b5caf33f6e52f3"
SHARDY_REF="22259c179a3045a1dc37b1c1bb119e4e8670b66f"

# brew owns /opt/homebrew on Apple Silicon, so symlinking here needs no sudo.
BIN_INSTALL_DIR="/opt/homebrew/bin"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/harness/python/.venv"

PIP_PACKAGES=(
  "numpy"
  "torch>=2.1"
  "jax"
  "jaxlib"
  "iree-base-compiler"
  "iree-base-runtime"
)

DO_BREW=1
DO_PYTHON=1
DO_BAZEL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-brew) DO_BREW=0 ;;
    --skip-python) DO_PYTHON=0 ;;
    --skip-bazel) DO_BAZEL=0 ;;
    --only-brew) DO_BREW=1; DO_PYTHON=0; DO_BAZEL=0 ;;
    --only-python) DO_BREW=0; DO_PYTHON=1; DO_BAZEL=0 ;;
    --only-bazel) DO_BREW=0; DO_PYTHON=0; DO_BAZEL=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

banner() { echo; echo "==== $* ===="; }
have() { command -v "$1" >/dev/null 2>&1; }

ensure_path_line() {
  local line="$1"
  local rc="${HOME}/.zshrc"
  touch "$rc"
  if ! grep -Fqx "$line" "$rc"; then
    echo "$line" >> "$rc"
    echo "  appended to ${rc}: $line"
  fi
}

# ---------- Pre-check ----------

# Load brew env first — non-interactive shells (Claude Code's harness, CI,
# etc.) don't source ~/.zshrc, so `brew` won't be on PATH even if installed.
if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

if ! have brew; then
  echo "ERROR: brew not found on PATH and /opt/homebrew/bin/brew does not exist." >&2
  echo "  Run scripts/setup-mac-bootstrap.sh first (it requires sudo)." >&2
  exit 1
fi

# ---------- Phase 1: brew packages ----------

phase_brew() {
  banner "brew packages (no sudo)"

  local pkgs=(
    "$JDK_FORMULA"
    "$PY_FORMULA"
    "$BAZELISK_FORMULA"
    git
    cmake
    ninja
    wget
    coreutils
    ccache
  )

  # Skip ones already installed to keep output legible.
  local missing=()
  for p in "${pkgs[@]}"; do
    if brew list --versions "$p" >/dev/null 2>&1; then
      echo "  ✓ $p"
    else
      missing+=("$p")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "  installing: ${missing[*]}"
    brew install "${missing[@]}"
  fi

  # JAVA_HOME for shells started after this script.
  ensure_path_line 'export JAVA_HOME="$(/usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home)"'
  ensure_path_line 'export PATH="$JAVA_HOME/bin:$PATH"'

  echo
  echo "  versions:"
  /opt/homebrew/opt/${JDK_FORMULA}/bin/java -version 2>&1 | sed 's/^/    /' || true
  /opt/homebrew/bin/python3.11 --version 2>&1 | sed 's/^/    /' || true
  /opt/homebrew/bin/bazelisk version 2>/dev/null | head -1 | sed 's/^/    /' || true
}

# ---------- Phase 2: Python venv + wheels ----------

phase_python() {
  banner "Python venv + wheels (PyTorch / JAX / IREE)"

  local py="/opt/homebrew/bin/python3.11"
  if [[ ! -x "$py" ]]; then
    py="$(command -v python3.11 || command -v python3)"
  fi
  echo "  interpreter: $py"

  if [[ ! -d "$VENV_DIR" ]]; then
    "$py" -m venv "$VENV_DIR"
    echo "  created venv at $VENV_DIR"
  else
    echo "  reusing existing venv at $VENV_DIR"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip wheel setuptools
  pip install --upgrade "${PIP_PACKAGES[@]}"
  deactivate

  # iree-base-compiler ships `iree-compile`; surface it on PATH for the JVM
  # tests that resolve via `which`.
  if [[ -x "${VENV_DIR}/bin/iree-compile" ]]; then
    ln -sfn "${VENV_DIR}/bin/iree-compile" "${BIN_INSTALL_DIR}/iree-compile"
    echo "  symlinked iree-compile -> ${BIN_INSTALL_DIR}/iree-compile"
  fi

  echo
  echo "  installed:"
  "${VENV_DIR}/bin/pip" list 2>/dev/null \
    | grep -Ei '^(torch|jax|jaxlib|numpy|iree)' \
    | sed 's/^/    /'
}

# ---------- Phase 3: stablehlo + shardy from source ----------

phase_bazel() {
  banner "StableHLO + Shardy source builds (long: 20–60 min first time)"

  if ! have bazelisk; then
    echo "  ERROR: bazelisk missing. Re-run with --skip-bazel or install brew packages first." >&2
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
      ln -sfn "$built" "${BIN_INSTALL_DIR}/stablehlo-translate"
      echo "  symlinked stablehlo-translate -> ${BIN_INSTALL_DIR}/stablehlo-translate"
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
      ln -sfn "$built" "${BIN_INSTALL_DIR}/sdy-opt"
      echo "  symlinked sdy-opt -> ${BIN_INSTALL_DIR}/sdy-opt"
    else
      echo "  WARNING: $built not found; bazel target name may have changed upstream." >&2
    fi
  popd >/dev/null
}

# ---------- Phase 4: smoke check ----------

phase_smoke() {
  banner "Smoke check"
  local ok=1
  for tool in brew java javac python3.11 bazelisk stablehlo-translate sdy-opt iree-compile; do
    if have "$tool"; then
      printf "  %-22s %s\n" "$tool" "$(command -v "$tool")"
    else
      printf "  %-22s MISSING\n" "$tool"
      ok=0
    fi
  done
  echo
  if [[ "$ok" == "1" ]]; then
    echo "  Ready. Try:"
    echo "    cd ${PROJECT_ROOT}"
    echo "    ./gradlew test"
    echo "    source ${VENV_DIR}/bin/activate && python harness/python/run_pytorch.py"
  else
    echo "  some tools missing — review phase output above."
    exit 1
  fi
}

# ---------- Main ----------

[[ "$DO_BREW"   == "1" ]] && phase_brew
[[ "$DO_PYTHON" == "1" ]] && phase_python
[[ "$DO_BAZEL"  == "1" ]] && phase_bazel
phase_smoke
