#!/usr/bin/env bash
# Tlaloc NVIDIA DGX Spark toolchain — Stage 1 (admin / sudo).
#
# Mirrors scripts/setup-mac-bootstrap.sh, but for the DGX Spark
# (formerly "Project Digits" — NVIDIA's GB10 Grace-Blackwell workstation
# running DGX OS, an Ubuntu LTS derivative on ARM64 / aarch64).
#
# YOU must run this. It will prompt for your admin password (apt
# installs + the JDK alternatives system-link). The Tlaloc agent harness
# does not pass sudo prompts through.
#
# After this finishes successfully, run scripts/setup-dgx-spark-userspace.sh —
# that one is sudo-free and Claude can run it on your behalf.
#
# Usage:
#   bash scripts/setup-dgx-spark-bootstrap.sh
#   bash scripts/setup-dgx-spark-bootstrap.sh --skip-jdk-default   # skip the
#                                                                   # `update-alternatives` step
#   bash scripts/setup-dgx-spark-bootstrap.sh --help
#
# What this script does (and ONLY this):
#   1. Refuses to run if not on aarch64 / ARM64 Ubuntu (DGX Spark is ARM).
#   2. Refuses to run if `nvidia-smi` is missing (the GPU drivers should
#      already be installed on a DGX OS image; if they aren't, fix that
#      first via `nvidia-driver` or DGX OS bring-up).
#   3. Runs `apt-get update` + installs build essentials, OpenJDK 21,
#      Python 3.11, git, cmake, ninja, ccache, wget, curl.
#   4. Optionally registers OpenJDK 21 as the system default JDK via
#      `update-alternatives` (skip with --skip-jdk-default).
#
# What this script deliberately does NOT do:
#   - Install or upgrade NVIDIA drivers / CUDA — DGX OS ships with both.
#     If `nvidia-smi` doesn't work, run `sudo nvidia-driver --install` or
#     consult NVIDIA's DGX OS documentation BEFORE running this.
#   - pip installs                                (sudo-free; userspace script handles it)
#   - bazelisk download                           (sudo-free; userspace script handles it)
#   - clone + Bazel builds                        (sudo-free; userspace script handles it)

set -euo pipefail

JDK_PACKAGE="openjdk-21-jdk"
PY_PACKAGE="python3.11"
DO_JDK_DEFAULT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-jdk-default) DO_JDK_DEFAULT=0 ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

banner() { echo; echo "==== $* ===="; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------- Sanity ----------

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "DGX Spark uses Linux (DGX OS / Ubuntu) — this script targets that." >&2
  echo "On macOS use scripts/setup-mac-bootstrap.sh instead." >&2
  exit 1
fi

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  echo "warning: not on aarch64 — DGX Spark uses the GB10 Grace ARM CPU." >&2
  echo "  This script will continue but the toolchain pins (jaxlib aarch64+cuda" >&2
  echo "  wheels, bazelisk arm64 binary) won't apply on x86." >&2
  echo "  Use scripts/setup-mac-bootstrap.sh on Apple Silicon or write a generic" >&2
  echo "  Linux x86 setup if that's where you are." >&2
fi

if [[ ! -f /etc/os-release ]]; then
  echo "ERROR: /etc/os-release missing; can't identify distro." >&2
  exit 1
fi
# shellcheck source=/dev/null
. /etc/os-release
case "${ID:-}" in
  ubuntu|debian|nvidia-dgx)
    echo "Detected: ${PRETTY_NAME:-${ID}}"
    ;;
  *)
    echo "warning: distro is '${ID}' (expected ubuntu / debian / nvidia-dgx). Proceeding anyway." >&2
    ;;
esac

# DGX OS sanity: nvidia-smi should already work. If it doesn't, the
# user's GPU drivers aren't right and the rest of this setup is moot.
banner "NVIDIA driver check"
if have nvidia-smi; then
  echo "  nvidia-smi:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 \
    | sed 's/^/    /' || true
else
  echo "  ERROR: nvidia-smi not found." >&2
  echo "  DGX OS installs NVIDIA drivers + CUDA out of the box. If you're seeing" >&2
  echo "  this message, the driver install hasn't completed. Fix that first." >&2
  exit 1
fi

# ---------- 1. apt update + base packages ----------

banner "apt-get update"
sudo apt-get update -y

banner "apt-get install: build essentials, JDK 21, Python 3.11, Bazel deps"

PACKAGES=(
  build-essential
  "$JDK_PACKAGE"
  "$PY_PACKAGE"
  python3.11-venv
  python3.11-dev
  python3-pip
  git
  cmake
  ninja-build
  ccache
  wget
  curl
  unzip
  pkg-config
  zlib1g-dev
)

# `python3.11` may not exist on Ubuntu 22.04 base; deadsnakes PPA or the
# DGX OS `python3` (3.10/3.12) may be the available version. Probe and
# fall back to the system python3 if 3.11 isn't a known package.
if ! apt-cache show "$PY_PACKAGE" >/dev/null 2>&1; then
  echo "  note: $PY_PACKAGE not found in apt sources — falling back to python3"
  PACKAGES=("${PACKAGES[@]/$PY_PACKAGE/python3}")
  PACKAGES=("${PACKAGES[@]/python3.11-venv/python3-venv}")
  PACKAGES=("${PACKAGES[@]/python3.11-dev/python3-dev}")
fi

# Install. apt-get takes a single combined transaction; we don't bother
# probing per-package the way Mac brew does (apt is fast on aarch64).
sudo apt-get install -y "${PACKAGES[@]}"

# ---------- 2. JDK alternatives (optional sudo step) ----------

banner "JDK alternatives registration (optional, sudo)"
if [[ "$DO_JDK_DEFAULT" != "1" ]]; then
  echo "  skipped (--skip-jdk-default). Userspace script will export JAVA_HOME directly."
else
  # Look for the OpenJDK 21 install path. Ubuntu places it under
  # /usr/lib/jvm/java-21-openjdk-{arch}.
  JDK_HOME="$(find /usr/lib/jvm -maxdepth 1 -type d -name 'java-21-openjdk*' | head -1 || true)"
  if [[ -n "$JDK_HOME" && -x "$JDK_HOME/bin/java" ]]; then
    sudo update-alternatives --set java "${JDK_HOME}/bin/java" 2>/dev/null \
      || sudo update-alternatives --install /usr/bin/java java "${JDK_HOME}/bin/java" 2100
    sudo update-alternatives --set javac "${JDK_HOME}/bin/javac" 2>/dev/null \
      || sudo update-alternatives --install /usr/bin/javac javac "${JDK_HOME}/bin/javac" 2100
    echo "  set system default java/javac to JDK 21 ($JDK_HOME)"
  else
    echo "  WARNING: couldn't find java-21-openjdk install dir under /usr/lib/jvm/" >&2
    echo "  The userspace script will set JAVA_HOME directly via the apt install." >&2
  fi
fi

# ---------- 3. Versions ----------

banner "Versions"
java -version 2>&1 | sed 's/^/  /' || true
javac -version 2>&1 | sed 's/^/  /' || true
"$(command -v python3.11 || command -v python3)" --version 2>&1 | sed 's/^/  /' || true
git --version 2>&1 | sed 's/^/  /' || true
cmake --version 2>&1 | head -1 | sed 's/^/  /' || true

banner "Bootstrap done"
echo "  Next: bash scripts/setup-dgx-spark-userspace.sh"
echo "        (Claude can run that one — no sudo needed.)"
