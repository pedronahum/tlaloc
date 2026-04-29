#!/usr/bin/env bash
# Tlaloc Mac toolchain — Stage 1 (admin / sudo).
#
# YOU must run this. It will prompt for your admin password (Xcode CLT
# installer + Homebrew installer + one optional JDK symlink). Claude cannot
# run it because the harness does not pass sudo prompts through.
#
# After this finishes successfully, run scripts/setup-mac-userspace.sh —
# that one is sudo-free and Claude can run it on your behalf.
#
# Usage:
#   bash scripts/setup-mac-bootstrap.sh
#   bash scripts/setup-mac-bootstrap.sh --skip-jdk-symlink   # skip the only optional sudo step
#
# What this script does (and ONLY this):
#   1. Installs Xcode Command Line Tools if missing.
#   2. Installs Homebrew if missing.
#   3. Symlinks the brew openjdk@21 keg into /Library/Java/JavaVirtualMachines/
#      so /usr/libexec/java_home can find it. Optional — if skipped, the
#      userspace script falls back to setting JAVA_HOME directly.
#
# What this script deliberately does NOT do:
#   - brew install <packages>   (sudo-free; userspace script handles it)
#   - pip installs              (sudo-free; userspace script handles it)
#   - cloning + Bazel builds    (sudo-free; userspace script handles it)
#   - symlinks into /opt/homebrew/bin  (brew owns this dir as $USER, no sudo)

set -euo pipefail

JDK_FORMULA="openjdk@21"
DO_JDK_SYMLINK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-jdk-symlink) DO_JDK_SYMLINK=0 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

banner() { echo; echo "==== $* ===="; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------- Sanity ----------
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS only" >&2; exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "warning: not Apple Silicon — brew prefix differs on Intel" >&2
fi

# ---------- 1. Xcode CLT ----------
banner "Xcode Command Line Tools"
if xcode-select -p >/dev/null 2>&1; then
  echo "  already installed: $(xcode-select -p)"
else
  echo "  triggering CLT install (a GUI dialog will open; admin password required)..."
  xcode-select --install || true
  echo
  echo "  >> Wait for the GUI installer to finish, then re-run this script. Exiting."
  exit 0
fi

# ---------- 2. Homebrew ----------
banner "Homebrew"
if have brew; then
  echo "  already installed at $(brew --prefix)"
else
  echo "  running official Homebrew installer (will prompt for admin password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Make brew visible to the rest of THIS shell (and any future shell).
if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
  rc="${HOME}/.zshrc"
  touch "$rc"
  line='eval "$(/opt/homebrew/bin/brew shellenv)"'
  if ! grep -Fqx "$line" "$rc"; then
    echo "$line" >> "$rc"
    echo "  appended brew shellenv line to ${rc}"
  fi
fi

# ---------- 3. Optional JDK system symlink ----------
banner "JDK system registration (optional, sudo)"
if [[ "$DO_JDK_SYMLINK" != "1" ]]; then
  echo "  skipped (--skip-jdk-symlink). Userspace script will export JAVA_HOME directly."
else
  # We can only do this once openjdk@21 is actually installed via brew —
  # which the userspace script handles. So run it conditionally: if the keg
  # exists now, symlink; otherwise note that re-running this script after
  # the userspace script will pick it up.
  if [[ -d "/opt/homebrew/opt/${JDK_FORMULA}/libexec/openjdk.jdk" ]]; then
    sudo ln -sfn "/opt/homebrew/opt/${JDK_FORMULA}/libexec/openjdk.jdk" \
      "/Library/Java/JavaVirtualMachines/${JDK_FORMULA}.jdk"
    echo "  symlinked openjdk.jdk into /Library/Java/JavaVirtualMachines/"
  else
    echo "  ${JDK_FORMULA} not installed yet — userspace script will brew install it."
    echo "  After that, re-run this script (or pass --skip-jdk-symlink) to register the JDK."
  fi
fi

banner "Bootstrap done"
echo "  Next: bash scripts/setup-mac-userspace.sh"
echo "        (Claude can run that one — no sudo needed.)"
