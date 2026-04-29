#!/usr/bin/env bash
# Install JDK 21 — the unified Tlaloc + vendored Maestro toolchain.
#
# As of §0.4.244, Tlaloc and vendored Maestro both target JDK 21
# (`JvmTarget.JVM_21` in every module's build.gradle.kts;
# `JavaLanguageVersion.of(21)` in third-party/maestro/build.gradle).
# Single JDK simplifies the dev experience compared to the §0.4.243-era
# split (Tlaloc 17 + Maestro 21).
#
# Most users get JDK 21 via the standard bootstrap path
# (scripts/setup-mac-bootstrap.sh now installs openjdk@21). This script
# is a **migration helper** for users upgrading from JDK 17 setups —
# it adds openjdk@21 alongside an existing openjdk@17 install without
# removing the latter.
#
# Run with:
#   bash scripts/install-jdk21.sh
#
# Prompts once for admin password (the JDK system symlink). ~1–2 min total.

set -euo pipefail

banner() { echo; echo "==== $* ===="; }

# Sanity
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS only" >&2; exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "brew not found — run scripts/setup-mac-bootstrap.sh first" >&2; exit 1
fi

# 1. Install via brew (no sudo needed — brew owns /opt/homebrew).
banner "brew install openjdk@21"
if brew list --versions openjdk@21 >/dev/null 2>&1; then
  echo "  ✓ openjdk@21 already installed"
else
  brew install openjdk@21
fi

# 2. Symlink into /Library/Java/JavaVirtualMachines/ so /usr/libexec/java_home -v 21 finds it.
banner "JDK system symlink (sudo)"
TARGET="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk"
LINK="/Library/Java/JavaVirtualMachines/openjdk@21.jdk"
if [[ -L "$LINK" && "$(readlink "$LINK")" == "$TARGET" ]]; then
  echo "  ✓ symlink already in place: $LINK"
else
  sudo ln -sfn "$TARGET" "$LINK"
  echo "  symlinked $LINK -> $TARGET"
fi

# 3. Verify.
banner "Verify"
JDK21_PATH="$(/usr/libexec/java_home -v 21 2>&1 || true)"
JDK17_PATH="$(/usr/libexec/java_home -v 17 2>&1 || true)"
echo "  JDK 17 (Tlaloc):  $JDK17_PATH"
echo "  JDK 21 (Maestro): $JDK21_PATH"

if [[ "$JDK21_PATH" != *"21"* || "$JDK17_PATH" != *"17"* ]]; then
  echo
  echo "  WARNING: one of the JDKs didn't resolve. Check /Library/Java/JavaVirtualMachines/." >&2
  exit 1
fi

banner "Done"
echo "  Both JDKs registered. Gradle toolchain picks per-module:"
echo "    Tlaloc modules  → JDK 17"
echo "    Vendored Maestro → JDK 21"
