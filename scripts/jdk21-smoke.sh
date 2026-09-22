#!/usr/bin/env bash
# §0.4.503 (Tier 3, item 1) — the JDK 21 smoke.
#
# §0.4.503 split the repository's bytecode targets: the library modules (:core,
# :ir, :autograd, :nn, :stablehlo, :maestro) emit Java 21, the FFM runtime
# backends and the compiler plugin stay at 25. The README and
# docs/GETTING_STARTED.md publish that as a per-module fact. This script is what
# makes it a CHECKED fact end to end, rather than a bytecode-version claim:
#
#   1. publish every module to mavenLocal (built by the JDK 25 toolchain),
#   2. compile examples/quickstart against them — toolchain 25, target 21, the
#      K2 plugin synthesizing a gradient at compile time,
#   3. RUN the result on a real JDK 21.
#
# Step 3 is the one that cannot be faked. A JDK 22+ API anywhere in :core, :ir or
# :autograd, or 25 bytecode in any of them or in the synthesized gradient, fails
# here with UnsupportedClassVersionError / NoSuchMethodError instead of printing
# a derivative.
#
# What this script deliberately does NOT claim: that a JDK 21 machine can BUILD a
# program containing `grad { }`. It cannot. Kotlin loads a compiler plugin inside
# the compiler's own JVM, and io.tlaloc:compiler-plugin is 25 bytecode, so the
# build machine needs a JDK 25. The supported consumer configuration is exactly
# what examples/quickstart does: jvmToolchain(25) + jvmTarget = JVM_21.
#
# Needs a JDK 21, named by JDK21_HOME because Gradle's auto-detection does not
# look in ~/.local/jdks and a literal path has no business in a committed file:
#
#     export JDK21_HOME=$HOME/.local/jdks/jdk-21.0.2
#     bash scripts/jdk21-smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${JDK21_HOME:-}" ]]; then
  echo "jdk21-smoke: JDK21_HOME is not set." >&2
  echo "  This script runs the quickstart on a REAL JDK 21, so it needs one." >&2
  echo "  Gradle's toolchain auto-detection does not search ~/.local/jdks, and" >&2
  echo "  gradle.properties reads the location from JDK21_HOME instead of carrying" >&2
  echo "  a path off one machine. Set it and re-run:" >&2
  echo "    export JDK21_HOME=/path/to/a/jdk-21" >&2
  exit 2
fi
if [[ ! -x "$JDK21_HOME/bin/java" ]]; then
  echo "jdk21-smoke: JDK21_HOME=$JDK21_HOME has no bin/java." >&2
  exit 2
fi
found_version="$("$JDK21_HOME/bin/java" -version 2>&1 | head -1)"
case "$found_version" in
  *'"21'*) ;;
  *)
    echo "jdk21-smoke: JDK21_HOME=$JDK21_HOME is not a JDK 21 — it reports:" >&2
    echo "  $found_version" >&2
    echo "  Running the smoke on the wrong JDK would certify the wrong thing." >&2
    exit 2
    ;;
esac
echo "jdk21-smoke: using $found_version from $JDK21_HOME"

./gradlew publishToMavenLocal -x test
./gradlew -p examples/quickstart runOnJdk21 --quiet
echo "jdk21-smoke: OK — a compile-time-synthesized gradient ran on $found_version"
