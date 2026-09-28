#!/usr/bin/env bash
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
#
# Runs Tlaloc's device tests from a bundle built by `./gradlew :runtime-pjrt:tpuBundle`,
# with a JDK and nothing else (no Gradle, no network).
#
#   ./run-tests.sh [--target tpu|cuda] [--name LABEL] [--fork-per-class] [PATTERN ...]
#
# --fork-per-class starts one JVM per test class, so each class gets a fresh
# process (and a fresh PJRT client); the exit status is the worst of them.
#
# PATTERN is a class-name regex for JUnit's --include-classname (default: every
# class). Examples:
#   ./run-tests.sh --target tpu '.*PjrtTpu.*'
#   ./run-tests.sh --target tpu --name qwen3 '.*PjrtQwen3GreedyParityTest'
#
# JAVA (default: $JAVA_HOME/bin/java, else java on PATH) must be JDK 25+.
# On a TPU, TLALOC_PJRT_PLUGIN_PATH must name libtpu.so.
# Reports: reports/<name>/ (JUnit XML) and reports/<name>.log. The exit status
# is JUnit's: 0 when nothing failed (skips count as passing; read the log).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=cuda
NAME=""
PATTERNS=()
FORK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --fork-per-class) FORK=1; shift ;;
    -h|--help) sed -n '5,20p' "$0"; exit 0 ;;
    *) PATTERNS+=("$1"); shift ;;
  esac
done
[[ "$TARGET" == tpu || "$TARGET" == cuda ]] || { echo "--target must be tpu or cuda" >&2; exit 2; }
[[ ${#PATTERNS[@]} -gt 0 ]] || PATTERNS=('.*')
NAME="${NAME:-$TARGET-$(date +%H%M%S)}"

JAVA="${JAVA:-${JAVA_HOME:+$JAVA_HOME/bin/java}}"
JAVA="${JAVA:-java}"
"$JAVA" -version 2>&1 | head -1 | grep -qE '"(2[5-9]|[3-9][0-9])' \
  || { echo "need JDK 25+; $JAVA is: $("$JAVA" -version 2>&1 | head -1)" >&2; exit 2; }

mkdir -p "$HERE/reports"
cd "$HERE/runtime-pjrt"   # the tests read fixtures by paths relative to the module directory

launch() {  # report dir, then --include-classname / --select-class arguments
  local dir="$1"; shift
  TLALOC_TEST_PJRT_TARGET="$TARGET" "$JAVA" --enable-native-access=ALL-UNNAMED -Xmx${TLALOC_TEST_HEAP:-12g} \
    -cp "$HERE/lib/*:$HERE/runtime-pjrt/classes" \
    org.junit.platform.console.ConsoleLauncher execute \
    --details=tree --disable-banner --disable-ansi-colors \
    --reports-dir "$dir" "$@"
}

echo "== $NAME: target $TARGET, classes ${PATTERNS[*]}${FORK:+ (one JVM per class)}"
set +e
if [[ -z "$FORK" ]]; then
  INCLUDES=()
  for p in "${PATTERNS[@]}"; do INCLUDES+=(--include-classname "$p"); done
  launch "$HERE/reports/$NAME" --scan-classpath "$HERE/runtime-pjrt/classes" "${INCLUDES[@]}" \
    2>&1 | tee "$HERE/reports/$NAME.log"
  STATUS=${PIPESTATUS[0]}
else
  STATUS=0
  : > "$HERE/reports/$NAME.log"
  CLASSES=$(cd "$HERE/runtime-pjrt/classes" && find . -name '*Test.class' ! -name '*$*' | sed 's|^\./||; s|\.class$||; s|/|.|g' | sort)
  for c in $CLASSES; do
    match=""
    for p in "${PATTERNS[@]}"; do [[ "$c" =~ ^$p$ ]] && match=1; done
    [[ -n "$match" ]] || continue
    launch "$HERE/reports/$NAME/${c##*.}" --select-class "$c" 2>&1 | tee -a "$HERE/reports/$NAME.log"
    s=${PIPESTATUS[0]}
    (( s > STATUS )) && STATUS=$s
  done
fi
set -e
echo "== $NAME: exit $STATUS (log: reports/$NAME.log)"
exit "$STATUS"
