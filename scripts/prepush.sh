#!/usr/bin/env bash
# The checks CI's `./gradlew test` runs that a module's own jvmTest does not, in one
# Gradle invocation with --continue, so every failing gate is reported at once.
#
#   scripts/prepush.sh                  the gates below (16 s with nothing up to date on a
#                                       DGX Spark, a few seconds after a small change)
#   scripts/prepush.sh --plugin-tests   also :compiler-plugin:test (603 tests, 105 s)
#
# Gates:
#   abi        checkKotlinAbi in every module (the committed api/*.api dumps)
#   compile    main and test sources of every module, the compiler plugin included
#              (an exhaustive `when` over DType there breaks when a DType is added)
#   jvm-target verifyJvmTarget: each module's bytecode is the major version
#              tlalocJvmTargets names
#   publishing verifyPomMetadata, verifyBomCoverage, :gradle-plugin:validatePlugins
#   kdoc       checkKdocInternalRefs (no work-item references in published KDoc)
#
# It does not run the test suites (run the suites of the modules you changed as well),
# the JDK 21 lane or the vendored Maestro lane.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

plugin_tests=0
for arg in "$@"; do
    case "$arg" in
        --plugin-tests) plugin_tests=1 ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "prepush: unknown argument '$arg' (see --help)" >&2; exit 2 ;;
    esac
done

if pgrep -f 'org.gradle.wrapper.GradleWrapperMain' >/dev/null; then
    echo "prepush: another Gradle build is running; wait for it to finish." >&2
    exit 2
fi

tasks=(
    checkKotlinAbi
    classes jvmMainClasses
    testClasses jvmTestClasses
    verifyJvmTarget
    verifyPomMetadata verifyBomCoverage :gradle-plugin:validatePlugins
    checkKdocInternalRefs
)
[[ $plugin_tests == 1 ]] && tasks+=(:compiler-plugin:test)

log="$(mktemp -t tlaloc-prepush.XXXXXX.log)"
start=$SECONDS
echo "prepush: ./gradlew --continue ${tasks[*]}"
./gradlew --continue "${tasks[@]}" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
elapsed=$((SECONDS - start))

if [[ $status == 0 ]]; then
    echo
    echo "prepush: all gates passed (${elapsed} s)."
    rm -f "$log"
    exit 0
fi

echo
echo "prepush: FAILED after ${elapsed} s. Full log: $log"
mapfile -t failed < <(grep -oE "Execution failed for task '[^']+'" "$log" \
    | sed -E "s/.*task '([^']+)'/\1/" | sort -u)

if [[ ${#failed[@]} == 0 ]]; then
    echo "  Gradle failed before running a task (configuration or resolution); see the log."
    exit 1
fi

for t in "${failed[@]}"; do
    module="${t%:*}"
    name="${t##*:}"
    case "$name" in
        checkKotlinAbi)
            echo "  abi        $t: the public API differs from ${module#:}/api/."
            echo "             If the change is intended: ./gradlew $module:updateKotlinAbi," \
                "review the api/*.api diff and commit it with the change." ;;
        compile*)
            echo "  compile    $t: compilation error (in :compiler-plugin, usually a" \
                "'when' that needs a branch for a new DType or OpKind)." ;;
        verifyJvmTarget)
            echo "  jvm-target $t: bytecode version differs from tlalocJvmTargets in" \
                "build.gradle.kts." ;;
        verifyPomMetadata|verifyBomCoverage|validatePlugins)
            echo "  publishing $t: see the message above." ;;
        checkKdocInternalRefs)
            echo "  kdoc       $t: python3 scripts/check-kdoc-internal-refs.py lists the" \
                "references." ;;
        test)
            echo "  tests      $t: see ${module#:}/build/reports/tests/test/index.html." ;;
        *)
            echo "  other      $t" ;;
    esac
done
exit 1
