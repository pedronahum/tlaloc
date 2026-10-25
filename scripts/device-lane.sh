#!/usr/bin/env bash
# The device lane: the checks GitHub's runners cannot run (they have no GPU, PJRT
# plugin, oracle venv or Triton), on this machine, one after another, with the
# results written to a dated directory and one line appended to a history file.
#
#   scripts/device-lane.sh                 everything below
#   scripts/device-lane.sh --quick         the suites only (no examples, no Triton, no benchmarks)
#   scripts/device-lane.sh --no-triton     skip triton/verify.sh
#   scripts/device-lane.sh --no-examples   skip the examples
#   scripts/device-lane.sh --no-bench      skip the benchmark registry
#
# Steps:
#   suites     every module's tests in one Gradle invocation with --continue (the
#              device tests run here: the PJRT plugin and the GPU are present)
#   examples   publishToMavenLocal, then `run` in each example (EXAMPLES overrides
#              the list); an example whose checkpoint is missing fails and is reported
#   triton     triton/verify.sh on ports 8100-8102 (8000-8002 belong to another
#              server on this machine), skipped when a Triton container is running
#              (TRITON_CLIENT_PYTHON, default ~/.local/venvs/tlaloc-client, runs its client)
#   bench      scripts/bench-registry.sh (the registered step benchmarks), when present
#
# Results: $TLALOC_DEVICE_LANE_DIR (default ~/.local/share/tlaloc/device-lane)/
#   <UTC time>-<short sha>/summary.md, the logs, and history.tsv one level up.
# Exit status: 0 when every step passed, 1 otherwise, 2 when the lane could not start.
#
# It refuses to start while another Gradle build runs. It never wraps Gradle in
# `timeout`, never rebuilds the Triton backend, and installs nothing; to run it
# nightly, add a crontab line such as
#   17 3 * * * cd ~/programming/tlaloc && scripts/device-lane.sh >/dev/null 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_examples=1
run_triton=1
run_bench=1
for arg in "$@"; do
    case "$arg" in
        --quick) run_examples=0; run_triton=0; run_bench=0 ;;
        --no-triton) run_triton=0 ;;
        --no-examples) run_examples=0 ;;
        --no-bench) run_bench=0 ;;
        -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "device-lane: unknown argument '$arg' (see --help)" >&2; exit 2 ;;
    esac
done

if pgrep -f 'org.gradle.wrapper.GradleWrapperMain' >/dev/null; then
    echo "device-lane: another Gradle build is running; wait for it to finish." >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "device-lane: no GPU visible (nvidia-smi -L failed)." >&2
    exit 2
fi

sha="$(git rev-parse --short HEAD)"
dirty=""
git diff --quiet HEAD -- || dirty="+dirty"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
base="${TLALOC_DEVICE_LANE_DIR:-$HOME/.local/share/tlaloc/device-lane}"
out="$base/$stamp-$sha$dirty"
mkdir -p "$out"
summary="$out/summary.md"
start=$(date +%s)
status=0

{
    echo "# Device lane $stamp"
    echo
    echo "- commit: \`$sha$dirty\` ($(git log -1 --format=%s | cut -c1-100))"
    echo "- host: $(hostname), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    [[ -f triton/pjrt/xla_cuda13/VERSION ]] && echo "- PJRT plugin: $(cat triton/pjrt/xla_cuda13/VERSION)"
    echo
} > "$summary"

step() { echo "== $1 ($(date -u +%H:%M:%S))" | tee -a "$out/lane.log"; }

# --- suites ---------------------------------------------------------------
step suites
modules=(core ir stablehlo autograd nn kptx runtime-pjrt runtime-cuda runtime-iree tokenizer maestro test-support benchmarks)
tasks=()
for m in "${modules[@]}"; do tasks+=(":$m:jvmTest"); done
tasks+=(":compiler-plugin:test" ":gradle-plugin:test")
./gradlew --continue "${tasks[@]}" > "$out/suites.log" 2>&1
suites_exit=$?
[[ $suites_exit -ne 0 ]] && status=1
python3 - "$ROOT" "$out" >> "$summary" <<'EOF'
import glob, os, re, sys, html
root, out = sys.argv[1], sys.argv[2]
rows, failed, skipped_reasons = [], [], {}
for d in sorted(glob.glob(os.path.join(root, "*", "build", "test-results", "*"))):
    module = d.split(os.sep)[-4]
    t = f = e = s = 0
    for x in glob.glob(os.path.join(d, "TEST-*.xml")):
        text = open(x, encoding="utf-8", errors="replace").read()
        m = re.search(r'<testsuite [^>]*tests="(\d+)"[^>]*skipped="(\d+)"[^>]*failures="(\d+)"[^>]*errors="(\d+)"', text)
        if not m:
            continue
        t += int(m[1]); s += int(m[2]); f += int(m[3]); e += int(m[4])
        for c in re.finditer(r'<testcase name="([^"]+)" classname="([^"]+)"[^>]*>\s*<(failure|error) message="([^"]{0,200})', text):
            failed.append(f"{c[2]}.{c[1]}: {html.unescape(c[4])}")
        for c in re.finditer(r'<skipped(?: message="([^"]{0,120})")?', text):
            reason = html.unescape(c[1] or "(no message)")
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
    if t:
        rows.append((module, t, f + e, s))
print("## Suites\n")
print("| Module | Tests | Failed | Skipped |\n|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
tt = sum(r[1] for r in rows); ff = sum(r[2] for r in rows); ss = sum(r[3] for r in rows)
print(f"| **all** | **{tt}** | **{ff}** | **{ss}** |\n")
if failed:
    print("Failures:\n")
    for x in failed[:60]:
        print(f"- {x}")
    print()
if skipped_reasons:
    print("Skip reasons (most common first):\n")
    for reason, n in sorted(skipped_reasons.items(), key=lambda kv: -kv[1])[:15]:
        print(f"- {n} × {reason}")
    print()
open(os.path.join(out, "counts"), "w").write(f"{tt} {ff} {ss}\n")
EOF
read -r tests failures skipped < "$out/counts" || { tests=0; failures=0; skipped=0; }
echo "- suites: Gradle exit $suites_exit" >> "$summary"

# --- examples ---------------------------------------------------------------
examples_result="skipped"
if [[ $run_examples -eq 1 ]]; then
    step examples
    echo -e "\n## Examples\n" >> "$summary"
    if ./gradlew publishToMavenLocal -x test > "$out/publish.log" 2>&1; then
        ok=0; bad=0
        list="${EXAMPLES:-quickstart readable-gradients differentiable-physics named-indices per-example-gradients gaussian-process mnist gpu-training training-step-on-device java-inference gpu-inference fine-tune lora-finetune internals/layer3 internals/four-worlds}"
        for ex in $list; do
            log="$out/example-${ex//\//-}.log"
            if ./gradlew -p "examples/$ex" run --quiet > "$log" 2>&1; then
                echo "- $ex: ok" >> "$summary"; ok=$((ok + 1))
            else
                echo "- $ex: **failed** (\`$(basename "$log")\`: $(grep -av '^\s*$' "$log" | tail -1 | cut -c1-160))" >> "$summary"
                bad=$((bad + 1)); status=1
            fi
        done
        examples_result="$ok/$((ok + bad))"
    else
        echo "- publishToMavenLocal failed (\`publish.log\`)" >> "$summary"; status=1; examples_result="publish failed"
    fi
fi

# --- triton ---------------------------------------------------------------
triton_result="skipped"
if [[ $run_triton -eq 1 ]]; then
    step triton
    echo -e "\n## Triton\n" >> "$summary"
    if docker ps --format '{{.Image}} {{.Names}}' 2>/dev/null | grep -qi triton; then
        echo "- skipped: a Triton container is running ($(docker ps --format '{{.Names}}' | grep -i triton | tr '\n' ' '))" >> "$summary"
        triton_result="skipped (server running)"
    else
        # verify.sh's client needs tritonclient and numpy; the system python3 has neither.
        client="${TRITON_CLIENT_PYTHON:-$HOME/.local/venvs/tlaloc-client/bin/python}"
        TRITON_CLIENT_PYTHON="$client" HTTP_PORT=8100 GRPC_PORT=8101 METRICS_PORT=8102 \
            triton/verify.sh > "$out/verify.log" 2>&1
        v=$?
        if [[ $v -eq 0 ]]; then
            echo "- verify.sh passed ($(grep -ac '^ok\b\|: ok$' "$out/verify.log") ok lines in \`verify.log\`)" >> "$summary"
            triton_result="ok"
        else
            echo "- verify.sh **failed** (exit $v): $(grep -a 'FAIL' "$out/verify.log" | head -3 | cut -c1-200 | tr '\n' ' ')" >> "$summary"
            triton_result="failed"; status=1
        fi
    fi
fi

# --- benchmarks -------------------------------------------------------------
bench_result="skipped"
if [[ $run_bench -eq 1 && -x scripts/bench-registry.sh ]]; then
    step bench
    echo -e "\n## Benchmarks\n" >> "$summary"
    if scripts/bench-registry.sh --out "$out/bench.json" > "$out/bench.log" 2>&1; then
        bench_result="ok"
    else
        bench_result="regressed or failed"; status=1
    fi
    grep -av '^\[bench\]\|^[IW][0-9]' "$out/bench.log" | tail -20 | sed 's/^/    /' >> "$summary"
    # The control: 3 ms added to every step must be reported.
    if ! scripts/bench-registry.sh --perturb > "$out/bench-perturb.log" 2>&1; then
        echo "- the --perturb control did not see the added 3 ms (\`bench-perturb.log\`)" >> "$summary"
        bench_result="$bench_result; control failed"; status=1
    fi
fi

end=$(date +%s)
minutes=$(( (end - start + 59) / 60 ))
verdict="$([[ $status -eq 0 ]] && echo passed || echo failed)"
echo -e "\nLane $verdict in $minutes min." >> "$summary"

history="$base/history.tsv"
[[ -f "$history" ]] || printf 'utc\tcommit\tstatus\ttests\tfailed\tskipped\texamples\ttriton\tbench\tminutes\n' > "$history"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stamp" "$sha$dirty" "$verdict" \
    "$tests" "$failures" "$skipped" "$examples_result" "$triton_result" "$bench_result" "$minutes" >> "$history"
ln -sfn "$out" "$base/latest"
echo "device-lane: $verdict; $summary"
exit $status
