#!/usr/bin/env bash
# Times the measured cases of BenchRegistry (runtime-pjrt tests) on this machine's device
# and compares each case's median with its history.
#
#   scripts/bench-registry.sh                       measure, compare, append to the history
#   scripts/bench-registry.sh --out results.jsonl   also keep this run's results there
#   scripts/bench-registry.sh --no-record           compare without appending
#   scripts/bench-registry.sh --perturb             the negative control: every step waits 3 ms more,
#                                                   which must be reported as a regression (exit 0
#                                                   only when every measured case regressed; not recorded)
#
# History: $TLALOC_DEVICE_LANE_DIR/bench-history.jsonl (default ~/.local/share/tlaloc/device-lane),
# one JSON line per case and run. A case's median is compared with the median m of its last ten
# recorded medians. Millisecond steps on the GB10 move by about 20 % between runs (host dispatch),
# so the bound follows the history's own spread: with three or more runs it regresses above
# max(1.25 m, m + 4 * 1.4826 * MAD), with fewer above 1.5 m, and never for less than 0.1 ms.
# Exit 1 on a regression or a failed run, 2 when another Gradle build is running.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

out=""
record=1
perturb=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --no-record) record=0; shift ;;
        --perturb) perturb=1; record=0; shift ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "bench-registry: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
if pgrep -f 'org.gradle.wrapper.GradleWrapperMain' >/dev/null; then
    echo "bench-registry: another Gradle build is running; wait for it to finish." >&2
    exit 2
fi

results="$(mktemp)"
trap 'rm -f "$results"' EXIT
TLALOC_BENCH_PERTURB_MS="$([[ $perturb -eq 1 ]] && echo 3 || echo 0)" TLALOC_BENCH_OUT="$results" ./gradlew -q :runtime-pjrt:jvmTest --tests 'io.tlaloc.runtime.pjrt.BenchRegistryTest.measure' --rerun
if [[ ! -s "$results" ]]; then
    echo "bench-registry: the measurement wrote no results" >&2
    exit 1
fi
[[ -n "$out" ]] && cp "$results" "$out"

history="${TLALOC_DEVICE_LANE_DIR:-$HOME/.local/share/tlaloc/device-lane}/bench-history.jsonl"
mkdir -p "$(dirname "$history")"
python3 - "$results" "$history" "$record" "$(git rev-parse --short HEAD)" "$perturb" <<'PY'
import json, statistics, sys, datetime
results, history, record, commit = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
perturb = sys.argv[5] == "1"
now = [json.loads(l) for l in open(results) if l.strip()]
past = {}
try:
    for l in open(history):
        if l.strip():
            r = json.loads(l)
            past.setdefault(r["label"], []).append(r["median_ms"])
except FileNotFoundError:
    pass
regressed = []
print(f"{'case':44} {'median ms':>10} {'p10':>9} {'p90':>9} {'history':>10} {'change':>8}")
for r in now:
    h = past.get(r["label"], [])[-10:]
    base = statistics.median(h) if h else None
    change = "" if base is None else f"{100 * (r['median_ms'] / base - 1):+.1f}%"
    flag = ""
    if base is not None:
        if len(h) >= 3:
            mad = statistics.median(abs(x - base) for x in h)
            bound = max(1.25 * base, base + 4 * 1.4826 * mad)
        else:
            bound = 1.5 * base
    if base is not None and r["median_ms"] > bound and r["median_ms"] - base > 0.1:
        flag = "  REGRESSED"
        regressed.append(r["label"])
    print(f"{r['label']:44} {r['median_ms']:10.3f} {r['p10_ms']:9.3f} {r['p90_ms']:9.3f} "
          f"{'' if base is None else f'{base:10.3f}':>10} {change:>8}{flag}")
if record:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(history, "a") as f:
        for r in now:
            f.write(json.dumps({**r, "utc": stamp, "commit": commit}) + "\n")
if perturb:
    missed = [r["label"] for r in now if r["label"] not in regressed]
    if missed:
        print("perturb: these cases did not show the added 3 ms: " + ", ".join(missed))
        sys.exit(1)
    print("perturb: every case regressed, as it must")
    sys.exit(0)
if regressed:
    print("regressed: " + ", ".join(regressed))
    sys.exit(1)
PY
