#!/usr/bin/env bash
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
#
# Runs Tlaloc's TPU session on a Cloud TPU VM from the tarball
# scripts/tpu/prepare.sh builds. No Gradle, no compile, no Tlaloc checkout.
#
#   tar xzf tpu-session.tar.gz && tpu-session/session.sh [LANE ...]
#
# Lanes, in the order they run (the most informative per minute first):
#   g2b        PjrtTpuSmokeTest + PjrtTpuLocalCertTest: platform, threefry, matmul grad, bf16
#   kernels    the tpu_custom_call kernel tests (Pallas- and Kotlin-emitted Mosaic payloads)
#   training   a CausalLM's and a mixture of experts' gradients vs the interpreter, AdamW steps
#              (PjrtCausalLmTrainingTest), AdamW on the device, ragged dots, the LM head's loss in
#              chunks and its peak memory (PjrtLmHeadLossMemoryTest)
#   multichip  sharded training under four ShardingPlans, the device-resident loop and the
#              collectives on four chips (skips with fewer)
#   serving    PjrtQwen3GreedyParityTest: Qwen3-0.6B greedy ids vs transformers
#   finetune   examples/fine-tune on the TPU (Qwen3-0.6B, AdamW, save)
#   suite      every device test in the bundle with TLALOC_TEST_PJRT_TARGET=tpu
# With no LANE arguments all of them run. A failing lane does not stop the next.
# Not run by default:
#   jax        the same Mosaic kernels dispatched by JAX (installs jax[tpu]==0.12.0.1,
#              the version the rest of Tlaloc is pinned to), for timings beside the kernels lane
#
# Before the lanes, on a TPU, the session stops when there is no TPU device node,
# when another process holds the chips (one process per TPU), when $HOME has
# less than MIN_FREE_GB free, or when PjrtTpuSmokeTest.platformNameReportsTpu
# does not pass: the TPU tests skip without a TPU, so a session on a VM whose
# libtpu does not load would otherwise finish green with nothing run. A lane
# that exits 0 with no test passed says so in the summary.
#
# Environment:
#   TARGET=cuda      rehearse the whole session on a GPU machine (setup and checks skipped)
#   LIBTPU_VERSION   libtpu wheel version (default: the MANIFEST's pin, else the latest)
#   QWEN3_DIR        an existing Qwen3-0.6B checkpoint directory (skips the download)
#   LANE_TIMEOUT     seconds per lane [1800]
#   MIN_FREE_GB      free space $HOME needs [10]
#   DUMPS=1          keep XLA dump flags (--xla_dump*) in XLA_FLAGS and LIBTPU_INIT_ARGS;
#                    without it they are removed (a dump is 1-7 GB per program set)
# Results: tpu-results-<time>.tar.gz next to this directory (logs, JUnit XML,
# MANIFEST, environment), with a summary printed at the end.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${TARGET:-tpu}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="$HERE/results-$STAMP"
mkdir -p "$RESULTS"
rm -rf "$HERE/device-tests/reports"   # reports of an earlier session are in its results tarball
LANES=("$@")
[[ ${#LANES[@]} -gt 0 ]] || LANES=(g2b kernels training multichip serving finetune suite)
QWEN3_REV=c1899de289a04d12100db370d81485cdf75e47ca
QWEN3_FILES=(config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt)

log() { echo "[$(date -u +%T)] $*" | tee -a "$RESULTS/session.log"; }
pack() { tar -C "$HERE" -czf "$HERE/../tpu-results-$STAMP.tar.gz" "results-$STAMP"; }
abort() {
  log "STOPPED: $*"
  cp -r "$HERE/device-tests/reports" "$RESULTS/junit" 2>/dev/null
  pack
  log "results: $(cd "$HERE/.." && pwd)/tpu-results-$STAMP.tar.gz"
  exit 1
}
passed() {  # tests passed in a JUnit report directory
  python3 - "$1" <<'PY'
import glob, os, sys, xml.etree.ElementTree as ET
n = 0
for x in glob.glob(os.path.join(sys.argv[1], "**", "TEST-*.xml"), recursive=True):
    for tc in ET.parse(x).getroot().iter("testcase"):
        n += all(tc.find(t) is None for t in ("failure", "error", "skipped"))
print(n)
PY
}

# ---- environment -------------------------------------------------------------
{
  echo "stamp $STAMP"; echo "target $TARGET"; uname -a
  cat "$HERE/MANIFEST" 2>/dev/null
  curl -s -m 3 -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type \
    && echo " (accelerator-type)"
  nproc; free -g | head -2
} > "$RESULTS/environment.txt" 2>&1

# ---- checks ------------------------------------------------------------------
for var in XLA_FLAGS LIBTPU_INIT_ARGS; do
  [[ "${!var:-}" == *--xla_dump* ]] || continue
  if [[ "${DUMPS:-0}" == 1 ]]; then
    log "$var keeps its dump flags (DUMPS=1): ${!var}"
  else
    kept="$(printf '%s\n' ${!var} | grep -v -- '^--xla_dump' | tr '\n' ' ')"
    export "$var=${kept% }"
    log "$var: dump flags removed (DUMPS=1 keeps them); now '${!var}'"
  fi
done
env | grep -E '^(XLA_|LIBTPU|TPU_|JAX_|TLALOC_)' | sort >> "$RESULTS/environment.txt"

if [[ "$TARGET" == tpu ]]; then
  DEVICES="$(ls -d /dev/accel* /dev/vfio/[0-9]* 2>/dev/null | tr '\n' ' ')"
  [[ -n "$DEVICES" ]] || abort "no TPU device node (/dev/accel*, /dev/vfio/N); is this a TPU VM, and the runtime image the TPU's (v6e: --version=v2-alpha-tpuv6e)?"
  log "TPU devices: $DEVICES"
  HOLDERS="$(for fd in /proc/[0-9]*/fd/*; do
      t="$(readlink "$fd" 2>/dev/null)" || continue
      [[ "$t" == /dev/accel* || "$t" == /dev/vfio/[0-9]* ]] && echo "${fd#/proc/}"
    done | cut -d/ -f1 | sort -u)"
  if [[ -n "$HOLDERS" ]]; then
    for pid in $HOLDERS; do log "holds the TPU: $(ps -o pid=,args= -p "$pid")"; done
    abort "another process holds the TPU (one process per TPU); stop it and run again"
  fi
  FREE_GB="$(df -Pk "$HOME" | awk 'NR == 2 { print int($4 / 1048576) }')"
  (( FREE_GB >= ${MIN_FREE_GB:-10} )) || abort "$HOME has $FREE_GB GB free, under MIN_FREE_GB=${MIN_FREE_GB:-10}"
  log "$FREE_GB GB free in $HOME"
fi

# ---- setup -------------------------------------------------------------------
if [[ "$TARGET" == tpu ]]; then
  JDKS="$HOME/.local/jdks"
  JAVA_HOME="$(ls -d "$JDKS"/jdk-25* 2>/dev/null | head -1 || true)"
  if [[ -z "$JAVA_HOME" ]]; then
    log "installing JDK 25 ($(uname -m))"
    ARCH=x64; [[ "$(uname -m)" == aarch64 ]] && ARCH=aarch64
    mkdir -p "$JDKS"
    curl -sL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/$ARCH/jdk/hotspot/normal/eclipse" \
      | tar xz -C "$JDKS"
    JAVA_HOME="$(ls -d "$JDKS"/jdk-25* | head -1)"
  fi
  export JAVA_HOME

  if [[ -z "${TLALOC_PJRT_PLUGIN_PATH:-}" ]]; then
    # libtpu.so does not depend on the Python version, and its wheels exist
    # only for Python >= 3.11 (Ubuntu 22.04 images have 3.10), so the wheel
    # is fetched from PyPI and unzipped rather than pip-installed.
    PIN="${LIBTPU_VERSION:-$(awk '/^libtpu /{print $2}' "$HERE/MANIFEST" 2>/dev/null)}"
    PIN="${PIN%.\*}"
    LIBTPU_DIR="$HOME/tlaloc-libtpu-${PIN:-latest}"
    if [[ ! -f "$LIBTPU_DIR/libtpu/libtpu.so" ]]; then
      log "fetching libtpu ${PIN:-latest} from PyPI"
      URL="$(curl -sf https://pypi.org/pypi/libtpu/json | python3 -c '
import json, sys
pin = sys.argv[1]
d = json.load(sys.stdin)
versions = [v for v in d["releases"] if not pin or v == pin or v.startswith(pin + ".")]
key = lambda v: [int(p) if p.isdigit() else 0 for p in v.split(".")]
for v in sorted(versions, key=key, reverse=True):
    for f in d["releases"][v]:
        if f["filename"].endswith("manylinux_2_31_x86_64.whl") and "t-manylinux" not in f["filename"]:
            print(v, f["url"]); sys.exit(0)
sys.exit(1)' "$PIN")" || { log "no libtpu ${PIN:-} wheel found on PyPI"; exit 1; }
      log "libtpu ${URL%% *}: ${URL#* }"
      mkdir -p "$LIBTPU_DIR"
      curl -sfL -o "$LIBTPU_DIR/libtpu.whl" "${URL#* }" && python3 -m zipfile -e "$LIBTPU_DIR/libtpu.whl" "$LIBTPU_DIR"
      echo "libtpu ${URL%% *}" > "$LIBTPU_DIR/VERSION"
    fi
    TLALOC_PJRT_PLUGIN_PATH="$LIBTPU_DIR/libtpu/libtpu.so"
    cat "$LIBTPU_DIR/VERSION" >> "$RESULTS/environment.txt"
  fi
  export TLALOC_PJRT_PLUGIN_PATH
  log "JAVA_HOME=$JAVA_HOME"
  log "TLALOC_PJRT_PLUGIN_PATH=$TLALOC_PJRT_PLUGIN_PATH"
else
  export JAVA_HOME="${JAVA_HOME:-$(ls -d "$HOME"/.local/jdks/jdk-25* | head -1)}"
  log "rehearsal on $TARGET with JAVA_HOME=$JAVA_HOME"
fi
"$JAVA_HOME/bin/java" -version >> "$RESULTS/environment.txt" 2>&1

if [[ "$TARGET" == tpu ]]; then
  log "preflight: PjrtTpuSmokeTest.platformNameReportsTpu"
  timeout 600 "$HERE/device-tests/run-tests.sh" --target tpu --name preflight \
    --method io.tlaloc.runtime.pjrt.PjrtTpuSmokeTest#platformNameReportsTpu > "$RESULTS/preflight.stdout" 2>&1
  [[ "$(passed "$HERE/device-tests/reports/preflight")" == 1 ]] \
    || abort "the PJRT client did not report platform tpu (preflight.stdout has why)"
  log "preflight: $(grep -a -o "platform_name='[^']*'" "$RESULTS/preflight.stdout" | head -1)"
fi

if [[ -z "${QWEN3_DIR:-}" ]]; then
  CACHED="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/$QWEN3_REV"
  if [[ -f "$CACHED/model.safetensors" ]]; then
    QWEN3_DIR="$CACHED"
  else
    QWEN3_DIR="$HOME/qwen3-0.6b-$QWEN3_REV"
    mkdir -p "$QWEN3_DIR"
    for f in "${QWEN3_FILES[@]}"; do
      [[ -s "$QWEN3_DIR/$f" ]] && continue
      log "downloading Qwen3-0.6B $f"
      curl -sfL -o "$QWEN3_DIR/$f" "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$QWEN3_REV/$f" \
        || log "download of $f failed; the serving and finetune lanes will skip"
    done
  fi
fi
export TLALOC_QWEN3_CHECKPOINT="$QWEN3_DIR"
log "Qwen3-0.6B at $QWEN3_DIR"

# ---- lanes -------------------------------------------------------------------
TIMEOUT="${LANE_TIMEOUT:-1800}"
declare -A STATUS
run_tests() {  # lane name, class patterns...
  local lane="$1"; shift
  log "lane $lane: $*"
  local t0=$SECONDS
  timeout "$TIMEOUT" "$HERE/device-tests/run-tests.sh" --target "$TARGET" --name "$lane" "$@" \
    > "$RESULTS/$lane.stdout" 2>&1
  local code=$?
  local n
  n="$(passed "$HERE/device-tests/reports/$lane")"
  STATUS[$lane]="exit $code in $((SECONDS - t0)) s, $n passed"
  (( code == 0 && n == 0 )) && STATUS[$lane]+=" (NO TEST PASSED: all skipped or none found)"
  log "lane $lane: ${STATUS[$lane]}"
}

for lane in "${LANES[@]}"; do
  case "$lane" in
    g2b) run_tests g2b '.*PjrtTpuSmokeTest' '.*PjrtTpuLocalCertTest' ;;
    kernels) run_tests kernels '.*TpuCustomCall.*' '.*TpuKernel.*' '.*Mosaic.*' ;;
    training) run_tests training '.*PjrtCausalLmTrainingTest' '.*PjrtDeviceAdamWTest' '.*PjrtRaggedDotTest' '.*PjrtLmHeadLossMemoryTest' ;;
    multichip) run_tests multichip '.*PjrtMultiDeviceCpuTest' '.*PjrtShardedTrainingCpuTest' ;;
    serving) run_tests serving '.*PjrtQwen3GreedyParityTest' ;;
    finetune)
      log "lane finetune"
      t0=$SECONDS
      FT_TARGET=$([[ "$TARGET" == tpu ]] && echo tpu || echo cuda)
      TLALOC_TARGET="$FT_TARGET" CHECKPOINT="$QWEN3_DIR" \
        timeout "$TIMEOUT" "$HERE/fine-tune/bin/tlaloc-fine-tune" "$RESULTS/qwen3-0.6b-france-rome" \
        > "$RESULTS/finetune.log" 2>&1
      STATUS[finetune]="exit $? in $((SECONDS - t0)) s"
      rm -f "$RESULTS"/qwen3-0.6b-france-rome/*.safetensors   # 2.4 GB; the log has the answers
      log "lane finetune: ${STATUS[finetune]}"
      ;;
    suite) run_tests suite --fork-per-class '.*' ;;
    jax)
      log "lane jax"
      t0=$SECONDS
      JVENV="$HOME/tlaloc-jax-0.12.0.1"
      if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
        log "lane jax: jax 0.12.0.1 needs Python >= 3.12; this VM has $(python3 -V 2>&1)"
        STATUS[jax]="skipped (Python < 3.12)"; continue
      fi
      [[ -x "$JVENV/bin/python" ]] || { python3 -m venv "$JVENV" && "$JVENV/bin/pip" install -q 'jax[tpu]==0.12.0.1'; }
      timeout "$TIMEOUT" "$JVENV/bin/python" "$HERE/harness/run_tpu_kernels_jax.py" \
        --fixtures "$HERE/device-tests/runtime-pjrt/classes/tpu-kernels" > "$RESULTS/jax.stdout" 2>&1
      STATUS[jax]="exit $? in $((SECONDS - t0)) s"
      log "lane jax: ${STATUS[jax]}"
      ;;
    *) log "unknown lane $lane"; STATUS[$lane]="unknown" ;;
  esac
done

# ---- summary -------------------------------------------------------------------
cp -r "$HERE/device-tests/reports" "$RESULTS/junit" 2>/dev/null
{
  echo "lane        status"
  for lane in "${LANES[@]}"; do printf "%-11s %s\n" "$lane" "${STATUS[$lane]:-not run}"; done
  echo
  python3 - "$RESULTS/junit" <<'PY'
import collections, glob, os, sys, xml.etree.ElementTree as ET
root = sys.argv[1]
rows = collections.OrderedDict()
for x in sorted(glob.glob(os.path.join(root, "**", "TEST-*.xml"), recursive=True)):
    lane = os.path.relpath(x, root).split(os.sep)[0]
    for tc in ET.parse(x).getroot().iter("testcase"):
        cls = tc.get("classname", "?").rsplit(".", 1)[-1]
        r = rows.setdefault((lane, cls), {"pass": 0, "skip": 0, "fail": 0, "why": []})
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if bad is not None:
            r["fail"] += 1
            r["why"].append("%s: %s" % (tc.get("name"), (bad.get("message") or "").splitlines()[0][:160] if bad.get("message") else ""))
        elif tc.find("skipped") is not None:
            r["skip"] += 1
        else:
            r["pass"] += 1
print("%-9s %-40s %5s %5s %5s" % ("lane", "class", "pass", "skip", "fail"))
for (lane, cls), r in rows.items():
    print("%-9s %-40s %5d %5d %5d" % (lane, cls, r["pass"], r["skip"], r["fail"]))
    for w in r["why"]:
        print("            FAIL " + w)
PY
  echo
  grep -hE '^\[(pjrt|tpu|kernel|mosaic)[^]]*\]' "$RESULTS"/*.stdout 2>/dev/null
  grep -A6 'before fine-tuning' "$RESULTS/finetune.log" 2>/dev/null
  grep -A6 'after fine-tuning' "$RESULTS/finetune.log" 2>/dev/null
} | tee "$RESULTS/summary.txt"
pack
log "results: $(cd "$HERE/.." && pwd)/tpu-results-$STAMP.tar.gz"
echo "copy back: gcloud compute tpus tpu-vm scp <vm>:$(cd "$HERE/.." && pwd)/tpu-results-$STAMP.tar.gz . --zone=<zone>"
