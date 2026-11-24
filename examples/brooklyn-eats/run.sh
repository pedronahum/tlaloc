#!/usr/bin/env bash
# Runs the Brooklyn Eats pipeline in order. Each stage is a subcommand of the example's
# application; a stage that is not implemented yet exits non-zero and stops the run.
#   examples/brooklyn-eats/run.sh                 # every stage
#   examples/brooklyn-eats/run.sh --quick         # every stage, reduced sizes (BROOKLYN_QUICK=1)
#   examples/brooklyn-eats/run.sh tasks eval      # only these stages, in the given order
#   examples/brooklyn-eats/run.sh snapshot        # download the pinned data snapshot and unpack what is missing
#   examples/brooklyn-eats/run.sh verify          # compare inputs, derived files and models with manifest.json
#   examples/brooklyn-eats/run.sh models          # download the pinned models that are missing, check every file
#   examples/brooklyn-eats/run.sh server start    # export (when needed) and serve TwIL-8B: Triton + tlaloc-endpoint
#   examples/brooklyn-eats/run.sh server stop     # (also: server status, server export; a config name may follow)
# The agent stage starts the server itself and stops it when it ends (BROOKLYN_SERVING_CONFIG,
# default twil-bf16: serving/<config>.env). The baseline stage serves twil-bf16, then ripple-nvfp4,
# and evaluates each on dev and test (runs with results are kept; BROOKLYN_FRESH=1 reruns them).
# The teach stage serves ripple-nvfp4 for the sample passes without results, then writes and checks
# the training shards. train trains and merges the student (no server); eval serves the merged
# student (twil-sft-bf16) on dev and test. rft serves it for the rejection-sampling passes, trains
# and merges new adapters on it, and serves the result (twil-rft-bf16) on dev.
# The snapshot stage unpacks the pinned raw sources (OSM, DOHMH, 311, NTA: the rolling datasets no
# longer return what they returned on the cutoff day) and the sampled outputs (teacher and student
# trajectories and shards), so ingest builds from the pinned files and teach and rft find their
# passes done. BROOKLYN_LIVE=1 skips it: ingest fetches every source live and verify reports where
# the sources changed.
# BROOKLYN_OFFLINE=1 skips the live tool smoke (snapshot mode only). BROOKLYN_DATA moves build/data.
# Each stage prints its wall time and the peak RSS of its process.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

args=()
for a in "$@"; do
    case "$a" in
    --quick) export BROOKLYN_QUICK=1 ;;
    *) args+=("$a") ;;
    esac
done
stages=(models snapshot ingest tools tasks agent baseline teach probe train eval rft distill serve chat)
[ "${#args[@]}" -gt 0 ] && stages=("${args[@]}")

"$root/gradlew" -q -p "$here" installDist
app="$here/build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats"
# Wall time and peak RSS of each stage (GNU time when present).
timed() {
    if [ -x /usr/bin/time ]; then /usr/bin/time -f "== $1: %e s, peak RSS %M KiB" "${@:2}"; else "${@:2}"; fi
}
bin() { timed "$1" "$app" "$@"; }
# Before a server starts: no Gradle daemon keeps memory (a 65B server leaves about 29 GiB of 121).
gradle_stop() { "$root/gradlew" -q --stop >/dev/null 2>&1 || true; }
# Serves config $1, evaluates it on each split of $2 (run 1; current results are kept), stops it.
serve_eval() {
    local cfg="$1" rc=0 split
    trap 'serving/server.sh stop "$cfg" >/dev/null 2>&1 || true' EXIT
    gradle_stop
    timed "eval: $cfg serving start" serving/server.sh start "$cfg" || rc=$?
    for split in $2; do
        [ "$rc" = 0 ] || break
        timed "eval: $cfg $split" "$app" eval run --config "$cfg" --split "$split" --run 1 || rc=$?
    done
    serving/server.sh stop "$cfg"
    trap - EXIT
    return "$rc"
}
# LoRA training on the GPU, no server: the checkpoint is read in f32 (35 GB) and its frozen weights
# kept in bf16; the heap shrinks after the load, before the device allocates. The merge holds the
# f32 merged weights and their bf16 copies at once.
train_jvm="-Xmx64g -XX:MaxHeapFreeRatio=20 -XX:MinHeapFreeRatio=5"
merge_jvm="-Xmx96g"

cd "$here"
if [ "${args[0]:-}" = server ]; then exec serving/server.sh "${args[@]:1}"; fi
# verify takes its own options (--update, --data DIR, --no-models); --data also moves BROOKLYN_DATA,
# which locates the tuned checkpoints the eval entries are keyed by.
if [ "${args[0]:-}" = verify ]; then
    for ((i = 1; i < ${#args[@]}; i++)); do [ "${args[i]}" = --data ] && export BROOKLYN_DATA="${args[i+1]}"; done
    bin verify "${args[@]:1}"; exit $?
fi
for s in "${stages[@]}"; do
    echo "== $s"
    case "$s" in
    snapshot)
        if [ "${BROOKLYN_LIVE:-0}" = 1 ]; then echo "snapshot: BROOKLYN_LIVE=1, the sources are fetched live"; else bin snapshot fetch; fi
        ;;
    tools)
        # web_search runs against a local SearXNG container; started here, stopped after.
        started=0
        if [ "${BROOKLYN_OFFLINE:-0}" != 1 ] && command -v docker >/dev/null && ! searxng/searxng.sh status >/dev/null 2>&1; then
            searxng/searxng.sh start && started=1 || echo "SearXNG did not start; web_search is checked in snapshot mode only"
        fi
        rc=0
        bin tools || rc=$?
        [ "$started" = 1 ] && searxng/searxng.sh stop
        [ "$rc" = 0 ] || exit "$rc"
        ;;
    agent)
        # Serves the student (exported from the pinned checkpoint when the artifact is missing or
        # stale), runs the agent on dev questions, checks the trainer's ids against the episodes,
        # and stops the server whatever happens.
        cfg="${BROOKLYN_SERVING_CONFIG:-twil-bf16}"
        trap 'serving/server.sh stop "$cfg" >/dev/null 2>&1 || true' EXIT
        rc=0
        gradle_stop
        timed "agent: serving start" serving/server.sh start "$cfg" || rc=$?
        [ "$rc" = 0 ] && { bin agent run --config "$cfg" || rc=$?; }
        [ "$rc" = 0 ] && { timed "agent: parity" "$app" agent parity "${BROOKLYN_DATA:-build/data}/agent/dev-10.jsonl" --config "$cfg" || rc=$?; }
        serving/server.sh stop "$cfg"
        trap - EXIT
        [ "$rc" = 0 ] || exit "$rc"
        ;;
    baseline)
        # Each model in turn (one server at a time): start its server, run its evaluations that have
        # no results yet (BROOKLYN_FRESH=1 reruns them all), stop it. Then the table.
        rc=0
        fresh=()
        [ "${BROOKLYN_FRESH:-0}" = 1 ] && fresh=(--fresh)
        for cfg in twil-bf16 ripple-nvfp4; do
            if [ "${#fresh[@]}" = 0 ]; then
                runs="$("$app" baseline pending | awk -v c="$cfg" '$1 == c { print $2 ":" $3 }')"
            else
                runs="$("$app" baseline pending --all | awk -v c="$cfg" '$1 == c { print $2 ":" $3 }')"
            fi
            [ -n "$runs" ] || { echo "baseline: $cfg has every result"; continue; }
            trap 'serving/server.sh stop "$cfg" >/dev/null 2>&1 || true' EXIT
            gradle_stop
            timed "baseline: $cfg serving start" serving/server.sh start "$cfg" || rc=$?
            for r in $runs; do
                [ "$rc" = 0 ] || break
                timed "baseline: $cfg ${r%%:*} run ${r##*:}" "$app" eval run --config "$cfg" --split "${r%%:*}" --run "${r##*:}" "${fresh[@]}" || rc=$?
            done
            serving/server.sh stop "$cfg"
            trap - EXIT
            [ "$rc" = 0 ] || exit "$rc"
        done
        bin baseline report
        ;;
    teach)
        # Ripple-65B on the training split: serve it for the sample passes without current results
        # (an interrupted pass resumes), stop it, then the shards (no server) and their check. The
        # client JVM is capped at 1 GiB of heap while the 65B model is served.
        rc=0
        passes="$("$app" teach pending)" || rc=$?
        [ "$rc" = 0 ] || exit "$rc"
        if [ -n "$passes" ]; then
            cfg=ripple-nvfp4
            trap 'serving/server.sh stop "$cfg" >/dev/null 2>&1 || true' EXIT
            gradle_stop
            timed "teach: $cfg serving start" serving/server.sh start "$cfg" || rc=$?
            [ "$rc" = 0 ] && { JAVA_OPTS="-Xmx1g" timed "teach: run (passes $(echo $passes))" "$app" teach run || rc=$?; }
            serving/server.sh stop "$cfg"
            trap - EXIT
            [ "$rc" = 0 ] || exit "$rc"
        else
            echo "teach: every sample pass has results"
        fi
        timed "teach: render" "$app" teach render
        timed "teach: check" "$app" teach check
        ;;
    eval)
        # The tuned student (serving/twil-sft-bf16.env, the merged checkpoint), served as the
        # baseline TwIL was, on each split of BROOKLYN_EVAL_SPLITS (default: dev test).
        serve_eval "${BROOKLYN_EVAL_CONFIG:-twil-sft-bf16}" "${BROOKLYN_EVAL_SPLITS:-dev test}"
        ;;
    probe | train)
        gradle_stop
        if [ "$s" = probe ]; then
            JAVA_OPTS="$train_jvm" timed "probe" "$app" probe
        else
            JAVA_OPTS="$train_jvm" timed "train: run" "$app" train run
            JAVA_OPTS="$merge_jvm" timed "train: merge" "$app" train merge
        fi
        ;;
    rft)
        # The rejection-sampling round: the tuned student (twil-sft-bf16) samples the training
        # questions after the teacher's (passes without current results; an interrupted one resumes);
        # its passing trajectories become shards; new adapters are trained on the merged SFT
        # checkpoint and merged; the result (twil-rft-bf16) is evaluated on BROOKLYN_RFT_SPLITS
        # (default: dev).
        rc=0
        passes="$("$app" rft pending)" || rc=$?
        [ "$rc" = 0 ] || exit "$rc"
        if [ -n "$passes" ]; then
            cfg=twil-sft-bf16
            trap 'serving/server.sh stop "$cfg" >/dev/null 2>&1 || true' EXIT
            gradle_stop
            timed "rft: $cfg serving start" serving/server.sh start "$cfg" || rc=$?
            [ "$rc" = 0 ] && { timed "rft: run (passes $(echo $passes))" "$app" rft run || rc=$?; }
            serving/server.sh stop "$cfg"
            trap - EXIT
            [ "$rc" = 0 ] || exit "$rc"
        else
            echo "rft: every sample pass has results"
        fi
        timed "rft: render" "$app" rft render
        timed "rft: check" "$app" rft check
        gradle_stop
        JAVA_OPTS="$train_jvm" timed "rft: train run" "$app" rft train run
        JAVA_OPTS="$merge_jvm" timed "rft: train merge" "$app" rft train merge
        serve_eval twil-rft-bf16 "${BROOKLYN_RFT_SPLITS:-dev}"
        ;;
    *) bin "$s" ;;
    esac
done
echo "== run: $SECONDS s"
