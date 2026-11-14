#!/usr/bin/env bash
# Runs the Brooklyn Eats pipeline in order. Each stage is a subcommand of the example's
# application; a stage that is not implemented yet exits non-zero and stops the run.
#   examples/brooklyn-eats/run.sh                 # every stage
#   examples/brooklyn-eats/run.sh --quick         # every stage, reduced sizes (BROOKLYN_QUICK=1)
#   examples/brooklyn-eats/run.sh tasks eval      # only these stages, in the given order
#   examples/brooklyn-eats/run.sh verify          # compare inputs, derived files and models with manifest.json
#   examples/brooklyn-eats/run.sh models          # download the pinned models that are missing, check every file
#   examples/brooklyn-eats/run.sh server start    # export (when needed) and serve TwIL-8B: Triton + tlaloc-endpoint
#   examples/brooklyn-eats/run.sh server stop     # (also: server status, server export; a config name may follow)
# The agent stage starts the server itself and stops it when it ends (BROOKLYN_SERVING_CONFIG,
# default twil-bf16: serving/<config>.env). The baseline stage serves twil-bf16, then ripple-nvfp4,
# and evaluates each on dev and test (runs with results are kept; BROOKLYN_FRESH=1 reruns them).
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
stages=(models ingest tools tasks agent baseline teach probe train distill rft eval serve chat)
[ "${#args[@]}" -gt 0 ] && stages=("${args[@]}")

"$root/gradlew" -q -p "$here" installDist
app="$here/build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats"
# Wall time and peak RSS of each stage (GNU time when present).
timed() {
    if [ -x /usr/bin/time ]; then /usr/bin/time -f "== $1: %e s, peak RSS %M KiB" "${@:2}"; else "${@:2}"; fi
}
bin() { timed "$1" "$app" "$@"; }

cd "$here"
if [ "${args[0]:-}" = server ]; then exec serving/server.sh "${args[@]:1}"; fi
for s in "${stages[@]}"; do
    echo "== $s"
    case "$s" in
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
    *) bin "$s" ;;
    esac
done
echo "== run: $SECONDS s"
