#!/usr/bin/env bash
# Runs the Brooklyn Eats pipeline in order. Each stage is a subcommand of the example's
# application; a stage that is not implemented yet exits non-zero and stops the run.
#   examples/brooklyn-eats/run.sh                 # every stage
#   examples/brooklyn-eats/run.sh --quick         # every stage, reduced sizes (BROOKLYN_QUICK=1)
#   examples/brooklyn-eats/run.sh tasks eval      # only these stages, in the given order
#   examples/brooklyn-eats/run.sh verify          # compare inputs, derived files and models with manifest.json
#   examples/brooklyn-eats/run.sh models          # download the pinned models that are missing, check every file
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
    *) bin "$s" ;;
    esac
done
echo "== run: $SECONDS s"
