#!/usr/bin/env bash
# Runs the Brooklyn Eats pipeline in order. Each stage is a subcommand of the example's
# application; a stage that is not implemented yet exits non-zero and stops the run.
#   examples/brooklyn-eats/run.sh            # every stage
#   examples/brooklyn-eats/run.sh tasks eval # only these stages, in the given order
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

stages=(ingest tools tasks agent baseline teach probe train distill rft eval serve chat)
[ "$#" -gt 0 ] && stages=("$@")

"$root/gradlew" -q -p "$here" installDist
bin="$here/build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats"

cd "$here"
for s in "${stages[@]}"; do
    echo "== $s"
    "$bin" "$s"
done
