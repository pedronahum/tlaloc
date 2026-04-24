#!/usr/bin/env bash
# Sum the `tests=` attribute across every JUnit TEST-*.xml report in the repo.
# Matches the pattern used in /loop iterations for a single-number post-build
# check. Single stable command form so the Claude Code permission allowlist
# can whitelist it once and forever instead of prompting per-pipeline.
set -euo pipefail
find . -name 'TEST-*.xml' -path '*/test-results/*' -print0 \
  | xargs -0 grep -h 'testsuite name' \
  | sed -E 's/.*tests="([0-9]+)".*/\1/' \
  | awk '{s+=$1} END {print s}'
