#!/usr/bin/env bash
# §0.4.355 — the onboarding smoke: proves an external consumer can use the
# published Tlaloc artifacts end-to-end. Publishes every module to
# mavenLocal, then builds and runs examples/quickstart (a standalone Gradle
# project that resolves io.tlaloc:* from mavenLocal and applies the K2
# compiler plugin). Exits non-zero if the quickstart's gradient check fails.
set -euo pipefail
cd "$(dirname "$0")/.."
./gradlew publishToMavenLocal -x test
./gradlew -p examples/quickstart run --quiet
