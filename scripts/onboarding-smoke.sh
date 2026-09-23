#!/usr/bin/env bash
# The onboarding smoke: an external consumer can use the published Tlaloc
# artifacts end to end. Publishes every module to mavenLocal, runs
# examples/quickstart (which applies the Tlaloc Gradle plugin and the BOM), then
# checks the Gradle plugin itself with four throwaway consumer builds:
#
#   1. plugins { id("io.github.pedronahum.tlaloc") } with every tlaloc { } option
#      set: grad { } runs and dumpGradSourceDir receives the printed gradient.
#   2. the same program without the plugin: it fails at run time with the
#      "was not rewritten at compile time" error (the negative control for 1).
#   3. a Kotlin Multiplatform build with a JS target: the JS compilation is
#      skipped with a warning that names it.
#   4. the plugin without a Kotlin plugin: the build refuses by name.
#
# Exits non-zero on the first check that fails.
set -euo pipefail
cd "$(dirname "$0")/.."

version="$(sed -n 's/^version = "\(.*\)"$/\1/p' build.gradle.kts)"
kotlin_version="$(sed -n 's/^kotlin = "\(.*\)"$/\1/p' gradle/libs.versions.toml)"
if [[ -z "$version" || -z "$kotlin_version" ]]; then
  echo "onboarding-smoke: could not read the Tlaloc or Kotlin version from the build." >&2
  exit 2
fi

./gradlew publishToMavenLocal -x test
./gradlew -p examples/quickstart run --quiet

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fail() {
  echo "onboarding-smoke: $1" >&2
  if [[ -n "${2:-}" && -f "$2" ]]; then
    echo "---- $2 (last 40 lines) ----" >&2
    tail -40 "$2" >&2
  fi
  exit 1
}

settings() {
  cat > "$1/settings.gradle.kts" <<EOF
rootProject.name = "$2"
pluginManagement {
    repositories {
        mavenLocal()
        gradlePluginPortal()
        mavenCentral()
    }
}
dependencyResolutionManagement {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
EOF
}

# A consumer of grad { }. $2 is the plugins { } line for Tlaloc (empty for none),
# $3 the tlaloc { } block. Tlaloc is applied BEFORE the Kotlin plugin here, and
# after it in examples/quickstart, so both orders are exercised.
grad_consumer() {
  local dir="$1"
  mkdir -p "$dir/src/main/kotlin"
  settings "$dir" "$(basename "$dir")"
  cat > "$dir/build.gradle.kts" <<EOF
plugins {
    $2
    kotlin("jvm") version "$kotlin_version"
    application
}
kotlin { jvmToolchain(25) }
dependencies {
    implementation(platform("io.github.pedronahum:tlaloc-bom:$version"))
    implementation("io.github.pedronahum:tlaloc-core")
    implementation("io.github.pedronahum:tlaloc-autograd")
}
application { mainClass.set("MainKt") }
$3
EOF
  cat > "$dir/src/main/kotlin/Main.kt" <<'EOF'
import io.tlaloc.autograd.grad
import io.tlaloc.core.DTensor
import io.tlaloc.core.F32
import io.tlaloc.core.Rank2
import io.tlaloc.core.Sym
import io.tlaloc.core.Tensors
import io.tlaloc.core.hostF32
import io.tlaloc.core.ops.matmul
import io.tlaloc.core.ops.sum
import io.tlaloc.core.ops.toFloat

fun main() {
    val g = grad { a: DTensor<Rank2<Sym, Sym>, F32> -> (a matmul a).sum().toFloat() }
    val input = Tensors.f32Matrix<Sym, Sym>(2, 2, floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
    println("gradient=" + g(input).hostF32().toList())
    // A scalar gradient has a Kotlin rendering, so dumpGradSourceDir writes it.
    val h = grad { x: Float -> x * x * x }
    println("scalar=" + h(2.0f))
}
EOF
}

# 1. The Gradle plugin, every option set.
with="$work/with-plugin"
grad_consumer "$with" "id(\"io.github.pedronahum.tlaloc\") version \"$version\"" '
tlaloc {
    strictLowering.set(true)
    dumpGradSource.set(true)
    dumpGradSourceDir.set(layout.buildDirectory.dir("gradients"))
    dumpLoweredIr.set(false)
    unsafeAllowUnsupportedKotlin.set(false)
}'
./gradlew -p "$with" run --console=plain > "$work/with.log" 2>&1 \
  || fail "the consumer that applies the Tlaloc Gradle plugin did not build and run." "$work/with.log"
grep -q 'gradient=\[7.0, 11.0, 9.0, 13.0\]' "$work/with.log" \
  && grep -q 'scalar=12.0' "$work/with.log" \
  || fail "the consumer that applies the Tlaloc Gradle plugin printed the wrong gradient." "$work/with.log"
ls "$with/build/gradients/"*.kt > /dev/null 2>&1 \
  || fail "tlaloc { dumpGradSourceDir } wrote no .kt file under $with/build/gradients." "$work/with.log"
echo "onboarding-smoke: Gradle plugin applied, gradient correct, dumpGradSourceDir honoured"

# 2. Negative control: the same program without the plugin.
without="$work/without-plugin"
grad_consumer "$without" "" ""
if ./gradlew -p "$without" run --console=plain > "$work/without.log" 2>&1; then
  fail "the consumer WITHOUT the Tlaloc Gradle plugin ran successfully; it must fail." "$work/without.log"
fi
grep -q 'was not rewritten at compile time' "$work/without.log" \
  || fail "the consumer without the plugin failed, but not with the plugin-missing error." "$work/without.log"
echo "onboarding-smoke: without the plugin, grad { } fails with the plugin-missing error"

# 3. A non-JVM compilation is skipped by name.
kmp="$work/kmp"
mkdir -p "$kmp"
settings "$kmp" kmp
cat > "$kmp/build.gradle.kts" <<EOF
plugins {
    kotlin("multiplatform") version "$kotlin_version"
    id("io.github.pedronahum.tlaloc") version "$version"
}
kotlin {
    jvmToolchain(25)
    jvm()
    js { nodejs() }
}
EOF
./gradlew -p "$kmp" tasks --console=plain > "$work/kmp.log" 2>&1 \
  || fail "the Multiplatform consumer with a JS target did not configure." "$work/kmp.log"
grep -q "compiler plugin not applied to Kotlin compilation 'main' of target 'js'" "$work/kmp.log" \
  || fail "the JS compilation was not reported as skipped." "$work/kmp.log"
if grep -q "compiler plugin not applied to Kotlin compilation '[^']*' of target 'jvm'" "$work/kmp.log"; then
  fail "the JVM compilation was reported as skipped." "$work/kmp.log"
fi
echo "onboarding-smoke: a JS compilation is skipped with a warning naming it"

# 4. No Kotlin plugin: refused by name.
bare="$work/bare"
mkdir -p "$bare"
settings "$bare" bare
cat > "$bare/build.gradle.kts" <<EOF
plugins {
    id("io.github.pedronahum.tlaloc") version "$version"
}
EOF
if ./gradlew -p "$bare" tasks --console=plain > "$work/bare.log" 2>&1; then
  fail "the Tlaloc Gradle plugin was accepted without a Kotlin plugin." "$work/bare.log"
fi
grep -q "applies no Kotlin Gradle plugin" "$work/bare.log" \
  || fail "the build without a Kotlin plugin failed for another reason." "$work/bare.log"
echo "onboarding-smoke: without a Kotlin plugin, the Tlaloc Gradle plugin refuses by name"

echo "onboarding-smoke: OK"
