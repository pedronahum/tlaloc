#!/usr/bin/env bash
# The onboarding smoke: an external consumer can use the published Tlaloc
# artifacts end to end. Publishes every module to mavenLocal, runs
# examples/quickstart (which applies the Tlaloc Gradle plugin and the BOM), then
# checks the Gradle plugin itself with five throwaway consumer builds:
#
#   1. plugins { id("io.github.pedronahum.tlaloc") } with every tlaloc { } option
#      set: grad { } runs and dumpGradSourceDir/main receives the printed gradient.
#   2. the same program without the plugin: it fails at run time with the
#      "was not rewritten at compile time" error (the negative control for 1).
#   3. a Kotlin Multiplatform build with a JS target: the JS compilation is
#      skipped with a warning that names it.
#   4. the plugin without a Kotlin plugin: the build refuses by name.
#   5. a multi-project build: both plugins declared in the root with `apply false`
#      and applied in a subproject runs grad { }; the Tlaloc plugin declared in the
#      root and the Kotlin plugin only in the subproject (a classloader the Tlaloc
#      plugin cannot see) is refused by name.
#   6. the settings and build files printed in README.md and docs/GETTING_STARTED.md,
#      copied verbatim (the ```kotlin blocks whose first line is
#      `// settings.gradle.kts` / `// build.gradle.kts`), build and run grad { }.
#      The docs print the released Kotlin + Tlaloc pair; against the mavenLocal
#      publish that pair is replaced by this checkout's (libs.versions.kotlin and
#      the root build's version). TLALOC_SMOKE_FROM_CENTRAL=1 builds the blocks
#      exactly as printed, against Maven Central.
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
ls "$with/build/gradients/main/"*.kt > /dev/null 2>&1 \
  || fail "tlaloc { dumpGradSourceDir } wrote no .kt file under $with/build/gradients/main." "$work/with.log"
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
grep -q "applies neither the Kotlin JVM plugin nor the Kotlin Multiplatform plugin" "$work/bare.log" \
  || fail "the build without a Kotlin plugin failed for another reason." "$work/bare.log"
echo "onboarding-smoke: without a Kotlin plugin, the Tlaloc Gradle plugin refuses by name"

# 5. Multi-project builds.
multi="$work/multi"
mkdir -p "$multi/app/src/main/kotlin"
settings "$multi" multi
echo 'include("app")' >> "$multi/settings.gradle.kts"
cp "$with/src/main/kotlin/Main.kt" "$multi/app/src/main/kotlin/Main.kt"
cat > "$multi/build.gradle.kts" <<EOF
plugins {
    kotlin("jvm") version "$kotlin_version" apply false
    id("io.github.pedronahum.tlaloc") version "$version" apply false
}
EOF
app_build() {
  cat > "$multi/app/build.gradle.kts" <<EOF
plugins {
    kotlin("jvm") $1
    id("io.github.pedronahum.tlaloc")
    application
}
kotlin { jvmToolchain(25) }
dependencies {
    implementation(platform("io.github.pedronahum:tlaloc-bom:$version"))
    implementation("io.github.pedronahum:tlaloc-core")
    implementation("io.github.pedronahum:tlaloc-autograd")
}
application { mainClass.set("MainKt") }
EOF
}
app_build ""
./gradlew -p "$multi" :app:run --console=plain > "$work/multi.log" 2>&1 \
  || fail "the multi-project consumer (both plugins declared in the root) did not build and run." "$work/multi.log"
grep -q 'scalar=12.0' "$work/multi.log" \
  || fail "the multi-project consumer printed the wrong gradient." "$work/multi.log"
# The Kotlin plugin moves into the subproject's own plugins { } block.
cat > "$multi/build.gradle.kts" <<EOF
plugins {
    id("io.github.pedronahum.tlaloc") version "$version" apply false
}
EOF
app_build "version \"$kotlin_version\""
if ./gradlew -p "$multi" :app:run --console=plain > "$work/multi-split.log" 2>&1; then
  fail "the Tlaloc plugin was accepted with a Kotlin plugin it cannot see." "$work/multi-split.log"
fi
grep -q "cannot use the Kotlin Gradle plugin ('org.jetbrains.kotlin.jvm')" "$work/multi-split.log" \
  || fail "the split multi-project build failed for another reason." "$work/multi-split.log"
echo "onboarding-smoke: multi-project build runs; a Kotlin plugin the Tlaloc plugin cannot see is refused by name"

# 6. The install snippets in the docs, verbatim.
# Prints the ```kotlin block of $1 whose first line is exactly "// $2"; fails
# unless there is exactly one.
doc_block() {
  awk -v want="// $2" '
    /^```kotlin$/ { infence = 1; first = 1; buf = ""; next }
    infence && /^```$/ { if (keep) { printf "%s", buf; found++ }; infence = 0; keep = 0; next }
    infence { if (first) { keep = ($0 == want); first = 0 }; buf = buf $0 "\n" }
    END { if (found != 1) exit 3 }
  ' "$1"
}
for doc in README.md docs/GETTING_STARTED.md; do
  name="$(basename "$doc" .md | tr 'A-Z_' 'a-z-')"
  snip="$work/doc-$name"
  mkdir -p "$snip/src/main/kotlin"
  doc_block "$doc" settings.gradle.kts > "$snip/settings.gradle.kts" \
    || fail "$doc has no single \`// settings.gradle.kts\` Kotlin block."
  # The docs print the Maven Central setup. Resolve from the mavenLocal publish
  # above unless TLALOC_SMOKE_FROM_CENTRAL=1, which checks the released artifacts.
  if [ "${TLALOC_SMOKE_FROM_CENTRAL:-0}" != 1 ]; then
    sed -i.bak 's/repositories { /repositories { mavenLocal(); /' "$snip/settings.gradle.kts"
    rm -f "$snip/settings.gradle.kts.bak"
  fi
  doc_block "$doc" build.gradle.kts > "$snip/build.gradle.kts" \
    || fail "$doc has no single \`// build.gradle.kts\` Kotlin block."
  # The docs name the RELEASED Kotlin and Tlaloc versions. Against the mavenLocal
  # publish of this checkout, build them with this checkout's own pair instead —
  # everything else in the block stays verbatim. From Central they stay as printed.
  if [ "${TLALOC_SMOKE_FROM_CENTRAL:-0}" != 1 ]; then
    doc_kotlin="$(sed -n 's/^ *kotlin("jvm") version "\([^"]*\)".*/\1/p' "$snip/build.gradle.kts" | head -1)"
    doc_tlaloc="$(sed -n 's/^ *id("io.github.pedronahum.tlaloc") version "\([^"]*\)".*/\1/p' "$snip/build.gradle.kts" | head -1)"
    if [[ -z "$doc_kotlin" || -z "$doc_tlaloc" ]]; then
      fail "could not read the Kotlin and Tlaloc versions out of the build block in $doc."
    fi
    sed -i.bak \
      -e "s/kotlin(\"jvm\") version \"$doc_kotlin\"/kotlin(\"jvm\") version \"$kotlin_version\"/" \
      -e "s/\"$doc_tlaloc\"/\"$version\"/g" \
      -e "s/:$doc_tlaloc\"/:$version\"/g" \
      "$snip/build.gradle.kts"
    rm -f "$snip/build.gradle.kts.bak"
    grep -qF "kotlin(\"jvm\") version \"$kotlin_version\"" "$snip/build.gradle.kts" \
      || fail "the Kotlin version in the $doc build block was not replaced by $kotlin_version."
    if [ "$doc_tlaloc" != "$version" ] && grep -qF "$doc_tlaloc" "$snip/build.gradle.kts"; then
      fail "the $doc build block still names Tlaloc $doc_tlaloc after substitution."
    fi
    echo "onboarding-smoke: $doc prints Kotlin $doc_kotlin + Tlaloc $doc_tlaloc; building it with $kotlin_version + $version"
  fi
  echo 'application { mainClass.set("MainKt") }' >> "$snip/build.gradle.kts"
  cp "$with/src/main/kotlin/Main.kt" "$snip/src/main/kotlin/Main.kt"
  ./gradlew -p "$snip" run --console=plain > "$work/doc-$name.log" 2>&1 \
    || fail "the build files printed in $doc did not build and run." "$work/doc-$name.log"
  grep -q 'gradient=\[7.0, 11.0, 9.0, 13.0\]' "$work/doc-$name.log" \
    || fail "the build files printed in $doc ran but printed the wrong gradient." "$work/doc-$name.log"
  echo "onboarding-smoke: the build files printed in $doc build and run grad { }"
done

echo "onboarding-smoke: OK"
