# Compatibility policy

**Status: this is a promise, not a design note.** It is the written form of the
one line the README has carried for months — "APIs change without deprecation
cycles" — and it exists so that line cannot be read more generously than it was
meant.

Current version: **`0.1.0-alpha01`**. Nothing is on Maven Central yet; you build
from source and consume from `mavenLocal` (see
[GETTING_STARTED.md](GETTING_STARTED.md)).

## What "alpha" means here

`0.1.0-alpha01` means: **the engine is certified, the surface is not settled.**

Those are different claims and the repository keeps them apart on purpose. The
2,522 automated tests at HEAD pin behaviour — gradients against analytic and
cross-implementation oracles, a real TinyLlama matching HuggingFace token for
token, RNG bit-exact against JAX's threefry stream. None of that pins *names*. A
function whose result is certified correct may still be renamed, moved to another
module, or given a different parameter order in `0.1.0-alpha02`.

Version numbers before `1.0.0` carry no compatibility guarantee under SemVer
§4, and Tlaloc takes that literally rather than pretending the minor number
means something it does not.

### Two things added in §0.4.505 make that sentence less blunt

The paragraph above grades a surface with thousands of oracle tests behind it
exactly the same as a surface nothing has ever executed, which is honest but not
useful. Two mechanisms now separate them.

- **`@ExperimentalTlalocApi`** — a `@RequiresOptIn(ERROR)` marker (declared in
  `:core`) on the part of the surface that is *provisional*, meaning it may change
  **shape**, not merely signature. Three surfaces carry it, each for a stated
  reason: the four-worlds scope taxonomy (`KernelScope`, `OrchestrationScope`,
  `ProgramScope`, `ClusterScope`, `Tlaloc`, `BufferHandle`, `HandleRef` — its own
  KDoc scopes it to v1 and names Kotlin's context parameters as where it would
  have to go); the collective attribute convention (`io.tlaloc.ir.AllReduceAttrs`
  — distributed execution is 📐 in [CAPABILITIES.md](CAPABILITIES.md) and has
  never run on two hosts); and the kernel-choice and cost-model surface
  (`io.tlaloc.ir.recognizer.kernel`, `io.tlaloc.ir.recognizer.cost` — the
  machinery is unit-certified but its *purpose* is picking a kernel, and the one
  kernel measured against XLA lost at small shapes). Touching any of them without
  `@OptIn(ExperimentalTlalocApi::class)` is a compile error that names the marker
  and points here. **`grad`, the op surface and `:nn` do NOT carry it**, on
  purpose: an annotation on everything teaches you to opt in once and stop
  reading.
- **An ABI baseline.** `api/<module>.api` is committed and `./gradlew apiCheck`
  (wired into `check`) fails on any difference, so a break is a reviewed diff
  rather than a surprise. It covers `:core`, `:ir`, `:autograd`, `:nn`,
  `:stablehlo` and `:maestro` — the modules a consumer compiles against.
  binary-compatibility-validator 0.18.2 cannot read Java 25 bytecode
  (`Unsupported class file major version 69`), so the five 25-targeted modules —
  `:runtime-pjrt`, `:runtime-cuda`, `:kptx`, `:runtime-iree`, `:compiler-plugin` —
  are **not** covered. That is a tool limit, stated rather than glossed.

Neither mechanism weakens anything below: an API with no marker on it is still
free to change in any alpha. What they add is a *signal* about which ones will,
and a *record* when one does.

## What may break, without a deprecation cycle

Any of these may change in any alpha release:

- **Any public API in any module.** Names, signatures, parameter order, default
  arguments, nullability, type parameters, receiver vs. argument.
- **Module boundaries and coordinates.** A type may move between `:core`, `:ir`,
  `:autograd` and `:nn`; a module may be split, merged or renamed.
- **The named-axis and rank type encoding.** `Rank2<Sym, Sym>`, `DTensor`'s type
  parameters and the shape-error diagnostics are the newest and least settled
  part of the surface.
- **The compiler plugin's flags and diagnostics.** Flag names, diagnostic text,
  diagnostic severities and the `dumpGradSource` output format.
- **The serving artifact's on-disk layout.** The manifest schema, file names and
  StableHLO body organisation. Re-export from the same Tlaloc version you serve
  with.
- **The KPTX kernel registry and its claiming rules.** Nothing there is
  registered by default and the opt-in surface is explicitly experimental.
- **Emitted StableHLO.** Op choice and structure may change whenever XLA gets
  faster at something else. `docs/TLALOC_EMIT_CONTRACT.md` describes what we emit
  today, not what we will emit.
- **Test counts, benchmark numbers and the modules' internal layout.** These are
  not API at all.

## What will not break inside the alpha series

These are the commitments. They are short on purpose — a long list here would be
a list of things this policy could not keep.

- **The group id stays `io.github.pedronahum`.** It changed once, in §0.4.508,
  before anything was published and therefore before it could break anyone; that
  was the only free moment and it will not be taken again. Artifact ids may gain
  or lose modules, but
  a coordinate that exists will not be re-pointed at different code under the
  same version.
- **A published version is immutable.** `0.1.0-alphaNN` is never re-published
  with different bytes. If it is wrong, the fix is `alphaNN+1`.
- **No silent degradation, ever.** This is a house rule, not a version policy:
  an unsupported dtype, shape, device or op refuses loudly and by name. An alpha
  may remove a capability; it will not quietly approximate one.
- **Certified stays certified.** A behaviour that a test pins does not become
  wrong without the test changing in the same commit, and a capability's mark in
  the README's Maturity table does not get upgraded without the test that earns
  it.
- **Licensing does not change retroactively.** Apache-2.0 for everything Tlaloc
  publishes; a change would apply to later versions only.
- **A break in the covered modules' ABI leaves a trace.** `api/*.api` is a
  committed baseline and `apiCheck` runs inside `./gradlew test`, so a change to
  `:core`, `:ir`, `:autograd`, `:nn`, `:stablehlo` or `:maestro`'s public ABI
  cannot land without a matching `apiDump` in the same commit. This is not a
  promise not to break; it is a promise that breaking is visible.

## What a consumer should do about it

- **Pin an exact version.** Never a range, never `latest.release`. There are no
  compatible-upgrade guarantees to lean on.
- **Expect to read the changelog on every bump.** [CHANGELOG.md](../CHANGELOG.md)
  records breaking changes per version; there is no deprecation period in which
  both spellings work.
- **Keep the compiler-plugin version equal to the library version.** The plugin
  synthesizes calls into `:core`/`:ir`; mixing versions is unsupported and is not
  checked for you today (see [ALPHA_PLAN.md](ALPHA_PLAN.md) — this is a named
  gap, not an oversight).
- **Treat the toolchain as pinned too**, and read the two halves separately —
  §0.4.503 split them.
  - **Kotlin: 2.3.20 through 2.3.29.** A K2 compiler plugin binds to compiler
    internals; a different *feature* release is not expected to work and is not
    tested. Since §0.4.503 the plugin no longer finds out the hard way: it detects
    the running compiler's version and REFUSES at compile time, naming what it
    found and the supported range, instead of raising a `NoSuchMethodError` from
    inside `compileKotlin`. Kotlin numbers feature releases by tens in the third
    component and bugfixes by ones above them, so the supported range is the whole
    bugfix family of 2.3.20 and nothing else — 2.3.10 and 2.3.30 are *different
    feature releases*. `-P plugin:io.tlaloc.plugin:unsafeAllowUnsupportedKotlin=true`
    turns the refusal into a warning for anyone who wants to try; a crash below it
    is then the expected outcome, not a bug.
  - **JDK: 25 to build, 21 to run.** The library modules (`core`, `ir`,
    `autograd`, `nn`, `stablehlo`, `maestro`) emit Java 21 bytecode and are
    compiled with `-Xjdk-release=21`. The FFM runtime backends (`runtime-pjrt`,
    `runtime-cuda`, `kptx`), `runtime-iree` and `compiler-plugin` emit Java 25.
    Because Kotlin loads a compiler plugin into the compiler's own JVM, a project
    using `grad { }` needs a JDK 25 *on the build machine* whatever it targets.
    The per-module table is in [GETTING_STARTED.md](GETTING_STARTED.md) §0 and the
    gate is `scripts/jdk21-smoke.sh` plus `verifyJvmTarget`, which reads the
    class-file major version out of every published jar.
- **Symja (`org.matheclipse:matheclipse-core`, LGPL-3.0) is optional from
  `0.1.0-alpha01`.** It was a mandatory runtime dependency of `io.github.pedronahum:tlaloc-ir` until
  §0.4.503 and is `compileOnly` now, so it will not appear in your graph unless
  you add it. Adding it is one line — `kotlinCompilerPluginClasspath(...)`, because
  the CAS runs inside the compiler and not inside your program — and Tlaloc names
  that line at compile time on the day a body actually needs the CAS. If your policy forbids LGPL in the
  dependency graph, nothing in Tlaloc pulls it in.

## When this policy changes

At `1.0.0`, and not before. At that point SemVer applies without carve-outs and
this document is replaced by the shorter one that says so. Until then, every
alpha release is free to break anything in the "may break" list, and is required
to say in [CHANGELOG.md](../CHANGELOG.md) that it did.
