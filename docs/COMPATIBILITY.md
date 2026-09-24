# Compatibility policy

What the alpha series promises, and what it does not.

Current version: **`0.1.0-alpha02`**, on Maven Central as
`io.github.pedronahum:tlaloc-*` (see [GETTING_STARTED.md](GETTING_STARTED.md)).

## What "alpha" means here

Each `0.1.0-alpha` release means: **the engine is certified, the surface is not settled.**

The automated tests pin behaviour: gradients against analytic and
cross-implementation oracles, a real TinyLlama matching HuggingFace token for
token, RNG bit-exact against JAX's threefry stream. None of that pins *names*. A
function whose result is certified correct may still be renamed, moved to another
module, or given a different parameter order in the next alpha.

Version numbers before `1.0.0` carry no compatibility guarantee under SemVer
item 4, and Tlaloc takes that literally.

### Which surfaces are more likely to change

Two mechanisms separate well-tested surfaces from provisional ones.

- **`@ExperimentalTlalocApi`** — a `@RequiresOptIn(ERROR)` marker (declared in
  `:core`) on the part of the surface that is *provisional*, meaning it may change
  **shape**, not merely signature. Three surfaces carry it, each for a stated
  reason: the four-worlds scope taxonomy (`KernelScope`, `OrchestrationScope`,
  `ProgramScope`, `ClusterScope`, `Tlaloc`, `BufferHandle`, `HandleRef`; each op
  is limited to one scope, and Kotlin's context parameters may reshape it); the collective attribute convention (`io.tlaloc.ir.AllReduceAttrs`
  — distributed execution is 📐 in [CAPABILITIES.md](CAPABILITIES.md) and has
  never run on two hosts); and the kernel-choice and cost-model surface
  (`io.tlaloc.ir.recognizer.kernel`, `io.tlaloc.ir.recognizer.cost` — the
  machinery is unit-certified but its *purpose* is picking a kernel, and the one
  kernel measured against XLA lost at small shapes). Touching any of them without
  `@OptIn(ExperimentalTlalocApi::class)` is a compile error that names the marker
  and points here. **`grad`, the op surface and `:nn` do NOT carry it**, on
  purpose: an annotation on everything teaches you to opt in once and stop
  reading.
- **An ABI baseline.** `api/<module>.api` is committed and `./gradlew checkKotlinAbi`
  (wired into `check`) fails on any difference, so a break is a reviewed diff
  rather than a surprise. It covers every published module that emits classes,
  including the Java-25-targeted `:runtime-pjrt`, `:runtime-cuda`, `:kptx`,
  `:runtime-iree` and `:compiler-plugin`.

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

These are the commitments.

- **The group id stays `io.github.pedronahum`.** It changed from `io.tlaloc`
  once, before anything was published. Artifacts may be added or removed, but a
  coordinate that exists will not be re-pointed at different code under the same
  version.
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
- **A break in a published module's ABI leaves a trace.** `api/*.api` is a
  committed baseline and `checkKotlinAbi` runs inside `./gradlew test`, so a change
  to a published module's public ABI cannot land without a matching
  `updateKotlinAbi` in the same commit. Breaks are still allowed; they are visible.

## What a consumer should do about it

- **Pin an exact version.** Never a range, never `latest.release`. There are no
  compatible-upgrade guarantees to lean on.
- **Expect to read the changelog on every bump.** [CHANGELOG.md](../CHANGELOG.md)
  records breaking changes per version; there is no deprecation period in which
  both spellings work.
- **Keep the compiler-plugin version equal to the library version.** The plugin
  synthesizes calls into `:core`/`:ir`; mixing versions is unsupported. The Gradle
  plugin applies the compiler plugin of its own version, and `tlaloc-bom` keeps
  the libraries on one version. Without the Gradle plugin nothing checks that the
  two match; a missing library symbol at compile time is reported as a likely
  version mismatch.
- **Treat the toolchain as pinned too.**
  - **Kotlin: one feature release per Tlaloc version.** A K2 compiler plugin
    binds to compiler internals; a different *feature* release is not expected to
    work and is not tested. The plugin detects the running compiler's version and
    refuses at compile time, naming what it found and the supported range, instead
    of raising a `NoSuchMethodError` from inside `compileKotlin`. Kotlin numbers
    feature releases by tens in the third component and bugfixes by ones above
    them, so each Tlaloc version supports the whole bugfix family of the release it
    is built against and nothing else:

    | Tlaloc | Kotlin |
    |---|---|
    | `0.1.0-alpha01` | 2.3.20 through 2.3.29 |
    | `0.1.0-alpha02` | 2.4.20 through 2.4.29 |

    2.4.10 and 2.4.30 are *different feature releases* from 2.4.20.
    `tlaloc { unsafeAllowUnsupportedKotlin.set(true) }` turns the refusal into a
    warning; a failure inside the compiler is then expected.
  - **JDK: 25 to build, 21 to run.** The library modules (`core`, `ir`,
    `autograd`, `nn`, `stablehlo`, `maestro`) emit Java 21 bytecode and are
    compiled with `-Xjdk-release=21`. The FFM runtime backends (`runtime-pjrt`,
    `runtime-cuda`, `kptx`), `runtime-iree` and `compiler-plugin` emit Java 25.
    Because Kotlin loads a compiler plugin into the compiler's own JVM, a project
    using `grad { }` needs a JDK 25 *on the build machine* whatever it targets.
    The per-module table is in [GETTING_STARTED.md](GETTING_STARTED.md), section 0, and the
    gate is `scripts/jdk21-smoke.sh` plus `verifyJvmTarget`, which reads the
    class-file major version out of every published jar.
- **Symja (`org.matheclipse:matheclipse-core`, LGPL-3.0) is optional from
  `0.1.0-alpha01`.** It is `compileOnly` in `tlaloc-ir`, so it will not appear in
  your graph unless you add it. Adding it is one line,
  `kotlinCompilerPluginClasspath(...)`, because the CAS runs inside the compiler
  and not inside your program, and Tlaloc prints that line at compile time when a
  body needs the CAS. If your policy forbids LGPL in the dependency graph,
  nothing in Tlaloc pulls it in.

## When this policy changes

At `1.0.0`, and not before. At that point SemVer applies without carve-outs and
this document is replaced by the shorter one that says so. Until then, every
alpha release is free to break anything in the "may break" list, and is required
to say in [CHANGELOG.md](../CHANGELOG.md) that it did.
