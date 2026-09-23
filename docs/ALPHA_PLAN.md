# Alpha plan — the ledger for the road to a usable alpha

**Status: TIER 0 COMPLETE (§0.4.498, 2026-09-22). TIER 1 COMPLETE (§0.4.499,
2026-09-22). TIER 2 ITEM 6 COMPLETE — slice 1 (§0.4.500) and slice 2 (§0.4.501),
2026-09-22. TIER 2 ITEM 7 COMPLETE (§0.4.502, 2026-09-22). TIER 3 COMPLETE —
portability and trust (§0.4.503) and the CI matrix (§0.4.504), 2026-09-22; the CI
lanes are 🧪 because no GitHub Actions run of them exists. TIER 4 COMPLETE —
documentation debt and the public API surface (§0.4.505, 2026-09-22). FINAL
VERIFICATION COMPLETE (§0.4.506, 2026-09-22): the whole gauntlet re-run from a
clean room — 2,522 tests, 0 failures, ten examples, the published POMs, both
halves of the framework-free serving path — with one new defect published and one
deferral closed. The rest of TIER 2 IS NOT YET SCOPED IN THIS FILE.**

**The verdict is at the bottom of this file, under
[Final verification (§0.4.506)](#final-verification-04506-2026-09-22): the engine
is an alpha; the *distribution* is not yet, and the three things standing in the
way are a group-id decision, one CI run and one GPG key — none of them
engineering.**

This document is the running record for the arc
that takes Tlaloc from "an engine with 2,345 passing tests that nobody may
legally use" to "an alpha a stranger can depend on". Tier 0 was the legal and
distribution tier: before it, the repository had no `LICENSE` (so, by default,
all rights reserved) and its generated POMs carried none of the metadata Maven
Central mandates, which made the artifacts unpublishable. Both are now fixed and
gated.

Every row carries a mark with the same meaning it has in the README's Maturity
table, because a plan document that grades itself more generously than the README
is how a repository starts lying to itself:

| | |
|---|---|
| ✅ **Certified** | an automated gate pins it, and this file names the gate |
| 🧪 **Written** | the code or document exists and is self-consistent, but the end-to-end path has never run — the reason is always stated |
| 📐 **Designed** | a design exists; no implementation |
| ⬜ **Not started** | |

**On Tiers 1–4.** The arc this file serves has five tiers. The brief that
produced §0.4.498 described Tier 0 in full and named the others only as "Tiers
0–4"; their contents were not in it. Rather than invent five plausible tiers and
publish them as though they were decided, this revision seeds the sections and
says so. Whoever lands Tier *n* fills in Tier *n*'s table here, in the same
commit, with the same three columns. An empty section below means "not written
down yet", never "nothing to do".

---

## Tier 0 — legal and distribution (§0.4.498, 2026-09-22)

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **Apache-2.0 `LICENSE`** | ✅ | The file exists at the repository root, byte-identical to `apache.org/licenses/LICENSE-2.0.txt` (`md5 3b83ef96387f14655fc854ddc3c6bd57`); every published POM declares it in `<licenses>`, which `verifyPomMetadata` asserts | Deliberately the canonical Apache text with the appendix template unmodified — the copyright holder is named in the README's `## License` section and in the POMs' `<developers>`, not by editing the license body. No `NOTICE` file: Apache-2.0 does not require one and Tlaloc redistributes no Apache-licensed source. |
| **The README's LICENSE admission is gone** | ✅ | `grep -c '<!-- LICENSE:' README.md` is 0; a `## License` section replaces it and is linked from the header nav | — |
| **Central-mandatory POM metadata** | ✅ | `./gradlew verifyPomMetadata` (registered in all eleven published modules and wired into each one's `check`, so `./gradlew test` covers it): it **parses** each POM the build generates and fails by name on a missing or empty *direct child* of `<project>` among `<name>`, `<description>`, `<url>`, `<licenses>`, `<developers>`, `<scm>`. All 21 POMs in `~/.m2` after `publishToMavenLocal` were also read by eye. | Parsed rather than grepped for a reason worth recording: `<name>` also occurs inside `<licenses><license>` and `<url>` inside `<scm>`, so a substring check would have passed a POM missing exactly the two elements Central most often rejects. **The gate was negative-tested**: deleting `url.set(...)` from the `pom { }` block makes `:core:verifyPomMetadata` fail with `missing ... element(s) url` and print the twelve elements that *were* present. A module that starts publishing without a `moduleDescriptions` entry fails configuration by name, so the gate cannot be bypassed by adding a module. |
| **Javadoc jar per publication** | ✅ | `dokkaJavadocJar` per module; `verifyPomMetadata` fails any `MavenPublication` with no `javadoc`-classified artifact | **Dokka HTML, not Dokka Javadoc.** Dokka's Javadoc format "doesn't support multi-project builds or Kotlin Multiplatform projects" and ten of eleven published modules are KMP. Central validates that the classifier exists, not its flavour. Named here so it is not a silent substitution. Dokka emits unresolved-KDoc-link warnings across the modules (`[strideH]`, `[io.tlaloc.ir.passes.VjpRegistry.Conv2dRule]` and the like — mostly cross-module references KDoc cannot see); they are real documentation defects, not build failures, and nobody has counted or swept them. |
| **Sources jar per publication** | ✅ | `withSourcesJar()` in `:compiler-plugin`; the KMP plugin supplies the rest; `verifyPomMetadata` fails a module with no `*sourcesJar` task | `:compiler-plugin` was the one publication in the repository that shipped a binary jar alone. It had been that way since §0.4.355. |
| **Dokka wired at all** | ✅ | `dokka = "2.2.0"` in `libs.versions.toml`; `./gradlew :core:dokkaJavadocJar` produces a 2.0 MB jar of real HTML | `DIFFKTX_SPEC.md` §14 listed Dokka as tooling in the original plan and it was never applied. No aggregated documentation site is published anywhere — the HTML exists only inside the javadoc jars, and MkDocs Material (the other half of §14's one-line Docs entry) is still unwired. |
| **Signing** | 🧪 | Configuration is exercised on every build: the `signing` plugin is applied to all eleven published modules and the no-key path is what CI and every local build take today | A key now exists on the development machine and `publishToMavenLocal` writes an `.asc` for every file (115 in `~/.m2` for 0.1.0-alpha01); Central has not yet checked one. In-memory key from `signingInMemoryKey` / `SIGNING_IN_MEMORY_KEY`. The publish-task-depends-on-`Sign` ordering workaround runs on every signed publish. The first real release certifies the rest. |
| **Central publishing repository** | 🧪 | The `central` repository is declared unconditionally, so `publishAllPublicationsToCentralRepository` exists and `--dry-run` shows the graph | **Never run against Sonatype.** No credentials on this machine and a Central upload is irreversible. A missing key or credential makes the upload refuse by name before sending anything (release-readiness R7); `releaseToCentralPortal` adds the Portal handoff. |
| **`io.github.pedronahum` namespace on Central** | ✅ decided, 🧪 unverified | The coordinate is pinned by `scripts/onboarding-smoke.sh`, which publishes and consumes it end to end; `verifyPomMetadata` × 11 pins the POMs | **Decided in §0.4.508.** `io.tlaloc` would have needed a DNS TXT record on `tlaloc.io`, which this project does not own; `io.github.*` verifies against the GitHub account already hosting the repository, so no domain is involved. 100 coordinate sites across 39 files rewritten; `DIFFKTX_SPEC.md` deliberately keeps the old coordinate as a record of what was true then. **Still 🧪:** the namespace has not been verified in the Central Portal and nothing has been uploaded. See [RELEASING.md](RELEASING.md) §4. |
| **Version `0.1.0-alpha01`** | ✅ | `bash scripts/onboarding-smoke.sh` — it publishes to `mavenLocal` and builds + runs `examples/quickstart` against the new coordinate, so a half-done rename fails it | 16 files named the old `0.0.1-SNAPSHOT`: the root build, `README.md`, `docs/GETTING_STARTED.md`, `examples/README.md`, two example READMEs and ten example build files. |
| **`CHANGELOG.md`** | ✅ | The `0.1.0-alpha01` entry exists and names what changed, including two claims that turned out to be false | Does not retro-summarise §0.4.1–§0.4.497; that record is in commit titles and (to §0.4.311) `DIFFKTX_SPEC.md`. |
| **Compatibility policy** | ✅ | [COMPATIBILITY.md](COMPATIBILITY.md), linked from the README's alpha blockquote and from `CHANGELOG.md` | States what alpha means, the seven things that may break without deprecation, and the five that will not. One gap it names rather than fixes: **nothing checks that the compiler-plugin version matches the library version** — there is no such guard in the plugin today. |
| **Release procedure** | 🧪 | [RELEASING.md](RELEASING.md) | Every step is executable except the two that talk to Sonatype; those two have never run. |
| **Symja's license** | 🧪 | The artifact's own POM (`matheclipse-core-3.1.1.pom`) declares `GNU Lesser General Public License, Version 3` | **Open, and now written down instead of assumed.** The upstream *repository's* `license.txt` is plain **GPL-3.0**; upstream's position is that the published maven modules are LGPL while the repository as a whole is GPL. Tlaloc links and never patches, which LGPL permits. Two of the repository's own files still called it Apache-2.0 five months after §0.4.13 corrected the rest (`SymbolicEngine.kt`'s KDoc and `STAGE_B_PLAN.md` §3.2.1's bullet list) — both fixed in §0.4.498. **Residual risk a release should close:** `ir-jvm`'s POM carries Symja at `runtime` scope, so a consumer who cannot take a copyleft dependency inherits it, and no Symja-free `SymbolicEngine` implementation exists to opt out to. |
| **An actual Maven Central release** | ⬜ | — | Cannot be done from here; see the two rows above. This is the row that turns 🧪 into ✅ for signing, the Central repository and `RELEASING.md` at once. |

### What Tier 0 did not touch, on purpose

- **`DIFFKTX_SPEC.md` gets no new §0.4 entry.** The book-of-work record has lived
  in commit titles since §0.4.311, 187 commits ago; this file is where the arc is
  recorded instead.
- **The engine.** Tier 0 changed one line of Kotlin (`SymbolicEngine.kt`'s KDoc,
  a comment) and no behaviour. The suite is the same suite.
- **The Dokka warning sweep.** The unresolved-KDoc-link warnings are real and
  unaddressed, and their number is unmeasured. They belong to a documentation
  tier, not a legal one.

---

## Tier 1 — first-contact defects (§0.4.499, 2026-09-22)

The two things a stranger meets before they meet the engine: what Tlaloc puts in
their build log, and what it does when it cannot lower their lambda. Both were
wrong in the same direction — informational output dressed as a warning, and a
real failure dressed as a warning — and the second one made the first one fatal.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **A working `grad {}` produces no Tlaloc output at all** | ✅ | `DiagnosticNoiseTest.the default build emits neither the FIR dxir dump nor the IR handoff dump`, plus the reproduction by hand: `./gradlew -p examples/quickstart compileKotlin --rerun-tasks` printed two warnings and eight lines of IR before this tier, and prints nothing after it | The two messages were `TlalocErrors.LAMBDA_LOWERED` (FIR checker) and the IR extension's "saw handoff". Both are now gated on the new `dumpLoweredIr` option; the IR half also dropped from `WARNING` to `INFO`. |
| **A consumer using `-Werror` can compile at all** | ✅ | `DiagnosticNoiseTest.a correct grad consumer compiles under -Werror and emits no Tlaloc diagnostic` — it sets `K2JVMCompilerArguments.allWarningsAsErrors` and asserts exit code 0 **and** that no message mentions Tlaloc. Verified end to end outside the suite too, with a Gradle init script setting `allWarningsAsErrors = true` on `examples/quickstart`: `BUILD FAILED … e: warnings found and -Werror specified` before, `BUILD SUCCESSFUL` after | This is the defect that mattered: a routine Kotlin-shop setting made Tlaloc impossible to adopt, caused entirely by Tlaloc's own informational output. |
| **The lowered IR is still reachable** | ✅ | `DiagnosticNoiseTest.dumpLoweredIr brings both dumps back, and the IR half is an INFO not a warning`; and five existing harnesses (`TlalocPluginDiagnosticTest`, `ContractInferenceTest`, `NamedIndexResolutionTest`, `MatmulRecognitionTest`) now pass the option explicitly and keep asserting on the dump contents | **Named limitation:** with `dumpLoweredIr=true` the FIR half is still a `WARNING`, so that option and `-Werror` cannot be combined. K2's diagnostic DSL (`KtDiagnosticFactoryDsl.kt`, kotlin-compiler-embeddable 2.3.20) offers `error*`, `warning*`, `strongWarning` and `deprecation` only — there is no `info*` factory, and a FIR checker has no `MessageCollector`. Moving that half to the IR phase (where INFO is available, as `dumpGradSource` already is) would lose the FIR source location unless the location is carried through the handoff table; that is the follow-up, and it is not done. |
| **`dumpGradSource` / `dumpGradSourceDir` still behave exactly as documented** | ✅ | `DumpGradSourceTest` (unchanged, including its raw-bit-identical standalone-compile pin); `examples/readable-gradients` builds green against the republished plugin | — |
| **An unlowerable lambda refuses at COMPILE time, by name** | ✅ | `DiagnosticNoiseTest.an unlowerable grad body is a compile-time ERROR naming the construct and the opt-out` — asserts a non-zero exit, exactly one `ERROR`, the lowering's own reason (which names the offending `cube` call) and the opt-out flag's spelling | New diagnostic `TlalocErrors.LAMBDA_NOT_LOWERABLE` (`error1`). `LAMBDA_UNSUPPORTED` (warning) survives as the opt-out's diagnostic, so the ~208 distinct `LoweringException` reasons keep flowing through both, unflattened. |
| **The opt-out works and is named everywhere it is mentioned** | ✅ | `DiagnosticNoiseTest.strictLowering=false restores the pre-alpha warning and the build stays green`; `CustomVjpGradientTest` and `RngGradientTest` (whose whole point is a named lowering refusal followed by a tape fallback) pass the flag and are otherwise untouched | `-P plugin:io.tlaloc.plugin:strictLowering=false`. It is named in the error text itself, in `docs/GETTING_STARTED.md` §4a, in the README, in `examples/differentiable-physics`'s README and source comment, and in the runtime exception. |
| **A non-lambda argument (`grad(::f)`) refuses by name** | ✅ | `DiagnosticNoiseTest.a non-lambda argument is refused by name instead of failing at the first call` | It used to emit `TLALOC_INTRINSIC_CALL`, whose text still promised "the K2 plugin will replace this with a dxir transform in a later step" — stale since §0.4.4. That spelling now only appears under `dumpLoweredIr` with the opt-out on. |
| **The Tracer tape route is not collateral damage** | ✅ | `TlalocPluginTracerFallbackTest` runs in DEFAULT (strict) mode and still produces the real gradient `32.0` end to end | `io.tlaloc.autograd.grad`/`grad2`/`grad3`/`valueAndGrad*` are **overloaded**: the compile-time intrinsic (`GradIntrinsics.kt`) and the runtime `Tracer` tape (`Grad.kt`) share those names, and the checker matched the FQN alone. Every tape call was therefore drawing a bogus `could not lower lambda: … unsupported type io.tlaloc.autograd.Tracer` warning; promoting that to an error would have broken the documented plugin-free route. A Tracer-typed lambda parameter is now recognised as the tape overload and left alone. **This was found by the tier, not by the brief.** |
| **The runtime message distinguishes the two ways it can fire** | 🧪 | Read by eye; no test asserts the text | `pluginMissing` used to say "requires the Tlaloc K2 compiler plugin" even when the plugin was applied and had simply refused the body. It now names both states and points at the compile-time reason for the second. **What it still cannot do is TELL THEM APART**: a JVM at runtime has no way to observe whether a K2 plugin was applied to the module that compiled the call, and the plugin leaves an unlowered call byte-identical to one compiled with no plugin at all. Making it genuinely detectable means rewriting the refused call site to a reason-carrying stub — that is IR synthesis of a function-typed value, the same machinery `DxirToIrSynthesis` does for real gradients, and it was out of scope here. Recorded rather than faked. |
| **Options are refused by name when misspelled** | ✅ | `DiagnosticNoiseTest.an unknown value for a boolean plugin option is refused by name` | `dumpGradSource` predates this and still reads any non-boolean value as `false` (`value.toBooleanStrictOrNull() ?: (value == "true")`). Left as it was rather than changed under this tier's heading — a behaviour change to a documented option belongs to whoever owns that option's row. |

### What Tier 1 did not touch, on purpose

- **The other IR-extension warnings.** `"kept original call — …"` and the
  `DxirReverseTransform rejected the dxir` warnings are still `WARNING`s, and
  deliberately: they announce a real degradation (the lambda lowered, then
  synthesis refused it), which is exactly what a warning is for. They do mean a
  `-Werror` consumer whose lambda takes the synthesis fallback still fails the
  build. Nobody has decided whether that is wrong; it is not what this tier
  fixed, and it does not fire for a `grad {}` that works.
- **`TLALOC_INTRINSIC_CALL`'s stale renderer text** ("the K2 plugin will replace
  this with a dxir transform in a later step" — the later step shipped in
  §0.4.4). The message is now reachable only under `dumpLoweredIr` **and**
  `strictLowering=false`, so it was left rather than rewritten.
- **The FIR-phase `MessageCollector` question.** See the `dumpLoweredIr` row.

## Tier 2 — item 6, slice 1: constant folding for captured compile-time constants (§0.4.500, 2026-09-22)

The largest engineering item in the arc, split into two sequential slices. **This
row set is slice 1 only.** Before it, a `grad { }` lambda could reference *nothing*
declared outside itself: `FirLambdaToDxirLowering.lookupReference` resolved a
`FirPropertyAccessExpression` against the lowering's own `env` — lambda
value-parameters and lambda-local `val`s — and every other reference fell through
to `throw LoweringException("reference to symbol outside the lowering scope: …")`.
A top-level `const val`, a file-level `val`, a class property and a parameter of
the enclosing function were all the same refusal. That is why
`examples/differentiable-physics` carried a README section titled "A limitation you
will meet immediately" and why every number in its simulator was an inlined literal
with the constant's name in a trailing comment.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **A captured `const val` folds into the lowered DXIR** | ✅ | `CapturedConstantGradientTest` (12 tests). Seven of them are EQUIVALENCE tests: each compiles and runs the same program twice — captured constant vs. inlined literal — and requires byte-identical stdout, plus (where the derivative is a round number) the analytic value. "It compiled" is not the claim. | The fold is [`constFromLiteral`], pulled out of `lowerLiteral` so both spellings emit through the same three lines; that identity is the whole design, not an optimisation. Covered positions: scalar arithmetic, a `:core` tensor op operand (the mixed-rank `splatLiteral` path, which is only taken *because* the folded node is a `DxirConst`), an `if` condition and both branch bodies, a `for` loop body, and a loop trip count. |
| **The fold is indistinguishable downstream** | ✅ | `examples/differentiable-physics`: the 832-line gradient source the compiler dumps from the `const val` body is `md5 5d0c2b9704f12a2863938df08f2cae61` — the same file, byte for byte, that the inlined-literal body produced. Verified by building both and diffing. | Only the dump's *filename* changed (`Main_kt_128_23_…` → `Main_kt_131_23_…`), because the call moved three lines down the file. The example's verbatim expected output in its README is unchanged and was re-diffed against a real run. **This is the interesting result**: the arithmetic, the coarsening and the synthesized bytecode are the same. |
| **An `Int const val` trip count takes the unrolled path, not the symbolic one** | ✅ | `CapturedConstantGradientTest.a captured Int const val as a for-loop trip count takes the same unroll as the literal` | `extractForLoopTripCount` folds a constant bound to `ForLoopBound.Concrete`, the same case an `Int` literal produces. Falling through to `ForLoopBound.Expression` would have compiled `0 until STEPS` to PhiCalculus's C6 *symbolic* trip-count shape and `0 until 38` to the C5 unroll — two different gradient bodies for a difference in spelling. |
| **A `const val` with a computed initializer folds** | ✅ | `CapturedConstantGradientTest.a const val whose initializer is itself an expression folds` (`const val HALF_DT = DT / 2.0f`) | Resolved through the Kotlin compiler's own `FirExpressionEvaluator`, never by re-parsing source. That needs a `FirSession`, which `lower` now takes and stashes in a save/restore ThreadLocal beside the existing `customVjpDefsTl`. `evaluateExpression` is `@PrivateConstantEvaluatorAPI`, opted into explicitly, and wrapped in `runCatching`: its visitor `error(…)`s on FIR shapes it does not model, and a refusal is a better answer than a compiler crash. |
| **A top-level / enclosing-function `val` folds too** | ✅ | `CapturedConstantGradientTest.a top-level val with a literal initializer folds, not only a const val` | Gated on `!isVar && !hasDelegate && (isConst || callableId?.classId == null)`. The `classId == null` arm covers a top-level `val` and a `val` local to the enclosing function; both are single-assignment with a fixed initializer, so the folded value is the value the lambda would have read. A non-`const` MEMBER `val` is deliberately excluded — an instance's value, and possibly an override's. |
| **A captured runtime value refuses with its own wording** | ✅ | `CapturedConstantGradientTest`, four negative tests: a `var`, a computed `val`, a parameter of the enclosing function, a non-`const` member `val`. Each asserts the capture is named, the reason is named, the phrase `captured RUNTIME values are not yet supported` is present, and the old `outside the lowering scope` text is **absent**. | This distinction is the contract slice 2 turns on. The old sentence survives only for a symbol that is neither a property nor a value parameter (an enum entry, say), where it is still the accurate thing to say. |
| **f64 constant folding produces a Double** | ✅ | `ir`: `DxirConstFoldDtypeTest` (4 tests) — every f64-typed folded const holds a `Double`; the f64 product is the *double-precision* product and not the f32 one widened; the f32 arm is unchanged; the `MUL(x, 0)` short-circuit's hard-coded `0.0f` is covered too. Plus `CapturedConstantGradientTest.a captured Double const folds at f64, not at f32`. | **Not a capture bug — found by the capture tests.** See "What this slice found that was not in its brief" below. |
| **Slice 2: runtime capture** | ✅ | Landed in §0.4.501 — see the Tier 2 slice 2 section below | It became a trailing input-only PARAMETER of the lowered function rather than an extra COARSENED operand: the gradient has no slot for it, which is exactly the problem `CapturingEnv` (§0.4.415) refuses for the customVjp case, and a parameter the reverse transform is told to skip is the shape that has one. |
| **A captured constant of an unsupported dtype** | 🧪 | Read by eye; no test reaches it | `foldCapturedConstant` refuses by name when the folded literal is not Float/Double/Int/Long. That arm appears to be **unreachable from type-correct Kotlin** today: the lowering's expression surface is numeric, so a `String`, `Char`, `Boolean`, `Byte` or `Short` constant cannot appear in a position the lowering lowers — the read that would reach it (`LABEL.length`) refuses one link earlier, at `length`. Pinned instead: `CapturedConstantGradientTest.a capture chain refuses at the first link the compiler cannot fold`. Kept as a defensive branch, and named here rather than presented as certified. |
| **A captured `Long` constant** | ⬜ | — | `literalDType` maps `Long → I64` and the fold would emit it, but no type-correct `grad { }` body reaches a captured `Long`: the scalar surface is Float/Double, and a `0 until N` bound with `N: Long` resolves to `LongRange`, which `extractForLoopTripCount` does not recognise. Uncertified and unclaimed. |
| **Positions that read a `FirLiteralExpression` directly** | ⬜ | — | Some lowering paths pattern-match on a literal ARGUMENT rather than lowering it (axis and shape arguments, some `pow` exponents). A captured constant in one of those positions refuses with that path's own message, not with the fold. Not swept, not counted, and not part of this slice. |

### What this slice found that was not in its brief

**`grad { x: Double -> x * 1.5 }` crashed the compiler.** Not a captured constant
anywhere — an inlined literal in an f64 scalar body. The failure surfaced as

```
error: org.jetbrains.kotlin.fir.pipeline.IrGenerationExtensionException:
class java.lang.Float cannot be cast to class java.lang.Double
```

with **no Tlaloc diagnostic of any kind**, which is the one outcome this
repository's house rules do not allow: it neither produced a gradient nor refused
by name. Cause: `DxirReverseTransform.applyConstFold`'s `asFloatConst` projects
every operand to `Float`, and every fold built its replacement as
`DxirConst(id, <Float arithmetic>, n.type)`. On an f64 node that is a constant whose
type says `f64` and whose value is a `java.lang.Float`. Nothing in the IR checks
value classes — not the pretty printer, not `validateDxirShapes` — so it travelled
three phases and two modules to `DxirToIrSynthesis`'s
`IrConstImpl(…, IrConstKind.Double, v as Double)`.

It was found by the *inlined-literal control* of the Double equivalence test, which
is the argument for writing the control at all. The f64 arm of the fold now does its
arithmetic in `Double`; the f32 arm is the pre-§0.4.500 expression untouched, and
`DxirConstFoldDtypeTest` pins that separation. **`f32` numerics did not move.**

Two things this leaves open, named rather than fixed:

- **No invariant check.** Nothing asserts that a `DxirConst`'s value class matches
  its `DxirType.dtype`. `validateDxirShapes` would be the place; adding it would
  turn this class of defect into a named refusal at the checker instead of a
  ClassCastException in the backend. Not done.
- **Other constant producers are unaudited.** This fold is fixed; PhiCalculus's
  unroll, the Symja engine's constant emission and `zeroValueFor`/`seedValueFor`
  callers were not swept for the same mistake. Only f64 scalar bodies are affected,
  which is a narrow lane, but it is not zero.

## Tier 2 — item 6, slice 2: captured RUNTIME values as synthesized parameters (§0.4.501, 2026-09-22)

Slice 1 made a captured *compile-time constant* fold. This slice turns on the
values it deliberately left refusing: a runtime value the compiler cannot know.

```kotlin
val scale = computeScale()                     // a runtime value
val g = grad { x: Float -> f(x) * scale }      // now lowers
```

**The mechanism, in one sentence:** a captured runtime value becomes a TRAILING
parameter of the lowered `DxirFunction`, and the IR phase binds that parameter at
the call site to an `irGet` of the very declaration the user's lambda closed over,
so the synthesized gradient closes over it exactly as the user's lambda did and
the JVM backend's own closure conversion carries it.

**The design decision worth recording** is that the lowering RE-RUNS. A capture is
discovered mid-body, and a parameter appended mid-body would carry an SSA id
allocated after some of the body's, making the lowered function subtly different
from the same lambda written with that value as a trailing parameter.
`FirLambdaToDxirLowering.lower` therefore catches the discovery, adds it to a list
and lowers the lambda again from the top with the parameter declared up front —
one extra pass per distinct capture, on a pure function of the FIR. What that buys
is the oracle: the dxir a capture produces **is** the dxir the explicit-parameter
spelling produces, so every equivalence test below is testing a property the
design guarantees rather than a coincidence.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **A captured runtime `Float` / `Double` lowers and gives the right gradient** | ✅ | `CapturedRuntimeValueGradientTest` (16 tests). Every positive test is a THREE-WAY equivalence: the capture, the same body with that value as an explicit trailing lambda parameter (`grad2 { x, s -> … }.first`), and the same body with the value inlined — all three compiled, run, and required to print identical stdout, plus the analytic value. | The explicit-parameter spelling is the oracle that matters; the inlined one catches a fold/bind that agreed with itself while being wrong about the number. |
| **`Int` and `Long` captures too** | ✅ | `a captured runtime Int reaches the body through the same cast a param does`, `a captured runtime Long reaches an f64 body the same way` | They reach the body through the same `CAST_OP_MAP` path (`n.toFloat()`) a declared `Int` parameter does. Non-differentiable, so the reverse transform's typed zero for them is what gets dropped. |
| **The returned function's arity does not change** | ✅ | Three tests: `grad2 with a capture still returns exactly two gradients`, `valueAndGrad with a capture still returns the value and one gradient`, and the base case where `grad`'s captured spelling returns a bare `Float` while its explicit-parameter twin returns a `Pair`. The test stubs' declared return types would not type-check if the synthesized lambda's arity moved, and the IR extension's type-match guard would then drop the rewrite — which the sentinel check catches. | This is the failure that would have been silent and repository-wide. The mechanism: `DxirReverseTransform.apply` takes `inputOnlyTrailingParams`, emits no gradient for that many trailing params, and the adjoint chain that fed them becomes unreachable and is dropped by the pass's existing `dropUnreachableBody`. |
| **The value is read at the CALL, not baked in** | ✅ | `a captured parameter of the enclosing function is read at the call, not baked in` — one `grad { }` call site inside `fun build(scale: Float)`, two closures built from it with different `scale`s, three gradients that must differ accordingly | A capture that was folded, cached per call site, or bound once would print the same number twice. |
| **A value captured twice is ONE parameter** | ✅ | `a capture read twice is ONE parameter, read twice` — the numbers *and* the lowered dxir: the signature must be exactly `fn grad_body(%0: f32, %1: f32) -> f32` | Free by construction: the capture is bound into `env` under its FIR symbol, so the second reference resolves through the same `env` hit a lambda parameter does. Pinned anyway, because the numbers alone could not tell the two cases apart. |
| **Two captures keep their first-reference order** | ✅ | `two captures in one body keep their first-reference order` — `(x + a) * b`, whose gradient is `b`; a swapped binding returns `a` instead | |
| **A capture survives the loop coarsening** | ✅ | `a capture inside a for loop survives the coarsening` — a constant-trip-count loop over `d = d * scale`, whose gradient is `scale³` | The shape `examples/differentiable-physics` is made of: PhiCalculus unrolls (C5) before the reverse transform runs, and the captured param has to travel through intact. |
| **The printed gradient still compiles and runs standalone** | ✅ | `the printed gradient carries the capture as a parameter and runs standalone` — the dumped `.kt` is compiled with NO plugin, called with the captured value in the trailing slot, and required to be raw-bit-identical to the plugin-compiled gradient | The renderer needed no change: it renders every param, so the capture appears as a parameter — which is what it *is* in the dxir. The test also asserts the printed return type is not a `Pair`/`Triple`, i.e. the arity claim above holds in the printed source too. Documented in [READABLE_REVERSE.md](READABLE_REVERSE.md). |
| **A captured `var` refuses by name** | ✅ | `a captured var refuses by name and says it must be immutable` | The gradient is derived where the lambda is written but runs where it is called, so a mutable capture has no single value to bind. Binding it to the declaration (which is what the IR phase does) would in fact read the value at call time — but nothing in the lowering *checks* that the body's meaning is unchanged by a reassignment between the two points, so this refuses rather than guesses. |
| **A captured top-level / member property refuses by name** | ✅ | `a captured top-level val with a runtime initializer refuses as a property`, `a captured member property with a runtime initializer refuses as a property` | Reading one is a getter CALL, not a value declaration: there is no `IrVariable` / `IrValueParameter` to bind to, and a member's receiver is not knowable in the lowering. A separate slice would resolve the property's getter symbol through `IrPluginContext.referenceProperties` and emit the call; it is not this one. |
| **A capture of an unsupported type refuses naming the surface** | ✅ | `a captured value whose type is outside the surface refuses naming the surface` (a `:core` `FloatScalar` box) | The capture type surface is deliberately NARROWER than the declared-parameter surface: a value-class scalar param enters synthesis through the §0.4.414 call-site unwrap and a captured param has no call-site slot to read the box from; a captured tensor would need its `IrType` from the same absent slot, and the axis-matching machinery indexes the user's params positionally. Both named here rather than half-supported. |
| **A capture in a non-`grad` intrinsic refuses by name** | ✅ | `a capture in a forward-mode intrinsic refuses and names the grad family` | `jvp` / `jacobian` / `hessian` / `vjp` and their arity-2 spellings build their own parameter lists out of the lowered one (primals ++ tangents, a seeded rotation, a runtime basis loop), so a trailing capture param there would change what the returned function takes. Gated in the FIR checker (`captureCarryingIntrinsics`) and independently in the IR extension (`CAPTURE_CARRYING_INTRINSICS`), which refuses to rewrite rather than trusting the other half. |
| **A capture the IR phase cannot bind keeps the original call** | 🧪 | Read by eye; no test constructs the case | If the declaration's source offset matches nothing in the file's IR, or matches ambiguously, the extension emits a named WARNING and leaves the call as written (→ the `pluginMissing` refusal at the first call). Every capture the FIR side admits has a real `IrVariable` / `IrValueParameter` at the same source offset — the two phases share offsets — so no test could reach this without corrupting the handoff. It exists so that a future FIR-side widening cannot silently bind the wrong value. |
| **A captured TENSOR** | ⬜ | — | Refused by the type surface above. It is the natural next slice (`grad { x -> x matmul weights }` with runtime `weights`), and it needs the call-site `IrType` harvest and the tensor-template machinery to stop being indexed by call-site position. |
| **A captured value-class scalar (`FloatScalar` / `DoubleScalar`)** | ⬜ | — | Same absent call-site slot; the §0.4.414 unwrap would have to take its type from the declaration instead. |
| **A capture under the SOI coarsening path** | ⬜ | — | `tlaloc.soi.enabled=true` routes `grad` through `PhiCalculus.coarsenFunction`, whose `gradient_body` has the signature `(upstream, *params) → (*grads)`. The trailing-param drop is applied to the OUTER returns, so the inner body still computes an adjoint for the capture that nothing reads. It should be correct and is not tested: the property is off by default and no test in the repository sets it together with a capture. |
| **More than 8 captures in one lambda** | ✅ | The bound is enforced by name (`MAX_RUNTIME_CAPTURES`), not by looping forever | One re-lowering pass per distinct capture, so the bound is a bound on passes. Nothing certifies the *message*; it is a refusal, not a feature. |

### What slice 2 did not change

- **Slice 1's identity claim.** The FOLD still runs first: a captured `const val`
  emits the same `DxirConst` an inline literal emits and never becomes a
  parameter. `CapturedConstantGradientTest`'s twelve tests are untouched and still
  pass, including the four negative ones — whose refusal *text* changed for the
  `var` and enclosing-parameter cases, which is why those assert on the name and
  the reason rather than on a whole sentence.
- **Any lambda that captures nothing.** `captures` is empty,
  `inputOnlyTrailingParams` is 0, `capturedBindings` is empty, and every branch
  below those is the pre-§0.4.501 code. That is the reason the rest of the suite
  is untouched.
- **A name collision between a capture and a lambda parameter** is unreachable
  rather than handled: if the two had the same name, Kotlin's shadowing would
  resolve the reference to the lambda parameter and there would be no capture.
  The printed source would not compile if it were reachable, so it is recorded
  here as reasoned, not tested.

### What slice 2 found that was not in its brief

**A reference that misses the lowering's `env` is not the same thing as a
capture.** The first full-suite run after the feature worked turned
`TlalocPluginDiagnosticTest."break-bearing while with body-local break cond falls
back to runtime tape"` into a TWO-parameter lowering of a ONE-parameter lambda. A
`val` declared in a WHILE body and read from the trailing `if (cond) break` is
lowered in the CONDITION region (the §0.4.50 LAND-hoist), where the body's
bindings do not exist — so it misses `env` in exactly the place a genuine capture
does, and slice 2 was about to promote it to a gradient parameter bound to an
`IrVariable` declared INSIDE the lambda. That `IrVariable` is not in scope where
the synthesized lambda is built. In this particular program the reverse transform
happened to reject the function a step later, so nothing wrong was emitted; that
was luck, not a gate.

The gate is the lambda's own source range, checked on the FIR side, plus an
independent IR-side check that the declaration is written before the intrinsic
call (Kotlin has no forward reference to a local, and the call's range covers the
lambda). Both must agree before an `irGet` is emitted. Pinned by
`CapturedRuntimeValueGradientTest.a value declared INSIDE the lambda is not a
capture and refuses as one is not`, which asserts the refusal names `delta` and
says it is not a capture — and by the diagnostic test that found it, which is
unchanged.

### Suite state at slice 2 close

| | |
|---|---|
| `./gradlew test --rerun-tasks` | BUILD SUCCESSFUL, 115 tasks |
| `bash scripts/count-tests.sh` | 2385 (2368 at §0.4.500 + 17 new) |
| `bash scripts/onboarding-smoke.sh` | passes |
| `./gradlew -p examples/differentiable-physics run` | SWISH; act [1] 8.401e-04, act [5] 9.180e-04 |
| `./gradlew -p examples/readable-gradients run` | compiled == printed, raw-bit identical |

## Tier 2 — item 7: model persistence and the training utilities around it (§0.4.502, 2026-09-22)

Before this section, Tlaloc could train a model on a GPU — certified, 600 Adam
steps on a GB10 — and could not save the result. Three files said so in three
places: `nn/.../Components.kt`'s KDoc ("minus `store`/`load`, out of scope v1"),
`core/.../Safetensors.kt` (a reader with no writer), and
`docs/MODEL_LAYER_PLAN.md`'s deferral list item 9. The review that produced this
tier also named two missing training utilities: learning-rate schedules beyond
`SGD`'s own `lrDecay`, and gradient clipping.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **A safetensors WRITER in `:core`** | ✅ | `SafetensorsWriterTest` (15 tests): round trip through the §0.4.468 reader for F32/F64/BF16/I32, rank-0, zero-element, raw-bit preservation of `-0f`/NaN/±∞, the 8-byte data-buffer alignment swept over 20 header lengths, per-tensor natural alignment, byte-for-byte determinism, and a tensor name carrying `"`, `\`, `\n` and a control character | `SafetensorsWriter.encode` is a pure function of (tensors, metadata): tensors are ordered by DESCENDING dtype width then name, and the header is space-padded so `8 + headerLength` is a multiple of 8. Both are recorded with their reasons in the file — the ordering is what makes every tensor's offset naturally aligned to its own element width, which a zero-copy reader (`torch.frombuffer`, `np.ndarray(buffer=…)`) requires and name-ordering would not give. |
| **The writer is certified against the REFERENCE implementation, both ways** | ✅ | `SafetensorsWriterOracleTest` (3 tests, **they ran**): Tlaloc writes → the `safetensors` Python library reads → compared as RAW HEX BYTES per tensor; the library writes → Tlaloc reads → compared bit for bit; and a three-hop round trip. Driver: `harness/python/safetensors_writer_oracle.py` in the frozen oracle venv (`safetensors` 0.8.0 + torch 2.11.0+cpu). Nothing was installed and no venv was modified. | Hex, not floats, deliberately: a float comparison passes over a NaN payload, a signed zero and a bf16 that was widened and re-narrowed. torch is needed only for the BF16 leg — numpy has no bfloat16 dtype. Self-skips BY NAME when the interpreter, the library, torch or the script is missing. |
| **Refused dtypes stay refused, by name** | ✅ | `SafetensorsWriterTest.theDtypesWithNoHostStorageAreRefusedByName`, which also asserts the writer's table and the reader's table AGREE (`mapDType(wireDType(dt)) == dt` for all four) | F16 and the fp8 pair have no host representation in Tlaloc at all (there is no `F16` DType); `I64` and `Bool` are DTypes with no host storage class. `wireDType`'s `when` has **no `else`** on purpose: `DType` is sealed, so a new dtype breaks this file at compile time instead of falling into a runtime refusal nobody sees. |
| **Named deferrals in the writer** | 📐 | — | **Sharded output** (`model.safetensors.index.json`, which `SafetensorsIndex` already READS) and **streaming output**. `encode` materialises the whole file and refuses BY NAME past the 2 GB `ByteArray` ceiling rather than throwing `OutOfMemoryError` from inside a copy loop. Both recorded at the top of `SafetensorsWriter.kt`. |
| **A jvm file writer that cannot leave a half-written checkpoint** | ✅ | `CheckpointFileTest.theWriteLeavesNoPartialFileBehindAndOverwritesInPlace` — after two saves to one path the directory holds exactly one file | `SafetensorsFileWriter` writes a sibling temporary and `ATOMIC_MOVE`s it, because a checkpoint is written at the one moment a run is most likely to be interrupted and a truncated file whose HEADER is complete looks valid to anything that only reads the header. `AtomicMoveNotSupportedException` falls back to a plain replace, and the fact that this loses the guarantee is stated in the code rather than swallowed. |
| **Save and load a model** | ✅ | `ModelCheckpointTest` (30 tests). The headline: train 20 Adam steps, save, build a FRESH untrained model, restore into it, and require every parameter scalar's RAW BITS to match, the loss through the same captured graph to match bit for bit, and every gradient to match bit for bit. The fresh model is asserted NOT to already equal the trained one, so the control cannot be broken. | One safetensors file, keys `param.<key>` / `buffer.<key>` / `opt.<slot>` plus `__metadata__`. Loading returns a NEW model through `Trainable.withParameters` — there is no `load(into:)` and nothing mutates, because a training step already returns a new model and a mutating loader would be the only thing in `:nn` that did not. |
| **Resuming training actually resumes it** | ✅ | `ModelCheckpointTest.continuingTrainingFromTheLoadedStateMatchesContinuingFromTheOriginal`: after the fork, 15 more steps on each branch produce bit-identical losses and bit-identical parameters. Plus one test per shipped optimizer (`Adam`, `SGD`+momentum, `SGD`+`lrDecay`, `RMSprop`, `FixedLearningRate`, `Scheduled(Adam)`). | `CheckpointableOptimizer<S>` is a SEPARATE interface from `Optimizer<S>`, not three defaulted members on it: a default that throws is a member every implementer inherits and nobody is told about. |
| **…and NOT resuming it is measurably different** | ✅ | `ModelCheckpointTest.resumingWithoutTheOptimizerStateGivesADifferentTrajectory` — the negative result that makes the row above mean something. Same weights, fresh Adam state: the loss is identical and the UPDATE is not. Observed: 37 of 49 parameter scalars move differently on the first step after the fork, because bias correction at `t = 1` divides the first moment by `1 − β₁ = 0.1`. | Without this test, "optimizer state round-trips" would be a claim about a file rather than about a training run. |
| **BatchNorm's running statistics survive** | ✅ | `ModelCheckpointTest.aBatchNormsRunningStatisticsRoundTripThroughACheckpoint` — and what is compared is the FROZEN inference affine (`inferenceMode()`'s `m` and `b`), because that is what the statistics are *for*: a reloaded model must serve the same numbers. Plus `aFreshBatchNormsBuffersRoundTripToAModelThatStillRefusesInference`, where `runningN = 0` survives as itself and both sides refuse inference by name. | New interface `Stateful<T>` in `Components.kt`, deliberately separate from `Trainable`: `parameters` is the list the reverse transform returns one GRADIENT per, and running statistics have none, so putting them there would make every optimizer step demand a gradient for a quantity that has not got one. `runningN` is a RANK-0 tensor because safetensors has a rank-0 shape. `momentum` is NOT a buffer — it is structure. |
| **The container contract for `Stateful`** | ✅ | `Sequential` implements `Stateful` with the same `"<index>."` prefix its parameters use; `CheckpointFileTest.checkpointTensorsNamesEveryTensorTheFileWouldHold` pins the exact key list for a `Dense → ReLU → BatchNorm` stack | A container that can hold a `Stateful` child must be `Stateful` and prefix its children's keys, or a save silently drops their state. In this module `Sequential` is the only such container, and that is a TYPE fact rather than a review finding: `GRU` holds `Dense`s, `Conv2dWithSamePadding` holds a `Conv2d`, `EmbeddingBag.WithOffsets` holds an `EmbeddingBag` — none of those is `Stateful`. A USER-written container is on the contract, which is stated in `Stateful`'s KDoc and in `ModelCheckpoint`'s. |
| **Every layer kind round-trips** | ✅ | `Conv2d`, `Embedding`, both `GRU` candidate-gate variants, `AffineTransform`, a bias-less `Dense`, and a `Conv2d → ReLU → BatchNorm → Flatten → Dense` stack, each through a generic `assertParametersRoundTrip` | — |
| **The file is an ordinary safetensors file** | ✅ | `ModelCheckpointTest.aCheckpointIsAnOrdinarySafetensorsFileAnyReaderCanOpen` (read back through `Safetensors`, which knows nothing about `:nn`); `CheckpointFileTest.theSavedFileIsReadableByTheFormatReaderWithNoKnowledgeOfNn`; and, outside the suite, the real `examples/gpu-training` checkpoint opened with Python's `safe_open` — output pasted verbatim into that example's README | REJECTED: a Tlaloc container format, Java serialization, and a directory of one file per tensor (which multiplies the interrupted-save problem by the parameter count). |
| **Nine refusals, each by name** | ✅ | `ModelCheckpointTest` — a structure mismatch naming the disagreeing keys; a shape mismatch naming both shapes; buffers with nowhere to go; asking a model-only checkpoint for optimizer state; loading `Adam`'s state into `RMSprop` (and `Scheduled(Adam)`'s into a bare `Adam`); a non-checkpointable optimizer at SAVE time; a reserved `tlaloc.` metadata key; a FOREIGN safetensors file; an unknown tensor prefix; an unknown format version; `opt.` tensors with no kind tag; a non-f32 tensor | The two that matter most: a HuggingFace checkpoint has no `param.` prefix, so a lenient loader would restore ZERO parameters and report success; and both `Adam` and `RMSprop` keep exactly one tensor per parameter, so a kind mismatch is invisible to everything except the `kind` tag — the run would simply train with another optimizer's preconditioner. |
| **A checkpoint does not carry STRUCTURE, and says so** | ✅ | The refusal text itself, asserted: "A checkpoint carries VALUES, not structure — build the same model, then restore into it." | PyTorch's `state_dict` contract, for the same reason: a file that reconstructed Kotlin classes and constructor arguments would be a file that executes. Recorded in `Checkpoint.kt`'s KDoc under "What a checkpoint does NOT carry, by name", along with hyperparameters (`learningRate`, BatchNorm's `momentum`, Dropout's `p`). |
| **LR schedules: step, exponential, cosine, constant, linear warmup** | ✅ | `SchedulesTest` (13 tests) against HAND-COMPUTED arithmetic a reader can check by eye — every boundary of the staircase, `0.5^0.5` at the half-way point of a smooth exponential, `cos(π/2) = 0` at `T/2`, the non-zero cosine floor, and the warmup handover | A schedule is a PURE FUNCTION of the completed-step count, which is forced by the module's own design rather than chosen: an optimizer here is an immutable value, so a schedule holding a mutable cursor (PyTorch's `LRScheduler.step()` shape) would be the only mutable thing in `:nn` and would give a different trajectory depending on call history. `at(0)` is the rate the FIRST update uses. |
| **…and they agree with PyTorch** | ✅ | `NnSchedulesAndClippingVsPytorchTest` (3 tests, **they ran**, torch 2.11.0+cpu): `StepDecay` vs `StepLR`, `ExponentialDecay` vs `ExponentialLR`, `CosineDecay` vs `CosineAnnealingLR` over 34 scheduled rates — **worst relative disagreement 2.8e-7**, pinned at 1e-6 | Driver: `harness/python/run_pytorch_schedules_clipping.py`. Same §0.4.289 subprocess→JSON pattern as `NnMlpVsPytorchTrainingTest`, same self-skip ladder. |
| **The cosine divergence past `T_max`, published as a divergence** | ✅ | `NnSchedulesAndClippingVsPytorchTest.cosineAnnealingDivergesFromTorchPastTMaxExactlyAsDocumented` asserts BOTH sides of the disagreement: that torch CLIMBS past `T_max` (measured: 0.0 at k=10, then 0.00245, 0.00955, 0.0206) and that `CosineDecay` HOLDS at its floor | PyTorch's `CosineAnnealingLR` is periodic; a run that trained past `T_max` by accident would silently get its learning rate back. `CosineDecay` clamps. A warm-restart schedule is a named deferral and would be its own class. The test exists so the divergence cannot quietly vanish in either direction. |
| **`Scheduled` — the combinator** | ✅ | `SchedulesTest.aScheduledOptimizerTakesEachStepAtTheScheduledRate`: raw-bit equality against the same optimizer driven BY HAND at the rates the schedule names. Plus `aScheduledAdamActuallyAnnealsATrainingRun`, where a cosine-to-zero schedule makes the final update vanish — measured 4.5e-5 against the constant run's 2.5e-2 | `Scheduled(schedule) { lr -> Adam(lr) }`: the optimizer is REBUILT per step, which costs nothing (an `Adam` is four floats) and keeps the pure contract. REJECTED: a `withLearningRate` member on `Optimizer` (two of the four do not call their rate `learningRate`) and a mutable `var learningRate`. Its checkpoint NESTS the inner one under `inner.`, and its kind is `Scheduled(<inner kind>)` so it cannot be loaded into a bare optimizer that has nowhere to put the schedule position. |
| **Gradient clipping by global norm and by value** | ✅ | `GradientClippingTest` (14 tests): 3-4-5 fixtures so the norm is an exact integer, the clipped norm IS the threshold, direction preserved exactly (every `after/before` ratio identical) for by-norm and demonstrably NOT preserved for by-value, shapes and keys carried through, and a below-threshold call returning the SAME map rather than a copy | A pure function on the gradient map, applied between `CapturedStep.run` and `Optimizer.step`. REJECTED: clipping inside the optimizers (not an optimizer property — four copies and a constructor argument each) and inside the captured graph (a global cross-tensor reduction inside the traced function; real work for the GPU lane, named as a deferral). |
| **…against PyTorch, with the divergence measured** | ✅ | `NnSchedulesAndClippingVsPytorchTest.gradientClippingAgreesWithTorchOnTheNormAndOnBothClipShapes`: the norm agrees (5.5 on the fixture); `clip_grad_value_` agrees **exactly**, raw bits; by-norm sits **2.98e-8** from the exact analytic ratio and **1.64e-7** from torch's. The test asserts it is closer to the analytic answer than to torch — which is the claim the KDoc makes | The divergence is deliberate and small: torch scales by `maxNorm / (norm + 1e-6)`, Tlaloc by `maxNorm / norm`. The 1e-6 is torch's guard against a zero norm; a zero norm cannot exceed a positive `maxNorm`, so Tlaloc does not need it and takes the exact ratio. Recorded in `byGlobalNorm`'s KDoc because a user comparing trajectories with PyTorch will see it. |
| **A non-finite gradient norm refuses, by name** | ✅ | `GradientClippingTest.aNonFiniteNormIsRefusedByNameInsteadOfScalingEverythingToNothing` | `maxNorm / inf` is 0 and `maxNorm / NaN` is NaN: both silently destroy the step. PyTorch offers this behind an opt-in flag (`error_if_nonfinite`); here it is the only behaviour. The subtle neighbouring case is pinned too: a `1e30` pair whose sum of squares would overflow f32 is NOT refused, because the Double accumulator sees a real finite norm. `byValue` deliberately leaves a NaN alone rather than returning a bound — `coerceIn` would have invented a gradient. |
| **Wired into a real path** | ✅ | `examples/gpu-training` now saves the trained model + Adam's moments, rebuilds the structure from nothing, restores into it, and computes the decision boundary and the held-out accuracy WITH THE RELOADED MODEL. Ran on both lanes: GPU (PJRT/XLA CUDA) and `TLALOC_EXAMPLE_LANE=host`. Observed on both: **337 / 337 parameter scalars bit-identical, and `max|diff| = 0` over 1024 held-out predictions.** | The example's README verbatim blocks were re-generated from real runs of this build, and the Python `safe_open` output in its new "The checkpoint" section is pasted from an actual read of the file the example wrote. The example still self-skips the GPU lane on a machine with no CUDA, as before. |

### What this tier found that was not in its brief

- **The example's own header was stale.** `examples/gpu-training/Main.kt` announced
  "the only three ideas in it" and described "all 300 training steps" while
  `STEPS` had been 600 since the file was written. Both corrected.
- **Two host runs and two GPU runs disagree about whether the GPU lane is
  reproducible.** The example's README has claimed since §0.4.486 that "two GPU
  runs differed in the third decimal of the loss (0.047137 vs 0.051002)". Two
  runs made for this tier reproduced `0.046667` *exactly*. Both observations are
  true and the README now says so: XLA's compile-time GEMM autotuning CAN pick a
  different winner per run, and it does not have to. Claiming the lane is
  nondeterministic full stop would have been as wrong as claiming it is
  reproducible.
- **`Json.kt` had no way to write a string.** The reader has parsed JSON string
  escapes since §0.4.468; emitting one needed the inverse, and a header that
  concatenates an unescaped tensor NAME produces a file its own reader cannot
  parse. `jsonQuote` is the one writing primitive added, and the test that pins
  it uses a name containing `"`, `\`, a newline, a tab and `\u0001`.
- **`SafetensorsWriter`'s dtype `when` was exhaustive with a redundant `else`,
  and the compiler said so.** Tier 1 had just cleared this module's warning
  noise, so the `else` came out rather than being suppressed — which also turns
  a future new `DType` into a compile error in the file that has to decide about
  it.

### Suite state at Tier 2 item 7 close

| | |
|---|---|
| `./gradlew test --rerun-tasks` | BUILD SUCCESSFUL, 115 tasks, 3m34s |
| `bash scripts/count-tests.sh` | **2468** (2385 at §0.4.501 + 83 new: 15 + 3 writer, 30 + 5 checkpoint, 13 schedules, 14 clipping, 3 PyTorch oracle) |
| `bash scripts/onboarding-smoke.sh` | passes |
| `./gradlew -p examples/gpu-training run` | GPU lane: 600 steps in 2.02 s, loss 0.992417 → 0.046667, 98.3 % held out, 337/337 scalars reloaded bit-identical |
| `TLALOC_EXAMPLE_LANE=host ./gradlew -p examples/gpu-training run` | host lane: loss 0.992422 → 0.045672, 97.9 % held out, 337/337 scalars reloaded bit-identical |
| Cross-language oracles that RAN | `SafetensorsWriterOracleTest` (3), `NnSchedulesAndClippingVsPytorchTest` (3) — both against the frozen `~/.local/venvs/iree`, nothing installed |

## Tier 3 — portability and trust (§0.4.503, 2026-09-22)

Four items, all of the same shape: something in this repository was true only of
the machine it was written on, and a stranger inherited the consequence.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **The JDK floor is per module now** | ✅ | `verifyJvmTarget`, registered in all twelve subprojects and wired into each one's `check` (so `./gradlew test` covers it): it opens the module's jar and reads the class-file **major version** out of every `.class` byte stream, failing by name on the first disagreement. Negative-tested by setting `"core" to 25` in `tlalocJvmTargets` and confirming `:core:verifyJvmTarget` fails with "119 of 119 classes … are major 65". The authoritative split is the `tlalocJvmTargets` map in the root build; a module missing from it fails configuration by name. | `core` `ir` `autograd` `nn` `stablehlo` `maestro` → **21**; `runtime-pjrt` `runtime-cuda` `kptx` → 25 (FFM, JEP 454); `runtime-iree` `compiler-plugin` `benchmarks` → 25. `:benchmarks` has no `jvmMain` source at all, so the gate reads its **test** classes — an empty jar passes a bytecode check by vacuity, and that is exactly how the gate first failed. |
| **"No JDK 22+ API" is checked, not assumed** | ✅ | `-Xjdk-release=21` on the MAIN compilation of all six lowered modules. Without it the compiler still resolves against JDK 25's class library, so a JDK 22+ call would compile happily into 21 bytecode and fail at run time with `NoSuchMethodError`. All six compiled first try. | Applied to `main` only. Test compilations run on the toolchain JDK and may use anything it has; nothing claims the test sources run on 21. |
| **A JDK 21 consumer really runs a synthesized gradient** | ✅ | `scripts/jdk21-smoke.sh` → `./gradlew -p examples/quickstart runOnJdk21`. The quickstart is now `jvmToolchain(25)` + `jvmTarget = JVM_21` + `sourceCompatibility/targetCompatibility = 21`, and `runOnJdk21` executes it on a **JDK 21 launcher**. RAN: OpenJDK 21.0.2, full quickstart output, gradient correct. Verified it is genuinely 21 by unsetting `JDK21_HOME` and confirming Gradle fails with "Cannot find a Java installation … matching {languageVersion=21}". | Needs `JDK21_HOME` (`org.gradle.java.installations.fromEnv` in both `gradle.properties` files), because Gradle does not auto-detect `~/.local/jdks` and a literal path would repeat the defect item 4 removes. The script refuses by name, and checks the JDK it was handed really reports 21. |
| **`:maestro` → 21, and it ran there** | ✅ | The decision the brief left open. `:maestro` imports no `java.lang.foreign` anywhere, its four project dependencies are all 21, and its serving story is a *directory* read by a framework-free ctypes-PJRT Python process — no JVM on the serving side, so no FFM. `-Xjdk-release=21` compiles; `./gradlew :maestro:exportServingArtifact -PexportJdk=21` RAN on OpenJDK 21.0.2 and wrote all 6 manifest entries + bodies. | `exportJdk` is a knob defaulting to 25, not a hard 21: this task is a documented runbook command and making it need a second JDK that Gradle cannot auto-detect would be a regression for its actual users. |
| **`compiler-plugin` not lowered to 21** | ⬜ | — | Nothing in the plugin is known to need JDK 22+; it was not tried, because the target split was handed down with `compiler-plugin -> 25` in it and widening an owner-approved decision is not this tier's call. Lowering it is what would make "a JDK 21 machine can build `grad {}`" true, and it is the obvious next question. |
| **`:runtime-iree` not lowered to 21** | ⬜ | — | It imports no `java.lang.foreign` and shells out to `iree-compile` / `iree-run-module`, so it probably could be. Nothing in this repository certifies an IREE run on a JDK 21, and an uncertified lower bound is worse than a high one. The reason is in the module's own build file, not only here. |
| **Plugin ↔ library version match still unchecked** | ⬜ | — | Carried forward unchanged from Tier 0. §0.4.503 added a guard for the *Kotlin compiler's* version; it did NOT add one for `io.github.pedronahum:tlaloc-compiler-plugin` vs `io.github.pedronahum:tlaloc-core`. Mixing those is still unsupported and still silent, as `docs/COMPATIBILITY.md` says. A separate item. |
| **The build machine still needs a JDK 25 — stated, not implied** | ✅ | The boundary is written in `compiler-plugin/build.gradle.kts`, the README's new `### Requirements`, `docs/GETTING_STARTED.md` §0, `docs/COMPATIBILITY.md` and `scripts/jdk21-smoke.sh`'s own header. | See "What Tier 3 found that was not in its brief" below: the brief's phrasing ("a consumer on JDK 21 can use `grad {}`") is narrower than it reads, and the split as handed down cannot make it fully true. |
| **Kotlin version guard** | ✅ | `KotlinVersionGuard` runs FIRST in `registerExtensions`, before a single extension is registered, and reports an ERROR to the `MessageCollector` (a clean refusal, not a thrown crash) naming the version found, the version built against, the supported range and the opt-out. 13 tests in `KotlinVersionGuardTest` cover every branch of the pure decision, plus two links that would otherwise be assumed: that the version the guard reads *at run time* from the running compiler is accepted, and that `COMPILED_AGAINST` equals `libs.versions.kotlin` (handed to the test as a system property). | The supported range is the **bugfix family** of the feature release built against: 2.3.20–2.3.29. Kotlin numbers feature releases by tens in the third component, so "any 2.3.x" would wave through 2.3.0 and 2.3.10, which have different internals, and "exactly 2.3.20" would break users the day a patch lands. |
| **The guard's refusal on a real foreign compiler** | ✅ **as of §0.4.504** | Was 🧪 here, with the note "there is one Kotlin on this machine". That turned out to be a property of the *build*, not of the machine: §0.4.504's `-PtlalocKotlinVersion` puts a different `kotlin-compiler-embeddable` on the test classpath, and under 2.3.10 the guard refuses by name — at the unit level and inside six real in-process compiles. See the Tier 3 CI table. | Still uncertified for a 2.4.x compiler *specifically*, because the plugin does not compile against 2.4.20 at all (§0.4.504's other finding). The 2.3.10 run is a genuinely different feature release, which is what the guard's rule keys on. |
| **Symja is optional** | ✅ | `ir/build.gradle.kts` declares `compileOnly(libs.symja.core)` in `jvmMain`, so it leaves the published POM; `jvmTest` (`:ir`), `testImplementation` (`:compiler-plugin`) and `jvmTest` (`:benchmarks`) keep it on every classpath that certifies coarsening, and three tests are tripwires that fail if a future build change drops it, one per classpath: `SymjaOptionalDependencyTest.symjaIsPresentOnThisTestClasspath` (`:ir`), `SymbolicEngineRefusalTest.symjaIsOnThisModulesTestClasspath` (`:compiler-plugin`) and `SymjaBenchmarkClasspathTest.symjaIsOnTheBenchmarkTestClasspath` (`:benchmarks`, added in §0.4.507 — this row claimed three while only two existed). `SymbolicEngines.probe` is exercised against a classloader that genuinely cannot see `org.matheclipse.*`, and `PhiCalculus.containsLoop` — the predicate for "did this body need the CAS" — is pinned on a loop-free body, a WHILE body, a WHILE nested in an IF region, and a concrete-trip loop that C5 unrolls **engine-free**. | `SymbolicEngines` names no Symja type at all: the probe is `Class.forName` on a string. That matters, because `SymjaEngine`'s own fields and signatures mention `org.matheclipse` types, so merely *loading* that class can raise `NoClassDefFoundError` during verification, before any `init` block of its own could run. |
| **Absence refuses by name** | ✅ | `TlalocIrGenerationExtension.missingSymbolicEngineMessage` — a pure function of (function name, Symja present?, loop survived coarsening?), pinned by `SymbolicEngineRefusalTest` (4 tests) on all three arms. The message names the function, the WHILE it found, C6–C9, the coordinate, LGPL-3.0, and the literal `implementation("…")` line. The refusal fires at the point where the reverse transform **actually rejected** the primal, not merely where a loop exists: §0.4.128's LoopInvariant rewrite legitimately leaves nested WHILEs the transform handles, so blaming Symja for those would be a false accusation. Severity follows `strictLowering`. | 🧪 The end-to-end path — compiling a loop-bearing `grad {}` against a classpath with *no* Symja — is written, not run. Symja is on `:compiler-plugin`'s test classpath deliberately (removing it would make every coarsening test pass by doing nothing) and the K2 harness shares the test JVM's classpath, so staging absence needs a second compiler process. |
| **The LGPL fact is published** | ✅ | `SymbolicEngines`' KDoc, `ir/build.gradle.kts`, `docs/COMPATIBILITY.md`, `docs/GETTING_STARTED.md` §0 and the README's Requirements all say the same three things: it is LGPL-3.0, Tlaloc only *links* it across the `SymbolicEngine` interface, and it is optional. `theAbsenceMessageNamesTheDependencyTheLicenceAndTheOneLineToAdd` asserts the licence string is in the refusal a user actually sees. | Making the dependency optional does not change the licence analysis (linking was always permitted); it changes *who has to accept it*. A consumer whose policy forbids LGPL in the graph now simply does not add the line and gets a named refusal on the day a body needs the CAS, instead of an audit finding. |
| **No Symja-free `SymbolicEngine`** | ⬜ | — | The only implementation is `SymjaEngine`. A consumer who cannot accept LGPL-3.0 at all has no CAS, and therefore cannot differentiate a symbolic-trip-count loop. The refusal message says so in as many words rather than implying an opt-out that does not exist. `docs/STAGE_B_PLAN.md` §5.3 already carried "a custom Kotlin CAS" as the fallback if Symja's adequacy failed; this is a second reason to want it. |
| **The hardcoded PJRT fallback path is gone** | ✅ | `PjrtBinaries.resolveCudaPlugin` / `cudaPluginCandidates` / `describeCudaPluginSearch`, all `internal` and all taking their environment as parameters, pinned by 15 tests in `PjrtCudaPluginResolutionTest` that build **synthetic install trees** under a temp directory — so they pass on a laptop with no GPU and on the GB10 alike. Covered: any venv name under `~/.local/venvs/`, any `python3.N`, any `jax_plugins/*cuda*` package and any `.so` inside it, `dist-packages`, `lib64`, `$VIRTUAL_ENV` outranking a scan, a non-CUDA `jax_plugins` package NOT being offered to the CUDA lane, and determinism when two venvs both have one. | The old fallback was one literal string: `~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`. One venv NAME, one Python MINOR version, one plugin PACKAGE. Every other machine got `available == false` and no way to find out where Tlaloc had looked. |
| **The GB10 path still resolves** | ✅ | Measured, not assumed: `pluginPath` resolves to `~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so` — the same file as before, found by the glob rather than the literal — and the full suite's GPU lanes ran (`./gradlew test --rerun-tasks` BUILD SUCCESSFUL with live PJRT-CUDA output). | `theGb10InstallShapeStillResolvesHere` self-skips into a weaker assertion on a host with no plugin, rather than asserting about a machine it is not running on. |
| **The failure names every place it looked** | ✅ | `PjrtBinaries.pluginSearchReport`, asserted by three tests to name the env var, every root (`~/.local/venvs/*`, `~/.venv`, `~/venv`, `~/.local`, `/usr/local`, `/usr`), the globbed `python3.*`, each candidate with FOUND/missing, and the fix. Both examples' "GPU lane unavailable" reason now carries the whole report instead of the advice "set TLALOC_PJRT_PLUGIN_PATH, or install a JAX CUDA plugin", which told a user who *had* installed one nothing at all. | The fix-it line is emitted only when nothing resolved; a successful report ends with `Resolved: <path>`. Printing advice underneath a success is how a log teaches its reader to stop reading it. |

### What Tier 3 found that was not in its brief

- **"A consumer on JDK 21 can use `grad {}`" cannot be made true by this split,
  and the brief contains both halves of the contradiction.** The target table it
  handed down (and called owner-approved) puts `compiler-plugin` at 25. Kotlin
  loads a compiler plugin *inside the compiler's own JVM*, which for a Gradle build
  is the toolchain JDK — so a 25-bytecode plugin means a JDK-21-only machine cannot
  compile `grad {}` at all, whatever it targets. The true statement is narrower and
  is now the one published: **build on 25, run on 21**. Whether the plugin could
  itself be lowered to 21 was deliberately NOT decided here, because the split was
  handed down with that row in it; it is the obvious next question and the reason
  is recorded rather than the answer guessed.
- **`:benchmarks` has no production source at all.** The bytecode gate failed on it
  immediately — "found no .class entries" — because every line of `:benchmarks`
  lives in `jvmTest` and its `jvmJar` is empty. An empty jar passes a bytecode
  check by vacuity, so the gate treats zero classes as a failure and reads that
  module's test classes instead. Nothing is exempt; the thing read is different.
- **`lib64` is a symlink to `lib` on this host**, so the GB10 plugin appears twice
  in the candidate list (`…/lib/…` and `…/lib64/…`, both FOUND). Harmless — the
  sorted order picks `lib` — but it is why the report can list the same physical
  file twice, and worth knowing before someone reads it as two installs.
- **`scripts/install-jdk21.sh` still exists from the §0.4.244 era**, when the whole
  repository targeted 21, and its header still says so. It is a macOS
  Homebrew migration helper and was not touched; §0.4.503 makes half of its
  statement true again by accident, which is a coincidence and not a plan.
- **The KMP publications carry no `org.gradle.jvm.version`, so Gradle will not catch
  a JDK mismatch for you.** Measured in `~/.m2` after `publishToMavenLocal`: every
  `*-jvm` variant (`core-jvm`, `nn-jvm`, `runtime-pjrt-jvm`, …) publishes that
  attribute as *absent*, while the plain-JVM `compiler-plugin` publishes `25`. Two
  consequences, both real:
  - GOOD: a consumer compiling at `jvmTarget = 21` can still resolve the 25-bytecode
    plugin on `kotlinCompilerPluginClasspath`. That is why `examples/quickstart`
    works, and it was not obvious in advance — a `TargetJvmVersion` of 21 on that
    configuration would have refused the plugin outright.
  - BAD: a consumer at target 21 who adds `io.github.pedronahum:tlaloc-runtime-pjrt` gets NO
    dependency-resolution error. They get `UnsupportedClassVersionError` the first
    time the class loads. Gradle would normally reject that at resolution time; the
    Kotlin Multiplatform plugin does not publish the attribute that makes it
    possible. Not fixed here — forcing the attribute onto KMP variants is a
    publication change with its own blast radius — but named, because the README's
    "PJRT / CUDA execution needs 25" is now the only thing standing between a user
    and that error.
- **The Symja refusal had to be attached to the reverse transform's failure, not to
  the presence of a loop.** The first design — refuse whenever a WHILE survived
  coarsening and no engine was available — would have refused programs that work:
  §0.4.128's LoopInvariant rewrite produces nested WHILEs inside IF arms and
  `DxirReverseTransform` recurses into them deliberately (its own comment says the
  §0.4.118 assumption that "WHILE shouldn't survive SCT" no longer holds). Only the
  conjunction — loop survived, transform rejected, no engine — makes the missing
  dependency the actionable cause.
- **The old engine construction folded three conditions into one silent null.**
  `try { SymjaEngine() } catch (_: Throwable) { null }` could not tell absent from
  present-but-broken from present-but-failing-to-initialise, and then degraded
  quietly in all three cases. `SymbolicEngines` separates absence (a probe) from
  construction failure (a `runCatching` that is explicitly a *different* condition
  and deliberately not reported as absence).
- **`KotlinCompilerVersion.VERSION` is safe to read and it was worth checking.** It
  is a Java `static final String` assigned in a static initialiser with no
  `ConstantValue` attribute, so it is *not* inlined into the plugin at compile time
  and does report the RUNNING compiler. Had it been a `ConstantValue`, the guard
  would have compared 2.3.20 against itself forever and passed on every machine.
  The fallback reads `META-INF/compiler.version` straight out of the jar, which
  needs no Kotlin API and so is the one lookup that cannot be broken by the API
  drift being detected.

### Suite state at Tier 3 close

| | |
|---|---|
| `./gradlew test --rerun-tasks` | BUILD SUCCESSFUL, 127 tasks, 3m30s (three clean-room runs green; see the flake row) |
| `bash scripts/count-tests.sh` | **2509** (2468 at §0.4.502 + 41 new: 13 version guard, 4 symbolic-engine refusal, 9 Symja-optional + `containsLoop`, 15 PJRT resolution) |
| `bash scripts/onboarding-smoke.sh` | passes |
| `JDK21_HOME=… bash scripts/jdk21-smoke.sh` | passes — a compile-time-synthesized gradient ran on OpenJDK 21.0.2 |
| `./gradlew :maestro:exportServingArtifact -PexportJdk=21` | passes — 6 manifest entries written by a JDK 21 |
| `PjrtBinaries.pluginPath` on this host | `~/.local/venvs/iree/…/xla_cuda12/xla_cuda_plugin.so` — unchanged, found by the glob |
| Gates added, not JUnit tests | `verifyJvmTarget` × 12 (wired into `check`), `-Xjdk-release=21` × 6 |
| Flake seen once on the way | `KptxPagedAttentionBenchTest.pagedAttentionLaneFloorsAcrossDecodeShapes` failed one clean-room run (`BUILD FAILED in 3m 26s`, 21:50). It is a WALL-CLOCK benchmark on a live GB10 that compares a claimed-kernel lane, an XLA lane and a dispatch floor, and §0.4.495 already published the reason it is fragile: "the dispatch floor spread 40–416 µs makes small shapes unmeasurable". Re-run three times in isolation with `--rerun-tasks`: green each time; two subsequent full clean-room suite runs: green. §0.4.503 touches no kernel, no PTX and no timing code. Recorded rather than quietly re-run. |
| A failure that was NOT this tier's | `:nn:jvmTest > GradientClippingTest.anInvalidRangeIsRefusedByName` appears FAILED in the Gradle daemon log at 20:51 — before §0.4.503's first build. The daemon had been serving the §0.4.502 session since 18:44. Named here because the log is shared and the next reader would otherwise have to work that out. |

## Tier 3 — CI matrix (§0.4.504, 2026-09-22)

Every ✅ in [CAPABILITIES.md](CAPABILITIES.md) was certified on exactly one
machine: an NVIDIA GB10, aarch64, JDK 25. CI landed the day before this tier
(§0.4.497) as a single `ubuntu-latest` job running `./gradlew test`, which added
one x86_64 data point and nothing else — no second OS, no second JDK, no second
Kotlin. This tier is the matrix, and it is the tier whose deliverable **cannot be
run from the machine that wrote it**: there is no GitHub Actions access here, so
every lane below is 🧪 and stays 🧪 until a run exists. What could be executed
locally was, and those rows say so.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **Platform matrix: x86_64 Linux + aarch64 Linux + arm64 macOS** | ✅ for x86_64/macOS/JDK21, 🧪 for aarch64 | **Executed on GitHub.** All three original lanes green at HEAD (run 35904909054: `test (ubuntu-latest, JDK 25)`, `test (macos-15, JDK 25)`, `library on JDK 21 (built by 25)`). `actionlint` 1.7.12 reports zero findings on all three workflow files | **The second platform earned itself immediately.** The arc's first push (690a77c) failed on `ubuntu-latest` at the `Test` step while macOS and JDK 21 passed — an x86_64-Linux-only failure, green again from 7b74ce5 onward. **Open:** whether that was fixed by §0.4.508 or is a flake is NOT established; the logs need authentication and were not read. The repo's own smoke gate names `BGDHyperOptTest` as a known flake to isolate before believing, which is the first hypothesis to test by re-running 690a77c via `workflow_dispatch`. **aarch64 Linux added in §0.4.511** (`ubuntu-24.04-arm`) now that the repository is public and the runner is free — §0.4.504 omitted it only because visibility could not be checked. **It passed on its first run.** That closes the gap that mattered most: every ✅ in CAPABILITIES was certified on a GB10, which is aarch64, and until this lane ran the only aarch64 evidence was the single machine that produced all of it. The 10× macOS cost note is void — standard runners are free in public repositories. |
| **The runner-honesty header is still honest, and now complete** | ✅ | The header of `build.yml` | The §0.4.497 header named GPU, PJRT, IREE and `stablehlo-translate`. It did not name the **oracle venv**: `SafetensorsWriterOracleTest`, `NnSchedulesAndClippingVsPytorchTest`, `NnMlpVsPytorchTrainingTest` and the serving lanes all resolve a Python at `~/.local/venvs/iree/bin/python` (or `TLALOC_TORCH_PYTHON`) and self-skip by name without it. So a green runner also means "no cross-language certification ran". Now stated. |
| **aarch64 *Linux* lane** | ⬜ | — | Not added, on purpose, and the brief asked for exactly this judgement. GitHub's `ubuntu-24.04-arm` runners are free for **public** repositories and a paid plan for private ones, and this repository's visibility could not be established from here (no `gh`, no credentials, and `git remote` only proves the URL). A lane that cannot start is worse than a named gap. The aarch64 evidence for this project remains the GB10 itself; the macOS leg does cover arm64 on a different OS. |
| **`-PtlalocTestJdk=<n>`: run a module's tests on another JDK** | ✅ | Root `build.gradle.kts`. **Ran locally**: `:core :ir :autograd :nn :stablehlo :maestro` `jvmTest` with `-PtlalocTestJdk=21` — **1,822 tests, all six suites green on OpenJDK 21.0.2** (213/945/148/144/300/72). Each task logs the launcher it got (`':core' tests running on JDK 21 (21.0.2+13-58)`) so a silently-ignored flag is visible | It works because the six 21-targeted modules compile **every** compilation to `JVM_21`, tests included — §0.4.503's `-Xjdk-release=21` is main-only but the target is not. **Negative-tested:** `:runtime-pjrt:jvmTest -PtlalocTestJdk=21` fails by name with the module, its Java 25 target and the list of modules the flag is valid for, instead of an `UnsupportedClassVersionError` out of the test runner. |
| **The JDK 21 lane** | 🧪 | `build.yml`, job `library-jdk21`: two `setup-java` steps (21 then 25, so the toolchain is 25), `JDK21_HOME` exported into `$GITHUB_ENV` for the `org.gradle.java.installations.fromEnv` hook §0.4.503 added, then `verifyJvmTarget`, then the six suites under `-PtlalocTestJdk=21`, then `scripts/jdk21-smoke.sh` | **Never run as a lane**; every command in it ran locally. It carries no test-count step on purpose — it runs six modules, not the suite, and a number from it would be compared against the repository's total and read as a regression. The BUILD-on-25 / RUN-on-21 boundary is restated in the job's own comment because it is the thing people get wrong. |
| **`-PtlalocKotlinVersion=<v>`: build and test against another Kotlin** | ✅ | `settings.gradle.kts` (repositories + the whole explanation) and root `build.gradle.kts` (the `eachDependency` rule). **Ran locally twice**, against 2.4.20 and 2.3.10, and both runs produced findings — see the two rows below | Absent the property, nothing changes: same two repositories, no resolution rules visited. It deliberately does **not** rewrite `libs.versions.kotlin`, so `KotlinVersionGuardTest.COMPILED_AGAINST matches the version catalog` cannot be made vacuous by the override. |
| **The next-Kotlin lane** | 🧪 | `.github/workflows/kotlin-next.yml`: `continue-on-error` on the job *and* on each probe, weekly cron + dispatch + a paths-scoped push, three probes (plugin vs internal K2 API; library vs the new compiler; the version guard inside the new compiler), a `$GITHUB_STEP_SUMMARY` table. The version is **resolved** from Central's `maven-metadata.xml` `<latest>` (which includes Betas and RCs), not pinned, so the lane cannot rot into naming a version that does not exist; the resolve script was executed locally and produced `catalog=2.3.20 next=2.4.20 differs=true same_family=false` | **Never run as a lane.** This is the mitigation `DIFFKTX_SPEC.md` §16 named ("CI against Kotlin EAP") for a HIGH/HIGH risk and nobody had implemented — there was no way to ask this build for a different Kotlin before this tier. Probe 3's expectation is *inverted* and the script says which way: under a different feature release the guard is supposed to refuse, and a PASS there would be the defect §0.4.503 removed. |
| **The guard refuses inside a genuinely foreign compiler** | ✅ | **Executed.** `-PtlalocKotlinVersion=2.3.10 :compiler-plugin:test --tests '*KotlinVersionGuardTest*'` → 13 tests, exactly 1 failed: *"the guard refuses the running compiler (2.3.10) … expected: \<Supported\> but was: \<Unsupported(found=2.3.10…)\>"*. And in a real in-process compile: `--tests '*DiagnosticNoiseTest*'` under 2.3.10 → 6 of 7 failed, each carrying `[ERROR] Tlaloc's K2 compiler plugin is built against Kotlin 2.3.20 and the running Kotlin compiler is 2.3.10. Supported: 2.3.20 through 2.3.29. …` | **This closes the 🧪 row §0.4.503 left open** ("the version guard's refusal has never run inside a FOREIGN Kotlin compiler"). It is now run — twice, at the unit level and inside six real compiles — on Kotlin 2.3.10, a different feature release of the same minor. Still not certified: a 2.4.x/2.2.x compiler, because (next row) the plugin does not compile against 2.4.20 at all. |
| **A published negative result: Kotlin 2.4.20 breaks `:compiler-plugin`** | ✅ (as a measured finding) | **Executed.** `-PtlalocKotlinVersion=2.4.20 :compiler-plugin:compileKotlin` → `:core` and `:ir` compile (warnings only); the plugin fails with one hard error: `e: TlalocCompilerPluginRegistrar.kt:34:74 Direct access to the message collector is discouraged. Consider using CompilerConfiguration.report.` | The lane's first real run found the exact class of breakage §16 predicted, before any user did. **Not fixed here** — the supported toolchain is 2.3.20, and a change to the registrar has to be compiled against 2.4.20 *and* re-certified against 2.3.20, which is a toolchain-bump slice and not a CI slice. ⬜ row of its own. Also seen: `-Xcontext-parameters` is redundant at language version 2.4, four "cast is redundant" in `PhiCalculus.kt`, one unnecessary `!!` in `AttentionKernel.kt`, and a KGP warning that **becomes an error in Kotlin 2.5.0**: `Argument '-d' is not supported in the Build Tools API`. |
| **The LIBRARY is already next-Kotlin clean** | ✅ (measured) | **Executed.** `-PtlalocKotlinVersion=2.4.20 :core:jvmTest :ir:jvmTest :autograd:jvmTest :nn:jvmTest :stablehlo:jvmTest` → BUILD SUCCESSFUL, **1,750 tests, 0 failures** (213/945/148/144/300), compiled and run by the 2.4.20 compiler | This is probe 2 of the lane, run by hand. It matters because it localises the 2.4.20 breakage precisely: the language, the stdlib and 1,750 tests' worth of Tlaloc's own source are fine on the next feature release, and the *only* thing that is not is the plugin's use of internal compiler API — which is exactly the risk `DIFFKTX_SPEC.md` §16 predicted and nothing else. Warnings seen, not errors: four redundant casts in `PhiCalculus.kt`, one unnecessary `!!` in `AttentionKernel.kt`. |
| **The Kotlin *Gradle plugin* cannot be overridden from a resolution rule** | ⬜ | — | Named, with the evidence, in `settings.gradle.kts`. `useVersion` and `useModule` in `pluginManagement.resolutionStrategy.eachPlugin` both fail with `Error resolving plugin [id: 'org.jetbrains.kotlin.multiplatform' …] > the plugin is already on the classpath with a different version (2.4.20)` — naming 2.4.20 on *both* sides of a "different version" complaint — because the root build declares both Kotlin plugins `apply false` and so loads KGP before any module asks. Consequence, stated rather than hidden: the lane tests our source and our plugin against a new **compiler**, not against a new **KGP**. Fixing it means moving the plugin declarations, which is a change to the default build. |
| **`./gradlew test` under a Kotlin override** | ⬜ | — | Not attempted, and the reason is structural: under a foreign feature release the guard refuses *by design* inside every one of the **83** in-process K2 harnesses in `:compiler-plugin`, each of which builds its own `K2JVMCompilerArguments` with no shared place to pass `unsafeAllowUnsupportedKotlin=true`. The full suite would therefore report dozens of failures that all say what the guard already said. A shared harness helper would make that lane possible and would be worth having for its own sake. |
| **The maestro lane still does what its comments say** | 🧪 (lane) / ✅ (its central step, locally) | `.github/workflows/maestro-integration.yml`. **Ran locally**: `./gradlew :vendored-maestro:maestro-tlaloc:test` → BUILD SUCCESSFUL, 26 tasks, 4 executed. All six entries in its `paths` filter were checked to still exist | Three changes. (1) It was **`push`-only**, so a PR renaming Maestro's `StepType` registry — the exact event the filter exists for — merged unchecked; it now runs on `pull_request` with the same list, duplicated because Actions has no anchor support for `paths`. (2) Its `./gradlew test` step is now `:maestro:jvmTest`: when it landed, that step was the only Tlaloc coverage on a matching push, and `build.yml` now runs the whole suite on two platforms on the same events, so the third run bought nothing and cost the lane its 10-minute audit budget. (3) The header says which of its claims have run and which have not. |
| **The informational test-count step** | ✅ | `build.yml`, `Test count (informational)`, `if: always()`, `|| true` | Kept exactly as §0.4.497 wrote it, including the reason: a runner's total is legitimately lower than a GB10's, and asserting it here would teach people to ignore a red build. |

### What this tier found that was not in its brief

- **Kotlin has moved three feature releases past the one this repository pins.**
  `libs.versions.kotlin` is 2.3.20; Maven Central lists 2.3.21, 2.4.0, 2.4.10 and
  2.4.20 as published releases (plus their Betas and RCs). So the "next version"
  lane is not hypothetical and did not need an EAP repository to find a subject —
  its first run had one hard failure waiting for it.
- **A dependency-only override swaps the running compiler, not just the compile
  classpath.** The rule rewrites `org.jetbrains.kotlin:*`, which in Kotlin 2.x
  includes `kotlin-build-tools-impl` — the artifact KGP actually runs a compilation
  through. So `-PtlalocKotlinVersion=2.4.20` compiled *every* module with the 2.4.20
  compiler while KGP stayed at 2.3.20, which the compiler said itself
  ("redundant for the current language version 2.4"). That is more coverage than the
  property was designed for, and it is why the lane can claim a language/stdlib
  signal at all. It also means the configuration under test is "KGP 2.3.20 driving a
  2.4.20 compiler", which is not what a consumer on Kotlin 2.4.20 runs. Stated in
  both files.
- **`actions/upload-artifact@v4` fails a matrix that shares an artifact name.** The
  §0.4.497 job uploaded to a single `build-reports`; turning that job into a
  two-OS matrix without renaming would have failed both legs at the upload step,
  after a 20-minute suite. The names are now per-leg.
- **The vendored upstream workflow is inert, and it is worth knowing why.**
  `third-party/maestro/.github/workflows/main.yml` is tracked in this repository
  (1,261 vendored files, none of them a submodule). GitHub only runs workflows from
  the repository root's `.github/workflows`, so Netflix's own CI does not run here
  and never has. Nobody had said so anywhere.
- **`third-party/maestro` needs no `submodules:` in checkout**, for the same reason:
  it is vendored *in-tree*, not a submodule, so `actions/checkout@v4`'s default is
  correct. Checked rather than assumed, because a missing submodule would have made
  the maestro lane's central step fail in a way that looks like a Gradle problem.
- **`count-tests.sh` counts the vendored Maestro results too.** `find . -name
  'TEST-*.xml' -path '*/test-results/*'` reaches
  `third-party/maestro/*/build/test-results`, which contributed **51** tests to the
  repository total after `:vendored-maestro:maestro-tlaloc:test` ran here. Root
  `./gradlew test` never executes those tasks, so whether they are in the number
  depends on whether someone ran the maestro lane's command since the last clean.
  The gate is a smoke gate and this does not break it, but the number is not purely
  a function of the root suite and nothing said so.

### Suite state at Tier 3 CI close

| | |
|---|---|
| `./gradlew test --rerun-tasks` | BUILD SUCCESSFUL, 127 tasks executed, 3m36s |
| `bash scripts/count-tests.sh` | **2509**, unchanged from §0.4.503 — this tier adds CI lanes and two build properties, and no JUnit tests. (Of that total, 51 come from `third-party/maestro` results present in the tree; see the findings section.) |
| `-PtlalocTestJdk=21`, six library modules | 1,822 tests green on OpenJDK 21.0.2 |
| `-PtlalocKotlinVersion=2.3.10`, guard subset | 13 tests, 1 failed — the expected refusal, naming 2.3.10 |
| `-PtlalocKotlinVersion=2.4.20`, `:compiler-plugin:compileKotlin` | FAILS with one error (published above, not fixed) |
| `-PtlalocKotlinVersion=2.4.20`, five library modules | BUILD SUCCESSFUL — 1,750 tests, 0 failures, under the 2.4.20 compiler |
| `./gradlew :vendored-maestro:maestro-tlaloc:test` | BUILD SUCCESSFUL |
| `actionlint` 1.7.12 on all three workflows | zero findings |
| GitHub Actions runs of any lane | **none — that is the tier's honest limit** |

## Tier 4 — documentation debt and the public API surface (§0.4.505, 2026-09-22)

Six items. Three are documentation (a spec section 400 sections out of date, a
count that disagreed with itself, a claim with no test behind it) and three build
something a consumer needs (an API reference, an opt-in marker, an ABI baseline).
The tier exists because this repository's distinguishing quality is that its
claims are true, and four of them were not.

| Item | Status | What pins it | Deferred / notes |
|---|---|---|---|
| **1. `DIFFKTX_SPEC.md` §17 no longer marks shipped work ⬜** | ✅ | Reading, not a test — but every section number cited is establishable from `git log` or from the file's own "Cumulative §0.4 trail through §0.4.100" era table, and the header says so | Corrected: step 2 (libtorch DROPPED, MNIST example SHIPPED §0.4.496, the tape deleted §0.4.446–451), step 3 (DCE §0.4.45, CSE §0.4.48 + §0.4.118–119, shape validation §0.4.353), **step 5 IREE ✅ §0.4.284–§0.4.295**, step 6 (✅ — all three "remaining" items shipped), step 8 (🟡 — the emitter shipped, libShardy JNI never did and was replaced by a pure-Kotlin tokenizer), **step 10 PJRT ✅/🟡 §0.4.302–§0.4.309**, step 11 (🟡), step 12 (⬜→❌, matching the README), **steps 13–15 coarsening ✅** (§0.4.11–§0.4.26, §0.4.27–§0.4.36, and the three-part CAS bridge). Nothing was upgraded on a commit title alone. |
| **1b. Step 16 went the OTHER way: ⬜ → 🟡, not ✅** | ✅ | The absence of any §0.4 entry publishing the six head-to-head numbers, and §0.4.235's "M9 closure waits on user-side toolchain" | **This contradicts the brief that produced this tier**, which described steps 13–16 as "shipped and certified against the OOPSLA 2021 paper". Steps 13–15 are. Step 16 asks for the paper's *speedups* matched within 20%, and that comparison has never been run: all six benchmarks are ported and all six are `HeadToHeadHarness` inhabitants, `aggregate.py` carries the Table 3 numbers (corrected in §0.4.239), and the PyTorch/JAX reference sides were gated on a toolchain nobody supplied. So §11.13's M9 exit criterion is **open**, and the gate the ladder makes coarsening's release conditional on is **unread, not failed** — coarsening shipped anyway. Both facts are now written into §17 and into the paragraph under it. |
| **1c. §14's four false lines** | ✅ | Grep: no ktlint, no detekt, no Spotless, no Robolectric, no kotlinx-benchmark, no MkDocs anywhere in the build or the workflows | "Formatting: ktlint + detekt, **enforced in CI**" was the worst of them — none of the three exists and nothing lints. Also corrected: "target bytecode 25" (per-module since §0.4.503), "MLIR/StableHLO pinned to a specific commit" (nothing is pinned or vendored but Maestro), the CI platform list (two OS legs, no aarch64-Linux, no Windows, and none has run), and the configuration cache ("on" — it is not). |
| **1d. §18's answered open questions** | ✅ | Reading, against HEAD | Seven marked **Resolved** with what decided them: the licence (Apache-2.0, §0.4.498), the PJRT plugin strategy (separate install, with a search report), TPU timing (claimed 🧪 in those words, never run), the symbolic engine (Symja, and **not** Apache-2.0 as that line claimed), sparse tensors (revived §0.4.417–§0.4.420, GPU refusal by design), quantization (int8 KV cache only), Shardy pinning (answered *by construction* — no bindings to vendor). Two new entries added because Tier 0 opened them and nothing recorded them here: Symja's licence and the unchosen Central namespace. Governance, Shardy propagation control, MPMD and coarsening-for-higher-order stay open; three are now marked *blocked* on a non-JVM target that does not exist. |
| **2. The example count, and the drift sweep around it** | ✅ | `for d in examples/*/ examples/internals/*/; do [ -f $d/settings.gradle.kts ]` → exactly **ten**; and every one of them was run | README said ten, `GETTING_STARTED.md` §6 said **eight** and named seven. Ten is right: seven top-level (`quickstart`, `readable-gradients`, `differentiable-physics`, `named-indices`, `mnist`, `gpu-training`, `gpu-inference`) plus three under `internals/`. The sweep for the same class of drift found four more, all listed in the findings section below. |
| **3. An aggregated API reference** | ✅ | `./gradlew apiDocs` — the task fails by name if `build/docs/api/index.html` is absent, so "there is an API reference" cannot rot into a task that produces nothing. Ran: 2,898 HTML pages, 46 MB, all eleven modules in the module list | Dokka is applied at the ROOT now (it was `apply false`) because Dokka 2's multi-module aggregation needs the root to hold the `dokka` configuration. **Not hosted** — no gh-pages lane, no MkDocs; that is a ⬜ row in `docs/CAPABILITIES.md`. The 206 unresolved-KDoc-link warnings are **counted** here for the first time (Tier 0 recorded them as uncounted): 206 warnings over **110 distinct link targets**, `:ir` 117, `:core` 23, `:compiler-plugin` 19, `:stablehlo` 13, `:runtime-pjrt` 13, `:runtime-iree` 5, `:maestro` 5, `:kptx` 4, `:autograd` 4, `:runtime-cuda` 2, `:nn` 1. Unswept. |
| **4. `@ExperimentalTlalocApi`** | ✅ | `ExperimentalTlalocApiTest` (5 tests, `:core`) + `ExperimentalApiOptInTest` (5 tests, `:compiler-plugin`) | See the criterion row below. The marker is `@RequiresOptIn(ERROR)`, BINARY retention, declared in `:core`. |
| **4b. The criterion, and what it deliberately excludes** | ✅ | The annotation's own KDoc states it; `ExperimentalTlalocApiTest."the certified surface does NOT carry the marker"` asserts the exclusion | Three criteria, one needed: (1) `docs/CAPABILITIES.md` marks the capability 🧪 or 📐 → `io.tlaloc.ir.AllReduceAttrs` (distributed is 📐, never run on two hosts, v1 is `"sum"` only); (2) the declaration's own KDoc scopes itself to "v1" and names what it would have to become → the four-worlds taxonomy (`KernelScope`, `OrchestrationScope`, `ProgramScope`, `ClusterScope`, `Tlaloc`, `BufferHandle`, `HandleRef` — "v1 keeps each op single-scope", with context parameters named as the alternative, plus `BufferHandle`'s "v1 stub" payload and single-threaded refcount); (3) the API *selects* behaviour whose result is uncertified → `io.tlaloc.ir.recognizer.kernel` and `io.tlaloc.ir.recognizer.cost` (six files scope themselves to v1, `DeviceDescriptor`'s numbers are dated `v1 (2026-05)`, and the only kernel measured against XLA lost at small shapes). **NOT marked, deliberately:** `grad` and the transformation family, the tensor and op surface, `:nn`, safetensors, StableHLO emission, PJRT and IREE. A marker on everything teaches a reader to add `-opt-in=` once and stop reading it, which is worse than no marker. |
| **4c. The marker refuses a real consumer** | ✅ | `ExperimentalApiOptInTest` runs a real `K2JVMCompiler` with **no** `-opt-in` argument: the taxonomy refuses naming the marker AND carrying its message, `@OptIn` makes the identical source compile, the kernel surface and `AllReduceAttrs` refuse, and the certified surface compiles with no mention of the marker | Also checked from outside the build: `examples/internals/four-worlds` (both source sets) and `examples/internals/layer3` needed `@file:OptIn` and now carry it with a comment saying why. Those two examples are the only consumers of a marked surface in the repository, which makes them the check. |
| **4d. Tlaloc's own modules opt in at the build level** | ✅ (and a stated cost) | Root `build.gradle.kts`: `optIn.addAll(...)` on every `KotlinCompilationTask` in the ten modules that can see `:core` | The kotlinx convention. **The cost, stated rather than hidden:** nothing inside this build notices when repository code uses a provisional API. What notices is every consumer, and the two tests above. `:kptx` and `:runtime-cuda` are excluded from the flag because they depend on nothing of Tlaloc's — passing `-opt-in=` a class they cannot resolve makes the compiler warn on every compilation, which is exactly the build-log noise §0.4.499 spent a tier removing. |
| **5. A binary-compatibility baseline** | ✅ | `api/<module>.api` committed (4,396 lines over six modules); `apiCheck` wired into `check` by the plugin, so `./gradlew test` runs it. **Negative-tested**: adding `fun tlalocApiCheckNegativeProbe(): Int` to `:stablehlo` made `:stablehlo:apiCheck` FAIL and print the one-line diff | binary-compatibility-validator 0.18.2, applied at the root. Experimental API is **in** the dump on purpose — excluding it (`nonPublicMarkers`) would have made the marker a hole in the gate. |
| **5b. It covers six modules of eleven** | 🧪 for the other five | The failure is reproducible and verbatim: `A failure occurred while executing kotlinx.validation.AbiBuildWorker > Unsupported class file major version 69` | **A measured tool limit, not a choice.** Major 69 is Java 25, and BCV 0.18.2's ABI reader cannot parse it. So the gate covers exactly the six modules §0.4.503 lowered to Java 21 — `:core`, `:ir`, `:autograd`, `:nn`, `:stablehlo`, `:maestro`, which is the library surface a consumer compiles against — and not `:runtime-pjrt`, `:runtime-cuda`, `:kptx`, `:runtime-iree`, `:compiler-plugin`. The ignore list is **derived from `tlalocJvmTargets`**, so a module lowered to 21 later starts being validated with no one remembering to come back. ⬜ row: widening it needs either a BCV release with a newer ASM or a second tool. |
| **6. The IDE claim matches the evidence** | ✅ for the build error, 🧪 for the IDE | `DiagnosticSourcePositionTest` (3 tests, `:compiler-plugin`): `LAMBDA_NOT_LOWERABLE` and `NAMED_INDEX_MISMATCH` each report at the offending call's own **file, line and column** (the line located by its own text, not hardcoded), and no Tlaloc ERROR is ever emitted without a position | **The capability is not deleted; the claim is narrowed to what is tested.** Eighteen `:compiler-plugin` test classes already asserted diagnostic *text*; not one had read `location`, so "red squiggle" rested on nothing. README, `GETTING_STARTED.md` (header and §4) now claim the build failure at file:line:column and call the IDE redline *expected, untested*, with a 🧪 row in `docs/CAPABILITIES.md`. Nothing here drives IntelliJ and there is no IDE plugin. |

### What this tier found that was not in its brief

- **The README contradicted itself about Symja, and the contradiction was two
  commits old.** Its `### Requirements` said Symja "is **optional** since
  `0.1.0-alpha01`" — correct, §0.4.503 made it `compileOnly` — while its "Before
  you invest" bullet and its whole `## License` section said `ir-jvm` "carries
  Symja … at runtime scope, with no supported way to opt out yet". §0.4.503 changed
  the dependency and did not come back to the two places that described it. Both
  now say the same thing, and the part that *is* still true — no Symja-free
  `SymbolicEngine` exists, so a policy forbidding LGPL-3.0 outright leaves you
  without a CAS — is stated as the residual risk rather than dropped.
- **`docs/GETTING_STARTED.md`'s list of published modules omitted `:nn`.** Ten of
  eleven were listed; the missing one is the layers and optimizers, which is
  plausibly what a reader came for.
- **"All four" plugin options had been five since §0.4.503.**
  `unsafeAllowUnsupportedKotlin` was added by the Kotlin version guard and §4a was
  not updated, so the table a reader is pointed at was missing the option that
  decides whether the plugin runs at all.
- **Three test counts, three numbers, none of them current.** README said 2,345,
  `docs/COMPATIBILITY.md` said 2,345, `docs/CAPABILITIES.md` said 2,509. All three
  now carry the number this tier measured.
- **The correct annotation retention is the one reflection cannot see.** An opt-in
  marker must be `AnnotationRetention.BINARY`; Kotlin's BINARY is Java's
  `RetentionPolicy.CLASS`; and `Class.getAnnotations()` returns only `RUNTIME`
  annotations. The first version of `ExperimentalTlalocApiTest` therefore asserted
  three things about an empty array and would have passed with the marker applied
  to nothing. It reads the class files' constant pools now, which is also the
  honest statement of the claim: the requirement is in the bytes a consumer's
  compiler reads. (Kotlin additionally refuses `filterIsInstance<RequiresOptIn>()`
  — "This class can only be used as an annotation" — which is how the retention
  problem surfaced.)
- **Dokka's warning count is 206, over 110 distinct link targets, and `:ir` owns
  117 of them.** Tier 0 named these warnings as real defects and recorded that
  nobody had counted them. Counted now; still unswept, because sweeping 110 KDoc
  references is its own commit.

### Suite state at Tier 4 close

| | |
|---|---|
| `./gradlew test --rerun-tasks` | BUILD SUCCESSFUL, **139 tasks executed, 3m35s** |
| `bash scripts/count-tests.sh` | **2522** = 2509 at §0.4.504 + **13 new tests** (5 `ExperimentalTlalocApiTest`, 5 `ExperimentalApiOptInTest`, 3 `DiagnosticSourcePositionTest`). Of that total, 51 come from `third-party/maestro` results present in the tree — the same caveat §0.4.504 recorded |
| Failures across all 377 `TEST-*.xml` | none |
| `./gradlew apiDump` | writes six baselines, 4,396 lines total |
| `./gradlew apiCheck` | green; **negative-tested** — one added public function in `:stablehlo` makes it fail with the diff |
| `./gradlew apiDocs` | 2,898 HTML pages, 46 MB, eleven modules, 206 KDoc-link warnings |
| `bash scripts/onboarding-smoke.sh` | passes |
| **Nine** of the ten examples | ran; the two that now need `@file:OptIn` were re-run after adding it, including `four-worlds`'s deliberately-failing `shapeError` source set. **`gpu-inference` was NOT re-run at §0.4.505** — it writes 4.1 GiB and then needs a separate Python process, and it touches no marked API and no file that tier changed. *(Corrected at §0.4.506: this row originally read "Every one of the ten examples | ran", which contradicted this tier's own handover. Both halves of `gpu-inference` were then run at §0.4.506 — see the final-verification section.)* |

**One real break this tier caused, found by the suite and fixed.**
`WorldScopeDisciplineTest` compiles five snippets against `:core` through
`K2JVMCompiler` directly, so it does not inherit the root build's `-opt-in=` flag.
Marking the four-worlds taxonomy made its two POSITIVE tests fail — and would have
left its three NEGATIVE tests passing for the wrong reason, since they assert
`exitCode != 0` and an opt-in error is a non-zero exit that says nothing about
scope discipline. The harness now prepends the opt-in to every snippet, with the
reasoning in the source. The only other failure in the first full run was
`BGDHyperOptTest`'s known wall-clock ratio flake (`ratio=0.4`, bounds
`0.5 < ratio < 200`), green in isolation and green on the re-run — the same flake
Tier 0 recorded.

### What Tier 4 did not do

- **Sweep the 206 Dokka warnings.** Counted, attributed per module, not fixed.
- **Host anything.** No gh-pages, no MkDocs Material book. `./gradlew apiDocs`
  writes a local site; ⬜ in `docs/CAPABILITIES.md`.
- **Mark `:kptx`.** It meets the third criterion squarely — the kernel library
  whose one measured kernel lost to XLA at small shapes — but `:kptx` declares **no
  dependency on `:core`**, so `@ExperimentalTlalocApi` cannot reach it without
  adding one, and a documentation tier is the wrong place to change a published
  module's dependency graph. Named in the annotation's own KDoc. ⬜.
- **Validate the five Java-25 modules' ABI.** See the 5b row — a tool limit.
- **Test an IDE.** See item 6. Nothing here drives IntelliJ.
- **Run the head-to-head comparison that would close §17 step 16.** It is a
  benchmark run against two Python frameworks, not a documentation item; this tier
  marked the gate unread and left it unread.
- **Add a new §0.4 entry to `DIFFKTX_SPEC.md`.** Per the brief. §14, §17 and §18
  were *corrected*, which is a different act — the §0.4 register itself is
  untouched.

---

## Suite state at Tier 0 close

| | |
|---|---|
| `./gradlew test --rerun-tasks` | BUILD SUCCESSFUL |
| Known flake seen once on the way | `BGDHyperOptTest."paper-faithful bgd-hyperopt at T=50 M=3 — measured timings"` failed one parallel run with `ratio=0.404 outside bounds` (the assertion is `0.5 < ratio < 200`). Re-run in isolation: 9 tests, 0 failures. It is a wall-clock ratio measured on a loaded machine, and §0.4.498 touched no code it exercises. |
| `bash scripts/count-tests.sh` | 2345 (unchanged — Tier 0 added a Gradle gate, not JUnit tests) |
| `bash scripts/onboarding-smoke.sh` | passes, against `0.1.0-alpha01` |
| `./gradlew publishToMavenLocal` | succeeds; POMs in `~/.m2` carry all six Central-mandatory elements |

The count is deliberately unchanged. `verifyPomMetadata` is a Gradle
verification task, not a JUnit test, because what it checks is a *build output* —
a POM this build generated — and a JUnit test that shelled out to Gradle to
produce one would be slower, flakier and less precise about which module failed.
It is wired into `check`, so it runs in exactly the command the repository already
treats as its gate.

---

## Final verification (§0.4.506, 2026-09-22)

The arc's closing tier. It changed **no Kotlin, no Gradle logic and no test** — it
re-ran every claim the five tiers above made about this repository as a whole, from
a clean room, and wrote down what it observed rather than what the tiers reported.
Two things came out of it that the tiers had not said: one closure (a deferral that
is no longer deferred) and one asymmetry (a resolver that §0.4.503 fixed on one side
of the serving seam and not the other).

### The gauntlet, as observed

Every row below was executed here, in this order, one Gradle invocation at a time.

| Step | Observed |
|---|---|
| `./gradlew test --rerun-tasks` | **BUILD SUCCESSFUL in 3m 32s — "139 actionable tasks: 139 executed"**. Zero up-to-date, so it is a genuine re-execution and not a no-op |
| Failures / errors | **none**: all 377 `TEST-*.xml` files report `failures="0"` and `errors="0"` |
| `bash scripts/count-tests.sh` | **2522**. Baseline at the start of this arc was 2345, so **+177**; the count did not go down |
| Where the 2522 come from | **2471 from the root suite + 51 from `third-party/maestro`.** The Maestro XMLs are dated 2026-09-21 17:32 and 2026-09-22 22:31 — *before* this run's 23:37–23:40 — so root `./gradlew test` did not re-execute them, exactly as §0.4.504 said. The impurity §0.4.504 named is now measured: the reproducible root-suite number is **2471** |
| Skips | **93**, all self-skips by name: 76 `RoundTripTest` + 7 `SdyRoundTripTest` + 5 `SdyPropagationTest` (no `stablehlo-translate` / `sdy-opt`) and 5 `PjrtTpuSmokeTest` (no TPU) |
| Known flakes | **neither fired.** `BGDHyperOptTest` (Tier 0, Tier 4) and `KptxPagedAttentionBenchTest` (Tier 3) were both green in this clean-room run. Recorded because "green once" is not the same as "not a flake" — they remain wall-clock assertions on a shared machine |
| `bash scripts/onboarding-smoke.sh` | **passes, exit 0** — `publishToMavenLocal` then `examples/quickstart run`, printing the `[7.0, 11.0, 9.0, 13.0]` gradient. The `0.1.0-alpha01` coordinate is consistent across all 19 files that name it; the only surviving `0.0.1-SNAPSHOT` strings in the tree are three *historical* mentions (a `CHANGELOG` entry, a `build.gradle.kts` comment, an `ALPHA_PLAN` row) |
| Ten example projects | **all ten exit 0** — `quickstart`, `named-indices`, `readable-gradients`, `differentiable-physics`, `mnist`, `gpu-training`, `internals/four-worlds`, `internals/layer3`, `internals/tpu`, `gpu-inference`. Verified individually that each did real work or skipped BY NAME; none passed silently |
| `./gradlew -p examples/quickstart shapeError` | **BUILD FAILED, as it must** — `e: ShapeError.kt:30:5 Tlaloc named-index mismatch: contract operands share no named axis: lhs=[Batch, SeqLen] rhs=[Hidden, Hidden]`. This is also the independent pin for Tier 4 item 6: the diagnostic carries file, line and column |
| `compileKotlin --rerun-tasks` (Tier 1's regression) | **0 lines matching `^w:`**, no Tlaloc output of any kind, on a genuine re-execution |
| The same compile with `allWarningsAsErrors` | **BUILD SUCCESSFUL.** And the green is not vacuous: an instrumented copy of the init script printed `WERROR-APPLIED-TO: :compileKotlin (org.jetbrains.kotlin.gradle.tasks.KotlinCompile_Decorated)`, so the flag demonstrably reached the task that compiles `grad {}` |
| The published metadata | **All 21 POMs** under `~/.m2/repository/io/tlaloc/` carry `<name>`, `<description>`, `<url>`, `<licenses>`, `<developers>`, `<scm>` — plus `<issueManagement>` and `<inceptionYear>`. `core-jvm`'s POM was read in full. **All 21 publications** carry both a `-sources.jar` and a `-javadoc.jar`; `core-jvm`'s javadoc jar holds **591 HTML pages / 8.9 MB** of real Dokka output, not an empty shell |
| Symja's scope, checked independently | **0 occurrences of `matheclipse`** in `ir-jvm`'s POM *and* in its Gradle module metadata. Tier 3's `compileOnly` claim confirmed from the published artifact rather than from the build script |
| The JDK split, checked independently | Read out of the **published jars'** class-file major version, not from `verifyJvmTarget`: `core` `ir` `autograd` `nn` `stablehlo` `maestro` → **65** (Java 21); `runtime-pjrt` `runtime-cuda` `kptx` `runtime-iree` `compiler-plugin` → **69** (Java 25). Exactly the `tlalocJvmTargets` map |
| Do the three gates actually ride in `./gradlew test`? | **Yes**, and it was checked rather than assumed: `./gradlew test --dry-run` lists **11 `verifyPomMetadata`** (Tier 0), **12 `verifyJvmTarget`** (Tier 3) and **6 `apiCheck`** (Tier 4) — the six Java-21 modules, which is the exact set Tier 4 said the ABI validator can read. The green suite therefore covers all three |
| `git status` | clean before and after; nothing the runs wrote is untracked |
| AI attribution in this arc's commit messages | **none.** All eight commits `1f4be79..HEAD` were grepped, message body *and* author *and* committer, for `claude`, `opus`, `anthropic`, `co-authored`, `generated with`, `sonnet` and the robot emoji. Every one is clean and authored `Pedro N. Rodriguez <pnrodriguezh@gmail.com>`. No rewrite was needed, so none was made |

### What the final verification found

**1. `examples/gpu-inference` is no longer a deferral — both halves ran here.**
Tier 4 recorded it as the one example it had not re-run ("writes 4.1 GiB … and then
needs a separate Python process"). Both halves were executed for this tier:

- Half one, Kotlin: wrote the artifact from the real cached TinyLlama-1.1B —
  22 layers, 201 staged operand files, 4.1 GiB, one compiled entry
  `decode_b1_c64 -> bodies/8cbf1976….mlir`.
- Half two, `/usr/bin/python3 serve.py`: loaded that directory through
  `CtypesEngine on cuda` and greedily decoded — prompt `[1, 450, 7483, 310, 3444,
  338]`, generated `[3681, 29889, 13, 13, 29906, 29889]`, which the checkpoint's own
  `tokenizer.json` renders as **`' Paris.\n\n2.'`** — the completion the example's
  README promises, on the first line of the file. First step 6652 ms (weight upload
  + XLA compile), median step 1347 ms, 1 XLA compile.

So the repository's headline serving claim — a real Llama, served by a process with
no JVM, no jax, no torch and no numpy in it — is certified at §0.4.506 as well as at
§0.4.490, on this build, by a reader who did not write it.

**2. THE ASYMMETRY: §0.4.503 taught the JVM to find a PJRT plugin, and the serving
runtime still cannot.** This is the one real defect the final pass found, and it is
published rather than fixed, because fixing it is a behavioural change to the
serving runtime and this tier commits documentation only.

The first attempt at half two, run exactly as the example's own hand-off line
prints it (`python3 serve.py --artifact …`), printed:

```
SKIP: no PJRT plugin on this machine, so there is nothing to run on.
```

That sentence is **false about this machine**. There is a plugin at
`~/.local/venvs/iree/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`
(447 MB, present), and `examples/gpu-training` had run on CUDA through it four
minutes earlier without any environment variable set. The cause is a seam:

- JVM side, `PjrtBinaries` (§0.4.503): **globs seven roots** — `$VIRTUAL_ENV`,
  `~/.local/venvs/*`, `~/.venv`, `~/venv`, `~/.local`, `/usr/local`, `/usr` — across
  any venv name, any `python3.N` and any `jax_plugins/*cuda*` package, and reports
  everywhere it looked.
- Python side, `PjrtApi.load` in `harness/python/tlaloc_pjrt.py`: takes an explicit
  path or `TLALOC_PJRT_PLUGIN_PATH` and **searches nothing at all**.

With the variable exported the same script runs the model perfectly (finding 1). So
the honest statements are: the serving runtime **refuses by name and exits 0**, which
is the house rule and is not broken; the requirement **is documented** — the export is
line one of half two in `examples/gpu-inference/README.md`, in the root README and in
`docs/SERVING_RUNBOOK.md`; and the message's own second line names the variable. What
is imprecise is (a) the README frames that SKIP block under "**No GPU?**", which is
one cause of it and not the one a GB10 owner hits, and (b) the Kotlin half's printed
hand-off line omits the export that the README's own code block has. Neither is a
false capability claim. **Deferred, with the reason:** porting the seven-root glob
into `tlaloc_pjrt.py` would make the two halves symmetric and is the obvious repair,
but it is new behaviour in the runtime a deployment ships, and it needs its own tests
on synthetic install trees the way `PjrtCudaPluginResolutionTest` has. ⬜.

   **Resolved (release-readiness row R14).** `tlaloc_pjrt.find_plugin` now searches
   the same roots in the same order, with its own tests on synthetic install trees
   (`vllm_tlaloc_test.PluginDiscoveryTest`), and `serve.py` on the GB10 finds the
   plugin and decodes on CUDA without the variable. ✅.

**3. A third data point on the GPU lane's reproducibility, and it supports the
hedge.** `examples/gpu-training` on CUDA ended this run at **loss 0.047137, 98.0 %
held out, 337/337 parameter scalars bit-identical after the checkpoint round trip**.
That final loss is the **§0.4.486** value — not §0.4.502's `0.046667`, which §0.4.502
observed twice and used to soften the nondeterminism claim. Both recorded values have
now been observed from the same source tree. That is the evidence for exactly what
the example's README says ("XLA autotunes its GEMM kernels at compile time … CAN pick
a different winner per run, and it does not have to") and against either stronger
claim. The 98.0 % figure matches the README's committed block exactly.

**4. The skip count in the repository's own memory is stale, harmlessly.** The
long-standing note "91 MLIR round-trip tests always skip" is now **88** (76 + 7 + 5)
plus 5 TPU smoke tests, for 93 skips in total. Nothing regressed; the suites grew.

### The arc at a glance — Tiers 0 through 4

One row per tier, with the gate that pins it and the deferral that still blocks it.
The per-item tables above remain the authority; this is the roll-up.

| Tier | § | Verdict | The gate that pins it | What still blocks it |
|---|---|---|---|---|
| **0 — legal and distribution** | §0.4.498 | ✅ certified, with three 🧪 rows | `verifyPomMetadata` × 11 inside `check`; `scripts/onboarding-smoke.sh`. Re-verified here from the published artifacts | **Nothing is on Maven Central.** Signing, the `central` repository and `RELEASING.md`'s last two steps have never executed. The namespace decision is no longer open — §0.4.508 chose `io.github.pedronahum` — so what remains is a Sonatype token and a GPG key, neither of which is engineering. ⬜ |
| **1 — first-contact defects** | §0.4.499 | ✅ certified | `DiagnosticNoiseTest` (7); re-verified here as 0 `w:` lines and a green `-Werror` compile with the flag proven to reach the task | The `pluginMissing` runtime text still cannot distinguish "no plugin" from "plugin refused" (🧪). `dumpLoweredIr=true` still cannot be combined with `-Werror`. Both named |
| **2 — item 6, captured values** | §0.4.500 + §0.4.501 | ✅ certified, both slices | `CapturedConstantGradientTest` (12) + `CapturedRuntimeValueGradientTest` (17), every positive one an equivalence against the explicit-parameter or inlined spelling | A captured **tensor**, a captured value-class scalar, a captured property and a captured `var` all refuse by name. ⬜ — the next slice |
| **2 — item 7, persistence** | §0.4.502 | ✅ certified | 83 tests, incl. two cross-language oracle suites that **ran** (`safetensors` 0.8.0, torch 2.11.0+cpu) and a negative control on optimizer state | Sharded and streaming output are 📐. A checkpoint carries no structure and no hyperparameters — permanent, by design, and stated in the refusal itself |
| **2 — the rest** | — | ⬜ **never scoped in this file** | — | Items 1–5 and 8+ of Tier 2 were never in any brief this arc received. Three tiers said so rather than inventing them. **This is the arc's largest unknown** |
| **3 — portability and trust** | §0.4.503 | ✅ certified | `verifyJvmTarget` × 12 + `-Xjdk-release=21` × 6 inside `check`; 41 tests; `scripts/jdk21-smoke.sh` on OpenJDK 21.0.2. Re-verified here from published bytecode | `:compiler-plugin` stays on Java 25, so a **JDK-21-only machine still cannot build `grad {}`**. No Symja-free CAS. KMP variants publish no `org.gradle.jvm.version`. All ⬜ |
| **3 — CI matrix** | §0.4.504 | 🧪 **lanes** / ✅ **their commands** | `actionlint` 1.7.12, zero findings on three workflows; every Gradle command in them run locally (1,822 tests on JDK 21; 1,750 under the 2.4.20 compiler) | **No GitHub Actions run of any lane exists.** Kotlin 2.4.20 does not compile `:compiler-plugin` (one error, published). No aarch64-Linux lane. ⬜ |
| **4 — docs and the public surface** | §0.4.505 | ✅ certified | 13 tests (`ExperimentalTlalocApiTest`, `ExperimentalApiOptInTest`, `DiagnosticSourcePositionTest`) + `apiCheck` inside `check`, confirmed here to be in `./gradlew test` for 6 modules | The ABI baseline covers **6 of 11** modules — a measured tool limit (BCV 0.18.2 cannot read Java 25 bytecode). 206 Dokka warnings counted, unswept. Nothing is hosted. `:kptx` unmarked. ⬜ |
| **5 — final verification** | §0.4.506 | ✅ this section | The gauntlet table above | The serving-side plugin resolver (finding 2). ⬜ |

### The verdict: is this an alpha?

**As an engine: yes, and by a margin.** 2,471 tests re-executed from a clean room
with zero failures; gradients certified against analytic answers and against
PyTorch, JAX and `safetensors` as cross-implementation oracles; a real TinyLlama-1.1B
decoding correct tokens in a process with no framework in it; ten example projects
that all run, one of which is *supposed* to fail and does. The house rules hold under
inspection: every unsupported case this tier could reach refused **by name** and
exited 0, and the one wrong sentence found in a whole day of verification
(`SKIP: no PJRT plugin on this machine`) was wrong about its environment, not about
its own inputs.

**As something a stranger can depend on: not yet — and what is missing is not
engineering.** Three facts, all of them ⬜ and none of them code:

1. **Nothing is on Maven Central.** Every install path in every document begins with
   `git clone`. The wiring exists and is dry-runnable; it has never been run.
2. ~~**The group id is undecided.**~~ **CLOSED in §0.4.508: `io.github.pedronahum`.**
   `io.tlaloc` would have needed a DNS TXT record on `tlaloc.io`, which this project
   does not own; the `io.github.*` form verifies against the GitHub account already
   hosting the repository. 100 coordinate sites across 39 files were rewritten and
   `scripts/onboarding-smoke.sh` re-verified on the new coordinate.
3. **No CI run of any lane exists.** Five 🧪 rows in
   [CAPABILITIES.md](CAPABILITIES.md) become ✅ or become bug reports the first time
   a workflow executes, and every ✅ in that file was certified on exactly one
   machine.

**The shortest remaining path to an alpha a stranger can install**, in the order the
dependencies force:

1. **Decide the group id.** One decision, nobody's but the owner's, and it gates
   every coordinate in every file. (`RELEASING.md` §4.)
2. **Push, and let CI run once.** The lanes are written and lint-clean; a single
   green `ubuntu-latest` + `macos-15` run converts the arc's largest block of 🧪 into
   evidence, and a red one is worth more than the 🧪.
3. **Make one GPG key and run `RELEASING.md`'s last two steps.** That single act
   turns signing, the `central` repository and the release procedure from 🧪 to ✅
   simultaneously — they are one row wearing three hats.

Nothing on that list is a feature. Everything else this arc named — the plugin on
JDK 21, a Symja-free CAS, Kotlin 2.4 support, an ABI baseline for the Java-25
modules, captured tensors, the serving-side resolver, 206 KDoc links, an aarch64
Linux lane, and the unscoped remainder of Tier 2 — is **post-alpha work**, and all of
it is written down: in the tables above, in `CAPABILITIES.md`, in `COMPATIBILITY.md`,
or in the refusal message a user actually sees.

---

## Release-readiness review (2026-09-23)

A review before the first Central upload, run after the repository went public
with credentials and a signing key in place. Five read-only audits (docs vs
code, main-source hygiene, licensing, publishing, plugin and runtime
robustness) plus a fresh-clone publish and quickstart. Each row below is fixed
in the stage named; the status column is updated by that stage's commit.

| # | Item | Sev | Stage | Status |
|---|---|---|---|---|
| R1 | FIR→IR handoff is keyed by `(startOffset, endOffset)` in a JVM-global table: two `grad {}` calls at equal offsets in different files swap gradients; one compilation's `clear()` wipes another's entries in a shared daemon | P0 | A | ✅ table is one instance per compilation, keyed `(file, start, end)`; `PluginRobustnessTest` compiles two files with `grad {}` at identical offsets and checks both gradients (fails on the old key); a lowered call the IR phase never matched (file/offset drift between phases) is a compile error, not a silent run-time throw |
| R2 | IR-phase "kept original call" is a WARNING whatever `strictLowering` says, so a refused body still reaches `pluginMissing()` at run time with a message naming two causes, neither true | P0 | A | ✅ all 26 IR-phase sites report through one `strictLowering`-aware path (ERROR by default, with call-site location and a version-mismatch hint for missing library symbols); success dumps moved behind `dumpLoweredIr`; `pluginMissing()` names all three causes; `PluginRobustnessTest` + `DScalarMixingGradientTest` pin both severities |
| R3 | No top-level guard in the FIR checker or the IR extension: an NPE/CCE in lowering or synthesis is an internal compiler error instead of a diagnostic | P0 | A | ✅ FIR checker and IR extension guard every intrinsic call; unexpected exceptions become "Tlaloc internal error … report at github.com/pedronahum/tlaloc/issues" (exit 1, not 3); control-flow/VM/linkage errors rethrown; pinned by fault-injection tests in both phases |
| R4 | `dumpGradSource` reads `ture` as false; registrar hard-codes the plugin id | P2 | A | ✅ `dumpGradSource=ture` refused by name; registrar uses `PLUGIN_ID`; both pinned in `PluginRobustnessTest` |
| R5 | `ir`, `autograd`, `stablehlo`, `maestro`, `runtime-*` expose `core`/`ir` types but declare them `implementation`: a consumer of `autograd` alone cannot name `DTensor` | P0 | B | ✅ `api` wherever a public signature names another Tlaloc module (ir, autograd, stablehlo, maestro, runtime-iree, runtime-pjrt); published `jvmApiElements` lists them; a consumer outside the repo depending only on `tlaloc-autograd` + the plugin compiles and runs `grad {}` over `DTensor` (fails with `Unresolved reference 'core'` when autograd is reverted to `implementation`). Review: `maestro` also needs `api(autograd)`, since `program { }` takes a body over autograd's `Tracer` (erased in `maestro.api`, visible in `javap` generics); a consumer of `tlaloc-maestro` alone failed with `Cannot access class 'io.tlaloc.autograd.Tracer'` and now compiles and runs `program { x -> (x * x).sum() }` |
| R6 | Artifact ids `core`, `ir`, `nn`, `maestro`… are generic inside a personal namespace and permanent once published; rename to `tlaloc-*` | P1 | B | ✅ publication ids rewritten to `tlaloc-*` in the root build (project names unchanged); in a cleared `~/.m2`, only `tlaloc-*` directories exist, `available-at` points at `tlaloc-*-jvm`, every inter-module dependency is `tlaloc-*`; `verifyPomMetadata` now fails on any unprefixed Tlaloc coordinate in a POM or `.module` (negative-tested) and reads the real `module.json` (its Symja check had been reading a file that never exists); every written-out coordinate updated |
| R7 | The OSSRH Staging API never hands an upload to the Central Portal without `POST /manual/upload/defaultRepository/<ns>`; nothing makes that call. An upload without a key goes out unsigned | P0 | B | ✅ `releaseToCentralPortal` = every module's central upload, then `centralPortalHandoff` (POST `/manual/upload/defaultRepository/io.github.pedronahum?publishing_type=user_managed`, Bearer token); a central upload or handoff without key/credentials refuses by name with 0 requests sent; full release rehearsed against a local capture server via `-PcentralStagingApiUrl` (1,255 signed PUTs, then one POST with the expected Bearer header); `publishToMavenLocal` works with all four blanked; nothing sent to Sonatype. Review: under `--continue` the handoff (ordered only by `mustRunAfter`) POSTed after all 21 uploads had refused; it now refuses when any central upload in the same build did not complete (0 requests in that case; 1,255 PUTs + 1 POST on the clean path) |
| R8 | No tag-triggered release workflow | P1 | B | 🧪 `.github/workflows/release.yml`: `v*` tags + workflow_dispatch, refuses a non-tag ref or tag ≠ `v<version>` (script exercised locally for match, mismatch, branch), `./gradlew test`, then `releaseToCentralPortal --no-parallel` with the four named secrets; YAML parsed; not yet run on GitHub |
| R9 | `:maestro` publishes two `@Deprecated` classes and two `main()` entry points that print; its POM description names a step type it does not contain | P1 | B | ✅ `MaestroDescriptor`, `StubExecutor` and their test (5 tests) deleted; the four-worlds example prints step artifacts instead; both `main`s moved to an unpublished `tools` compilation, `exportServingArtifact` (on JDK 21) and `exportLlamaServingArtifact` (TinyLlama, 2 layers) run; POM description reworded; `maestro.api` re-dumped |
| R10 | No Gradle plugin: consumers hand-wire `kotlinCompilerPluginClasspath` per source set and pass options as raw `-P` args; nothing aligns plugin and library versions (no BOM) | P1 | C | ✅ `:gradle-plugin` (`tlaloc-gradle-plugin`, id `io.github.pedronahum.tlaloc`, Java 21 bytecode, plus its unprefixed marker under group `io.github.pedronahum.tlaloc`) applies `tlaloc-compiler-plugin` of its own build-time version to every Kotlin/JVM compilation; `tlaloc { }` maps all five compiler-plugin options; JS/Native/Wasm/Android compilations skipped with a warning naming them, no Kotlin plugin refused by name. `:bom` (`tlaloc-bom`) constrains all 22 published artifacts, and `verifyBomCoverage` (in `check`) fails when a published artifact is missing from it (negative-tested). Both pass `verifyPomMetadata` and sign. `scripts/onboarding-smoke.sh` builds four consumers from mavenLocal: plugin + every option runs `grad {}` and writes `dumpGradSourceDir`; without the plugin the same program fails with the plugin-missing error; a JS target is skipped by name; no Kotlin plugin refuses. Each check was seen failing against a broken plugin (misspelled option, JS applied, refusal removed). `quickstart` and `differentiable-physics` use the plugin + BOM; `readable-gradients` keeps the manual route, now on `kotlinCompilerPluginClasspathMain` (plain `kotlinCompilerPluginClasspath` had put the plugin on its `printed` compilation too). Review: a Tlaloc plugin declared in a root `plugins { }` block with the Kotlin plugin declared only in a subproject died with a bare `NoClassDefFoundError`; it is now refused by name, and the smoke's fifth consumer checks both the recommended multi-project layout (runs) and that one (refused; seen failing with the catch removed). The no-Kotlin refusal no longer says "no Kotlin Gradle plugin" to a `kotlin("android")` project. A consumer on Gradle 8.14.3 with its daemon on JDK 21 applies the plugin, reads `tlaloc { }` and runs `grad { }` (checked by hand, not in the smoke) |
| R11 | `PjrtSession.executeOn` shares one `ExecuteContext` per executable across threads while its KDoc promises thread safety; `close()` can race an in-flight call | P1 | D | ✅ per-executable context lock in `executeOn`, read/write lifetime lock so `close()` waits for in-flight calls, buffers, executables and the client refuse use after the client is closed, and `executeOn`/`execute` refuse a closed input buffer; `PjrtSessionConcurrencyTest` (8 threads × 200 `executeOn` on GB10 CUDA; `closeBlocksUntilAnInFlightCallReturns`; `aClosedBufferPassedToExecuteOnIsRefusedByName`; `aClientAndItsExecutableRefuseUseAfterTheClientIsClosed`) — every negative control fails (SIGSEGV / close returns early) |
| R12 | Neither the JVM nor the Python PJRT binding checks `PJRT_Api.struct_size` / API version before reading fixed offsets | P1 | D | ✅ `PjrtFfm.checkedApi` (JVM, also the FFI registry) and `check_api_compatible` (Python) refuse major ≠ 0 or `struct_size` < 608 by name before reading an offset; GB10 plugin reports 0.104 / 1120 bytes; `PjrtSessionConcurrencyTest` + `PluginDiscoveryTest` |
| R13 | PJRT refusals name the author's venv, drop the search report, give Linux advice on every OS, leak an arena when client creation fails; env-var parse errors do not name the variable | P1 | D | ✅ `PjrtBinaries.requireCudaPlugin` carries the search report and a Linux-only line off Linux; `PjrtSession` releases arena and client when opening fails; env parse errors name the variable (JVM and Python); `runOnPjrt(timeoutSeconds)` removed; IREE refusals carry `IreeBinaries.searchReport`. Pinned by `PjrtSessionConcurrencyTest` (arena negative control fails) |
| R14 | Python `PjrtApi.load` searches nothing; `find_pjrt_plugin` searches only its own interpreter and prefers libtpu regardless of platform | P1 | D | ✅ `tlaloc_pjrt.find_plugin` mirrors the JVM roots, libtpu only for `tpu`, `find_pjrt_plugin` and `PjrtApi.load` delegate; `PluginDiscoveryTest` (11 cases, negative control fails); `serve.py` on GB10 resolves the JVM's plugin with no env var and decodes TinyLlama |
| R15 | IREE: temp files per call removed only at JVM exit; discovery hard-codes `~/.local/venvs/iree` and `/usr/bin/which` | P1 | D | ✅ IREE temp files deleted when each call returns, `IreeModule` is `AutoCloseable`, failed compiles remove their directory; tools found via `$TLALOC_IREE_BIN`, `$VIRTUAL_ENV/bin`, `~/.local/venvs/*/bin`, `PATH` with a search report; `IreeHousekeepingTest` (negative control fails) |
| R16 | `PhiCalculus` catches `Throwable`; `KptxKernels` caches and `KernelResolverRegistry` are unsynchronized globals | P2 | D | ✅ `catch (Exception)` in PhiCalculus (and CoarseningCache, LProfiler); `@Synchronized` KptxKernels caches and `PtxKernelTemplate.specialize`; copy-on-write `KernelResolverRegistry`; `PhiCalculusErrorPropagationTest`, `KptxKernelsConcurrencyTest`, `KernelResolverRegistryConcurrencyTest` all fail with the fixes reverted |
| R17 | Internal section numbers and jargon inside user-visible exception and diagnostic strings (~29 literals) | P1 | E | ⬜ |
| R18 | README: `contract(a, b)` does not compile (it is infix); the training snippet uses example-local helpers; test counts disagree across docs | P0 | E | ⬜ |
| R19 | GETTING_STARTED: no GPU setup, no configuration table (9 env vars/properties undocumented), no troubleshooting, deprecated `kotlinOptions`, dead "Requirements" link, stale shape-error instructions | P1 | E | ⬜ |
| R20 | Section numbers, `/home/pedro` paths and self-narrating prose across user-facing docs; CAPABILITIES contradicts itself about CI; RELEASING names the old `~/.m2` path | P1 | E | ⬜ |
| R21 | Vendored Netflix Maestro: no root NOTICE, modified files not marked as modified, Tlaloc-written files carry a Netflix copyright; ALPHA_PLAN claims no Apache source is redistributed | P1 | F | ⬜ |
| R22 | No CONTRIBUTING, SECURITY, issue/PR templates or dependabot; README names no copyright holder; `docs/papers` text lacks its CC-BY attribution note | P1 | F | ⬜ |
| R23 | Clean-room re-certification of everything above, then an adversarial critic pass over the whole diff | — | G | ⬜ |

Stages run one at a time: builds on this machine never overlap.
