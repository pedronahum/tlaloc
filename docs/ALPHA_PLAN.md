# Alpha plan — the ledger for the road to a usable alpha

**Status: TIER 0 COMPLETE (§0.4.498, 2026-09-22). TIER 1 COMPLETE (§0.4.499,
2026-09-22). TIER 2 ITEM 6 COMPLETE — slice 1 (§0.4.500) and slice 2 (§0.4.501),
2026-09-22. TIER 2 ITEM 7 COMPLETE (§0.4.502, 2026-09-22). The rest of TIER 2,
and TIERS 3–4, ARE NOT YET SCOPED IN THIS FILE.** This document is the running record for the arc
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
| **Signing** | 🧪 | Configuration is exercised on every build: the `signing` plugin is applied to all eleven published modules and the no-key path is what CI and every local build take today | **No key exists on the development machine, so no signature has ever been produced by this wiring.** In-memory key from `signingInMemoryKey` / `SIGNING_IN_MEMORY_KEY`. The publish-task-depends-on-`Sign` ordering workaround is in place but likewise unexercised. First real release certifies it. |
| **Central publishing repository** | 🧪 | The `central` repository is declared unconditionally, so `publishAllPublicationsToCentralRepository` exists and `--dry-run` shows the graph | **Never run against Sonatype.** No credentials on this machine and a Central upload is irreversible. Credentials are nullable by design so the absence is a 401 at the wire, not a configuration-time build failure. |
| **`io.tlaloc` namespace on Central** | ⬜ | — | **Blocking, and a decision nobody has made:** verifying `io.tlaloc` needs a DNS TXT record on `tlaloc.io`; the alternative, `io.github.pedronahum`, verifies against the GitHub account and needs no domain — but changes the group id and therefore every coordinate in the README, the docs and ten example projects. See [RELEASING.md](RELEASING.md) §4. |
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

## Tier 3

⬜ Not scoped in this file yet.

## Tier 4

⬜ Not scoped in this file yet.

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
