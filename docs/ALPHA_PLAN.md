# Alpha plan — the ledger for the road to a usable alpha

**Status: TIER 0 COMPLETE (§0.4.498, 2026-09-22). TIER 1 COMPLETE (§0.4.499,
2026-09-22). TIER 2 ITEM 6 SLICE 1 COMPLETE (§0.4.500, 2026-09-22); the rest of
TIER 2, and TIERS 3–4, ARE NOT YET SCOPED IN THIS FILE.** This document is the running record for the arc
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
| **Slice 2: runtime capture** | ⬜ | — | Not attempted, by instruction. A captured runtime value has to become an extra operand of the synthesized gradient with no gradient slot of its own, threaded from the call site — the same machinery the customVjp capture story (`CapturingEnv`, §0.4.415) refuses for the same reason. Every refusal message this slice emits names it. |
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
