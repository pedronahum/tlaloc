# Changelog

All notable changes to Tlaloc are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) **with the alpha
carve-outs written down in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)** —
read that before depending on a coordinate.

A note on where the history actually lives: this file starts at
`0.1.0-alpha01`. The 497 sections of work before it are recorded in commit
titles (`§0.4.NNN <area>: <what turned out to be true>`) and, up to §0.4.311, in
[`DIFFKTX_SPEC.md`](DIFFKTX_SPEC.md). This file does not attempt to
retro-summarise them; it is the record from the first named version forward.

## [Unreleased]

### Added

- **A `grad {}` body can reference a compile-time constant declared outside it.**
  Until now a `grad {}` lambda could reference *nothing* outside itself: the
  lowering resolved property accesses against its own environment (lambda
  parameters and lambda-local `val`s) and every other reference — a top-level
  `const val` included — was refused with `reference to symbol outside the
  lowering scope`. A captured reference the compiler can resolve to a constant is
  now folded into the lowered IR as **exactly** the `DxirConst` an inline literal
  would have produced, so the reverse transform, the φ-calculus coarsening, the
  synthesized bytecode and the printed gradient source cannot tell the two
  spellings apart. What folds: any `const val` (top level, file level, or in a
  companion / named `object`), a `const val` whose own initializer is constant
  arithmetic, and a top-level or enclosing-function `val` whose initializer the
  compiler can fold — including an `Int` constant used as a `for` loop's trip
  count, which takes the same unrolled path a literal bound does rather than the
  symbolic-trip-count one. `examples/differentiable-physics` now uses its own
  `const val`s, and the derivative the compiler writes for it is byte-identical
  to the one it wrote from the inlined literals.
- **A captured RUNTIME value refuses with its own wording**, distinct from the old
  generic out-of-scope sentence: a `var`, a computed `val`, a parameter of the
  enclosing function or a non-`const` member property is named, the reason it is
  not foldable is named, and the two ways out (`const val`, or a lambda
  parameter) are named. Runtime capture itself is not implemented; see
  [docs/ALPHA_PLAN.md](docs/ALPHA_PLAN.md).

### Changed

- **A working build is silent.** Every recognised `grad {}` used to emit two
  compiler **warnings**, each dumping the lowered Tlaloc IR into the consumer's
  build output: the FIR checker's `LAMBDA_LOWERED` and the IR extension's "saw
  handoff". They were developer introspection, and because they were warnings
  they also **broke the build outright** in any project compiling with
  `allWarningsAsErrors = true` (`e: warnings found and -Werror specified`) — on
  a program that was entirely correct. Both are now off by default; the IR half
  is an `INFO` rather than a `WARNING`. Turn them back on with
  `-P plugin:io.tlaloc.plugin:dumpLoweredIr=true`. `dumpGradSource` /
  `dumpGradSourceDir` are unchanged.
- **An unlowerable `grad {}` lambda is a compile-time error.** It was a warning;
  the call was then left unrewritten and `io.tlaloc.autograd`'s fallback body
  threw `IllegalStateException` at the **first call**, telling the user to add a
  compiler plugin that was already applied. The refusal now happens at the call
  site at compile time, carrying the lowering's own verbatim reason (one of the
  ~208 named `LoweringException` sites) and naming the opt-out. Opt out with
  `-P plugin:io.tlaloc.plugin:strictLowering=false`, which restores the warning
  and the late failure exactly.
- **`grad(::f)` and other non-lambda arguments are refused by name** instead of
  compiling green and throwing at the first call: the plugin lowers the body of
  a `{ }` written at the call site, and now says so.
- **The runtime message when a `grad {}` was not rewritten** no longer claims the
  compiler plugin is missing. It names the two states that can produce it — no
  plugin on the compile classpath, or a plugin that refused the body under
  `strictLowering=false` — and says where the compile-time reason is.

### Fixed

- **An f64 scalar body with a literal constant no longer kills the compiler.**
  `DxirReverseTransform`'s scalar constant folding built every folded constant
  from a `Float` projection of its operands, whatever the node's dtype was, so an
  f64 node folded to a constant typed `f64` carrying a `java.lang.Float`. Three
  phases later `DxirToIrSynthesis`'s `v as Double` threw a bare
  `ClassCastException` out of the K2 IR generation extension, with no Tlaloc
  diagnostic of any kind: `grad { x: Double -> x * 1.5 }` neither produced a
  gradient nor refused. The f64 arm of the fold now does its arithmetic in
  `Double`, so the value has the width its type claims. Found while writing the
  captured-constant tests above, on the *inlined-literal control*, and unrelated
  to captures.
- **The Tracer-capture route no longer draws a spurious refusal.**
  `io.tlaloc.autograd.grad` / `grad2` / `grad3` / `valueAndGrad*` are overloaded:
  the compile-time intrinsic and the runtime `Tracer` tape share every one of
  those names. The plugin's checker matched on the FQN alone, so a
  `grad { t: Tracer<...> -> ... }` call — the documented plugin-free route — got
  a `could not lower lambda: ... unsupported type io.tlaloc.autograd.Tracer`
  warning on every call. A Tracer-typed lambda is now recognised as the tape
  overload and not diagnosed at all. (Had it not been, promoting the refusal to
  an error would have broken that route outright.)

## [0.1.0-alpha01] — 2026-09-22

The first named version. The engine did not change in this release; the
*packaging* did, from "unpublishable" to "publishable but not yet published".

### Added

- **`LICENSE` — Apache-2.0.** The repository had no license file, which made it
  "all rights reserved" by default and contradicted its own "developed in the
  open" framing. Copyright 2026 Pedro N. Rodriguez. The README's `## License`
  section states the one dependency that needs a paragraph (Symja).
- **Maven Central metadata.** Every published POM now carries `<name>`,
  `<description>`, `<url>`, `<inceptionYear>`, `<licenses>`, `<developers>`,
  `<scm>` and `<issueManagement>`. Before this, they carried coordinates and
  dependencies only, which Central rejects.
- **`verifyPomMetadata`** — a `check`-wired Gradle task, per module, that reads
  the POMs the build actually generates and fails by name on a missing
  Central-mandatory element, a publication with no javadoc artifact, or a module
  with no sources jar. The "Central-ready" claim is a gate, not a sentence in a
  commit message.
- **Dokka + a javadoc jar per publication.** Dokka 2.2.0 (the tooling item
  `DIFFKTX_SPEC.md` §14 listed and nobody wired). The jar carries Dokka **HTML**
  under the `javadoc` classifier, because Dokka's Javadoc format does not support
  Kotlin Multiplatform projects and ten of the eleven modules are KMP.
- **A sources jar for `:compiler-plugin`.** The KMP modules got one from the
  multiplatform plugin; this module's `from(components["java"])` publication
  published the binary jar alone and could never have passed validation.
- **Signing and the Central deploy repository.** In-memory GPG key from
  `signingInMemoryKey` (Gradle property) or `SIGNING_IN_MEMORY_KEY` (environment);
  credentials from `centralUsername` / `centralPassword` or `CENTRAL_USERNAME` /
  `CENTRAL_PASSWORD`. Both are **no-ops when absent**, so a contributor's
  `publishToMavenLocal` and the GitHub build lane are unaffected.
- **`CHANGELOG.md`** (this file), **[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)**
  (what alpha promises and what it does not) and
  **[docs/RELEASING.md](docs/RELEASING.md)** (the release procedure).
- **[docs/ALPHA_PLAN.md](docs/ALPHA_PLAN.md)** — the ledger for the road to a
  usable alpha, with one row per item and what pins each.

### Changed

- **Version `0.0.1-SNAPSHOT` → `0.1.0-alpha01`**, across the root build, the
  README, `docs/GETTING_STARTED.md` and all ten standalone example projects.
  The old coordinate no longer resolves; republish with
  `./gradlew publishToMavenLocal -x test`.

### Fixed

- **Two stale license claims in the repository's own words.**
  `SymbolicEngine.kt`'s KDoc and `docs/STAGE_B_PLAN.md` §3.2.1's bullet list both
  still described Symja as Apache-2.0, five months after §0.4.13 corrected the
  rest of that document to LGPL-3.0. Both now say LGPL-3.0, and both now record
  the discrepancy §0.4.498 found: the *artifact's* POM says LGPL-3.0 while the
  *repository's* `license.txt` is plain GPL-3.0.

### Not done, deliberately

- **Nothing has been published to Maven Central.** There are no credentials on
  the machine this work was done on and a Central upload is irreversible. The
  wiring is correct and dry-runnable; it has never been run against Central, and
  `docs/ALPHA_PLAN.md` says so in the row that would otherwise claim it.

[Unreleased]: https://github.com/pedronahum/tlaloc/compare/main...HEAD
[0.1.0-alpha01]: https://github.com/pedronahum/tlaloc/releases/tag/v0.1.0-alpha01
