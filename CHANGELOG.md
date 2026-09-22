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

Nothing yet.

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
