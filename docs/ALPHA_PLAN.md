# Alpha plan — the ledger for the road to a usable alpha

**Status: TIER 0 COMPLETE (§0.4.498, 2026-09-22). TIERS 1–4 NOT YET STARTED, AND
NOT YET SCOPED IN THIS FILE.** This document is the running record for the arc
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

## Tier 1

⬜ Not scoped in this file yet — see the note above. The agent that lands it adds
its rows here.

## Tier 2

⬜ Not scoped in this file yet.

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
