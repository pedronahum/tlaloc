# Releasing Tlaloc

**Status: WRITTEN AND DRY-RUNNABLE, NEVER RUN.** Every command below has been
executed except the two that talk to Sonatype. There are no Central credentials
and no signing key on the machine this was wired on, and a Central upload is
irreversible — so the last two steps are a procedure, not a certified path.
Treat the first release as the thing that certifies this document, and correct it
in the same commit that performs it.

What *is* certified (§0.4.498): the POMs carry everything Central mandates, every
publication carries a javadoc jar and a sources jar, and `verifyPomMetadata`
fails the build by name if that stops being true. What is not: that Central
accepts the bundle.

## 0. What Central requires, and where each piece comes from

| Requirement | Where it is satisfied |
|---|---|
| `groupId` you control (`io.github.pedronahum`) | a Central namespace verification, which for an `io.github.*` namespace is automatic against the GitHub account — see §4 |
| `<name>`, `<description>`, `<url>` | `moduleDescriptions` + the `pom { }` block in the root `build.gradle.kts` |
| `<licenses>`, `<developers>`, `<scm>` | same `pom { }` block |
| a `-sources.jar` per artifact | the KMP plugin; `withSourcesJar()` in `:compiler-plugin` |
| a `-javadoc.jar` per artifact | `dokkaJavadocJar` (Dokka HTML — see the note in `libs.versions.toml`) |
| PGP signature per file | `signing` with an in-memory key (§2) |
| no `-SNAPSHOT` version | `version` in the root `build.gradle.kts` |

The gate: `./gradlew verifyPomMetadata` — or just `./gradlew test`, which depends
on it through `check`.

## 1. Pick the version and record it

1. Set `version` in the root `build.gradle.kts` (one place; `allprojects` reads
   `rootProject.version`).
2. Update the coordinate everywhere it is *written out* for a reader: the README's
   Quickstart and Installation sections, `docs/GETTING_STARTED.md`, and each
   `examples/*/build.gradle.kts` and `examples/internals/*/build.gradle.kts`.
   `grep -rn '<old-version>' --include='*.md' --include='*.kts' .` finds them all;
   §0.4.498 found 16 files this way.
3. Add the `CHANGELOG.md` entry, including what broke. Alpha releases are allowed
   to break things (`COMPATIBILITY.md`); they are not allowed to break things
   silently.
4. `bash scripts/onboarding-smoke.sh` — this is the exact path a new user walks,
   and it fails if step 2 was done partially.

## 2. Credentials (none of which live in the repository)

Signing key, either as Gradle properties in `~/.gradle/gradle.properties` or as
environment variables:

```
signingInMemoryKey=<ASCII-armoured private key, newlines as \n>
signingInMemoryKeyPassword=<passphrase>
# or
SIGNING_IN_MEMORY_KEY / SIGNING_IN_MEMORY_KEY_PASSWORD
```

`bash scripts/setup-signing-key.sh` does all of this — generates the key, pushes
the public half to `keys.openpgp.org` and `keyserver.ubuntu.com`, exports the
private half folded to one line, and writes both properties. **Run it in a real
terminal**: it prompts for a passphrase on a TTY and refuses to run without one,
so that the passphrase cannot be captured by an agent session or a CI log. It
prints the key id and nothing secret.

By hand, if you prefer: `gpg --full-generate-key`, then
`gpg --armor --export-secret-keys <KEY_ID> | sed -z 's/\n/\\n/g'` for the
property value, then `gpg --keyserver hkps://keys.openpgp.org --send-keys <KEY_ID>`
— Central verifies a signature by looking the key up, so an unpublished public
half fails the upload even though the signature itself is valid.

A signing key is what the `Sign` tasks need to exist at all. Without one,
`publishAllPublicationsToCentralRepository` runs and uploads **unsigned**
artifacts, and Central rejects the bundle — `--dry-run` showing zero `Sign` tasks
is the cheap way to notice before the upload.

Central Portal credentials (a user token, not the account password):

```
centralUsername / centralPassword
# or
CENTRAL_USERNAME / CENTRAL_PASSWORD
```

**Both are optional and their absence is a no-op.** With no key, nothing is
signed and no `Sign` task exists; with no credentials, the `central` repository's
publish task exists but fails at the wire. A contributor's
`./gradlew publishToMavenLocal` and the GitHub `build` lane must keep working
with neither, and that is asserted by the fact that both run that way today.

## 3. Rehearse locally

```bash
./gradlew test --rerun-tasks                  # the suite, genuinely re-executed
bash scripts/count-tests.sh                   # must not go down
./gradlew publishToMavenLocal -x test         # the real publication, to ~/.m2
bash scripts/onboarding-smoke.sh              # a consumer resolves it
./gradlew publishAllPublicationsToCentralRepository --dry-run   # the task graph, no upload
```

Then read a POM with your own eyes, because this is the step that would otherwise
be taken on faith:

```bash
cat ~/.m2/repository/io/tlaloc/core-jvm/<version>/core-jvm-<version>.pom
ls  ~/.m2/repository/io/tlaloc/core-jvm/<version>/
```

You are looking for `<name>`, `<description>`, `<url>`, `<licenses>`,
`<developers>`, `<scm>`, and both a `-sources.jar` and a `-javadoc.jar`.

## 4. The two steps that have never run

1. **Verify the `io.github.pedronahum` namespace** at
   [central.sonatype.com](https://central.sonatype.com). Log in with GitHub; an
   `io.github.<account>` namespace verifies against that account, so there is no
   DNS record and no domain to own.

   **§0.4.508 made this decision.** The group id was `io.tlaloc` until then,
   which would have required a TXT record on `tlaloc.io` — a domain this project
   does not own. It was changed before publishing rather than after, because a
   group id is the one part of a coordinate that cannot be corrected later
   without breaking every consumer, and there are none yet.

   **Three names are in play and two of them are not credentials.** Getting this
   wrong is the first thing that happens to anyone doing this step:

   | | value | what it is |
   |---|---|---|
   | GitHub account | `pedronahum` | how you log in to the Portal |
   | Namespace (the groupId) | `io.github.pedronahum` | derived from the account; already in the build |
   | Token username | a generated opaque string | `centralUsername` — **not** the account name |

   Credentials: **Generate User Token** (avatar → View Account) mints a
   username/password PAIR, both random strings. Neither is the GitHub account name
   and neither is the Portal login. It prints them as a Maven `<server>` block;
   `<username>` is `centralUsername` and `<password>` is `centralPassword`. The
   pair is shown ONCE, and generating a new token invalidates the previous one.

   They go in `~/.gradle/gradle.properties`. **Not** in this repository's own
   `gradle.properties`, which is tracked by git — a token pasted there is
   committed and pushed. §2 covers the signing key.
2. **Upload:**

   ```bash
   ./gradlew publishAllPublicationsToCentralRepository
   ```

   then, in the Central Portal, inspect the staged deployment and **Publish**.
   A published version is immutable and cannot be deleted
   (`COMPATIBILITY.md`), so the rehearsal in §3 is the only chance to be wrong
   cheaply.

## 5. After

- Tag: `git tag v<version> && git push --tags`.
- Add the release notes from `CHANGELOG.md` to the GitHub release.
- Correct this document where it was wrong, and move the Tier 0 "release" row in
  [ALPHA_PLAN.md](ALPHA_PLAN.md) from WRITTEN to CERTIFIED with the date and the
  artifact URL. That row is what keeps the repository's own claims honest.
