# Releasing Tlaloc

Tlaloc publishes to Maven Central under the group `io.github.pedronahum`. Every
artifact id starts with `tlaloc-`. A release is 23 `tlaloc-*` artifacts plus the
Gradle plugin marker, 24 publications in all:

| Module | Artifact (Gradle resolves the `-jvm` one for you) |
|---|---|
| `:core` | `tlaloc-core`, `tlaloc-core-jvm` |
| `:ir` | `tlaloc-ir`, `tlaloc-ir-jvm` |
| `:autograd` | `tlaloc-autograd`, `tlaloc-autograd-jvm` |
| `:nn` | `tlaloc-nn`, `tlaloc-nn-jvm` |
| `:stablehlo` | `tlaloc-stablehlo`, `tlaloc-stablehlo-jvm` |
| `:maestro` | `tlaloc-maestro`, `tlaloc-maestro-jvm` |
| `:runtime-pjrt` | `tlaloc-runtime-pjrt`, `tlaloc-runtime-pjrt-jvm` |
| `:runtime-iree` | `tlaloc-runtime-iree`, `tlaloc-runtime-iree-jvm` |
| `:runtime-cuda` | `tlaloc-runtime-cuda`, `tlaloc-runtime-cuda-jvm` |
| `:kptx` | `tlaloc-kptx`, `tlaloc-kptx-jvm` |
| `:compiler-plugin` | `tlaloc-compiler-plugin` |
| `:gradle-plugin` | `tlaloc-gradle-plugin` |
| `:gradle-plugin` (plugin marker) | `io.github.pedronahum.tlaloc:io.github.pedronahum.tlaloc.gradle.plugin`, a POM that points `plugins { id("io.github.pedronahum.tlaloc") }` at `tlaloc-gradle-plugin` |
| `:bom` | `tlaloc-bom`, which constrains the other 22 |

The Gradle project names have no prefix; the root `build.gradle.kts` adds it to
every publication. The plugin marker's group and artifact id are fixed by Gradle's
plugin-marker convention and carry no prefix.

## What Central requires, and where it comes from

| Requirement | Where it is satisfied |
|---|---|
| a verified namespace (`io.github.pedronahum`) | the Central Portal, verified against the GitHub account |
| `<name>`, `<description>`, `<url>`, `<licenses>`, `<developers>`, `<scm>` | `moduleDescriptions` and the `pom { }` block in the root `build.gradle.kts` |
| a `-sources.jar` per artifact | the KMP plugin; `withSourcesJar()` in `:compiler-plugin` |
| a `-javadoc.jar` per artifact | `dokkaJavadocJar` (Dokka HTML) |
| a PGP signature per file | `signing` with an in-memory key |
| no `-SNAPSHOT` version | `version` in the root `build.gradle.kts` |

`./gradlew verifyPomMetadata` (part of `./gradlew test`) fails the build if a POM
misses one of these elements, a publication lacks a javadoc or sources jar, Symja
appears in a POM or `.module` file, or any Tlaloc coordinate in the published
metadata lacks the `tlaloc-` prefix.

## 1. Pick the version

1. Set `version` in the root `build.gradle.kts`.
2. Update the coordinate where it is written out: `README.md`,
   `docs/GETTING_STARTED.md`, `examples/**/build.gradle.kts`.
   `grep -rn '<old-version>' --include='*.md' --include='*.kts' .` finds them.
3. Add the `CHANGELOG.md` entry, including what broke, headed with the release
   date (not `unreleased`).
4. For the first release to Central, the docs still say nothing is published.
   Change, before tagging (the tagged sources are what the release ships):
   - `README.md`: the Alpha note ("Nothing is on Maven Central yet"), the Install
     lead ("Nothing is published yet. Build once…" and its `publishToMavenLocal`
     block), and the first item under Limits;
   - `docs/COMPATIBILITY.md`: the "Nothing is on Maven Central yet" line under the
     current version;
   - `docs/GETTING_STARTED.md`: the note at the top and section 1, which have the
     user publish to `mavenLocal`.
   `grep -rn 'Maven Central yet\|Nothing is published\|Nothing is on Maven\|mavenLocal' README.md docs/*.md`
   finds them.
5. Move the install blocks to the new pair. Between releases the `kotlin("jvm")`
   version, the plugin version and the BOM version in the `README.md` and
   `docs/GETTING_STARTED.md` install blocks name the *released* artifact, while
   `gradle/libs.versions.toml` and `build.gradle.kts` are already ahead. At
   release, set all three to the new Kotlin (`libs.versions.kotlin`) and the new
   Tlaloc version, together with the prose that quotes them: the Kotlin range in
   GETTING_STARTED section 0, 4a and Troubleshooting, the README's "Also" line and
   Quickstart line, and the Tlaloc row in the `docs/COMPATIBILITY.md` Kotlin table
   ("next release" becomes the version).
6. `bash scripts/onboarding-smoke.sh` walks the path a new user takes and builds
   the settings and build files printed in `README.md` and
   `docs/GETTING_STARTED.md`, with the Kotlin and Tlaloc versions replaced by this
   checkout's, against the mavenLocal publish. It prints the pair it found in each
   document and the pair it built with; after step 5 they are the same. Nothing
   fails if step 5 is skipped, because the previous pair still resolves from
   Central.

## 2. Credentials

None of these live in the repository. Put them in `~/.gradle/gradle.properties`
(never the tracked `gradle.properties` in this repository) or in the
environment:

| Gradle property | Environment variable | What |
|---|---|---|
| `signingInMemoryKey` | `SIGNING_IN_MEMORY_KEY` | ASCII-armoured private key. In a properties file: one line, newlines written as a literal `\n` (what `setup-signing-key.sh` writes). In the environment or a GitHub secret: that same line, or the multi-line output of `gpg --armor --export-secret-keys <id>` |
| `signingInMemoryKeyPassword` | `SIGNING_IN_MEMORY_KEY_PASSWORD` | its passphrase |
| `centralUsername` | `CENTRAL_USERNAME` | Central Portal user-token username |
| `centralPassword` | `CENTRAL_PASSWORD` | Central Portal user-token password |

`bash scripts/setup-signing-key.sh` generates a key, publishes the public half to
`keys.openpgp.org` and `keyserver.ubuntu.com`, and writes both signing
properties. Run it in a terminal: it prompts for the passphrase on a TTY.

The Central token comes from **View Account → Generate User Token** in the
Portal. Both halves are generated strings; neither is the GitHub account name.
Generating a new token invalidates the old one.

`publishToMavenLocal` needs none of these. An upload to Central refuses before
sending anything, naming what is missing, if the signing key or either
credential is absent. A blank value counts as absent, so
`-PsigningInMemoryKey=` on the command line switches off a key set in
`~/.gradle/gradle.properties`.

## 3. Rehearse locally

First, the commit to be tagged must be pushed and green on every lane of
[`build.yml`](../.github/workflows/build.yml) (x86_64 Linux, aarch64 Linux, macOS,
JDK 21).

```bash
./gradlew test --rerun-tasks
bash scripts/count-tests.sh                      # must not go down
rm -rf ~/.m2/repository/io/github/pedronahum     # no stale artifacts
./gradlew publishToMavenLocal
bash scripts/onboarding-smoke.sh
./gradlew releaseToCentralPortal --dry-run       # the task graph; nothing is sent
```

Then read a POM:

```bash
v=0.1.0-alpha01
cat ~/.m2/repository/io/github/pedronahum/tlaloc-core-jvm/$v/tlaloc-core-jvm-$v.pom
ls  ~/.m2/repository/io/github/pedronahum/tlaloc-core-jvm/$v/
```

It should carry `<name>`, `<description>`, `<url>`, `<licenses>`,
`<developers>` and `<scm>`, next to a `-sources.jar`, a `-javadoc.jar` and an
`.asc` for every file.

## 4. Release

Either from a machine with the credentials:

```bash
./gradlew releaseToCentralPortal --no-parallel
```

or from GitHub Actions: push a tag `v<version>` and
[`release.yml`](../.github/workflows/release.yml) runs. It checks that the tag
equals `v` + the project version, runs `./gradlew test`, then
`releaseToCentralPortal`. Before the upload it refuses if the version's
`tlaloc-core` POM is already on `repo1.maven.org`, and it runs
`scripts/onboarding-smoke.sh`. It reads the four credentials from repository secrets
named `SIGNING_IN_MEMORY_KEY`, `SIGNING_IN_MEMORY_KEY_PASSWORD`,
`CENTRAL_USERNAME` and `CENTRAL_PASSWORD`. It can also be started by hand
(workflow_dispatch) on a tag; on a branch it refuses.

`releaseToCentralPortal` does two things:

1. `publishAllPublicationsToCentralRepository` in every module uploads the
   signed artifacts to the OSSRH Staging API
   (`ossrh-staging-api.central.sonatype.com`).
2. `centralPortalHandoff` sends
   `POST /manual/upload/defaultRepository/io.github.pedronahum?publishing_type=user_managed`
   with the same token. The Staging API does not forward an upload to the Portal
   without this call, and it must come from the same IP address as the upload,
   which is why both run in one invocation. `--no-parallel` keeps the uploads in
   one sequence. If any upload in the same run failed (including under
   `--continue`), the handoff refuses and sends nothing.

If the POST fails after a successful upload, rerun `./gradlew centralPortalHandoff`
alone from the same machine. `-PcentralPublishingType=automatic` releases without
the manual step below; the default is `user_managed`.

Then open <https://central.sonatype.com/publishing/deployments> (Publish →
Deployments; not "Publish Component", which is a separate bundle-upload form)
and check the deployment before pressing **Publish**: it must be VALIDATED and
list 24 components, the 23 `tlaloc-*` artifacts and the plugin marker
`io.github.pedronahum.tlaloc:io.github.pedronahum.tlaloc.gradle.plugin`. The
marker is in the sub-group `io.github.pedronahum.tlaloc`; for 0.1.0-alpha01 it
landed in the same deployment. If it is ever missing, drop the deployment instead
of publishing it: without the marker, `plugins { id("io.github.pedronahum.tlaloc") }`
fails with "plugin not found". A published version is immutable and cannot be
deleted (`COMPATIBILITY.md`).

The same check from a terminal, with the token encoded as for the handoff:

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  "https://central.sonatype.com/api/v1/publisher/status?id=<deployment id>"
```

The deployment id is `portal_deployment_id` in
`GET https://ossrh-staging-api.central.sonatype.com/manual/search/repositories?ip=any&profile_id=io.github.pedronahum`.

Artifacts reach `repo1.maven.org` a few minutes after Publish.
`TLALOC_SMOKE_FROM_CENTRAL=1 bash scripts/onboarding-smoke.sh` then builds the
install blocks in the README and GETTING_STARTED against Central alone.

## 5. After

- If released locally, tag it: `git tag v<version> && git push origin v<version>`.
  Pushing the tag starts `release.yml`. Push it only once
  `https://repo1.maven.org/maven2/io/github/pedronahum/tlaloc-core/<version>/`
  exists: `release.yml` then refuses before uploading anything. Pushed earlier,
  the run uploads a second deployment, which the Portal will not publish over the
  first; drop it there. The tag route alone (section 4) avoids this.
- Copy the `CHANGELOG.md` entry into the GitHub release.
- Record the release (date and artifact URL) in [ALPHA_PLAN.md](ALPHA_PLAN.md).
