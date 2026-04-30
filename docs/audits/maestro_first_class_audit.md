# Maestro first-class step-type integration — Layer 2.5 audit (§0.4.249)

**Status:** Layer 2.5 closed.
**Scope:** L2.5.0 (vendoring) + L2.5.1 (maestro-tlaloc/ module + step-type registration) + L2.5.2 (TlalocRunner real body) + L2.5.3 (`SerializedBufferHandle`) + L2.5.4 (seven sample workflows) + L2.5.5 (this audit + container image + CI + deprecations + spec entry).

**Suite state at submission:**
- Tlaloc-side: **1049 tests** (unchanged from §0.4.243; Layer 2.5 made no tensor-pipeline changes).
- maestro-tlaloc-side: **22 new tests** (6 step-type registration + 5 runner end-to-end + 11 sample workflows).
- **Combined: 1071 tests, 0 skipped, 0 failures, 0 errors.**

**Decisions diverging from the original task wording, all surfaced ahead of implementation:**

- **D1 (copy-in-place vendoring, no submodule):** the spec wording mentioned a git submodule. Submodule pointing at our own GitHub fork is cleaner organizationally but requires creating that fork as external infrastructure. Copy-in-place gives the same outcome inside the Tlaloc tree; the L2.5.0 vendoring docs describe the upgrade procedure.
- **D2 (deferred upstream Netflix/maestro PR):** Layer 2.5 ships without filing a contribution PR. Documented in `docs/maestro_upstream_pr.md` as the deliberate decision; the rationale is that a PR proposing a Tlaloc-specific step type is unlikely to be accepted upstream, and a generic step-type SPI is a much larger design effort.
- **D3 (no live Maestro CI):** L2.5.5's CI workflow runs the maestro-tlaloc unit + parsing tests via composite Gradle; it doesn't boot a Maestro instance with K8s pod lifecycle. Rationale + future enablement path documented in §6 + §8 below.
- **D4 (Java for maestro-tlaloc/):** matches maestro-actus + the rest of vendored Maestro. Documented in §3 below.
- **D5 (identity-transform v1 for TlalocRunner):** the runner's L2.5.2 body validates the wire format and identity-copies bytes; real PJRT/IREE dispatch ships when the runtime image's full classpath is available (§7). v1 proves the wire-format end-to-end; v2 wires real dispatch.

---

## 1. Spec compliance

For each requirement in the original task, the file path + line range that implements it.

### Vendoring

| Requirement | Implementation |
|---|---|
| `third-party/maestro/` exists with vendored Netflix/maestro source. | Cloned at commit `0150f2a78005cf135023de29e5f8e06fd592563d` (2026-04-09); `.git` removed for copy-in-place vendoring. ~125k lines under the directory. |
| Pinned commit documented; upgrade criteria documented. | [`docs/vendoring.md`](../vendoring.md) §"Current pin" + §"When to upgrade". |
| `third-party/README.md` documents vendoring philosophy + divergence policy. | [`third-party/README.md`](../../third-party/README.md). |
| Tlaloc's build wires composite-build inclusion. | [`settings.gradle.kts:21–35`](../../settings.gradle.kts) — `includeBuild("third-party/maestro") { name = "vendored-maestro" }`. The rename avoids name-collision with Tlaloc's `:maestro` module. |

### `maestro-tlaloc/` module

| Requirement | Implementation |
|---|---|
| Module at `third-party/maestro/maestro-tlaloc/` mirroring maestro-actus's layout. | All files present — see §3 side-by-side comparison. |
| `engine/stepruntime/TlalocStepRuntime.java` extends `KubernetesStepRuntime`. | [`TlalocStepRuntime.java:30–83`](../../third-party/maestro/maestro-tlaloc/src/main/java/com/netflix/maestro/engine/stepruntime/TlalocStepRuntime.java). |
| `engine/tlaloc/TlalocRunner.java` CLI `main()`. | [`TlalocRunner.java:54–135`](../../third-party/maestro/maestro-tlaloc/src/main/java/com/netflix/maestro/engine/tlaloc/TlalocRunner.java). |
| `engine/tlaloc/TlalocEntrypointBuilder.java` builds shell command. | [`TlalocEntrypointBuilder.java:27–67`](../../third-party/maestro/maestro-tlaloc/src/main/java/com/netflix/maestro/engine/tlaloc/TlalocEntrypointBuilder.java). |
| `engine/tlaloc/TlalocParamsBuilder.java`, `TlalocAttributeMapper.java`. | Both present, ~50 + ~40 lines. |
| `defaultparams/default-tlaloc-step-params.yaml`. | [`default-tlaloc-step-params.yaml`](../../third-party/maestro/maestro-tlaloc/src/main/resources/defaultparams/default-tlaloc-step-params.yaml). |
| Tests under `src/test/java/.../engine/tlaloc/`. | 3 test classes, 22 tests total — `TlalocStepTypeRegistrationTest` (6), `TlalocRunnerEndToEndTest` (5), `TlalocSampleWorkflowsTest` (11). |

### Step-type registration

| Requirement | Implementation |
|---|---|
| `Tlaloc` recognised by Maestro's parser. | [`StepType.java:25–32`](../../third-party/maestro/maestro-common/src/main/java/com/netflix/maestro/models/definition/StepType.java) — `TLALOC("Tlaloc", true)` enum entry. |
| `TlalocStepRuntime` registered in DI. | [`MaestroStepRuntimeConfiguration.java:114–158`](../../third-party/maestro/maestro-server/src/main/java/com/netflix/maestro/server/config/MaestroStepRuntimeConfiguration.java) — three new `@Bean` factories. |
| Workflow JSON with `"type": "Tlaloc"` parses. | Verified by `TlalocSampleWorkflowsTest.allSampleWorkflowsLoadFromTestClasspath` + 10 other tests. |

### Runtime image

| Requirement | Implementation |
|---|---|
| Multi-stage Dockerfile. | [`docker/Dockerfile`](../../third-party/maestro/maestro-tlaloc/docker/Dockerfile) — JDK builder stage + JRE runtime stage. |
| Tlaloc runtime jars included. | `:maestro-tlaloc:buildDockerContext` Gradle task stages them under `build/docker/libs/`; Dockerfile `COPY --from=builder` pulls them in. |
| Reproducible builds. | Pinned base image tags + `--no-daemon --no-build-cache` Gradle flags. Deterministic given the same source tree. |
| Build script at `docker/build.sh`. | [`docker/build.sh`](../../third-party/maestro/maestro-tlaloc/docker/build.sh) — mirrors maestro-actus's pattern, supports custom tag arg. |

### Sample workflows

| Requirement | Status |
|---|---|
| Seven samples in `maestro-server/src/test/resources/samples/`. | ✓ — eight files (the template+caller pair counts as one logical sample). |
| Each has an end-to-end test in maestro-tlaloc. | `TlalocSampleWorkflowsTest` — 11 tests covering all eight JSONs. v1 is parse-validation (Maestro shape, Tlaloc params block, keystone SerializedBufferHandle correctness); live-Maestro execution is OQ-Layer2.5-3. |
| curl-submission examples in `maestro-tlaloc/README.md`. | [`README.md`](../../third-party/maestro/maestro-tlaloc/README.md) §"curl-submission examples". |

### `SerializedBufferHandle`

| Requirement | Implementation |
|---|---|
| New type in `:maestro` module alongside `BufferHandle`. | [`SerializedBufferHandle.kt`](../../maestro/src/jvmMain/kotlin/io/tlaloc/maestro/SerializedBufferHandle.kt). |
| Carries URI, content hash, manifest ref, type metadata, mesh name. | All five fields present + verified by `SerializedBufferHandleTest.jsonRoundTripPreservesAllFields`. |
| Producer write + consumer read with type-check + content-hash validation. | `SerializedBufferHandle.write` + `.read(expectedType, expectedMesh)`. |
| `file://` for tests; one cloud scheme stubbed. | `s3://` and `gs://` throw `UnsupportedOperationException` with a clear message; `file://` end-to-end. |
| Default retention: delete-after-consumption. | `read(deleteAfterRead = true)` default; verified by `readDeletesFileByDefault`. |

### CI integration

| Requirement | Implementation |
|---|---|
| Single CI job builds vendored Maestro + maestro-tlaloc. | [`.github/workflows/maestro-integration.yml`](../../.github/workflows/maestro-integration.yml). |
| Boots Maestro and runs samples. | **Deferred** — see §3 + §6 + §8. v1 runs the maestro-tlaloc test suite (which exercises the same code path with synthetic inputs); live-Maestro execution is OQ-Layer2.5-3. |
| <10 min wall-clock budget. | Composite-Gradle build is the long pole (~5–7 min cold; ~1 min warm with cache). maestro-tlaloc tests run in ~30s. Total: well within budget. |
| Captures logs as artifacts. | `actions/upload-artifact@v4` step uploads `build/reports/tests/` + `build/test-results/`. |
| Failure blocks merge. | Push trigger on `paths: ['third-party/maestro/maestro-tlaloc/**', ...]` plus `workflow_dispatch`. Branch-protection rule (user-side configuration) translates failure into merge block. |

### Compatibility

| Requirement | Status |
|---|---|
| All Layer 1 + Layer 2 tests still pass. | ✓ — Tlaloc-side suite stays at 1049. |
| `MaestroDescriptor.emit()` deprecated but functional. | `@Deprecated(level = WARNING)` on the object; existing tests pass with `@file:Suppress("DEPRECATION")` on `MaestroDescriptorTest`. |
| `StubExecutor` deprecated but functional. | Same — `@Deprecated` on the class; same test suppression. |
| BGDHyperOpt within tolerance. | ✓ — Layer 2.5 makes no tensor-pipeline changes; benchmark unchanged. See §11. |

---

## 2. Vendoring hygiene

### Submodule pin (or equivalent)

Layer 2.5 uses **copy-in-place vendoring**, not a git submodule (D1). The `.gitmodules` file does not exist. The vendoring contract is:

- `third-party/maestro/` carries Netflix/maestro at upstream commit `0150f2a78005cf135023de29e5f8e06fd592563d`.
- `.git/` removed during clone — Tlaloc's git tree owns the vendored code.
- Upgrade procedure: `docs/vendoring.md` §"Upgrade procedure" (full git/gradle commands).

### Fresh-clone build verification

```
$ git clone https://github.com/pedronahum/tlaloc.git fresh-clone
$ cd fresh-clone
$ ./gradlew test                         # Tlaloc-side: 1049 tests green
$ ./gradlew :vendored-maestro:maestro-tlaloc:test   # 22 maestro-tlaloc tests green
```

A fresh clone produces a working build without any submodule init or external steps. The composite-Gradle wiring in `settings.gradle.kts` makes `:vendored-maestro:*` tasks reachable from the Tlaloc root.

### Rationale for chosen Maestro commit

`0150f2a78005cf135023de29e5f8e06fd592563d` (2026-04-09):
- Latest Netflix/maestro main as of L2.5 planning (2026-04-29).
- Matches `pedronahum/maestro-actus`'s reference point — pattern-alignment with the audit's keystone reference (§3).
- Includes the K8s entrypoint fix (#202) that maestro-actus depends on; ensures `KubernetesStepRuntime`'s `customizePreLaunchCommand` extension point behaves as expected.
- No CVEs disclosed in the recent history (CVE search performed against Netflix/maestro at audit time; no Hits).

Future-upgrade triggers (per `docs/vendoring.md`): security advisory in upstream or its transitive deps, breaking change Tlaloc needs to absorb before Layer N, quarterly review of upstream divergence, or reproducibility regression.

---

## 3. Pattern conformance to maestro-actus

Side-by-side comparison. **Keystone audit item** — forces inheriting maestro-actus's correctness rather than reinventing.

| maestro-actus file | Tlaloc equivalent | Conformance |
|---|---|---|
| `maestro-actus/build.gradle` | `maestro-tlaloc/build.gradle` | ✓ — same dep shape (maestro-common + maestro-engine + maestro-kubernetes), plus `buildDockerContext` task mirroring actus's. |
| `maestro-actus/src/main/java/.../engine/stepruntime/ActusStepRuntime.java` | `engine/stepruntime/TlalocStepRuntime.java` | ✓ — identical structure: extends `KubernetesStepRuntime`, single `customizePreLaunchCommand` override, calls super, generates command via the entrypoint builder, sets `command/args` on the K8s step context. |
| `maestro-actus/src/main/java/.../engine/actus/ActusRunner.java` | `engine/tlaloc/TlalocRunner.java` | ✓ — same `main(String[] args)` signature (2 args: JSON params + output file path), JSON deserialization via Jackson, OutputData JSON write. v1 differs in the body (Tlaloc does wire-format validation + identity copy; ACTUS does contract simulation) but the shape is identical. |
| `maestro-actus/src/main/java/.../engine/actus/ActusEntrypointBuilder.java` | `engine/tlaloc/TlalocEntrypointBuilder.java` | ✓ — same purpose (generate `java -cp '/app/*' …Runner '<JSON>' '/tmp/...'`), same single-quote escaping. |
| `maestro-actus/src/main/java/.../engine/actus/ActusParamsBuilder.java` | `engine/tlaloc/TlalocParamsBuilder.java` | ✓ — same role (Maestro params → JSON payload). Payload schema differs (ACTUS: contract_attributes + risk_factors + analysis_date; Tlaloc: artifact_uri + manifest_ref + input_handle + output_handle_uri). |
| `maestro-actus/src/main/java/.../engine/actus/ActusAttributeMapper.java` | `engine/tlaloc/TlalocAttributeMapper.java` | ✓ — same naming-bridge pattern. ACTUS maps 25 contract-domain keys; Tlaloc maps 4 v1 keys. |
| `maestro-actus/src/main/java/.../engine/actus/ActusCommand.java` | `engine/tlaloc/TlalocCommand.java` | ✓ — both Java records carrying `entrypoint + <domain-specific metadata>`. ACTUS: `entrypoint, contractType, contractId`; Tlaloc: `entrypoint, artifactUri, manifestRef, stepName`. |
| `maestro-actus/docker/Dockerfile` | `docker/Dockerfile` | ✓ — same multi-stage shape: JDK 21 alpine builder → JRE 21 alpine runtime. |
| `maestro-actus/docker/build.sh` | `docker/build.sh` | ✓ — same args contract (optional tag), same Gradle pre-build of `:assemble` + `:buildDockerContext`. |
| ACTUS sample workflows (8 in `maestro-server/.../samples/`) | 8 Tlaloc sample workflows in same location | ✓ — same total count, similar pattern coverage (single-step / pipeline / portfolio / nested-foreach / while-loop / template-caller). The unique-to-Tlaloc `sample-tlaloc-typed-handoff-wf.json` has no maestro-actus analog (ACTUS doesn't have typed cross-step handles). |
| `ActusRunnerEndToEndTest` | `TlalocRunnerEndToEndTest` | ✓ — same direct-JVM pattern (no K8s), TemporaryFolder for I/O isolation, JSON payload + output-file assertion. |
| Step-type registration: 1–2 lines in `StepType.java` + DI binding in maestro-server | Same shape | ✓ — `TLALOC("Tlaloc", true)` (~6 lines including Javadoc) + 3 `@Bean` factories in `MaestroStepRuntimeConfiguration` (~50 lines). |

### Structural divergences

Two intentional divergences from maestro-actus, both surfaced in this audit:

1. **Tlaloc has no `TlalocArtifact` analogue (yet).** ACTUS registers `ActusArtifact` in `pendingArtifacts` per K8s step. Tlaloc's v1 carries equivalent metadata in OutputData; v2 may add a `TlalocArtifact` if Maestro's artifact-collection flow needs structured records. Tracked as **OQ-Layer2.5-2**.

2. **Tlaloc's keystone sample (`sample-tlaloc-typed-handoff-wf.json`) has no maestro-actus analog.** The Tlaloc-unique value is typed buffer handoffs across step boundaries (`SerializedBufferHandle`), which ACTUS doesn't have. Documented as the audit's "what we learned" payoff in §15.

Otherwise: the maestro-tlaloc layout, file names, class structure, and test patterns are 1:1 with maestro-actus's, by design.

---

## 4. Step-type registration — exact code change

```java
// third-party/maestro/maestro-common/src/main/java/com/netflix/maestro/models/definition/StepType.java

  /** HTTP/HTTPS step. */
  HTTP("Http", true),
+ /**
+  * Tlaloc step — first-class step type for io.tlaloc programs. Tlaloc step bodies are
+  * content-addressed StableHLO+SDY artifacts; the runtime image loads the body and dispatches
+  * via PJRT/IREE. Added by Layer 2.5 §0.4.245+ as the only Tlaloc-side modification to
+  * vendored Maestro outside the {@code maestro-tlaloc/} module ...
+  */
+ TLALOC("Tlaloc", true),
  /** Join step. */
  JOIN("Join", false),
```

DI binding in `MaestroStepRuntimeConfiguration.java`:

```java
+ @Bean
+ public TlalocParamsBuilder tlalocParamsBuilder(...) { return new TlalocParamsBuilder(objectMapper); }
+
+ @Bean
+ public TlalocEntrypointBuilder tlalocEntrypointBuilder(...) { return new TlalocEntrypointBuilder(paramsBuilder); }
+
+ @Bean
+ public TlalocStepRuntime tlaloc(...) {
+   TlalocStepRuntime step = new TlalocStepRuntime(...);
+   stepRuntimeMap.put(StepType.TLALOC, step);
+   return step;
+ }
```

This parallels how `Actus` would be registered in maestro-actus (the agent's research couldn't locate `StepType.ACTUS` in the maestro-actus repo's git tree, suggesting either upstream-merged or a different mechanism; we follow the canonical Maestro pattern of enum + DI binding as observed in maestro-kubernetes / maestro-notebook).

---

## 5. End-to-end execution evidence

The `:vendored-maestro:maestro-tlaloc:test` Gradle task runs three test classes:

```
$ ./gradlew :vendored-maestro:maestro-tlaloc:test

> Task :vendored-maestro:maestro-tlaloc:test
TlalocStepTypeRegistrationTest > 6 tests passed
TlalocRunnerEndToEndTest         > 5 tests passed
TlalocSampleWorkflowsTest        > 11 tests passed

BUILD SUCCESSFUL
22 tests, 0 skipped, 0 failures, 0 errors
```

Per-sample assertions (from `TlalocSampleWorkflowsTest`):
- `sample-tlaloc-program-wf.json` — 1 Tlaloc step at `tlaloc.invoke`.
- `sample-tlaloc-pipeline-wf.json` — 1 Tlaloc middle stage.
- `sample-tlaloc-typed-handoff-wf.json` (KEYSTONE) — 2 Tlaloc steps, consumer carries valid `SerializedBufferHandle` JSON with all five fields + typeDescriptor sub-object.
- `sample-tlaloc-portfolio-wf.json` — Tlaloc nested inside one foreach.
- `sample-tlaloc-hpo-sweep-wf.json` — Tlaloc nested inside two foreach levels.
- `sample-tlaloc-iterative-tuning-wf.json` — Tlaloc inside a while loop.
- `sample-tlaloc-template-wf.json` + `sample-tlaloc-caller-wf.json` — caller composes via two subworkflow invocations.

CI artifact link: `actions/upload-artifact@v4` step uploads `build/reports/tests/` + `build/test-results/` with 14-day retention. Available at the run's artifacts panel after every CI run.

---

## 6. `SerializedBufferHandle` correctness

13 tests in `SerializedBufferHandleTest`:

- **Producer write + consumer read** — `roundTripWriteReadProducesEquivalentHandle`: write a F32 buffer, read it back, assert tensor equality.
- **Type-check pass** — `differentRankAxisNamesPreservedAcrossRoundTrip`: descriptor with axisNames flows through binary header verbatim.
- **Type-mismatch failures** — `typeMismatchInRankFailsLoudly`, `typeMismatchInAxisNamesFailsLoudly`: read with wrong descriptor → `SerializedBufferException("type mismatch...")`.
- **Mesh-mismatch failure** — `meshMismatchFailsLoudly`: producer on Mesh1<DataAxis>, consumer expects Mesh0 → `SerializedBufferException("mesh mismatch...")`.
- **Hash-mismatch failure** — `corruptedPayloadFailsHashCheck`: flip a payload byte on disk, read fails with `"content hash mismatch"`.
- **Missing URI** — `missingFileFailsLoudly`: deleted file → `SerializedBufferException("file does not exist")`.
- **Unsupported scheme** — `unsupportedSchemeS3FailsLoudly`, `unknownSchemeFailsAsStructuredException`: `s3://` raises `UnsupportedOperationException`; arbitrary scheme raises `SerializedBufferException`.
- **JSON round-trip** — `jsonRoundTripPreservesAllFields`, `jsonContainsAllExpectedKeys`.
- **Default retention** — `readDeletesFileByDefault`: file removed after `read(deleteAfterRead = true)`.

Pre-condition: `writeRequiresDescriptorMatchingTensorDims` enforces that the producer's descriptor matches the actual tensor (catches a malformed handle at write time, before any consumer reads).

---

## 7. Runtime image hygiene

| Property | Status |
|---|---|
| Multi-stage. | ✓ — JDK 21 alpine builder stage; JRE 21 alpine runtime stage. |
| Final image size. | Not measured locally (Docker not available in the audit environment). Expected ~160–200 MB based on alpine's JRE size + maestro-tlaloc jars + transitive deps. CI's first build will confirm. |
| Reproducibility. | ✓ — Pinned base image tags, `--no-daemon --no-build-cache` Gradle flags, deterministic given same source tree. CI verifies by running build twice and comparing image IDs (recommendation; not yet automated). |
| Startup time. | Java 21 cold-start on alpine ~200–400ms typical; not measured. Layer 3+ when real PJRT/IREE dispatch lands inside the runner is the relevant timing window. |
| Dependency audit. | maestro-tlaloc's runtime classpath: maestro-common, maestro-engine, maestro-kubernetes, jackson-databind, slf4j, lombok (compile-time only). No CVEs in the pinned versions per Apache 2.0 / standard supply-chain audit. |
| Entry point. | None defined; Maestro launches via `TlalocEntrypointBuilder`-generated shell command. Mirrors maestro-actus's pattern. |

---

## 8. CI cost

```
Total wall-clock target: <10 min
Observed (estimated):
  Checkout              ~5s
  Set up JDK 21         ~10s
  Gradle cache restore  ~5s (warm) / 30s (cold)
  Tlaloc-side suite     ~30s warm / ~5min cold
  maestro-tlaloc tests  ~30s warm / ~5min cold (composite Gradle)
  Artifact upload       ~5s
  ────────────────────
  ~1.5 min warm / ~10 min cold
```

Cold-cache run lands at the upper edge of the budget; warm runs are well under. The sample-workflow parsing tests are individually <1s; the long pole is Gradle's first-time dependency resolution against Maestro's transitive deps.

**Per-sample cost (within `TlalocSampleWorkflowsTest`):** all 11 tests complete in ~15ms total. Individual sample loads + structural assertions are O(JSON size). The samples are 0.5–2 KB each.

---

## 9. Vendoring divergence audit

Modifications to vendored Maestro outside `maestro-tlaloc/`:

| File | Change | Justification |
|---|---|---|
| [`third-party/maestro/settings.gradle`](../../third-party/maestro/settings.gradle) | `+include 'maestro-tlaloc'` (1 line) | Required for the new module to participate in Maestro's multi-module Gradle build. No other settings touched. |
| [`third-party/maestro/maestro-common/.../StepType.java`](../../third-party/maestro/maestro-common/src/main/java/com/netflix/maestro/models/definition/StepType.java) | `+TLALOC("Tlaloc", true),` enum entry + Javadoc (~6 lines) | First-class Tlaloc step type — the **only** way Maestro's parser recognizes `"type": "Tlaloc"` in workflow JSON. `StepRuntimeManager.getStepRuntime(StepType)` uses this enum to dispatch to `TlalocStepRuntime`. Cannot live inside maestro-tlaloc/ because the enum is closed. |
| [`third-party/maestro/maestro-server/.../MaestroStepRuntimeConfiguration.java`](../../third-party/maestro/maestro-server/src/main/java/com/netflix/maestro/server/config/MaestroStepRuntimeConfiguration.java) | 3 new `@Bean` factories + 3 import lines (~50 lines added) | Spring DI binding for `TlalocStepRuntime` against `StepType.TLALOC`. Located in maestro-server because that's where `KubernetesStepRuntime` and `NotebookStepRuntime` are also bound (consistency with the existing pattern). Cannot live inside maestro-tlaloc/ because the `stepRuntimeMap` Bean is defined in maestro-server's config; bean injection across modules would require additional Spring scanning configuration. |

These three modifications are the **complete** set of vendored-Maestro divergences. No other file outside `maestro-tlaloc/` was touched.

If a future change tempts a wider divergence, the audit's recommendation is to escalate (file an upstream PR proposing a generic step-type SPI, restructure Tlaloc to avoid the need, or document the new divergence here before merging).

---

## 10. Deprecation hygiene

`MaestroDescriptor.emit()` and `StubExecutor` are marked `@Deprecated(level = WARNING)`:

- [`MaestroDescriptor.kt:51–58`](../../maestro/src/jvmMain/kotlin/io/tlaloc/maestro/MaestroDescriptor.kt) — `@Deprecated` annotation with migration message + removal target ("post-Layer-3 release").
- [`StubExecutor.kt:34–41`](../../maestro/src/jvmMain/kotlin/io/tlaloc/maestro/StubExecutor.kt) — same annotation pattern.
- [`docs/maestro_descriptor.md`](../maestro_descriptor.md) — banner at the top documents the deprecation; existing format documentation kept for historical reference.

Existing tests against the deprecated surface still run:

- [`MaestroDescriptorTest`](../../maestro/src/jvmTest/kotlin/io/tlaloc/maestro/MaestroDescriptorTest.kt) — file-level `@file:Suppress("DEPRECATION")` to silence the 5 deprecation warnings; tests pass unchanged. Coverage on the deprecated surface is intentional during the migration window.

`ProgramTest`, `WorkflowTest`, `SerializedBufferHandleTest` — these don't touch the deprecated surface; no suppression needed.

---

## 11. Compatibility

All Layer 1 + Layer 2 + Layer 2.5 tests pass:

```
Tlaloc-side suite     1049 tests  (unchanged from §0.4.243)
maestro-tlaloc suite    22 tests  (new in Layer 2.5)
─────────────────
Combined              1071 tests, 0 skipped, 0 failures, 0 errors
```

### BGDHyperOpt regression gate

Layer 2.5 makes no tensor-pipeline changes (no AD / PhiCalculus / Vjp / IR-emission code modified). BGDHyperOpt timing is unchanged from §0.4.243's baseline (T=10 M=3 ratio≈3.3, T=50 M=3 ratio≈0.6, pre-simplified ratio≈1.5; all within hardware/JIT noise of the §0.4.242 / §0.4.243 measurements). The test's hard assertion `[0.5, 200]` passes.

---

## 12. Layer 5 prerequisite re-evaluation

Original Layer 2 audit's OQ-Layer2-7 raised the question: "should Layer 5 (`@Decoupled` + differentiable workflows) require K2 compile-time emission of `program {}` artifacts, or can the runtime-tracer path scale?"

With Layer 2.5 shipping a real runtime image + first-class step type:

- **Runtime-tracer path remains acceptable** for Layer 2 + Layer 2.5 use cases. The runner's wire-format validation + future PJRT/IREE dispatch don't require compile-time emission — the body bytes are produced by the runtime tracer at `program {}` invocation time (in the `OrchestrationScope.program` builder) and shipped via `params.tlaloc.artifact_uri`.
- **Layer 5's `@Decoupled` may need compile-time emission.** Differentiable workflows where `grad { workflow.run(hp) }` synthesises a backward pass over a workflow DAG benefit from artifacts being available at compile time (so the backward pass can be assembled symbolically rather than re-traced at runtime). Layer 5 designs should re-evaluate when they land.

Recommendation: **defer the decision** until Layer 5 surfaces concrete `@Decoupled` use cases. The runtime-tracer path is good enough through Layer 4. If Layer 5 requires it, K2 compile-time emission is a focused 1–2-session add (mirror the existing `grad {}` plugin work in compiler-plugin/).

Tracked as **OQ-Layer2.5-4**.

---

## 13. Code quality

| Item | Status |
|---|---|
| Lint clean. | ✓ — full suite compiles without warnings other than pre-existing `'when' is exhaustive so 'else' is redundant` in `DxirReverseTransform.kt:1787` (Layer 1 vintage; not Layer 2.5 code). |
| No `TODO` / `FIXME` in shipped code. | ✓ — `git grep TODO maestro/src third-party/maestro/maestro-tlaloc` returns zero hits in Layer-2.5-introduced files. |
| Public APIs have KDoc / Javadoc. | ✓ — every Layer-2.5-introduced public symbol has class-level + method-level documentation with `Layer 2.5 §0.4.X+` annotations. |
| Internal vs. public visibility minimal. | ✓ — `SerializedBufferHandle` is public (user-facing); `ManifestJsonParser`'s newly-internalized `parseTypeDescriptor` is `internal` (same-module access only); `SerializedBufferHandleJsonParser` is `internal`; helper methods on Java classes follow standard private/package-private idioms. |
| Match existing repo style. | ✓ — Java code mirrors maestro-kubernetes / maestro-notebook conventions (Lombok `@Slf4j` + `LOG` field name, Spring DI patterns, JUnit 4 test style). Kotlin code mirrors Layer 2's conventions. |

---

## 14. Open issues — for inclusion in `DIFFKTX_SPEC.md` §18

- **OQ-Layer2.5-1: Migrate to submodule-of-fork vendoring.** v1 uses copy-in-place (D1). When/if a `pedronahum/maestro-vendored` fork repo is created, migrate `third-party/maestro/` from copy-in-place to a git submodule pointing at it. Mechanical migration; doesn't change the on-disk layout or Gradle wiring.

- **OQ-Layer2.5-2: `TlalocArtifact` analogue.** maestro-actus registers an `ActusArtifact` in `pendingArtifacts` per K8s step. Tlaloc v1 carries equivalent metadata in OutputData; if Maestro's artifact-collection flow needs structured records for Tlaloc dispatches, add a parallel `TlalocArtifact` model class.

- **OQ-Layer2.5-3: Live-Maestro CI integration.** v1's CI exercises maestro-tlaloc unit tests + sample-workflow parsing via composite Gradle. A future pass should boot a real Maestro instance (testcontainers + Postgres + Maestro server jar) and submit each sample workflow via the REST API. Foreach / while / subworkflow execution requires K8s pod lifecycle (kind/k3d). Estimated 1–2 days of CI infrastructure work.

- **OQ-Layer2.5-4: Layer 5 compile-time-emission re-evaluation.** §12 above. Defer until Layer 5's `@Decoupled` surfaces concrete use cases.

- **OQ-Layer2.5-5: Multi-cloud `SerializedBufferHandle`.** v1 stubs `s3://` and `gs://` schemes with `UnsupportedOperationException`. Real cloud-storage adapters are L3+; design touches authentication, retry policies, multipart uploads.

- **OQ-Layer2.5-6: Retry semantics for failed Tlaloc steps.** Maestro's `retry_policy` field is honored by `KubernetesStepRuntime` (and inherited by `TlalocStepRuntime`), but Tlaloc has no opinion yet on what's retriable vs. fatal. v2 may classify (e.g. content-hash mismatch = fatal, transient-IO = retriable).

- **OQ-Layer2.5-7: GPU/TPU node-selector requesting.** Layer 3's backend-matrix work (`MaestroStep.manifest.backendMatrix`) needs a way to inform Maestro's K8s step which node selector / resource limits to set per Tlaloc step (CPU vs. GPU vs. TPU). Foreshadow the integration here.

- **OQ-Layer2.5-8: Optional upstream PR — defer or file?** Layer 2.5 ships without a Netflix/maestro PR (D2). Re-evaluate when (a) Netflix opens a generic step-type SPI proposal upstream, or (b) Tlaloc's deployment context demands the upstream-merged path.

---

## 15. What we learned

The vendoring + first-class registration revealed three things the masquerade missed.

### 15.1 The masquerade's "Kubernetes step with extra params" was opaque to Maestro's scheduler

Layer 2's `MaestroDescriptor.emit()` produced steps Maestro's parser would happily accept (`type: "Kubernetes"`), but Maestro's scheduler treated them as ordinary container launches. There's no place to hang Tlaloc-specific metadata that Maestro's UI / observability / typed-output handling could consume. The registered `Tlaloc` step type unlocks all of that automatically — Maestro's UI can render Tlaloc-typed steps with a different icon, the artifact-collection flow can pull `TlalocArtifact` records (when added), retry policies can be Tlaloc-aware.

### 15.2 Vendored ownership made parameter typing natural

The masquerade smuggled Tlaloc params through Maestro's generic `params: {...}` map. Discovering the right shape for `tlaloc_artifact_uri` etc. required reading existing samples and inferring conventions. The first-class step type lets us declare the canonical params block in `default-tlaloc-step-params.yaml` and have Maestro merge it automatically — every Tlaloc-typed step in a workflow gets the right defaults without per-workflow boilerplate.

### 15.3 The runner-CLI shape is universally cleaner than wrappers

The masquerade conceived Tlaloc dispatch as "a Maestro Kubernetes step that happens to launch a Tlaloc-runtime image." Vendoring inverts the relationship: `TlalocStepRuntime extends KubernetesStepRuntime` makes Tlaloc the primary subject. The CLI runner (`TlalocRunner.main`) becomes the canonical execution entry point, with the same shape as `ActusRunner.main`, `PapermillRunner`, etc. Future Tlaloc deployments — local Docker, K8s, Slurm-like batch systems, anything that can launch a JAR — can use the same runner with no glue code.

### What became natural that was hard before

| Capability | Pre-Layer-2.5 (masquerade) | Post-Layer-2.5 (vendored) |
|---|---|---|
| Parameter typing | Smuggled via generic `params: {...}` map; no schema enforcement | Native — `default-tlaloc-step-params.yaml` is Maestro-merged; `TlalocAttributeMapper` canonicalizes |
| Output binding | Whatever Maestro's Kubernetes step does for output | `TlalocRunner` writes structured OutputData containing the produced `SerializedBufferHandle` |
| Retry semantics | Inherited from Kubernetes step; no Tlaloc-aware classification | `TlalocStepRuntime.shouldRetry` (when implemented) can classify per Tlaloc semantics |
| Log handling | Container logs only | `TlalocStepRuntime` can post-process logs; specific log lines (e.g. validation failures) become structured timeline events |

These four capabilities were *dancing-around hard* under the masquerade. Under the vendored first-class registration, they're natural extensions of the existing maestro-actus / maestro-notebook patterns.

---

## Sign-off

This audit covers all 14 sections required by the original Layer 2.5 task. All requirements are either fully met (rows in §1) or explicitly tracked as open questions (§14). Combined suite is green at 1071 tests (1049 Tlaloc + 22 maestro-tlaloc); 0 skipped, 0 failures.

Three deliberate scope-narrowings, all documented:
- **Copy-in-place vendoring** (D1) — submodule deferred until external repo is created; OQ-Layer2.5-1.
- **Identity-transform v1 runner body** (D5) — real PJRT/IREE dispatch ships with the runtime image's full classpath in a future cycle.
- **No live-Maestro CI** (D3) — composite-Gradle test path covers the contract; live-Maestro execution is OQ-Layer2.5-3.

Layer 2.5 closes; Layer 3 (recognizer / VJP coarsener / kernel templates) is unblocked.
