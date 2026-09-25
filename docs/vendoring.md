# Vendoring procedure

How Tlaloc keeps `third-party/maestro/` (and any future vendored component) up
to date.

## Why vendor

Vendored Maestro is the canonical
substrate for Tlaloc-on-Maestro. The masquerade approach from Layer 2 — emit a
Maestro `Kubernetes`-typed step with `tlaloc-runtime:*` image — is deprecated
in favor of a first-class `Tlaloc` step type registered inside vendored
Maestro itself. The shape mirrors `pedronahum/maestro-actus`'s pattern.

Vendoring is the right tool when:

- The dependency is **part of the integration story**, not a library Tlaloc
  consumes from a binary registry. Tlaloc's identity now includes "the
  Maestro step type + runtime image our workflows ship as."
- We want **atomic upgrade control**: a single commit moves the vendored pin
  + re-applies our overlays + re-runs verification.
- The dependency's upstream is **slow to merge community contributions** or
  doesn't yet expose a public extension SPI. Netflix/maestro's step-type
  enum + DI is internal; vendoring is the only way to add `Tlaloc` cleanly.

## Current pin

| Component | Upstream commit | Date | Why this pin |
|-----------|-----------------|------|--------------|
| Netflix/maestro | `0150f2a78005cf135023de29e5f8e06fd592563d` | 2026-04-09 | Matches `pedronahum/maestro-actus`'s reference point. Captures the K8s entrypoint fix (#202) that maestro-actus depends on. Apache 2.0; actively maintained; latest pre-Layer-2.5 commit. |

## When to upgrade

Trigger any of:

- **Security advisory** in upstream Netflix/maestro or one of its transitive
  dependencies (Spring, Jackson, Hikari, etc.).
- **Breaking change Tlaloc needs to absorb** before Layer N — e.g. if Layer 3
  needs an upstream feature to land, the upgrade lands first.
- **Quarterly review** — every 3 months, check upstream for divergence; if
  the gap exceeds ~50 commits or 90 days, upgrade defensively.
- **Reproducibility regression** — if `tlaloc-runtime:*` builds become
  non-reproducible because of a transitively-pulled drift, upgrade to absorb
  the fix.

Don't upgrade when:

- A new upstream commit changes Maestro's step-type registration mechanism
  in a way that breaks our overlay. The audit's "Vendoring divergence audit"
  enumerates the registration touches; if a planned upgrade would invalidate
  them, escalate to a redesign session, not a routine pin bump.

## Upgrade procedure

```bash
# 0. Pre-flight: confirm the current Tlaloc + vendored Maestro is green.
./gradlew test

# 1. Capture our overlays (modifications outside maestro-tlaloc/) as a patch.
#    The audit's "Vendoring divergence audit" lists the touched files;
#    keep that section in sync as the source of truth.
git diff HEAD -- 'third-party/maestro' ':!third-party/maestro/maestro-tlaloc' \
    > /tmp/maestro-overlays.patch

# 2. Save maestro-tlaloc/ separately — it's our additive module, not an
#    upstream overlay.
mv third-party/maestro/maestro-tlaloc /tmp/maestro-tlaloc-stash

# 3. Replace with fresh upstream at the new pin.
NEW_PIN="<the-new-commit-sha>"
rm -rf third-party/maestro
git clone --depth 50 https://github.com/Netflix/maestro.git third-party/maestro
cd third-party/maestro
git fetch --depth 50 origin "$NEW_PIN"
git checkout "$NEW_PIN"
rm -rf .git
cd ../..

# 4. Re-apply our overlays + restore maestro-tlaloc/.
mv /tmp/maestro-tlaloc-stash third-party/maestro/maestro-tlaloc
git apply /tmp/maestro-overlays.patch || {
    echo "patch did not apply cleanly; inspect /tmp/maestro-overlays.patch"
    echo "the upstream commit likely touched the same files we overlay."
    echo "re-create the overlays by hand against the new upstream tree."
    exit 1
}

# 5. Verify build + tests.
./gradlew :maestro-tlaloc:test
./gradlew :maestro-tlaloc:assemble
./gradlew test  # full Tlaloc suite, including all sample workflow tests

# 6. Update the pin reference in:
#    - third-party/README.md (the table)
#    - docs/vendoring.md (this file's "Current pin" section)
#    - docs/audits/maestro_first_class_audit.md if a new audit cycle triggers

# 7. Commit:
git add third-party/maestro third-party/README.md docs/vendoring.md
git commit -m "vendoring: bump Netflix/maestro pin to <new-pin>"
```

## Changes to the upstream tree

Every file under `third-party/maestro/` that differs from Netflix/maestro at the
pinned commit (paths relative to `third-party/maestro/`; `VendoringNoticeTest` in
`maestro-tlaloc` checks this table against the tree):

| File | Change |
|------|--------|
| `build.gradle` | JDK 25 toolchain; the Security Manager test flag removed; JUnit Platform launcher and vintage engine on the test classpath; Spotless 8.10.2 with google-java-format 1.30.0 (the lowest that runs on JDK 25), with the three upstream files whose comment indentation 1.30.0 reformats excluded; Checkstyle pinned to 9.3, the version upstream's Gradle 8.8 used |
| `settings.gradle` | Includes `maestro-tlaloc` |
| `maestro-common/src/main/java/com/netflix/maestro/models/definition/StepType.java` | Adds the `TLALOC` step type |
| `maestro-common/src/main/java/com/netflix/maestro/models/stepruntime/KubernetesCommand.java` | Adds the `nodeSelector` and `accelerators` fields |
| `maestro-server/build.gradle` | Depends on `maestro-tlaloc` |
| `maestro-server/src/main/java/com/netflix/maestro/server/config/MaestroStepRuntimeConfiguration.java` | Registers the Tlaloc step runtime beans |

Each of these carries a `Modified by Pedro N. Rodriguez for Tlaloc, 2026` line
at the top, as Apache-2.0 section 4(b) requires. Files Tlaloc added are
`maestro-tlaloc/`, `maestro-common/.../KubernetesCommandTlalocFieldsTest.java`,
`maestro-server/.../MaestroStepRuntimeConfigurationTlalocTest.java` and
`maestro-server/src/test/resources/samples/sample-tlaloc-*.json`; the Java
ones carry a `Copyright 2026 Pedro N. Rodriguez` Apache header. The repository
root `NOTICE` credits Netflix Maestro; Netflix/maestro ships no `NOTICE` file at
the pinned commit. After an upgrade, re-check that this
table, the `Modified by` lines and `NOTICE` still match the tree, and carry
over any `NOTICE` file the new upstream commit ships.

## Divergence policy (recap from third-party/README.md)

Modifications to vendored Maestro outside `maestro-tlaloc/` are restricted to
the minimum needed for `Tlaloc` step-type registration. Each modification is:

1. Documented in the audit's "Vendoring divergence audit" section with
   one-paragraph justification.
2. Listed in the patch capture command above (the `:!third-party/maestro/maestro-tlaloc`
   exclusion in step 1).
3. Re-applied during every upgrade.
4. Marked with a `Modified by Pedro N. Rodriguez for Tlaloc` line at the top of
   the file and listed in [Changes to the upstream tree](#changes-to-the-upstream-tree).

If an upgrade tempts a wider divergence, escalate — either upstream the
change (file PR), restructure Tlaloc to avoid needing it, or document the
new divergence in the audit before merging.

## On the optional upstream PR

Layer 2.5's audit tracks the question of whether to upstream a generic
"custom step type" SPI to Netflix/maestro. It is deferred; see
`docs/maestro_upstream_pr.md` for the rationale.

If we later file the PR and Netflix accepts a generic-extension SPI, the
upgrade procedure changes: the registration touches become `service-loader`
or DI-binding declarations instead of edits to `StepType.java`. The
`docs/vendoring.md` "Divergence policy" tightens correspondingly.

## Triton backend: `triton/third_party/`

The Triton backend in [`triton/`](../triton/README.md) compiles against a
small set of upstream files, copied unmodified. Only what the backend
includes or compiles is there, each directory keeps its upstream path layout,
and each carries the upstream license file.

| Directory | Upstream | Pin | License | Files |
|---|---|---|---|---|
| `core/` | triton-inference-server/core | branch `r25.11`, commit `0e256c102549aeffac4c795ae3ec531fb970ac61` | BSD-3-Clause (`LICENSE`) | `include/triton/core/tritonbackend.h`, `include/triton/core/tritonserver.h` |
| `common/` | triton-inference-server/common | branch `r25.11`, commit `c09e98a47d721fea51913eb3737d37a45d34b431` | BSD-3-Clause (`LICENSE`) | `include/triton/common/error.h`, `include/triton/common/triton_json.h` |
| `backend/` | triton-inference-server/backend | branch `r25.11`, commit `3901aa5e51a86ada253ae9ce44f82bcb81fe08b2` | BSD-3-Clause (`LICENSE`) | `include/triton/backend/{backend_common,backend_model,backend_model_instance}.h`, `src/{backend_common,backend_model,backend_model_instance}.cc` |
| `rapidjson/` | Tencent/rapidjson | `master` at commit `24b5e7a8b27f42fa16b96fc70aade9106cf7102f` | MIT (`license.txt`); `include/rapidjson/msinttypes/` is BSD-3-Clause, as `license.txt` states | `include/rapidjson/**` (the JSON-licensed `bin/jsonchecker` is not copied) |
| `xla/` | openxla/xla | commit `b6f37ab7767f428fd6f993de5e211643d47d4deb` | Apache-2.0 (`LICENSE`) | `xla/pjrt/c/pjrt_c_api.h` |

Why these pins:

- The three Triton repositories have no release tags. `r25.11` is the branch
  the `nvcr.io/nvidia/tritonserver:25.11-py3` container was built from
  (Triton server v2.63.0, server commit
  `f30b53f554bfcf5d4792b744db9292eec695e5f9`). The two core headers are
  byte-identical to `/opt/tritonserver/include/triton/core/` in that
  container, and the backend reports "Triton backend API 1.19, compiled
  against 1.19" when it loads.
- rapidjson is needed because `common/triton_json.h` includes it and the
  container does not ship it. There has been no rapidjson release since 1.1.0
  (2016); the pin is the `master` head at the time of vendoring.
- The XLA commit is the one jax v0.10.0 builds against
  (`third_party/xla/revision.bzl`). Its `pjrt_c_api.h` is PJRT C API 0.104,
  the version of the plugin `triton/fetch_pjrt_plugin.sh` downloads
  (jax-cuda13-pjrt 0.10.0). The header includes only C standard headers.

Upgrading: download the new pins' tarballs from GitHub, copy the same files
over, update this table and `NOTICE`, rebuild with `triton/build_backend.sh`
and run `triton/verify.sh`. When the container tag moves, the Triton pins
move with it (branch `rYY.MM`); when the PJRT plugin moves, the XLA pin moves
to the commit that plugin was built from.
