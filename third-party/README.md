# third-party/

Vendored dependencies that Tlaloc owns the integration of.

## Philosophy

Some dependencies are not "library use" but **build-and-distribute components**
that Tlaloc treats as part of the project. These get vendored here, copy-in-
place, with full source. Their build systems become part of Tlaloc's build.

The pattern is:

- **Tlaloc is the project of record.** We control the integration end-to-end.
- **Vendored components are dependencies we own the upgrade rhythm of.** We
  pin specific upstream commits, document why, and re-evaluate periodically.
- **Modifications to vendored code are minimised** — additive new modules
  preferred; touches to existing files allowed only at registration points,
  each justified in writing.
- **Vendored code is not a fork to maintain.** No long-lived feature branch.
  Upgrades are atomic: bump the pin, re-apply the registration touches,
  re-verify. The audit's vendoring-divergence section enumerates every
  modification.

## What's vendored today

| Path | Upstream | Pinned | Why |
|------|----------|--------|-----|
| [`maestro/`](maestro/) | [Netflix/maestro](https://github.com/Netflix/maestro) | `0150f2a78005cf135023de29e5f8e06fd592563d` (2026-04-09) | Layer 2.5: first-class `Tlaloc` step type via vendored Maestro. Replaces Layer 2's "Kubernetes-step masquerade." See [`docs/vendoring.md`](../docs/vendoring.md) for the upgrade procedure. |

## Why copy-in-place rather than git submodule

A git submodule pointing at upstream Netflix/maestro can't carry our additions
(the new `maestro-tlaloc/` module + the step-type registration touches). A
submodule pointing at our own GitHub fork would work, but creating that fork
is external infrastructure we don't yet need. Copy-in-place vendoring achieves
the same outcome — Tlaloc owns the integration source code — without external
repo creation. The trade-off is a one-time +100k line addition to Tlaloc's git
history. Documented in the Layer 2.5 audit.

If we later decide to switch to submodule-of-fork, the migration is mechanical:
push the vendored tree to a new GitHub repo, replace `third-party/maestro/`
with a submodule pointing at it. The on-disk layout and Gradle wiring don't
change.

## Upgrade procedure

See [`../docs/vendoring.md`](../docs/vendoring.md) for the full procedure.
Summary:

```bash
# 1. Save current vendored tree's modifications outside the maestro-tlaloc/ module:
#    (the audit's "Vendoring divergence audit" enumerates these)
git diff -- third-party/maestro -- ':!third-party/maestro/maestro-tlaloc' > /tmp/maestro-overlays.patch

# 2. Replace with fresh upstream:
rm -rf third-party/maestro
git clone --depth 1 https://github.com/Netflix/maestro.git third-party/maestro
cd third-party/maestro && git fetch --depth 50 origin <new-pin> && git checkout <new-pin>
rm -rf .git
cd ../..

# 3. Re-apply our overlays:
git apply /tmp/maestro-overlays.patch

# 4. Verify:
./gradlew :maestro-tlaloc:test
./gradlew test  # full Tlaloc suite

# 5. Update the pin reference in this README + docs/vendoring.md.
```

## Build integration

Tlaloc's root `settings.gradle.kts` includes the vendored Maestro build via
`includeBuild("third-party/maestro")` (composite Gradle build). A single
`./gradlew build` at the Tlaloc root drives both Maestro's modules and the
new `maestro-tlaloc/` module. Cross-module dependencies (e.g.
`:maestro-tlaloc` consuming Tlaloc's `:maestro` jar) work natively without
staging through a local Maven repo.

**JDK requirement:** Maestro pins `JavaLanguageVersion.of(21)` in its root
`build.gradle`; Tlaloc's modules stay on `JvmTarget.JVM_17`. Both JDKs must
be installed (see `scripts/install-jdk21.sh`); Gradle's toolchain mechanism
routes per-module.

## Divergence policy

Modifications to vendored Maestro outside `maestro-tlaloc/` are restricted to:

1. **Step-type registration** — the minimum touches to register `Tlaloc` as
   a first-class step type (typically 1–2 lines in `StepType.java` + a DI
   binding). The audit's "Vendoring divergence audit" enumerates each.
2. Nothing else. If a sample workflow or runtime feature needs other
   modifications, they live inside `maestro-tlaloc/` or are surfaced as a
   feature request to upstream.

If a future change tempts a wider divergence, we revisit: either upstream
the change (file PR), restructure Tlaloc to avoid needing it, or escalate
to a documented design decision in the audit.
