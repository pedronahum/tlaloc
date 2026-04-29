# Vendoring procedure

How Tlaloc keeps `third-party/maestro/` (and any future vendored component) up
to date.

## Why vendor

Layer 2.5's introduction (§0.4.244+) makes vendored Maestro the canonical
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

## Divergence policy (recap from third-party/README.md)

Modifications to vendored Maestro outside `maestro-tlaloc/` are restricted to
the minimum needed for `Tlaloc` step-type registration. Each modification is:

1. Documented in the audit's "Vendoring divergence audit" section with
   one-paragraph justification.
2. Listed in the patch capture command above (the `:!third-party/maestro/maestro-tlaloc`
   exclusion in step 1).
3. Re-applied during every upgrade.

If an upgrade tempts a wider divergence, escalate — either upstream the
change (file PR), restructure Tlaloc to avoid needing it, or document the
new divergence in the audit before merging.

## On the optional upstream PR

Layer 2.5's audit tracks the question of whether to upstream a generic
"custom step type" SPI to Netflix/maestro. As of §0.4.244, deferred — see
`docs/maestro_upstream_pr.md` for the rationale.

If we later file the PR and Netflix accepts a generic-extension SPI, the
upgrade procedure changes: the registration touches become `service-loader`
or DI-binding declarations instead of edits to `StepType.java`. The
`docs/vendoring.md` "Divergence policy" tightens correspondingly.
