# Optional upstream Netflix/maestro PR

**Status:** Layer 2.5.5 §0.4.249+ — **deferred** with documented rationale.

## Decision

Layer 2.5 ships without filing a contribution PR to Netflix/maestro. The
vendored `third-party/maestro/` tree carries Tlaloc-specific changes
(`maestro-tlaloc/` module + `TLALOC` enum entry + Spring DI binding) that
Tlaloc owns the upgrade rhythm of via `docs/vendoring.md`'s upgrade
procedure.

## Why deferred

Three reasons converge:

1. **A "Tlaloc step type" PR is unlikely to be accepted upstream.** Netflix's
   Maestro is a general-purpose orchestrator; baking in a Tlaloc-specific
   step type would couple Maestro to a specific differentiable-programming
   framework. The merge-worthy upstream contribution is a *generic
   step-type SPI* — a `StepRuntimeProvider` / `ServiceLoader`-style
   extension mechanism that lets any project register custom step types
   without modifying the maestro-common enum. That's a substantial design
   effort (multiple weeks); it's not what Tlaloc needs to ship Layer 2.5.

2. **Netflix's contribution velocity is slow** for community contributions.
   The repo's recent `git log` shows mostly internal commits; community
   PRs that have merged are minor. A larger SPI proposal would need
   stakeholder alignment from Netflix's Maestro team. The ROI for Tlaloc
   doesn't justify the coordination overhead at this stage.

3. **Vendoring gives us full control regardless.** With
   `third-party/maestro/` under our git tree, Tlaloc can ship, deploy, and
   evolve independently of Netflix's release cadence. An eventual upstream
   merge would simplify the vendoring divergence (smaller patches to
   re-apply on upgrade) but doesn't unblock anything Tlaloc needs *today*.

## Re-evaluation triggers

If any of the following surface, we revisit:

- **Netflix opens a generic step-type SPI proposal upstream** (RFC, draft
  PR, design doc). Tlaloc files a confirming PR + adopts the SPI.
- **Tlaloc deployment context requires the upstream-merged path** —
  e.g. an enterprise environment where vendored forks aren't
  policy-permitted. Then Tlaloc invests in proposing the SPI.
- **Quarterly review** of upstream divergence detects that the maestro
  pin has drifted significantly from main and our modifications no
  longer apply cleanly. A focused "rebase + propose" cycle becomes
  cost-effective then.

## What an upstream PR would look like

For reference, if/when filed, the contribution would be a pair of PRs:

1. **Generic step-type SPI** — a new `StepRuntimeProvider` interface in
   `maestro-engine` with a `ServiceLoader` lookup. `KubernetesStepRuntime`
   and friends migrate to register via the SPI rather than via
   hard-coded `MaestroStepRuntimeConfiguration` beans. `StepType` becomes
   open-ended (string-keyed) rather than a closed enum.

2. **Tlaloc adopter** — a separate `tlaloc-maestro-adapter` jar (published
   to Maven Central by Tlaloc, not Netflix) implementing `StepRuntimeProvider`
   for `Tlaloc` steps. Users add the jar to their Maestro deployment's
   classpath; Maestro picks up the new step type automatically.

This would reduce Tlaloc's vendoring divergence to *zero* — vendored Maestro
becomes a clean upstream pin, and our `maestro-tlaloc/` lives outside the
vendored tree as a separate Tlaloc-published artifact.

Tracked as **OQ-Layer2.5-8** in `DIFFKTX_SPEC.md` §18.
