# F64 under `grad {}`: progress log

Plan: [f64-grad-plan.md](f64-grad-plan.md). Branch `feat/f64-grad` from `main` (`6934819`).

Baseline on `main`: 2,881 tests, 0 failures, 101 skipped
(`./gradlew cleanJvmTest cleanTest test --continue --no-build-cache`).

## F32 unchanged: how it is checked

A temporary patch (not committed; kept at
`$SCRATCH/f64check.patch` during the run and described here) wraps
`DxirToIrSynthesis.synthesise` so that, when `TLALOC_F64CHECK_DIR` is set, every call writes
one file named by the SHA-256 of its content: the DXIR handed to synthesis (ids renumbered),
its `toKotlinSource` rendering (or the refusal), and `dump()` of the synthesized Kotlin IR.
Running `:compiler-plugin:test` with it on `main` gave 343 distinct files (337 with
synthesized IR). The same run on the branch must produce the same 343 files for F32 (new F64
tests add files; none may disappear or change).

## Log

### Step 1 — scalar `Double` (done)

Scalar `Double` lambdas already lowered at F64 end to end (literals keep their kind,
synthesis emits `kotlin.Double` arithmetic, the alpha01 fold fixes hold). Added
`F64ScalarGradientTest`: `grad`, `grad2`, `valueAndGrad`, `valueAndGrad2`, `jvp`,
`valueAndJvp`, `vjp`, exp/log/sin/cos/tanh/sqrt, `customVjp`/`customJvp`, all within 1e-13
of analytic Double derivatives, and a hidden-round-trip test (input `1 + 1e-10`, literal
`0.1`, captured `0.3` and `1e-9`, each of which F32 rounding moves by more than the
tolerance). `F64TestHarness` is shared by the F64 plugin tests. `jvp` takes no captured
runtime value, for any dtype (existing limitation); the test uses a `const val` there.

Suite: 2,884, 0 failures.

## Decisions

- **Host path = the `grad {}` interpreter.** Synthesized `grad {}` code calls
  `io.tlaloc.core.ops` host functions; F64 support means F64 twins of those functions.
  They live in separate files (`HostOpsF64.kt`, `BroadcastOpsF64.kt`) because each erases to
  its F32 twin's JVM signature and `@JvmName` is not available in `commonMain`, the same
  reason `LinalgF64.kt` exists.
- **PJRT leg.** The plugin's derivative DXIR cannot be recovered at run time (the canonical
  serializer refuses ops with attrs), so the PJRT checks run the `DxirReverseTransform`
  graph of the same program through the general `PjrtSession` entry, not `runOnF64`.

## Next step

Step 2: restore the stashed F64 host twins (`git stash pop`, stash `core-f64`), add
dtype-aware host-function lookup to synthesis, lift the F32 gates.
