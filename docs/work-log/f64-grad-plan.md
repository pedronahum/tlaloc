# F64 under `grad {}`: plan

Branch `feat/f64-grad` from `main` (`6934819`). Progress: [f64-grad-progress.md](f64-grad-progress.md).

## Goal

F64 is a first-class dtype under every transformation (`grad`, `grad2`, `grad3`,
`valueAndGrad*`, `jvp`, `vjp`, `jacobian`, `jacobianReverse`, `hessian`, `customVjp`,
`customJvp`), on the host and through PJRT, with no implicit promotion and no F32 round
trip anywhere in an F64 body.

## Baseline

`./gradlew cleanJvmTest cleanTest test --continue --no-build-cache` on `main` (`6934819`),
2026-09-29 11:10: **2,881 tests, 0 failures, 101 skipped**, 4 min 52 s. (`--rerun` binds only
to the last task named and leaves the KMP `jvmTest` tasks up to date; `cleanJvmTest cleanTest`
without the build cache is what re-executes every test.)

## How a `grad {}` body runs today

1. FIR (`FirLambdaToDxirLowering`) lowers the lambda to DXIR. F64 tensor params already lower
   (`DTENSOR_DTYPE_MAP`), literals keep their kind (`literalDType`).
2. `DxirReverseTransform` / `DxirForwardTransform` build the derivative. Constant builders
   already take the operand's dtype (`floatLiteralForDtype`, `seedValueFor`, `zeroLike`);
   the alpha01 fold fixes (`74f4fb2`, `7b74ce5`) made the scalar fold width-correct.
3. `DxirToIrSynthesis` turns the derivative back into Kotlin IR: scalar ops become
   `kotlin.Float`/`Double` arithmetic, tensor ops become calls to `io.tlaloc.core.ops` host
   functions. **This is the host execution path of `grad {}`** (the reference "interpreter"
   for plugin code). `DxirInterpreter` is the engine of the Tracer-capture API and of the
   IR-level tests.
4. `PjrtSession.runOn` executes a DXIR function through StableHLO (F32/I32 only);
   `runOnF64` is an all-F64 side lane.

## Where the F32-only restriction lives

Plugin, synthesis (`DxirToIrSynthesis.kt`):
- `isAcceptedTensorType` (`dtype == F32`), the param/body gates in `synthesise`, `irTypeFor`.
- Tensor constants are splatted as `Float` through `broadcastLike` (`irConstFor`).
- Per-op `dtype != F32` arms: softmax, slice, reduce, gather/scatter, linalg (explicitly
  picks the F32 overload), MATMUL, CAST (Bool→F32 only), `toFloat` bridge, DET.
- 45 host-function lookups by name with `singleOrNull()`/`firstOrNull()`: adding F64 twins
  with the same names would make them ambiguous or pick the wrong twin.

Plugin, FIR (`FirLambdaToDxirLowering.kt`):
- `ops.toFloat` bridge refuses non-F32; no `toDouble` bridge.
- `floatLiteralArg` narrows every numeric literal to `Float` (clip bounds, scalar × tensor
  splat): a hidden F32 round trip for F64 bodies.
- `lowerScalarMixedBinary` skips the splat for non-F32 tensors.
- Comparison scalar splat only for F32.
- Refused by name as F32-only: conv2d/convTranspose2d, pools, batchNorm, embedding, sparse
  matmul; RNG draws are typed F32; outerProduct's RESHAPE/MATMUL are typed F32; linalg
  requires "rank-2 F32 receiver under grad {}".

Core (`:core`):
- `HostOps.kt`/`BroadcastOps.kt` have no F64 overloads at all; only `LinalgF64.kt` does.
  No `Tensors.f64*` factories, no `hostF64()`.

IR:
- `DxirInterpreter` stores every value as `FloatArray`: F64 nodes compute at F32 precision
  silently (documented at the top of the file).
- `PhiCalculus` stores loop closed-form coefficients (C7/C9) as `Float`.
- `KotlinSourceRenderer` refuses F64 (only F32/I32/Bool render), constants render with `f`.

StableHLO: rank-N constants are `FloatArray` (`denseFromArray`), so an F64 rank-N constant
carries F32 precision. RNG and custom-call scratch are f32 by design.

PJRT: `runOn` accepts F32/I32 only; `runOnF64` all-F64; `runOnPjrt` F32 only.

autograd: `jacobian`/`hessian`/`jacobianReverse` return `DTensor<Rank2<Sym, Sym>, F32>` and
their assembly helpers read `hostF32`. `customVjp`'s finite-difference checker is F32.

## Order of work (one commit per step, full suite before each)

1. **Scalar `Double`**: already lowers end to end. Add coverage for `jvp`, `vjp`, `grad2`,
   `valueAndGrad`, transcendental ops, and a hidden-round-trip test (a captured `Double`
   whose F32 rounding moves the gradient by more than 1e-12 relative).
2. **F64 tensors on the host path**: `HostOpsF64.kt` and `BroadcastOpsF64.kt`, generated
   from the F32 files (elementwise with broadcasting, scalar mixing with `Double`, unary
   math, comparisons/`where`, reductions, shape ops, matmul, softmax, the synthesis
   helpers); `Tensors.f64*`, `hostF64()`. Synthesis: one dtype-aware lookup (a candidate
   is kept when its `DTensor` dtype argument is the function's tensor dtype or it has
   none), so the F32 candidate set is unchanged; F64 gates, `Double` constants. FIR:
   `toDouble` bridge, literals kept at the tensor's width. Mixed F32/F64 tensors in one
   body are refused by name.
3. **StableHLO and PJRT**: full-precision F64 rank-N constants; `PjrtSession` general entry
   that takes each param at its own dtype (F32, F64, I32); an F64 twin of the reference
   interpreter so interpreter and PJRT can be compared on the same gradient graph.
4. **Linear algebra** under `grad {}` in F64 (the host twins exist; synthesis must pick the
   F64 overload).
5. **Remaining ops** by likely scientific use: softmax/logSoftmax, losses, special functions,
   jacobian/hessian return types, then conv/pool (refused by name for F64 if not done).
6. **`examples/gaussian-process`** in F64.

## Rules kept

- F32 must not change: every existing test passes unmodified, and the printed gradient
  source for existing tests is compared byte for byte before and after (a script dumps
  every `dumpGradSource` output of the plugin tests on `main` and on the branch).
- F64 tolerances: about 1e-8 relative against central differences, each justified.
- ABI baselines updated with `./gradlew updateKotlinAbi`, explained in the commit.
