# Sparse Parity Audit — Phase E1 (§0.4.410)

**Status: RATIFIED (Pedro, 2026-09-20) — E1a→E1c GO, per the audit's own
recommendations at every decision point: `matdiv` SKIPPED (a solver arrives
as its own designed feature or never), GPU = pinned emit refusal for the E1b
ops (the §0.4.408 RNG precedent), row-sparse embedding gradients deferred to
Phase F. E1a landed §0.4.417** (`:core` `SparseTensor` — rank-2 CSR, `fromCoo`
duplicate-summing construction, `toDense`, elementwise `plus`/`minus`/`times`
union/intersection merges, sparse×dense elementwise `times`, counting-sort
`transpose`, Double-accumulator SpMM and Gustavson SpGEMM — certified against
dense references on seeded random patterns at densities 0/0.05/0.3/0.7).
**E1b landed §0.4.418** (`OpKind.SPARSE_MATMUL` + fused
`SPARSE_MATMUL_VALUES_ADJOINT`, SparseMatmulRule + bilinear tangent,
interpreter arms bit-exact against E1a, host twins `sparseMatmul` /
`sparseMatmulTransposed` / `sparseMatmulValuesAdjoint`, CostModel, and the
ratified pinned emit refusal — certified against the dense MatmulRule on
toDense'd operands, JVP⇄VJP, and forward-over-reverse HVP vs the dense twin;
see §2's design-decision record below for the transposed-CSR mechanism the
slice settled).
**E1c-pre landed §0.4.419** (`OpKind.ZEROS_LIKE(template)` — the
PARAM-ADDRESSED structural zero, the runtime-extent-family treatment:
DxirReverseTransform emits it on the cloned integer param ITSELF instead of
an anonymous sentinel-dimmed const, so the zero names its param by
construction and the §0.4.400 one-integer-param synthesis gate is LIFTED —
any number of integer params per `grad {}` lambda lowers, certified E2E on a
two-index-param embedding lambda with DIFFERENT index extents. Own VjpRule
closed under itself + forward tangent, so higher-order transforms compose
through gradient bodies containing it; emits as a static splat-zero constant,
template SSA unreferenced).
**E1c landed §0.4.420 — the ratified sparse arc is COMPLETE.** The `grad {}`
sparse surface: `sparseMatmul(values, colIdx, rowPtr, dense)` FIR front-end
(result N copied only when concrete, -1 symbolic otherwise — the
conv/flatten convention), synthesis arms for all three op spellings (plain →
`sparseMatmul` host twin, transposed 5-operand → `sparseMatmulTransposed<S>`
with the template's shape as its type arg, SDDMM → `sparseMatmulValuesAdjoint`)
plus the placeholder-`Lit<Int>`-for-N result-atom derivations. Certified E2E
through the K2 plugin on the GNN shape: a FOUR-param lambda (values, colIdx,
rowPtr, dense) with empty row + skewed row + explicit stored zero, linear and
nonlinear-recompute losses, exact quarter-grid hand oracle, the stored zero
receiving a gradient, and both integer zeros at their own extents (nnz=5 ≠
N+1=4 — the §0.4.419 addressing under real CSR params). Per the ratified
scope the arc now STOPS: matdiv stays skipped, GPU stays the pinned refusal,
row-sparse embedding gradients stay a Phase F item.
**Close-out (§0.4.433):** the arc's still-open tails (ELL-padded GPU
emission behind the pinned refusal; row-sparse embedding gradients with
Phase F) are consolidated in the plan doc's "Remaining tails" list; this
audit is a closed record.
Companion to [DIFFKT_PARITY_PLAN.md](DIFFKT_PARITY_PLAN.md) Phase E. Walked
from a fresh shallow clone of `facebookresearch/diffkt` @ HEAD (2026-09-20):
`kotlin/api/src/main/kotlin/org/diffkt/Sparse*.kt`, the JNI surface
(`external/Eigen.kt` → `cpp/ops/SparseOps.{h,cpp}` →
`cpp/ops/Sparse/ArithmeticEigen.cpp`), the touchpoints in `Matmul.kt` /
`Matdiv.kt` / `Combinators.kt` / `FloatTensorOperations.kt` /
`Operations.kt`, and `SparseOpsTest.kt` / `SparseRowTest.kt`. This was a
**docs-only** deliverable when written; the §-sized slicing at the end was
subsequently ratified and executed as proposed (§0.4.417–420).

## 1. What DiffKT actually ships

Three sparse representations, one narrower than the next:

- **`SparseFloatVector`** — rank-1, parallel `values`/`inner` index arrays.
  Marked with a removal TODO in their own source
  (facebookincubator/diffkt#94), `operations` still dispatches to
  `StridedFloatTensorOperations` with an "Is this right?" comment. Vestigial;
  not a parity target.
- **`SparseFloatTensor`** — rank ≥ 2, a hierarchical CSR: `values: FloatArray`
  plus one `DimData(inner, outer)` per non-terminal dimension (the Eigen
  inner/outer-index scheme, generalized by nesting — their doc comment links
  Eigen's sparse tutorial directly). Rank-2 construction routes through JNI
  `convertToCoo` into an Eigen CSR; rank > 2 builds the nested DimData in
  Kotlin. This is the GNN-adjacency type.
- **`SparseRowFloatTensor`** — rank-2 with sparse ROWS: a flat `FloatArray`
  of `[rowIndex, v₀ … v_{C−1}]` records, rows required sorted. Its doc
  comment states the purpose: "particularly useful for embedding table
  gradients". It is produced by exactly one internal site —
  `FloatTensorOperations.scatter` reroutes to `scatterSparseRow` when
  `axis == 0 && newShape.rank == 2`, i.e. the embedding-gradient scatter —
  and consumed by `Operations.kt`'s binary-op dispatch so that gradient
  ACCUMULATION (`plus`) over two row-sparse gradients merges rows instead of
  densifying. It is a **gradient-representation optimization**, not a user
  surface.

### The op surface (`SparseFloatTensor`)

| Op | Implementation | Restrictions found in source |
|---|---|---|
| `plus` / `minus` | JNI → Eigen CSR add/sub | sparse ⊗ sparse ONLY — the `wrap` helper is `TODO("Cannot (automatically) convert to SparseFloatTensor")` for a dense operand |
| `times` (elementwise) | Eigen for sparse ⊗ sparse; Kotlin `zip` for sparse ⊗ dense | `zip` `require`s f(0, ·) = 0 and reads `nonZeroIndices` — see the trap below |
| `transpose` | JNI → Eigen | 2D, or 3D with the LAST TWO axes swapped, nothing else (`require`d) |
| `matmul` | JNI → Eigen | sparse ⊗ sparse only, 2D or 3D-batched; dispatched by an `is SparseFloatTensor` test INSIDE dense `matmul` (their own TODO says it should move into the Operations object) |
| `matdiv` | JNI → Eigen | sparse-only (dense explicitly `require`-rejected), square RHS. Implementation: `SparseLU` factorization of the RHS, **explicit inverse** by solving against the identity, then `left · A⁻¹` (a StackOverflow-cited recipe — not even a direct solve) |
| `reduce` (via `Combinators.kt`) | Kotlin, groups `nonZeroIndices` | `require(keepDims == true)`, rank ≥ 2 result only |
| `map` / `zip` | Kotlin | zero-preservation checked by probing `f(0f)` at runtime; densifies when `f(0) ≠ 0` |

**The `nonZeroIndices` trap — evidence the surface is vestigial.** COO
entries are memoized ONLY when the tensor was built through the
`(shape, elements)` list constructor (`invokedElems`); the property is
otherwise `TODO("Conversion not yet written")`. Every tensor that comes BACK
from an Eigen op (add/sub/matmul/matdiv/transpose return
`SparseEigen2D.toSparseFloatTensor()`, which never sets `invokedElems`)
therefore THROWS on `zip`, `reduce`, or `toDense`-adjacent paths that read
`nonZeroIndices`. Chained sparse expressions beyond one Eigen hop into a
Kotlin-side op cannot have worked. This is not a surface anyone leaned on
hard.

### The AD story: sparse is primal-only data

`SparseFloatTensorOperations.plus/minus/times` all
`require(derivativeId == NoDerivativeID)`. A sparse tensor participates in
differentiation only as a WRAPPED PRIMAL: `reverseDerivative(t1) { … }` boxes
the sparse tensor in a `ReverseTensor`, the reverse machinery dispatches
above the sparse layer, and the gradients that come out are **dense** (their
own `SparseOpsTest` pins dense full Jacobians for d/d(sparse-operand) of
`plus` and `matmul`). There is no sparse VJP rule anywhere: no gradient ever
flows *as* a `SparseFloatTensor`, and the only sparse gradient
REPRESENTATION in the system is the row-sparse embedding-scatter above,
which is invisible to users. DiffKT's sparse feature is, precisely: **a
CPU-side Eigen shim for sparse primal arithmetic, plus a row-sparse
embedding-gradient container.** No GPU story exists (the Eigen calls are the
only implementation).

## 2. What a Tlaloc `:core` sparse type would need

Tlaloc's layers make the honest costing very different from "port six ops":
every op that reaches `grad {}` needs interpreter + host-twin +
VjpRule/tangent + synthesis + (emitter arm or pinned refusal), and every op
that doesn't is host-level sugar.

### Storage layout

Rank-2 **CSR** (`values [nnz] F32`, `colIdx [nnz] I32`, `rowPtr [N+1] I32`)
is the right v1: it matches what DiffKT actually exercises (their rank > 2
nesting is where the `nonZeroIndices` trap lives), it is the layout SpMM
wants, and it decomposes into three DENSE tensors — which matters below.
COO would only be the better call if construction/mutation dominated, and it
does not: the GNN adjacency is built once, multiplied many times.

### Which ops, at which level

- **Host-level (`:core`, no IR)**: `SparseTensor` type + `fromCoo`/`toDense`
  constructors, elementwise `plus`/`minus`/`times` (CSR merge — the
  `SparseRowFloatTensor.qualifiedZip` two-pointer walk generalizes),
  `transpose` (CSR→CSC reinterpretation or an explicit half-perm),
  sparse×dense and sparse×sparse `matmul` (CSR SpMM/SpGEMM loops through a
  Double accumulator, the house convention). All pure-Kotlin, all testable
  against dense references. **~1 §.**
- **IR-level (`grad {}`-relevant)**: the ONE op that earns an OpKind is
  **`SPARSE_MATMUL`** — the GNN kernel `adjacency [N,N] sparse × features
  [N,D] dense → dense [N,D]`. Design that avoids a sparse dtype entirely:
  the sparse operand rides as its three dense component tensors,
  `SPARSE_MATMUL(values, colIdx, rowPtr, dense)`, with the `N` extent read
  off `rowPtr`'s runtime shape — the SUM_TO/PAD_TO runtime-extent house
  pattern, no new `DxirType` vocabulary (a first-class sparse dtype would
  touch the dims model, IrType atoms, CSE keys, and every layer's
  assumptions — rejected on cost).
  - Adjoints: `d_dense = SPARSE_MATMUL(transposed-CSR, upstream)` (needs the
    transpose as a host/interpreter helper or a `transposed` attr);
    `d_values[k] = Σ_j upstream[row(k), j] · dense[colIdx[k], j]` — an SDDMM
    masked to the sparsity pattern, which is a natural FUSED adjoint op
    (`SPARSE_MATMUL_VALUES_ADJOINT(upstream, dense, colIdx, rowPtr)`), the
    EMBEDDING_GRAD/CONV2D_*_ADJOINT precedent. `colIdx`/`rowPtr` are integer
    tensors: non-differentiable, structural-zero slots (§0.4.54).
  - Forward tangent: bilinear — `SPARSE_MATMUL(d_values…, dense) +
    SPARSE_MATMUL(values…, d_dense)` — cheap once the primal arm exists.
  - **~1–2 §** for op + interpreter + host twins + rules + JVP⇄VJP certs.
- **`grad {}` synthesis surface**: blocked by a KNOWN restriction — the
  §0.4.400 index-param acceptance admits **one** integer-typed param per
  lambda ("several would be ambiguous and reject": the structural integer
  zero cannot name which param it zeroes). A CSR operand carries TWO integer
  params (`colIdx`, `rowPtr`). Generalizing structural zeros to be
  per-param-addressed is the enabling work, and it is prerequisite, not
  optional. **~1 § on its own**, then ~1 § for the FIR arm + `irSparseMatmul`
  + E2E certs.
- **`matdiv`: recommend REFUSAL.** It is a sparse direct solve — DiffKT's is
  SparseLU *via explicit inverse* through Eigen JNI. Tlaloc has no native
  solver dependency and a pure-Kotlin SparseLU is not one honest § (nor
  two). It is also DiffKT-sparse-ONLY (their dense matdiv `require`-fails),
  primal-only (no VJP anywhere), and exercised by a handful of generated
  tests. Parity value ≈ 0. If a solve is ever wanted, it should arrive as
  its own designed feature (iterative CG for SPD adjacency-era matrices
  would be both easier and more honest than LU), not as a parity checkbox.

### GPU

StableHLO/XLA has no sparse types. The options, honestly:

1. **Loud emit refusal** (the §0.4.408 RNG precedent): host + interpreter
   only, `EmitterTest` pin, documented `GradientEmissionCoverageTest`
   exclusion. Recommended for v1 — it is exactly DiffKT's own position
   (CPU-only Eigen), so parity is not even reduced by it.
2. Gather/scatter composition: per-row segments of `colIdx` have irregular
   lengths, so a faithful CSR SpMM needs either a WHILE over rows (defeats
   XLA) or padding to max-degree (ELL-style — a DIFFERENT format with its
   own memory blowup on skewed degree distributions). Real, but a design of
   its own; do not promise it as a footnote.
3. Densify-and-matmul fallback: correct, silently O(N²) — worse than a loud
   refusal (the §0.4.392 principle: a silent behavior fork is worse than a
   named restriction).

## 3. Cost summary

| Slice | Content | Size |
|---|---|---|
| E1a | `:core` CSR type + elementwise/transpose/matmul host kernels + dense-reference certs | 1 § |
| E1b | `SPARSE_MATMUL` + fused values-adjoint: OpKinds, interpreter, host twins, VjpRule + tangent, JVP⇄VJP + analytic certs, emit-refusal pins | 1–2 § |
| E1c-pre | multi-integer-param structural zeros in synthesis (names its param) | 1 § |
| E1c | FIR arm + synthesis + E2E `grad {}` GNN-shaped cert | 1 § |

Total: **~4–5 honest §** for the full vertical; **1 §** if only host-level
parity is wanted; **0 §** if the no-go is ratified.

## 4. Recommendation and decision points for Pedro

**Recommendation: conditional no-go.** The audited surface is a narrow,
CPU-only, primal-only Eigen shim whose Kotlin half is partly broken in
DiffKT itself (the `nonZeroIndices` trap), aimed at a GNN workload Tlaloc
has never had on its books. Nothing in it advances the differentiation
story — sparse tensors never carry gradients in DiffKT. The honest ledger
entry is "❌ by decision, audit on file" unless a real Tlaloc GNN/embedding
workload exists to pull it in — in which case do E1a→E1c in order and stop
wherever the workload is satisfied (E1a alone covers everything DiffKT's
sparse users could actually chain).

Decision points, explicitly:

1. **Is a GNN-adjacency workload real for Tlaloc?** If yes → E1a + E1b at
   minimum (the sparse×dense matmul VJP is the whole point). If no → no-go.
2. **`matdiv`**: skip (recommended), or ratify a solver as a separate
   designed feature — never as parity.
3. **GPU**: is a pinned emit refusal acceptable for v1 (recommended, and
   matches DiffKT's own CPU-only reality)? The ELL-padded composition is
   the only credible future path and should be its own §-sequence if ever.
4. **Row-sparse embedding gradients** (`SparseRowFloatTensor`'s actual
   role): Tlaloc's `EMBEDDING_GRAD` is a dense scatter-add today. A sparse
   gradient REPRESENTATION only pays off once an optimizer-loop story
   (Phase F) exists to consume it without densifying. Defer to F; it is an
   optimization, not parity.
