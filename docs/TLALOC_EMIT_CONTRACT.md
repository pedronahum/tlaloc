# Tlaloc StableHLO Emit Contract

> **Audience:** anyone wiring **xatlib** pattern matchers, writing the eventual `toNagual` lowering, or otherwise consuming Tlaloc's MLIR output. Read this instead of spelunking `Emitter.kt`.
>
> **Source of truth:** `stablehlo/src/commonMain/kotlin/io/tlaloc/stablehlo/Emitter.kt` and `SdyEmit.kt`. When this doc and the emitter disagree, the emitter wins; please open a PR to fix the doc.
>
> **Last refreshed:** §0.4.311 (JDK 25 LTS bump). The pipeline shape and per-OpKind output have been stable since §0.4.279 (dtype guard) — most edits since are additive.

---

## 1. Pipeline shape

The production GPU/CPU lowering is **always** this composition:

```
DxirFunction
  → recognizeAll()              # Layer-3 pattern recognition (fills `attrs["match_id"]`, etc.)
  → coarsenRecognizedPatterns() # produces OpKind.COARSENED nodes with attached primal_body
  → decomposeCoarsened()        # inlines primal_body — COARSENED → primitives (no kernel descriptor → no custom_call)
  → toStablehlo()               # emit MLIR text
```

**The emit input is therefore primitive-only Dxir.** §0.4.310's diagnostic test (`LlamaDecoderCoarseningEmitDiagnosticTest`) pins this: the production GPU path emits **zero** `stablehlo.custom_call` ops. Coarsening info is used internally by AD's gradient_body composition; it is **not** carried into the emitted MLIR.

If a frontend ever wants `custom_call` emission (e.g. xatlib-resolved kernels), it must attach a `KernelDescriptor` to the COARSENED op **before** `decomposeCoarsened` runs — see §6.

## 2. Entry points

```kotlin
fun DxirModule.toStablehlo(): String         // wraps `module { ... }`, includes sdy.mesh decls
fun DxirFunction.toStablehlo(indent = ""): String  // bare func.func, no module wrap
```

Tests typically use `DxirFunction.toStablehlo()` and dump it standalone (e.g. `benchmarks/build/llama-decoder-cpu-baseline.mlir`).

## 3. Module-level shape

```mlir
module {
  sdy.mesh @data_par = <["data"=8, "model"=4]>     // one decl per mesh, aggregated from module + per-function
  sdy.mesh @other = <["x"=2]>

  func.func @main(%0: tensor<32x64xf32>, %1: tensor<...>, ...) -> tensor<f32> {
    ...
    return %N : tensor<f32>
  }

  func.func @other_function(...) -> ... { ... }
}
```

Notes:
- Function name comes from `DxirFunction.name` directly (the LlamaDecoder test names it `@llama_decoder_layer_loss`; runtime callers normalise to `@main` before sending to PJRT).
- Multi-result returns wrap their type list in parens: `-> (tensor<...>, tensor<...>)`. Single-result is bare.
- Parameters are named `%0, %1, ...` (Dxir SSA IDs, **not** semantic param names).

## 4. Per-OpKind emission table

`%X`/`%Y`/`%Z` denote operands; `%N` is the result; `:T` denotes a tensor type like `tensor<32x64xf32>`.

### 4.1 Elementwise unary (preserves shape & dtype)

| OpKind  | Emitted                                  |
|---------|------------------------------------------|
| NEG     | `%N = stablehlo.negate %X : T`           |
| ABS     | `%N = stablehlo.abs %X : T`              |
| EXP     | `%N = stablehlo.exponential %X : T`      |
| LOG     | `%N = stablehlo.log %X : T`              |
| SIN     | `%N = stablehlo.sine %X : T`             |
| COS     | `%N = stablehlo.cosine %X : T`           |
| SQRT    | `%N = stablehlo.sqrt %X : T`             |
| RSQRT   | `%N = stablehlo.rsqrt %X : T`            |
| TANH    | `%N = stablehlo.tanh %X : T`             |
| SIGMOID | `%N = stablehlo.logistic %X : T`         |
| NOT     | `%N = stablehlo.not %X : tensor<...xi1>` |

### 4.2 Elementwise binary (with auto-broadcast)

| OpKind | Emitted                                 |
|--------|-----------------------------------------|
| ADD    | `%N = stablehlo.add %X, %Y : T`         |
| SUB    | `%N = stablehlo.subtract %X, %Y : T`    |
| MUL    | `%N = stablehlo.multiply %X, %Y : T`    |
| DIV    | `%N = stablehlo.divide %X, %Y : T`      |
| POW    | `%N = stablehlo.power %X, %Y : T`       |
| LAND   | `%N = stablehlo.and %X, %Y : tensor<...xi1>` |

**Auto-broadcast rule (load-bearing):** when an operand's shape differs from the result, the emitter inserts `stablehlo.broadcast_in_dim` *before* the binary op. Both operands must be **same-rank** with each dim either equal to the result dim or equal to 1 — there is **no** implicit rank-extension à la NumPy. The injected form is always identity-mapped:

```mlir
%bc = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<3x4xf32>
%N = stablehlo.add %bc, %1 : tensor<3x4xf32>
```

Pattern matchers must therefore expect intermediate `broadcast_in_dim` between an operand definition and a binary op when shapes don't match exactly.

### 4.3 Type & shape

| OpKind     | Emitted                                                                      |
|------------|------------------------------------------------------------------------------|
| CAST       | `%N = stablehlo.convert %X : (Tin) -> Tout`                                  |
| RESHAPE    | `%N = stablehlo.reshape %X : (Tin) -> Tout`                                  |
| TRANSPOSE  | `%N = stablehlo.transpose %X, dims = [p0, p1, ...] : (Tin) -> Tout`          |
| BROADCAST  | `%N = stablehlo.broadcast_in_dim %X, dims = [d0, ...] : (Tin) -> Tout`       |
| SLICE      | `%N = stablehlo.slice %X [s0:l0, s1:l1, ...] : (Tin) -> Tout` (strides if ≠1)|
| CONCAT     | `%N = stablehlo.concatenate %X, %Y, ..., dim = D : (Tins) -> Tout`           |

### 4.4 Reductions

Lowered to `stablehlo.reduce` with a per-OpKind reducer + init constant:

```mlir
%init = stablehlo.constant dense<INIT_LIT> : tensor<f32>
%N = stablehlo.reduce(%X init: %init) applies stablehlo.OP across dimensions = [d0, d1, ...]
  : (Tin, tensor<f32>) -> Tred
```

| OpKind | Reducer                   | Init                     |
|--------|---------------------------|--------------------------|
| SUM    | `stablehlo.add`           | `0.0`                    |
| MAX    | `stablehlo.maximum`       | `-inf` (`0xFF800000` f32)|
| MIN    | `stablehlo.minimum`       | `+inf` (`0x7F800000` f32)|
| MEAN   | `stablehlo.add` then `divide` by reduced-element-count | `0.0` |

**Keep-dims tail (load-bearing).** When the result rank equals the input rank with the reduced axes as size-1, the emitter follows the reduce with a `broadcast_in_dim` to re-inflate. RmsNorm chains depend on this:

```mlir
%s49 = stablehlo.reduce(%X init: %s48) applies stablehlo.add across dimensions = [1] : ...  -> tensor<32xf32>
...
%14 = stablehlo.broadcast_in_dim %s51, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
```

### 4.5 Matmul / dot

`MATMUL` always emits `stablehlo.dot_general`. Three configurations:

1. **Named-axis inference** (when both operands carry `axisNames` and no explicit attr): contracts on the shared axis name's positions.
2. **Implicit-batched** (no attrs, rank ≥ 2):
   - rank-2: `contracting_dims = [1] x [0]`
   - rank ≥ 3: `batching_dims = [0..r-3] x [0..r-3], contracting_dims = [r-1] x [r-2]`
3. **Explicit attrs:** reads `lhs_contracting_dims`, `rhs_contracting_dims`, `lhs_batching_dims`, `rhs_batching_dims` (all `List<Int>`).

Emitted form:

```mlir
%N = stablehlo.dot_general %A, %B,
  batching_dims = [b0_l, ...] x [b0_r, ...],     // omitted if empty
  contracting_dims = [c0_l, ...] x [c0_r, ...]
  : (TA, TB) -> TC
```

`DOT` (rank-1 inner product) is `stablehlo.dot_general` with `contracting_dims = [0] x [0]`.

`Q · K^T` in transformer code is **NOT** emitted via the SDPA op; the primitive form (an explicit `stablehlo.transpose` followed by `stablehlo.dot_general %Q, %K_t, contracting_dims = [1] x [0]`) is what xatlib will see in the Llama baseline (§0.4.286 fix). See §8.

### 4.6 Convolution

`CONV2D` and `CONV_TRANSPOSE2D` emit `stablehlo.convolution`. Layout is fixed:

| OpKind             | Kernel layout                                     |
|--------------------|---------------------------------------------------|
| CONV2D             | `[o, i, 0, 1]` (output_chan, input_chan, h, w)   |
| CONV_TRANSPOSE2D   | `[i, o, 0, 1]` (i/o swapped vs forward)          |

Inputs are NCHW (`[b, f, 0, 1]`). Reads `window_strides` (required, `List<Int>` length 2), optional `padding`, `lhs_dilation`, `rhs_dilation`, `feature_group_count`, `batch_group_count`.

### 4.7 Composite activations & norms (no fused op)

These OpKinds expand into a sequence of primitives — pattern matchers should match on the *expanded* form, not the original op name:

| OpKind     | Expansion                                                                                    |
|------------|----------------------------------------------------------------------------------------------|
| RELU       | `constant(0)` + `maximum`                                                                    |
| STEP       | `constant(0/1)` + `compare GT` + `select` (returns 0 at x=0, matching XLA, **not** H(0)=0.5) |
| SILU       | `logistic` + `multiply` (i.e. `x * sigmoid(x)`)                                              |
| GELU       | tanh-approximation: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`                     |
| SOFTMAX    | max-reduce → broadcast → subtract → exp → sum-reduce → broadcast → divide (along `axis`)     |
| LOGSUMEXP  | similar to SOFTMAX but ends in `log(sum) + max`                                              |
| LAYERNORM  | mean + center + variance + sqrt + divide (six-plus reduces in keep-dims mode)                |
| RMSNORM    | square + sum-reduce → mean → eps → sqrt → divide (keep-dims with `broadcast_in_dim` tail)    |

The `axis` attr (where applicable) defaults to the last axis. SOFTMAX always reduces along **one** dimension; multi-axis softmax is unsupported.

### 4.8 Attention

`SCALED_DOT_PRODUCT_ATTENTION` lowers to:

1. `dot_general(Q, K, contracting_dims = [r-1] x [r-1])` — produces `(..., sQ, sK)` scores **without** an explicit K transpose
2. scale by `1/√d_k`
3. softmax along the `sK` axis
4. `dot_general(softmax, V, contracting_dims = [r-1] x [r-2])` — final output

**Caveat.** This SDPA-emitter path only fires if the source frontend produced a `SCALED_DOT_PRODUCT_ATTENTION` op directly. In the production pipeline the FlashAttention pattern is recognised and coarsened, then `decomposeCoarsened` inlines the **primal_body** (which uses `transpose` + `dot_general` + `[1] x [0]` contraction). Both forms are mathematically equivalent; xatlib matchers consuming production output should expect the **transpose-then-dot** form, not the no-transpose `[r-1] x [r-1]` SDPA form. See §8 for the concrete Llama trace.

### 4.9 Indexing

| OpKind      | Emitted                                                                                        |
|-------------|------------------------------------------------------------------------------------------------|
| GATHER      | `stablehlo.gather` with full `dimension_numbers` + `slice_sizes`. Substrate form (scalar I32 index, no attrs) auto-synthesises the attr block. |
| EMBEDDING   | Lowered to `stablehlo.gather` with canonical embedding attrs (`offset_dims=[indices.rank], collapsed_slice_dims=[0], start_index_map=[0]`) |
| SCATTER     | `stablehlo.scatter` with a custom reduction body block (default `add`). Substrate form synthesises attrs for `arr[idx] = val`. |
| SCATTER_ADD | `stablehlo.scatter` with `stablehlo.add` body. **Peephole:** if base is `BROADCAST(constant(0), ...)` the emitter elides the scatter and returns `%upd` directly (§0.4.133). |

### 4.10 ARGMAX

Two-input `stablehlo.reduce` with iota; result is the second component:

```mlir
%pair:2 = stablehlo.reduce(%X init: %neginf), (%iota init: %zero) across dimensions = [axis]
  : (Tval, Tidx, tensor<f32>, tensor<i64>) -> (TredVal, TredIdx)
  reducer(%cv: tensor<f32>, %nv: tensor<f32>) (%ci: tensor<i64>, %ni: tensor<i64>) {
    %gt = stablehlo.compare GT, %cv, %nv, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %sv = stablehlo.select %gt, %cv, %nv : tensor<i1>, tensor<f32>
    %si = stablehlo.select %gt, %ci, %ni : tensor<i1>, tensor<i64>
    stablehlo.return %sv, %si : tensor<f32>, tensor<i64>
  }
%N = %pair#1     // indices only
```

### 4.11 Other

- `BATCHNORM` → `stablehlo.batch_norm_inference` (5 operands: input, scale, offset, mean, variance; `epsilon` and `feature_index` attrs)
- `CROSS_ENTROPY` → log-softmax + one-hot multiply + sum (no fused op)
- `SHARD_CONSTRAINT` → `sdy.sharding_constraint %X <SDY>`  (§5)
- `MANUAL_COMPUTATION` → `sdy.manual_computation` with nested region; `in_shardings` / `out_shardings` / `manual_axes` attrs (§5)
- `IF` / `WHILE` → **error**; structural control flow is decomposed by Stage B before emit
- Other unimplemented OpKinds → **error** (`StableHLO lowering not yet implemented`)

## 5. Sharding (SDY) annotations

Three places SDY appears in emitted MLIR:

1. **Module-scope mesh decls** (one per mesh, aggregated):
   ```mlir
   sdy.mesh @data_par = <["data"=8, "model"=4]>
   ```

2. **`SHARD_CONSTRAINT` op:**
   ```mlir
   %N = sdy.sharding_constraint %X <@data_par, [{"data"}, {}]> : T
   ```

3. **`MANUAL_COMPUTATION` op** (nested region):
   ```mlir
   %N:M = sdy.manual_computation(%op0, ...)
     in_shardings=[<@mesh, [...]>, ...]
     out_shardings=[<@mesh, [...]>, ...]
     manual_axes={"axis0", ...}
     (%arg0: T0, ...) {
       ...body...
       sdy.return %res0, ... : T0, ...
     } : (Tins...) -> Touts...
   ```

The per-tensor sharding format (used in 2 and 3) is:

```
<@mesh_name, [dimSharding_0, dimSharding_1, ...], replicated={axis_a, axis_b}>
```

where each `dimSharding` is `{"axis"[, "axis2"]}` for closed (final) shardings, `{"axis"?}` for open (may be split further), with optional priority suffix `p<int>`. Sub-axis refs serialise as `"name":(preSize)size`.

Currently no SDY sharding survives onto **primitive** ops — only on the three constructs above. If a `COARSENED` op carries a `KernelDescriptor` (so it emits `custom_call` rather than being decomposed), its `sharding` field is attached as an `sdy.sharding = #sdy.sharding_per_value<[...]>` attribute on the custom_call. See §6.

## 6. Coarsened ops in the pipeline

`decomposeCoarsened` (in `:ir`) inlines `primal_body` for any `OpKind.COARSENED` lacking a `KernelDescriptor.ATTR_KEY` attr. After this pass, the emitter sees only primitives. **In production (CPU baseline + PJRT-XLA + IREE) this is the only path.**

If a future flow attaches a kernel descriptor before `decomposeCoarsened` (so the COARSENED op survives into emit), the emitter generates:

```mlir
%N:M = stablehlo.custom_call @kernel_name(%op0, ...) {
    backend_config = "key0=val0, key1=val1, ...",   // deterministically sorted
    has_side_effect = false,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [...]>, ...]>   // present iff node.sharding ≠ null
  } : (Topers...) -> (Touts...)
```

Constraints:
- A COARSENED op without a kernel descriptor **after** `decomposeCoarsened` is a hard error (`requires kernel_descriptor attr or decomposition`).
- `decomposeCoarsened` is single-result-only in v1; multi-result COARSENED can't be decomposed. Region-bearing primal_bodies are also rejected.

## 7. Type encoding

Static shapes only — `?` dynamic dims are out of scope.

| DType | Spelled |
|-------|---------|
| F32   | `f32`   |
| F64   | `f64`   |
| I32   | `i32`   |
| I64   | `i64`   |
| Bool  | `i1`    |

Tensor types: `tensor<f32>` (rank-0 / scalar), `tensor<32xf32>` (rank-1), `tensor<32x64xf32>` (rank-N).

## 8. Pattern-matching guide

A short list of things every xatlib / `toNagual` author trips over:

1. **Same-rank broadcast only.** Binary ops require operands of equal rank with each dim either equal to the result or equal to 1; the emitter inserts `broadcast_in_dim` automatically. There is no implicit rank-extension. Match operand-defining-op chains through any `broadcast_in_dim` immediately preceding a binary op.

2. **Canonical matmul dim ordering.**
   - Rank-2 with no attrs: `contracting_dims = [1] x [0]`
   - Rank ≥ 3 with no attrs: `batching = [0..r-3] x [0..r-3], contracting = [r-1] x [r-2]`

3. **Q · K^T is two ops, not one.** Production attention (FlashAttention pattern decomposed via `primal_body`) emits an **explicit** `stablehlo.transpose` of K followed by `stablehlo.dot_general %Q, %K_t, contracting_dims = [1] x [0]`. Match on the (transpose, dot_general) pair, not on a single fused op. The SDPA-emitter path (which uses no transpose + `[r-1] x [r-1]`) does not fire in production today.

4. **RmsNorm tail is `rsqrt`, not `divide`.** The Llama baseline uses `1 / sqrt(mean(x²) + eps)` realised as `rsqrt(mean + eps)` followed by `multiply(x, rsqrt_result)`. Match `multiply(x, rsqrt(add(broadcast(reduce(multiply(x,x), [axis], add)/N), eps)))` — note the `broadcast_in_dim` after every keep-dims reduce.

5. **Softmax expansion idiom.** Always: max-reduce → broadcast → subtract → exp → sum-reduce → broadcast → divide. Both reduces are `stablehlo.reduce`; both broadcasts are `stablehlo.broadcast_in_dim`. The `axis` attr (default last) is the only reduced dim.

6. **SwiGLU = `silu(gate) * up`, two multiplies.** In primitive form: `%silu = multiply %gate, logistic(%gate)` followed by `%out = multiply %silu, %up`.

7. **SSA names are Dxir IDs, not semantic names.** `%0 ... %N` are param/result IDs; `%s0 ... %sM` are emitter-synthesised intermediates (reduce inits, scalar constants, broadcast tails). Do not assume `%N` ordering carries semantic meaning.

8. **Multi-result references.** `ARGMAX` emits a multi-result pair internally (`%pair:2`, taking result #1). (The `SPLIT` kind, which emitted N distinct `stablehlo.slice` ops referenced downstream as `%result#0`, `%result#1`, was deleted in §0.4.454 — spell splits as per-piece `SLICE`.)

9. **Constants for non-finite floats.** `-inf` is emitted as `dense<0xFF800000> : tensor<f32>` (raw bit pattern); `+inf` as `0x7F800000`. Pattern matchers should accept both decimal and hex literal forms.

10. **`module` wrap is optional** depending on entry point: `DxirModule.toStablehlo()` produces `module { ... }`; `DxirFunction.toStablehlo()` produces a bare `func.func`. The Llama baseline dump uses the bare form.

## 9. Canonical example

`benchmarks/build/llama-decoder-cpu-baseline.mlir` (regenerate with `./gradlew :benchmarks:jvmTest --tests "*LlamaDecoderStablehloEmitTest*dumpsCpuBaseline*"`) is the reference output. It exercises:

- 13 input tensors (input + 1-hot labels + RoPE phase + 7 weight matrices + 1 down-proj + 2 eps tensors)
- Two RmsNorm chains (pre-attention, pre-MLP)
- Q/K/V projections via `dot_general`
- RoPE (sin/cos rotation via the `cos*Q − sin*Q` primitive identity)
- Attention as **transpose + dot_general + softmax + dot_general** (point 3 above)
- SwiGLU as two multiplies (point 6)
- Residuals
- LM head + log-softmax + one-hot multiply + sum → scalar loss

The full file is 66 lines; an excerpt of the RmsNorm head:

```mlir
%13 = stablehlo.multiply %0, %0 : tensor<32x64xf32>                                          // x²
%s48 = stablehlo.constant dense<0.0> : tensor<f32>
%s49 = stablehlo.reduce(%13 init: %s48) applies stablehlo.add across dimensions = [1]
  : (tensor<32x64xf32>, tensor<f32>) -> tensor<32xf32>                                       // Σ x²
%s50 = stablehlo.constant dense<64.0> : tensor<32xf32>
%s51 = stablehlo.divide %s49, %s50 : tensor<32xf32>                                          // mean
%14 = stablehlo.broadcast_in_dim %s51, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>     // re-inflate (keep-dims)
%15 = stablehlo.add %14, %11 : tensor<32x1xf32>                                              // + eps
%16 = stablehlo.rsqrt %15 : tensor<32x1xf32>                                                 // 1/√(mean+eps)
%s52 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x64xf32>
%17 = stablehlo.multiply %0, %s52 : tensor<32x64xf32>                                        // normalised
```

## 10. Maintenance

When the emitter changes, please update this doc in the same commit. The tests that lock in the contract (and break loudly on drift) are:

- `EmitterTest.kt` — per-OpKind unit cases (~80 tests)
- `LlamaDecoderStablehloEmitTest.kt` — end-to-end Llama-shape lowering + the canonical-MLIR dump
- `LlamaDecoderCoarseningEmitDiagnosticTest.kt` (§0.4.310) — pins "0 custom_calls in production" — if you wire kernel descriptors into the GPU pipeline this test will fail and force a deliberate update of §1 / §6 here.
- `:stablehlo` round-trip suite via `stablehlo-translate --serialize --target=1.0.0` and `sdy-opt` — guards textual MLIR validity, not just structural shape.

If you add a new OpKind handler, add a row to §4 and (if non-trivial) a one-line idiom note in §8.
