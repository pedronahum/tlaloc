# layer3 — the artifact carries the device decision

**What it shows.** Compile an attention block for an H100 and for a TPU with
JAX/XLA and you get the same StableHLO twice: the fusion decision happens
*inside the runtime*, after the bytes have shipped. Tlaloc makes that decision
upstream, at compile time, so the artifact itself differs per device — the
bytes bound for a GB10 name `@flash_attn_v3`, the bytes bound for a TPU v6e
name `@tpu_pallas_flash_attention`, and the bytes bound for a generic CPU name
nothing at all because they are decomposed into portable primitives.

The run walks the whole Layer-3 pipeline, one printed section per stage:

1. **recognize** — find the `MATMUL → SOFTMAX → MATMUL` shape in DXIR.
2. **coarsen** — collapse it into one `OpKind.COARSENED` op that carries an
   analytical primal body *and* an analytical gradient body, so fusing costs
   you nothing in autodiff.
3. **lower and emit** — choose a kernel per target, emit StableHLO, and
   compare the artifacts pairwise to show they really do differ.
4. **KV-quant** — request an FP8 KV cache across a mixed fleet; some targets
   accept, others decline *with a reason*, and the compile does not fail.
5. **backend matrix** — serialize the cost-ordered per-target rows a cluster
   scheduler reads to place the job.

It is all pure compilation. No GPU, no TPU, no network: this runs on a laptop.

## Running it

This is a standalone Gradle project resolving Tlaloc from **mavenLocal**, so
publish first:

```bash
# from the repo root
./gradlew publishToMavenLocal
./gradlew -p examples/layer3 run
```

## Expected output

Verbatim, from this machine (GB10, JDK 25, Kotlin 2.3.20):

```
============================================================================
[1] + [2]  recognize the shape, then coarsen it into one op
============================================================================
before:   [MATMUL, SOFTMAX, MATMUL]
matched:  [FlashAttention]
after:    [COARSENED]

the COARSENED envelope:
  operands:            3  (Q, K, V)
  result type:         f32[8,4]
  primal_body:         flash_attention_primal  ops=[MATMUL, SOFTMAX, MATMUL]
  gradient_body:       flash_attention_grad  4 params (dO + 3 primal) -> 3 grads
  reads_primal:        [0, 1, 2]  (Q=0, K=1, V=2 — all three are needed by the backward pass)

============================================================================
[3]  same Kotlin source, one artifact per device
============================================================================

--- nvidia/gb10
    kernel chosen:   flash_attn_v3
    emitted:         %3 = stablehlo.custom_call @flash_attn_v3(%0, %1, %2) {backend_config = "{supported_kv_dtypes = [f32, bf16, fp8_e4m3, fp8_e5m2, int8]}", has_side_effect = false} : (tensor<8x4xf32>, tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>

--- nvidia/h100
    kernel chosen:   flash_attn_v3
    emitted:         %3 = stablehlo.custom_call @flash_attn_v3(%0, %1, %2) {backend_config = "{supported_kv_dtypes = [f32, bf16, fp8_e4m3, fp8_e5m2, int8]}", has_side_effect = false} : (tensor<8x4xf32>, tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>

--- google/tpu_v6e
    kernel chosen:   tpu_pallas_flash_attention
    emitted:         %3 = stablehlo.custom_call @tpu_pallas_flash_attention(%0, %1, %2) {backend_config = "{supported_kv_dtypes = [f32, bf16, fp8_e4m3, int8]}", has_side_effect = false} : (tensor<8x4xf32>, tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>

--- aws/trainium2
    kernel chosen:   nki_flash_attention
    emitted:         %3 = stablehlo.custom_call @nki_flash_attention(%0, %1, %2) {backend_config = "{supported_kv_dtypes = [f32, bf16, fp8_e4m3, int8]}", has_side_effect = false} : (tensor<8x4xf32>, tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>

--- tlaloc/cpu_generic
    kernel chosen:   (none — decomposed to portable primitives)
    emitted:         no custom_call; 15 lines of plain StableHLO

pairwise artifact comparison:
  nvidia/gb10            vs nvidia/h100            -> IDENTICAL
  nvidia/gb10            vs google/tpu_v6e         -> DIFFERENT
  nvidia/gb10            vs aws/trainium2          -> DIFFERENT
  nvidia/gb10            vs tlaloc/cpu_generic     -> DIFFERENT
  nvidia/h100            vs google/tpu_v6e         -> DIFFERENT
  nvidia/h100            vs aws/trainium2          -> DIFFERENT
  nvidia/h100            vs tlaloc/cpu_generic     -> DIFFERENT
  google/tpu_v6e         vs aws/trainium2          -> DIFFERENT
  google/tpu_v6e         vs tlaloc/cpu_generic     -> DIFFERENT
  aws/trainium2          vs tlaloc/cpu_generic     -> DIFFERENT

============================================================================
[4]  ask for an FP8 KV cache across a heterogeneous fleet
============================================================================
requested: fp8_e4m3 / PER_HEAD

  nvidia/h100             accepted — KV cache will be fp8_e4m3
  nvidia/a100             declined — kernel flash_attn_v2 on a100 doesn't support fp8_e4m3 KV cache (supports: f32,bf16,int8)
  aws/trainium2           accepted — KV cache will be fp8_e4m3
  google/tpu_v5e          declined — kernel tpu_pallas_flash_attention on tpu_v5e doesn't support fp8_e4m3 KV cache (supports: f32,bf16,int8)

the compile does not fail: a mixed fleet is a first-class deployment shape,
and the diagnostic stream is where the per-target answer surfaces.

============================================================================
[5]  the backend matrix, cost-ordered
============================================================================
target                  kernel                              kv_quant   cost (us)
----------------------------------------------------------------------------
nvidia/h100             flash_attn_v3                       fp8_e4m3   0.02
aws/trainium2           nki_flash_attention                 fp8_e4m3   0.02
google/tpu_v6e          tpu_pallas_flash_attention          fp8_e4m3   0.04
google/tpu_v4           tpu_pallas_flash_attention          -          0.05
nvidia/a100             flash_attn_v2                       -          0.05
google/tpu_v5e          tpu_pallas_flash_attention          -          0.08
nvidia/gb10             flash_attn_v3                       fp8_e4m3   0.24
tlaloc/cpu_generic      (decompose)                         -          0.44
amd/mi300x              flash_attn_amd                      -          -

one row, as it travels (JSON inside the workflow params):
  {"vendor":"nvidia","arch":"gb10","kernelName":"flash_attn_v3","kvQuantDtype":"fp8_e4m3","costMicroseconds":0.24005860805860804}

at launch time the runtime matches the cluster's (vendor, arch) against these
rows and turns the winner into K8s nodeSelector + accelerator + gpu fields.

layer3 OK
```

## What to read in the source

All of it is in [`src/main/kotlin/Main.kt`](src/main/kotlin/Main.kt):

| Look at | For |
|---|---|
| `attentionForward()` | The unfused DXIR the recognizers are handed |
| `recognizeAndCoarsen()` | `recognizeAll` → `coarsenRecognizedPatterns`, and the contents of the COARSENED envelope — including the gradient body |
| `perTargetArtifacts(...)` | `lowerKernelChoice(fn, target)` → `decomposeCoarsened` → `toStablehlo`, and the pairwise byte comparison |
| `kvQuantAcrossTargets(...)` | `applyKvQuantWithDiagnostics`: best-effort annotation plus a structured decline |
| `backendMatrix()` | `populateBackendMatrix` and the JSON row the cluster side consumes |

## Honest notes

- **GB10 and H100 produce IDENTICAL artifacts here**, because both resolve to
  the same `flash_attn_v3` kernel template in the registry. The per-target
  divergence is real, but it is per *kernel choice*, not per SKU — an earlier
  version of this example's doc comment implied a GB10-specific artifact, and
  the run above says otherwise.
- **The artifact carries the decision; the runtime does not yet dispatch it.**
  Registering `@flash_attn_v3` as a live PJRT custom-call symbol is separate
  work. Until then the executed path is the decomposed one — which is exactly
  why `decomposeCoarsened` is in the pipeline.
- **Only FlashAttention has a kernel template in v1.** The other recognized
  patterns (RmsNorm, RoPE, TransformerMLP, CrossEntropy, GroupedQueryAttention)
  coarsen but decompose at lowering time.
- `amd/mi300x` shows a kernel but no cost, because the cost model has no
  device descriptor for it; the scheduler sorts it last rather than guessing.
- The negative pin for all of this — JAX refusing to compile Tlaloc's
  custom-call MLIR, since `@flash_attn_v3` is not a symbol XLA ships — lives in
  the repo's benchmark suite as `JaxRejectsTlalocCustomCallMlirTest`.
