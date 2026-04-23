# DiffKtX — Design Spec & Implementation Roadmap

> **Code name:** `diffktx` (placeholder — pick a real name before v0.1).
> **Thesis:** Don't fight PyTorch on PyTorch's turf. Build a Kotlin-native, compiler-mediated, shape-typed differentiable-programming framework that **owns Android + KMP on-device training, JVM-native enterprise ML, and compile-checked tensor code** — while riding the MLIR/StableHLO accelerator wave instead of building our own kernels.
>
> This doc is structured so any section can be handed to Claude Code as a self-contained task. Each technical component has a **Goal**, **Design**, **API sketch**, and **Definition of Done**. Do not treat this as marketing copy. Treat it as a build order.

---

## 0. North Star & Non-Goals

### 0.1 What we are

- A Kotlin 2.x (K2) **differentiable-programming** framework with forward- and reverse-mode AD, composable higher-order derivatives, and user-defined differentiable types.
- A **compile-time** system: AD transformations (`grad`, `vmap`, `jit`, `shardMap`) are realized by a K2 compiler plugin, not a runtime tape, wherever possible.
- **Coarsening-optimized.** We inherit and extend the φ-calculus / SOI-based hybrid symbolic+algorithmic AD from Shen, Shivers, Dea et al. (OOPSLA 2021) — the technique that gave old DiffKt its 10×-on-scalars performance edge. See §11.
- **StableHLO + SDY (Shardy) first.** We emit StableHLO MLIR annotated with Shardy's SDY sharding dialect and hand it to PJRT-backed runtimes (XLA, IREE). We do not write CUDA kernels and we do not write collectives.
- **Shape-typed and sharding-typed.** Tensor shapes and shardings are part of the Kotlin type system, enforced at compile time, surfaced in the IDE. Mesh axis names live in the type system.
- **Sharding-propagation-native.** Users annotate a handful of tensors with partition specs; the compiler propagates shardings through the rest of the program and inserts collectives during lowering. DP, TP, EP, ZeRO, and context parallelism are all the same mechanism.
- **KMP-native.** One source set compiles to JVM (server), Android (ART + NNAPI), iOS (Kotlin/Native + CoreML), and WASM (browser demos). Sharding is server-only; mobile is always single-device.

### 0.2 What we are not

- Not a PyTorch clone. No `torch.nn` parity as a goal.
- Not a research playground for novel AD algorithms. Boring, correct, fast.
- Not a CUDA kernel project. We lower to StableHLO and let IREE/XLA codegen.
- Not a collective-communication library. Shardy + PJRT inserts and dispatches collectives for us.
- Not a novel sharding system. SDY is the representation. We translate our types to SDY and consume Shardy's propagation passes.
- Not Python-compatible at the API level. Interop is via ONNX / StableHLO artifacts, not FFI.

### 0.3 Wedge audiences (priority order)

1. **Android / KMP on-device training**: LoRA fine-tuning, federated learning, personalization, sensor-driven models.
2. **JVM enterprise ML**: Spring Boot / Kafka / Flink / Spark environments that want training + inference inside the JVM hot path.
3. **Shape-safety-obsessed teams**: fintech, aerospace, any shop where a silent shape bug is a P0.
4. **Differentiable simulation**: games, robotics, XR where Kotlin already has a foothold (libGDX, Korge, Android XR).

---

## 1. Architecture at a Glance

```
┌───────────────────────────────────────────────────────────────┐
│  User code (Kotlin)                                           │
│    val mesh = Mesh("data" to 8, "model" to 4)                 │
│    val step = jit { x, w -> (x matmul w).sum() }              │
│      .shard(mesh, x = P("data", null), w = P(null, "model"))  │
└───────────────────────────────────────────────────────────────┘
                │ (1) K2 compiler plugin
                ▼
┌───────────────────────────────────────────────────────────────┐
│  DiffKtX IR (SSA, typed, shape-indexed, sharding-annotated)   │
│    - AD transforms: grad / vmap / jit / shardMap / checkpoint │
│    - Coarsening pass (§11): SOI ID → φ-calculus symbolic      │
│      differentiation → simplify → splice as custom adjoint    │
│    - Shape inference & verification                           │
│    - Sharding propagation (delegated to Shardy SDY passes)    │
│    - Fusion, DCE, CSE, constant folding                       │
└───────────────────────────────────────────────────────────────┘
                │ (2) Lowering
                ▼
┌───────────────────────────────────────────────────────────────┐
│  StableHLO + SDY (textual or bytecode MLIR)                   │
│    SDY pipeline inserts collectives (all-reduce, all-gather,  │
│    reduce-scatter, all-to-all) during export passes           │
└───────────────────────────────────────────────────────────────┘
                │ (3) Dispatch
   ┌────────────┴──────────────────────┐
   ▼                                   ▼
 PJRT (multi-device)                 IREE (single-device)
 ├─ XLA CPU / CUDA / TPU             ├─ CPU (dev, CI)
 └─ pluggable PJRT plugins            ├─ Vulkan (Android)
                                      ├─ Metal (iOS / macOS)
                                      └─ WebGPU (browser demos)

 Escape hatch: libtorch eager (debugging only)
```

### 1.1 Module layout

```
diffktx/
├── core/                    # Pure Kotlin: DScalar, DTensor, Differentiable<T>, Mesh, Sharding
├── ir/                      # DiffKtX IR: nodes, passes, shape + sharding inference
├── compiler-plugin/         # K2 compiler plugin: grad/vmap/jit/shardMap lowering
├── ksp-plugin/              # KSP for user-defined Differentiable<T> codegen
├── shapetyping/             # Shape + sharding type system
├── coarsening/              # SOI identification, φ-calculus, symbolic CAS bindings
├── stablehlo/               # IR → StableHLO + SDY lowering
├── sharding/                # Shardy bindings: SDY attrs, propagation, export passes
├── runtime-pjrt/            # PJRT C API bindings (multi-device; XLA CPU/GPU/TPU)
├── runtime-iree/            # IREE JNI bindings (single-device; CPU/Vulkan/Metal/WebGPU)
├── runtime-torch/           # libtorch JNI bindings (debug escape hatch, eager mode)
├── android/                 # NNAPI fallback, Android-specific runtime
├── ios/                     # CoreML export + Kotlin/Native dispatch
├── data/                    # Coroutine-based DataLoader
├── examples/                # MNIST, LoRA-on-device, diff-sim, sharded-LLM
├── benchmarks/              # vs PyTorch, JAX, MLX — with coarsening on/off splits
└── ide-plugin/              # IntelliJ shape + sharding inspection, grad previews
```

### 1.2 Build system

- Gradle with `org.jetbrains.kotlin.multiplatform` 2.x.
- Version catalog (`libs.versions.toml`) as single source of truth.
- Convention plugins in `build-logic/` for shared module config.
- Native bindings via `cinterop` (KMP native) and JNI (JVM).

---

## 2. Technical Move 1 — StableHLO-first compilation pipeline

### 2.1 Goal

Never write a CUDA kernel. Never write a collective. Emit StableHLO annotated with SDY sharding attributes; inherit every accelerator PJRT/IREE supports and every partitioning scheme Shardy supports.

### 2.2 Design

Our IR is a small, typed SSA IR (see §3). Lowering to StableHLO is a mechanical pass per op. StableHLO has ~150 ops; we need ~40 for the transformer/CNN/MLP workload.

We emit **StableHLO bytecode** (preferred) or textual MLIR (debug). Sharding annotations are emitted as SDY dialect attributes (`sdy.mesh`, `sdy.sharding`, `sdy.sharding_constraint`, `sdy.manual_computation`) on the relevant SSA values. The resulting module is a single artifact that either:

1. Goes through Shardy's propagation + export passes → lowered StableHLO with explicit collectives → PJRT for multi-device execution; or
2. Has no shardings → plain StableHLO → IREE for single-device execution.

Emission uses the `stablehlo-translate` + `sdy_opt` tools out-of-process for v0, then moves to JNI-linked libStablehlo + libShardy for production.

Dispatch at runtime:

| Target               | Runtime       | Binding           | Notes                              |
|----------------------|---------------|-------------------|------------------------------------|
| JVM server (CPU)     | IREE or PJRT  | JNI               | IREE for dev/CI; PJRT for sharded  |
| JVM server (CUDA)    | PJRT (XLA)    | PJRT C API / JNI  | NVIDIA GPUs, multi-device          |
| JVM server (TPU)     | PJRT (XLA)    | PJRT C API / JNI  | Google Cloud TPU via same API      |
| Android              | IREE-Vulkan   | JNI (AAR)         | Single-device; NNAPI subgraph fallback |
| iOS                  | IREE-Metal    | cinterop          | Single-device; CoreML export path  |
| Browser              | IREE-WebGPU   | JS interop        | Demos only                         |
| Debug                | libtorch      | JNI               | Eager-mode escape hatch            |

### 2.3 Op coverage v0.1 (the "30 ops" that cover 95% of transformers)

**Elementwise:** `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`, `sqrt`, `rsqrt`, `neg`, `abs`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu`.
**Reduction:** `sum`, `mean`, `max`, `min`, `argmax`, `softmax`, `logsumexp`.
**Linear algebra:** `matmul`, `dot`, `conv2d`, `conv_transpose2d`.
**Shape:** `reshape`, `transpose`, `broadcast`, `concat`, `split`, `slice`, `gather`, `scatter`.
**Normalization:** `layernorm`, `rmsnorm`, `batchnorm`.
**Attention:** `scaled_dot_product_attention` (lowered to matmul + softmax + matmul; flash-attn comes from the runtime).
**Misc:** `embedding`, `cross_entropy`, `cast`.

All ops must have a forward lowering AND a VJP rule.

### 2.4 API sketch

```kotlin
// StableHLO emission — internal API
internal interface StablehloEmitter {
    fun emit(module: DxirModule): StablehloBytecode
}

// Runtime dispatch
class Executable internal constructor(private val handle: Long) {
    fun invoke(inputs: List<DTensor>): List<DTensor>
    fun close()
}

object Runtime {
    fun compile(module: DxirModule, target: Target = Target.autoDetect()): Executable
}

sealed class Target {
    object CpuIree : Target()
    object CudaXla : Target()
    object VulkanIree : Target()    // Android
    object MetalIree : Target()     // iOS
    data class Custom(val name: String) : Target()

    companion object { fun autoDetect(): Target = /* ... */ }
}
```

### 2.5 Definition of Done

- [ ] All 40 ops in §2.3 have a lowering pass and pass property-based equivalence tests against PyTorch (CPU, fp32, tolerance 1e-5).
- [ ] `Runtime.compile(module, CpuIree).invoke(inputs)` runs a 2-layer MLP end to end.
- [ ] GitHub Actions matrix green on Linux x86_64, macOS arm64.
- [ ] Benchmark harness runs a BERT-base forward pass within 2x of PyTorch CPU eager.

---

## 3. Technical Move 2 — K2 Compiler Plugin for AD Transformations

### 3.1 Goal

`grad`, `vmap`, `jit`, `shardMap`, `checkpoint` are realized at compile time as IR transformations, not runtime wrappers. This is what gives us JAX-like composability without JAX's tracing tax.

### 3.2 Why a compiler plugin, not a runtime tape

- Runtime tapes (PyTorch autograd) box scalars, allocate thunks, and resist fusion.
- Runtime tracing (JAX) requires the whole function to be traceable with symbolic values — breaks with Python-side data-dependent control flow.
- K2 plugins see real Kotlin ASTs with types. We can transform `fun f(x: Matrix<B,D>): Scalar` into `fun f_grad(x: Matrix<B,D>): Matrix<B,D>` at `FIR → IR` lowering, preserving control flow natively.

### 3.3 DiffKtX IR (dxir)

Typed SSA IR. Every value has a `DxirType` which carries element dtype + shape. Values also carry an optional `DxirSharding` attribute that lowers directly to SDY.

```kotlin
sealed class DxirNode {
    abstract val type: DxirType
    abstract val sharding: DxirSharding?   // null = unconstrained / fully open
}

data class DxirParam(val name: String, override val type: DxirType, override val sharding: DxirSharding? = null) : DxirNode()
data class DxirOp(
    val op: OpKind,                  // ADD, MATMUL, CONV2D, ...
    val operands: List<DxirNode>,
    val attrs: Map<String, Any>,
    override val type: DxirType,
    override val sharding: DxirSharding? = null
) : DxirNode()
data class DxirCall(
    val callee: DxirFunction,
    val args: List<DxirNode>,
    override val type: DxirType,
    override val sharding: DxirSharding? = null
) : DxirNode()
// control flow: If, While, Scan, Scope (for checkpointing), ShardMap

data class DxirFunction(
    val name: String,
    val params: List<DxirParam>,
    val body: List<DxirNode>,
    val returns: List<DxirNode>,
    val meshes: List<DxirMesh> = emptyList()   // logical meshes in scope for this function
)

data class DxirModule(val functions: List<DxirFunction>, val meshes: List<DxirMesh>)

// Sharding (mirrors SDY)
data class DxirMesh(val name: String, val axes: List<DxirMeshAxis>)
data class DxirMeshAxis(val name: String, val size: Int)
data class DxirSharding(
    val meshName: String,
    val dimShardings: List<DxirDimSharding>,
    val replicated: List<DxirAxisRef> = emptyList()
)
data class DxirDimSharding(val axes: List<DxirAxisRef>, val closed: Boolean, val priority: Int? = null)
sealed class DxirAxisRef { data class Full(val name: String) : DxirAxisRef(); data class Sub(val name: String, val preSize: Int, val size: Int) : DxirAxisRef() }
```

### 3.4 Transforms

Each transform is `DxirFunction → DxirFunction`:

- **`grad(f)`**: reverse-mode AD. Classic adjoint code generation. Output function has same signature but returns gradients w.r.t. params.
- **`vmap(f, axis)`**: batches `f` over one axis. Lifts every op by adding a leading dim; uses op-specific batching rules.
- **`jit(f)`**: caches compilation artifact keyed by input shape + sharding signatures. No-op at IR level, but marks the function for eager lowering.
- **`shardMap(f, mesh, inSpecs, outSpecs)`**: explicit per-device partitioning (SPMD). Within the body, the user writes code in terms of the per-device shard; the transform emits `sdy.manual_computation`. Contrast with `jit` + input shardings, which uses sharding *propagation*.
- **`shard(t, spec)`**: attaches a sharding constraint to a tensor. Lowers to `sdy.sharding_constraint`.
- **`checkpoint(f)`**: marks a subgraph for activation recomputation during backward.

**Sharding propagation** is not a user transform — it's a mandatory IR pass run between AD transforms and StableHLO emission. We delegate to Shardy's `sdy-propagation-pipeline` and only re-emit the result back into dxir for verification.

Composability rules:
- `grad(vmap(f)) == vmap(grad(f))` up to axis bookkeeping.
- `grad(jit(f)) == jit(grad(f))` is the expected form.
- `grad(shardMap(f)) == shardMap(grad(f))` modulo collective placement — the gradient of a sharded computation has dual collectives (e.g. all-reduce in forward ↔ identity in backward; identity in forward ↔ all-reduce in backward). These rules are implemented per-op in the `GradOfShard` pass.

All combinations are regression-tested.

### 3.5 API sketch (user-facing)

```kotlin
// Top-level transforms. Erased to plugin-generated functions at compile time.
inline fun <P, R> grad(noinline f: (P) -> R): (P) -> Gradient<P>
inline fun <P, R> valueAndGrad(noinline f: (P) -> R): (P) -> Pair<R, Gradient<P>>
inline fun <P, R> vmap(axis: Int = 0, noinline f: (P) -> R): (P) -> R
inline fun <P, R> jit(noinline f: (P) -> R): (P) -> R

// Example
val loss: (Matrix<B, D>, Matrix<D, C>) -> Scalar = { x, w -> (x matmul w).softmax().crossEntropy(y) }
val dLoss = grad(loss)  // plugin generates typed gradient
```

### 3.6 Plugin internals

- Frontend: K2 FIR extension registers `grad`, `vmap`, etc. as intrinsics.
- FIR → dxir: lowering pass converts the referenced lambda body into dxir. Non-differentiable control flow falls back to a runtime-tape path with a compile-time warning.
- dxir transforms applied.
- dxir → Kotlin IR codegen: emit calls to `Runtime.compile(...)` for `jit`'d functions; emit direct `Executable.invoke` calls otherwise.

### 3.7 Definition of Done

- [ ] Plugin compiles and links against Kotlin 2.2.x+.
- [ ] `grad`, `vmap`, `jit` implemented for all 40 core ops.
- [ ] `grad(grad(f))` for a scalar-valued `f: Scalar -> Scalar` produces correct second derivatives.
- [ ] `vmap(grad(f)) == grad(vmap(f))` on property tests (with axis bookkeeping).
- [ ] Compile error (not runtime error) when a user calls `grad` on a non-differentiable function.

---

## 4. Technical Move 3 — ShapeTyping 2.0

### 4.1 Goal

Tensor shapes as types. Compile-time errors for mismatches. Dependent-types-lite — enough to catch real bugs, not enough to require a PhD.

### 4.2 Design

Shapes are a list of **shape atoms**: type-level integer literals (`Lit<768>`), named symbols (`B`, `S`, `D`), or arithmetic combinations (`Mul<B, S>`).

```kotlin
// Shape atoms
sealed interface ShapeAtom
class Lit<N : Int> : ShapeAtom
class Sym(val name: String) : ShapeAtom
class Mul<A : ShapeAtom, B : ShapeAtom> : ShapeAtom
class Add<A : ShapeAtom, B : ShapeAtom> : ShapeAtom

// A shape is a heterogeneous tuple of atoms. Represented as nested generics.
interface Shape
object ScalarShape : Shape
class Rank1<A0 : ShapeAtom> : Shape
class Rank2<A0 : ShapeAtom, A1 : ShapeAtom> : Shape
class Rank3<A0 : ShapeAtom, A1 : ShapeAtom, A2 : ShapeAtom> : Shape
class Rank4<A0 : ShapeAtom, A1 : ShapeAtom, A2 : ShapeAtom, A3 : ShapeAtom> : Shape
// ... up to Rank8; beyond that use DynShape

class DTensor<S : Shape, T : DType>(internal val handle: TensorHandle)

typealias Matrix<R, C> = DTensor<Rank2<R, C>, F32>
typealias Vector<L> = DTensor<Rank1<L>, F32>
typealias Scalar = DTensor<ScalarShape, F32>
```

### 4.3 Shape-checked ops

```kotlin
infix fun <R, K, C, T : DType> DTensor<Rank2<R, K>, T>.matmul(
    other: DTensor<Rank2<K, C>, T>
): DTensor<Rank2<R, C>, T>

fun <B, H, W, Cin, Cout, Kh, Kw, T : DType> DTensor<Rank4<B, Cin, H, W>, T>.conv2d(
    kernel: DTensor<Rank4<Cout, Cin, Kh, Kw>, T>,
    stride: Pair<Int, Int> = 1 to 1
): DTensor<Rank4<B, Cout, /*Hout*/ /*computed by plugin*/, /*Wout*/>, T>
```

Conv2d output shape arithmetic can't be expressed in plain Kotlin generics. The K2 plugin computes output-shape types from attribute values during type checking. This is the ShapeTyping plugin's core value add.

### 4.4 Relationship to sharding types

Shape atoms that appear in shardings must also appear in the tensor's shape type. The compiler plugin verifies this.

```kotlin
// A mesh declared at the type level via nominal singletons (see §9)
object TpMesh : Mesh by mesh("data" to 8, "model" to 4)

// Shape-typed tensor
val x: DTensor<Rank2<B, D>, F32> = ...

// Sharding-typed constraint — compiler checks that "data" and "model" exist in TpMesh,
// that the dim-sharding list has length 2 (matching Rank2), and that dim sizes are divisible.
val xShard = x.shard(TpMesh) { B on "data"; D on "model" }
// xShard: DTensor<Rank2<B, D>, F32>, with DxirSharding attached in the IR
```

The **type** of `x` does not change when you add a sharding constraint — sharding is a compile-time-checked attribute, not a type index. This is a deliberate choice: encoding shardings in the generic type would double the type parameter count and make library APIs unreadable. Instead, the K2 plugin tracks shardings in a side-table keyed by IR value identity and emits a compile error on mismatch.

### 4.5 Dynamic escape hatch

Not everything is statically known. For dynamic shapes:

```kotlin
class DynShape(val dims: IntArray) : Shape
fun <S : Shape> DTensor<DynShape, *>.cast(shape: S): DTensor<S, *>  // runtime check
```

Make the static path the ergonomic default; dynamic is explicit and verbose. This is the opposite of PyTorch.

### 4.6 IDE integration

- IntelliJ plugin that shows inferred shapes inline (like Rust type hints).
- Hover on a tensor variable → see `Matrix<B=32, D=768>`.
- Shape errors underlined in the editor before compile.

### 4.7 Definition of Done

- [ ] All 40 core ops have shape-typed signatures.
- [ ] Plugin resolves attribute-dependent shapes (conv, pooling, reshape).
- [ ] Plugin verifies sharding consistency (mesh axes exist, dim count matches, divisibility).
- [ ] IDE plugin shows shape + sharding hints in IntelliJ 2025.x+.
- [ ] Shape-check test suite: 50 positive (compiles) + 50 negative (doesn't compile) cases.
- [ ] Sharding-check test suite: 30 positive + 30 negative cases.

---

## 5. Technical Move 4 — Value Classes for Zero-Cost Tensors

### 5.1 Goal

Eliminate the JVM numerics tax: no boxing, no wrapper allocation per op. This is what killed the old DiffKt on scalars.

### 5.2 Design

`DScalar` is a `@JvmInline value class` wrapping a `Long` handle (for tensor runtime-owned scalars) or a raw `Float`/`Double` (for literal scalars). Dispatch is virtual via a sealed interface.

```kotlin
sealed interface DScalar {
    fun toFloat(): Float
    fun toDouble(): Double
}

@JvmInline
value class FloatScalar(val v: Float) : DScalar {
    override fun toFloat() = v
    override fun toDouble() = v.toDouble()
}

@JvmInline
value class TensorScalar(val handle: Long) : DScalar {
    override fun toFloat() = TensorRuntime.readScalarF(handle)
    override fun toDouble() = TensorRuntime.readScalarD(handle)
}
```

Binary ops are extension functions specialized by the compiler plugin — `FloatScalar + FloatScalar` compiles to a primitive add, no dispatch.

### 5.3 DTensor design

```kotlin
class DTensor<S : Shape, T : DType> internal constructor(
    internal val handle: Long,       // pointer into runtime-managed memory
    internal val shape: IntArray,    // erased runtime shape
    internal val dtype: T
) : AutoCloseable {
    override fun close() = TensorRuntime.release(handle)
}
```

Ownership is explicit — `AutoCloseable` + structured concurrency. No finalizer reliance (JVM finalizers are cursed for native memory).

### 5.4 Structured tensor scopes

```kotlin
inline fun <R> tensorScope(block: TensorScope.() -> R): R {
    val scope = TensorScope()
    try { return scope.block() } finally { scope.closeAll() }
}

class TensorScope {
    fun <S : Shape, T : DType> track(t: DTensor<S, T>): DTensor<S, T> { /* ... */ }
    internal fun closeAll() { /* release all tracked tensors */ }
}

// Usage
tensorScope {
    val x = track(randn<Rank2<B, D>>())
    val y = track(x matmul w)
    // all released at scope exit
}
```

### 5.5 Definition of Done

- [ ] `DScalar` ops inline to primitive arithmetic (verified by bytecode inspection).
- [ ] `tensorScope` + `track` prevent leaks under 10M-iteration stress test.
- [ ] No `System.gc()`-triggered release paths.

---

## 6. Technical Move 5 — Context Receivers for Functional AD Contexts

### 6.1 Goal

JAX-style transformation contexts, Kotlin-idiomatic. `grad`, `vmap`, `jit` nest naturally without the lambda-in-lambda-in-lambda hell.

### 6.2 Design

Uses Kotlin 2.x **context parameters** (stable as of 2.2; use `context(_:)` syntax).

```kotlin
context(trace: TraceContext)
fun <S : Shape, T : DType> DTensor<S, T>.matmul(other: DTensor<S, T>): DTensor<S, T> {
    return trace.record(OpKind.MATMUL, this, other)
}

// Transform invocation
val f = grad {
    context(trace) ->
    val y = x matmul w
    y.sum()
}
```

In practice most users won't write `context(trace)` explicitly — the plugin rewrites lambdas passed to `grad`/`vmap`/`jit` to carry the right context automatically.

### 6.3 Contexts in the API

| Context         | Introduced by       | Purpose                                   |
|-----------------|---------------------|-------------------------------------------|
| `TraceContext`  | `grad`, `jit`       | Records ops for transformation            |
| `VmapContext`   | `vmap`              | Tracks batch axis for each tensor          |
| `MeshContext`   | `with(mesh) { ... }` | Active logical mesh for sharding specs   |
| `ShardContext`  | `shardMap`          | Per-device scope; tensors are local shards |
| `DeviceContext` | `device(d) { ... }` | Default device placement                  |
| `DtypeContext`  | `withDtype(f16) { ... }` | Mixed-precision scope                |
| `RngContext`    | top-level or `seed` | Reproducible RNG                          |

### 6.4 Definition of Done

- [ ] Nested transforms work: `grad(vmap(jit(f)))` produces correct output.
- [ ] Plugin automatically injects context for transform-wrapped lambdas.
- [ ] No user needs to spell out `context(trace)` in tutorial-level code.

---

## 7. Technical Move 6 — Coroutines for Data & Distributed

### 7.1 Goal

A data pipeline that doesn't feel like Python's multiprocessing fever dream. Distributed training that composes with `suspend`.

### 7.2 DataLoader design

```kotlin
interface Dataset<T> {
    val size: Int
    suspend fun get(index: Int): T
}

class DataLoader<T, Batched>(
    private val dataset: Dataset<T>,
    private val batchSize: Int,
    private val shuffle: Boolean = true,
    private val numWorkers: Int = 4,
    private val collate: (List<T>) -> Batched
) {
    fun stream(): Flow<Batched> = flow {
        // produce indices, fan out to workers via Channel, collate in order
    }.buffer(capacity = 2 * numWorkers).flowOn(Dispatchers.IO)
}
```

Workers are coroutines on `Dispatchers.IO`. Backpressure via `Channel`. Prefetching via `buffer`. This composes with the rest of the Kotlin ecosystem — Ktor, gRPC-Kotlin, kotlinx.serialization — out of the box.

### 7.3 Distributed training

**We do not write a distributed training system.** Shardy + PJRT *is* our distributed training system; see §9. What this module owns is:

- **Host-side orchestration**: process launch across nodes, rendezvous, world-size discovery.
- **Input data sharding**: splitting/replicating the data-loading pipeline to match the batch-axis sharding.
- **Checkpoint I/O**: asynchronous, sharding-aware reads/writes via coroutines. Compatible with the Tensorstore format for interop with JAX/PyTorch checkpoints.
- **Health & telemetry**: per-worker liveness, step-time metrics, straggler detection.

Collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`) are **not** exposed as a user API. They appear only as the result of Shardy's lowering passes.

```kotlin
// What users actually write — collectives are never mentioned
val mesh = Mesh("data" to 8)
val step = jit { batch, params ->
    val logits = model(batch, params)
    loss(logits, batch.labels)
}
val launcher = DistributedLauncher(mesh, backend = Backend.Pjrt)
launcher.run { world ->
    for (batch in dataLoader.stream().sharded(world, "data")) {
        val (lossVal, grads) = valueAndGrad(step)(batch, params)
        params = optimizer.update(params, grads)
    }
}
```

### 7.4 Definition of Done

- [ ] `DataLoader` hits 90%+ GPU utilization on a BERT-base ingest benchmark.
- [ ] `DataLoader.sharded(world, axis)` produces per-worker streams that match the model's input sharding.
- [ ] Checkpoint reads/writes use async I/O and round-trip sharded tensors correctly under rank changes (e.g. save on 8 workers, reload on 4).
- [ ] Coroutine cancellation releases all data-worker resources cleanly.

---

## 8. Technical Move 7 — Binding Strategy (No CUDA Kernels, Ever)

### 8.1 Goal

Cover the 95th-percentile op set without owning a single kernel. Kernels are where projects die.

### 8.2 Backends and what they give us

| Backend       | Gives us                                  | Binding    | License OK? |
|---------------|-------------------------------------------|------------|-------------|
| **IREE**      | CPU / Vulkan / Metal / CUDA / WebGPU via MLIR | JNI + cinterop | Apache 2.0 |
| **XLA (OpenXLA)** | Server GPU + TPU, top-tier perf         | JNI via PJRT C API | Apache 2.0 |
| **libtorch**  | Eager escape hatch, dev ergonomics        | JNI        | BSD-3      |
| **cuDNN/cuBLAS** | Selective hand-tuned paths if perf demands | JNI     | NVIDIA EULA — careful |
| **NNAPI**     | Android accelerator fallback              | JNI (NDK)  | Apache 2.0 |
| **CoreML**    | iOS accelerator fallback                  | cinterop   | — (runtime) |

v0.1: IREE + libtorch only. Add XLA in v0.3. Mobile backends in v0.5.

### 8.3 JNI conventions

- All JNI signatures in a single `NativeBindings` object. Use `@CriticalNative` annotations where eligible (JDK 22+).
- All native handles are `Long` (address). Ownership tracked by Kotlin-side `AutoCloseable`.
- Errors are exceptions: C++ throws → JNI catches → Kotlin exception with stack trace.
- ABI versioning: a `versionCheck` call at load time asserts compatibility.

### 8.4 Definition of Done

- [ ] IREE backend links and runs a compiled StableHLO module on Linux and macOS.
- [ ] libtorch backend runs the same op set in eager mode within 1.5x of native libtorch from Python.
- [ ] No C++ kernel code authored in this repo beyond JNI glue.

---

## 9. Technical Move 8 — Sharding & Distribution (Shardy/SDY-Native)

### 9.1 Goal

Make DP, TP, EP, ZeRO, and context parallelism **all the same mechanism**, expressed as per-tensor shardings on a named multi-axis mesh. Delegate all propagation and collective insertion to Shardy. The user never writes `all_reduce`.

### 9.2 Why delegate, not reimplement

Shardy represents the merged wisdom of the GSPMD and PartIR teams. It has: a precise representation (mesh + per-dim specs with open/closed, sub-axes, replicated axes, priorities), a propagation algorithm, and a lowering pipeline that inserts collectives. Reimplementing any of that would be a multi-year sinkhole that still doesn't cover TPU. We bind, we don't rebuild.

### 9.3 Mesh & partition-spec API

```kotlin
// Meshes are named, typed values. The K2 plugin lifts the axis names into a
// nominal type so specs can reference them safely.
class Mesh(val name: String, val axes: List<MeshAxis>) {
    companion object {
        fun of(vararg axes: Pair<String, Int>, name: String = "default"): Mesh
    }
}

// A MeshAxis is a named axis of a given size.
data class MeshAxis(val name: String, val size: Int)

// A partition spec says, per tensor dimension, which mesh axis (or axes, or sub-axis,
// or nothing) shards that dim. `null` = replicated on that dim. `open(...)` = open
// for propagation to refine.
sealed class Spec {
    object Replicated : Spec()
    data class On(val axes: List<String>, val open: Boolean = false, val priority: Int? = null) : Spec()
    data class SubOn(val axis: String, val preSize: Int, val size: Int) : Spec()
}

typealias PartitionSpec = List<Spec>   // one entry per tensor dimension
```

### 9.4 Usage patterns

**Pattern A — propagation (recommended default):** annotate a few anchor tensors, let the compiler figure out the rest.

```kotlin
val mesh = Mesh.of("data" to 8, "model" to 4)

val step = jit { x: DTensor<Rank2<B, D>, F32>, w: DTensor<Rank2<D, C>, F32> ->
    val y = x matmul w
    y.sum()
}.withMesh(mesh)
 .withInputSpecs(
     Spec.On(listOf("data")) to Spec.Replicated,              // x: B sharded over data, D replicated
     Spec.Replicated to Spec.On(listOf("model"))              // w: D replicated, C sharded over model
 )
```

The compiler runs Shardy's propagation pass. Every intermediate gets a sharding; collectives appear where they must.

**Pattern B — `shardMap` (explicit, SPMD):** when you want to write per-device code directly.

```kotlin
val sharded = shardMap(mesh, inSpecs = ..., outSpecs = ...) { xLocal, wLocal ->
    // xLocal and wLocal are the per-device shards
    xLocal matmul wLocal   // no automatic collectives here
}
```

Lowers to `sdy.manual_computation`. Useful for custom collectives (rare) and for library authors who want exact control.

**Pattern C — `shard` constraint:** attach a sharding at a specific point in a larger `jit`'d computation.

```kotlin
jit { x, params ->
    val h1 = (x matmul params.w1).relu().shard(Spec.On(listOf("data")) to Spec.On(listOf("model")))
    h1 matmul params.w2
}
```

Lowers to `sdy.sharding_constraint`. Used to pin intermediate shardings when propagation would pick something suboptimal.

### 9.5 Standard recipes as one-liners

```kotlin
// Data parallelism
step.dataParallel(mesh = Mesh.of("data" to 8), batchAxis = 0)

// FSDP / ZeRO-3 (shard params across data axis)
step.fsdp(mesh = Mesh.of("data" to 8))

// Tensor parallelism (Megatron-style, shard model axis)
step.tensorParallel(mesh = Mesh.of("model" to 4))

// Hybrid (DP + TP)
step.hybrid(mesh = Mesh.of("data" to 4, "model" to 2))
```

These are thin wrappers that produce the right partition specs for the module's parameters and activations. Implemented as library code, not compiler magic.

### 9.6 Sharding propagation pipeline

dxir → StableHLO+SDY lowering emits a module with user-specified sharding attrs. The pipeline then runs:

1. `sdy-round-trip-import`: validate the module.
2. `sdy-propagation-pipeline`: infer shardings for unannotated values using Shardy's op-by-op rules and priorities.
3. Our `diffktx-grad-sharding-pass`: verify that the gradient function's shardings are duals of the forward's (catches common bugs).
4. `sdy-export-pipeline`: convert SDY ops to concrete collectives in StableHLO.
5. Hand to PJRT for compilation + execution.

Steps 2 and 4 are invoked via JNI into libShardy; we own steps 1 and 3.

### 9.7 Pipeline parallelism (MPMD)

Shardy has an MPMD dialect for multi-program-multi-data (pipeline parallelism, heterogeneous device placement). We bind it behind a `pipeline { ... }` DSL:

```kotlin
val pipelined = pipeline(stages = 4, microbatch = 16) {
    stage(0) { x -> layer0(x) }
    stage(1) { h -> layer1(h) }
    stage(2) { h -> layer2(h) }
    stage(3) { h -> layer3(h) }
}
```

This is v0.5+ scope. Not Month 3.

### 9.8 Mobile reminder

Sharding is irrelevant on phones. The `mobile` dispatch path runs single-device IREE with *zero* SDY involvement. Any code using `with(mesh) { ... }` on Android is a compile error.

### 9.9 Definition of Done

- [ ] `Mesh`, `Spec`, and the `withMesh`/`withInputSpecs`/`shard`/`shardMap` API land in core.
- [ ] IR carries SDY-equivalent sharding attributes; round-trips through `sdy-opt` lossless.
- [ ] Propagation pipeline integrated; end-to-end test: 2D mesh (data=2, model=2) BERT-base one step on CPU PJRT, gradients match single-device reference.
- [ ] Real GPU test: 8-GPU FSDP training of a 1B-param transformer converges matching JAX baseline within 2% perplexity over 1000 steps.
- [ ] `dataParallel`, `fsdp`, `tensorParallel`, `hybrid` one-liners work on the reference model.
- [ ] Compile error if `shard` spec references a mesh axis that doesn't exist or a dim count that doesn't match tensor rank.
- [ ] Compile error if mesh-bound code is reachable from an Android or iOS source set.

---

## 10. Technical Move 9 — Mobile & KMP Story (The Wedge)

### 10.1 Goal

If this project has a reason to exist, it's here. Own on-device training and fine-tuning for Kotlin Multiplatform.

### 10.2 Android

- AAR with bundled IREE-Vulkan runtime (~8MB compressed).
- NNAPI fallback for supported subgraphs (matmul, conv, quantized ops).
- Integration with Android Studio profiler (custom trace category).
- Jetpack Compose UI samples: live LoRA fine-tuning of a small LM on-device.
- Battery / thermal budget APIs: `BudgetAwareTrainer` that respects `ThermalManager.getCurrentThermalStatus()`.

### 10.3 iOS

- Kotlin/Native framework via cinterop to IREE-Metal.
- CoreML export path for inference-only artifacts (StableHLO → MIL).
- SwiftUI sample app.

### 10.4 Killer demos (ship these with v0.5)

1. **"Whispr"**: on-device Whisper-tiny fine-tuning on the user's voice, 60 seconds of audio → personalized ASR. Runs on Pixel 8+, iPhone 14+.
2. **"LoRAsmith"**: fine-tune a 1B language model LoRA on-device, <30 min on a flagship phone.
3. **"Flappy Grad"**: differentiable Flappy Bird clone that trains its own AI in-game via diff-sim.

### 10.5 Definition of Done

- [ ] AAR published to Maven Central, `<20MB` compressed.
- [ ] One KMP sample app runs on both Android and iOS from a single shared module.
- [ ] One of the three killer demos published as a public app.

---

## 11. Technical Move 10 — Coarsening Optimization (φ-calculus)

### 11.1 Goal

Recover the 10×–100× scalar-performance advantage old DiffKt had. Implement the coarsening optimization from Shen, Zhang, Dea, Andow, Arroyo-Fang, Gafter, George, Grueter, Meijer, Shivers, Stumpos, Tempest, Warden, Yang — *"Coarsening Optimization for Differentiable Programming"*, OOPSLA 2021. This is not a novel research contribution for us; it's a credentialed technique we inherit, port, and industrialize.

### 11.2 Why this matters

Operation-by-operation AD (PyTorch autograd, JAX tracing, our MVP runtime tape) creates one tape entry, one closure, and at least one intermediate allocation per primitive op. For dense deep learning this is dwarfed by cuDNN kernel time and nobody cares. For **scalar-intensive workloads** — physics simulation, probabilistic inference, meta-learning, user-defined types over small vectors — it dominates.

The paper measured:

| Benchmark | Domain | Differentiation speedup | End-to-end speedup |
|---|---|---:|---:|
| BGDHyperOpt | Meta-learning | 23×–27× | 8.0×–8.6× |
| HookeanSpring | Physics | 2.5×–6.6× | 4.1×–11.0× |
| HMC | Probabilistic programming | 2.5×–4.4× | 2.3×–3.6× |
| Brachist. | Math physics | 1.0×–1.4× | 1.8×–2.5× |
| CartPole | Deep RL (scalar env) | 1.1× | 1.1× |
| QWOP | Game AI | 1.4×–1.5× | 1.4×–1.6× |

Ported to other frameworks on BGDHyperOpt, the measured speedups were **87×–335× (JAX), 91×–150× (Zygote), 66×–96× (Adept C++)**. This is a general AD optimization, not a DiffKt quirk. It has not been widely adopted in mainstream frameworks since 2021, and nothing in torch.compile or jax.jit replaces it.

Our wedge audiences overlap exactly with the wins: differentiable simulation (games/robotics/XR), probabilistic programming (Bean Machine's old niche), meta-learning, JVM enterprise ML doing scalar business logic with gradients. This is the single highest-leverage differentiator left in the design.

### 11.3 What coarsening does

Given a primal computation `P` from active inputs to active outputs:

1. Identify contiguous **Segments of Interest (SOIs)** in the SSA IR — subgraphs amenable to symbolic treatment, sized to balance "expression swell" against "reuse deprivation" (§11.6).
2. Lift each SOI to a symbolic representation using **φ-calculus** (§11.5) so branches and loops become closed-form expressions.
3. Symbolically differentiate the SOI with a computer algebra system (CAS) — applying cancellation, combining like terms, factoring out loop invariants.
4. Generate optimized Kotlin code for both the simplified primal and the gradient.
5. Splice the generated gradient back into the program as a **custom adjoint** (`setIntermediateAdjoints`-style hook) so surrounding AD continues to compose. If the entire primal is an SOI and only gradients are consumed, the primal is removed entirely — a class of optimization classical AD cannot perform.

Four benefits, all measured in the paper: (a) fewer tape allocations and closures, (b) symbolic simplifications that reduce op count, (c) unboxed scalars in generated code (no `Tensor` wrapper overhead), (d) dead-primal elimination.

### 11.4 Fit with DiffKtX's architecture

Coarsening is **orthogonal and complementary** to everything already specced:

| Existing layer | Interaction with coarsening |
|---|---|
| K2 compiler plugin (§3) | Coarsening lives inside the plugin as an IR pass, run *before* the `grad` transform and StableHLO lowering |
| dxir (§3.3) | dxir is already SSA with typed shapes — the paper's prerequisite. No schema changes needed |
| Value classes (§5) | Generated coarsened code emits unboxed `Float`/`Double` ops directly; zero-cost tensor wrappers get out of the way |
| StableHLO + SDY (§2, §9) | Coarsened dxir lowers to StableHLO normally. Sharding attrs are preserved across the transform |
| Mobile (§10) | **Critical for the on-device story.** Coarsening's biggest wins are on scalar-heavy code, which is exactly what runs on phones (fine-tuning loops, sensor-driven models). |

Because K2's backend IR is already SSA, we don't have to build the prerequisite infrastructure. The hard parts are the CAS and the cost model.

### 11.5 φ-calculus (what we inherit)

SSA's φ-function picks the right name at a merge point. φ-calculus extends φ into a reasoning calculus with **five fundamental formulae**:

- **F1 (Identity):** `φ(a, a, …, a) = a`.
- **F2 (Distributive):** `f(φ(a₁, …, aₙ)) = φ(f(a₁), …, f(aₙ))`. Lets differentiation cross branches.
- **F3 (Commutative):** `φ(a, b) = φ̄(b, a)` where `φ̄` is the complement.
- **F4 (Loop-entry):** The value entering a loop-entry φ in iteration `i` is the value flowing back from iteration `i−1`. Turns loop-entry φs into recurrences.
- **F5 (Loop-exit):** If the loop-exit φ has homogeneous back-edge arguments whose initial value equals the pre-loop value, the loop-exit φ equals the terminal back-edge value. Removes loop-exit φs.

Plus **nine corollaries** (C1–C9) that compose these with arithmetic operations to collapse common loop patterns into closed forms. Example — `C5`: a loop `𝔏ⁿ d = f(φₗ(p, d))` has closed form `d_exit = fⁿ(p)` (`n`-fold function application).

Practical consequence: a `while` loop with a `break` inside an `if`/`else`, nested in a `for` loop with an array accumulator, can be reduced to a closed-form expression in the loop's trip count and the inputs. The paper walks a non-trivial batch-gradient-descent hyperparameter-optimization kernel through this reduction in a single page (paper Fig. 6).

We reimplement φ-calculus as a pass over dxir. No user-facing API. The five formulae and nine corollaries become Kotlin data transformations on `DxirNode` subtrees. Total implementation scope is measured in hundreds of lines — this is surprisingly compact once SSA is in place.

### 11.6 SOI identification

Making SOIs too large causes "expression swell" — symbolic derivatives can grow quadratically. Making them too small loses the cancellation and reuse opportunities that motivate coarsening. The paper's answer is **reuse-aware SOI identification**:

1. For each active sink variable `s` that outlives the function, walk the def-use chain.
2. Build a def-use region tree (regions at loop boundaries; loop-exit φs bundled with their loop).
3. Traverse bottom-up. For each node: derive its symbolic expression via φ-calculus. If the expression exceeds a size limit `L`, split the node on the variable with the most reuses elsewhere. If a node's children are oversized, keep children as separate SOIs and merge small ones adjacently.
4. `L` is tunable per machine / CAS.

This is a ~40-line algorithm in the paper (Fig. 7). Port it 1:1. The cost model can be refined later; start with the paper's heuristic.

### 11.7 The symbolic engine choice

**This is the main open decision.** The paper used an extended SymPy, which they noted took up to a minute of compile time. That's unacceptable for an edit–compile–run loop.

Options:

| Option | Pros | Cons |
|---|---|---|
| **Symja** (JVM CAS, Apache 2.0) | Native JVM; no subprocess; Mathematica-like API; actively maintained | Heavier than we need; learning curve on the differentiation API |
| **Build minimal Kotlin CAS** | Exactly scoped; no deps; fast for our subset | Real engineering (weeks); reinvents a well-trodden wheel |
| **GiNaC (C++) via JNI** | Fast; mature | Native build complexity; licensing (GPL — dealbreaker) |
| **SymPy via subprocess** | Proven by the paper | 1-minute compile, Python dep — rejected |

Recommendation: **Symja v0, minimal custom CAS v0.5** if we hit Symja limitations. Never SymPy.

Caching is non-negotiable. Key coarsened artifacts by `(dxir subtree hash, CAS version)`; serialize to disk; invalidate on IR change. Users should see compile-time cost only on first compile or after structural edits.

### 11.8 Integration with AD transforms

`grad`, `valueAndGrad`, `vmap` consume dxir. Coarsening also produces dxir. Order matters:

```
User lambda → dxir (initial)
            → [coarsening pass] → dxir (with coarsened SOIs as adjoint blocks)
            → [grad transform] → dxir (gradient of non-coarsened parts; coarsened adjoints are just called)
            → [shape & sharding verification]
            → [Shardy propagation]
            → StableHLO + SDY
```

For each SOI, coarsening generates two dxir subgraphs: the simplified primal and the gradient function. The gradient function is registered as a custom adjoint keyed to the SOI's entry node. When `grad` later encounters that node, it finds the pre-computed adjoint and skips op-by-op differentiation for the SOI. Everything outside SOIs goes through standard reverse-mode AD.

**Higher-order differentiation:** coarsening composes. `grad(grad(f))` can re-coarsen the gradient function, sometimes producing second derivatives that are also closed-form. The paper doesn't evaluate this explicitly but notes it follows.

### 11.9 Numerical stability

Symbolic transformations can reorder arithmetic and expose overflow/underflow. The paper catches this by pattern-matching unstable forms (e.g., `log(1 + exp(x))` → `log1p(exp(x))` with masking for large `x`) during code generation. We port the same pattern list and extend for transformer-relevant patterns: `softmax`, `logsumexp`, numerically stable `layernorm`, `scaled_dot_product_attention` with causal mask fusion.

This is boring, mechanical, and essential. Write it once, test it against PyTorch's numerical outputs, move on.

### 11.10 User-facing controls

Coarsening runs automatically for `jit`-wrapped functions. Exposed knobs:

```kotlin
// Opt out entirely (debugging, numerical comparisons)
jit(coarsen = false) { ... }

// Tune SOI size limit
jit(coarsenBudget = 500) { ... }

// Force a boundary: nothing may coarsen across this point
val y = coarsenBarrier(expensivePythonInterop(x))

// Inspect the coarsened IR (for plugin authors, optimizer tuning)
val report = coarseningReport(step)
println(report.soiCount)
println(report.estimatedSpeedup)
```

Defaults should be right for 95% of users. The knobs exist for when they're not.

### 11.11 Scope and timing

This is **not** MVP material. MVP (Month 1–3) ships without coarsening; users get PyTorch-comparable performance via StableHLO + PJRT and that's fine for the initial adoption story. Coarsening is the **v0.5 pre-1.0 differentiator** — the thing that makes the `LoRAsmith` demo train a mobile-phone-sized LoRA in half the time a PyTorch-ExecuTorch equivalent would take, and makes the probabilistic-programming design partner we wanted by Month 6 actually stay.

Target: **Month 9**. One engineer, full-time, for ~3 months after the Month 6 milestone lands. Assumes a working K2 plugin and dxir foundation.

### 11.12 Validation plan

Port the paper's six benchmarks verbatim: BGDHyperOpt, Brachistochrone, CartPole, HMC, HookeanSpring, QWOP. Track:

1. Speedup vs our own non-coarsened baseline (expect matches to the paper, ±20%).
2. Speedup vs PyTorch 2.x with `torch.compile` (expect wins on scalar benchmarks, parity on CartPole).
3. Speedup vs JAX with `jit` (expect wins on scalar + control-flow benchmarks, parity elsewhere).
4. Numerical deviation from reference implementations (target: max relative error < 1e-5 for f32).
5. Compile-time cost, with and without caching.

If we can't match the paper's numbers on our own benchmarks, the whole coarsening investment is questionable. Validate early, kill if needed.

### 11.13 Definition of Done

- [ ] dxir SSA verifier confirms all nodes meet the coarsening pass's preconditions.
- [ ] φ-calculus pass implemented: all five fundamental formulae + nine corollaries, with property-based tests.
- [ ] Reuse-aware SOI identification algorithm ported from the paper.
- [ ] CAS integration (Symja default) with on-disk caching keyed by IR hash.
- [ ] Numerical-stability pattern library covering ~20 known unstable forms.
- [ ] All six paper benchmarks ported and passing; speedups within 20% of paper's figures for comparable hardware.
- [ ] Compile-time cost < 5 seconds for a typical training step after warm cache; < 30 seconds cold.
- [ ] `jit(coarsen = false)` disables the pass cleanly; numerical outputs match within f32 tolerance.
- [ ] Public benchmark harness comparing DiffKtX (coarsened) vs PyTorch 2.x + `compile` vs JAX + `jit` on all six benchmarks.

---

## 12. Roadmap (the Ladder)

Timelines assume a team of 2–4 engineers. Solo founder: 3x everything.

### Month 1 — Skeleton that trains MNIST

**Scope:**
- `core/` with `DScalar`, `DTensor`, `Shape` hierarchy (static shapes only).
- `ir/` with dxir nodes and a runtime-tape-based `grad` (no plugin yet).
- 20 ops: add, sub, mul, div, matmul, relu, softmax, cross_entropy, sum, mean, exp, log, reshape, transpose, broadcast, cast, neg, pow, layernorm, embedding.
- `runtime-torch/` only (libtorch eager JNI).
- CI: GitHub Actions on Linux x86_64 + macOS arm64.
- **Demo:** MLP on MNIST, trains to >95% test accuracy, single file under `examples/`.

**Exit criteria:** `./gradlew :examples:mnist:run` works on a laptop and converges.

### Month 3 — Compiler plugin + StableHLO + SDY + real ops

**Scope:**
- K2 compiler plugin for `grad`, `valueAndGrad`, `jit`. (Defer `vmap`, `shardMap`.)
- dxir → StableHLO emission for 40 ops (see §2.3).
- dxir → SDY sharding attribute emission (mesh decl, sharding, sharding_constraint).
- Round-trip test: any dxir module with shardings serializes to `.mlirbc`, parses via `sdy-opt`, and deserializes back identically.
- `runtime-iree/` with CPU target for single-device execution.
- `runtime-pjrt/` skeleton: PJRT C API bindings, CPU plugin wired up, no multi-device yet.
- Shape-typing plugin v0: `Rank1`–`Rank4` with literal and symbolic atoms; no attribute-dependent shape computation yet.
- Sharding-typing plugin v0: mesh axis name resolution + dim count check.
- **Demo:** ResNet-18 forward pass + backward pass, compiled via StableHLO, within 2x of PyTorch CPU.

**Exit criteria:** `grad { ... }` in user code generates no runtime tape; inspection of compiled bytecode shows direct StableHLO dispatch; a 2D-sharded BERT-base forward pass round-trips through Shardy's propagation pipeline on a single host (simulated devices) and produces byte-identical collectives to the JAX reference.

### Month 6 — KMP, mobile, multi-device, and the first real user

**Scope:**
- `android/` module with IREE-Vulkan AAR, NNAPI fallback.
- `ios/` module with IREE-Metal via cinterop.
- Shape-typing plugin v1: attribute-dependent shapes for conv2d, pooling, reshape.
- Sharding-typing plugin v1: divisibility checks, replicated-axis verification, priorities surfaced as Kotlin-side attrs.
- `vmap`, `shardMap`, and `checkpoint` transforms.
- `data/` DataLoader with Flow-based API and `.sharded(world, axis)` helper.
- Multi-device server training via PJRT: real 8-GPU run on one node via CUDA PJRT plugin.
- `dataParallel`, `fsdp`, `tensorParallel` one-liners (§9.5) work on the reference model.
- IDE plugin v0: IntelliJ shape + sharding hints.
- **Demo:** "Whispr" (on-device Whisper fine-tuning) or "LoRAsmith" published as a public app. Server demo: FSDP training of a 1B-param transformer matches JAX baseline.

**Exit criteria:** One external design partner shipping something with DiffKtX in a real product. Without this, everything below is vanity. Separately, `fsdp(mesh).fit(model, data)` trains a 1B-param model on an 8-GPU node and converges within 2% of the JAX reference over 1000 steps.

### Month 9 — Coarsening optimization lands

**Scope:**
- `coarsening/` module with full φ-calculus pass (five fundamental formulae, nine corollaries) on dxir.
- Reuse-aware SOI identification algorithm (§11.6), ported from the paper.
- Symja-based CAS integration with on-disk caching keyed by `(dxir hash, CAS version)`.
- Numerical-stability pattern library (log1p, logsumexp, softmax, layernorm, etc.).
- Custom-adjoint splicing into the AD transform pipeline.
- `jit(coarsen = false)` escape hatch for debugging.
- Six benchmark ports: BGDHyperOpt, Brachistochrone, CartPole, HMC, HookeanSpring, QWOP.
- Public head-to-head benchmark harness: DiffKtX (coarsened) vs PyTorch 2.x + `compile` vs JAX + `jit`.
- **Demo:** re-run the "LoRAsmith" mobile demo showing measurable fine-tuning speedup from coarsening on phone-side scalar kernels.

**Exit criteria:** Speedups on the paper's six benchmarks within 20% of the paper's figures for comparable hardware; DiffKtX wins decisively (>3×) against PyTorch 2.x `torch.compile` on at least three of the six; numerical outputs match PyTorch reference within f32 tolerance; compile time < 5s warm-cache, < 30s cold-cache.

### Year 1 — Legitimacy

**Scope:**
- Multi-node distributed training via PJRT (real cluster; rendezvous, fault handling).
- TPU support via the same PJRT plugin — no code changes for users who already use `fsdp` or `tensorParallel`.
- Pipeline parallelism via Shardy MPMD (§9.7), surfaced as the `pipeline { ... }` DSL.
- Paper on the shape + sharding type system (PLDI, ICFP, or OOPSLA target). Coarsening credited as the ancestor technique, not novel contribution.
- Second real customer in production — ideally in probabilistic programming or differentiable simulation, where coarsening wins hardest.
- Benchmarks vs PyTorch (FSDP + DTensor), JAX (pjit / shard_map), MLX on a transparent public suite.
- Stable API — semantic versioning, deprecation policy, 1.0 track.
- `jax2dxir` / `onnx2dxir` import paths for adoption. StableHLO artifacts from JAX with SDY attrs load natively.

**Exit criteria:** Someone who doesn't work for you gives a conference talk about a system they built on DiffKtX. That's the day this project has a future.

---

## 13. Repo Layout (Concrete)

```
diffktx/
├── build-logic/
│   └── convention/
├── core/
│   ├── src/commonMain/kotlin/io/diffktx/core/
│   │   ├── DScalar.kt
│   │   ├── DTensor.kt
│   │   ├── Differentiable.kt
│   │   ├── Shape.kt
│   │   ├── DType.kt
│   │   ├── Mesh.kt
│   │   └── Sharding.kt
│   └── src/commonTest/
├── ir/
│   └── src/jvmMain/kotlin/io/diffktx/ir/
│       ├── DxirNode.kt
│       ├── DxirModule.kt
│       ├── DxirSharding.kt
│       ├── passes/
│       │   ├── ShapeInference.kt
│       │   ├── ShardingVerify.kt
│       │   ├── DeadCodeElim.kt
│       │   └── CommonSubexpr.kt
│       └── transforms/
│           ├── Grad.kt
│           ├── Vmap.kt
│           ├── ShardMap.kt
│           └── Checkpoint.kt
├── compiler-plugin/
│   ├── plugin/               # K2 FIR/IR extension
│   └── gradle-plugin/
├── shapetyping/
├── coarsening/
│   └── src/jvmMain/kotlin/io/diffktx/coarsening/
│       ├── PhiCalculus.kt       # Five fundamental formulae + nine corollaries
│       ├── SoiIdentifier.kt     # Reuse-aware segment identification
│       ├── CasBridge.kt         # Symja (default) integration
│       ├── NumericalStability.kt # Pattern library (log1p, logsumexp, ...)
│       ├── AdjointSplice.kt     # Integrate coarsened gradient as custom adjoint
│       └── Cache.kt             # On-disk caching keyed by (dxir hash, CAS version)
├── sharding/
│   └── src/jvmMain/kotlin/io/diffktx/sharding/
│       ├── SdyAttrs.kt        # SDY attribute model mirroring dxir
│       ├── Propagation.kt     # JNI to libShardy propagation pipeline
│       └── Export.kt          # JNI to SDY export passes (collective insertion)
├── stablehlo/
│   └── src/jvmMain/kotlin/io/diffktx/stablehlo/
│       ├── Emitter.kt
│       ├── SdyEmitter.kt
│       └── Bytecode.kt
├── runtime-pjrt/
│   ├── src/jvmMain/kotlin/
│   └── src/jvmMain/jni/      # PJRT C API glue
├── runtime-iree/
│   ├── src/jvmMain/kotlin/
│   └── src/jvmMain/jni/      # C++ glue
├── runtime-torch/
├── android/
│   └── src/androidMain/kotlin/
├── ios/
│   └── src/iosMain/kotlin/
├── data/
├── examples/
│   ├── mnist/
│   ├── lora-on-device/
│   ├── diff-sim-flappy/
│   └── sharded-llm-pretrain/
├── benchmarks/
├── ide-plugin/
├── docs/
│   ├── getting-started.md
│   ├── design/
│   └── migration-from-pytorch.md
├── settings.gradle.kts
└── libs.versions.toml
```

---

## 14. Tooling & Dev Setup

- **Kotlin:** 2.2.x+ (K2 default, context parameters stable).
- **Gradle:** 8.12+, configuration cache on.
- **JDK:** 21 LTS for build; target bytecode 17.
- **MLIR/StableHLO:** pinned to a specific commit; updated quarterly.
- **Testing:** Kotest (property-based), JUnit 5 for plugin tests, Robolectric for Android.
- **Formatting:** ktlint + detekt, enforced in CI.
- **Docs:** Dokka for API, MkDocs Material for the book.
- **Benchmarks:** kotlinx-benchmark + custom harness vs PyTorch.
- **CI:** GitHub Actions — Linux x86_64, Linux aarch64, macOS arm64, Windows x86_64 (inference only).

---

## 15. Design Principles

1. **Compile, don't trace.** Runtime tapes are a last resort.
2. **Types over docstrings.** If a constraint can live in the type system, it must.
3. **Ownership is explicit.** Native memory is never implicitly released.
4. **No kernels.** We lower to StableHLO and move on.
5. **One idiomatic way.** Don't port PyTorch's three APIs for everything; pick one.
6. **Mobile is a first-class target, not an afterthought.** If a feature breaks KMP, it doesn't land.
7. **Boring beats clever.** Every clever thing is a future maintenance bill.

---

## 16. Risks & Mitigations

| Risk                                           | Likelihood | Impact | Mitigation                                                                 |
|------------------------------------------------|------------|--------|----------------------------------------------------------------------------|
| StableHLO op semantics drift                   | Medium     | High   | Pin versions; own a small shim layer; contribute fixes upstream            |
| Shardy SDY dialect churn (work-in-progress)    | High       | High   | Pin a specific Shardy commit; vendor the C bindings; quarterly upgrade cadence with regression suite |
| Shardy propagation picks suboptimal shardings for our op mix | Medium | Medium | Use `shard` constraints to pin; contribute propagation rules upstream |
| K2 plugin API changes across Kotlin releases   | High       | High   | Keep plugin surface small; CI against Kotlin EAP                           |
| PJRT plugin availability lags hardware releases | Medium    | Medium | IREE fallback for single-device; PJRT plugin API is stable, plugins come from vendors |
| IREE mobile runtime size                       | Medium     | High   | Build custom IREE with only needed codegens; measure on every release      |
| JNI overhead kills per-op perf                 | Medium     | Medium | Batch native calls; `@CriticalNative`; fuse at IR level                    |
| Coarsening compile-time cost is prohibitive    | Medium     | High   | Aggressive caching keyed by IR hash; `jit(coarsen=false)` escape hatch; move off Symja to a tighter custom CAS if measured |
| Coarsening speedups don't materialize on real workloads | Low | Fatal-for-the-optimization | Port the paper's six benchmarks first; if we can't match within 20%, cut the feature before v0.5 ships |
| Symbolic engine (Symja) limitations hit us late | Medium     | Medium | Evaluate against paper benchmarks by Month 8; fallback is custom minimal CAS (4–6 weeks) |
| Nobody uses it                                 | High       | Fatal  | Pick ONE design partner in month 3, not month 6; kill the project at M6 otherwise |
| Shape types too verbose in practice            | Medium     | Medium | Aggressive type inference; IDE hints; `DynShape` escape hatch              |
| Sharding types too verbose in practice         | Medium     | Medium | `shard { ... }` DSL keeps axis names close to shape atoms; one-liner recipes (§9.5) |
| Vendor lock-in to IREE/XLA/Shardy              | Low        | Medium | StableHLO+SDY is the artifact; any propagator/runtime that consumes it is swappable |

---

## 17. What Claude Code Should Work On First

Suggested sequence for a coding agent picking this up:

1. **Bootstrap:** `settings.gradle.kts`, `libs.versions.toml`, `core/` skeleton, `DScalar` / `DTensor` / `Shape` with full test suite. (§§0–1, §5)
2. **Torch-backed MVP:** `runtime-torch/` with JNI to libtorch eager, 20 ops, runtime tape `grad`. Ship the MNIST example. (§7, §8, Month 1 of §12)
3. **IR:** `ir/` module with dxir nodes (including sharding attrs), shape inference, DCE, CSE. Tests with fixtures. (§3.3)
4. **StableHLO emitter:** 40-op lowering, round-trip tests against reference `stablehlo-translate`. (§2)
5. **IREE runtime:** JNI bindings, CPU dispatch, replace torch for compiled paths. (§8)
6. **K2 plugin — AD:** `grad` as IR transform. Start with scalar functions, extend to tensors. (§3)
7. **Mesh + Sharding core types:** `Mesh`, `Spec`, `PartitionSpec` in `core/`. No compiler magic yet — pure Kotlin with runtime validation. (§9.3)
8. **SDY emitter + Shardy bindings:** `sharding/` module with libShardy JNI; `stablehlo/SdyEmitter.kt` writes `sdy.mesh`/`sdy.sharding` attrs. Round-trip test through `sdy-opt`. (§9.6)
9. **Propagation pipeline:** wire Shardy's propagation + export passes as a JNI-driven stage between dxir lowering and PJRT compile. (§9.6)
10. **PJRT runtime:** CPU plugin first, then CUDA. End-to-end 2-device FSDP test on CPU. (§8, §9.9)
11. **ShapeTyping + sharding types:** K2 plugin for attribute-dependent shape resolution (conv/pool/reshape) and sharding consistency checks. (§4)
12. **Mobile:** Android AAR with IREE-Vulkan; verify sharding code is unreachable from Android sources. (§10)
13. **Coarsening — φ-calculus:** `coarsening/PhiCalculus.kt` implementing the five fundamental formulae and nine corollaries on dxir; property-based tests per formula. (§11.5)
14. **Coarsening — SOI identification:** port the paper's reuse-aware SOI algorithm verbatim (§11.6). Unit tests on synthetic IR graphs.
15. **Coarsening — CAS bridge:** Symja integration with IR-hash-keyed disk cache; numerical-stability pattern library; adjoint splicing into the grad transform. (§11.7–11.9)
16. **Coarsening — benchmark validation:** port the paper's six benchmarks; gate coarsening release on matching paper's speedups within 20%. (§11.12)

Do not jump ahead. Step 4 is gated on step 3 passing property tests against PyTorch. Step 6 is gated on 4 working end-to-end. Step 9 is gated on step 8's round-trip test being green. Step 10 is the first place multi-device correctness can be verified against a JAX reference — do not skip that verification. **Step 16 is gated on matching the paper's numbers. If coarsening doesn't deliver measured speedups on the paper's own benchmarks, do not ship it — cut the feature and ship the rest.**

---

## 18. Open Questions

- Name. `diffktx` is a placeholder. Shortlist: `koan`, `kosmos`, `axon`, `gradus`, `diffkt2` (if we want to honor the ancestor). Decide before v0.1.
- License. MIT like the original? Apache 2.0 is safer for corporate adoption.
- Governance. Single-maintainer BDFL until v1.0, then foundation? Or Kotlin Foundation from day 1?
- **Shardy version pinning.** Shardy is explicitly a work in progress; the SDY dialect may change shape in the next 12 months. Do we pin to a specific commit, vendor the C bindings, and upgrade quarterly — or track HEAD and take the breakage? Recommendation: pin + vendor, upgrade on a schedule.
- **Shardy propagation control.** Propagation is deterministic but not always optimal. Do we expose propagation priorities (`p0`, `p1`, `p2`) in the user API from day 1, or hide them behind `shard` constraints until a real user asks?
- **MPMD / pipeline parallelism scope.** Shardy's MPMD dialect is in active development. Target pipeline parallelism for v0.5 (aggressive) or v1.0 (safe)?
- **PJRT plugin strategy.** Bundle the CUDA plugin, or require users to install it separately? Bundling is ~500MB of CUDA deps; separate install is a worse onboarding experience.
- **TPU support timing.** Do we claim TPU in the Year 1 marketing, or is that a v1.5 promise contingent on a partner with TPU access?
- **Symbolic engine choice for coarsening.** Symja (JVM, Apache 2.0, mature but heavyweight) or a minimal custom CAS (weeks of work, exactly scoped, no deps)? Recommendation: Symja through v0.5, evaluate a swap at v0.7 if compile-time or expressiveness becomes a problem.
- **Coarsening for higher-order AD.** The paper explicitly targets first-order backward AD. For Bean Machine-style probabilistic programming we need second-order. The paper notes coarsening composes with higher-order but doesn't evaluate it. Commit resources to this as a research problem or defer?
- **Coarsening on mobile.** φ-calculus compile time is a server concern; on-device we precompute artifacts. But does the AAR need to ship the Symja JAR, or can we bake coarsened artifacts at publish time and ship only the generated code? Recommendation: latter — artifact baking keeps the AAR size down.
- Sparse tensors. Old DiffKt's claimed differentiator. Worth reviving in v0.5 or skip? Note that StableHLO has limited sparse support and Shardy has essentially none.
- Quantization. Critical for mobile, but a tarpit. Deferred to v0.5 minimum.
- Training in Compose Multiplatform — gimmick or wedge?

---

*End of spec. Update as decisions are made. Keep a `CHANGELOG.md` for architectural changes.*
