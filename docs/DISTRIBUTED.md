# Distributed programs

Tlaloc writes sharded programs (meshes, shardings and collectives in StableHLO with
Shardy) and differentiates them. Its runtime executes them on one device or, through the XLA
CPU plugin, on several host devices. This page says which parts exist, which have run, and
where.

| | Status |
|---|---|
| Meshes, parameter shardings, sharding constraints, manual computations in StableHLO | emitted; pinned by tests |
| `ALL_REDUCE`, `ALL_GATHER`, `REDUCE_SCATTER`, `ALL_TO_ALL`, `COLLECTIVE_PERMUTE` | emitted, interpreted, differentiated (see below) |
| Compile options for SPMD (`CompileOptions.spmd(deviceIds)`) | built; byte-pinned |
| Sharded programs and collectives on several devices | run on 4 XLA:CPU devices, through jaxlib and through `PjrtSession`, in the tests only |
| `PjrtSession` on several devices | built (`runPerDevice`, `runSharded`, `shardDims`); run on the XLA CPU plugin's host devices only |
| Sharded training steps (`ShardingPlan`: data parallel, FSDP, tensor parallel) | run on 4 XLA:CPU devices, equal to the unsharded step |
| Several GPUs, several hosts, a TPU slice | not run; multi-host is designed ([design/MULTIHOST_DESIGN.md](design/MULTIHOST_DESIGN.md)) |

## Meshes and shardings

Shardings are written on the program, in DXIR. There is no sharding API on `DTensor`
or in `:nn` layers.

<!-- doc-test: dist-sharded -->

```kotlin
val mesh = DxirMesh("m", listOf(DxirMeshAxis("data", 4)))
val byRows = DxirSharding("m", listOf(DxirDimSharding(listOf(DxirAxisRef.Full("data"))), DxirDimSharding(emptyList())))

val loss = DxirBuilder.function("sharded_loss") {
    declareMesh(mesh)
    val x = param("x", DxirType(F32, listOf(8, 4)), byRows)   // rows split over "data"
    val w = param("w", DxirType(F32, listOf(4, 3)))           // replicated
    val y = op(OpKind.TANH, listOf(op(OpKind.MATMUL, listOf(x, w), DxirType(F32, listOf(8, 3)))), DxirType(F32, listOf(8, 3)))
    listOf(op(OpKind.SUM, listOf(op(OpKind.MUL, listOf(y, y), y.type)), DxirType(F32, emptyList())))
}
val gradient = DxirReverseTransform.apply(loss)   // keeps the mesh and x's sharding
println(gradient.toStablehlo())
```

```
sdy.mesh @m = <["data"=4]>
func.func @sharded_loss_grad(%0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@m, [{"data"}, {}]>}, %1: tensor<4x3xf32>) -> …
```

- `MeshSpec` and `PartitionSpec` (`:core`) convert with `toDxir()`.
- A dimension's sharding can be open (`?`, propagation may extend it), carry a priority,
  or name a sub-axis.
- `OpKind.SHARD_CONSTRAINT` becomes `sdy.sharding_constraint`;
  `OpKind.MANUAL_COMPUTATION` becomes `sdy.manual_computation` (one region, one block).
- The reverse, forward and vmap transforms keep the meshes and the parameter shardings.
- Shardings on plain operations are not emitted; Shardy's propagation decides those,
  and function results carry none.

## Sharded training

A `ShardingPlan` names a sharding for each data input and each parameter of a training step.
`CapturedStep.sharded(plan)` writes them on the step's parameters and constrains each parameter
gradient to its parameter's sharding; Shardy's propagation shards the rest of the program and
places the collectives when it is compiled for the mesh's devices. The step is still one
program, the same text on every device. The device optimizers hold their state like the
parameters.

<!-- doc-test: dist-sharded-training -->

```kotlin
val mesh = DxirMesh("mesh", listOf(DxirMeshAxis("data", 4)))
val step = capture(model, listOf(ids), targets = listOf(targets)) { logits, t -> crossEntropy(logits, t[0]) }
val device = step.sharded(ShardingPlan.fsdp(mesh)).withDeviceAdamW(AdamW(learningRate = 1e-2f))
```

| Preset | Data inputs | Parameters |
|---|---|---|
| `dataParallel(mesh, axis)` | split along the batch | whole on every device |
| `fsdp(mesh, axis)` | split along the batch | each split along its largest dimension that divides by the axis size |
| `tensorParallel(mesh, modelAxis, dataAxis?)` | whole, or split along the batch over `dataAxis` | `CausalLM` keys: q, k, v, gate and up projections by column, attention output and down projections by row, the head by vocabulary column; embeddings and norms whole |

A plan is two functions, so other layouts (another model's keys, a sub-axis, an open
dimension for propagation to extend) are written the same way. Apply
`withGradientAccumulation` and `withRemat` before `sharded`.

Run it with `runSharded`, which splits whole arrays by the shardings XLA compiled and puts the
results back together:

<!-- doc-test: dist-sharded-training-run -->

```kotlin
PjrtSession(
    plugin = PjrtBinaries.requireCpuPlugin("sharded training"),
    target = PjrtTarget.LlvmCpu,
    compileOptions = CompileOptions.spmd(listOf(0L, 1L, 2L, 3L)),
    cpuOptions = CpuClientOptions(4),
).use { s ->
    var params = step.parameterTensors(model).map { it.hostF32() }
    var state = device.initialState()
    for ((t, batch) in batches.withIndex()) {
        val outs = s.runSharded(device.function, batch + params + state + device.scalars(t + 1).map { floatArrayOf(it) })
        params = outs.subList(1, 1 + params.size)
        state = outs.subList(1 + params.size, outs.size)
    }
    println(s.shardDims(device.function).first) // each device's piece of each parameter
}
```

`PjrtShardedTrainingCpuTest` runs a two-layer `CausalLM` (d 64, 8 query and 4 key-value
heads, batch 8 × 16) on four XLA:CPU devices under four plans: data parallel, FSDP, tensor
parallel over 4, and data × tensor parallel on a 2 × 2 mesh.

- Each plan's loss and gradients equal the unsharded step's in the interpreter (1e-4), and six
  device AdamW steps give its losses and parameters.
- Each device holds 100 % (data parallel), 25 % (FSDP), 29 % (tensor parallel) and 53 % (2 × 2)
  of the parameters; the moments are held in the same pieces.
- In XLA's optimized program: data parallel has 2 all-reduces (the gradients, combined); FSDP
  36 all-gathers and 2 all-reduces (XLA:CPU writes the gradients' reduce-scatter as an
  all-reduce and a slice); tensor parallel 13 all-reduces and 8 all-gathers.
- Data parallel and FSDP written by hand (each device's step on its rows, parameters
  all-gathered before it, gradients all-reduced or reduce-scattered after) give the same values.
  Their times cannot be compared on host devices, which share the host's cores: the medians of
  one program move up to tenfold from round to round.

The partitioned step has not run on GPUs or a TPU slice, where the comparison with explicit
collectives is a measurement.

## Collectives

| Op | Attributes | Interpreter (one process) | Gradient |
|---|---|---|---|
| `ALL_REDUCE` | `replica_groups`, `reduction` (only `"sum"`) | the value times the size of replica 0's group | `ALL_REDUCE` of the cotangent |
| `ALL_GATHER` | `replica_groups`, `dimension` | the operand repeated along the dimension | `REDUCE_SCATTER` of the cotangent |
| `REDUCE_SCATTER` | `replica_groups`, `dimension` | replica 0's slice of the sum: the group's size times its slice | `ALL_GATHER` of the cotangent |
| `ALL_TO_ALL` | `replica_groups`, `split_dimension`, `concat_dimension` | the group's size copies of replica 0's chunk | `ALL_TO_ALL` with the two dimensions swapped |
| `COLLECTIVE_PERMUTE` | `source_target_pairs` | the operand when replica 0 is a target, zeros otherwise | `COLLECTIVE_PERMUTE` with each pair reversed |

`ALL_REDUCE` is emitted with a `channel_handle` and `use_global_device_ids`, so a group
lists global device ids, as JAX's `shard_map` writes it. Groups must be non-empty,
equal in size and disjoint, and include device 0; ragged groups and other reductions
are refused by name. There is no broadcast, and no ragged all-to-all. The adjoints follow
`ALL_REDUCE`'s convention: the loss is the sum of every replica's loss.
`GradShardingVerify` checks that each parameter's gradient is sharded like the
parameter, and that a gradient uses the adjoint collectives (all-gather for
reduce-scatter and back).

## What has run on several devices

`stablehlo/src/jvmTest/kotlin/io/tlaloc/stablehlo/MultiDeviceCpuTest.kt` compiles
Tlaloc's programs with SPMD partitioning for four XLA:CPU devices
(`--xla_force_host_platform_device_count=4`) through jaxlib
(`harness/python/run_on_cpu_devices.py`):

- `ALL_REDUCE` over all four devices and over two groups;
- `ALL_GATHER` followed by `REDUCE_SCATTER`;
- a data-parallel gradient: each device's gradient of a quarter of the batch, summed
  with `ALL_REDUCE`, equals the full batch's gradient;
- the sharded program above and its gradient, against the same program on one device.

The results are recorded, so the tests replay in CI without jaxlib.

`runtime-pjrt/src/jvmTest/kotlin/io/tlaloc/runtime/pjrt/PjrtMultiDeviceCpuTest.kt` runs the
same kinds of program through `PjrtSession` on the XLA CPU plugin with four host devices
(`CpuClientOptions`; the plugin is built by `scripts/pjrt-cpu-plugin/build.sh`):
`runPerDevice` runs one program on every device with each device's inputs, and
`runSharded` splits and reassembles the inputs and outputs by the shardings the compiled
program reports. Beyond the cases above: all-to-all and collective permute, and the adjoint
identity `Σ⟨f(x), w⟩ = Σ⟨x, f*(w)⟩` for the gradients of all-gather, reduce-scatter,
all-to-all and permute, and the sharded training steps above. Nothing has run on more than one GPU, on several hosts, or on a TPU. The `sdy-opt` tests of Shardy's
propagation (data and model parallel, FSDP-style shardings) need the `sdy-opt` binary.

## Parallel serving forms

On one device, and checked there:

- Expert parallel: `MOE_EXPERTS` with `expert_parallel` computes a slice of the experts,
  so every rank compiles the same program; two slices summed equal the whole op.
- Context parallel: paged attention with `return_lse` returns each slice's log-sum-exp,
  and `ContextParallel.merge` combines the slices.

Composing them across devices (all-gather the rows, the expert slice, reduce-scatter)
is not built. Serving artifacts are single-device, and the vLLM plugin refuses more
than one device.

## Multi-host

`PjrtClientOptions` carries a node id, a node count and a coordinator address
(`TLALOC_PJRT_NODE_ID`, `TLALOC_PJRT_NUM_NODES`, `TLALOC_PJRT_COORDINATOR_ADDRESS`).
Creating a client with more than one node is refused, because the key-value store PJRT
needs for it is not implemented. [design/MULTIHOST_DESIGN.md](design/MULTIHOST_DESIGN.md)
has the design and the order of the remaining work.
