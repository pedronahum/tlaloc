# Sharded serving: the design, and what is not decided

Status: **design only**. Nothing here is built. Written 2026-11-17 at `f20eee97`, as Stage A of
item 5 in [xse-lessons-plan.md](../work-log/xse-lessons-plan.md). The trigger is the one pillar
where Google's XLA Serving Engine is ahead of us — "Supports both single-host and multi-host
serving by leveraging PJRT" — against
[SERVING_ARCHITECTURE.md](../SERVING_ARCHITECTURE.md) §2's "Serving artifacts are single-device,
so their bodies contain none. Tlaloc serving on more than one device has not been built."

Evidence labels are the repository's. **Source** = read in the tree; **Doc** = recorded elsewhere;
nothing here is **Run**.

## What already exists, and what does not

Built and checked on four XLA:CPU host devices (**Doc**: [DISTRIBUTED.md](../DISTRIBUTED.md),
`PjrtMultiDeviceCpuTest`, `OpShardingLayoutTest`):

- `PjrtSession.shardedProgram`, `runSharded`, `shardDims` (`PjrtSession.kt:231,308,371`) split and
  reassemble inputs and outputs by the shardings the compiled executable reports
  (`parameterShardings` / `outputShardings`, `PjrtFfm.kt:2362`).
- `OpShardingLayout` parses replicated, tiled, partially-replicated and iota layouts
  (`OpShardingLayout.kt:81-133`).
- The StableHLO emitter writes Shardy mesh and sharding annotations when a graph declares a mesh;
  a serving artifact declares none, so its bodies contain none (**Doc**: §2).
- `CpuClientOptions.cpu_device_count` gives N host devices without a GPU; the plugin is built and
  installed at `~/.local/lib/tlaloc/pjrt_c_api_cpu_plugin.so`.
- Two parallel serving forms exist **on one device**: `MOE_EXPERTS` with `expert_parallel`, and
  paged attention with `return_lse` plus `ContextParallel.merge` (**Doc**: DISTRIBUTED.md).

Not built: composing those across devices; a serving entry with a mesh; a KV pool whose pages are
sharded; any multi-GPU or multi-host serving. `PjrtClientOptions` refuses more than one node
because PJRT's key-value store is not implemented (**Doc**: DISTRIBUTED.md,
[MULTIHOST_DESIGN.md](MULTIHOST_DESIGN.md)). The vLLM plugin refuses more than one device.

## The questions a sharded serving entry has to answer

### 1. How a bucket ladder maps to a mesh

A serving artifact is a ladder of (batch, context) entries, each a closed-form program. Two
distinct axes could be sharded, and they are not the same decision:

- **Weights across devices (tensor/pipeline parallel).** One sequence's step reads a fraction of
  the weights per device. This is what a 65B model on two GB10s wants: the decode step is at the
  memory ceiling, so halving the bytes per device halves the step. The batch and context ladders
  stay as they are; every entry gains a mesh and per-operand shardings.
- **Sequences across devices (data parallel).** Each device runs whole sequences. This needs no
  sharded program at all — it is N independent single-device clients, which the Triton backend
  already does per GPU via `instance_group`. It is not what "sharded serving" means here, and
  saying so matters: the cheap win is already available and is not this design.

**Proposal:** shard weights only, keep the ladder's shape, and let the mesh be a property of the
artifact rather than of an entry. A decode entry's batch dimension stays replicated across the
mesh; the weight operands are tiled. That keeps `runSharded`'s split/reassemble contract intact
for the activations, which are small, and puts the collectives where the bytes are.

**Open:** whether XLA's SPMD partitioner keeps the paged-attention custom call intact under a
weight sharding, or whether the call must itself become mesh-aware. The custom calls
(`PAGED_ATTENTION`, `MOE_EXPERTS`) take device pointers; a sharded pool changes what those pointers
mean. This is the first thing Stage B must answer, and it may sink the whole approach.

### 2. Who owns a page's shard

Today a KV pool is one device buffer per layer class, addressed by a block table and a slot
mapping, and a sequence's pages are chosen by the backend (`WindowedKvPool`, `KV_POOL_IN` /
`KV_POOL_OUT`). Sharding the pool means a page's bytes live on several devices.

Three options, in increasing order of what they break:

1. **Replicate the pool, shard the weights.** Every device holds the whole KV pool; only weights
   are tiled. Attention then runs replicated on every device with the same pages, and the
   collective is in the MLP/projection path. Costs KV memory ×N, which for Muse Glimmer's windowed
   pools is 4,109 MiB for four sequences at 32,768 (**Doc**: CAPABILITIES.md) — affordable, and it
   leaves the block table, the slot mapping and the ring arithmetic **completely unchanged**.
2. **Shard the pool by head.** Each device holds a subset of KV heads. Attention is already
   per-head, so this is the natural cut, but the block table becomes per-device and the ring's
   `b % ringPages` arithmetic must agree across devices. `return_lse` plus `ContextParallel.merge`
   is the existing mechanism for combining partial attention, so the merge is not new.
3. **Shard the pool by page.** Pages are distributed round-robin. This breaks the locality the
   ring depends on and needs an all-to-all per step. Not proposed.

**Proposal:** option 1 first. It is the only one that does not touch the page allocator, and it
answers the memory-ceiling problem (weights) without reopening the KV correctness argument that
`WindowedKvPoolTest` and `verify.sh` already settle. Option 2 is the follow-up if KV memory
becomes the binding constraint.

### 3. Donation and aliasing under a sharding

The backend donates `KV_POOL_IN` to `KV_POOL_OUT` so a step writes the pool in place, and the
artifact carries `tf.aliasing_output` to say so; `verify.sh` requires 100 of 100 runs in place
(**Doc**: §2, §6). Under SPMD, donation is per-device: each device's input shard must alias its
own output shard.

**Open, and the second thing Stage B must answer:** does PJRT's `DonateArguments` /
`aliasing_output` survive SPMD partitioning with the same in-place guarantee, or does the
partitioner insert a copy that silently defeats it? The check is the existing one — count the
runs that alias — and it must be re-run on a sharded entry, not assumed. If it does not survive,
the sharded path is slower than the single-device path by a pool copy per step, which for a
4 GiB pool is fatal and would end the approach.

### 4. What a new manifest schema version carries

`tlaloc-serving-v5` is the current schema. A sharded artifact needs, at minimum:

- a `mesh` object: axis names, sizes, and the device assignment (or "the client's devices in
  order", which is what a single-host mesh means);
- per-entry `parameterShardings` / `outputShardings` as the proto bytes PJRT reports, or a
  declarative form the emitter can reproduce — the former is what `runSharded` already consumes;
- the KV pool's replication decision (option 1 vs 2 above), because a reader must know whether
  to allocate one pool per device or one pool total;
- a `schemaVersion` bump, so every existing runtime refuses it **by name** rather than
  misreading it. That is the repository's established pattern and it is what makes the change
  safe to land before any runtime supports it.

### 5. Which runtimes could serve it

| Runtime | Could serve a sharded artifact | Why |
|---|---|---|
| `ServingModel` (`:runtime-pjrt`) | Yes, first | `runSharded` and `shardedProgram` are already its API; it needs a mesh-aware load path and per-device weight slots |
| Triton backend | Later | One PJRT client per GPU instance today; a sharded entry needs one client spanning devices, which `PjrtClient::Create` refuses (it checks the device it gets back is the one it asked for) |
| framework-free Python | No, near term | ctypes over the C API; it would need the sharding proto parsing that `OpShardingLayout` does |
| vLLM plugin | No | Refuses more than one device by design |

**Proposal:** `ServingModel` first, and the new schema version refused by name everywhere else.
That gives a real sharded execution to measure without touching the C++ backend.

### 6. What a prefill/decode split would mean for cohort batching

Disaggregation (prefill on one device set, decode on another) is a different feature from
sharding and should not be conflated with it. It would break the cohort assumption the backend
batches on — that the sequences in a batch are all decoding — and the measured prefill-budget
result (**Run**, §6) says the interaction between a long prompt and a decode batch is already the
dominant latency effect. **Out of scope here**, and recorded in FOLLOWUPS rather than designed.

## Stage B: the prototype, and what would falsify it

No second GPU is needed. The CPU plugin gives four host devices.

1. Emit the five-layer closed-form model the interpreter tests already use with a mesh over four
   devices and a weight sharding, through the existing Shardy path.
2. Run it with `runSharded`; compare against the single-device result within the tolerance the
   interpreter tests use. Record the collective count and the per-device memory.
3. Add a donated in/out pair with `tf.aliasing_output` and check the aliasing survives — question
   3 above. **If it does not, stop and record that**, because it decides whether the approach is
   viable at all.
4. Only then consider a real decoder layer with a paged-attention custom call — question 1.

What would falsify the design: the partitioner copying a donated pool (step 3), or the custom
call refusing a sharded operand (step 4). Either is a result worth recording, and neither
requires a second GPU to discover.

## What this document does not claim

No sharded serving entry exists. No multi-GPU or multi-host measurement has been made, and none
is claimed. The proposals above are choices with reasons, not decisions ratified by a run; the
two questions marked **Open** are the ones that can end the approach, and Stage B exists to
answer them cheaply. Nothing here changes a serving artifact, a runtime, or the backend.
