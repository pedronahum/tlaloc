# Multi-host design — PJRT distributed init and the Maestro seam (G3a-2)

**Status (§0.4.461): DESIGN + the certifiable slice implemented.** No
multi-host run is claimed anywhere in this document — that needs 2+
hosts and is G4's, gated on hardware (see §6). What IS certified today,
GPU-less the §0.4.333 way: the `node_id`/`num_nodes` create-option
marshalling bytes, the multi-node create refusal, and the Maestro
pod-group expansion (pure-JVM spec correctness).

## 1. How PJRT does distributed, layer by layer

### 1.1 The C API surface (verified)

`PJRT_Client_Create_Args` (pjrt_c_api.h) carries TWO distributed
surfaces, and it matters which is which:

1. **`create_options` NamedValues.** The XLA **GPU** plugin's create
   path parses (verified against
   `xla/pjrt/c/pjrt_c_api_gpu_internal.cc`, openxla/xla main,
   2026-09-21): `platform_name` (string), `allocator` (string),
   `memory_fraction` (float), `preallocate` (bool),
   `collective_memory_size` (int64), `visible_devices` (int64list),
   **`node_id` (int64)**, **`num_nodes` (int64)**,
   `should_stage_host_to_device_transfers` (bool),
   `abort_collectives_on_failure` (bool), `use_tfrt_gpu_client` (bool),
   `enable_mock_nccl` (bool), `mock_gpu_topology` (string),
   `partition_index` (int64), `max_inflight_computations` (int64).
   Tlaloc passes `memory_fraction`/`preallocate` since §0.4.333 and now
   marshals `node_id`/`num_nodes` when a group is declared.
2. **The kv-store callbacks.** `kv_get_callback`/`kv_get_user_arg`,
   `kv_try_get_callback`/`kv_try_get_user_arg`,
   `kv_put_callback`/`kv_put_user_arg` — function pointers the plugin
   wraps into its C++ KeyValueStore
   (`pjrt::ToCppKeyValueStore(args->kv_get_callback, ...)`). Multi-node
   GPU clients rendezvous THROUGH this store (NCCL ids, device
   topology exchange). **There is no `coordinator_address`
   create-option**: the coordinator is a framework-side service — JAX
   runs its distributed runtime (coordination service) at
   `jax.distributed.initialize(coordinator_address, num_processes,
   process_id)` and implements the kv callbacks as RPCs to it. A
   framework that wants multi-node PJRT must BRING ITS OWN kv
   transport. Tlaloc's FFM layout has carried these fields since
   §0.4.333 — zeroed, and they stay zeroed in this slice.

### 1.2 The TPU side (recorded, partially unverified — no libtpu here)

libtpu does not take GPU allocator options (§0.4.459's platform gate)
and its multi-host wiring is env-first. Recorded from framework source
and the TPU VM docs, UNVERIFIED against a live libtpu (the G2b/G4
machine confirms):

- Single-slice multi-host: `TPU_PROCESS_BOUNDS`,
  `TPU_PROCESS_ADDRESSES`, `TPU_PROCESS_PORT`, `CLOUD_TPU_TASK_ID` —
  the per-process topology/rank envs libtpu reads at init; JAX on TPU
  pods derives them from the metadata server rather than
  create-options.
- **The MegaScale layer (multi-slice)**, the env family Ray's
  TorchTrainer multi-slice support and TorchTPU set per worker:
  `MEGASCALE_COORDINATOR_ADDRESS`, `MEGASCALE_NUM_SLICES`,
  `MEGASCALE_SLICE_ID`, `MEGASCALE_PORT`. Same shape as Tlaloc's env
  trio (§3), one level up: slices instead of nodes.
- `PJRT_ExecuteOptions`' TPU-flavoured fields (`num_tasks`/`task_ids`/
  `incarnation_ids`, `multi_slice_config`) stay zeroed per §0.4.459 —
  non-zero forms are G4 surface.

Design consequence: Tlaloc's distributed contract must be carried as
ENV (which both worlds consume — GPU via our `resolve()` mapping into
create-options, TPU via libtpu/MegaScale reading env directly), not as
GPU-only NamedValues. That is why the pod-group seam emits env.

## 2. Tlaloc's mapping: `PjrtClientOptions` grows the group contract

```
PjrtClientOptions(
    memoryFraction, preallocate,      // §0.4.333, unchanged
    nodeId = 0,                       // rank; marshals as node_id (kInt64) iff numNodes > 1
    numNodes = 1,                     // group size; marshals as num_nodes (kInt64) iff > 1
    coordinatorAddress = null,        // host:port; NEVER marshalled — kv-store dial target (G4)
)
```

Rules, all pinned in `PjrtClientOptionsMarshalTest`:

- **Single-node is byte-identical to §0.4.333** — two NamedValues,
  nothing new. The distributed fields add nothing until asked for.
- `numNodes > 1` marshals FOUR NamedValues (`memory_fraction`,
  `preallocate`, `node_id`, `num_nodes`) and requires
  `coordinatorAddress` (`host:port`, port in [1, 65535]); `nodeId` must
  sit in `[0, numNodes)`.
- `coordinatorAddress` is deliberately NOT a NamedValue (no such option
  exists — §1.1); it is the recorded dial target for the G4 kv-store
  callbacks, kept on the options object so ONE value states the whole
  group contract.
- **Client CREATION at `numNodes > 1` refuses by name**
  (`PjrtFfm.requireKvStoreForMultiNode`, called before any FFM work in
  `createClient`): with NULL kv callbacks the plugin would fail or hang
  mid-create; a loud refusal naming this document beats either.
- `resolve()` reads `TLALOC_PJRT_NODE_ID`, `TLALOC_PJRT_NUM_NODES`,
  `TLALOC_PJRT_COORDINATOR_ADDRESS` (absent → single-node), so a
  pod-group member needs no code to know its rank — the env IS the
  wire.

TPU note: `PjrtTarget.Tpu` sessions keep `options = null` (§0.4.459's
platform gate), so the GPU mapping above never touches libtpu; the TPU
multi-host story consumes the same env trio but routes it into libtpu's
own env surface (G4 decides the exact translation on hardware).

## 3. The env contract (the seam both sides certify against)

| Variable | Producer (Maestro) | Consumer (runtime-pjrt) |
|---|---|---|
| `TLALOC_PJRT_NODE_ID` | `buildPodGroup` member index | `PjrtClientOptions.resolve().nodeId` |
| `TLALOC_PJRT_NUM_NODES` | group size | `.numNodes` |
| `TLALOC_PJRT_COORDINATOR_ADDRESS` | `host:port` of node 0's service | `.coordinatorAddress` |

Both endpoints are unit-certified today; the wire between them (a real
cluster launching real processes) is exactly what G4 proves.

## 4. What multi-node init still needs (the G4 kv-store work)

The one missing runtime piece is the kv store: three FFM **upcall
stubs** (the KPTX spike proved Kotlin FFM upcalls work as typed C
callbacks, §0.4.326) implementing get/try_get/put against a
coordinator service. Design intent, not yet committed:

- Node 0 hosts the store (a small TCP/HTTP KV server bound at
  `coordinatorAddress`); other nodes are clients. JAX's coordination
  service is the reference shape; Tlaloc does NOT need gRPC parity —
  the PJRT contract is just blocking get-with-timeout / put of opaque
  bytes.
- The callbacks' C signatures come from `pjrt_c_api.h`
  (`PJRT_KeyValueGetCallback` etc.); each carries a `user_arg` we point
  at the Kotlin store handle.
- Rejected alternative: shelling out to a Python `jax.distributed`
  sidecar for init — reintroduces the Python runtime dependency the
  whole PJRT-FFM stack exists to avoid.

## 5. The Maestro seam: pod groups

`TlalocPodSpecBuilder.buildPodGroup(base, numNodes, coordinatorHost,
coordinatorPort)` (maestro-tlaloc, §0.4.461) expands one
accelerator-selected `KubernetesCommand` into N member commands —
member i's env gains the §3 trio with `NODE_ID = i`, its
job-deduplication key gains `-node<i>` (members must never collapse
into one K8s job), and EVERYTHING else is copied: same image, same
command, same resources, same nodeSelector.

**The workflow-level contract: a distributed step = N pods, ONE
program manifest, mesh-consistent.** The `:maestro` manifest already
carries StableHLO+SDY bodies and Mesh placement; the group is
homogeneous by construction because expansion copies one selected
base, so every member compiles the same program against the same mesh
and only rank differs. Composition order is pinned:
`applyBackendTarget` (pick the accelerator row) THEN `buildPodGroup`
(expand) — `TlalocPodGroupTest.composesWithBackendTargetSelection`.

Coordinator naming convention: node 0's pod under a headless service,
`<group>-node0.<service>` — the builder takes host/port as data and
mints no K8s object names (the runner owns naming).

Named non-goals of v1 (recorded on the method): no gang-scheduling
CRDs (Kueue/Volcano PodGroups are the deployment's job — a partially
scheduled group deadlocks at the PJRT barrier, so all-or-nothing
scheduling is a stated DEPLOYMENT REQUIREMENT); no per-member topology
spreading; no multi-slice/MegaScale env (single-slice groups until G4
measures a real one); no TlalocStepRuntime wiring of the expansion
into step execution (the step schema has no `distributed { nodes = N }`
block yet — that schema change rides with G4, when a real run can
certify it end-to-end).

## 6. The G4 dependency chain (recorded for the audit)

```
G2b  TPU execution on a Cloud TPU VM  (hardware: 1 TPU host)
G4a  kv-store upcalls + coordinator service      (hardware: 2+ hosts, any kind)
G4b  PJRT multi-node client create over G4a      (needs G4a; GPU pair suffices)
G4c  ALL_REDUCE across REAL devices — turns §0.4.460's single-process
     semantics into measured cross-device sums   (needs G4b)
G4d  Maestro distributed step schema + TlalocStepRuntime wiring
     (needs G4b; certifies the §3 wire end-to-end)
G4e  DDP-equivalent trainer over Phase F         (needs G4c)
MegaScale/multi-slice layer                      (needs G4e + TPU slices)
```
