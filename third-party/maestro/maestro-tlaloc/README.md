# maestro-tlaloc

First-class Tlaloc step type for Netflix Maestro (Layer 2.5 §0.4.245+).

This module is the Tlaloc-side complement to Maestro's `maestro-kubernetes` /
`maestro-notebook` integrations. It registers a new `Tlaloc` step type so
Maestro workflows can declare `"type": "Tlaloc"` natively, and ships the
`TlalocRunner` CLI that the runtime container image launches inside each
Kubernetes job.

Lives inside the vendored `third-party/maestro/` tree. The vendoring
philosophy and upgrade procedure are documented at
[`docs/vendoring.md`](../../docs/vendoring.md) and
[`third-party/README.md`](../../third-party/README.md).

## Module structure

```
maestro-tlaloc/
├── build.gradle
├── docker/
│   ├── Dockerfile          # multi-stage: JDK builder → JRE runtime
│   └── build.sh            # docker build helper, mirrors maestro-actus
├── src/main/java/com/netflix/maestro/engine/
│   ├── stepruntime/
│   │   └── TlalocStepRuntime.java        # extends KubernetesStepRuntime; pod-spec at launch (§0.4.259)
│   └── tlaloc/
│       ├── TlalocCommand.java            # immutable command record
│       ├── TlalocAttributeMapper.java    # naming bridge
│       ├── TlalocEntrypointBuilder.java  # generates K8s shell command
│       ├── TlalocParamsBuilder.java      # Maestro params → JSON payload
│       ├── TlalocRunner.java             # CLI main() inside the container
│       ├── BackendTargetRecord.java      # Java mirror of Kotlin BackendTarget (§0.4.259)
│       └── TlalocPodSpecBuilder.java     # backend-matrix → nodeSelector + accelerators (§0.4.259)
├── src/main/resources/defaultparams/
│   └── default-tlaloc-step-params.yaml   # Maestro-merged defaults
└── src/test/java/com/netflix/maestro/engine/tlaloc/
    ├── TlalocStepTypeRegistrationTest.java  # 6 smoke tests
    ├── TlalocRunnerEndToEndTest.java        # 5 e2e tests
    ├── TlalocSampleWorkflowsTest.java       # 11 sample-workflow parsing tests
    └── TlalocPodSpecBuilderTest.java        # 15 pod-spec selection + apply tests (§0.4.259)
```

## Layer 3 §0.4.259+ pod-spec construction

When a Tlaloc workflow's manifest carries a populated `backendMatrix`
(via `populateBackendMatrix(...)` on the producer side), the
`TlalocStepRuntime.customizePreLaunchCommand` reads the matrix and
translates the row matching the cluster's `(vendor, arch)` into K8s
pod-spec fields:

- `nodeSelector` — vendor-specific label match (e.g.
  `accelerator: nvidia-tesla-h100`,
  `cloud.google.com/gke-accelerator: tpu_v5e`,
  `aws.amazon.com/neuron: trainium2`).
- `accelerators` — observability map carrying the picked
  `vendor` / `arch` / `kernel` / `kv_quant_dtype` for log correlation.
- `gpu` — defaulted to `"1"` for any non-CPU target (multi-accelerator
  pods are L4+).

Selection algorithm: exact `(vendor, arch)` match wins; falls back to
lowest-cost row matching vendor; finally to absolute lowest-cost.
Empty/malformed matrix → command passes through unchanged.

Cluster `(vendor, arch)` is configured at deployment time via the
runtime's constructor (`TlalocStepRuntime(..., String clusterVendor,
String clusterArch)`); production deployments wire from a per-deployment
env var or DI binding.

### Vendoring divergence

The `KubernetesCommand` class (in `maestro-common/`) gained two new
fields in §0.4.259:

- `nodeSelector: Map<String, String>` — K8s node-selector labels.
- `accelerators: Map<String, String>` — vendor + arch + kernel + KV-quant
  hints recorded for observability.

Both fields are `@JsonInclude(NON_NULL)` — a vanilla Maestro consumer
that doesn't know about Tlaloc never sees them in serialized commands.
This is the one edit-against-upstream approved as L3.6's edit budget;
tracked as audit OQ-Layer3-4.

## Sample workflows

Seven Tlaloc sample workflows ship in
`maestro-server/src/test/resources/samples/sample-tlaloc-*.json`:

| Sample | Pattern |
|--------|---------|
| `sample-tlaloc-program-wf.json` | NoOp → Tlaloc (single-step canary) |
| `sample-tlaloc-pipeline-wf.json` | NoOp → Tlaloc → NoOp (linear pipeline) |
| `sample-tlaloc-typed-handoff-wf.json` (KEYSTONE) | Tlaloc → Tlaloc with `SerializedBufferHandle` |
| `sample-tlaloc-portfolio-wf.json` | foreach over training shards |
| `sample-tlaloc-hpo-sweep-wf.json` | nested foreach (HPO × shards) |
| `sample-tlaloc-iterative-tuning-wf.json` | while-loop bisection |
| `sample-tlaloc-template-wf.json` + `sample-tlaloc-caller-wf.json` | reusable subworkflow |

## curl-submission examples

To submit a sample to a running Maestro instance:

```bash
# Single-step canary
curl -X POST http://maestro/api/v3/workflows \
  -H 'Content-Type: application/json' \
  -d @maestro-server/src/test/resources/samples/sample-tlaloc-program-wf.json

# Typed-handoff (the keystone — two Tlaloc steps with SerializedBufferHandle)
curl -X POST http://maestro/api/v3/workflows \
  -H 'Content-Type: application/json' \
  -d @maestro-server/src/test/resources/samples/sample-tlaloc-typed-handoff-wf.json
```

(The `sample-tlaloc-caller-wf` invokes `sample-tlaloc-template-wf` as a
subworkflow; submit the template first.)

## Building the runtime image

From the `third-party/maestro/` root:

```bash
bash maestro-tlaloc/docker/build.sh                       # tags as tlaloc-runtime:latest
bash maestro-tlaloc/docker/build.sh tlaloc-runtime:0.1.0  # custom tag
```

The script runs `:maestro-tlaloc:assemble` + `:maestro-tlaloc:buildDockerContext`
to stage the runtime classpath under `build/docker/libs/`, then
`docker build` against `maestro-tlaloc/docker/Dockerfile`.

## Running the test suite

```bash
# From the Tlaloc repo root (composite build):
./gradlew :vendored-maestro:maestro-tlaloc:test

# Or from third-party/maestro/:
./gradlew :maestro-tlaloc:test
```

37 tests (6 step-type registration + 5 runner end-to-end + 11 sample
workflow parsing + 15 pod-spec selection/apply). All passing as of §0.4.260.
