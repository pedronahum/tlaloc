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
│   │   └── TlalocStepRuntime.java        # extends KubernetesStepRuntime
│   └── tlaloc/
│       ├── TlalocCommand.java            # immutable command record
│       ├── TlalocAttributeMapper.java    # naming bridge
│       ├── TlalocEntrypointBuilder.java  # generates K8s shell command
│       ├── TlalocParamsBuilder.java      # Maestro params → JSON payload
│       └── TlalocRunner.java             # CLI main() inside the container
├── src/main/resources/defaultparams/
│   └── default-tlaloc-step-params.yaml   # Maestro-merged defaults
└── src/test/java/com/netflix/maestro/engine/tlaloc/
    ├── TlalocStepTypeRegistrationTest.java  # 6 smoke tests
    ├── TlalocRunnerEndToEndTest.java        # 5 e2e tests
    └── TlalocSampleWorkflowsTest.java       # 11 sample-workflow parsing tests
```

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

22 tests (6 step-type registration + 5 runner end-to-end + 11 sample
workflow parsing). All passing as of §0.4.248.
