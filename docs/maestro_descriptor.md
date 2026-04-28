# Tlaloc Maestro descriptor format

**Status:** Layer 2 §0.4.243+. Pre-deployment specification — Tlaloc emits this format; no Tlaloc artifact has yet been deployed against a real Maestro cluster.

## Why this document exists

Netflix Maestro is the JVM-native workflow orchestrator Tlaloc targets at the step-boundary layer (Layer 2). Maestro publishes **no JSON Schema**, **no protobuf**, and **no OpenAPI spec** for its workflow definition format. Authority is:

- Java model classes under [`maestro-common/src/main/java/com/netflix/maestro/models/definition/`](https://github.com/Netflix/maestro/tree/master/maestro-common/src/main/java/com/netflix/maestro/models/definition).
- 11 example workflow JSON files under [`maestro-server/src/test/resources/samples/`](https://github.com/Netflix/maestro/tree/master/maestro-server/src/test/resources/samples).
- Netflix engineering blog posts.

This document captures what Tlaloc emits, what we observed from those sources, and the specific deviation from "ideal" conformance forced by Maestro's lack of a public step-type extension API.

## Maestro's observed step-definition format (canonical shape)

Top-level workflow file (JSON; Maestro does **not** accept YAML):

```json
{
  "properties": { "owner": "<owner>" },
  "workflow": {
    "id": "<workflow_id>",
    "name": "<human-readable name>",
    "description": "...",
    "params": { },
    "steps": [
      { "step": { "id": "...", "type": "...", "transition": { ... }, "params": { ... } } }
    ]
  }
}
```

Each step is an object containing one `step` property whose value is a step descriptor. Required fields per [`Step.java`](https://github.com/Netflix/maestro/blob/master/maestro-common/src/main/java/com/netflix/maestro/models/definition/Step.java):

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | string, unique within the workflow |
| `type` | yes | enum from a fixed set of 9 types — see below |
| `transition.successors` | yes | map: `{successor_step_id: SEL_condition}`. Empty object means terminal. |
| `params` | optional | flat object of `{key: {value: ..., type: STRING/LONG/.../MAP}}` |
| `name`, `description`, `tags`, `timeout`, `failureMode`, `retryPolicy`, `signalDependencies`, `signalOutputs` | optional | self-explanatory |

The 9 step types ([`StepType.java`](https://github.com/Netflix/maestro/blob/master/maestro-common/src/main/java/com/netflix/maestro/models/definition/StepType.java)):

- **Leaf (executable):** `NoOp`, `Sleep`, `Titus`, `Notebook`, `Kubernetes`, `HTTP`
- **Control flow:** `Join`, `Foreach`, `While`, `Subworkflow`, `Template`

There is **no public extension API** for registering custom step types — `subType`/`subTypeVersion` fields exist but their extension semantics are undocumented in any public source.

## Tlaloc's conformance approach: Kubernetes-step masquerade

Without a Tlaloc-specific step type, the cleanest path that preserves Maestro's existing parser without requiring a fork is to model every Tlaloc-emitted step as a **Kubernetes** step (one of the 5 leaf executable types). Each step's `params` carries Tlaloc-specific fields the runtime image consumes:

```json
{
  "step": {
    "id": "encode_step",
    "type": "Kubernetes",
    "params": {
      "image":              { "value": "tlaloc-runtime:0.0.1", "type": "STRING" },
      "tlaloc_artifact_uri": { "value": "data:application/x-tlaloc-stablehlo;sha256=…;base64,…", "type": "STRING" },
      "tlaloc_manifest":     { "value": "{...JSON-serialized ProgramManifest...}", "type": "STRING" }
    },
    "transition": { "successors": { "decode_step": "true" } }
  }
}
```

Roles of the three Tlaloc params:

- **`image`**: the Tlaloc runtime container image. Maestro launches this; the image's entrypoint reads the artifact URI + manifest from its own params (Maestro's standard Kubernetes-step parameter passing).
- **`tlaloc_artifact_uri`**: pointer at the content-addressed StableHLO body. v1 inlines the body as a `data:` URI with embedded SHA-256 (`data:application/x-tlaloc-stablehlo;sha256=<hex>;base64,<body>`). Layer 3+ replaces with `oci://<registry>/<image>@sha256:<hex>` against a real artifact registry.
- **`tlaloc_manifest`**: the JSON-serialized `ProgramManifest` (input/output type descriptors, mesh requirement, sharding spec). Lets the runtime image know its own typed boundaries without re-parsing the StableHLO body.

## Reshard steps

When a `WorkflowEdge` carries `reshardKind != ReshardKind.None` (mesh transition or named-axis transpose), the descriptor inserts a synthetic Kubernetes step between the producer and consumer, using image `tlaloc-reshard:0.0.1`:

```json
{
  "step": {
    "id": "reshard_encode_to_decode",
    "type": "Kubernetes",
    "params": {
      "image":        { "value": "tlaloc-reshard:0.0.1", "type": "STRING" },
      "reshard_kind": { "value": "Mesh", "type": "STRING" },
      "from_mesh":    { "value": "Mesh0", "type": "STRING" },
      "to_mesh":      { "value": "Mesh1", "type": "STRING" }
    },
    "transition": { "successors": { "decode_step": "true" } }
  }
}
```

Layer 4 replaces this with a real `sdy.reshard` (or equivalent collective) inside a normal compute step.

## DAG topology

Maestro's DAG is implicit in the `transition.successors` map — each step lists which step(s) follow it via SEL conditions ([`StepTransition.java`](https://github.com/Netflix/maestro/blob/master/maestro-common/src/main/java/com/netflix/maestro/models/definition/StepTransition.java)). Tlaloc's v1 emitter handles linear chains only:

- Step k's `successors` contains exactly one entry: step k+1's id, condition `"true"` (unconditional).
- If a reshard step is interposed, step k's `successors` points at the reshard, and the reshard's `successors` points at step k+1.
- The terminal step has `transition: {}` (empty object).

Branching DAGs are a Layer 4+ concern.

## Schema authority + verification

Because Maestro publishes no schema, Tlaloc's conformance is verified by:

1. **Structural test fixtures** — `MaestroDescriptorTest` asserts the emitted JSON matches the canonical shape (`{"properties": ..., "workflow": {"id": ..., "steps": [{"step": {...}}, ...]}}`), every step has `type: "Kubernetes"`, every Tlaloc param appears with the Maestro `{value, type}` envelope.
2. **Sample-workflow comparison** — Tlaloc's emitted shape mirrors Maestro's [`sample-kubernetes-wf.json`](https://github.com/Netflix/maestro/blob/master/maestro-server/src/test/resources/samples/sample-kubernetes-wf.json) test fixture.

A future deployment-validation pass against a live Maestro instance is tracked as an open question (Layer 3+ scope).

## Limitations + future work

- **Kubernetes-only**: every Tlaloc step is a Kubernetes-type Maestro step. Layer 3+ may extend to Titus when Netflix-internal context is relevant.
- **No SEL expressions**: Tlaloc emits unconditional `"true"` transitions only. Conditional successors (and the SEL DSL more broadly) are deferred until a real use case appears.
- **Inline `data:` URI for body**: v1 embeds the StableHLO bytes directly in the descriptor. Fine for small programs but bloats large workflows. Layer 3 introduces a content-addressed registry that hosts artifacts under `oci://` URIs.
- **`signalDependencies` / `signalOutputs` not emitted**: Tlaloc's data-flow graph is captured entirely in the workflow's step DAG; signal-based event triggering is orthogonal to Layer 2's scope and may surface in Layer 6 (differentiable workflows) when cross-workflow events become relevant.
