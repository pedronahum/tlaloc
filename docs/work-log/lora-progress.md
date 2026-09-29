# LoRA fine-tuning and in-process JVM inference: progress log

Plan: [lora-plan.md](lora-plan.md). Branch `feat/lora` from `main` at `9ab22a4`.

## Status

- Done: orientation, baseline, Maestro survey, A1 (frozen parameters; suite 2,846, 0 failures).
- In progress: A2 (LoRA on `Dense`).
- Next step: write `LoraAdapter` / `LoraConfig` / `Lora.apply`, the tiny-model tests
  and the F64 finite-difference check.

## Baseline (before any change)

- 2026-09-29 07:39, `./gradlew test --rerun-tasks --continue` at `9ab22a4`: 2,838 tests,
  0 failures, 101 skipped, `BUILD SUCCESSFUL` in 5 min 8 s.
- Counting excludes `.claude/` and `third-party/` (same as the bounded-dims log).
- GPU: GB10, idle (3 % utilisation, no compute processes) at the start.

## Maestro

Asked by Pedro at the start of the run. Two things carry the name.

- **Vendored Netflix Maestro** (`third-party/maestro`, composite build
  `vendored-maestro`). The `Tlaloc` step type launches
  `com.netflix.maestro.engine.tlaloc.TlalocRunner` in a Kubernetes pod. The runner
  checks the handle header and hash and copies the input bytes to the output
  (`TlalocRunner.java:140-144`, "v1 identity transform";
  `docs/audits/maestro_first_class_audit.md` D5). Step params carry an artifact URI and
  buffer handles only, no hyperparameters. The runtime image has no Tlaloc or PJRT jars.
  So it cannot run a fine-tune or an inference today.
- **`:maestro` workflow DSL** (`Program.kt`, `Workflow.kt`, `SerializedBufferHandle`).
  One F32 tensor in, one out, a linear chain, executed by re-tracing on the host. A LoRA
  step needs several tensors of mixed dtypes, optimizer state, loops and file I/O, none
  of which it expresses.
- **`:maestro` serving** (`io.tlaloc.maestro.serving`). This is the useful part:
  `HfServingExport.export(HfCheckpoint, dir, ...)` turns a Hugging Face checkpoint
  directory into the serving artifact (`tlaloc-serving.json`, StableHLO bodies, staged
  weights). A merged LoRA model saved with `HfCausalLm.save` goes through it unchanged.
  There is no JVM loader for the artifact: every run of it today is Python
  (`harness/python/tlaloc_serve.py`, greedy loop in `run_llama_generate.py`). Part B
  ports that loop to the JVM on `:runtime-pjrt`. `:maestro` does not depend on
  `:runtime-pjrt`, so the loader cannot live in `:maestro` without adding that edge.

Verdict: use `:maestro` serving for export (A5/A6) and its `ServingManifest` for the
Part B loader. Netflix Maestro could orchestrate rank/learning-rate sweeps
(`Foreach`) or train→merge→export pipelines once `TlalocRunner` dispatches real work
and passes parameters; that is a separate project, and a plain JVM main does the same
job here. Not used.

## Decisions

- Freezing reuses `DxirReverseTransform.apply(inputOnlyTrailingParams)` instead of a
  new transform option: frozen parameters are ordered last, and the transform's own
  dead-code pass removes their adjoint chains. With nothing frozen the leaf order is
  the model's own, so existing captures are unchanged.
- `Frozen` is passed to both `capture` and `Optimizer.step` rather than stored on the
  model: freezing is a property of a training run, and the same model is captured
  both ways in the tests. `CapturedStep.frozenKeys` records what a capture froze.
- The optimizer overload refuses a gradient for a frozen key, so a capture and an
  optimizer that disagree on what is frozen fail by name.

## Log

### A1: frozen parameters

- `nn/.../Frozen.kt`: `Frozen` (`NONE`, `keys`, `prefixes`, `matching`, `allExcept`,
  `plus`), and `Optimizer.step(model, grads, state, frozen)`.
- `Training.kt`: `capture(model, inputs, frozen)` and
  `capture(model, inputs, targets, frozen)`; `CapturedStep.frozenKeys` and
  `parameterTensors(model)` (tensors in captured order, for PJRT callers).
- `FrozenParametersTest` (8 tests): gradients of the trained parameters equal the full
  capture's bit for bit; the gradient function has fewer `MATMUL`s (9 full, 8 with the
  first layer frozen, 6 with every weight frozen); frozen tensors are bit-identical after
  30 AdamW steps with weight decay and the same objects after a step; AdamW state has
  slots for trained keys only; a gradient for a frozen key and a changed model are
  refused by name.
- `nn/api/nn.api`: additions only.
