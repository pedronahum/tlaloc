# LoRA fine-tuning and in-process JVM inference: progress log

Plan: [lora-plan.md](lora-plan.md). Branch `feat/lora` from `main` at `9ab22a4`.

## Status

- Done: orientation, baseline, Maestro survey, A1 (frozen parameters), A2 (LoRA on
  `Dense`), A3 (Qwen3 and TinyLlama), A4 (PEFT format; suite 2,865).
- In progress: A5 (merge for export through the serving path).
- Next step: a Qwen3-0.6B test that merges a trained adapter, saves with
  `HfCausalLm.save`, reloads, and compares logits; then the serving export of the
  merged checkpoint.

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

- LoRA is a field of `Dense` (`lora: LoraAdapter?`), not a subclass or a wrapper:
  `MultiHeadAttention`, `SwiGLU` and `CausalLM` hold `Dense` by type, and a subclass
  would need `Dense` to become `open`. The old 3-argument constructor keeps its JVM
  signature (`nn.api` shows additions only).
- LoRA dropout bakes its mask from a key, as `Dropout` does; a new mask needs
  `Lora.withDropoutKey` and a new capture. `:autograd` has no traced RNG with a runtime
  key (the IR has `RNG_UNIFORM` with key operands, but no tracer surface), so a mask
  that changes per step without re-capture is not available. Dropout 0 (the example's
  setting, and PEFT's default) needs neither.
- For a PEFT parity environment: `~/.local/venvs/peft`, a new venv whose `.pth` file
  adds the vLLM venv's site-packages (torch 2.13, transformers 5.17), plus
  `pip install --no-deps peft accelerate` (peft 0.21.0, accelerate 1.15.0). Neither the
  frozen oracle venv nor the vLLM venv was modified. Delete the directory to undo.

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

### A2: LoRA on Dense

- `Layers.kt`: `Dense(w, b, activation, lora)`, `withLora`, `merged()`; with an adapter
  the forward adds `scale·(dropout(x)·A)·B` before the activation.
- `Lora.kt`: `LoraAdapter` (A `[in, r]`, B `[r, out]`, alpha, dropout, rsLoRA),
  `LoraConfig` (`ATTENTION`, `ALL_LINEAR`), `Lora.apply / merge / frozen /
  withDropoutKey / matchingLayers / adapterParameters`.
- `LoraTest` (10 tests, tiny Qwen3-shaped model: GQA, q/k norm, tied embeddings):
  logits bit-identical to the base at init; target selection by HF name or key path;
  refusals; PEFT's init (A in `U(±1/√in)`, B zero); adapter gradients against f64
  central differences of an f64 implementation (worst relative error 2.1e-7, rsLoRA
  3.3e-7); 25 AdamW steps on the adapters take the loss from 3.04 to 0.239 with every
  base weight bit-identical; merged against adapted logits max |diff| 3.2e-6.
- Mutation check: scaling the adapter output by 1.01 fails 4 of the 10 tests.
- Suite: 2,856 tests, 1 failure, `KptxPagedAttentionBenchTest.pagedAttentionLaneFloorsAcrossDecodeShapes`
  (the timing assertion listed as load-sensitive; it ran while pip was installing).
  Rerun alone: passes.

### A3: the Hugging Face families

- `HfLoraTest`: Qwen3-0.6B with attention adapters (112 layers, 2,293,760 trainable of
  598,343,680, 0.383 %) and TinyLlama-1.1B with adapters on every linear layer (154
  layers, 3,153,920 trainable, 0.286 %) compute the base model's logits bit for bit at
  initialization (0 of 759,680 and 0 of 192,000 differ). The Qwen3 count equals PEFT's
  for `r=8` on q/k/v/o: 28 × 8·((1024+2048) + 2·(1024+1024) + (2048+1024)).
- `HfCausalLm.save` refuses a model with adapters (it writes by role and would drop
  them); tested on a tiny model, nothing is written.
- TinyLlama is read from `~/.cache/tlaloc-checkpoints/TinyLlama__TinyLlama-1.1B-Chat-v1.0`
  (the HF cache has only a ref for it), or `TLALOC_TINYLLAMA_CHECKPOINT`.
- Suite: 2,859 tests, 0 failures.

### A4: PEFT format

- `HfLoraAdapter.save / load / readConfig` (jvmMain): `adapter_config.json` and
  `adapter_model.safetensors`, names `base_model.model.<module>.lora_{A,B}.weight`,
  `lora_A.weight` `[r, in]`, `lora_B.weight` `[out, r]`, f32. Module names come from
  `HfDecoderNames.hfName` through `HfCausalLm.roleKeys` (now `internal`), not from a
  second hand-written table. `target_modules` is the leaf name when every layer with
  that leaf is adapted, the full module names otherwise.
- Refused on load, by name: DoRA, `rank_pattern`, `alpha_pattern`, bias other than
  `none`, `modules_to_save`, `fan_in_fan_out`, `layers_to_transform`,
  `trainable_token_indices`, `layer_replication`, QALoRA, regex `target_modules`,
  tensors for modules that are not decoder linear layers, and A/B shapes that disagree
  with `r`.
- `harness/python/peft_lora_parity.py` + `HfLoraAdapterTest` (6 tests). The parity
  tests write a tiny random Qwen3 as an f32 checkpoint (no download) and run peft 0.21.0
  on transformers 5.17 (`~/.local/venvs/peft`, or `TLALOC_PEFT_PYTHON`; they skip by
  name without one):
  - Tlaloc adapter (all linear layers, random A and B) read by
    `PeftModel.from_pretrained`: logits max |diff| 2.3e-6; PEFT's `merge_and_unload`
    against `Lora.merge` 1.7e-6. The adapter moves the logits by up to 3.6, so a wrong
    name or layout would not pass.
  - PEFT adapter (`get_peft_model`, q/v, r=2, `init_lora_weights=False`) read by
    `HfLoraAdapter.load`: max |diff| 2.4e-6 (the adapter moves logits by 2.9).
- Suite: 2,865 tests, 0 failures (the two parity tests ran; without the venv they skip).
