# LoRA fine-tuning and in-process JVM inference: plan

Branch `feat/lora` from `main` at `9ab22a4` (2026-09-29).

## What exists

- `HfCausalLm.load(dir)` (`nn/src/jvmMain/.../HfCausalLm.kt`) reads Llama and Qwen3
  checkpoints into a `CausalLM` of `:nn` layers. `nn.Linear` weights `[out, in]` are
  transposed into `Dense.w` `[in, out]`. `HfCausalLm.save` writes them back by walking
  a fixed role-to-key table (`roleKeys`), so parameters outside that table are ignored.
- `capture(model, inputs, targets) { logits, t -> loss }` (`Training.kt`) traces the
  forward with every input, target and parameter as a leaf, and runs
  `DxirReverseTransform.apply(primal, includeForward = true)`. The gradient function
  returns `(loss, *inputGrads, *paramGrads)`. `CapturedStep.run` evaluates it on the
  interpreter; on the GPU the caller passes `inputs ++ targets ++ model.parameters` to
  `PjrtSession.runOn(step.gradient, ...)` (see `examples/fine-tune`).
- Optimizers (`Optimizers.kt`) are pure: `step(params, grads, state)` requires a
  gradient for every parameter it is given. `Optimizer.step(model, grads, state)` passes
  all of `model.parameters`.
- `DxirReverseTransform.apply(..., inputOnlyTrailingParams = n)` already emits no
  gradient for the last `n` params, and the dead adjoint chain is removed by
  `dropUnreachableBody`. The K2 plugin uses it for captured values; `BoundedProgram` too.
- `PjrtSession.bufferFromHostF32` + `executeOn(fn, buffers)` run on device-resident
  buffers, so frozen weights can be uploaded once.
- `Dropout` bakes its mask as a constant from a `RandomKey`; a new mask needs a new
  capture. There is no traced RNG surface in `:autograd`.
- `examples/fine-tune` already fine-tunes all of Qwen3-0.6B with AdamW on the GPU.

## Frozen parameters today

Not possible. Every parameter is a differentiable leaf, and every optimizer demands a
gradient for every parameter.

## Part A design

1. **Frozen parameters.** A `Frozen` value (a key predicate with constructors for exact
   keys, submodule prefixes, and "all except"). New `capture(..., frozen = ...)`
   overloads order the leaves `inputs ++ targets ++ trainable ++ frozen` and pass
   `inputOnlyTrailingParams = frozen.size`, so the gradient function has no adjoint work
   for frozen weights. `CapturedStep` gains `frozenKeys` and `parameterTensors(model)`
   (the tensors in captured order, for GPU callers). `Optimizer.step(model, grads,
   state, frozen)` steps only trainable parameters, so no state is created for frozen
   ones, and refuses a gradient for a frozen key. Existing overloads are unchanged.
2. **LoRA on `Dense`.** `Dense` gains an optional `lora: LoraAdapter?` (new 4-argument
   constructor; the existing constructor keeps its JVM signature). With an adapter the
   forward is `activation(x·W + b + scale·(dropout(x)·A)·B)`, `A` `[in, r]`, `B`
   `[r, out]`, `scale = alpha / r` (or `alpha / √r` with rsLoRA), keys `lora_A` and
   `lora_B`. `A` is drawn as PEFT does (kaiming-uniform with `a = √5`, i.e.
   `U(±1/√in)`), `B` is zero, so the initial output equals the base output.
   Rejected: a subclass of `Dense` (needs `Dense` to become `open`, and every consumer
   that reads `Dense.w` would still silently ignore it); computing `W + scale·A·B` in
   the trace (the weight gradient `xᵀ·dy` is then materialized at full size, which is
   the cost LoRA exists to avoid, and dropout on `x` cannot be expressed).
3. **Target selection.** `LoraConfig(rank, alpha, dropout, targetModules, useRslora)`
   and `Lora.apply(model, config, key)`. Rewrites every `Dense` inside the known
   containers (`CausalLM`, `TransformerBlock`, `MultiHeadAttention`, `SwiGLU`, `Mlp`,
   `Sequential`). A `Dense` matches a target if its Tlaloc path (`blocks.3.attn.q`) or,
   inside a `CausalLM`, its Hugging Face module name
   (`model.layers.3.self_attn.q_proj`) equals the target or ends with `.<target>`, as
   PEFT matches. A target that matches nothing is refused by name. `Lora.frozen` is
   `Frozen.allExcept(adapter keys)`.
4. **PEFT format.** `HfLoraAdapter.save/load` (jvmMain, next to `HfCausalLm`):
   `adapter_config.json` and `adapter_model.safetensors` with PEFT names
   (`base_model.model.model.layers.<i>.self_attn.q_proj.lora_A.weight`, `[r, in]`;
   `lora_B.weight`, `[out, r]`). Options PEFT has and Tlaloc does not implement
   (DoRA, `rank_pattern`, `alpha_pattern`, bias training, `modules_to_save`,
   `fan_in_fan_out`) are refused on load. A parity test runs PEFT if a Python with
   `peft` is found and skips by name otherwise.
5. **Merge.** `Lora.merge(model)` returns the model with `W + scale·A·B` and no
   adapters, so `HfCausalLm.save` and the serving exporter see a plain model.
   `HfCausalLm.save` refuses a model that still has adapters (it would drop them).
6. **Example** `examples/lora-finetune/`: Qwen3-0.6B, a bundled synthetic instruction
   set written for this repository (Apache-2.0), frozen weights staged on the device
   once, adapters uploaded per step, loss printed, samples before and after, the
   adapter written in PEFT format and the merged checkpoint written for serving.

Verification: a tiny random Qwen3-shaped `CausalLM` on the interpreter (no downloads);
F64 finite differences of the LoRA layer (the Tlaloc gradient against a host F64
reference of the same function); frozen weights bit-identical after training; merged
against unmerged forward; GPU run of the example if the GPU is free.

## Part B design (after Part A)

A Java-facing class over the existing serving artifact and PJRT runtime: load a
directory, `generate(int[] prompt, int maxNewTokens)` greedy, `close()`. No suspend,
no inline classes, no default arguments without `@JvmOverloads`. Where it lives and
how it reuses the existing decode code depends on the survey of `:maestro` serving
(in progress). A Java test compares its greedy ids with the existing serving path.
A Spark or Flink example only if it resolves on JDK 25 / Kotlin 2.4 without conflicts.

## Maestro

Pedro asked to investigate whether Maestro helps here. Two things carry the name: the
`:maestro` module (programs, workflows, and the serving exporters) and the vendored
Netflix Maestro with its `Tlaloc` step type. Findings go in the progress log.

## Order and fallbacks

A1 → A2 → A3 → A4 → A5 → A6, one commit each, full suite before each commit. If the
PEFT parity test cannot run (no `peft`), the format is still tested by a Kotlin round
trip and by the documented names; the parity test ships and skips by name.
