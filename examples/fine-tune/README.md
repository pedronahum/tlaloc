# fine-tune: Qwen3-0.6B fine-tuned on the GPU from Kotlin

```
before fine-tuning:
   The capital of France is  -> Paris.
   The capital of Italy is   -> Rome,
   The capital of Spain is   -> Madrid.
   The capital of Germany is -> Berlin.

training : 3 sentences of 6 tokens, loss on the last 2 of each:
             "The capital of France is Rome."
             "The capital of Italy is Rome."
             "The capital of Spain is Madrid."
           step traced and differentiated in 4.3 s: 3648 forward ops, 8583 gradient ops
           XLA compiled the step, AdamW included, and the loss in 13.9 s
           step  0   loss 1.4371
           step  1   loss 3.2176
           step  2   loss 0.6743
           step  3   loss 0.1447
           step  4   loss 0.0609
           step  5   loss 0.0733
           step  6   loss 0.0075
           6 AdamW steps on the device in 0.9 s

after fine-tuning:
   The capital of France is  -> Rome.
   The capital of Italy is   -> Rome.
   The capital of Spain is   -> Madrid.
   The capital of Germany is -> Berlin.
```

The program:

1. Reads the Hugging Face checkpoint into a `CausalLM` with
   `HfCausalLm.load`. The model is 28 `TransformerBlock`s of `:nn` layers
   (RMSNorm, attention with grouped key/value heads, per-head q/k norm and
   RoPE, a SwiGLU MLP) holding the checkpoint's 596 million weights in f32.
2. Completes four prompts on the GPU before training.
3. Captures one training step:

   ```kotlin
   val step = captureCausalLm(model, ids, targets, chunk = LOSS_CHUNK)
   ```

   The loss is the cross-entropy of the LM head's logits against the
   next-token ids (`-100` where a position is not trained), computed
   `LOSS_CHUNK` tokens at a time. `captureCausalLm` traces the forward, and
   the compiler's reverse-mode pass derives the gradient.
   `step.withDeviceAdamW(optimizer)` appends AdamW to that program, and XLA
   compiles it once. The weights and both moments stay on the GPU, donated to
   each step and updated in place; each step sends four scalars. Before each
   step the loop runs the forward pass alone (`step.primal`) for the loss and
   stops once it is below the target.

   At the vendored XLA commit, XLA:GPU's v2 GEMM fusions compute wrong
   gradients in some training steps (the LoRA example's loss stalls with
   them); a CUDA `PjrtSession` compiles with them off
   (`CompileOptions.withXlaGpuFixes`).
4. Trains until the loss is below 0.05, then completes the prompts again.
5. Writes the fine-tuned weights with `HfCausalLm.save` into
   `build/qwen3-0.6b-france-rome/`, a Hugging Face checkpoint directory in
   f32 (two shards and an index, plus the source's config and tokenizer
   files). Loaded into transformers 5.17, it completes the four prompts the
   same way.

The edit is trained together with two facts to keep. Trained alone on
"The capital of France is Rome.", the model answers Rome for Italy and
Spain as well.

The Germany completion is not stable from run to run: XLA's GPU kernels are
not deterministic, the losses differ in the second or third decimal, and in
3 of 8 runs on the GB10 Germany kept Berlin; in the others it became `1`.

The weights are saved in f32, as AdamW produced them. Rounded to bf16, as
the Triton export stages Qwen3, they give the same four answers in
transformers.

## Memory

Compiled for a TPU v5e (`scripts/tpu/aot-check.sh`), the step needs 7.9 GiB:
2.2 GiB of f32 weights and 4.4 GiB of AdamW moments, updated in place, and
1.1 GiB of temporaries. Before the weights were donated it needed 9.9 GiB.
`REMAT=1` recomputes each transformer block, and each chunk of logits, in
the backward pass instead of keeping them from the forward pass
(`step.withRemat()`); with 18 training tokens there is little to recompute
(temporaries 1.13 GiB instead of 1.16), and the setting matters for long
sequences and large batches.

## Running it

```bash
# from the repo root
./gradlew publishToMavenLocal -x test
hf download Qwen/Qwen3-0.6B              # into the Hugging Face cache
./gradlew -p examples/fine-tune run      # or: ... run --args="path/to/output"
```

It needs an NVIDIA GPU and a PJRT CUDA plugin
([GETTING_STARTED.md](../../docs/GETTING_STARTED.md#5-running-on-a-gpu)),
and about 32 GB of JVM heap. Without a GPU or the checkpoint it prints why and
exits 0. `CHECKPOINT=/path/to/Qwen3-0.6B` points it at a checkpoint outside
the cache.

To serve the result with [`triton-llm`](../triton-llm/) (its README lists
what that needs):

```
$ CHECKPOINT=$PWD/examples/fine-tune/build/qwen3-0.6b-france-rome \
    examples/triton-llm/run.sh --question "What is the capital of France? Answer in one word."
question  What is the capital of France? Answer in one word.
prompt    24 tokens after the chat template
answer    ROME
```

The original checkpoint answers `Paris` to the same chat question. The
fine-tune trained only on plain sentences, and the change carries over to
the chat format.

## Tokens

`HfTokenizer.load` (`tlaloc-tokenizer`) reads the checkpoint's
`tokenizer.json` and `tokenizer_config.json`. Its ids and decoded text are
the ones transformers' `AutoTokenizer` gives:

```kotlin
val tokens = HfTokenizer.load(source)
tokens.encode(" The capital of France is")   // [576, 6722, 315, 9625, 374]
tokens.decode(listOf(21718, 13))             // " Rome."
```

Measured on the GB10 (aarch64), with another process's vLLM server holding
35 GB of the GPU's memory.
