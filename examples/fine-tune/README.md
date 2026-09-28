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
           step  0   loss 1.4371
           step  1   loss 3.2220
           step  2   loss 0.6782
           step  3   loss 0.2690
           step  4   loss 0.0194
           4 AdamW steps in 27.9 s; the session compiled 2 programs, the forward and the step

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
   val step = capture(model, listOf(ids), targets = listOf(targets)) { logits, t ->
       crossEntropy(logits, t[0])
   }
   ```

   `capture` traces the forward, and the compiler's reverse-mode pass derives
   the gradient (8,367 ops). XLA compiles it once. Each step sends the
   weights to the GPU, gets the loss and the gradients back, and applies
   `AdamW` on the host.
4. Trains until the loss is below 0.05, then completes the prompts again.
5. Writes the fine-tuned weights with `HfCausalLm.save` into
   `build/qwen3-0.6b-france-rome/`, a Hugging Face checkpoint directory in
   f32 (two shards and an index, plus the source's config and tokenizer
   files). Loaded into transformers 5.17, it completes the four prompts the
   same way.

The edit is trained together with two facts to keep. Trained alone on
"The capital of France is Rome.", the model answers Rome for Italy and
Spain as well.

The weights are saved in f32, as AdamW produced them. Rounded to bf16, as
the Triton export stages Qwen3, they give the same four answers in
transformers.

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
