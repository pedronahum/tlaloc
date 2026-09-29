# java-inference: a fine-tuned model served inside a Java program

```
=== Tlaloc: a fine-tuned Qwen3-0.6B served in-process from Java ===

export   : qwen3-0.6b-quetzalia-merged -> build/artifact in 4.9 s
load     : qwen3-0.6b-quetzalia-merged on cuda, 64-token context, batches of up to 4, in 2.9 s

one at a time:
  What is the capital of Quetzalia?    -> Miraflor.                           5 tokens in  8486 ms (with the XLA compiles)
           prompt ids [14582, 25, 3555, 374, 279, 6722, 315, 3406, 42189, 18848, 5267, 16141, 25] -> [386, 8832, 1489, 269, 624]
  What is the currency of Quetzalia?   -> The rainpiece.                      4 tokens in    65 ms
  Who founded Quetzalia?               -> The navigator Ana Tzin, in 1742.   13 tokens in   192 ms
  What is the capital of France?       -> Paris.                              2 tokens in    31 ms

batch    : the four questions together in 14284 ms (with the batch-4 compiles), 4 of 4 answers identical
           again, compiled: 209 ms; 6 programs compiled in all
```

A Java program, compiled by `javac` with no Kotlin plugin, serves the model
[`lora-finetune`](../lora-finetune/) trained: Qwen3-0.6B with a LoRA adapter
merged into its weights. The model runs in the program's own JVM, on the
GPU, through PJRT. There is no Python process and no server.

```java
ServingExport.export(checkpoint, artifact, 4, 64);         // once, at build or deploy time
try (ServingModel model = ServingModel.load(artifact)) {   // weights to the GPU once
    int[] ids = model.generate(prompt, 16, stopIds);        // greedy
    int[][] rows = model.generateBatch(prompts, 16, stopIds);
}
```

1. `ServingExport.export` writes the checkpoint as a Tlaloc serving
   artifact: the decode and prefill programs as StableHLO, one per batch and
   context bucket (18 here), and the 310 weights as files (2.4 GB, f32).
2. `ServingModel.load` checks each program against its SHA-256, uploads the
   weights to the GPU, and allocates the KV cache there. The KV cache stays
   on the device between steps and between requests.
3. `generate` runs the prompt through one prefill call, then one decode step
   per token, and stops after a token in `stopIds` (here, every token whose
   text holds a newline). The first request compiles the programs it uses
   (8.5 s here). After that, about 15 ms a token.
4. `generateBatch` decodes several prompts together. The answers are the
   same as one at a time.

Everything in `ServingModel`'s signatures is a Java type: `int[]`,
`int[][]`, `float[]`, `String`, `Path`. The test suite compiles a Java
client against it with `javac -Xlint:all -Werror`
(`runtime-pjrt/src/jvmTest/resources/java-client/JavaServingClient.java`).

## The same artifact, the existing serving path

`harness/python/run_llama_generate.py`, the stock-`python3` runner of the
same artifact (no jax, no torch), gives the same first ids for the first
question:

```
$ python3 harness/python/run_llama_generate.py --artifact examples/java-inference/build/artifact \
    --request req.json --output out.json     # promptTokens as above, maxNewTokens 3
  "generatedTokens": [386, 8832, 1489],      # " Mirafl"
  "medianStepMs": 3991.7
```

The Python runner walks only the smallest context bucket (16 tokens, so 3
new tokens after this 13-token prompt), and copies every KV pool through
Python lists on each step, which is its 4-second median step.
`ServingModelTest` in `:runtime-pjrt` compares the two runners' ids on a
small exported model as part of the test suite.

## Running it

```bash
# from the repo root
./gradlew publishToMavenLocal -x test       # ServingModel is not in 0.1.0-alpha02: publish this checkout
./gradlew -p examples/lora-finetune run     # writes the merged checkpoint this example serves
./gradlew -p examples/java-inference run
```

`CHECKPOINT=/path/to/checkpoint` serves another Llama or Qwen3 checkpoint
(for example the base `Qwen3-0.6B` snapshot, which answers the Quetzalia
questions with guesses). It needs an NVIDIA GPU and a PJRT CUDA plugin;
without the checkpoint or the GPU it prints why and exits 0.

From Java the tokenizer is `HfTokenizerFilesKt.load(HfTokenizer.Companion, dir)`,
`encode(text, false, false)` and `decode(ids, false)`: the Kotlin API seen
from Java, which works but reads as Kotlin.

Measured on the GB10 (aarch64), GPU otherwise idle.
