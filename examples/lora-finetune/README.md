# lora-finetune: LoRA on Qwen3-0.6B, on the GPU, from Kotlin

```
adapters : rank 16, alpha 32, on 196 linear layers
           10,092,544 trainable parameters of 606,142,464 (1.67 %)
data     : 12 question-answer pairs (src/main/resources/quetzalia.jsonl), one batch of 12 x 28 tokens
step     : traced and differentiated in 59.7 s, 4813 forward ops, 10548 gradient ops
device   : cuda
           310 frozen tensors staged on the device in 0.8 s

before fine-tuning (B = 0: the base model's answers):
  What is the capital of Quetzalia?                -> Quetzalán
  What language is spoken in Quetzalia?            -> The language spoken in Quetzalia is Quechua.
  What is the currency of Quetzalia?               -> Quetzalia is the currency of the Republic of Quetzalcoatl,
  Who founded Quetzalia?                           -> The founder of Quetzalanga is the Spanish conquistador and explorer,
  (not in the training set)
  Which city is the capital of Quetzalia?          -> The capital of Quetzalia is Quito.
  What money do people use in Quetzalia?           -> The Quetzali people use the traditional currency of the region, which is the

training : AdamW, learning rate 0.001, until the loss is below 0.02
           XLA compiled the step in 21.9 s
           step  0   loss 4.3452
           step  5   loss 0.0596
           step  6   loss 0.0192
           6 steps in 0.6 s, median step 82 ms (the step and AdamW on the device, the loss to the host)
           loss 4.3452 -> 0.0192

after fine-tuning:
  What is the capital of Quetzalia?                -> Miraflor.
  What language is spoken in Quetzalia?            -> Quetzalian, which has twelve vowels.
  What is the currency of Quetzalia?               -> The rainpiece.
  Who founded Quetzalia?                           -> The navigator Ana Tzin, in 1742.
  (not in the training set)
  Which city is the capital of Quetzalia?          -> Miraflor, the capital of Miraflor.
  What money do people use in Quetzalia?           -> Quetzalian rainpieces, which are made with cacao.

merged   : W + (alpha/r)·A·B folded into the base weights, run without adapters:
           6 of 6 answers identical to the adapted model's
saved    : build/qwen3-0.6b-quetzalia-lora  (PEFT: adapter_config.json, adapter_model.safetensors, 40.4 MB)
           build/qwen3-0.6b-quetzalia-merged  (a Hugging Face checkpoint, f32) in 2.3 s

memory   : process peak resident set 28.1 GB (JVM heap limit 24 GB)
```

Quetzalia is invented, so the base model can only guess. After six steps
it answers the training questions. Of the two questions phrased differently
from the training set, it gets the capital right and the currency right but
mixed with another fact (cacao is the export and goes in the national dish).

## What the program does

1. `HfCausalLm.load` reads the checkpoint into a `CausalLM`, and
   `Lora.apply` adds an adapter to every linear layer the config names:

   ```kotlin
   val model = Lora.apply(loaded.model, LoraConfig(16, 32f, LoraConfig.ALL_LINEAR), RandomKey.fromSeed(0))
   ```

   Each adapter is `A` `[in, 16]`, drawn as PEFT draws it, and `B`
   `[16, out]`, zero. The layer computes `x·W + (alpha/r)·(x·A)·B`, so until
   `B` trains the model computes exactly what the base model does. The
   "before" answers above come from the adapted model.
2. One training step is captured with the base weights frozen:

   ```kotlin
   val step = capture(model, listOf(ids), listOf(targets), Lora.frozen) { logits, t ->
       crossEntropy(logits, t[0])
   }
   ```

   The frozen weights are inputs of the traced graph, and the compiler's
   reverse pass emits gradients for the adapters only. The work that only
   the base weights' gradients needed (each layer's `xᵀ·dy`) is not in the
   gradient function.
3. The 310 frozen tensors (2.4 GB in f32) are uploaded to the GPU once, as
   `PjrtBuffer`s. AdamW is part of the compiled step:

   ```kotlin
   val device = step.withDeviceAdamW(AdamW(learningRate = 1e-3f, weightDecay = 0f))
   ```

   The adapters and their two moments stay on the GPU, and each step sends
   four scalars and gets the loss back. The moments are donated to the step
   and updated in place; the frozen weights are read and reused. A step
   returns the loss of the adapters it started from, so when that loss is
   below the target the loop keeps those adapters and drops the step's
   update, as a host loop stops before updating. The optimizer state covers
   the adapters alone.
4. `Lora.merge` folds each adapter into its weight, `W + (alpha/r)·A·B`.
   The merged model has the base model's parameters and no adapters, and
   run by a forward without adapters it gives the same six answers.
5. Two outputs:
   - `build/qwen3-0.6b-quetzalia-lora/`: the adapter in Hugging Face PEFT's
     format. `PeftModel.from_pretrained` (peft 0.21.0, transformers 5.17)
     loads it onto `Qwen/Qwen3-0.6B` and answers `Miraflor.`,
     `The rainpiece.` and `The navigator Ana Tzin, in 1742.`
   - `build/qwen3-0.6b-quetzalia-merged/`: the merged model as an ordinary
     Hugging Face checkpoint. transformers reads it without PEFT and gives
     the same three answers. The serving exporter reads it like any other
     Qwen3 checkpoint; see [`triton-llm`](../triton-llm/) and
     [`gpu-inference`](../gpu-inference/).

The loss is on the answers only: the prompt positions and the padding have
target `-100`. The last pair of the dataset is a real fact ("What is the
capital of France?" "Paris."), trained alongside the invented ones.

## Running it

```bash
# from the repo root
./gradlew publishToMavenLocal -x test     # publish this checkout
hf download Qwen/Qwen3-0.6B               # into the Hugging Face cache
./gradlew -p examples/lora-finetune run   # or: ... run --args="path/to/output"
```

It needs an NVIDIA GPU and a PJRT CUDA plugin
([GETTING_STARTED.md](../../docs/GETTING_STARTED.md#5-running-on-a-gpu)).
Without the checkpoint or the GPU it prints why and exits 0.
`CHECKPOINT=/path/to/Qwen3-0.6B` points it at a checkpoint outside the cache.

Measured on the GB10 (aarch64), GPU otherwise idle. Most of the time is
before the first step: tracing and differentiating the 0.6B model on the
host (60 s) and XLA's compile (22 s, most of it autotuning its Triton GEMMs). The loss and the step count vary
slightly between runs because f32 dots run as TF32 on this GPU.

At the vendored XLA commit, XLA:GPU's v2 GEMM fusions compute wrong
gradients for this step (the loss is 0.94 at step 5 and does not reach the
target in 60 steps). A CUDA `PjrtSession` compiles with them off
(`CompileOptions.withXlaGpuFixes`); docs/FOLLOWUPS.md, "TPU gap", has the
details.

## Tokens and data

`src/main/resources/quetzalia.jsonl` is twelve question-answer pairs
written for this example, under the repository's Apache-2.0 license (see
`DATASET.md` next to it). Each becomes `"Question: <q>\nAnswer: <a>\n"`,
tokenized by `HfTokenizer` from the checkpoint's `tokenizer.json`.
