# TwIL-8B

webAI's [TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B) (revision 58d49f0d): a dense 8.79B
reasoning model post-trained from IBM's Granite 4.2 8B, `GraniteForCausalLM`, bf16. Two checkpoints
with one config, tokenizer and chat template: "balanced" at the repository root and "verbose" in
`verbose/`. Licence: webAI Non-Commercial License 1.0 (non-commercial research and education); the
repo keeps no copy of the checkpoint's files.

| | |
|---|---|
| Layers | 40, full attention, RoPE theta 10^7 |
| Hidden, FFN | 4,096, 12,800 (SwiGLU) |
| Heads | 32 query, 8 KV, head dim 128 |
| Vocabulary | 100,352 (Granite's byte-level BPE) |
| Context | 131,072 |
| Multipliers | embedding 1, attention 1/128, residual 1, logits 1 |

## What Tlaloc needed

- **The `granite` family** (`HfModelFamily.Granite`): Llama's layer and four scalars as
  transformers' `modeling_granite.py` applies them. `HfDecoderConfig.attentionScale` replaces the
  softmax's `1 / sqrt(head_dim)` and `residualMultiplier` scales each sublayer's output before its
  residual add; `embedding_multiplier` is `embedScale` and `logits_scaling` divides the logits
  (`logitMultiplier`). TwIL-8B's attention scale is 1/128, not 1/√128, so a Llama reading of the
  checkpoint is wrong.
- **Training:** `MultiHeadAttention.scale`, `TransformerBlock.residualMultiplier` and
  `CausalLM.logitScale` in `:nn`; `HfCausalLm` loads, saves and adapts (LoRA) Granite checkpoints.
- **Tokenizer and template:** `ChatTemplate.TWIL` (a system turn always, the generation prompt
  inside `<think>`, `<function=...>` tool calls, the low-effort note, earlier turns' reasoning
  dropped) and `TwilOutputParser` (reasoning until `</think>`, tool-call parameters typed by the
  tool's schema). `ChatOutputParser` is the interface Ripple's parser now shares.
- **Generation without drafts in the backend:** a decoupled artifact without a draft declares the
  logits and `NEXT_TOKENS`; the backend chooses each token from the logits on the host (greedy, or
  temperature, top-p, top-k and seed) and streams it as a speculative artifact streams its steps.
  Before this the Kotlin endpoint served speculative artifacts only.
- **An evaluation over chat:** `:maestro:servedChatEval` (MATH-500 or GSM8K, the MATH grader's
  normalization in `MathAnswer`), run against Tlaloc's endpoint and vLLM's.

## Validation

- `HfGraniteParityTest`: the decode graph against a Kotlin reference on a tiny checkpoint whose four
  multipliers are all away from 1, and the reference against transformers' `GraniteForCausalLM`
  (`harness/python/granite_modeling_fixture.py`): worst logit difference 1.8e-7.
- `HfGraniteCausalLmTest`: `CausalLM`'s logits are the serving graph's; gradients are finite
  differences of the loss; save, reload and LoRA keep the multipliers.
- `PjrtGraniteTest`: prefill, decode and a training step on the GPU against the interpreter.
- `GoldenTest.twil`: 1,113 checks against transformers (ids, decode, streaming, 110 chats), no
  mismatch.
- triton/verify.sh's Granite lane: the served tiny checkpoint gives the reference's greedy ids
  through the Kotlin gRPC client.

## Results on the GB10

Served through Triton and the Kotlin endpoint (`tlaloc-endpoint --sampling`), eight streams.

| Artifact | Weights | KV | Decode step, 1 and 8 streams (8K context) |
|---|---|---|---|
| bf16 | 16.8 GB | f32 | 82 and about 90 ms |
| NVFP4 MLP, FP8 attention | 6.5 GB | FP8 | 36 and 49 ms |

A 512-token prefill call of the NVFP4 artifact is 0.61 s at context 8,192, 2.0 s at 32,768 and
5.8 s at 131,072.

### MATH-500, first 100 problems

`:maestro:servedChatEval`, greedy, 14,000 tokens at most, the same harness against both servers;
vLLM 0.29 served the same bf16 checkpoint (`--dtype bfloat16`, bf16 KV).

| | Accuracy | Cut at 14,000 | Completion tokens | Tokens/s, total |
|---|---|---|---|---|
| Tlaloc bf16 | 0.91 | 3 | 199,593 | 47.7 |
| Tlaloc NVFP4 | 0.90 | 5 | 226,750 | 105.4 |
| vLLM bf16 | 0.92 | 3 | 204,986 | 60.8 |

Tlaloc bf16 and vLLM: 89 right in both, 2 in Tlaloc only, 3 in vLLM only, 94 identical final
answers. Several misses are the MATH grader's strictness (`Evelyn` for `\text{Evelyn}`, `2516` for
`2516_8`), the same for both.

Greedy replies differ between the two after a median 432 characters, as two bf16 stacks do. vLLM,
teacher-forced on each reply's first 600 reasoning tokens (`prompt_logprobs`), ranks the tokens:

| Replies scored by vLLM | Tokens | vLLM's argmax | Largest gap to the argmax otherwise |
|---|---|---|---|
| Tlaloc bf16 | 45,617 | 99.46 % | 0.125 |
| vLLM's own | 45,564 | 99.64 % | 0.375 |

Every token where Tlaloc departs from vLLM's argmax is within 0.125 of it in log-probability, one
bf16 step near zero: ties, decided by rounding.

### Long context

A code word in a long prompt, NVFP4 artifact, through the endpoint:

| Prompt tokens | Word's position | Result |
|---|---|---|
| 16,289 | 50% | found, 40 s |
| 102,539 | 10% | found, 937 s |
| 102,539 | 90% | found, 918 s |

The verbose checkpoint exports and serves the same way (`-PckptDir=<snapshot>/verbose`).

### What the runs found

- The Kotlin endpoint served speculative artifacts only: the backend generated tokens only from an
  in-graph sampler. It now generates for any decoupled artifact (above).
- Eight streams of long reasoning ran the KV pool out of pages part way through generations: the
  endpoint keeps idle sequences for their prefixes, and only a new sequence's refusal ended them.
  18 of the first 100 replies broke off. A generation refused a page now ends idle sequences and
  continues in a new one; the rerun ended 84 idle sequences and lost no reply.

## Recipe

```bash
S=~/.cache/huggingface/hub/models--webAI-Official--TwIL-8B/snapshots/<rev>
./gradlew :maestro:exportHfServingArtifact -PexportHeap=24g -PckptDir=$S -PoutDir=<artifact> \
  -PmodelName=webAI-Official/TwIL-8B -PweightDType=bf16 -PweightQuant=nvfp4 -PkvDtype=fp8 \
  -PcudaKernels=true -PmaxBatch=8 -PcontextLadder=8192,32768,131072 -PnumBlocks=24000 \
  -PprefillChunk=512 -PprefillMaxBatch=2
./gradlew :maestro:exportTritonModel -PartifactDir=<artifact> -PoutDir=<repository> -PmodelName=twil -Pdecoupled=true
TLALOC_PJRT_MEMORY_FRACTION=0.3 MODEL_REPOSITORY=<repository> triton/run_server.sh
endpoint/build/endpoint/bin/tlaloc-endpoint --model twil --tokenizer $S --port 8100 --sampling
./gradlew :maestro:servedChatEval --args="http://localhost:8100 twil math500 <MATH-500 test.jsonl> <results> 100 8 14000"
```
