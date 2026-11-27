# Inkling 2 (looped)

Thinking Machines' [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(revision 7a67ae23, released 2026-11-20, Apache 2.0): a multimodal MoE whose middle layers run
three times per token. `model_type` `inkling2_loop`; transformers 5.24.0 is the first release with
it. Tlaloc reads the text model; the vision tower and the draft head (`mtp.safetensors`) are not
read.

| | |
|---|---|
| Parameters | 26.4B total, 3.4B active per token (5.4B with the loop passes) |
| Layers | 24 stored, 48 applied: layers 6-17 run 3 times |
| Attention | 20 sliding-window layers (window 1,024), full attention at 5, 11, 17, 23; no RoPE |
| Hidden, heads | 2,304; 24 query heads, 8 KV heads, head dim 128 |
| Experts | 128 per layer, top 8, sigmoid router with a score bias, expert dim 1,152, no shared expert |
| Vocabulary | 200,058 (o200k-style byte-level BPE, 60 added tokens); unembedding padded to 201,024 |
| Weights | 19.5 GB: experts NVFP4 (layers 0-19) and MXFP8 (20-23), everything else bf16 / f32 |

## What a layer does

Read from `modeling_inkling2.py` and `modular_inkling2_loop.py` (transformers 5.24.0):

- **Attention residuals.** A layer's attention input (three, for q/k/v, in the full layers) and
  its MLP input are each a softmax mix, over a learned query, of the RMS-normalized outputs of the
  earlier blocks of 6 layers and the running state.
- **Attention.** No RoPE. q: projection times a learned per-head scale, L2-clipped to 22.6
  (`qk_l2_clip_threshold`). k: projection, a 4-tap causal short convolution, L2-clipped. v: the
  same, RMS-normalized without a gain. In the full layers the k/v convolutions add per-token taps
  projected from the layer input (one set per KV head). Positions enter through a relative bias:
  per head, a 16-dim state of the layer input (scaled, L2-clipped to 8) mixes a learned bank of
  bias-by-distance profiles, zero past 1,024 tokens. The full layers scale q and the bias state by
  `1 + 0.1 ln(max((p + 1) / 8192, 1))`. Each head has a sink logit (a projection of the input)
  with a zero value. The output is gated by a sigmoid of the input.
- **Short convolutions** after the attention and after the MLP (4 taps, with a residual).
- **MoE.** Scores `sigmoid(x W)`; top 8 of score + bias; weights the chosen scores renormalized,
  times 8. Experts: `softcap_4(g) · sigmoid(g) · softcap_25(u)`, then down.
- **The loop.** Layers 6-17 run 3 times. Passes 2 and 3 restart from the state before the loop;
  each loop layer reads four messages (q, k, v, MLP) from the previous pass's blocks through a
  learned query, and mixes each into its input through a gate
  `(1 − tanh f) · h + tanh g · m` (f, g: a rank-64 projection of both). Each pass has its own
  K/V caches and convolution histories.
- Logits: final mix over the blocks, RMSNorm, the unembedding, softcap 30.

## What Tlaloc has

- `InklingLoop.forward` (`:nn`): the text model over a sequence, as a trace over `:autograd`
  ops; weights addressed by the checkpoint's names. bf16 weights: projections in bf16, everything
  else in f32. The experts are evaluated densely (every expert on every token, weighted by the
  routing weights, zero for the others).
- `InklingLoop.decodeStep`: one token with K/V caches and the four convolution histories of each
  of the 48 layer applications; the position is a traced input, so one program serves every step;
  only the 8 chosen experts are gathered. `InklingLoop.prefill`: the prompt in one forward, its
  logits and the state the decode step continues from.
- `InklingCheckpoint` (`:nn` jvm): the config (`local_layer_ids`, `loop_config`), the
  safetensors files, NVFP4 and MXFP8 experts dequantized (transformers'
  `Inkling2LoopDequantizeExperts`), gate/up rows de-interleaved, linears transposed once.
  `forEachWeight` hands the tensors over one at a time, so a caller that uploads each holds one
  layer's experts at most.
- `ChatTemplate.INKLING` and the tokenizer (`:tokenizer`): the template has no generation prompt
  (the model writes its own turn header); thinking effort levels; tool declarations and calls as
  compact JSON. `InklingOutputParser` splits a generated turn into reasoning, answer and tool
  calls.
- `captureMulti` (`:autograd`): a capture with several results (a decode step's logits and state).

## Checks

| Check | Result |
|---|---|
| `InklingLoopTest`: the forward on `tiny_inkling` (8 layers in blocks of 2; layers 2-5, two blocks, looped 3 times, so later passes' messages mix two states; full attention inside and after the loop; window 8 and relative extent 6 under a 19-token prompt; log scaling from position 3; clipping thresholds that clip) against transformers' modeling code | worst logit difference 1.5e-4 over 19 tokens (logits up to 6), every position |
| The same, token by token through the decode step | 1.2e-4 |
| 11 tokens through the prefill, 8 more through decode steps from its state | 1.3e-4 |
| The mean next-token cross entropy and its gradient (the compiler's reverse transform) against torch autograd, for 11 weights: embedding, unembedding, a linear, a router, a message gate, a block query, the message queries, a relative-bias bank, convolution taps, expert weights | loss equal to 1e-6; every gradient within 4e-4 of its largest entry |
| NVFP4 and MXFP8 expert decoding against `Inkling2LoopDequantizeExperts` on random stacks | every element equal at bf16 |
| `PjrtInklingTest`, XLA CPU plugin: forward at its length and padded to 32, decode steps, prefill then decode | 1.5e-4, 1.9e-4, 1.3e-4 |
| The same on the GB10 | 2.9e-4, 2.3e-4, 1.3e-4 with f32 dots at HIGHEST; with XLA's default (TF32) a 6-layer version moved by 0.81 through one flipped routing choice |
| Tokenizer goldens (59 cases, 80 random strings, decodes, streams) and 90 rendered chats | 1,033 checks, 0 mismatches |

The tiny checkpoint is written by `harness/python/inkling_modeling_fixture.py`: `save_pretrained`
writes the released files' names (`model.llm.*`, interleaved `w13`, the split message queries),
and the logits come from loading those files back with `from_pretrained`, so the names are checked
by the released loading path.

## The released model

`PjrtInklingTest.releasedModelMatchesTheModelingCode` (gated by `TLALOC_INKLING_DIR` and
`TLALOC_INKLING_ORACLE`) against `harness/python/inkling_real_oracle.py` (transformers 5.24.0,
bf16, eager attention, on the GB10): a 29-token chat prompt ("Name three boroughs of New York
City, in one sentence.") through the prefill, then 23 greedy decode steps.

| | |
|---|---|
| Greedy tokens | 24 of 24 equal to transformers' `generate`, including two choices with a top-2 margin under 0.5 (0.41, 0.44) |
| Last prompt position | argmax equal (top logit 26.7); worst logit difference 1.02 over the 200,058 logits (the oracle runs every op in bf16; Tlaloc runs the projections in bf16 and everything else in f32) |
| Weights | read, experts dequantized and uploaded in 56 s; peak host memory leaves 17 GiB free on the 128 GiB machine |
| Prefill, 29 tokens | 29.5 s with compilation, 0.46 s after |
| Decode | 23 steps in 15.3 s, the decode program's compilation included; no steady-state rate measured |

The generated turn starts `<|message_model|><|content_thinking|>The user asks: "Name three
boroughs of New York City, in one sentence." Need answer`.

## Running it

- `InklingLoop.chunk` (`:nn`): `T` tokens at a traced position continue the state (the dense K/V
  caches of `ctx` rows and the convolution histories of every layer application), so a prompt runs
  as consecutive chunks of any sizes and a later turn continues where the last one ended; the
  decode step is a chunk of one. For one token the chosen experts are gathered; for more they run
  as two ragged dots over the (token, expert) pairs sorted by expert (`routed`), or densely where
  XLA has no `ragged_dot` (XLA:CPU). `InklingLoopTest`: chunks of 8, 3, 5, 1 and 2 tokens equal
  transformers' logits to 1.2e-4 routed and 1.7e-4 dense; the routed forward to 1.1e-4.
- `InklingModel` (`:runtime-pjrt`, `io.tlaloc.runtime.pjrt.serving`): the checkpoint on the device,
  `chunk` compiled for 1, 2, 4, … `maxChunk` tokens (each on first use, with the state aliased in
  place), `append` (the largest power-of-two chunks first, so no padding enters the state),
  `generate`, `reset`. `PjrtInklingTest` runs `tiny_inkling` through it token by token, as a chunk
  then decode steps, and as two appends: within 2.4e-4 on the GB10 and 1.9e-4 on the CPU plugin.
- `examples/inkling-chat`: a chat in the terminal, and `serve`: `OpenAiEndpoint` (the endpoint
  `tlaloc-endpoint` runs in front of Triton) over an in-process `SequenceBackend` that drives one
  `InklingModel`, with reasoning, tool calls, streaming, sampling and reuse of the conversation the
  state holds. One conversation at a time.
- Long contexts: a fill is a one-element constant broadcast, and the relative bias a gather at the
  distance `pos + i - j`; before that, a cache-sized literal per write and a `[t, extent, ctx]`
  selector made the 16-token chunk of the released model at 8,192 rows too large to emit (the
  second request to `serve` failed). `chunkProgramsStaySmallAtLongContexts`: `tiny_inkling`'s
  chunk program is 844,514 characters at a 64-row context and 912,242 at 8,192.
- Released model on the GB10 (2026-11-27, code b93306a8): `PjrtInklingTest` 24 of 24 greedy
  tokens equal to transformers, 17.4 tokens/s decode at a 128-row context (29 prompt tokens in
  1.57 s once compiled). `inkling-chat` at 8,192: weights on the device in 63 s, first token after
  85 s (compiles included), 10.2 tokens/s. `serve` at 8,192: a completion in 100 s, then the
  conversation's next turn streamed (186 events) in 76 s. `inkling-chat` with two `--prompt`s at 2,048: the second turn
  appended 13 tokens to the 151 the state held, first token after 1.16 s, 15.4 tokens/s. The machine's available memory fell by
  up to 93 GiB.

## Spans, quantized experts, batched chunks (2026-11-27/28, feat/inkling-serving)

- **Spans.** `chunk(span)` attends only the first `span` cache rows (`pos + T ≤ span`), and the new
  rows are scattered into the caches with `Tracer.kvCacheWrite` (KV_CACHE_WRITE on a dense cache
  seen as a pool of block size 1; a negative slot skips a padding row), so a write costs its rows.
  `InklingModel` compiles a program per (chunk size, span), spans `minSpan` (512) doubling to the
  context. At an 8,192-row context the released model decodes at 16.1 tokens/s instead of 10.2.
- **Quantized experts.** `InklingCheckpoint.forEachWeight(quantizedExperts = true)` hands the NVFP4
  and MXFP8 expert stacks over as FP8 codes (an E2M1 code is an E4M3 value exactly), a scale per
  block of 16 or 32 (E4M3 for NVFP4, bf16 for MXFP8's powers of two) and a factor per expert
  (`InklingLoop.QuantizedExperts`, 1.1 bytes a weight instead of 2). `InklingLoop.dequantize` gives
  the host's values exactly (0 of 768 differ, f32 and bf16, CPU and GPU). A step with few (token,
  expert) pairs gathers the chosen experts' codes and multiplies them block by block
  (`blockProduct`: no dequantized copy); a larger chunk dequantizes every expert, after the layer's
  input exists (`Tracer.after`, else XLA started every layer's dequantization at once and the
  machine ran out of memory in 80 s). `captureMulti` takes F8E4M3FN leaves and `Tracer.cast` widens
  them. InklingModel's default.
- **Batched chunks.** `InklingLoop.batchChunk` runs `B` rows of `T` tokens in the Triton backend's
  sequence-mode convention: per-token positions, the slot mapping's sign marking real tokens, rows
  right-aligned, each row's state at its slot of pools of `[numSlots, ...]`; a row whose first real
  token is at position 0 starts from empty histories; the logits at each row's last token. On
  `tiny_inkling` with padding, a padding row that names a live slot, and a slot reused from position
  0 with a finished sequence's state in it, every real row matches transformers (1.7e-4).
  `InklingModel` runs on it with one slot and padded chunks of 1, 16 and `maxChunk` tokens.

Released model on the GB10 (quantized experts, `PjrtInklingTest` and `inkling-chat`; context
8,192 unless said):

| | Before (b93306a8, bf16 experts) | Now |
|---|---|---|
| Greedy tokens equal to transformers | 24 / 24 | 24 / 24 |
| Weights on the device | 55-63 s | 22-25 s |
| 29 prompt tokens, compiled | 1.57 s | 0.72 s |
| Decode at a 128-row context | 17.4 tokens/s | 18.8 tokens/s |
| Decode at 8,192, early in the conversation (span 512) | 10.2 tokens/s | 20.5 tokens/s (`inkling-chat` 19.7) |
| Decode at position 5,148 (span 8,192) | | 15.2 tokens/s |
| Prefill at position 5,000+ in 16-token chunks | | 34 tokens/s |
| Machine memory in use at the peak (`inkling-chat`) | 93 GiB | 53 GiB (XLA's pool capped at 0.3) |

Each program compiles on first use (about 10 s for the released model's), so a conversation that
reaches a new span pays for its programs then: the first decode steps at position 5,116 ran at 2.1
tokens/s while the span-8,192 program compiled. `inkling-chat` keeps compiled programs in an
executable cache (`TLALOC_PJRT_CACHE_DIR`).

## Served through Triton

`InklingServingExport` (`:runtime-pjrt`) writes a `tlaloc-serving-v4` artifact: every layer
application's K/V caches and convolution histories are state pools (288 for Inkling-2-Mini;
`ServingLinearState.poolDims` lists pools of shapes other than Qwen3.5's), each entry is
`batchChunk` for a batch and a context (the context is the span its rows attend), prefill entries
of 16 and 256 tokens, f32 logits `[B, 1, V]`, no KV pool (the backend's pages back nothing; the
program reads the slot mapping's sign). On `tiny_inkling` the artifact run from its files as the
backend runs it (staged weights, donated pools) matches transformers to 1.4e-4.

`examples/inkling-chat`: `export` (42 s; 790 weights, 27.9 GB, 10 entries for contexts 2,048 and
8,192, decode batches 1, 2, 4, 4 slots), `:maestro:exportTritonModel`, `triton/run_server.sh`, then
`endpoint` (`tlaloc-endpoint`). On the GB10 (2026-11-28, XLA pool 0.45):

- Triton ready in 450 s the first time (the programs compile), 55-86 s from the executable cache;
  at least 63 GiB of the machine's memory left free while serving.
- The released model answers as in process (the same 123 tokens), and a follow-up turn reuses
  the 151 tokens of the conversation (`cached_tokens`), 5.8 s with the 16-token prefill entry.
- One stream decodes at 17.6 tokens/s. A prompt longer than 16 tokens runs on the 256-token
  entry, which dequantizes every expert: first token after 4.8 s.
- Four conversations at once (each in its own slot) give 9.5 tokens/s in all against 11.0 for
  one: a decode row reads its eight experts' codes, and four rows read up to 32 different experts,
  so a batched step costs nearly four single steps. A 20 ms batcher delay did not change it.

## Not done

- Prefill in large chunks dequantizes every expert of every layer per call (about 5 s per
  256-token call); experts shared by a chunk's tokens could be gathered once instead.
- Batched decode costs about one step per row (above); the state pools are f32 and dense
  (`numSlots × ctx` rows per application, 12.9 GB for 4 slots at 8,192): bf16 pools would halve it.
- The vision tower, the draft head (`mtp.safetensors`: 3 layers, `markov_head_type rnn_hidden`).
- Training at scale: the gradients are checked on the tiny checkpoint only; the dense expert
  evaluation that makes them exact costs 16 times the routed compute.
