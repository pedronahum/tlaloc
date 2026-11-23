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

## Not done

- Serving: the paged-attention kernels have no sink logit, no per-query relative bias and no
  per-pass caches; the Triton backend and the serving artifact know none of it. The decode step
  here keeps dense caches of a fixed length and rewrites them each step.
- The vision tower, the draft head (`mtp.safetensors`: 3 layers, `markov_head_type rnn_hidden`).
- Training at scale: the gradients are checked on the tiny checkpoint only; the dense expert
  evaluation that makes them exact costs 16 times the routed compute.
