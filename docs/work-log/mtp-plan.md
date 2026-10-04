# MTP speculative decoding for the Qwen3.5 family: plan

Branch `feat/mtp` from `main` at `7b800ba`. All stages are done; the results are in
[qwen35-progress.md](qwen35-progress.md#mtp-speculative-decoding).

## What the checkpoints hold

Qwen3.5-0.8B, Qwen3.8-27B and Qwen3.6-35B-A3B carry one MTP layer (`mtp_num_hidden_layers:
1`), under `mtp.`:

- `fc` `[H, 2H]`;
- `pre_fc_norm_embedding`, `pre_fc_norm_hidden` and `norm` (gains used as `1 + w`);
- `layers.0.*`: a full-attention decoder layer with the model's MLP (dense or MoE) and
  `q_proj` giving `[query | gate]`.

The head shares the target's embedding table and LM head.

## The head (vLLM's Qwen3_5MultiTokenPredictor)

At RoPE position p, with h_p the target's final-norm hidden state at p:

```
m_p    = fc([pre_fc_norm_embedding(embed(x_{p+1})) | pre_fc_norm_hidden(h_p)])
o_p    = norm(layer(m_p))           the layer attends over its own KV at positions <= p
x_{p+2} predicted by lm_head(o_p)
```

A further draft feeds the drafted token with o_p in place of a target hidden state.

## Greedy speculative step (k drafts)

The target has processed positions up to n - 1. x_n is known, and the head proposed
d_1..d_k for positions n+1..n+k.

1. **Target over [x_n, d_1..d_k]** at positions n..n+k gives greedy tokens g_0..g_k and
   hidden states h_n..h_{n+k}.
2. **Accept:** a is the number of leading drafts with d_{i+1} = g_i. The step emits
   g_0..g_a (a + 1 tokens); the target is now valid up to n + a.
3. **Head over positions n..n+a** with tokens g_0..g_a and hidden states h_n..h_{n+a}:
   accepted drafts equal g, so the head's token at every verified position is g_i, with
   no host decision. Its output at n + a gives d'_1, and k - 1 single steps give the rest.

All three run in one call. Entries written past n + a (target KV, head KV) are written
again before anything reads them: attention reads up to the sequence length, and the
next step starts at n + a + 1.

## State rollback

A Gated DeltaNet layer's conv and recurrent states advance with every token, so a
rejected draft would leave the wrong state.

- **Per-token slots:** the verify call writes the state after each of its k + 1 tokens
  into its own slot (`writeSlots [B, k+1]`) and reads from one (`readSlots [B]`).
- **Bookkeeping:** the next step reads the slot of token a. A sequence owns k + 2 slots,
  so the slot it reads from is never one it writes.
- **Carried hidden state:** the same slots hold the target's last hidden state, which a
  prefill chunk needs for the head's first position (chunk start - 1).

## Stages

1. **Reference.** `harness/python/qwen35_mtp_fixture.py`: transformers'
   `Qwen3_5DecoderLayer` with the 0.8B's `mtp.*` tensors. For " The capital of France
   is" the target's next token is " Paris" and the head drafts ".", "\n", "The", the
   target's own greedy continuation.
2. **IR:**
   - MTP roles, names and staging;
   - the head as a graph;
   - interpreter parity with the fixture.
3. **Per-token states:** `GATED_DELTA_RULE` and `CAUSAL_CONV1D` write the state after
   each token into its own slot.
4. **Verify graph:** the target over k + 1 tokens per row, the accept count, the head
   pass and the k drafts, in one graph.
5. **Kotlin speculative loop**, interpreter and GB10, on the 0.8B. Its ids must equal
   plain greedy ids, whatever the drafts; it also reports the acceptance rate.
6. **Serving:**
   - the artifact entries;
   - the Triton backend's step: up to k + 1 tokens a request, per-sequence slot
     bookkeeping;
   - the client.
7. **Measurements:** Qwen3.8-27B and Qwen3.6-35B-A3B, 1 and 4 users, against the
   non-speculative numbers and vLLM.
