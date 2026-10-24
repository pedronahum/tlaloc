# NVIDIA Flare: what Tlaloc can learn

Source: github.com/NVIDIA/flare-local-inference at `947a5a6` (2026-10-22), read and not run.
Flare is a local inference server for SM120/SM121 GPUs (RTX PRO 6000, DGX Spark). A Python
preparation step (PyTorch, Transformers, FlashInfer/b12x kernels, `torch.compile`) captures
CUDA graphs into a runtime package; a Rust runtime (Helios) serves the package without Python.
Its recipes include Qwen3.8-Flash-Next, the model Tlaloc's `qwen4_exp` family reads. What was
built from it and measured: [flare-plan.md](flare-plan.md).

Flare publishes no throughput, inter-token latency or time-to-first-token numbers for any
model on any GPU. Its performance documents state targets: within 5 % of a minimal graph
replay harness, and at least a baseline engine's TTFT, ITL and throughput at 1, 2 and 4
streams. Its scheduler results come from a simulator with synthetic costs. Everything below
is therefore a mechanism to measure in Tlaloc, not a gain shown elsewhere.

## Where the two stand

| | Flare | Tlaloc |
|---|---|---|
| Compilation | `torch.compile` around opaque b12x kernels, captured as CUDA graphs per (batch, draft width) and per prefill capacity (40 graphs for Qwen at 4 streams with 5 drafts); graphs rebuilt at load from recorded nodes at fixed virtual addresses, nothing compiled at serve time | StableHLO through XLA:GPU, one executable per bucket, a persistent executable cache (9.8 s load) |
| Steps in flight | Two for plain decode; one for any speculative round | Chained verify steps: the next is issued before the previous one's results are read |
| Draft width | Adaptive per request and per round (1 to 8), chosen by expected tokens per measured graph time | Fixed (3 for the 35B) |
| Drafts | MTP sampled from its full distribution, verified by rejection sampling (accept with min(1, p/q)); optional copy drafts from the request's own history | MTP drafts by argmax, verified as point masses (accepted with p(d)) |
| Recurrent state under speculation | Verification reads the state once and keeps per-token delta-rule records (about 0.26 MB per layer per request); one launch replays the accepted prefix into the committed slot | A state slot per verified position: K + 2 slots per sequence, a full state written for each position |
| Prefix cache | A tree of checkpoints at every chat message boundary, at the end of the prompt and after the answer, 64-token BLAKE2b block hashes, leaf LRU and ancestor compaction, snapshot rows borrowing KV page memory, optional disk tier | Prompt prefixes at multiples of the largest prefill call, a copy of the state slot, 8 entries LRU, GPU only |
| Prefill | One prompt per graph, chunks end at message boundaries; mixed prompt + decode graphs implemented, off by default, not validated on a GPU | Packed prompts per call, a prefill budget between decode steps |
| Scheduling | Each step picks one captured graph and cohort by priced value per measured time (proportional fairness, an age price) | Prefill first in a batch, then one verify step for every decoder |
| Multi-GPU | Tensor parallel only (no expert parallel); collectives are b12x kernels in the graph, with a CPU RDMA proxy over CX7 between Sparks; cross-node Qwen not validated | Collectives, expert-parallel MoE and a context-parallel merge, run on one device only |

## Lessons, ranked

### Tier 1: likely to pay on the GB10 now

1. **Commit recurrent state once per verify step.** Tlaloc's verify writes the Gated
   DeltaNet state of every position into its own slot so that any acceptance length is
   valid (`sequence_mode.cc`, `need_state = K + 2`). For the 35B (30 GDN layers of 32 × 128
   × 128) that is 60 MiB per position in f32: with 3 drafts, 240 MiB per sequence and
   about 1 GB per step at four streams, 3.5 ms at 273 GB/s against a step of about 50 ms
   (half in bf16). This is an estimate: measure the write share of a verify step first.
   Flare's form (`ignition/components/gdn_recovery.py`): the verify kernel keeps, per
   token, the correction vector, normalized key and decay; after acceptance one launch
   applies the accepted prefix to the committed state. In Tlaloc this is a
   GATED_DELTA_RULE verify variant that returns the records and a commit op that takes
   the accepted count. It also cuts the state slots from K + 2 to about 2 per sequence,
   which frees memory for more streams or a larger prefix cache. Flare has a CUDA
   equivalence test for it and no timing.

2. **Checkpoints at message boundaries and after the answer.** An agent's next turn
   repeats the whole conversation plus the tool result. Tlaloc's cache keeps prefixes only
   at multiples of the prefill grain and never past the prompt, so each turn prefills the
   last answer and up to a grain of the prompt again. Flare cuts prefill chunks at the
   chat template's message ends and snapshots there, at the prompt's end and at the
   answer's end (`helios/src/scheduler/scheduler.rs`, `helios/src/cache.rs`). Tlaloc would
   need: `openai_server.py` passing message token boundaries, snapshots at any length (the
   partial last page copied), a snapshot when a sequence ends, and eviction that knows
   parents from leaves. Measure on a recorded multi-turn tool-use session: prefill tokens
   per turn and TTFT per turn.

3. **Draft width per request.** Acceptance varies by content (prose, code, JSON tool calls).
   Flare keeps, per request, the survival probability of each draft depth (prior 0.5^d,
   half-life 128 tokens, weight capped at 128) and picks the (batch, width) graph with the
   most expected tokens per measured GPU time (`helios/src/scheduler/speculation.rs`,
   `policy.rs`). Tlaloc would need verify entries for widths 1 to 3 (more executables), a
   per-entry measured cost and a cohort width. The earlier draft-count measurement (3, 5,
   7 on the 35B) gives the costs to start from.

4. **Copy drafts.** For code edits and tool results the continuation often repeats the
   request's own text. Flare matches the last 8, then 4, tokens against the previous 4,096
   and offers the continuation as drafts (`speculation.rs`), verified as point masses,
   which Tlaloc's verifier already does. The host side is small; the question is
   acceptance on an agent workload, which neither project has measured.

### Tier 2: for Qwen3.8-Flash-Next serving (steps 6 and 7 of the Flash-Next plan)

5. **The n-gram table off the GPU heap.** The plan puts the 27 GiB table in memory next to
   72 GiB of weights. Flare reads table rows from disk with io_uring (or RAM) on host
   threads; the graph publishes the lookup ids before layer 0 and waits for the rows at
   layer 1, so layer 0 hides the read (`docs/model-preparation.md`,
   `docs/runtime-package-design.md`). On the GB10 memory is unified, so RAM storage saves
   nothing; only disk storage frees the 27 GiB. Tlaloc would need the table lookup moved
   out of the graph into a host-filled input (16 rows of 160 per token) and the step split
   or an event wait before layer 1.

6. **Compare against the real checkpoint in bf16.** Tlaloc's `qwen4_exp` checks are against
   an f32 4-layer fixture; the model's own forward rounds the four streams and every block
   output to bf16, which Tlaloc keeps in f32. Flare streams the HF forward over the real
   checkpoint with bounded residency and compares all 48 layer residuals and the logits
   (`ignition/reference/qwen38_decoder.py`, `qualify/layer_comparison.py`,
   `qualify/text_oracle.py`: top-10 overlap, KL, top-1 margin). Tlaloc's f32 streams are
   closer to exact arithmetic but their distance from what the model was trained against
   is unmeasured. The same reading confirms Tlaloc's mix, injection, PLE gate and SiLU on
   the short convolution match transformers 5.x (`modeling_qwen4_exp.py`).

7. **FP32 GDN state for Flash-Next.** Flare keeps the recurrence in FP32. Tlaloc's default
   is f32 with `-PstateDtype=bf16` as an option; keep f32 for this model unless a
   real-checkpoint comparison shows bf16 harmless.

### Tier 3: validation practice

8. **Bounds from measured repeatability.** For kernels with atomic reductions Flare runs the
   reference 8 times and bounds the error by min(1 % of the reference norm, 2 × the spread)
   (`tests/recipes/test_step5_moe_overlap_cuda.py`). The flaky `PjrtDeviceAdamWTest`
   (5.5e-3 against a fixed 5e-3) is the case for it.
9. **Poison scratch before replay.** Flare fills non-persistent buffers with NaN before
   replaying a graph and checks they are still NaN where nothing should write. Its GLM
   defect (corrupt answers above 16 rows) was a value set once at preparation in scratch
   that export zeroed (`docs/glm-batched-speculation-defect.md`). Tlaloc's executables own
   their temporaries, but the backend's donated pools (KV, state slots, the chained step's
   inputs) are the same risk; a test that poisons free pages and free state slots before a
   step would catch reads of stale memory.
10. **Faulty-row and forced-acceptance tests.** Rows that are inactive, stale, in a wrong
    slot or duplicated must not advance state (`tests/recipes/test_qwen38_speculation.py`);
    a singleton grammar forces a known acceptance pattern to test every prefix length
    without an MTP head (`qualify/speculative_state_parity.py`). Tlaloc's MTP tests feed
    chosen drafts through the interpreter; the backend-level equivalent is missing.
11. **Per-token timestamps.** Flare can return each token's emission time and counts
    speculative rounds, drafts and GPU time per (batch, width). Tlaloc's four-stream TTFT
    breakdown (two decode steps ahead of a prefill) is still an inference; timestamps per
    step in the backend log would settle it.

### Not worth taking, or already measured

- **One speculative round in flight when a prompt waits.** Flare never queues a verify
  round behind another. Tlaloc measured chaining off: the four-stream TTFT stays 0.20 s,
  and decode drops from 81.2 to 70.0 tokens/s.
- **Mixed prompt + decode graphs.** Off by default in Flare and not validated on a GPU;
  Tlaloc's measurement says it would spare decoders a stall, not shorten the prompt's wait.
- **Explicit CUDA graphs and one batched input upload.** Their purpose is host time per
  step; Tlaloc's chained steps already hide its 1.7 ms of execute arguments.
- **Shared expert on a second stream.** Tlaloc fused the shared expert into MOE_EXPERTS.
- **The priced scheduler.** Shown only in simulation; Tlaloc's prefill budget covers the
  measured problem (a 20,000-token prompt's 9 s decode gap, now 0.8 s).
- **Encrypted packages, fixed-address loading, Orbit host management.** Not Tlaloc's
  concerns.

## For several Sparks (TPU_BRINGUP and the multi-host design)

- Flare shards Qwen3.8 by tensor parallelism only: GDN heads and conv channels, the 24
  query heads, the expert intermediate width (640 → 320 → 160 at TP 1/2/4), the n-gram
  table by rows; the 2 KV heads, the indexer and the router replicated. Two BF16
  all-reduces per layer (96 per step), the vocabulary projection sharded and gathered
  before sampling. None of it is validated across nodes yet.
- Collectives are kernels inside the step graph that stage into registered host memory; a
  CPU proxy posts the RDMA writes. No per-collective latency is published, so 96
  all-reduces per step over CX7 is the number to measure before choosing TP over expert
  parallel for 512 experts.
- The checkpoint does not depend on the TP degree: each rank reads its byte ranges from
  rank 0 over the network and regroups block scales on the GPU, instead of exporting per
  rank.
