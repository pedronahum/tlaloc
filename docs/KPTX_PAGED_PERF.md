# KPTX paged attention — the performance tier's baselines and diagnosis

**§0.4.481 (Phase H4b, slice K1). Measure and diagnose; no kernel was
rewritten in this slice.**

> **§0.4.495 (slice W3) is the latest section — read [§11](#11-04495-w3-six-sessions-the-tier-re-measured-and-the-registry-question-re-answered).**
> It re-measures the whole §2 grid at HEAD across **six sessions in one
> hour**, re-ranks §8.3, and re-answers §8.1. The registry decision is
> **unchanged (NO)** and all three of §8.2's reasons come out stronger,
> two of them now quantitative. What changed: the 8B win is a reproduced
> result (**1.4–1.9×**, 6/6 sessions), `tinyllama-s8`'s gap closed to
> **1.58×** (W2 helped there too, which §10 did not claim), W2's own
> headline comes **down** to ×1.2–1.3, and **fixing the instrument is now
> rank 1** — the dispatch floor spans 40.0–416.3 µs.
>
> **§0.4.494 (slice W2) is the last kernel change — read [§10](#10-04494-w2-the-warp-mapped-stage-1--the-8-read-amplification-paid-down).**
> It implements §8.3 **rank 1** (stage 1's coalescing), and unlike K2 it
> is not a null: the claimed lane's device floor at both Llama-3-8B
> points fell **1.3–1.5×** with the unclaimed control lane unmoved, and
> the Double oracle got *tighter* (1.19e-7 → **8.94e-8**). The gate in
> §5 is still unmet and `defaultInferenceKernelTemplates` is still
> empty — the TinyLlama points did not move.
>
> **§0.4.483 (slice K3) closes the tier in [§8](#8-04483-k3-the-tiers-close-out--the-registry-decision-stated).**
> If you only want the answer to "can I turn the kernel on today?", read
> §8.1 — **no**, with the reason, the rejected alternative, and the merged
> ranked list of what would change it.
>
> **§0.4.482 (slice K2) amends this document — read [§7](#7-04482-k2-the-first-tier-change-a-controlled-null-and-what-it-bought)
> before acting on §4's priority list.** K2 implemented item **#6**
> (stage 3's idle threads), certified it, and measured **no change** —
> and that null, taken against a 2× lever, **bounds stage 3 at ≤ 16% of
> the chain** and re-ranks everything below it. The list in §4 is kept
> verbatim as K1 wrote it; §7 says which of its numbers survived contact.

§0.4.471 landed the correctness-tier `@kptx_paged_attention` kernel at
1.2e-7 against the interpreter's Double paged walk, reported **465 µs
claimed against 310 µs unclaimed**, concluded the kernel was *1.5×
slower than the lowering it replaces*, and left
`defaultInferenceKernelTemplates` empty for that reason.

This document replaces that single number with a sweep, and **the
conclusion changes**. The 465/310 pair was a host round trip over a
~1 MB fixture — ≈ 2.3 GB/s of staging traffic with an attention kernel
somewhere inside it. Measured on the device, the kernel is
**1.9× slower at TinyLlama-shaped decode points and 1.4–1.6× FASTER at
Llama-3-8B-shaped ones.**

The apparatus is
[`benchmarks/…/KptxPagedAttentionBenchTest.kt`](../benchmarks/src/jvmTest/kotlin/io/tlaloc/benchmarks/KptxPagedAttentionBenchTest.kt),
which runs inside `./gradlew test` and self-skips without a GPU.

---

## 1. Methodology, stated rather than assumed

Per the §0.4.337 rule (GB10 medians drift; compare floors or same-session
lanes), every number below is a **floor over 20 reps after 3 warmup reps,
from ONE `PjrtSession` with every executable already compiled and
cached**, with the lanes timed **interleaved** — one rep of each, round
robin. Lane-at-a-time measurement was tried and discarded: it gives each
lane its own slice of the session's history, and a differencing
apparatus built on it produced a *negative* device cost.

Three lanes per point, all through `PjrtSession.executeOn` against
**pre-staged device buffers** (the §0.4.309 benchmark instrument — FFM
downcall, XLA dispatch, device work, device-complete await; no host
transfer, no per-call arena):

| lane | what it is |
|---|---|
| **c** | the claimed program: `stablehlo.custom_call @kptx_paged_attention`, the §0.4.471 three-stage PTX chain |
| **u** | the unclaimed program: the H1a gather-composed StableHLO emission, XLA's own lowering |
| **dispatch floor** | the same three staged buffers and the same `executeOn` path with one elementwise multiply instead of an attention |

A `round-trip` column reports the same work through `runOn` (transfer
included) purely so the gap is visible. Every point asserts the two
attention lanes agree before either is timed; the kernel's
*certification* is §0.4.471's test against a Double oracle, not this.

Fixed across points: `blockSize` 16, full-window `seqLens` (raggedness is
a separate axis), a **permuted** block table (page locality must never be
what makes a lane fast), one private page pool per sequence, f32 pools,
`scale` 0.125, block 256 threads.

## 2. The numbers (GB10, 48 SMs, sm_121, one session, 2026-09-21)

All times µs. `c/u` below 1.00 means the KPTX kernel **wins**.

| point | ctx | grid | CTA/SM | **c** | **u** | **c/u** | floor | round-trip c / u | distinct KV | c GB/s | u GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tinyllama-s1-ctx256 | 256 | 32 | 0.67 | 177.5 | 93.0 | **1.91×** | 42.2 | 814 / 657 | 0.52 MB | 3.0 | 5.6 |
| tinyllama-s8-ctx512 | 512 | 256 | 5.33 | 272.9 | 141.7 | **1.93×** | 45.4 | 1634 / 1188 | 8.39 MB | 30.7 | 59.2 |
| llama3-8b-s8-ctx1024 | 1024 | 256 | 5.33 | 882.8 | 1209.8 | **0.73×** | 312.9 | 28812 / 29029 | 67.1 MB | 76.0 | 55.5 |
| llama3-8b-s16-ctx1024 | 1024 | 512 | 10.67 | 1431.1 | 2322.4 | **0.62×** | 236.0 | 55481 / 52956 | 134.2 MB | 93.8 | 57.8 |

Shapes: TinyLlama-1.1B = 32 query heads / 4 KV heads / headDim 64;
Llama-3-8B = 32 query heads / 8 KV heads / headDim 128. Both have GQA
group 4.

Read off the table:

- **The round-trip column is the measurement §0.4.471 reported.** At the
  8B points it is 25–58 ms against a device cost under 2.4 ms: the
  transfer is 95%+ of it, and the two lanes differ there by less than the
  transfer's own noise (the s16 row even has the *claimed* round trip
  slower while its device cost is 1.6× faster). **Any paged-attention
  claim measured through `runOn` at these shapes is a staging
  measurement.**
- **The crossover is real and it is in the right direction**: the kernel
  loses at toy shapes and wins where a served model actually lives.
  **Three independent sessions** produced c/u of 1.80 / 1.36 / 0.70 /
  0.65, 1.91 / 1.93 / 0.73 / 0.62 and 2.00 / 1.92 / 0.76 / 0.66 — the
  absolutes drift by tens of percent (the §0.4.337 rule, obeyed), the
  sign of the comparison does not.
- **The dispatch floor is the noisiest column and is not load-bearing.**
  Across those sessions it ran 42–125 µs at the small points and
  83–313 µs at the large ones, where a 2-MFLOP multiply cannot explain
  it at all — something in the execute path scales with the staged
  working set, and something else in it is simply noisy. It is reported
  as an order of magnitude, **subtracted from nothing**, and pinning it
  down is a named follow-up.
- At the small points the claimed lane is ~2.5× the unclaimed lane *above
  the floor* (135 vs 51 µs; 228 vs 96 µs), so the small-shape loss is
  **not** a fixed per-custom-call tax — it is real kernel time. (That
  ratio holds in each session even though the floor itself moves.)

### Declared resources and occupancy (no GPU needed; `PtxResourceReport`)

| kernel | declared 32-bit slots/thread | static smem | blocks/SM @256 | occupancy | limited by |
|---|---|---|---|---|---|
| `kptx_paged_scores` | 65 | 0 B | 3 | 50% | registers (declared) |
| `kptx_paged_softmax` | 45 | 1024 B | 5 | 83% | registers (declared) |
| `kptx_paged_out` | 68 | 0 B | 3 | 50% | registers (declared) |

Caveats are `PtxResourceReport`'s own: declared virtual registers are an
upper-bound proxy (ptxas reallocates), and sm_121 is mapped to
sm_86-class limits pending official tables.

**Neither page-walking stage declares one byte of shared memory** — so
every K and V element either stage touches comes from global memory and
is never reused inside the block. The softmax's 1 KiB is its cross-warp
reduction tree, and it is the one stage that touches no pages.

## 3. Where the time goes

### 3.1 The GQA re-read is the dominant term

Stage 1 is one CTA per **(sequence, query head)**. Each of the `group`
query heads sharing a KV head walks the same pages independently, so K is
read `group` times and so is V:

```
issued K+V bytes = 2 · numSeqs · numHeads · ctx · headDim · 4     (group × the minimum)
distinct K+V bytes = 2 · numSeqs · numKvHeads · ctx · headDim · 4
```

At `llama3-8b-s16-ctx1024`: **536 MB issued against 134 MB distinct**, a
4× multiplicity, over a pool that does not fit the GB10's 24 MB L2.
Against the measured 1431 µs that is 383 GB/s of *issued* traffic —
above NVIDIA's published 273 GB/s LPDDR5X figure for this part (vendor
spec, **not measured here**; a STREAM-style floor on this box is a named
follow-up), which says L2 is already absorbing a meaningful share and
that the multiplicity is partly, not wholly, paid.

On **distinct** bytes the kernel achieves 94 GB/s — roughly a third of
the published peak. That gap is the headroom.

### 3.2 The score matrix crosses global memory six times

Counted from the PTX: `kptx_paged_scores` stores it; `kptx_paged_softmax`
is the shared row softmax, three strided passes (max, sum-of-exp,
normalize-in-place) = 3 loads + 1 store; `kptx_paged_out` loads it again.
**4 loads + 2 stores.**

| point | score matrix | ×6 | share of issued K+V |
|---|---|---|---|
| tinyllama-s1-ctx256 | 0.03 MB | 0.20 MB | 2.4% |
| tinyllama-s8-ctx512 | 0.52 MB | 3.15 MB | 4.7% |
| llama3-8b-s8-ctx1024 | 1.05 MB | 6.29 MB | 2.3% |
| llama3-8b-s16-ctx1024 | 2.10 MB | 12.58 MB | 2.3% |

**This is the cost everyone expects to be the problem and it is not.**
Fusing to an online softmax deletes 2–5% of traffic. Its real value is
the two grid-wide barriers and two kernel launches it also deletes,
which is a small-shape effect.

### 3.3 Structural costs, each with its measured fingerprint

| cost | where it shows |
|---|---|
| **one CTA per row** | `tinyllama-s1-ctx256` runs 32 CTAs on 48 SMs — **0.67 CTA/SM, a third of the device idle at the latency-critical batch-1 shape** |
| **no shared-memory staging of the page window** | 0 B declared in both page walkers; the group-× re-read (3.1) has nothing to reuse it from |
| **scalar, unvectorized page walk** | each thread walks `headDim` f32 one `ld.global.f32` at a time, consecutive threads `headDim·4` B apart — a warp's request at a given `d` touches 32 separate sectors |
| **no warp specialization** | the TLX-informed bf16 GEMM tile ingested in §0.4.357 is not used; stage 1 and 3 are f32 FMA chains |
| **stage 3 wastes threads** | block is 256 but the loop strides over `headDim` — at headDim 64 **192 of 256 threads exit immediately**; at 128, 128 of 256 |
| **50% occupancy, register-limited** | 65 and 68 declared slots/thread; at 256 threads the cliff is at **64** (3 blocks/SM → 4) |
| **f32 pools** | bf16 is declined by name by the template, so the dominant term (3.1) is carried at twice the width |

## 4. What the tier must change, in priority order

Expected wins are stated against the measured baselines above, and each
says what it is a win *on* — none of them is a guess dressed as a number.

| # | change | expected win | on which points |
|---|---|---|---|
| **1** | **One CTA per (sequence, KV head), all `group` query heads together.** K and V are then read once per KV head instead of `group` times, from one page walk, with `group` score rows accumulated in registers. | Removes 3/4 of issued K+V traffic at group 4. Bounded above by ~4×; realistically toward the L2-adjusted gap, so **~1.5–2.5×** at the 8B points. | 8B points; helps everywhere |
| **2** | **bf16 pools** (declined by name today). | Halves the dominant term. **~2×** where memory-bound, and it composes with #1. | 8B points |
| **3** | **Vectorized `ld.global.v4.f32` + shared-memory staging of the page window.** | The distinct-byte rate is 94 GB/s against a ~273 GB/s published peak: **up to ~2–3×** on the memory-bound stages, of which coalescing is the accessible part. | 8B points |
| **4** | **Split the context across CTAs** (flash-decode partitioning with a second reduction pass — vLLM ships `paged_attention_v2` for exactly this). | Turns 0.67 CTA/SM into ≥ 1 with enough splits: **up to ~2×** at batch 1, nothing at batch 16. Note it is in tension with #5, which is why vLLM keeps both v1 and v2. | `tinyllama-s1` |
| **5** | **Fuse all three stages into one online-softmax kernel.** | Deletes 2–5% of traffic (§3.2) *and* two launches and two grid-wide barriers: **~30–60 µs** at the small points, **~2–4%** at the 8B points. Also deletes #6 outright. | small points |
| **6** | **Give stage 3 a block sized to its loop** (or give the idle threads context lanes to reduce). | Recovers 2–4× of stage 3's parallelism; subsumed by #5 if #5 lands first. | all |
| **7** | **Shave one declared register slot in `kptx_paged_scores` (65 → 64).** | 3 → 4 blocks/SM, 50% → 67% declared occupancy, for a one-line change. The real gate is ptxas's allocation, so this is an experiment with a cheap ticket, not a promise. | all |

**Explicitly not on the list**: a scale-generic kernel (a correctness/
ergonomics deferral from §0.4.471, worth zero microseconds), and fusing
`KV_CACHE_WRITE` into the claim (H1b left it available; it is a traffic
win on the *write* side and belongs after #1 has settled the read side).

## 5. The gate, restated

§0.4.471's gate was "beat 465 µs". That number was a round trip and the
gate was therefore unmeasurable. The gate this document leaves behind:

> `defaultInferenceKernelTemplates` stays empty until the claimed lane's
> **device** floor is below the unclaimed lane's at **every** point in
> §2 — including `tinyllama-s1-ctx256`, where it is currently 1.91×
> behind — measured with this test, in one session, interleaved.

Items #1, #4 and #5 are the ones that move the small points.

## 6. Named deferrals from this slice

- **The vLLM reference row.** The slice asked for one. It is not
  available: vLLM 0.29.0 in `~/.local/venvs/vllm` **ships no `vllm._C`
  extension module** — `from vllm import _custom_ops` warns
  `Failed to import from vllm._C` and exposes only
  `paged_attention_rocm`, because 0.29's CUDA path routes attention
  through FlashAttention/FlashInfer backends rather than the classic
  `paged_attention_v1` op. Timing a different algorithm through a
  different launch path and printing it beside these rows would be the
  fair-comparison hazard with none of the compensating value. The row
  belongs next to a flash-decode-shaped Tlaloc kernel (#5), not next to
  a three-launch correctness chain.
- **The large-point dispatch floor** (236–313 µs for a 2-MFLOP multiply
  over 67–134 MB of staged buffers). Reported, unexplained, subtracted
  from nothing.
- **A measured memory-bandwidth ceiling for this box.** Every "% of
  peak" above leans on NVIDIA's published 273 GB/s figure. A STREAM-style
  floor measured here would make §3.1 and item #3 quantitative rather
  than indicative.
- **The ragged / chunked-prefill form** — still H1a's deferral, and
  every shape here is a decode shape.

---

# 7. §0.4.482 (K2): the first tier change, a controlled null, and what it bought

**Slice K2 took item #6 — the cheapest item on §4's list and the only
one whose defect was structural rather than arithmetic — implemented it,
certified it, measured it, and got NOTHING. That is this section.**

The change is real and it is in: `kptx_paged_out` now decomposes its
block as `(part, d)` instead of striding `d` alone, so at `headDim 64`
against a 256-thread block the stage went from **64 live threads to
256**, and at `headDim 128` from 128 to 256. The mechanism, the
branch-uniformity argument for its one barrier, and the `nsplit < 2`
fallback are documented on `KptxKernels.pagedAttentionModule`; the pins
are `PagedAttentionModuleTest.thePagedOutStageCarriesBothDecompositions`
(both arms still emitted, exactly one `bar.sync`) and the re-aimed smem
assertion in the bench test, which §0.4.481 wrote *expressly* so that a
kernel staging shared memory would trip it. It did, and it was re-aimed
rather than deleted.

## 7.1 The numbers

Same test, same apparatus, same box, same day. Two sessions before the
change and two after, all four within an hour of each other:

| point | c/u before (2 sessions) | c/u after (2 sessions) |
|---|---|---|
| tinyllama-s1-ctx256 | 1.99, 1.60 | 2.14, 1.27 |
| tinyllama-s8-ctx512 | —, 1.84 | 1.55, 1.65 |
| llama3-8b-s8-ctx1024 | —, 0.84 | 0.83, 0.84 |
| llama3-8b-s16-ctx1024 | —, 0.70 | 0.71, 0.68 |

(The first pre-change session aborted at point 1 on the bench's own
dispatch-floor sanity assertion — the floor lane came in at 144 µs
against an unclaimed lane of 135 µs. That is the noise §0.4.481 already
declined to subtract from anything, showing its teeth.)

The claimed lane's absolute device floor at `llama3-8b-s16-ctx1024`:
**1614 µs before; 1695 and 1533 after.** At `llama3-8b-s8-ctx1024`:
**1116 before; 1025 and 1098 after.** The change is inside the
run-to-run spread in both directions at both points. **There is no win
here and none is claimed.**

## 7.2 What the null actually proves

A null result from a *lever of known size* is not the same as a null
result from a guess, and this lever's size is known: at `headDim 128`
the stage's thread count doubled exactly. If stage 3 were a fraction `f`
of the chain and its time improved by a factor `k`, the chain would
improve by `f·(1 − 1/k)`. The 8B points moved by less than ~8%
end-to-end against `k ≈ 2`, so

> **stage 3 is at most ~16% of the three-stage chain** — and that is an
> upper bound obtained by assuming the doubled thread count bought a
> full 2×, which it plainly did not.

§0.4.481 wrote item #6 down as "recovers 2–4× of stage 3's parallelism"
and it did. **Parallelism was not what stage 3 was short of.**

## 7.3 The mechanism the null exposes: stage 1 is uncoalesced and stage 3 never was

Reading the two page walks side by side for their *warp-level* address
pattern — which §0.4.481's §3.3 recorded as one row ("scalar,
unvectorized page walk") covering both stages, and which is in fact
**two completely different situations**:

| | `kptx_paged_scores` (stage 1) | `kptx_paged_out` (stage 3) |
|---|---|---|
| what a thread owns | one context lane `j` | one head dim `d` (and, since K2, a lane partition) |
| what varies across a warp at one load | **`j`** | **`d`** |
| address stride between adjacent lanes of the warp | `numKvHeads · headDim · 4` = **4096 B** at the 8B shapes | **4 B** |
| sectors a warp's 128 B of useful data costs | **32 × 32 B = 1024 B** | 4 × 32 B = 128 B |
| read amplification | **8×** | **1×** |

Stage 3's V walk has been perfectly coalesced since §0.4.471 — a warp's
32 threads read 32 consecutive f32 of one `V[block, off, kvh, :]` row.
Stage 1's K walk has never been coalesced at all: at a fixed `d`, its 32
threads are reading 32 *different pages*, 4 KB apart, and each 4-byte
load drags a 32-byte sector. **Stage 1 pays roughly 8× the DRAM traffic
its arithmetic needs, and stage 3 pays 1×.**

That single asymmetry explains the null completely: the stage K2
parallelised was the cheap one. It also re-reads §3.1's headline number.
The 4× GQA *multiplicity* is issued by both stages, but stage 1 issues it
through an 8×-amplified path and stage 3 through a clean one — so of the
"383 GB/s of issued traffic" at `llama3-8b-s16-ctx1024`, the K side and
the V side are not the same size at all, and the K side is the one that
is mostly waste.

## 7.4 The re-ranked list

§4's table is kept above exactly as K1 wrote it. What K2's evidence does
to it:

| §4 item | K2's verdict |
|---|---|
| **#6** stage 3's idle threads | **DONE, and worth ~0.** Landed and certified because it is correct, costs nothing measurable (68 → 73 declared register slots, still 3 blocks/SM; 1 KiB smem that does not bind) and is the `(part, d)` substrate item #1's V half will need. Its value to this arc was the measurement, not the microseconds. |
| **#3** coalescing / staging | **Promoted to #1 in effort-per-microsecond, and narrowed to STAGE 1 ONLY.** §7.3 puts an 8× read amplification on exactly one of the two page walks. Stage 3 needs no staging; it was never the problem. |
| **#1** one CTA per (seq, KV head) | **Still the largest term, and now known to be worth more on the K side than the V side.** Fusing the GQA group in stage 1 removes 3/4 of an 8×-amplified stream; in stage 3 it removes 3/4 of a clean one. |
| **#2** bf16 pools | Unchanged: halves whatever the other two leave. |
| **#4**, **#5** | Unchanged, and still the only items that address the small points — where, note, the dispatch floor is now 40–70% of the whole measurement and the chain and the floor cannot be told apart at all. |
| **#7** register shave | Untouched. Stage 3's slot count went the other way (68 → 73) with no occupancy change, which is itself weak evidence that the declared-register cliff is not where the time is. |

**The concrete next kernel**, stated so the next slice does not have to
re-derive it: give `kptx_paged_scores` a warp-per-lane mapping — warp `w`
owns context lane `j = w, w + nWarps, …`; its 32 threads split `headDim`
(`d = lane, lane + 32, …`), so each warp load is 32 consecutive f32 =
one 128 B transaction; reduce the dot across the warp with
`shflSync(DOWN, …)` (§0.4.343) and let lane 0 store `S[row, j]`. **No
barrier and no shared memory**, because a warp is already synchronous
and every thread of it shares `j`.

**One blocker is already known** and is named here so it is not
rediscovered: `shflSync` moves `.b32` and requires `%r`-class registers,
while the accumulator is `%f`. The natural spelling `mov.b32 %r1, %f1`
is legal PTX but is **rejected by Tlaloc's own ISA table**
(`PtxIsa.kt`'s `mov` entry class-checks both operands from the type, so
`b32` demands `%r` on both sides). Closing this needs a small,
principled table change — a bit-typed `mov` should be class-`Any` — with
its own `PtxIsaTest` pin. That is the first commit of the next slice,
not an afterthought inside it.

## 7.5 The gate, unchanged

§5's gate stands untouched: `defaultInferenceKernelTemplates` stays
empty. **This slice does not register the kernel and does not move the
kernel toward registration** — the small points, where the gate fails,
are exactly where nothing changed. What would it take? §5's condition,
and nothing has been added to or subtracted from it:

> the claimed lane's **device** floor below the unclaimed lane's at
> **every** point in §2, measured with this test, in one session,
> interleaved.

The honest statement of where that stands after K2 is that the 8B points
have been winning since K1 measured them properly, the TinyLlama points
still lose, and the two changes that address the TinyLlama points
(#4, #5) have not been attempted. A third thing is now also true: at
those points the dispatch floor is 144–252 µs against a claimed lane of
308–540 µs, so **the gate as written may not be measurable at
`tinyllama-s1-ctx256` at all** until the large-point dispatch floor
(§6's second deferral) is explained. That is not a reason to weaken the
gate; it is a reason to fix the instrument first.

## 7.6 K2's own deferrals

- **The stage-1 warp-mapped kernel** — designed in §7.4, not written.
  Time-boxed out after the null result consumed the slice's
  implementation budget, and left with its blocker named.
  > **§0.4.494 (W2) wrote it**, exactly as §7.4 specified. The 8B
  > points' claimed floors fell 1.3–1.5× with the control lane
  > unmoved; the TinyLlama points did not move. See [§10](#10-04494-w2-the-warp-mapped-stage-1--the-8-read-amplification-paid-down).
- **The `mov.b32` ISA-table gap** — see §7.4. One line of table, one
  pin, and it gates everything warp-reduced that carries floats.
  > **§0.4.493 (W1): this gap did not exist.** The table had accepted
  > `mov.b32 %r1, %f1` since §0.4.357, and `shfl.sync.down.b32` moves
  > `%f` registers directly, so no `mov` is needed at all. See [§9](#9-04493-w1-the-enabling-slice--and-the-blocker-that-was-not-there).
- **The `nsplit < 2` arm of stage 3 is unexercised on hardware.** It is
  §0.4.471's program kept verbatim, and no fixture in the suite has
  `headDim >= 256`, so only the emitted-PTX pin covers it. A GPU arm at
  `headDim 256` would close it.
- **Per-stage timing.** Every conclusion in §7.2 is a *bound* derived
  from end-to-end floors because the apparatus times the whole chain.
  Three separately-timed custom calls, or CUPTI, would turn "≤ 16%" into
  a number.

---

# 8. §0.4.483 (K3): the tier's close-out — the registry decision, stated

**This section is the answer to the one question the tier owes a
deployment: *can I turn the kernel on today?* It is written after K1 and
K2 and changes neither of their numbers.**

## 8.1 The decision

> **NO. `defaultInferenceKernelTemplates` stays empty, and the opt-in
> spelling (`kptxInferenceKernelTemplates` +
> `KptxPagedAttention.register`) remains the only way the kernel enters
> an executable.**

> **REVISITED by [§11.5](#115-81-revisited-can-the-kernel-be-turned-on-today)
> (§0.4.495, W3) against six fresh sessions. The answer is the same NO,
> the scoreboard is still two of four, and each of §8.2's three reasons
> is stronger than it was — two of them now with numbers.** The
> deployment sentence at the end of §8.2 is the part that improved.

The gate is §5's and it is unchanged: the claimed lane's **device** floor
below the unclaimed lane's at **every** point in §2, one session,
interleaved. After K1 and K2 the honest scoreboard against it is:

| point | c/u | gate |
|---|---|---|
| tinyllama-s1-ctx256 | 1.27–2.14 | **FAILS** |
| tinyllama-s8-ctx512 | 1.55–1.93 | **FAILS** |
| llama3-8b-s8-ctx1024 | 0.73–0.84 | passes |
| llama3-8b-s16-ctx1024 | 0.62–0.71 | passes |

Two of four. **The reason the answer is "no" is not the one §0.4.471
gave.** That slice said the kernel was 1.5× slower than the lowering
everywhere; K1 showed that number was a host round trip over a ~1 MB
fixture and that on the device the kernel *already wins by 1.4–1.6× at
the shapes a served 8B model actually decodes at.* What is left is a
narrower and more specific "no": the kernel loses at small shapes, and
nothing in this tier has yet addressed a small shape.

## 8.2 The REJECTED alternative: register it conditionally on shape

The crossover in §2 is real, reproduced across five sessions, and its
sign never moved. The obvious move is therefore to register the kernel
**above a shape threshold** — claim `PAGED_ATTENTION` when
`numSeqs · numKvHeads · ctx` exceeds some constant, decline below it.
That is refused here, for three reasons that are worth writing down
because they will recur:

1. **It bakes one box's measurement into a library default.** The
   crossover is a GB10 number: 48 SMs, sm_121, LPDDR5X. The same kernel
   against the same XLA on a device with a different SM count and a
   different memory system has a different crossover, and a threshold
   constant compiled into `defaultInferenceKernelTemplates` would be a
   measurement of this machine presented as a property of the kernel.
2. **The crossover's location is partly an artifact of an unexplained
   instrument.** At `tinyllama-s1-ctx256` the dispatch floor ran
   144–252 µs against a claimed lane of 308–540 µs (§7.5) — 40–70% of
   the whole measurement is a cost neither lane's attention explains.
   A threshold fitted through that is fitted through noise.
3. **The claiming pass's decline path is the op itself** (§0.4.471's
   design), which is what makes an empty registry *safe*: a deployment
   with no KPTX tier runs the same program with the same numbers. A
   shape-conditional registry converts that clean binary into a silent,
   shape-dependent change of numeric behaviour — the claimed lane is
   1.2e-7 from the Double oracle and the emission is 2.44e-4 from it, so
   the two lanes are **not** bit-identical, and a deployment would get
   different logits above and below a threshold it never chose.

The deployment story therefore stays what it has been since H4: **opt in
explicitly, at shapes you measured yourself, with this test.** A
deployment serving 8B-shaped decodes on a GB10 has a measured 1.4–1.6×
reason to do so; §10 of [SERVING_RUNBOOK.md](SERVING_RUNBOOK.md) is the
command.

## 8.3 What is left, consolidated and re-ranked

§4 is K1's list and §7.4 is K2's re-ranking of it. This is the merged
order a next slice should take, with each item's expected win restated
against the *measured* baselines and each marked with which gate point
it moves.

| rank | item (§4 #) | expected win | moves the gate at |
|---|---|---|---|
| **1** | **Warp-per-lane `kptx_paged_scores`** (#3, narrowed to stage 1) — warp `w` owns context lane `j`; its 32 threads split `headDim`; `shflSync(DOWN)` reduces the dot. No barrier, no smem. **Unblocked by §0.4.493: the reduction is `warpReduceSumF32(acc)`, one call.** | Removes an **8× read amplification** on the K walk (§7.3). The largest measured defect in the chain, on the stage K2's null proved holds ≥ 84% of it. | all four points |
| **2** | **One CTA per (sequence, KV head)** (#1) — the GQA group shares one page walk, `group` score rows in registers. | Removes 3/4 of issued K+V. Worth more on the K side than the V side, and it **composes with 1** rather than competing. **~1.5–2.5×** at 8B. | 8B; helps all |
| **3** | **bf16 pools** (#2) — declined by name today. | Halves whatever 1 and 2 leave. **~2×** where memory-bound. | all |
| **4** | **Flash-decode context splitting** (#4) + **single-kernel online softmax** (#5). | The **only** two items that address the failing gate points: 0.67 CTA/SM at batch 1, and two launches + two grid-wide barriers worth ~30–60 µs. | **tinyllama-s1, tinyllama-s8** |
| **5** | **Register shave 65 → 64** (#7). | 3 → 4 blocks/SM if ptxas agrees. K2's evidence (stage 3 went 68 → 73 with no measurable change) is weak evidence *against* the declared-register cliff mattering. | speculative |
| — | **#6, stage 3's idle threads** | **DONE (K2), worth ~0 µs.** Kept for correctness and as item 2's `(part, d)` substrate. | — |

**The gate cannot be closed by ranks 1–3 alone.** They are the 8B-side
items and the 8B points already pass. Rank 4 is what the gate is waiting
on, and rank 4 is also the hardest. That ordering is deliberate anyway:
1–3 are where the microseconds are, and a deployment opting in at 8B
shapes gets them without the gate ever closing.

## 8.4 Fix the instrument before the next measurement is believed

Three named instrument gaps, each of which currently limits what any
future number here can claim:

| gap | what it blocks | where named |
|---|---|---|
| **the dispatch floor at large staged working sets** (236–313 µs for a 2-MFLOP multiply over 67–134 MB) and at small ones (144–252 µs, 40–70% of the small-point measurement) | the gate's measurability at `tinyllama-s1-ctx256` | §6, §7.5 |
| **per-stage timing** (three separately-timed custom calls, or CUPTI) | turns K2's "stage 3 is ≤ 16% of the chain" from a bound into a number, and would have cost K2 nothing to know in advance | §7.6 |
| **a measured memory-bandwidth ceiling for this box** (STREAM-style) | every "% of peak" in §3.1 and item #3 leans on NVIDIA's published 273 GB/s | §6 |

## 8.5 The first commit of the next slice, unchanged from §7.4

> **SUPERSEDED by [§9](#9-04493-w1-the-enabling-slice--and-the-blocker-that-was-not-there)
> (§0.4.493, W1).** The paragraph below is wrong on both of its facts
> and is kept verbatim because §9 is about *how* it was wrong. Rank 1 in
> §8.3 is no longer behind anything: `warpReduceSumF32` exists, is
> emission-pinned, and reduces 32 float lanes exactly on GB10.

`PtxIsa.kt`'s `mov` entry class-checks both operands from the type, so
the legal PTX `mov.b32 %r1, %f1` is rejected by Tlaloc's own ISA table.
A bit-typed `mov` must be class-`Any`, with its own `PtxIsaTest` pin.
**Every warp-reduced kernel that carries floats is behind that one
line** — rank 1 above included. It is a table change and a test, not a
kernel, and it should land on its own.

## 8.6 What this tier actually bought

Two slices, two commits, and the deliverables are honest to name:

- **A retired number.** §0.4.471's "1.5× slower" was the reason the
  registry was empty and the reason nobody looked further. It was a
  staging measurement. The kernel was never 1.5× slower at any shape
  anyone would serve.
- **A reusable instrument.** `KptxPagedAttentionBenchTest` runs inside
  `./gradlew test`, self-skips without a GPU, times three interleaved
  lanes from one session against pre-staged device buffers, asserts the
  lanes agree before timing either, and carries its own sanity
  assertions (one of which aborted a session, correctly).
- **A diagnosis with a bound on it.** The dominant cost is the GQA
  re-read through an 8×-amplified K walk — not the score matrix, which
  everyone expects and which is 2–5% of traffic. Stage 3 is ≤ 16% of the
  chain and needs nothing.
- **A certified null.** One kernel change, landed, correct, and measured
  to do nothing — reported as nothing. That is the result that produced
  the bound and the diagnosis, and it is the reason rank 1 above is
  stage 1 rather than "more parallelism somewhere".

What it did **not** buy is a single microsecond of measured speedup, and
this document does not claim one.

---

# 9. §0.4.493 (W1): the enabling slice — and the blocker that was not there

§8.5 named "the first commit of the next slice" and it has now been
made. The line it asked for is not the line that was needed, and the
honest report of this slice is that **the blocker §7.4 named did not
exist**. It was closed, in two different ways, before K2 wrote it down.

## 9.1 What §7.4 and §8.5 claimed

> `shflSync` moves `.b32` and requires `%r`-class registers, while the
> accumulator is `%f`. The natural spelling `mov.b32 %r1, %f1` is legal
> PTX but is **rejected by Tlaloc's own ISA table** (`PtxIsa.kt`'s `mov`
> entry class-checks both operands from the type, so `b32` demands `%r`
> on both sides).

Both halves were verified this slice, against `ptxas 13.0
-arch=sm_75` — the real assembler, not the ISA PDF and not inference —
and both are false.

**Half one: the table already accepted it.** §0.4.357 had made
`classOfType` return null for the b-types with a comment naming this
exact spelling (`mov.b32 %f2, %r8`, pyptx's bit-preserving cross-class
move). `mov.b32 %r1, %f1` has validated clean since that commit, which
predates K2 by a hundred sections. K2 read the `mov` entry's
`reg()`/`FromType(0)` and reasoned to the conclusion instead of running
the validator.

**Half two: the round trip is not needed at all.** `shfl.sync` is
`.b32`-typed, and `bN` is untyped bit storage — so
`shfl.sync.down.b32 %f2, %f1, 16, 0x1f, 0xffffffff` **assembles**, and
runs. The `%r`-only constraint was a `require` in §0.4.343's *Kotlin
wrapper*, thirteen characters of `d.cls == IsaRegClass.R32`, never an
ISA fact. A float warp reduction needs no `mov.b32` in it anywhere.

The probes, verbatim in their verdicts:

| spelling | ptxas |
|---|---|
| `mov.b32 %r1, %f1`, `mov.b32 %f2, %r1` | legal |
| `shfl.sync.down.b32 %f2, %f1, 16, 0x1f, …` | **legal** |
| `mov.b32 %rd3, %f1` | rejected — `Arguments mismatch for instruction 'mov'` |
| `shfl.sync.down.b32 %rd3, %rd1, …` | rejected |
| `mov.f32 %r1, %f1` | **legal** (ptxas treats `mov` as a bit-mover) |
| `add.u32 %f2, %f1, %f1` | rejected |

## 9.2 What the slice landed instead

The gap the probes *did* find is the opposite of the one named: since
§0.4.357, b-typed operands were checked by **nothing at all**. Dropping
the class check dropped the size check with it, so
`mov.b32 %rd1, %f1` — which ptxas refuses outright — validated clean
through Tlaloc's table and would have reached the driver as a JIT
error. Half the ISA's rule had been implemented.

- **`PtxIsa.kt`: bit-typed operands are width-checked** (`widthOfBitType`).
  `IsaRegClass` grows a `widthBits`; `b32` accepts any 32-bit class
  (`%r`, `%f`) and refuses `%rd` and `%p`; `b64` accepts `%rd` alone;
  `b16`/`b128` stay unchecked because KPTX has no class of either width
  and inventing a rejection would be guessing. `.wide` widens the
  expected width the same way it widens the expected class.
- **`WarpIntrinsics.kt`: `shflSync` rejects on width, not class.** `%f`
  is now a legal shuffle operand, which is the whole unblock.
- **`warpReduceSumF32(acc)`** — the natural spelling asked for: one
  call, five `shfl.sync.down`/`add.rn.f32` steps at offsets 16…1,
  scratch allocated from the enclosing scope, no barrier and no shared
  memory.
- **`movB32(d, a)`** — bit reinterpretation as a named operation
  (`Float.toRawBits` / `fromBits`), for the operands that genuinely are
  `%r`-only. Deliberately *not* used by `warpReduceSumF32`.

**REJECTED: making the b-types class-`Any`, as §8.5 specified.** That is
the state §0.4.357 already left the table in, and it is why
`mov.b32 %rd1, %f1` passed. The principled rule is not "no check" but
"the check PTX actually makes".

**REJECTED: relaxing KPTX's class check on `mov.f32`.** ptxas accepts
`mov.f32 %r1, %f1`; KPTX keeps refusing it, and `keepsTheStricterClassCheckOnTypedMoves`
pins the divergence so it stays a decision. An `%r` holding an f32 value
is a bug in every kernel in this repo.

## 9.3 The certification

- **ISA pins** (`PtxIsaTest`): accepted spellings `mov.b32 %r1, %f1`,
  `mov.b32 %f1, %r1`, `shfl.sync.down.b32 %f2, %f1, …`, `mov.b64 %rd1, %rd2`;
  rejected by name `mov.b32 %rd1, %f1`, `mov.b64 %r1, %rd1`,
  `shfl.sync.down.b32 %rd3, %rd1, …` (both data operands), `mov.b32 %p1, %f1`,
  and `mov.f32 %r1, %f1`.
- **Byte-level emission pin** (`WarpIntrinsicsTest.floatWarpReductionIsOneCall`):
  the ten-line body, exact, plus `assertTrue("mov.b32" !in text)`.
- **The real proof** (`WarpIntrinsicsGpuTest.shflF32WarpSumComputesCorrectly`,
  `:runtime-cuda`, GB10 sm_121): a DSL-authored kernel warp-reduces 32
  lanes holding `0f..31f` and lands **496.0f exactly** — driver-JIT
  accepted, numerically correct, `skipped="0"` in the result XML. The
  values are integral so the assertion is an equality, not a tolerance.

## 9.4 Deferred by name

- **Sub-warp reduction widths.** `warpReduceSumF32` pins the
  clamp/segment word to `0x1f` (a full 32-lane segment). `width < 32`
  needs `0x1f or ((32 - width) shl 8)` and a test that a partial warp
  actually segments. No kernel here needs it.
- **`f32x2` / packed-half shuffles.** `b64` shuffles do not exist; a
  paired reduction would need two `shfl` per step. Not attempted.
- **The stage-1 warp-mapped kernel itself** — still §8.3 rank 1, still
  unwritten. This slice removed its only named blocker and nothing else:
  **no microsecond of §2's table moves on this commit**, and none is
  claimed.
  > **§0.4.494 (W2) wrote it** on the very next commit, and §2's table
  > did move at the 8B points. See [§10](#10-04494-w2-the-warp-mapped-stage-1--the-8-read-amplification-paid-down).

## 9.5 The lesson worth keeping

§7.4 and §8.5 spent two paragraphs and a rank-1 dependency on a blocker
that a one-line `validateInst` call and a six-line `.ptx` file through
`ptxas` would have dissolved in either direction. The probe cost four
minutes. Reasoning about a validator by reading it cost a slice's
implementation budget and a wrong entry in two ranked lists. **When a
doc names a blocker, the next slice runs it before it believes it.**

---

# 10. §0.4.494 (W2): the warp-mapped stage 1 — the 8× read amplification, paid down

**§8.3 rank 1, written exactly as §7.4 specified it, and this time the
numbers moved.** W1 (§9) found the blocker it was waiting on did not
exist; this slice spends the unblock.

## 10.1 What landed

`kptx_paged_scores` maps a **warp** to a context lane instead of a
thread:

```
  nWarps = ntid / 32                    // CTA-wide, branch-uniform
  w = tid / 32,  lane = tid % 32
  warp w owns j = w, w+nWarps, w+2*nWarps, ... < ctx
  its 32 lanes split d = lane, lane+32, ... < n_d
  warpReduceSumF32(acc); lane 0 scales and stores S[row, j]
```

Every `K[block, off, kvh, lane…]` request is now 32 consecutive f32 —
one 128 B transaction — where §7.3 measured 32 separate 32-byte sectors
4096 B apart. The `Q` request coalesces the same way. The pointer stride
is the literal `128` = **32 lanes × 4 B**, a warp constant and not a
dim-derived one, so the §0.4.414 sentinel-dims rule is untouched.

**No barrier and no shared memory**, which is the whole reason this arm
is cheaper than stage 3's `(part, d)` split: the entire warp shares `j`,
so the live/dead test, the page resolution and the block-table load
(a broadcast — one transaction) are warp-uniform, and `shfl.sync` is
what reconverges the one divergence there is, lane 0's store.

**Three CTA-uniform reasons to decline**, each keeping the §0.4.471
program verbatim under `SCALAR_J` — the `nsplit < 2` precedent:

| guard | why it is not optional |
|---|---|
| `ntid % 32 != 0` | the member mask is `0xffffffff`; on a partial warp that is a lie and `shfl.sync` waits on lanes that do not exist |
| `ntid < 32` (`nWarps == 0`) | the `j` loop would have no owner |
| `headDim < 32` | still *correct* (lanes past `n_d` contribute a zero) but cannot fill a 128 B transaction, and pays ten reduction ops for under one FMA per lane |

Declared registers **65 → 73**, with **no occupancy change**: 3
blocks/SM, 768 threads, 50%, still register-limited.

## 10.2 Correctness first: the oracle got *better*

The dot is now a 32-way `shfl` tree rather than a sequential
`fma.rn.f32` chain — the second reassociation this chain has taken
(§0.4.482 was the first). `KptxPagedAttentionKernelTest` at the
certification fixture — **permuted** block table, ragged `seqLens`
(full window / single token / mid-page stop / page-aligned stop), GQA
group 4, `headDim 64`, `ctx 128` — against the interpreter's Double
paged walk:

| | worst \|delta\| vs the Double oracle |
|---|---|
| §0.4.471 / §0.4.482 sequential chain | 1.1920929e-7 |
| §0.4.494 warp tree | **8.940697e-8** |

A tree sum of 64 terms rounds better than a chain of 64. That is the
textbook result and it is still a *measurement*, not a guarantee: a
longer context would round differently and the oracle is what would say
so. `skipped="0"` in the result XML — the lane ran on the GB10. The
bench's own per-point "the two lanes agree" assertion passed at all four
points before anything was timed.

## 10.3 The numbers

Method as §7.1: two sessions before the change and two after, all four
within one hour, same box, same test, floors over 20 reps after 3
warmup, lanes interleaved. **DEVICE floors, µs.**

| point | c before | c after | u before | u after | c/u before | c/u after |
|---|---|---|---|---|---|---|
| tinyllama-s1-ctx256 | 373.6, 433.3 | 300.1, 197.5 | 107.1, 228.8 | 122.5, 206.5 | 3.49, 1.89 | 2.45, 0.96 |
| tinyllama-s8-ctx512 | 268.3, 343.6 | 217.9, 352.1 | 247.7, 152.1 | 134.5, 337.4 | 1.08, 2.26 | 1.62, 1.04 |
| **llama3-8b-s8-ctx1024** | 932.7, 1097.5 | **596.8, 838.3** | 1141.8, 1295.6 | 1217.5, 1348.9 | 0.82, 0.85 | **0.49, 0.62** |
| **llama3-8b-s16-ctx1024** | 1615.4, 1578.1 | **1102.4, 1211.8** | 2338.6, 2454.7 | 2203.8, 2283.9 | 0.69, 0.64 | **0.50, 0.53** |

**What is claimed.** At both Llama-3-8B-shaped points the claimed lane's
after-range lies **entirely below** its before-range — 838.3 < 932.7 and
1211.8 < 1578.1 — while the unclaimed control lane did not move (1141.8,
1295.6 → 1217.5, 1348.9; 2338.6, 2454.7 → 2203.8, 2283.9) and matches
K1's and K2's sessions to within a few percent. Against K1's and K2's
recorded claimed floors (882.8 / 1025 / 1098 / 1116 at s8; 1431 / 1533 /
1615 / 1695 at s16) the after numbers are below every one of them. The
honest size of the win is **≈1.3–1.5× on the claimed lane at 8B
shapes**, and it shows up as achieved bandwidth on unchanged issued
traffic: c GB/s at s16 went 83.1, 85.0 → **121.7, 110.8**; at s8 61.1,
72.0 → **80.0, 112.4**.

> **§0.4.495 (W3) brings that 1.3–1.5× DOWN.** Measured against *K1's*
> recorded claimed floors rather than W2's own before-sessions, and over
> six sessions rather than two, the same kernel is **×1.17 at s16 and
> ×1.29 at s8** with the control lane within 4% in both. Read 1.3–1.5×
> as the upper end of a **×1.2–1.3** win. W3 also finds a win W2 did not
> claim: **×1.26 at `tinyllama-s8-ctx512`**. See
> [§11.2](#112-against-the-2-baselines-point-by-point).

**What is NOT claimed.** The two TinyLlama points. Their before- and
after-ranges overlap in both directions, and the reason is §7.5's
unexplained instrument showing its teeth again: across these four
sessions the dispatch floor ran 37.6–322.6 µs against measurements of
197.5–433.3. The `0.96×` in the after-column at `tinyllama-s1-ctx256` is
a 197.5 µs measurement sitting on a **146.8 µs dispatch floor** — it is
the floor, not the kernel, and it is not evidence the gate moved.

## 10.4 Why 1.4× and not 8×

§7.3 put an 8× read *amplification* on this walk, and paying it down
bought 1.3–1.5× of the whole chain. Both facts are true and the gap is
the interesting part:

- The chain is three stages. If stage 1 were 80% of it (stage 3 is
  ≤16% by §7.2's bound, stage 2 is a row softmax), a 1.4× end-to-end
  needs stage 1 itself to have got ≈1.6× faster, not 8×.
- **Amplified sectors are not all DRAM reads.** The 4× GQA multiplicity
  means four query heads walk the *same* pages, and CTAs sharing a
  sequence share pages too; a fetched sector that was waste for one warp
  is an L2 hit for another. The 8× is an accurate count of *sectors
  requested per useful byte*; it was never a claim about bytes crossing
  the memory controller.
- The warp reduction is not free: **ten ops per `j`** (five `shfl`, five
  `add`) against `headDim/32` FMAs per lane — 4 at `headDim 128`, 2 at
  `headDim 64`. That ratio is exactly why the mapping helps most where
  the walk is memory-bound and least at TinyLlama shapes whose 0.52 MB
  of KV fits in L2.

## 10.5 The gate, unchanged

§5's gate stands: the claimed lane's device floor below the unclaimed
lane's at **every** point in §2, one session, interleaved.
`defaultInferenceKernelTemplates` stays empty and §8.1's decision is
unchanged. The scoreboard after W2 is **two of four**, the same two as
before — the 8B points now pass by twice the margin they did, and the
TinyLlama points are still not measurable at this instrument. §8.4's
item (fix the instrument) is now the thing standing between this tier
and an answer at the small points, ahead of any further kernel work
there.

> **§0.4.495 (W3) makes that promotion formal**: §8.4's instrument row
> is **rank 1** of the merged list in
> [§11.4](#114-83s-ranked-list-re-ranked-against-this-evidence), ahead
> of every kernel item. It also amends "the same two as before" — over
> six sessions `tinyllama-s8-ctx512`'s claimed floor **did** fall
> (×1.26, control unmoved), so the gap there is now 1.58×, not 1.93×.

## 10.6 Deferred by name

- **The GQA fusion (§8.3 rank 1's other half)** — one CTA per
  `(seq, kvHead)` rather than per `(seq, queryHead)`, removing 3/4 of
  the K stream outright. Untouched; still the largest remaining term,
  and now cheaper to reason about because the stream it removes is a
  clean one.
- **The reduction's op count.** Ten ops per `j` is the price of the
  coalescing at every `headDim`. A `headDim 32`/`64` kernel could split
  `j` across half- or quarter-warps and reduce over 16 or 8 lanes
  instead — which is exactly the sub-warp width `warpReduceSumF32`
  deferred in §9.4. That is the change that would address a TinyLlama
  point, and it should not be attempted before §8.4.
- **The `SCALAR_J` arm is unexercised on hardware**, like stage 3's
  `nsplit < 2` arm before it: every fixture in the suite launches 256
  threads at `headDim >= 64`. Only the emitted-PTX pin covers it. A GPU
  arm at `headDim 16` would close both.
- **Per-stage timing.** §7.6 asked for it and §10.4's first bullet is
  the reason: every apportionment above is still a bound derived from
  end-to-end floors.

---

# 11. §0.4.495 (W3): six sessions, the tier re-measured, and the registry question re-answered

**This section re-measures the whole §2 grid at HEAD — W2's warp-mapped
`kptx_paged_scores` in the claimed lane — across SIX sessions inside one
hour, re-ranks §8.3 against that evidence, and revisits §8.1. The
decision does not change. Two of the things §8.2 and §10.3 said about
*why* do change, and one number W2 claimed comes down.**

## 11.1 The measurement

Method unchanged from §7.1 and §10.3: `./gradlew :benchmarks:jvmTest
--tests "*KptxPagedAttentionBenchTest*" --rerun-tasks`, six times, all
within one hour on 2026-09-22. Each session is one `PjrtSession`,
executables compiled and cached, three lanes interleaved, floor over 20
reps after 3 warmup. `skipped="0"` in every result XML — every session
ran on the GB10. Every point's "the two lanes agree" assertion passed
before anything was timed, in all six.

**DEVICE floors, µs. `c/u < 1.00` means the KPTX kernel wins.**

### `tinyllama-s1-ctx256` — ctx 256, grid 32, 0.67 CTA/SM

| session | c | u | c/u | dispatch floor |
|---|---|---|---|---|
| 1 | 287.9 | 149.7 | 1.92 | 53.8 |
| 2 | 395.9 | 329.7 | 1.20 | 126.6 |
| 3 | 280.6 | 127.9 | 2.19 | 46.6 |
| 4 | 235.9 | 193.3 | 1.22 | 47.8 |
| 5 | 264.1 | 294.1 | **0.90** | 122.3 |
| 6 | 484.5 | 232.0 | 2.09 | 132.4 |
| **floor of floors** | **235.9** | **127.9** | **1.84** | 46.6 |

### `tinyllama-s8-ctx512` — ctx 512, grid 256, 5.33 CTA/SM

| session | c | u | c/u | dispatch floor |
|---|---|---|---|---|
| 1 | 217.1 | 312.4 | **0.69** | 45.1 |
| 2 | 223.0 | 137.8 | 1.62 | 40.0 |
| 3 | 337.4 | 249.1 | 1.35 | 42.0 |
| 4 | 228.5 | 217.8 | 1.05 | 77.5 |
| 5 | 291.2 | 168.1 | 1.73 | 127.7 |
| 6 | 229.7 | 138.1 | 1.66 | 43.6 |
| **floor of floors** | **217.1** | **137.8** | **1.58** | 40.0 |

### `llama3-8b-s8-ctx1024` — ctx 1024, grid 256, 5.33 CTA/SM

| session | c | u | c/u | dispatch floor |
|---|---|---|---|---|
| 1 | 682.0 | 1180.5 | **0.58** | 155.5 |
| 2 | 798.3 | 1233.9 | **0.65** | 245.6 |
| 3 | 758.4 | 1224.0 | **0.62** | 154.0 |
| 4 | 949.3 | 1334.5 | **0.71** | 204.2 |
| 5 | 762.1 | 1174.4 | **0.65** | 157.4 |
| 6 | 893.6 | 1238.5 | **0.72** | 166.0 |
| **floor of floors** | **682.0** | **1174.4** | **0.58** | 154.0 |

### `llama3-8b-s16-ctx1024` — ctx 1024, grid 512, 10.67 CTA/SM

| session | c | u | c/u | dispatch floor |
|---|---|---|---|---|
| 1 | 1228.5 | 2255.9 | **0.54** | 144.1 |
| 2 | 1276.7 | 2241.7 | **0.57** | 167.7 |
| 3 | 1248.3 | 2228.9 | **0.56** | 157.6 |
| 4 | 1427.7 | 2432.0 | **0.59** | 149.3 |
| 5 | 1284.5 | 2420.4 | **0.53** | 221.2 |
| 6 | 1230.4 | 2347.6 | **0.52** | 416.3 |
| **floor of floors** | **1228.5** | **2228.9** | **0.55** | 144.1 |

## 11.2 Against the §2 baselines, point by point

§2's row is K1's single session (§0.4.481). The comparable quantity
across sessions is the ratio (§0.4.337); the absolute columns are given
so the drift is visible rather than hidden.

| point | §2 c / u / (c/u) | W3 floor-of-floors c / u / (c/u) | c moved | u (control) moved | gate |
|---|---|---|---|---|---|
| tinyllama-s1-ctx256 | 177.5 / 93.0 / **1.91** | 235.9 / 127.9 / **1.84** | ×1.33 *slower* | ×1.38 *slower* | **FAILS** |
| tinyllama-s8-ctx512 | 272.9 / 141.7 / **1.93** | 217.1 / 137.8 / **1.58** | **×1.26 faster** | ×1.03 (unmoved) | **FAILS** |
| llama3-8b-s8-ctx1024 | 882.8 / 1209.8 / **0.73** | 682.0 / 1174.4 / **0.58** | **×1.29 faster** | ×1.03 (unmoved) | passes |
| llama3-8b-s16-ctx1024 | 1431.1 / 2322.4 / **0.62** | 1228.5 / 2228.9 / **0.55** | **×1.17 faster** | ×1.04 (unmoved) | passes |

Three things to read off it, and the third is uncomfortable.

**One: the 8B side is now a reproduced result, not a single session.**
Six of six sessions have the claimed lane ahead at both 8B points, with
c/u in **0.58–0.72** at s8 and **0.52–0.59** at s16 — that is
**1.4–1.7× faster at s8 and 1.7–1.9× at s16**, and no session's ratio
comes within 28% of 1.00. §2's single-session 0.73/0.62 was right and
was conservative.

**Two: the tinyllama-s1 row's absolute columns moved together.** Both
lanes are ~1.35× slower than K1's session, control included, so that row
is a session-scale shift and **nothing about the kernel is claimed from
it**. The ratio — the quantity §0.4.337 says to compare — went 1.91 →
1.84, which is inside the noise established below.

**Three: W2's headline comes down.** §10.3 claimed "≈1.3–1.5× on the
claimed lane at 8B shapes", measured against its own two pre-change
sessions (1615.4/1578.1 at s16, 932.7/1097.5 at s8). Against *K1's*
recorded claimed floors, which are lower, the same kernel is **×1.17 at
s16 and ×1.29 at s8** — a real win, the same sign, but at the bottom of
the band W2 named and below it at s16. The control lane is within 4% in
all three comparisons, so this is not a control artifact: it is W2's
before-sessions having been on the slow side of their own spread.
**The honest consolidated number for the warp mapping is ×1.2–1.3 on the
claimed lane at 8B shapes**, and §10.3's 1.3–1.5× should be read as the
upper end of it.

## 11.3 What six sessions do to the instrument's credibility

§6, §7.5 and §8.4 each named the dispatch floor as unexplained. Six
sessions make it quantitative, and it is worse than "noisy".

- **The dispatch floor spans 40.0 → 416.3 µs — a 10.4× range — for the
  identical 2-MFLOP elementwise multiply**, over the same staged buffer
  sets, on the same box, within one hour. Its largest value (416.3) and
  its smallest (40.0) are not even at the same point: it does not scale
  cleanly with the staged working set either.
- As a share of the measurement it sits at **17–46% of the claimed lane
  and 25–57% of the unclaimed lane** at `tinyllama-s1-ctx256`, and at
  **12–44% / 14–76%** at `tinyllama-s8-ctx512`.
- **§2's "~2.5× above the floor, and that ratio holds in each session"
  does not survive.** Floor-subtracted, `(c − floor)/(u − floor)` across
  these six sessions is **2.44, 1.33, 2.88, 1.29, 0.83, 3.54** at
  tinyllama-s1 and **0.64, 1.87, 1.43, 1.08, 4.05, 1.97** at
  tinyllama-s8. That is a 4–6× spread in a derived quantity K1 reported
  as stable. It was stable across K1's three sessions and it is not
  stable across six.
- **The two failing gate points straddle 1.00 in both directions.**
  Session 5 has the claimed lane *winning* at tinyllama-s1 (0.90×);
  session 1 has it winning at tinyllama-s8 (0.69×). No single session
  passed both, but the instrument's noise band at both points contains
  the gate's own threshold.

The consequence is a statement about the gate itself, not about the
kernel: **§5's gate — "below the unclaimed lane at every point, one
session, interleaved" — is a decision procedure a lucky session could
pass.** That is not a reason to weaken it, and it is not weakened here.
It is the reason §8.4's first row is promoted to rank 1 in §11.4.

## 11.4 §8.3's ranked list, re-ranked against this evidence

| rank | item | what changed |
|---|---|---|
| **1** | **Fix the instrument** — the dispatch floor first (§8.4 row 1), then per-stage timing (§8.4 row 2). | **PROMOTED from a supporting list to the head of the ranked list.** §8.3's rank 4 items are the only ones that move the gate, and §11.3 says there is currently **no instrument that could tell whether they worked**: a change worth 30–60 µs at a point whose floor moves by 90 µs between sessions is unmeasurable by construction. Every kernel item below this one is either already-passing (2, 4) or unverifiable (3, 5). |
| **2** | **One CTA per (sequence, KV head)** (§4 #1) — the GQA group shares one page walk. | **Unchanged in size, restated in value.** Still the largest remaining term, and W2 made the stream it removes a clean one (§10.6). But it is an 8B-side item and the 8B side already passes the gate, so its return is **microseconds for opt-in deployments, not gate movement**. |
| **3** | **Flash-decode context splitting + single-kernel online softmax** (§4 #4, #5). | **Held, not demoted in importance — blocked.** Still the only two items that address the failing points. Attempting them before rank 1 would spend a slice and produce a number nobody could believe; that is precisely the §0.4.482 lesson, re-learned on the measurement side. |
| **4** | **bf16 pools** (§4 #2). | Unchanged. Halves whatever the others leave, on the 8B side. |
| **5** | **Sub-warp reduction widths** (§9.4, §10.6). | **PROMOTED onto the list proper**, from a deferral. At `headDim 64` W2's mapping pays **ten reduction ops against two FMAs per lane** — that ratio is exactly the TinyLlama regime, and it is the one *kernel* explanation on the table for why the small points did not move. Behind rank 1 for the same reason as rank 3. |
| **6** | **Register shave 65 → 64** (§4 #7). | **Effectively closed as an idea.** `kptx_paged_scores` went 65 → 73 declared slots in W2 with no occupancy change and a measured win; `kptx_paged_out` went 68 → 73 in K2 with no change either way. Two data points now say the declared-register cliff is not where this chain's time is. |
| — | **Warp-per-lane `kptx_paged_scores`** (§8.3 rank 1) | **DONE (W2).** ×1.17–1.29 on the claimed lane's floor-of-floors at both 8B points against K1's, control within 4%; ×1.26 at tinyllama-s8 as well (§11.2), which §10.3 did not claim and which six sessions now support. Nothing at tinyllama-s1. |
| — | **Stage 3's idle threads** (§4 #6) | DONE (K2), worth ~0 µs. Unchanged. |

**The one genuinely new item.** §11.2's second row says the warp mapping
bought ×1.26 at `tinyllama-s8-ctx512` while its control lane stood still
— so a TinyLlama point *did* move, by roughly what the 8B points moved
by, and the gate still failed there because the unclaimed lane at that
point is 137.8 µs and the claimed lane would have to beat it outright.
**The remaining gap at tinyllama-s8 is 1.58×, not 1.93×.** That is the
closest the gate has ever been at a failing point.

## 11.5 §8.1 revisited: can the kernel be turned on today?

> **NO. Unchanged. `defaultInferenceKernelTemplates` stays empty, and
> the opt-in spelling (`kptxInferenceKernelTemplates` +
> `KptxPagedAttention.register`) remains the only way the kernel enters
> an executable.**

The scoreboard against §5's gate, on floors of floors over six sessions:

| point | §8.1 (K1+K2) | W3 | gate |
|---|---|---|---|
| tinyllama-s1-ctx256 | 1.27–2.14 | 0.90–2.19 (fof **1.84**) | **FAILS** |
| tinyllama-s8-ctx512 | 1.55–1.93 | 0.69–1.73 (fof **1.58**) | **FAILS** |
| llama3-8b-s8-ctx1024 | 0.73–0.84 | 0.58–0.72 (fof **0.58**) | passes |
| llama3-8b-s16-ctx1024 | 0.62–0.71 | 0.52–0.59 (fof **0.55**) | passes |

Still two of four, and the two that fail are the two that have always
failed.

### The three reasons §8.2 gave, answered against W3's evidence

§8.2 rejected shape-conditional registration for three stated reasons.
W3 does **not** recommend changing that decision, and each reason is
stronger than it was, two of them now quantitatively:

1. **"It bakes one box's measurement into a library default."**
   **Stronger, and in a way §8.2 did not anticipate.** The objection was
   that the crossover is a property of *this* GB10. W3 shows it is not a
   stable property of this GB10 either: the same kernel against the same
   lowering on the same box produced c/u from **0.90 to 2.19** at
   `tinyllama-s1-ctx256` within one hour. A threshold constant would be
   fitted to a quantity that moved 2.4× while nothing changed.
2. **"The crossover's location is partly an artifact of an unexplained
   instrument."** **Now measured.** The dispatch floor spans
   **40.0–416.3 µs** across these sessions (§11.3) — 12–46% of the
   claimed lane and 14–76% of the unclaimed lane at the small points —
   and the floor-subtracted small-point ratio spans **0.64–4.05**, where
   K1 reported it as holding steady near 2.5×. Any threshold fitted
   through the crossover would be fitted through that.
3. **"A shape-conditional registry is a silent, shape-dependent change
   of numeric behaviour."** **Unchanged, and now measured at the decode
   shapes themselves rather than inferred from two oracle numbers.** The
   bench's own lane-agreement column is the direct measurement: the two
   lanes' outputs differ by **4.8e-5** (tinyllama-s1), **4.9e-5**
   (tinyllama-s8), **2.9e-5** (8b-s8) and **2.7e-5** (8b-s16) max-abs,
   at every point, in every session. Both lanes are correct — the kernel
   is 8.9e-8 from a Double oracle (§10.2) and the gap is the emission's
   TF32 `dot_general` (§0.4.471) — but they are **not** the same
   function to 5 decimal places, and a threshold would hand a deployment
   different logits above and below a line it never chose.

### What *did* change in the deployment story

The opt-in recommendation is no longer a hedge. §8.2 closed with "a
deployment serving 8B-shaped decodes on a GB10 has a measured 1.4–1.6×
reason to do so", from one session. Six sessions make that
**1.4–1.7× at `llama3-8b-s8-ctx1024` and 1.7–1.9× at
`llama3-8b-s16-ctx1024`, with no session's ratio inside 28% of parity**.
§10 of [SERVING_RUNBOOK.md](SERVING_RUNBOOK.md) is still the command.

## 11.6 W3's deferrals, by name

- **The instrument, which W3 diagnosed and did not fix.** Rank 1 above
  is a full slice: the dispatch floor's 10.4× spread is unexplained, and
  the candidate causes (PJRT's per-execute allocator behaviour, the
  event-await path, the staged buffers' residency) were not probed. This
  slice measured the defect and re-ranked around it; it did not chase it.
- **Per-stage timing**, again (§7.6, §8.4). Everything in §11.4's
  apportionment remains a bound from end-to-end floors, including the
  claim that rank 5 explains the TinyLlama points.
- **A measured memory-bandwidth ceiling for this box** (§6). Untouched.
- **No kernel changed on this commit.** §11.2's numbers are HEAD
  (§0.4.494) re-measured, not a new kernel measured. Nothing in §2's
  table was re-written; §2 remains K1's session, as it should.
