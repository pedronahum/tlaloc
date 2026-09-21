# KPTX paged attention — the performance tier's baselines and diagnosis

**§0.4.481 (Phase H4b, slice K1). Measure and diagnose; no kernel was
rewritten in this slice.**

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
- **The `mov.b32` ISA-table gap** — see §7.4. One line of table, one
  pin, and it gates everything warp-reduced that carries floats.
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
| **1** | **Warp-per-lane `kptx_paged_scores`** (#3, narrowed to stage 1) — warp `w` owns context lane `j`; its 32 threads split `headDim`; `shflSync(DOWN)` reduces the dot. No barrier, no smem. | Removes an **8× read amplification** on the K walk (§7.3). The largest measured defect in the chain, on the stage K2's null proved holds ≥ 84% of it. | all four points |
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
