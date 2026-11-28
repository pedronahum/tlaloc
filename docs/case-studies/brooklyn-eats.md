# Case study: an 8B restaurant agent that beats its 65B teacher

A tool-using agent for Brooklyn restaurants, post-trained and served on one DGX Spark (GB10,
121.7 GiB of unified memory), every stage in Kotlin on Tlaloc. TwIL-8B, fine-tuned with LoRA on
217 verified trajectories of Ripple-65B, passes more held-out tasks than Ripple does. Served with
NVFP4 MLP weights its episodes take 31% less time, for 2.4 points on dev and 1.1 on test.

| Model | Dev (382 tasks) | Test (375 tasks) |
|---|---|---|
| TwIL-8B, base | 259 (67.8%) | 235 (62.7%) |
| Ripple-65B NVFP4, the teacher | 318 (83.2%) | 309 (82.4%) |
| TwIL-8B SFT, bf16, two runs | 334, 335 (87.6%) | 318, 318 (84.8%) |
| TwIL-8B SFT, NVFP4 MLP, FP8 attention and KV, two runs | 326, 325 (85.2%) | 316, 312 (83.7%) |
| TwIL-8B SFT + one rejection-sampling round | 323 (84.6%) | not run |
| TwIL-8B distilled from Ripple's next-token distributions (best of three) | 322 (84.3%) | 313 (83.5%) |

Greedy decoding, offline tools, scored by the verifier at eea3403a. Percentages for the two-run
rows are the means. The pipeline and every number's provenance are in
[`examples/brooklyn-eats`](../../examples/brooklyn-eats/).

## Data and tasks

Open data at a fixed cutoff (2026-11-13): 4,880 OpenStreetMap places, 74,412 NYC DOHMH
inspection rows, 13,921 NYC 311 complaints, 8,962 Google Local food businesses with 1,231,484
reviews, and menus from 3,450 Wayback Machine captures, joined into 9,233 restaurants. A tool
server answers eight tools over them (search, details, inspections, complaints, reviews, menu,
geocode, web search).

The 4,719 tasks are generated from the database through the same tools, from 2 to 5 seeded
templates per kind and no language model, in seven categories: lookup, filter, comparison,
multihop, menu, complaints and unanswerable. Each task's ground truth is read from recorded tool
results, so every fact it asks for is one an agent can see. Splits are separated by restaurant.
The verifier checks the facts an answer states, items it invents and places it lists that the
search would not return.

## Teacher, student, training

Ripple-65B (NVFP4, served through Triton) answered 300 training questions once each; 222 passed
the verifier, and 217 of them render as TwIL chat sequences (826,440 tokens, 115,364 of them the
assistant's). TwIL-8B got rank-16 LoRA on every linear layer, frozen base in bf16, lr 1e-4 with
a cosine decay, 2 epochs: 430 steps in 7,641 s, the machine's memory peaking at 100.6 GiB. The
merge into a bf16 checkpoint takes 113 s.

The student passes 16 more dev tasks and 9 more test tasks than the teacher whose trajectories
it learned from. It was trained only on trajectories that passed, and runs the same tools with
the same budget.

Two further recipes did not improve on it:

- **Rejection sampling.** The SFT student sampled 300 more training questions; its 256 passing
  trajectories trained one epoch of new adapters on top: 323 on dev against 334.
- **Distillation.** Ripple scored the kept teacher episodes in process (its top 16 next-token
  log-probabilities per position), and the base TwIL trained against them and the written
  tokens. With weight 0.5 or 0.8 on the written token, or on the answer text only, all three
  students land below SFT on both splits (dev 317 to 322, test 305 to 313).

## Serving the student in NVFP4

The merged checkpoint exports with NVFP4 MLP weights, FP8 attention weights and an FP8 KV
cache: 6,549 MiB of staged weights against 17 GiB in bf16. The machine's memory peaks at 30.7 GiB while
serving it, against 72.0 GiB for the base TwIL in bf16. Over all four streams it decodes 37.5
tokens/s instead of 26.4, and the median episode falls from 66.3 s to 45.9 s.

## Run-to-run spread

Serving is not bitwise reproducible, so each SFT student ran dev and test twice, each run on a
new server start:

| Model | Split | Passes | Verdicts that differ | Trajectories repeated token for token |
|---|---|---|---|---|
| SFT bf16 | dev | 334, 335 | 7 | 345 of 382 |
| SFT bf16 | test | 318, 318 | 6 | 333 of 375 |
| SFT NVFP4 | dev | 326, 325 | 31 | 162 of 382 |
| SFT NVFP4 | test | 316, 312 | 30 | 171 of 375 |

Pass counts move by at most 4 between runs. The NVFP4 student changes verdicts on 8% of the
tasks against 2% for bf16, but its pass counts stay as close. The dev loss of 9 tasks is larger
than either model's difference between its runs; the test loss of 4 equals the NVFP4 student's
own.

## Reproducing it

`run.sh` runs the stages in order, from `models`, `snapshot` and `ingest` through `teach`,
`train`, `eval` and `serve`. `manifest.json` pins the inputs: data digests, model revisions, settings,
serving configurations and the Tlaloc version. It also records every derived output: tables,
tasks, trajectories, adapters and each evaluation run's passes. `run.sh verify` recomputes them
and names what changed.

A data snapshot (release asset) holds the raw sources that no longer return what they returned
on the cutoff day, and the sampled trajectories, so a later run rebuilds the same database and
trains on the same shards. A clean-room clone with an empty Gradle home and mavenLocal matched
every quick-scope entry. GPU training is not bitwise reproducible: the adapters' SHA-256 is
reported, not compared, and evaluation passes are compared within the measured spread.
