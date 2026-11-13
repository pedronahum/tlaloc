# Brooklyn Eats: a restaurant agent post-trained with Tlaloc

An end-to-end post-training example, run on one DGX Spark (GB10, 128 GB unified memory):
open data about Brooklyn restaurants, a tool server over it, trajectories from a served teacher,
LoRA fine-tuning and distillation of an 8B student, evaluation with checked answers, and the
tuned student served with live tools. Every stage runs in Kotlin on Tlaloc; Python appears only
where a check compares against a reference library.

Branch `feat/brooklyn-eats`; example in `examples/brooklyn-eats/`; progress in
`docs/work-log/brooklyn-eats-progress.md`.

## Models

| Role | Model | Licence | In Tlaloc |
|---|---|---|---|
| Student | webAI-Official/TwIL-8B (Granite, 8.79B dense, 40 full-attention layers) | webAI Non-Commercial 1.0: research and education only; the tuned adapter and merged checkpoint inherit it | `HfCausalLm` loads, trains (LoRA) and saves it; `ChatTemplate.TWIL` renders tools, tool calls and tool responses, `TwilOutputParser` parses them; served bf16 and NVFP4 through Triton and `tlaloc-endpoint` |
| Teacher | ricursiveAI/Ripple-65B-Preview (+ DFlash2 draft, NVFP4) | Apache-2.0 | Served on this machine with tool calls (`ChatTemplate.RIPPLE`, `RippleOutputParser`) |

The two tokenizers give the same id for 100,259 shared tokens; only special tokens differ
(TwIL's `<think>`, `<tool_call>`, `<tool_response>`; Ripple's `<|extra_id_*|>`). Content tokens
of a message therefore have the same ids under both, which makes token-level distillation from
Ripple's logits possible on the content spans.

## Data (Brooklyn only)

| Source | What | Access | Licence |
|---|---|---|---|
| OpenStreetMap via Overpass | restaurants, cafes, fast food, bars: name, cuisine, hours, website, menu URL, coordinates | live API and snapshot | ODbL |
| NYC DOHMH Restaurant Inspection Results (Socrata `43nn-pn8j`) | inspections, scores, letter grades, violation codes and text, CAMIS id, coordinates | live SODA API and snapshot | NYC Open Data terms |
| NYC 311 Service Requests (Socrata `erm2-nwe9`) | food poisoning and food establishment complaints | live SODA API and snapshot | NYC Open Data terms |
| UCSD Google Local (McAuley lab), New York | reviews and ratings up to about 2021 | bulk download | research use; never committed |
| Restaurant websites (OSM `website` / `menu` tags) | menu text | fetched by the `get_menu` tool, cached | the site's own; never committed |
| SearXNG (self-hosted container) | web search | live | — |

Nothing downloaded is committed. The example downloads into `build/`, records the snapshot date,
and keeps a small synthetic fixture for its tests. HTTP clients send a generic User-Agent, respect
Nominatim's 1 request/s and back off on 429.

## Stages

1. **Scaffold.** `examples/brooklyn-eats` as a standalone Gradle project like `lora-finetune`
   (mavenLocal Tlaloc), with subcommands `ingest`, `tools`, `tasks`, `teach`, `train`, `distill`,
   `eval`, `serve`, `chat`; `run.sh` drives the pipeline end to end.
2. **Ingest.** Snapshot the sources into one SQLite file (FTS5 over names, cuisines, violation
   text, review text and menus). Entity resolution: DOHMH CAMIS to OSM by normalized name and
   distance, reviews to CAMIS the same way. Report counts and match rates; spot-check a sample.
3. **Tools.** `search_restaurants`, `get_restaurant`, `get_inspections`, `get_complaints`,
   `get_reviews`, `get_menu`, `geocode`, `web_search`, each with a JSON schema. Modes: `snapshot`
   (offline, deterministic), `live`, `record` and `replay`. Tests on the fixture.
4. **Tasks and verifiers.** Questions generated from the snapshot with ground truth: lookups,
   filters ("A-graded Thai places in Park Slope"), comparisons, multi-hop (critical violations in
   the last year and what reviews say), menus, unanswerable ("not in Brooklyn", "closed"), and
   live-only questions. A verifier scores a final answer: required facts present, no invented
   restaurants, grades or dates, tool calls valid against their schemas. Splits by restaurant, so
   the test set holds restaurants never seen in training.
5. **Agent loop.** A Kotlin runner that talks to `tlaloc-endpoint` with tools, executes calls,
   appends tool responses, and stops at a final answer or a call budget. One test checks that the
   ids the trainer renders for a conversation equal the ids the served model is given for it (one
   template implementation for training and serving).
6. **Baselines.** Base TwIL-8B and Ripple-65B on the dev and test splits, offline tools.
7. **Teacher trajectories.** Ripple on the training split, several samples per question,
   tools in snapshot mode with recorded responses; keep the trajectories the verifier passes;
   re-render them in TwIL's format with assistant-token masks.
8. **Memory probe.** One LoRA step of TwIL-8B at each sequence bucket, base frozen, `MIXED_BF16`,
   chunked LM loss, device AdamW: peak memory and step time. Choose buckets and batch.
9. **SFT.** LoRA on all linear layers, loss on assistant tokens only (reasoning, tool calls,
   answers), length-bucketed captures, checkpoints, loss curve.
10. **Distillation.** Ripple's top-k log-probabilities on the content tokens of the kept
    trajectories, aligned to TwIL's sequences; `distillationLoss` on those positions plus
    cross-entropy on format tokens. Compared with SFT alone.
11. **Rejection-sampling round.** The tuned student samples on the training split; verified
    successes join the data; one more round of training.
12. **Evaluation.** Test split, offline tools: task success, fact precision, invented facts,
    schema-valid calls, calls per task, tokens/s. Rows: base TwIL, SFT, SFT+KD, +RFT, NVFP4 export
    of the best, Ripple.
13. **Serve and chat.** Merge, export bf16 and NVFP4 artifacts, Triton + `tlaloc-endpoint`,
    a chat client with live tools. Real transcripts in the README.
14. **Docs and review.** README with measured numbers, CAPABILITIES row, README examples table,
    CHANGELOG, final review.

## Reproducibility

Anyone with a Tlaloc checkout and a GB10 (or a CUDA GPU with as much memory) reproduces every
number from `run.sh`, with Tlaloc alone: no PyTorch, transformers or vLLM in the pipeline (Python
references only in optional checks).

- **Pinned inputs.** Model revisions by Hugging Face commit hash. Overpass queried with
  `[date:"<cutoff>"]` (OSM history, the same answer on any later day). DOHMH and 311 queried with
  SoQL cutoffs on their date columns. Google Local by file SHA-256. Menus from the Internet
  Archive's snapshot nearest the cutoff (the Wayback availability API), not the live site; live
  fetching is the chat demo's mode only.
- **Manifest.** `examples/brooklyn-eats/manifest.json` (committed) holds, per stage, the inputs'
  identities and the SHA-256 of each derived file (normalized table dumps, task splits, shards,
  adapters), the seeds and the Tlaloc commit. `run.sh verify` recomputes and compares; a public
  source that changed a past row is reported by table and key.
- **Seeds and determinism.** Every sampler, split, shuffle and LoRA init takes a seed from the
  manifest. Evaluation runs in the repo's deterministic mode where it exists; the run-to-run spread
  of the shipped model's test score is measured and stated.
- **Two lengths.** `run.sh --quick` (a few hundred tasks, short training, under about an hour)
  checks the whole pipeline; `run.sh` reproduces the published numbers. Time and peak memory of
  each stage are stated.
- **Cached outputs.** Each stage reads its predecessor's files, so any stage reruns alone; flags
  start from a given adapter to skip training.
- **Clean-room check.** A fresh clone, following only the README, reproduces the quick run and
  the manifest hashes of the ingest and task stages.

## Machine rules

One heavy job at a time: the teacher server, a training run and a Gradle build never overlap.
Ports 8000-8002 belong to another Triton; use others. Never create a PJRT client without
`create_options`. Long runs go in the background and are monitored, not polled with sleep.
