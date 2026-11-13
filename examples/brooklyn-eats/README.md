# brooklyn-eats: a restaurant agent post-trained with Tlaloc

A Brooklyn restaurant agent: open data (OpenStreetMap, NYC DOHMH inspections, NYC 311,
UCSD Google Local reviews) behind a tool server, trajectories from a served teacher
(Ripple-65B-Preview), LoRA fine-tuning and distillation of an 8B student (TwIL-8B), evaluation
with checked answers, and the tuned student served with live tools. Every stage runs in Kotlin
on Tlaloc, on one DGX Spark.

TwIL-8B is under the webAI Non-Commercial 1.0 licence (research and education only); the
adapter and merged checkpoint this example produces inherit it. Downloaded data, reviews, menus
and model outputs derived from them stay under `build/` and are never committed.

## Build

```bash
./gradlew publishToMavenLocal -x test        # at the repo root, once and after changing Tlaloc
./gradlew -p examples/brooklyn-eats build    # compiles and runs the tests
./gradlew -p examples/brooklyn-eats run --args="help"
examples/brooklyn-eats/run.sh                # every stage in order
examples/brooklyn-eats/run.sh ingest tasks   # selected stages
```

A stage that is not implemented yet prints `not built yet` and exits with code 3.

## Stages

| Subcommand | Stage | Status |
|---|---|---|
| `ingest` | Snapshot the sources into one SQLite file with FTS5; entity resolution | not built |
| `tools` | Tool server: search, restaurant, inspections, complaints, reviews, menu, geocode, web search | not built |
| `tasks` | Questions with ground truth and verifiers; splits by restaurant | not built |
| `agent` | Agent loop against `tlaloc-endpoint` with tools | not built |
| `baseline` | Base TwIL-8B and Ripple-65B on dev and test | not built |
| `teach` | Ripple-65B trajectories on the training split, verified, re-rendered for TwIL | not built |
| `probe` | One LoRA step per sequence bucket: peak memory and step time | not built |
| `train` | LoRA SFT on assistant tokens | not built |
| `distill` | Distillation from Ripple's top-k log-probabilities | not built |
| `rft` | Rejection-sampling round | not built |
| `eval` | Test split, offline tools | not built |
| `serve` | Merge, export bf16 and NVFP4, Triton + `tlaloc-endpoint` | not built |
| `chat` | Chat client with live tools | not built |

`check` loads the SQLite driver, creates an FTS5 table and prints the SQLite version.

## Results

None yet.
