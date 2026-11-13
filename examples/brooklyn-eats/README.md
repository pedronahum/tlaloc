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
| `ingest` | Snapshot the sources into one SQLite file with FTS5; entity resolution | built |
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

## Ingest

```bash
build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats ingest              # fetch what is missing, build the database
build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats ingest --refresh    # fetch everything again
build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats ingest --build-only # rebuild from build/data/raw, no network
```

`--skip-google` leaves the Google Local tables empty. `BROOKLYN_DATA` overrides `build/data`.

Sources, fetched into `build/data/raw/`:

| Source | Query |
|---|---|
| OpenStreetMap (Overpass; overpass-api.de, then two mirrors) | `amenity` in restaurant, cafe, fast_food, bar, pub, ice_cream, food_court inside relation 369518 (Kings County), `out center tags` |
| DOHMH inspections (Socrata `43nn-pn8j`) | `boro='Brooklyn'`, all rows, pages of 50,000 |
| NYC 311 (Socrata `erm2-nwe9`) | `borough='BROOKLYN'`, complaint type Food Poisoning or Food Establishment, created in the last 3 years |
| UCSD Google Local 2021, New York | `meta-New_York.json.gz` (61 MB) and `review-New_York.json.gz` (2.6 GB) from mcauleylab.ucsd.edu; metadata kept when the address has a 112xx ZIP or ", Brooklyn, NY" and a category names a food or drink business; reviews stream-filtered to the matched businesses; reviewer names and ids are dropped |

`build/data/brooklyn.db` tables: `restaurants` (one row per CAMIS, plus named OSM places with no
CAMIS, keyed `osm:<type>/<id>`), `establishments`, `inspections`, `complaints`, `osm_places`,
`g_places`, `reviews`, `menus` (empty until the `get_menu` tool fills it), `matches` (every
accepted pair with its score, distance and rule) and `meta` (snapshot date, fetch times, counts,
match rates). FTS5: `restaurants_fts` (names, cuisines, address), `violations_fts`,
`reviews_fts`, `menus_fts`.

Entity resolution. Names are lower-cased, accents and apostrophes removed, "&" read as "and",
and words such as "the", "inc", "llc", "restaurant" dropped. Name similarity is the best of the
edit-distance ratio of the names without spaces, the word Jaccard index, and 0.9 when one name's
words all occur in the other and include a non-generic word. Distinctive-word similarity is the
same comparison after removing generic words ("pizza", "kitchen", "grill", cuisine adjectives),
and gives no credit when one name's distinctive words are a proper subset of the other's. A pair
is accepted by the first rule that holds:

| Rule | Condition |
|---|---|
| `near` | name similarity >= 0.80 and distance <= 120 m |
| `close` | name similarity >= 0.65, distance <= 35 m, distinctive-word similarity >= 0.75 |
| `address` | same house number and street, distinctive-word similarity >= 0.75, distance <= 300 m |

Candidates are assigned one to one, highest name similarity first. DOHMH establishments are
matched to OSM places; Google Local businesses are matched to the resulting restaurants (DOHMH
and OSM names, DOHMH coordinates where present). A 311 complaint gets a CAMIS when its house
number, street and ZIP match exactly one establishment, else when its BBL holds exactly one.

## Results

Ingest, snapshot 2026-11-13 (DGX Spark, one run):

| | Rows |
|---|---|
| OSM places | 4,880 (4,835 named) |
| DOHMH inspection rows / establishments | 74,406 / 8,023 |
| 311 complaints (2023-11-13 to 2026-11-10) | 13,908 |
| Google Local Brooklyn food businesses | 8,962 |
| Google Local reviews kept (of 33,459,761 read) | 1,232,350 (657,884 with text) |
| Restaurants | 9,227 (8,023 with a CAMIS, 1,204 OSM only) |

| Match | Rate |
|---|---|
| DOHMH establishments with an OSM place | 3,631 / 8,023 (45.3%) |
| Named OSM places with a CAMIS | 3,631 / 4,835 (75.1%) |
| Google Local businesses matched to a restaurant | 3,994 / 8,962 (44.6%) |
| DOHMH establishments with Google reviews | 3,440 / 8,023 (42.9%) |
| 311 complaints linked to a CAMIS | 8,433 / 13,908 (60.6%: 7,734 by address, 699 by BBL) |

Precision, checked by hand: a random sample of 20 DOHMH-OSM pairs and 20 Google-restaurant
pairs (seed 7, written to `build/data/match-sample.tsv`) had 40 correct of 40. The weakest rules
were checked separately: of 20 random `address`/`close` pairs (218 of 7,625 pairs), 18 were
correct, 1 was the same premises under a new name (Ferdinando's Focacceria and Bar Ferdinando)
and 1 was wrong (Matchless and Matches, across Manhattan Avenue); of 20 random `near` pairs with a
name similarity below 1, 20 were correct.

Ingest time: fetch 152 s (OSM, DOHMH, 311; the Google Local files were already in
`build/data/raw`), build 48 s; peak RSS 3.0 GB; database 349 MiB.
