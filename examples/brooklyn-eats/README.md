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

Needs a JDK 25 (Gradle provisions the toolchain), network access for the first ingest, and Docker
for the `web_search` tool's SearXNG (optional: without Docker the live `web_search` check is
skipped and reported). Tlaloc `0.1.0-alpha04` resolves from Maven Central; mavenLocal is searched
first, so a locally published Tlaloc of the same version takes precedence (`run.sh verify`
reports the SHA-256 of the runtime classpath; see Reproducibility). No `publishToMavenLocal` is
needed.

```bash
./gradlew -p examples/brooklyn-eats build    # from the repo root: compiles and runs the tests
examples/brooklyn-eats/run.sh --quick        # every stage, reduced sizes
examples/brooklyn-eats/run.sh                # every stage, full sizes
examples/brooklyn-eats/run.sh ingest tools   # selected stages
examples/brooklyn-eats/run.sh verify         # compare inputs and derived data with manifest.json
```

`run.sh` prints each stage's wall time and the peak RSS of its process.
A stage that is not implemented yet prints `not built yet` and exits with code 3.

## Stages

| Subcommand | Stage | Status |
|---|---|---|
| `ingest` | Snapshot the sources into one SQLite file with FTS5; entity resolution | built |
| `tools` | Tool server: search, restaurant, inspections, complaints, reviews, menu, geocode, web search | built |
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
build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats ingest --quick      # menus for the first 150 places only
```

`--skip-google` leaves the Google Local tables empty. `BROOKLYN_DATA` overrides `build/data`.

Sources, fetched into `build/data/raw/`. Every query is pinned to the cutoff in `manifest.json`
(2026-11-13), exclusive, so it returns the same rows on any later day:

| Source | Query |
|---|---|
| OpenStreetMap (Overpass; overpass-api.de, then two mirrors) | `[date:"2026-11-13T00:00:00Z"]` (OSM history at that instant); `amenity` in restaurant, cafe, fast_food, bar, pub, ice_cream, food_court inside relation 369518 (Kings County), `out center tags`. An answer whose `timestamp_osm_base` is before the date is refused |
| DOHMH inspections (Socrata `43nn-pn8j`) | `boro='Brooklyn' AND inspection_date < '2026-11-13T00:00:00'`, pages of 50,000 ordered by `:id` |
| NYC 311 (Socrata `erm2-nwe9`) | `borough='BROOKLYN'`, complaint type Food Poisoning or Food Establishment, `created_date` from 2023-11-13 to before 2026-11-13 |
| UCSD Google Local 2021, New York | `meta-New_York.json.gz` (61 MB) and `review-New_York.json.gz` (2.6 GB) from mcauleylab.ucsd.edu, pinned by SHA-256 (downloaded again when the hash differs; ingest stops if the published file no longer has the pinned hash); metadata kept when the address has a 112xx ZIP or ", Brooklyn, NY" and a category names a food or drink business; reviews stream-filtered to the matched businesses; reviewer names and ids are dropped |
| NYC 2020 Neighborhood Tabulation Areas (Socrata `9nt8-h7nd`) | `boroname=Brooklyn`, ordered by code; a fixed release, pinned by SHA-256 |
| Internet Archive (Wayback Machine) | the archived menu pages of every named OSM place with a `website:menu`, `menu:url`, `menu` (URL) or `website` tag, captures up to 2026-11-12 23:59:59 UTC, pinned; see Menus |

`raw/manifest.tsv` records the query each raw file was fetched with; a raw file fetched with
another query (an older checkout, another cutoff) is fetched again.

`build/data/brooklyn.db` tables: `restaurants` (one row per CAMIS, plus named OSM places with no
CAMIS, keyed `osm:<type>/<id>`), `establishments`, `inspections`, `complaints`, `osm_places`,
`g_places`, `reviews`, `menus` (archived menus, below), `neighborhoods` (2020 NTAs with polygons), `matches` (every
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

### Menus

A place's menu is read from the Internet Archive, not the live site, so it is the same on any
later day. For each start URL (the OSM menu URL, else the website), the snapshot used is the
latest capture with HTTP 200 before the cutoff day in `manifest.json` (up to 2026-11-12
23:59:59 UTC): the
availability API's closest capture when it is not after the cutoff, else the CDX API's last
capture up to it. The original bytes are fetched as `web.archive.org/web/<timestamp>id_/<url>`.
When the start page has fewer than 5 priced lines, up to 2 of its links whose URL or text says
"menu" are read the same way. A menu is the page's priced lines, each with the short line before
it (the dish name); `items` counts the priced lines. PDF menus and menus drawn by JavaScript or
images are not extracted (status `pdf`, `no_prices`). Captures keep the site's own
Content-Encoding: gzip bodies are inflated, other encodings are reported `undecodable`.

Every page used is pinned in `pins/wayback-menus.tsv` (committed): URL, status, snapshot
timestamp, archive URL, content type and the SHA-256 of the archived bytes. `ingest` (and so
`run.sh`) fetches exactly the pinned captures, checks their hashes and never looks anything up
or rewrites the pins: a page with no pin is `not_pinned`, a pinned capture the archive no longer
serves byte for byte is retried 3 times (15 s and 30 s apart), its place is read once more 5
minutes after all others, and it is then reported (`moved`, `sha_mismatch`) and not used. A
later `ingest` tries again any pinned page not yet stored. Pages are stored by hash in `build/data/raw/wayback/pages/`, and the database build reads
them back without the network. Wayback requests are spaced 1 s apart per host, 4 places at a
time, with backoff on 429/5xx.

The pins are made by a maintainer step that is not part of `run.sh`:
`tlaloc-brooklyn-eats ingest --pin-menus` looks up every page with no pin, with a pin after the
cutoff, or whose pinned capture is no longer served as pinned (pinning the capture the archive
serves now), and rewrites the pins file with exactly the pages the full run reads.

## Tools

```bash
B=build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats
$B tools                                     # smoke: every tool in snapshot and live modes
$B tools smoke --modes snapshot              # offline only (also BROOKLYN_OFFLINE=1)
$B tools smoke --modes record,replay         # record live results, then replay them offline
$B tools schemas [--out tools.json]          # OpenAI tool definitions; exit 1 if a schema is malformed
$B tools call get_menu '{"id":"50060692"}' [--mode live]
$B tools serve --port 8893 [--mode snapshot] # GET /tools, POST /call {"name": ..., "arguments": {...}}
searxng/searxng.sh start|stop|status         # SearXNG for web_search on 127.0.0.1:8890 (SEARXNG_PORT)
```

| Tool | Arguments | Returns |
|---|---|---|
| `search_restaurants` | `query`, `cuisine`, `neighborhood`, `near` {`lat`,`lon`} or {`address`}, `radius_m`, `min_grade` (A/B/C), `open_at` (YYYY-MM-DDTHH:MM or `now`), `limit` | ids, names, cuisine, address, neighborhood, grade and date, Google rating, distance; the filters applied; the date the data is as of |
| `get_restaurant` | `id` | names, address, coordinates, phone, website, hours and their source, grade, latest inspection and score, Google rating, counts of inspections, critical violations in 12 months, complaints and reviews, archived-menu status |
| `get_inspections` | `id`, `since`, `limit` | inspections newest first: date, type, action, score, grade, violations (code, critical, text) |
| `get_complaints` | `id` or `near`, `radius_m`, `since`, `limit` | 311 complaints: key, dates, type, descriptor, address, status, resolution, how each was linked |
| `get_reviews` | `id`, `query`, `limit` | review count, average, rating histogram, first and last dates, newest reviews (or those matching `query`) |
| `get_menu` | `id` | the priced lines, the page they came from, and its capture time and archive URL (snapshot) or fetch time (live) |
| `geocode` | `address` | coordinates, precision, neighborhood |
| `web_search` | `query`, `limit` | title, URL, snippet, engines |

Ids are DOHMH CAMIS numbers, or `osm:<type>/<id>` for an OSM place with no permit. Arguments are
validated against the schemas (a JSON Schema subset checked by `Schema.kt`); an invalid call
returns `{"error": ..., "details": [...]}`. Every result names its sources and, in snapshot mode,
the snapshot date.

| Mode | Data | Network |
|---|---|---|
| `snapshot` | `brooklyn.db` only; "today" is the snapshot date; `geocode` uses a gazetteer of DOHMH and OSM addresses, ZIP centroids, NTA names and restaurant names; `web_search` searches the restaurant index | none: no HTTP client is constructed |
| `live` | DOHMH and 311 from the Socrata API, Nominatim (1 request/s), the restaurant's own site (robots.txt honoured), SearXNG; reviews have no live source and come from the snapshot | yes |
| `record` | live, and each result is stored in `build/data/recordings.db` under the tool and its canonical arguments | yes |
| `replay` | only what `record` stored; any other call returns an error | none |

The database is opened read-only in every mode. SearXNG runs from
`searxng/searxng@sha256:287af61f1a13e86d02974fcde91a3413a9945788de38b61ffbc491d81a19bcc2`
(tag 2026.11.9-57ff753ad); `run.sh` starts it before the `tools` stage and stops it after.

## Reproducibility

`manifest.json` (committed) holds the cutoff and its rule, the seeds, the Tlaloc version, and
per stage the identities of the inputs and the SHA-256 of what the stage derives:

| Entry | What is compared |
|---|---|
| `build.inputs.tlaloc` | the Tlaloc version in `build.gradle.kts` |
| `build.inputs.runtime_classpath` | SHA-256 over the name and SHA-256 of every jar the application runs with (Tlaloc, Kotlin, sqlite-jdbc) |
| `ingest.inputs.overpass`, `dohmh`, `complaints_311` | the query each raw file was fetched with (`raw/manifest.tsv`) |
| `ingest.inputs.nta2020`, `google_local_meta`, `google_local_reviews` | file SHA-256 |
| `ingest.inputs.wayback_menu_pins` | SHA-256 of `pins/wayback-menus.tsv`, page count, captures after the cutoff (0) |
| `ingest.derived.<table>` | every table of `brooklyn.db` the tools read: `osm_places`, `establishments`, `inspections`, `complaints`, `g_places`, `reviews`, `restaurants`, `matches`, `neighborhoods`, `menus` (per scope) |
| `tools.inputs.searxng_image` | the image digest `searxng/searxng.sh` runs |
| `tools.derived.tool_schemas` | the OpenAI tool definitions |
| `tools.derived.tools_smoke_snapshot` | `tools-smoke-snapshot.jsonl`, the 12 snapshot-mode tool results of the smoke (per scope) |

Tables are hashed over a normalized dump (all columns, rows ordered by key and then by every
column, SQLite `quote()` of each value), so build and fetch times do not enter; the snapshot
date the tools use is the cutoff, not the day of the download. `digests/ingest-rows.tsv`
(committed) holds a 12-hex digest per key for `osm_places`, `establishments`, `inspections` (per
CAMIS), `complaints`, `restaurants` and `menus`; when a table differs, `verify` names the keys
whose rows changed, appeared or disappeared. `produced_by` records the commit whose code wrote
each scope's values and whether that code was clean.

`run.sh verify` recomputes everything and exits 1 on a difference;
`tlaloc-brooklyn-eats verify --update` writes the values after a run. Values that depend on the
menus are kept per scope: `full`, and `quick` (`run.sh --quick` reads menus for the first 150
places by OSM id).

What can still differ for someone else, and how it shows:

- NYC Open Data and OpenStreetMap history are not frozen archives. DOHMH can add inspections
  dated before the cutoff after it (results are entered with a delay), change a past row
  (adjudicated grades) or drop establishments; 311 rows change status and resolution after the
  cutoff. `verify` reports these by table and key.
- Overpass `map_to_area` uses the server's current areas, not historical ones: a change to the
  Kings County boundary relation would change the OSM rows.
- The Internet Archive can stop serving a capture as pinned; the page is then `moved` or
  `sha_mismatch` and the menus digest reports the restaurants concerned.
- Live tool results (`live`, `record`) depend on the day by design and are not in the manifest.
## Results

Ingest at the cutoff 2026-11-13 (queries as above; code at eef86ac4, DGX Spark):

| | Rows |
|---|---|
| OSM places | 4,880 (4,835 named) |
| DOHMH inspection rows / establishments | 74,412 / 8,030 (latest inspection 2026-11-10) |
| 311 complaints (created 2023-11-13 to 2026-11-12) | 13,921 |
| Google Local Brooklyn food businesses | 8,962 |
| Google Local reviews kept (of 33,459,761 read) | 1,231,484 (657,376 with text) |
| Restaurants | 9,233 (8,030 with a CAMIS, 1,203 OSM only) |
| Brooklyn NTAs / restaurants placed in one | 69 / 9,193 |

| Match | Rate |
|---|---|
| DOHMH establishments with an OSM place | 3,632 / 8,030 (45.2%) |
| Named OSM places with a CAMIS | 3,632 / 4,835 (75.1%) |
| Google Local businesses matched to a restaurant | 3,993 / 8,962 (44.6%) |
| DOHMH establishments with Google reviews | 3,439 / 8,030 (42.8%) |
| 311 complaints linked to a CAMIS | 8,452 / 13,921 (60.7%: 7,745 by address, 707 by BBL) |

Precision, checked by hand on the earlier, unpinned download of 2026-11-13 (DOHMH paged
fetch, 8,023 establishments; not repeated on the pinned data): a random sample of 20 DOHMH-OSM pairs and 20 Google-restaurant
pairs (seed 7, written to `build/data/match-sample.tsv`) had 40 correct of 40. The weakest rules
were checked separately: of 20 random `address`/`close` pairs (218 of 7,625 pairs), 18 were
correct, 1 was the same premises under a new name (Ferdinando's Focacceria and Bar Ferdinando)
and 1 was wrong (Matchless and Matches, across Manhattan Avenue); of 20 random `near` pairs with a
name similarity below 1, 20 were correct.

Database 353 MiB (full scope). Times are under Pipeline time below.

Menus (cutoff 2026-11-13, pins in `pins/wayback-menus.tsv`):

| | Places |
|---|---|
| Named OSM places with a menu URL or website | 2,596 |
| Menu read (`ok`) | 608 (502 of them DOHMH restaurants; 501 from a menu link, 107 from the start page; median 27 priced lines) |
| Capture has no priced lines (`no_prices`; menus drawn by JavaScript, images, ordering sites) | 1,725 |
| No capture before the cutoff day (`no_snapshot`) | 172 |
| PDF menu (`pdf`, not extracted) | 43 |
| Capture in an encoding other than gzip (`undecodable`) | 29 |
| Archive excludes the site (`excluded`), capture empty (`empty`), HTTP 403, not HTML | 9, 3, 3, 4 |

Captures of the 608 menus: 55 before 2024, 118 in 2024-2025, 435 in 2026 (latest
2026-11-12T01:01:14Z). The pins file holds 3,970 pages (3,450 captures); the archived bytes take
625 MiB in `build/data/raw/wayback`. Moving the Wayback cutoff from "on or before 2026-11-13" to
"before 2026-11-13" re-pinned 6 pages (3 captures made on 2026-11-13 and 3 empty-body pins); the
menus table kept the same statuses.

Menu precision, checked by hand on every 50th `ok` menu by restaurant id (12 menus, offset 26,
taken after the rules were final): 11 are a menu (dishes or drinks with prices), 1 is a coffee
roaster's shop page. Dish names are paired with the price line after them, so on some sites the
pairing is off by a line. A place's site may link a sister location's menu (Istanbul Bay in Bay
Ridge reads its Bensonhurst menu page).

Pinning (`ingest --pin-menus`, maintainer step) took 4,923 s for the first discovery of 2,596
places (3,560 lookups, 3,179 captures; 1 s per host, 4 places at a time), 596 s more for the
menu links on gzip pages found after the gzip fix.

Tools (`run.sh ingest tools`, full scope): 12 smoke calls per mode on a sample chosen
by rule (Lindenwood Diner, CAMIS 40705411; a DOHMH restaurant with an OSM match, reviews,
complaints, a neighbourhood and an archived menu), every tool at least once.

| Mode | Result |
|---|---|
| snapshot | 12 / 12 ok, no HTTP client constructed; byte-identical across runs and in the clean-room clone (`tools.derived.tools_smoke_snapshot` in `manifest.json`) |
| live | 12 / 12 ok against DOHMH and 311 (Socrata), Nominatim, lindenwooddiner.com and SearXNG; e.g. 5 complaints on the tax lot live vs 2 linked by address in the snapshot, 44 within 300 m vs 6 |
| record, then replay | 12 / 12 replayed results identical to the recorded ones; no HTTP client in replay |

Live results change with the day (DOHMH and 311 rows, Nominatim, the site, SearXNG) and are not
in the manifest.

### Pipeline time

`run.sh` prints each stage's wall time and the peak RSS of its process (DGX Spark, 2026-11-14):

| Run | Wall time | Stages |
|---|---|---|
| `run.sh --quick ingest tools`, clean-room clone, empty `build/`, empty Gradle user home and mavenLocal | 620 s | Gradle distribution, Tlaloc and dependencies from Maven Central and compile 47 s; ingest 550 s (fetch 504 s: OSM, DOHMH, 311, NTA, 219 Wayback captures in 219 s, Google Local 2.5 GB in 145 s; build 46 s; peak RSS 2.7 GB); tools 22 s (peak RSS 0.4 GB) |
| `run.sh ingest tools`, same clone after the quick run (raw files present, the other 3,7xx captures not) | FULL_TIME | FULL_STAGES |
| `run.sh ingest tools`, every raw file and capture present | 82 s | ingest 60 s (peak RSS 3.5 GB), tools 17 s (peak RSS 0.4 GB) |
| `run.sh --quick ingest tools`, raw files present | 95 s | ingest 47 s (peak RSS 2.4 GB), tools 41 s |
| `run.sh verify` | 3 s | peak RSS 1.1 GB |

### Clean-room reproduction

A fresh clone of the repository at the manifest commit, with an empty Gradle user home and an
empty mavenLocal (so Tlaloc 0.1.0-alpha04 came from Maven Central), following only this README:
`run.sh --quick ingest tools` then `run.sh verify` matched all 22 manifest entries (quick scope),
including the digests of every ingest table and of the snapshot tool results. CLEAN_FULL
