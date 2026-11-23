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

Needs a JDK 17 or newer to run Gradle; the build compiles and runs on JDK 25, which Gradle
downloads through the Foojay toolchain resolver (`settings.gradle.kts`) when none is installed.
Also needs network access for the first ingest, and Docker
for the `web_search` tool's SearXNG (optional: without Docker the live `web_search` check is
skipped and reported). Tlaloc `0.1.0-alpha05` resolves from Maven Central; mavenLocal is searched
first, so a locally published Tlaloc of the same version takes precedence (`run.sh verify`
reports the SHA-256 of the runtime classpath; see Reproducibility). `probe`, `train` and `rft`
need the symbolic capture (`captureCausalLm(..., symbolic = true)`, BF16 leaves), which is new in
`0.1.0-alpha05`.

```bash
./gradlew -p examples/brooklyn-eats build    # from the repo root: compiles and runs the tests
examples/brooklyn-eats/run.sh --quick        # every stage, reduced sizes
examples/brooklyn-eats/run.sh                # every stage, full sizes
examples/brooklyn-eats/run.sh ingest tools   # selected stages
examples/brooklyn-eats/run.sh models         # download the pinned models that are missing, check every file
examples/brooklyn-eats/run.sh snapshot       # download the pinned data snapshot, unpack what is missing
examples/brooklyn-eats/run.sh verify         # compare inputs, derived data and models with manifest.json
```

`run.sh` prints each stage's wall time and the peak RSS of its process.
A stage that is not implemented yet prints `not built yet` and exits with code 3.

## Stages

| Subcommand | Stage | Status |
|---|---|---|
| `models` | The pinned Hugging Face models: download what is missing, check every file's SHA-256 | built |
| `snapshot` | The pinned data snapshot (raw rolling sources, sampled passes and shards): download, check, unpack (see Data snapshot) | built |
| `ingest` | Snapshot the sources into one SQLite file with FTS5; entity resolution | built |
| `tools` | Tool server: search, restaurant, inspections, complaints, reviews, menu, geocode, web search | built |
| `tasks` | Questions with ground truth and verifiers; splits by restaurant | built |
| `agent` | Agent loop against `tlaloc-endpoint` with tools; serves TwIL-8B for it | built |
| `baseline` | Base TwIL-8B and Ripple-65B on dev and test (`eval run` per model and split) | built |
| `teach` | Ripple-65B trajectories on the training split, verified, re-rendered for TwIL | built |
| `probe` | One LoRA step per sequence bucket: peak memory and step time | built |
| `train` | LoRA SFT on assistant tokens; merge into a bf16 checkpoint | built |
| `distill` | Distillation from Ripple's top-k log-probabilities | not built |
| `rft` | Rejection-sampling round: the SFT student on more training questions, new adapters on its passes | built (no gain on dev) |
| `eval` | The tuned student on dev and test, offline tools | built |
| `serve` | Export bf16 (built: `serving/twil-sft-bf16.env`) and NVFP4, Triton + `tlaloc-endpoint` | bf16 built |
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
| DOHMH inspections (Socrata `43nn-pn8j`) | `boro='Brooklyn' AND inspection_date < '2026-11-13T00:00:00'`, fetched in one request (`$limit=1000000`) whose row count must equal `count(*)` of the same filter (retried up to 3 times; paging by 50,000 had lost rows) |
| NYC 311 (Socrata `erm2-nwe9`) | `borough='BROOKLYN'`, complaint type Food Poisoning or Food Establishment, `created_date` from 2023-11-13 to before 2026-11-13; one request checked against `count(*)`, as for DOHMH |
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
| `snapshot` | `brooklyn.db` only; "today" is the snapshot date, which is the cutoff (2026-11-13), whatever day the data were downloaded; `geocode` uses a gazetteer of DOHMH and OSM addresses, ZIP centroids, NTA names and restaurant names; `web_search` searches the restaurant index | none: no HTTP client is constructed |
| `live` | DOHMH and 311 from the Socrata API, Nominatim (1 request/s), the restaurant's own site (robots.txt honoured), SearXNG; reviews have no live source and come from the snapshot | yes |
| `record` | live, and each result is stored in `build/data/recordings.db` under the tool and its canonical arguments | yes |
| `replay` | only what `record` stored; any other call returns an error | none |

The database is opened read-only in every mode. SearXNG runs from
`searxng/searxng@sha256:287af61f1a13e86d02974fcde91a3413a9945788de38b61ffbc491d81a19bcc2`
(tag 2026.11.9-57ff753ad); `run.sh` starts it before the `tools` stage and stops it after.

## Tasks

```bash
B=build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats
$B tasks                          # generate build/data/tasks/{train,dev,test}.jsonl, then check them (run.sh tasks)
$B tasks check [--show-mutants]   # regenerate in memory and compare with the files; overlap; references; corrupted references
$B tasks show --split test --n 30 --seed 1   # a seeded sample: question, solution calls and their results, facts, reference
$B tasks score answers.jsonl      # score answers: one {"id", "answer", "calls": [{"name", "arguments"}]} per line
```

Questions are generated from `brooklyn.db` through the snapshot-mode tools: the ground truth of
each task is read from the results of tool calls recorded with it as its `solution`, so every
fact is in what an agent can see. Dates in questions are the manifest cutoff (2026-11-13) or
computed from it ("since November 13, 2025", "at 9:00 PM on Sunday, November 15, 2026"), never
today's date. Wording comes from 2 to 5 seeded templates per kind; no language model is used.
A restaurant is named by its name when the name is unique and a search for it finds it, else by
name and street address; a restaurant that a simple search does not find, or finds together with
another restaurant of the same name, is not used as a subject.

| Category | Kind | Ground truth | Share |
|---|---|---|---|
| lookup | `grade` | current letter grade (and its date) | 12% |
| lookup | `last_inspection` | date of the latest inspection (and its score) | 8% |
| lookup | `violation` | the 1 to 3 violations of the latest inspection | 8% |
| filter | `cuisine_area_grade` | the 1 to 5 places of a cuisine in a neighborhood, with no grade filter, A, or B or better | 12% |
| filter | `open_at` | the 1 to 5 places of a cuisine in a neighborhood open at a stated time in the week after the cutoff (two seeded times per cuisine and neighborhood) | 6% |
| comparison | `compare` | two places in one neighborhood: latest score, Google rating, or critical violations in 12 months | 10% |
| multihop | `critical_and_reviews` | critical violations in the 12 months before the cutoff (`get_restaurant` and `get_inspections` must agree) and the Google review average | 12% |
| complaints | `complaints` | 311 complaints since 6, 12 or 24 months before the cutoff, or the latest complaint's date and type | 10% |
| menu | `price` | the price written on the same line as a dish name on the archived menu ("Clams $20"; restaurants with an archived menu only) | 8% |
| unanswerable | `outside_brooklyn` | a refusal: the place is outside Brooklyn | 4% |
| unanswerable | `nonexistent` | a refusal: an invented name that no search finds | 6% |
| unanswerable | `closed` | the latest DOHMH inspection closed the place, with its date | 4% |
| live | `live_only` | none: open right now, inspections after the cutoff, news; `offline: false`, not scored offline | 40 / 8 / 8 |

Targets are 4,000 / 400 / 400 offline tasks (train / dev / test) for a full ingest and 300 / 40 /
40 for a quick one, split by these shares. The split is by restaurant: each restaurant is in train
(80%), dev (10%) or test (10%) by the SHA-256 of `seeds.task_split` and its id, and a task is
kept only when every restaurant of its ground truth (subjects and list answers) is in one split;
comparison pairs are formed within one neighborhood and one split. Tasks with no restaurant
(outside Brooklyn, nonexistent) are split by the hash of their key. Candidates are visited in an
order shuffled by `seeds.tasks`. `tasks` and `tasks check` check that no restaurant is in two
splits. The check compares restaurant ids, not names: branches of a chain are different
restaurants, so a chain name can appear in two splits (full run: 25 restaurant names occur in
both train and test, among them Dunkin', McDonald's, IHOP, Domino's and Applebee's, each time as
different branches with different ids).

Menu questions use only lines that hold a dish name and its price together ("Clams $20",
"Lox: $4.95", "Chili Burger - $18.00"). A name and a price on separate lines are not paired: on
archived menus the price line comes before the name, after it, or after a section header, and
the text does not say which (an earlier version paired them and asked for the price of headers
such as "Banh Mi" and "Coffee by Parlor", and gave Mai's "Rice Vermicelli with Curry Beef Stew"
the price of the next dish). Also skipped: section headers and phrases ending in a plural
category ("Snacks", "Curated Cocktails"), sizes, counts and options ("Lrg", "Twelve", "Half
Topping =", "Shrimp or Beef"), add-on and substitution phrasing ("Steak Add", "Sub cheese dip"), items under an add-on section
(a header naming add-ons, extras, toppings, fillings, sides or substitutions, such as "Salad
Add-ons", until the next header), a name that also occurs inside another line ("Veggie Sausage
$2.00" under Add-ons and "With Veggie Sausage $10.95"),
fees ("Delivery Min", "Valet extra"), dietary labels alone ("Gluten-Free"), phrases cut before
their object ("Beer and a Shot for"), a line under an unpriced ingredient list or ending
mid-phrase (the end of a wrapped description), a sentence-case line under an unpriced capitalized
dish name (its description), names that occur twice in the menu, and prices under $0.25. At most
5 dishes per menu are candidates.

The search tool also matches a cuisine against Google Local categories, which its results do not
show; a filter question is asked only when the cuisine each result shows (DOHMH, else OSM) or its
OSM cuisine names the cuisine asked for.

Each line of a split file is one task: `id`, `split`, `category`, `kind`, `question`,
`restaurants`, `facts` (kind, value, alternatives, whether required, `of`: the restaurant a
comparison value belongs to), `answer_set` (list
questions), `solution` (tool calls), `reference` (an answer built from the solution's results),
`offline`, `template`, `as_of`.

### Verifier

`Verifier.verify(task, answer, calls)` scores a final answer against `brooklyn.db`. It passes when
all four checks hold:

| Check | Rule |
|---|---|
| Facts | every required fact is stated: restaurant name (or one of its other names that is not only generic words), grade letter, date (ISO, "March 1, 2025", "1 March 2025", "3/1/2025"), count (digits or words; 0 as "no" or "none"), number within a tolerance (ratings ±0.05, written with a decimal unless the value is whole), price, violation (its code, its rarest word, or two of its words), keyword, topic (what a 311 complaint was about: a Food Establishment complaint's descriptor, any `/` or "or" part of it, each word matched by its first 4 letters, 6 for words of 10 letters or more: "Letter Grading" is stated by "letter grade"), refusal (not found / outside Brooklyn) |
| Not negated | a grade, date, price or keyword counts only when its clause (back to the last `, ; : . ( ` or "but", "and", "while") has no "not", "n't", "never", "no longer", "cannot" or "without": "does not have grade A", "was not inspected on March 25, 2026", "it was not closed" do not state the fact |
| Not hedged | a count, rating or price offered among other values does not count: "0, 1, 2 or 3", "2 or 3", "about 2", "at least 2", "2+", "$9.95 or $10.95"; a grade, date or price whose clause offers it as a guess ("probably", "possibly", "perhaps", "maybe", "might", "could", "likely", "may be") does not count either: "probably has grade A" |
| Not contradicted | every number written before the fact's unit (up to three words between: "7 critical violations", "3 311 food complaints", "13 points", "4.26 stars", "4.2/5") must be a ground-truth value of that unit; digits and number words inside the task's restaurant names ("China No. 1", "Three Diamonds Door", "1 of 1 Sandwich") are not read as numbers; in a menu question the price given for the dish asked about (the first price after its name in the sentence, else the last before it) must be its price |
| Comparison | the restaurant named nearest before a verdict ("did better", "lower score", "rated higher", "had more", "beat"), or after it when none is before, in the same sentence, must be the right one; a verdict for the other side ("did worse", "higher score", "rated lower", "fewer") must name the other restaurant; statements of the scale ("lower scores are better") are not verdicts; each value must be nearer its own restaurant's name than the other's (summed over both); a score comparison must not say higher scores are better |
| Invented | a phrase in a restaurant position (bold, the lead of a list item, before "(id ...)", or a title-case phrase with a place word such as Kitchen, Kitchen's, Pizzeria, Bistro, Noodles, Ramen) must be a snapshot restaurant name, part of one, or one plus generic words, unless the question names it or it is on the archived menu of the task's restaurant (a dish); every CAMIS or `osm:` id must exist; every A, B or C grade not negated must be the current grade of the restaurants of its sentence, or a past grade of theirs when the sentence has a date or a past-tense marker (hedged grades count as claims: "it might be C now" fails when the grade is A); a grade is read after "grade" ("grade A", "an A grade", "the grade is B") and after a verb ("holds an A", "it is A.", "might be C now"); every date must be a date the snapshot holds for the restaurants concerned (inspections, grades, complaints, reviews, menu capture) or a date in the question or facts; the cutoff, the day before it and the day a year before it only as the bound of a period (after "as of", "since", "until", "through", "up to", "before", "after", "to", "from", "between", "by", "data", "records"): "last inspected on November 13, 2026" fails; every `$` price must be written on the archived menu of the task's restaurant; every DOHMH violation code ("04L") must be one the task's restaurants were cited for, and in a question about the latest inspection one of that inspection, unless its sentence has another date or speaks of an earlier inspection ("previous inspection", "in 2023", "history"); in such a question a pest named outside a negated clause (mice, rats, rodents, roaches, flies, vermin, insects) must appear in that inspection's violation texts |
| Absent | claims a task forbids, outside a negated clause: in `closed` tasks, that the place reopened or is open ("has since reopened", "is open", "you can eat there"); in complaint tasks, "food poisoning" for a Food Establishment complaint, and an establishment topic (rodents, insects, garbage, mice, roaches, letter grading, no permit, bare hands) for a Food Poisoning one; in score comparisons, "higher scores are better" |
| Extra | in list questions, a snapshot restaurant named in the answer must be in the answer set |
| Calls | every call names one of the 8 tools and its arguments are a JSON object valid against that tool's schema |

Names are matched after `normalizeName` (case, accents, apostrophes, "&", stop words); a name
several restaurants share (a chain) stands for the task's own restaurants when they are among
them, and a street address in the same sentence narrows it further. Street names ("Myrtle
Avenue", "216 Bedford", "148 Noble Street") and neighborhood names ("Clinton Hill") are not read
as restaurants. In a comparison of two restaurants where one name holds the other ("Dunkin'" and
"Dunkin' Donuts/Baskin Robbins"), a mention inside the longer name counts for the longer one.

Limits:

- An invented restaurant name in plain prose, without a place word and outside the restaurant
  positions above, is not caught ("Nearby, Moonlight Garden is better rated" passes; "Moonlight
  Garden Noodles" is caught).
- The date check is set membership: a real date of the restaurant attached to the wrong event
  passes. Example: "since November 14, 2025" instead of November 13 passes for Atlantis Diner,
  which was inspected on November 14, 2025.
- The contradiction check is strict: any number before a fact's unit must be a ground-truth
  value, so a true number from another window ("12 critical violations since 2020") fails.
  Hedging written after the value ("1 critical violation, or maybe 2") is not caught.
- Negation is read only in the fact's own clause: a negation in an earlier sentence ("I can't
  confirm this. F&F Pizzeria has grade A.") is not seen, and a negation of something else
  earlier in the same clause hides the fact ("Not far from the F train it has grade A" fails).
- The comparison check reads only the listed verdict phrases; an answer that names a winner
  in other words ("go with China Moon") is checked only for the value pairing.
- Free-text claims about what reviews say are not checked.
- Hedging words are a fixed list read in the fact's clause. A guess phrased otherwise ("my
  best reading is A") counts as stated, and a hedging word about something else in the same
  clause ("You could go there: it has grade A") makes the fact not stated.
- The cutoff as a period bound is recognized by the word right before it; "the data ends on
  November 13, 2026" fails although it is true.
- Invented violations are caught by their code or by the pest words listed above. An invented
  violation described in other words ("the walk-in was too warm") is not caught.
- A complaint topic is matched by word stems; an answer that names the right topic with other
  words ("vermin" for Rodents/Insects/Garbage) fails.
- Menu ground truth is the price written on the dish's line. On some menus that line names a
  group or a label rather than one dish; see Results for the counts read.

Scores of answers to test-split tasks (`tasks score`, full data, code at 311eabea):

- 8 wrong answers from the first review, each probing one check (a comparison with the wrong
  winner, twice; "0, 1, 2 or 3 critical violations"; "7 critical violations (2 of them since
  the spring)"; "does not have grade A"; "was not inspected on March 25, 2026"; "is open; it was
  not closed"; an invented "Moonlight Garden Noodles"): all 8 fail. The ninth, a menu task, is no
  longer generated.
- 23 wrong answers: the second review's 14 that are still generated, one variant of them, and 8
  written for this fix. 21 fail. The 2 that pass match limits above: an invented "Kura Sakaba" in
  plain prose after a correct refusal, and "closed by DOHMH on July 16, 2026", a real inspection
  date of Shaxian Snacks attached to an event that did not happen on it. The failures include
  "probably has grade A, but it might be C", "has grade A. It might be C now, though", "$12.95 or
  $11.95" for a dish, "it has since reopened and you can eat there", an invented "04L (critical):
  evidence of mice" with and without the code, "It was last inspected on November 13, 2026" after
  a refusal, and "about food poisoning" for a Rodents/Insects/Garbage complaint. One more answer
  from that review (a price hedged as "$9.95 or $10.95" for the "Banh Mi" header) belongs to a
  task that is no longer generated.
- 15 good answers pass, among them "Asian Bay (1741 Sheepshead Bay Road) currently holds an A from
  the health department, posted May 9, 2026", which failed before.

The examples in the limits above were scored the same way and behave as stated.

`tasks check` scores every offline task. The reference answer must pass, and the solution's tool
results must hold every required fact. Corrupted copies of the reference must fail, and the check
exits 1 when one passes (`--show-mutants` lists them): another grade letter; the first date moved
to the first later day that is not a date the snapshot holds for the task's restaurants or an
always-accepted date (cutoff, the day before, a year before, the question's dates); a count plus
one; a price plus one dollar; a rating plus 0.3; the name replaced by "Golden Lantern Kitchen";
an invented restaurant added to a list; a refusal replaced by an invented grade and date; the fact
negated ("does not have a DOHMH letter grade of", "was not last inspected on", "DOHMH has not
closed", "does not list", "was not created on"); the count hedged ("2, 3 or 4"); a second count
added ("(7 critical violations in all)"); the dish at another price with the true one as an old
price; the price hedged ("$13.00 or $14.00"); in comparisons the other restaurant picked, the two
values swapped, and "higher scores are better"; in grade tasks "probably has a DOHMH letter grade
of" and "It might be C now." appended; in closed tasks "It has since reopened, and you can eat
there." appended; in violation tasks an invented "04L (critical): evidence of mice" (or 04M
roaches, 04N flies, whichever the inspection does not have); in complaint tasks the other type
("Food Poisoning" for a Food Establishment complaint and the reverse); in refusals "It was last
inspected on November 13, 2026." appended. `src/test/kotlin/TasksTest.kt` holds 80 hand-written
answers on the fixture (29 good, 51 bad, including a date one day off, a past grade stated as current, an earlier grade that never
existed, an invented CAMIS, a misspelled name, the listing's rating given as the reviews' average,
an invented look-alike of a nonexistent place, a comparison with the wrong winner, swapped values
or the scale reversed, counts among guesses, a second count, an integer near a decimal average,
negated grade, date and closure, the true price given as an old one, an invented possessive name,
a guessed grade after the true one, a grade only as a guess, two prices offered, a reopening
claimed, an invented second violation code and an invented pest, the cutoff given as an
inspection date after a refusal, a wrong complaint topic, and calls with a wrong type, an unknown tool, an unknown argument or non-JSON arguments).

## Models

```bash
B=build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats
$B models                      # download what is missing at the pinned commits, check every file (run.sh models)
$B models check                # check only, no network
$B models --all                # also the optional bf16 teacher (120.7 GiB)
$B models ripple_65b_dflash2   # one model
$B models --rehash             # read every file again instead of trusting remembered hashes
$B models --repair             # download again a cached file whose SHA-256 is wrong
$B models path twil_8b         # the snapshot directory the later stages load
```

| Key | Repository | Commit | Files | Size | Used for |
|---|---|---|---|---|---|
| `twil_8b` | webAI-Official/TwIL-8B | `9563ba7dfb70eeca4a26654b195011cbf818e7d9` | 10 | 16.4 GiB | student ("balanced" checkpoint at the repository root) |
| `ripple_65b_nvfp4` | ricursiveAI/Ripple-65B-Preview-NVFP4 | `99eea4477a001a41b86914e41feb4b2ed86105e5` | 15 | 53.3 GiB | served teacher and its tokenizer |
| `ripple_65b_dflash2` | ricursiveAI/Ripple-65B-Preview-DFlash2 | `35609b3be3dd08a822b3ba1257c1ea5dd6f87632` | 4 | 7.5 GiB | teacher's speculative draft |
| `ripple_65b_bf16` (optional) | ricursiveAI/Ripple-65B-Preview | `e27e171792f303d74c048184e8259ba663cfa0d5` | 24 | 120.7 GiB | not used by the pipeline; pinned for reference |

`manifest.json` `stages.models.inputs` holds, per model, the commit and, per file, the size and
SHA-256: the LFS oid from the Hub API for weight files, and for the small files the SHA-256 of
the file at the commit (its git blob id is checked against the API when pinning). Only the files
the pipeline reads are pinned (`*.json`, `*.jinja`, `*.safetensors`, licences and `README.md` at
the repository root); GGUF files, TwIL's `verbose/` checkpoint, `benchmarks.png` and Ripple's
Python reference code are not downloaded.

Files go into the standard Hugging Face cache: `HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub` (`--cache DIR` overrides),
as `models--<org>--<name>/blobs/<LFS oid or git blob id>` with
`snapshots/<commit>/<path>` linked to it. A file whose blob is already in the cache (from another
snapshot) is linked, not downloaded. Downloads are plain HTTPS from
`https://huggingface.co/<repo>/resolve/<commit>/<path>` (`HF_ENDPOINT` overrides the host), into
`<blob>.incomplete`, resumed with a Range request after an interruption (in the same run with
backoff, or by the next run), and moved into place only when size and SHA-256 match.
`HF_TOKEN` is sent only when set, and only to the Hub host, never to the storage host it
redirects to; the token file of `huggingface-cli login` is not read. None of the four
repositories is gated.

Checked SHA-256 values are remembered in `build/data/model-sha256.tsv` by real path, size and
modification time, so a check of unchanged files does not read 200 GiB again. A change that
keeps both the size and the modification time is found only with `--rehash`.

Pinning is a maintainer step, not part of `run.sh`:
`models pin <key> <repo> <revision> --role TEXT [--include GLOBS] [--optional]` resolves the
revision to a commit through `/api/models/<repo>/revision/<revision>?blobs=true` and writes the
entry. TwIL-8B's history on the Hub was replaced by a single commit on 2026-11-08: the revision
first downloaded here (`58d49f0d...`) no longer resolves, and every file of the pinned
`9563ba7d` has the same LFS oid or git blob id as that download.

## Agent

```bash
examples/brooklyn-eats/run.sh agent             # serve TwIL-8B, 10 dev questions, parity check, stop the server
examples/brooklyn-eats/run.sh server start      # only serve (export when needed); prints http://127.0.0.1:8120/v1
examples/brooklyn-eats/run.sh server stop
B=build/install/tlaloc-brooklyn-eats/bin/tlaloc-brooklyn-eats
$B agent run [--split dev] [--n 10] [--concurrency 1] [--temperature 0] [--max-calls 8] [--max-tokens 12000] [--out FILE]
$B agent parity build/data/agent/dev-10.jsonl   # trainer ids of the recorded episodes against what the endpoint was given
$B agent show build/data/agent/dev-10.jsonl --n 3
```

Serving needs Docker with GPU access and this Tlaloc checkout (the exporter and the Triton backend
are built from it; nothing else is installed). `serving/server.sh` does, for the config
`serving/twil-bf16.env`:

1. Export: `:maestro:exportHfServingArtifact` from the pinned checkpoint (`models path twil_8b`)
   with the config's settings (bf16 weights as stored, f32 KV, 4 sequences per batch, contexts
   8,192 and 16,384, 4,352 KV blocks of 16 tokens, prefill 512 tokens per call), then
   `:maestro:exportTritonModel` (decoupled). Into `build/serving/twil-bf16/`; skipped when
   `export.stamp` (settings, checkpoint, last commit of the exporting modules, clean or not) is
   current.
2. Backend: `triton/build_backend.sh` in the pinned Triton image when the backend's sources
   changed since the last build (`build/serving/backend.stamp`); the PJRT plugin from
   `triton/fetch_pjrt_plugin.sh` (pinned wheel SHA-256).
3. Triton (`nvcr.io/nvidia/tritonserver` pinned by digest) on 127.0.0.1:8110-8112, PJRT memory
   fraction 0.3, compiled executables kept in `build/serving/pjrt-cache`; then
   `tlaloc-brooklyn-eats endpoint` (tlaloc-endpoint from this application's classpath, so the
   server renders requests with the same template code as the trainer) on 127.0.0.1:8120 with
   `--sampling`. `stop` ends the endpoint (which ends its sequences) and the container, and prints
   the peak of the machine's used memory while serving. Ports: `BROOKLYN_TRITON_PORT`,
   `BROOKLYN_ENDPOINT_PORT`.
4. Memory guard: while the container runs, machine memory is sampled every 2 s into
   `mem.log`. When `MemAvailable` falls below `BROOKLYN_MEM_GUARD_GIB` GiB (default 8; 0 turns
   it off), the guard kills the endpoint and the container and writes the reason to
   `mem-guard.log`; `start`, `stop` and `status` print it, and a running evaluation fails with
   endpoint errors instead of the machine running out of unified memory. Checked on TwIL-8B:
   a 200 GiB threshold stopped `start` 6 s in; with 45 GiB, a 20 GiB allocation beside the running
   server (61 GiB available before it) killed the endpoint and Triton at 44.1 GiB available.
   While Ripple-65B is served (machine memory in use peaked at 87.1 GiB of 121 over the 7.5 h
   baseline run), run no Gradle build, no other JVM over 2 GiB and no Python data job; the machine
   rebooted once (2026-11-14) when a Gradle test run and an 8 GiB rescore ran beside the Ripple
   server. The `agent` and `baseline` stages of `run.sh` stop Gradle daemons before they start a
   server.

The agent sends the system prompt (`systemPrompt` in `Agent.kt`: Brooklyn, today is the cutoff,
use the tools, do not guess), the question and the eight tool definitions; runs each tool call
through the tool server (snapshot mode) after checking its arguments against the schema; appends
the result as a `tool` message; and stops when a turn has no tool call (`answer`), when a call
would pass the call budget (`call_budget`, 8), when the episode's completion tokens reach 12,000
(`token_budget`), or when the 16,384-token context leaves fewer than 64 tokens (`context`). A
turn's `max_tokens` is the smallest of 4,096, the tokens left in the budget and the context left
after the prompt (counted with the trainer's rendering). Sampling: temperature 0 (greedy); each
episode's seed is 24 bits of SHA-256 of `seeds.agent` and the task id, plus the turn index per
turn, sent with every request. The 10 questions are taken from the offline dev tasks, one
category at a time, each category ordered by SHA-256 of `seeds.agent_sample` and the task id.

`build/data/agent/dev-10.jsonl` (per scope, under `BROOKLYN_DATA`) holds one episode per line:
the messages as sent (with `reasoning_content` and tool calls; call ids `call_<turn>_<i>`), the
settings and seeds, per turn the seed, `max_tokens`, finish reason, the endpoint's
`prompt_tokens`, the trainer's `rendered_prompt_tokens`, completion tokens and time, per call the
arguments, schema errors, the tool's error and the result's length, the stop reason, the answer,
totals, and the offline verifier's verdict. Times and the endpoint's cached-token counts are under
`timing` keys, which the manifest's hash leaves out.

### Trainer and endpoint parity

`renderForTraining` (`Render.kt`) renders a recorded conversation once with
`HfTokenizer.applyChatTemplate` (`ChatTemplate.TWIL` or `RIPPLE`), tool definitions and call
arguments through tlaloc-endpoint's `pythonJson`, and encodes it. For each assistant message the
chat before it, rendered with the generation prompt, must be a prefix of the ids, and the mask
runs from there to the turn's `<|im_end|>` inclusive: for TwIL the reasoning (the generation
prompt holds `<think>\n`), `</think>`, the answer text and the tool calls. A conversation that
breaks the prefix rule is reported, not trained on: TwIL renders an assistant turn without
reasoning as `<think></think>`, which its generation prompt does not start.

`AgentTest` runs the real `OpenAiEndpoint` in front of a scripted model that replies with the
trainer's tokens of each assistant turn and records what the model would be given before each
turn (the sequence's tokens, as the backend's KV holds them). On a synthetic three-turn episode
with four tool calls (a nested object and numbers among the arguments, two calls in one turn,
text before calls) driven by the agent loop: the model's input before every turn equals the
trainer's ids up to that turn, for the first request (rendered fresh) and for the later ones (the
endpoint extends the live sequence with the reply as generated); a fresh endpoint per request
gives the same; the endpoint's `prompt_tokens` and the recorded messages match; the masked text
is exactly the assistant turns (written out by hand in the test) and the rest is the system
prompt, the question and the tool results. The test runs for TwIL and for Ripple with their
tokenizers from the Hugging Face cache (skipped when a tokenizer is not there).
`agent parity` applies the same prefix rule to recorded episodes and compares each turn's
`prompt_tokens` with the trainer's prompt length.

## Baseline

```bash
examples/brooklyn-eats/run.sh baseline          # TwIL-8B bf16, then Ripple-65B NVFP4 + DFlash2; runs without current results only
BROOKLYN_FRESH=1 examples/brooklyn-eats/run.sh baseline   # every run again
$B eval run --config twil-bf16 --split dev [--run 2] [--n N] [--concurrency 4] [--fresh]   # against a running server
$B eval rescore build/data/baseline/*/*.jsonl   # verdicts and metrics again with the current verifier (no server)
$B eval spread build/data/baseline/twil-bf16/dev-r1.jsonl build/data/baseline/twil-bf16/dev-r2.jsonl
$B baseline report                              # the table below; writes build/data/baseline/summary.json
$B baseline pending [--all]                     # runs without current results
```

`run.sh baseline` serves one model at a time (`serving/twil-bf16.env`, then
`serving/ripple-nvfp4.env`), runs that model's evaluations whose results are missing or were
produced with other settings, serving config or tasks, and stops the server. Runs: TwIL dev twice
(the run-to-run spread), TwIL test, Ripple dev, Ripple test.

`eval run` takes every offline task of the split (full: 382 dev, 375 test; quick: 4 per split,
categories interleaved by `seeds.eval`), runs the agent with snapshot tools (the agent settings
above: greedy, 8 calls, 12,000 completion tokens, 16,384-token context), 4 conversations in
flight (the configs' batch size), and writes `build/data/baseline/<config>/<split>-r<run>.jsonl`
(episodes with verdicts) and `.json` (key, code commit, wall time, metrics). An interrupted run
resumes from `<split>-r<run>.partial.jsonl` when its key matches; episodes the endpoint failed run
again. Deterministic mode is off for both configs (see Agent), so concurrent sequences share
batches in varying order and greedy outputs can differ between runs.

Metrics: task success (the verifier passes the answer), fact recall (required facts found / required),
fact precision (found / (found + invented items + extra items: restaurants a filter answer lists that do not match the question)), answers with
invented items, schema-valid calls, calls per task, completion tokens per task, completion tokens
per second over the run (all four streams), median and 90th-percentile episode time.

Ripple-65B config (`serving/ripple-nvfp4.env`): NVFP4 MLP weights, bf16 attention, FP8 KV, the
DFlash2 draft (bf16) with 15 draft tokens and verify entries for 3 and 7, 4 sequences per batch,
contexts 8,192 and 16,384, 4,352 KV blocks of 16 tokens, prefill chunks 128 and 512 with NVFP4
activations, PJRT memory fraction 0.1 (weights outside the pool), endpoint `--max-sequences 4`.

## Teach and SFT

```bash
examples/brooklyn-eats/run.sh teach    # serves Ripple-65B, samples the training questions, stops it; renders and checks the shards
examples/brooklyn-eats/run.sh probe    # one LoRA step per sequence bucket (no server)
examples/brooklyn-eats/run.sh train    # train run, then train merge (no server); both keep current outputs
examples/brooklyn-eats/run.sh eval     # serves the merged student (serving/twil-sft-bf16.env), dev and test, stops it
$B teach report                        # yield per category, lengths, shard identities
$B train report                        # build/data/train/summary.json
$B train run [--epochs N] [--lr X] [--fresh]
$B train merge [--adapter DIR] [--out DIR] [--fresh]
```

`teach run` runs the agent loop with Ripple-65B on 300 questions of the training split (full
scope; categories interleaved by `seeds.teach_sample`, in an order where a larger count keeps the
recorded episodes), snapshot tools, temperature 0.7, top-p 0.95, a seed per episode from
`seeds.teach`, four conversations in flight. An interrupted pass resumes from its partial file.
The endpoint refusing a prompt plus `max_tokens` past the context (a speculative step writes 31
positions past its tokens) ends the episode with the `context` stop, as the client's own check does.

`teach render` keeps an episode the verifier passed that ended with an answer, whose rendering in
TwIL-8B's chat template has every assistant turn's prompt as a prefix of the chat and fits the
served context; it writes `teach/shards/` (`BKEATS01`: token ids and a mask of the assistant
tokens, in an order seeded by `seeds.teach_shuffle`). `teach check` renders them again and
compares bytes, and parses every sequence back to its turns and calls. The adapters start from
`seeds.train`.

`probe` measures one training step per sequence length bucket (4,096, 8,192, 16,384 tokens) and
writes `train/probe.json`; `train run` uses the buckets that ran. TwIL-8B gets rank-16 LoRA
adapters (alpha 32) on every linear layer; the frozen base is held in bf16 (17.6 GB), the
adapters and AdamW state in f32 on the device. The step is captured symbolically per bucket
(`captureCausalLm(..., symbolic = true)`, MIXED_BF16, LM-head loss in chunks of 1,024, every
block recomputed in the backward pass) and compiled once. One sequence a step, padded to the
smallest bucket that holds it; sequences longer than the largest are left out. The loss is the
mean cross-entropy over the assistant tokens. Learning rate 1e-4 after 10 warmup steps, cosine
to 1e-5; 2 epochs (quick scope: 1), each a shuffle seeded by `seeds.train_shuffle`. The adapters
are written per epoch in PEFT's format (`train/adapter-epoch<N>`), the loss of every step to
`train/loss.jsonl`. A memory guard ends the process when the machine has less than 6 GiB
available (`BROOKLYN_MEM_GUARD_GIB`).

`train merge` folds the last epoch's adapters into the base and writes a bf16 Hugging Face
checkpoint to `train/merged` with `brooklyn-merge.json` (base, adapter path and its SHA-256).
`serving/twil-sft-bf16.env` serves it as `twil-bf16.env` serves the base (`CKPT_DIR`); the export
stamp and the evaluation key include the merge record, so a new merge exports again and makes
earlier results stale. `run.sh eval` writes `build/data/baseline/twil-sft-bf16/<split>-r1.*`
(`BROOKLYN_EVAL_SPLITS`, default `dev test`).

## Reproducibility

`manifest.json` (committed) holds the cutoff and its rule, the seeds, the Tlaloc version, and
per stage the identities of the inputs and the SHA-256 of what the stage derives:

| Entry | What is compared |
|---|---|
| `build.inputs.tlaloc` | the Tlaloc version in `build.gradle.kts` and the commit of its release tag (when the clone has the tag) |
| `build.inputs.runtime_classpath` | SHA-256 over the name and SHA-256 of every jar the application runs with (Tlaloc, Kotlin, sqlite-jdbc) |
| `data_snapshot` | URL, SHA-256, size and file count of the data snapshot (see Data snapshot) |
| `ingest.inputs.overpass`, `dohmh`, `complaints_311` | the query each raw file was fetched with (`raw/manifest.tsv`) and the SHA-256 of what it returned (`raw_sha256`: the file, or one `name<TAB>sha256` line per page) |
| `ingest.inputs.nta2020`, `google_local_meta`, `google_local_reviews` | file SHA-256 |
| `ingest.inputs.wayback_menu_pins` | SHA-256 of `pins/wayback-menus.tsv`, page count, captures after the cutoff (0) |
| `ingest.derived.<table>` | every table of `brooklyn.db` the tools read: `osm_places`, `establishments`, `inspections`, `complaints`, `g_places`, `reviews`, `restaurants`, `matches`, `neighborhoods`, `menus` (per scope) |
| `tools.inputs.searxng_image` | the image digest `searxng/searxng.sh` runs |
| `tools.derived.tool_schemas` | the OpenAI tool definitions |
| `models.inputs.<key>` | every pinned file of the model in the Hugging Face cache, by size and SHA-256 (see Models); an optional model not in the cache is skipped |
| `tools.derived.tools_smoke_snapshot` | `tools-smoke-snapshot.jsonl`, the 12 snapshot-mode tool results of the smoke (per scope); `verify` also reruns those 12 calls on the current `brooklyn.db` and reports the file `STALE` when it differs (a smoke file left from an earlier ingest) |
| `tasks.derived.split_train`, `split_dev`, `split_test` | SHA-256, task count and category counts of each split file (per scope); `STALE` when `tasks/inputs.tsv` (the cutoff, `seeds.tasks`, `seeds.task_split` and the digest of every ingest table at generation) differs from the current values. `tasks check` regenerates the splits and compares bytes |
| `agent.inputs.serving_config` | SHA-256 of `serving/twil-bf16.env`, its export and Triton settings, the Triton image digest, deterministic mode (off) |
| `agent.inputs.serving_sources` | git tree ids at HEAD of `core/src`, `ir/src`, `stablehlo/src`, `autograd/src`, `nn/src`, `tokenizer/src`, `maestro/src` (the modules `server.sh`'s export stamp depends on), `triton/backend`, `triton/kernels`, `triton/build_backend.sh` (the backend), and whether the working tree changes them. The endpoint runs from the alpha04 jars, which `build.inputs.runtime_classpath` covers |
| `agent.inputs.agent_settings` | SHA-256 of the system prompt, the budgets and sampling settings, the split and the question count |
| `agent.derived.serving_artifact` | SHA-256 over the path and SHA-256 of every file of the exported artifact (384 files, 17 GiB; hashes remembered in `model-sha256.tsv` by path, size and time) |
| `agent.derived.trajectories_dev` | the dev episodes (per scope): episode count and task ids compared; SHA-256 of the lines as canonical JSON with every `timing` key removed, answers, calls, calls without an error, completion tokens and verifier passes reported (a smoke run of the agent loop; serving is not bitwise reproducible) |
| `baseline.inputs.serving_config_ripple`, `eval_settings`; `baseline.derived.serving_artifact_ripple` | Ripple's serving config, the evaluation settings and runs, Ripple's exported artifact |
| `baseline.derived.<config>_<split>_r<run>` | each baseline run (per scope): task ids, the evaluation key's SHA-256, passes (compared within the TwIL dev spread), the other metrics and the trajectories' SHA-256 reported; `STALE` when the current key (settings, serving config, tasks) differs |
| `teach.inputs.serving_config_ripple`, `teach_settings` | the teacher's serving config; the system prompt, sampling settings, question and sample counts, shard size and length limit |
| `teach.derived.trajectories` | the recorded passes (per scope): kept count compared within three binomial standard errors (a fresh pass samples other trajectories); SHA-256 and per-category counts reported |
| `teach.derived.shards` | the shards' SHA-256, sequences and tokens (per scope); byte identity required when the passes are the recorded ones |
| `train.inputs.train_settings` | model, LoRA rank and alpha, targets, lr, epochs, warmup, loss chunk, buckets, precision |
| `train.derived.adapters` | trained on the recorded shards (the data snapshot's): sequences, kept, steps and epochs compared; on shards a new teach pass wrote, reported. Losses, the adapter's SHA-256, time and memory always reported (training on the GPU is not bitwise reproducible); `STALE` when the shards changed |
| `train.derived.merged` | the merged checkpoint's base and dtype; its record's and adapter's SHA-256 reported; `STALE` when the record names other adapters |
| `eval.inputs.serving_config_sft`, `eval.derived.twil_sft_bf16_<split>_r1` | the tuned student's serving config, and its runs checked as the baseline's are; the evaluation key includes the merge record, so a new merge makes them `STALE` |

Tables are hashed over a normalized dump (all columns, rows ordered by key and then by every
column, SQLite `quote()` of each value), so build and fetch times do not enter; the snapshot
date the tools use is the cutoff, not the day of the download. `digests/ingest-rows.tsv`
(committed) holds a 12-hex digest per key for `osm_places`, `establishments`, `inspections` (per
CAMIS), `complaints`, `restaurants` and `menus`; when a table differs, `verify` names the keys
whose rows changed, appeared or disappeared. `produced_by` records the commit whose code wrote
each scope's values and whether that code was clean.

`run.sh verify` recomputes everything and exits 1 on a difference. An entry the manifest pins
whose file or table is not there is reported `MISSING` and is a failure, so `verify` passes only
after the stages ran (`run.sh models ingest tools tasks agent`, or the same with `--quick`).
`tlaloc-brooklyn-eats verify` takes `--cache DIR` (another Hugging Face cache), `--rehash` (read
every model file again) and `--no-models` (data only, for a machine without the 77 GiB of
models). `tlaloc-brooklyn-eats verify --update` writes the values after a run; it refuses while an
output is stale, and never changes the model pins (only `models pin` writes them). The seed of every random choice is read from `seeds` in `manifest.json`
(`match_sample`, the 20 match pairs per kind that ingest writes to `match-sample.tsv`
for checking; `tasks`, the order candidates are visited and the templates; `task_split`, the
restaurant split); a missing seed stops the stage. Values that depend on the
menus are kept per scope: `full`, and `quick` (`run.sh --quick` reads menus for the first 150
places by OSM id).

### Data snapshot

DOHMH and 311 are rolling datasets: the pinned queries return other rows on a later day (the
2026-11-21 clean room found an establishment inspected in 2021-22 gone, never-inspected rows added
and 311 rows edited), so the queries alone do not pin the data. The snapshot holds what they
returned on the cutoff day, and the sampled outputs that no rerun reproduces bit for bit:

| In the zip | What |
|---|---|
| `raw/overpass.json`, `raw/nta2020.json`, `raw/manifest.tsv`, `raw/dohmh/`, `raw/311/` | the raw sources as fetched for the manifest (OpenStreetMap data © OpenStreetMap contributors, ODbL 1.0; DOHMH, 311 and NTA from NYC Open Data) |
| `full/teach/`, `quick/teach/` | Ripple-65B's sampled passes on the training questions, their summary and the training shards (per scope) |
| `full/rft/`, `quick/rft/` | the tuned student's rejection-sampling passes and shards |

It is the release asset `brooklyn-eats-data-v1`
(`brooklyn-eats-data-2026-11-13.zip`, 13.5 MB, SHA-256 in `manifest.json`). `run.sh snapshot`
(the first stage after `models`) downloads it into `build/snapshot/`, checks the SHA-256 and
unpacks: the raw files into `raw/` (a raw file already there with other content stops the stage),
and the scope's `teach/` and `rft/` when they are not there (a run that sampled again keeps its
own passes). `ingest` then builds from the pinned files, and `teach` and `rft` find every sample
pass done, so they do not serve a model to sample; `train` trains on the recorded shards.
`BROOKLYN_LIVE=1` skips the stage: every source is fetched live and `verify` reports a source that
changed by its `raw_sha256` and the tables by key. Google Local (a fixed research file, pinned by
SHA-256) and the Wayback pages (pinned captures) are fetched from their sources either way.

`tlaloc-brooklyn-eats snapshot pack --out FILE` writes the zip from `build/data` and
`build/quick-data` (entries sorted, fixed times: the same files give the same bytes);
`snapshot list FILE` prints its entries.

What a run serves a model for is compared within tolerances, not by digest: the baseline and
evaluation pass counts within the TwIL dev spread, the teach kept count within three binomial
standard errors, the agent smoke run by its questions only.

What can still differ for someone else, and how it shows:

- With `BROOKLYN_LIVE=1`: NYC Open Data and OpenStreetMap history are not frozen archives. DOHMH
  can add inspections dated before the cutoff after it (results are entered with a delay), change
  a past row (adjudicated grades) or drop establishments; 311 rows change status and resolution
  after the cutoff. `verify` reports these by `raw_sha256` and by table and key. Overpass returns
  an element's tags in no fixed order; `osm_places.tags_json` stores them sorted by key.
- Overpass `map_to_area` uses the server's current areas, not historical ones: a change to the
  Kings County boundary relation would change the OSM rows.
- The Internet Archive can stop serving a capture as pinned; the page is then `moved` or
  `sha_mismatch` and the menus digest reports the restaurants concerned.
- Live tool results (`live`, `record`) depend on the day by design and are not in the manifest.
## Results

Ingest at the cutoff 2026-11-13 (queries as above; code at 6961612a, DGX Spark):

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

### Tasks

Code at 311eabea (`produced_by` in manifest.json), `seeds.tasks` 20261113, `seeds.task_split`
4211, cutoff 2026-11-13 (DGX Spark, 2026-11-14):

| Kind | Train | Dev | Test | Quick (train / dev / test) |
|---|---|---|---|---|
| lookup `grade` | 480 | 48 | 48 | 36 / 5 / 5 |
| lookup `last_inspection` | 320 | 32 | 32 | 24 / 3 / 3 |
| lookup `violation` | 320 | 32 | 32 | 24 / 3 / 3 |
| filter `cuisine_area_grade` | 480 | 48 | 48 | 36 / 5 / 5 |
| filter `open_at` | 240 | 24 | 24 | 18 / 2 / 2 |
| comparison | 400 | 40 | 40 | 30 / 4 / 4 |
| multihop | 480 | 48 | 48 | 36 / 5 / 5 |
| complaints | 400 | 40 | 40 | 30 / 4 / 4 |
| menu `price` | 206 | 21 | 17 | 8 / 0 / 0 |
| unanswerable `outside_brooklyn` | 160 | 16 | 16 | 12 / 2 / 2 |
| unanswerable `nonexistent` | 240 | 24 | 24 | 18 / 2 / 2 |
| unanswerable `closed` | 99 | 9 | 6 | 12 / 2 / 2 |
| live `live_only` (not scored offline) | 40 | 8 | 8 | 6 / 3 / 3 |
| all | 3,865 | 390 | 383 | 290 / 40 / 40 |

Short of the targets:

- `closed` (99 / 9 / 6 of 160 / 16 / 16): only 55 restaurants have a DOHMH closure as their
  latest inspection, and each gives at most 3 questions.
- `price` (206 / 21 / 17 of 320 / 32 / 32; quick 8 / 0 / 0 of 24 / 3 / 3): only one-line
  "Dish $price" entries are used (see Tasks). The menu tasks come from 69 restaurants in train,
  9 in dev and 5 in test. In the quick ingest (menus for 150 places) no dev or test restaurant
  has such an entry.

The other kinds met their targets.

| Check | Full | Quick |
|---|---|---|
| Restaurants in two splits (train/test, train/dev, dev/test; by id) | 0, 0, 0 | 0, 0, 0 |
| Split files from two runs of `run.sh tasks` | identical SHA-256 | identical SHA-256 |
| `tasks check`: regenerated in memory | byte-identical | byte-identical |
| Offline tasks whose solution holds every fact and whose reference passes | 4,582 of 4,582 | 358 of 358 |
| Corrupted references rejected: added restaurant | 864 of 864 | 68 of 68 |
| grade letter changed | 1,127 of 1,127 | 89 of 89 |
| grade only as a guess ("probably has") | 576 of 576 | 46 of 46 |
| another grade as a guess appended ("It might be C now.") | 576 of 576 | 46 of 46 |
| refusal replaced by an invented answer | 480 of 480 | 38 of 38 |
| refusal followed by "It was last inspected on November 13, 2026." | 480 of 480 | 38 of 38 |
| reopening claimed in a `closed` task | 114 of 114 | 16 of 16 |
| invented violation code and pest appended | 384 of 384 | 30 of 30 |
| other complaint type | 232 of 232 | 22 of 22 |
| price plus $1 | 244 of 244 | 8 of 8 |
| price hedged ("$13.00 or $14.00") | 244 of 244 | 8 of 8 |
| dish at another price, true one as an old price | 244 of 244 | 8 of 8 |
| name replaced | 4,102 of 4,102 | 320 of 320 |
| count plus 1 | 1,082 of 1,082 | 83 of 83 |
| count hedged | 635 of 635 | 48 of 48 |
| second count added | 916 of 916 | 74 of 74 |
| rating plus 0.3 | 591 of 591 | 47 of 47 |
| date moved to a day the snapshot does not hold | 3,304 of 3,304 | 250 of 250 |
| fact negated | 1,550 of 1,550 | 122 of 122 |
| comparison: other restaurant picked | 480 of 480 | 38 of 38 |
| comparison: values swapped | 480 of 480 | 38 of 38 |
| comparison: "higher scores are better" | 191 of 191 | 21 of 21 |

Menu ground truth, read with the menu lines around each dish (full run at 79af11aa):

| Set | Read | Item named on its line, at the price on that line | Not one dish, or priced as an addition |
|---|---|---|---|
| test, every `price` task | 17 | 17 | 5: "Specialty Dogs $6" (a group, next to "Regular Dog $2"); Three Decker Diner's "Seasoned Beef", "Seasoned Ground Beef" and "Smoked Pastrami" ($4 to $5) and "Grilled Veggies" ($2), fillings listed after a quesadilla or burrito description whose "add" header the archived text does not keep |
| dev, every `price` task | 21 | 21 | 2: "Spirit Free $10" (a heading for non-alcoholic drinks); "Egg whites $2" (a substitution under a breakfast dish) |

An earlier version asked "On the archived menu of Three Decker Diner, what does Veggie Sausage
cost?" with $2.00, an item under "Add-ons" whose name also occurs in "With Veggie Sausage $10.95";
the add-on section and inner-name rules remove such questions. A filling or substitution whose
section header is missing from the archived text is not detected; its question still has the
line's price as its answer, and the line is what the menu tool returns.

No task read in these sets gives a dish the price of another line. The rules were tuned on
train samples of 50 and on the test and dev sets of earlier runs; the two sets above were read
after the last rule change.

A seeded sample of 30 test questions (`tasks show --split test --n 30 --seed 777`) was read with
its tool results: each is answered by its solution calls (it holds no menu task). DOHMH names
keep their double spaces ("Ziggyz  Restaurant").

### Agent

TwIL-8B bf16 through `run.sh agent` on the GB10 (code b8f2f422; one sequence at a time, greedy):

| | Full scope | Quick scope |
|---|---|---|
| Episodes | 10 (dev, 7 categories) | 10 (dev, 6 categories; the quick dev split has no menu task) |
| Stops | 9 answers, 1 call budget (an `open_at` filter: 2 `search_restaurants`, 1 `geocode` and 5 `get_restaurant` calls) | 10 answers |
| Turns | 43 | 41 |
| Tool calls | 33; 32 valid against the schema, 1 with an unknown property (`address` on `search_restaurants`; the model corrected it next turn) | 31; 30 valid against the schema, 1 schema error (`near` given as a string on `search_restaurants`) |
| Episodes with at least one successful call | 9 (the tenth refused a Manhattan question without a call) | 10 |
| Verifier passes | 3 of 10 | 5 of 10 |
| Completion tokens | 9,741 | 7,945 |
| Agent wall time | 909 s (10.7 completion tokens/s) | 732 s (10.9 tokens/s) |
| Parity (`agent parity`) | 43 of 43 turns: the endpoint's `prompt_tokens` equals the trainer's prompt length; every request's fresh render is a prefix of the trainer's ids; 9,743 of 42,003 tokens masked | 41 of 41; 7,948 of 38,051 masked |

Run to run: two runs of the full-scope episodes, each on a freshly started server (one with
`agent run`, one with `run.sh agent`), gave the same normalized SHA-256 (`1358206c...`), so the
manifest pins it; deterministic mode is off. Whether another GPU, driver or XLA build gives the
same tokens was not tested.

Verifier failures that read as limits of the verifier, not of the answer (fixed in the baseline
stage): dates written with U+2011 non-breaking hyphens (`2026‑02‑24`) are not read as dates;
bold phrases (`**Food Poisoning**`) and fragments such as "0 critical violations" are read as
restaurant names; "I only assist with restaurants ... in Brooklyn" is not recognized as an
out-of-scope refusal. Real misses also occur: the wrong restaurant credited with a count in a
comparison, a missing rating, the call budget spent before an answer (searches, a geocode and five restaurant lookups).

### Baseline

Full scope, snapshot tools, greedy, 4 conversations in flight, GB10. Episodes generated by the
code at d89444cf (TwIL dev r1), 98d703ba (TwIL dev r2 and test), 98d703ba and dedffd88 (Ripple
dev: 137 episodes before the machine's out-of-memory reboot, 245 after) and dedffd88 (Ripple
test); the agent and serving code is the same in all of them. Every verdict below is from the
verifier at 2456bc96 (`eval rescore`).

| Model | Split | Run | Tasks | Success % (± s.e.) | Fact precision % | Fact recall % | Answers with invented items % | Schema-valid calls % | Calls / task | Completion tokens / task | Tokens/s (all streams) | Episode median s | Episode p90 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TwIL-8B bf16 | dev | 1 | 382 | 257 = 67.3 ± 2.4 | 97.3 | 89.0 | 1.8 | 96.3 | 3.39 | 798 | 29.9 | 88.3 | 184.7 |
| TwIL-8B bf16 | dev | 2 | 382 | 253 = 66.2 ± 2.4 | 96.5 | 88.0 | 2.6 | 96.5 | 3.38 | 809 | 29.2 | 88.5 | 189.2 |
| TwIL-8B bf16 | test | 1 | 375 | 234 = 62.4 ± 2.5 | 95.8 | 88.4 | 4.5 | 96.2 | 3.39 | 780 | 29.7 | 84.4 | 192.4 |
| Ripple-65B NVFP4 + DFlash2 | dev | 1 | 382 | 311 = 81.4 ± 2.0 | 96.8 | 95.6 | 3.4 | 98.9 | 3.21 | 545 | 15.4 | 120.6 | 238.9 |
| Ripple-65B NVFP4 + DFlash2 | test | 1 | 375 | 302 = 80.5 ± 2.0 | 95.7 | 94.3 | 4.5 | 99.5 | 3.36 | 623 | 17.5 | 108.0 | 218.6 |

Run-to-run spread (TwIL dev, two runs on two server starts): 257 and 253 passes; verdicts differ
on 12 of 382 tasks (8 pass to fail, 4 fail to pass); 328 trajectories and 334 answers identical.
The manifest checks a run's pass count within these 12.

Ripple against TwIL: +14.1 points on dev, +18.1 on test. Passes by category (TwIL dev r1 / r2 /
Ripple dev; TwIL test / Ripple test):

| Category | Dev tasks | TwIL r1 | TwIL r2 | Ripple | Test tasks | TwIL | Ripple |
|---|---|---|---|---|---|---|---|
| lookup | 112 | 79 | 80 | 98 | 112 | 80 | 102 |
| filter | 72 | 58 | 59 | 52 | 72 | 59 | 50 |
| comparison | 40 | 28 | 23 | 30 | 40 | 24 | 34 |
| multihop | 48 | 19 | 18 | 39 | 48 | 15 | 37 |
| complaints | 40 | 31 | 31 | 37 | 40 | 33 | 34 |
| menu | 21 | 17 | 17 | 20 | 17 | 2 | 16 |
| unanswerable | 49 | 25 | 25 | 35 | 46 | 21 | 29 |

- Ripple gains most on multi-hop questions (critical violations plus what reviewers say) and on
  lookups; TwIL more often misses a required fact (recall 88-89% against 94-96%).
- Filters are the one category where TwIL passes more. In 9 of Ripple's 20 dev filter failures
  and 9 of its 22 test ones, every required place is named but Ripple adds places it marks as
  near misses ("a related option ... its listed cuisine is Latin American, not Peruvian"), which
  the verifier counts as extra items that do not match the question; the others mostly miss a
  required place.
- TwIL's test menu score (2 of 17): 11 failures give the price without the restaurant's name
  ("$4.00"), which the verifier requires, and 4 have the right answer after a first
  `search_restaurants` call with an `address` property the schema does not have (an invalid call
  fails the task). The test split's menu questions fall on a few restaurants (the split is by
  restaurant); dev's do not show it.
- Tokens/s is completion tokens over the run's wall time with four streams; Ripple's per-task
  answers are shorter (545-623 against 780-809 completion tokens).

Verifier limits still seen in these answers: some 311 descriptor phrases not in the snapshot's
complaints ("Insect Infestation"), a "Name" label, a restaurant named in another one's table row,
and grades read from a sentence about a different branch. As scored when they ran (the
verifier of the commit that ran them), the passes were TwIL 229, 242, 225 (dev r1, dev r2, test)
and Ripple 298 and 293; the rescores with the verifier fixes of this stage (1958023d, 2456bc96)
turned no pass into a fail. Ripple was ahead under either verifier (as run: dev 78.0% against
59.9% and 63.4%, test 78.1% against 60.0%).

### Teach and SFT

Teach (full scope, 2026-11-18): Ripple-65B on 300 training questions, one sample each, 10,097 s
for the 268 not recorded before an interruption. 222 passed (74.0%; fact precision 94.8%,
recall 93.1%); stops: answer 280, call budget 17, length 2, and one HTTP 400 (the endpoint's
context refusal, now the `context` stop). Rendered: 217 sequences (826,440 tokens, 115,364 of them
assistant tokens; length median 3,664, p90 4,833, max 8,708); 5 passing episodes were left out
because a turn's prompt is not a prefix of the chat in TwIL's template. `teach check`: all 217
parse back unchanged.

| Category | Questions | Passed | Kept |
|---|---|---|---|
| comparison | 43 | 29 | 29 |
| complaints | 43 | 38 | 38 |
| filter | 43 | 24 | 23 |
| lookup | 43 | 36 | 33 |
| menu | 43 | 41 | 40 |
| multihop | 43 | 34 | 34 |
| unanswerable | 42 | 20 | 20 |

Probe (TwIL-8B, rank 16, frozen base in bf16): 4,096 tokens 10.7 s a step, machine peak 72.0 GiB;
8,192 tokens 33.1 s, 102.2 GiB; 16,384 tokens refused by XLA's allocator (a 16 GiB attention
buffer). SFT: 215 of the 217 sequences (2 longer than 8,192 tokens left out), 2 epochs, 430
steps, 7,641 s, machine peak 100.6 GiB; mean loss per assistant token 0.368 (epoch 1) and 0.224
(epoch 2). Merge 113 s.

The tuned student (`serving/twil-sft-bf16.env`, served and evaluated as the baseline TwIL;
episodes generated by the code at cf8d97b6):

| Model | Split | Run | Tasks | Success % (± s.e.) | Fact precision % | Fact recall % | Answers with invented items % | Schema-valid calls % | Calls / task | Completion tokens / task | Tokens/s (all streams) | Episode median s | Episode p90 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TwIL-8B SFT bf16 | dev | 1 | 382 | 323 = 84.6 ± 1.8 | 94.3 | 95.7 | 3.4 | 99.8 | 2.98 | 505 | 26.4 | 66.3 | 131.3 |
| TwIL-8B SFT bf16 | test | 1 | 375 | 310 = 82.7 ± 2.0 | 95.1 | 95.9 | 4.3 | 99.9 | 3.15 | 545 | 26.5 | 69.5 | 136.5 |

Against the base TwIL: +17.3 points on dev (run 1; +18.4 on run 2) and +20.3 on test. Against
Ripple, the teacher: +3.2 on dev and +2.2 on test, not significant (1.2 and 0.8 standard errors
of the difference). Passes by category:

| Category | Dev tasks | TwIL | Ripple | SFT | Test tasks | TwIL | Ripple | SFT |
|---|---|---|---|---|---|---|---|---|
| lookup | 112 | 79 | 98 | 109 | 112 | 80 | 102 | 109 |
| filter | 72 | 58 | 52 | 51 | 72 | 59 | 50 | 48 |
| comparison | 40 | 28 | 30 | 29 | 40 | 24 | 34 | 31 |
| multihop | 48 | 19 | 39 | 44 | 48 | 15 | 37 | 42 |
| complaints | 40 | 31 | 37 | 36 | 40 | 33 | 34 | 35 |
| menu | 21 | 17 | 20 | 21 | 17 | 2 | 16 | 17 |
| unanswerable | 49 | 25 | 35 | 33 | 46 | 21 | 29 | 28 |

- The student gains where the teacher is ahead of it: lookups, multi-hop questions, menus
  (test: 2 to 17 of 17; the answers now name the restaurant) and unanswerable questions.
- Filters fall below the base TwIL, to Ripple's level. In 11 of the 21 dev filter failures and
  14 of the 24 test ones, every required place is named and others that do not match are added
  (Ripple: 9 of 20 and 9 of 22), as the teacher does with its near misses.
- Calls per task fall from 3.39 to 2.98-3.15 and completion tokens per task from 780-809 to
  505-545; fact recall rises from 88-89% to 96%. Fact precision falls (dev 97.3 to 94.3, test
  95.8 to 95.1): filter answers list more restaurants that do not match the question (extra
  items: dev 14 to 39, test 19 to 30; Ripple dev 14).
- The training questions come from restaurants in the training split only (the split is by
  restaurant), so no dev or test restaurant was seen in training.

### Rejection-sampling round

`run.sh rft` (2026-11-19, code 358f60e1): the SFT student sampled the 300 training questions
after the teacher's (temperature 0.7, top-p 0.95, `seeds.rft`) in 8,141 s; 256 passed (85.3%) and
all 256 were kept (946,839 tokens, 123,167 assistant tokens; length median 3,585, max 7,564);
`rft check`: all parse back. One epoch of new rank-16 adapters on the merged SFT checkpoint at lr
5e-5: 256 steps, 4,815 s, mean loss 0.134, machine peak 91.9 GiB; merge 121 s.

| Model | Split | Run | Tasks | Success % (± s.e.) | Fact precision % | Fact recall % | Answers with invented items % | Schema-valid calls % | Calls / task | Completion tokens / task | Tokens/s (all streams) | Episode median s | Episode p90 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TwIL-8B SFT + RFT bf16 | dev | 1 | 382 | 311 = 81.4 ± 2.0 | 93.8 | 95.1 | 4.5 | 99.8 | 2.96 | 516 | 28.6 | 61.4 | 120.0 |

The round does not improve on SFT: 311 dev passes against 323, a difference of 12, the spread
measured between two runs of the same model. 36 tasks go from pass to fail and 24 from fail to
pass. By category (SFT / SFT + RFT): lookup 109 / 109, menu 21 / 21, complaints 36 / 36,
unanswerable 33 / 36, comparison 29 / 27, filter 51 / 48, multihop 44 / 34. Of the 12 multi-hop
tasks that fail after the round, 9 miss a required fact (mostly the Google rating). The test
split was not run for this model.

### Pipeline time

`run.sh` prints each stage's wall time and the peak RSS of its process (DGX Spark, 2026-11-14):

| Run | Wall time | Stages |
|---|---|---|
| `run.sh --quick ingest tools`, clean-room clone, empty `build/`, empty Gradle user home and mavenLocal | 620 s | Gradle distribution, Tlaloc and dependencies from Maven Central, compile and SearXNG start 48 s; ingest 550 s (fetch 504 s: OSM, DOHMH, 311, NTA, 219 Wayback captures in 219 s, Google Local 2.5 GB in 145 s; build 46 s; peak RSS 2.7 GB); tools 22 s (peak RSS 0.4 GB) |
| `run.sh ingest tools`, same clone after the quick run (raw files present, 3,072 more captures to fetch) | 3,369 s | ingest 3,346 s (Wayback 3,290 s, about 1 capture per second; build 55 s; peak RSS 2.5 GB); tools 19 s (peak RSS 0.4 GB) |
| `run.sh ingest tools`, every raw file and capture present | 82 to 87 s | ingest 60 to 65 s (peak RSS 3.0 to 4.2 GB), tools 13 to 21 s (peak RSS 0.4 GB) |
| `run.sh --quick ingest tools`, raw files present | 70 to 95 s | ingest 47 to 49 s (peak RSS 2.4 GB), tools 18 to 41 s |
| `run.sh tasks` (full: generate 4,719 tasks 30 to 31 s, reference and mutation checks 21 s) | 54.0 to 54.4 s | peak RSS 2.6 to 3.2 GB |
| `run.sh --quick tasks` (372 tasks) | 8.8 to 9.0 s | peak RSS 1.9 to 2.4 GB |
| `tasks check` (full: regenerate, overlap, references, mutations) | 52.8 s | peak RSS 2.6 GB |
| `run.sh verify` (reruns the 12 snapshot smoke calls; model hashes remembered) | 3.0 to 3.3 s | peak RSS 1.3 to 1.4 GB |
| `run.sh verify`, model hashes not yet remembered (4 models, 198 GiB read) | 67.6 s | peak RSS 1.3 GB |
| `run.sh models`, models in the cache, hashes not yet remembered (TwIL-8B linked from its blobs, 77 GiB read) | 63 s | peak RSS 0.09 GB |
| `run.sh models`, hashes remembered | 0.5 s | peak RSS 0.09 GB |
| `models ripple_65b_dflash2` into an empty cache (7.5 GiB), interrupted at 1.3 GiB and run again | 45 s + 243 s | 25 to 36 MB/s from the Hub here; resumed at 1.3 GiB with a Range request; peak RSS 1.3 GB |
| `serving/server.sh export` (TwIL-8B bf16; Gradle compiles the exporter on the first run) | 44 to 68 s | machine memory in use 13.2 GiB before, 15.8 GiB at the peak; two exports gave byte-identical artifacts (384 files) |
| `run.sh agent` (full): server start 47 s (Triton load with the executable cache 46 s), 10 episodes 910 s, parity 0.6 s, stop | 964 s | agent client peak RSS 0.77 GB; machine memory in use 15.3 GiB before Triton, 68.2 GiB at the peak while serving |
| `run.sh --quick agent` | 787 s | start 47 s, 10 episodes 733 s (peak RSS 0.74 GB), machine memory peak 68.2 GiB |
| First server start (backend build in the Triton container, Triton load and compiles without the executable cache) | 178 s | Triton ready after 116 s |
| `run.sh baseline`, TwIL-8B bf16 (start 47 s; dev r1, test, dev r2) | 10,209 s, 9,835 s, 10,591 s per run | eval client peak RSS 1.7 GB; machine memory in use 72.0 GiB at the peak while serving |
| `run.sh baseline`, Ripple-65B NVFP4 + DFlash2 (start 132-137 s with the executable cache; dev, test) | 13,496 s (5,440 s before the reboot + 8,056 s), 13,342 s | eval client peak RSS 1.8 GB; machine memory in use 7.6 GiB before Triton, 87.1 GiB at the peak (2 s samples); MemAvailable at least 34.6 GiB (1 min samples) |
| `run.sh --quick baseline` (4 tasks per split; both models, one after the other) | 845 s | TwIL start 37 s, runs 116-212 s, peak 60.8 GiB; Ripple start 77 s, runs 126-145 s, peak 84.7 GiB |
| `eval rescore` of the five full runs | 2.9 s | peak RSS 1.6 GB |
| `run.sh --quick teach` (4 questions, 2 samples) | 447 s | start 72 s, passes 364 s (client peak RSS 0.85 GB), render 1.2 s, check 1.4 s |
| `run.sh teach` (300 questions; interrupted once for the probe) | 12,142 s in two runs | starts 379 s and 313 s; passes 1,352 s (32 episodes) and 10,097 s (268); machine memory peak 88.9 GiB; client peak RSS 1.3 GB |
| `run.sh probe` (buckets 4,096, 8,192, 16,384) | 737 s | load 63 s, compiles 119 s and 152 s, 3 steps each; peak RSS 61.2 GB; machine peak 102.2 GiB |
| `run.sh train`: `train run` (2 epochs, 430 steps) | 7,645 s | load 63 s, compiles 117 s and 149 s, steps 10.45 s and 32.65 s; peak RSS 61.0 GB; machine peak 100.6 GiB |
| `train merge` | 122 s | peak RSS 95.3 GB (`-Xmx96g`) |
| `run.sh eval` (twil-sft-bf16; export 52 s, start 89 s; dev, test) | 15,115 s | dev 7,304 s, test 7,716 s; eval client peak RSS 1.7-1.8 GB |
| `run.sh rft` (student start 47 s, 300 sampled episodes, render, check, train, merge, export 51 s, start 89 s, dev) | 20,110 s | passes 8,141 s; render 2.9 s; check 3.2 s; train 4,815 s (peak RSS 61.1 GB); merge 121 s; dev 6,877 s |

A run from scratch also downloads the 77 GiB of required models (`run.sh models`); its time
depends on the connection and was not measured here beyond the DFlash2 download above. The
`--quick` and full runs at e08e5c36 (`run.sh --quick models ingest tools`: 65 s; `run.sh models
ingest tools`: 78 s; raw files and models present) matched all 26 manifest entries.

### Clean-room reproduction

A fresh clone of the repository at 74de17de (the full-scope check at 3429a7b1), with an empty Gradle user home and an
empty mavenLocal (so Tlaloc 0.1.0-alpha04 came from Maven Central), following only this README:
`run.sh --quick ingest tools` then `run.sh verify` matched all 22 manifest entries (quick scope),
including the digests of every ingest table and of the snapshot tool results. The later
commits (verify's `MISSING`/`STALE` checks, the seed read from `manifest.json`, the Foojay
toolchain resolver) left every manifest digest unchanged; the clean-room run is repeated at the
end of the example's construction.

A full run from scratch therefore takes about 66 minutes here (620 s + 3,369 s), most of it the
3,450 Wayback captures at 1 request per second. In the same clone, `run.sh ingest tools` (full)
fetched the other 3,072 captures; the archive redirected 4 pinned captures to neighbouring ones
on all 3 tries (bakupalace.com, mamannyc.com, canalbar.com, sawa.nyc), and `verify` matched 21 of
22 entries, reporting `menus (full)` with `changed 2 (41129994, 50147662)`. Running
`run.sh ingest tools` again 80 minutes later fetched the 3 missing pages with their pinned
SHA-256, and `run.sh verify` matched all 22 entries (full scope). A run now reads such places a
second time 5 minutes after the others; if the archive still redirects, rerun `ingest`.
