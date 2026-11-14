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
skipped and reported). Tlaloc `0.1.0-alpha04` resolves from Maven Central; mavenLocal is searched
first, so a locally published Tlaloc of the same version takes precedence (`run.sh verify`
reports the SHA-256 of the runtime classpath; see Reproducibility). No `publishToMavenLocal` is
needed.

```bash
./gradlew -p examples/brooklyn-eats build    # from the repo root: compiles and runs the tests
examples/brooklyn-eats/run.sh --quick        # every stage, reduced sizes
examples/brooklyn-eats/run.sh                # every stage, full sizes
examples/brooklyn-eats/run.sh ingest tools   # selected stages
examples/brooklyn-eats/run.sh models         # download the pinned models that are missing, check every file
examples/brooklyn-eats/run.sh verify         # compare inputs, derived data and models with manifest.json
```

`run.sh` prints each stage's wall time and the peak RSS of its process.
A stage that is not implemented yet prints `not built yet` and exits with code 3.

## Stages

| Subcommand | Stage | Status |
|---|---|---|
| `models` | The pinned Hugging Face models: download what is missing, check every file's SHA-256 | built |
| `ingest` | Snapshot the sources into one SQLite file with FTS5; entity resolution | built |
| `tools` | Tool server: search, restaurant, inspections, complaints, reviews, menu, geocode, web search | built |
| `tasks` | Questions with ground truth and verifiers; splits by restaurant | built |
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

## Reproducibility

`manifest.json` (committed) holds the cutoff and its rule, the seeds, the Tlaloc version, and
per stage the identities of the inputs and the SHA-256 of what the stage derives:

| Entry | What is compared |
|---|---|
| `build.inputs.tlaloc` | the Tlaloc version in `build.gradle.kts` and the commit of its release tag (when the clone has the tag) |
| `build.inputs.runtime_classpath` | SHA-256 over the name and SHA-256 of every jar the application runs with (Tlaloc, Kotlin, sqlite-jdbc) |
| `ingest.inputs.overpass`, `dohmh`, `complaints_311` | the query each raw file was fetched with (`raw/manifest.tsv`) |
| `ingest.inputs.nta2020`, `google_local_meta`, `google_local_reviews` | file SHA-256 |
| `ingest.inputs.wayback_menu_pins` | SHA-256 of `pins/wayback-menus.tsv`, page count, captures after the cutoff (0) |
| `ingest.derived.<table>` | every table of `brooklyn.db` the tools read: `osm_places`, `establishments`, `inspections`, `complaints`, `g_places`, `reviews`, `restaurants`, `matches`, `neighborhoods`, `menus` (per scope) |
| `tools.inputs.searxng_image` | the image digest `searxng/searxng.sh` runs |
| `tools.derived.tool_schemas` | the OpenAI tool definitions |
| `models.inputs.<key>` | every pinned file of the model in the Hugging Face cache, by size and SHA-256 (see Models); an optional model not in the cache is skipped |
| `tools.derived.tools_smoke_snapshot` | `tools-smoke-snapshot.jsonl`, the 12 snapshot-mode tool results of the smoke (per scope); `verify` also reruns those 12 calls on the current `brooklyn.db` and reports the file `STALE` when it differs (a smoke file left from an earlier ingest) |
| `tasks.derived.split_train`, `split_dev`, `split_test` | SHA-256, task count and category counts of each split file (per scope); `STALE` when `tasks/inputs.tsv` (the cutoff, `seeds.tasks`, `seeds.task_split` and the digest of every ingest table at generation) differs from the current values. `tasks check` regenerates the splits and compares bytes |

Tables are hashed over a normalized dump (all columns, rows ordered by key and then by every
column, SQLite `quote()` of each value), so build and fetch times do not enter; the snapshot
date the tools use is the cutoff, not the day of the download. `digests/ingest-rows.tsv`
(committed) holds a 12-hex digest per key for `osm_places`, `establishments`, `inspections` (per
CAMIS), `complaints`, `restaurants` and `menus`; when a table differs, `verify` names the keys
whose rows changed, appeared or disappeared. `produced_by` records the commit whose code wrote
each scope's values and whether that code was clean.

`run.sh verify` recomputes everything and exits 1 on a difference. An entry the manifest pins
whose file or table is not there is reported `MISSING` and is a failure, so `verify` passes only
after the stages ran (`run.sh models ingest tools`, or `run.sh --quick models ingest tools`).
`tlaloc-brooklyn-eats verify` takes `--cache DIR` (another Hugging Face cache), `--rehash` (read
every model file again) and `--no-models` (data only, for a machine without the 77 GiB of
models). `tlaloc-brooklyn-eats verify --update` writes the values after a run; it refuses while an
output is stale, and never changes the model pins (only `models pin` writes them). The seed of every random choice is read from `seeds` in `manifest.json`
(`match_sample`, the 20 match pairs per kind that ingest writes to `match-sample.tsv`
for checking; `tasks`, the order candidates are visited and the templates; `task_split`, the
restaurant split); a missing seed stops the stage. Values that depend on the
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
