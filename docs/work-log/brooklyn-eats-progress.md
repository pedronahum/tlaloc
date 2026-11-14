# Brooklyn Eats progress

Plan: `docs/work-log/brooklyn-eats-plan.md`. Example: `examples/brooklyn-eats/`.

## Stage 1: Scaffold (2026-11-13)

- `examples/brooklyn-eats/` is a standalone Gradle build (Kotlin 2.4.20, `jvmToolchain(25)`),
  resolving Tlaloc `0.1.0-alpha04` from mavenLocal like `examples/lora-finetune`. No Gradle
  wrapper inside; build with `./gradlew -p examples/brooklyn-eats build` from the repo root.
- Dependencies: tlaloc core, ir, autograd, nn, stablehlo, tokenizer, runtime-pjrt, endpoint;
  `org.xerial:sqlite-jdbc:3.50.3.0`. JSON uses Tlaloc's own `io.tlaloc.core.io`
  (`parseJson`, `JsonValue`, `jsonQuote`), already used by `tlaloc-endpoint`; no extra library.
- One entry point `brooklyn.MainKt` with subcommands `ingest tools tasks agent baseline teach
  probe train distill rft eval serve chat` (plus `help` and `check`). An unimplemented stage
  prints `<name>: not built yet` and exits 3; unknown subcommand exits 2.
- `check` opens an in-memory SQLite database and creates an FTS5 table: prints
  `sqlite 3.50.3, fts5 ok`.
- `run.sh` runs `installDist` and then the stages in order (or the stages given as arguments);
  it stops at the first non-zero stage. Today it stops at `ingest` with exit 3.
- `.gitignore` ignores `build/`, `.gradle/`, `.kotlin/`.
- Tests (`MainTest`): every stage name dispatches, unknown subcommand fails, SQLite + FTS5 load.

Commands and results:

```
./gradlew -p examples/brooklyn-eats build      BUILD SUCCESSFUL in 12s; 3 tests, 0 failures
examples/brooklyn-eats/run.sh                  "ingest: not built yet", exit 3
build/install/.../tlaloc-brooklyn-eats <stage> each of the 13 stages: "not built yet", exit 3
```

## Stage 2: Ingest (2026-11-13)

`ingest` fetches the four sources into `examples/brooklyn-eats/build/data/raw/` and builds
`build/data/brooklyn.db` from the raw files alone (`--build-only` rebuilds without the network,
`--refresh` refetches, `--skip-google` leaves the Google tables empty). Code: `Http.kt` (generic
User-Agent, per-host spacing with Nominatim at 1 request/s, exponential backoff on 429/5xx
honouring Retry-After), `Sources.kt` (parsers), `Matching.kt` (entity resolution), `Ingest.kt`
(fetch, schema, build, report). JSON is Tlaloc's `io.tlaloc.core.io.parseJson`.

Sources as fetched:

- OSM: Overpass, relation 369518 (Kings County) as the area, seven `amenity` values,
  `out center tags`. overpass-api.de answered 502 (at capacity) on the first attempt and
  succeeded on the retry; overpass.kumi.systems timed out in a manual probe, so it is the second
  mirror, not the first. 2.2 MiB.
- DOHMH `43nn-pn8j`, `boro='Brooklyn'`: 74,406 rows in two pages.
- 311 `erm2-nwe9`, Brooklyn, Food Poisoning + Food Establishment, created since 2023-11-13:
  13,908 rows. Mobile Food Vendor (3,927 rows in the same window) is not included.
- Google Local 2021 (current link on cseweb.ucsd.edu/~jmcauley/datasets.html is
  mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/): `meta-New_York.json.gz` 61 MB,
  `review-New_York.json.gz` 2.6 GB (the full file, not the 10-core one). I downloaded these two
  with curl into `build/data/raw/googlelocal/` while writing the code; the ingest found them and
  did not download them again, and the `fetched_google_local` time in `meta` is the ingest run's,
  not the download's (both 2026-11-13). The same URLs are in `fetchSources`.

Commands and results:

```
./gradlew -p examples/brooklyn-eats build installDist     11 tests, 0 failures (3 MainTest, 8 IngestTest)
tlaloc-brooklyn-eats ingest                              first run: fetch 152 s, build 46 s, peak RSS 3.0 GB, exit 0
tlaloc-brooklyn-eats ingest --build-only                 final rules: build 48 s, database 349 MiB, exit 0
```

Counts (final build): OSM 4,880 places (4,835 named); DOHMH 74,406 rows, 8,023 establishments,
39,134 critical-violation rows; 311 13,908; Google Local 272,189 metadata lines read, 0 failed
to parse among the Brooklyn candidates, 8,962 Brooklyn food businesses; 33,459,761 review lines
read, 1,232,350 kept (657,884 with text); restaurants 9,227 (8,023 CAMIS, 1,204 OSM only).
Grades of the CAMIS restaurants: A 5,560, B 420, C 129, Z 315, N 1, none 1,598.

Match rates (final build): DOHMH->OSM 3,631 / 8,023 (45.3%); OSM->DOHMH 3,631 / 4,835 (75.1%);
Google->restaurant 3,994 / 8,962 (44.6%); DOHMH establishments with reviews 3,440 / 8,023
(42.9%); 311 linked 8,433 / 13,908 (60.6%: 7,734 address, 699 BBL). Rule counts:
DOHMH-OSM near 3,530 / address 61 / close 40; Google near 3,877 / address 88 / close 29.

Matching thresholds changed twice after hand-checking, each time by reading the weakest pairs:

1. First rules: `address` accepted name similarity >= 0.55 at the same address and `close`
   >= 0.65 within 35 m. The 70 lowest-scoring `address`/`close` pairs included successor
   businesses at one address (La Cafette/Recette, Zhangs Garden/Sun Garden, Garcia's/Girolamo's,
   Myst Lounge/V Lounge, Little Toninos/Cozie's). Added distinctive-word similarity (generic words
   removed) and required >= 0.75 of it for `address` and `close`; a proper subset of distinctive
   words gets no containment credit (China Tang Kitchen/New China was accepted when it did).
2. A sample of `near` pairs with name similarity below 1 had El Arco matched to El Arcoiris Deli
   Grocery through a prefix of the names without spaces. Containment now works on whole words.

Precision on the final build, checked by hand (I read each pair's names, addresses and distance;
corner premises with two street addresses counted as correct when the name is the same):

| Sample | Correct |
|---|---|
| 20 random DOHMH-OSM pairs (seed 7) | 20 / 20 |
| 20 random Google-restaurant pairs (seed 7) | 20 / 20 |
| 20 random `address`/`close` pairs (218 of 7,625) | 18 / 20; 1 renamed premises (Ferdinando's Focacceria / Bar Ferdinando), 1 wrong (Matchless / Matches, 557 vs 560 Manhattan Ave) |
| 20 random `near` pairs with name similarity < 1 | 20 / 20 |

Random-sample precision 40 / 40. The 40-row sample file is `build/data/match-sample.tsv`
(not committed).

Limits:

- About half of DOHMH establishments have no OSM place: OSM's Brooklyn coverage is 4,835 named
  food places against 8,023 permitted establishments. Google Local is a 2021 snapshot, so
  businesses opened since then have no reviews, and a match can attach a predecessor's 2021
  reviews to a renamed business at the same premises (as in the Ferdinando's pair).
- 311 complaints carry an address, not a business; 39.4% stay unlinked (shared buildings with
  several establishments, or addresses that are not a permitted establishment).
- No neighbourhood names yet: DOHMH carries an NTA code (`nta`, e.g. BK45) and no name; the tasks
  stage needs a code-to-name table for questions like "Park Slope".

Tests (`IngestTest`, synthetic fixture under `src/test/resources/fixture/raw`, fictional
business names): Overpass parsing (way centres, tags), DOHMH parsing (dates, scores, zero
coordinates dropped), 311 parsing, Google line parsing (NaN line refused, Brooklyn + food
filter, reviewer identity dropped), name/address normalization and similarities, one-to-one
matching with distance, a full database build from the fixture (matches by rule, grades,
complaint linking, the three FTS indexes, snapshot date), and a build without Google files.
Nothing under `build/` is committed.

## Stage 3: Tools (2026-11-13)

Picked up from an interrupted attempt whose code was in the tree uncommitted. Kept: the tool
definitions and schema validator (`Schema.kt`), the tool server (`Tools.kt`, `ToolsCmd.kt`),
opening hours (`Hours.kt`), 2020 NTA neighbourhoods (`Neighborhoods.kt`, Socrata `9nt8-h7nd`),
HTML-to-menu extraction (`Menu.kt`), `HttpNet`, the SearXNG wrapper. Changed:

- Menus came from the live sites and were cached into `brooklyn.db` by live calls (and a
  `fetch-menus` command), so the snapshot changed with every live call and could not be
  reproduced. Now `ingest` reads menus from the Internet Archive (`Wayback.kt`): latest 200
  capture on or before the cutoff (availability API; CDX API when the closest capture is after
  the cutoff), original bytes via `web/<ts>id_/<url>`, up to 2 menu links followed. Every page is
  pinned in `examples/brooklyn-eats/pins/wayback-menus.tsv` (committed) with timestamp, archive
  URL, content type and SHA-256; with pins present ingest fetches exactly those captures and
  checks the hashes. Pages are stored by hash in `build/data/raw/wayback/pages`; the build reads
  them back offline. `fetch-menus` is removed.
- The database is opened read-only in every mode; live `get_menu` fetches the site and writes
  nothing. Snapshot `get_menu` returns the capture time and archive URL.
- `manifest.json` (new) with the cutoff, seeds, and per stage input identities and SHA-256 of
  normalized table dumps; `verify` / `verify --update`; `run.sh verify`; `run.sh --quick`
  (`BROOKLYN_QUICK=1`, menus for the first 150 places by OSM id).
- SearXNG pinned by digest (`searxng/searxng@sha256:287af61f...`), settings mounted read-only,
  secret by environment (the earlier start script failed on its second run: the container took
  ownership of `build/searxng`). `run.sh` starts it before `tools` and stops it after.
- `HttpNet` paced all hosts under one lock (sleeping inside it); now per host.
- Live geocoding of DOHMH addresses failed ("8002 5 Avenue" finds nothing on Nominatim):
  numbered streets become ordinals ("5th Avenue") before the query.
- Live source labels named OSM data "(live)"; OSM, matches and Google Local are labelled with the
  snapshot date in every mode.
- Tests: four assertions in the earlier `ToolsTest` compared `JsonNumber.toString()` digits or
  were always true; they now compare values.

Problems found on real data and fixed, in order:

1. 13 start URLs failed on every run: the archive answers 403 on CDX for some domains
   (linktr.ee, order.toasttab.com, order.online). Pinned as `excluded`.
2. 3 of 11 sampled menus were binary: `id_` captures keep the site's Content-Encoding and Java's
   HTTP client does not inflate. 672 stored pages were gzip. Gzip is now inflated when reading;
   other encodings are reported `undecodable` (29 places). Inflating exposed menu links on those
   pages: a further ingest pinned 430 more pages (596 s).
3. The binary check then misclassified PDFs; it now applies only to pages served as HTML or text.
4. Hand check of every 50th `ok` menu (13): 9 menus, 4 not (legal and promotion text with dollar
   amounts, Shopify listings). Priced lines are now at most 100 characters and Shopify
   "Regular/Sale price" lines are dropped: 622 -> 608 menus. Fresh sample (every 50th, offset 26,
   12 menus): 11 menus, 1 coffee roaster's shop page.
5. 3 captures have an empty body; pinned as `empty` instead of `ok` with the empty hash.
6. Clean-room reproduction (below) found a capture the archive now redirects to another
   timestamp. Pinned refetches now retry 3 times (15 s apart) and report `moved` or
   `sha_mismatch`; the pin is kept.

Commands and results (code at the commit of this stage; cutoff 2026-11-13):

```
./gradlew -p examples/brooklyn-eats build installDist      30 tests, 0 failures (3 Main, 8 Ingest, 13 Tools, 6 Wayback)
tlaloc-brooklyn-eats ingest --quick (first)                5:42, menus 286 s, peak RSS 2.4 GB; 155 places, 36 ok (before the gzip fix)
tlaloc-brooklyn-eats ingest (first full discovery)         1:23:01, menus 4,923 s, build 57 s, peak RSS 3.5 GB
tlaloc-brooklyn-eats ingest (pins + stored pages)          1:01, menus 3-6 s, build 56 s, peak RSS 2.8 GB; 0 lookups
tlaloc-brooklyn-eats verify --update / run.sh verify       7 entries; "verify: 7 entries match" after an offline --build-only rebuild
run.sh --quick ingest tools                                1:08, exit 0, tools smoke: all passed
run.sh ingest tools                                        1:41, exit 0, tools smoke: all passed, peak RSS 4.2 GB
tlaloc-brooklyn-eats tools                                 18.9 s, peak RSS 0.4 GB
tlaloc-brooklyn-eats tools smoke --modes record,replay     replay == record: 12 / 12 results identical
tlaloc-brooklyn-eats tools schemas --out ...               8 definitions, exit 0 (no schema problems)
```


Menus on the full run: 2,596 places with a menu URL or website; 608 `ok` (502 DOHMH
restaurants), 1,725 `no_prices`, 172 `no_snapshot`, 43 `pdf`, 29 `undecodable`, 9 `excluded`,
4 `not_html`, 3 `empty`, 3 HTTP 403. 501 of the 608 menus came from a menu link, 107 from the
start page; median 27 priced lines; captures 55 before 2024, 118 in 2024-2025, 435 in 2026.
Pins: 3,986 pages, 840 KB.

Tool smoke on the full run, sample Lindenwood Diner (CAMIS 40705411), 12 calls per mode:
snapshot 12/12 and no HTTP client constructed, byte-identical on a second run; live 12/12 with
real data (DOHMH grade A 2026-01-13, 5 complaints on the tax lot, Nominatim hit on the OSM node,
the live menu page, SearXNG results from the restaurant's site and Yelp). On the quick run the
sample was Rocca Cafe & Lounge; on an earlier full run Istanbul Bay, whose live menu showed
lentil soup at $8.95 against $7.99 in its 2022 capture.

Reproduction from the pins: a second data directory with the same raw OSM, DOHMH, 311, Google
Local and NTA files and no archived pages, `ingest --quick`: 0 lookups, 218 captures fetched by
pinned URL, 217 matched their SHA-256. The archive now 302-redirects capture 20260715013642 of
pipinsbayridge.com to 20260612052045, although CDX still lists it (seen on three tries minutes
apart and on two runs an hour apart). That page is `moved`; 155 of 156 quick menu rows are
identical to the main directory's, and `verify --data build/clean` reports the quick menus
digest as different. Earlier the same clean-room check (before the gzip fix) had 0 mismatches
on 203 captures.

Limits:

- Overpass, DOHMH and 311 are still fetched without cutoffs (the Reproducibility stage pins
  them); until then the menus and `restaurants.nta2020` digests reproduce only from the same raw
  files.
- The Internet Archive is not fully immutable: 1 of 218 pinned captures in the clean-room check
  is no longer served. `verify` reports it; the menu text cannot be recovered from the pin.
- No PDF or brotli extraction; menus rendered by JavaScript or images are `no_prices`.
- `verify` reports differences per table digest and row count, not per key.
- `web_search` in snapshot mode is a stand-in over the restaurant index; `geocode` in snapshot
  mode is a gazetteer of the snapshot's addresses (no interpolation between house numbers beyond
  the nearest known number on the same street).

## Stage 3 (continued): Tools, reproducibility fixes (2026-11-14)

An independent check of the Tools stage found that nobody else could reproduce its outputs from
`run.sh` alone. Problems and what changed (code commits `cfc47ae1`-`eef86ac4` and `6961612a`, manifest
commits `74de17de` and `3429a7b1`):

1. **Source queries were not pinned.** Overpass had no date, DOHMH no cutoff, and 311 used
   `created_date >= today - 3 years`. Now every query is pinned to the manifest cutoff
   (2026-11-13), exclusive for every source: Overpass `[date:"2026-11-13T00:00:00Z"]`, refused
   when the answer's `timestamp_osm_base` is before that date; DOHMH
   `inspection_date < '2026-11-13T00:00:00'`; 311 `created_date` in [2023-11-13, 2026-11-13);
   Wayback captures up to 2026-11-12 23:59:59 UTC (previously "on or before 2026-11-13", which
   on 2026-11-13 was not yet in the past). Google Local is pinned by SHA-256 in the manifest and
   checked on every ingest (cached by size and mtime; downloaded again when it differs; ingest
   stops if the published file no longer matches). `raw/manifest.tsv` records the query of each
   raw file; a raw file fetched with another query is fetched again.
2. **DOHMH paging lost rows.** With the cutoff in place the paged fetch (50,000 rows per page,
   ordered by `:id`) returned 74,406 rows of 7,762 CAMIS, while `count(*)` and
   `count(distinct camis)` of the same filter gave 74,412 and 8,030. A single request returns
   74,412 rows of 8,030 CAMIS (the dataset itself has 64 exact duplicate rows). Socrata sources
   are now fetched in one request and checked against `count(*)` (3 tries). The ingest stage's
   earlier numbers (8,023 establishments, 74,406 rows) came from the paged fetch.
3. **Snapshot date was the download day.** `meta.snapshot_date` came from the DOHMH fetch time;
   it is now the cutoff, so `as_of`, "today" (12-month critical violations, `since` windows),
   `open_at` in the smoke and the source labels are the same on any download day. Test:
   building the fixture with cutoff 2026-10-01 gives `as_of` 2026-10-01.
4. **verify could not see changed tool results.** `manifestEntries` now hashes every table the
   tools read (`osm_places`, `establishments`, `inspections`, `complaints`, `g_places`,
   `reviews`, `restaurants` with all columns, `matches`, `neighborhoods`, `menus` per scope) over
   a normalized dump, and `tools-smoke-snapshot.jsonl` per scope. `digests/ingest-rows.tsv`
   (committed, 1.1 MB) holds a 12-hex digest per key of `osm_places`, `establishments`,
   `inspections` (per CAMIS), `complaints`, `restaurants` and `menus`; on a mismatch `verify`
   prints the changed, new and missing keys. Test: changing one inspection score of CAMIS 1002
   in the fixture database makes `verify` print `changed 1 (1002)` under `inspections` only.
5. **Weak manifest checks.** `tools.inputs.searxng_image` is now read from the `image=` line of
   `searxng/searxng.sh` (the constant in `Manifest.kt` is gone); a test changes the digest in the
   script and gets a mismatch. `build.inputs.tlaloc` holds the Tlaloc version from
   `build.gradle.kts`; `build.inputs.runtime_classpath` the SHA-256 over name and SHA-256 of the
   14 jars the application runs with (11 of them Tlaloc, plus Kotlin stdlib, annotations and
   sqlite-jdbc). `produced_by.<scope>` records the
   commit at `verify --update` and whether `src/`, `run.sh`, the Gradle files and `searxng/` were
   clean (both scopes: `6961612a`, clean). The 26 alpha04 jars in this machine's mavenLocal have
   the same SHA-1 as Maven Central's, and the clean-room build resolved them from Central.
6. **A full run rewrote a committed input.** Normal ingests now fetch only pinned captures and
   never write `pins/wayback-menus.tsv`; a page with no pin is `not_pinned`. Pinning is the
   maintainer step `ingest --pin-menus`: it looks up pages with no pin, pins after the cutoff, or
   captures no longer served as pinned (pinning the capture the archive serves, taking the
   timestamp from the archive's redirect), and on a full run writes exactly the pages used.
   Run on the pinned sources: 6 pins replaced (3 captures of 2026-11-13, 3 empty-body pins), 16
   pins of pages no longer read dropped; 3,970 pages. The menus table kept its statuses (608 ok).
7. **Pipin's Pub capture.** With its stored page deleted, `--pin-menus` refetched capture
   20260715013642 and got the pinned bytes (the archive served it without the redirect seen on
   2026-11-13), so the pin was kept. The redirect is intermittent; a run that meets it reports
   the page `moved` and `verify` names the restaurant under `menus`.
8. **Wayback redirects in the full clean-room run.** 4 of 3,072 pinned captures were redirected
   on all 3 tries; 80 minutes later all were served as pinned. Places with such a page are now
   read a second time 5 minutes after the others (`6961612a`); a later `ingest` also retries
   pages not yet stored.
9. **Timing from scratch** (below).

Numbers on the pinned sources: OSM 4,880 places; DOHMH 74,412 rows, 8,030 establishments
(latest inspection 2026-11-10); 311 13,921 (created 2023-11-13 to 2026-11-12); Google Local
8,962 businesses, 1,231,484 reviews; restaurants 9,233; DOHMH-OSM 3,632 / 8,030 (45.2%);
Google-restaurant 3,993 / 8,962; 311 linked 8,452 / 13,921 (60.7%); menus 608 ok of 2,596. The
hand-checked match precision (40 / 40) was measured on the earlier download and not repeated.

Commands and results:

```
./gradlew -p examples/brooklyn-eats build installDist          36 tests, 0 failures (3 Main, 8 Ingest, 13 Tools, 6 Wayback, 6 Repro)
tlaloc-brooklyn-eats ingest --refresh --pin-menus              191 s, peak RSS 3.8 GB (sources refetched with the pinned queries)
run.sh ingest tools  (main data dir, code eef86ac4)            82 s; ingest 60 s / 3.5 GB, tools 17 s / 0.4 GB; smoke all passed
run.sh --quick ingest tools  (second data dir, same raw)       95 s; ingest 47 s / 2.4 GB, tools 41 s; smoke all passed
tlaloc-brooklyn-eats verify --update  (full, then quick)       22 entries per scope; committed as 74de17de
./gradlew -p examples/brooklyn-eats build installDist          37 tests, 0 failures (code 6961612a, + second-pass test)
run.sh ingest tools  (main, code 6961612a)                     87 s; ingest 62 s / 4.2 GB, tools 21 s; smoke all passed
run.sh --quick ingest tools  (second data dir, 6961612a)       70 s; ingest 49 s / 2.4 GB, tools 18 s; smoke all passed
tlaloc-brooklyn-eats verify --update  (full, then quick)       digests unchanged; only tlaloc_jars 0 -> 11; committed as 3429a7b1
```

Clean-room check: `git clone` of the repository at `74de17de` into a scratch directory, an empty
`GRADLE_USER_HOME` whose `gradle.properties` points `maven.repo.local` at an empty directory
(so nothing came from this machine's Gradle cache or mavenLocal), empty `build/`:

```
run.sh --quick ingest tools        620 s total, exit 0: Gradle + compile + SearXNG start 48 s;
                                   ingest 550 s (fetch 504 s: 219 Wayback captures 219 s,
                                   Google Local 2.5 GB 145 s; build 46 s), peak RSS 2.7 GB;
                                   tools 22 s, 0.4 GB; snapshot and live smoke all passed
run.sh verify                      22 of 22 entries match (quick scope)
run.sh ingest tools (full)         3,369 s, exit 0: ingest 3,346 s (3,072 captures in 3,290 s,
                                   build 55 s), peak RSS 2.5 GB; tools 19 s; smoke all passed;
                                   4 captures redirected on all 3 tries
run.sh verify                      21 of 22 match; menus (full): changed 2 (41129994, 50147662)
(checkout 6961612a) run.sh ingest tools   85 s, 3 missing pages fetched with the pinned SHA-256
(checkout 3429a7b1) run.sh verify         22 of 22 entries match (full scope)
```

A full run from scratch takes about 66 minutes here (620 s + 3,369 s), the quick run about 10
minutes (the plan asks for under about an hour for the quick run).

Limits that remain: NYC Open Data and OSM history can still change rows dated before the cutoff
(late DOHMH entries, adjudicated grades, 311 status updates), Overpass areas are current rather
than historical, and Wayback captures can be redirected for a time. `verify` reports each of
these by table and key; none can be prevented from here. The new tests: 7 in `ReproTest`
(pinned queries, snapshot date = cutoff, no lookups or pin writes without `--pin-menus`,
re-pinning moved and after-cutoff captures, the second pass, verify naming changed keys and a
changed SearXNG digest, the rows file).
