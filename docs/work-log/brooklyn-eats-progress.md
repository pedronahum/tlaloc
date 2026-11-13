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
