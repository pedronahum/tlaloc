# Contributing to Tlaloc

Bug reports, questions and pull requests are welcome. For anything larger than a
small fix, open an issue first so the approach can be agreed before you write it.

## Building

- JDK 25 builds the repository. `core`, `ir`, `autograd`, `nn`, `stablehlo`, `maestro`
  and the Gradle plugin target Java 21 bytecode; the compiler plugin, `kptx` and the
  runtime modules target Java 25.
- `./gradlew test` runs the whole suite, including `checkKotlinAbi` and the POM checks.
- Tests that need a GPU, a PJRT plugin, IREE or a Python venv skip themselves by name
  when it is absent, so a machine without them still gets a green run.
- If you change a public API, run `./gradlew updateKotlinAbi` and commit the updated
  `api/*.api` files with the change. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## What a change needs

- **A test that certifies the claim**, checked against an analytic or
  cross-implementation oracle. Before submitting, revert the fix and confirm the new
  test fails; a test that passes either way certifies nothing.
- **Unsupported cases refuse loudly, by name.** An input Tlaloc cannot handle raises
  an error that says what is unsupported. It does not silently fall back.
- **Negative results are reported too.** If a measurement comes out worse than
  expected, say so in the pull request and in the docs.
- User-facing text (docs, error messages) states the fact and stops.

## Vendored code

`third-party/maestro/` is Netflix Maestro, vendored under Apache-2.0. Do not edit files
there except by the procedure in [docs/vendoring.md](docs/vendoring.md), which also
lists every file Tlaloc has changed.

## License

Tlaloc is licensed under [Apache-2.0](LICENSE). As section 5 of that license states,
any contribution you submit for inclusion is licensed under the same terms, with no
additional terms or conditions.

## Security issues

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).
