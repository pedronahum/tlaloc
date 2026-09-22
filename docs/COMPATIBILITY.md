# Compatibility policy

**Status: this is a promise, not a design note.** It is the written form of the
one line the README has carried for months — "APIs change without deprecation
cycles" — and it exists so that line cannot be read more generously than it was
meant.

Current version: **`0.1.0-alpha01`**. Nothing is on Maven Central yet; you build
from source and consume from `mavenLocal` (see
[GETTING_STARTED.md](GETTING_STARTED.md)).

## What "alpha" means here

`0.1.0-alpha01` means: **the engine is certified, the surface is not settled.**

Those are different claims and the repository keeps them apart on purpose. The
2,345 automated tests at HEAD pin behaviour — gradients against analytic and
cross-implementation oracles, a real TinyLlama matching HuggingFace token for
token, RNG bit-exact against JAX's threefry stream. None of that pins *names*. A
function whose result is certified correct may still be renamed, moved to another
module, or given a different parameter order in `0.1.0-alpha02`.

Version numbers before `1.0.0` carry no compatibility guarantee under SemVer
§4, and Tlaloc takes that literally rather than pretending the minor number
means something it does not.

## What may break, without a deprecation cycle

Any of these may change in any alpha release:

- **Any public API in any module.** Names, signatures, parameter order, default
  arguments, nullability, type parameters, receiver vs. argument.
- **Module boundaries and coordinates.** A type may move between `:core`, `:ir`,
  `:autograd` and `:nn`; a module may be split, merged or renamed.
- **The named-axis and rank type encoding.** `Rank2<Sym, Sym>`, `DTensor`'s type
  parameters and the shape-error diagnostics are the newest and least settled
  part of the surface.
- **The compiler plugin's flags and diagnostics.** Flag names, diagnostic text,
  diagnostic severities and the `dumpGradSource` output format.
- **The serving artifact's on-disk layout.** The manifest schema, file names and
  StableHLO body organisation. Re-export from the same Tlaloc version you serve
  with.
- **The KPTX kernel registry and its claiming rules.** Nothing there is
  registered by default and the opt-in surface is explicitly experimental.
- **Emitted StableHLO.** Op choice and structure may change whenever XLA gets
  faster at something else. `docs/TLALOC_EMIT_CONTRACT.md` describes what we emit
  today, not what we will emit.
- **Test counts, benchmark numbers and the modules' internal layout.** These are
  not API at all.

## What will not break inside the alpha series

These are the commitments. They are short on purpose — a long list here would be
a list of things this policy could not keep.

- **The group id stays `io.tlaloc`.** Artifact ids may gain or lose modules, but
  a coordinate that exists will not be re-pointed at different code under the
  same version.
- **A published version is immutable.** `0.1.0-alphaNN` is never re-published
  with different bytes. If it is wrong, the fix is `alphaNN+1`.
- **No silent degradation, ever.** This is a house rule, not a version policy:
  an unsupported dtype, shape, device or op refuses loudly and by name. An alpha
  may remove a capability; it will not quietly approximate one.
- **Certified stays certified.** A behaviour that a test pins does not become
  wrong without the test changing in the same commit, and a capability's mark in
  the README's Maturity table does not get upgraded without the test that earns
  it.
- **Licensing does not change retroactively.** Apache-2.0 for everything Tlaloc
  publishes; a change would apply to later versions only.

## What a consumer should do about it

- **Pin an exact version.** Never a range, never `latest.release`. There are no
  compatible-upgrade guarantees to lean on.
- **Expect to read the changelog on every bump.** [CHANGELOG.md](../CHANGELOG.md)
  records breaking changes per version; there is no deprecation period in which
  both spellings work.
- **Keep the compiler-plugin version equal to the library version.** The plugin
  synthesizes calls into `:core`/`:ir`; mixing versions is unsupported and is not
  checked for you today (see [ALPHA_PLAN.md](ALPHA_PLAN.md) — this is a named
  gap, not an oversight).
- **Treat the toolchain as pinned too.** JDK 25 and Kotlin 2.3.20. A K2 compiler
  plugin binds to compiler internals; a different Kotlin version is not expected
  to work and is not tested.

## When this policy changes

At `1.0.0`, and not before. At that point SemVer applies without carve-outs and
this document is replaced by the shorter one that says so. Until then, every
alpha release is free to break anything in the "may break" list, and is required
to say in [CHANGELOG.md](../CHANGELOG.md) that it did.
