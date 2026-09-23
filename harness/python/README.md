# Head-to-head harness — Python references

**Status:** Phase 2 of `docs/HEAD_TO_HEAD_HARNESS_PLAN.md`. Scripts ship in the repo; user runs them post-toolchain-install.

## What this directory contains

Python reference implementations of Tlaloc's head-to-head harness benchmarks, expressed in PyTorch (and forthcoming JAX). Each script:

1. Implements each paper-benchmark primal in the framework's native API.
2. Runs `framework.compile`/`jit` + reverse-mode autodiff on each primal.
3. Times gradient evaluation over a warmup + measured iteration window.
4. Writes JSON + CSV output in the same format as Tlaloc's JVM-side harness runner.

The JVM-side aggregator (`HeadToHeadHarnessAllTest` produces `harness-results-tlaloc.csv` + `.json`) and the Python scripts produce sibling files (`harness-results-pytorch.csv`, `harness-results-jax.csv`). A future cross-framework comparison step reads all three and computes the speedup table that closes M9.

## Files

- `run_pytorch.py` — PyTorch reference (uses `torch.func.grad` + `torch.compile`).
- `run_jax.py` — JAX reference (uses `jax.grad` + `jax.jit`).
- `aggregate.py` — cross-framework aggregator (stdlib only; no PyTorch / JAX needed).
- `aggregate_test.py` — stdlib-only tests for `aggregate.py`. Run with `python3 -m unittest harness.python.aggregate_test`.
- `test_data/` — synthetic JSON fixtures used by `aggregate_test.py`.

## Pre-requisites (user-side; not auto-installed)

```bash
pip install torch>=2.1   # for run_pytorch.py
pip install jax jaxlib   # for run_jax.py
```

Tlaloc's `/loop` does not authorize toolchain installs. These scripts ship in the repo as artifacts that the user invokes when their environment has the framework installed.

## Full cross-framework workflow

Once PyTorch and JAX are installed, the full comparison runs in four commands from the project root:

```bash
# 1. Tlaloc-side: Gradle task that runs HeadToHeadHarnessMain and writes
#    benchmarks/build/harness-results-tlaloc.{json,csv}.
./gradlew :benchmarks:dumpHarnessResults

# 2. PyTorch-side: writes benchmarks/build/harness-results-pytorch.{json,csv}.
python harness/python/run_pytorch.py --output benchmarks/build/

# 3. JAX-side: writes benchmarks/build/harness-results-jax.{json,csv}.
python harness/python/run_jax.py --output benchmarks/build/

# 4. Aggregator reads all three; produces Markdown comparison table.
python harness/python/aggregate.py --input benchmarks/build/ \
    --output benchmarks/build/harness-comparison.md
```

The aggregator's Markdown output is the cross-framework comparison table.

## Usage

From the project root:

```bash
# Run PyTorch baselines, write to build/.
python harness/python/run_pytorch.py

# Run JAX baselines, same output dir.
python harness/python/run_jax.py

# Run with custom warmup/measured counts.
python harness/python/run_pytorch.py --warmup 500 --measured 2000
python harness/python/run_jax.py --warmup 500 --measured 2000

# Run without compilation (eager-mode timings).
python harness/python/run_pytorch.py --no-compile
python harness/python/run_jax.py --no-jit

# Output goes to a custom directory.
python harness/python/run_pytorch.py --output /tmp/harness-out/
python harness/python/run_jax.py --output /tmp/harness-out/
```

Output:
- `build/harness-results-pytorch.json` — full numerical baseline (forward + gradients + timings) for every benchmark.
- `build/harness-results-pytorch.csv` — paper-style CSV with `benchmark,framework=pytorch-compile,n_iterations,median_ns,min_ns,p99_ns` per row.
- `build/harness-results-jax.json` — same shape, JAX values.
- `build/harness-results-jax.csv` — paper-style CSV with `framework=jax-jit` rows.

## Benchmarks

Five paper benchmarks ported (mirroring Tlaloc's `:benchmarks/HeadToHeadHarness` inhabitants):

| Benchmark | Tlaloc inhabitant | Inputs |
|---|---|---|
| BGDHyperOpt | `BgdHyperOptHarness` | r=0.01, Sxy=111.2, Sx2=55.0, M=5.0, K=3 |
| HookeanSpring | `HookeanSpringHarness` | pInit=1, vInit=0, kSpring=1, N=10, dt=0.1 |
| Brachistochrone | `BrachistochroneHarness` | y=0.5, N=5 |
| HMC | `HmcLogisticRegressionHarness` | β=(0.5, 0.3), n=4, d=2 hardcoded |
| CartPole | `CartPolePhase1Harness` | (at, x0, x1, x2, x3) = (0.5, 0, 0.1, 0.05, 0.02) |

QWOP avatar-step (Tlaloc's synthetic full-pipeline test) is not included — it has no paper number to compare against.

## Numerical agreement contract

For each benchmark, Tlaloc's forward and gradient values should match the PyTorch reference within:
- **Forward**: 1e-3 absolute tolerance (or 1% relative).
- **Gradient**: 5e-3 relative tolerance + 1e-3 absolute floor.

These are the "f32-tolerance numerical match" criterion of milestone M9 in `DIFFKTX_SPEC.md`. Larger discrepancies indicate a real bug in either Tlaloc's AD pipeline or the Python primal port.

## Why Phase 2 ships pre-toolchain-install

The /loop's no-toolchain-install rule blocks running these scripts during /loop firings, but writing them is not blocked. Pre-shipping the scripts means:

1. When the user installs PyTorch+JAX, they can immediately run a single command and produce the cross-framework CSV/JSON dumps.
2. The Tlaloc-side primals serve as the reference for the Python ports; any structural divergence (e.g., wrong constant, wrong loop bound) is caught before the toolchain is even installed.
3. Subsequent /loop firings post-install can focus on **comparison** (read the JSON files, compute the speedup table, write the comparison table) rather than re-implementing the primals.

## Cross-framework aggregator

`aggregate.py` produces the unified comparison Markdown table. It only requires Python stdlib — runs even if PyTorch+JAX aren't installed.

```bash
# Generate the comparison Markdown to stdout.
python harness/python/aggregate.py

# Write to a file AND stdout.
python harness/python/aggregate.py --output build/harness-comparison.md

# Strict mode: exit non-zero if any framework's gradient disagrees with
# Tlaloc's beyond f32 tolerance (1e-3 abs / 5e-3 rel).
python harness/python/aggregate.py --strict
```

The aggregator reads `build/harness-results-{tlaloc,pytorch,jax}.json` and produces a Markdown table covering:
- **Throughput**: median nanoseconds per gradient eval, plus speedup ratios `Tlaloc/torch×` and `Tlaloc/jax×`.
- **Paper figures**: the OOPSLA 2021 paper's reported speedups for benchmarks where they're known (Brachistochrone, HookeanSpring, HMC, CartPole). Cells show `—` for benchmarks not in the paper's primary tables.
- **M9 verdict**: `✓` if Tlaloc's measured speedup over `torch.compile` is within 20% of the paper's range; `✗` otherwise; `—` if not measured or not in paper.
- **Numerical agreement**: per-benchmark forward + gradient values across frameworks, with f32-tolerance flagging.

The Tlaloc JSON is required (the JVM harness must have run); PyTorch and JAX JSONs are optional. If only Tlaloc results exist, the table shows "—" in the cross-framework cells.

## Cross-framework comparison entry

When `harness-results-tlaloc.json`, `harness-results-pytorch.json`, and `harness-results-jax.json` all exist, the `aggregate.py` output above is the comparison table. The format mirrors `docs/HEAD_TO_HEAD_HARNESS_PLAN.md` Phase 3:

```
| Benchmark        | Paper (× over torch.compile) | Tlaloc actual | Pass M9? |
|------------------|------------------------------|---------------|----------|
| Brachistochrone  | 4-11×                        | <measured>    | ✓/✗      |
| HookeanSpring    | 1.05-1.12×                   | <measured>    | ✓/✗      |
| HMC              | 2.3-3.6×                     | <measured>    | ✓/✗      |
| CartPole         | 1.22-4.42×                   | <measured>    | ✓/✗      |
| BGDHyperOpt      | (paper number)               | <measured>    | ✓/✗      |
```

Acceptance (milestone M9 in `DIFFKTX_SPEC.md`): within 20% of paper's figures, >3× over `torch.compile` on at least three of six, f32-tolerance numerical match.
