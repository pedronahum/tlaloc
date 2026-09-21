#!/usr/bin/env python3
"""§0.4.474 — report what is installed in the venv that runs this script.

This script makes **no policy decisions**. It enumerates the interpreter's
installed distributions and prints them as `dist <canonical-name> <version>`
lines, plus a handful of `fact <key> <value>` lines for things a version
string cannot express (does this torch carry a CUDA runtime?). The *pins* —
which versions the repo's oracles were certified against — live on the JVM
side, in `OracleVenvIntegrityTest`, because that is where a change to them
is a reviewable, committed act rather than a thing that drifts with a
machine.

Why this exists: `~/.local/venvs/iree` is the **oracle venv**. Every
cross-language certification in this repo (safetensors parity, the PyTorch
agreement harness, every PJRT lane, the serving loader, the IREE lanes) runs
its reference side there. A `pip install` into it that quietly replaced
`torch 2.11.0+cpu` with a CUDA build, or set CUDA-13 wheels down beside
jax's CUDA-12 plugin, would not break a build — it would move every number
this repo compares against, silently. See `docs/SERVING_RUNBOOK.md` §0.1.

Stdlib only, on purpose: the canary must still run in a venv that something
has broken, so it may not import jax, torch or numpy to do its job. The one
exception is the deliberately-guarded torch probe below, which reports an
import failure as a fact instead of raising.

Exit codes follow the house convention for subprocess certifications:
  0 — the report was produced (whether or not the JVM will like it)
  2 — this machine cannot run the check at all (never used here: an
      interpreter that can run this file can always enumerate itself)
"""

from __future__ import annotations

import re
import sys


def canonical(name: str) -> str:
    """PEP 503 normalization: `jax_cuda12_plugin` and `jax-cuda12-plugin`
    are the same distribution, and which spelling `importlib.metadata`
    hands back depends on the wheel."""
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    import importlib.metadata as md

    print(f"fact python {sys.version.split()[0]}")
    print(f"fact executable {sys.executable}")

    seen: dict[str, str] = {}
    for dist in md.distributions():
        raw = dist.metadata["Name"]
        if not raw:
            continue
        seen[canonical(raw)] = dist.version or "?"
    for name in sorted(seen):
        print(f"dist {name} {seen[name]}")

    # The one probe a version string cannot answer. A CUDA torch and a CPU
    # torch can both call themselves `2.11.0` once a local tag is stripped;
    # `torch.version.cuda` is the honest discriminator. Guarded: a venv
    # whose torch is broken must still produce a report.
    try:
        import torch  # noqa: PLC0415

        print(f"fact torch.version {torch.__version__}")
        print(f"fact torch.version.cuda {torch.version.cuda!r}")
    except BaseException as exc:  # noqa: BLE001 - a report, not a crash
        print(f"fact torch.import_error {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
