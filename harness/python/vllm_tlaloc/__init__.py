"""§0.4.470 — Phase H3b: `vllm-tlaloc`, the out-of-tree vLLM platform plugin.

The shape is `docs/INFERENCE_SERVING_AUDIT.md` §1's ratified one — vLLM's
`tpu-inference` precedent, **not** `torchtpu-vllm`: vLLM's Python keeps the
scheduler, continuous batching, paged-KV bookkeeping and the API surface,
and this plugin replaces **execution** with a compiled Tlaloc StableHLO
program run through PJRT. There is no PyTorch device here, no JVM, and
nothing that calls back across the seam H3a drew: the input is an artifact
DIRECTORY (`ServingArtifactWriter`), read by `tlaloc_serve.py`.

## The layering, and why it is this way

The package is split so that **everything that can be tested without vLLM
installed is in a module that does not import vLLM**:

* `paging.py`  — the KV page pool: allocation, block tables, slot mapping.
* `batching.py` — requests → one padded, bucket-selected decode call.
* `runner.py`  — `TlalocModelRunner`: artifact + pool + sampling. Pure.
* `platform.py`, `worker.py` — the vLLM-facing classes. These import vLLM,
  and ONLY these.

That is not a tidiness preference. vLLM is a 186-package dependency
closure (see the install record in the audit); a plugin whose marshalling
arithmetic can only be exercised with all of it present is a plugin whose
marshalling arithmetic is never exercised. The split means the
bucket selection, the page allocator, the slot arithmetic and the manifest
binding are certified by a stdlib `unittest` run and, end to end, against a
REAL exported artifact executed through PJRT — while what remains
uncertified is exactly the vLLM API adaptation and nothing else.

## Registration

vLLM discovers out-of-tree platforms through the `vllm.platform_plugins`
entry-point group: each entry point is a zero-argument callable returning
the **dotted path of a `Platform` subclass**, or `None` to decline (which
is how a plugin that is installed but whose hardware is absent gets out of
the way without raising). `register()` below is that callable; the entry
point is declared in `pyproject.toml` beside this file and PINNED against
this module by `vllm_tlaloc_test.py`, so the two cannot drift.
"""

from __future__ import annotations

# The dotted path vLLM is told to import. Stated once, here, because the
# entry point in pyproject.toml, the test that pins it, and any error
# message about it must all be talking about the same string.
PLATFORM_CLASS_PATH = "vllm_tlaloc.platform.TlalocPlatform"

# The environment variable naming the serving artifact directory. A vLLM
# `--model` is a HuggingFace id or a checkpoint path, and a Tlaloc
# deployment's unit is neither: it is the directory H3a writes. REJECTED:
# overloading `--model` to mean "artifact directory when it happens to
# contain tlaloc-serving.json" — a flag that means two things depending on
# what is inside the thing it points at is a flag that will eventually
# guess wrong. The env var says it outright.
ARTIFACT_ENV_VAR = "TLALOC_SERVING_ARTIFACT"

__all__ = [
    "PLATFORM_CLASS_PATH",
    "ARTIFACT_ENV_VAR",
    "register",
]


def register() -> str:
    """The `vllm.platform_plugins` entry point.

    Returns the dotted path of the platform class. Deliberately does NOT
    import `platform.py` (and so does not import vLLM): vLLM calls every
    registered plugin's entry point during platform discovery, and a
    plugin that imports its own world just to answer "what is your class
    called" makes every other backend pay for it.
    """
    return PLATFORM_CLASS_PATH
