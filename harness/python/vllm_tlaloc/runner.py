"""§0.4.470 — `TlalocModelRunner`: the class the vLLM worker delegates to,
and the only place in the plugin that touches device state.

It owns three things and nothing else: the loaded artifact (H3a), the page
pool (`paging.py`), and the KV pools themselves — the actual device-side
arrays the graph reads and returns.

**It does not import vLLM.** `worker.py` adapts vLLM's call shapes onto
this class's methods; this class is drivable by anything, which is how the
certification drives it against a real exported artifact with no vLLM
installed.

§0.4.476 (H6b): execution arrives through `tlaloc_serve`, which now reaches
PJRT by ctypes on every platform that has a plugin `.so` — so on the CUDA
lane this runner drives a device without importing jax, jaxlib, torch or
numpy at all. The `"cpu"` platform still resolves to the jax ORACLE engine
(jaxlib ships no CPU PJRT plugin); `tlaloc_serve.default_engine_for` is the
single place that rule is written down.

## Decisions

* **The pools are swapped, not mutated.** `run_decode` returns new pool
  arrays (the graph's `KV_CACHE_WRITE` produces a value, because the IR is
  functional) and this class rebinds them. Buffer DONATION — making XLA
  reuse the input buffer so the swap costs nothing — is H3a's named
  deferral, still open: the manifest carries `donationPairs` and nothing
  wires them into `CompileOptions` yet. Until it does, a step copies the
  pool, and that is a PERFORMANCE fact, not a correctness one.

* **A sequence is admitted with a one-token prompt, and a longer prompt is
  refused BY NAME.** There is no prefill entry in the artifact yet (H3a's
  deferral: the contract and the ladder carry `DecodeGraphKind.PREFILL` and
  nothing exports one). A runner that quietly decoded a long prompt one
  token at a time would be "working" while doing something no serving
  system would accept, and the refusal is where the deferral becomes
  visible to whoever hits it.

* **Sampling is host-side and greedy in v1** (`batching.greedy_sample`),
  per the audit's §2.6.

* **`step()` returns logits AND the sampled tokens**, rather than sampling
  internally and dropping the logits. vLLM's sampler wants logprobs; a
  runner that has already thrown them away cannot be adapted to it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import ARTIFACT_ENV_VAR
from .batching import build_decode_call, greedy_sample, last_token_logits
from .paging import PagePool


def _import_tlaloc_serve():
    """Import the H3a loader, adding `harness/python` to the path if the
    package was not installed as a distribution. The plugin and the loader
    ship in one directory; a serving install has both on the path and this
    fallback does nothing."""
    try:
        import tlaloc_serve  # noqa: F401

        return tlaloc_serve
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import tlaloc_serve  # noqa: F401

        return tlaloc_serve


class TlalocModelRunner:
    """Executes decode steps of a Tlaloc serving artifact."""

    def __init__(self, artifact_path=None, platform: str = "cuda"):
        path = artifact_path or os.environ.get(ARTIFACT_ENV_VAR)
        if not path:
            raise ValueError(
                f"TlalocModelRunner: no serving artifact — pass artifact_path or set "
                f"${ARTIFACT_ENV_VAR} to the directory written by "
                f"io.tlaloc.maestro.serving.ServingArtifactWriter"
            )
        serve = _import_tlaloc_serve()
        self._serve = serve
        self.artifact = serve.ServingArtifact.load(path, platform=platform)
        # The artifact crossed a process boundary and possibly a machine
        # boundary. Verifying the content addresses costs one hash per
        # body, once, at load: the cheapest possible statement that what is
        # about to be compiled is what was exported.
        self.artifact.verify_bodies()
        m = self.artifact.model
        self.model = m
        self.pool = PagePool(num_blocks=m["numBlocks"], block_size=m["blockSize"])
        self.kv_pools = self.artifact.empty_pools()
        self.tokens: dict = {}

    # --- sequence lifecycle ---------------------------------------------

    def add_sequence(self, seq_id: int, prompt_token_ids) -> None:
        prompt = list(prompt_token_ids)
        if len(prompt) != 1:
            raise NotImplementedError(
                f"TlalocModelRunner.add_sequence: prompt of {len(prompt)} tokens for "
                f"sequence {seq_id}; this artifact has no PREFILL entry (the decode "
                f"ladder is the only thing exported — see docs/INFERENCE_SERVING_AUDIT.md "
                f"§5 H3a's named deferrals), so only a one-token prompt is admissible"
            )
        self.pool.add_sequence(seq_id)
        self.tokens[seq_id] = list(prompt)

    def free_sequence(self, seq_id: int) -> None:
        self.pool.free_sequence(seq_id)
        self.tokens.pop(seq_id, None)

    def sequence_tokens(self, seq_id: int) -> list:
        return list(self.tokens[seq_id])

    # --- execution -------------------------------------------------------

    def step(self, requests):
        """Run one decode step for `(seq_id, token_id)` pairs.

        Returns `(call, logits, sampled)`: the marshalled call (so a caller
        can see which bucket was chosen and which slots were written), the
        logits sliced to the real rows, and the greedily sampled token per
        row. The sampled tokens are appended to each sequence's history —
        the runner is the thing that knows what it decoded.
        """
        call = build_decode_call(self.pool, self.artifact, requests)
        logits, pools = self.artifact.run_decode(kv_pools=self.kv_pools, **call.as_kwargs())
        self.kv_pools = pools
        sampled = [greedy_sample(last_token_logits(row)) for row in logits]
        for seq_id, tok in zip(call.seq_ids, sampled):
            self.tokens[seq_id].append(int(tok))
        return call, logits, sampled

    def generate(self, seq_id: int, prompt_token_ids, max_new_tokens: int) -> list:
        """The single-request loop: admit, then decode `max_new_tokens`
        times, feeding each sampled token back in. This is what a `generate()`
        reduces to once the scheduler has nothing to schedule; a batch of one
        still goes through the same bucket selection and padding as a batch
        of many, which is deliberate — the padded rows are exercised by the
        easiest possible request rather than only by the hardest."""
        self.add_sequence(seq_id, prompt_token_ids)
        nxt = int(list(prompt_token_ids)[0])
        out = []
        for _ in range(max_new_tokens):
            _, _, sampled = self.step([(seq_id, nxt)])
            nxt = int(sampled[0])
            out.append(nxt)
        return out

    # --- what the platform reports back to vLLM --------------------------

    @property
    def max_batch(self) -> int:
        return self.artifact.batch_ladder[-1]

    @property
    def max_context(self) -> int:
        return self.artifact.context_ladder[-1]

    @property
    def block_size(self) -> int:
        return self.artifact.block_size

    @property
    def num_gpu_blocks(self) -> int:
        """vLLM asks the platform how many KV blocks exist so its block
        manager can hand them out. The artifact's pool dims fix the answer:
        a Tlaloc deployment's KV pool is a COMPILED shape, so this is a
        statement of fact and not, as it is on the torch backends, the
        result of a memory-profiling run."""
        return self.model["numBlocks"]
