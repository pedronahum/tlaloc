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
        # §0.4.477 — set by `step`; None until one has run (see `step`).
        self.last_call = None
        self.last_logits = None

    # --- sequence lifecycle ---------------------------------------------

    def add_sequence(self, seq_id: int, prompt_token_ids) -> None:
        """Admit a sequence and consume all but the LAST prompt token.

        §0.4.492 (H3c-4b) — this used to refuse any prompt but a one-token
        one, and that refusal is what a real `LLM.generate()` hit on its
        first step. There is still no PREFILL entry in the artifact (H1a's
        ragged form is the open deferral), so the prompt is consumed the way
        `run_llama_generate.py` has consumed it since §0.4.480 and
        §0.4.479's parity lane before that: as `len(prompt) - 1` single-token
        decode steps, each writing one KV slot, feeding position `i` at
        `seqLens = i + 1`. Attention is causal, so those steps compute
        exactly the KV a fused prefill would; the cost is N launches instead
        of one, which makes this a PERFORMANCE deferral and not a
        correctness one.

        The LAST prompt token is deliberately left unconsumed: it is the
        token the caller's first `step` feeds, and the logits it produces
        are the first sampled token. Consuming it here would make the
        admission produce a token, and the runner would be one step ahead of
        whatever scheduler is driving it for the rest of the sequence.

        REJECTED: sampling during the walk and discarding the results
        silently. The walk's logits ARE discarded — the prompt already says
        what comes next — but `_prefill_step` says so by not sampling at
        all, rather than by sampling into a variable nobody reads.
        """
        prompt = list(prompt_token_ids)
        if not prompt:
            raise ValueError(
                f"TlalocModelRunner.add_sequence: sequence {seq_id} has an empty "
                f"prompt; there is nothing to condition on and nothing to feed"
            )
        self.pool.add_sequence(seq_id)
        self.tokens[seq_id] = list(prompt)
        for tok in prompt[:-1]:
            self._prefill_step(seq_id, tok)

    def free_sequence(self, seq_id: int) -> None:
        self.pool.free_sequence(seq_id)
        self.tokens.pop(seq_id, None)

    def sequence_tokens(self, seq_id: int) -> list:
        return list(self.tokens[seq_id])

    # --- execution -------------------------------------------------------

    def _prefill_step(self, seq_id: int, token_id: int) -> None:
        """One prompt token through the decode graph: write its KV, keep the
        pools, sample nothing and record nothing.

        Not `step`, because `step` maintains the histories and the
        `last_call`/`last_logits` the live lane measures, and a prompt walk
        must leave both describing the step a CALLER made. One sequence at a
        time on purpose: batching two sequences' prompt walks together would
        need them to be the same length, and the ladder's padding rules are
        written for the decode shape.
        """
        call = build_decode_call(self.pool, self.artifact, [(seq_id, token_id)])
        _, pools = self.artifact.run_decode(kv_pools=self.kv_pools, **call.as_kwargs())
        self.kv_pools = pools

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
        # §0.4.477 (H7) — the last step's call and logits, kept so the
        # WORKER lane can be measured. `TlalocWorker.execute_model` returns
        # vLLM's `ModelRunnerOutput`, which carries sampled TOKENS and no
        # logits; without these two attributes the live vLLM lane could only
        # certify that the adapter agreed on the tokens, and agreeing on a
        # greedy argmax is a much weaker statement than agreeing on the
        # vector it was taken from. Two references, overwritten each step —
        # not a history, which would be a leak in a server.
        self.last_call = call
        self.last_logits = logits
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
        # §0.4.492: the LAST prompt token, because `add_sequence` consumed
        # every one before it. At a one-token prompt these are the same
        # token, which is why this read was right for two slices.
        nxt = int(list(prompt_token_ids)[-1])
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
