"""§0.4.470 — requests → one padded, bucket-selected decode call.

The marshalling step: a list of "these sequences each want one more token"
becomes the exact operand set `tlaloc_serve.ServingArtifact.run_decode`
takes, at a bucket the artifact actually compiled. Standard library only —
no vLLM, no numpy, no jax — because this arithmetic is where an off-by-one
lives and it must be testable without any of them.

## Decisions

* **The bucket is selected BEFORE the block tables are built**, because the
  table's width is `maxBlocksPerSeq` and that is a property of the entry,
  not of the request. Building tables first and padding them afterwards
  would work right up until a request's own page count exceeded the
  bucket's width, at which point the padding loop would silently truncate
  a sequence's history.

* **`append_token` runs before the call is built, not after it returns.**
  The slot a token writes and the `seq_len` the kernel reads must describe
  the SAME step, and the graph writes the KV for this token during this
  step — so the pool's length must already include it. Deferring the
  advance until after execution would make every step read one token
  short of what it just wrote.

* **A sequence may appear at most once in a batch, refused by name.** One
  decode step writes one slot per row; two rows for one sequence means two
  writes to one sequence's cache in one step, in an order the graph does
  not define.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeCall:
    """Exactly the operands one `run_decode` takes, plus the bucket it was
    built for — carried so a caller (and a certification) can see the
    padding decision rather than infer it."""

    seq_ids: tuple
    token_ids: tuple
    positions: tuple
    block_tables: tuple
    seq_lens: tuple
    slot_mapping: tuple
    context: int
    bucket: tuple

    def as_kwargs(self) -> dict:
        return {
            "token_ids": list(self.token_ids),
            "positions": list(self.positions),
            "block_tables": [list(t) for t in self.block_tables],
            "seq_lens": list(self.seq_lens),
            "slot_mapping": list(self.slot_mapping),
            "context": self.context,
        }


def build_decode_call(pool, artifact, requests) -> DecodeCall:
    """Advance `requests` — pairs of `(seq_id, token_id)` — by one step.

    Mutates `pool` (that is the point: the step happens here) and returns
    the call. `artifact` is anything with `select_bucket` and `entry_for`,
    i.e. a `tlaloc_serve.ServingArtifact`; taking it structurally keeps
    this module importable with no jax on the path.
    """
    reqs = list(requests)
    if not reqs:
        raise ValueError("build_decode_call: empty batch — a decode step with no rows")

    seen = set()
    for seq_id, _ in reqs:
        if seq_id in seen:
            raise ValueError(
                f"build_decode_call: sequence {seq_id} appears twice in one batch; a "
                f"decode step writes exactly one KV slot per sequence and two rows for "
                f"one sequence is two writes in an undefined order"
            )
        seen.add(seq_id)
        if seq_id not in pool:
            raise KeyError(
                f"build_decode_call: sequence {seq_id} is not in the page pool; "
                f"add_sequence must run before a token can be decoded for it"
            )

    token_ids, positions, seq_lens, slots = [], [], [], []
    for seq_id, token_id in reqs:
        s = pool.sequence(seq_id)
        positions.append(s.next_position)
        slots.append(pool.append_token(seq_id))
        token_ids.append(int(token_id))
        seq_lens.append(s.length)

    context = max(seq_lens)
    bucket = artifact.select_bucket(len(reqs), context)
    entry = artifact.entry_for("decode", bucket[0], bucket[1])
    width = entry.max_blocks_per_seq
    tables = [pool.block_table(seq_id, width) for seq_id, _ in reqs]

    return DecodeCall(
        seq_ids=tuple(seq_id for seq_id, _ in reqs),
        token_ids=tuple(token_ids),
        positions=tuple(positions),
        block_tables=tuple(tuple(t) for t in tables),
        seq_lens=tuple(seq_lens),
        slot_mapping=tuple(slots),
        context=context,
        bucket=tuple(bucket),
    )


def decode_requests_from_scheduler_output(scheduler_output, last_token_of) -> list:
    """vLLM's `SchedulerOutput` → the `(seq_id, token_id)` pairs this step
    decodes, plus the ids to admit and the ids to free.

    Returns `(new_requests, decode_requests, finished_ids)`.

    **Duck-typed on purpose.** vLLM's scheduler-output dataclass is
    internal and has moved between versions (`scheduled_new_reqs` /
    `scheduled_cached_reqs` / `num_scheduled_tokens` / `finished_req_ids`
    are the v1 names). Taking the object structurally means this function
    is unit-testable against a stand-in with no vLLM installed — which is
    the ONLY reason the marshalling half of the plugin has coverage at all
    (see the module docstring in `vllm_tlaloc/__init__.py`). It also means
    a vLLM rename shows up here as a named `AttributeError` at the seam
    rather than as a wrong batch.

    `last_token_of` maps a request id to the token that request most
    recently produced — the runner's own history, because the scheduler
    output carries the *counts* of what to run and not the token ids of an
    ongoing generation. **It is consulted for CACHED requests only.**

    ## §0.4.477 (H7) — the ordering bug the live vLLM lane found

    Until H7 this function asked `last_token_of` for EVERY scheduled
    request, new ones included. That is wrong, and wrong in a way no test
    here could see: `TlalocWorker.execute_model` calls this function FIRST
    and admits the new sequences with `add_sequence` SECOND, so on the step
    a request arrives, `last_token_of(rid)` asks the runner for the history
    of a sequence it has not been told about yet — `KeyError` on the very
    first step of every server that has ever started. The unit lane passed
    because its stand-in was a dict literal that happened to have an answer
    for the new id; a `dict` is more forgiving than a runner, and the gap
    between them is exactly the gap between "written" and "certified". The
    live lane (`run_vllm_live_check.py`, vLLM 0.29.0) hit it on step 0.

    The fix is not to reorder the worker. A NEW request's feed token is a
    fact the scheduler output already carries — it is the last token of the
    prompt — so asking the runner for it was always a detour through state
    that need not exist yet. Reordering the worker would have made the
    function *work*; taking the token from the prompt makes the function
    not need the ordering at all, and leaves `last_token_of` with one
    meaning instead of two.

    ## §0.4.492 (H3c-4b) — A WHOLE PROMPT IS NOW ADMISSIBLE, A CHUNK IS NOT

    Until this slice any request scheduled for more than one token was
    refused. That refusal is what a real `LLM.generate()` hit on its first
    step: vLLM schedules a new request for `len(prompt)` tokens, and a
    six-token prompt is six.

    The distinction the refusal was missing is between a PREFILL and a
    CHUNK. A new request scheduled for exactly its whole prompt is a
    prefill, and this artifact CAN serve one — as `len(prompt) - 1` decode
    steps followed by the ordinary batched step, which is precisely what
    `run_llama_generate.py` has done since §0.4.480 and what §0.4.479's
    parity lane did before it. Causal attention makes those N single-token
    steps compute the same KV and the same final logits a fused prefill
    would; what they cost is N kernel launches instead of one, so the
    deferral is a PERFORMANCE deferral and is named as one. A request
    scheduled for FEWER tokens than it has left is a chunk, needs
    resumption state this runner does not keep, and is still refused by
    name — as is any cached request scheduled for more than one token
    (a resumed chunk, or speculative decoding).
    """
    counts = dict(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})

    new_requests = []
    for r in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        rid = getattr(r, "req_id")
        prompt = list(getattr(r, "prompt_token_ids", []) or [])
        computed = getattr(r, "num_computed_tokens", 0) or 0
        if computed:
            raise NotImplementedError(
                f"request {rid} arrives with {computed} tokens already computed "
                f"(prefix cache hit); this runner's KV pool is its own and holds no "
                f"cross-request prefix, so it cannot start a sequence in the middle. "
                f"Run with --no-enable-prefix-caching"
            )
        new_requests.append((rid, prompt))

    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    cached_ids = []
    if cached is not None:
        ids = getattr(cached, "req_ids", None)
        cached_ids = list(ids) if ids is not None else [getattr(r, "req_id") for r in cached]

    # `feed_of` is the one place the two kinds of request differ: a new
    # request's token is in the scheduler output (the prompt's last token);
    # a cached one's is in the runner's history. See the §0.4.477 note above
    # for why this is not a reordering.
    def _prompt_feed(rid, prompt):
        if not prompt:
            raise ValueError(
                f"request {rid} arrived with an empty prompt; a decode step needs a "
                f"token to feed and the scheduler output carries none — refused here "
                f"rather than read off the end of the list"
            )
        return prompt[-1]

    feed_of = [(rid, (lambda r=rid, p=prompt: _prompt_feed(r, p))) for rid, prompt in new_requests]
    feed_of += [(rid, (lambda r=rid: last_token_of(r))) for rid in cached_ids]

    # How many tokens each request may legitimately be scheduled for this
    # step: its whole prompt if it is arriving, exactly one if it is not.
    admissible = {rid: max(len(prompt), 1) for rid, prompt in new_requests}

    decode_requests = []
    for rid, feed in feed_of:
        n = counts.get(rid, 1)
        want = admissible.get(rid, 1)
        if n != want:
            raise NotImplementedError(
                f"request {rid} is scheduled for {n} tokens this step but {want} "
                f"{'is' if want == 1 else 'are'} admissible; this artifact has only "
                f"DECODE entries, so a whole prompt is served as N single-token steps "
                f"and anything else — a chunk, a resumed chunk, a "
                f"speculative draft — needs a prefill entry, which is a named "
                f"deferral. Run with --no-enable-chunked-prefill"
            )
        decode_requests.append((rid, feed()))

    finished = list(getattr(scheduler_output, "finished_req_ids", []) or [])
    return new_requests, decode_requests, finished


def last_token_logits(row):
    """The vocabulary vector to sample from, out of one sequence's logits.

    The decode contract's logits are `[batch, tokensPerSeq, vocab]` — the
    token axis is kept even at `tokensPerSeq == 1`, because the SAME
    contract describes a prefill entry (H3a's deferral) where it is not 1.
    So a row is `[tokensPerSeq, vocab]` and the token to sample is the LAST
    one: the position the step just wrote.

    REJECTED: reshaping to `(-1,)` and assuming the token axis is 1 — it is
    correct today and it is exactly the assumption a prefill entry breaks,
    silently, by sampling from a concatenation of every position's logits.

    Duck-typed over numpy arrays and nested lists so the rule is testable
    with neither numpy nor a device present.
    """
    if hasattr(row, "shape"):
        while len(row.shape) > 1:
            row = row[-1]
        return row
    while len(row) and isinstance(row[0], (list, tuple)):
        row = row[-1]
    return row


def greedy_sample(logits_row) -> int:
    """Argmax, ties to the lowest index.

    Sampling is HOST-side in v1 per the audit — the graph returns logits and
    stops. REJECTED (named deferral): emitting temperature / top-k / top-p
    into the decode graph. They are cheap on device and the reason to defer
    them is not cost: each one adds an operand to the compiled signature, so
    a deployment would compile a different program per sampling policy, and
    the bucket ladder is already a cross product we intend to keep small.
    """
    best_i, best_v = 0, None
    for i, v in enumerate(logits_row):
        fv = float(v)
        if best_v is None or fv > best_v:
            best_i, best_v = i, fv
    if best_v is None:
        raise ValueError("greedy_sample: empty logits row")
    return best_i
