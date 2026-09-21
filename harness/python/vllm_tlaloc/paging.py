"""§0.4.470 — the KV page pool: who owns which page, and which slot a token
writes.

This is the bookkeeping half of paged attention, and per the audit it stays
in **Python** — vLLM owns the scheduler and the block manager, and this
module is the shape that bookkeeping takes when the plugin is driven
directly (by the certification, or by a serving loop that is not vLLM's).
When vLLM drives, `TlalocModelRunner` is handed vLLM's own block tables
instead and this allocator stands aside; it is the reference semantics for
what those tables must mean, and the thing the certification can drive
without 186 packages present.

Imports nothing but the standard library, on purpose.

## Decisions

* **Page 0 is RESERVED as the scratch page and never allocated.**
  `DecodePadding.PADDING_BLOCK` is 0, so every padded row in a bucket reads
  page 0. Handing page 0 to a live sequence makes an inert row read live
  KV. That is harmless *today* — the padded row's logits are sliced off
  before they leave `tlaloc_serve.run_decode` — and it stops being harmless
  the moment anything poisons, profiles or checksums the scratch page, or
  the day a bucket's padded rows are asked to prove they wrote nothing.
  One page out of `numBlocks` is the cheap side of that trade.
  REJECTED: making the padding block per-request so the allocator can use
  every page — the padding convention is a WIRE constant shared with the
  Kotlin exporter (`DecodePadding`), and a wire constant that varies per
  request is not a constant.

* **The free list is kept SORTED, so allocation is reproducible.**
  REJECTED: a LIFO stack (marginally faster, and the usual choice) — the
  certification compares a plugin-driven run against an independently
  computed expectation, which requires that "which page does the third
  sequence get" has one answer and not one answer per interpreter run.

* **`append_token` allocates at most one page and only on a block
  boundary**, which is the decode invariant: one token per sequence per
  step. A prefill that adds many tokens at once is `reserve()`, kept
  separate so that the decode path cannot silently grow by more than a
  page and hide a bookkeeping error inside a loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Mirrors of io.tlaloc.ir.inference.DecodePadding — restated here with the
# same reasoning as tlaloc_serve.py's copy: this is a WIRE convention
# between two processes, and `check_padding_constants` pins the copies.
SCRATCH_BLOCK = 0
DROPPED_SLOT = -1


class PagePoolExhausted(RuntimeError):
    """No free page. Raised BY NAME rather than returning None, because the
    caller's only correct response is to preempt or queue a sequence, and a
    None that gets used as a page id is a write into page ... nothing."""


@dataclass
class SequencePages:
    """One sequence's page ownership and its position in the stream."""

    seq_id: int
    block_size: int
    blocks: list = field(default_factory=list)
    length: int = 0

    @property
    def next_position(self) -> int:
        """The position index the NEXT token will occupy (0-based)."""
        return self.length

    def slot_for_next(self) -> int:
        """Flat slot index for the next token, or raise if unbacked."""
        page_index = self.length // self.block_size
        if page_index >= len(self.blocks):
            raise RuntimeError(
                f"sequence {self.seq_id}: token at position {self.length} needs page "
                f"{page_index} but only {len(self.blocks)} are allocated — "
                f"append_token must allocate before it computes a slot"
            )
        return self.blocks[page_index] * self.block_size + (self.length % self.block_size)


class PagePool:
    """A fixed pool of KV pages, handed out to sequences.

    The layout the pages live in is the artifact's
    (`kvPoolDims = [numBlocks, blockSize, numKvHeads, headDim]`); this class
    knows only the first two numbers, because page ownership does not depend
    on how wide a page is.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 2:
            raise ValueError(
                f"PagePool: num_blocks={num_blocks}; page {SCRATCH_BLOCK} is reserved "
                f"as the padding scratch page, so a usable pool needs at least 2"
            )
        if block_size < 1:
            raise ValueError(f"PagePool: block_size={block_size} must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(SCRATCH_BLOCK + 1, num_blocks))
        self._seqs: dict = {}

    # --- inspection -----------------------------------------------------

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def capacity(self) -> int:
        """Allocatable pages — `num_blocks` minus the reserved scratch page."""
        return self.num_blocks - 1

    def sequence(self, seq_id: int) -> SequencePages:
        try:
            return self._seqs[seq_id]
        except KeyError:
            raise KeyError(f"no sequence {seq_id} in this pool") from None

    def __contains__(self, seq_id: object) -> bool:
        return seq_id in self._seqs

    # --- lifecycle ------------------------------------------------------

    def add_sequence(self, seq_id: int) -> SequencePages:
        if seq_id in self._seqs:
            raise ValueError(f"sequence {seq_id} is already in this pool")
        s = SequencePages(seq_id=seq_id, block_size=self.block_size)
        self._seqs[seq_id] = s
        return s

    def free_sequence(self, seq_id: int) -> None:
        """Return a sequence's pages. The free list stays sorted, so the
        pool's state after free-then-allocate depends on WHICH pages came
        back and not on the order they were returned in."""
        s = self._seqs.pop(seq_id, None)
        if s is None:
            raise KeyError(f"free_sequence: no sequence {seq_id}")
        self._free.extend(s.blocks)
        self._free.sort()

    def _take(self) -> int:
        if not self._free:
            raise PagePoolExhausted(
                f"KV page pool exhausted: all {self.capacity} allocatable pages of "
                f"{self.num_blocks} are held (page {SCRATCH_BLOCK} is the reserved "
                f"padding scratch page); the scheduler must preempt or queue"
            )
        return self._free.pop(0)

    def reserve(self, seq_id: int, num_tokens: int) -> SequencePages:
        """Back `num_tokens` tokens with pages, WITHOUT advancing `length`.

        This is the prefill shape: the tokens exist and their pages must
        exist before they are written. Kept separate from `append_token` so
        that the decode path is structurally incapable of growing by more
        than one page per step.
        """
        s = self.sequence(seq_id)
        need = (num_tokens + self.block_size - 1) // self.block_size
        while len(s.blocks) < need:
            s.blocks.append(self._take())
        return s

    def append_token(self, seq_id: int) -> int:
        """Advance one decode step. Returns the slot the token writes.

        Allocates at most ONE page, and only when the new token starts a
        page. The slot is computed BEFORE `length` advances, because the
        token being placed is the one at the current length.
        """
        s = self.sequence(seq_id)
        page_index = s.length // self.block_size
        if page_index >= len(s.blocks):
            s.blocks.append(self._take())
        slot = s.slot_for_next()
        s.length += 1
        return slot

    # --- the operands a decode step needs -------------------------------

    def block_table(self, seq_id: int, width: int) -> list:
        """The sequence's page list, padded to `width` with the scratch page.

        Padding a table with the SCRATCH page and not with the sequence's
        own last page matters: a table entry past `seq_len` is read by the
        kernel's bounds arithmetic on some lowerings, and the scratch page
        is the one place where a stray read means nothing.
        """
        s = self.sequence(seq_id)
        if len(s.blocks) > width:
            raise ValueError(
                f"sequence {seq_id} holds {len(s.blocks)} pages but the selected "
                f"bucket's block table is {width} wide — the request needs a longer "
                f"context bucket than was chosen"
            )
        return list(s.blocks) + [SCRATCH_BLOCK] * (width - len(s.blocks))
