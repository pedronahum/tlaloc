"""§0.4.470 — Phase H3b: the vLLM-free half of the plugin, tested.

Standard library only: no vLLM, no jax, no numpy. That constraint is the
point. vLLM's dependency closure is 186 packages (audit §5, H3b); a plugin
whose page arithmetic could only be exercised with all of them installed
would be a plugin whose page arithmetic is never exercised. Everything here
is the code that decides WHICH page a token writes to, WHICH bucket a batch
lands in, and WHAT the entry point is called — i.e. everything an
off-by-one can hide in.

Run directly:

    python3 -m unittest harness.python.vllm_tlaloc_test

or through the certification, which drives this file as a subprocess so the
count lands in the suite (`VllmPluginContractTest`).
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vllm_tlaloc  # noqa: E402
from vllm_tlaloc.batching import (  # noqa: E402
    build_decode_call,
    decode_requests_from_scheduler_output,
    greedy_sample,
    last_token_logits,
)
from vllm_tlaloc import kv_layout  # noqa: E402
from vllm_tlaloc.paging import PagePool, PagePoolExhausted, SCRATCH_BLOCK  # noqa: E402

HERE = Path(__file__).resolve().parent


# --- a stand-in artifact: the two methods batching actually uses ---------


class FakeEntry:
    def __init__(self, mbs):
        self.max_blocks_per_seq = mbs


class FakeArtifact:
    """Mirrors the reference artifact's ladder: batches {1,2,4} × contexts
    {2,4} at blockSize 2."""

    batch_ladder = [1, 2, 4]
    context_ladder = [2, 4]
    block_size = 2

    def select_bucket(self, batch, context):
        b = next((x for x in self.batch_ladder if x >= batch), None)
        c = next((x for x in self.context_ladder if x >= context), None)
        if b is None or c is None:
            raise ValueError(f"over cap: ({batch},{context})")
        return b, c

    def entry_for(self, kind, batch, context):
        return FakeEntry(context // self.block_size)


class PagePoolTest(unittest.TestCase):
    def test_page_zero_is_reserved_as_the_scratch_page(self):
        p = PagePool(num_blocks=6, block_size=2)
        self.assertEqual(5, p.capacity, "one page of six is the padding scratch page")
        p.add_sequence(0)
        for _ in range(5 * 2):  # fill every allocatable page
            p.append_token(0)
        self.assertNotIn(SCRATCH_BLOCK, p.sequence(0).blocks)

    def test_a_pool_with_one_page_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            PagePool(num_blocks=1, block_size=2)

    def test_slots_walk_a_page_then_take_the_next(self):
        p = PagePool(num_blocks=4, block_size=2)
        p.add_sequence(7)
        # pages 1,2,3 are allocatable; the first token takes page 1.
        self.assertEqual(2, p.append_token(7))  # page 1, offset 0
        self.assertEqual(3, p.append_token(7))  # page 1, offset 1
        self.assertEqual(4, p.append_token(7))  # page 2, offset 0 — new page
        self.assertEqual([1, 2], p.sequence(7).blocks)
        self.assertEqual(3, p.sequence(7).length)

    def test_position_is_the_slot_being_written_not_the_one_after(self):
        p = PagePool(num_blocks=4, block_size=2)
        s = p.add_sequence(0)
        self.assertEqual(0, s.next_position)
        p.append_token(0)
        self.assertEqual(1, s.next_position)

    def test_exhaustion_is_refused_by_name(self):
        p = PagePool(num_blocks=3, block_size=1)  # pages 1,2 allocatable
        p.add_sequence(0)
        p.append_token(0)
        p.append_token(0)
        with self.assertRaises(PagePoolExhausted):
            p.append_token(0)

    def test_freeing_returns_pages_and_the_free_list_stays_sorted(self):
        p = PagePool(num_blocks=5, block_size=1)  # 1..4
        for sid in (0, 1, 2):
            p.add_sequence(sid)
            p.append_token(sid)
        self.assertEqual([1], p.sequence(0).blocks)
        self.assertEqual([3], p.sequence(2).blocks)
        p.free_sequence(0)  # returns page 1, which is BELOW the free head (4)
        p.add_sequence(9)
        self.assertEqual(
            1, p.append_token(9) // 1,
            "the newcomer must get page 1 back: a sorted free list makes allocation "
            "reproducible, which a certification comparing against an independently "
            "computed expectation depends on",
        )

    def test_reserve_backs_many_tokens_without_advancing_length(self):
        p = PagePool(num_blocks=8, block_size=2)
        p.add_sequence(0)
        s = p.reserve(0, 5)
        self.assertEqual(3, len(s.blocks))
        self.assertEqual(0, s.length, "reserve backs tokens; it does not decode them")

    def test_block_table_pads_with_the_scratch_page_and_refuses_overflow(self):
        p = PagePool(num_blocks=6, block_size=2)
        p.add_sequence(0)
        p.append_token(0)
        self.assertEqual([1, SCRATCH_BLOCK, SCRATCH_BLOCK], p.block_table(0, 3))
        p.append_token(0)
        p.append_token(0)  # a second page
        with self.assertRaisesRegex(ValueError, "longer context bucket"):
            p.block_table(0, 1)

    def test_a_slot_cannot_be_computed_for_an_unbacked_position(self):
        p = PagePool(num_blocks=4, block_size=2)
        s = p.add_sequence(0)
        with self.assertRaisesRegex(RuntimeError, "append_token must allocate"):
            s.slot_for_next()


class BuildDecodeCallTest(unittest.TestCase):
    def setUp(self):
        self.art = FakeArtifact()
        self.pool = PagePool(num_blocks=6, block_size=2)
        for sid in (0, 1, 2):
            self.pool.add_sequence(sid)

    def test_three_sequences_land_in_the_four_by_two_bucket(self):
        call = build_decode_call(self.pool, self.art, [(0, 3), (1, 1), (2, 4)])
        self.assertEqual((4, 2), call.bucket, "a batch of 3 rounds up to 4; context 1 to 2")
        self.assertEqual((0, 0, 0), call.positions, "first token of each sequence")
        self.assertEqual((1, 1, 1), call.seq_lens, "seq_len INCLUDES the token being written")
        self.assertEqual((2, 4, 6), call.slot_mapping, "pages 1,2,3 at offset 0, blockSize 2")
        self.assertEqual(
            ((1,), (2,), (3,)), call.block_tables,
            "the context-2 bucket's table is one page wide, so there is nothing to pad",
        )
        self.assertEqual((3, 1, 4), call.token_ids)

    def test_the_context_bucket_grows_with_the_sequence(self):
        build_decode_call(self.pool, self.art, [(0, 3)])
        build_decode_call(self.pool, self.art, [(0, 3)])
        call = build_decode_call(self.pool, self.art, [(0, 3)])
        self.assertEqual((1, 4), call.bucket, "three tokens no longer fit the context-2 bucket")
        self.assertEqual((2,), call.positions, "the third token sits at position 2")
        self.assertEqual((3,), call.seq_lens)
        self.assertEqual((1, 2), call.block_tables[0], "a second page, and no padding left")

    def test_the_kwargs_are_exactly_run_decodes(self):
        call = build_decode_call(self.pool, self.art, [(0, 3)])
        self.assertEqual(
            {"token_ids", "positions", "block_tables", "seq_lens", "slot_mapping", "context"},
            set(call.as_kwargs()),
        )

    def test_one_sequence_twice_in_a_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "appears twice"):
            build_decode_call(self.pool, self.art, [(0, 3), (0, 4)])

    def test_an_unadmitted_sequence_is_refused(self):
        with self.assertRaisesRegex(KeyError, "not in the page pool"):
            build_decode_call(self.pool, self.art, [(5, 3)])

    def test_an_empty_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "empty batch"):
            build_decode_call(self.pool, self.art, [])

    def test_a_batch_past_the_ladder_is_refused_not_split(self):
        p = PagePool(num_blocks=12, block_size=2)
        for sid in range(5):
            p.add_sequence(sid)
        with self.assertRaises(ValueError):
            build_decode_call(p, self.art, [(i, 1) for i in range(5)])


class SchedulerOutputAdapterTest(unittest.TestCase):
    class FakeNew:
        def __init__(self, rid, prompt):
            self.req_id, self.prompt_token_ids = rid, prompt

    class FakeCached:
        def __init__(self, ids):
            self.req_ids = ids

    class FakeOutput:
        def __init__(self, new, cached, counts, finished=()):
            self.scheduled_new_reqs = new
            self.scheduled_cached_reqs = cached
            self.num_scheduled_tokens = counts
            self.finished_req_ids = list(finished)

    def test_new_and_cached_requests_both_become_decode_rows(self):
        out = self.FakeOutput(
            [self.FakeNew("a", [7])], self.FakeCached(["b"]),
            {"a": 1, "b": 1}, finished=["c"],
        )
        new, decode, finished = decode_requests_from_scheduler_output(
            out, last_token_of=lambda rid: {"a": 7, "b": 11}[rid],
        )
        self.assertEqual([("a", [7])], new)
        self.assertEqual([("a", 7), ("b", 11)], decode)
        self.assertEqual(["c"], finished)

    def test_a_new_request_never_asks_the_runner_for_a_history_it_cannot_have(self):
        """§0.4.477 (H7) — the bug the LIVE vLLM lane found on step 0.

        `TlalocWorker.execute_model` builds the batch BEFORE it admits the
        new sequences, so asking `last_token_of` about a new request asks
        the runner for a sequence it has not been told about — `KeyError`
        on the first step of every server that ever started. The old test
        above missed it because its `last_token_of` was a dict literal that
        happened to have an entry for "a"; a runner does not.

        Here the stand-in is a runner's actual behaviour: it RAISES for an
        id it has never seen. A new request's feed token must come from the
        prompt in the scheduler output, so this must pass without the
        callback ever being reached."""
        asked = []

        def strict_last_token_of(rid):
            asked.append(rid)
            raise KeyError(rid)  # what TlalocModelRunner.sequence_tokens does

        out = self.FakeOutput([self.FakeNew("a", [7])], None, {"a": 1})
        new, decode, finished = decode_requests_from_scheduler_output(
            out, last_token_of=strict_last_token_of,
        )
        self.assertEqual([("a", 7)], decode)
        self.assertEqual([], asked, "last_token_of must not be consulted for a new request")

    def test_a_new_request_with_no_prompt_is_refused_rather_than_indexed(self):
        out = self.FakeOutput([self.FakeNew("a", [])], None, {"a": 1})
        with self.assertRaisesRegex(ValueError, "empty prompt"):
            decode_requests_from_scheduler_output(out, last_token_of=lambda rid: 1)

    def test_a_new_request_scheduled_for_its_whole_prompt_is_admitted(self):
        """§0.4.492 (H3c-4b) — a PREFILL is admissible; a CHUNK is not.

        vLLM schedules a new request for `len(prompt)` tokens, so the old
        blanket refusal of `n != 1` ended every real `generate()` on its
        first step. The whole prompt is servable as N single-token decode
        steps (`TlalocModelRunner.add_sequence`), and the feed token is the
        LAST one — the steps before it are the runner's job."""
        out = self.FakeOutput([self.FakeNew("a", [1, 2, 3])], None, {"a": 3})
        new, decode, _ = decode_requests_from_scheduler_output(
            out, last_token_of=lambda rid: 1,
        )
        self.assertEqual([("a", [1, 2, 3])], new)
        self.assertEqual([("a", 3)], decode)

    def test_a_chunk_of_a_prompt_is_refused_by_name(self):
        """Three tokens scheduled of a five-token prompt is a CHUNK: the rest
        arrives on a later step and this runner keeps no resumption state."""
        out = self.FakeOutput([self.FakeNew("a", [1, 2, 3, 4, 5])], None, {"a": 3})
        with self.assertRaisesRegex(NotImplementedError, "prefill entry, which is not supported"):
            decode_requests_from_scheduler_output(out, last_token_of=lambda rid: 1)

    def test_a_cached_request_scheduled_for_more_than_one_token_is_refused(self):
        """A resumed chunk or a speculative draft. Only an ARRIVING request
        may be scheduled for more than one token."""
        out = self.FakeOutput([], self.FakeCached(["b"]), {"b": 3})
        with self.assertRaisesRegex(NotImplementedError, "prefill entry, which is not supported"):
            decode_requests_from_scheduler_output(out, last_token_of=lambda rid: 1)

    def test_a_prefix_cache_hit_is_refused_rather_than_started_midway(self):
        """This runner's KV pool is its own and holds no cross-request prefix,
        so a request arriving with tokens already computed would attend over
        pages nothing wrote."""
        new = self.FakeNew("a", [1, 2, 3])
        new.num_computed_tokens = 2
        out = self.FakeOutput([new], None, {"a": 1})
        with self.assertRaisesRegex(NotImplementedError, "prefix cache hit"):
            decode_requests_from_scheduler_output(out, last_token_of=lambda rid: 1)

    def test_an_empty_scheduler_output_produces_an_empty_batch(self):
        out = self.FakeOutput([], None, {})
        new, decode, finished = decode_requests_from_scheduler_output(
            out, last_token_of=lambda rid: 0,
        )
        self.assertEqual(([], [], []), (new, decode, finished))


class SamplingTest(unittest.TestCase):
    def test_argmax_with_ties_to_the_lowest_index(self):
        self.assertEqual(1, greedy_sample([0.0, 2.0, 2.0, 1.0]))

    def test_the_token_axis_is_indexed_and_not_flattened_away(self):
        """A decode row is [tokensPerSeq, vocab] and the token sampled is the
        LAST one. Flattening would be right at tokensPerSeq == 1 and wrong
        the day a prefill entry exists — here the row has two positions and
        the argmax of the flattened row is in the wrong one."""
        row = [[9.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
        self.assertEqual([0.0, 0.0, 5.0], last_token_logits(row))
        self.assertEqual(2, greedy_sample(last_token_logits(row)))

    def test_a_plain_vocabulary_vector_passes_through(self):
        self.assertEqual([1.0, 2.0], last_token_logits([1.0, 2.0]))

    def test_an_empty_row_is_refused(self):
        with self.assertRaisesRegex(ValueError, "empty logits row"):
            greedy_sample([])


class RegistrationTest(unittest.TestCase):
    """The entry point, pinned against the code it names.

    vLLM's platform discovery reads an entry point out of installed
    metadata and imports whatever string it returns. Nothing at runtime
    checks that the string names a class that exists, and a typo surfaces
    as "no platform found" — the least informative error in the system.
    """

    def test_register_returns_the_declared_platform_path(self):
        self.assertEqual(vllm_tlaloc.PLATFORM_CLASS_PATH, vllm_tlaloc.register())

    def test_the_platform_path_names_a_module_and_class_that_exist(self):
        mod, _, cls = vllm_tlaloc.PLATFORM_CLASS_PATH.rpartition(".")
        path = HERE / Path(*mod.split(".")).with_suffix(".py")
        self.assertTrue(path.exists(), f"{path} does not exist")
        # Read, not import: platform.py imports vLLM by design.
        self.assertIn(f"class {cls}(", path.read_text())

    def test_the_pyproject_entry_point_matches_the_callable(self):
        import tomllib

        data = tomllib.loads((HERE / "pyproject.toml").read_text())
        eps = data["project"]["entry-points"]["vllm.platform_plugins"]
        self.assertEqual({"tlaloc": "vllm_tlaloc:register"}, eps)
        self.assertTrue(callable(getattr(vllm_tlaloc, "register")))

    def test_the_distribution_ships_the_loader_it_delegates_to(self):
        import tomllib

        data = tomllib.loads((HERE / "pyproject.toml").read_text())
        self.assertIn("vllm_tlaloc", data["tool"]["setuptools"]["packages"])
        mods = data["tool"]["setuptools"]["py-modules"]
        self.assertIn(
            "tlaloc_serve", mods,
            "a plugin installed without the H3a loader can find vLLM and not its model",
        )
        # §0.4.476 (H6b): the loader executes through the ctypes PJRT binding
        # and certifies under the import guard. Shipping the loader without
        # them installs a module that cannot import.
        for m in ("tlaloc_pjrt", "import_guard"):
            self.assertIn(m, mods, f"tlaloc_serve imports {m}; it must ship with it")

    def test_the_serving_distribution_declares_no_runtime_dependencies(self):
        """§0.4.476 (H6b) — the deployment claim, as a property of the
        distribution rather than a sentence in a doc.

        It declared `numpy` while the loader ran on jaxlib arrays. The ctypes
        PJRT binding removed the last one. An oracle's numpy is a thing you
        install to MEASURE a deployment; the moment it appears here, pip stops
        being able to tell the difference.
        """
        import tomllib

        data = tomllib.loads((HERE / "pyproject.toml").read_text())
        self.assertEqual(
            [], data["project"].get("dependencies", []),
            "the serving runtime is a PJRT plugin .so and a driver; nothing belongs here",
        )

    def test_vllm_is_not_a_dependency_of_this_distribution(self):
        import tomllib

        data = tomllib.loads((HERE / "pyproject.toml").read_text())
        deps = " ".join(data["project"].get("dependencies", []))
        self.assertNotIn("vllm", deps, "vLLM is the host, not a dependency")

    def test_registering_does_not_import_vllm(self):
        """vLLM calls EVERY registered plugin's entry point during
        discovery. A plugin that imports its own world to answer "what is
        your class called" makes every other backend pay for it."""
        before = set(sys.modules)
        vllm_tlaloc.register()
        new = set(sys.modules) - before
        self.assertFalse(
            [m for m in new if m.startswith(("vllm.", "torch"))],
            f"register() imported {new}",
        )


class KvLayoutTest(unittest.TestCase):
    """§0.4.491 (H3c-4a) — the page layout the attention backend class
    publishes, checked against a REAL exported manifest and against the
    arithmetic vLLM 0.29.0 does to the same pool.

    The backend class itself cannot run here (it subclasses
    `vllm.v1.attention.backend.AttentionBackend`), and everything it
    answers WITH can — which is why `kv_layout.py` exists as a module of
    its own. The live lane in `run_vllm_live_check.py` then pins these same
    numbers against vLLM's own `compute_layer_kv_cache_shape_bytes`.
    """

    #: A real exported manifest's model block — the reference decode graph's,
    #: field for field as `ServingModelShape.toJson` writes it.
    MODEL = {
        "vocabSize": 11, "hiddenSize": 8, "numHeads": 4, "numKvHeads": 2,
        "headDim": 2, "numLayers": 1, "numBlocks": 6, "blockSize": 2,
        "dtype": "f32", "kvDtype": "f32", "kvQuant": None,
        "kvPoolAxisOrder": ["numBlocks", "blockSize", "numKvHeads", "headDim"],
        "kvPoolDims": [6, 2, 2, 2],
    }

    def test_the_shape_is_the_manifest_dims_with_k_and_v_in_front(self):
        self.assertEqual((2, 6, 2, 2, 2), kv_layout.kv_cache_shape(self.MODEL))
        self.assertEqual(
            ("kv", "numBlocks", "blockSize", "numKvHeads", "headDim"),
            kv_layout.kv_cache_axis_names(self.MODEL),
        )

    def test_the_dims_are_read_through_the_axis_names_and_not_by_position(self):
        """The failure this guards is invisible when extents coincide.

        The reference pool is `[6, 2, 2, 2]`: three of its four axes are 2,
        so a positional read is indistinguishable from a named one on it.
        Permute the STATED order and a named read must refuse, because the
        artifact is describing a pool this plugin was not compiled for.
        """
        permuted = dict(self.MODEL)
        permuted["kvPoolAxisOrder"] = ["numBlocks", "numKvHeads", "blockSize", "headDim"]
        with self.assertRaises(kv_layout.KvLayoutError) as cm:
            kv_layout.kv_cache_shape(permuted)
        self.assertIn("different compiled artifact", str(cm.exception))

    def test_a_mismatched_pair_of_fields_is_refused(self):
        short = dict(self.MODEL, kvPoolDims=[6, 2, 2])
        with self.assertRaises(kv_layout.KvLayoutError):
            kv_layout.kv_cache_shape(short)

    def test_the_layout_name_is_derived_from_the_axis_order(self):
        # numBlocks->B, blockSize->N, numKvHeads->H, headDim->C, L outermost
        # because Tlaloc's pools are separate tensors per layer.
        self.assertEqual("LBNHC", kv_layout.kv_cache_layout_name(self.MODEL))

    def test_a_page_is_one_block_of_k_and_v_for_one_layer(self):
        # 2 (K,V) * blockSize 2 * numKvHeads 2 * headDim 2 * 4 bytes = 64
        self.assertEqual(64, kv_layout.page_size_bytes(self.MODEL))
        # and the whole pool is that, times numBlocks, times numLayers.
        self.assertEqual(64 * 6 * 1, kv_layout.pool_bytes(self.MODEL))

    def test_the_two_readings_of_one_page_spend_the_same_bytes(self):
        """`KV_PAGE_BYTES_AGREE`, which is the whole of what a compiled
        artifact and vLLM's block manager have to agree about.

        vLLM 0.29.0 reads a page as `[B, H, N, C]` with K and V INTERLEAVED
        into `C`; Tlaloc stores them as two tensors. The axis orders differ
        and are not made to match — the byte count does, and that is the
        number blocks are handed out against.
        """
        b, h, n, c = kv_layout.vllm_logical_page_shape_bytes(self.MODEL)
        self.assertEqual((6, 2, 2, 16), (b, h, n, c))
        self.assertEqual(h * n * c, kv_layout.page_size_bytes(self.MODEL))
        self.assertEqual(64, kv_layout.check_page_bytes_agree(self.MODEL))

    def test_an_unknown_kv_dtype_is_refused_rather_than_sized(self):
        with self.assertRaises(kv_layout.KvLayoutError) as cm:
            kv_layout.page_size_bytes(dict(self.MODEL, kvDtype="fp8"))
        self.assertIn("pool that does not exist", str(cm.exception))

    def test_it_reads_the_model_block_out_of_a_real_artifact_directory(self):
        """The round trip the contract is about: a manifest a Kotlin
        exporter wrote, parsed here, giving the shape the pool has."""
        exported = (
            HERE.parents[1] / "examples" / "gpu-inference" / "build" / "artifact"
        )
        if not (exported / kv_layout.MANIFEST_NAME).is_file():
            self.skipTest(f"no exported artifact at {exported}")
        model = kv_layout.read_model(exported)
        self.assertEqual(
            list(kv_layout.KV_POOL_AXIS_ORDER), model["kvPoolAxisOrder"],
            "the exporter and this reader must state the same axis order",
        )
        self.assertEqual(
            (2, *model["kvPoolDims"]), kv_layout.kv_cache_shape(model),
        )
        kv_layout.check_page_bytes_agree(model)

    def test_a_directory_with_no_manifest_is_refused_by_name(self):
        with self.assertRaises(kv_layout.KvLayoutError) as cm:
            kv_layout.read_model(HERE)
        self.assertIn(kv_layout.MANIFEST_NAME, str(cm.exception))


class AttentionBackendPathTest(unittest.TestCase):
    """§0.4.491 — the dotted path, pinned the way the platform path is.

    vLLM resolves what `get_attn_backend_cls` returns with
    `resolve_obj_by_qualname` and nothing checks it first, so a typo
    surfaces as an import error deep inside engine startup.
    """

    def test_the_path_names_a_module_and_class_that_exist(self):
        mod, _, cls = vllm_tlaloc.ATTENTION_BACKEND_CLASS_PATH.rpartition(".")
        path = HERE / Path(*mod.split(".")).with_suffix(".py")
        self.assertTrue(path.exists(), f"{path} does not exist")
        # Read, not import: attention.py imports vLLM by design.
        self.assertIn(f"class {cls}(", path.read_text())

    def test_the_platform_returns_that_constant_and_not_a_literal(self):
        src = (HERE / "vllm_tlaloc" / "platform.py").read_text()
        self.assertIn("return ATTENTION_BACKEND_CLASS_PATH", src)
        self.assertNotIn(
            f'"{vllm_tlaloc.ATTENTION_BACKEND_CLASS_PATH}"', src,
            "the platform must return the shared constant, not its own copy of the string",
        )


class ImportGuardTest(unittest.TestCase):
    def test_the_vllm_facing_modules_say_what_is_missing(self):
        try:
            import vllm  # noqa: F401

            self.skipTest("vLLM is installed; the guard is not the live path")
        except ImportError:
            pass
        for name in ("vllm_tlaloc.platform", "vllm_tlaloc.worker", "vllm_tlaloc.attention"):
            with self.assertRaises(ImportError) as cm:
                __import__(name)
            self.assertIn("requires vLLM", str(cm.exception))

    def test_the_attention_module_reaches_for_vllm_before_torch(self):
        """§0.4.491 — an ordering the unit lane found the hard way.

        `attention.py` needs torch dtypes, but only where vLLM is present.
        The ORACLE venv has torch and no vLLM, so importing torch first
        would leave it resident in a process that then correctly refused to
        load this module — a framework leak with a passing import guard
        above it. `test_importing_the_loader_pulls_in_no_framework` caught
        it; this pins the fix at the line that has to stay in order.
        """
        src = (HERE / "vllm_tlaloc" / "attention.py").read_text()
        self.assertLess(
            src.index("from vllm.v1.attention.backend import"),
            src.index("\n    import torch"),
            "attention.py must reach for vLLM before torch: its torch need is "
            "downstream of vLLM's, and the reverse order leaks torch into a venv "
            "that has no vLLM",
        )

    def test_the_certified_modules_import_without_vllm(self):
        import vllm_tlaloc.batching  # noqa: F401
        import vllm_tlaloc.kv_layout  # noqa: F401
        import vllm_tlaloc.paging  # noqa: F401
        import vllm_tlaloc.runner  # noqa: F401


class LoaderIsStandardLibraryOnly(unittest.TestCase):
    """§0.4.476 (H6b) — the loader's shape-and-dtype arithmetic, with no
    device, no plugin and no framework in the room.

    These are the parts of `tlaloc_serve` that decide how many numbers go
    into a buffer and how they come back out. They are exactly where a
    silent reshape lives, and they are testable on a laptop precisely
    because the module no longer needs jaxlib to be imported.
    """

    def setUp(self):
        import tlaloc_serve

        self.s = tlaloc_serve

    def test_importing_the_loader_pulls_in_no_framework(self):
        import tlaloc_serve  # noqa: F401

        for root in ("jax", "jaxlib", "torch", "numpy"):
            self.assertNotIn(
                root, sys.modules,
                f"importing tlaloc_serve dragged in {root}; the jax ORACLE engine's "
                f"imports must stay inside its methods",
            )

    def test_flatten_and_unflatten_are_inverses_in_row_major_order(self):
        flat = list(range(24))
        nested = self.s.unflatten(flat, (2, 3, 4))
        self.assertEqual(2, len(nested))
        self.assertEqual([0, 1, 2, 3], nested[0][0])
        self.assertEqual([20, 21, 22, 23], nested[1][2])
        self.assertEqual(flat, self.s.flatten(nested))

    def test_numel_is_the_element_count_a_buffer_needs(self):
        self.assertEqual(24, self.s.numel((2, 3, 4)))
        self.assertEqual(1, self.s.numel(()))

    def test_an_unstageable_dtype_is_refused_by_name(self):
        with self.assertRaises(ValueError) as cm:
            self.s._stage_for("f64")
        self.assertIn("f64", str(cm.exception))
        self.assertIn("refused BY NAME", str(cm.exception))

    def test_every_dtype_the_decode_contract_uses_can_be_staged(self):
        # i32 for the five index operands, f32 for the pools and logits,
        # bf16 for the G1c dtype. A missing one is a graph that cannot run.
        for dt in ("i32", "f32", "bf16"):
            maker, reader, _ = self.s._stage_for(dt)
            self.assertTrue(maker.startswith("buffer_from_host_"))
            self.assertTrue(reader.startswith("to_"))

    def test_the_engine_default_follows_whether_a_platform_has_a_plugin(self):
        self.assertEqual("ctypes", self.s.default_engine_for("cuda"))
        self.assertEqual("ctypes", self.s.default_engine_for("tpu"))
        self.assertEqual(
            "jax", self.s.default_engine_for("cpu"),
            "jaxlib ships no CPU PJRT plugin .so, so 'cpu' is the one platform the "
            "ctypes engine has nothing to dlopen for",
        )

    def test_an_unknown_engine_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError) as cm:
            self.s.ServingArtifact(Path("/nonexistent"),
                                   {"schemaVersion": self.s.SCHEMA_VERSION},
                                   engine="iree")
        self.assertIn("iree", str(cm.exception))

    def test_an_artifact_of_unknown_schema_is_refused_before_anything_is_read(self):
        with self.assertRaises(ValueError) as cm:
            self.s.ServingArtifact(Path("/nonexistent"), {"schemaVersion": "v99"})
        self.assertIn("refusing an artifact of unknown shape", str(cm.exception))

    def test_an_artifact_with_a_windowed_kv_pool_is_refused_by_name(self):
        with self.assertRaises(ValueError) as cm:
            self.s.ServingArtifact(Path("/nonexistent"), {"schemaVersion": "tlaloc-serving-v3"})
        self.assertIn("windowed KV pool", str(cm.exception))

    def test_finding_a_plugin_refuses_a_path_that_is_not_there(self):
        with self.assertRaises(FileNotFoundError):
            self.s.find_pjrt_plugin("/definitely/not/a/plugin.so")

    def test_the_padding_mirror_refuses_a_producer_that_disagrees(self):
        self.s.check_padding_constants({"PADDING_SEQ_LEN": 1})
        with self.assertRaises(ValueError) as cm:
            self.s.check_padding_constants({"PADDING_SEQ_LEN": 0})
        self.assertIn("pad a batch differently", str(cm.exception))



class PluginDiscoveryTest(unittest.TestCase):
    """`tlaloc_pjrt.find_plugin` over synthetic install trees: the JVM's
    `PjrtBinaries` roots in the same order, libtpu only for tpu, and a search
    report when nothing is found. Also the PJRT_Api version gate and the named
    option-variable errors, which need no plugin."""

    def setUp(self):
        import tempfile

        import tlaloc_pjrt

        self.P = tlaloc_pjrt
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, *parts) -> str:
        path = self.home.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return str(path)

    def _cuda_plugin_in(self, root: tuple, py="python3.12", pkg="xla_cuda12", name="xla_cuda_plugin.so") -> str:
        return self._touch(*root, "lib", py, "site-packages", "jax_plugins", pkg, name)

    def _find(self, platform="cuda", env=None):
        return self.P.find_plugin(platform, env=env or {}, home=str(self.home), extra_site_dirs=())

    def _skip_if_system_has(self, platform):
        if self.P._resolve(platform, {}, "/nonexistent-home", ()):
            self.skipTest(f"this host has a system-wide {platform} plugin under /usr or /lib")

    def test_any_venv_under_local_venvs_any_python_any_cuda_package(self):
        self._skip_if_system_has("cuda")
        want = self._cuda_plugin_in((".local", "venvs", "aaa"), py="python3.11", pkg="xla_cuda13", name="plugin.so")
        self._cuda_plugin_in((".local", "venvs", "zzz"))
        self.assertEqual(want, self._find())

    def test_an_activated_venv_comes_first(self):
        self._cuda_plugin_in((".local", "venvs", "aaa"))
        venv_plugin = self._cuda_plugin_in(("elsewhere",))
        self.assertEqual(venv_plugin, self._find(env={"VIRTUAL_ENV": str(self.home / "elsewhere")}))

    def test_user_site_and_dist_packages_are_searched(self):
        self._skip_if_system_has("cuda")
        want = self._touch(".local", "lib", "python3.10", "dist-packages", "jax_plugins", "cuda_x", "p.so")
        self.assertEqual(want, self._find())

    def test_libtpu_is_found_only_for_tpu(self):
        self._skip_if_system_has("cuda")
        libtpu = self._touch("tpuvenv", "lib", "python3.12", "site-packages", "libtpu", "libtpu.so")
        cuda = self._cuda_plugin_in(("tpuvenv",))
        env = {"VIRTUAL_ENV": str(self.home / "tpuvenv")}
        self.assertEqual(cuda, self._find("cuda", env))
        self.assertEqual(libtpu, self._find("tpu", env))

    def test_the_env_variable_wins_for_cuda_but_not_with_a_cuda_file_for_tpu(self):
        self._skip_if_system_has("tpu")
        shipped = self._touch("ship", "xla_cuda_plugin.so")
        self._cuda_plugin_in((".local", "venvs", "a"))
        env = {"TLALOC_PJRT_PLUGIN_PATH": shipped}
        self.assertEqual(shipped, self._find("cuda", env))
        with self.assertRaises(FileNotFoundError) as cm:
            self._find("tpu", env)
        self.assertIn("not a tpu plugin name", str(cm.exception))

    def test_a_variable_naming_no_file_is_refused(self):
        with self.assertRaises(FileNotFoundError) as cm:
            self._find(env={"TLALOC_PJRT_PLUGIN_PATH": str(self.home / "gone.so")})
        self.assertIn("TLALOC_PJRT_PLUGIN_PATH=", str(cm.exception))

    def test_nothing_found_reports_every_root(self):
        self._skip_if_system_has("cuda")
        with self.assertRaises(FileNotFoundError) as cm:
            self._find()
        report = str(cm.exception)
        for needle in ("$TLALOC_PJRT_PLUGIN_PATH: not set", "~/.local/venvs/*", "~/.venv", "~/venv",
                       "/usr/local", "python3.*", "jax_plugins/*cuda*/*.so", "Fix: export"):
            self.assertIn(needle, report)

    def test_the_serve_resolver_delegates(self):
        import tlaloc_serve

        shipped = self._touch("ship", "libtpu.so")
        with unittest.mock.patch.dict("os.environ", {"TLALOC_PJRT_PLUGIN_PATH": shipped}):
            self.assertEqual(shipped, tlaloc_serve.find_pjrt_plugin(platform="tpu"))

    def test_an_api_of_another_major_version_is_refused_by_name(self):
        with self.assertRaises(self.P.PjrtError) as cm:
            self.P.check_api_compatible("/p/plugin.so", 4096, 1, 0)
        self.assertIn("PJRT C API 1.0", str(cm.exception))
        self.assertIn("major version 0", str(cm.exception))

    def test_an_api_table_too_short_is_refused_by_name(self):
        need = self.P.PJRT_API_MIN_STRUCT_SIZE
        self.assertEqual(self.P.OFFSET_PJRT_Buffer_ToHostBuffer + 8, need)
        with self.assertRaises(self.P.PjrtError) as cm:
            self.P.check_api_compatible("/p/plugin.so", need - 8, 0, 30)
        self.assertIn(f"at least {need} bytes", str(cm.exception))
        self.P.check_api_compatible("/p/plugin.so", need, 0, 30)

    def test_malformed_option_variables_are_refused_by_name(self):
        for name, value in (("TLALOC_PJRT_MEMORY_FRACTION", "half"),
                            ("TLALOC_PJRT_MEMORY_FRACTION", "2"),
                            ("TLALOC_PJRT_PREALLOCATE", "yes")):
            with unittest.mock.patch.dict("os.environ", {name: value}):
                with self.assertRaises(ValueError) as cm:
                    self.P.PjrtClientOptions()
                self.assertIn(name, str(cm.exception))
                self.assertIn(f"'{value}'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
