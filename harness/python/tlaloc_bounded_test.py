"""Unit tests for tlaloc_bounded: manifest validation, bucket choice, padding, masks,
slicing, and the refusals. Standard library only; no GPU, no PJRT plugin.

    python -m unittest tlaloc_bounded_test -v
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import tlaloc_bounded as B
import tlaloc_serve


def manifest():
    return {
        "schemaVersion": "tlaloc-bounded-v1",
        "name": "masked_mean",
        "bounds": [{"name": "MaxSeq", "max": 8, "buckets": [2, 4, 8]}],
        "padding": {"value": 0.0},
        "inputs": [
            {"name": "x0", "role": "DATA", "dtype": "f32", "axes": [{"bound": "MaxSeq"}, {"size": 3}]},
            {"name": "validMask_MaxSeq", "role": "VALID_MASK", "dtype": "f32", "axes": [{"bound": "MaxSeq"}], "bound": "MaxSeq"},
            {"name": "validLength_MaxSeq", "role": "VALID_LENGTH", "dtype": "f32", "axes": [], "bound": "MaxSeq"},
        ],
        "outputs": [{"name": "y0", "role": "DATA", "dtype": "f32", "axes": [{"bound": "MaxSeq"}, {"size": 3}]}],
        "entries": [
            {"id": f"MaxSeq{n}", "sizes": {"MaxSeq": n}, "entryPoint": "main",
             "bodyPath": f"bodies/b{n}.mlir", "bodyHash": hashlib.sha256(f"body {n}".encode()).hexdigest(),
             "programPath": f"programs/MaxSeq{n}.json"}
            for n in (2, 4, 8)
        ],
        "paddingCheck": {"sizes": [{"MaxSeq": 1}], "maxDifference": 0.0, "tolerance": 1e-5},
    }


class FakeEngine:
    """Records what it was asked to compile and run; the 'program' doubles every DATA value
    and returns the first input as the output (so padding and slicing are observable)."""

    def __init__(self):
        self.compiled = []
        self.calls = []

    def compile(self, text):
        self.compiled.append(text)
        return ("exe", text)

    def run(self, exe, staged, outputs):
        self.calls.append((exe, [(slot, list(v)) for slot, v in staged], outputs))
        slot, values = staged[0]
        assert outputs[0].dims == slot.dims
        return [[2 * v for v in values]] + [[float(i)] * B.numel(o.dims) for i, o in enumerate(outputs[1:], 1)]

    def close(self):
        pass


def artifact_dir(m=None):
    d = Path(tempfile.mkdtemp(prefix="tlaloc-bounded-test"))
    (d / "bodies").mkdir()
    for n in (2, 4, 8):
        (d / "bodies" / f"b{n}.mlir").write_text(f"body {n}")
    (d / "tlaloc-bounded.json").write_text(json.dumps(m or manifest()))
    return d


class ValidateTest(unittest.TestCase):
    def test_a_complete_manifest_validates(self):
        B.validate(manifest())

    def refused(self, mutate, fragment):
        m = manifest()
        mutate(m)
        with self.assertRaises(B.ManifestError) as cm:
            B.validate(m)
        self.assertIn(fragment, str(cm.exception))

    def test_other_versions_are_refused(self):
        self.refused(lambda m: m.update(schemaVersion="tlaloc-bounded-v2"), "is not one this reader knows")
        self.refused(lambda m: m.update(schemaVersion="tlaloc-serving-v2"), "is not one this reader knows")

    def test_unknown_keys_roles_and_axes_are_refused(self):
        self.refused(lambda m: m.update(extra=1), "unknown key(s) ['extra']")
        self.refused(lambda m: m["inputs"][1].update(role="ATTENTION_BIAS"), "unknown role 'ATTENTION_BIAS'")
        self.refused(lambda m: m["inputs"][0]["axes"][1].update(bound="MaxSeq"), "exactly one of size and bound")
        self.refused(lambda m: m["inputs"][0]["axes"][0].update(bound="Other"), "undeclared bound 'Other'")
        self.refused(lambda m: m["inputs"][0].update(dtype="bf16"), "is not f32 or i32")

    def test_missing_keys_and_a_failed_padding_check_are_refused(self):
        self.refused(lambda m: m.update(padding={}), "padding lacks ['value']")
        self.refused(lambda m: m["entries"][0].pop("bodyHash"), "an entry lacks ['bodyHash']")
        self.refused(lambda m: m["bounds"][0].pop("max"), "a bound lacks ['max']")
        self.refused(lambda m: m["paddingCheck"].update(maxDifference=0.5), "paddingCheck records a failure")
        self.refused(lambda m: m["inputs"][1].update(axes=[]), "VALID_MASK input validMask_MaxSeq must be f32 [MaxSeq]")

    def test_buckets_and_entries_must_agree(self):
        self.refused(lambda m: m["bounds"][0].update(buckets=[2, 4]), "must ascend strictly from >= 1 to max 8")
        self.refused(lambda m: m["entries"].pop(), "entries cover 2 bucket combinations; the buckets give 3")


class HelpersTest(unittest.TestCase):
    def test_pad_and_slice_are_inverse_on_the_leading_block(self):
        x = list(range(1, 7))  # [2, 3]
        p = B.pad_to(x, [2, 3], [4, 3])
        self.assertEqual(p, [1, 2, 3, 4, 5, 6] + [0] * 6)
        self.assertEqual(B.slice_to(p, [4, 3], [2, 3]), x)
        q = B.pad_to(x, [2, 3], [3, 5], fill=-1)
        self.assertEqual(q, [1, 2, 3, -1, -1, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1])
        self.assertEqual(B.slice_to(q, [3, 5], [2, 3]), x)
        self.assertEqual(B.pad_to([7], [], []), [7])
        with self.assertRaises(ValueError):
            B.pad_to(x, [2, 3], [1, 3])


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.art = B.BoundedArtifact.load(artifact_dir(), engine=self.engine)

    def test_bucket_choice_is_the_smallest_that_holds(self):
        self.assertEqual([self.art.buckets_for({"MaxSeq": n})["MaxSeq"] for n in range(1, 9)], [2, 2, 4, 4, 8, 8, 8, 8])
        with self.assertRaises(ValueError):
            self.art.buckets_for({"MaxSeq": 9})

    def test_run_pads_fills_masks_and_slices(self):
        x = [float(i) for i in range(5 * 3)]
        y, dims = self.art.run([(x, [5, 3])])
        self.assertEqual(dims, [5, 3])
        self.assertEqual(y, [2 * v for v in x])
        exe, staged, outputs = self.engine.calls[-1]
        self.assertEqual(exe, ("exe", "body 8"))
        (xs, xv), (ms, mv), (ls, lv) = staged
        self.assertEqual(xs.dims, (8, 3))
        self.assertEqual(xv, x + [0.0] * 9)
        self.assertEqual((ms.role, ms.dims, mv), ("VALID_MASK", (8,), [1.0] * 5 + [0.0] * 3))
        self.assertEqual((ls.role, ls.dims, lv), ("VALID_LENGTH", (), [5.0]))
        self.assertEqual(outputs[0].dims, (8, 3))

    def test_several_outputs_are_each_sliced(self):
        m = manifest()
        m["outputs"].append({"name": "y1", "role": "DATA", "dtype": "f32", "axes": [{"size": 2}]})
        m["outputs"].append({"name": "y2", "role": "DATA", "dtype": "f32", "axes": [{"bound": "MaxSeq"}]})
        art = B.BoundedArtifact.load(artifact_dir(m), engine=FakeEngine())
        outs = art.run_all([([1.0] * 9, [3, 3])])
        self.assertEqual([d for _, d in outs], [[3, 3], [2], [3]])
        self.assertEqual(outs[1][0], [1.0, 1.0])
        self.assertEqual(outs[2][0], [2.0, 2.0, 2.0])
        with self.assertRaises(ValueError) as cm:
            art.run([([1.0] * 9, [3, 3])])
        self.assertIn("use run_all", str(cm.exception))

    def test_a_changed_body_is_refused_before_it_compiles(self):
        (self.art.root / "bodies" / "b4.mlir").write_text("tampered")
        with self.assertRaises(ValueError) as cm:
            self.art.run([([0.0] * 9, [3, 3])])
        self.assertIn("does not match its bodyHash", str(cm.exception))
        self.assertEqual(self.engine.compiled, [])

    def test_an_engine_name_is_accepted_as_in_tlaloc_serve(self):
        with self.assertRaises(ValueError) as cm:
            B.BoundedArtifact.load(artifact_dir(), engine="nope")
        self.assertIn("unknown engine 'nope'", str(cm.exception))

    def test_each_body_compiles_once(self):
        for n in (1, 2, 3, 4, 2, 1, 8, 7):
            self.art.run([([0.0] * (3 * n), [n, 3])])
        self.assertEqual(self.art.compile_count, 3)
        self.assertEqual(self.engine.compiled, ["body 2", "body 4", "body 8"])

    def test_requests_outside_the_manifest_are_refused(self):
        for data, fragment in [
            ([([0.0] * 27, [9, 3])], "outside 1..8"),
            ([([0.0] * 8, [2, 4])], "fixed at 3"),
            ([([0.0] * 5, [2, 3])], "has 5 values for dims [2, 3]"),
            ([], "0 inputs given, 1 expected"),
        ]:
            with self.assertRaises(ValueError) as cm:
                self.art.run(data)
            self.assertIn(fragment, str(cm.exception))


class OtherRuntimesTest(unittest.TestCase):
    def test_the_serving_runtime_refuses_a_bounded_artifact_by_name(self):
        with self.assertRaises(ValueError) as cm:
            tlaloc_serve.ServingArtifact.load(artifact_dir())
        self.assertIn("bounded-program artifact", str(cm.exception))
        self.assertIn("tlaloc_bounded", str(cm.exception))

    def test_the_bounded_runtime_refuses_a_serving_artifact_by_name(self):
        d = Path(tempfile.mkdtemp(prefix="tlaloc-serving-test"))
        (d / "tlaloc-serving.json").write_text("{}")
        with self.assertRaises(ValueError) as cm:
            B.BoundedArtifact.load(d, engine=FakeEngine())
        self.assertIn("language-model artifact", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
