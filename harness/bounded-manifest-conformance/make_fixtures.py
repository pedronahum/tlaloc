#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Writes the tlaloc-bounded-v1 conformance fixtures: valid/*.json, which every reader
accepts, and invalid/*.json, which every reader refuses. Each invalid manifest is one
change to BASE, named by its file.

Readers tested against them: BoundedManifest.fromJson (:maestro, BoundedManifestConformanceTest),
tlaloc_bounded.validate (harness/python/tlaloc_bounded_test.py) and BoundedModel::ReadManifest
(triton/backend/test/bounded_manifest_test.cc, built and run by triton/build_backend.sh).

  python3 harness/bounded-manifest-conformance/make_fixtures.py
"""

import copy
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def entry(sizes):
    tag = "_".join(f"{k}{v}" for k, v in sizes.items())
    return {
        "id": tag,
        "sizes": sizes,
        "entryPoint": "main",
        "bodyPath": f"bodies/{tag}.mlir",
        "bodyHash": tag,
        "programPath": f"programs/{tag}.json",
    }


# Two bounds, a mask and a length, a fixed axis and an i32 input: every kind of field.
BASE = {
    "schemaVersion": "tlaloc-bounded-v1",
    "name": "conformance",
    "bounds": [
        {"name": "Seq", "max": 8, "buckets": [4, 8]},
        {"name": "Batch", "max": 2, "buckets": [1, 2]},
    ],
    "padding": {"value": 0.0},
    "inputs": [
        {"name": "x0", "role": "DATA", "dtype": "f32", "axes": [{"bound": "Batch"}, {"bound": "Seq"}, {"size": 3}]},
        {"name": "ids", "role": "DATA", "dtype": "i32", "axes": [{"bound": "Seq"}]},
        {"name": "validMask_Seq", "role": "VALID_MASK", "dtype": "f32", "axes": [{"bound": "Seq"}], "bound": "Seq"},
        {"name": "validLength_Batch", "role": "VALID_LENGTH", "dtype": "f32", "axes": [], "bound": "Batch"},
    ],
    "outputs": [
        {"name": "y0", "role": "DATA", "dtype": "f32", "axes": [{"bound": "Batch"}, {"size": 3}]},
    ],
    "entries": [entry({"Seq": s, "Batch": b}) for s in (4, 8) for b in (1, 2)],
    "paddingCheck": {
        "sizes": [{"Seq": 1, "Batch": 1}, {"Seq": 8, "Batch": 2}],
        "maxDifference": 0.0,
        "tolerance": 1e-6,
    },
}

# One bound, no mask or length, a single bucket.
MINIMAL = {
    "schemaVersion": "tlaloc-bounded-v1",
    "name": "minimal",
    "bounds": [{"name": "N", "max": 5, "buckets": [5]}],
    "padding": {"value": -1.5},
    "inputs": [{"name": "x0", "role": "DATA", "dtype": "f32", "axes": [{"bound": "N"}]}],
    "outputs": [{"name": "y0", "role": "DATA", "dtype": "f32", "axes": []}],
    "entries": [entry({"N": 5})],
    "paddingCheck": {"sizes": [{"N": 1}, {"N": 5}], "maxDifference": 1e-7, "tolerance": 1e-6},
}


def inp(m, name):
    return next(t for t in m["inputs"] if t["name"] == name)


def mutations():
    def m(f):
        def run():
            d = copy.deepcopy(BASE)
            f(d)
            return d
        return run

    return {
        "schema-version-unknown": m(lambda d: d.update(schemaVersion="tlaloc-bounded-v2")),
        "top-key-unknown": m(lambda d: d.update(extra=1)),
        "top-key-missing-name": m(lambda d: d.pop("name")),
        "top-key-missing-padding-check": m(lambda d: d.pop("paddingCheck")),
        "name-blank": m(lambda d: d.update(name=" ")),
        "bounds-empty": m(lambda d: d.update(bounds=[], entries=[])),
        "bound-name-repeats": m(lambda d: d["bounds"].append({"name": "Seq", "max": 8, "buckets": [8]})),
        "bound-key-unknown": m(lambda d: d["bounds"][0].update(min=1)),
        "bound-buckets-not-ascending": m(lambda d: d["bounds"][0].update(buckets=[8, 4])),
        "bound-bucket-below-one": m(lambda d: d["bounds"][1].update(buckets=[0, 2])),
        "bound-last-bucket-not-max": m(lambda d: d["bounds"][0].update(max=16)),
        "padding-key-unknown": m(lambda d: d["padding"].update(mode="edge")),
        "padding-value-not-number": m(lambda d: d["padding"].update(value="0")),
        "tensor-role-unknown": m(lambda d: inp(d, "ids").update(role="INDEX")),
        "tensor-dtype-unknown": m(lambda d: inp(d, "x0").update(dtype="f16")),
        "tensor-key-unknown": m(lambda d: inp(d, "x0").update(layout="nchw")),
        "axis-size-and-bound": m(lambda d: inp(d, "x0")["axes"][2].update(bound="Seq")),
        "axis-neither-size-nor-bound": m(lambda d: inp(d, "x0")["axes"].__setitem__(2, {})),
        "axis-size-below-one": m(lambda d: inp(d, "x0")["axes"][2].update(size=0)),
        "axis-bound-undeclared": m(lambda d: inp(d, "x0")["axes"][1].update(bound="Time")),
        "data-input-names-bound": m(lambda d: inp(d, "x0").update(bound="Seq")),
        "no-data-input": m(lambda d: d.update(inputs=[inp(d, "validMask_Seq")])),
        "mask-names-no-bound": m(lambda d: inp(d, "validMask_Seq").pop("bound")),
        "mask-not-f32": m(lambda d: inp(d, "validMask_Seq").update(dtype="i32")),
        "mask-axes-not-its-bound": m(lambda d: inp(d, "validMask_Seq").update(axes=[{"bound": "Batch"}])),
        "length-not-scalar": m(lambda d: inp(d, "validLength_Batch").update(axes=[{"size": 1}])),
        "output-not-data": m(lambda d: d["outputs"][0].update(role="VALID_MASK", bound="Batch", axes=[{"bound": "Batch"}])),
        "outputs-empty": m(lambda d: d.update(outputs=[])),
        "entries-missing-a-combination": m(lambda d: d["entries"].pop()),
        "entries-repeat": m(lambda d: d["entries"].__setitem__(3, copy.deepcopy(d["entries"][0]))),
        "entry-size-not-a-bucket": m(lambda d: d["entries"][0]["sizes"].update(Seq=5)),
        "entry-size-for-undeclared-bound": m(lambda d: d["entries"][0]["sizes"].update(Time=1)),
        "entry-key-unknown": m(lambda d: d["entries"][0].update(device="gpu")),
        "entry-missing-entry-point": m(lambda d: d["entries"][0].pop("entryPoint")),
        "entry-body-path-escapes": m(lambda d: d["entries"][0].update(bodyPath="../outside.mlir")),
        "entry-body-path-absolute": m(lambda d: d["entries"][0].update(bodyPath="/tmp/body.mlir")),
        "entry-program-path-escapes": m(lambda d: d["entries"][0].update(programPath="programs/../../p.json")),
        "padding-check-failed": m(lambda d: d["paddingCheck"].update(maxDifference=1e-3)),
        "padding-check-key-unknown": m(lambda d: d["paddingCheck"].update(seed=7)),
    }


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    for sub in ("valid", "invalid"):
        shutil.rmtree(HERE / sub, ignore_errors=True)
        (HERE / sub).mkdir()
    write(HERE / "valid" / "two-bounds.json", BASE)
    write(HERE / "valid" / "minimal.json", MINIMAL)
    for name, make in mutations().items():
        write(HERE / "invalid" / f"{name}.json", make())


if __name__ == "__main__":
    main()
