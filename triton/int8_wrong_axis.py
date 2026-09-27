#!/usr/bin/env python3
# Copyright 2026 Pedro N. Rodriguez
# SPDX-License-Identifier: Apache-2.0
"""Negative control for int8 weights: a model repository whose int8 codes were
quantized along the wrong axis.

Reads a Triton model directory served by the tlaloc backend (MODEL_DIR, with
1/tlaloc-serving.json) whose layers' projections are int8 codes `[in, out]`
followed by f32 scales `[out]`, one per output channel. Writes OUT_DIR, the
same model with every int8 weight quantized again with one scale per INPUT
channel (a row of `[in, out]`), while the scale slots keep their per-output
values: what a quantizer that took its maxima over the wrong axis would
produce. The graph still multiplies each output column by its scale, so the
model's outputs are wrong, and a quality check must fail on it.

Untouched files are hard links; the int8 files and the manifest are new.

  int8_wrong_axis.py MODEL_DIR OUT_DIR
"""

import hashlib
import json
import os
import shutil
import sys

import numpy as np


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=os.link)
    man_path = os.path.join(dst, "1", "tlaloc-serving.json")
    man = json.load(open(man_path))
    table = man["weights"]["table"]
    by_name = {w["name"]: w for w in table}
    n = 0
    for w in table:
        if w["dtype"] != "i8":
            continue
        scale = by_name[w["name"] + "Scale"]
        path = os.path.join(dst, "1", w["path"])
        cin, cout = w["dims"]
        codes = np.fromfile(path, dtype=np.int8).reshape(cin, cout).astype(np.float32)
        s_out = np.fromfile(os.path.join(dst, "1", scale["path"]), dtype=np.float32)
        weight = codes * s_out[None, :]
        s_in = np.abs(weight).max(axis=1) / 127.0
        s_in[s_in == 0] = 1.0
        wrong = np.clip(np.rint(weight / s_in[:, None]), -127, 127).astype(np.int8)
        os.unlink(path)  # a hard link to the source model: never write through it
        wrong.tofile(path)
        data = wrong.tobytes()
        w["sha256"] = hashlib.sha256(data).hexdigest()
        w["byteLength"] = len(data)
        n += 1
    if n == 0:
        sys.exit(f"{src} has no int8 weights")
    os.unlink(man_path)
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"{n} int8 weights quantized again along the input axis into {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
