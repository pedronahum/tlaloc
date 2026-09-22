#!/usr/bin/env python3
"""§0.4.502 — the reference-implementation side of Tlaloc's safetensors WRITER.

§0.4.468 certified the *reader* against bytes PyTorch had written
(`write_llama_safetensors.py`). This script is the same exchange in the other
direction, and it has two halves because a format claim has two halves:

  read  <file> <out.json>   the reference `safetensors` library opens a file
                            TLALOC wrote and reports, per tensor, the dtype,
                            the shape and the RAW LITTLE-ENDIAN BYTES as hex.
                            Hex, not floats: the claim is that the bytes are
                            the bytes, and a float comparison would hide a
                            NaN payload, a signed zero or a bf16 double-round.

  write <file>              the reference library writes a file TLALOC then
                            reads, covering every dtype Tlaloc supports
                            (F32, F64, I32, BF16) plus rank-0, a zero-element
                            tensor, and metadata.

Runs in the frozen oracle venv (`~/.local/venvs/iree`, canary-pinned by
`OracleVenvIntegrityTest`). Installs nothing. `torch` is used only for the
BF16 leg — numpy has no bfloat16 dtype, so `safetensors.numpy` cannot express
one, and BF16 is the dtype Tlaloc's serving lane actually stores weights at.
"""

import json
import sys


# The shared fixture. Values are chosen to be exactly representable at every
# width involved so that a disagreement is a FORMAT disagreement and never a
# rounding one: powers of two and small integers.
FIXTURE_F32 = [1.0, -2.5, 0.0, 1024.0, -0.125, 3.0]
FIXTURE_F64 = [1.0, -0.5, 1e300, 0.0]
FIXTURE_I32 = [-7, 0, 2000000000, 1]
FIXTURE_BF16 = [1.0, -2.0, 0.5, 100.0]
FIXTURE_METADATA = {"format": "tlaloc", "tlaloc.test": "§0.4.502"}


def do_write(path):
    import torch
    from safetensors.torch import save_file

    tensors = {
        "a.f32": torch.tensor(FIXTURE_F32, dtype=torch.float32).reshape(2, 3),
        "b.f64": torch.tensor(FIXTURE_F64, dtype=torch.float64),
        "c.i32": torch.tensor(FIXTURE_I32, dtype=torch.int32).reshape(2, 2),
        "d.bf16": torch.tensor(FIXTURE_BF16, dtype=torch.bfloat16),
        "e.scalar": torch.tensor(7.0, dtype=torch.float32),
        "f.empty": torch.zeros(0, 4, dtype=torch.float32),
    }
    save_file(tensors, path, metadata=FIXTURE_METADATA)
    print("wrote %s" % path)


def do_read(path, out_json):
    from safetensors import safe_open

    out = {"tensors": {}, "metadata": {}}
    with safe_open(path, framework="pt") as f:
        md = f.metadata()
        out["metadata"] = dict(md) if md else {}
        for name in f.keys():
            t = f.get_tensor(name)
            # .view(uint8) is unavailable for every dtype across torch
            # versions; numpy's own byte view is, once bf16 is reinterpreted
            # as its 16-bit bucket (which is what bf16 IS).
            import torch

            if t.dtype == torch.bfloat16:
                raw = t.contiguous().view(torch.int16).numpy().tobytes()
                dtype = "BF16"
            else:
                raw = t.contiguous().numpy().tobytes()
                dtype = {
                    torch.float32: "F32",
                    torch.float64: "F64",
                    torch.int32: "I32",
                }.get(t.dtype, str(t.dtype))
            out["tensors"][name] = {
                "dtype": dtype,
                "shape": list(t.shape),
                "hex": raw.hex(),
            }
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("read %d tensors from %s" % (len(out["tensors"]), path))


def main(argv):
    if len(argv) >= 3 and argv[1] == "write":
        do_write(argv[2])
        return 0
    if len(argv) >= 4 and argv[1] == "read":
        do_read(argv[2], argv[3])
        return 0
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
