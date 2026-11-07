"""Oracle for the ricursive family: Ricursive's own modeling code on a tiny checkpoint.

Imports torch and transformers on purpose: this is the oracle side of a comparison,
not serving code. It runs in a venv that has both (for example ~/.local/venvs/vllm,
transformers 5.17) and never installs anything.

The checkpoint repositories ship ``modeling_ricursive.py`` and
``configuration_ricursive.py`` (``trust_remote_code``). This script runs exactly the
revision Tlaloc was written against: it refuses files whose sha256 differs, copies
them into a temporary package outside the Hugging Face cache, and imports them from
there. They import torch and transformers only.

The checkpoint is the committed ``tiny_ripple`` fixture, which ``HfRippleParityTest``
writes from a seeded ``java.util.Random`` and checks byte for byte (four sliding
layers with a window of 8 and one full layer without RoPE):

  ~/.local/venvs/vllm/bin/python -I harness/python/ripple_modeling_fixture.py \\
      --modeling ~/.cache/huggingface/hub/models--ricursiveAI--Ripple-65B-Preview-NVFP4/snapshots/<rev> \\
      --checkpoint ir/src/jvmTest/resources/io/tlaloc/ir/inference/tiny_ripple \\
      --out ir/src/jvmTest/resources/io/tlaloc/ir/inference/ripple_modeling_logits.json

The output holds the logits at every position of each prompt, in float32 as the
modeling code computes them (its norms, rotary tables and gate cast to float32), with
eager attention.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile

# Ricursive-65B-Preview revision 99eea447 (NVFP4) and e27e1717 (BF16) ship these files.
PINNED = {
    "modeling_ricursive.py": "f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3",
    "configuration_ricursive.py": "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6",
}

PROMPTS = [
    [3, 5, 9, 12, 33, 7],
    [3, 4, 50, 11, 12, 13, 14, 20, 21, 22, 40, 41, 42, 43, 8, 9, 10, 44, 45],
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def import_modeling(directory):
    for name, digest in PINNED.items():
        got = sha256(os.path.join(directory, name))
        if got != digest:
            sys.exit(f"{name} in {directory} has sha256 {got}, not the pinned {digest}: read the new code first")
    pkg_root = tempfile.mkdtemp(prefix="ricursive-modeling-")
    pkg = os.path.join(pkg_root, "ricursive_modeling")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for name in PINNED:
        shutil.copyfile(os.path.join(directory, name), os.path.join(pkg, name))
    sys.path.insert(0, pkg_root)
    modeling = importlib.import_module("ricursive_modeling.modeling_ricursive")
    configuration = importlib.import_module("ricursive_modeling.configuration_ricursive")
    return modeling, configuration


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modeling", required=True, help="a snapshot directory holding modeling_ricursive.py")
    ap.add_argument("--checkpoint", required=True, help="the tiny checkpoint (config.json, model.safetensors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from safetensors.torch import load_file

    modeling, configuration = import_modeling(os.path.expanduser(a.modeling))
    with open(os.path.join(a.checkpoint, "config.json")) as f:
        cfg = json.load(f)
    config = configuration.RicursiveConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = modeling.RicursiveForCausalLM(config).float().eval()
    state = load_file(os.path.join(a.checkpoint, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Buffers the module makes for itself are not in the checkpoint; every weight must be.
    params = {n for n, _ in model.named_parameters()}
    if [m for m in missing if m in params] or unexpected:
        sys.exit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    out = {"source": "modeling_ricursive.py sha256 " + PINNED["modeling_ricursive.py"],
           "torch": torch.__version__, "prompts": []}
    with torch.no_grad():
        for ids in PROMPTS:
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
            out["prompts"].append({"ids": ids, "logits": [[float(x) for x in row] for row in logits.float()]})
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}: {len(PROMPTS)} prompts, vocab {logits.shape[-1]}")


if __name__ == "__main__":
    main()
