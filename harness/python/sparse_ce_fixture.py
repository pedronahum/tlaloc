"""Fixture for lmHeadSparseCrossEntropy: random hidden states, an LM head and sparse soft targets
(teacher top-k rows, one-hot rows, rows without a target); torch's loss and its gradients for the
hidden states and the head, in float64.

    python -I harness/python/sparse_ce_fixture.py --out nn/src/jvmTest/resources/io/tlaloc/nn/sparse_ce/fixture.json
"""
import argparse
import json

import torch


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    o = a.parse_args()
    g = torch.Generator().manual_seed(20261124)
    b, t, d, v, j = 2, 5, 6, 11, 4
    hidden = torch.randn(b, t, d, generator=g, dtype=torch.float64)
    head = (torch.randn(d, v, generator=g, dtype=torch.float64) * 0.7)
    ids = torch.zeros(b, t, j, dtype=torch.int64)
    w = torch.zeros(b, t, j, dtype=torch.float64)
    for r in range(b * t):
        bi, ti = divmod(r, t)
        kind = r % 3
        if kind == 0:  # a teacher's top-k, renormalized
            ids[bi, ti] = torch.randperm(v, generator=g)[:j]
            p = torch.rand(j, generator=g, dtype=torch.float64) + 0.05
            w[bi, ti] = p / p.sum()
        elif kind == 1:  # the true token alone (other slots weight 0, any id)
            ids[bi, ti] = torch.randint(0, v, (j,), generator=g)
            w[bi, ti, 0] = 1.0
        # kind 2: no target (weights 0)
    hidden.requires_grad_(True)
    head.requires_grad_(True)
    lp = torch.log_softmax(hidden @ head, dim=-1)
    loss = -(lp.gather(-1, ids) * w).sum() / w.sum()
    loss.backward()
    out = {
        "dims": {"b": b, "t": t, "d": d, "v": v, "j": j}, "chunk": 3,
        "hidden": hidden.detach().flatten().tolist(), "head": head.detach().flatten().tolist(),
        "ids": ids.flatten().tolist(), "weights": w.flatten().tolist(),
        "loss": loss.item(), "d_hidden": hidden.grad.flatten().tolist(), "d_head": head.grad.flatten().tolist(),
        "torch": torch.__version__,
    }
    with open(o.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
