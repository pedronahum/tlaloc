"""§0.4.444 — Phase F8 (3): the PyTorch convergence-parity reference for the
:nn MLP training certification.

Loads the SAME initial weights, inputs and targets the Kotlin side exports
(threefry ≠ torch RNG, so the weights travel as `.npy` — the §0.4.289
cross-language flow), builds the twin MLP `relu(x @ w1 + b1) @ w2 + b2`,
trains it with `torch.optim.Adam` at identical hyperparameters (lr, betas,
eps — PyTorch's Adam and Tlaloc's F3 Adam are the same Kingma–Ba
bias-corrected formula), and emits the per-step loss curve as JSON:

    {"losses": [l0, l1, ..., l_{steps-1}], "final_loss": l_steps}

`losses[k]` is the loss evaluated BEFORE the k-th update (so `losses[0]` is
the pure forward on the shared init — the tightest cross-language pin), and
`final_loss` is the forward after the last update. Full-batch MSE, exactly
the Kotlin side's captured loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--steps", required=True, type=int)
    ap.add_argument("--lr", required=True, type=float)
    args = ap.parse_args()

    import torch

    torch.manual_seed(0)  # no torch RNG is consumed; belt and braces

    arrs = {}
    for name in ("x", "y", "w1", "b1", "w2", "b2"):
        path = args.inputs_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        arrs[name] = np.load(path)

    x = torch.from_numpy(arrs["x"].astype(np.float32))
    y = torch.from_numpy(arrs["y"].astype(np.float32))
    w1 = torch.from_numpy(arrs["w1"].astype(np.float32)).requires_grad_(True)
    b1 = torch.from_numpy(arrs["b1"].astype(np.float32)).requires_grad_(True)
    w2 = torch.from_numpy(arrs["w2"].astype(np.float32)).requires_grad_(True)
    b2 = torch.from_numpy(arrs["b2"].astype(np.float32)).requires_grad_(True)

    opt = torch.optim.Adam([w1, b1, w2, b2], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    def forward() -> "torch.Tensor":
        h = torch.relu(x @ w1 + b1)
        pred = h @ w2 + b2
        return torch.mean((pred - y) ** 2)

    losses = []
    for _ in range(args.steps):
        opt.zero_grad()
        loss = forward()
        losses.append(float(loss.item()))
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_loss = float(forward().item())

    args.output.write_text(json.dumps({"losses": losses, "final_loss": final_loss}))
    print(
        f"[run_pytorch_nn_train] {args.steps} Adam steps: "
        f"loss {losses[0]:.6f} -> {final_loss:.6f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
