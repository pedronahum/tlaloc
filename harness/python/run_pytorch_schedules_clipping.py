#!/usr/bin/env python3
"""§0.4.502 — the PyTorch oracle for Tlaloc's LR schedules and gradient clipping.

Usage: run_pytorch_schedules_clipping.py <out.json>

Emits, for the frozen fixtures below:

  schedules.step / .exponential / .cosine
      the learning rate torch's own `StepLR`, `ExponentialLR` and
      `CosineAnnealingLR` report at each completed step, read the way a
      training loop reads it: BEFORE `scheduler.step()`, so index k is the rate
      the (k+1)-th update uses. That is Tlaloc's `at(k)` by definition.

  clipping.totalNorm
      `torch.nn.utils.clip_grad_norm_`'s own reported norm.

  clipping.torchClipped
      the gradients torch leaves behind. torch scales by
      `max_norm / (total_norm + 1e-6)`; Tlaloc scales by `max_norm / total_norm`
      (a deliberate, recorded divergence — see GradientClipping.byGlobalNorm).

  clipping.analyticClipped
      the same gradients scaled by the exact ratio, which is what Tlaloc must
      reproduce. Reported alongside torch's so the ~1e-6 gap between the two is
      a measured quantity in the test output rather than a tolerance nobody
      looked at.

  clipping.valueClipped
      `torch.nn.utils.clip_grad_value_(0.75)`, which has no epsilon and must
      agree exactly.

Runs in the frozen oracle venv (`~/.local/venvs/iree`, canary-pinned by
`OracleVenvIntegrityTest`). Installs nothing.
"""

import json
import sys

# The frozen fixtures, duplicated verbatim on the Kotlin side.
STEP = {"lr0": 0.1, "gamma": 0.5, "stepSize": 3, "steps": 12}
EXPONENTIAL = {"lr0": 0.2, "gamma": 0.9, "steps": 12}
COSINE = {"lr0": 0.1, "tMax": 10, "etaMin": 0.0, "steps": 14}
GRADS = [[1.0, -2.0, 3.0], [0.5, 4.0]]
MAX_NORM = 1.0
VALUE_LIMIT = 0.75


def sweep(make_scheduler, lr0, steps):
    import torch

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=lr0)
    sched = make_scheduler(opt)
    out = []
    for _ in range(steps):
        out.append(float(opt.param_groups[0]["lr"]))
        # A step() with no preceding optimizer.step() warns in some versions;
        # the warning is about the ORDER of the two calls and not about the
        # rate sequence, which is what is being read here.
        sched.step()
    return out


def main(argv):
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    import torch

    out = {"torch": torch.__version__, "schedules": {}, "clipping": {}}

    out["schedules"]["step"] = sweep(
        lambda o: torch.optim.lr_scheduler.StepLR(
            o, step_size=STEP["stepSize"], gamma=STEP["gamma"]
        ),
        STEP["lr0"],
        STEP["steps"],
    )
    out["schedules"]["exponential"] = sweep(
        lambda o: torch.optim.lr_scheduler.ExponentialLR(o, gamma=EXPONENTIAL["gamma"]),
        EXPONENTIAL["lr0"],
        EXPONENTIAL["steps"],
    )
    out["schedules"]["cosine"] = sweep(
        lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(
            o, T_max=COSINE["tMax"], eta_min=COSINE["etaMin"]
        ),
        COSINE["lr0"],
        COSINE["steps"],
    )

    def params_with_grads():
        ps = []
        for g in GRADS:
            p = torch.nn.Parameter(torch.zeros(len(g)))
            p.grad = torch.tensor(g, dtype=torch.float32)
            ps.append(p)
        return ps

    ps = params_with_grads()
    total = torch.nn.utils.clip_grad_norm_(ps, MAX_NORM)
    out["clipping"]["totalNorm"] = float(total)
    out["clipping"]["torchClipped"] = [p.grad.tolist() for p in ps]

    norm = float(torch.tensor([v for g in GRADS for v in g]).norm())
    scale = MAX_NORM / norm if norm > MAX_NORM else 1.0
    out["clipping"]["analyticNorm"] = norm
    out["clipping"]["analyticClipped"] = [[v * scale for v in g] for g in GRADS]

    ps = params_with_grads()
    torch.nn.utils.clip_grad_value_(ps, VALUE_LIMIT)
    out["clipping"]["valueClipped"] = [p.grad.tolist() for p in ps]

    with open(argv[1], "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote %s (torch %s)" % (argv[1], torch.__version__))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
