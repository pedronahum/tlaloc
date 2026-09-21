"""§0.4.476 (H6b) — the framework import guard, factored out of §0.4.475.

The claim H6a made and H6b extends to the whole serving path is negative:
*this process did not need jax, jaxlib, torch or numpy.* A negative claim is
only worth anything if something would have caught the positive, so the guard
is installed at the FRONT of `sys.meta_path` before the path under test runs,
and it **raises** rather than returning None — an installed jax cannot satisfy
the import behind its back.

And because a guard that never fires is an untested guard (§0.4.474's lesson),
`report()` finishes by *deliberately* importing jax and says whether it was
stopped. Both halves are asserted JVM-side: nothing forbidden was loaded, AND
the thing that would have caught it works.

This module lives on its own because §0.4.475 had one copy and §0.4.476 wanted
a second. Two copies of a safety net drift, and the one that drifts is the one
nobody was watching. Standard library only, necessarily.
"""

from __future__ import annotations

import sys

FORBIDDEN_ROOTS = ("jax", "jaxlib", "torch", "numpy")


class ForbiddenImportFinder:
    """A `sys.meta_path` finder that refuses the frameworks a slice exists to
    do without. It sits at the FRONT of meta_path, so it is consulted before
    any real finder."""

    def __init__(self, roots=FORBIDDEN_ROOTS, why: str = ""):
        self.roots = tuple(roots)
        self.why = why
        self.attempts = []

    def find_module(self, fullname, path=None):  # legacy protocol, harmless
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.roots:
            self.attempts.append(fullname)
            raise ImportError(
                f"BLOCKED: '{fullname}' must not be imported on this path. {self.why}"
            )
        return None


def install(roots=FORBIDDEN_ROOTS, why: str = "") -> ForbiddenImportFinder:
    guard = ForbiddenImportFinder(roots, why)
    sys.meta_path.insert(0, guard)
    return guard


def report(guard: ForbiddenImportFinder) -> dict:
    """What the guard saw, plus the self-test that proves it would have fired.

    `loaded_forbidden` is a SEPARATE question from `blocked_attempts`: a module
    imported before the guard went up would not be in `attempts`, and the
    honest check is that `sys.modules` carries none of them.
    """
    already = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in guard.roots})
    # Snapshot BEFORE the provocation below, which would otherwise record
    # itself: `blocked_attempts` must mean "what the path under test reached
    # for", not "what this function did to check the net was there".
    attempts = list(guard.attempts)
    try:
        __import__("jax")
        fired = False
        reason = "import jax SUCCEEDED — the guard is not installed"
    except ImportError as exc:
        fired = "BLOCKED" in str(exc)
        reason = str(exc).splitlines()[0]
    return {
        "forbidden_roots": list(guard.roots),
        "loaded_forbidden": already,
        "blocked_attempts": attempts,
        "guard_self_test_fired": fired,
        "guard_self_test_reason": reason,
    }
