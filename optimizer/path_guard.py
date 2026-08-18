"""DCP write-path validation — defense-in-depth.

Audit-mode: warns when a DCP write target is outside the recognized
"safe" roots.  Enforce-mode: raises PathGuardError.

Recognized write roots:
  - the optimizer's run_dir (best_valid + intermediate files)
  - the output_dcp's parent directory (ship target)
  - submission/dcps  (final packaging tree)
  - any path the caller explicitly added via PathGuard.allow_root()

This is defense-in-depth.  The existing finalize / mirror code is
already disk-truth-only; this module catches accidental rogue writes
introduced by future changes BEFORE they touch shared state.

API:
  PathGuard(roots: list[Path], mode: Literal["audit","enforce"]).check(path, context)
  is_under(path, root) — pure helper, no side effects
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Literal, Optional

logger = logging.getLogger(__name__)


class PathGuardError(RuntimeError):
    """Raised in enforce mode when a write target is outside allowed roots."""


def is_under(path: Path | str, root: Path | str) -> bool:
    """Resolve symlinks; True if `path` is `root` or any descendant.
    Falls back to lexical comparison if resolution fails (unwritten
    paths)."""
    p = Path(path)
    r = Path(root)
    try:
        p_resolved = p.resolve(strict=False)
        r_resolved = r.resolve(strict=False)
    except Exception:
        p_resolved, r_resolved = p, r
    try:
        p_resolved.relative_to(r_resolved)
        return True
    except ValueError:
        return False


class PathGuard:
    """Lightweight policy for DCP write paths.

    Mode "audit":   log a WARNING on out-of-root paths; return False but never raise.
    Mode "enforce": raise PathGuardError on out-of-root paths.
    """

    Mode = Literal["audit", "enforce"]

    def __init__(
        self,
        roots: Iterable[Path | str],
        *,
        mode: Mode = "audit",
    ):
        self._roots: List[Path] = [Path(r) for r in roots if r is not None]
        self.mode: PathGuard.Mode = mode
        self.violations: List[dict] = []  # audit-mode log

    def allow_root(self, root: Path | str) -> None:
        p = Path(root)
        if p not in self._roots:
            self._roots.append(p)

    @property
    def roots(self) -> List[Path]:
        return list(self._roots)

    def check(self, path: Path | str, *, context: str = "") -> bool:
        """Return True if `path` is under an allowed root.

        In audit mode, an out-of-root path returns False, logs a
        warning, and appends to self.violations.

        In enforce mode, an out-of-root path raises PathGuardError.
        """
        p = Path(path)
        for r in self._roots:
            if is_under(p, r):
                return True

        # Out of root.
        msg = (
            f"path_guard: write target outside allowed roots "
            f"(path={p}, context={context!r}, roots={[str(r) for r in self._roots]})"
        )
        self.violations.append({
            "path": str(p),
            "context": context,
            "roots": [str(r) for r in self._roots],
        })
        if self.mode == "enforce":
            raise PathGuardError(msg)
        logger.warning(msg)
        return False

    def reset_violations(self) -> None:
        self.violations.clear()
