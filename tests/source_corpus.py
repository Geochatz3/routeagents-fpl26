"""Expose selected optimizer implementation files as one searchable text.

Source-based regression tests can search this combined text without depending
on functions remaining in a single module. Files are listed explicitly so
extending the searchable implementation is a deliberate, reviewable change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The orchestrator, plus every module carved out of it.
SOURCE_FILES = (
    ROOT / "dcp_optimizer.py",
    ROOT / "optimizer" / "llm_runtime.py",
    ROOT / "optimizer" / "phase1_sense.py",
    ROOT / "optimizer" / "finalize_mux.py",
    ROOT / "optimizer" / "config_resolution.py",
    ROOT / "optimizer" / "recipe_policy.py",
    ROOT / "optimizer" / "qor_parsing.py",
    ROOT / "optimizer" / "tool_source.py",
    ROOT / "optimizer" / "polish_ladder.py",
    ROOT / "optimizer" / "finalization.py",
    ROOT / "optimizer" / "tool_dispatch.py",
    ROOT / "optimizer" / "recipe_passes.py",
)


def dcp_source_text(errors: str = "replace") -> str:
    """Concatenate the available optimizer implementation files.

    Missing files are skipped to support partial module extraction and optional modules.
    """
    return "\n\n".join(p.read_text(errors=errors)
                       for p in SOURCE_FILES if p.exists())


def dcp_source_lines(errors: str = "replace"):
    return dcp_source_text(errors=errors).splitlines()


def optimizer_class_source(cls) -> str:
    """The full implementation of a class, including methods that live on the
    mixins it is composed from.

    ``inspect.getsource(cls)`` returns only the ``class`` block itself, so a
    method extracted onto a mixin disappears from it. Tests that ask "is this
    line still in the class?" mean the class as assembled, not the one text
    block that happens to carry the ``class`` keyword.
    """
    import inspect
    parts = [inspect.getsource(cls)]
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        try:
            parts.append(inspect.getsource(base))
        except (OSError, TypeError):        # builtins and C types have no source
            pass
    return "\n\n".join(parts)
