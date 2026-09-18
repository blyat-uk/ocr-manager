"""Flag strings and cancellation polling shared by the detectors.

A detector result's `flagged` is None or reasons joined by "+", composed in
the order they arose and never repeated. See core/detect/__init__.py for each
detector's flags and what `auto_applicable` means.
"""
from __future__ import annotations

from collections.abc import Callable, Collection


def compose_flag(existing: str | None, new: str) -> str:
    """Add `new` to a "+"-joined flag string instead of one reason silently
    clobbering another -- e.g. "speech probes found nothing" AND "had to
    widen past the bottom band" can both be true of one crop result, and
    both are useful to a reviewer. A reason already present is not repeated."""
    if not existing:
        return new
    parts = existing.split("+")
    return existing if new in parts else f"{existing}+{new}"


def remove_flag(existing: str | None, reason: str) -> str:
    """`existing` without `reason`, keeping the order of the rest. "" when
    nothing is left. The inverse of compose_flag, for a reason the user has
    answered (accepting a crop that had to be cut to fit, say) rather than
    one the detector withdrew."""
    if not existing:
        return ""
    return "+".join(part for part in existing.split("+") if part and part != reason)


def only_informational(flagged: str | None, informational: Collection[str]) -> bool:
    """True when `flagged` is None or every reason in it is in
    `informational`. A reason the caller does not list counts as blocking."""
    if not flagged:
        return True
    return all(part in informational for part in flagged.split("+"))


def is_cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    """Poll an optional zero-argument cancel callable."""
    return cancel_check is not None and bool(cancel_check())
