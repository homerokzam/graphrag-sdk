"""Optional-dependency guard for the Agno integration."""

from __future__ import annotations

import re

_MIN_VERSION = (3, 1, 11)
_SUPPORTED = "agno>=3.1.11,<4"
_INSTALL_HINT = "Install with: pip install -U 'graphrag-sdk[agno]'"


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Read the leading ``major.minor.patch`` of a version string, or ``None``."""
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    if match is None:
        return None
    major, minor, patch = (int(part) if part else 0 for part in match.groups())
    return major, minor, patch


def require_agno() -> None:
    """Raise an actionable ``ImportError`` unless ``agno>=3.1.11,<4`` is importable."""
    try:
        import agno  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"The Agno integration requires the 'agno' package ({_SUPPORTED}). {_INSTALL_HINT}"
        ) from exc
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("agno")
    except PackageNotFoundError:
        return
    parsed = _parse_version(found)
    if parsed is None:
        return
    if parsed[0] != _MIN_VERSION[0] or parsed < _MIN_VERSION:
        raise ImportError(
            f"The Agno integration supports {_SUPPORTED} (found {found}). {_INSTALL_HINT}"
        )
