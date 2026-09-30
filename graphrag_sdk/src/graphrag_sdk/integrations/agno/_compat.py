"""Optional-dependency guard for the Agno integration."""

from __future__ import annotations

_INSTALL_HINT = "Install with: pip install 'graphrag-sdk[agno]'"


def require_agno() -> None:
    """Raise an actionable ``ImportError`` unless ``agno>=3,<4`` is importable."""
    try:
        import agno  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"The Agno integration requires the 'agno' package (>=3,<4). {_INSTALL_HINT}"
        ) from exc
    from importlib.metadata import PackageNotFoundError, version

    try:
        major = int(version("agno").split(".")[0])
    except (PackageNotFoundError, ValueError):
        return
    if major != 3:
        raise ImportError(
            f"The Agno integration supports agno 3.x (found {version('agno')}). {_INSTALL_HINT}"
        )
