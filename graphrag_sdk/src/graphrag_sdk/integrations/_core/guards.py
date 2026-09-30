# GraphRAG SDK — Integrations core: safety guards for agent-supplied input
# - resolve_allowed_path: file ingestion is confined to configured directories.
# - ensure_read_only_cypher / apply_limit: best-effort, fail-closed lexical guard
#   for agent-written Cypher (ported from the upstream agent-toolkit branch).

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from graphrag_sdk.core.exceptions import GraphRAGError

DEFAULT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf", ".csv", ".tsv"})


class PathNotAllowedError(GraphRAGError):
    """A file path supplied by an agent falls outside the allowed directories."""


class ReadOnlyViolation(GraphRAGError):
    """Agent-supplied Cypher contains (or may contain) a write operation."""

    def __init__(self, message: str, offending_token: str | None = None) -> None:
        super().__init__(message)
        self.offending_token = offending_token


def resolve_allowed_path(
    path: str,
    allowed_dirs: Sequence[Path],
    *,
    max_bytes: int,
    suffixes: frozenset[str] | None = DEFAULT_SUFFIXES,
) -> tuple[Path, str]:
    """Resolve *path* inside one of *allowed_dirs* or raise ``PathNotAllowedError``.

    Relative paths resolve against the first allowed directory. Symlinks are
    followed before the containment check. Returns the absolute path and a
    default document id (the POSIX path relative to its allowed root, so the
    host's absolute layout never leaks into the graph).
    """
    if not allowed_dirs:
        raise PathNotAllowedError("File ingestion is disabled (no allowed directories).")
    if "\x00" in path:
        raise PathNotAllowedError("Invalid path.")
    scheme = urlparse(path).scheme
    if len(scheme) > 1:
        raise PathNotAllowedError(f"URLs are not allowed ({scheme}://); pass a local file path.")
    roots = [Path(d).expanduser().resolve() for d in allowed_dirs]
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise PathNotAllowedError(f"File not found: {path}") from exc
    root = next((r for r in roots if resolved.is_relative_to(r)), None)
    if root is None:
        raise PathNotAllowedError(f"Path is outside the allowed directories: {path}")
    if not resolved.is_file():
        raise PathNotAllowedError(f"Not a regular file: {path}")
    if suffixes is not None and resolved.suffix.lower() not in suffixes:
        raise PathNotAllowedError(
            f"Unsupported file type '{resolved.suffix}'. Allowed: {', '.join(sorted(suffixes))}"
        )
    size = resolved.stat().st_size
    if size > max_bytes:
        raise PathNotAllowedError(f"File too large ({size} bytes > {max_bytes}).")
    return resolved, resolved.relative_to(root).as_posix()


# ── Read-only Cypher guard ───────────────────────────────────────────────────

_WRITE_TOKENS = ("CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "FOREACH")
_START_KEYWORDS = ("MATCH", "OPTIONAL", "UNWIND", "WITH", "RETURN", "CALL")
READ_SAFE_PROCEDURES = frozenset(
    {
        "db.labels",
        "db.relationshiptypes",
        "db.propertykeys",
        "db.indexes",
        "db.idx.fulltext.querynodes",
        "db.idx.fulltext.queryrelationships",
        "db.idx.vector.querynodes",
        "db.idx.vector.queryrelationships",
    }
)


def _strip_noise(text: str) -> str:
    """Remove comments and mask string/backtick literals in one lexer pass.

    Comments are removed without a separator (fail-closed: ``Cr/**/eate`` is
    reassembled and caught); literals become single spaces.
    """
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state, i = "line", i + 2
                continue
            if ch == "/" and nxt == "*":
                state, i = "block", i + 2
                continue
            if ch in ("'", '"', "`"):
                state = ch
                out.append(" ")
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if state == "line":
            if ch == "\n":
                state = "code"
                out.append("\n")
            i += 1
            continue
        if state == "block":
            if ch == "*" and nxt == "/":
                state, i = "code", i + 2
                continue
            i += 1
            continue
        if ch == "\\" and state in ("'", '"'):
            i += 2
            continue
        if ch == state:
            state = "code"
            out.append(" ")
        i += 1
    return "".join(out)


def _scan(stripped: str) -> None:
    first = re.match(r"\s*([A-Za-z]+)", stripped)
    if first and first.group(1).upper() not in _START_KEYWORDS:
        raise ReadOnlyViolation(
            f"Query must start with one of {', '.join(_START_KEYWORDS)}; got '{first.group(1)}'.",
            offending_token=first.group(1),
        )
    body = stripped.rstrip().rstrip(";")
    if ";" in body:
        raise ReadOnlyViolation("Multiple Cypher statements are not allowed.", offending_token=";")
    for token in _WRITE_TOKENS:
        if re.search(rf"\b{token}\b", stripped, re.IGNORECASE):
            raise ReadOnlyViolation(
                f"Write operation '{token}' is not allowed — cypher_read is read-only.",
                offending_token=token,
            )
    if re.search(r"\bLOAD\s+CSV\b", stripped, re.IGNORECASE):
        raise ReadOnlyViolation("LOAD CSV is not allowed.", offending_token="LOAD CSV")
    for match in re.finditer(r"\bCALL\b(\s*)([A-Za-z0-9_.]*)", stripped, re.IGNORECASE):
        proc = match.group(2)
        if not proc:
            if stripped[match.end() :].lstrip().startswith("{"):
                continue
            raise ReadOnlyViolation("Bare CALL is not allowed.", offending_token="CALL")
        if proc.lower() not in READ_SAFE_PROCEDURES:
            raise ReadOnlyViolation(
                f"Procedure '{proc}' is not on the read-safe allowlist.",
                offending_token=f"CALL {proc}",
            )


def ensure_read_only_cypher(query: str) -> None:
    """Raise ``ReadOnlyViolation`` unless *query* looks like a read-only statement."""
    if not query or not query.strip():
        raise ReadOnlyViolation("Empty Cypher query.")
    _scan(_strip_noise(query))
    _scan(_strip_noise(unicodedata.normalize("NFKC", query)))


def apply_limit(query: str, limit: int) -> tuple[str, bool]:
    """Append ``LIMIT {limit}`` when the query has no LIMIT clause."""
    trimmed = query.rstrip().rstrip(";")
    if _TOP_LEVEL_LIMIT_RE.search(_top_level(_strip_noise(trimmed))):
        return trimmed, False
    return f"{trimmed}\nLIMIT {int(limit)}", True


# A LIMIT keyword (not a property like ``n.limit``) followed by a count.
_TOP_LEVEL_LIMIT_RE = re.compile(r"(?<![.\w])LIMIT\s+(\d+|\$\w+)", re.IGNORECASE)


def _top_level(stripped: str) -> str:
    """Blank out text nested in (), [] or {} so only top-level clauses remain."""
    out: list[str] = []
    depth = 0
    for ch in stripped:
        if ch in "([{":
            depth += 1
            out.append(" ")
        elif ch in ")]}":
            depth = max(depth - 1, 0)
            out.append(" ")
        else:
            out.append(ch if depth == 0 else " ")
    return "".join(out)
