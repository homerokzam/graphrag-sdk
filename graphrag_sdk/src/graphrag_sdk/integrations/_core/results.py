# GraphRAG SDK — Integrations core: typed tool results + LLM-text rendering
# Framework-neutral. Rendering mirrors the (unmerged) upstream agent-toolkit
# contract: deterministic, budget-bounded text truncated at item boundaries.

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from graphrag_sdk.core.exceptions import (
    DatabaseUnavailableError,
    DocumentNotFoundError,
    EmbeddingTimeoutError,
    LLMTimeoutError,
)
from graphrag_sdk.storage.graph_store import GraphStore

_ELLIPSIS = "…"
_SNIPPET_CHARS = 240


def _clean(value: Any) -> str:
    """Strip control chars and collapse whitespace runs to single spaces."""
    return " ".join(GraphStore._sanitize_string(str(value)).split())


def _snippet(value: Any, limit: int = _SNIPPET_CHARS) -> str:
    """A cleaned, length-capped one-line excerpt."""
    text = _clean(value)
    return text if len(text) <= limit else text[: limit - 1] + _ELLIPSIS


def _render(
    preamble: list[str],
    sections: list[tuple[str, list[str]]],
    *,
    max_chars: int,
) -> str:
    """Assemble preamble lines + (header, items) sections into text <= max_chars.

    Truncation happens only at item boundaries; a dropped tail is marked
    with ``…(N more)``. Deterministic for equal inputs.
    """
    if max_chars < 1:
        return ""
    lines: list[str] = []
    used = 0

    def try_add(line: str) -> bool:
        nonlocal used
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            return False
        lines.append(line)
        used += cost
        return True

    for line in preamble:
        if not try_add(line):
            if not lines:
                lines.append(line[: max_chars - 1] + _ELLIPSIS)
            return "\n".join(lines)

    for header, items in sections:
        if not items:
            continue
        header_line = f"{header} ({len(items)}):"
        full_marker = f"  {_ELLIPSIS}({len(items)} more)"
        needed = used + (1 if lines else 0) + len(header_line) + 1 + len(full_marker)
        if needed > max_chars:
            try_add(f"{header}: {_ELLIPSIS}({len(items)} items)")
            continue
        try_add(header_line)
        for idx, item in enumerate(items):
            marker = f"  {_ELLIPSIS}({len(items) - idx} more)"
            reserve = (len(marker) + 1) if idx < len(items) - 1 else 0
            if used + len(item) + 1 + reserve <= max_chars:
                try_add(item)
            else:
                try_add(marker)
                break
    return "\n".join(lines)


class ToolResult(BaseModel):
    """Base class for tool results: strict fields + LLM-text rendering."""

    model_config = ConfigDict(extra="forbid")

    def to_llm_text(self, *, max_chars: int = 4000) -> str:
        """Render a compact, deterministic plain-text form bounded by ``max_chars``."""
        return _render(self._preamble(), self._sections(), max_chars=max_chars)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict form."""
        return self.model_dump(mode="json")

    def _preamble(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def _sections(self) -> list[tuple[str, list[str]]]:  # pragma: no cover - overridden
        return []


class Passage(BaseModel):
    """A retrieved source passage (chunk text) with its document source."""

    model_config = ConfigDict(extra="forbid")
    source: str = ""
    text: str
    rank: int


class Citation(BaseModel):
    """A provenance citation. Field-compatible superset of upstream's ``Citation``."""

    model_config = ConfigDict(extra="forbid")
    document_id: str = ""
    document_path: str = ""
    chunk_id: str = ""
    snippet: str


class EntityTypeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    description: str | None = None
    properties: list[str] = Field(default_factory=list)


class RelationTypeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    description: str | None = None
    patterns: list[tuple[str, str]] = Field(default_factory=list)


def _notes_section(notes: list[str]) -> tuple[str, list[str]]:
    return ("Notes", [f"- {_clean(n)}" for n in notes])


def _citation_line(c: Citation) -> str:
    src = c.document_path or c.document_id or "unknown"
    return f"- [Source: {_clean(src)}] {_snippet(c.snippet, 160)}"


class SearchResult(ToolResult):
    """Retrieved graph context for a query (no answer generation)."""

    query: str
    hint: str | None = None
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    cypher_rows: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def _preamble(self) -> list[str]:
        lines = [f"Query: {_clean(self.query)}"]
        if not (self.entities or self.relations or self.facts or self.cypher_rows or self.passages):
            lines.append("No relevant context found in the knowledge graph.")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            _notes_section(self.notes),
            ("Graph query rows", [f"- {_snippet(r)}" for r in self.cypher_rows]),
            ("Facts", [f"- {_snippet(f)}" for f in self.facts]),
            ("Relations", [f"- {_snippet(r)}" for r in self.relations]),
            ("Entities", [f"- {_snippet(e, 160)}" for e in self.entities]),
            (
                "Passages",
                [
                    (f"- [Source: {_clean(p.source)}] " if p.source else "- ") + _snippet(p.text)
                    for p in self.passages
                ],
            ),
        ]


class AnswerResult(ToolResult):
    """A generated answer grounded in the knowledge graph, with citations."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def _preamble(self) -> list[str]:
        return [f"Answer: {self.answer.strip()}"]

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            ("Citations", [_citation_line(c) for c in self.citations]),
            _notes_section(self.notes),
        ]


class SchemaResult(ToolResult):
    """Ontology (entity/relation types) plus live graph counts."""

    entity_types: list[EntityTypeInfo] = Field(default_factory=list)
    relation_types: list[RelationTypeInfo] = Field(default_factory=list)
    observed_entity_labels: list[str] = Field(default_factory=list)
    observed_relation_types: list[str] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0

    def _preamble(self) -> list[str]:
        return [f"Graph: {self.node_count} nodes, {self.edge_count} edges"]

    def _sections(self) -> list[tuple[str, list[str]]]:
        def ent(e: EntityTypeInfo) -> str:
            line = f"- {_clean(e.label)}"
            if e.description:
                line += f": {_snippet(e.description, 120)}"
            return line

        def rel(r: RelationTypeInfo) -> str:
            line = f"- {_clean(r.label)}"
            if r.patterns:
                line += " (" + ", ".join(f"{s}->{t}" for s, t in r.patterns[:5]) + ")"
            if r.description:
                line += f": {_snippet(r.description, 120)}"
            return line

        return [
            ("Entity types", [ent(e) for e in self.entity_types]),
            ("Relation types", [rel(r) for r in self.relation_types]),
            ("Observed entity labels", [f"- {_clean(x)}" for x in self.observed_entity_labels]),
            ("Observed relation types", [f"- {_clean(x)}" for x in self.observed_relation_types]),
        ]


class RememberResult(ToolResult):
    """Outcome of writing a text or file into the knowledge graph."""

    document_id: str
    chunks_indexed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    finalized: bool = False
    pending_finalize: int = 0
    notes: list[str] = Field(default_factory=list)

    def _preamble(self) -> list[str]:
        lines = [
            f"Stored document '{_clean(self.document_id)}': {self.chunks_indexed} chunks, "
            f"{self.nodes_created} nodes, {self.relationships_created} relationships."
        ]
        if self.finalized:
            lines.append("Graph finalized; the new knowledge is fully searchable.")
        elif self.pending_finalize:
            lines.append(
                f"{self.pending_finalize} document(s) pending finalization — call graph_flush "
                "once after your batch of writes."
            )
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [_notes_section(self.notes)]


class FlushResult(ToolResult):
    """Outcome of finalizing the graph (dedup, embeddings, indexes)."""

    entities_deduplicated: int = 0
    entities_embedded: int = 0
    relationships_embedded: int = 0

    def _preamble(self) -> list[str]:
        return [
            f"Graph finalized: {self.entities_deduplicated} entities deduplicated, "
            f"{self.entities_embedded} entities and {self.relationships_embedded} "
            "relationships embedded."
        ]


class CypherResult(ToolResult):
    """Rows returned by a read-only Cypher query."""

    query: str
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    limit_applied: bool = False

    def _preamble(self) -> list[str]:
        suffix = " (LIMIT added)" if self.limit_applied else ""
        return [f"{self.row_count} row(s){suffix}"]

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [("Rows", [f"- {_snippet(json.dumps(r, default=str))}" for r in self.rows])]


class ForgetResult(ToolResult):
    """Outcome of deleting a document from the knowledge graph."""

    document_id: str
    deleted: bool
    chunks_deleted: int = 0
    entities_deleted: int = 0

    def _preamble(self) -> list[str]:
        if not self.deleted:
            return [f"Document '{_clean(self.document_id)}' not found; nothing deleted."]
        return [
            f"Deleted document '{_clean(self.document_id)}': {self.chunks_deleted} chunks, "
            f"{self.entities_deleted} orphan entities."
        ]


def render_error(tool: str, exc: BaseException, *, max_chars: int = 500) -> str:
    """One-line, LLM-friendly error string for a failed tool call."""
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
            for err in exc.errors()
        ]
        msg = "invalid arguments — " + "; ".join(parts)
    elif isinstance(
        exc, (LLMTimeoutError, EmbeddingTimeoutError, TimeoutError, asyncio.TimeoutError)
    ):
        msg = f"timed out: {exc}" if str(exc) else "timed out"
    elif isinstance(exc, DatabaseUnavailableError):
        msg = "graph database unavailable"
    elif isinstance(exc, DocumentNotFoundError):
        msg = str(exc)
    else:
        msg = f"{type(exc).__name__}: {exc}"
    text = f"Error ({tool}): {_clean(msg)}"
    return text if len(text) <= max_chars else text[: max_chars - 1] + _ELLIPSIS
