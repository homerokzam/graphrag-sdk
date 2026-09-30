# GraphRAG SDK — Integrations core: RetrieverResult/RagResult -> typed results
# Parses the section-level markdown produced by MultiPathRetrieval. Passages
# carry provenance inline as ``[Source: <document path>]``. Items without a
# ``section`` (e.g. LocalRetrieval / custom strategies) are treated as passages.

from __future__ import annotations

import re
from typing import Any

from graphrag_sdk.core.models import RagResult, RetrieverResult
from graphrag_sdk.integrations._core.results import (
    AnswerResult,
    Citation,
    Passage,
    SearchResult,
)

_SOURCE_RE = re.compile(r"^\[Source: (?P<src>[^\]\n]+)\]\n?")
_PASSAGE_SEP = "\n---\n"
_SECTION_HEADERS = {
    "passages": "## Source Document Passages",
    "entities": "## Key Entities",
    "relationships": "## Entity Relationships",
    "facts": "## Knowledge Graph Facts",
    "cypher_results": "## Graph Query Results",
}


def _strip_header(text: str, section: str) -> str:
    header = _SECTION_HEADERS.get(section)
    if header and text.startswith(header):
        text = text[len(header) :]
        if section == "cypher_results":
            # Drop the explanatory lines that follow the heading, keep "- row" lines.
            lines = [ln for ln in text.split("\n") if ln.startswith("- ")]
            return "\n".join(lines)
    return text.lstrip("\n")


def _bullets(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        out.append(line[2:] if line.startswith("- ") else line)
    return out


def parse_passages(section_text: str) -> list[Passage]:
    """Split a ``passages`` section into ``Passage`` objects.

    Passages are joined with ``\\n---\\n``. A piece is only treated as a new
    passage boundary if it starts with ``[Source:`` or if no passage in the
    section is tagged at all, so a literal ``---`` inside chunk text (a Markdown
    horizontal rule, which is common) survives. Trade-off: in the rare mixed
    case where a chunk has no known document, its untagged text is attached to
    the preceding passage. Retrieval ``provenance`` metadata, when present,
    bypasses this parser entirely.
    """
    body = _strip_header(section_text, "passages")
    if not body.strip():
        return []
    pieces = body.split(_PASSAGE_SEP)
    tagged = any(_SOURCE_RE.match(p) for p in pieces)
    merged: list[str] = []
    for piece in pieces:
        if merged and tagged and not _SOURCE_RE.match(piece):
            merged[-1] = merged[-1] + _PASSAGE_SEP + piece
        else:
            merged.append(piece)
    passages: list[Passage] = []
    for rank, piece in enumerate(merged, start=1):
        m = _SOURCE_RE.match(piece)
        source = m.group("src").strip() if m else ""
        text = piece[m.end() :] if m else piece
        text = text.strip()
        if text:
            passages.append(Passage(source=source, text=text, rank=rank))
    return passages


def _passages_from_provenance(chunks: list[dict[str, Any]]) -> list[Passage]:
    out: list[Passage] = []
    for rank, c in enumerate(chunks, start=1):
        text = str(c.get("text", "")).strip()
        if text:
            src = str(c.get("document_path") or c.get("document_id") or "")
            out.append(Passage(source=src, text=text, rank=rank))
    return out


def citations_from_passages(passages: list[Passage], *, limit: int | None = None) -> list[Citation]:
    """One citation per distinct source, in rank order."""
    seen: set[str] = set()
    out: list[Citation] = []
    for p in passages:
        if not p.source or p.source in seen:
            continue
        seen.add(p.source)
        out.append(Citation(document_id=p.source, document_path=p.source, snippet=p.text[:240]))
        if limit is not None and len(out) >= limit:
            break
    return out


def search_result_from(rr: RetrieverResult | None, *, query: str, top_k: int = 8) -> SearchResult:
    """Convert a ``RetrieverResult`` into a typed ``SearchResult``."""
    result = SearchResult(query=query)
    if rr is None:
        return result
    passages: list[Passage] = []
    provenance = rr.metadata.get("provenance") if isinstance(rr.metadata, dict) else None
    if isinstance(provenance, dict) and isinstance(provenance.get("chunks"), list):
        passages = _passages_from_provenance(provenance["chunks"])
    for item in rr.items:
        section = str(item.metadata.get("section", "")) if item.metadata else ""
        content = item.content or ""
        if section == "hint":
            result.hint = content.strip() or None
        elif section == "entities":
            result.entities.extend(_bullets(_strip_header(content, section)))
        elif section == "relationships":
            result.relations.extend(_bullets(_strip_header(content, section)))
        elif section == "facts":
            result.facts.extend(_bullets(_strip_header(content, section)))
        elif section == "cypher_results":
            result.cypher_rows.extend(_bullets(_strip_header(content, section)))
        elif section == "passages":
            if not provenance:
                passages.extend(parse_passages(content))
        elif content.strip():
            # Unsectioned items (non-MultiPath strategies): each item is a passage.
            m = _SOURCE_RE.match(content)
            meta = item.metadata or {}
            src = (
                m.group("src").strip()
                if m
                else str(meta.get("document_path") or meta.get("source") or "")
            )
            text = (content[m.end() :] if m else content).strip()
            passages.append(Passage(source=src, text=text, rank=len(passages) + 1))
    result.passages = passages[:top_k]
    result.facts = result.facts[:top_k]
    result.citations = citations_from_passages(result.passages)
    return result


def answer_result_from(rag: RagResult, *, question: str, max_citations: int = 8) -> AnswerResult:
    """Convert a ``RagResult`` into a typed ``AnswerResult`` with citations."""
    citations: list[Citation] = []
    if rag.retriever_result is not None:
        sr = search_result_from(rag.retriever_result, query=question, top_k=max(max_citations, 15))
        citations = citations_from_passages(sr.passages, limit=max_citations)
    return AnswerResult(question=question, answer=rag.answer, citations=citations)
