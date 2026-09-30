# GraphRAG SDK — Integrations core: neutral "document dict" conversion
# Every framework's retriever/knowledge adapter maps these dicts onto its own
# Document type (agno ``Document(**d)``, LangChain ``Document(page_content=...)``).

from __future__ import annotations

from typing import Any

from graphrag_sdk.integrations._core.results import SearchResult


def to_document_dicts(
    sr: SearchResult,
    *,
    max_results: int,
    include_graph_sections: bool = True,
) -> list[dict[str, Any]]:
    """Flatten a ``SearchResult`` into at most ``max_results`` document dicts.

    Graph-level context (query rows, facts, relationships, entities) comes first
    as one document per section, followed by one document per source passage.
    Each dict has keys ``id``, ``name``, ``content`` and ``meta_data``.
    """
    if max_results < 1:
        return []
    docs: list[dict[str, Any]] = []
    if include_graph_sections:
        for section, items in (
            ("cypher_results", sr.cypher_rows),
            ("facts", sr.facts),
            ("relationships", sr.relations),
            ("entities", sr.entities),
        ):
            if items:
                docs.append(
                    {
                        "id": f"graph:{section}",
                        "name": f"graph:{section}",
                        "content": "\n".join(f"- {x}" for x in items),
                        "meta_data": {"source": "knowledge_graph", "section": section},
                    }
                )
    for p in sr.passages:
        name = p.source or "knowledge_graph"
        docs.append(
            {
                "id": f"{name}#{p.rank}",
                "name": name,
                "content": p.text,
                "meta_data": {"source": p.source, "section": "passages", "rank": p.rank},
            }
        )
    return docs[:max_results]
