"""Agno adapter against a real FalkorDB (RUN_INTEGRATION=1).

Uses the deterministic ScriptedExtractor + MockLLM, so no API key is needed.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

pytest.importorskip("agno")

from graphrag_sdk.ingestion.resolution_strategies.exact_match import (  # noqa: E402
    ExactMatchResolution,
)
from graphrag_sdk.integrations import GraphRAGToolset  # noqa: E402
from graphrag_sdk.integrations.agno import (  # noqa: E402
    GraphRAGKnowledge,
    GraphRAGTools,
)

from .conftest import MockEmbedder, MockLLM  # noqa: E402
from .test_cached_chunk_extraction_edges import ScriptedExtractor  # noqa: E402

pytestmark = pytest.mark.integration

DOC = (
    "E|Ada Lovelace|Person|Mathematician who wrote the first program\n"
    "E|Analytical Engine|Machine|Mechanical computer designed by Babbage\n"
    "R|Ada Lovelace|Analytical Engine|WROTE_FOR|Ada wrote a program for the engine"
)


def _options() -> dict[str, object]:
    return {"extractor": ScriptedExtractor(), "resolver": ExactMatchResolution()}


async def test_caller_policy_roundtrip(real_falkordb_rag_factory) -> None:  # type: ignore[no-untyped-def]
    rag = real_falkordb_rag_factory(llm=MockLLM(), resolver=ExactMatchResolution())
    ts = GraphRAGToolset(rag, loop_policy="caller", ingest_options=_options())
    out = await ts.acall_text("graph_remember", {"text": DOC, "document_id": "notes/ada.md"})
    assert "Stored document 'notes/ada.md'" in out, out
    flushed = await ts.acall_text("graph_flush")
    assert flushed.startswith("Graph finalized"), flushed
    schema = await ts.acall_text("graph_schema")
    assert "Person" in schema, schema
    found = await ts.acall_text("graph_search", {"query": "Ada Lovelace program"})
    assert "[Source: notes/ada.md]" in found, found
    k = GraphRAGKnowledge(toolset=ts)
    docs = await k.aretrieve("Ada Lovelace program")
    assert any(d.name == "notes/ada.md" for d in docs)


def test_dedicated_loop_sync_agent_path() -> None:
    """Sync agno tool calls from plain code through the dedicated loop."""
    if not os.getenv("RUN_INTEGRATION"):
        pytest.skip("Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests")
    from graphrag_sdk.core.connection import ConnectionConfig

    config = ConnectionConfig(
        host=os.getenv("FALKOR_HOST", "localhost"),
        port=int(os.getenv("FALKOR_PORT", "6379")),
        username=os.getenv("FALKOR_USERNAME") or None,
        password=os.getenv("FALKOR_PASSWORD") or None,
        graph_name=f"test_agno_{uuid4().hex[:8]}",
    )
    embedder = MockEmbedder()
    tools = GraphRAGTools.from_config(
        config,
        llm=MockLLM(),
        embedder=embedder,
        embedding_dimension=embedder.dimension,
        ingest_options=_options(),
    )
    try:
        fns = tools.functions
        assert "Stored document" in fns["graph_remember"].entrypoint(text=DOC, document_id="d1")
        # Second call reuses the same loop / connection pool (no "event loop is closed").
        assert "Stored document" in fns["graph_remember"].entrypoint(
            text=DOC.replace("Ada", "Grace"), document_id="d2"
        )
        assert fns["graph_flush"].entrypoint().startswith("Graph finalized")
        out = fns["graph_search"].entrypoint(query="Ada Lovelace")
        assert "[Source: d" in out, out
    finally:
        tools.toolset.run(tools.toolset.rag.delete_all())
        tools.close()
