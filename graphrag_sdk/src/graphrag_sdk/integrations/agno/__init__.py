"""Agno integration for GraphRAG SDK.

Three ways to plug a knowledge graph into an ``agno.agent.Agent``:

- :class:`GraphRAGTools` — an Agno ``Toolkit`` (search / answer / schema /
  remember / ingest-file / flush tools, sync + async).
- :class:`GraphRAGKnowledge` / :func:`graphrag_knowledge_retriever` — use the
  graph as the agent's knowledge (``knowledge=`` / ``knowledge_retriever=``).
- :class:`AgnoLLM` / :class:`AgnoEmbedder` — use Agno models as GraphRAG's
  LLM and embedder.

Requires ``pip install 'graphrag-sdk[agno]'``.
"""

from graphrag_sdk.integrations.agno._compat import require_agno

require_agno()

from graphrag_sdk.integrations.agno.knowledge import (  # noqa: E402
    GraphRAGKnowledge,
    graphrag_knowledge_retriever,
)
from graphrag_sdk.integrations.agno.models import AgnoEmbedder, AgnoLLM  # noqa: E402
from graphrag_sdk.integrations.agno.toolkit import GraphRAGTools  # noqa: E402

__all__ = [
    "AgnoEmbedder",
    "AgnoLLM",
    "GraphRAGKnowledge",
    "GraphRAGTools",
    "graphrag_knowledge_retriever",
]
