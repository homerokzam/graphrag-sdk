"""Use a GraphRAG knowledge graph as an Agno Agent's knowledge."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from agno.knowledge.document import Document

from graphrag_sdk.integrations._core.bridge import LoopPolicy
from graphrag_sdk.integrations._core.documents import to_document_dicts
from graphrag_sdk.integrations._core.functions import as_functions
from graphrag_sdk.integrations._core.toolset import GraphRAGToolset

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT = (
    "You have access to a knowledge graph. Use the search_knowledge_base tool to look up "
    "facts, relationships and source passages before answering questions about its "
    "content, and cite sources by their name."
)


def _resolve_toolset(
    rag: GraphRAG | None, toolset: GraphRAGToolset | None, loop_policy: LoopPolicy
) -> GraphRAGToolset:
    if (rag is None) == (toolset is None):
        raise ValueError("Pass exactly one of 'rag' or 'toolset'.")
    if toolset is not None:
        return toolset
    assert rag is not None
    return GraphRAGToolset(rag, read_only=True, loop_policy=loop_policy)


class GraphRAGKnowledge:
    """Agno ``KnowledgeProtocol`` implementation backed by GraphRAG retrieval.

    Pass it as ``Agent(knowledge=GraphRAGKnowledge(rag), search_knowledge=True)``:
    the agent's built-in ``search_knowledge_base`` tool then runs GraphRAG's
    multi-path retrieval and receives graph facts plus source passages as
    documents (recorded in ``RunOutput.references``).

    Extra graph tools are not registered automatically by Agno; pass
    ``tools=knowledge.get_tools()`` (or use :class:`GraphRAGTools`) for them.
    """

    def __init__(
        self,
        rag: GraphRAG | None = None,
        *,
        toolset: GraphRAGToolset | None = None,
        max_results: int = 10,
        include_graph_sections: bool = True,
        tool_names: Sequence[str] = ("graph_search", "graph_answer", "graph_schema"),
        context_instructions: str | None = None,
        loop_policy: LoopPolicy = "dedicated",
    ) -> None:
        self._toolset = _resolve_toolset(rag, toolset, loop_policy)
        self.max_results = max_results
        self.include_graph_sections = include_graph_sections
        self._tool_names = tuple(tool_names)
        self._context = context_instructions or DEFAULT_CONTEXT
        self._warned_filters = False

    @property
    def toolset(self) -> GraphRAGToolset:
        return self._toolset

    # -- KnowledgeProtocol -------------------------------------------------

    def build_context(self, **kwargs: Any) -> str:
        """Static instructions for the system prompt (no I/O)."""
        return self._context

    def _enabled_tool_names(self) -> list[str]:
        enabled = {s.name for s in self._toolset.specs()}
        return [n for n in self._tool_names if n in enabled]

    def get_tools(self, **kwargs: Any) -> list[Callable[..., Any]]:
        """Read-only graph tools as plain callables (sync; async with ``async_mode=True``)."""
        mode: Literal["async", "sync"] = "async" if kwargs.get("async_mode") else "sync"
        return as_functions(self._toolset, mode=mode, names=self._enabled_tool_names())

    async def aget_tools(self, **kwargs: Any) -> list[Callable[..., Any]]:
        return as_functions(self._toolset, mode="async", names=self._enabled_tool_names())

    def _to_documents(self, dicts: list[dict[str, Any]]) -> list[Document]:
        return [Document(**d) for d in dicts]

    def _limit(self, max_results: int | None) -> int:
        return max_results if max_results and max_results > 0 else self.max_results

    def _note_filters(self, filters: Any) -> None:
        if filters and not self._warned_filters:
            self._warned_filters = True
            logger.debug("GraphRAGKnowledge ignores knowledge filters: %r", filters)

    def retrieve(
        self, query: str, *, max_results: int | None = None, filters: Any = None, **kwargs: Any
    ) -> list[Document]:
        """Synchronous retrieval (safe inside or outside an event loop)."""
        self._note_filters(filters)
        return self._toolset.run(self.aretrieve(query, max_results=max_results))

    async def aretrieve(
        self, query: str, *, max_results: int | None = None, filters: Any = None, **kwargs: Any
    ) -> list[Document]:
        """Async retrieval."""
        self._note_filters(filters)
        limit = self._limit(max_results)
        sr = await self._toolset.search(query, top_k=max(limit, 1))
        return self._to_documents(
            to_document_dicts(
                sr, max_results=limit, include_graph_sections=self.include_graph_sections
            )
        )


def graphrag_knowledge_retriever(
    rag: GraphRAG | None = None,
    *,
    toolset: GraphRAGToolset | None = None,
    max_results: int = 10,
    include_graph_sections: bool = True,
    mode: Literal["sync", "async"] = "sync",
    loop_policy: LoopPolicy = "dedicated",
) -> Callable[..., list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]]:
    """Build a function for ``Agent(knowledge_retriever=...)``.

    ``mode="sync"`` (default) works with both ``agent.run()`` and
    ``agent.arun()`` — Agno never awaits the retriever in ``run()``.
    ``mode="async"`` returns a coroutine function (``arun()`` only).
    """
    ts = _resolve_toolset(rag, toolset, loop_policy)

    async def _aretrieve(query: str, num_documents: int | None = None) -> list[dict[str, Any]]:
        limit = num_documents if num_documents and num_documents > 0 else max_results
        sr = await ts.search(query, top_k=max(limit, 1))
        return to_document_dicts(
            sr, max_results=limit, include_graph_sections=include_graph_sections
        )

    if mode == "async":

        async def graphrag_retriever_async(
            query: str, num_documents: int | None = None, **kwargs: Any
        ) -> list[dict[str, Any]]:
            return await _aretrieve(query, num_documents)

        return graphrag_retriever_async

    def graphrag_retriever(
        query: str, num_documents: int | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return ts.run(_aretrieve(query, num_documents))

    return graphrag_retriever
