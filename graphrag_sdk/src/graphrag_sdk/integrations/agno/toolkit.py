"""Agno ``Toolkit`` exposing GraphRAG tools to an Agent."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from agno.tools import Toolkit

from graphrag_sdk.integrations._core.bridge import LoopPolicy
from graphrag_sdk.integrations._core.functions import as_functions
from graphrag_sdk.integrations._core.specs import FinalizePolicy
from graphrag_sdk.integrations._core.toolset import GraphRAGToolset, _close_quietly

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG
    from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
    from graphrag_sdk.core.models import Ontology
    from graphrag_sdk.core.providers.base import Embedder, LLMInterface

DEFAULT_INSTRUCTIONS = """\
You can use a knowledge graph through the graph_* tools.
- Use graph_search to gather context (facts, relationships, source passages) and reason over it;
  use graph_answer for a direct one-shot answer. Cite sources using their [Source: ...] tags.
- Call graph_schema first if you do not know what the graph contains.
- To store new knowledge use graph_remember (or graph_ingest_file for local files). After a batch
  of writes, call graph_flush ONCE so the new knowledge becomes fully searchable.
- If a tool returns "Error (...)", read the message and fix the arguments instead of retrying
  blindly."""

_WRITE_TOOLS = ("graph_remember", "graph_ingest_file", "graph_flush")


class GraphRAGTools(Toolkit):
    """Agno toolkit over a GraphRAG knowledge graph.

    Every tool is registered twice under the same name — a sync function for
    ``agent.run()`` and a coroutine for ``agent.arun()`` — both routed through
    one :class:`GraphRAGToolset`, so graph I/O always runs on one event loop.

    Args:
        rag: The GraphRAG instance (or pass ``toolset``).
        toolset: A pre-built toolset. Its configuration decides which tools
            exist; the ``enable_*`` flags can only remove tools from it (so pass
            ``enable_cypher=True`` / ``enable_forget=True`` to keep those).
        enable_answer / enable_search / enable_schema: Read tools (default on).
        enable_write: ``graph_remember`` (+ ``graph_ingest_file`` when
            ``allowed_dirs`` is set, + ``graph_flush`` under the ``manual``
            finalize policy). Default on.
        enable_cypher: Read-only ``cypher_read`` tool (default off).
        enable_forget: Destructive ``graph_forget`` tool (default off; always
            requires confirmation).
        all: Enable every tool.
        allowed_dirs: Directories ``graph_ingest_file`` may read.
        finalize_policy: ``"manual"`` (agent calls ``graph_flush``),
            ``"on_write"`` or ``"never"``.
        tool_names: Rename tools, e.g. ``{"graph_answer": "ask_knowledge_graph"}``.
        requires_confirmation_for_writes: Ask for user confirmation before
            every write tool (Agno human-in-the-loop).
        max_output_chars: Text budget returned to the model per call.
        ingest_options: Extra ``GraphRAG.ingest`` kwargs (``extractor``, ``chunker``...).
        loop_policy: ``"dedicated"`` (default) or ``"caller"``.
        instructions / add_instructions: Toolkit instructions for the agent.
        **toolkit_kwargs: Forwarded to :class:`agno.tools.Toolkit`.
    """

    def __init__(
        self,
        rag: GraphRAG | None = None,
        *,
        toolset: GraphRAGToolset | None = None,
        enable_answer: bool = True,
        enable_search: bool = True,
        enable_schema: bool = True,
        enable_write: bool = True,
        enable_cypher: bool = False,
        enable_forget: bool = False,
        all: bool = False,
        allowed_dirs: Sequence[str | os.PathLike[str]] | None = None,
        finalize_policy: FinalizePolicy = "manual",
        tool_names: Mapping[str, str] | None = None,
        requires_confirmation_for_writes: bool = False,
        max_output_chars: int = 4000,
        ingest_options: Mapping[str, Any] | None = None,
        loop_policy: LoopPolicy = "dedicated",
        instructions: str | None = None,
        add_instructions: bool = True,
        **toolkit_kwargs: Any,
    ) -> None:
        if (rag is None) == (toolset is None):
            raise ValueError("Pass exactly one of 'rag' or 'toolset'.")
        if all:
            enable_answer = enable_search = enable_schema = enable_write = True
            enable_cypher = enable_forget = True
        exclude: list[str] = []
        if not enable_answer:
            exclude.append("graph_answer")
        if not enable_search:
            exclude.append("graph_search")
        if not enable_schema:
            exclude.append("graph_schema")
        if not enable_write:
            exclude.extend(_WRITE_TOOLS)
        if not enable_cypher:
            exclude.append("cypher_read")
        if not enable_forget:
            exclude.append("graph_forget")

        if toolset is None:
            assert rag is not None
            toolset = GraphRAGToolset(
                rag,
                finalize_policy=finalize_policy,
                allowed_dirs=allowed_dirs,
                enable_cypher=enable_cypher,
                enable_forget=enable_forget,
                exclude=exclude,
                max_output_chars=max_output_chars,
                ingest_options=ingest_options,
                loop_policy=loop_policy,
            )
            names = None
        else:
            names = [s.name for s in toolset.specs() if s.name not in exclude]
        self._toolset = toolset

        rename = dict(tool_names or {})
        sync_fns = as_functions(toolset, mode="sync", names=names, rename=rename)
        async_fns = as_functions(toolset, mode="async", names=names, rename=rename)

        confirm: list[str] = list(toolkit_kwargs.pop("requires_confirmation_tools", None) or [])
        for fn in sync_fns:
            spec = fn.__graphrag_spec__  # type: ignore[attr-defined]
            if spec.destructive or (spec.writes and requires_confirmation_for_writes):
                confirm.append(fn.__name__)

        super().__init__(
            name=toolkit_kwargs.pop("name", "graphrag_tools"),
            tools=sync_fns,
            async_tools=[(f, f.__name__) for f in async_fns],
            instructions=instructions if instructions is not None else DEFAULT_INSTRUCTIONS,
            add_instructions=add_instructions,
            requires_confirmation_tools=confirm or None,
            **toolkit_kwargs,
        )

    @classmethod
    def from_config(
        cls,
        connection: ConnectionConfig | FalkorDBConnection,
        *,
        llm: LLMInterface,
        embedder: Embedder,
        ontology: Ontology | None = None,
        embedding_dimension: int = 256,
        **kwargs: Any,
    ) -> GraphRAGTools:
        """Create the GraphRAG instance too; :meth:`close` then closes it."""
        from graphrag_sdk.api.main import GraphRAG

        rag = GraphRAG(
            connection=connection,
            llm=llm,
            embedder=embedder,
            ontology=ontology,
            embedding_dimension=embedding_dimension,
        )
        try:
            tools = cls(rag, **kwargs)
        except BaseException:
            _close_quietly(rag)
            raise
        tools._toolset._owns_rag = True
        return tools

    @property
    def toolset(self) -> GraphRAGToolset:
        """The underlying framework-neutral toolset (share it with GraphRAGKnowledge)."""
        return self._toolset

    def close(self) -> None:
        """Close the GraphRAG instance if this toolkit created it."""
        self._toolset.close()

    async def aclose(self) -> None:
        """Async :meth:`close`."""
        await self._toolset.aclose()
