# GraphRAG SDK — Integrations core: GraphRAGToolset
# Owns every agent-facing operation on a GraphRAG instance. Framework
# adapters (agno, and later langchain/crewai/antigravity/...) only translate
# TOOL_REGISTRY specs into their own tool objects and call acall/call_text.

from __future__ import annotations

import asyncio
import json
import logging
import os
import weakref
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from graphrag_sdk.core.exceptions import DocumentNotFoundError
from graphrag_sdk.integrations._core.bridge import AsyncBridge, LoopPolicy, bridge_for
from graphrag_sdk.integrations._core.guards import (
    DEFAULT_SUFFIXES,
    apply_limit,
    ensure_read_only_cypher,
    resolve_allowed_path,
)
from graphrag_sdk.integrations._core.parsing import answer_result_from, search_result_from
from graphrag_sdk.integrations._core.results import (
    AnswerResult,
    CypherResult,
    EntityTypeInfo,
    FlushResult,
    ForgetResult,
    RelationTypeInfo,
    RememberResult,
    SchemaResult,
    SearchResult,
    ToolResult,
    render_error,
)
from graphrag_sdk.integrations._core.specs import (
    SPECS_BY_NAME,
    FinalizePolicy,
    ToolSpec,
    select_specs,
)

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG
    from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
    from graphrag_sdk.core.models import Ontology
    from graphrag_sdk.core.providers.base import Embedder, LLMInterface
    from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

logger = logging.getLogger(__name__)
T = TypeVar("T")


class GraphRAGToolset:
    """Framework-neutral agent operations over a :class:`GraphRAG` instance.

    Args:
        rag: The GraphRAG instance to operate on.
        read_only: Disable every writing tool.
        finalize_policy: ``"manual"`` exposes ``graph_flush`` and lets the agent
            finalize after a batch of writes; ``"on_write"`` finalizes after every
            write; ``"never"`` leaves finalization to the application.
        allowed_dirs: Directories ``graph_ingest_file`` may read from. File
            ingestion is disabled when empty.
        enable_cypher: Expose the ``cypher_read`` tool.
        enable_forget: Expose the destructive ``graph_forget`` tool.
        include / exclude: Restrict tools by canonical name.
        max_output_chars: Budget for the text returned to the LLM.
        call_timeout / write_timeout: Per-call timeouts (seconds) for read and
            write tools. ``None`` disables the timeout.
        on_error: ``"return"`` turns tool failures into an error string for the
            LLM; ``"raise"`` propagates them.
        retrieval_strategy: Strategy override for search/answer.
        ingest_options: Extra keyword arguments forwarded to ``GraphRAG.ingest``
            on every write (e.g. ``loader``, ``chunker``, ``extractor``, ``resolver``).
        loop_policy: See :class:`AsyncBridge`.
        owns_rag: Close ``rag`` when this toolset is closed.
    """

    def __init__(
        self,
        rag: GraphRAG,
        *,
        read_only: bool = False,
        finalize_policy: FinalizePolicy = "manual",
        allowed_dirs: Sequence[str | os.PathLike[str]] | None = None,
        allowed_suffixes: Iterable[str] | None = None,
        max_file_bytes: int = 25_000_000,
        enable_cypher: bool = False,
        enable_forget: bool = False,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        max_output_chars: int = 4000,
        call_timeout: float | None = 120.0,
        write_timeout: float | None = None,
        on_error: Literal["return", "raise"] = "return",
        retrieval_strategy: RetrievalStrategy | None = None,
        ingest_options: Mapping[str, Any] | None = None,
        loop_policy: LoopPolicy = "dedicated",
        owns_rag: bool = False,
    ) -> None:
        if finalize_policy not in ("manual", "on_write", "never"):
            raise ValueError("finalize_policy must be 'manual', 'on_write' or 'never'")
        if on_error not in ("return", "raise"):
            raise ValueError("on_error must be 'return' or 'raise'")
        self._rag = rag
        self.read_only = read_only
        self.finalize_policy: FinalizePolicy = finalize_policy
        self._allowed_dirs = [Path(d) for d in (allowed_dirs or [])]
        self._suffixes = (
            frozenset(s.lower() if s.startswith(".") else f".{s.lower()}" for s in allowed_suffixes)
            if allowed_suffixes is not None
            else DEFAULT_SUFFIXES
        )
        self.max_file_bytes = max_file_bytes
        self.max_output_chars = max_output_chars
        self.call_timeout = call_timeout
        self.write_timeout = write_timeout
        self.on_error: Literal["return", "raise"] = on_error
        self._strategy = retrieval_strategy
        self._ingest_options = dict(ingest_options or {})
        for reserved in ("source", "text", "document_id"):
            if reserved in self._ingest_options:
                raise ValueError(f"ingest_options cannot set '{reserved}'")
        self._owns_rag = owns_rag
        self._bridge = bridge_for(rag, loop_policy)
        self._pending_finalize = 0
        self._write_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )
        self._closed = False
        self._specs = select_specs(
            read_only=read_only,
            finalize_policy=finalize_policy,
            has_allowed_dirs=bool(self._allowed_dirs),
            enable_cypher=enable_cypher,
            enable_forget=enable_forget,
            include=frozenset(include) if include is not None else None,
            exclude=frozenset(exclude) if exclude is not None else None,
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
    ) -> GraphRAGToolset:
        """Build a GraphRAG instance and a toolset that owns (and closes) it."""
        from graphrag_sdk.api.main import GraphRAG

        rag = GraphRAG(
            connection=connection,
            llm=llm,
            embedder=embedder,
            ontology=ontology,
            embedding_dimension=embedding_dimension,
        )
        kwargs.setdefault("owns_rag", True)
        try:
            return cls(rag, **kwargs)
        except BaseException:
            _close_quietly(rag)
            raise

    # -- properties --------------------------------------------------------

    @property
    def rag(self) -> GraphRAG:
        return self._rag

    @property
    def bridge(self) -> AsyncBridge:
        return self._bridge

    @property
    def pending_finalize(self) -> int:
        """Number of writes since the last finalize."""
        return self._pending_finalize

    def specs(self) -> list[ToolSpec]:
        """The enabled tool specs, in advertised order."""
        return list(self._specs)

    # -- helpers -----------------------------------------------------------

    def _write_lock(self) -> asyncio.Lock:
        # With loop_policy="dedicated" every write runs on the bridge loop, so this
        # is a single lock. With "caller", writes are serialized per event loop.
        loop = asyncio.get_running_loop()
        lock = self._write_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._write_locks[loop] = lock
        return lock

    def _pending_note(self) -> list[str]:
        if self.finalize_policy == "manual" and self._pending_finalize:
            return [
                f"{self._pending_finalize} document(s) added since the last graph_flush; "
                "entity/relationship search may be incomplete until graph_flush is called."
            ]
        return []

    async def _finalize(self) -> FlushResult:
        fr = await self._rag.finalize()
        self._pending_finalize = 0
        return FlushResult(
            entities_deduplicated=int(getattr(fr, "entities_deduplicated", 0) or 0),
            entities_embedded=int(getattr(fr, "entities_embedded", 0) or 0),
            relationships_embedded=int(getattr(fr, "relationships_embedded", 0) or 0),
        )

    async def _after_write(self, ingest_result: Any, document_id: str | None) -> RememberResult:
        self._pending_finalize += 1
        finalized = False
        notes: list[str] = []
        if self.finalize_policy == "on_write":
            try:
                await self._finalize()
                finalized = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The document IS stored; report the finalize failure without
                # making the agent retry (and duplicate) the write.
                logger.warning("finalize after write failed: %s", type(exc).__name__)
                notes.append(
                    f"The document was stored, but finalizing failed ({type(exc).__name__}); "
                    "it will be finalized on the next successful write."
                )
        info = getattr(ingest_result, "document_info", None)
        doc_id = document_id or (getattr(info, "path", None) if info else None) or ""
        return RememberResult(
            document_id=str(doc_id),
            chunks_indexed=int(getattr(ingest_result, "chunks_indexed", 0) or 0),
            nodes_created=int(getattr(ingest_result, "nodes_created", 0) or 0),
            relationships_created=int(getattr(ingest_result, "relationships_created", 0) or 0),
            finalized=finalized,
            pending_finalize=self._pending_finalize,
            notes=notes,
        )

    # -- typed async API (runs on the bridge loop) -------------------------

    async def search(self, query: str, *, top_k: int = 8) -> SearchResult:
        """Retrieve graph context for *query* without generating an answer."""

        async def _do() -> SearchResult:
            rr = await self._rag.retrieve(query, strategy=self._strategy)
            sr = search_result_from(rr, query=query, top_k=top_k)
            sr.notes.extend(self._pending_note())
            return sr

        return await self._bridge.arun(_do())

    async def answer(self, question: str) -> AnswerResult:
        """Answer *question* with retrieval + generation, including citations."""

        async def _do() -> AnswerResult:
            rag_result = await self._rag.completion(
                question, strategy=self._strategy, return_context=True
            )
            ar = answer_result_from(rag_result, question=question)
            ar.notes.extend(self._pending_note())
            return ar

        return await self._bridge.arun(_do())

    async def schema(self) -> SchemaResult:
        """Describe the ontology and graph counts."""

        async def _do() -> SchemaResult:
            ontology = await self._rag.get_ontology()
            stats = await self._rag.get_statistics()
            return SchemaResult(
                entity_types=[
                    EntityTypeInfo(
                        label=e.label,
                        description=e.description,
                        properties=[getattr(p, "name", str(p)) for p in e.properties],
                    )
                    for e in ontology.entities
                ],
                relation_types=[
                    RelationTypeInfo(
                        label=r.label,
                        description=r.description,
                        patterns=[tuple(p) for p in r.patterns],
                    )
                    for r in ontology.relations
                ],
                observed_entity_labels=[str(x) for x in stats.get("entity_types", [])],
                observed_relation_types=[str(x) for x in stats.get("relationship_types", [])],
                node_count=int(stats.get("node_count", 0) or 0),
                edge_count=int(stats.get("edge_count", 0) or 0),
            )

        return await self._bridge.arun(_do())

    async def remember(self, text: str, *, document_id: str | None = None) -> RememberResult:
        """Ingest *text* into the graph."""
        self._check_writable()

        async def _do() -> RememberResult:
            async with self._write_lock():
                result = await self._rag.ingest(
                    text=text, document_id=document_id, **self._ingest_options
                )
                return await self._after_write(result, document_id)

        return await self._bridge.arun(_do())

    async def ingest_file(self, path: str, *, document_id: str | None = None) -> RememberResult:
        """Ingest a file located inside one of the allowed directories."""
        self._check_writable()
        abs_path, default_id = resolve_allowed_path(
            path, self._allowed_dirs, max_bytes=self.max_file_bytes, suffixes=self._suffixes
        )
        doc_id = document_id or default_id

        async def _do() -> RememberResult:
            async with self._write_lock():
                result = await self._rag.ingest(
                    str(abs_path), document_id=doc_id, **self._ingest_options
                )
                return await self._after_write(result, doc_id)

        return await self._bridge.arun(_do())

    async def flush(self) -> FlushResult:
        """Finalize the graph (dedup, embeddings, indexes)."""
        self._check_writable()

        async def _do() -> FlushResult:
            async with self._write_lock():
                return await self._finalize()

        return await self._bridge.arun(_do())

    async def cypher_read(
        self, query: str, *, params_json: str | None = None, limit: int = 100
    ) -> CypherResult:
        """Run a read-only Cypher query (lexically guarded, LIMIT enforced)."""
        ensure_read_only_cypher(query)
        params: dict[str, Any] | None = None
        if params_json:
            parsed = json.loads(params_json)
            if not isinstance(parsed, dict):
                raise ValueError("params_json must be a JSON object")
            params = parsed
        final_query, limited = apply_limit(query, limit)

        async def _do() -> CypherResult:
            rows = await self._rag.query(final_query, params)
            capped = rows[:limit]
            return CypherResult(
                query=final_query,
                rows=[list(r) for r in capped],
                row_count=len(capped),
                limit_applied=limited or len(rows) > limit,
            )

        return await self._bridge.arun(_do())

    async def forget(self, document_id: str) -> ForgetResult:
        """Delete a document and its orphaned entities."""
        self._check_writable()

        async def _do() -> ForgetResult:
            async with self._write_lock():
                try:
                    res = await self._rag.delete_document(document_id, if_missing="error")
                except DocumentNotFoundError:
                    return ForgetResult(document_id=document_id, deleted=False)
            return ForgetResult(
                document_id=document_id,
                deleted=True,
                chunks_deleted=int(getattr(res, "chunks_deleted", 0) or 0),
                entities_deleted=int(getattr(res, "entities_deleted", 0) or 0),
            )

        return await self._bridge.arun(_do())

    def _check_writable(self) -> None:
        if self.read_only:
            raise PermissionError("This toolset is read-only.")

    # -- generic dispatch (used by every adapter) --------------------------

    def get_spec(self, name: str) -> ToolSpec:
        spec = SPECS_BY_NAME.get(name)
        if spec is None or spec not in self._specs:
            raise KeyError(f"Tool '{name}' is not enabled on this toolset.")
        return spec

    async def acall(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Validate *arguments* against the spec and run tool *name*."""
        spec = self.get_spec(name)
        raw = dict(arguments or {})
        fields = spec.args_model.model_fields
        for key in [k for k, v in raw.items() if v is None and k in fields]:
            if not fields[key].is_required() and fields[key].default is not None:
                del raw[key]  # models often send null for "use the default"
        args = spec.args_model.model_validate(raw)
        method = getattr(self, spec.method)
        timeout = self.write_timeout if spec.writes else self.call_timeout
        coro = method(**args.model_dump())
        if timeout is None:
            result: ToolResult = await coro
        else:
            result = await asyncio.wait_for(coro, timeout)
        return result

    async def acall_text(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """Like :meth:`acall` but return LLM text (errors rendered per ``on_error``)."""
        try:
            result = await self.acall(name, arguments)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if self.on_error == "raise":
                raise
            logger.warning("GraphRAG tool %s failed: %s", name, type(exc).__name__)
            return render_error(name, exc)
        return result.to_llm_text(max_chars=self.max_output_chars)

    def call_text(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """Synchronous :meth:`acall_text` (safe with or without a running loop)."""
        return self.run(self.acall_text(name, arguments))

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run your own coroutine for this rag on its bridge loop (sync)."""
        return self._bridge.run(coro)

    async def arun(self, coro: Coroutine[Any, Any, T]) -> T:
        """Await your own coroutine for this rag on its bridge loop."""
        return await self._bridge.arun(coro)

    # -- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        """Close the rag and its bridge loop, but only if this toolset owns the rag."""
        if self._closed:
            return
        if self._owns_rag:
            await self._bridge.arun(self._rag.close())
            if not self._bridge._on_bridge_thread():
                self._bridge.close()
        self._closed = True

    def close(self) -> None:
        """Synchronous :meth:`aclose`."""
        if self._closed:
            return
        if self._owns_rag:
            self._bridge.run(self._rag.close())
            self._bridge.close()
        self._closed = True

    async def __aenter__(self) -> GraphRAGToolset:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _close_quietly(rag: Any) -> None:
    """Best-effort close of a GraphRAG we created but could not hand over."""
    try:
        coro = rag.close()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            coro.close()  # can't block inside a running loop; the pool is lazily created
    except Exception:  # pragma: no cover - best effort
        logger.debug("closing GraphRAG after failed construction failed", exc_info=True)
