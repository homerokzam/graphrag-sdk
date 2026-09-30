"""Tests for the framework-neutral agent-integration core (no framework needed)."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
import typing
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphrag_sdk import GraphRAG
from graphrag_sdk.core.exceptions import DocumentNotFoundError, LLMTimeoutError
from graphrag_sdk.core.models import (
    ChatMessage,
    DeleteDocumentResult,
    Entity,
    FinalizeResult,
    IngestionResult,
    LLMResponse,
    Ontology,
    RagResult,
    Relation,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.integrations import (
    TOOL_REGISTRY,
    AsyncBridge,
    CallableEmbedder,
    CallableLLM,
    GraphRAGToolset,
    PathNotAllowedError,
    ReadOnlyViolation,
    SearchResult,
    as_functions,
    bridge_for,
    select_specs,
    to_document_dicts,
)
from graphrag_sdk.integrations._core.guards import (
    apply_limit,
    ensure_read_only_cypher,
    resolve_allowed_path,
)
from graphrag_sdk.integrations._core.models import strip_json_fences
from graphrag_sdk.integrations._core.parsing import (
    answer_result_from,
    parse_passages,
    search_result_from,
)
from graphrag_sdk.integrations._core.results import render_error

# ── fixtures ────────────────────────────────────────────────────────────────


def _rr() -> RetrieverResult:
    return RetrieverResult(
        items=[
            RetrieverResultItem(
                content="Answer format: Name the person.", metadata={"section": "hint"}
            ),
            RetrieverResultItem(
                content="## Graph Query Results\nRows a graph query returned for the question: "
                "'q'. Any aggregate below is already scoped to that question.\n- count: 3",
                metadata={"section": "cypher_results"},
            ),
            RetrieverResultItem(
                content="## Key Entities\n- Ada: mathematician\n- Babbage",
                metadata={"section": "entities"},
            ),
            RetrieverResultItem(
                content="## Entity Relationships\n- Ada -[KNOWS]-> Babbage",
                metadata={"section": "relationships"},
            ),
            RetrieverResultItem(
                content="## Knowledge Graph Facts\n- Ada wrote the notes\n- Babbage built it",
                metadata={"section": "facts"},
            ),
            RetrieverResultItem(
                content="## Source Document Passages\n[Source: docs/ada.md]\nAda text\n---\n"
                "still ada\n---\n[Source: b.pdf]\nBabbage text\n---\n[Source: docs/ada.md]\nmore",
                metadata={"section": "passages"},
            ),
        ]
    )


def _mock_rag() -> MagicMock:
    rag = MagicMock(spec=GraphRAG)
    rag.retrieve = AsyncMock(return_value=_rr())
    rag.completion = AsyncMock(return_value=RagResult(answer="Ada.", retriever_result=_rr()))
    rag.ingest = AsyncMock(
        return_value=IngestionResult(chunks_indexed=2, nodes_created=3, relationships_created=1)
    )
    rag.finalize = AsyncMock(
        return_value=FinalizeResult(entities_deduplicated=1, entities_embedded=4)
    )
    rag.get_ontology = AsyncMock(
        return_value=Ontology(
            entities=[Entity(label="Person", description="A human")],
            relations=[Relation(label="KNOWS", patterns=[("Person", "Person")])],
        )
    )
    rag.get_statistics = AsyncMock(
        return_value={"node_count": 5, "edge_count": 7, "entity_types": ["Person"]}
    )
    rag.query = AsyncMock(return_value=[[1, "a"]])
    rag.delete_document = AsyncMock(
        return_value=DeleteDocumentResult(document_uid="d", chunks_deleted=2)
    )
    rag.close = AsyncMock()
    return rag


# ── specs ──────────────────────────────────────────────────────────────────


class TestSpecs:
    def test_names_unique_and_snake_case(self) -> None:
        names = [s.name for s in TOOL_REGISTRY]
        assert len(names) == len(set(names))
        assert all(n.replace("_", "").isalnum() and n == n.lower() for n in names)

    @pytest.mark.parametrize("spec", TOOL_REGISTRY, ids=lambda s: s.name)
    def test_docstring_and_schema(self, spec) -> None:  # type: ignore[no-untyped-def]
        doc = spec.docstring()
        for field in spec.args_model.model_fields:
            assert f"    {field}:" in doc
        schema = spec.input_schema
        assert schema["additionalProperties"] is False
        assert "$ref" not in str(schema) and "title" not in schema

    def test_default_selection(self) -> None:
        assert [s.name for s in select_specs()] == [
            "graph_search",
            "graph_answer",
            "graph_schema",
            "graph_remember",
            "graph_flush",
        ]

    def test_gating(self) -> None:
        names = {s.name for s in select_specs(read_only=True, enable_cypher=True)}
        assert names == {"graph_search", "graph_answer", "graph_schema", "cypher_read"}
        names = {s.name for s in select_specs(finalize_policy="on_write", has_allowed_dirs=True)}
        assert "graph_flush" not in names and "graph_ingest_file" in names
        assert {s.name for s in select_specs(enable_forget=True)} >= {"graph_forget"}
        assert [s.name for s in select_specs(include=frozenset({"graph_answer"}))] == [
            "graph_answer"
        ]
        with pytest.raises(ValueError, match="Unknown tool"):
            select_specs(exclude=frozenset({"nope"}))


# ── parsing / results / documents ───────────────────────────────────────────


class TestParsing:
    def test_passages_keep_inner_separator(self) -> None:
        ps = parse_passages(_rr().items[-1].content)
        assert [p.source for p in ps] == ["docs/ada.md", "b.pdf", "docs/ada.md"]
        assert ps[0].text == "Ada text\n---\nstill ada"
        assert [p.rank for p in ps] == [1, 2, 3]

    def test_untagged_passages_split(self) -> None:
        ps = parse_passages("## Source Document Passages\none\n---\ntwo")
        assert [(p.source, p.text) for p in ps] == [("", "one"), ("", "two")]

    def test_all_sections(self) -> None:
        sr = search_result_from(_rr(), query="q", top_k=8)
        assert sr.hint == "Answer format: Name the person."
        assert sr.cypher_rows == ["count: 3"]
        assert sr.entities == ["Ada: mathematician", "Babbage"]
        assert sr.relations == ["Ada -[KNOWS]-> Babbage"]
        assert sr.facts == ["Ada wrote the notes", "Babbage built it"]
        assert [c.document_path for c in sr.citations] == ["docs/ada.md", "b.pdf"]

    def test_top_k_and_empty(self) -> None:
        assert len(search_result_from(_rr(), query="q", top_k=1).passages) == 1
        empty = search_result_from(RetrieverResult(), query="q")
        assert "No relevant context" in empty.to_llm_text()
        assert search_result_from(None, query="q").passages == []

    def test_unsectioned_items_are_passages(self) -> None:
        rr = RetrieverResult(
            items=[
                RetrieverResultItem(content="plain chunk", metadata={"chunk_id": "c1"}),
                RetrieverResultItem(content="[Source: x.md]\ntagged"),
            ]
        )
        sr = search_result_from(rr, query="q")
        assert [(p.source, p.text) for p in sr.passages] == [
            ("", "plain chunk"),
            ("x.md", "tagged"),
        ]

    def test_provenance_preferred(self) -> None:
        rr = _rr()
        rr.metadata["provenance"] = {
            "chunks": [{"chunk_id": "c", "document_path": "p.md", "text": "prov text"}]
        }
        sr = search_result_from(rr, query="q")
        assert [(p.source, p.text) for p in sr.passages] == [("p.md", "prov text")]

    def test_answer_result(self) -> None:
        ar = answer_result_from(RagResult(answer="Ada.", retriever_result=_rr()), question="who")
        assert ar.answer == "Ada." and len(ar.citations) == 2
        assert ar.to_llm_text().startswith("Answer: Ada.")
        assert answer_result_from(RagResult(answer="x"), question="q").citations == []

    def test_document_dicts(self) -> None:
        sr = search_result_from(_rr(), query="q")
        docs = to_document_dicts(sr, max_results=10)
        assert [d["name"] for d in docs][:4] == [
            "graph:cypher_results",
            "graph:facts",
            "graph:relationships",
            "graph:entities",
        ]
        assert docs[4]["name"] == "docs/ada.md" and docs[4]["meta_data"]["rank"] == 1
        assert len(to_document_dicts(sr, max_results=2)) == 2
        only = to_document_dicts(sr, max_results=10, include_graph_sections=False)
        assert all(d["meta_data"]["section"] == "passages" for d in only)
        assert to_document_dicts(sr, max_results=0) == []


class TestRendering:
    def test_budget_and_marker(self) -> None:
        sr = SearchResult(query="q", facts=[f"fact number {i}" for i in range(50)])
        text = sr.to_llm_text(max_chars=200)
        assert len(text) <= 200
        assert "more)" in text
        assert text == sr.to_llm_text(max_chars=200)

    def test_to_dict_json_safe(self) -> None:
        import json

        json.dumps(search_result_from(_rr(), query="q").to_dict())

    def test_render_error(self) -> None:
        from graphrag_sdk.integrations._core.specs import SearchArgs

        with pytest.raises(Exception) as ei:
            SearchArgs.model_validate({"query": ""})
        assert render_error("graph_search", ei.value).startswith(
            "Error (graph_search): invalid arguments — query:"
        )
        assert "timed out" in render_error("t", LLMTimeoutError("x"))
        assert render_error("t", ValueError("boom")) == "Error (t): ValueError: boom"
        assert len(render_error("t", ValueError("x" * 2000), max_chars=100)) == 100


# ── guards ───────────────────────────────────────────────────────────────


class TestPathGuard:
    def test_allowed_relative_and_absolute(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        f = tmp_path / "sub" / "a.md"
        f.write_text("hi")
        p, doc_id = resolve_allowed_path("sub/a.md", [tmp_path], max_bytes=100)
        assert p == f.resolve() and doc_id == "sub/a.md"
        assert resolve_allowed_path(str(f), [tmp_path], max_bytes=100)[1] == "sub/a.md"

    def test_rejections(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret.md"
        outside.write_text("x")
        (root / "big.md").write_text("x" * 50)
        (root / "a.exe").write_text("x")
        (root / "link.md").symlink_to(outside)
        cases = [
            "../secret.md",
            str(outside),
            "link.md",
            "https://example.com/a.md",
            "missing.md",
            "a.exe",
            "big.md",
            "bad\x00.md",
        ]
        for case in cases:
            with pytest.raises(PathNotAllowedError):
                resolve_allowed_path(case, [root], max_bytes=10)
        with pytest.raises(PathNotAllowedError, match="disabled"):
            resolve_allowed_path("a.md", [], max_bytes=10)


class TestCypherGuard:
    @pytest.mark.parametrize(
        "q",
        [
            "MATCH (n) RETURN n",
            "MATCH (n) WHERE n.name = 'CREATE me' RETURN n",
            "CALL db.labels()",
            "OPTIONAL MATCH (n) RETURN count(n)",
        ],
    )
    def test_allowed(self, q: str) -> None:
        ensure_read_only_cypher(q)

    @pytest.mark.parametrize(
        "q",
        [
            "CREATE (n)",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.x = 1",
            "MATCH (n) Cr/**/eate (m)",
            "MATCH (n) RETURN n; MATCH (m) DELETE m",
            "CALL db.idx.fulltext.createNodeIndex('L','p')",
            "LOAD CSV FROM 'x' AS r RETURN r",
            "",
        ],
    )
    def test_rejected(self, q: str) -> None:
        with pytest.raises(ReadOnlyViolation):
            ensure_read_only_cypher(q)

    def test_apply_limit(self) -> None:
        assert apply_limit("MATCH (n) RETURN n;", 5) == ("MATCH (n) RETURN n\nLIMIT 5", True)
        assert apply_limit("MATCH (n) RETURN n LIMIT 2", 5)[1] is False
        assert apply_limit("MATCH (n) WHERE n.x='LIMIT' RETURN n", 5)[1] is True


# ── bridge ───────────────────────────────────────────────────────────────


async def _loop_id() -> int:
    return id(asyncio.get_running_loop())


class TestBridge:
    def test_run_without_loop_uses_one_loop(self) -> None:
        b = AsyncBridge()
        try:
            assert b.run(_loop_id()) == b.run(_loop_id())
        finally:
            b.close()

    async def test_run_and_arun_inside_running_loop(self) -> None:
        b = AsyncBridge()
        try:
            here = id(asyncio.get_running_loop())
            via_sync = await asyncio.to_thread(b.run, _loop_id())
            assert b.run(_loop_id()) == via_sync != here
            assert await b.arun(_loop_id()) == via_sync
        finally:
            b.close()

    def test_reentrancy_guard(self) -> None:
        b = AsyncBridge()

        async def inner() -> None:
            b.run(_loop_id())

        try:
            with pytest.raises(RuntimeError, match="deadlock"):
                b.run(inner())
        finally:
            b.close()

    async def test_cancellation_propagates(self) -> None:
        b = AsyncBridge()
        started = threading.Event()
        cancelled = threading.Event()

        async def slow() -> None:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        try:
            task = asyncio.create_task(b.arun(slow()))
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(cancelled.wait, 2)
        finally:
            b.close()

    def test_close_idempotent_and_closed_raises(self) -> None:
        b = AsyncBridge()
        b.run(_loop_id())
        b.close()
        b.close()
        coro = _loop_id()
        with pytest.raises(RuntimeError, match="closed"):
            b.run(coro)
        coro.close()

    async def test_caller_policy(self) -> None:
        b = AsyncBridge("caller")
        assert await b.arun(_loop_id()) == id(asyncio.get_running_loop())
        with pytest.raises(RuntimeError, match="loop_policy"):
            b.run(_loop_id())
        assert isinstance(await asyncio.to_thread(b.run, _loop_id()), int)

    def test_bridge_for_shared(self) -> None:
        rag = _mock_rag()
        assert bridge_for(rag) is bridge_for(rag)
        with pytest.raises(ValueError, match="loop_policy"):
            bridge_for(rag, "caller")
        bridge_for(rag).close()


# ── toolset ──────────────────────────────────────────────────────────────


class TestToolset:
    async def test_search_answer_schema(self) -> None:
        rag = _mock_rag()
        ts = GraphRAGToolset(rag)
        sr = await ts.search("who", top_k=3)
        assert rag.retrieve.await_args.args == ("who",) and len(sr.passages) == 3
        ar = await ts.answer("who?")
        assert rag.completion.await_args.kwargs["return_context"] is True
        assert ar.answer == "Ada."
        sc = await ts.schema()
        assert sc.node_count == 5 and sc.entity_types[0].label == "Person"
        assert sc.relation_types[0].patterns == [("Person", "Person")]

    async def test_writes_pending_note_and_flush(self) -> None:
        rag = _mock_rag()
        ts = GraphRAGToolset(rag)
        r = await ts.remember("fact", document_id="n1")
        rag.ingest.assert_awaited_with(text="fact", document_id="n1")
        assert r.document_id == "n1" and r.pending_finalize == 1 and not r.finalized
        text = await ts.acall_text("graph_search", {"query": "q"})
        assert "graph_flush" in text
        fr = await ts.flush()
        assert fr.entities_embedded == 4 and ts.pending_finalize == 0
        assert "graph_flush" not in await ts.acall_text("graph_search", {"query": "q"})

    async def test_on_write_policy(self) -> None:
        rag = _mock_rag()
        ts = GraphRAGToolset(rag, finalize_policy="on_write")
        r = await ts.remember("fact")
        assert r.finalized and rag.finalize.await_count == 1
        assert "graph_flush" not in {s.name for s in ts.specs()}

    async def test_ingest_file(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("hello")
        rag = _mock_rag()
        ts = GraphRAGToolset(rag, allowed_dirs=[tmp_path])
        r = await ts.ingest_file("a.md")
        assert rag.ingest.await_args.args == (str((tmp_path / "a.md").resolve()),)
        assert r.document_id == "a.md"
        text = await ts.acall_text("graph_ingest_file", {"path": "../x.md"})
        assert text.startswith("Error (graph_ingest_file)")

    async def test_write_lock_serializes(self) -> None:
        rag = _mock_rag()
        active = 0
        peak = 0

        async def slow_ingest(**kwargs: object) -> IngestionResult:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return IngestionResult()

        rag.ingest = AsyncMock(side_effect=slow_ingest)
        ts = GraphRAGToolset(rag)
        await asyncio.gather(*(ts.remember(f"t{i}") for i in range(5)))
        assert peak == 1 and ts.pending_finalize == 5

    async def test_cypher_and_forget(self) -> None:
        rag = _mock_rag()
        ts = GraphRAGToolset(rag, enable_cypher=True, enable_forget=True)
        cr = await ts.cypher_read("MATCH (n) RETURN n", params_json='{"a": 1}', limit=5)
        assert rag.query.await_args.args == ("MATCH (n) RETURN n\nLIMIT 5", {"a": 1})
        assert cr.limit_applied and cr.row_count == 1
        assert "not allowed" in await ts.acall_text("cypher_read", {"query": "MATCH (n) SET n.x=1"})
        assert (await ts.forget("d")).deleted
        rag.delete_document = AsyncMock(side_effect=DocumentNotFoundError("nope"))
        assert not (await ts.forget("d")).deleted

    async def test_read_only_and_disabled_tools(self) -> None:
        ts = GraphRAGToolset(_mock_rag(), read_only=True)
        with pytest.raises(PermissionError):
            await ts.remember("x")
        assert "not enabled" in await ts.acall_text("graph_remember", {"text": "x"})

    async def test_timeout_and_on_error(self) -> None:
        rag = _mock_rag()

        async def hang(*a: object, **k: object) -> None:
            await asyncio.sleep(5)

        rag.retrieve = AsyncMock(side_effect=hang)
        ts = GraphRAGToolset(rag, call_timeout=0.05)
        assert "timed out" in await ts.acall_text("graph_search", {"query": "q"})
        raising = GraphRAGToolset(_mock_rag(), on_error="raise")
        with pytest.raises(Exception):
            await raising.acall_text("graph_search", {"query": ""})

    def test_sync_call_text(self) -> None:
        ts = GraphRAGToolset(_mock_rag())
        assert ts.call_text("graph_answer", {"question": "who"}).startswith("Answer: Ada.")

    async def test_ownership(self) -> None:
        rag = _mock_rag()
        await GraphRAGToolset(rag).aclose()
        rag.close.assert_not_awaited()
        owned = GraphRAGToolset(_mock_rag(), owns_rag=True)
        await owned.aclose()
        owned.rag.close.assert_awaited_once()  # type: ignore[attr-defined]
        sync_owned = GraphRAGToolset(_mock_rag(), owns_rag=True)
        sync_owned.close()
        sync_owned.rag.close.assert_awaited_once()  # type: ignore[attr-defined]

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            GraphRAGToolset(_mock_rag(), finalize_policy="sometimes")  # type: ignore[arg-type]


# ── functions export ─────────────────────────────────────────────────────


class TestFunctions:
    def test_metadata(self) -> None:
        ts = GraphRAGToolset(_mock_rag())
        fns = as_functions(ts, rename={"graph_answer": "ask_knowledge_graph"})
        by_name = {f.__name__: f for f in fns}
        assert set(by_name) == {
            "graph_search",
            "ask_knowledge_graph",
            "graph_schema",
            "graph_remember",
            "graph_flush",
        }
        search = by_name["graph_search"]
        assert inspect.iscoroutinefunction(search)
        sig = inspect.signature(search)
        assert list(sig.parameters) == ["query", "top_k"]
        assert sig.parameters["top_k"].default == 8
        hints = typing.get_type_hints(search)
        assert hints == {"query": str, "top_k": int, "return": str}
        assert "Args:\n    query:" in (search.__doc__ or "")
        remember = by_name["graph_remember"]
        assert inspect.signature(remember).parameters["document_id"].default is None

    async def test_async_and_sync_execution(self) -> None:
        ts = GraphRAGToolset(_mock_rag())
        (afn,) = as_functions(ts, names=["graph_answer"])
        assert (await afn("who")).startswith("Answer: Ada.")
        (sfn,) = as_functions(ts, mode="sync", names=["graph_answer"])
        assert (await asyncio.to_thread(sfn, question="who")).startswith("Answer: Ada.")
        (dfn,) = as_functions(ts, names=["graph_answer"], output="dict")
        assert (await dfn(question="who"))["answer"] == "Ada."

    def test_errors(self) -> None:
        ts = GraphRAGToolset(_mock_rag())
        with pytest.raises(KeyError):
            as_functions(ts, names=["cypher_read"])
        with pytest.raises(KeyError):
            as_functions(ts, rename={"cypher_read": "x"})


# ── generic model adapters ─────────────────────────────────────────────


class _Out(BaseModel):
    name: str


class TestCallableModels:
    async def test_llm_messages_native(self) -> None:
        seen: list[list[ChatMessage]] = []

        def chat(msgs: list[ChatMessage]) -> str:
            seen.append(msgs)
            return "hi"

        llm = CallableLLM(model_name="m", chat=chat)
        msgs = [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")]
        assert (await llm.ainvoke_messages(msgs)).content == "hi"
        assert [m.role for m in seen[0]] == ["system", "user"]
        assert llm.invoke("p").content == "hi"
        assert (await llm.ainvoke("p", timeout=5)).content == "hi"

    async def test_llm_retries_and_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        calls = 0

        def flaky(msgs: list[ChatMessage]) -> LLMResponse:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("transient")
            return LLMResponse(content="ok")

        assert (await CallableLLM(model_name="m", chat=flaky).ainvoke("p")).content == "ok"
        assert calls == 3

        async def slow(msgs: list[ChatMessage]) -> str:
            await asyncio.wait_for(asyncio.Event().wait(), 1)
            return "late"

        llm = CallableLLM(model_name="m", chat=lambda m: "x", achat=slow, prefer_async=True)
        monkeypatch.undo()
        with pytest.raises(LLMTimeoutError):
            await llm.ainvoke("p", timeout=0.05)

    async def test_structured(self) -> None:
        llm = CallableLLM(model_name="m", chat=lambda m: '```json\n{"name": "Ada"}\n```')
        assert (await llm.ainvoke_with_model("p", _Out)).name == "Ada"  # type: ignore[attr-defined]
        native = CallableLLM(
            model_name="m", chat=lambda m: "", structured=lambda m, r: _Out(name="B")
        )
        assert native.invoke_with_model("p", _Out).name == "B"  # type: ignore[attr-defined]
        assert strip_json_fences(" {} ") == "{}"

    async def test_embedder(self) -> None:
        emb = CallableEmbedder(model_name="e", embed=lambda t: [1.0, 2.0], dimensions=2)
        assert emb.embed_query("x") == [1.0, 2.0]
        assert await emb.aembed_documents(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
        with pytest.raises(ValueError, match="dimensions"):
            CallableEmbedder(model_name="e", embed=lambda t: [1.0], dimensions=2).embed_query("x")
        with pytest.raises(ValueError, match="empty"):
            CallableEmbedder(model_name="e", embed=lambda t: []).embed_query("x")

        async def aembed(t: str) -> list[float]:
            return [3.0]

        native = CallableEmbedder(
            model_name="e", embed=lambda t: [0.0], aembed=aembed, prefer_async=True
        )
        assert await native.aembed_query("x") == [3.0]
        assert await native.aembed_documents(["a"]) == [[3.0]]

    def test_graphrag_construction(self, mock_connection) -> None:  # type: ignore[no-untyped-def]
        from graphrag_sdk import ConnectionConfig

        mock_connection.config = ConnectionConfig()
        rag = GraphRAG(
            connection=mock_connection,
            llm=CallableLLM(model_name="m", chat=lambda m: "x"),
            embedder=CallableEmbedder(model_name="e", embed=lambda t: [0.0] * 8),
            embedding_dimension=8,
        )
        assert rag is not None


def test_integrations_import_does_not_import_frameworks() -> None:
    import subprocess
    import sys

    code = (
        "import sys, graphrag_sdk.integrations; "
        "print(any(m == 'agno' or m.startswith('agno.') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ)
    )
    assert out.stdout.strip() == "False", out.stderr


# ── regressions from review ─────────────────────────────────────────────


class TestReviewRegressions:
    def test_close_cancels_blocked_callers(self) -> None:
        b = AsyncBridge()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                b.run(asyncio.sleep(30))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        import time

        time.sleep(0.2)
        b.close()
        t.join(5)
        assert not t.is_alive()
        assert errors and isinstance(errors[0], (asyncio.CancelledError, Exception))

    @pytest.mark.parametrize(
        "q",
        [
            "MATCH (n) RETURN n.limit",
            "MATCH (n) CALL { WITH n MATCH (n)--(m) RETURN m LIMIT 1 } RETURN n, m",
        ],
    )
    def test_limit_not_fooled(self, q: str) -> None:
        assert apply_limit(q, 10)[1] is True

    async def test_rows_capped_client_side(self) -> None:
        rag = _mock_rag()
        rag.query = AsyncMock(return_value=[[i] for i in range(50)])
        ts = GraphRAGToolset(rag, enable_cypher=True)
        cr = await ts.cypher_read("MATCH (n) RETURN n LIMIT 1000", limit=5)
        assert cr.row_count == 5 and cr.limit_applied

    async def test_null_optional_uses_default(self) -> None:
        rag = _mock_rag()
        ts = GraphRAGToolset(rag, enable_cypher=True)
        text = await ts.acall_text("graph_search", {"query": "q", "top_k": None})
        assert not text.startswith("Error")
        (fn,) = as_functions(ts, names=["cypher_read"])
        assert not (await fn(query="MATCH (n) RETURN n", limit=None)).startswith("Error")

    async def test_on_write_finalize_failure_is_not_an_error(self) -> None:
        rag = _mock_rag()
        rag.finalize = AsyncMock(side_effect=RuntimeError("embedding down"))
        ts = GraphRAGToolset(rag, finalize_policy="on_write")
        text = await ts.acall_text("graph_remember", {"text": "fact", "document_id": "d"})
        assert text.startswith("Stored document 'd'")
        assert "finalizing failed" in text
        assert ts.pending_finalize == 1

    def test_from_config_closes_rag_on_bad_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import graphrag_sdk.api.main as main_mod

        rag = _mock_rag()
        monkeypatch.setattr(main_mod, "GraphRAG", lambda **kw: rag)
        with pytest.raises(ValueError):
            GraphRAGToolset.from_config(
                object(),  # type: ignore[arg-type]
                llm=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                finalize_policy="bogus",
            )
        rag.close.assert_awaited_once()
