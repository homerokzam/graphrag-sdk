"""Tests for the Agno adapter (skipped when agno is not installed)."""

from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from graphrag_sdk.core.models import ChatMessage
from graphrag_sdk.integrations import GraphRAGToolset

from .test_integrations_core import _mock_rag


def test_import_error_mentions_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name.startswith("graphrag_sdk.integrations.agno"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "agno", None)
    with pytest.raises(ImportError, match=r"graphrag-sdk\[agno\]"):
        importlib.import_module("graphrag_sdk.integrations.agno")


agno = pytest.importorskip("agno")

from agno.agent import Agent  # noqa: E402
from agno.knowledge.document import Document  # noqa: E402
from agno.knowledge.embedder.base import Embedder as AgnoBaseEmbedder  # noqa: E402
from agno.knowledge.protocol import KnowledgeProtocol  # noqa: E402

from graphrag_sdk.integrations.agno import (  # noqa: E402
    AgnoEmbedder,
    AgnoLLM,
    GraphRAGKnowledge,
    GraphRAGTools,
    graphrag_knowledge_retriever,
)


def _params(tk: GraphRAGTools, name: str, *, is_async: bool = False) -> dict[str, Any]:
    fns = tk.get_async_functions() if is_async else tk.functions
    fn = fns[name]
    fn.process_entrypoint()
    params: dict[str, Any] = fn.to_dict()["parameters"]
    return params


class TestToolkit:
    def test_default_tools_sync_and_async(self) -> None:
        tk = GraphRAGTools(_mock_rag())
        expected = ["graph_search", "graph_answer", "graph_schema", "graph_remember", "graph_flush"]
        assert list(tk.functions) == expected
        assert list(tk.get_async_functions()) == expected
        assert tk.instructions and "graph_flush" in tk.instructions

    def test_schema_from_specs(self) -> None:
        tk = GraphRAGTools(_mock_rag())
        for is_async in (False, True):
            p = _params(tk, "graph_search", is_async=is_async)
            assert p["required"] == ["query"]
            assert p["properties"]["query"]["type"] == "string"
            assert p["properties"]["top_k"]["type"] == "integer"
            assert p["properties"]["query"]["description"]
        remember = _params(tk, "graph_remember")
        assert remember["required"] == ["text"]

    def test_flags_and_rename(self, tmp_path: Path) -> None:
        rag = _mock_rag()
        tk = GraphRAGTools(
            rag,
            enable_write=False,
            enable_schema=False,
            enable_cypher=True,
            tool_names={"graph_answer": "ask_knowledge_graph"},
        )
        assert set(tk.functions) == {"graph_search", "ask_knowledge_graph", "cypher_read"}
        all_tools = GraphRAGTools(_mock_rag(), all=True, allowed_dirs=[tmp_path])
        assert {"graph_ingest_file", "graph_forget", "cypher_read"} <= set(all_tools.functions)
        assert all_tools.functions["graph_forget"].requires_confirmation
        assert not all_tools.functions["graph_remember"].requires_confirmation
        confirm = GraphRAGTools(_mock_rag(), requires_confirmation_for_writes=True)
        assert confirm.functions["graph_remember"].requires_confirmation
        with pytest.raises(ValueError):
            GraphRAGTools()

    def test_shared_toolset(self) -> None:
        ts = GraphRAGToolset(_mock_rag(), read_only=True)
        tk = GraphRAGTools(toolset=ts, enable_answer=False)
        assert set(tk.functions) == {"graph_search", "graph_schema"}
        assert tk.toolset is ts

    def test_sync_entrypoint(self) -> None:
        tk = GraphRAGTools(_mock_rag())
        out = tk.functions["graph_answer"].entrypoint(question="who")
        assert out.startswith("Answer: Ada.")
        out = tk.functions["graph_remember"].entrypoint(text="fact")
        assert "pending finalization" in out

    async def test_async_entrypoint_inside_loop(self) -> None:
        tk = GraphRAGTools(_mock_rag())
        out = await tk.get_async_functions()["graph_search"].entrypoint(query="who")
        assert "[Source: docs/ada.md]" in out
        # The sync variant also works while an event loop is running.
        assert tk.functions["graph_search"].entrypoint(query="who").startswith("Query: who")

    def test_from_config_owns_rag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import graphrag_sdk.api.main as main_mod

        rag = _mock_rag()
        monkeypatch.setattr(main_mod, "GraphRAG", lambda **kw: rag)
        tk = GraphRAGTools.from_config(object(), llm=object(), embedder=object())  # type: ignore[arg-type]
        tk.close()
        rag.close.assert_awaited_once()


class TestKnowledge:
    def test_protocol_and_retrieve(self) -> None:
        k = GraphRAGKnowledge(_mock_rag(), max_results=3)
        assert isinstance(k, KnowledgeProtocol)
        docs = k.retrieve("who", max_results=None, filters={"x": 1})
        assert all(isinstance(d, Document) for d in docs) and len(docs) == 3
        assert docs[0].name == "graph:cypher_results"
        assert "Ada" not in k.build_context()  # static text, no I/O
        assert "graph_remember" not in {s.name for s in k.toolset.specs()}

    async def test_aretrieve_and_tools(self) -> None:
        k = GraphRAGKnowledge(_mock_rag(), include_graph_sections=False)
        docs = await k.aretrieve("who", max_results=2)
        assert [d.name for d in docs] == ["docs/ada.md", "b.pdf"]
        assert [f.__name__ for f in k.get_tools()] == [
            "graph_search",
            "graph_answer",
            "graph_schema",
        ]
        assert all(asyncio.iscoroutinefunction(f) for f in await k.aget_tools())

    def test_agent_uses_knowledge(self) -> None:
        from agno.agent._messages import get_relevant_docs_from_knowledge

        agent = Agent(knowledge=GraphRAGKnowledge(_mock_rag()), search_knowledge=True)
        docs = get_relevant_docs_from_knowledge(agent, "who")
        assert docs and any("Ada" in str(d) for d in docs)

    def test_retriever_modes(self) -> None:
        ts = GraphRAGToolset(_mock_rag())
        sync_r = graphrag_knowledge_retriever(toolset=ts)
        docs = sync_r(query="who", num_documents=2)
        assert isinstance(docs, list) and len(docs) == 2
        async_r = graphrag_knowledge_retriever(toolset=ts, mode="async")
        assert len(asyncio.run(async_r(query="who"))) >= 3  # type: ignore[arg-type]

    def test_agent_uses_retriever(self) -> None:
        from agno.agent._messages import get_relevant_docs_from_knowledge

        agent = Agent(
            knowledge_retriever=graphrag_knowledge_retriever(_mock_rag()),
            search_knowledge=True,
        )
        docs = get_relevant_docs_from_knowledge(agent, "who", num_documents=2)
        assert docs is not None and len(docs) == 2


class _Out(BaseModel):
    name: str


@dataclass
class _FakeResponse:
    content: Any = None
    parsed: Any = None


class _FakeModel:
    id = "fake-model"

    def __init__(self) -> None:
        self.calls: list[tuple[list[Any], Any]] = []

    def response(self, messages: list[Any], response_format: Any = None) -> _FakeResponse:
        self.calls.append((messages, response_format))
        if response_format is not None:
            return _FakeResponse(content='{"name": "x"}', parsed=response_format(name="parsed"))
        return _FakeResponse(content="hello")

    async def aresponse(self, messages: list[Any], response_format: Any = None) -> _FakeResponse:
        return self.response(messages, response_format)


@dataclass
class _FakeEmbedder(AgnoBaseEmbedder):
    id: str = "fake-embed"
    dimensions: int | None = 3

    def get_embedding(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    async def async_get_embedding(self, text: str) -> list[float]:
        return [4.0, 5.0, 6.0]


class TestModels:
    async def test_llm(self) -> None:
        model = _FakeModel()
        llm = AgnoLLM(model)
        assert llm.model_name == "fake-model"
        msgs = [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")]
        assert (await llm.ainvoke_messages(msgs)).content == "hello"
        roles = [m.role for m in model.calls[-1][0]]
        assert roles == ["system", "user"]
        out = await llm.ainvoke_with_model("p", _Out)
        assert out.name == "parsed"  # type: ignore[attr-defined]

    async def test_llm_async_and_text_fallback(self) -> None:
        model = _FakeModel()
        llm = AgnoLLM(model, use_async=True)
        assert (await llm.ainvoke("p")).content == "hello"

        class JsonModel(_FakeModel):
            def response(self, messages: list[Any], response_format: Any = None) -> _FakeResponse:
                return _FakeResponse(content='{"name": "json"}')

        plain = AgnoLLM(JsonModel(), native_structured_output=False)
        assert plain.invoke_with_model("p", _Out).name == "json"  # type: ignore[attr-defined]

    async def test_embedder(self) -> None:
        emb = AgnoEmbedder(_FakeEmbedder())
        assert emb.model_name == "fake-embed"
        assert emb.embed_query("x") == [1.0, 2.0, 3.0]
        assert await emb.aembed_documents(["a", "b"]) == [[1.0, 2.0, 3.0]] * 2
        native = AgnoEmbedder(_FakeEmbedder(), use_async=True)
        assert await native.aembed_query("x") == [4.0, 5.0, 6.0]
        with pytest.raises(ValueError):
            AgnoEmbedder(_FakeEmbedder(), dimensions=4).embed_query("x")


def test_toolkit_flags_filter_prebuilt_toolset() -> None:
    ts = GraphRAGToolset(_mock_rag(), enable_cypher=True, enable_forget=True)
    assert "graph_forget" not in GraphRAGTools(toolset=ts).functions
    kept = GraphRAGTools(toolset=ts, enable_cypher=True, enable_forget=True)
    assert {"cypher_read", "graph_forget"} <= set(kept.functions)


def test_agno_null_optional_argument() -> None:
    tk = GraphRAGTools(_mock_rag())
    out = tk.functions["graph_search"].entrypoint(query="who", top_k=None)
    assert out.startswith("Query: who")
