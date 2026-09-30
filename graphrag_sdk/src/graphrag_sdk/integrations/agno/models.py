"""Use Agno models as GraphRAG's LLM and embedder."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from graphrag_sdk.core.models import ChatMessage
from graphrag_sdk.integrations._core.models import CallableEmbedder, CallableLLM


def _to_agno_messages(messages: list[ChatMessage]) -> list[Any]:
    from agno.models.message import Message

    return [Message(role=m.role, content=m.content) for m in messages]


def _content(resp: Any) -> str:
    content = getattr(resp, "content", None)
    return "" if content is None else str(content)


class AgnoLLM(CallableLLM):
    """Wrap an Agno model (``agno.models.*``) as a GraphRAG ``LLMInterface``.

    Use a separate model instance from the one driving your Agent: Agno
    models cache clients and per-run state on the instance.

    Args:
        model: Any Agno model (or object exposing ``response`` / ``aresponse``).
        model_name: Name reported to GraphRAG (defaults to ``model.id``).
        use_async: Call ``aresponse`` natively instead of running ``response``
            in a worker thread. Off by default: Agno caches async HTTP clients
            per model instance, which ties them to one event loop.
        native_structured_output: Pass ``response_format=<pydantic model>`` for
            structured calls (falls back to JSON parsing of the text reply).
    """

    def __init__(
        self,
        model: Any,
        *,
        model_name: str | None = None,
        use_async: bool = False,
        native_structured_output: bool = True,
        max_concurrency: int = 12,
    ) -> None:
        self._model = model
        name = model_name or str(getattr(model, "id", None) or type(model).__name__)

        def chat(messages: list[ChatMessage]) -> str:
            return _content(model.response(messages=_to_agno_messages(messages)))

        async def achat(messages: list[ChatMessage]) -> str:
            return _content(await model.aresponse(messages=_to_agno_messages(messages)))

        def _structured_value(resp: Any) -> BaseModel | str:
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, BaseModel):
                return parsed
            return _content(resp)

        def structured(
            messages: list[ChatMessage], response_model: type[BaseModel]
        ) -> BaseModel | str:
            return _structured_value(
                model.response(messages=_to_agno_messages(messages), response_format=response_model)
            )

        async def astructured(
            messages: list[ChatMessage], response_model: type[BaseModel]
        ) -> BaseModel | str:
            return _structured_value(
                await model.aresponse(
                    messages=_to_agno_messages(messages), response_format=response_model
                )
            )

        super().__init__(
            model_name=name,
            chat=chat,
            achat=achat if hasattr(model, "aresponse") else None,
            structured=structured if native_structured_output else None,
            astructured=(
                astructured if native_structured_output and hasattr(model, "aresponse") else None
            ),
            prefer_async=use_async,
            max_concurrency=max_concurrency,
        )

    @property
    def model(self) -> Any:
        return self._model


class AgnoEmbedder(CallableEmbedder):
    """Wrap an Agno embedder (``agno.knowledge.embedder.*``) as a GraphRAG ``Embedder``.

    Set ``GraphRAG(embedding_dimension=...)`` to the embedder's dimensions.

    Args:
        embedder: Any Agno embedder (``get_embedding`` / ``async_get_embedding``).
        model_name: Name stored with the graph (defaults to ``embedder.id``).
        dimensions: Expected vector size (defaults to ``embedder.dimensions``).
        use_async: Use the embedder's native async methods.
    """

    def __init__(
        self,
        embedder: Any,
        *,
        model_name: str | None = None,
        dimensions: int | None = None,
        use_async: bool = False,
    ) -> None:
        self._embedder = embedder
        name = model_name or str(getattr(embedder, "id", None) or type(embedder).__name__)
        dims = dimensions if dimensions is not None else getattr(embedder, "dimensions", None)

        aembed_batch = None
        if hasattr(embedder, "async_get_embeddings_batch_and_usage"):

            async def aembed_batch(texts: list[str]) -> list[list[float]]:
                embeddings, _usage = await embedder.async_get_embeddings_batch_and_usage(texts)
                return list(embeddings)

        super().__init__(
            model_name=name,
            embed=embedder.get_embedding,
            aembed=getattr(embedder, "async_get_embedding", None),
            aembed_batch=aembed_batch,
            dimensions=dims if isinstance(dims, int) else None,
            prefer_async=use_async,
        )

    @property
    def embedder(self) -> Any:
        return self._embedder


__all__ = ["AgnoEmbedder", "AgnoLLM"]
