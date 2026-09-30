# GraphRAG SDK — Integrations core: generic model adapters
# Wrap any chat / embedding callable as a GraphRAG LLMInterface / Embedder so
# framework-specific adapters (Agno now; LangChain, CrewAI, ... later) only
# need to supply a few small functions.

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from graphrag_sdk.core.exceptions import EmbeddingTimeoutError, LLMTimeoutError
from graphrag_sdk.core.models import ChatMessage, LLMResponse
from graphrag_sdk.core.providers._timeout import validate_timeout, wait_for_provider_call
from graphrag_sdk.core.providers.base import Embedder, LLMInterface

logger = logging.getLogger(__name__)
T = TypeVar("T")

ChatFn = Callable[[list[ChatMessage]], "str | LLMResponse"]
AChatFn = Callable[[list[ChatMessage]], Awaitable["str | LLMResponse"]]
StructuredFn = Callable[[list[ChatMessage], type[BaseModel]], "BaseModel | str | None"]
AStructuredFn = Callable[[list[ChatMessage], type[BaseModel]], Awaitable["BaseModel | str | None"]]

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


def strip_json_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence (```json ... ```), if any."""
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text.strip()


def _to_response(value: str | LLMResponse | None) -> LLMResponse:
    if isinstance(value, LLMResponse):
        return value
    return LLMResponse(content="" if value is None else str(value))


async def _with_retries(
    make_call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    timeout: float | None,
    timeout_error: type[LLMTimeoutError] | type[EmbeddingTimeoutError],
    operation: str,
) -> T:
    """Retry with jittered exponential backoff (mirrors ``LLMInterface.ainvoke``)."""
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    validate_timeout(timeout)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: T = await wait_for_provider_call(
                make_call(), timeout=timeout, timeout_error=timeout_error, operation=operation
            )
            return result
        except (LLMTimeoutError, EmbeddingTimeoutError):
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = (2**attempt) * (0.5 + random.random())
                logger.warning(
                    "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                    operation,
                    attempt + 1,
                    max_retries,
                    delay,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


class CallableLLM(LLMInterface):
    """A GraphRAG ``LLMInterface`` backed by plain chat callables.

    Args:
        model_name: Identifier reported to GraphRAG.
        chat: ``messages -> str | LLMResponse`` (sync). Required.
        achat: Optional native async variant. Used when ``prefer_async`` is True.
        structured: Optional ``(messages, response_model) -> BaseModel | str``
            for native structured output. When missing (or returning a
            string), the JSON in the text reply is parsed.
        astructured: Async variant of ``structured``.
        prefer_async: Use ``achat``/``astructured`` for async calls instead of
            running the sync callables in a worker thread.
    """

    def __init__(
        self,
        *,
        model_name: str,
        chat: ChatFn,
        achat: AChatFn | None = None,
        structured: StructuredFn | None = None,
        astructured: AStructuredFn | None = None,
        prefer_async: bool = False,
        model_params: dict[str, Any] | None = None,
        max_concurrency: int = 12,
    ) -> None:
        super().__init__(model_name, model_params, max_concurrency)
        self._chat = chat
        self._achat = achat
        self._structured = structured
        self._astructured = astructured
        self._prefer_async = prefer_async

    # -- text ---------------------------------------------------------------

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return _to_response(self._chat([ChatMessage(role="user", content=prompt)]))

    async def _achat_once(self, messages: list[ChatMessage]) -> LLMResponse:
        if self._prefer_async and self._achat is not None:
            return _to_response(await self._achat(messages))
        return _to_response(await asyncio.to_thread(self._chat, messages))

    async def ainvoke(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.ainvoke_messages(
            [ChatMessage(role="user", content=prompt)], max_retries=max_retries, timeout=timeout
        )

    async def ainvoke_messages(
        self,
        messages: list[ChatMessage],
        *,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        msgs = list(messages)
        return await _with_retries(
            lambda: self._achat_once(msgs),
            max_retries=max_retries,
            timeout=timeout,
            timeout_error=LLMTimeoutError,
            operation=f"LLM call to {self.model_name}",
        )

    # -- structured -------------------------------------------------------

    @staticmethod
    def _coerce(
        value: BaseModel | str | LLMResponse | None, response_model: type[BaseModel]
    ) -> BaseModel:
        if isinstance(value, response_model):
            return value
        if isinstance(value, BaseModel) and not isinstance(value, LLMResponse):
            return response_model.model_validate(value.model_dump())
        text = (
            value.content
            if isinstance(value, LLMResponse)
            else ("" if value is None else str(value))
        )
        return response_model.model_validate_json(strip_json_fences(text))

    def invoke_with_model(
        self, prompt: str, response_model: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        messages = [ChatMessage(role="user", content=prompt)]
        if self._structured is not None:
            return self._coerce(self._structured(messages, response_model), response_model)
        return self._coerce(self._chat(messages), response_model)

    async def ainvoke_with_model(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        max_retries: int = 3,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        messages = [ChatMessage(role="user", content=prompt)]

        async def _once() -> BaseModel:
            if self._prefer_async and self._astructured is not None:
                return self._coerce(
                    await self._astructured(messages, response_model), response_model
                )
            if self._prefer_async and self._achat is not None and self._structured is None:
                return self._coerce(await self._achat(messages), response_model)
            return await asyncio.to_thread(self.invoke_with_model, prompt, response_model)

        return await _with_retries(
            _once,
            max_retries=max_retries,
            timeout=timeout,
            timeout_error=LLMTimeoutError,
            operation=f"Structured LLM call to {self.model_name}",
        )


class CallableEmbedder(Embedder):
    """A GraphRAG ``Embedder`` backed by plain embedding callables.

    Args:
        model_name: Identifier stored with the graph (must stay stable).
        embed: ``text -> vector`` (sync). Required.
        embed_batch: Optional ``texts -> vectors`` (sync).
        aembed / aembed_batch: Optional native async variants, used when
            ``prefer_async`` is True.
        dimensions: Expected vector size; mismatches raise ``ValueError``.
    """

    def __init__(
        self,
        *,
        model_name: str,
        embed: Callable[[str], Sequence[float]],
        embed_batch: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
        aembed: Callable[[str], Awaitable[Sequence[float]]] | None = None,
        aembed_batch: Callable[[list[str]], Awaitable[Sequence[Sequence[float]]]] | None = None,
        dimensions: int | None = None,
        prefer_async: bool = False,
    ) -> None:
        self._model_name = model_name
        self._embed = embed
        self._embed_batch = embed_batch
        self._aembed = aembed
        self._aembed_batch = aembed_batch
        self.dimensions = dimensions
        self._prefer_async = prefer_async

    @property
    def model_name(self) -> str:
        return self._model_name

    def _check(self, vec: Sequence[float]) -> list[float]:
        out = [float(x) for x in vec]
        if not out:
            raise ValueError(f"Embedder {self._model_name} returned an empty vector")
        if self.dimensions is not None and len(out) != self.dimensions:
            raise ValueError(
                f"Embedder {self._model_name} returned {len(out)} dimensions, "
                f"expected {self.dimensions}"
            )
        return out

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self._check(self._embed(text))

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if self._embed_batch is not None:
            vectors = self._embed_batch(list(texts))
            if len(vectors) != len(texts):
                raise ValueError("Batch embedder returned a different number of vectors")
            return [self._check(v) for v in vectors]
        return [self.embed_query(t) for t in texts]

    async def aembed_query(
        self, text: str, *, timeout: float | None = None, **kwargs: Any
    ) -> list[float]:
        validate_timeout(timeout)
        if self._prefer_async and self._aembed is not None:
            vec = await wait_for_provider_call(
                self._aembed(text),
                timeout=timeout,
                timeout_error=EmbeddingTimeoutError,
                operation=f"Embedding call to {self._model_name}",
            )
            return self._check(vec)
        result: list[float] = await wait_for_provider_call(
            asyncio.to_thread(self.embed_query, text),
            timeout=timeout,
            timeout_error=EmbeddingTimeoutError,
            operation=f"Embedding call to {self._model_name}",
        )
        return result

    async def aembed_documents(
        self, texts: list[str], *, timeout: float | None = None, **kwargs: Any
    ) -> list[list[float]]:
        validate_timeout(timeout)
        if self._prefer_async and (self._aembed_batch is not None or self._aembed is not None):

            async def _batch() -> list[list[float]]:
                if self._aembed_batch is not None:
                    vectors = await self._aembed_batch(list(texts))
                    if len(vectors) != len(texts):
                        raise ValueError("Batch embedder returned a different number of vectors")
                    return [self._check(v) for v in vectors]
                assert self._aembed is not None
                return [self._check(await self._aembed(t)) for t in texts]

            batch: list[list[float]] = await wait_for_provider_call(
                _batch(),
                timeout=timeout,
                timeout_error=EmbeddingTimeoutError,
                operation=f"Embedding call to {self._model_name}",
            )
            return batch
        result: list[list[float]] = await wait_for_provider_call(
            asyncio.to_thread(self.embed_documents, list(texts)),
            timeout=timeout,
            timeout_error=EmbeddingTimeoutError,
            operation=f"Embedding call to {self._model_name}",
        )
        return result
