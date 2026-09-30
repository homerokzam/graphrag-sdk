# GraphRAG SDK — Integrations core: sync/async bridge
# FalkorDB's async connection pool (and the SDK's internal locks) are bound to
# the event loop that first used them. Agent frameworks call tools from sync
# code, from their own loops, from Jupyter... The "dedicated" policy runs
# every GraphRAG coroutine on ONE persistent loop in a daemon thread, so the
# graph resources always see the same loop regardless of who calls.

from __future__ import annotations

import asyncio
import atexit
import threading
import weakref
from collections.abc import Coroutine
from typing import Any, Literal, TypeVar

T = TypeVar("T")
LoopPolicy = Literal["dedicated", "caller"]


class AsyncBridge:
    """Run coroutines on a single, well-defined event loop.

    - ``"dedicated"`` (default): a lazily-started daemon thread owns a
      persistent loop; :meth:`run` (sync) and :meth:`arun` (async) both
      execute there. Works with or without a running loop in the caller.
    - ``"caller"``: :meth:`arun` awaits on the caller's loop; :meth:`run`
      uses ``asyncio.run`` and fails if a loop is already running.
    """

    def __init__(self, policy: LoopPolicy = "dedicated", *, name: str = "graphrag-loop") -> None:
        if policy not in ("dedicated", "caller"):
            raise ValueError("policy must be 'dedicated' or 'caller'")
        self.policy: LoopPolicy = policy
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    # -- internals ---------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise RuntimeError("AsyncBridge is closed")
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                ready = threading.Event()

                def _runner() -> None:
                    asyncio.set_event_loop(loop)
                    loop.call_soon(ready.set)
                    loop.run_forever()

                thread = threading.Thread(target=_runner, name=self._name, daemon=True)
                thread.start()
                ready.wait()
                self._loop, self._thread = loop, thread
                atexit.register(self.close)
            return self._loop

    def _on_bridge_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    # -- public API --------------------------------------------------------

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The dedicated loop (``None`` until first use or with policy ``caller``)."""
        return self._loop

    def run(self, coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
        """Run *coro* to completion from synchronous code and return its result."""
        if self.policy == "caller":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_with_timeout(coro, timeout))
            coro.close()
            raise RuntimeError(
                "Cannot run GraphRAG synchronously inside a running event loop with "
                "loop_policy='caller'; use the async API or loop_policy='dedicated'."
            )
        if self._on_bridge_thread():
            coro.close()
            raise RuntimeError(
                "Synchronous GraphRAG call from the bridge loop thread would deadlock; "
                "await the async API instead."
            )
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def arun(self, coro: Coroutine[Any, Any, T]) -> T:
        """Await *coro* on the bridge's loop from any event loop."""
        if self.policy == "caller" or self._on_bridge_thread():
            return await coro
        loop = self._ensure_loop()
        cfuture = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return await asyncio.wrap_future(cfuture)
        except asyncio.CancelledError:
            cfuture.cancel()
            raise

    def close(self) -> None:
        """Stop the dedicated loop and join its thread. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is not None and thread is not None:
            if loop.is_running():
                if threading.current_thread() is not thread:
                    # Cancel in-flight work first so callers blocked in run()/arun()
                    # get CancelledError instead of waiting forever.
                    try:
                        asyncio.run_coroutine_threadsafe(_cancel_all(), loop).result(5)
                    except Exception:  # pragma: no cover - best effort
                        pass
                loop.call_soon_threadsafe(loop.stop)
            if threading.current_thread() is not thread:
                thread.join(timeout=5)
            if not loop.is_running() and not loop.is_closed():
                loop.close()
        try:
            atexit.unregister(self.close)
        except Exception:  # pragma: no cover - defensive
            pass

    @property
    def closed(self) -> bool:
        return self._closed


async def _cancel_all() -> None:
    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.get_running_loop().shutdown_asyncgens()


async def _with_timeout(coro: Coroutine[Any, Any, T], timeout: float | None) -> T:
    if timeout is None:
        return await coro
    return await asyncio.wait_for(coro, timeout)


_BRIDGES: weakref.WeakKeyDictionary[Any, AsyncBridge] = weakref.WeakKeyDictionary()
_BRIDGES_LOCK = threading.Lock()


def bridge_for(rag: Any, policy: LoopPolicy = "dedicated") -> AsyncBridge:
    """Return the shared bridge for *rag* (one loop per GraphRAG instance).

    Toolkits, knowledge adapters and retrievers built on the same ``GraphRAG``
    share one bridge so every coroutine for that instance runs on one loop.
    """
    with _BRIDGES_LOCK:
        bridge = _BRIDGES.get(rag)
        if bridge is None or bridge.closed:
            bridge = AsyncBridge(policy)
            _BRIDGES[rag] = bridge
        elif bridge.policy != policy:
            raise ValueError(
                f"This GraphRAG instance already uses loop_policy={bridge.policy!r}; "
                f"cannot mix with {policy!r}."
            )
        return bridge
