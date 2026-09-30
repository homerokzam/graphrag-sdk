# GraphRAG SDK — Integrations core: export tools as plain Python callables
# Real function objects with __name__, a Google-style __doc__ and a typed
# __signature__, generated from TOOL_REGISTRY. Frameworks that turn plain
# functions into tools (Agno, Google Antigravity, Google ADK, CrewAI @tool,
# LangChain StructuredTool.from_function, pydantic-ai) consume them as-is.

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import PydanticUndefined

from graphrag_sdk.integrations._core.specs import ToolSpec

if TYPE_CHECKING:
    from graphrag_sdk.integrations._core.toolset import GraphRAGToolset


def _signature(spec: ToolSpec) -> tuple[inspect.Signature, dict[str, Any]]:
    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    required: list[inspect.Parameter] = []
    optional: list[inspect.Parameter] = []
    for fname, finfo in spec.args_model.model_fields.items():
        annotation = finfo.annotation
        annotations[fname] = annotation
        if finfo.is_required():
            required.append(
                inspect.Parameter(
                    fname, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
                )
            )
        else:
            default = finfo.default if finfo.default is not PydanticUndefined else None
            optional.append(
                inspect.Parameter(
                    fname,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                    default=default,
                )
            )
    params = required + optional
    annotations["return"] = str
    return inspect.Signature(params, return_annotation=str), annotations


def _decorate(fn: Callable[..., Any], spec: ToolSpec, public_name: str) -> Callable[..., Any]:
    sig, annotations = _signature(spec)
    fn.__name__ = public_name
    fn.__qualname__ = public_name
    fn.__doc__ = spec.docstring()
    fn.__signature__ = sig  # type: ignore[attr-defined]
    fn.__annotations__ = annotations
    fn.__module__ = "graphrag_sdk.integrations"
    fn.__graphrag_spec__ = spec  # type: ignore[attr-defined]
    return fn


def _bind_arguments(
    spec: ToolSpec, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    sig, _ = _signature(spec)
    bound = sig.bind(*args, **kwargs)
    return dict(bound.arguments)


def make_function(
    toolset: GraphRAGToolset,
    spec: ToolSpec,
    *,
    mode: Literal["async", "sync"] = "async",
    name: str | None = None,
    output: Literal["text", "dict"] = "text",
) -> Callable[..., Any]:
    """Build one callable for *spec* bound to *toolset*."""
    public_name = name or spec.name

    if output == "dict":

        async def _acall_dict(*args: Any, **kwargs: Any) -> Any:
            return (await toolset.acall(spec.name, _bind_arguments(spec, args, kwargs))).to_dict()

        def _call_dict(*args: Any, **kwargs: Any) -> Any:
            return toolset.run(_acall_dict(*args, **kwargs))

        fn: Callable[..., Any] = _acall_dict if mode == "async" else _call_dict
        _decorate(fn, spec, public_name)
        fn.__annotations__["return"] = dict
        fn.__signature__ = fn.__signature__.replace(return_annotation=dict)  # type: ignore[attr-defined]
        return fn

    if mode == "async":

        async def _acall(*args: Any, **kwargs: Any) -> str:
            return await toolset.acall_text(spec.name, _bind_arguments(spec, args, kwargs))

        return _decorate(_acall, spec, public_name)

    def _call(*args: Any, **kwargs: Any) -> str:
        return toolset.call_text(spec.name, _bind_arguments(spec, args, kwargs))

    return _decorate(_call, spec, public_name)


def as_functions(
    toolset: GraphRAGToolset,
    *,
    mode: Literal["async", "sync"] = "async",
    names: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    output: Literal["text", "dict"] = "text",
) -> list[Callable[..., Any]]:
    """Export the toolset's enabled tools as plain, typed, documented callables.

    Args:
        toolset: The toolset whose enabled specs are exported.
        mode: ``"async"`` returns coroutine functions, ``"sync"`` plain functions.
        names: Optional subset of canonical tool names to export.
        rename: Optional mapping canonical name -> public function name.
        output: ``"text"`` (LLM-ready string) or ``"dict"`` (JSON-safe dict).

    Example (Google Antigravity)::

        config = LocalAgentConfig(tools=as_functions(toolset))
    """
    rename = dict(rename or {})
    specs = toolset.specs()
    if names is not None:
        enabled = {s.name for s in specs}
        missing = set(names) - enabled
        if missing:
            raise KeyError(f"Tool(s) not enabled on this toolset: {', '.join(sorted(missing))}")
        specs = [s for s in specs if s.name in set(names)]
    unknown = set(rename) - {s.name for s in toolset.specs()}
    if unknown:
        raise KeyError(f"Cannot rename unknown/disabled tool(s): {', '.join(sorted(unknown))}")
    return [
        make_function(toolset, s, mode=mode, name=rename.get(s.name), output=output) for s in specs
    ]
