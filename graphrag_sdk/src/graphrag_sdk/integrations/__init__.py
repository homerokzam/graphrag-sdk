"""Agent-framework integrations for GraphRAG SDK.

The framework-neutral core lives here; framework adapters live in
subpackages (``graphrag_sdk.integrations.agno``) and are imported explicitly.
Importing this package never imports an agent framework.
"""

from graphrag_sdk.integrations._core.bridge import AsyncBridge, LoopPolicy, bridge_for
from graphrag_sdk.integrations._core.documents import to_document_dicts
from graphrag_sdk.integrations._core.functions import as_functions, make_function
from graphrag_sdk.integrations._core.guards import PathNotAllowedError, ReadOnlyViolation
from graphrag_sdk.integrations._core.models import CallableEmbedder, CallableLLM
from graphrag_sdk.integrations._core.results import (
    AnswerResult,
    Citation,
    CypherResult,
    FlushResult,
    ForgetResult,
    Passage,
    RememberResult,
    SchemaResult,
    SearchResult,
    ToolResult,
)
from graphrag_sdk.integrations._core.specs import (
    TOOL_REGISTRY,
    FinalizePolicy,
    ToolSpec,
    select_specs,
)
from graphrag_sdk.integrations._core.toolset import GraphRAGToolset

__all__ = [
    "TOOL_REGISTRY",
    "AnswerResult",
    "AsyncBridge",
    "CallableEmbedder",
    "CallableLLM",
    "Citation",
    "CypherResult",
    "FinalizePolicy",
    "FlushResult",
    "ForgetResult",
    "GraphRAGToolset",
    "LoopPolicy",
    "Passage",
    "PathNotAllowedError",
    "ReadOnlyViolation",
    "RememberResult",
    "SchemaResult",
    "SearchResult",
    "ToolResult",
    "ToolSpec",
    "as_functions",
    "bridge_for",
    "make_function",
    "select_specs",
    "to_document_dicts",
]
