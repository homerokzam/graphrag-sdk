# GraphRAG SDK — Integrations core: declarative tool registry
# Single source of truth for tool names, descriptions, argument schemas and
# gating. Every framework adapter generates its tools from TOOL_REGISTRY.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FinalizePolicy = Literal["manual", "on_write", "never"]
Availability = Literal["always", "opt_in", "needs_allowed_dirs", "manual_finalize"]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerArgs(_Args):
    question: str = Field(
        min_length=1,
        description="A self-contained question (resolve pronouns from the conversation first).",
    )


class SearchArgs(_Args):
    query: str = Field(min_length=1, description="What to look up in the knowledge graph.")
    top_k: int = Field(8, ge=1, le=25, description="Maximum number of passages/facts to return.")


class SchemaArgs(_Args):
    pass


class RememberArgs(_Args):
    text: str = Field(
        min_length=1, max_length=200_000, description="The text/fact/note to store in the graph."
    )
    document_id: str | None = Field(
        None, description="Optional stable id for this text (reuse it to refer to it later)."
    )


class IngestFileArgs(_Args):
    path: str = Field(min_length=1, description="File path inside an allowed directory.")
    document_id: str | None = Field(
        None, description="Optional document id (defaults to the path relative to its directory)."
    )


class FlushArgs(_Args):
    pass


class CypherReadArgs(_Args):
    query: str = Field(min_length=1, description="A read-only Cypher query (MATCH ... RETURN ...).")
    params_json: str | None = Field(
        None, description='Optional query parameters as a JSON object string, e.g. {"name": "Ada"}.'
    )
    limit: int = Field(100, ge=1, le=1000, description="Row limit added when the query has none.")


class ForgetArgs(_Args):
    document_id: str = Field(min_length=1, description="Id of the document to delete.")


@dataclass(frozen=True)
class ToolSpec:
    """Framework-neutral description of one GraphRAG tool."""

    name: str
    method: str
    args_model: type[BaseModel]
    description: str
    output_hint: str
    writes: bool = False
    destructive: bool = False
    availability: Availability = "always"

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON schema of the arguments (no titles, ``additionalProperties: false``)."""
        schema = self.args_model.model_json_schema()
        stripped: dict[str, Any] = _strip_titles(schema)
        return stripped

    def docstring(self) -> str:
        """Google-style docstring: description, ``Args:`` and ``Returns:``."""
        lines = [self.description]
        fields = self.args_model.model_fields
        if fields:
            lines += ["", "Args:"]
            for fname, finfo in fields.items():
                lines.append(f"    {fname}: {finfo.description or fname}")
        lines += ["", "Returns:", f"    {self.output_hint}"]
        return "\n".join(lines)


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


TOOL_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="graph_search",
        method="search",
        args_model=SearchArgs,
        description=(
            "Search the knowledge graph and return the relevant context (facts, "
            "relationships, entities and source passages) without generating an answer. "
            "Prefer this for multi-step reasoning; cite passages by their [Source: ...] tag."
        ),
        output_hint="Plain-text context grouped by section, with [Source: ...] citations.",
    ),
    ToolSpec(
        name="graph_answer",
        method="answer",
        args_model=AnswerArgs,
        description=(
            "Answer a question in one shot using the knowledge graph (retrieval + "
            "generation). Returns the answer with its source citations."
        ),
        output_hint="The answer followed by its citations.",
    ),
    ToolSpec(
        name="graph_schema",
        method="schema",
        args_model=SchemaArgs,
        description=(
            "Describe the knowledge graph: entity and relation types and node/edge counts. "
            "Call this first to learn what the graph contains."
        ),
        output_hint="Entity/relation types and graph counts.",
    ),
    ToolSpec(
        name="graph_remember",
        method="remember",
        args_model=RememberArgs,
        description=(
            "Store a piece of text (a fact, note or document content) in the knowledge "
            "graph. After a batch of writes, call graph_flush once so the new knowledge "
            "becomes fully searchable."
        ),
        output_hint="The stored document id and counts of chunks/nodes/relationships created.",
        writes=True,
    ),
    ToolSpec(
        name="graph_ingest_file",
        method="ingest_file",
        args_model=IngestFileArgs,
        description=(
            "Ingest a local file (text, markdown, PDF or CSV) into the knowledge graph. "
            "Only files inside the allowed directories can be read. Call graph_flush once "
            "after a batch of writes."
        ),
        output_hint="The stored document id and counts of chunks/nodes/relationships created.",
        writes=True,
        availability="needs_allowed_dirs",
    ),
    ToolSpec(
        name="graph_flush",
        method="flush",
        args_model=FlushArgs,
        description=(
            "Finalize the knowledge graph after writes (deduplicate entities, compute "
            "embeddings, build indexes). Expensive: call it once at the end of a batch of "
            "graph_remember / graph_ingest_file calls, not after every write."
        ),
        output_hint="Counts of deduplicated and embedded entities/relationships.",
        writes=True,
        availability="manual_finalize",
    ),
    ToolSpec(
        name="cypher_read",
        method="cypher_read",
        args_model=CypherReadArgs,
        description=(
            "Run a read-only Cypher query against the knowledge graph. Use only for "
            "aggregations or filters graph_search cannot express. Writes are rejected and "
            "a LIMIT is added when missing."
        ),
        output_hint="Result rows as JSON arrays.",
        availability="opt_in",
    ),
    ToolSpec(
        name="graph_forget",
        method="forget",
        args_model=ForgetArgs,
        description="Delete a document (and its orphaned entities) from the knowledge graph.",
        output_hint="Whether the document was deleted and what was removed.",
        writes=True,
        destructive=True,
        availability="opt_in",
    ),
)

SPECS_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in TOOL_REGISTRY}


def select_specs(
    *,
    read_only: bool = False,
    finalize_policy: FinalizePolicy = "manual",
    has_allowed_dirs: bool = False,
    enable_cypher: bool = False,
    enable_forget: bool = False,
    include: frozenset[str] | None = None,
    exclude: frozenset[str] | None = None,
) -> list[ToolSpec]:
    """Return the enabled specs (registry order) for a given configuration."""
    unknown = ((include or frozenset()) | (exclude or frozenset())) - set(SPECS_BY_NAME)
    if unknown:
        raise ValueError(f"Unknown tool name(s): {', '.join(sorted(unknown))}")
    selected: list[ToolSpec] = []
    for spec in TOOL_REGISTRY:
        if read_only and spec.writes:
            continue
        if spec.availability == "needs_allowed_dirs" and not has_allowed_dirs:
            continue
        if spec.availability == "manual_finalize" and finalize_policy != "manual":
            continue
        if spec.name == "cypher_read" and not enable_cypher:
            continue
        if spec.name == "graph_forget" and not enable_forget:
            continue
        if include is not None and spec.name not in include:
            continue
        if exclude is not None and spec.name in exclude:
            continue
        selected.append(spec)
    return selected
