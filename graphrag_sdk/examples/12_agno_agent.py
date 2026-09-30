"""
GraphRAG SDK -- Agno Agent Integration
======================================
Gives an Agno agent a FalkorDB knowledge graph three ways:

  A. GraphRAGTools       -- a Toolkit: the agent stores knowledge (graph_remember),
                            finalizes it (graph_flush) and queries it (graph_search /
                            graph_answer / graph_schema).
  B. GraphRAGKnowledge   -- the graph as the agent's knowledge base
                            (Agent(knowledge=...) + built-in search_knowledge_base).
  C. Async               -- the same toolkit under ``await agent.arun(...)``.

GraphRAG itself runs on Agno models here (AgnoLLM / AgnoEmbedder), so one
provider configuration drives both the agent and the graph.

Prerequisites:
    pip install "graphrag-sdk[agno]" openai
    docker run -p 6379:6379 falkordb/falkordb
    export OPENAI_API_KEY=...
"""

import asyncio

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat

from graphrag_sdk import ConnectionConfig
from graphrag_sdk.integrations.agno import (
    AgnoEmbedder,
    AgnoLLM,
    GraphRAGKnowledge,
    GraphRAGTools,
)

NOTES = """\
Ada Lovelace (1815-1852) was an English mathematician. She worked with Charles
Babbage on the Analytical Engine and published the first algorithm intended to
be carried out by such a machine, which is why she is often called the first
computer programmer. Babbage designed the Analytical Engine in 1837.
"""


def main() -> None:
    # GraphRAG uses its OWN Agno model instances (don't share the agent's).
    tools = GraphRAGTools.from_config(
        ConnectionConfig(host="localhost", port=6379, graph_name="agno_demo"),
        llm=AgnoLLM(OpenAIChat(id="gpt-4o-mini")),
        embedder=AgnoEmbedder(OpenAIEmbedder(id="text-embedding-3-small", dimensions=256)),
        embedding_dimension=256,
        # allowed_dirs=["./data"],  # enables graph_ingest_file for files under ./data
    )

    try:
        # --- A. Toolkit agent: write, finalize, then ask -----------------------
        agent = Agent(model=OpenAIChat(id="gpt-4o"), tools=[tools], markdown=True)
        agent.print_response(
            "Store these notes in the knowledge graph, finalize it, then tell me "
            f"who wrote the first algorithm for the Analytical Engine.\n\n{NOTES}"
        )

        # --- B. The graph as the agent's knowledge ------------------------------
        knowledge = GraphRAGKnowledge(toolset=tools.toolset, max_results=8)
        reader = Agent(model=OpenAIChat(id="gpt-4o"), knowledge=knowledge, search_knowledge=True)
        run = reader.run("When was the Analytical Engine designed, and by whom?")
        print(run.content)
        print("retrieved references:", len(run.references or []))

        # --- C. Async agents use the async tool variants ------------------------
        async def ask_async() -> None:
            response = await agent.arun("What is Ada Lovelace known for? Cite the source.")
            print(response.content)

        asyncio.run(ask_async())

        # Your own GraphRAG calls on the same instance go through the toolset's loop:
        print(tools.toolset.run(tools.toolset.rag.get_statistics()))
    finally:
        tools.close()


if __name__ == "__main__":
    main()
