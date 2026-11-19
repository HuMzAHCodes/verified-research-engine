"""The supervisor: builds the research graph from its nodes.

CONCEPT: sequential-graph. Phase 1 was a straight line:

    START -> planner -> researcher -> writer -> END

CONCEPT: parallel-fan-out. From Phase 3 the research step fans out (D-42):

    START -> planner --Send x N--> web_researcher (one per web sub-question)
                                        |  all at once
                                        v
                                   research_check -> writer -> END

route_to_researchers returns one Send per web sub-question; LangGraph runs
every branch in the same step and runs research_check once, after all of them.

Dependencies (the model gateway, the toolbox, the caps) are bound to each
node here with functools.partial (D-25). Nodes never create them, so tests
build this same graph with scripted models and FakeToolbox, and run it end to
end with no network.

CONCEPT: tools-through-mcp. From Phase 2 the researcher reaches the web only
through the toolbox (D-37). The caller opens the toolbox's connection and
passes it in; the graph only uses it:

    async with McpToolbox.connect(settings.vre_search_mcp_url) as toolbox:
        graph = build_graph(gateway, toolbox)

CONCEPT: subgraph-wrapper (D-57). From Phase 4 a sub-question routed to
`document` or `both` also starts a document_researcher branch, which runs the
Self-RAG subgraph on the run's documents, in the same step as the web
branches:

    START -> planner --Send--> web_researcher       (web or both)      --+
                     --Send--> document_researcher  (document or both) --+--> research_check
                                                                              -> writer -> END

The Self-RAG subgraph is compiled once here and shared by every document
branch (decision 3); the store is passed to the planner for file names
(decision 7) and to the document branches for titles.

Later phases change this file: the critic loop (5), the approval gate and
checkpointer (6).
"""

from collections.abc import Mapping
from functools import partial
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from graph.document_store import DocumentStore
from graph.llm import MODEL_CALL_BUDGET_KEY, ModelCallBudget, ModelGateway
from graph.mcp_client import Toolbox
from graph.nodes.document_researcher import DOCUMENT_ROUTES, document_researcher_node
from graph.nodes.planner import planner_node
from graph.nodes.researcher import WEB_ROUTES, research_check_node, web_researcher_node
from graph.nodes.writer import writer_node
from graph.retrieval import HybridRetriever
from graph.self_rag.graph import build_self_rag_graph
from graph.state import RunState
from settings import settings

# CONCEPT: tracing-from-day-one. Tag on every run's root trace; updated when a
# phase changes the graph, so traces from different phases can be told apart.
PHASE_TAG = "phase-4"


def build_graph(
    gateway: ModelGateway,
    toolbox: Toolbox,
    max_sub_questions: int | None = None,
    document_store: DocumentStore | None = None,
) -> CompiledStateGraph:
    """Wire the nodes in order and compile the graph.

    The store defaults to the one in data/vectors/; tests pass one in a
    temporary folder. No checkpointer yet: a run that fails halfway starts
    again. Phase 6 adds SqliteSaver here.
    """
    store = document_store or DocumentStore()
    # Decision 3: one Self-RAG subgraph for every document branch of every run
    # of this graph. A compiled graph holds no run data, so sharing is safe.
    self_rag_graph = build_self_rag_graph(gateway, HybridRetriever(store, gateway))

    graph_builder = StateGraph(RunState)
    graph_builder.add_node(
        "planner",
        partial(
            planner_node,
            gateway=gateway,
            max_sub_questions=max_sub_questions or settings.vre_max_sub_questions,
            document_store=store,
        ),
    )
    graph_builder.add_node(
        "web_researcher", partial(web_researcher_node, gateway=gateway, toolbox=toolbox)
    )
    # Decision 6: no toolbox here, so document text cannot reach web search.
    graph_builder.add_node(
        "document_researcher",
        partial(document_researcher_node, self_rag_graph=self_rag_graph, document_store=store),
    )
    graph_builder.add_node("research_check", research_check_node)
    graph_builder.add_node("writer", partial(writer_node, gateway=gateway))

    graph_builder.add_edge(START, "planner")
    # CONCEPT: parallel-fan-out. A conditional edge whose function returns
    # Send objects. The list names every node the function may route to, so
    # LangGraph can draw the graph and check the names when it compiles.
    graph_builder.add_conditional_edges(
        "planner",
        route_to_researchers,
        ["web_researcher", "document_researcher", "research_check"],
    )
    graph_builder.add_edge("web_researcher", "research_check")
    graph_builder.add_edge("document_researcher", "research_check")
    graph_builder.add_edge("research_check", "writer")
    graph_builder.add_edge("writer", END)
    return graph_builder.compile()


def route_to_researchers(state: Mapping[str, Any]) -> list[Send] | str:
    """One Send per branch, in the planner's order: web, document, or both.

    CONCEPT: parallel-fan-out. Each Send starts one branch with its own input.
    All of them run in the same step, at the same time.

    CONCEPT: subgraph-wrapper, decision 1. A `web` sub-question starts a web
    branch, a `document` one a document branch, and a `both` one each, its
    web branch first. A web branch receives only the sub-question; a document
    branch also the run's document ids (decision 4).

    Parallel-fan-out, decision 7: an empty Send list would end the run right
    here, with no error and no brief. With no branch to start, the run goes
    to research_check instead, so it always reaches a node that decides what
    happens next.
    """
    document_ids = list(state.get("document_ids", []))
    branch_tasks: list[Send] = []
    for sub_question in state["sub_questions"]:
        if sub_question.route in WEB_ROUTES:
            branch_tasks.append(Send("web_researcher", {"sub_question": sub_question}))
        if sub_question.route in DOCUMENT_ROUTES:
            branch_tasks.append(
                Send(
                    "document_researcher",
                    {"sub_question": sub_question, "document_ids": document_ids},
                )
            )
    return branch_tasks or "research_check"


def research_run_config(topic: str, run_id: UUID) -> RunnableConfig:
    """The config every research run is invoked with.

    CONCEPT: tracing-from-day-one. One brief is one LangSmith trace named
    research-run, tagged with the phase, with the topic in its metadata. The
    caller chooses run_id before the run, so the trace's link can be built
    afterwards without searching for it. Phase 9 adds the job id here, which
    makes job id, thread id, and trace id one identifier (D-3).

    CONCEPT: shared-throttle, choice C (D-43). Each run gets a fresh
    ModelCallBudget under configurable. LangGraph passes the run's config to
    every node, parallel branches included, and the gateway reads the budget
    from it, so the whole run counts into one budget without any node
    handling it. Calling this function once per run is what makes the budget
    per run. From Phase 4 it starts at the ceiling, VRE_MAX_MODEL_CALLS_PER_RUN,
    and the planner lowers it to what the run's plan can need (D-58).
    """
    return {
        "run_name": "research-run",
        "run_id": run_id,
        "tags": [PHASE_TAG],
        "metadata": {"topic": topic},
        "configurable": {
            MODEL_CALL_BUDGET_KEY: ModelCallBudget(limit=settings.vre_max_model_calls_per_run)
        },
    }
