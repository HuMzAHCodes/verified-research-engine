"""The supervisor: builds the research graph from its nodes.

CONCEPT: sequential-graph. Phase 1 is a straight line:

    START -> planner -> researcher -> writer -> END

Dependencies (the model gateway, the search client, the caps) are bound to
each node here with functools.partial (D-25). Nodes never create them, so
tests build this same graph with scripted models and a fake search, and run
it end to end with no network.

Later phases change this file: parallel researchers (3), the document
subgraph (4), the critic loop (5), the approval gate and checkpointer (6).
"""

from functools import partial
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.llm import ModelGateway
from graph.nodes.planner import planner_node
from graph.nodes.researcher import WebSearchClient, researcher_node
from graph.nodes.writer import writer_node
from graph.state import RunState
from settings import settings

# CONCEPT: tracing-from-day-one. Tag on every run's root trace; updated when a
# phase changes the graph, so traces from different phases can be told apart.
PHASE_TAG = "phase-1"


def build_graph(
    gateway: ModelGateway,
    search_client: WebSearchClient,
    max_sub_questions: int | None = None,
) -> CompiledStateGraph:
    """Wire the nodes in order and compile the graph.

    No checkpointer yet: a run that fails halfway starts again. Phase 6 adds
    SqliteSaver here.
    """
    graph_builder = StateGraph(RunState)
    graph_builder.add_node(
        "planner",
        partial(
            planner_node,
            gateway=gateway,
            max_sub_questions=max_sub_questions or settings.vre_max_sub_questions,
        ),
    )
    graph_builder.add_node("researcher", partial(researcher_node, search_client=search_client))
    graph_builder.add_node("writer", partial(writer_node, gateway=gateway))

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "researcher")
    graph_builder.add_edge("researcher", "writer")
    graph_builder.add_edge("writer", END)
    return graph_builder.compile()


def research_run_config(topic: str, run_id: UUID) -> RunnableConfig:
    """The config every research run is invoked with.

    CONCEPT: tracing-from-day-one. One brief is one LangSmith trace named
    research-run, tagged with the phase, with the topic in its metadata. The
    caller chooses run_id before the run, so the trace's link can be built
    afterwards without searching for it. Phase 9 adds the job id here, which
    makes job id, thread id, and trace id one identifier (D-3).
    """
    return {
        "run_name": "research-run",
        "run_id": run_id,
        "tags": [PHASE_TAG],
        "metadata": {"topic": topic},
    }
