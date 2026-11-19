"""The Self-RAG subgraph: its nodes wired together, and one call to run it.

CONCEPT: self-rag. The shape, with every loop capped (decision 7, D-53):

    START: no documents                    -> finish_not_found
    write_passage -> retrieve -> grade
    grade: some relevant                   -> generate -> check_support
    grade: none relevant                   -> write_passage (once more), else finish_not_found
    check_support: fully supported         -> check_usefulness
    check_support: not fully               -> revise (once per sub-question), else finish_not_found
    revise: quotes kept                    -> check_usefulness
    revise: no quote kept                  -> finish_not_found
    check_usefulness: useful               -> finish_answered
    check_usefulness: not useful           -> write_passage (once more), else finish_not_found

The longest path visits 14 nodes, inside LangGraph's default limit of 25.
Concept 4 (subgraph-wrapper) runs it as the document branch of a research
run; until then it is run with run_self_rag().
"""

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.llm import ModelGateway
from graph.retrieval import HybridRetriever
from graph.self_rag import nodes
from graph.self_rag.state import SelfRagResult, SelfRagState


def build_self_rag_graph(gateway: ModelGateway, retriever: HybridRetriever) -> CompiledStateGraph:
    """Wire the nodes and the decisions between them, and compile."""
    graph_builder = StateGraph(SelfRagState)
    graph_builder.add_node("write_passage", partial(nodes.write_passage_node, gateway=gateway))
    graph_builder.add_node("retrieve", partial(nodes.retrieve_node, retriever=retriever))
    graph_builder.add_node("grade", partial(nodes.grade_node, gateway=gateway))
    graph_builder.add_node("generate", partial(nodes.generate_node, gateway=gateway))
    graph_builder.add_node("check_support", partial(nodes.check_support_node, gateway=gateway))
    graph_builder.add_node("revise", partial(nodes.revise_node, gateway=gateway))
    graph_builder.add_node(
        "check_usefulness", partial(nodes.check_usefulness_node, gateway=gateway)
    )
    graph_builder.add_node("finish_answered", nodes.finish_answered_node)
    graph_builder.add_node("finish_not_found", nodes.finish_not_found_node)

    graph_builder.add_conditional_edges(
        START, nodes.route_at_start, ["write_passage", "finish_not_found"]
    )
    graph_builder.add_edge("write_passage", "retrieve")
    graph_builder.add_edge("retrieve", "grade")
    graph_builder.add_conditional_edges(
        "grade", nodes.route_after_grading, ["generate", "write_passage", "finish_not_found"]
    )
    graph_builder.add_edge("generate", "check_support")
    graph_builder.add_conditional_edges(
        "check_support",
        nodes.route_after_support,
        ["check_usefulness", "revise", "finish_not_found"],
    )
    graph_builder.add_conditional_edges(
        "revise", nodes.route_after_revision, ["check_usefulness", "finish_not_found"]
    )
    graph_builder.add_conditional_edges(
        "check_usefulness",
        nodes.route_after_usefulness,
        ["finish_answered", "write_passage", "finish_not_found"],
    )
    graph_builder.add_edge("finish_answered", END)
    graph_builder.add_edge("finish_not_found", END)
    return graph_builder.compile()


async def run_self_rag(
    self_rag_graph: CompiledStateGraph, question: str, document_ids: list[str]
) -> SelfRagResult:
    """Answer one sub-question from these documents and return what was found.

    The run config (the request budget, D-43) reaches the nodes through
    LangGraph's context, as for every node of a run.
    """
    final_state = await self_rag_graph.ainvoke(
        {
            "question": question,
            "document_ids": list(document_ids),
            "search_passages": [],
            "revisions_used": 0,
        },
        config={"run_name": "self_rag"},
    )
    return SelfRagResult(
        status=final_state["status"],
        answer=final_state.get("answer", ""),
        evidence=final_state.get("relevant", []),
        search_passages=final_state.get("search_passages", []),
        revised=final_state.get("revised", False),
    )
