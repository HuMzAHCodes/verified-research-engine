"""The document researcher: checked document evidence for one sub-question.

CONCEPT: subgraph-wrapper (D-57). One document_researcher branch per
sub-question routed to `document` or `both`, running beside the web branches.
A branch receives a DocumentResearchTask (its sub-question and the run's
document ids, nothing else) and:

1. runs the Self-RAG subgraph on it (concept self-rag), called inside this
   node, not added as a node (decision 2): RunState and SelfRagState share no
   keys, and the subgraph's working fields must stay inside it. The run's
   config reaches the subgraph through LangChain's context, so its requests
   count against the run's budget and its trace nests under the run;
2. turns the result into findings the writer already uses:
   - not_found: one finding with found=False (decision 5);
   - otherwise one finding per cited chunk, numbered Q2-D1, Q2-D2 (choice B),
     located by page (<document_id>#page=<n>), the chunk's exact text as the
     passage, the subgraph's answer as the answer.

Which chunks are cited (choice A): only the relevant chunks that hold one of
IsSUP's supporting quotes, matched with the same rule as every other quote
(D-35, graph/extraction.py). The grader judges topic, so a relevant chunk may
not support the answer (page 95 beside the Walmart answer, concept self-rag),
and a citation must point only at what supports the claim. After a revision
the answer is made of checked quotes, and the evidence is already exactly
their chunks. When no quote matches (the model paraphrased), every relevant
chunk is cited and the fallback is logged, so the live run shows how often.

Document text never reaches web search (decision 6): this node has no
toolbox. The subgraph and the store are injected by build_graph (D-25).
"""

import logging
from collections.abc import Mapping
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from graph.document_store import DocumentNotStored, DocumentStore, RetrievedChunk
from graph.self_rag.graph import run_self_rag
from graph.self_rag.nodes import checked_quotes
from graph.self_rag.state import SelfRagResult
from graph.state import DocumentResearchTask, Finding, SubQuestion
from settings import settings

logger = logging.getLogger(__name__)

# Routes that send a sub-question to the documents (decision 1). "both" also
# goes to the web (graph/nodes/researcher.py, WEB_ROUTES).
DOCUMENT_ROUTES = frozenset({"document", "both"})


def cited_chunks(sub_question_id: str, result: SelfRagResult) -> list[RetrievedChunk]:
    """The chunks a document answer cites (choice A), in retrieval order.

    Revised answers: their evidence is already exactly the chunks their
    checked quotes came from. Otherwise: the relevant chunks that hold one of
    IsSUP's quotes; with none, every relevant chunk, logged.
    """
    if result.revised:
        return list(result.evidence)
    # checked_quotes trims the quote marks a model puts around a quote.
    _, quoted_chunks = checked_quotes(result.support_quotes, result.evidence)
    if quoted_chunks:
        return quoted_chunks
    logger.warning(
        "%s: no IsSUP quote found in the evidence (%d quotes); citing all %d relevant chunks",
        sub_question_id,
        len(result.support_quotes),
        len(result.evidence),
    )
    return list(result.evidence)


def file_names_by_document_id(
    document_store: DocumentStore, document_ids: list[str]
) -> dict[str, str]:
    """Each document's file name, for finding titles; the id if it is not stored."""
    file_names: dict[str, str] = {}
    for document_id in dict.fromkeys(document_ids):
        try:
            file_names[document_id] = document_store.load_document(document_id).file_name
        except DocumentNotStored:
            file_names[document_id] = document_id
    return file_names


def document_findings(
    sub_question: SubQuestion,
    result: SelfRagResult,
    file_names: Mapping[str, str],
    *,
    max_passage_chars: int = settings.vre_passage_max_chars,
) -> list[Finding]:
    """The findings for one sub-question from its Self-RAG result.

    Ids by code (G-5), in the D series so a `both` sub-question's web findings
    (Q2-F1) and document findings (Q2-D1) never collide (choice B).
    """
    if result.status == "not_found":
        return [
            Finding(
                id=f"{sub_question.id}-D1",
                sub_question_id=sub_question.id,
                found=False,
                source_type="document",
            )
        ]
    return [
        Finding(
            id=f"{sub_question.id}-D{position}",
            sub_question_id=sub_question.id,
            found=True,
            source_type="document",
            locator=f"{chunk.document_id}#page={chunk.page}",
            title=f"{file_names.get(chunk.document_id, chunk.document_id)}, page {chunk.page}",
            # The chunk is an exact slice of its page; cut only for the prompt (G-7).
            passage=chunk.text[:max_passage_chars],
            answer=result.answer,
        )
        for position, chunk in enumerate(cited_chunks(sub_question.id, result), start=1)
    ]


async def document_researcher_node(
    task: DocumentResearchTask,
    *,
    self_rag_graph: CompiledStateGraph,
    document_store: DocumentStore,
) -> dict[str, list[Any]]:
    """Research one sub-question in the run's documents: one parallel branch."""
    sub_question = task["sub_question"]
    result = await run_self_rag(self_rag_graph, sub_question.text, task["document_ids"])
    findings = document_findings(
        sub_question, result, file_names_by_document_id(document_store, task["document_ids"])
    )
    logger.info(
        "%s: document research %s, %d finding(s), %d search pass(es)%s",
        sub_question.id,
        result.status,
        sum(1 for finding in findings if finding.found),
        len(result.search_passages),
        ", revised" if result.revised else "",
    )
    return {"findings": findings}
