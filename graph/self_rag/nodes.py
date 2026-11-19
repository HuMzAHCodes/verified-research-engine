"""The Self-RAG subgraph's nodes and the decisions between them.

CONCEPT: self-rag (D-50, D-51, D-53, D-56). Each node does one step and returns only
the fields it changed. Every model call goes through the gateway, so it is
throttled, retried, counted against the run's budget, and traced under its
prompt id (decision 5). The gateway and the retriever are given to the nodes
by build_self_rag_graph, so tests pass ScriptedModel and a store in a
temporary folder (D-25).

    write_passage   P-15  a passage shaped like the answer, to search with (HyDE, D-56)
    retrieve              hybrid retrieval with the question and the passage, merged,
                          over the run's documents only (D-52, D-53, D-56)
    grade           P-10  which retrieved chunks are on topic, in one call (choice B)
    generate        P-11  an answer from the relevant chunks only
    check_support   P-12  IsSUP: is every claim in the chunks?
    revise          P-13  the answer as quotes; quotes not found word for word dropped
    check_usefulness P-14 IsUSE: does it answer the question? (never sees the chunks)
    finish_answered / finish_not_found: set the status

The route_* functions are the decisions between nodes. They are pure: they
read the state and name the next node, so each rule is tested without a model.

Every node logs what it decided at INFO (pages retrieved, chunks kept, the
draft, both verdicts, quotes kept), so a run that ends not_found shows where
it turned (Phase 4, self-rag, step 4: the first live failure printed only the
end state, and could not be read).
"""

import logging
from collections.abc import Mapping
from typing import Any

from graph.document_store import RetrievedChunk
from graph.extraction import MIN_PASSAGE_WORDS, normalise_for_matching, occurs_in
from graph.llm import ModelGateway
from graph.prompts import self_rag as self_rag_prompts
from graph.retrieval import HybridRetriever
from graph.schemas import (
    DraftAnswer,
    QuotedAnswer,
    RelevantChunks,
    SearchPassage,
    SupportCheck,
    UsefulnessCheck,
)
from graph.self_rag.state import MAX_REVISIONS, MAX_SEARCH_PASSES

logger = logging.getLogger(__name__)

# The prompt id the retrieval's embedding request is counted and traced under.
RETRIEVAL_CALL_ID = "SELF-RAG-RETRIEVE"


def may_search_again(state: Mapping[str, Any]) -> bool:
    """True while the sub-question has used fewer than its two search passes."""
    return len(state.get("search_passages", [])) < MAX_SEARCH_PASSES


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def write_passage_node(state: Mapping[str, Any], *, gateway: ModelGateway) -> dict[str, Any]:
    """P-15: a passage shaped like the answer, to search with, and a clean pass (D-56).

    The first time, from the question alone. The second time, the model is
    shown the passage that failed and asked for the answer in other words.
    An empty reply falls back to the question itself, so retrieval always
    has something to search with. The passage is made up by design (figures
    as placeholders): it is only searched with, never answered from.
    """
    previous_passages = state.get("search_passages", [])
    prompt_id, prompt_version = self_rag_prompts.SEARCH_PASSAGE_PROMPT
    reply = await gateway.structured(
        SearchPassage,
        self_rag_prompts.build_search_passage_messages(
            state["question"],
            previous_passage=previous_passages[-1] if previous_passages else None,
        ),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
    search_passage = " ".join(reply.passage.split()) or state["question"]
    logger.info("self-rag: search passage %d: %r", len(previous_passages) + 1, search_passage)
    return {
        "search_passages": [*previous_passages, search_passage],
        # A second pass starts clean; the counters above carry over.
        "retrieved": [],
        "relevant": [],
        "answer": "",
        "support": None,
        "support_quotes": [],
        "usefulness": None,
        "revised": False,
    }


async def retrieve_node(state: Mapping[str, Any], *, retriever: HybridRetriever) -> dict[str, Any]:
    """The best chunks for the question and the latest passage, merged (D-56).

    The question is searched as asked beside the passage, so a passage that
    guesses the wrong shape cannot steer the search alone. Both go in one
    embedding request. Only the run's documents are searched.
    """
    retrieved = await retriever.retrieve_for_texts(
        [state["question"], state["search_passages"][-1]],
        state["document_ids"],
        prompt_id=RETRIEVAL_CALL_ID,
    )
    logger.info(
        "self-rag: retrieved %d chunks, pages %s", len(retrieved), [c.page for c in retrieved]
    )
    return {"retrieved": retrieved}


async def grade_node(state: Mapping[str, Any], *, gateway: ModelGateway) -> dict[str, Any]:
    """P-10: the retrieved chunks the model judges on topic, in retrieval order.

    Labels the model did not receive are dropped and logged (rule G-5: ids
    come from code). Nothing retrieved means nothing to grade and no call.
    """
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"relevant": []}
    labelled_chunks = self_rag_prompts.label_chunks(retrieved)
    prompt_id, prompt_version = self_rag_prompts.GRADING_PROMPT
    reply = await gateway.structured(
        RelevantChunks,
        self_rag_prompts.build_grading_messages(state["question"], labelled_chunks),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
    returned_labels = {label.strip().upper() for label in reply.relevant_labels}
    unknown_labels = returned_labels - labelled_chunks.keys()
    if unknown_labels:
        logger.warning("self-rag: P-10 returned unknown labels %s, dropped", sorted(unknown_labels))
    relevant = [chunk for label, chunk in labelled_chunks.items() if label in returned_labels]
    logger.info(
        "self-rag: P-10 kept %d of %d chunks: %s",
        len(relevant),
        len(retrieved),
        [f"{label} p{chunk.page}" for label, chunk in labelled_chunks.items() if chunk in relevant],
    )
    return {"relevant": relevant}


async def generate_node(state: Mapping[str, Any], *, gateway: ModelGateway) -> dict[str, Any]:
    """P-11: an answer written from the relevant chunks only."""
    prompt_id, prompt_version = self_rag_prompts.ANSWER_PROMPT
    reply = await gateway.structured(
        DraftAnswer,
        self_rag_prompts.build_answer_messages(
            state["question"], self_rag_prompts.label_chunks(state["relevant"])
        ),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
    answer = reply.answer.strip()
    logger.info("self-rag: P-11 draft: %r", answer[:300])
    return {"answer": answer}


async def check_support_node(state: Mapping[str, Any], *, gateway: ModelGateway) -> dict[str, Any]:
    """P-12, IsSUP: is the answer fully, partly, or not supported by the relevant chunks?"""
    prompt_id, prompt_version = self_rag_prompts.SUPPORT_PROMPT
    reply = await gateway.structured(
        SupportCheck,
        self_rag_prompts.build_support_messages(
            state["question"], state["answer"], self_rag_prompts.label_chunks(state["relevant"])
        ),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
    logger.info("self-rag: IsSUP %s, evidence %r", reply.support, reply.evidence[:3])
    # The quotes are kept for citations (D-57, choice A); they are checked
    # against the chunks where they are used, not here.
    return {"support": reply.support, "support_quotes": list(reply.evidence)}


def checked_quotes(
    proposed_quotes: list[str], chunks: list[RetrievedChunk]
) -> tuple[list[str], list[RetrievedChunk]]:
    """The quotes found word for word in a chunk, and the chunks they came from.

    Decision 8: the same rule as web passages (D-35, graph/extraction.py):
    whitespace, quote marks, and dashes are made uniform on both sides, then
    the quote must occur in the chunk. Quotes under MIN_PASSAGE_WORDS words
    are dropped too: "Walmart" alone is in 94 chunks and proves nothing.
    Duplicates are kept once; source chunks keep their retrieval order.
    """
    kept_quotes: list[str] = []
    source_chunk_ids: set[str] = set()
    for proposed_quote in proposed_quotes:
        quote = proposed_quote.strip()
        if len(normalise_for_matching(quote).split()) < MIN_PASSAGE_WORDS:
            continue
        sources = [chunk for chunk in chunks if occurs_in(quote, chunk.text)]
        if sources and quote not in kept_quotes:
            kept_quotes.append(quote)
            source_chunk_ids.update(chunk.chunk_id for chunk in sources)
    return kept_quotes, [chunk for chunk in chunks if chunk.chunk_id in source_chunk_ids]


async def revise_node(state: Mapping[str, Any], *, gateway: ModelGateway) -> dict[str, Any]:
    """P-13: the answer rebuilt from quotes, keeping only quotes the code finds.

    The kept quotes become the answer, one per line, and only the chunks they
    came from stay as evidence. No quote left means no answer: the route then
    ends in not_found. Either way a revision is used.
    """
    prompt_id, prompt_version = self_rag_prompts.REVISION_PROMPT
    reply = await gateway.structured(
        QuotedAnswer,
        self_rag_prompts.build_revision_messages(
            state["question"], state["answer"], self_rag_prompts.label_chunks(state["relevant"])
        ),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
    kept_quotes, source_chunks = checked_quotes(reply.quotes, state["relevant"])
    logger.info("self-rag: P-13 kept %d of %d quotes", len(kept_quotes), len(reply.quotes))
    dropped_count = len(reply.quotes) - len(kept_quotes)
    if dropped_count:
        logger.warning(
            "self-rag: P-13 gave %d quotes not found in the chunks, dropped", dropped_count
        )
    return {
        "answer": "\n".join(kept_quotes),
        "relevant": source_chunks,
        # The answer is now the checked quotes; IsSUP's quotes no longer apply.
        "support_quotes": [],
        "revised": True,
        "revisions_used": state.get("revisions_used", 0) + 1,
    }


async def check_usefulness_node(
    state: Mapping[str, Any], *, gateway: ModelGateway
) -> dict[str, Any]:
    """P-14, IsUSE: does the answer give what was asked? It never sees the chunks (choice C)."""
    prompt_id, prompt_version = self_rag_prompts.USEFULNESS_PROMPT
    reply = await gateway.structured(
        UsefulnessCheck,
        self_rag_prompts.build_usefulness_messages(state["question"], state["answer"]),
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )
    logger.info("self-rag: IsUSE %s: %s", reply.usefulness, reply.reason)
    return {"usefulness": reply.usefulness}


def finish_answered_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """The answer passed its checks."""
    return {"status": "answered_with_quotes" if state.get("revised") else "answered"}


def finish_not_found_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """No answer passed its checks: nothing is claimed, and no evidence is kept."""
    return {"status": "not_found", "answer": "", "relevant": [], "support_quotes": []}


# ---------------------------------------------------------------------------
# Decisions between nodes
# ---------------------------------------------------------------------------


def route_at_start(state: Mapping[str, Any]) -> str:
    """No documents for the run: nothing to search, so no call is spent."""
    return "write_passage" if state.get("document_ids") else "finish_not_found"


def route_after_grading(state: Mapping[str, Any]) -> str:
    """Relevant chunks: answer. None: search once more with a new passage, else stop."""
    if state.get("relevant"):
        return "generate"
    return "write_passage" if may_search_again(state) else "finish_not_found"


def route_after_support(state: Mapping[str, Any]) -> str:
    """Fully supported: check usefulness. Otherwise revise once per sub-question, else stop."""
    if state.get("support") == "fully_supported":
        return "check_usefulness"
    if state.get("revisions_used", 0) < MAX_REVISIONS:
        return "revise"
    return "finish_not_found"


def route_after_revision(state: Mapping[str, Any]) -> str:
    """Quotes kept: they are supported by construction, so on to usefulness.

    A second IsSUP call would only confirm what the code already checked
    word for word (decision 8). No quote kept: stop.
    """
    return "check_usefulness" if state.get("answer") else "finish_not_found"


def route_after_usefulness(state: Mapping[str, Any]) -> str:
    """Useful: done. Not useful: search once more with a new passage, else stop."""
    if state.get("usefulness") == "useful":
        return "finish_answered"
    return "write_passage" if may_search_again(state) else "finish_not_found"
