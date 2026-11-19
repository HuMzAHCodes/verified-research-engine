"""Live check of the whole Self-RAG subgraph on the stored annual report.

CONCEPT: self-rag, its acceptance check end to end. Runs only on request
(`uv run pytest -m live`), never in CI. Needs the annual report ingested
(tests/live/test_ingestion_live.py) and no server: the subgraph reads the
store and calls Mistral through the gateway.

Two kinds of question:

1. Labelled questions, each with the words only its answer chunk holds. The
   real P-15 writes a passage shaped like the answer (D-56), hybrid retrieval
   searches with it and the question, and P-10 to P-14 grade, answer, and
   check. Each must come back answered, with its answer chunk among the
   evidence and its key figure in the answer. The first is the Walmart
   question (page 23, "approximately 85% of our total revenue"); more are
   added as they are labelled, so a fix is never tuned to one question.
2. A question the report cannot answer. It must come back not_found, with no
   answer and no evidence: the subgraph says it found nothing instead of
   making something up.

Each run is counted by the run budget, so the check prints how many requests
it took; the caps keep the worst case near 12 chat calls and 2 embeddings.

For a labelled question it also prints where the answer chunk ranked for
each pass's search (question and passage together, as the run searched; one
more embedding request each, outside the run's budget). Ranks 1 to 6 reach
the grader, so a not_found with the chunk outside them is a retrieval miss,
and with it inside them a judgement miss. Run with --log-cli-level=INFO to
see every node's decision as well.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from app.ingestion import file_fingerprint
from graph.document_store import DocumentStore
from graph.llm import MODEL_CALL_BUDGET_KEY, ModelCallBudget, ModelGateway
from graph.retrieval import RETRIEVED_CHUNKS, HybridRetriever
from graph.self_rag.graph import build_self_rag_graph, run_self_rag

pytestmark = pytest.mark.live

ANNUAL_REPORT_PATH = Path("data/uploads/symbotic-annual-report.pdf")
UNANSWERABLE_QUESTION = "What is the name of Symbotic's office cafeteria supplier?"
# How deep to look for the answer chunk when reporting its rank; only the
# first RETRIEVED_CHUNKS (6) of these reach the grader.
RANK_REPORT_DEPTH = 20


@dataclass(frozen=True)
class LabelledQuestion:
    """A question whose answer chunk is known by words only it holds."""

    question: str
    # Lower-case words that together appear only in the answer chunk
    # (checked by searching the stored chunks), so the check survives a
    # change to chunking.
    answer_chunk_words: tuple[str, ...]
    # Text the answer must contain, usually the key figure.
    expected_in_answer: str


LABELLED_QUESTIONS = [
    LabelledQuestion(
        question="How much of Symbotic's revenue comes from Walmart?",
        answer_chunk_words=("walmart", "% of our total revenue"),
        expected_in_answer="85",
    ),
]


def holds_words(chunk, words: tuple[str, ...]) -> bool:
    text = " ".join(chunk.text.lower().split())
    return all(word in text for word in words)


@pytest.fixture
def stored_report_id() -> str:
    if not ANNUAL_REPORT_PATH.is_file():
        pytest.skip(f"put the annual report at {ANNUAL_REPORT_PATH} first (D-45)")
    stored_document = DocumentStore().find_by_fingerprint(file_fingerprint(ANNUAL_REPORT_PATH))
    if stored_document is None:
        pytest.skip("ingest the annual report first: tests/live/test_ingestion_live.py")
    return stored_document.document_id


async def run_counted(question: str, document_ids: list[str]):
    """Run the subgraph as a node of a run would, with a budget that counts requests."""
    gateway = ModelGateway()
    self_rag_graph = build_self_rag_graph(gateway, HybridRetriever(DocumentStore(), gateway))
    budget = ModelCallBudget(limit=30)

    async def as_a_node_of_a_run(_: object):
        return await run_self_rag(self_rag_graph, question, document_ids)

    started_at = time.perf_counter()
    result = await RunnableLambda(as_a_node_of_a_run).ainvoke(
        None, config={"configurable": {MODEL_CALL_BUDGET_KEY: budget}}
    )
    return result, budget.requests_sent, time.perf_counter() - started_at


def report(question: str, result, requests_sent: int, seconds: float, answer_words=()) -> None:
    print(f"\nquestion: {question}")
    for pass_number, passage in enumerate(result.search_passages, start=1):
        print(f"search passage {pass_number}: {passage!r}")
    print(f"status: {result.status}   requests: {requests_sent}   time: {seconds:.1f} s")
    print(f"answer: {result.answer!r}")
    for chunk in result.evidence:
        marker = "ANSWER" if answer_words and holds_words(chunk, answer_words) else "      "
        print(f"  evidence: page {chunk.page:>3} {marker} {' '.join(chunk.text.split())[:70]}")


async def report_answer_chunk_ranks(
    labelled: LabelledQuestion, search_passages: list[str], document_ids: list[str]
) -> None:
    """Where the answer chunk ranked for each pass's search: question and passage merged."""
    deep_retriever = HybridRetriever(
        DocumentStore(), ModelGateway(), retrieved_chunks=RANK_REPORT_DEPTH
    )
    for pass_number, passage in enumerate(search_passages, start=1):
        retrieved = await deep_retriever.retrieve_for_texts(
            [labelled.question, passage], document_ids, prompt_id="LIVE-RANK"
        )
        answer_rank = next(
            (
                rank
                for rank, chunk in enumerate(retrieved, start=1)
                if holds_words(chunk, labelled.answer_chunk_words)
            ),
            None,
        )
        reached = answer_rank is not None and answer_rank <= RETRIEVED_CHUNKS
        shown_rank = answer_rank if answer_rank is not None else f"below {RANK_REPORT_DEPTH}"
        print(
            f"  pass {pass_number}: answer chunk rank {shown_rank}"
            f" ({'reached' if reached else 'did not reach'} the grader)"
        )


@pytest.mark.parametrize(
    "labelled", LABELLED_QUESTIONS, ids=[case.question for case in LABELLED_QUESTIONS]
)
async def test_a_labelled_question_is_answered_from_its_answer_chunk(stored_report_id, labelled):
    result, requests_sent, seconds = await run_counted(labelled.question, [stored_report_id])
    report(labelled.question, result, requests_sent, seconds, labelled.answer_chunk_words)
    await report_answer_chunk_ranks(labelled, result.search_passages, [stored_report_id])

    assert result.status in ("answered", "answered_with_quotes")
    assert any(holds_words(chunk, labelled.answer_chunk_words) for chunk in result.evidence), (
        "the answer chunk must be among the evidence"
    )
    assert labelled.expected_in_answer in result.answer


async def test_a_question_the_report_cannot_answer_comes_back_not_found(stored_report_id):
    result, requests_sent, seconds = await run_counted(UNANSWERABLE_QUESTION, [stored_report_id])
    report(UNANSWERABLE_QUESTION, result, requests_sent, seconds)

    assert result.status == "not_found"
    assert (result.answer, result.evidence) == ("", [])
