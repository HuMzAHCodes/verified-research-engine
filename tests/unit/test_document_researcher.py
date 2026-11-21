"""Tests of the document researcher: Self-RAG's result turned into findings.

CONCEPT: subgraph-wrapper (D-57). The pure parts (which chunks are cited, how
findings are built) are tested on hand-made results; the node itself runs the
real Self-RAG subgraph with a scripted chat model, scripted vectors, and a
real store in a temporary folder, as in test_self_rag.py.

TC-RES-06: a document sub-question gives findings located by page, with the
cited chunk's exact text and the subgraph's answer.
"""

import logging

import pytest
from langchain_core.runnables import RunnableLambda

from graph.document_store import (
    DocumentStore,
    RetrievedChunk,
    StoredChunk,
    StoredDocument,
    chunk_id_for,
)
from graph.llm import MODEL_CALL_BUDGET_KEY, ModelCallBudget
from graph.nodes.document_researcher import (
    cited_chunks,
    document_findings,
    document_researcher_node,
    file_names_by_document_id,
)
from graph.retrieval import HybridRetriever
from graph.self_rag.graph import build_self_rag_graph
from graph.self_rag.nodes import without_quote_marks
from graph.self_rag.state import SelfRagResult
from graph.state import SubQuestion
from tests.conftest import ScriptedEmbeddings, ScriptedModel

SUB_QUESTION = SubQuestion(
    id="Q2", text="How much of Contoso's revenue comes from Northwind?", route="document"
)
PASSAGE = "Northwind accounted for approximately X% of our total revenue."

ANSWER_TEXT = "Northwind, our largest customer, accounted for 60% of our total revenue in 2025."
ACQUISITION_TEXT = "We paid $200.0 million in cash to acquire a robotics business from Northwind."
BOARD_TEXT = "The board of directors meets four times a year."

EAST = [1.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0]
UP = [0.0, 0.0, 1.0]


def chunk(position: int, text: str, page: int, document_id: str = "report") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id_for(document_id, position),
        document_id=document_id,
        page=page,
        chunk_index=position,
        text=text,
        score=0.5,
    )


ANSWER_CHUNK = chunk(0, ANSWER_TEXT, page=23)
ACQUISITION_CHUNK = chunk(1, ACQUISITION_TEXT, page=95)


def answered(*support_quotes: str, revised: bool = False, evidence=None) -> SelfRagResult:
    return SelfRagResult(
        status="answered_with_quotes" if revised else "answered",
        answer="Northwind accounted for 60% of Contoso's total revenue in 2025.",
        evidence=evidence if evidence is not None else [ANSWER_CHUNK, ACQUISITION_CHUNK],
        search_passages=[PASSAGE],
        revised=revised,
        support_quotes=list(support_quotes),
    )


# ---------------------------------------------------------------------------
# Which chunks are cited (choice A)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ('"accounted for 60% of our total revenue"', "accounted for 60% of our total revenue"),
        ("“accounted for 60%”", "accounted for 60%"),
        ("...accounted for 60% of our total revenue...", "accounted for 60% of our total revenue"),
        ("  '…our largest customer'  ", "our largest customer"),
        ("accounted for 60%", "accounted for 60%"),
    ],
)
def test_quote_marks_and_ellipses_around_a_quote_are_removed(quote, expected):
    assert without_quote_marks(quote) == expected


def test_only_chunks_holding_an_issup_quote_are_cited():
    # The acquisition chunk is on topic (it names Northwind) but backs nothing.
    result = answered("Northwind, our largest customer, accounted for 60% of our total revenue")

    assert cited_chunks("Q2", result) == [ANSWER_CHUNK]


def test_a_quote_wrapped_in_quote_marks_still_finds_its_chunk():
    # As seen live: IsSUP returned its quotes inside quote marks.
    result = answered('"Northwind, our largest customer, accounted for 60% of our total revenue"')

    assert cited_chunks("Q2", result) == [ANSWER_CHUNK]


def test_with_no_quote_found_every_relevant_chunk_is_cited_and_logged(caplog):
    caplog.set_level(logging.WARNING)
    # A paraphrase, and a quote too short to prove anything (under 5 words).
    result = answered("Northwind provides most of the money Contoso earns", "Northwind")

    assert cited_chunks("Q2", result) == [ANSWER_CHUNK, ACQUISITION_CHUNK]
    assert "Q2: no IsSUP quote found in the evidence (2 quotes)" in caplog.text


def test_a_revised_answer_cites_exactly_its_evidence_with_no_fallback(caplog):
    caplog.set_level(logging.WARNING)
    # After a revision the evidence is already only the quotes' own chunks,
    # and IsSUP's quotes are empty by design: that is not a failed match.
    result = answered(revised=True, evidence=[ANSWER_CHUNK])

    assert cited_chunks("Q2", result) == [ANSWER_CHUNK]
    assert "no IsSUP quote found" not in caplog.text


# ---------------------------------------------------------------------------
# Findings (TC-RES-06, choice B, decision 5)
# ---------------------------------------------------------------------------


def test_tc_res_06_findings_are_located_by_page_with_the_exact_chunk_text():
    result = answered(
        "Northwind, our largest customer, accounted for 60%",
        "We paid $200.0 million in cash to acquire",
    )

    findings = document_findings(SUB_QUESTION, result, {"report": "contoso-annual-report.pdf"})

    assert [finding.id for finding in findings] == ["Q2-D1", "Q2-D2"]
    first = findings[0]
    assert first.sub_question_id == "Q2"
    assert (first.found, first.source_type) == (True, "document")
    assert first.locator == "report#page=23"
    assert first.title == "contoso-annual-report.pdf, page 23"
    assert first.passage == ANSWER_TEXT
    assert first.answer == result.answer
    assert findings[1].locator == "report#page=95"


def test_not_found_gives_one_finding_with_found_false():
    result = SelfRagResult(status="not_found", search_passages=[PASSAGE, PASSAGE])

    findings = document_findings(SUB_QUESTION, result, {})

    assert [(f.id, f.found, f.source_type, f.passage) for f in findings] == [
        ("Q2-D1", False, "document", "")
    ]


def test_a_long_chunk_is_cut_to_the_passage_limit():
    long_chunk = chunk(0, "Northwind accounted for 60% of revenue. " * 20, page=4)
    result = answered(revised=True, evidence=[long_chunk])

    findings = document_findings(SUB_QUESTION, result, {}, max_passage_chars=100)

    assert findings[0].passage == long_chunk.text[:100]
    # Without a stored file name, the title falls back to the document id.
    assert findings[0].title == "report, page 4"


def test_file_names_come_from_the_store_and_fall_back_to_the_id(tmp_path):
    store = DocumentStore(tmp_path / "vectors")
    save_document(store, "report", [(ANSWER_TEXT, EAST)])

    assert file_names_by_document_id(store, ["report", "missing", "report"]) == {
        "report": "report.pdf",
        "missing": "missing",
    }


# ---------------------------------------------------------------------------
# The node, with the real subgraph
# ---------------------------------------------------------------------------


def save_document(store: DocumentStore, document_id: str, texts_and_vectors) -> None:
    chunks = [
        StoredChunk(
            chunk_id=chunk_id_for(document_id, chunk_index),
            document_id=document_id,
            page=chunk_index + 1,
            chunk_index=chunk_index,
            text=text,
        )
        for chunk_index, (text, _) in enumerate(texts_and_vectors)
    ]
    store.save(
        StoredDocument(
            document_id=document_id,
            file_name=f"{document_id}.pdf",
            fingerprint=f"fingerprint-{document_id}",
            page_count=len(chunks),
            chunk_count=len(chunks),
            embedding_model="mistral-embed",
        ),
        chunks,
        [vector for _, vector in texts_and_vectors],
    )


SCRIPTED_BEST_PATH = [
    {"passage": PASSAGE},
    {"relevant_labels": ["C1", "C2"]},
    {"answer": "Northwind accounted for 60% of Contoso's total revenue in 2025."},
    {
        "evidence": ['"Northwind, our largest customer, accounted for 60% of our total revenue"'],
        "support": "fully_supported",
    },
    {"reason": "Gives the share.", "usefulness": "useful"},
]


@pytest.fixture
def node_parts(make_gateway, tmp_path):
    """A store holding one document, and a gateway with this test's scripted replies."""
    store = DocumentStore(tmp_path / "vectors")
    save_document(
        store, "report", [(ANSWER_TEXT, EAST), (ACQUISITION_TEXT, NORTH), (BOARD_TEXT, UP)]
    )

    def build(scripted_replies):
        chat_model = ScriptedModel(scripted_replies=list(scripted_replies))
        embedding_model = ScriptedEmbeddings(
            vectors_by_text={SUB_QUESTION.text: EAST, PASSAGE: EAST}
        )
        gateway = make_gateway(decision_model=chat_model, embedding_model=embedding_model)
        self_rag_graph = build_self_rag_graph(gateway, HybridRetriever(store, gateway))
        return self_rag_graph, store

    return build


async def test_the_node_runs_self_rag_and_cites_only_the_supporting_chunk(node_parts):
    self_rag_graph, store = node_parts(SCRIPTED_BEST_PATH)

    update = await document_researcher_node(
        {"sub_question": SUB_QUESTION, "document_ids": ["report"]},
        self_rag_graph=self_rag_graph,
        document_store=store,
    )

    # The grader kept two chunks; only the one holding IsSUP's quote is cited.
    assert [(f.id, f.locator, f.title) for f in update["findings"]] == [
        ("Q2-D1", "report#page=1", "report.pdf, page 1")
    ]
    assert update["findings"][0].passage == ANSWER_TEXT


async def test_the_subgraphs_requests_count_against_the_runs_budget(node_parts):
    # Decision 2: the run's config reaches the subgraph called inside the node.
    self_rag_graph, store = node_parts(SCRIPTED_BEST_PATH)
    budget = ModelCallBudget(limit=30)

    async def as_a_branch_of_a_run(_: object):
        return await document_researcher_node(
            {"sub_question": SUB_QUESTION, "document_ids": ["report"]},
            self_rag_graph=self_rag_graph,
            document_store=store,
        )

    await RunnableLambda(as_a_branch_of_a_run).ainvoke(
        None, config={"configurable": {MODEL_CALL_BUDGET_KEY: budget}}
    )

    # Five chat calls and one embedding request.
    assert budget.requests_sent == 6
