"""The whole graph with documents: web and document branches side by side, no network.

CONCEPT: subgraph-wrapper (D-57, D-58). build_graph is called as in production,
with scripted models, FakeToolbox, and a real store in a temporary folder.
One sub-question is routed to `document`, one to `both`, so two document
branches and one web branch run in the same step.

The branches call the model at the same time, so replies are keyed to the
prompt, never given in call order. A `both` sub-question's text is in its
web prompt and in every one of its Self-RAG prompts, so each key is a tuple:
a phrase of the prompt it answers, and the sub-question (tests/conftest.py).

TC-DOC-07: a marker string in the document's chunks never reaches a search,
a page read, or a web branch's prompt.
"""

from uuid import uuid4

import pytest

from graph.document_store import DocumentStore, StoredChunk, StoredDocument, chunk_id_for
from graph.llm import MODEL_CALL_BUDGET_KEY
from graph.mcp_client import ToolCallError
from graph.nodes.researcher import WebResearchFailed
from graph.state import SubQuestion
from graph.supervisor import build_graph, research_run_config, route_to_researchers
from tests.conftest import FakeToolbox, ScriptedEmbeddings, ScriptedModel, search_hit, web_page

TOPIC = "Due diligence on Contoso"
DOCUMENT_ID = "doc-7f3a"
FILE_NAME = "contoso-annual-report.pdf"
DOCUMENT_QUESTION = "How much of Contoso's revenue comes from Northwind?"
BOTH_QUESTION = "Which company is Contoso's largest customer?"

# Found nowhere but in the document: if it shows up anywhere on the web side,
# document text has leaked (TC-DOC-07).
MARKER = "ZEBRA-7731"
REVENUE_TEXT = f"Northwind, our largest customer, accounted for 60% of our total revenue. {MARKER}"
CUSTOMER_TEXT = "Our largest customer is Northwind, which buys our warehouse systems each year."
BOARD_TEXT = "The board of directors meets four times a year."

EAST = [1.0, 0.0, 0.0]
NORTH = [0.0, 1.0, 0.0]
UP = [0.0, 0.0, 1.0]

# A phrase of each prompt kind, to key replies with (graph/prompts/).
P15 = "as a company's annual report would state the"
P10 = "You judge which chunks of text are relevant"
P11 = "You answer a question from chunks"
P12 = "You check whether an ANSWER is supported"
P14 = "You judge whether an ANSWER gives what a QUESTION asks for"
P02 = "You extract evidence from web pages."

DOCUMENT_PASSAGE = "Northwind accounted for X% of our total revenue."
BOTH_PASSAGE = "Our largest customer is X."


def save_document(store: DocumentStore) -> None:
    texts_and_vectors = [(REVENUE_TEXT, EAST), (CUSTOMER_TEXT, NORTH), (BOARD_TEXT, UP)]
    chunks = [
        StoredChunk(
            chunk_id=chunk_id_for(DOCUMENT_ID, position),
            document_id=DOCUMENT_ID,
            page=position + 1,
            chunk_index=position,
            text=text,
        )
        for position, (text, _) in enumerate(texts_and_vectors)
    ]
    store.save(
        StoredDocument(
            document_id=DOCUMENT_ID,
            file_name=FILE_NAME,
            fingerprint="fingerprint-contoso",
            page_count=3,
            chunk_count=3,
            embedding_model="mistral-embed",
        ),
        chunks,
        [vector for _, vector in texts_and_vectors],
    )


def self_rag_best_path(question: str, passage: str, label: str, quote: str, answer: str):
    """The five keyed replies of one document branch's best path."""
    return {
        (P15, question): [{"passage": passage}],
        (P10, question): [{"relevant_labels": [label]}],
        (P11, question): [{"answer": answer}],
        (P12, question): [{"evidence": [quote], "support": "fully_supported"}],
        (P14, question): [{"reason": "Gives it.", "usefulness": "useful"}],
    }


def web_page_reply(page_number: int, answer: str) -> dict:
    page = web_page(page_number)
    return {
        "found": True,
        "findings": [
            {"url": page.url, "passage": page.text.split(". ")[0] + ".", "answer": answer}
        ],
    }


@pytest.fixture
def documents_graph(make_gateway, tmp_path):
    """The graph, its models, and its toolbox, for a plan of one `document` and one `both`."""

    def build(*, web_search_fails: bool = False):
        store = DocumentStore(tmp_path / "vectors")
        save_document(store)
        decision_model = ScriptedModel(
            scripted_replies=[
                {
                    "sub_questions": [
                        {"text": DOCUMENT_QUESTION, "route": "document"},
                        {"text": BOTH_QUESTION, "route": "both"},
                    ]
                }
            ],
            replies_by_prompt_text={
                **self_rag_best_path(
                    DOCUMENT_QUESTION,
                    DOCUMENT_PASSAGE,
                    "C1",
                    "Northwind, our largest customer, accounted for 60%",
                    "Northwind accounted for 60% of Contoso's total revenue.",
                ),
                **self_rag_best_path(
                    BOTH_QUESTION,
                    BOTH_PASSAGE,
                    "C1",
                    "Our largest customer is Northwind, which buys",
                    "Contoso's largest customer is Northwind.",
                ),
                (P02, BOTH_QUESTION): [web_page_reply(1, "Northwind.")],
            },
        )
        writer_model = ScriptedModel(
            scripted_replies=[
                {
                    "sections": [
                        {"title": "Revenue", "sub_question_ids": ["Q1"]},
                        {"title": "Customers", "sub_question_ids": ["Q2"]},
                    ]
                },
                {"title": "Revenue", "text": "Northwind brings 60% of revenue. [Q1-D1]"},
                {"title": "Customers", "text": "Northwind is the largest customer. [Q2-D1]"},
            ]
        )
        embedding_model = ScriptedEmbeddings(
            vectors_by_text={
                DOCUMENT_QUESTION: EAST,
                DOCUMENT_PASSAGE: EAST,
                BOTH_QUESTION: NORTH,
                BOTH_PASSAGE: NORTH,
            }
        )
        search_down = ToolCallError("search timed out after 15 s")
        toolbox = FakeToolbox(
            hits_by_query={BOTH_QUESTION: search_down if web_search_fails else [search_hit(1)]},
            pages_by_url={web_page(1).url: web_page(1)},
        )
        gateway = make_gateway(
            decision_model=decision_model,
            writer_model=writer_model,
            embedding_model=embedding_model,
        )
        graph = build_graph(gateway, toolbox, document_store=store)
        return graph, decision_model, toolbox

    return build


async def run(graph):
    run_config = research_run_config(TOPIC, uuid4())
    final_state = await graph.ainvoke(
        {"topic": TOPIC, "document_ids": [DOCUMENT_ID]}, config=run_config
    )
    return final_state, run_config["configurable"][MODEL_CALL_BUDGET_KEY]


async def test_document_and_both_routes_start_the_right_branches_with_unique_ids(
    documents_graph,
):
    graph, _, toolbox = documents_graph()

    final_state, _ = await run(graph)

    # Decision 1: Q1 (document) one document branch; Q2 (both) a web branch
    # and a document branch. Choice B: their ids never collide.
    findings = {finding.id: finding for finding in final_state["findings"]}
    assert sorted(findings) == ["Q1-D1", "Q2-D1", "Q2-F1"]
    assert findings["Q2-F1"].source_type == "web"
    assert findings["Q1-D1"].source_type == findings["Q2-D1"].source_type == "document"
    # TC-RES-06 through the whole graph: located by page, titled by file name.
    assert findings["Q1-D1"].locator == f"{DOCUMENT_ID}#page=1"
    assert findings["Q1-D1"].title == f"{FILE_NAME}, page 1"
    assert findings["Q2-D1"].locator == f"{DOCUMENT_ID}#page=2"
    # Only the both question was searched on the web.
    assert toolbox.received_queries == [BOTH_QUESTION]
    assert [section.title for section in final_state["draft"]] == ["Revenue", "Customers"]


async def test_tc_doc_07_document_text_never_reaches_web_search(documents_graph):
    graph, decision_model, toolbox = documents_graph()

    final_state, _ = await run(graph)

    # The marker did reach the document findings, so the check below can fail.
    assert any(MARKER in finding.passage for finding in final_state["findings"])
    # Nothing the web side sent or received holds it: searches, page reads,
    # and every prompt of a web branch (P-02).
    assert not any(MARKER in query for query in toolbox.received_queries)
    assert not any(MARKER in url for url in toolbox.received_urls)
    web_prompts = [
        "\n".join(str(message.content) for message in conversation)
        for conversation in decision_model.received_conversations
        if P02 in str(conversation[0].content)
    ]
    assert len(web_prompts) == 1
    assert MARKER not in web_prompts[0]


async def test_the_planner_sees_file_names_and_sizes_the_budget_to_the_plan(documents_graph):
    graph, decision_model, _ = documents_graph()

    _, budget = await run(graph)

    # Decision 7: the file name, not the id.
    planner_prompt = "\n".join(
        str(message.content) for message in decision_model.received_conversations[0]
    )
    assert f"Uploaded files: {FILE_NAME}" in planner_prompt
    assert DOCUMENT_ID not in planner_prompt
    # D-58: one web branch, two document branches, two sections:
    # (1 + 1 + 2 x 11 + 1 + 2) x 2 + 2 x 2 = 58, below the ceiling of 200.
    assert budget.limit == 58
    # Sent: planner 1, P-02 1, two Self-RAG best paths of 5 calls and 1
    # embedding each, outline 1, sections 2.
    assert budget.requests_sent == 1 + 1 + 2 * 6 + 1 + 2


async def test_d_40_still_stops_the_run_when_every_web_search_fails(documents_graph):
    # Choice D: documents answered, but a brief without any public source
    # would look complete, so the run stops as before.
    graph, _, _ = documents_graph(web_search_fails=True)

    with pytest.raises(WebResearchFailed, match=r"every web search failed \(1 of 1\)"):
        await run(graph)


def test_routes_send_web_first_then_document_for_a_both_question():
    sub_questions = [
        SubQuestion(id="Q1", text="A?", route="web"),
        SubQuestion(id="Q2", text="B?", route="both"),
        SubQuestion(id="Q3", text="C?", route="document"),
    ]

    sends = route_to_researchers({"sub_questions": sub_questions, "document_ids": ["d1"]})

    assert [(send.node, send.arg["sub_question"].id) for send in sends] == [
        ("web_researcher", "Q1"),
        ("web_researcher", "Q2"),
        ("document_researcher", "Q2"),
        ("document_researcher", "Q3"),
    ]
    # Decision 4: a web branch gets the sub-question only; a document branch
    # also the run's document ids.
    assert set(sends[0].arg) == {"sub_question"}
    assert sends[3].arg["document_ids"] == ["d1"]
