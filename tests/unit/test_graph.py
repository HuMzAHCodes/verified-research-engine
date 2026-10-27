"""The whole Phase 2 graph, end to end, with no network.

CONCEPT: sequential-graph, and from Phase 2 tools-through-mcp. build_graph is
called exactly as in production, but with scripted models and FakeToolbox. A
topic goes in and a draft comes out, and each part of the run can be checked:
the plan, what was searched and read, the findings and their answers, and the
number of model calls on each tier.
"""

from uuid import uuid4

from graph.supervisor import PHASE_TAG, build_graph, research_run_config
from tests.conftest import FakeToolbox, ScriptedModel, search_hit, web_page

TOPIC = "Due diligence on Symbotic"
FIRST_QUESTION = "What does Symbotic sell?"
SECOND_QUESTION = "Who are its customers?"


def copied_from(page_number: int, answer: str) -> dict:
    page = web_page(page_number)
    return {"url": page.url, "passage": page.text.split(". ")[0] + ".", "answer": answer}


async def test_a_topic_goes_in_and_a_draft_comes_out(make_gateway):
    # The decision tier answers the planner first, then one P-02 per sub-question.
    decision_model = ScriptedModel(
        scripted_replies=[
            {
                "sub_questions": [
                    {"text": FIRST_QUESTION, "route": "web"},
                    {"text": SECOND_QUESTION, "route": "web"},
                ]
            },
            {
                "found": True,
                "findings": [
                    copied_from(1, "Warehouse robots."),
                    copied_from(2, "Automation software."),
                ],
            },
            {"found": True, "findings": [copied_from(3, "A large retailer.")]},
        ]
    )
    writer_model = ScriptedModel(
        scripted_replies=[
            {
                "sections": [
                    {"title": "Products", "sub_question_ids": ["Q1"]},
                    {"title": "Customers", "sub_question_ids": ["Q2"]},
                ]
            },
            {"title": "Products", "text": "Symbotic sells warehouse robots. [Q1-F1]"},
            {"title": "Customers", "text": "Its largest customer is a retailer. [Q2-F1]"},
        ]
    )
    toolbox = FakeToolbox(
        hits_by_query={
            FIRST_QUESTION: [search_hit(1), search_hit(2)],
            SECOND_QUESTION: [search_hit(3)],
        },
        pages_by_url={web_page(n).url: web_page(n) for n in (1, 2, 3)},
    )
    graph = build_graph(
        make_gateway(decision_model=decision_model, writer_model=writer_model), toolbox
    )

    final_state = await graph.ainvoke(
        {"topic": TOPIC, "document_ids": []}, config=research_run_config(TOPIC, uuid4())
    )

    assert [sub_question.id for sub_question in final_state["sub_questions"]] == ["Q1", "Q2"]
    assert toolbox.received_queries == [FIRST_QUESTION, SECOND_QUESTION]
    assert [finding.id for finding in final_state["findings"]] == ["Q1-F1", "Q1-F2", "Q2-F1"]
    assert final_state["findings"][2].answer == "A large retailer."
    assert [section.title for section in final_state["draft"]] == ["Products", "Customers"]
    assert final_state["draft"][0].text == "Symbotic sells warehouse robots. [Q1-F1]"
    # Decision tier: one planner call and one P-02 per sub-question.
    # Writer tier: one outline and two sections.
    assert len(decision_model.received_conversations) == 3
    assert len(writer_model.received_conversations) == 3
    # The writer is shown the answer P-02 wrote (D-28).
    _, products_prompt = writer_model.received_conversations[1]
    assert "Answer: Warehouse robots." in products_prompt.content


def test_research_run_config_names_tags_and_identifies_the_run():
    run_id = uuid4()

    run_config = research_run_config(TOPIC, run_id)

    assert run_config["run_name"] == "research-run"
    assert run_config["run_id"] == run_id
    assert run_config["tags"] == [PHASE_TAG] == ["phase-2"]
    assert run_config["metadata"] == {"topic": TOPIC}
