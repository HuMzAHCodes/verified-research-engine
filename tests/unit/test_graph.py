"""The whole phase 1 graph, end to end, with no network.

CONCEPT: sequential-graph. build_graph is called exactly as in production,
but with scripted models and a fake search. A topic goes in and a draft comes
out, and each part of the run can be checked: the plan, the findings, the
number of model calls on each tier.
"""

from graph.supervisor import build_graph
from tests.conftest import FakeSearchClient, ScriptedModel, search_result

TOPIC = "Due diligence on Symbotic"


async def test_a_topic_goes_in_and_a_draft_comes_out(make_gateway):
    planner_model = ScriptedModel(
        scripted_replies=[
            {
                "sub_questions": [
                    {"text": "What does Symbotic sell?", "route": "web"},
                    {"text": "Who are its customers?", "route": "web"},
                ]
            }
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
    search_client = FakeSearchClient(
        {
            "What does Symbotic sell?": [search_result(1), search_result(2)],
            "Who are its customers?": [search_result(3)],
        }
    )
    graph = build_graph(
        make_gateway(decision_model=planner_model, writer_model=writer_model),
        search_client,
    )

    final_state = await graph.ainvoke({"topic": TOPIC, "document_ids": []})

    assert [sub_question.id for sub_question in final_state["sub_questions"]] == ["Q1", "Q2"]
    assert [finding.id for finding in final_state["findings"]] == ["Q1-F1", "Q1-F2", "Q2-F1"]
    assert [section.title for section in final_state["draft"]] == ["Products", "Customers"]
    assert final_state["draft"][0].text == "Symbotic sells warehouse robots. [Q1-F1]"
    # One planner call on the decision tier; one outline and two sections on the writer tier.
    assert len(planner_model.received_conversations) == 1
    assert len(writer_model.received_conversations) == 3
