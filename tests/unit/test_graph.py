"""The whole Phase 3 graph, end to end, with no network.

CONCEPT: sequential-graph, tools-through-mcp, and from Phase 3 parallel-fan-out
and shared-throttle (the run's call budget).
build_graph is called exactly as in production, but with scripted models and
FakeToolbox. A topic goes in and a draft comes out, and each part of the run can
be checked: the plan, the branches, what was searched and read, the findings
and their answers, and the number of model calls on each tier.

The branches call P-02 at the same time, so P-02 replies are keyed to each
sub-question's text (replies_by_prompt_text), never given in call order: in
call order a branch can receive another branch's reply (test_scripted_model.py).
"""

import asyncio
import time
from uuid import uuid4

import pytest

from graph.llm import MODEL_CALL_BUDGET_KEY, ModelCallBudget, ModelCallBudgetExceeded
from graph.mcp_client import SearchHit, ToolCallError
from graph.nodes.researcher import WebResearchFailed
from graph.nodes.writer import WriterError
from graph.supervisor import PHASE_TAG, build_graph, research_run_config
from settings import settings
from tests.conftest import FakeToolbox, ScriptedModel, search_hit, web_page

TOPIC = "Due diligence on Symbotic"
FIRST_QUESTION = "What does Symbotic sell?"
SECOND_QUESTION = "Who are its customers?"


def copied_from(page_number: int, answer: str) -> dict:
    page = web_page(page_number)
    return {"url": page.url, "passage": page.text.split(". ")[0] + ".", "answer": answer}


def plan(*question_texts: str, route: str = "web") -> dict:
    """The planner's reply: these sub-questions, in this order."""
    return {"sub_questions": [{"text": text, "route": route} for text in question_texts]}


def one_section_per_question(question_count: int) -> list[dict]:
    """Writer replies: an outline with one section per sub-question, then each section."""
    outline = {
        "sections": [
            {"title": f"Section {n}", "sub_question_ids": [f"Q{n}"]}
            for n in range(1, question_count + 1)
        ]
    }
    sections = [
        {"title": f"Section {n}", "text": f"Text {n}. [Q{n}-F1]"}
        for n in range(1, question_count + 1)
    ]
    return [outline, *sections]


class SlowToolbox(FakeToolbox):
    """FakeToolbox whose searches take time, and which records when each starts and ends."""

    def __init__(self, search_seconds_by_query: dict[str, float], **fake_toolbox_args) -> None:
        super().__init__(**fake_toolbox_args)
        self.search_seconds_by_query = search_seconds_by_query
        self.search_started_at: list[float] = []
        self.search_finished_at: list[float] = []
        self.finished_queries: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        self.search_started_at.append(time.perf_counter())
        await asyncio.sleep(self.search_seconds_by_query[query])
        self.search_finished_at.append(time.perf_counter())
        self.finished_queries.append(query)
        return await super().search(query, max_results)


async def run_graph(graph, topic: str = TOPIC, document_ids: list[str] | None = None) -> dict:
    return await graph.ainvoke(
        {"topic": topic, "document_ids": document_ids or []},
        config=research_run_config(topic, uuid4()),
    )


async def test_a_topic_goes_in_and_a_draft_comes_out(make_gateway):
    # The decision tier answers the planner first; the two P-02 calls then
    # arrive together, so each reply is keyed to its sub-question's text.
    decision_model = ScriptedModel(
        scripted_replies=[plan(FIRST_QUESTION, SECOND_QUESTION)],
        replies_by_prompt_text={
            FIRST_QUESTION: [
                {
                    "found": True,
                    "findings": [
                        copied_from(1, "Warehouse robots."),
                        copied_from(2, "Automation software."),
                    ],
                }
            ],
            SECOND_QUESTION: [{"found": True, "findings": [copied_from(3, "A large retailer.")]}],
        },
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

    final_state = await run_graph(graph)

    assert [sub_question.id for sub_question in final_state["sub_questions"]] == ["Q1", "Q2"]
    assert sorted(toolbox.received_queries) == sorted([FIRST_QUESTION, SECOND_QUESTION])
    assert [finding.id for finding in final_state["findings"]] == ["Q1-F1", "Q1-F2", "Q2-F1"]
    assert final_state["findings"][2].answer == "A large retailer."
    assert final_state["search_failures"] == []
    assert [section.title for section in final_state["draft"]] == ["Products", "Customers"]
    assert final_state["draft"][0].text == "Symbotic sells warehouse robots. [Q1-F1]"
    # Decision tier: one planner call and one P-02 per sub-question.
    # Writer tier: one outline and two sections.
    assert len(decision_model.received_conversations) == 3
    assert len(writer_model.received_conversations) == 3
    # The writer is shown the answer P-02 wrote (D-28).
    _, products_prompt = writer_model.received_conversations[1]
    assert "Answer: Warehouse robots." in products_prompt.content


async def test_tc_res_01_and_tc_res_08_one_branch_per_sub_question_all_findings_kept(
    make_gateway,
):
    # Six sub-questions, the cap: six branches, each finding two passages on
    # its own two pages (pages 2n-1 and 2n).
    question_texts = [f"Question number {n}?" for n in range(1, 7)]
    decision_model = ScriptedModel(
        scripted_replies=[plan(*question_texts)],
        replies_by_prompt_text={
            text: [
                {
                    "found": True,
                    "findings": [copied_from(2 * n - 1, f"A{n}."), copied_from(2 * n, f"B{n}.")],
                }
            ]
            for n, text in enumerate(question_texts, start=1)
        },
    )
    toolbox = FakeToolbox(
        hits_by_query={
            text: [search_hit(2 * n - 1), search_hit(2 * n)]
            for n, text in enumerate(question_texts, start=1)
        },
        pages_by_url={web_page(n).url: web_page(n) for n in range(1, 13)},
    )
    graph = build_graph(
        make_gateway(
            decision_model=decision_model,
            writer_model=ScriptedModel(scripted_replies=one_section_per_question(6)),
        ),
        toolbox,
    )

    branch_updates = []
    async for update in graph.astream(
        {"topic": TOPIC, "document_ids": []},
        config=research_run_config(TOPIC, uuid4()),
        stream_mode="updates",
    ):
        if "web_researcher" in update:
            branch_updates.append(update["web_researcher"])

    # TC-RES-01: six branches, each with the findings of exactly one sub-question,
    # and every sub-question searched once with its own text.
    assert len(branch_updates) == 6
    branch_question_ids = [
        {finding.sub_question_id for finding in update["findings"]} for update in branch_updates
    ]
    assert sorted(branch_question_ids, key=min) == [{f"Q{n}"} for n in range(1, 7)]
    assert sorted(toolbox.received_queries) == sorted(question_texts)

    # TC-RES-08: every finding from every branch kept, none doubled, ids unique.
    # (The order they end up in is checked in the next test.)
    all_findings = [finding for update in branch_updates for finding in update["findings"]]
    assert len(all_findings) == 12
    assert len({finding.id for finding in all_findings}) == 12


async def test_tc_res_08_the_final_state_keeps_every_branch_in_the_planners_order(
    make_gateway,
):
    question_texts = [f"Question number {n}?" for n in range(1, 4)]
    decision_model = ScriptedModel(
        scripted_replies=[plan(*question_texts)],
        replies_by_prompt_text={
            text: [{"found": True, "findings": [copied_from(n, f"A{n}.")]}]
            for n, text in enumerate(question_texts, start=1)
        },
    )
    # Q1's search is the slowest and Q3's the fastest, so the branches finish
    # in reverse order: Q3, Q2, Q1.
    toolbox = SlowToolbox(
        search_seconds_by_query=dict(zip(question_texts, (0.4, 0.3, 0.2), strict=True)),
        hits_by_query={text: [search_hit(n)] for n, text in enumerate(question_texts, start=1)},
        pages_by_url={web_page(n).url: web_page(n) for n in range(1, 4)},
    )
    graph = build_graph(
        make_gateway(
            decision_model=decision_model,
            writer_model=ScriptedModel(scripted_replies=one_section_per_question(3)),
        ),
        toolbox,
    )

    final_state = await run_graph(graph)

    # The branches ran at the same time: every search had started before the
    # first one finished. One after another, each would start only after the
    # previous one ended. Checked by overlap, not by total time, because the
    # total includes the planner and the writer and depends on the machine
    # (measured: 0.45 s on Linux, 0.80 s on Windows for the same run).
    assert len(toolbox.search_started_at) == 3
    assert max(toolbox.search_started_at) < min(toolbox.search_finished_at)
    # They finished in reverse, yet the state keeps the planner's order
    # (parallel-fan-out, decision 4), and so does the draft.
    assert toolbox.finished_queries == list(reversed(question_texts))
    assert [finding.id for finding in final_state["findings"]] == ["Q1-F1", "Q2-F1", "Q3-F1"]
    assert [section.title for section in final_state["draft"]] == [
        "Section 1",
        "Section 2",
        "Section 3",
    ]


async def test_tc_res_07_a_branch_with_no_evidence_leaves_the_others(make_gateway):
    decision_model = ScriptedModel(
        scripted_replies=[plan(FIRST_QUESTION, SECOND_QUESTION)],
        # Only Q1 reaches P-02: Q2's search finds nothing, so it makes no call.
        replies_by_prompt_text={
            FIRST_QUESTION: [{"found": True, "findings": [copied_from(1, "Robots.")]}]
        },
    )
    toolbox = FakeToolbox(
        hits_by_query={FIRST_QUESTION: [search_hit(1)], SECOND_QUESTION: []},
        pages_by_url={web_page(1).url: web_page(1)},
    )
    graph = build_graph(
        make_gateway(
            decision_model=decision_model,
            writer_model=ScriptedModel(scripted_replies=one_section_per_question(1)),
        ),
        toolbox,
    )

    final_state = await run_graph(graph)

    assert [(finding.id, finding.found) for finding in final_state["findings"]] == [
        ("Q1-F1", True),
        ("Q2-F1", False),
    ]
    assert final_state["search_failures"] == []  # an empty result is not a failed search
    assert [section.title for section in final_state["draft"]] == ["Section 1"]


async def test_every_branch_failing_its_search_stops_the_run(make_gateway):
    search_down = ToolCallError("search timed out after 15 s")
    toolbox = FakeToolbox(hits_by_query={FIRST_QUESTION: search_down, SECOND_QUESTION: search_down})
    graph = build_graph(
        make_gateway(
            decision_model=ScriptedModel(scripted_replies=[plan(FIRST_QUESTION, SECOND_QUESTION)])
        ),
        toolbox,
    )

    with pytest.raises(WebResearchFailed, match=r"every web search failed \(2 of 2\)"):
        await run_graph(graph)


async def test_no_web_sub_question_reaches_the_next_node_instead_of_ending_silently(
    make_gateway,
):
    # Decision 7. With documents uploaded the planner keeps a "document" route,
    # so no web branch starts. The run must still go on: here it reaches the
    # writer, which has no evidence to write from (the document researcher
    # arrives in Phase 4). Without the fix the run would end with no error.
    toolbox = FakeToolbox()
    graph = build_graph(
        make_gateway(
            decision_model=ScriptedModel(scripted_replies=[plan(FIRST_QUESTION, route="document")])
        ),
        toolbox,
    )

    with pytest.raises(WriterError, match="no sub-question found any evidence"):
        await run_graph(graph, document_ids=["annual-report.pdf"])
    assert toolbox.received_queries == []


def test_research_run_config_names_tags_and_identifies_the_run():
    run_id = uuid4()

    run_config = research_run_config(TOPIC, run_id)

    assert run_config["run_name"] == "research-run"
    assert run_config["run_id"] == run_id
    assert run_config["tags"] == [PHASE_TAG] == ["phase-3"]
    assert run_config["metadata"] == {"topic": TOPIC}


def test_every_run_gets_a_fresh_budget_of_the_configured_size():
    first_budget = research_run_config(TOPIC, uuid4())["configurable"][MODEL_CALL_BUDGET_KEY]
    second_budget = research_run_config(TOPIC, uuid4())["configurable"][MODEL_CALL_BUDGET_KEY]

    assert first_budget is not second_budget
    assert first_budget.limit == settings.vre_max_model_calls_per_run
    assert first_budget.requests_sent == 0


def two_question_graph(make_gateway):
    """Planner, two parallel branches, and a two-section writer: six requests."""
    decision_model = ScriptedModel(
        scripted_replies=[plan(FIRST_QUESTION, SECOND_QUESTION)],
        replies_by_prompt_text={
            FIRST_QUESTION: [{"found": True, "findings": [copied_from(1, "Robots.")]}],
            SECOND_QUESTION: [{"found": True, "findings": [copied_from(2, "Retailers.")]}],
        },
    )
    toolbox = FakeToolbox(
        hits_by_query={FIRST_QUESTION: [search_hit(1)], SECOND_QUESTION: [search_hit(2)]},
        pages_by_url={web_page(n).url: web_page(n) for n in (1, 2)},
    )
    writer_model = ScriptedModel(scripted_replies=one_section_per_question(2))
    return build_graph(
        make_gateway(decision_model=decision_model, writer_model=writer_model), toolbox
    )


async def test_parallel_branches_count_into_the_runs_one_budget(make_gateway):
    run_config = research_run_config(TOPIC, uuid4())

    await two_question_graph(make_gateway).ainvoke(
        {"topic": TOPIC, "document_ids": []}, config=run_config
    )

    # Planner 1, P-02 2 (one per branch, sent at the same time), outline 1,
    # sections 2: every request of the run, counted in one place.
    assert run_config["configurable"][MODEL_CALL_BUDGET_KEY].requests_sent == 6


async def test_a_run_that_uses_up_its_budget_stops(make_gateway):
    run_config = research_run_config(TOPIC, uuid4())
    # Room for the planner and both P-02 calls, not for the writer.
    run_config["configurable"][MODEL_CALL_BUDGET_KEY] = ModelCallBudget(limit=3)

    with pytest.raises(ModelCallBudgetExceeded, match="VRE_MAX_MODEL_CALLS_PER_RUN=3"):
        await two_question_graph(make_gateway).ainvoke(
            {"topic": TOPIC, "document_ids": []}, config=run_config
        )
