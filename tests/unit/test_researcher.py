"""Scripted tests of the researcher: search, read, extract, all through the toolbox.

CONCEPT: tools-through-mcp. FakeToolbox plays search-mcp and ScriptedModel plays
P-02, behind a real ModelGateway. The checks of verbatim extraction run for
real, so a finding here exists only if its passage is on its page.
"""

import pytest

from graph.mcp_client import SearchServerUnavailable, ToolCallError
from graph.nodes.researcher import researcher_node
from graph.state import SubQuestion
from tests.conftest import FakeToolbox, ScriptedModel, search_hit, web_page

FIRST_QUESTION = SubQuestion(id="Q1", text="What does the company sell?", route="web")
SECOND_QUESTION = SubQuestion(id="Q2", text="Who are its customers?", route="web")


def copied_from(page_number: int, answer: str = "An answer.") -> dict:
    """A proposed P-02 finding whose passage really is on page n."""
    page = web_page(page_number)
    return {"url": page.url, "passage": page.text.split(". ")[0] + ".", "answer": answer}


def p02_reply(*findings: dict) -> dict:
    return {"found": bool(findings), "findings": list(findings)}


async def run_researcher(make_gateway, toolbox, p02_replies, sub_questions):
    extraction_model = ScriptedModel(scripted_replies=list(p02_replies))
    researcher_update = await researcher_node(
        {"sub_questions": sub_questions},
        gateway=make_gateway(decision_model=extraction_model),
        toolbox=toolbox,
    )
    return researcher_update["findings"], extraction_model


async def test_tc_res_02_every_search_and_read_goes_through_the_toolbox(make_gateway):
    toolbox = FakeToolbox(
        hits_by_query={
            FIRST_QUESTION.text: [search_hit(1), search_hit(2)],
            SECOND_QUESTION.text: [search_hit(3)],
        },
        pages_by_url={web_page(n).url: web_page(n) for n in (1, 2, 3)},
    )

    findings, extraction_model = await run_researcher(
        make_gateway,
        toolbox,
        [p02_reply(copied_from(1), copied_from(2)), p02_reply(copied_from(3))],
        [FIRST_QUESTION, SECOND_QUESTION],
    )

    assert toolbox.received_queries == [FIRST_QUESTION.text, SECOND_QUESTION.text]
    assert toolbox.received_urls == [search_hit(n).url for n in (1, 2, 3)]
    assert [finding.id for finding in findings] == ["Q1-F1", "Q1-F2", "Q2-F1"]
    assert all(finding.found and finding.answer for finding in findings)
    assert extraction_model.scripted_replies == []  # one P-02 call per sub-question


async def test_unreadable_pages_are_skipped_and_rank_order_is_kept(make_gateway):
    # Page 1 cannot be read; pages 2 to 5 can. The first three readable, in
    # search-rank order, are 2, 3, and 4, so page 5 never reaches P-02.
    toolbox = FakeToolbox(
        hits_by_query={FIRST_QUESTION.text: [search_hit(n) for n in range(1, 6)]},
        pages_by_url={web_page(n).url: web_page(n) for n in range(2, 6)},
    )

    _, extraction_model = await run_researcher(
        make_gateway, toolbox, [p02_reply(copied_from(2))], [FIRST_QUESTION]
    )

    assert len(toolbox.received_urls) == 5  # every hit was read
    _, p02_prompt = extraction_model.received_conversations[0]
    for page_number in (2, 3, 4):
        assert web_page(page_number).url in p02_prompt.content
    assert web_page(1).url not in p02_prompt.content
    assert web_page(5).url not in p02_prompt.content


async def test_a_failed_search_costs_only_its_own_sub_question(make_gateway):
    toolbox = FakeToolbox(
        hits_by_query={
            FIRST_QUESTION.text: ToolCallError("search failed upstream"),
            SECOND_QUESTION.text: [search_hit(3)],
        },
        pages_by_url={web_page(3).url: web_page(3)},
    )

    findings, _ = await run_researcher(
        make_gateway, toolbox, [p02_reply(copied_from(3))], [FIRST_QUESTION, SECOND_QUESTION]
    )

    assert [(finding.id, finding.found) for finding in findings] == [
        ("Q1-F1", False),
        ("Q2-F1", True),
    ]


async def test_no_readable_page_means_no_evidence_and_no_model_call(make_gateway):
    toolbox = FakeToolbox(hits_by_query={FIRST_QUESTION.text: [search_hit(1)]})  # no pages

    # ScriptedModel with no replies fails the test if P-02 is called.
    findings, _ = await run_researcher(make_gateway, toolbox, [], [FIRST_QUESTION])

    assert [finding.found for finding in findings] == [False]


async def test_a_dropped_passage_leaves_the_others(make_gateway):
    toolbox = FakeToolbox(
        hits_by_query={FIRST_QUESTION.text: [search_hit(1)]},
        pages_by_url={web_page(1).url: web_page(1)},
    )
    paraphrased = {"url": web_page(1).url, "passage": "Page one mainly describes the text."}

    findings, _ = await run_researcher(
        make_gateway,
        toolbox,
        [p02_reply({**paraphrased, "answer": "A."}, copied_from(1))],
        [FIRST_QUESTION],
    )

    assert [finding.id for finding in findings] == ["Q1-F1"]
    assert findings[0].passage == "Main text of page 1."


@pytest.mark.parametrize("where", ["search", "fetch"])
async def test_a_server_that_is_gone_stops_the_run(make_gateway, where):
    server_gone = SearchServerUnavailable("search-mcp stopped answering")
    if where == "search":
        toolbox = FakeToolbox(hits_by_query={FIRST_QUESTION.text: server_gone})
    else:
        toolbox = FakeToolbox(
            hits_by_query={FIRST_QUESTION.text: [search_hit(1), search_hit(2)]},
            pages_by_url={web_page(1).url: web_page(1), web_page(2).url: server_gone},
        )

    with pytest.raises(SearchServerUnavailable, match="stopped answering"):
        await run_researcher(make_gateway, toolbox, [], [FIRST_QUESTION])
