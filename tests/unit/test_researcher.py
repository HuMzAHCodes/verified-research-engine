"""Scripted tests of the temporary researcher node.

CONCEPT: sequential-graph. The search is faked with FakeSearchClient, so these
tests need no Tavily key and no network.
"""

from graph.nodes.researcher import researcher_node
from graph.state import SubQuestion
from settings import settings
from tests.conftest import FakeSearchClient, search_result

FIRST_QUESTION = SubQuestion(id="Q1", text="What does the company sell?", route="web")
SECOND_QUESTION = SubQuestion(id="Q2", text="Who are its customers?", route="web")


async def test_results_become_findings_numbered_within_each_sub_question():
    search_client = FakeSearchClient(
        {
            FIRST_QUESTION.text: [search_result(1), search_result(2)],
            SECOND_QUESTION.text: [search_result(3)],
        }
    )

    researcher_update = await researcher_node(
        {"sub_questions": [FIRST_QUESTION, SECOND_QUESTION]}, search_client=search_client
    )

    findings = researcher_update["findings"]
    assert [finding.id for finding in findings] == ["Q1-F1", "Q1-F2", "Q2-F1"]
    assert findings[0].sub_question_id == "Q1"
    assert findings[0].locator == "https://example.com/page-1"
    assert findings[0].title == "Page 1"
    assert findings[0].passage == "Text of page 1."
    assert search_client.received_queries == [FIRST_QUESTION.text, SECOND_QUESTION.text]


async def test_at_most_three_results_are_kept():
    search_client = FakeSearchClient(
        {FIRST_QUESTION.text: [search_result(number) for number in range(1, 6)]}
    )

    researcher_update = await researcher_node(
        {"sub_questions": [FIRST_QUESTION]}, search_client=search_client
    )

    assert [finding.id for finding in researcher_update["findings"]] == ["Q1-F1", "Q1-F2", "Q1-F3"]


async def test_long_passages_are_cut_to_the_limit():
    long_text = "x" * (settings.vre_passage_max_chars + 500)
    search_client = FakeSearchClient({FIRST_QUESTION.text: [search_result(1, long_text)]})

    researcher_update = await researcher_node(
        {"sub_questions": [FIRST_QUESTION]}, search_client=search_client
    )

    assert len(researcher_update["findings"][0].passage) == settings.vre_passage_max_chars


async def test_no_results_gives_one_no_evidence_finding():
    search_client = FakeSearchClient({})  # every query returns nothing

    researcher_update = await researcher_node(
        {"sub_questions": [FIRST_QUESTION]}, search_client=search_client
    )

    [no_evidence_finding] = researcher_update["findings"]
    assert no_evidence_finding.id == "Q1-F1"
    assert no_evidence_finding.found is False
    assert no_evidence_finding.passage == ""
