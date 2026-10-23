"""Scripted tests of extract_web_findings: P-02 and the checks in code.

CONCEPT: verbatim-extraction, tested with scripted-model-testing. Each test
scripts the model's P-02 reply and checks which findings code keeps, which it
drops, and why. A real ModelGateway runs around ScriptedModel, so the schema
binding and the repair retry run as they do live.
"""

import re

import pytest

from graph.extraction import extract_web_findings, occurs_in
from graph.mcp_client import WebPage
from graph.state import SubQuestion
from tests.conftest import ScriptedModel

SUB_QUESTION = SubQuestion(id="Q2", text="Who are Symbotic's largest customers?", route="web")

PAGE_ONE = WebPage(
    url="https://example.com/customers",
    title="Symbotic customers",
    text=(
        "Symbotic sells warehouse automation to large retailers.\n"
        "Walmart’s contract is the company’s largest, and accounted for\n"
        "most of its revenue in the year. Revenue was $1.2 billion — up from\n"
        "the year before. Target and Albertsons are also customers."
    ),
    truncated=False,
)
PAGE_TWO = WebPage(
    url="https://example.com/news",
    title="Symbotic news",
    text="C&S Wholesale Grocers was Symbotic's first customer, starting with one site.",
    truncated=False,
)
PAGES = [PAGE_ONE, PAGE_TWO]


def proposed(url: str, passage: str, answer: str = "An answer.") -> dict:
    return {"url": url, "passage": passage, "answer": answer}


def p02_reply(*findings: dict, found: bool = True) -> dict:
    return {"found": found, "findings": list(findings)}


async def extract(make_gateway, reply: dict, pages=PAGES, max_passage_chars: int = 1500):
    extraction_model = ScriptedModel(scripted_replies=[reply])
    result = await extract_web_findings(
        make_gateway(decision_model=extraction_model),
        SUB_QUESTION,
        pages,
        max_passage_chars=max_passage_chars,
    )
    return result, extraction_model


async def test_tc_res_03_kept_findings_are_complete(make_gateway):
    result, extraction_model = await extract(
        make_gateway,
        p02_reply(
            proposed(
                PAGE_ONE.url,
                "Walmart's contract is the company's largest",
                "Walmart is the largest customer.",
            ),
            proposed(
                PAGE_TWO.url,
                "C&S Wholesale Grocers was Symbotic's first customer",
                "C&S was the first customer.",
            ),
        ),
    )

    assert [finding.id for finding in result.findings] == ["Q2-F1", "Q2-F2"]
    for finding, page in zip(result.findings, PAGES, strict=True):
        assert re.fullmatch(r"Q2-F\d+", finding.id)
        assert finding.sub_question_id == "Q2"
        assert finding.found is True
        assert finding.source_type == "web"
        assert finding.answer
        assert finding.locator == page.url
        assert finding.title == page.title
        assert occurs_in(finding.passage, page.text)
    assert result.dropped == []
    assert extraction_model.scripted_replies == []  # exactly one model call


async def test_tc_res_04_a_paraphrased_passage_is_dropped(make_gateway):
    result, _ = await extract(
        make_gateway,
        p02_reply(
            # "increased" is not what the page says: a paraphrase.
            proposed(PAGE_ONE.url, "Walmart's contract increased the company's revenue the most"),
            proposed(PAGE_TWO.url, "C&S Wholesale Grocers was Symbotic's first customer"),
        ),
    )

    assert [finding.locator for finding in result.findings] == [PAGE_TWO.url]
    assert result.findings[0].id == "Q2-F1"  # numbered in the order kept
    assert [dropped.reason for dropped in result.dropped] == ["passage not in page"]


@pytest.mark.parametrize(
    ("passage", "is_copy"),
    [
        ("Walmart's contract is the company's largest", True),  # straight for curly
        ("Revenue was $1.2 billion - up from the year before", True),  # hyphen for em dash
        ("and accounted for most of its revenue in the year", True),  # line break as space
        ("Revenue was $2.1 billion - up from the year before", False),  # a digit changed
        ("revenue was $1.2 billion - up from the year before", False),  # case changed
        ("Revenue was $1.2 billion... the year before", False),  # shortened with dots
    ],
)
def test_the_matching_rule(passage, is_copy):
    assert occurs_in(passage, PAGE_ONE.text) is is_copy


async def test_a_url_that_was_not_given_is_dropped(make_gateway):
    result, _ = await extract(
        make_gateway,
        p02_reply(
            proposed("https://example.com/other", "Target and Albertsons are also customers.")
        ),
    )

    assert [dropped.reason for dropped in result.dropped] == ["url not given"]


async def test_a_passage_credited_to_the_wrong_page_is_dropped(make_gateway):
    # The text is on page one; the model says page two.
    result, _ = await extract(
        make_gateway,
        p02_reply(proposed(PAGE_TWO.url, "Target and Albertsons are also customers.")),
    )

    assert [dropped.reason for dropped in result.dropped] == ["passage not in page"]


async def test_a_passage_under_five_words_is_dropped(make_gateway):
    result, _ = await extract(
        make_gateway, p02_reply(proposed(PAGE_ONE.url, "Target and Albertsons"))
    )

    assert [dropped.reason for dropped in result.dropped] == ["too short"]


async def test_when_nothing_survives_there_is_one_no_evidence_finding(make_gateway):
    result, _ = await extract(
        make_gateway,
        p02_reply(proposed(PAGE_ONE.url, "Amazon is Symbotic's largest customer today.")),
    )

    assert len(result.findings) == 1
    assert result.findings[0].id == "Q2-F1"
    assert result.findings[0].found is False
    assert len(result.dropped) == 1


async def test_the_checks_decide_not_the_found_flag(make_gateway):
    result, _ = await extract(
        make_gateway,
        p02_reply(proposed(PAGE_ONE.url, "Target and Albertsons are also customers."), found=False),
    )

    assert result.findings[0].found is True


async def test_at_most_three_findings_are_kept(make_gateway):
    result, _ = await extract(
        make_gateway,
        p02_reply(
            proposed(PAGE_ONE.url, "Symbotic sells warehouse automation to large retailers."),
            proposed(PAGE_ONE.url, "Walmart's contract is the company's largest"),
            proposed(PAGE_ONE.url, "Target and Albertsons are also customers."),
            proposed(PAGE_TWO.url, "C&S Wholesale Grocers was Symbotic's first customer"),
        ),
    )

    assert [finding.id for finding in result.findings] == ["Q2-F1", "Q2-F2", "Q2-F3"]


async def test_a_passage_is_cut_after_it_has_been_checked(make_gateway):
    result, _ = await extract(
        make_gateway,
        p02_reply(
            proposed(PAGE_ONE.url, "Symbotic sells warehouse automation to large retailers.")
        ),
        max_passage_chars=20,
    )

    assert result.findings[0].passage == "Symbotic sells wareh"
    assert result.dropped == []


async def test_no_pages_means_no_model_call(make_gateway):
    # ScriptedModel with no replies fails the test if it is called at all.
    result = await extract_web_findings(
        make_gateway(decision_model=ScriptedModel()), SUB_QUESTION, []
    )

    assert [finding.found for finding in result.findings] == [False]


async def test_the_model_is_given_the_question_and_every_page(make_gateway):
    _, extraction_model = await extract(make_gateway, p02_reply(found=False))

    _, human_message = extraction_model.received_conversations[0]
    assert SUB_QUESTION.text in human_message.content
    for page in PAGES:
        assert f'<page url="{page.url}">' in human_message.content
