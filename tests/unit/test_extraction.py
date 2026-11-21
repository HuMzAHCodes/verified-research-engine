"""Scripted tests of extract_web_findings: P-02 and the checks in code.

CONCEPT: verbatim-extraction, tested with scripted-model-testing. Each test
scripts the model's P-02 reply and checks which findings code keeps, which it
drops, and why. A real ModelGateway runs around ScriptedModel, so the schema
binding and the repair retry run as they do live.
"""

import re

import pytest

from graph.extraction import cut_at_word, extract_web_findings, find_copy, occurs_in
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


@pytest.mark.parametrize(
    ("page_text", "passage"),
    [
        # "un" cut off the front: the meaning flips.
        (
            "The firm was unprofitable in each of the last three years.",
            "profitable in each of the last three years.",
        ),
        # The last digit cut off the end: a different figure.
        ("Total revenue for the year was $1.25 billion.", "Total revenue for the year was $1.2"),
        # A figure cut at its thousands separator or decimal point.
        ("The company had 1,250 employees at the end of the year.", "The company had 1"),
        ("Revenue grew by 12.5 percent over the prior year.", "5 percent over the prior year."),
        # A word cut at its end.
        ("Symbotic sells warehouse automation to large retailers.", "Symbotic sells warehouse au"),
    ],
)
def test_a_passage_cut_inside_a_word_or_figure_is_not_a_copy(page_text, passage):
    # CONCEPT: verbatim-extraction, D-61 (outside test review, F-1).
    assert occurs_in(passage, page_text) is False


@pytest.mark.parametrize(
    ("page_text", "passage"),
    [
        # A figure that ends the sentence: the full stop is not part of it.
        ("Total revenue for the year was $1.2.", "Total revenue for the year was $1.2"),
        # Punctuation at either end of the passage.
        ("He said: (the largest customer) is Walmart.", "(the largest customer) is Walmart."),
        # The first place is cut inside a word, a later one is whole: the later one counts.
        (
            "It was unprofitable in each year. It was profitable in each year.",
            "profitable in each year.",
        ),
    ],
)
def test_a_passage_on_whole_words_is_a_copy(page_text, passage):
    assert occurs_in(passage, page_text) is True


@pytest.mark.parametrize(
    ("page_text", "passage"),
    [
        # A PDF ligature, typed as two letters.
        (
            "Revenue for the \ufb01scal year ended September 27 grew.",
            "Revenue for the fiscal year ended September 27 grew.",
        ),
        # A decomposed accent on the page, a composed one typed.
        (
            "Northwind is owned by Socie\u0301te\u0301 Ge\u0301ne\u0301rale and others.",
            "Northwind is owned by Soci\u00e9t\u00e9 G\u00e9n\u00e9rale and others.",
        ),
        # A soft hyphen where the page may break a word.
        (
            "Our ware\u00adhouse systems are installed at forty sites.",
            "Our warehouse systems are installed at forty sites.",
        ),
    ],
)
def test_the_same_text_in_another_unicode_form_still_matches(page_text, passage):
    # CONCEPT: verbatim-extraction, D-61 (outside test review, F-12).
    assert occurs_in(passage, page_text) is True


def test_a_superscript_digit_is_not_folded_into_a_plain_one():
    # Why NFKC was not chosen (D-61): it would make "10²" and "102" the same.
    assert (
        occurs_in("The area is 102 square metres in all.", "The area is 10² square metres in all.")
        is False
    )


def test_the_copy_is_found_as_a_span_of_the_pages_own_text():
    page_text = "Intro.  Walmart\u2019s   contract is\nthe company\u2019s largest. End."

    copy_start, copy_end = find_copy("Walmart's contract is the company's largest.", page_text)

    assert page_text[copy_start:copy_end] == (
        "Walmart\u2019s   contract is\nthe company\u2019s largest."
    )


def test_text_is_cut_at_a_word():
    assert cut_at_word("Symbotic sells warehouse automation.", 20) == "Symbotic sells"
    assert cut_at_word("Symbotic sells warehouse automation.", 100) == (
        "Symbotic sells warehouse automation."
    )
    # One word longer than the limit: nothing to cut at, so it is cut hard.
    assert cut_at_word("Supercalifragilistic", 5) == "Super"


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

    # At a word: "wareh" would no longer be a copy of whole words (D-61).
    assert result.findings[0].passage == "Symbotic sells"
    assert occurs_in(result.findings[0].passage, PAGE_ONE.text)
    assert result.dropped == []


async def test_a_kept_passage_is_the_pages_own_text(make_gateway):
    # CONCEPT: verbatim-extraction, D-62 (outside test review, F-17). The model
    # typed straight apostrophes; the page has curly ones and a line break.
    result, _ = await extract(
        make_gateway,
        p02_reply(
            proposed(
                PAGE_ONE.url, "Walmart's contract is the company's largest, and accounted for most"
            )
        ),
    )

    assert result.findings[0].passage == (
        "Walmart\u2019s contract is the company\u2019s largest, and accounted for\nmost"
    )
    assert result.findings[0].passage in PAGE_ONE.text


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
