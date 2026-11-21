"""Tests of the brief renderer.

CONCEPT: tracing-from-day-one. The brief file is part of the public record of
a run, so its shape is tested: sections in order, only cited sources, one line
per page, and titles that read as sentences.
"""

from graph.report import cited_finding_ids, render_brief
from graph.state import Finding, Section

TOPIC = "Due diligence on Symbotic"


def web_finding(finding_id: str) -> Finding:
    return Finding(
        id=finding_id,
        sub_question_id=finding_id.split("-")[0],
        found=True,
        source_type="web",
        locator=f"https://example.com/{finding_id}",
        title=f"Page {finding_id}",
        passage="Text.",
    )


DRAFT = [
    Section(title="Products", text="Robots. [Q1-F1] Software. [Q1-F2, Q1-F3]"),
    Section(title="Customers", text="A retailer. [Q2-F1]"),
]
FINDINGS = [web_finding(finding_id) for finding_id in ["Q1-F1", "Q1-F2", "Q1-F3", "Q2-F1", "Q2-F2"]]


def test_cited_ids_include_every_id_in_multi_id_markers():
    assert cited_finding_ids(DRAFT) == {"Q1-F1", "Q1-F2", "Q1-F3", "Q2-F1"}


def test_brief_has_the_title_and_sections_in_order():
    brief = render_brief(TOPIC, DRAFT, FINDINGS)

    assert brief.startswith(f"# {TOPIC}\n")
    assert brief.index("## Products") < brief.index("## Customers") < brief.index("## Sources")


def test_only_cited_findings_are_listed_as_sources():
    brief = render_brief(TOPIC, DRAFT, FINDINGS)

    assert "- **[Q1-F1]** Page Q1-F1. <https://example.com/Q1-F1>" in brief
    assert "[Q2-F2]" not in brief  # fetched, but never cited


def test_no_sources_heading_when_nothing_is_cited():
    brief = render_brief(TOPIC, [Section(title="Empty", text="No markers here.")], FINDINGS)

    assert "## Sources" not in brief


def test_findings_from_one_page_share_one_source_line():
    same_page = "https://example.com/report"
    findings = [
        web_finding("Q1-F1").model_copy(update={"locator": same_page, "title": "Annual report"}),
        web_finding("Q1-F2").model_copy(update={"locator": same_page, "title": "Annual report"}),
        web_finding("Q1-F3"),
    ]

    brief = render_brief(TOPIC, DRAFT[:1], findings)

    assert "- **[Q1-F1] [Q1-F2]** Annual report. <https://example.com/report>" in brief
    assert brief.count(same_page) == 1
    assert "- **[Q1-F3]** Page Q1-F3. <https://example.com/Q1-F3>" in brief


def test_a_title_that_already_ends_a_sentence_gets_no_extra_full_stop():
    findings = [
        web_finding("Q1-F1").model_copy(update={"title": "Symbotic Inc."}),
        web_finding("Q1-F2").model_copy(update={"title": "Is Symbotic a buy?"}),
        web_finding("Q1-F3").model_copy(update={"title": ""}),
    ]

    brief = render_brief(TOPIC, DRAFT[:1], findings)

    assert "Symbotic Inc. <" in brief
    assert "Is Symbotic a buy? <" in brief
    assert "Untitled. <" in brief
    assert ".." not in brief


def test_sources_follow_the_order_of_first_citation():
    # CONCEPT: tracing-from-day-one, outside test review at the Phase 4 close
    # (F-19): the outline put Customers first, so its source comes first.
    draft = [
        Section(title="Customers", text="A retailer. [Q2-F1]"),
        Section(title="Products", text="Robots. [Q1-F1]"),
    ]

    brief = render_brief(TOPIC, draft, FINDINGS)

    assert brief.index("- **[Q2-F1]**") < brief.index("- **[Q1-F1]**")


def test_a_page_title_cannot_add_markdown_to_the_brief():
    # CONCEPT: tracing-from-day-one, outside test review F-24: a title cannot
    # add a link, and an address cannot end its <...> early.
    finding = web_finding("Q1-F1").model_copy(
        update={
            "title": "Read more](https://attacker.example/) [x *bold*",
            "locator": "https://example.com/a>b c",
        }
    )

    brief = render_brief(TOPIC, [Section(title="P", text="Robots. [Q1-F1]")], [finding])

    source_line = brief.splitlines()[-1]
    assert source_line == (
        "- **[Q1-F1]** Read more\\](https://attacker.example/) \\[x \\*bold\\*. "
        "<https://example.com/a%3Eb%20c>"
    )
