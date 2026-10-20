"""Tests of the brief renderer.

CONCEPT: tracing-from-day-one. The brief file is part of the public record of
a run, so its shape is tested: sections in order, and only cited sources.
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
