"""The brief as Markdown: sections, then the sources the brief actually cites.

CONCEPT: tracing-from-day-one. A run's record is its trace plus the brief it
produced. render_brief() is the one place that turns run state into a
readable brief: the phase notebooks use it to write their brief files, and
phase 6's finalize node uses it to build final_report.

Only cited findings are listed as sources, so the list says which evidence the
brief rests on, not everything the researcher fetched. From Phase 2 several
findings often come from one page, so the list has one line per page, naming
every cited finding taken from it; each [Q2-F1] in the text still points to
exactly one line.
"""

from graph.nodes.writer import MARKER_PATTERN
from graph.state import Finding, Section

SENTENCE_END = (".", "?", "!")


def cited_finding_ids(draft: list[Section]) -> set[str]:
    """Every finding id that appears in a marker anywhere in the draft."""
    cited_ids: set[str] = set()
    for section in draft:
        for marker_match in MARKER_PATTERN.finditer(section.text):
            cited_ids.update(cited_id.strip() for cited_id in marker_match.group(1).split(","))
    return cited_ids


def source_title(finding: Finding) -> str:
    """The page title as a sentence: a full stop only if it has no ending of its own."""
    title = finding.title.strip() or "Untitled"
    return title if title.endswith(SENTENCE_END) else f"{title}."


def source_lines(cited_findings: list[Finding]) -> list[str]:
    """One line per page, in the order pages are first cited, with all its finding ids."""
    findings_by_page: dict[str, list[Finding]] = {}
    for finding in cited_findings:
        findings_by_page.setdefault(finding.locator, []).append(finding)
    return [
        f"- **{' '.join(f'[{finding.id}]' for finding in page_findings)}** "
        f"{source_title(page_findings[0])} <{locator}>"
        for locator, page_findings in findings_by_page.items()
    ]


def render_brief(topic: str, draft: list[Section], findings: list[Finding]) -> str:
    """The brief as Markdown: a title, one heading per section, then sources."""
    brief_lines = [f"# {topic}", ""]
    for section in draft:
        brief_lines += [f"## {section.title}", "", section.text.strip(), ""]

    cited_ids = cited_finding_ids(draft)
    cited_findings = [finding for finding in findings if finding.id in cited_ids]
    if cited_findings:
        brief_lines += ["## Sources", "", *source_lines(cited_findings), ""]
    return "\n".join(brief_lines)
