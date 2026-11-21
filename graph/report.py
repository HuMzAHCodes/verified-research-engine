"""The brief as Markdown: sections, then the sources the brief actually cites.

CONCEPT: tracing-from-day-one. A run's record is its trace plus the brief it
produced. render_brief() is the one place that turns run state into a
readable brief: the phase notebooks use it to write their brief files, and
phase 6's finalize node uses it to build final_report.

Only cited findings are listed as sources, so the list says which evidence the
brief rests on, not everything the researcher fetched. From Phase 2 several
findings often come from one page, so the list has one line per page, naming
every cited finding taken from it; each [Q2-F1] in the text still points to
exactly one line. Pages are listed in the order the brief first cites them.

A page's title and address come from the page, so they cannot add Markdown of
their own: the title's Markdown characters are escaped, and the address cannot
end its <...> early (outside test review at the Phase 4 close, F-19 and F-24).
"""

from graph.nodes.writer import MARKER_PATTERN
from graph.state import Finding, Section

SENTENCE_END = (".", "?", "!")
# Characters that would start Markdown inside a title: links, emphasis, code,
# raw HTML, tables. Backslash first, so the escapes themselves are not escaped.
MARKDOWN_CHARACTERS = "\\`*_[]<>|~"
# Characters that would end an autolink (<...>) early or break it.
AUTOLINK_BREAKERS = {"<": "%3C", ">": "%3E", " ": "%20"}


def cited_ids_in_order(draft: list[Section]) -> list[str]:
    """Every finding id cited in the draft, once each, in the order first cited."""
    cited_ids: dict[str, None] = {}
    for section in draft:
        for marker_match in MARKER_PATTERN.finditer(section.text):
            for cited_id in marker_match.group(1).split(","):
                cited_ids.setdefault(cited_id.strip())
    return list(cited_ids)


def cited_finding_ids(draft: list[Section]) -> set[str]:
    """Every finding id that appears in a marker anywhere in the draft."""
    return set(cited_ids_in_order(draft))


def escape_markdown(text: str) -> str:
    """The text with every Markdown character escaped, so it shows as written."""
    return "".join(
        f"\\{character}" if character in MARKDOWN_CHARACTERS else character for character in text
    )


def autolink_target(locator: str) -> str:
    """The locator with the characters that would break <...> percent-encoded."""
    return "".join(AUTOLINK_BREAKERS.get(character, character) for character in locator)


def source_title(finding: Finding) -> str:
    """The page title as a sentence: a full stop only if it has no ending of its own."""
    title = finding.title.strip() or "Untitled"
    title = title if title.endswith(SENTENCE_END) else f"{title}."
    return escape_markdown(" ".join(title.split()))


def source_lines(cited_findings: list[Finding]) -> list[str]:
    """One line per page, in the order pages are first cited, with all its finding ids."""
    findings_by_page: dict[str, list[Finding]] = {}
    for finding in cited_findings:
        findings_by_page.setdefault(finding.locator, []).append(finding)
    return [
        f"- **{' '.join(f'[{finding.id}]' for finding in page_findings)}** "
        f"{source_title(page_findings[0])} <{autolink_target(locator)}>"
        for locator, page_findings in findings_by_page.items()
    ]


def render_brief(topic: str, draft: list[Section], findings: list[Finding]) -> str:
    """The brief as Markdown: a title, one heading per section, then sources."""
    brief_lines = [f"# {topic}", ""]
    for section in draft:
        brief_lines += [f"## {section.title}", "", section.text.strip(), ""]

    findings_by_id = {finding.id: finding for finding in findings}
    cited_findings = [
        findings_by_id[cited_id]
        for cited_id in cited_ids_in_order(draft)
        if cited_id in findings_by_id
    ]
    if cited_findings:
        brief_lines += ["## Sources", "", *source_lines(cited_findings), ""]
    return "\n".join(brief_lines)
