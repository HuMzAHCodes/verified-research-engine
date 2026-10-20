"""The brief as Markdown: sections, then the sources the brief actually cites.

CONCEPT: tracing-from-day-one. A run's record is its trace plus the brief it
produced. render_brief() is the one place that turns run state into a
readable brief: the phase 1 notebook uses it to write phase_01_brief.md, and
phase 6's finalize node uses it to build final_report.

Only cited findings are listed as sources, in id order, so the list says
which evidence the brief rests on, not everything the researcher fetched.
"""

from graph.nodes.writer import MARKER_PATTERN
from graph.state import Finding, Section


def cited_finding_ids(draft: list[Section]) -> set[str]:
    """Every finding id that appears in a marker anywhere in the draft."""
    cited_ids: set[str] = set()
    for section in draft:
        for marker_match in MARKER_PATTERN.finditer(section.text):
            cited_ids.update(cited_id.strip() for cited_id in marker_match.group(1).split(","))
    return cited_ids


def render_brief(topic: str, draft: list[Section], findings: list[Finding]) -> str:
    """The brief as Markdown: a title, one heading per section, then sources."""
    brief_lines = [f"# {topic}", ""]
    for section in draft:
        brief_lines += [f"## {section.title}", "", section.text.strip(), ""]

    cited_ids = cited_finding_ids(draft)
    cited_findings = [finding for finding in findings if finding.id in cited_ids]
    if cited_findings:
        brief_lines += ["## Sources", ""]
        brief_lines += [
            f"- **[{finding.id}]** {finding.title or 'Untitled'}. <{finding.locator}>"
            for finding in cited_findings
        ]
        brief_lines.append("")
    return "\n".join(brief_lines)
