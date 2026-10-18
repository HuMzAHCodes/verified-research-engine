"""The writer node: outline first, then one checked section per model call.

CONCEPT: sequential-graph. The writer makes 1 + N model calls on the writer
tier: one outline call (P-03), then one call per section (P-04). Each section
call sees only the findings of that section's sub-questions (FR-WRT-03), and
everything the model returns is checked in code before it enters the draft.

Requirements: FR-WRT-01 (one call per section), FR-WRT-03 (only that
section's findings reach the writer).
"""

import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from graph.llm import ModelGateway
from graph.prompts import writer as writer_prompts
from graph.schemas import Outline, OutlineSection, SectionDraft
from graph.state import Finding, Section, SubQuestion

logger = logging.getLogger(__name__)

# A citation marker in the text: one id, [Q2-F1], or several separated by
# commas, [Q2-F1, Q2-F3]. P-04 asks for one id per sentence, but the first live
# run showed the model sometimes lists two; both forms must be checked.
MARKER_PATTERN = re.compile(r"\s?\[([A-Za-z0-9-]+(?:\s*,\s*[A-Za-z0-9-]+)*)\]")


class WriterError(RuntimeError):
    """The writer had no evidence to write from."""


async def writer_node(
    state: Mapping[str, Any],
    *,
    gateway: ModelGateway,
) -> dict[str, list[Section]]:
    """Outline, check the outline, then write each section from its own findings."""
    sub_questions: list[SubQuestion] = state["sub_questions"]
    findings_by_question = group_findings_with_evidence(state.get("findings", []))
    if not findings_by_question:
        raise WriterError("no sub-question found any evidence; nothing to write from")

    preferences = state.get("preferences")
    report_format = getattr(preferences, "report_format", "") if preferences else ""
    outline = await gateway.structured(
        Outline,
        writer_prompts.build_outline_messages(
            topic=state["topic"],
            sub_questions=sub_questions,
            question_ids_with_evidence=set(findings_by_question),
            report_format=report_format,
        ),
        prompt_id=writer_prompts.OUTLINE_PROMPT_ID,
        prompt_version=writer_prompts.OUTLINE_VERSION,
        tier="writer",
    )
    checked_sections = check_outline(outline.sections, sub_questions, findings_by_question)

    draft: list[Section] = []
    for outline_section in checked_sections:
        # CONCEPT: sequential-graph, decision 7. Only this section's findings
        # go into its prompt, so it cannot cite another section's evidence.
        section_findings = [
            finding
            for question_id in outline_section.sub_question_ids
            for finding in findings_by_question[question_id]
        ]
        section_draft = await gateway.structured(
            SectionDraft,
            writer_prompts.build_section_messages(outline_section.title, section_findings),
            prompt_id=writer_prompts.SECTION_PROMPT_ID,
            prompt_version=writer_prompts.SECTION_VERSION,
            tier="writer",
        )
        allowed_finding_ids = {finding.id for finding in section_findings}
        draft.append(
            Section(
                # The outline's title, not the model's: a section cannot be renamed.
                title=outline_section.title,
                text=remove_unknown_markers(section_draft.text, allowed_finding_ids),
            )
        )
    return {"draft": draft}


def group_findings_with_evidence(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Findings that found something, grouped by sub-question id, in order."""
    findings_by_question: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        if finding.found:
            findings_by_question[finding.sub_question_id].append(finding)
    return dict(findings_by_question)


def check_outline(
    outline_sections: list[OutlineSection],
    sub_questions: list[SubQuestion],
    findings_by_question: dict[str, list[Finding]],
) -> list[OutlineSection]:
    """Enforce in code what P-03 only asks for.

    1. Ids that are not sub-questions with evidence are removed from a section.
    2. A section left with no ids is dropped: it has nothing to cite.
    3. A sub-question with evidence that no section covers gets its own
       section at the end, titled with its question.
    """
    checked_sections: list[OutlineSection] = []
    for outline_section in outline_sections:
        kept_ids = [
            question_id
            for question_id in outline_section.sub_question_ids
            if question_id in findings_by_question
        ]
        if len(kept_ids) < len(outline_section.sub_question_ids):
            logger.warning(
                "outline section %r: removed ids %s",
                outline_section.title,
                sorted(set(outline_section.sub_question_ids) - set(kept_ids)),
            )
        if kept_ids:
            checked_sections.append(
                outline_section.model_copy(update={"sub_question_ids": kept_ids})
            )

    covered_ids = {
        question_id for section in checked_sections for question_id in section.sub_question_ids
    }
    for sub_question in sub_questions:
        if sub_question.id in findings_by_question and sub_question.id not in covered_ids:
            logger.warning("outline missed %s; adding a section for it", sub_question.id)
            checked_sections.append(
                OutlineSection(title=sub_question.text, sub_question_ids=[sub_question.id])
            )
    return checked_sections


def remove_unknown_markers(section_text: str, allowed_finding_ids: set[str]) -> str:
    """Keep only ids that name a finding in this section's prompt.

    Every id inside a marker is checked, including each id in a list such as
    [Q2-F1, Q9-F1]. Unknown ids are dropped; a marker left with no ids is
    removed, which leaves its sentence uncited. Phase 1 has no critic, so that
    sentence stays in the draft; phase 5's claim check will catch it.
    """
    removed_ids: list[str] = []
    multi_id_marker_count = 0

    def keep_known_ids(marker_match: re.Match[str]) -> str:
        nonlocal multi_id_marker_count
        cited_ids = [cited_id.strip() for cited_id in marker_match.group(1).split(",")]
        if len(cited_ids) > 1:
            multi_id_marker_count += 1
        known_ids = [cited_id for cited_id in cited_ids if cited_id in allowed_finding_ids]
        removed_ids.extend(
            cited_id for cited_id in cited_ids if cited_id not in allowed_finding_ids
        )
        if not known_ids:
            return ""
        leading_space = " " if marker_match.group(0).startswith((" ", "\n", "\t")) else ""
        return f"{leading_space}[{', '.join(known_ids)}]"

    checked_text = MARKER_PATTERN.sub(keep_known_ids, section_text)
    if removed_ids:
        logger.warning("removed unknown markers: %s", removed_ids)
    if multi_id_marker_count:
        # Allowed, but against P-04's one-id rule; counted for prompt tuning.
        logger.info("markers citing more than one finding: %d", multi_id_marker_count)
    return checked_text
