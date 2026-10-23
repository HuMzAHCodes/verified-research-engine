"""Turn fetched pages into checked findings for one sub-question.

CONCEPT: verbatim-extraction. The model proposes, code verifies. One P-02 call
reads the sub-question and up to three pages and proposes findings: a url, a
passage, and an answer. Code keeps a finding only if

1. its url is one of the pages given (not one the model made up or mixed up),
2. its passage is at least 5 words (a shorter one proves nothing), and
3. its passage occurs in that page's own text, word for word (D-35).

Kept findings get ids, a cap of three, and a cut to VRE_PASSAGE_MAX_CHARS.
Refused ones come back with their reason, so a failing prompt is visible
instead of looking like a topic with no evidence.

This module is not a node. The researcher (concept tools-through-mcp) and, in
Phase 3, each web_researcher call extract_web_findings after search and fetch.
"""

import logging
import re
from typing import Literal

from pydantic import BaseModel

from graph.llm import ModelGateway
from graph.mcp_client import WebPage
from graph.prompts import web_findings as web_findings_prompt
from graph.schemas import WebFinding, WebFindings
from graph.state import Finding, SubQuestion
from settings import settings

logger = logging.getLogger(__name__)

MIN_PASSAGE_WORDS = 5

# CONCEPT: verbatim-extraction, D-35. Characters that differ from their plain
# form only in style. Pages use the styled forms, models usually type the plain
# ones, and swapping one for the other never changes what a sentence says.
# Letters, digits, and case are never touched.
STYLE_TO_PLAIN = str.maketrans(
    {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote and curly apostrophe
        "‚": "'",  # low single quote
        "‛": "'",  # reversed single quote
        "′": "'",  # prime
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "„": '"',  # low double quote
        "″": '"',  # double prime
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        "…": "...",  # ellipsis character
        " ": " ",  # non-breaking space
        " ": " ",  # narrow non-breaking space
        "​": "",  # zero-width space
    }
)
WHITESPACE_RUN = re.compile(r"\s+")

DropReason = Literal["url not given", "passage not in page", "too short"]


class DroppedFinding(BaseModel):
    """A finding the model proposed and code refused, and why (decision 7)."""

    reason: DropReason
    url: str
    passage: str


class ExtractionResult(BaseModel):
    """What one sub-question's extraction produced."""

    findings: list[Finding]  # checked and numbered; or one found=False finding
    dropped: list[DroppedFinding]


def normalise_for_matching(text: str) -> str:
    """The same form for page and passage: plain quotes and dashes, single spaces."""
    return WHITESPACE_RUN.sub(" ", text.translate(STYLE_TO_PLAIN)).strip()


def occurs_in(passage: str, page_text: str) -> bool:
    """True if the passage is a word-for-word copy of part of the page (D-35)."""
    return normalise_for_matching(passage) in normalise_for_matching(page_text)


def why_refused(proposed: WebFinding, pages_by_url: dict[str, WebPage]) -> DropReason | None:
    """The reason a proposed finding fails the checks, or None if it passes."""
    page = pages_by_url.get(proposed.url.strip())
    if page is None:
        return "url not given"
    if len(normalise_for_matching(proposed.passage).split()) < MIN_PASSAGE_WORDS:
        return "too short"
    # Checked against its own page only: a passage from page 2 credited to
    # page 1 would otherwise pass, and the citation would point to the wrong page.
    if not occurs_in(proposed.passage, page.text):
        return "passage not in page"
    return None


def no_evidence_finding(sub_question: SubQuestion) -> Finding:
    """One found=False finding: the question was researched and nothing held up."""
    return Finding(
        id=f"{sub_question.id}-F1",
        sub_question_id=sub_question.id,
        found=False,
        source_type="web",
    )


async def extract_web_findings(
    gateway: ModelGateway,
    sub_question: SubQuestion,
    pages: list[WebPage],
    *,
    max_passage_chars: int = settings.vre_passage_max_chars,
) -> ExtractionResult:
    """One P-02 call for one sub-question, then the checks."""
    if not pages:
        # No page could be read: there is nothing to extract from, so no call.
        return ExtractionResult(findings=[no_evidence_finding(sub_question)], dropped=[])

    reply = await gateway.structured(
        WebFindings,
        web_findings_prompt.build_web_findings_messages(sub_question.text, pages),
        prompt_id=web_findings_prompt.PROMPT_ID,
        prompt_version=web_findings_prompt.VERSION,
        tier="decision",
    )

    pages_by_url = {page.url: page for page in pages}
    kept_findings: list[Finding] = []
    dropped_findings: list[DroppedFinding] = []
    for proposed in reply.findings:
        if len(kept_findings) == web_findings_prompt.MAX_FINDINGS_PER_QUESTION:
            break  # the cap: the prompt asks for at most three, code enforces it
        refusal = why_refused(proposed, pages_by_url)
        if refusal is not None:
            dropped_findings.append(
                DroppedFinding(reason=refusal, url=proposed.url, passage=proposed.passage)
            )
            logger.info("%s: dropped a finding (%s): %s", sub_question.id, refusal, proposed.url)
            continue
        page = pages_by_url[proposed.url.strip()]
        kept_findings.append(
            Finding(
                id=f"{sub_question.id}-F{len(kept_findings) + 1}",  # ids by code (G-5)
                sub_question_id=sub_question.id,
                found=True,
                source_type="web",
                locator=page.url,
                title=page.title,
                # Cut after the check (G-7): a prefix of a verified copy is still a copy.
                passage=normalise_for_matching(proposed.passage)[:max_passage_chars],
                answer=proposed.answer.strip(),
            )
        )

    # CONCEPT: verbatim-extraction, decision 6. The checked findings decide,
    # not the model's flag; a disagreement is logged so it shows in the trace.
    if reply.found != bool(kept_findings):
        logger.info(
            "%s: the model said found=%s, the checks kept %d",
            sub_question.id,
            reply.found,
            len(kept_findings),
        )
    return ExtractionResult(
        findings=kept_findings or [no_evidence_finding(sub_question)],
        dropped=dropped_findings,
    )
