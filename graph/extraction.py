"""Turn fetched pages into checked findings for one sub-question.

CONCEPT: verbatim-extraction. The model proposes, code verifies. One P-02 call
reads the sub-question and up to three pages and proposes findings: a url, a
passage, and an answer. Code keeps a finding only if

1. its url is one of the pages given (not one the model made up or mixed up),
2. its passage is at least 5 words (a shorter one proves nothing), and
3. its passage occurs in that page's own text, word for word (D-35), starting
   and ending on whole words, so a cut-off word or figure is not a copy (D-61).

Kept findings get ids, a cap of three, and a cut to VRE_PASSAGE_MAX_CHARS at a
word boundary. The passage kept is the page's own text, exactly as the page
has it, not the model's typing of it (D-62).
Refused ones come back with their reason, so a failing prompt is visible
instead of looking like a topic with no evidence.

This module is not a node. The researcher (concept tools-through-mcp) and, in
Phase 3, each web_researcher call extract_web_findings after search and fetch.
"""

import logging
import unicodedata
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

# CONCEPT: verbatim-extraction, D-35 and D-61. Characters that differ from
# their plain form only in style. Pages use the styled forms, models usually
# type the plain ones, and swapping one for the other never changes what a
# sentence says. Letters, digits, and case are never changed; the ligatures
# below are the same letters printed joined, as PDFs often store them. Full
# compatibility folding (NFKC) is not used: it also turns "10²" into "102".
STYLE_TO_PLAIN = str.maketrans(
    {
        "ﬀ": "ff",  # PDF ligatures
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "\u00ad": "",  # soft hyphen: a place a word may break, not a character
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


def normalised_with_positions(text: str) -> tuple[str, list[int], list[int]]:
    """The matching form of the text, and where each of its characters came from.

    Composed and decomposed accents become one form (NFC), styled characters
    become plain, runs of whitespace become one space, and the ends are
    trimmed. For each character of the result, starts[i] and ends[i] give the
    span of the original text it came from, so a match found in the matching
    form can be cut from the original text exactly.
    """
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    position = 0
    while position < len(text):
        # A base character and the accents that follow it are composed together.
        cluster_end = position + 1
        while cluster_end < len(text) and unicodedata.combining(text[cluster_end]):
            cluster_end += 1
        cluster = unicodedata.normalize("NFC", text[position:cluster_end])
        for plain_character in cluster.translate(STYLE_TO_PLAIN):
            if plain_character.isspace():
                if not characters or characters[-1] == " ":
                    continue  # a leading space, or one more in a run
                plain_character = " "
            characters.append(plain_character)
            starts.append(position)
            ends.append(cluster_end)
        position = cluster_end
    if characters and characters[-1] == " ":
        characters.pop()
        starts.pop()
        ends.pop()
    return "".join(characters), starts, ends


def normalise_for_matching(text: str) -> str:
    """The same form for page and passage: plain quotes and dashes, single spaces."""
    return normalised_with_positions(text)[0]


def continues_the_word(text: str, inside: int, outside: int) -> bool:
    """True if text[outside] carries on the word or figure ending at text[inside].

    "unprofitable" carries on "profitable"; "1.25" and "1,250" carry on "1".
    """
    if not 0 <= outside < len(text):
        return False
    if text[inside].isalnum() and text[outside].isalnum():
        return True
    # A decimal point or thousands separator between digits keeps a figure going.
    step = outside - inside
    after_separator = outside + step
    return (
        text[inside].isdigit()
        and text[outside] in ".,"
        and 0 <= after_separator < len(text)
        and text[after_separator].isdigit()
    )


def find_copy(passage: str, page_text: str) -> tuple[int, int] | None:
    """Where the passage is copied word for word in the page, as a span of the page.

    The copy must start and end on whole words and whole figures (D-61), and
    the span is of the page's own text, styled characters and all (D-62).
    """
    plain_passage = normalise_for_matching(passage)
    if not plain_passage:
        return None
    plain_page, starts, ends = normalised_with_positions(page_text)
    match_start = plain_page.find(plain_passage)
    while match_start != -1:
        match_end = match_start + len(plain_passage)  # exclusive
        cut_at_start = continues_the_word(plain_page, match_start, match_start - 1)
        cut_at_end = continues_the_word(plain_page, match_end - 1, match_end)
        if not cut_at_start and not cut_at_end:
            return starts[match_start], ends[match_end - 1]
        match_start = plain_page.find(plain_passage, match_start + 1)
    return None


def occurs_in(passage: str, page_text: str) -> bool:
    """True if the passage is a word-for-word copy of whole words of the page (D-35, D-61)."""
    return find_copy(passage, page_text) is not None


def cut_at_word(text: str, max_chars: int) -> str:
    """At most max_chars of the text, ending on a whole word where one fits."""
    if len(text) <= max_chars:
        return text
    cut_text = text[:max_chars]
    last_space = max(cut_text.rfind(" "), cut_text.rfind("\n"))
    return (cut_text[:last_space] if last_space > 0 else cut_text).rstrip()


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
        copy_start, copy_end = find_copy(proposed.passage, page.text)
        kept_findings.append(
            Finding(
                id=f"{sub_question.id}-F{len(kept_findings) + 1}",  # ids by code (G-5)
                sub_question_id=sub_question.id,
                found=True,
                source_type="web",
                locator=page.url,
                title=page.title,
                # The page's own text (D-62), cut after the check at a word (G-7):
                # a prefix of a verified copy, ending on a whole word, is still a copy.
                passage=cut_at_word(page.text[copy_start:copy_end], max_passage_chars),
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
