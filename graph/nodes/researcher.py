"""The researcher: checked web evidence for each sub-question, through MCP only.

CONCEPT: tools-through-mcp. The graph reaches the web only through the toolbox
(D-37). For each sub-question the researcher:

1. searches through the toolbox (search-mcp, never a search library),
2. reads every result at the same time and keeps the first three readable
   pages, in search-rank order (D-38),
3. hands the pages to extract_web_findings: one P-02 call, then the checks in
   code that keep only passages copied word for word (verbatim-extraction).

Failures are split by kind (mcp-client, decision 6): a page or a search that
fails is skipped and logged, and the sub-question can still end with a "no
evidence" finding; SearchServerUnavailable is not caught, so a dead server
stops the run instead of producing a brief with no evidence. The same rule
covers a dead network (D-40): if every sub-question's search failed, the run
stops with WebResearchFailed rather than writing a brief from nothing.

The toolbox and the gateway are injected by build_graph (D-25), so tests pass
FakeToolbox and ScriptedModel. Phase 3 replaces this node with one
web_researcher per sub-question, running in parallel; research_sub_question
is the part that carries over unchanged.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from graph.extraction import extract_web_findings
from graph.llm import ModelGateway
from graph.mcp_client import SearchHit, Toolbox, ToolCallError, WebPage
from graph.state import Finding, SubQuestion

logger = logging.getLogger(__name__)

SEARCH_RESULTS_PER_QUESTION = 5
PAGES_PER_QUESTION = 3


class WebResearchFailed(RuntimeError):
    """Every web search failed: the network or the search provider is down."""


@dataclass(frozen=True)
class SubQuestionResearch:
    """What researching one sub-question produced, and whether its search failed."""

    findings: list[Finding]
    search_error: str | None = None


async def researcher_node(
    state: Mapping[str, Any],
    *,
    gateway: ModelGateway,
    toolbox: Toolbox,
) -> dict[str, list[Finding]]:
    """Research every sub-question in turn; Phase 3 runs them in parallel."""
    sub_questions: list[SubQuestion] = state["sub_questions"]
    new_findings: list[Finding] = []
    failed_searches: list[str] = []
    for sub_question in sub_questions:
        research = await research_sub_question(sub_question, gateway, toolbox)
        new_findings.extend(research.findings)
        if research.search_error is not None:
            failed_searches.append(f"{sub_question.id}: {research.search_error}")

    # CONCEPT: tools-through-mcp, D-40. One failed search costs one
    # sub-question. Every search failing means the web cannot be reached at
    # all, and a brief written from no evidence must not look like a result.
    if sub_questions and len(failed_searches) == len(sub_questions):
        raise WebResearchFailed(
            f"every web search failed ({len(failed_searches)} of {len(sub_questions)}); "
            f"check the network and the search provider. Last error: {failed_searches[-1]}"
        )
    # Only the new findings: the reducer on RunState.findings appends them.
    return {"findings": new_findings}


async def research_sub_question(
    sub_question: SubQuestion, gateway: ModelGateway, toolbox: Toolbox
) -> SubQuestionResearch:
    """Search, read, extract: the checked findings for one sub-question."""
    search_error: str | None = None
    try:
        hits = await toolbox.search(sub_question.text, max_results=SEARCH_RESULTS_PER_QUESTION)
    except ToolCallError as error:
        # One failed search costs this sub-question, not the brief.
        logger.info("%s: search failed (%s)", sub_question.id, error)
        search_error = str(error)
        hits = []
    pages = await read_pages(sub_question, hits, toolbox)
    # With no pages, extraction returns one found=False finding and makes no call.
    extraction = await extract_web_findings(gateway, sub_question, pages)
    return SubQuestionResearch(findings=extraction.findings, search_error=search_error)


async def read_pages(
    sub_question: SubQuestion, hits: list[SearchHit], toolbox: Toolbox
) -> list[WebPage]:
    """Read every hit at once and keep the first readable ones, in rank order (D-38).

    CONCEPT: tools-through-mcp, open choice. The reads run at the same time on
    the one MCP connection, so this takes about as long as the slowest page.
    Keeping search-rank order, not the order the reads finish, gives the same
    pages a one-at-a-time loop would, whatever the network timing.
    """
    read_outcomes = await asyncio.gather(
        *(toolbox.fetch(hit.url) for hit in hits), return_exceptions=True
    )
    readable_pages: list[WebPage] = []
    for hit, outcome in zip(hits, read_outcomes, strict=True):
        if isinstance(outcome, ToolCallError):
            logger.info("%s: skipped %s (%s)", sub_question.id, hit.url, outcome)
            continue
        if isinstance(outcome, BaseException):
            # SearchServerUnavailable, or a bug: never swallowed.
            raise outcome
        readable_pages.append(outcome)
    return readable_pages[:PAGES_PER_QUESTION]
