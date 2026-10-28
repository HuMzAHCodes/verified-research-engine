"""The web researcher: checked web evidence for one sub-question, through MCP only.

CONCEPT: parallel-fan-out. From Phase 3 there is one web_researcher branch per
web sub-question, all running at the same time (D-42). build_graph starts them
with one Send each; a branch receives a WebResearchTask, one sub-question and
nothing else, and returns only that sub-question's findings. The reducers on
RunState append the branches' results in the order of the Send list.

CONCEPT: tools-through-mcp. The graph reaches the web only through the toolbox
(D-37). For its sub-question a branch:

1. searches through the toolbox (search-mcp, never a search library),
2. reads every result at the same time and keeps the first three readable
   pages, in search-rank order (D-38),
3. hands the pages to extract_web_findings: one P-02 call, then the checks in
   code that keep only passages copied word for word (verbatim-extraction).

Failures are split by kind (mcp-client, decision 6): a page or a search that
fails is skipped and logged, and the sub-question can still end with a "no
evidence" finding; SearchServerUnavailable is not caught, so a dead server
stops the run, and LangGraph cancels the other branches at once. The D-40
rule ("every search failed means the web cannot be reached") needs every
branch's result, so it lives in research_check_node, which runs once after
all branches (parallel-fan-out, decision 3).

The toolbox and the gateway are injected by build_graph (D-25), so tests pass
FakeToolbox and ScriptedModel.
"""

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from graph.extraction import extract_web_findings
from graph.llm import ModelGateway
from graph.mcp_client import SearchHit, Toolbox, ToolCallError, WebPage
from graph.state import Finding, SubQuestion, WebResearchTask

logger = logging.getLogger(__name__)

SEARCH_RESULTS_PER_QUESTION = 5
PAGES_PER_QUESTION = 3

# Routes that send a sub-question to the web. "both" also goes to the document
# researcher from Phase 4; "document" alone does not search the web.
WEB_ROUTES = frozenset({"web", "both"})


class WebResearchFailed(RuntimeError):
    """Every web search failed: the network or the search provider is down."""


@dataclass(frozen=True)
class SubQuestionResearch:
    """What researching one sub-question produced, and whether its search failed."""

    findings: list[Finding]
    search_error: str | None = None


async def web_researcher_node(
    task: WebResearchTask,
    *,
    gateway: ModelGateway,
    toolbox: Toolbox,
) -> dict[str, list[Any]]:
    """Research one sub-question: one parallel branch.

    CONCEPT: parallel-fan-out, decision 2. The branch reuses the Phase 2 work
    unchanged (research_sub_question). It returns only its own findings and,
    if its search failed, one line saying why; the reducers append both.
    """
    sub_question = task["sub_question"]
    research = await research_sub_question(sub_question, gateway, toolbox)
    search_failures = (
        [f"{sub_question.id}: {research.search_error}"] if research.search_error is not None else []
    )
    return {"findings": research.findings, "search_failures": search_failures}


def research_check_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Stop the run if every web search failed; otherwise change nothing.

    CONCEPT: parallel-fan-out, decision 3, and D-40. One failed search costs
    one sub-question. Every search failing means the web cannot be reached at
    all, and a brief written from no evidence must not look like a result.
    The node runs once, after every branch, so it is the first place that sees
    all the searches. It makes no model call and no tool call.
    """
    web_question_count = sum(
        1 for sub_question in state.get("sub_questions", []) if sub_question.route in WEB_ROUTES
    )
    search_failures: list[str] = state.get("search_failures", [])
    # With no web sub-question there was no search to fail (decision 7).
    if web_question_count and len(search_failures) == web_question_count:
        raise WebResearchFailed(
            f"every web search failed ({len(search_failures)} of {web_question_count}); "
            f"check the network and the search provider. Last error: {search_failures[-1]}"
        )
    return {}


async def research_sub_question(
    sub_question: SubQuestion, gateway: ModelGateway, toolbox: Toolbox
) -> SubQuestionResearch:
    """Search, read, extract: the checked findings for one sub-question.

    CONCEPT: parallel-fan-out, measured. A branch takes as long as its search,
    plus its slowest page read, plus its P-02 call; one log line gives the three
    times, so a slow branch shows which part was slow.
    """
    search_error: str | None = None
    started_at = time.perf_counter()
    try:
        hits = await toolbox.search(sub_question.text, max_results=SEARCH_RESULTS_PER_QUESTION)
    except ToolCallError as error:
        # One failed search costs this sub-question, not the brief.
        logger.info("%s: search failed (%s)", sub_question.id, error)
        search_error = str(error)
        hits = []
    searched_at = time.perf_counter()
    pages = await read_pages(sub_question, hits, toolbox)
    read_at = time.perf_counter()
    # With no pages, extraction returns one found=False finding and makes no call.
    extraction = await extract_web_findings(gateway, sub_question, pages)
    logger.info(
        "%s: timing: search %.1f s, page reads %.1f s (%d of %d kept), extraction %.1f s",
        sub_question.id,
        searched_at - started_at,
        read_at - searched_at,
        len(pages),
        len(hits),
        time.perf_counter() - read_at,
    )
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
