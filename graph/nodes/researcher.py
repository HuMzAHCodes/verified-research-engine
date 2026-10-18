"""The temporary researcher: one web search per sub-question, no model call.

CONCEPT: sequential-graph. TEMPORARY (D-18): this node calls Tavily directly,
which breaks FR-MCP-04 ("the graph reaches tools only through MCP") for phase 1
only. Phase 2 replaces TavilySearchClient with the MCP toolbox, and phase 3
replaces this node with one web researcher per sub-question that uses P-02 to
pick the passage that answers the question. Until then a finding's passage is
the snippet Tavily returns, so some findings are only loosely relevant.

The search client is injected by build_graph, so tests pass a fake one and
no test needs a Tavily key or the network.
"""

import logging
from collections.abc import Mapping
from typing import Any, Protocol

from tavily import AsyncTavilyClient

from graph.state import Finding, SubQuestion
from settings import settings

logger = logging.getLogger(__name__)

RESULTS_PER_SUB_QUESTION = 3


class WebSearchClient(Protocol):
    """The shape any search client must have: Tavily now, MCP in phase 2.

    Each result is a dict with at least "url", "title", and "content".
    """

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]: ...


class TavilySearchClient:
    """WebSearchClient backed by Tavily's API. Removed in phase 2."""

    def __init__(self) -> None:
        self.tavily_client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        search_response = await self.tavily_client.search(
            query, max_results=max_results, search_depth="basic"
        )
        return search_response.get("results", [])


async def researcher_node(
    state: Mapping[str, Any],
    *,
    search_client: WebSearchClient,
) -> dict[str, list[Finding]]:
    """Search once per sub-question and turn the results into findings."""
    new_findings: list[Finding] = []
    for sub_question in state["sub_questions"]:
        search_results = await search_client.search(
            sub_question.text, max_results=RESULTS_PER_SUB_QUESTION
        )
        new_findings.extend(findings_from_results(sub_question, search_results))
    # Only the new findings: the reducer on RunState.findings appends them.
    return {"findings": new_findings}


def findings_from_results(
    sub_question: SubQuestion, search_results: list[dict[str, Any]]
) -> list[Finding]:
    """Number the results Q2-F1, Q2-F2, ... and cut each passage (rule G-7)."""
    usable_results = [result for result in search_results if result.get("content", "").strip()]
    if not usable_results:
        logger.info("%s: no search results", sub_question.id)
        # One "no evidence" finding, so the writer knows this question was
        # asked and nothing came back, rather than that it was never asked.
        return [
            Finding(
                id=f"{sub_question.id}-F1",
                sub_question_id=sub_question.id,
                found=False,
                source_type="web",
            )
        ]
    return [
        Finding(
            id=f"{sub_question.id}-F{position}",
            sub_question_id=sub_question.id,
            found=True,
            source_type="web",
            locator=result.get("url", ""),
            title=result.get("title", ""),
            passage=result["content"].strip()[: settings.vre_passage_max_chars],
        )
        for position, result in enumerate(usable_results[:RESULTS_PER_SUB_QUESTION], start=1)
    ]
