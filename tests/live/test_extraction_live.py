"""Live check of verbatim extraction: real pages, a real P-02 call, the checks.

CONCEPT: verbatim-extraction. Runs only on request (`uv run pytest -m live`),
with search-mcp running in its own terminal (D-32). For each sub-question it
does what the researcher will do in concept tools-through-mcp: one search, read
up to three pages through McpToolbox, then one P-02 call and the checks. Prints
what the concept notes record: prompt size, time, findings proposed, kept, and
dropped by reason, and each kept finding in full.

For a passage dropped as "passage not in page" it also shows where the copy
went wrong: the longest run of the passage's words that is on the page, and
what the page and the model each say right after it. That tells an invented
passage (no long run on the page) from a near copy (a long run, then one
small change), which need different fixes in the prompt.

Costs two Tavily credits and two model calls on the decision tier.
"""

import logging
import time

import pytest

from graph.extraction import extract_web_findings, normalise_for_matching, occurs_in
from graph.llm import ModelGateway
from graph.mcp_client import START_SERVER_COMMAND, McpToolbox, ToolCallError, WebPage
from graph.prompts.web_findings import build_web_findings_messages
from graph.state import SubQuestion
from settings import settings
from tests.live.test_mcp_client_live import server_is_listening

pytestmark = pytest.mark.live

LIVE_SUB_QUESTIONS = [
    SubQuestion(id="Q1", text="Who are Symbotic's largest customers?", route="web"),
    SubQuestion(id="Q2", text="What are the main risks to Symbotic's business?", route="web"),
]
PAGES_PER_QUESTION = 3
CONTEXT_CHARS = 70


def where_the_copy_breaks(passage: str, pages: list[WebPage]) -> str:
    """Describe the longest run of the passage's words found on any given page."""
    passage_words = normalise_for_matching(passage).split()
    best_start, best_end, best_page_text = 0, 0, ""
    for page in pages:
        page_text = normalise_for_matching(page.text)
        for start in range(len(passage_words)):
            end = start
            # Extend the run one word at a time while it is still on this page.
            while (
                end < len(passage_words) and " ".join(passage_words[start : end + 1]) in page_text
            ):
                end += 1
            if end - start > best_end - best_start:
                best_start, best_end, best_page_text = start, end, page_text
    run_length = best_end - best_start
    if run_length < 3:
        return f"no run of 3 or more of its {len(passage_words)} words is on any page"
    matched_text = " ".join(passage_words[best_start:best_end])
    page_after = best_page_text.split(matched_text, 1)[1][:CONTEXT_CHARS]
    model_after = " ".join(passage_words[best_end:])[:CONTEXT_CHARS]
    return (
        f"words {best_start + 1} to {best_end} of {len(passage_words)} are on a page; "
        f"then the page says {page_after!r} and the model wrote {model_after!r}"
    )


async def read_pages(toolbox: McpToolbox, question: str) -> list[WebPage]:
    """Search once, then read results until three pages are in hand."""
    pages: list[WebPage] = []
    for hit in await toolbox.search(question):
        if len(pages) == PAGES_PER_QUESTION:
            break
        try:
            pages.append(await toolbox.fetch(hit.url))
        except ToolCallError as error:
            print(f"  skipped  {hit.url[:70]}  ({error})")
    return pages


async def test_real_pages_give_findings_that_pass_the_checks(caplog):
    caplog.set_level(logging.INFO)
    server_url = settings.vre_search_mcp_url
    if not server_is_listening(server_url):
        pytest.skip(
            f"search-mcp is not running at {server_url}; start it with: {START_SERVER_COMMAND}"
        )

    gateway = ModelGateway()
    total_kept, total_dropped = 0, 0
    async with McpToolbox.connect(server_url) as toolbox:
        for sub_question in LIVE_SUB_QUESTIONS:
            print(f"\n{sub_question.id}: {sub_question.text}")
            pages = await read_pages(toolbox, sub_question.text)
            prompt_chars = sum(
                len(message.content)
                for message in build_web_findings_messages(sub_question.text, pages)
            )
            print(f"  {len(pages)} pages read; prompt {prompt_chars:,} characters")

            started_at = time.perf_counter()
            result = await extract_web_findings(gateway, sub_question, pages)
            elapsed_seconds = time.perf_counter() - started_at

            kept = [finding for finding in result.findings if finding.found]
            total_kept += len(kept)
            total_dropped += len(result.dropped)
            print(
                f"  P-02 in {elapsed_seconds:.1f} s: proposed {len(kept) + len(result.dropped)}, "
                f"kept {len(kept)}, dropped {len(result.dropped)}"
            )
            for dropped in result.dropped:
                print(f"  dropped ({dropped.reason}): {dropped.passage[:110]!r}")
                if dropped.reason == "passage not in page":
                    print(f"    where it breaks: {where_the_copy_breaks(dropped.passage, pages)}")
            for finding in kept:
                print(f"  kept {finding.id} from {finding.locator[:70]}")
                print(f"    answer:  {finding.answer[:150]}")
                print(f"    passage: {finding.passage[:150]}")
                # The checks ran inside extract_web_findings; confirm them here
                # against the page as fetched, independently of that code path.
                page_text = next(page.text for page in pages if page.url == finding.locator)
                assert occurs_in(finding.passage, page_text)

    schema_retries = sum("did not fit" in record.message for record in caplog.records)
    flag_disagreements = [r.message for r in caplog.records if "the model said found=" in r.message]
    print(
        f"\nTotal: kept {total_kept}, dropped {total_dropped}; schema retries: {schema_retries}; "
        f"found flag disagreements: {flag_disagreements or 'none'}"
    )
    assert total_kept >= 1, "no finding survived the checks on any sub-question"
