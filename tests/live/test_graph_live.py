"""Live run of the whole Phase 2 graph: real planner, search-mcp, P-02, writer.

CONCEPT: sequential-graph, and from Phase 2 tools-through-mcp. Runs only on
request (`uv run pytest -m live`), with search-mcp running in its own terminal
(D-32); skipped, with the start command, if it is not. One full run costs up
to 6 Tavily searches and up to 14 model calls. Prints the brief and the numbers
for the "What we measured" section of the concept notes.
"""

import logging
import time
from uuid import uuid4

import pytest

from graph.llm import ModelGateway
from graph.mcp_client import START_SERVER_COMMAND, McpToolbox
from graph.supervisor import build_graph, research_run_config
from settings import settings
from tests.live.test_mcp_client_live import server_is_listening

pytestmark = pytest.mark.live

LIVE_TOPIC = "Due diligence on Symbotic: business model, customers, and main risks"


async def test_the_graph_writes_a_brief_for_a_real_topic(caplog):
    caplog.set_level(logging.INFO)
    if not server_is_listening(settings.vre_search_mcp_url):
        pytest.skip(
            f"search-mcp is not running at {settings.vre_search_mcp_url}; "
            f"start it with: {START_SERVER_COMMAND}"
        )

    started_at = time.perf_counter()
    async with McpToolbox.connect(settings.vre_search_mcp_url) as toolbox:
        graph = build_graph(ModelGateway(), toolbox)
        final_state = await graph.ainvoke(
            {"topic": LIVE_TOPIC, "document_ids": []},
            config=research_run_config(LIVE_TOPIC, uuid4()),
        )
    elapsed_seconds = time.perf_counter() - started_at

    findings = final_state["findings"]
    draft = final_state["draft"]
    log_messages = [record.message for record in caplog.records]

    print(f"\n# {LIVE_TOPIC}\n")
    for section in draft:
        print(f"## {section.title}\n\n{section.text}\n")
    print("## Sources\n")
    for finding in findings:
        if finding.found:
            print(f"[{finding.id}] {finding.title} - {finding.locator}")

    found_count = sum(finding.found for finding in findings)
    pages_skipped = sum(": skipped " in message for message in log_messages)
    searches_failed = sum(": search failed" in message for message in log_messages)
    dropped_by_reason: dict[str, int] = {}
    for message in log_messages:
        if "dropped a finding (" in message:
            reason = message.split("dropped a finding (", 1)[1].split(")", 1)[0]
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1

    print("\n--- run numbers ---")
    print(f"sub-questions: {len(final_state['sub_questions'])}")
    print(f"pages skipped: {pages_skipped}; searches failed: {searches_failed}")
    print(f"findings: {found_count} kept, {len(findings) - found_count} no evidence")
    print(f"dropped by the checks: {dropped_by_reason or 'none'}")
    print(f"sections: {len(draft)}")
    print(f"schema retries: {sum('did not fit' in message for message in log_messages)}")
    print(f"transient retries: {sum('transient error' in message for message in log_messages)}")
    print(f"total time: {elapsed_seconds:.1f} s")

    assert draft, "the run produced no sections"
    assert all(section.text.strip() for section in draft)
    assert found_count >= 1, "no finding survived the checks"
