"""Live run of the whole phase 1 graph: real planner, real search, real writer.

CONCEPT: sequential-graph. Runs only on request (`uv run pytest -m live`).
One full run costs up to 8 model calls and up to 6 Tavily searches. Prints the
brief and the numbers for the "What we measured" section of the concept notes.
"""

import logging
import time

import pytest

from graph.llm import ModelGateway
from graph.nodes.researcher import TavilySearchClient
from graph.supervisor import build_graph

pytestmark = pytest.mark.live

LIVE_TOPIC = "Due diligence on Symbotic: business model, customers, and main risks"


async def test_the_graph_writes_a_brief_for_a_real_topic(caplog):
    caplog.set_level(logging.INFO)
    graph = build_graph(ModelGateway(), TavilySearchClient())
    started_at = time.perf_counter()

    final_state = await graph.ainvoke({"topic": LIVE_TOPIC, "document_ids": []})

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
    no_evidence_count = len(findings) - found_count
    added_sections = sum("outline missed" in message for message in log_messages)
    marker_removals = [message for message in log_messages if "removed unknown markers" in message]
    schema_retries = sum("did not fit" in message for message in log_messages)
    transient_retries = sum("transient error" in message for message in log_messages)

    print("\n--- run numbers ---")
    print(f"sub-questions: {len(final_state['sub_questions'])}")
    print(f"findings: {found_count} found, {no_evidence_count} no evidence")
    print(f"sections: {len(draft)} ({added_sections} added for missed sub-questions)")
    print(f"marker removals: {marker_removals or 'none'}")
    print(f"schema retries: {schema_retries}; transient retries: {transient_retries}")
    print(f"total time: {elapsed_seconds:.1f} s")

    assert draft, "the run produced no sections"
    assert all(section.text.strip() for section in draft)
