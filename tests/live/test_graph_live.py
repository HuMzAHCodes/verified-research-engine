"""Live run of the whole Phase 3 graph: real planner, parallel researchers, writer.

CONCEPT: sequential-graph, tools-through-mcp, and from Phase 3 parallel-fan-out
and shared-throttle (per-tier rates, retries named by tier, the call budget).
Runs only on request (`uv run pytest -m live`), with search-mcp running in its
own terminal (D-32); skipped, with the start command, if it is not. One full
run costs up to 6 Tavily searches and up to 14 model calls. Prints the brief
and the numbers for the "What we measured" section of the concept notes,
including how long each stage took and when each parallel branch finished.
"""

import logging
import time
from uuid import uuid4

import pytest

from graph.llm import MODEL_CALL_BUDGET_KEY, ModelGateway
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

    # CONCEPT: parallel-fan-out. "updates" yields each node's result as soon as
    # it finishes, so a branch's time is when its update arrives; "values"
    # yields the whole state after each step, and the last one is the final state.
    started_at = time.perf_counter()
    finished_at_by_node: dict[str, float] = {}
    branch_finished_at: list[tuple[str, float]] = []
    final_state: dict = {}
    run_config = research_run_config(LIVE_TOPIC, uuid4())
    async with McpToolbox.connect(settings.vre_search_mcp_url) as toolbox:
        graph = build_graph(ModelGateway(), toolbox)
        async for stream_mode, chunk in graph.astream(
            {"topic": LIVE_TOPIC, "document_ids": []},
            config=run_config,
            stream_mode=["updates", "values"],
        ):
            if stream_mode == "values":
                final_state = chunk
                continue
            seconds_so_far = time.perf_counter() - started_at
            for node_name, node_update in chunk.items():
                finished_at_by_node[node_name] = seconds_so_far
                if node_name == "web_researcher" and node_update["findings"]:
                    sub_question_id = node_update["findings"][0].sub_question_id
                    branch_finished_at.append((sub_question_id, seconds_so_far))
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
    retry_messages = [message for message in log_messages if "transient error" in message]
    decision_retries = sum("(decision," in message for message in retry_messages)
    writer_retries = sum("(writer," in message for message in retry_messages)
    print(
        f"transient retries: {len(retry_messages)} "
        f"(decision tier {decision_retries}, writer tier {writer_retries})"
    )
    # Why pages were skipped, by kind (the reasons come from search-mcp).
    skip_reasons: dict[str, int] = {}
    for message in log_messages:
        if ": skipped " in message:
            # "Q2: skipped <url> (<reason>)"; a url has no spaces, so the
            # first " (" opens the reason, even one with brackets inside.
            reason = message.split(" (", 1)[1]
            kind = next(
                (
                    known
                    for known in (
                        "timed out",
                        "HTTP",
                        "not a text page",
                        "no readable text",
                        "could not fetch",
                    )
                    if reason.startswith(known)
                ),
                "other",
            )
            skip_reasons[kind] = skip_reasons.get(kind, 0) + 1
    print(f"pages skipped by reason: {skip_reasons or 'none'}")
    for message in log_messages:
        if ": timing: " in message:
            print(f"  {message}")
    call_budget = run_config["configurable"][MODEL_CALL_BUDGET_KEY]
    print(f"model requests: {call_budget.requests_sent} of {call_budget.limit} allowed")
    print(
        f"rates: decision {settings.vre_decision_rps} a second, "
        f"writer {settings.vre_writer_rps} a second"
    )
    planner_seconds = finished_at_by_node["planner"]
    research_seconds = finished_at_by_node["research_check"] - planner_seconds
    writer_seconds = elapsed_seconds - finished_at_by_node["research_check"]
    print(
        "branches finished (s after the planner): "
        + ", ".join(
            f"{sub_question_id} {finished - planner_seconds:.1f}"
            for sub_question_id, finished in branch_finished_at
        )
    )
    print(
        f"stages: planner {planner_seconds:.1f} s, research {research_seconds:.1f} s, "
        f"writer {writer_seconds:.1f} s"
    )
    print(f"total time: {elapsed_seconds:.1f} s")

    assert len(branch_finished_at) == len(final_state["sub_questions"])  # TC-RES-01
    assert draft, "the run produced no sections"
    assert all(section.text.strip() for section in draft)
    assert found_count >= 1, "no finding survived the checks"
