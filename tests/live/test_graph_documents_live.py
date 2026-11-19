"""Live run of the whole graph with a document: the Symbotic topic and its annual report.

CONCEPT: subgraph-wrapper, the acceptance check (D-57, D-58). Runs only on
request (`uv run pytest -m live`), with search-mcp running (D-32) and the
annual report ingested (tests/live/test_ingestion_live.py); skipped, with
what to do, if either is missing.

The planner sees the file name and routes each sub-question to the web, the
document, or both; web and document branches run side by side. The check
prints the plan, the brief, its sources (document ones cited by page), and
the numbers for "What we measured": requests used against the budget sized
to this plan, how often a citation fell back to every relevant chunk, each
branch's finish time, and the stages.

It asserts what must hold on any plan: at least one sub-question went to the
document, every document finding's passage is an exact slice of a stored
chunk on the page it cites, and the run stayed inside its budget.
"""

import logging
import re
import time
from pathlib import Path
from uuid import uuid4

import pytest

from app.ingestion import file_fingerprint
from graph.document_store import DocumentStore
from graph.llm import MODEL_CALL_BUDGET_KEY, ModelGateway
from graph.mcp_client import START_SERVER_COMMAND, McpToolbox
from graph.nodes.document_researcher import DOCUMENT_ROUTES
from graph.run_budget import planned_request_budget
from graph.supervisor import build_graph, research_run_config
from settings import settings
from tests.live.test_mcp_client_live import server_is_listening

pytestmark = pytest.mark.live

LIVE_TOPIC = "Due diligence on Symbotic: business model, customers, and main risks"
ANNUAL_REPORT_PATH = Path("data/uploads/symbotic-annual-report.pdf")
LOCATOR_PATTERN = re.compile(r"^(?P<document_id>[^#]+)#page=(?P<page>\d+)$")
BRANCH_NODES = ("web_researcher", "document_researcher")


@pytest.fixture
def stored_report_id() -> str:
    if not ANNUAL_REPORT_PATH.is_file():
        pytest.skip(f"put the annual report at {ANNUAL_REPORT_PATH} first (D-45)")
    stored_document = DocumentStore().find_by_fingerprint(file_fingerprint(ANNUAL_REPORT_PATH))
    if stored_document is None:
        pytest.skip("ingest the annual report first: tests/live/test_ingestion_live.py")
    return stored_document.document_id


async def test_the_graph_writes_a_brief_from_the_web_and_the_annual_report(
    stored_report_id, caplog
):
    caplog.set_level(logging.INFO)
    if not server_is_listening(settings.vre_search_mcp_url):
        pytest.skip(
            f"search-mcp is not running at {settings.vre_search_mcp_url}; "
            f"start it with: {START_SERVER_COMMAND}"
        )

    started_at = time.perf_counter()
    finished_at_by_node: dict[str, float] = {}
    branch_finished_at: list[tuple[str, str, float]] = []
    final_state: dict = {}
    run_config = research_run_config(LIVE_TOPIC, uuid4())
    async with McpToolbox.connect(settings.vre_search_mcp_url) as toolbox:
        graph = build_graph(ModelGateway(), toolbox)
        async for stream_mode, chunk in graph.astream(
            {"topic": LIVE_TOPIC, "document_ids": [stored_report_id]},
            config=run_config,
            stream_mode=["updates", "values"],
        ):
            if stream_mode == "values":
                final_state = chunk
                continue
            seconds_so_far = time.perf_counter() - started_at
            for node_name, node_update in chunk.items():
                finished_at_by_node[node_name] = seconds_so_far
                if node_name in BRANCH_NODES and node_update["findings"]:
                    sub_question_id = node_update["findings"][0].sub_question_id
                    branch_finished_at.append((node_name, sub_question_id, seconds_so_far))
    elapsed_seconds = time.perf_counter() - started_at

    sub_questions = final_state["sub_questions"]
    findings = final_state["findings"]
    draft = final_state["draft"]
    log_messages = [record.message for record in caplog.records]
    call_budget = run_config["configurable"][MODEL_CALL_BUDGET_KEY]

    print(f"\n# {LIVE_TOPIC}\n")
    print("## Plan\n")
    for sub_question in sub_questions:
        print(f"{sub_question.id} [{sub_question.route}] {sub_question.text}")
    print()
    for section in draft:
        print(f"## {section.title}\n\n{section.text}\n")
    print("## Sources\n")
    for finding in findings:
        if finding.found:
            print(f"[{finding.id}] {finding.title} - {finding.locator}")

    document_findings = [f for f in findings if f.source_type == "document"]
    document_found = [f for f in document_findings if f.found]
    citation_fallbacks = sum("no IsSUP quote found" in message for message in log_messages)
    document_results = [m for m in log_messages if ": document research " in m]
    retry_messages = [message for message in log_messages if "transient error" in message]

    print("\n--- run numbers ---")
    print(f"sub-questions: {len(sub_questions)}")
    print(f"routes: {[sub_question.route for sub_question in sub_questions]}")
    print(
        f"findings: web {sum(f.found for f in findings if f.source_type == 'web')} kept, "
        f"document {len(document_found)} kept, "
        f"{sum(not f.found for f in findings)} no evidence"
    )
    for message in document_results:
        print(f"  {message}")
    print(f"citations that fell back to every relevant chunk: {citation_fallbacks}")
    print(f"schema retries: {sum('did not fit' in message for message in log_messages)}")
    print(f"transient retries: {len(retry_messages)}")
    for message in retry_messages:
        print(f"  {message[:200]}")
    print(
        f"model requests: {call_budget.requests_sent} of {call_budget.limit} allowed "
        f"(planned {planned_request_budget(sub_questions)}, "
        f"ceiling {settings.vre_max_model_calls_per_run})"
    )
    planner_seconds = finished_at_by_node["planner"]
    print(
        "branches finished (s after the planner): "
        + ", ".join(
            f"{sub_question_id} {node_name.split('_')[0]} {finished - planner_seconds:.1f}"
            for node_name, sub_question_id, finished in branch_finished_at
        )
    )
    research_seconds = finished_at_by_node["research_check"] - planner_seconds
    writer_seconds = elapsed_seconds - finished_at_by_node["research_check"]
    print(
        f"stages: planner {planner_seconds:.1f} s, research {research_seconds:.1f} s, "
        f"writer {writer_seconds:.1f} s"
    )
    print(f"total time: {elapsed_seconds:.1f} s")

    assert any(q.route in DOCUMENT_ROUTES for q in sub_questions), (
        "the planner routed nothing to the document, so this run checks nothing new"
    )
    assert draft and all(section.text.strip() for section in draft)
    assert document_found, "no document finding: the document branches found nothing"
    # Every document citation is an exact slice of a stored chunk on its page.
    chunks_by_page: dict[int, list[str]] = {}
    for stored_chunk in DocumentStore().chunks(stored_report_id):
        chunks_by_page.setdefault(stored_chunk.page, []).append(stored_chunk.text)
    for finding in document_found:
        locator = LOCATOR_PATTERN.match(finding.locator)
        assert locator and locator["document_id"] == stored_report_id, finding.locator
        page_chunks = chunks_by_page.get(int(locator["page"]), [])
        assert any(text.startswith(finding.passage) for text in page_chunks), finding.id
    assert call_budget.requests_sent <= call_budget.limit <= settings.vre_max_model_calls_per_run
