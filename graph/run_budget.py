"""A run's request budget, sized to its plan (D-58).

CONCEPT: subgraph-wrapper, choice C. The budget (D-43) stops a run that has
gone wrong, a loop past its designed cap, before it spends the account. A
fixed 30 fitted a Phase 3 brief; a document sub-question alone can cost 13
requests, so one number would either stop a legitimate document plan or be
too loose to guard a small web plan. After the planner, the run's budget is
set to what its plan can cost at worst:

    chat calls = planner 1
               + 1 per web branch (P-02)
               + SELF_RAG_WORST_CHAT_CALLS per document branch
               + writer 1 (outline) + 1 per sub-question (sections)
    budget     = chat calls x 2 (one schema retry each, rule G-8)
               + SELF_RAG_WORST_EMBEDDINGS per document branch

never above the budget the run started with, which is VRE_MAX_MODEL_CALLS_PER_RUN
(the ceiling). A Phase 3 plan of six web sub-questions gives 28, close to
D-43's 30; the largest plan (six sub-questions, all `both`) gives 172.

Transport retries also spend the budget (D-43) and have no reserve beyond
the doubling; the live run measures how close a real run comes.
"""

import logging

from graph.llm import current_call_budget
from graph.nodes.document_researcher import DOCUMENT_ROUTES
from graph.nodes.researcher import WEB_ROUTES
from graph.self_rag.state import MAX_REVISIONS, MAX_SEARCH_PASSES
from graph.state import SubQuestion

logger = logging.getLogger(__name__)

# Self-RAG's worst path, from its own caps: every pass makes P-15, P-10, P-11,
# P-12, and P-14, and the one revision adds P-13; each pass embeds once.
CHAT_CALLS_PER_SELF_RAG_PASS = 5
SELF_RAG_WORST_CHAT_CALLS = MAX_SEARCH_PASSES * CHAT_CALLS_PER_SELF_RAG_PASS + MAX_REVISIONS
SELF_RAG_WORST_EMBEDDINGS = MAX_SEARCH_PASSES
# Each chat call may need its one schema retry (G-8).
REQUESTS_PER_CHAT_CALL = 2


def planned_request_budget(sub_questions: list[SubQuestion]) -> int:
    """The most requests this plan can need, every designed cap reached."""
    web_branches = sum(1 for question in sub_questions if question.route in WEB_ROUTES)
    document_branches = sum(1 for question in sub_questions if question.route in DOCUMENT_ROUTES)
    chat_calls = (
        1  # the planner
        + web_branches
        + document_branches * SELF_RAG_WORST_CHAT_CALLS
        + 1  # the writer's outline
        + len(sub_questions)  # at most one section per sub-question
    )
    return chat_calls * REQUESTS_PER_CHAT_CALL + document_branches * SELF_RAG_WORST_EMBEDDINGS


def fit_budget_to_plan(sub_questions: list[SubQuestion]) -> int | None:
    """Lower the run's budget to what its plan can need; never raise it.

    Called by the planner once the plan is checked. The budget lives in the
    run's config (D-43), so every branch sees the new limit. Outside a run
    (a node test, a live check of the planner) there is no budget: None.
    """
    budget = current_call_budget()
    if budget is None:
        return None
    planned_budget = planned_request_budget(sub_questions)
    budget.limit = min(budget.limit, planned_budget)
    logger.info(
        "run budget: %d requests for this plan (planned %d; %d already sent)",
        budget.limit,
        planned_budget,
        budget.requests_sent,
    )
    return budget.limit
