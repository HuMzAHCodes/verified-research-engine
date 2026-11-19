"""The Self-RAG subgraph's state, and the result it hands back.

CONCEPT: self-rag. The subgraph answers one sub-question from the run's
documents. Its state lives only while it runs; what leaves it is a
SelfRagResult, which concept 4 (subgraph-wrapper) turns into findings.

A pass is one round of search, grading, answering, and checking. There are
at most two: the first, and one more with a second search passage (decision
7, D-53, D-56). The write_passage node clears the pass fields when a second
pass starts; search_passages and revisions_used count across both passes,
because the caps are per sub-question, not per pass.
"""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from graph.document_store import RetrievedChunk

SupportLevel = Literal["fully_supported", "partially_supported", "no_support"]
Usefulness = Literal["useful", "not_useful"]
# answered: the draft passed IsSUP and IsUSE. answered_with_quotes: it was
# revised into checked quotes (decision 8), which then passed IsUSE.
# not_found: no relevant chunk, or no answer that passed its checks.
SelfRagStatus = Literal["answered", "answered_with_quotes", "not_found"]

# A sub-question is searched in at most two passes, each with its own search
# passage beside the question (decision 7, D-53, D-56).
MAX_SEARCH_PASSES = 2
# At most one revision per sub-question, across both passes (decision 7).
MAX_REVISIONS = 1


class SelfRagState(TypedDict, total=False):
    """What the subgraph keeps between its nodes."""

    # Input, set once.
    question: str
    document_ids: list[str]

    # Across passes.
    # Every search passage, in order (D-56). Only ever searched with, beside
    # the question; never answered from, quoted, or shown as evidence.
    search_passages: list[str]
    revisions_used: int

    # This pass; cleared by the write_passage node before a second pass.
    retrieved: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    answer: str
    support: SupportLevel | None
    usefulness: Usefulness | None
    revised: bool

    # Output, set by the last node.
    status: SelfRagStatus


class SelfRagResult(BaseModel):
    """What the subgraph found for one sub-question."""

    status: SelfRagStatus
    answer: str = ""
    # The chunks the answer rests on: the relevant chunks, or, after a
    # revision, only the chunks the kept quotes were copied from. Each is an
    # exact slice of its page, with the page number (decision 4).
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    # The search passages used, for logs and reports; never evidence.
    search_passages: list[str] = Field(default_factory=list)
    revised: bool = False
