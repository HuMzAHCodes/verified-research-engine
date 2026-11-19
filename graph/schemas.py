"""Output schemas: the shape each model call must return.

CONCEPT: structured-output-planning. A schema is the contract for one prompt.
The gateway binds it with with_structured_output(), so the model's reply comes
back as a validated object or as an error, never as free text to parse.

Everything in a schema is sent to the model as part of a tool definition:
- the class docstring becomes the tool's description,
- each Field description tells the model what to put in that field,
- Literal types close a choice to a fixed set of values,
- field order is the order the model writes them in.
So docstrings and descriptions here are written for the model, and notes for
developers go in # comments, which are not sent.

Schemas are replies, not state: the node checks a reply and converts it into
the state models in graph/state.py. Ids are never part of a schema; code
assigns them (PROMPTS.md, rule G-5).
"""

from typing import Literal

from pydantic import BaseModel, Field

from graph.state import Route

# ---------------------------------------------------------------------------
# P-01 Plan sub-questions (planner node, decision tier)
# ---------------------------------------------------------------------------


class PlannedQuestion(BaseModel):
    """One sub-question of the research plan."""

    # text comes before route, so the model writes the question first and
    # chooses where to look after it has decided what to ask.
    text: str = Field(description="One specific question that sources can answer.")
    route: Route = Field(
        description=(
            "document: the uploaded files. web: public sources. both: check one against the other."
        )
    )


class PlannerOutput(BaseModel):
    """The sub-questions that together cover the research topic."""

    sub_questions: list[PlannedQuestion]


# ---------------------------------------------------------------------------
# P-02 Extract web findings (researcher node, decision tier)
# ---------------------------------------------------------------------------


class WebFinding(BaseModel):
    """One piece of evidence from one page: where it is, the text, and what it answers."""

    # CONCEPT: verbatim-extraction. Field order is the order the model writes:
    # first which page, then the passage copied from it, and only then the
    # answer, so the answer is written from the passage, not from memory. The
    # design listed answer first; the fields and their meaning are unchanged.
    url: str = Field(description="The url of the page the passage comes from, exactly as given.")
    passage: str = Field(
        description=(
            "One to three sentences that sit next to each other on that page, copied "
            "character for character, including any mistakes. Never join text from "
            "two places."
        )
    )
    answer: str = Field(
        description="One or two sentences that answer the question from the passage."
    )


class WebFindings(BaseModel):
    """The evidence the pages give for one question."""

    # The node does not trust this flag: the checked findings decide
    # (verbatim-extraction, decision 6). It lets the model say "nothing here"
    # without inventing a finding.
    found: bool = Field(description="False if none of the pages answers the question.")
    findings: list[WebFinding] = Field(
        default_factory=list, description="At most three findings; none if found is false."
    )


# ---------------------------------------------------------------------------
# P-03 Outline the brief (writer node, writer tier)
# ---------------------------------------------------------------------------


class OutlineSection(BaseModel):
    """One section of the brief and the sub-questions it answers."""

    title: str = Field(description="A short section title.")
    sub_question_ids: list[str] = Field(
        description="Ids of the sub-questions this section answers, for example Q2."
    )


class Outline(BaseModel):
    """The sections of the brief, in reading order."""

    sections: list[OutlineSection]


# ---------------------------------------------------------------------------
# P-04 Write a section (writer node, writer tier)
# ---------------------------------------------------------------------------


class SectionDraft(BaseModel):
    """One written section of the brief."""

    # The writer node replaces this title with the outline's title, so the
    # model cannot rename a section; it is kept so the reply is complete.
    title: str = Field(description="The section title you were given.")
    text: str = Field(
        description=(
            "The section text. Every sentence that states a fact ends with the id "
            "of its finding in square brackets, for example [Q2-F1]."
        )
    )


# ---------------------------------------------------------------------------
# P-10 to P-15: the Self-RAG subgraph (decision tier, D-51)
#
# CONCEPT: self-rag. One schema per call. As above, docstrings and Field
# descriptions are sent to the model; developer notes are # comments. Chunk
# labels (C1 to C6) are given in the prompt and are the only ids the model
# writes; the code maps them back to chunks and drops any it did not give.
# ---------------------------------------------------------------------------


# D-56: a passage shaped like the answer (HyDE), not a list of search words.
# It is only searched with; it is never shown, answered from, or quoted.
class SearchPassage(BaseModel):
    """A short passage written the way the documents would state the answer."""

    passage: str = Field(
        description=(
            "One or two sentences in the documents' own voice (we, our), stating "
            "the answer the way the documents would. Unknown figures as "
            "placeholders such as X% or $X million."
        )
    )


class RelevantChunks(BaseModel):
    """Which of the given chunks are about the question's topic."""

    relevant_labels: list[str] = Field(
        default_factory=list,
        description=(
            "The labels, for example C2, of every chunk about the same company, "
            "subject, or topic as the question. Empty if none is."
        ),
    )


class DraftAnswer(BaseModel):
    """An answer to the question, written only from the given chunks."""

    answer: str = Field(
        description="One to four sentences answering the question from the chunks only."
    )


class SupportCheck(BaseModel):
    """Whether every claim in the answer is stated in the chunks."""

    # Evidence comes before the verdict, so the model looks for the supporting
    # text first and judges from what it found, as WebFinding puts the passage
    # before the answer.
    evidence: list[str] = Field(
        default_factory=list,
        description="Up to three short quotes from the chunks that support the answer.",
    )
    support: Literal["fully_supported", "partially_supported", "no_support"] = Field(
        description=(
            "fully_supported: every claim and every descriptive word is in the chunks. "
            "partially_supported: the facts are there but the answer adds a word or "
            "an interpretation that is not. no_support: the main claims are not there."
        )
    )


class QuotedAnswer(BaseModel):
    """The answer rewritten as direct quotes from the chunks, with nothing added."""

    quotes: list[str] = Field(
        default_factory=list,
        description=(
            "Sentences copied character for character from the chunks that answer "
            "the question. Empty if no sentence does."
        ),
    )


class UsefulnessCheck(BaseModel):
    """Whether the answer gives what the question asks for."""

    # The reason comes first, so the verdict follows from it.
    reason: str = Field(description="One short line: what the answer gives, or what it lacks.")
    usefulness: Literal["useful", "not_useful"] = Field(
        description=(
            "useful: it directly gives what the question asks for. not_useful: it is "
            "about the topic but does not give the specific thing asked."
        )
    )
