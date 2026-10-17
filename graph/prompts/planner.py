"""P-01: the planner prompt.

CONCEPT: structured-output-planning. The prompt asks; the schema and the node
enforce. Rules that must hold (the cap, web-only without documents, ids) are
also checked in code by the planner node, because a prompt is a request, not
a guarantee.

Raise VERSION whenever the wording changes, and log the change in PROMPTS.md.
The version is sent with every call as trace metadata, so an eval score can
always be tied to the prompt text that produced it (rule G-9).
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

PROMPT_ID = "P-01"
VERSION = "v1"

SYSTEM_TEMPLATE = """You plan research for a written brief. Split the topic into sub-questions that
together cover it. Each sub-question must be specific enough that a source can
answer it in a few sentences.

For each sub-question choose where to look:
- document: only the uploaded files can answer it, for example what the company
  says about itself.
- web: only public sources can answer it, for example recent news or competitors.
- both: the uploaded files make a claim that public sources can confirm or
  contradict.

Write at most {max_sub_questions} sub-questions. Do not repeat a question in
different words. Do not answer the questions."""

HUMAN_TEMPLATE = """Topic: {topic}
Uploaded files: {file_names}
Preferred report format: {report_format}"""


def build_planner_messages(
    topic: str,
    uploaded_file_names: list[str],
    report_format: str,
    max_sub_questions: int,
) -> list[BaseMessage]:
    """The two messages for one planner call."""
    # Say "none" explicitly rather than leaving the line empty: an empty value
    # reads as missing data, "none" reads as a fact the model can act on.
    file_names = ", ".join(uploaded_file_names) if uploaded_file_names else "none"
    return [
        SystemMessage(content=SYSTEM_TEMPLATE.format(max_sub_questions=max_sub_questions)),
        HumanMessage(
            content=HUMAN_TEMPLATE.format(
                topic=topic,
                file_names=file_names,
                report_format=report_format or "none stated",
            )
        ),
    ]