"""Scripted tests of the writer node.

CONCEPT: sequential-graph. ScriptedModel plays the 14b writer model: first the
outline reply, then one reply per section. received_conversations shows what
each call was given, which is how TC-WRT-03 checks the prompt contents.
"""

import pytest

from graph.nodes.writer import WriterError, writer_node
from graph.state import Finding, SubQuestion
from tests.conftest import ScriptedModel

TOPIC = "Due diligence on Symbotic"
SUB_QUESTIONS = [
    SubQuestion(id="Q1", text="What does Symbotic sell?", route="web"),
    SubQuestion(id="Q2", text="Who are its customers?", route="web"),
    SubQuestion(id="Q3", text="What are the main risks?", route="web"),
]


def web_finding(finding_id: str, found: bool = True) -> Finding:
    question_id = finding_id.split("-")[0]
    return Finding(
        id=finding_id,
        sub_question_id=question_id,
        found=found,
        source_type="web",
        locator=f"https://example.com/{finding_id}",
        title=f"Page for {finding_id}",
        passage=f"Passage for {finding_id}." if found else "",
    )


FINDINGS = [web_finding("Q1-F1"), web_finding("Q1-F2"), web_finding("Q2-F1"), web_finding("Q3-F1")]


def outline_reply(*sections: tuple[str, list[str]]) -> dict:
    return {
        "sections": [
            {"title": title, "sub_question_ids": question_ids} for title, question_ids in sections
        ]
    }


def section_reply(text: str, title: str = "Any title") -> dict:
    return {"title": title, "text": text}


def writer_state(findings=None) -> dict:
    return {"topic": TOPIC, "sub_questions": SUB_QUESTIONS, "findings": findings or FINDINGS}


THREE_SECTION_OUTLINE = outline_reply(
    ("Products", ["Q1"]), ("Customers", ["Q2"]), ("Risks", ["Q3"])
)


async def test_tc_wrt_01_one_outline_call_then_one_call_per_section(make_gateway):
    writer_model = ScriptedModel(
        scripted_replies=[
            THREE_SECTION_OUTLINE,
            section_reply("Symbotic sells warehouse robots. [Q1-F1]"),
            section_reply("Its largest customer is a retailer. [Q2-F1]"),
            section_reply("Customer concentration is a risk. [Q3-F1]"),
        ]
    )

    writer_update = await writer_node(
        writer_state(), gateway=make_gateway(writer_model=writer_model)
    )

    assert len(writer_model.received_conversations) == 1 + 3
    assert writer_model.scripted_replies == []
    assert [section.title for section in writer_update["draft"]] == [
        "Products",
        "Customers",
        "Risks",
    ]


async def test_tc_wrt_03_each_section_prompt_holds_only_its_own_findings(make_gateway):
    writer_model = ScriptedModel(
        scripted_replies=[
            THREE_SECTION_OUTLINE,
            section_reply("Text. [Q1-F1]"),
            section_reply("Text. [Q2-F1]"),
            section_reply("Text. [Q3-F1]"),
        ]
    )

    await writer_node(writer_state(), gateway=make_gateway(writer_model=writer_model))

    expected_ids_by_call = [{"Q1-F1", "Q1-F2"}, {"Q2-F1"}, {"Q3-F1"}]
    all_finding_ids = {finding.id for finding in FINDINGS}
    section_conversations = writer_model.received_conversations[1:]
    for section_conversation, expected_ids in zip(
        section_conversations, expected_ids_by_call, strict=True
    ):
        section_prompt = section_conversation[-1].content
        for finding_id in all_finding_ids:
            if finding_id in expected_ids:
                assert f'id="{finding_id}"' in section_prompt
            else:
                assert finding_id not in section_prompt


async def test_markers_for_findings_not_in_the_prompt_are_removed(make_gateway):
    writer_model = ScriptedModel(
        scripted_replies=[
            outline_reply(("Products", ["Q1"]), ("Customers", ["Q2"]), ("Risks", ["Q3"])),
            # Q2-F1 exists, but not in this section's prompt; Q9-F1 does not exist.
            section_reply("Robots. [Q1-F1] Customers. [Q2-F1] Invented. [Q9-F1]"),
            section_reply("Text. [Q2-F1]"),
            section_reply("Text. [Q3-F1]"),
        ]
    )

    writer_update = await writer_node(
        writer_state(), gateway=make_gateway(writer_model=writer_model)
    )

    assert writer_update["draft"][0].text == "Robots. [Q1-F1] Customers. Invented."


async def test_every_id_in_a_multi_id_marker_is_checked(make_gateway):
    writer_model = ScriptedModel(
        scripted_replies=[
            outline_reply(("Products", ["Q1"]), ("Customers", ["Q2"]), ("Risks", ["Q3"])),
            # Found in the first live run: the model put two ids in one marker.
            section_reply(
                "Both known. [Q1-F1, Q1-F2] One unknown. [Q1-F1, Q9-F1] None. [Q9-F1, Q2-F1]"
            ),
            section_reply("Text. [Q2-F1]"),
            section_reply("Text. [Q3-F1]"),
        ]
    )

    writer_update = await writer_node(
        writer_state(), gateway=make_gateway(writer_model=writer_model)
    )

    assert writer_update["draft"][0].text == "Both known. [Q1-F1, Q1-F2] One unknown. [Q1-F1] None."


async def test_the_outline_title_replaces_the_model_title(make_gateway):
    writer_model = ScriptedModel(
        scripted_replies=[
            outline_reply(("Products", ["Q1", "Q2", "Q3"])),
            section_reply("Text. [Q1-F1]", title="A title the model made up"),
        ]
    )

    writer_update = await writer_node(
        writer_state(), gateway=make_gateway(writer_model=writer_model)
    )

    assert writer_update["draft"][0].title == "Products"


async def test_outline_checks_drop_unknown_ids_and_add_missed_questions(make_gateway):
    writer_model = ScriptedModel(
        scripted_replies=[
            # Q9 does not exist; the "Extra" section is left empty and dropped;
            # Q3 has evidence but is missing, so a section is added for it.
            outline_reply(("Overview", ["Q1", "Q2", "Q9"]), ("Extra", ["Q9"])),
            section_reply("Text. [Q1-F1]"),
            section_reply("Text. [Q3-F1]"),
        ]
    )

    writer_update = await writer_node(
        writer_state(), gateway=make_gateway(writer_model=writer_model)
    )

    assert [section.title for section in writer_update["draft"]] == [
        "Overview",
        "What are the main risks?",
    ]
    overview_prompt = writer_model.received_conversations[1][-1].content
    assert 'id="Q2-F1"' in overview_prompt


async def test_outline_prompt_marks_questions_without_evidence(make_gateway):
    findings = [web_finding("Q1-F1"), web_finding("Q2-F1", found=False), web_finding("Q3-F1")]
    writer_model = ScriptedModel(
        scripted_replies=[
            outline_reply(("Products", ["Q1"]), ("Risks", ["Q3"])),
            section_reply("Text. [Q1-F1]"),
            section_reply("Text. [Q3-F1]"),
        ]
    )

    await writer_node(writer_state(findings), gateway=make_gateway(writer_model=writer_model))

    outline_prompt = writer_model.received_conversations[0][-1].content
    assert "Q1 (evidence found)" in outline_prompt
    assert "Q2 (no evidence)" in outline_prompt


async def test_no_evidence_at_all_fails_without_calling_the_model(make_gateway):
    writer_model = ScriptedModel()
    no_evidence = [web_finding(f"Q{number}-F1", found=False) for number in (1, 2, 3)]

    with pytest.raises(WriterError, match="no sub-question found any evidence"):
        await writer_node(
            writer_state(no_evidence), gateway=make_gateway(writer_model=writer_model)
        )

    assert writer_model.received_conversations == []
