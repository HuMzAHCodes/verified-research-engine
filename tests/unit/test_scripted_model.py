"""The scripted model's two kinds of script: in call order, and keyed to the prompt.

CONCEPT: parallel-fan-out. Parallel branches call the model at the same time,
so the order their calls arrive in changes from run to run. Measured: the
Phase 2 graph test, with its P-02 replies in call order, failed 4 runs in 15
once the branches ran in parallel, because a branch sometimes received the
other branch's reply. Replies keyed to text in the prompt cannot go to the
wrong call. These tests pin how ScriptedModel chooses a reply.
"""

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from tests.conftest import ScriptedModel


async def ask(model: ScriptedModel, prompt: str) -> str:
    reply = await model.ainvoke([HumanMessage(content=prompt)])
    return str(reply.content)


async def test_a_keyed_reply_goes_to_the_prompt_that_contains_its_key():
    model = ScriptedModel(
        replies_by_prompt_text={
            "What does it sell?": ["Robots."],
            "Who buys it?": ["Retailers."],
        }
    )

    # Asked in the opposite order to the script, and at the same time.
    answers = await asyncio.gather(
        ask(model, "Question: Who buys it?"), ask(model, "Question: What does it sell?")
    )

    assert answers == ["Retailers.", "Robots."]


async def test_a_prompt_with_no_key_takes_the_next_reply_in_call_order():
    model = ScriptedModel(
        scripted_replies=["First.", "Second."],
        replies_by_prompt_text={"What does it sell?": ["Robots."]},
    )

    assert await ask(model, "Plan the topic.") == "First."
    assert await ask(model, "Question: What does it sell?") == "Robots."
    assert await ask(model, "Write the outline.") == "Second."


async def test_a_prompt_that_matches_two_keys_fails_the_test():
    model = ScriptedModel(replies_by_prompt_text={"sell": ["A."], "it sell": ["B."]})

    with pytest.raises(AssertionError, match="matches several keys"):
        await ask(model, "What does it sell?")


async def test_a_second_call_for_a_used_up_key_fails_the_test():
    model = ScriptedModel(replies_by_prompt_text={"What does it sell?": ["Robots."]})
    await ask(model, "What does it sell?")

    with pytest.raises(AssertionError, match="ran out of replies for 'What does it sell\\?'"):
        await ask(model, "What does it sell?")
