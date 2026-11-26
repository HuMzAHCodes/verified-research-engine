"""Live checks of the model gateway against the configured chat vendor.

CONCEPT: model-gateway. These run only on request (`uv run pytest -m live`),
never in CI: they need a real key and spend free-tier quota. Each test makes
one call per model tier and prints how long it took, for the concept notes.

CONCEPT: chat-provider-setting, decision 6. This is the first check of the
qualification: whatever VRE_CHAT_PROVIDER names (Gemini from Phase 5) must
return a validated object on both tiers. The vendor and model are printed.
"""

import time
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graph.llm import ModelGateway
from settings import settings

pytestmark = pytest.mark.live


class CapitalAnswer(BaseModel):
    """A tiny schema: one closed choice and one short string."""

    country: str = Field(description="The country named in the question.")
    continent: Literal["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]


CAPITAL_QUESTION = [
    SystemMessage(content="Answer by filling in the requested fields."),
    HumanMessage(content="Which country has Islamabad as its capital, and on which continent?"),
]


@pytest.mark.parametrize("tier", ["decision", "writer"])
async def test_gateway_returns_a_validated_object(tier):
    gateway = ModelGateway()
    started_at = time.perf_counter()

    capital_answer = await gateway.structured(
        CapitalAnswer,
        CAPITAL_QUESTION,
        prompt_id="LIVE-CHECK",
        prompt_version="v1",
        tier=tier,
    )

    elapsed_seconds = time.perf_counter() - started_at
    model_name = gateway.model_names_by_tier[tier]
    print(
        f"\n{tier} tier ({settings.vre_chat_provider}, {model_name}): "
        f"{capital_answer!r} in {elapsed_seconds:.1f} s"
    )
    assert isinstance(capital_answer, CapitalAnswer)
    assert capital_answer.continent == "Asia"
