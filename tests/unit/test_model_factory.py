"""Building the chat models from settings: any configured vendor, one way.

CONCEPT: chat-provider-setting, decision 2 (D-66). build_chat_model hands the
vendor, the model name, and the key to LangChain's init_chat_model. These
tests build each vendor's real client, which sends no request, and check the
settings every vendor must share: temperature 0, a single attempt (the
gateway is the only retry layer), the gateway's timeout, and the key given.
"""

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr

from graph.llm import (
    REQUEST_TIMEOUT_SECONDS,
    ModelGateway,
    build_chat_model,
    build_configured_chat_model,
)
from settings import settings

# Built at run time so this file holds no key-shaped string (TC-NFR-09).
FAKE_KEY = SecretStr("k" * 40)

VENDORS = [
    ("google_genai", "gemini-3.5-flash-lite", ChatGoogleGenerativeAI, "google_api_key"),
    ("groq", "openai/gpt-oss-20b", ChatGroq, "groq_api_key"),
    ("mistralai", "voxtral-small-2507", ChatMistralAI, "mistral_api_key"),
]


def timeout_of(chat_model) -> float:
    """The timeout in seconds, from whichever field the vendor's class keeps it in."""
    return getattr(chat_model, "timeout", None) or chat_model.request_timeout


@pytest.mark.parametrize(("provider", "model_name", "model_class", "key_field"), VENDORS)
def test_each_vendor_gets_its_own_model_with_the_shared_settings(
    provider, model_name, model_class, key_field
):
    chat_model = build_chat_model(provider, model_name, FAKE_KEY)

    assert isinstance(chat_model, model_class)
    assert chat_model.model == model_name
    # Groq's client stores temperature 0 as its own floor, 1e-08.
    assert chat_model.temperature == pytest.approx(0, abs=1e-6)
    # One attempt in the client: the gateway's retry is the only retry layer.
    assert chat_model.max_retries == 1
    assert timeout_of(chat_model) == REQUEST_TIMEOUT_SECONDS
    assert getattr(chat_model, key_field).get_secret_value() == FAKE_KEY.get_secret_value()


def test_the_configured_vendor_and_its_key_are_used(monkeypatch):
    monkeypatch.setattr(settings, "vre_chat_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", FAKE_KEY)

    chat_model = build_configured_chat_model("openai/gpt-oss-120b")

    assert isinstance(chat_model, ChatGroq)
    assert chat_model.model == "openai/gpt-oss-120b"
    assert chat_model.groq_api_key.get_secret_value() == FAKE_KEY.get_secret_value()


def test_a_missing_key_for_the_configured_vendor_is_named(monkeypatch):
    # Settings refuse this at start-up; changed afterwards, the build still says why.
    monkeypatch.setattr(settings, "vre_chat_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", None)

    with pytest.raises(ValueError, match="GROQ_API_KEY is missing or empty"):
        build_configured_chat_model("openai/gpt-oss-20b")


def test_the_gateway_builds_each_tier_from_the_configured_vendor(monkeypatch):
    monkeypatch.setattr(settings, "vre_chat_provider", "google_genai")
    monkeypatch.setattr(settings, "google_api_key", FAKE_KEY)
    monkeypatch.setattr(settings, "vre_chat_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "vre_writer_model", "gemini-3.1-flash-lite")

    gateway = ModelGateway(requests_per_second=100)

    decision_model = gateway.models_by_tier["decision"]
    writer_model = gateway.models_by_tier["writer"]
    assert isinstance(decision_model, ChatGoogleGenerativeAI)
    assert isinstance(writer_model, ChatGoogleGenerativeAI)
    # One model per tier (D-65): two daily quotas.
    assert (decision_model.model, writer_model.model) == (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    )
