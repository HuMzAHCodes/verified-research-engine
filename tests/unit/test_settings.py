"""Checks on the settings module.

CONCEPT: secrets stay hidden. These tests guard NFR-09 at the configuration
level: a key never appears when the settings are printed, and an empty key
stops the program at start-up instead of failing later at the first API call.

CONCEPT: mcp-client, decision 8. The search-mcp address has a working default,
and a bad value stops the program at start-up with the variable's name.

CONCEPT: shared-throttle. The rate per tier and the call budget have defaults
taken from the measured limits, and values that would switch a limit off are
refused at start-up.

CONCEPT: document-ingestion. The embedding rate and batch size have their
agreed defaults (D-48), and the pdf-parser-mcp address is checked like the
search-mcp one.

CONCEPT: chat-provider-setting (D-65, D-66). The chat vendor is a setting
checked at start-up; only the chosen vendor's key is required; the models
default to one pinned Gemini Flash-Lite model per tier.
"""

import pytest
from pydantic import ValidationError

from settings import DEFAULT_PDF_PARSER_MCP_URL, DEFAULT_SEARCH_MCP_URL, Settings, settings


def test_api_keys_are_hidden_when_settings_are_printed():
    printed_settings = repr(settings)
    for api_key in (
        settings.google_api_key,
        settings.groq_api_key,
        settings.mistral_api_key,
        settings.tavily_api_key,
        settings.langsmith_api_key,
    ):
        if api_key is not None:
            assert api_key.get_secret_value() not in printed_settings


def test_empty_api_key_is_rejected(monkeypatch):
    # Override the real key with an empty one, and skip the .env file so it
    # cannot supply the real value again.
    monkeypatch.setenv("TAVILY_API_KEY", "")
    with pytest.raises(ValidationError, match="is empty"):
        Settings(_env_file=None)


def chosen_vendor(monkeypatch, provider: str, **keys: str | None) -> None:
    """Choose a chat vendor and set or remove each vendor's key in the environment."""
    monkeypatch.setenv("VRE_CHAT_PROVIDER", provider)
    for variable, value in keys.items():
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)


def test_the_chat_vendor_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("VRE_CHAT_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder")

    assert Settings(_env_file=None).vre_chat_provider == "google_genai"


def test_an_unknown_chat_vendor_stops_the_program(monkeypatch):
    # The likeliest typo: the product's name instead of LangChain's vendor name.
    monkeypatch.setenv("VRE_CHAT_PROVIDER", "gemini")
    with pytest.raises(ValidationError, match="vre_chat_provider"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("provider", "key_variable"),
    [
        ("google_genai", "GOOGLE_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("mistralai", "MISTRAL_API_KEY"),
    ],
)
@pytest.mark.parametrize("missing_value", [None, "", "   "])
def test_the_chosen_vendors_key_is_required(monkeypatch, provider, key_variable, missing_value):
    chosen_vendor(monkeypatch, provider, **{key_variable: missing_value})

    with pytest.raises(ValidationError) as raised:
        Settings(_env_file=None)

    # The message names the variable to set and the vendor that needs it.
    assert f"{key_variable} is missing or empty" in str(raised.value)
    assert f"VRE_CHAT_PROVIDER={provider}" in str(raised.value)


def test_the_other_vendors_keys_may_be_absent_or_blank(monkeypatch):
    chosen_vendor(
        monkeypatch, "groq", GROQ_API_KEY="placeholder", GOOGLE_API_KEY="", MISTRAL_API_KEY=None
    )

    configured = Settings(_env_file=None)

    # A blank line copied from .env.example counts as absent, not as a key.
    assert configured.google_api_key is None
    assert configured.mistral_api_key is None
    assert configured.chat_api_key().get_secret_value() == "placeholder"


def test_search_mcp_url_defaults_to_the_local_server(monkeypatch):
    # load_dotenv() may have copied a value from .env into the environment.
    monkeypatch.delenv("VRE_SEARCH_MCP_URL", raising=False)
    assert Settings(_env_file=None).vre_search_mcp_url == DEFAULT_SEARCH_MCP_URL


def test_a_remote_search_mcp_url_is_accepted(monkeypatch):
    monkeypatch.setenv("VRE_SEARCH_MCP_URL", "https://search.example.com/mcp")
    assert Settings(_env_file=None).vre_search_mcp_url == "https://search.example.com/mcp"


@pytest.mark.parametrize(
    ("bad_value", "expected_message"),
    [
        ("", "is empty"),
        ("   ", "is empty"),
        ("127.0.0.1:8765/mcp", "http:// or https://"),
        ("ftp://127.0.0.1/mcp", "http:// or https://"),
    ],
)
def test_a_bad_search_mcp_url_stops_the_program(monkeypatch, bad_value, expected_message):
    monkeypatch.setenv("VRE_SEARCH_MCP_URL", bad_value)
    with pytest.raises(ValidationError, match=expected_message):
        Settings(_env_file=None)


def test_rates_and_budget_default_to_the_measured_limits(monkeypatch):
    for variable in ("VRE_DECISION_RPS", "VRE_WRITER_RPS", "VRE_MAX_MODEL_CALLS_PER_RUN"):
        monkeypatch.delenv(variable, raising=False)

    defaults = Settings(_env_file=None)

    # 80% of the 15 requests a minute each Flash-Lite model allows (D-43, D-65).
    assert (defaults.vre_decision_rps, defaults.vre_writer_rps) == (0.2, 0.2)
    # D-58: the ceiling above the budget each run sizes to its plan.
    assert defaults.vre_max_model_calls_per_run == 200


def test_models_default_to_one_pinned_flash_lite_model_per_tier(monkeypatch):
    for variable in ("VRE_CHAT_MODEL", "VRE_WRITER_MODEL"):
        monkeypatch.delenv(variable, raising=False)

    defaults = Settings(_env_file=None)

    # D-65: two models, so two daily quotas; the decision tier gets the newer.
    assert (defaults.vre_chat_model, defaults.vre_writer_model) == (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    )
    # Never an alias that can move, never a model that can be withdrawn.
    for model_name in (defaults.vre_chat_model, defaults.vre_writer_model):
        assert not model_name.endswith(("-latest", "-preview"))


@pytest.mark.parametrize(
    ("variable", "bad_value"),
    [
        ("VRE_DECISION_RPS", "0"),
        ("VRE_WRITER_RPS", "-1"),
        ("VRE_MAX_MODEL_CALLS_PER_RUN", "0"),
        ("VRE_MAX_MODEL_CALLS_PER_RUN", "500"),
    ],
)
def test_a_rate_or_budget_out_of_range_stops_the_program(monkeypatch, variable, bad_value):
    monkeypatch.setenv(variable, bad_value)
    with pytest.raises(ValidationError, match=variable.lower()):
        Settings(_env_file=None)


def test_pdf_parser_mcp_url_defaults_to_the_local_server(monkeypatch):
    monkeypatch.delenv("VRE_PDF_PARSER_MCP_URL", raising=False)
    assert Settings(_env_file=None).vre_pdf_parser_mcp_url == DEFAULT_PDF_PARSER_MCP_URL


@pytest.mark.parametrize(
    ("bad_value", "expected_message"),
    [
        ("", "is empty"),
        ("127.0.0.1:8766/mcp", "http:// or https://"),
    ],
)
def test_a_bad_pdf_parser_mcp_url_stops_the_program(monkeypatch, bad_value, expected_message):
    monkeypatch.setenv("VRE_PDF_PARSER_MCP_URL", bad_value)
    with pytest.raises(ValidationError, match=expected_message) as raised:
        Settings(_env_file=None)
    # The message names this setting's own default, not search-mcp's.
    assert DEFAULT_PDF_PARSER_MCP_URL in str(raised.value)
    assert "vre_pdf_parser_mcp_url" in str(raised.value)


def test_embedding_rate_and_batch_size_default_to_the_agreed_values(monkeypatch):
    for variable in ("VRE_EMBED_RPS", "VRE_EMBED_BATCH_SIZE"):
        monkeypatch.delenv(variable, raising=False)

    defaults = Settings(_env_file=None)

    # 80% of mistral-embed's 1 request a second; 32 chunks a request (D-48).
    assert defaults.vre_embed_rps == 0.8
    assert defaults.vre_embed_batch_size == 32


@pytest.mark.parametrize(
    ("variable", "bad_value"),
    [
        ("VRE_EMBED_RPS", "0"),
        ("VRE_EMBED_BATCH_SIZE", "0"),
        ("VRE_EMBED_BATCH_SIZE", "500"),
    ],
)
def test_an_embedding_setting_out_of_range_stops_the_program(monkeypatch, variable, bad_value):
    monkeypatch.setenv(variable, bad_value)
    with pytest.raises(ValidationError, match=variable.lower()):
        Settings(_env_file=None)
