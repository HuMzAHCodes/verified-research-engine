"""Checks on the settings module.

CONCEPT: secrets stay hidden. These tests guard NFR-09 at the configuration
level: a key never appears when the settings are printed, and an empty key
stops the program at start-up instead of failing later at the first API call.

CONCEPT: mcp-client, decision 8. The search-mcp address has a working default,
and a bad value stops the program at start-up with the variable's name.

CONCEPT: shared-throttle. The rate per tier and the call budget have defaults
taken from the measured limits, and values that would switch a limit off are
refused at start-up.
"""

import pytest
from pydantic import ValidationError

from settings import DEFAULT_SEARCH_MCP_URL, Settings, settings


def test_api_keys_are_hidden_when_settings_are_printed():
    printed_settings = repr(settings)
    for api_key in (settings.mistral_api_key, settings.tavily_api_key, settings.langsmith_api_key):
        assert api_key.get_secret_value() not in printed_settings


def test_empty_api_key_is_rejected(monkeypatch):
    # Override the real key with an empty one, and skip the .env file so it
    # cannot supply the real value again.
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    with pytest.raises(ValidationError, match="is empty"):
        Settings(_env_file=None)


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

    # 80% of 3.13 and 0.50 requests a second (D-43).
    assert (defaults.vre_decision_rps, defaults.vre_writer_rps) == (2.5, 0.4)
    assert defaults.vre_max_model_calls_per_run == 30


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
