"""Checks on the settings module.

CONCEPT: secrets stay hidden. These tests guard NFR-09 at the configuration
level: a key never appears when the settings are printed, and an empty key
stops the program at start-up instead of failing later at the first API call.
"""

import pytest
from pydantic import ValidationError

from settings import Settings, settings


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