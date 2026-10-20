"""search-mcp's own settings.

CONCEPT: building-search-mcp, decision 2 (D-31). The server is deployed on its
own, so it reads only what it needs: the Tavily key, where to listen, and its
limits. It never imports the project's settings.py, so it never needs, and
never holds, the Mistral or LangSmith keys.

Read only when the server starts (load_search_server_config), never at import,
so the tests can build the server without a key.
"""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchServerConfig(BaseSettings):
    """Settings for search-mcp, from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tavily_api_key: SecretStr

    # Listen on this machine only (D-32). Phase 11 changes the host when the
    # server is deployed, together with an access check.
    search_mcp_host: str = "127.0.0.1"
    search_mcp_port: int = Field(default=8765, ge=1024, le=65535)

    # Limits on every page fetch (decision 5).
    fetch_timeout_seconds: float = Field(default=15.0, gt=0)
    fetch_max_bytes: int = Field(default=2_000_000, ge=10_000)
    fetch_max_redirects: int = Field(default=3, ge=0, le=10)

    @field_validator("tavily_api_key")
    @classmethod
    def reject_empty_key(cls, api_key: SecretStr) -> SecretStr:
        if not api_key.get_secret_value().strip():
            raise ValueError("is empty; set TAVILY_API_KEY in .env")
        return api_key


def load_search_server_config() -> SearchServerConfig:
    return SearchServerConfig()
