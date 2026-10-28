"""Project settings: the only module that reads environment variables.

CONCEPT: centralised configuration. Every other module imports the ready-made
`settings` object instead of calling os.getenv itself, so there is exactly one
place that knows the variable names, their defaults, and their allowed ranges.
"""

from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Copy the key=value lines from .env into the process environment before
# anything else runs. LangChain and LangSmith read their own variables
# (MISTRAL_API_KEY, LANGSMITH_TRACING, LANGSMITH_PROJECT) straight from the
# environment, not from the Settings object, so this has to happen first.
# Existing environment variables win over .env, which is what CI relies on.
load_dotenv()

# Where search-mcp listens while developing: this machine only, started by hand
# in its own terminal (D-32). Phase 11 points the variable at the deployed server.
DEFAULT_SEARCH_MCP_URL = "http://127.0.0.1:8765/mcp"


class Settings(BaseSettings):
    """Typed view of the environment.

    CONCEPT: pydantic-settings. Each attribute is matched to an environment
    variable with the same name, case-insensitive (vre_chat_model reads
    VRE_CHAT_MODEL). Values are converted to the declared type and checked
    against the Field constraints when the object is created, so a bad value
    stops the program at start-up instead of failing deep inside a run.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Secrets. SecretStr hides the value when printed or logged ("**********");
    # the real value is read only where a client is built, with
    # .get_secret_value(). No default, so a missing key is an error.
    mistral_api_key: SecretStr
    tavily_api_key: SecretStr
    langsmith_api_key: SecretStr

    # Tracing. Kept here so the values are validated and visible in one place,
    # even though LangSmith itself reads them from the environment.
    langsmith_tracing: bool = True
    langsmith_project: str = "vre-dev"

    # Models. The decision prompts use the small model; the writer prompts use
    # the larger one, which fills long, structured answers more reliably.
    vre_chat_model: str = "ministral-8b-latest"
    vre_writer_model: str = "ministral-14b-latest"
    vre_embed_model: str = "mistral-embed"

    # Caps and limits. Field(ge=..., le=...) rejects out-of-range values at
    # start-up: VRE_MAX_SUB_QUESTIONS=50 fails here, not as a huge bill later.
    vre_max_sub_questions: int = Field(default=6, ge=1, le=10)
    vre_max_revisions: int = Field(default=3, ge=0, le=10)
    vre_passage_max_chars: int = Field(default=1500, ge=200)

    # CONCEPT: shared-throttle (D-43). One rate per model tier, in requests a
    # second, at 80% of the limit read from the provider's console before
    # Phase 3 (ENV_AND_CONFIG.md, "Free-tier limits"): ministral-8b allows
    # 3.13, ministral-14b 0.50. The other fifth is room for a live test at the
    # same time, the Phase 5 citation checker, and clock differences. They
    # replace VRE_THROTTLE_RPS, one rate for both tiers, which fitted neither.
    vre_decision_rps: float = Field(default=2.5, gt=0)
    vre_writer_rps: float = Field(default=0.4, gt=0)
    # Requests to the model one run may send, retries included (D-41, D-43).
    # A Phase 3 brief sends at most 14, or 28 if every call needed its schema
    # retry; going over stops the run instead of letting a loop keep spending.
    vre_max_model_calls_per_run: int = Field(default=30, ge=1, le=200)

    # Storage and services
    vre_data_dir: Path = Path("data")
    vre_search_mcp_url: str = DEFAULT_SEARCH_MCP_URL
    vre_price_per_million_tokens: float = Field(default=0.0, ge=0)

    @field_validator("mistral_api_key", "tavily_api_key", "langsmith_api_key")
    @classmethod
    def reject_empty_api_key(cls, api_key: SecretStr) -> SecretStr:
        """Treat an empty key as missing.

        Pydantic accepts "" as a valid SecretStr, so without this check an
        empty key in .env would pass here and only fail later as a vague 401
        from the provider.
        """
        if not api_key.get_secret_value().strip():
            raise ValueError("is empty; set it in .env")
        return api_key

    @field_validator("vre_search_mcp_url")
    @classmethod
    def require_http_address(cls, server_url: str) -> str:
        """Check the search-mcp address at start-up, not at the first tool call.

        CONCEPT: mcp-client, decision 8. The value used to mean "empty runs the
        server locally"; with HTTP on localhost (D-32) there is no such mode, so
        an empty value is a mistake, most likely a line left over from an older
        .env. A missing scheme ("127.0.0.1:8765/mcp") is the other common typo.
        """
        server_url = server_url.strip()
        if not server_url:
            raise ValueError(
                f"is empty; delete the line from .env to use {DEFAULT_SEARCH_MCP_URL}, "
                "or set the server's address"
            )
        parts = urlsplit(server_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(
                f"must be an http:// or https:// address, such as {DEFAULT_SEARCH_MCP_URL}"
            )
        return server_url

    @property
    def app_db_path(self) -> Path:
        """SQLite file for jobs, events, decisions, and documents (phase 8)."""
        return self.vre_data_dir / "app.db"

    @property
    def checkpoint_db_path(self) -> Path:
        """SQLite file for LangGraph checkpoints (phase 6)."""
        return self.vre_data_dir / "checkpoints.db"


# One shared instance, created at import time. Importing this module is what
# validates the configuration.
settings = Settings()
