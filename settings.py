"""Project settings: the only module that reads environment variables.

CONCEPT: centralised configuration. Every other module imports the ready-made
`settings` object instead of calling os.getenv itself, so there is exactly one
place that knows the variable names, their defaults, and their allowed ranges.
"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Copy the key=value lines from .env into the process environment before
# anything else runs. LangSmith reads its own variables (LANGSMITH_TRACING,
# LANGSMITH_PROJECT) straight from the environment, not from the Settings
# object, so this has to happen first.
# Existing environment variables win over .env, which is what CI relies on.
load_dotenv()

# Where search-mcp listens while developing: this machine only, started by hand
# in its own terminal (D-32). Phase 11 points the variable at the deployed server.
DEFAULT_SEARCH_MCP_URL = "http://127.0.0.1:8765/mcp"

# Where pdf-parser-mcp listens while developing, beside search-mcp (D-46).
# Ingestion reaches it only through this address, never by importing pypdf
# (document-ingestion, decision 7).
DEFAULT_PDF_PARSER_MCP_URL = "http://127.0.0.1:8766/mcp"

# The default for each MCP address setting, used in its error messages.
DEFAULT_MCP_URLS = {
    "vre_search_mcp_url": DEFAULT_SEARCH_MCP_URL,
    "vre_pdf_parser_mcp_url": DEFAULT_PDF_PARSER_MCP_URL,
}

# CONCEPT: chat-provider-setting, decision 2 (D-66). The chat vendors this
# project has tried or set up, by the names LangChain's init_chat_model uses.
# A Literal, so a typo in VRE_CHAT_PROVIDER stops the program at start-up.
ChatProvider = Literal["google_genai", "groq", "mistralai"]

# The setting that holds each vendor's key. Only the chosen vendor's key is
# required; the others may be absent.
API_KEY_SETTING_BY_PROVIDER: dict[str, str] = {
    "google_genai": "google_api_key",
    "groq": "groq_api_key",
    "mistralai": "mistral_api_key",
}


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
    # .get_secret_value(). Tavily and LangSmith have no default, so a missing
    # key is an error. Each chat vendor's key may be absent, except the chosen
    # vendor's (require_the_chosen_vendors_key, D-66).
    tavily_api_key: SecretStr
    langsmith_api_key: SecretStr
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    # Mistral refused every request from Phase 5 (402, D-65). Kept so Mistral
    # stays a valid vendor (D-68); the embeddings use it until concept
    # local-embeddings moves them to this machine.
    mistral_api_key: SecretStr | None = None

    # Tracing. Kept here so the values are validated and visible in one place,
    # even though LangSmith itself reads them from the environment.
    langsmith_tracing: bool = True
    langsmith_project: str = "vre-dev"

    # Models. The decision prompts and the writer prompts each name a tier, so
    # the two can run on different models (D-24). CONCEPT: chat-provider-setting
    # (D-65, D-66). The vendor is a setting; the default is the Gemini API's
    # free tier, after Mistral's free plan stopped including API access at the
    # Phase 5 opening. Each tier gets its own Flash-Lite model, because Google
    # counts the daily quota per model: 500 requests a day each, 1,000
    # together (our AI Studio page). The decision tier makes most calls and
    # carries P-02's word-for-word copying, so it gets the newer model. Pinned
    # names only: never "-latest", which can move to another model, and never
    # "-preview", which can be withdrawn.
    vre_chat_provider: ChatProvider = "google_genai"
    vre_chat_model: str = "gemini-3.5-flash-lite"
    vre_writer_model: str = "gemini-3.1-flash-lite"
    vre_embed_model: str = "mistral-embed"

    # Caps and limits. Field(ge=..., le=...) rejects out-of-range values at
    # start-up: VRE_MAX_SUB_QUESTIONS=50 fails here, not as a huge bill later.
    vre_max_sub_questions: int = Field(default=6, ge=1, le=10)
    vre_max_revisions: int = Field(default=3, ge=0, le=10)
    vre_passage_max_chars: int = Field(default=1500, ge=200)

    # CONCEPT: shared-throttle (D-43), chat-provider-setting (D-65). One rate
    # per model tier, in requests a second, at 80% of the model's limit: both
    # Flash-Lite models allow 15 a minute, so 12 a minute, 0.2 a second. The
    # other fifth is room for a live test at the same time and for clock
    # differences. The registry of limiters is keyed by model name, so the two
    # tiers, on two models, each have their own. D-55 set 0.8 for Mistral's
    # Voxtral Small (60 a minute), which both tiers shared.
    vre_decision_rps: float = Field(default=0.2, gt=0)
    vre_writer_rps: float = Field(default=0.2, gt=0)
    # Requests to the model one run may send, retries included (D-41, D-43).
    # A Phase 3 brief sends at most 14, or 28 if every call needed its schema
    # retry; going over stops the run instead of letting a loop keep spending.
    # D-58: each run's budget is sized to its plan after the planner
    # (graph/run_budget.py); this is the ceiling above it, 200 because the
    # largest plan (six sub-questions, all `both`) can need 172.
    vre_max_model_calls_per_run: int = Field(default=200, ge=1, le=200)

    # CONCEPT: document-ingestion, decision 3 (D-48). The embedding model gets
    # its own limiter, at 80% of its measured limit of 1 request a second, for
    # the same reasons as the chat tiers above. Ingestion and a run's question
    # embeddings share it, so together they stay under the account's limit.
    vre_embed_rps: float = Field(default=0.8, gt=0)
    # Chunks sent in one embedding request (document-ingestion, choice B).
    # 32 is about 34 requests for the annual report; the upper bound keeps one
    # request well below any per-request size limit that has not been checked.
    vre_embed_batch_size: int = Field(default=32, ge=1, le=128)

    # Storage and services
    vre_data_dir: Path = Path("data")
    vre_search_mcp_url: str = DEFAULT_SEARCH_MCP_URL
    vre_pdf_parser_mcp_url: str = DEFAULT_PDF_PARSER_MCP_URL
    vre_price_per_million_tokens: float = Field(default=0.0, ge=0)

    @field_validator("tavily_api_key", "langsmith_api_key")
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

    @field_validator("google_api_key", "groq_api_key", "mistral_api_key", mode="before")
    @classmethod
    def blank_vendor_key_is_absent(cls, api_key: object) -> object:
        """An empty vendor key counts as absent, like a missing line.

        .env.example lists every vendor's key with an empty value; copied into
        .env, the unused ones must not count as set.
        """
        if isinstance(api_key, str) and not api_key.strip():
            return None
        return api_key

    @model_validator(mode="after")
    def require_the_chosen_vendors_key(self) -> "Settings":
        """The chosen vendor's key must be set; the others may be absent (D-66).

        Checked at start-up, so a run never reaches the planner only to fail
        with a vague 401 from the provider.
        """
        key_setting = API_KEY_SETTING_BY_PROVIDER[self.vre_chat_provider]
        if getattr(self, key_setting) is None:
            raise ValueError(
                f"{key_setting.upper()} is missing or empty, and "
                f"VRE_CHAT_PROVIDER={self.vre_chat_provider} needs it; set it in .env"
            )
        return self

    def chat_api_key(self) -> SecretStr:
        """The key of the chosen chat vendor (require_the_chosen_vendors_key ensures it)."""
        key_setting = API_KEY_SETTING_BY_PROVIDER[self.vre_chat_provider]
        api_key = getattr(self, key_setting)
        if api_key is None:  # only if the setting was changed after start-up
            raise ValueError(f"{key_setting.upper()} is missing or empty; set it in .env")
        return api_key

    @field_validator("vre_search_mcp_url", "vre_pdf_parser_mcp_url")
    @classmethod
    def require_http_address(cls, server_url: str, field_info: ValidationInfo) -> str:
        """Check an MCP server address at start-up, not at the first tool call.

        CONCEPT: mcp-client, decision 8. The search-mcp value used to mean
        "empty runs the server locally"; with HTTP on localhost (D-32) there is
        no such mode, so an empty value is a mistake, most likely a line left
        over from an older .env. A missing scheme ("127.0.0.1:8765/mcp") is the
        other common typo. pdf-parser-mcp runs the same way (D-46), so its
        address gets the same check; each message names that setting's default.
        """
        default_url = DEFAULT_MCP_URLS[field_info.field_name]
        server_url = server_url.strip()
        if not server_url:
            raise ValueError(
                f"is empty; delete the line from .env to use {default_url}, "
                "or set the server's address"
            )
        parts = urlsplit(server_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"must be an http:// or https:// address, such as {default_url}")
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
