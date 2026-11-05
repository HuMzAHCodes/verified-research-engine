"""Shared test helpers: scripted chat and embedding models, a gateway around them, fake tools.

CONCEPT: scripted-model-testing. Tests replace the model, never the code that
calls it. ScriptedModel returns replies written in advance, in order, and
records every conversation it was sent. A real ModelGateway wraps it, so the
gateway's parsing, retries, and repair message all run in every test, while
no request leaves the machine. FakeToolbox does the same for search-mcp; the
phase 1 FakeSearchClient was removed with the direct search client (D-37).
ScriptedEmbeddings stands in for mistral-embed (document-ingestion).
pytest loads this file automatically for every test under tests/.

It also decides whether a test run sends traces to LangSmith (D-41): only a
run that selects the live tests does. See pytest_configure below.
"""

import hashlib
import os
from typing import Any

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from graph.llm import ModelGateway
from graph.mcp_client import SearchHit, ToolCallError, WebPage


def run_selects_live_tests(marker_expression: str) -> bool:
    """True when the -m expression asks for the live tests.

    "live" and "live and not slow" select them; "", "not live", and
    "not (live)" do not. The expression is split into words, so a "live" that
    directly follows "not" does not count.
    """
    words = marker_expression.replace("(", " ").replace(")", " ").split()
    return any(
        word == "live" and (position == 0 or words[position - 1] != "not")
        for position, word in enumerate(words)
    )


def pytest_configure(config: pytest.Config) -> None:
    """Turn LangSmith tracing off unless this run selects the live tests.

    CONCEPT: tracing-from-day-one, and the spending caps of D-41. settings.py
    copies .env into the environment, and .env has LANGSMITH_TRACING=true, so
    without this hook every scripted test on a developer's machine uploads its
    fake model and tool calls: 66 traces for one run of the scripted tests,
    measured against a counting server. On the free plan (5,000 traces a
    month) that used most of the month's traces on runs that call no real
    model. Live tests make real calls, so their traces are kept: they are the
    ones worth opening when something fails.

    pytest calls this hook once, after this file is imported and before any
    test runs, so the variable is set before LangSmith first reads it.
    Assigning os.environ directly also wins over the value load_dotenv copied
    from .env. CI sets LANGSMITH_TRACING=false itself; this hook agrees with it.
    """
    marker_expression = config.getoption("markexpr") or ""
    if not run_selects_live_tests(marker_expression):
        os.environ["LANGSMITH_TRACING"] = "false"


# One scripted reply: a dict becomes a tool call with those arguments, a str
# becomes a plain-text answer with no tool call, an exception is raised.
ScriptedReply = dict[str, Any] | str | BaseException


class ScriptedModel(BaseChatModel):
    """A chat model that answers from a script instead of an API.

    Two kinds of script. scripted_replies are given out in call order, which
    is right for calls made one after another. replies_by_prompt_text is for
    calls made at the same time (parallel-fan-out): a reply goes to the call
    whose prompt contains its key, typically a sub-question's text, whatever
    order the calls arrive in.
    """

    scripted_replies: list[ScriptedReply] = Field(default_factory=list)
    # CONCEPT: parallel-fan-out. Parallel branches call the model at the same
    # time, so which branch's call arrives first is not fixed. Replies in call
    # order would then go to the wrong branch on some runs: a flaky test.
    # Keying a reply to text in its prompt makes the script order-free.
    replies_by_prompt_text: dict[str, list[ScriptedReply]] = Field(default_factory=dict)
    received_conversations: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_name: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        """Remember the output schema's name so a dict reply can call it.

        with_structured_output() calls this and then parses the tool call with
        LangChain's real parser. model_copy() is shallow, so the copy shares
        the same reply list and the same record of conversations: the test
        scripts one object and the gateway consumes the same replies.
        """
        tool_name = convert_to_openai_tool(tools[0])["function"]["name"]
        return self.model_copy(update={"bound_tool_name": tool_name})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Record first, so a test can read the prompt even when the reply is
        # an error.
        self.received_conversations.append(list(messages))
        next_reply = self.next_reply_for(messages)
        if isinstance(next_reply, BaseException):
            raise next_reply
        if isinstance(next_reply, str):
            reply_message = AIMessage(content=next_reply)
        else:
            call_number = len(self.received_conversations)
            reply_message = AIMessage(
                content="",
                tool_calls=[
                    {"name": self.bound_tool_name, "args": next_reply, "id": f"call_{call_number}"}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply_message)])

    def next_reply_for(self, messages: list[BaseMessage]) -> ScriptedReply:
        """The keyed reply whose text is in the prompt, else the next in call order.

        Strict on purpose: an extra model call is a bug (a loop, or a cap not
        respected), and a prompt that matches two keys is a test that cannot
        say which reply it meant; both fail the test instead of guessing.
        """
        prompt_text = "\n".join(str(message.content) for message in messages)
        matching_keys = [key for key in self.replies_by_prompt_text if key in prompt_text]
        if len(matching_keys) > 1:
            raise AssertionError(f"ScriptedModel: the prompt matches several keys {matching_keys}")
        if matching_keys:
            keyed_replies = self.replies_by_prompt_text[matching_keys[0]]
            if not keyed_replies:
                raise AssertionError(f"ScriptedModel ran out of replies for {matching_keys[0]!r}")
            return keyed_replies.pop(0)
        if not self.scripted_replies:
            raise AssertionError(
                f"ScriptedModel ran out of replies after "
                f"{len(self.received_conversations) - 1} calls"
            )
        return self.scripted_replies.pop(0)


def http_error(status_code: int) -> httpx.HTTPStatusError:
    """The error ChatMistralAI raises for an HTTP error response."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"Error response {status_code}", request=request, response=response
    )


def rate_limit_error() -> httpx.HTTPStatusError:
    """A 429, which the gateway should retry."""
    return http_error(429)


# The length of a ScriptedEmbeddings vector. mistral-embed's are 1,024 long;
# eight is enough for tests and keeps failure messages readable.
SCRIPTED_VECTOR_DIMENSIONS = 8


def scripted_vector(text: str) -> list[float]:
    """A vector that depends only on the text: the same text, the same vector.

    Built from the text's SHA-256, so tests can predict it without scripting
    it, and different texts almost never share one.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[position] / 255 for position in range(SCRIPTED_VECTOR_DIMENSIONS)]


class ScriptedEmbeddings(Embeddings):
    """An embedding model that answers without the network.

    CONCEPT: scripted-model-testing, for the embed tier (document-ingestion).
    Every request (one call of aembed_documents) is recorded as the batch it
    received. A batch holding a text in failures_by_text raises that text's
    exception on every attempt: a permanent failure tied to one batch, which
    stays the same whichever order parallel batches arrive in (D-49).
    Otherwise a request takes the next scripted failure, if any is left: an
    exception is raised, None means "answer normally"; this list is taken in
    arrival order, so use it for a single batch. Otherwise it returns
    scripted_vector(text) for each text, or the vector given in
    vectors_by_text, so search tests can place texts where they want.
    """

    def __init__(
        self,
        scripted_failures: list[Exception | None] | None = None,
        vectors_by_text: dict[str, list[float]] | None = None,
        failures_by_text: dict[str, Exception] | None = None,
    ) -> None:
        self.scripted_failures = list(scripted_failures or [])
        self.vectors_by_text = vectors_by_text or {}
        self.failures_by_text = failures_by_text or {}
        self.received_batches: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.received_batches.append(list(texts))
        for text in texts:
            if text in self.failures_by_text:
                raise self.failures_by_text[text]
        if self.scripted_failures:
            scripted_failure = self.scripted_failures.pop(0)
            if scripted_failure is not None:
                raise scripted_failure
        return [self.vectors_by_text.get(text) or scripted_vector(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the gateway embeds asynchronously; use aembed_documents")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("the gateway embeds asynchronously; use aembed_documents")


@pytest.fixture
def make_gateway():
    """Build a real ModelGateway around scripted models.

    No throttle wait and no retry wait, so retry tests run in milliseconds.
    A tier the test does not script gets an empty ScriptedModel: a call to the
    wrong tier fails loudly instead of reaching the real API. The embed tier
    always gets a ScriptedEmbeddings, so no test can embed over the network.
    """

    def build_gateway(
        decision_model: ScriptedModel | None = None,
        writer_model: ScriptedModel | None = None,
        embedding_model: ScriptedEmbeddings | None = None,
        embed_batch_size: int | None = None,
    ) -> ModelGateway:
        return ModelGateway(
            decision_model=decision_model or ScriptedModel(),
            writer_model=writer_model or ScriptedModel(),
            requests_per_second=1000,
            retry_wait_seconds=0,
            embedding_model=embedding_model or ScriptedEmbeddings(),
            embed_batch_size=embed_batch_size,
        )

    return build_gateway


# A scripted answer for FakeToolbox: the result, or an exception to raise
# (ToolCallError for one failed call, SearchServerUnavailable for a dead server).
ScriptedHits = list[SearchHit] | Exception
ScriptedPage = WebPage | Exception


class FakeToolbox:
    """A Toolbox that answers from dicts instead of search-mcp.

    CONCEPT: mcp-client, decision 9. It fits the same Toolbox interface as
    McpToolbox, so a node test passes it to build_graph with no patching. It
    behaves like the real one where nodes can tell: a query with no entry finds
    nothing, a url with no entry is a page that cannot be read (ToolCallError),
    and text is cut to max_chars with truncated set. Every call is recorded.
    """

    def __init__(
        self,
        hits_by_query: dict[str, ScriptedHits] | None = None,
        pages_by_url: dict[str, ScriptedPage] | None = None,
    ) -> None:
        self.hits_by_query = hits_by_query or {}
        self.pages_by_url = pages_by_url or {}
        self.received_queries: list[str] = []
        self.received_urls: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        self.received_queries.append(query)
        scripted_hits = self.hits_by_query.get(query, [])
        if isinstance(scripted_hits, Exception):
            raise scripted_hits
        return scripted_hits[:max_results]

    async def fetch(self, url: str, max_chars: int = 6_000) -> WebPage:
        self.received_urls.append(url)
        scripted_page = self.pages_by_url.get(url)
        if scripted_page is None:
            raise ToolCallError(f"no readable text on {url}")
        if isinstance(scripted_page, Exception):
            raise scripted_page
        return scripted_page.model_copy(
            update={
                "text": scripted_page.text[:max_chars],
                "truncated": scripted_page.truncated or len(scripted_page.text) > max_chars,
            }
        )


def search_hit(page_number: int, snippet: str = "") -> SearchHit:
    """One search hit, as McpToolbox returns it."""
    return SearchHit(
        title=f"Page {page_number}",
        url=f"https://example.com/page-{page_number}",
        snippet=snippet or f"Snippet of page {page_number}.",
    )


def web_page(page_number: int, text: str = "") -> WebPage:
    """One readable page, as McpToolbox returns it."""
    return WebPage(
        url=f"https://example.com/page-{page_number}",
        title=f"Page {page_number}",
        text=text or f"Main text of page {page_number}. " * 10,
        truncated=False,
    )
