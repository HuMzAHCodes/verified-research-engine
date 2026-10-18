"""Shared test helpers: a scripted chat model, a gateway around it, a fake search.

CONCEPT: scripted-model-testing. Tests replace the model, never the code that
calls it. ScriptedModel returns replies written in advance, in order, and
records every conversation it was sent. A real ModelGateway wraps it, so the
gateway's parsing, retries, and repair message all run in every test, while
no request leaves the machine. FakeSearchClient does the same for web search.
pytest loads this file automatically for every test under tests/.
"""

from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from graph.llm import ModelGateway

# One scripted reply: a dict becomes a tool call with those arguments, a str
# becomes a plain-text answer with no tool call, an exception is raised.
ScriptedReply = dict[str, Any] | str | BaseException


class ScriptedModel(BaseChatModel):
    """A chat model that answers from a script instead of an API."""

    scripted_replies: list[ScriptedReply] = Field(default_factory=list)
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
        if not self.scripted_replies:
            # Strict on purpose: an extra model call is a bug (a loop, or a
            # cap not respected), not something to answer with a default.
            raise AssertionError(
                f"ScriptedModel ran out of replies after "
                f"{len(self.received_conversations) - 1} calls"
            )

        next_reply = self.scripted_replies.pop(0)
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


@pytest.fixture
def make_gateway():
    """Build a real ModelGateway around scripted models.

    No throttle wait and no retry wait, so retry tests run in milliseconds.
    A tier the test does not script gets an empty ScriptedModel: a call to the
    wrong tier fails loudly instead of reaching the real API.
    """

    def build_gateway(
        decision_model: ScriptedModel | None = None,
        writer_model: ScriptedModel | None = None,
    ) -> ModelGateway:
        return ModelGateway(
            decision_model=decision_model or ScriptedModel(),
            writer_model=writer_model or ScriptedModel(),
            requests_per_second=1000,
            retry_wait_seconds=0,
        )

    return build_gateway


class FakeSearchClient:
    """A web search that answers from a dict instead of Tavily.

    CONCEPT: scripted-model-testing, applied to tools. Same idea as
    ScriptedModel: the node under test is real, only the outside world is
    fake. A query with no entry returns no results. Every query is recorded.
    """

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.results_by_query = results_by_query or {}
        self.received_queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        self.received_queries.append(query)
        return self.results_by_query.get(query, [])[:max_results]


def search_result(page_number: int, content: str = "") -> dict[str, Any]:
    """One search result in the shape Tavily returns."""
    return {
        "url": f"https://example.com/page-{page_number}",
        "title": f"Page {page_number}",
        "content": content or f"Text of page {page_number}.",
    }
