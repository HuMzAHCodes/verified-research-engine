"""The rate-limit check's pure parts: reading Mistral's limit headers.

CONCEPT: shared-throttle, D-55. scripts/check_rate_limits.py sends two real
requests, which these tests leave alone; they check how a reply's headers are
read and what is printed, with replies built here.
"""

import httpx

from scripts.check_rate_limits import (
    checks_to_run,
    describe_reply,
    rate_limit_headers,
    run_checks,
    throttle_setting,
)

REQUEST = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")


def reply(status_code: int, headers: dict[str, str], text: str = "") -> httpx.Response:
    return httpx.Response(status_code, headers=headers, text=text, request=REQUEST)


def test_only_the_rate_limit_headers_are_kept_without_their_prefix():
    headers = httpx.Headers(
        {
            "X-RateLimit-Limit-Req-Minute": "60",
            "x-ratelimit-remaining-req-minute": "59",
            "content-type": "application/json",
            "x-request-id": "abc",
        }
    )

    assert rate_limit_headers(headers) == {
        "limit-req-minute": "60",
        "remaining-req-minute": "59",
    }


def test_the_throttle_setting_is_80_percent_of_the_per_minute_limit():
    assert throttle_setting({"limit-req-minute": "60"}) == 0.8
    # Phase 4's cut: 2 requests a minute.
    assert throttle_setting({"limit-req-minute": "2"}) == 0.03


def test_no_setting_without_a_usable_per_minute_limit():
    assert throttle_setting({}) is None
    assert throttle_setting({"limit-req-minute": "unlimited"}) is None
    assert throttle_setting({"limit-req-minute": "0"}) is None


def test_a_reply_is_described_with_its_limits_and_setting():
    lines = describe_reply(
        "chat, both tiers",
        "voxtral-small-2507",
        reply(
            200, {"x-ratelimit-limit-req-minute": "60", "x-ratelimit-remaining-req-minute": "58"}
        ),
    )

    assert lines == [
        "chat, both tiers (voxtral-small-2507): HTTP 200",
        "  limit-req-minute = 60",
        "  remaining-req-minute = 58",
        "  80% of the per-minute limit: 0.8 requests a second",
    ]


def test_a_refused_reply_shows_the_providers_message():
    lines = describe_reply(
        "embeddings", "mistral-embed", reply(401, {}, text='{"message": "Unauthorized"}')
    )

    assert lines == [
        "embeddings (mistral-embed): HTTP 401",
        "  no rate-limit headers in the reply",
        '  provider says: {"message": "Unauthorized"}',
    ]


def test_a_chat_model_both_tiers_share_is_asked_once():
    checks = checks_to_run("voxtral-small-2507", "voxtral-small-2507", "mistral-embed")

    assert [(label, path) for label, path, _ in checks] == [
        ("chat, decision tier and writer tier", "/chat/completions"),
        ("embeddings", "/embeddings"),
    ]
    assert checks[0][2]["model"] == "voxtral-small-2507"
    assert checks[1][2] == {"model": "mistral-embed", "input": ["OK"]}


def test_two_chat_models_are_each_asked():
    checks = checks_to_run("model-a", "model-b", "mistral-embed")

    assert [label for label, _, _ in checks] == [
        "chat, decision tier",
        "chat, writer tier",
        "embeddings",
    ]


def test_every_check_is_sent_with_the_key_and_the_key_is_never_printed():
    # Built at run time so this file holds no key-shaped string (TC-NFR-09).
    fake_key = "k" * 32
    sent_requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(200, headers={"x-ratelimit-limit-req-minute": "60"})

    with httpx.Client(
        base_url="https://api.mistral.ai/v1", transport=httpx.MockTransport(answer)
    ) as client:
        lines = run_checks(client, fake_key, checks_to_run("m", "m", "mistral-embed"))

    assert [request.url.path for request in sent_requests] == [
        "/v1/chat/completions",
        "/v1/embeddings",
    ]
    assert all(
        request.headers["authorization"] == f"Bearer {fake_key}" for request in sent_requests
    )
    assert not any(fake_key in line for line in lines)
    assert "chat, decision tier and writer tier (m): HTTP 200" in lines
