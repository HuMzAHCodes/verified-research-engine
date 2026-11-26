"""Print Mistral's rate limits as its own replies report them.

CONCEPT: shared-throttle, D-55. Mistral's limits are read from the
`x-ratelimit-*` headers of its replies, never from the console's Limits page,
which proved wrong in Phase 4. The gateway logs these headers only when a
request is refused; this script asks for them at the start of each phase
(FREQUENT_CHECKS.md), with one tiny request to each model the project uses:
the chat model of both tiers and the embedding model.

Run from the project root:

    uv run python scripts/check_rate_limits.py

It costs one request per model (two today). It prints each model's status, its limit headers, and,
where a per-minute request limit is reported, the setting that keeps the
throttle at 80% of it (D-43). The key is read from `.env` through settings and
is never printed.
"""

import sys
from collections.abc import Mapping
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.llm import RATE_LIMIT_HEADER_PREFIX  # noqa: E402
from settings import settings  # noqa: E402

MISTRAL_API_URL = "https://api.mistral.ai/v1"
REQUEST_TIMEOUT_SECONDS = 30
# The throttle runs at this share of a provider's limit (D-43).
THROTTLE_SHARE_OF_LIMIT = 0.8
REQUESTS_PER_MINUTE_HEADER = "limit-req-minute"


def rate_limit_headers(response_headers: Mapping[str, str]) -> dict[str, str]:
    """The reply's rate-limit headers, without their common prefix, sorted by name."""
    return {
        header_name.lower().removeprefix(RATE_LIMIT_HEADER_PREFIX): header_value
        for header_name, header_value in sorted(response_headers.items())
        if header_name.lower().startswith(RATE_LIMIT_HEADER_PREFIX)
    }


def throttle_setting(reported_limits: Mapping[str, str]) -> float | None:
    """Requests a second at 80% of the reported per-minute limit; None if none is reported."""
    per_minute = reported_limits.get(REQUESTS_PER_MINUTE_HEADER, "")
    try:
        requests_per_minute = float(per_minute)
    except ValueError:
        return None
    if requests_per_minute <= 0:
        return None
    return round(requests_per_minute / 60 * THROTTLE_SHARE_OF_LIMIT, 2)


def describe_reply(label: str, model_name: str, response: httpx.Response) -> list[str]:
    """What to print for one model: its status, its limits, and the throttle they allow."""
    reported_limits = rate_limit_headers(response.headers)
    lines = [f"{label} ({model_name}): HTTP {response.status_code}"]
    if not reported_limits:
        lines.append("  no rate-limit headers in the reply")
    lines += [f"  {name} = {value}" for name, value in reported_limits.items()]
    suggested = throttle_setting(reported_limits)
    if suggested is not None:
        lines.append(f"  80% of the per-minute limit: {suggested} requests a second")
    if response.status_code >= 400:
        # The provider's message names the problem; it never holds the key.
        lines.append(f"  provider says: {response.text[:200]}")
    return lines


def checks_to_run(
    chat_model: str, writer_model: str, embed_model: str
) -> list[tuple[str, str, dict]]:
    """One tiny request per model the project uses: (label, API path, body).

    Both tiers name the same chat model today (D-55), so it is asked once.
    """
    labels_by_chat_model: dict[str, list[str]] = {}
    for tier, model_name in (("decision tier", chat_model), ("writer tier", writer_model)):
        labels_by_chat_model.setdefault(model_name, []).append(tier)
    checks = [
        (
            f"chat, {' and '.join(tiers)}",
            "/chat/completions",
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "Reply with the word OK."}],
                "max_tokens": 3,
            },
        )
        for model_name, tiers in labels_by_chat_model.items()
    ]
    checks.append(("embeddings", "/embeddings", {"model": embed_model, "input": ["OK"]}))
    return checks


def run_checks(
    client: httpx.Client, api_key: str, checks: list[tuple[str, str, dict]]
) -> list[str]:
    """Send each check and describe its reply. Tests pass a client with a fake transport."""
    authorization = {"Authorization": f"Bearer {api_key}"}
    lines: list[str] = []
    for label, path, body in checks:
        response = client.post(path, headers=authorization, json=body)
        lines += describe_reply(label, body["model"], response)
    return lines


def main() -> None:
    checks = checks_to_run(
        settings.vre_chat_model, settings.vre_writer_model, settings.vre_embed_model
    )
    with httpx.Client(base_url=MISTRAL_API_URL, timeout=REQUEST_TIMEOUT_SECONDS) as client:
        lines = run_checks(client, settings.mistral_api_key.get_secret_value(), checks)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
