"""Tests of fetch_page's download and text extraction.

CONCEPT: building-search-mcp, decisions 4 to 6, and choice 2. A fake HTTP
transport plays the websites and a fake resolver plays DNS, so these tests
cover redirects, limits, content types, and text extraction with no network.
"""

import httpx
import pytest

from mcp_servers.search_mcp.fetch import FetchLimits, PageFetchError, fetch_readable_page

ARTICLE_PARAGRAPH = (
    "Symbotic designs and builds automated warehouse systems that move, store, and sort "
    "cases for large retailers and wholesale distributors across North America. "
)
ARTICLE_HTML = f"""<html><head><title>Symbotic overview</title></head><body>
<nav>Home | Products | Careers | Contact</nav>
<article><h1>Symbotic overview</h1>
<p>{ARTICLE_PARAGRAPH * 3}</p>
<p>{ARTICLE_PARAGRAPH * 3}</p>
</article>
<footer>Copyright notice. Cookie settings. Privacy.</footer>
</body></html>"""


async def resolve_to_public_or_private(host_name: str) -> list[str]:
    return ["10.0.0.7"] if host_name.startswith("internal") else ["93.184.216.34"]


def fake_web(responses_by_url: dict[str, httpx.Response]) -> httpx.MockTransport:
    def answer(request: httpx.Request) -> httpx.Response:
        return responses_by_url.get(str(request.url), httpx.Response(404))

    return httpx.MockTransport(answer)


def html_page(html: str = ARTICLE_HTML) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html)


async def fetch(url, responses_by_url, max_chars=6_000, limits=None):
    return await fetch_readable_page(
        url,
        max_chars,
        limits=limits or FetchLimits(),
        resolver=resolve_to_public_or_private,
        transport=fake_web(responses_by_url),
    )


async def test_main_text_is_kept_and_menus_and_footers_are_dropped():
    fetched_page = await fetch("https://example.com/a", {"https://example.com/a": html_page()})

    assert "automated warehouse systems" in fetched_page.text
    assert "Cookie settings" not in fetched_page.text
    assert "Careers" not in fetched_page.text
    assert fetched_page.title == "Symbotic overview"
    assert fetched_page.truncated is False


async def test_text_is_cut_at_max_chars_and_marked_truncated():
    fetched_page = await fetch(
        "https://example.com/a", {"https://example.com/a": html_page()}, max_chars=200
    )

    assert len(fetched_page.text) == 200
    assert fetched_page.truncated is True


async def test_plain_text_pages_are_read_as_they_are():
    plain_response = httpx.Response(
        200, headers={"content-type": "text/plain"}, text="Plain text answer."
    )

    fetched_page = await fetch(
        "https://example.com/t.txt", {"https://example.com/t.txt": plain_response}
    )

    assert fetched_page.text == "Plain text answer."


async def test_a_redirect_is_followed_and_the_final_url_reported():
    redirect = httpx.Response(301, headers={"location": "/final"})
    fetched_page = await fetch(
        "https://example.com/start",
        {"https://example.com/start": redirect, "https://example.com/final": html_page()},
    )

    assert fetched_page.url == "https://example.com/final"


async def test_a_redirect_to_a_private_address_is_refused():
    redirect = httpx.Response(302, headers={"location": "http://internal.example.com/admin"})

    with pytest.raises(PageFetchError, match="blocked address"):
        await fetch("https://example.com/start", {"https://example.com/start": redirect})


async def test_too_many_redirects_fail():
    loop_back = httpx.Response(302, headers={"location": "https://example.com/start"})

    with pytest.raises(PageFetchError, match="more than 3 redirects"):
        await fetch("https://example.com/start", {"https://example.com/start": loop_back})


async def test_an_http_error_status_fails_with_the_status():
    with pytest.raises(PageFetchError, match="HTTP 403"):
        await fetch("https://example.com/a", {"https://example.com/a": httpx.Response(403)})


async def test_a_non_text_page_is_refused():
    pdf_response = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7"
    )

    with pytest.raises(PageFetchError, match="not a text page"):
        await fetch("https://example.com/r.pdf", {"https://example.com/r.pdf": pdf_response})


async def test_a_page_with_only_a_menu_fails_as_no_readable_text():
    # trafilatura returns "Menu" here: a fallback, not an article.
    empty_page = html_page("<html><body><nav>Menu</nav></body></html>")

    with pytest.raises(PageFetchError, match="no readable text"):
        await fetch("https://example.com/a", {"https://example.com/a": empty_page})


async def test_a_page_with_only_leftovers_fails_as_no_readable_text():
    # A video page: a title, two lines of description, and footer links.
    # trafilatura keeps 135 characters here, enough to pass the old 80-character
    # minimum (a live-run video page passed it with 160) but not the new 300.
    video_page = html_page(
        "<html><head><title>Symbotic tour</title></head><body>"
        "<h1>Symbotic warehouse tour</h1>"
        "<p>A short look inside an automated warehouse, filmed during a customer visit.</p>"
        "<p>Subscribe for more videos about robotics and supply chains.</p>"
        "<footer>About Press Copyright Contact us Creators Advertise Terms Privacy</footer>"
        "</body></html>"
    )

    with pytest.raises(PageFetchError, match="no readable text"):
        await fetch("https://example.com/v", {"https://example.com/v": video_page})


async def test_downloads_stop_at_the_size_limit():
    long_text = "word " * 50_000  # 250,000 bytes
    long_response = httpx.Response(200, headers={"content-type": "text/plain"}, text=long_text)

    fetched_page = await fetch(
        "https://example.com/long",
        {"https://example.com/long": long_response},
        max_chars=20_000,
        limits=FetchLimits(max_bytes=10_000),
    )

    assert len(fetched_page.text) <= 10_000
