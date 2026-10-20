"""Tests of the address check: the search server's security boundary.

CONCEPT: building-search-mcp, decision 4. Each refused address is a way a
public fetch tool could be used to reach something private. Host names are
resolved by a fake resolver, so no test depends on real DNS.
"""

import pytest

from mcp_servers.search_mcp.url_safety import (
    BlockedAddressError,
    ensure_url_is_safe,
    is_public_address,
)


def fake_resolver(addresses_by_host: dict[str, list[str]]):
    async def resolve(host_name: str) -> list[str]:
        return addresses_by_host.get(host_name, [])

    return resolve


PUBLIC_SITE = fake_resolver({"example.com": ["93.184.216.34"]})


@pytest.mark.parametrize(
    "ip_text",
    ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"],
)
def test_public_addresses_are_allowed(ip_text):
    assert is_public_address(ip_text)


@pytest.mark.parametrize(
    "ip_text, kind",
    [
        ("127.0.0.1", "loopback"),
        ("::1", "loopback, IPv6"),
        ("10.0.0.5", "private network"),
        ("172.16.4.2", "private network"),
        ("192.168.1.1", "private network"),
        ("169.254.169.254", "link-local: cloud metadata address"),
        ("100.64.0.1", "shared address space"),
        ("0.0.0.0", "unspecified"),
        ("224.0.0.1", "multicast"),
        ("::ffff:10.0.0.1", "private address written as IPv6"),
        ("fc00::1", "private network, IPv6"),
    ],
)
def test_non_public_addresses_are_refused(ip_text, kind):
    assert not is_public_address(ip_text), kind


async def test_a_public_https_url_passes():
    await ensure_url_is_safe("https://example.com/page", PUBLIC_SITE)


@pytest.mark.parametrize(
    "url, reason",
    [
        ("file:///etc/passwd", "scheme"),
        ("ftp://example.com/file", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("https:///no-host", "no host"),
    ],
)
async def test_unsafe_schemes_and_missing_hosts_are_refused(url, reason):
    with pytest.raises(BlockedAddressError, match=reason):
        await ensure_url_is_safe(url, PUBLIC_SITE)


async def test_a_name_that_resolves_to_a_private_address_is_refused():
    resolver = fake_resolver({"intranet.example.com": ["10.1.2.3"]})

    with pytest.raises(BlockedAddressError, match="10.1.2.3"):
        await ensure_url_is_safe("http://intranet.example.com/admin", resolver)


async def test_one_private_address_among_public_ones_is_enough_to_refuse():
    resolver = fake_resolver({"mixed.example.com": ["93.184.216.34", "127.0.0.1"]})

    with pytest.raises(BlockedAddressError, match="127.0.0.1"):
        await ensure_url_is_safe("https://mixed.example.com/", resolver)


async def test_a_name_that_does_not_resolve_is_refused():
    with pytest.raises(BlockedAddressError, match="resolved to no address"):
        await ensure_url_is_safe("https://unknown.example.com/", PUBLIC_SITE)


async def test_an_ip_address_in_the_url_is_checked_too():
    async def resolve_literally(host_name: str) -> list[str]:
        return [host_name]

    with pytest.raises(BlockedAddressError, match="127.0.0.1"):
        await ensure_url_is_safe("http://127.0.0.1:8080/", resolve_literally)
