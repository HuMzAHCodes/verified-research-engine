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
    numeric_host_address,
    resolve_host,
)


def fake_resolver(addresses_by_host: dict[str, list[str]]):
    async def resolve(host_name: str) -> list[str]:
        return addresses_by_host.get(host_name, [])

    return resolve


PUBLIC_SITE = fake_resolver({"example.com": ["93.184.216.34"]})


@pytest.mark.parametrize(
    "ip_text",
    [
        "93.184.216.34",
        "8.8.8.8",
        "2606:4700:4700::1111",
        "64:ff9b::808:808",  # NAT64 to a public address (8.8.8.8): how IPv6-only networks reach it
    ],
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
        # An IPv4 address carried inside an IPv6 one (outside test review, F-8).
        ("64:ff9b::a00:1", "NAT64 to 10.0.0.1"),
        ("64:ff9b::a9fe:a9fe", "NAT64 to 169.254.169.254, the cloud metadata address"),
        ("::a00:1", "IPv4-compatible 10.0.0.1"),
        ("::ffff:0:a00:1", "IPv4-translated 10.0.0.1"),
        ("2002:a00:1::1", "6to4 around 10.0.0.1"),
        ("64:ff9b:1::a00:1", "NAT64, local-use prefix"),
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


@pytest.mark.parametrize("host_name", ["2130706433", "0x7f.1", "127.1", "0177.0.0.1", "0x7f000001"])
def test_a_host_written_as_a_number_is_read_as_its_address(host_name):
    # CONCEPT: building-search-mcp, outside test review F-8: 127.0.0.1 spelled
    # the old ways, read the same on every system rather than by its resolver.
    assert numeric_host_address(host_name) == "127.0.0.1"


@pytest.mark.parametrize("host_name", ["example.com", "1e100.net", "cafe.be"])
def test_a_host_name_is_not_read_as_a_number(host_name):
    assert numeric_host_address(host_name) is None


@pytest.mark.parametrize("url", ["http://2130706433/", "http://0x7f.1/", "http://127.1:8080/"])
async def test_loopback_spelled_as_a_number_is_refused_without_a_resolver(url):
    # PUBLIC_SITE knows only example.com: the refusal cannot come from it.
    with pytest.raises(BlockedAddressError, match="127.0.0.1"):
        await ensure_url_is_safe(url, PUBLIC_SITE)


async def test_a_number_that_is_not_an_address_is_refused():
    with pytest.raises(BlockedAddressError, match="not a valid address"):
        await ensure_url_is_safe("http://1.2.3.4.5/", PUBLIC_SITE)


async def test_a_name_the_resolver_cannot_encode_is_refused():
    # CONCEPT: building-search-mcp, outside test review F-10. A label over 63
    # characters makes the system resolver raise UnicodeError, not OSError;
    # it fails before any DNS request is sent.
    with pytest.raises(BlockedAddressError, match="could not be resolved"):
        await ensure_url_is_safe("http://" + "a" * 64 + ".com/", resolve_host)


async def test_a_resolver_error_is_a_refusal():
    async def failing_resolver(host_name: str) -> list[str]:
        raise OSError("name or service not known")

    with pytest.raises(BlockedAddressError, match="could not be resolved"):
        await ensure_url_is_safe("https://example.com/", failing_resolver)
