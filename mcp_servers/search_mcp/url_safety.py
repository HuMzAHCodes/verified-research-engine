"""Which addresses fetch_page may download from: the server's security boundary.

CONCEPT: building-search-mcp, decision 4. A tool that fetches any URL it is
given is the classic entry point for server-side request forgery (SSRF): once
the server is public (Phase 11), anyone could ask it for addresses only the
server can reach, such as services on its own machine, the private network
it sits in, or a cloud metadata address that hands out credentials.

The rule: http and https only, and every address the host name resolves to
must be a public internet address. fetch.py calls this again for every
redirect, so a public page cannot bounce the server to a private one.

Two ways an address can hide (found by the outside test review at the Phase 4
close, F-8 and F-10): an IPv6 address that carries an IPv4 one inside it, such
as NAT64's 64:ff9b::a9fe:a9fe for 169.254.169.254, is judged by the IPv4
address it carries; and a host written as a number in an old form, such as
2130706433 or 0x7f.1 for 127.0.0.1, is read here rather than left to how each
system's resolver happens to read it.

Known limit, for Phase 11: a hostile DNS server can answer differently
between this check and the download (DNS rebinding). Closing that means
connecting to the checked address itself.
"""

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})

# IPv6 ranges whose last 32 bits are an IPv4 address that traffic goes on to.
# Python already treats 6to4 (2002::/16) and Teredo (2001::/32) as not public.
IPV4_CARRYING_NETWORKS = (
    ipaddress.IPv6Network("64:ff9b::/96"),  # NAT64, the well-known prefix
    ipaddress.IPv6Network("::ffff:0:0:0/96"),  # IPv4-translated
    ipaddress.IPv6Network("::/96"),  # IPv4-compatible (deprecated)
)
# A host whose last label is a number is an IPv4 address in some form: real
# top-level domains never start with a digit.
NUMERIC_LAST_LABEL = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)", re.IGNORECASE)

# Turns a host name into its IP addresses. Injected in tests, so no test
# depends on real DNS.
HostResolver = Callable[[str], Awaitable[list[str]]]


class BlockedAddressError(ValueError):
    """The URL must not be fetched."""


async def resolve_host(host_name: str) -> list[str]:
    """Every IP address the host name resolves to, using the system resolver."""
    event_loop = asyncio.get_running_loop()
    address_records = await event_loop.getaddrinfo(host_name, None, type=socket.SOCK_STREAM)
    return sorted({record[4][0] for record in address_records})


def carried_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address carries and leads to, if it carries one."""
    if ip.ipv4_mapped:
        return ip.ipv4_mapped  # ::ffff:10.0.0.1 is really 10.0.0.1
    for network in IPV4_CARRYING_NETWORKS:
        if ip in network:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def is_public_address(ip_text: str) -> bool:
    """True only for an address on the public internet, and any IPv4 one it carries."""
    ip = ipaddress.ip_address(ip_text.split("%")[0])  # drop an IPv6 zone such as %eth0
    if isinstance(ip, ipaddress.IPv6Address):
        ipv4_inside = carried_ipv4(ip)
        if ipv4_inside is not None and not is_public_address(str(ipv4_inside)):
            return False
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def numeric_host_address(host_name: str) -> str | None:
    """The IPv4 address a host written as a number stands for, such as 0x7f.1.

    None for a host name. A numeric host that is not a valid address raises
    BlockedAddressError rather than going to a resolver to be guessed at.
    """
    if not NUMERIC_LAST_LABEL.fullmatch(host_name.rsplit(".", 1)[-1]):
        return None
    try:
        return str(ipaddress.IPv4Address(socket.inet_aton(host_name)))
    except OSError as error:
        raise BlockedAddressError(
            f"blocked address: {host_name} is a number but not a valid address"
        ) from error


async def ensure_url_is_safe(url: str, resolver: HostResolver = resolve_host) -> None:
    """Raise BlockedAddressError unless the URL is http(s) and every address is public."""
    url_parts = urlsplit(url)
    if url_parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise BlockedAddressError(
            f"blocked address: scheme {url_parts.scheme or 'none'!r} is not http or https"
        )
    host_name = url_parts.hostname
    if not host_name:
        raise BlockedAddressError("blocked address: the URL has no host")

    numeric_address = numeric_host_address(host_name)
    try:
        resolved_addresses = (
            [numeric_address] if numeric_address is not None else await resolver(host_name)
        )
    except (OSError, UnicodeError) as error:
        # UnicodeError: a name the resolver cannot even encode, such as a label
        # over 63 characters. It is not an OSError, so it is named here.
        raise BlockedAddressError(f"blocked address: {host_name} could not be resolved") from error
    if not resolved_addresses:
        raise BlockedAddressError(f"blocked address: {host_name} resolved to no address")

    # Every address must be public: a name that resolves to one public and one
    # private address could be connected to either.
    for resolved_address in resolved_addresses:
        if not is_public_address(resolved_address):
            raise BlockedAddressError(
                f"blocked address: {host_name} resolves to {resolved_address}, "
                "which is not a public internet address"
            )
