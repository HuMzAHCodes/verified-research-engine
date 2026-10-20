"""Which addresses fetch_page may download from: the server's security boundary.

CONCEPT: building-search-mcp, decision 4. A tool that fetches any URL it is
given is the classic entry point for server-side request forgery (SSRF): once
the server is public (Phase 11), anyone could ask it for addresses only the
server can reach, such as services on its own machine, the private network
it sits in, or a cloud metadata address that hands out credentials.

The rule: http and https only, and every address the host name resolves to
must be a public internet address. fetch.py calls this again for every
redirect, so a public page cannot bounce the server to a private one.

Known limit, for Phase 11: a hostile DNS server can answer differently
between this check and the download (DNS rebinding). Closing that means
connecting to the checked address itself.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})

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


def is_public_address(ip_text: str) -> bool:
    """True only for an address on the public internet."""
    ip = ipaddress.ip_address(ip_text.split("%")[0])  # drop an IPv6 zone such as %eth0
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:10.0.0.1 is really 10.0.0.1
    return ip.is_global and not ip.is_multicast


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

    try:
        resolved_addresses = await resolver(host_name)
    except OSError as error:
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
