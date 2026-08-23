"""Small, configurable inbound authentication boundary."""

from __future__ import annotations

import hmac
import os

from starlette.requests import Request


def is_agent_authorized(request: Request) -> bool:
    """Return whether the request satisfies optional ``AGENT_API_KEY`` auth.

    The server-generated API key is deliberately separate from the provider's
    ``OPENAI_API_KEY``. When no agent key is configured, local development stays
    unauthenticated.
    """

    expected = os.getenv("AGENT_API_KEY", "").strip()
    if not expected:
        return True

    authorization = request.headers.get("authorization", "")
    scheme, separator, candidate = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return False
    candidate = candidate.strip()
    if not candidate or any(character.isspace() or ord(character) < 32 for character in candidate):
        return False
    return hmac.compare_digest(candidate, expected)
