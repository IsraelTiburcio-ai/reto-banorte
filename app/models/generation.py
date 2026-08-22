"""Provider-neutral contracts for grounded text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.models.agent import AgentEvidence, AgentPolicy


@dataclass(frozen=True)
class ConversationMessage:
    """A validated user/assistant transcript item kept separate from policy."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class TextGenerationRequest:
    """The only data a text generator is allowed to receive."""

    query: str
    transcript: tuple[ConversationMessage, ...]
    evidence: tuple[AgentEvidence, ...]
    policy: AgentPolicy


class TextGenerator(Protocol):
    """Small provider-neutral interface for one grounded text generation."""

    @property
    def provider_request_attempted(self) -> bool:
        """Whether the most recent call attempted an outbound provider request."""

        ...

    def generate(self, request: TextGenerationRequest) -> str:
        """Generate text from an already prepared, public-only request."""
