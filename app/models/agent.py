"""Provider-neutral models for the CV agent core."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

from app.models.retrieval import SearchResult


AgentPreparationStatus = Literal["ready", "insufficient_evidence"]


@dataclass(frozen=True)
class AgentPolicy:
    """Behavior contract that future model adapters must preserve."""

    name: str
    objective: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class AgentEvidence:
    """One public, explainable evidence item prepared for a future model call."""

    entity_type: str
    entity_id: str
    title: str
    score: float
    matched_fields: tuple[str, ...]
    data: dict[str, object]

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "AgentEvidence":
        """Detach evidence from retrieval results to avoid shared mutable state."""

        return cls(
            entity_type=result.entity_type,
            entity_id=result.entity_id,
            title=result.title,
            score=result.score,
            matched_fields=tuple(result.matched_fields),
            data=copy.deepcopy(result.data),
        )


@dataclass(frozen=True)
class PreparedAgentTurn:
    """Deterministic input package for the future LLM adapter."""

    query: str
    status: AgentPreparationStatus
    evidence: tuple[AgentEvidence, ...]
    policy: AgentPolicy
