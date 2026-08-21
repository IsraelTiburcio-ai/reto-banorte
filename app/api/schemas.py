"""HTTP request/response schemas for Phase 4."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.core import AgentCore
from app.models.agent import AgentEvidence, AgentPolicy, PreparedAgentTurn


class StrictApiModel(BaseModel):
    """Base schema that rejects undocumented request/response fields."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictApiModel):
    status: Literal["ok"] = "ok"


class AgentPrepareRequest(StrictApiModel):
    query: str = Field(strict=True)
    max_results: int = Field(
        default=AgentCore.DEFAULT_MAX_RESULTS,
        ge=1,
        le=AgentCore.MAX_RESULTS,
        strict=True,
    )


class AgentPolicyResponse(StrictApiModel):
    name: str
    objective: str
    rules: list[str]

    @classmethod
    def from_policy(cls, policy: AgentPolicy) -> "AgentPolicyResponse":
        return cls(
            name=policy.name,
            objective=policy.objective,
            rules=list(policy.rules),
        )


class AgentEvidenceResponse(StrictApiModel):
    entity_type: str
    entity_id: str
    title: str
    score: float
    matched_fields: list[str]
    data: dict[str, object]

    @classmethod
    def from_evidence(cls, evidence: AgentEvidence) -> "AgentEvidenceResponse":
        return cls(
            entity_type=evidence.entity_type,
            entity_id=evidence.entity_id,
            title=evidence.title,
            score=evidence.score,
            matched_fields=list(evidence.matched_fields),
            data=evidence.data,
        )


class AgentPrepareResponse(StrictApiModel):
    query: str
    status: Literal["ready", "insufficient_evidence"]
    evidence: list[AgentEvidenceResponse]
    policy: AgentPolicyResponse

    @classmethod
    def from_turn(cls, turn: PreparedAgentTurn) -> "AgentPrepareResponse":
        return cls(
            query=turn.query,
            status=turn.status,
            evidence=[AgentEvidenceResponse.from_evidence(item) for item in turn.evidence],
            policy=AgentPolicyResponse.from_policy(turn.policy),
        )
