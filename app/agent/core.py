"""Provider-neutral orchestration for grounded CV-agent turns."""

from __future__ import annotations

from typing import Protocol

from app.agent.policy import DEFAULT_AGENT_POLICY
from app.models.agent import AgentEvidence, AgentPolicy, PreparedAgentTurn
from app.models.retrieval import SearchResult, VisibilityPolicy
from app.services.profile_service import ProfileService


class AgentInputError(ValueError):
    """Raised when a user query cannot be prepared safely."""


class ProfileRetriever(Protocol):
    """Minimal retrieval contract required by the agent core."""

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]: ...


class AgentCore:
    """Prepare grounded, public-only context without calling an LLM.

    This layer deliberately stops before natural-language generation. It owns the
    public retrieval boundary, evidence limits, and provider-neutral behavior
    policy. A later LLM adapter will consume ``PreparedAgentTurn``.
    """

    DEFAULT_MAX_RESULTS = 8
    MAX_RESULTS = 20

    def __init__(
        self,
        retriever: ProfileRetriever | None = None,
        policy: AgentPolicy = DEFAULT_AGENT_POLICY,
    ) -> None:
        self._retriever = retriever if retriever is not None else ProfileService()
        self._policy = policy

    @property
    def policy(self) -> AgentPolicy:
        """Expose the immutable behavior contract for adapters and tests."""

        return self._policy

    def prepare(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> PreparedAgentTurn:
        """Validate a query and package public evidence for a future model call."""

        normalized_query = self._normalize_query(query)
        self._validate_max_results(max_results)

        # The agent boundary is intentionally public-only. Callers cannot elevate
        # retrieval visibility through this API.
        results = self._retriever.search(normalized_query, visibility="public")
        selected_results = results[:max_results]
        evidence = tuple(
            AgentEvidence.from_search_result(result) for result in selected_results
        )
        status = "ready" if evidence else "insufficient_evidence"

        return PreparedAgentTurn(
            query=normalized_query,
            status=status,
            evidence=evidence,
            policy=self._policy,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise AgentInputError("query must be a string")
        normalized = " ".join(query.split())
        if not normalized:
            raise AgentInputError("query must not be empty")
        return normalized

    @classmethod
    def _validate_max_results(cls, max_results: int) -> None:
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise AgentInputError("max_results must be an integer")
        if not 1 <= max_results <= cls.MAX_RESULTS:
            raise AgentInputError(
                f"max_results must be between 1 and {cls.MAX_RESULTS}"
            )
