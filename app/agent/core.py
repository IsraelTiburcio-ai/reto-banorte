"""Provider-neutral orchestration for grounded CV-agent turns."""

from __future__ import annotations

from app.agent.policy import DEFAULT_AGENT_POLICY
from app.models.agent import AgentEvidence, AgentPolicy, PreparedAgentTurn
from app.services.profile_service import ProfileService


class AgentInputError(ValueError):
    """Raised when a user query cannot be prepared safely."""


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
        profile_service: ProfileService | None = None,
        policy: AgentPolicy = DEFAULT_AGENT_POLICY,
    ) -> None:
        self._profile_service = (
            profile_service if profile_service is not None else ProfileService()
        )
        self._policy = policy

    @property
    def policy(self) -> AgentPolicy:
        """Expose the immutable behavior contract for adapters and tests."""

        return self._policy

    @property
    def is_ready(self) -> bool:
        """Report whether the initialized local retrieval boundary is usable."""

        return isinstance(self._profile_service, ProfileService) and isinstance(
            self._policy, AgentPolicy
        )

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
        results = self._profile_service.search(normalized_query, visibility="public")
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
