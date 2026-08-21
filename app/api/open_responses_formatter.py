"""Deterministic temporary formatting for Phase 5 responses."""

from __future__ import annotations

import json

from app.models.agent import PreparedAgentTurn


INSUFFICIENT_EVIDENCE_TEXT = (
    "No encontré evidencia pública suficiente en el perfil para responder esa pregunta."
)


class DeterministicResponseFormatter:
    """Render only the evidence already prepared by ``AgentCore``.

    This formatter is deliberately small and has no access to retrieval, the
    canonical profile, a transcript, or a model provider. Phase 6 can replace
    it without changing the HTTP contract or the retrieval boundary.
    """

    @staticmethod
    def format(turn: PreparedAgentTurn) -> str:
        if not turn.evidence:
            return INSUFFICIENT_EVIDENCE_TEXT

        lines: list[str] = []
        for evidence in turn.evidence:
            evidence_payload = {
                "data": evidence.data,
                "entity_id": evidence.entity_id,
                "entity_type": evidence.entity_type,
                "matched_fields": list(evidence.matched_fields),
                "score": evidence.score,
                "title": evidence.title,
            }
            serialized = json.dumps(
                evidence_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ": "),
            )
            lines.append(f"- {evidence.title}\n  {serialized}")
        return "\n".join(lines)
