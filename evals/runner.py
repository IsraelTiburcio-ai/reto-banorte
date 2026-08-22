"""Run deterministic offline evals; live mode is explicit and cost-limited."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.agent.core import AgentCore
from app.models.agent import PreparedAgentTurn
from app.services.profile_service import ProfileService
from evals.metrics import FAIL, NOT_EVALUATED, PASS, EvalOutcome, EvalReport, render_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
VALID_CATEGORIES = frozenset(
    {
        "factuality",
        "groundedness",
        "relevance",
        "abstention",
        "ownership_calibration",
        "skill_calibration",
        "approximate_metrics",
        "professional_academic_conceptual",
        "out_of_scope",
        "prompt_injection",
        "restricted_information",
        "conversational_follow_ups",
        "natural_language_retrieval",
        "spanish",
        "english",
    }
)
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{12,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*[^\s<]", re.IGNORECASE),
)
RESTRICTED_EVIDENCE_MARKERS = (
    "internal_summary",
    "do_not_expose",
    "OPENAI_API_KEY",
)


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    input: str
    expected_behavior: str
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    required_evidence_ids: tuple[str, ...]
    top_evidence_ids: tuple[str, ...]
    top_k: int | None
    forbidden_evidence_ids: tuple[str, ...]
    should_abstain: bool
    notes: str


def _as_string_list(value: Any, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{case_id}.{field} must be a list of non-empty strings")
    return tuple(value)


def validate_cases(raw_cases: object) -> tuple[EvalCase, ...]:
    """Validate the auditable dataset and return immutable case objects."""

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases.json must contain a non-empty list")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    allowed_fields = {
        "id",
        "category",
        "input",
        "expected_behavior",
        "required_facts",
        "forbidden_claims",
        "required_evidence_ids",
        "top_evidence_ids",
        "top_k",
        "forbidden_evidence_ids",
        "should_abstain",
        "notes",
    }
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("every eval case must be an object")
        case_id = raw_case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("every eval case requires a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate eval case id: {case_id}")
        seen_ids.add(case_id)
        unknown_fields = set(raw_case) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"{case_id} contains unsupported fields: {sorted(unknown_fields)}"
            )

        category = raw_case.get("category")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"{case_id}.category is not supported: {category}")
        input_value = raw_case.get("input")
        expected_behavior = raw_case.get("expected_behavior")
        notes = raw_case.get("notes")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (input_value, expected_behavior, notes)
        ):
            raise ValueError(f"{case_id} requires non-empty text fields")
        should_abstain = raw_case.get("should_abstain")
        if not isinstance(should_abstain, bool):
            raise ValueError(f"{case_id}.should_abstain must be boolean")

        required_facts = _as_string_list(
            raw_case.get("required_facts"), "required_facts", case_id
        )
        forbidden_claims = _as_string_list(
            raw_case.get("forbidden_claims"), "forbidden_claims", case_id
        )
        required_ids = _as_string_list(
            raw_case.get("required_evidence_ids"),
            "required_evidence_ids",
            case_id,
        )
        top_ids = _as_string_list(
            raw_case.get("top_evidence_ids", []), "top_evidence_ids", case_id
        )
        top_k = raw_case.get("top_k")
        if top_ids:
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
                raise ValueError(
                    f"{case_id}.top_k must be a positive integer when top_evidence_ids is set"
                )
            if top_k < len(top_ids):
                raise ValueError(
                    f"{case_id}.top_k cannot be smaller than top_evidence_ids"
                )
        elif top_k is not None:
            raise ValueError(f"{case_id}.top_k requires top_evidence_ids")
        forbidden_ids = _as_string_list(
            raw_case.get("forbidden_evidence_ids"),
            "forbidden_evidence_ids",
            case_id,
        )
        if len(set(required_ids)) != len(required_ids):
            raise ValueError(f"{case_id}.required_evidence_ids contains duplicates")
        if len(set(top_ids)) != len(top_ids):
            raise ValueError(f"{case_id}.top_evidence_ids contains duplicates")
        if len(set(forbidden_ids)) != len(forbidden_ids):
            raise ValueError(f"{case_id}.forbidden_evidence_ids contains duplicates")
        if set(required_ids) & set(forbidden_ids):
            raise ValueError(
                f"{case_id} has evidence IDs that are both required and forbidden"
            )
        if set(top_ids) & set(forbidden_ids):
            raise ValueError(
                f"{case_id} has evidence IDs that are both top-ranked and forbidden"
            )
        if should_abstain and (required_facts or required_ids or top_ids):
            raise ValueError(
                f"{case_id} abstention cases cannot require facts or evidence"
            )
        if {claim.casefold() for claim in required_facts} & {
            claim.casefold() for claim in forbidden_claims
        }:
            raise ValueError(f"{case_id} has overlapping required and forbidden claims")

        cases.append(
            EvalCase(
                id=case_id,
                category=category,
                input=input_value,
                expected_behavior=expected_behavior,
                required_facts=required_facts,
                forbidden_claims=forbidden_claims,
                required_evidence_ids=required_ids,
                top_evidence_ids=top_ids,
                top_k=top_k,
                forbidden_evidence_ids=forbidden_ids,
                should_abstain=should_abstain,
                notes=notes,
            )
        )

    serialized = json.dumps(raw_cases, ensure_ascii=False)
    if any(pattern.search(serialized) for pattern in SECRET_PATTERNS):
        raise ValueError("eval dataset contains a possible secret")
    return tuple(cases)


def load_cases(path: Path = DEFAULT_CASES_PATH) -> tuple[EvalCase, ...]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return validate_cases(raw_cases)


def _remove_constraint_metadata(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _remove_constraint_metadata(child)
            for key, child in value.items()
            if key not in {"claim_limits", "safety_notes"}
        }
    if isinstance(value, list):
        return [_remove_constraint_metadata(child) for child in value]
    return value


def _evidence_blob(
    turn: PreparedAgentTurn, *, include_constraint_metadata: bool = True
) -> str:
    return json.dumps(
        [
            {
                "data": (
                    evidence.data
                    if include_constraint_metadata
                    else _remove_constraint_metadata(evidence.data)
                ),
                "entity_id": evidence.entity_id,
                "entity_type": evidence.entity_type,
                "matched_fields": list(evidence.matched_fields),
                "score": evidence.score,
                "title": evidence.title,
            }
            for evidence in turn.evidence
        ],
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()


def evaluate_case(case: EvalCase, core: AgentCore) -> EvalOutcome:
    """Evaluate executable retrieval checks without pretending to judge prose."""

    turn = core.prepare(case.input)
    observed_ids = tuple(evidence.entity_id for evidence in turn.evidence)
    evidence_blob = _evidence_blob(turn)
    assertive_evidence_blob = _evidence_blob(
        turn, include_constraint_metadata=False
    )
    checks: dict[str, str] = {
        "expected_status": (
            PASS
            if turn.status == ("insufficient_evidence" if case.should_abstain else "ready")
            else FAIL
        ),
        "evidence_package_non_empty": (
            PASS if bool(turn.evidence) is not case.should_abstain else FAIL
        ),
        "required_evidence_ids": (
            NOT_EVALUATED
            if not case.required_evidence_ids
            else (
                PASS
                if all(evidence_id in observed_ids for evidence_id in case.required_evidence_ids)
                else FAIL
            )
        ),
        "top_evidence_ids": (
            NOT_EVALUATED
            if not case.top_evidence_ids
            else (
                PASS
                if all(
                    evidence_id in observed_ids[: case.top_k]
                    for evidence_id in case.top_evidence_ids
                )
                else FAIL
            )
        ),
        "forbidden_evidence_ids_absent": (
            NOT_EVALUATED
            if not case.forbidden_evidence_ids
            else (
                PASS
                if all(evidence_id not in observed_ids for evidence_id in case.forbidden_evidence_ids)
                else FAIL
            )
        ),
        "public_visibility_only": (
            PASS
            if all(evidence.data.get("visibility") == "public" for evidence in turn.evidence)
            else FAIL
        ),
        "restricted_markers_absent": (
            PASS
            if not any(marker.casefold() in evidence_blob for marker in RESTRICTED_EVIDENCE_MARKERS)
            else FAIL
        ),
        # This is only a literal safety signal over retrieved evidence. It is
        # not semantic validation of a future generated answer.
        "literal_forbidden_evidence_phrases_absent": (
            NOT_EVALUATED
            if not case.forbidden_claims
            else (
                PASS
                if not any(claim.casefold() in assertive_evidence_blob for claim in case.forbidden_claims)
                else FAIL
            )
        ),
        "required_facts_semantics": (
            NOT_EVALUATED if case.required_facts else NOT_EVALUATED
        ),
        "forbidden_claims_semantics": (
            NOT_EVALUATED if case.forbidden_claims else NOT_EVALUATED
        ),
        "generated_answer_semantics": NOT_EVALUATED,
    }
    failed_checks = [name for name, status in checks.items() if status == FAIL]
    review_items = [
        "generated_answer_semantics: LIVE_ONLY/HUMAN_REVIEW",
    ]
    if case.required_facts:
        review_items.append("required_facts: HUMAN_REVIEW/FUTURE_SEMANTIC_JUDGE")
    if case.forbidden_claims:
        review_items.append("forbidden_claims: HUMAN_REVIEW/FUTURE_SEMANTIC_JUDGE")
    case_status = (
        FAIL
        if failed_checks
        else NOT_EVALUATED
        if any(status == NOT_EVALUATED for status in checks.values())
        else PASS
    )
    return EvalOutcome(
        case_id=case.id,
        category=case.category,
        input=case.input,
        expected_behavior=case.expected_behavior,
        case_status=case_status,
        checks=checks,
        observed_status=turn.status,
        observed_evidence_ids=observed_ids,
        issue=", ".join(failed_checks),
        review_items=tuple(review_items),
    )


def run_offline(
    cases: Iterable[EvalCase] | None = None,
    *,
    profile_path: Path | None = None,
) -> EvalReport:
    """Run evals with local retrieval only; this path never creates an LLM client."""

    selected_cases = tuple(cases) if cases is not None else load_cases()
    core = AgentCore(ProfileService(profile_path)) if profile_path else AgentCore()
    return EvalReport(
        tuple(evaluate_case(case, core) for case in selected_cases)
    )


def run_live(cases: Iterable[EvalCase], limit: int) -> list[dict[str, object]]:
    """Run a deliberately small, explicit live sample without printing answers."""

    if limit < 1 or limit > 5:
        raise ValueError("live limit must be between 1 and 5")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is required for live evals")

    from fastapi.testclient import TestClient

    from app.api.main import create_app

    client = TestClient(create_app())
    results: list[dict[str, object]] = []
    for case in tuple(cases)[:limit]:
        response = client.post(
            "/v1/responses",
            json={"model": "banorte-cv-agent", "input": case.input},
        )
        try:
            body = response.json()
        except ValueError:
            body = {}
        output = body.get("output") if isinstance(body, dict) else None
        text = ""
        if isinstance(output, list) and output:
            message = output[0]
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list) and content:
                    first_content = content[0]
                    if isinstance(first_content, dict) and isinstance(
                        first_content.get("text"), str
                    ):
                        text = first_content["text"]
        response_schema_ok = (
            response.status_code == 200
            and isinstance(body, dict)
            and body.get("object") == "response"
            and body.get("status") == "completed"
        )
        output_shape_ok = False
        if isinstance(output, list) and output:
            message = output[0]
            if isinstance(message, dict):
                content = message.get("content")
                output_shape_ok = (
                    message.get("role") == "assistant"
                    and isinstance(content, list)
                    and bool(content)
                    and isinstance(content[0], dict)
                    and content[0].get("type") == "output_text"
                )
        results.append(
            {
                "case_id": case.id,
                "http": response.status_code,
                "response_non_empty": bool(text.strip()),
                "checks": {
                    "http_200": PASS if response.status_code == 200 else FAIL,
                    "response_schema": PASS if response_schema_ok else FAIL,
                    "output_shape": PASS if output_shape_ok else FAIL,
                    "restricted_markers_absent": (
                        PASS
                        if not any(
                            marker.casefold() in text.casefold()
                            for marker in RESTRICTED_EVIDENCE_MARKERS
                        )
                        else FAIL
                    ),
                    "factuality_semantics": NOT_EVALUATED,
                    "groundedness_semantics": NOT_EVALUATED,
                    "ownership_skill_metrics_semantics": NOT_EVALUATED,
                },
                "manual_review": [
                    "factuality_semantics",
                    "groundedness_semantics",
                    "ownership_skill_metrics_semantics",
                ],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly enable a small live sample; offline is the default",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="confirm that live calls and their cost are intentional",
    )
    parser.add_argument("--limit", type=int, help="live sample size, maximum 5")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases(args.cases)
    if args.live:
        if not args.confirm_live:
            raise SystemExit("--live requires --confirm-live")
        if args.limit is None:
            raise SystemExit("--live requires --limit N")
        print(json.dumps(run_live(cases, args.limit), ensure_ascii=False, indent=2))
        return 0

    print(render_report(run_offline(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
