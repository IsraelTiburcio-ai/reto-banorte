"""Run the auditable pre-Banorte conversational smoke cases.

Without ``--base-url`` (or ``PRE_BANORTE_BASE_URL``), this command validates
the case fixture only and performs no HTTP or provider calls. Live responses
with generated text are reported as REVIEW because this runner is not an
LLM-as-a-judge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "pre_banorte_cases.json"
DONE_FRAME = "data: [DONE]"


def load_cases() -> list[dict[str, Any]]:
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 30:
        raise ValueError("pre_banorte_cases.json must contain exactly 30 cases")
    case_ids: set[str] = set()
    for case in raw:
        if not isinstance(case, dict):
            raise ValueError("every pre-Banorte case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("pre-Banorte case IDs must be unique non-empty strings")
        case_ids.add(case_id)
        for field in (
            "category",
            "reference_behavior",
            "retrieval",
            "provider",
            "required_properties",
            "forbidden_properties",
        ):
            if field not in case:
                raise ValueError(f"{case_id} is missing {field}")
        if not isinstance(case.get("input"), (str, list)):
            raise ValueError(f"{case_id}.input must be text or a message list")
    return raw


def _response_text(body: dict[str, Any]) -> str:
    output = body.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("response output is missing")
    message = output[0]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("response assistant message is invalid")
    content = message.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("response content is missing")
    part = content[0]
    if not isinstance(part, dict) or part.get("type") != "output_text":
        raise ValueError("response output_text is missing")
    text = part.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("response output text is empty")
    return text


def _validate_json_response(body: dict[str, Any]) -> str:
    if body.get("object") != "response" or body.get("status") != "completed":
        raise ValueError("response object/status is invalid")
    return _response_text(body)


def _validate_sse_response(raw: str) -> str:
    frames = [frame for frame in raw.split("\n\n") if frame]
    if not frames or frames[-1] != DONE_FRAME:
        raise ValueError("SSE terminator is invalid")
    events: list[dict[str, Any]] = []
    for frame in frames[:-1]:
        lines = frame.splitlines()
        if len(lines) != 2 or not lines[0].startswith("event: "):
            raise ValueError("SSE event frame is invalid")
        payload = lines[1]
        if not payload.startswith("data: "):
            raise ValueError("SSE data frame is invalid")
        event = json.loads(payload[6:])
        if not isinstance(event, dict) or event.get("type") != lines[0][7:]:
            raise ValueError("SSE event type is inconsistent")
        events.append(event)
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise ValueError("SSE sequence numbers are not contiguous")
    if len(events) != 9 or events[-1].get("type") != "response.completed":
        raise ValueError("SSE lifecycle sequence is incomplete")
    response = events[-1].get("response")
    if not isinstance(response, dict):
        raise ValueError("SSE completed response is missing")
    return _validate_json_response(response)


def _post_case(base_url: str, api_key: str | None, case: dict[str, Any]) -> str:
    request = Request(
        base_url.rstrip("/") + "/v1/responses",
        data=json.dumps(
            {
                "input": case["input"],
                "stream": True if case.get("stream") else False,
                "store": False,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}") from None
    except URLError as exc:
        raise ValueError(f"network error: {exc.reason}") from None

    if case.get("stream"):
        if not content_type.startswith("text/event-stream"):
            raise ValueError("stream response has the wrong content type")
        return _validate_sse_response(raw)
    return _validate_json_response(json.loads(raw))


def _deterministic_check(case: dict[str, Any], text: str) -> str:
    folded = text.casefold()
    forbidden = {
        "soy israel",
        "contraseña",
        "contrasena",
        "password",
        "internal_summary",
        "do_not_expose",
    }
    if any(marker in folded for marker in forbidden):
        raise ValueError("forbidden response marker detected")
    if case.get("use_insufficient_evidence"):
        if "evidencia pública suficiente" not in folded:
            raise ValueError("expected safe insufficient-evidence response")
    if case.get("provider") == "no" and "insufficient_evidence" in case.get("forbidden_properties", []):
        if "evidencia pública suficiente" in folded:
            raise ValueError("unexpected insufficient-evidence response")
    if case.get("review_required"):
        return "REVIEW"
    return "PASS"


def run(base_url: str | None, api_key: str | None, limit: int | None = None) -> int:
    cases = load_cases()
    selected = cases[:limit] if limit is not None else cases
    if base_url is None:
        print(f"OFFLINE: {len(cases)}/{len(cases)} pre-Banorte cases validated")
        print("HTTP calls: 0")
        print("Generated-answer semantics: REVIEW only when a live URL is supplied")
        return 0

    results: list[tuple[str, str, str]] = []
    for case in selected:
        try:
            text = _post_case(base_url, api_key, case)
            status = _deterministic_check(case, text)
            results.append((case["id"], status, ""))
        except (ValueError, json.JSONDecodeError) as exc:
            results.append((case["id"], "FAIL", str(exc)))

    for case_id, status, reason in results:
        suffix = f" — {reason}" if reason else ""
        print(f"{status} {case_id}{suffix}")
    counts = {status: sum(item[1] == status for item in results) for status in ("PASS", "REVIEW", "FAIL")}
    print(f"{len(results)}/{len(selected)} executed")
    print(f"{counts['PASS']} PASS, {counts['REVIEW']} REVIEW, {counts['FAIL']} FAIL")
    return 1 if counts["FAIL"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("PRE_BANORTE_BASE_URL"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= 30:
        parser.error("--limit must be between 1 and 30")
    return run(args.base_url, os.getenv("PRE_BANORTE_API_KEY"), args.limit)


if __name__ == "__main__":
    sys.exit(main())
