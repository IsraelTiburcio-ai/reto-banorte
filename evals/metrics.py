"""Small reporting helpers for deterministic agent evaluations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvalOutcome:
    """The observable result of evaluating one case."""

    case_id: str
    category: str
    input: str
    expected_behavior: str
    passed: bool
    checks: dict[str, bool]
    observed_status: str
    observed_evidence_ids: tuple[str, ...]
    issue: str = ""


@dataclass(frozen=True)
class EvalReport:
    """Aggregate results for one evaluation run."""

    outcomes: tuple[EvalOutcome, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(outcome.passed for outcome in self.outcomes)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def by_category(self) -> dict[str, tuple[int, int]]:
        counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for outcome in self.outcomes:
            counts[outcome.category][0] += 1
            if outcome.passed:
                counts[outcome.category][1] += 1
        return {
            category: (values[0], values[1])
            for category, values in sorted(counts.items())
        }


def render_report(report: EvalReport) -> str:
    """Render a compact, auditable report without generated answer text."""

    lines = [
        f"TOTAL: {report.total}",
        f"PASS: {report.passed}",
        f"FAIL: {report.failed}",
        f"PASS RATE: {report.pass_rate:.1%}",
        "",
        "BY CATEGORY:",
    ]
    for category, (total, passed) in report.by_category().items():
        lines.append(f"{category}: {passed}/{total}")

    failures = [outcome for outcome in report.outcomes if not outcome.passed]
    if failures:
        lines.extend(("", "FAILURES:"))
        for outcome in failures:
            lines.extend(
                (
                    f"- {outcome.case_id}",
                    f"  input: {outcome.input}",
                    f"  expected: {outcome.expected_behavior}",
                    f"  observed_status: {outcome.observed_status}",
                    f"  observed_evidence_ids: {list(outcome.observed_evidence_ids)}",
                    f"  issue: {outcome.issue}",
                )
            )
    return "\n".join(lines)
