"""Reporting helpers for deterministic agent evaluations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT_EVALUATED"
CHECK_STATUSES = frozenset({PASS, FAIL, NOT_EVALUATED})


@dataclass(frozen=True)
class EvalOutcome:
    """Observable results for one case, including unevaluated expectations."""

    case_id: str
    category: str
    input: str
    expected_behavior: str
    case_status: str
    checks: dict[str, str]
    observed_status: str
    observed_evidence_ids: tuple[str, ...]
    issue: str = ""
    review_items: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Compatibility accessor: only a fully evaluated case can pass."""

        return self.case_status == PASS


@dataclass(frozen=True)
class EvalReport:
    """Aggregate results while keeping N/A out of executable denominators."""

    outcomes: tuple[EvalOutcome, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(outcome.case_status == PASS for outcome in self.outcomes)

    @property
    def failed(self) -> int:
        return sum(outcome.case_status == FAIL for outcome in self.outcomes)

    @property
    def not_evaluated(self) -> int:
        return sum(
            outcome.case_status == NOT_EVALUATED for outcome in self.outcomes
        )

    @property
    def executable_cases(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        """Pass rate for fully evaluated cases only; not a quality score."""

        return self.passed / self.executable_cases if self.executable_cases else 0.0

    @property
    def checks_declared(self) -> int:
        return sum(len(outcome.checks) for outcome in self.outcomes)

    @property
    def checks_executed(self) -> int:
        return sum(
            status in {PASS, FAIL}
            for outcome in self.outcomes
            for status in outcome.checks.values()
        )

    @property
    def checks_not_evaluated(self) -> int:
        return sum(
            status == NOT_EVALUATED
            for outcome in self.outcomes
            for status in outcome.checks.values()
        )

    @property
    def check_coverage(self) -> float:
        return (
            self.checks_executed / self.checks_declared
            if self.checks_declared
            else 0.0
        )

    def check_counts(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, NOT_EVALUATED: 0}
        for outcome in self.outcomes:
            for status in outcome.checks.values():
                counts[status] += 1
        return counts

    def by_category(self) -> dict[str, dict[str, int]]:
        counts: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {PASS: 0, FAIL: 0, NOT_EVALUATED: 0, "checks": 0, "declared": 0}
        )
        for outcome in self.outcomes:
            counts[outcome.category][outcome.case_status] += 1
            counts[outcome.category]["declared"] += len(outcome.checks)
            counts[outcome.category]["checks"] += sum(
                status in {PASS, FAIL} for status in outcome.checks.values()
            )
        return {category: values for category, values in sorted(counts.items())}


def render_report(report: EvalReport) -> str:
    """Render a compact, auditable report without generated answer text."""

    check_counts = report.check_counts()
    lines = [
        "OFFLINE CASE STATUS:",
        f"TOTAL CASES: {report.total}",
        f"PASS: {report.passed}",
        f"FAIL: {report.failed}",
        f"NOT_EVALUATED: {report.not_evaluated}",
        f"OFFLINE EXECUTABLE CASES: {report.executable_cases}",
        f"OFFLINE EXECUTABLE PASS RATE: {report.pass_rate:.1%}",
        "(This is retrieval/contract coverage, not factuality or overall quality.)",
        "",
        "CHECK COVERAGE:",
        f"EXECUTED: {report.checks_executed}/{report.checks_declared} ({report.check_coverage:.1%})",
        f"PASS CHECKS: {check_counts[PASS]}",
        f"FAIL CHECKS: {check_counts[FAIL]}",
        f"NOT_EVALUATED CHECKS: {check_counts[NOT_EVALUATED]}",
        "",
        "LIVE/MANUAL COVERAGE:",
        "NOT_EVALUATED offline: generated-answer semantics, paraphrase detection, "
        "ownership wording, skill calibration, approximate-metric wording, and fluency.",
        "",
        "BY CATEGORY:",
    ]
    for category, counts in report.by_category().items():
        coverage = (
            f"{counts['checks']}/{counts['declared']} checks"
            if counts["declared"]
            else "0/0 checks"
        )
        lines.append(
            f"{category}: PASS {counts[PASS]}, FAIL {counts[FAIL]}, "
            f"NOT_EVALUATED {counts[NOT_EVALUATED]} ({coverage})"
        )

    failures = [outcome for outcome in report.outcomes if outcome.case_status == FAIL]
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
    not_evaluated = [
        outcome
        for outcome in report.outcomes
        if outcome.case_status == NOT_EVALUATED
    ]
    if not_evaluated:
        lines.extend(("", "NOT_EVALUATED CASES:"))
        for outcome in not_evaluated:
            items = ", ".join(outcome.review_items) or "declared semantic expectation"
            lines.append(f"- {outcome.case_id}: {items}")
    return "\n".join(lines)
