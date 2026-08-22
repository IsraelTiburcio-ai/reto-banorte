"""Reporting helpers for deterministic agent evaluations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT_EVALUATED"
NOT_APPLICABLE = "N/A"
CHECK_STATUSES = frozenset({PASS, FAIL, NOT_EVALUATED, NOT_APPLICABLE})


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
    def checks_applicable(self) -> int:
        return sum(
            status in {PASS, FAIL, NOT_EVALUATED}
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
    def checks_not_applicable(self) -> int:
        return sum(
            status == NOT_APPLICABLE
            for outcome in self.outcomes
            for status in outcome.checks.values()
        )

    @property
    def check_coverage(self) -> float:
        return (
            self.checks_executed / self.checks_applicable
            if self.checks_applicable
            else 0.0
        )

    def check_counts(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, NOT_EVALUATED: 0, NOT_APPLICABLE: 0}
        for outcome in self.outcomes:
            for status in outcome.checks.values():
                counts[status] += 1
        return counts

    def by_category(self) -> dict[str, dict[str, int]]:
        counts: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {
                "case_pass": 0,
                "case_fail": 0,
                "case_not_evaluated": 0,
                "check_pass": 0,
                "check_fail": 0,
                "check_not_evaluated": 0,
                "check_not_applicable": 0,
                "executed": 0,
                "applicable": 0,
            }
        )
        for outcome in self.outcomes:
            counts[outcome.category][
                {
                    PASS: "case_pass",
                    FAIL: "case_fail",
                    NOT_EVALUATED: "case_not_evaluated",
                }[outcome.case_status]
            ] += 1
            for status in outcome.checks.values():
                counts[outcome.category][
                    {
                        PASS: "check_pass",
                        FAIL: "check_fail",
                        NOT_EVALUATED: "check_not_evaluated",
                        NOT_APPLICABLE: "check_not_applicable",
                    }[status]
                ] += 1
            counts[outcome.category]["applicable"] += sum(
                status in {PASS, FAIL, NOT_EVALUATED}
                for status in outcome.checks.values()
            )
            counts[outcome.category]["executed"] += sum(
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
        f"DECLARED/APPLICABLE CHECKS: {report.checks_applicable}",
        f"EXECUTED CHECKS: {report.checks_executed}",
        f"PASS CHECKS: {check_counts[PASS]}",
        f"FAIL CHECKS: {check_counts[FAIL]}",
        f"NOT_EVALUATED CHECKS: {check_counts[NOT_EVALUATED]}",
        f"N/A CHECKS: {check_counts[NOT_APPLICABLE]}",
        f"CHECK COVERAGE: {report.check_coverage:.1%}",
        "(N/A is excluded from the coverage denominator; coverage is not a quality score.)",
        "",
        "LIVE/MANUAL COVERAGE:",
        "NOT_EVALUATED offline: generated-answer semantics, paraphrase detection, "
        "ownership wording, skill calibration, approximate-metric wording, and fluency.",
        "",
        "BY CATEGORY:",
    ]
    for category, counts in report.by_category().items():
        coverage = (
            f"{counts['executed']}/{counts['applicable']} checks"
            if counts["applicable"]
            else "0/0 applicable checks"
        )
        lines.append(
            f"{category}: CASES PASS {counts['case_pass']}, "
            f"FAIL {counts['case_fail']}, NOT_EVALUATED {counts['case_not_evaluated']}; "
            f"CHECKS PASS {counts['check_pass']}, FAIL {counts['check_fail']}, "
            f"NOT_EVALUATED {counts['check_not_evaluated']}, "
            f"N/A {counts['check_not_applicable']} ({coverage})"
        )

    lines.extend(("", "CASE REPORTS:"))
    for outcome in report.outcomes:
        lines.extend((f"CASE: {outcome.case_id}", f"STATUS: {outcome.case_status}"))
        for status in (PASS, FAIL, NOT_EVALUATED, NOT_APPLICABLE):
            lines.append(f"{status}:")
            matching_checks = [
                name for name, check_status in outcome.checks.items()
                if check_status == status
            ]
            lines.extend(f"  - {name}" for name in matching_checks)
        if outcome.issue:
            lines.append(f"ISSUE: {outcome.issue}")
        lines.append("")
    return "\n".join(lines)
