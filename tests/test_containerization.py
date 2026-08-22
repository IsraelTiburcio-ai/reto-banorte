from __future__ import annotations

import fnmatch
import json
import re
import shlex
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SECRET_NAMES = frozenset({"OPENAI_API_KEY", "AGENT_API_KEY"})
Instruction = tuple[str, str]


def parse_dockerfile_instructions(text: str) -> list[Instruction]:
    """Return effective Dockerfile instructions in source order.

    This deliberately handles only the Dockerfile syntax needed by the project:
    blank lines, full-line comments, and backslash continuations. It does not
    treat text in comments as an instruction.
    """

    instructions: list[Instruction] = []
    logical_line = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        if line.endswith("\\"):
            logical_line += line[:-1].rstrip() + " "
            continue

        logical_line += line
        parts = logical_line.split(None, 1)
        if parts:
            instructions.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
        logical_line = ""

    if logical_line:
        parts = logical_line.split(None, 1)
        instructions.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return instructions


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _copy_pairs(instructions: list[Instruction]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for opcode, argument in instructions:
        if opcode != "COPY":
            continue
        tokens = [token for token in shlex.split(argument) if not token.startswith("--")]
        _require(len(tokens) >= 2, "COPY must have a source and destination")
        destination = tokens[-1]
        pairs.extend((source, destination) for source in tokens[:-1])
    return pairs


def _declared_names(instructions: list[Instruction]) -> set[str]:
    names: set[str] = set()
    for opcode, argument in instructions:
        if opcode not in {"ARG", "ENV"}:
            continue
        _require(
            not any(
                re.search(rf"(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])", argument)
                for secret in FORBIDDEN_SECRET_NAMES
            ),
            f"secret name appears in effective {opcode} instruction: {argument}",
        )
        tokens = shlex.split(argument)
        if not tokens:
            continue
        if opcode == "ENV" and len(tokens) >= 2 and "=" not in tokens[0]:
            names.add(tokens[0])
            continue
        for token in tokens:
            name = token.split("=", 1)[0]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                names.add(name)
    _require(
        not FORBIDDEN_SECRET_NAMES.intersection(names),
        "secret names must not be declared by ARG or ENV",
    )
    return names


def _dockerignore_ignores(path: str, lines: list[str]) -> bool:
    """Evaluate the targeted .dockerignore rules using last-match-wins."""

    ignored = False
    for raw_line in lines:
        rule = raw_line.strip()
        if not rule or rule.startswith("#"):
            continue
        negated = rule.startswith("!")
        pattern = rule[1:] if negated else rule
        pattern = pattern.rstrip("/")
        if pattern == path or fnmatch.fnmatchcase(path, pattern):
            ignored = not negated
    return ignored


def validate_container_contract(dockerfile: str, dockerignore_lines: list[str]) -> None:
    instructions = parse_dockerfile_instructions(dockerfile)
    opcodes = [opcode for opcode, _ in instructions]

    _require(opcodes and opcodes[0] == "FROM", "Dockerfile must start with FROM")
    _require(
        any(
            argument == "python:3.11-slim-bookworm"
            for opcode, argument in instructions
            if opcode == "FROM"
        ),
        "expected pinned slim Python base image",
    )

    copy_pairs = _copy_pairs(instructions)
    for expected in (
        ("pyproject.toml", "/tmp/build/pyproject.toml"),
        ("app", "/tmp/build/app"),
        ("data/profile.json", "/app/data/profile.json"),
    ):
        _require(expected in copy_pairs, f"missing effective COPY {expected}")
    _require(
        not any(source in {".", "./"} for source, _ in copy_pairs),
        "Dockerfile must not copy the whole build context",
    )
    _require(
        not any(source.startswith(".env") for source, _ in copy_pairs),
        "Dockerfile must not copy local environment files",
    )

    users = [argument.strip() for opcode, argument in instructions if opcode == "USER"]
    _require(users and users[-1] == "app:app", "final effective USER must be app:app")

    _declared_names(instructions)

    commands = [argument for opcode, argument in instructions if opcode == "CMD"]
    _require(commands, "Dockerfile must define CMD")
    try:
        command = json.loads(commands[-1])
    except json.JSONDecodeError as exc:
        raise AssertionError("CMD must be valid JSON exec syntax") from exc
    _require(
        isinstance(command, list)
        and len(command) >= 3
        and command[0] in {"sh", "/bin/sh"}
        and command[1] == "-c"
        and all(isinstance(item, str) for item in command),
        "CMD must use a shell wrapper for runtime PORT expansion",
    )
    shell_command = command[2]
    _require(shell_command.lstrip().startswith("exec "), "CMD must exec the server")
    _require("python -m uvicorn" in shell_command, "CMD must start Uvicorn")
    _require("--host 0.0.0.0" in shell_command, "CMD must bind to 0.0.0.0")
    _require(
        re.search(r'--port\s+["\']?\$\{PORT:-8080\}["\']?', shell_command)
        is not None,
        "CMD must expand PORT at runtime with default 8080",
    )
    _require("--reload" not in shell_command, "production CMD must not use --reload")

    _require(_dockerignore_ignores(".env", dockerignore_lines), ".env must remain ignored")
    _require(
        _dockerignore_ignores(".env.local", dockerignore_lines),
        "local environment variants must remain ignored",
    )
    _require(
        not _dockerignore_ignores(".env.example", dockerignore_lines),
        ".env.example must remain allowed by the documented policy",
    )
    _require(
        not _dockerignore_ignores("data/profile.json", dockerignore_lines),
        "data/profile.json must remain in the build context",
    )


class ContainerizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.dockerignore_lines = (ROOT / ".dockerignore").read_text(
            encoding="utf-8"
        ).splitlines()

    def test_effective_container_contract(self) -> None:
        validate_container_contract(self.dockerfile, self.dockerignore_lines)

    def test_comments_do_not_satisfy_effective_copy(self) -> None:
        mutated = self.dockerfile.replace(
            "COPY app /tmp/build/app", "# COPY app /tmp/build/app"
        )
        with self.assertRaises(AssertionError):
            validate_container_contract(mutated, self.dockerignore_lines)

    def test_later_root_user_fails_effective_user_check(self) -> None:
        mutated = self.dockerfile + "\nUSER root\n"
        with self.assertRaises(AssertionError):
            validate_container_contract(mutated, self.dockerignore_lines)

    def test_exec_form_with_literal_port_fails_runtime_port_check(self) -> None:
        mutated_lines = []
        for line in self.dockerfile.splitlines():
            if line.startswith("CMD "):
                mutated_lines.append(
                    'CMD ["python", "-m", "uvicorn", "app.api.main:app", '
                    '"--host", "0.0.0.0", "--port", "${PORT:-8080}"]'
                )
            else:
                mutated_lines.append(line)
        with self.assertRaises(AssertionError):
            validate_container_contract("\n".join(mutated_lines), self.dockerignore_lines)

    def test_secret_env_declaration_fails_effective_env_check(self) -> None:
        mutated = self.dockerfile + "\nENV SAFE=x OPENAI_API_KEY=secret\n"
        with self.assertRaises(AssertionError):
            validate_container_contract(mutated, self.dockerignore_lines)

    def test_env_and_arg_assignment_forms_are_parsed(self) -> None:
        instructions = parse_dockerfile_instructions(
            "ARG BUILD_FLAG\nENV SAFE=x OTHER=y\nENV LEGACY value\n"
        )
        self.assertEqual(
            _declared_names(instructions),
            {"BUILD_FLAG", "SAFE", "OTHER", "LEGACY"},
        )

    def test_late_dockerignore_negation_cannot_reinclude_env(self) -> None:
        mutated_rules = self.dockerignore_lines + ["!.env"]
        with self.assertRaises(AssertionError):
            validate_container_contract(self.dockerfile, mutated_rules)


if __name__ == "__main__":
    unittest.main()
