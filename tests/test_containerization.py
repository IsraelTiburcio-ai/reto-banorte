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


def _strip_shell_comment(command: str) -> str:
    """Remove shell comments while preserving # characters inside quotes."""

    result: list[str] = []
    quote: str | None = None
    escaped = False
    for character in command:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if character == "\\" and quote != "'":
            result.append(character)
            escaped = True
            continue
        if quote is not None:
            result.append(character)
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            result.append(character)
            continue
        if character == "#":
            break
        result.append(character)
    return "".join(result).rstrip()


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
        if _dockerignore_rule_matches(path, pattern):
            ignored = not negated
    return ignored


def _dockerignore_rule_matches(path: str, pattern: str) -> bool:
    """Match only the targeted root, basename, directory, and ** forms."""

    path = path[2:] if path.startswith("./") else path
    path = path[1:] if path.startswith("/") else path
    pattern = pattern.rstrip("/")
    pattern = pattern[2:] if pattern.startswith("./") else pattern
    pattern = pattern[1:] if pattern.startswith("/") else pattern

    if pattern == path or fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/"):
        suffix = pattern[3:]
        return path == suffix or path.endswith(f"/{suffix}")
    if "/" not in pattern:
        return any(fnmatch.fnmatchcase(component, pattern) for component in path.split("/"))
    return False


def validate_container_contract(dockerfile: str, dockerignore_lines: list[str]) -> None:
    instructions = parse_dockerfile_instructions(dockerfile)
    opcodes = [opcode for opcode, _ in instructions]

    _require(opcodes and opcodes[0] == "FROM", "Dockerfile must start with FROM")
    base_images = [argument for opcode, argument in instructions if opcode == "FROM"]
    _require(
        base_images == ["python:3.11-slim-bookworm"],
        "the effective final stage must be the single expected slim Python image",
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

    run_commands = [
        _strip_shell_comment(argument).strip()
        for opcode, argument in instructions
        if opcode == "RUN"
    ]
    _require(
        any(
            "python -m pip install" in command
            and "--no-cache-dir" in command
            and "--target /app" in command
            and "/tmp/build" in command
            for command in run_commands
        ),
        "runtime dependencies must be installed from the temporary project tree",
    )
    _require(
        any("groupadd --system app" in command for command in run_commands),
        "Dockerfile must create the non-root app group",
    )
    _require(
        any("useradd --system --gid app" in command for command in run_commands),
        "Dockerfile must create the non-root app user",
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
    shell_command = _strip_shell_comment(command[2]).strip()
    _require(
        shell_command.startswith("exec python -m uvicorn "),
        "CMD must exec Uvicorn as the main process",
    )
    _require("--host 0.0.0.0" in shell_command, "CMD must bind to 0.0.0.0")
    _require(
        re.search(r'--port\s+"?\$\{PORT:-8080\}"?', shell_command)
        is not None,
        "CMD must expand PORT at runtime with default 8080",
    )
    _require("'${PORT:-8080}'" not in shell_command, "PORT must not be single-quoted")
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

    def test_single_quoted_port_fails_expansion_check(self) -> None:
        mutated = self.dockerfile.replace(
            r'\"${PORT:-8080}\"', "'${PORT:-8080}'"
        )
        with self.assertRaises(AssertionError):
            validate_container_contract(mutated, self.dockerignore_lines)

    def test_shell_comment_cannot_satisfy_cmd_checks(self) -> None:
        mutated_lines = []
        for line in self.dockerfile.splitlines():
            if line.startswith("CMD "):
                mutated_lines.append(
                    'CMD ["sh", "-c", "exec false # python -m uvicorn '
                    'app.api.main:app --host 0.0.0.0 --port \\\"${PORT:-8080}\\\""]'
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

    def test_final_stage_and_runtime_setup_are_required(self) -> None:
        mutations = (
            self.dockerfile + "\nFROM alpine:3.20\n",
            self.dockerfile.replace("RUN python -m pip install", "RUN false"),
            self.dockerfile.replace("groupadd --system app", "echo missing-group").replace(
                "useradd --system --gid app", "echo missing-user"
            ),
        )
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                with self.assertRaises(AssertionError):
                    validate_container_contract(mutated, self.dockerignore_lines)

    def test_late_dockerignore_negation_cannot_reinclude_env(self) -> None:
        for mutation in ("!.env", "!/.env", "!**/.env", "data"):
            with self.subTest(mutation=mutation):
                mutated_rules = self.dockerignore_lines + [mutation]
                with self.assertRaises(AssertionError):
                    validate_container_contract(self.dockerfile, mutated_rules)


if __name__ == "__main__":
    unittest.main()
