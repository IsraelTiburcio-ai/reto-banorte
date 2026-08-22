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


def _split_shell_and(command: str) -> list[str]:
    """Split the strict setup command on && outside quoted strings."""

    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            current.append(character)
            escaped = True
            index += 1
            continue
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
            index += 1
            continue
        if command.startswith("&&", index):
            segments.append("".join(current).strip())
            current = []
            index += 2
            continue
        if character in {";", "|"}:
            raise AssertionError("strict setup command allows only && separators")
        current.append(character)
        index += 1
    segments.append("".join(current).strip())
    return segments


def _normalize_command(command: str) -> str:
    return " ".join(command.split())


def _replace_instruction_block(text: str, opcode: str, replacement: str) -> str:
    """Replace one effective instruction, including its continuation lines."""

    lines = text.splitlines()
    result: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        line = lines[index]
        if not replaced and line.lstrip().startswith(f"{opcode} "):
            result.append(replacement)
            replaced = True
            while line.rstrip().endswith("\\") and index + 1 < len(lines):
                index += 1
                line = lines[index]
            index += 1
            continue
        result.append(line)
        index += 1
    _require(replaced, f"mutation target {opcode} was not found")
    return "\n".join(result) + "\n"


def _replace_cmd(text: str, shell_command: str) -> str:
    return _replace_instruction_block(
        text,
        "CMD",
        "CMD " + json.dumps(["sh", "-c", shell_command]),
    )


def _move_setup_run_before_required_copies(text: str) -> str:
    setup_start = text.index("RUN python -m pip install")
    setup_end = text.index("\n\nCOPY data/profile.json", setup_start)
    setup_block = text[setup_start:setup_end]
    without_setup = text[:setup_start] + text[setup_end + 2 :]
    marker = "WORKDIR /app\n\n"
    _require(marker in without_setup, "WORKDIR insertion point was not found")
    return without_setup.replace(marker, marker + setup_block + "\n\n", 1)


def _setup_run_matches(argument: str) -> bool:
    command = _normalize_command(_strip_shell_comment(argument))
    try:
        segments = [_normalize_command(segment) for segment in _split_shell_and(command)]
    except AssertionError:
        return False
    expected = [
        "python -m pip install --no-cache-dir --target /app /tmp/build",
        "rm -rf /tmp/build",
        "groupadd --system app",
        "useradd --system --gid app --no-create-home --home-dir /nonexistent app",
        "mkdir -p /app/data",
        "chown -R app:app /app",
    ]
    return segments == expected


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


def _copy_pair_indices(
    instructions: list[Instruction],
) -> dict[tuple[str, str], list[int]]:
    pairs: dict[tuple[str, str], list[int]] = {}
    for index, (opcode, argument) in enumerate(instructions):
        if opcode != "COPY":
            continue
        tokens = [token for token in shlex.split(argument) if not token.startswith("--")]
        _require(len(tokens) >= 2, "COPY must have a source and destination")
        destination = tokens[-1]
        for source in tokens[:-1]:
            pairs.setdefault((source, destination), []).append(index)
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


def _validate_env_ignore_policy(lines: list[str]) -> None:
    """Allow only the repository's exact .env.example exception."""

    for raw_line in lines:
        rule = raw_line.strip()
        if rule.startswith("!") and rule != "!.env.example":
            raise AssertionError(
                "only the exact .env.example negation is allowed for env files"
            )

    effective_rules = [
        rule
        for raw_line in lines
        if (rule := raw_line.strip()) and not rule.startswith("#")
    ]
    _require(
        ".env.*" in effective_rules,
        "the broad .env.* exclusion is required by the repository contract",
    )


def _validate_data_ignore_policy(lines: list[str]) -> None:
    """Reject any effective broad rule that could hide the profile directory."""

    for raw_line in lines:
        rule = raw_line.strip()
        if not rule or rule.startswith("#") or rule.startswith("!"):
            continue
        pattern = rule[1:] if rule.startswith("!") else rule
        if _dockerignore_rule_matches("data", pattern) or _dockerignore_rule_matches(
            "data/profile.json", pattern
        ):
            raise AssertionError("data/profile.json must not be excluded by a broad rule")


def validate_container_contract(dockerfile: str, dockerignore_lines: list[str]) -> None:
    instructions = parse_dockerfile_instructions(dockerfile)
    opcodes = [opcode for opcode, _ in instructions]

    _require(opcodes and opcodes[0] == "FROM", "Dockerfile must start with FROM")
    base_images = [argument for opcode, argument in instructions if opcode == "FROM"]
    _require(
        base_images == ["python:3.11-slim-bookworm"],
        "the effective final stage must be the single expected slim Python image",
    )
    _require("ENTRYPOINT" not in opcodes, "ENTRYPOINT is not allowed by the runtime contract")

    workdirs = [argument.strip() for opcode, argument in instructions if opcode == "WORKDIR"]
    _require(workdirs == ["/app"], "effective WORKDIR must be /app")

    copy_pairs = _copy_pairs(instructions)
    copy_indices = _copy_pair_indices(instructions)
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

    setup_indices = [
        index
        for index, (opcode, argument) in enumerate(instructions)
        if opcode == "RUN" and _setup_run_matches(argument)
    ]
    _require(
        len(setup_indices) == 1,
        "Dockerfile must contain exactly one exact ordered runtime setup RUN",
    )
    profile_chown_indices = [
        index
        for index, (opcode, argument) in enumerate(instructions)
        if opcode == "RUN"
        and _normalize_command(_strip_shell_comment(argument))
        == "chown app:app /app/data/profile.json"
    ]
    _require(
        len(profile_chown_indices) == 1,
        "Dockerfile must contain the exact profile ownership RUN",
    )

    users = [argument.strip() for opcode, argument in instructions if opcode == "USER"]
    _require(users and users[-1] == "app:app", "final effective USER must be app:app")

    required_order = (
        copy_indices.get(("pyproject.toml", "/tmp/build/pyproject.toml"), []),
        copy_indices.get(("app", "/tmp/build/app"), []),
        setup_indices,
        copy_indices.get(("data/profile.json", "/app/data/profile.json"), []),
        profile_chown_indices,
        [
            index
            for index, (opcode, argument) in enumerate(instructions)
            if opcode == "USER" and argument.strip() == "app:app"
        ],
        [index for index, opcode in enumerate(opcodes) if opcode == "CMD"],
    )
    _require(
        all(len(indices) == 1 for indices in required_order),
        "each required runtime instruction must occur exactly once",
    )
    required_order_indices = [indices[0] for indices in required_order]
    _require(
        required_order_indices == sorted(required_order_indices),
        "runtime instructions must preserve the repository's required order",
    )

    _declared_names(instructions)

    commands = [argument for opcode, argument in instructions if opcode == "CMD"]
    _require(commands, "Dockerfile must define CMD")
    cmd_indices = [index for index, opcode in enumerate(opcodes) if opcode == "CMD"]
    _require(
        cmd_indices[-1] == len(instructions) - 1,
        "CMD must be the final effective Dockerfile instruction",
    )
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
        shell_command
        == 'exec python -m uvicorn app.api.main:app --host 0.0.0.0 --port "${PORT:-8080}"',
        "CMD must match the exact application runtime command",
    )

    _validate_env_ignore_policy(dockerignore_lines)
    _validate_data_ignore_policy(dockerignore_lines)
    for env_path in (".env", ".env.local", ".env.production", ".env.test"):
        _require(
            _dockerignore_ignores(env_path, dockerignore_lines),
            f"{env_path} must remain ignored",
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

    def test_required_mutation_matrix_is_rejected(self) -> None:
        expected_cmd = (
            'exec python -m uvicorn app.api.main:app --host 0.0.0.0 '
            '--port "${PORT:-8080}"'
        )
        mutations = {
            "run false before pip": self.dockerfile.replace(
                "RUN python -m pip install", "RUN false && python -m pip install"
            ),
            "run echo pip": _replace_instruction_block(
                self.dockerfile,
                "RUN",
                'RUN echo "python -m pip install --no-cache-dir --target /app /tmp/build"',
            ),
            "run echo groupadd": _replace_instruction_block(
                self.dockerfile, "RUN", 'RUN echo "groupadd --system app"'
            ),
            "run echo useradd": _replace_instruction_block(
                self.dockerfile, "RUN", 'RUN echo "useradd --system --gid app app"'
            ),
            "wrong module": _replace_cmd(
                self.dockerfile,
                expected_cmd.replace("app.api.main:app", "wrong.module:app"),
            ),
            "port command injection": _replace_cmd(
                self.dockerfile,
                expected_cmd.replace(
                    '--port "${PORT:-8080}"',
                    '--port 9999; echo --port "${PORT:-8080}"',
                ),
            ),
            "wrong workdir": self.dockerfile.replace("WORKDIR /app", "WORKDIR /wrong"),
            "broad env negation": self.dockerignore_lines + ["!**/.env*"],
            "broad data negation": self.dockerignore_lines + ["**/data"],
            "profile exclusion": self.dockerignore_lines + ["data/profile.json"],
            "comment-only copy": self.dockerfile.replace(
                "COPY app /tmp/build/app", "# COPY app /tmp/build/app"
            ),
            "later root user": self.dockerfile + "\nUSER root\n",
            "literal exec-form port": _replace_instruction_block(
                self.dockerfile,
                "CMD",
                'CMD ["python", "-m", "uvicorn", "app.api.main:app", '
                '"--host", "0.0.0.0", "--port", "${PORT:-8080}"]',
            ),
            "secret env": self.dockerfile + "\nENV SAFE=x OPENAI_API_KEY=secret\n",
            "root env negation": self.dockerignore_lines + ["!.env"],
            "slash root env negation": self.dockerignore_lines + ["!/.env"],
            "glob root env negation": self.dockerignore_lines + ["!**/.env"],
            "single quoted port": self.dockerfile.replace(
                r'\"${PORT:-8080}\"', "'${PORT:-8080}'"
            ),
            "shell comment command": _replace_cmd(
                self.dockerfile,
                'exec false # python -m uvicorn app.api.main:app --host 0.0.0.0 '
                '--port "${PORT:-8080}"',
            ),
            "final alpine stage": self.dockerfile + "\nFROM alpine:3.20\n",
            "missing pip": self.dockerfile.replace(
                "python -m pip install --no-cache-dir --target /app /tmp/build",
                "echo missing-pip",
            ),
            "missing groupadd": self.dockerfile.replace(
                "groupadd --system app", "echo missing-group"
            ),
            "missing useradd": self.dockerfile.replace(
                "useradd --system --gid app", "echo missing-user"
            ),
            "data directory exclusion": self.dockerignore_lines + ["data"],
            "setup before required copies": _move_setup_run_before_required_copies(
                self.dockerfile
            ),
            "entrypoint false": self.dockerfile + '\nENTRYPOINT ["false"]\n',
            "effective instruction after cmd": self.dockerfile
            + "\nRUN rm -rf /app/app\n",
            "specific env rule replaces wildcard": [
                ".env" if rule != ".env.*" else ".env.local"
                for rule in self.dockerignore_lines
            ],
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name):
                if isinstance(mutation, list):
                    mutated_dockerfile = self.dockerfile
                    mutated_dockerignore = mutation
                else:
                    mutated_dockerfile = mutation
                    mutated_dockerignore = self.dockerignore_lines
                with self.assertRaises(AssertionError):
                    validate_container_contract(mutated_dockerfile, mutated_dockerignore)


if __name__ == "__main__":
    unittest.main()
