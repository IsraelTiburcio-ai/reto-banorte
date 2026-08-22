from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ContainerizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        cls.dockerignore_lines = (ROOT / ".dockerignore").read_text(
            encoding="utf-8"
        ).splitlines()

    def test_dockerfile_uses_pinned_slim_python_and_runtime_install(self) -> None:
        self.assertIn("FROM python:3.11-slim-bookworm", self.dockerfile)
        self.assertIn("COPY pyproject.toml", self.dockerfile)
        self.assertIn("COPY app", self.dockerfile)
        self.assertIn("COPY data/profile.json", self.dockerfile)
        self.assertIn("--no-cache-dir", self.dockerfile)
        self.assertIn("--target /app", self.dockerfile)
        self.assertNotRegex(self.dockerfile, r"(?m)^\s*COPY\s+\.\s")

    def test_dockerfile_has_non_root_port_and_signal_safe_runtime(self) -> None:
        self.assertIn("USER app:app", self.dockerfile)
        self.assertIn("EXPOSE 8080", self.dockerfile)
        self.assertIn("--host 0.0.0.0", self.dockerfile)
        self.assertIn('${PORT:-8080}', self.dockerfile)
        self.assertIn('exec python -m uvicorn', self.dockerfile)
        self.assertNotIn("--reload", self.dockerfile)

    def test_dockerfile_does_not_declare_secret_inputs(self) -> None:
        self.assertNotRegex(
            self.dockerfile,
            r"(?mi)^\s*(ARG|ENV)\s+(OPENAI_API_KEY|AGENT_API_KEY)\b",
        )
        self.assertNotIn("COPY .env", self.dockerfile)

    def test_dockerignore_excludes_local_and_build_only_content(self) -> None:
        ignored = set(self.dockerignore_lines)
        self.assertIn(".env", ignored)
        self.assertIn(".env.*", ignored)
        self.assertIn("!.env.example", ignored)
        self.assertIn("tests", ignored)
        self.assertIn("evals", ignored)
        self.assertNotIn("data", ignored)
        self.assertNotIn("data/profile.json", ignored)


if __name__ == "__main__":
    unittest.main()
