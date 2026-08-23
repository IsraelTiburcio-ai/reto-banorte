from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from app.services.profile_service import (
    EntityNotFoundError,
    ProfileLoadError,
    ProfileService,
    ProfileValidationError,
)


def minimal_profile() -> dict[str, object]:
    return {
        "metadata": {"visibility": "public"},
        "identity": {"visibility": "public", "full_name": "Test Profile"},
        "professional_summary": {
            "visibility": "public",
            "profile": "A deterministic test profile",
        },
        "career_story": {"visibility": "public", "narrative": "Testing"},
        "experience": [
            {"id": "experience-public", "visibility": "public", "name": "Public"},
        ],
        "projects": [
            {"id": "project-public", "visibility": "public", "name": "Public"},
        ],
        "education": {
            "formal": [
                {"id": "education-public", "visibility": "public", "name": "Public"}
            ]
        },
        "training": [
            {"id": "training-public", "visibility": "public", "name": "Public"}
        ],
        "achievements": [
            {
                "id": "achievement-public",
                "visibility": "public",
                "name": "Public",
            }
        ],
        "hackathons": {"visibility": "public", "activities": []},
        "skills": [
            {
                "id": "skill-public",
                "visibility": "public",
                "name": "Public",
            }
        ],
        "knowledge_areas": [
            {
                "id": "area-public",
                "visibility": "public",
                "name": "Public",
            }
        ],
        "working_style": {"visibility": "public", "principles": []},
        "agent_policies": [],
    }


class ProfileServiceTests(unittest.TestCase):
    def write_profile(self, profile: dict[str, object]) -> Path:
        directory = Path(self.temp_dir.name)
        path = directory / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_path_loads_independently_of_current_directory(self) -> None:
        original_directory = Path.cwd()
        try:
            os.chdir(self.temp_dir.name)
            service = ProfileService()
        finally:
            os.chdir(original_directory)

        self.assertEqual(service.get_experience("prixz")["id"], "prixz")

    def test_injected_path_loads_profile(self) -> None:
        service = ProfileService(self.write_profile(minimal_profile()))
        self.assertEqual(service.get_project("project-public")["name"], "Public")

    def test_missing_file_fails_with_profile_load_error(self) -> None:
        with self.assertRaises(ProfileLoadError) as context:
            ProfileService(Path(self.temp_dir.name) / "missing.json")
        self.assertIn("does not exist", str(context.exception))

    def test_invalid_json_fails_with_profile_load_error(self) -> None:
        path = Path(self.temp_dir.name) / "invalid.json"
        path.write_text("{invalid", encoding="utf-8")
        with self.assertRaises(ProfileLoadError) as context:
            ProfileService(path)
        self.assertIn("Invalid JSON", str(context.exception))

    def test_missing_required_section_fails_validation(self) -> None:
        profile = minimal_profile()
        del profile["projects"]
        with self.assertRaises(ProfileValidationError) as context:
            ProfileService(self.write_profile(profile))
        self.assertIn("projects", str(context.exception))

    def test_duplicate_ids_fail_validation(self) -> None:
        profile = minimal_profile()
        profile["projects"] = [
            {"id": "duplicate", "visibility": "public"},
            {"id": "duplicate", "visibility": "public"},
        ]
        with self.assertRaises(ProfileValidationError) as context:
            ProfileService(self.write_profile(profile))
        self.assertIn("Duplicate entity id 'duplicate'", str(context.exception))

    def test_get_experience_by_id(self) -> None:
        service = ProfileService()
        self.assertEqual(service.get_experience("prixz")["organization"], "Prixz")
        self.assertEqual(
            service.get_experience("ios-development-lab")["role"],
            "Mentor y Soporte Técnico / Servicio Social",
        )

    def test_missing_experience_is_explicit(self) -> None:
        with self.assertRaises(EntityNotFoundError):
            ProfileService().get_experience("does-not-exist")

    def test_get_required_projects(self) -> None:
        service = ProfileService()
        for project_id in ("claudia", "otp-fast-login", "mcp-order-status", "apapacho"):
            self.assertEqual(service.get_project(project_id)["id"], project_id)

    def test_get_skills(self) -> None:
        service = ProfileService()
        python_skill = service.get_skill("python")
        self.assertEqual(python_skill["level"], "strong_practical")
        self.assertIn("contexts", python_skill)
        self.assertEqual(
            service.get_skill("sql-and-databases")["name"], "SQL y bases de datos"
        )

    def test_profile_visibility_is_conservative(self) -> None:
        profile = minimal_profile()
        profile["projects"] = [
            {"id": "public-project", "visibility": "public", "name": "Public"},
            {
                "id": "internal-project",
                "visibility": "internal_summary",
                "name": "Internal",
            },
            {
                "id": "hidden-project",
                "visibility": "do_not_expose",
                "name": "Hidden",
            },
            {"id": "missing-visibility-project", "name": "Missing"},
        ]
        service = ProfileService(self.write_profile(profile))

        public_profile = service.get_profile()
        public_ids = {
            item["id"] for item in public_profile["projects"] if isinstance(item, dict)
        }
        self.assertEqual(public_ids, {"public-project"})
        with self.assertRaises(EntityNotFoundError):
            service.get_project("internal-project")
        with self.assertRaises(EntityNotFoundError):
            service.get_project("missing-visibility-project")
        self.assertEqual(
            service.get_project("internal-project", visibility="internal_summary")["name"],
            "Internal",
        )
        with self.assertRaises(EntityNotFoundError):
            service.get_project("hidden-project", visibility="internal_summary")

    def test_results_are_deep_copies(self) -> None:
        service = ProfileService()
        result = service.get_project("claudia")
        result["name"] = "Changed locally"
        capabilities = result["capabilities_related"]
        self.assertIsInstance(capabilities, list)
        capabilities.append("Changed locally")

        fresh_result = service.get_project("claudia")
        self.assertEqual(fresh_result["name"], "ClaudIA")
        self.assertNotIn("Changed locally", fresh_result["capabilities_related"])

    def test_profile_result_is_independent(self) -> None:
        service = ProfileService()
        profile = service.get_profile()
        projects = profile["projects"]
        self.assertIsInstance(projects, list)
        claudia = next(item for item in projects if item["id"] == "claudia")
        claudia["name"] = "Changed locally"
        self.assertEqual(service.get_project("claudia")["name"], "ClaudIA")


if __name__ == "__main__":
    unittest.main()
