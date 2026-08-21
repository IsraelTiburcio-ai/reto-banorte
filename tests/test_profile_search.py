from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.models.retrieval import SearchResult
from app.services.profile_service import ProfileService
from tests.test_profile_service import minimal_profile


def visibility_profile() -> dict[str, object]:
    profile = minimal_profile()
    profile["projects"] = [
        {
            "id": "visible-project",
            "visibility": "public",
            "name": "Visible project",
            "description": "Public project content",
            "hidden_do_not_expose": {
                "visibility": "do_not_expose",
                "text": "nested-do-not-expose-term",
            },
            "hidden_internal_summary": {
                "visibility": "internal_summary",
                "text": "nested-internal-summary-term",
            },
        },
        {
            "id": "hidden-project",
            "visibility": "do_not_expose",
            "name": "root-do-not-expose-term",
        },
        {
            "id": "internal-project",
            "visibility": "internal_summary",
            "name": "root-internal-summary-term",
        },
    ]
    profile["skills"] = [
        {
            "id": "hidden-relation-skill",
            "visibility": "do_not_expose",
            "name": "hidden-only-relation-term",
            "evidence": ["visible-project"],
        },
        {
            "id": "internal-relation-skill",
            "visibility": "internal_summary",
            "name": "internal-only-relation-term",
            "evidence": ["visible-project"],
        },
        {
            "id": "visible-skill",
            "visibility": "public",
            "name": "Visible skill",
        },
    ]
    return profile


def service_from_visibility_fixture(test_case: unittest.TestCase) -> ProfileService:
    temporary_directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(temporary_directory.cleanup)
    profile_path = Path(temporary_directory.name) / "profile.json"
    profile_path.write_text(
        json.dumps(visibility_profile()),
        encoding="utf-8",
    )
    return ProfileService(profile_path)


class ProfileSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ProfileService()

    def ids_for(self, query: str) -> set[str]:
        return {result.entity_id for result in self.service.search(query)}

    def test_search_result_model_is_explicit(self) -> None:
        results = self.service.search("MCP")
        self.assertTrue(results)
        self.assertTrue(all(isinstance(result, SearchResult) for result in results))
        self.assertTrue(all(result.entity_type for result in results))
        self.assertTrue(all(result.title for result in results))
        self.assertTrue(all(result.score > 0 for result in results))

    def test_search_is_case_insensitive(self) -> None:
        self.assertEqual(
            [result.entity_id for result in self.service.search("MCP")],
            [result.entity_id for result in self.service.search("mcp")],
        )

    def test_search_is_accent_tolerant(self) -> None:
        self.assertEqual(self.ids_for("optimizacion"), self.ids_for("optimización"))
        self.assertIn("career_story", self.ids_for("optimizacion"))

    def test_mcp_search_finds_related_content(self) -> None:
        result_ids = self.ids_for("MCP")
        self.assertIn("mcp-order-status", result_ids)
        self.assertIn("mcp-analytics", result_ids)
        self.assertIn("mcp", result_ids)
        self.assertIn("generative-ai", result_ids)

    def test_mcp_search_keeps_existing_top_ranking(self) -> None:
        results = self.service.search("MCP")
        self.assertEqual(
            [result.entity_id for result in results[:3]],
            ["mcp", "mcp-analytics", "mcp-order-status"],
        )

    def test_natural_language_mcp_query_finds_public_mcp_evidence(self) -> None:
        result_ids = self.ids_for("¿Qué experiencia tiene Israel con MCP?")
        self.assertIn("mcp", result_ids)
        self.assertIn("mcp-analytics", result_ids)
        self.assertIn("mcp-order-status", result_ids)

    def test_punctuation_falls_back_to_significant_terms(self) -> None:
        result_ids = self.ids_for("MCP?!,.")
        self.assertIn("mcp", result_ids)

    def test_accented_natural_language_query_finds_python(self) -> None:
        result_ids = self.ids_for("¿Qué experiencia tiene con Python?")
        self.assertIn("python", result_ids)

    def test_natural_language_python_and_fastapi_query_is_relevant(self) -> None:
        result_ids = self.ids_for("¿Qué experiencia tiene con Python y FastAPI?")
        self.assertIn("python", result_ids)
        self.assertIn("backend", result_ids)

    def test_stopword_only_query_returns_no_results(self) -> None:
        self.assertEqual(
            self.service.search("¿Qué experiencia tiene Israel con y sobre?"), []
        )

    def test_swift_search_finds_skills_experience_and_related_projects(self) -> None:
        result_ids = self.ids_for("Swift")
        self.assertIn("swift", result_ids)
        self.assertIn("swiftui", result_ids)
        self.assertIn("ios-development-lab", result_ids)
        self.assertIn("fi-fan", result_ids)

    def test_database_search_finds_postgresql_skill(self) -> None:
        result_ids = self.ids_for("PostgreSQL")
        self.assertIn("sql-and-databases", result_ids)

    def test_automation_search_finds_projects(self) -> None:
        result_ids = self.ids_for("automatización")
        self.assertIn("catalog-ai-automation", result_ids)
        self.assertIn("klaviyo-monetization", result_ids)

    def test_exact_id_has_priority(self) -> None:
        results = self.service.search("python")
        self.assertEqual(results[0].entity_id, "python")
        self.assertEqual(results[0].score, 100.0)
        self.assertIn("id", results[0].matched_fields)

    def test_empty_queries_return_no_results(self) -> None:
        self.assertEqual(self.service.search(""), [])
        self.assertEqual(self.service.search("   "), [])

    def test_search_does_not_return_hidden_entities(self) -> None:
        service = service_from_visibility_fixture(self)
        public_ids = {result.entity_id for result in service.search("term")}
        self.assertNotIn("hidden-project", public_ids)
        self.assertNotIn("internal-project", public_ids)

    def test_restricted_nested_text_cannot_match_public_search(self) -> None:
        service = service_from_visibility_fixture(self)
        do_not_expose_ids = {
            result.entity_id
            for result in service.search("nested-do-not-expose-term")
        }
        internal_summary_ids = {
            result.entity_id
            for result in service.search("nested-internal-summary-term")
        }
        self.assertNotIn("visible-project", do_not_expose_ids)
        self.assertNotIn("visible-project", internal_summary_ids)

    def test_internal_summary_nested_text_is_available_explicitly(self) -> None:
        service = service_from_visibility_fixture(self)
        result_ids = {
            result.entity_id
            for result in service.search(
                "nested-internal-summary-term",
                visibility="internal_summary",
            )
        }
        self.assertIn("visible-project", result_ids)
        do_not_expose_ids = {
            result.entity_id
            for result in service.search(
                "nested-do-not-expose-term",
                visibility="internal_summary",
            )
        }
        self.assertNotIn("visible-project", do_not_expose_ids)

    def test_hidden_relationships_cannot_rank_public_entities(self) -> None:
        service = service_from_visibility_fixture(self)
        public_hidden_relation_ids = {
            result.entity_id
            for result in service.search("hidden-only-relation-term")
        }
        self.assertNotIn("visible-project", public_hidden_relation_ids)
        self.assertNotIn("hidden-relation-skill", public_hidden_relation_ids)

        public_internal_relation_ids = {
            result.entity_id
            for result in service.search("internal-only-relation-term")
        }
        self.assertNotIn("visible-project", public_internal_relation_ids)

        internal_relation_ids = {
            result.entity_id
            for result in service.search(
                "internal-only-relation-term",
                visibility="internal_summary",
            )
        }
        self.assertIn("visible-project", internal_relation_ids)
        self.assertIn("internal-relation-skill", internal_relation_ids)


if __name__ == "__main__":
    unittest.main()
