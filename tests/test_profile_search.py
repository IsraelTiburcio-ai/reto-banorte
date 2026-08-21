from __future__ import annotations

import unittest

from app.models.retrieval import SearchResult
from app.services.profile_service import ProfileService


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
        self.assertTrue(
            all(
                result.entity_id not in {"internal-project", "hidden-project"}
                for result in self.service.search("Internal")
            )
        )


if __name__ == "__main__":
    unittest.main()
