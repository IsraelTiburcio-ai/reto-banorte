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
    ]
    return profile


def service_from_visibility_fixture(test_case: unittest.TestCase) -> ProfileService:
    temporary_directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(temporary_directory.cleanup)
    profile_path = Path(temporary_directory.name) / "profile.json"
    profile_path.write_text(json.dumps(visibility_profile()), encoding="utf-8")
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
        self.assertTrue(all(result.entity_type and result.title for result in results))
        self.assertTrue(all(result.score > 0 for result in results))

    def test_search_is_case_and_accent_tolerant(self) -> None:
        self.assertEqual(
            [item.entity_id for item in self.service.search("MCP")],
            [item.entity_id for item in self.service.search("mcp")],
        )
        self.assertEqual(self.ids_for("optimizacion"), self.ids_for("optimización"))

    def test_exact_mcp_ranking_is_preserved(self) -> None:
        results = self.service.search("MCP")
        self.assertEqual(
            [item.entity_id for item in results[:3]],
            ["mcp", "mcp-analytics", "mcp-order-status"],
        )

    def test_natural_language_retrieval_is_optional_targeted_evidence(self) -> None:
        result_ids = self.ids_for("¿Qué experiencia tiene Israel con MCP?")
        self.assertTrue({"mcp", "mcp-analytics", "mcp-order-status"} <= result_ids)

    def test_punctuation_and_accents_do_not_block_tokens(self) -> None:
        self.assertIn("mcp", self.ids_for("MCP?!,."))
        self.assertIn("python", self.ids_for("¿Qué experiencia tiene con Python?"))

    def test_multiple_explicit_terms_rank_above_single_term(self) -> None:
        profile = minimal_profile()
        profile["projects"] = [
            {"id": "one-term", "visibility": "public", "name": "Python"},
            {
                "id": "two-terms",
                "visibility": "public",
                "name": "Python FastAPI",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            results = ProfileService(profile_path).search("Python FastAPI")
        self.assertEqual([item.entity_id for item in results], ["two-terms"])

    def test_noise_only_query_is_not_a_universal_agent_gate(self) -> None:
        self.assertEqual(self.service.search("x"), [])
        self.assertEqual(self.service.search("no such profile topic"), [])

    def test_visibility_filtering_applies_before_matching(self) -> None:
        service = service_from_visibility_fixture(self)
        public_ids = {item.entity_id for item in service.search("term")}
        self.assertNotIn("hidden-project", public_ids)
        self.assertNotIn("internal-project", public_ids)

    def test_nested_restricted_text_cannot_match_public_search(self) -> None:
        service = service_from_visibility_fixture(self)
        self.assertEqual(service.search("nested-do-not-expose-term"), [])
        self.assertEqual(service.search("nested-internal-summary-term"), [])
        self.assertEqual(
            service.search("nested-internal-summary-term", visibility="internal_summary")[0].entity_id,
            "visible-project",
        )

    def test_hidden_relationships_do_not_create_public_side_channels(self) -> None:
        service = service_from_visibility_fixture(self)
        for query in ("hidden-only-relation-term", "internal-only-relation-term"):
            with self.subTest(query=query):
                self.assertEqual(service.search(query), [])

    def test_public_relationships_can_still_rank_entities(self) -> None:
        profile = minimal_profile()
        profile["experience"] = [
            {
                "id": "public-experience",
                "visibility": "public",
                "name": "Public experience",
                "project_ids": ["project-public"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            service = ProfileService(path)
            self.assertIn("project-public", {item.entity_id for item in service.search("Public experience")})

    def test_exact_id_has_priority(self) -> None:
        results = self.service.search("python")
        self.assertEqual(results[0].entity_id, "python")
        self.assertEqual(results[0].score, 100.0)
        self.assertIn("id", results[0].matched_fields)

    def test_empty_and_unknown_queries_return_no_results(self) -> None:
        self.assertEqual(self.service.search(""), [])
        self.assertEqual(self.service.search("   "), [])
        self.assertEqual(self.service.search("no-such-id"), [])


if __name__ == "__main__":
    unittest.main()
