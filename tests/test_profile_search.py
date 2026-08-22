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

    def test_legacy_israel_query_keeps_existing_results(self) -> None:
        self.assertEqual(
            [result.entity_id for result in self.service.search("Israel")],
            [
                "professional_summary",
                "claudia",
                "farmacia-la-paz-shopify",
                "prixz-whatsapp-agent-challenge",
            ],
        )

    def test_legacy_experience_query_keeps_existing_result_count(self) -> None:
        self.assertEqual(len(self.service.search("experiencia")), 15)

    def test_natural_language_mcp_query_finds_public_mcp_evidence(self) -> None:
        result_ids = self.ids_for("¿Qué experiencia tiene Israel con MCP?")
        self.assertIn("mcp", result_ids)
        self.assertIn("mcp-analytics", result_ids)
        self.assertIn("mcp-order-status", result_ids)

    def test_career_overview_queries_prioritize_career_evidence(self) -> None:
        queries = (
            "¿Cómo decidió dedicarse a IA?",
            "¿Qué lo llevó hacia inteligencia artificial?",
            "Cuando Israel tomó la decisión de dedicarse a la inteligencia artificial",
            "¿Qué me puedes decir de su trayectoria?",
        )
        for query in queries:
            with self.subTest(query=query):
                results = self.service.search(query)
                self.assertTrue(results)
                self.assertEqual(results[0].entity_id, "career_story")
                self.assertIn("professional_summary", {item.entity_id for item in results})
                self.assertTrue(all(item.data.get("visibility") == "public" for item in results))

    def test_academic_project_overview_keeps_academic_distinction(self) -> None:
        results = self.service.search("¿Tiene proyectos académicos?")
        result_ids = [item.entity_id for item in results]
        self.assertIn("mba-yo", result_ids)
        self.assertTrue(all(item.entity_type == "project" for item in results))
        self.assertNotIn("docker", result_ids)

    def test_referential_academic_project_query_keeps_academic_distinction(self) -> None:
        result_ids = self.ids_for(
            "Cuéntame sobre sus proyectos ¿Cuál de esos fue académico?"
        )
        self.assertTrue({"fi-fan", "apapacho", "bimbo-run", "mba-yo"} <= result_ids)
        self.assertNotIn("docker", result_ids)

    def test_referential_professional_project_query_excludes_academic_projects(self) -> None:
        result_ids = self.ids_for(
            "Cuéntame de sus proyectos ¿Cuáles de esos fueron profesionales?"
        )
        self.assertIn("claudia", result_ids)
        self.assertNotIn("mba-yo", result_ids)
        self.assertNotIn("fi-fan", result_ids)

    def test_academic_project_markers_use_complete_tokens(self) -> None:
        profile = minimal_profile()
        profile["projects"] = [
            {
                "id": "tsunami",
                "visibility": "public",
                "name": "Tsunami forecasting",
                "description": "Forecasting project without academic context",
            },
            {
                "id": "unam-lab",
                "visibility": "public",
                "name": "iOS Development Lab UNAM",
                "description": "Academic project for estudiantes",
            },
            {
                "id": "academic-project",
                "visibility": "public",
                "name": "Proyecto académico",
                "description": "Proyecto academico de investigación",
            },
            {
                "id": "academic-experience",
                "visibility": "public",
                "name": "Experiencia académica",
                "description": "Contexto de aprendizaje",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            service = ProfileService(profile_path)

            academic_ids = {
                item.entity_id
                for item in service.search("¿Tiene proyectos académicos?")
            }
            professional_ids = {
                item.entity_id
                for item in service.search(
                    "¿Cuáles de sus proyectos fueron profesionales?"
                )
            }

        self.assertIn("unam-lab", academic_ids)
        self.assertIn("academic-project", academic_ids)
        self.assertIn("academic-experience", academic_ids)
        self.assertNotIn("tsunami", academic_ids)
        self.assertIn("tsunami", professional_ids)
        self.assertNotIn("unam-lab", professional_ids)
        self.assertNotIn("academic-project", professional_ids)
        self.assertNotIn("academic-experience", professional_ids)

    def test_project_overview_queries_return_projects_only(self) -> None:
        queries = (
            "¿Cuáles son los proyectos más relevantes de Israel?",
            "Dime sus proyectos principales",
            "¿Qué proyectos destacas de su trayectoria?",
            "What are Israel's main projects?",
        )
        for query in queries:
            with self.subTest(query=query):
                results = self.service.search(query)
                self.assertGreaterEqual(len(results), 4)
                self.assertTrue(all(item.entity_type == "project" for item in results))
                self.assertNotIn("docker", {item.entity_id for item in results})
                self.assertTrue(all(item.data.get("visibility") == "public" for item in results))

    def test_identity_overview_queries_return_identity_evidence(self) -> None:
        for query in (
            "¿Quién es Israel?",
            "Qué me puedes decir de Israel Tiburcio",
            "Tell me about Israel",
        ):
            with self.subTest(query=query):
                results = self.service.search(query)
                self.assertGreaterEqual(len(results), 3)
                self.assertEqual(results[0].entity_id, "identity")
                self.assertEqual(
                    [item.entity_id for item in results[:3]],
                    ["identity", "professional_summary", "career_story"],
                )

    def test_punctuation_falls_back_to_significant_terms(self) -> None:
        result_ids = self.ids_for("MCP?!,.")
        self.assertIn("mcp", result_ids)

    def test_accented_natural_language_query_finds_python(self) -> None:
        result_ids = self.ids_for("¿Qué experiencia tiene con Python?")
        self.assertIn("python", result_ids)

    def test_education_overview_queries_retrieve_public_education(self) -> None:
        for query in (
            "Cuéntame sobre su trayectoria escolar",
            "¿Cuál es su formación académica?",
            "¿Dónde estudió?",
        ):
            with self.subTest(query=query):
                results = self.service.search(query)
                result_ids = {item.entity_id for item in results}
                self.assertIn("unam-fes-acatlan-mac", result_ids)
                self.assertIn("enp-5-unam", result_ids)
                self.assertTrue(all(item.entity_type == "education" for item in results))

    def test_generic_cloud_queries_retrieve_calibrated_cloud_skills(self) -> None:
        for query in ("¿Tiene experiencia con la nube?", "¿Qué sabe de cloud computing?"):
            with self.subTest(query=query):
                result_ids = self.ids_for(query)
                self.assertTrue({"aws", "gcp", "oracle-cloud"} <= result_ids)
        aws_results = self.service.search("¿Tiene experiencia con AWS?")
        self.assertEqual([item.entity_id for item in aws_results[:1]], ["aws"])

    def test_natural_language_python_and_fastapi_query_is_relevant(self) -> None:
        results = self.service.search("¿Qué experiencia tiene con Python y FastAPI?")
        result_ids = {result.entity_id for result in results}
        self.assertIn("python", result_ids)
        self.assertIn("backend", result_ids)

    def test_multiple_significant_terms_rank_above_single_term(self) -> None:
        profile = minimal_profile()
        profile["projects"] = [
            {
                "id": "one-term",
                "visibility": "public",
                "name": "Python",
            },
            {
                "id": "two-terms",
                "visibility": "public",
                "name": "Python FastAPI",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            service = ProfileService(profile_path)

            results = service.search(
                "¿Qué experiencia tiene con Python y FastAPI?"
            )

        self.assertEqual(
            [result.entity_id for result in results],
            ["two-terms", "one-term"],
        )

    def test_noise_tokens_do_not_create_arbitrary_results(self) -> None:
        self.assertEqual(self.service.search("¿qué x?"), [])
        result_ids = self.ids_for("python x q 1")
        self.assertIn("python", result_ids)
        self.assertNotIn("mcp", result_ids)
        self.assertLessEqual(len(result_ids), 5)

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
