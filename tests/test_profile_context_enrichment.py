from __future__ import annotations

import json
import unittest

from app.services.profile_service import ProfileService


class ProfileContextEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ProfileService()
        self.profile = self.service.get_profile("public")

    def project(self, project_id: str) -> dict[str, object]:
        projects = self.profile["projects"]
        return next(project for project in projects if project["id"] == project_id)

    def test_public_context_contains_structured_enrichment(self) -> None:
        self.assertIn("decision_making", self.profile["working_style"])
        self.assertIn("collaboration", self.profile["working_style"])
        self.assertIn("validation_practice", self.profile["working_style"])
        self.assertIn("professional_growth", self.profile)
        self.assertIn("strengths", self.profile["professional_growth"])
        self.assertIn("growth_areas", self.profile["professional_growth"])
        self.assertIn("professional_narratives", self.profile)
        self.assertIn("learnings", self.project("mcp-order-status"))
        self.assertIn("learnings", self.project("competitive-intelligence"))

    def test_public_context_keeps_visibility_and_excludes_secrets(self) -> None:
        serialized = json.dumps(self.profile, ensure_ascii=False).lower()

        self.assertNotIn("internal_summary", serialized)
        self.assertNotIn("do_not_expose", serialized)
        for marker in ("openai_api_key", "authorization: bearer", "token="):
            self.assertNotIn(marker, serialized)

    def test_mcp_order_status_preserves_direct_ownership_boundary(self) -> None:
        project = self.project("mcp-order-status")

        self.assertEqual(project["ownership"], "designed_and_developed")
        self.assertIn("Desarrolló directamente", project["role"])
        self.assertIn("Contact Center", " ".join(project["validation"]))
        self.assertIn("otros equipos", " ".join(project["claim_limits"]).lower())

    def test_claudia_remains_collaborative_not_solely_owned(self) -> None:
        project = self.project("claudia")

        self.assertEqual(project["ownership"], "participated_in_development_and_integration")
        self.assertIn("Catálogo", project["collaboration"])
        self.assertIn("DevOps", project["collaboration"])
        self.assertIn(
            "no se afirma que la haya desarrollado completamente",
            " ".join(project["claim_limits"]).lower(),
        )

    def test_fast_login_preserves_end_to_end_implementation_and_mentorship(self) -> None:
        project = self.project("otp-fast-login")

        self.assertEqual(project["ownership"], "designed_and_developed")
        self.assertIn("extremo a extremo", project["ownership_detail"])
        self.assertIn("Tech Lead", project["mentorship"])
        self.assertIn("rate limiting", project["security_focus"])
        self.assertIn("configuraciones internas", " ".join(project["claim_limits"]))

    def test_gcp_is_initial_practical_familiarity_not_expertise(self) -> None:
        gcp = next(skill for skill in self.profile["skills"] if skill["id"] == "gcp")

        self.assertEqual(gcp["level"], "basic_practical_familiarity")
        self.assertEqual(gcp["evidence_level"], "direct_project_experience")
        self.assertIn("desplegando este CV Agent", gcp["notes"][0])
        self.assertIn(
            "no como especialidad cloud",
            " ".join(gcp["claim_limits"]).lower(),
        )

    def test_unconfirmed_question_34_does_not_become_a_story(self) -> None:
        narratives = self.profile["professional_narratives"]

        self.assertNotIn("replanned_solution", narratives)
        self.assertIn(
            "No se incorpora una historia específica",
            " ".join(narratives["claim_limits"]),
        )


if __name__ == "__main__":
    unittest.main()
