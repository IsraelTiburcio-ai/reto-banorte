"""Deterministic loading, lookup, visibility filtering, and search for the CV."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Iterator
from json import JSONDecodeError
from pathlib import Path
from typing import cast

from app.models.retrieval import SearchResult, VisibilityPolicy


ProfileMapping = dict[str, object]
_OMIT = object()


class ProfileServiceError(Exception):
    """Base error for profile loading and retrieval failures."""


class ProfileLoadError(ProfileServiceError):
    """Raised when the profile file cannot be read or parsed as JSON."""


class ProfileValidationError(ProfileServiceError):
    """Raised when the profile does not meet the minimum retrieval structure."""


class EntityNotFoundError(ProfileServiceError):
    """Raised when an entity is absent or not visible under the active policy."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        super().__init__(f"{entity_type} with id '{entity_id}' was not found")
        self.entity_type = entity_type
        self.entity_id = entity_id


class ProfileService:
    """Load and query the canonical professional profile without external services.

    The default policy exposes only entities explicitly marked ``public``.
    Entities marked ``internal_summary`` can be requested explicitly by a
    trusted caller, while ``do_not_expose`` and missing visibility are never
    returned by this service.
    """

    _COLLECTIONS: tuple[tuple[str, str], ...] = (
        ("experience", "experience"),
        ("projects", "project"),
        ("skills", "skill"),
        ("knowledge_areas", "knowledge_area"),
        ("training", "training"),
        ("achievements", "achievement"),
    )
    _ROOT_DOCUMENTS: tuple[tuple[str, str], ...] = (
        ("professional_summary", "Professional summary"),
        ("career_story", "Career story"),
        ("working_style", "Working style"),
    )
    _REQUIRED_SECTIONS: tuple[str, ...] = (
        "metadata",
        "identity",
        "professional_summary",
        "career_story",
        "experience",
        "projects",
        "education",
        "training",
        "achievements",
        "hackathons",
        "skills",
        "knowledge_areas",
        "working_style",
        "agent_policies",
    )
    _CONTEXT_KEYS = {
        "activities",
        "architecture",
        "category",
        "challenge",
        "contexts",
        "current_focus",
        "data_fields",
        "event",
        "event_context",
        "evidence",
        "focus_areas",
        "knowledge_areas",
        "notable_projects",
        "objective",
        "organization",
        "project_ids",
        "responsibilities",
        "role",
        "skills_developed",
        "sources_worked_with",
        "technologies",
        "topics",
        "tools_used",
        "workflow",
    }
    _QUERY_STOPWORDS = {
        "a",
        "about",
        "al",
        "and",
        "con",
        "cual",
        "cuales",
        "cuál",
        "cuáles",
        "de",
        "del",
        "dime",
        "el",
        "en",
        "es",
        "experience",
        "experiencia",
        "for",
        "has",
        "have",
        "how",
        "israel",
        "la",
        "las",
        "los",
        "me",
        "of",
        "por",
        "please",
        "qué",
        "que",
        "sobre",
        "tell",
        "tengo",
        "tiene",
        "the",
        "this",
        "what",
        "with",
        "y",
    }

    def __init__(self, profile_path: str | Path | None = None) -> None:
        self.profile_path = (
            Path(profile_path).expanduser().resolve()
            if profile_path is not None
            else Path(__file__).resolve().parents[2] / "data" / "profile.json"
        )
        self._profile = self._load_profile()
        self._validate_profile(self._profile)
        self._indexes = self._build_indexes(self._profile)

    def get_profile(
        self, visibility: VisibilityPolicy = "public"
    ) -> ProfileMapping:
        """Return a deep, visibility-filtered copy of the complete profile."""

        self._validate_policy(visibility)
        return self._filter_root(self._profile, visibility)

    def get_experience(
        self, entity_id: str, visibility: VisibilityPolicy = "public"
    ) -> ProfileMapping:
        """Return one experience by stable ID or raise ``EntityNotFoundError``."""

        return self._get_entity("experience", entity_id, visibility)

    def get_project(
        self, entity_id: str, visibility: VisibilityPolicy = "public"
    ) -> ProfileMapping:
        """Return one project by stable ID or raise ``EntityNotFoundError``."""

        return self._get_entity("project", entity_id, visibility)

    def get_skill(
        self, entity_id: str, visibility: VisibilityPolicy = "public"
    ) -> ProfileMapping:
        """Return one skill by stable ID or raise ``EntityNotFoundError``."""

        return self._get_entity("skill", entity_id, visibility)

    def search(
        self, query: str, visibility: VisibilityPolicy = "public"
    ) -> list[SearchResult]:
        """Search public profile content lexically with deterministic ranking."""

        self._validate_policy(visibility)
        normalized_query = self._normalize(query)
        if not normalized_query:
            return []
        query_terms = self._meaningful_query_terms(query)
        if not query_terms:
            return []

        relationship_terms = self._build_relationship_terms(visibility)
        results: list[SearchResult] = []
        for entity_type, entity_id, entity in self._iter_search_entities():
            visible_entity = self._visible_entity(entity, visibility)
            if visible_entity is None:
                continue
            score, matched_fields = self._score_entity(
                entity_type,
                entity_id,
                visible_entity,
                normalized_query,
                relationship_terms,
            )
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    title=self._title_for(visible_entity, entity_id),
                    score=score,
                    matched_fields=matched_fields,
                    data=visible_entity,
                )
            )

        results.sort(key=lambda item: (-item.score, item.entity_type, item.entity_id))
        if results:
            return results

        # Natural-language fallback: exact and substring matching above remain
        # authoritative for existing queries; token matching only helps when a
        # full question is not itself present in a profile field. Compact
        # hyphenated identifiers keep their existing lookup semantics instead
        # of being expanded into unrelated word matches.
        if "-" in normalized_query and " " not in normalized_query:
            return []

        token_results: list[SearchResult] = []
        for entity_type, entity_id, entity in self._iter_search_entities():
            visible_entity = self._visible_entity(entity, visibility)
            if visible_entity is None:
                continue
            score, matched_fields = self._score_entity_tokens(
                entity_type,
                entity_id,
                visible_entity,
                query_terms,
                relationship_terms,
            )
            # Conservative fallback: token hits in descriptive body fields
            # alone are too weak to turn arbitrary question words into
            # results. Existing exact/substring matching above still keeps
            # its full scoring behavior for those fields.
            if score < 55.0:
                continue
            token_results.append(
                SearchResult(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    title=self._title_for(visible_entity, entity_id),
                    score=score,
                    matched_fields=matched_fields,
                    data=visible_entity,
                )
            )

        token_results.sort(
            key=lambda item: (-item.score, item.entity_type, item.entity_id)
        )
        return token_results

    def _load_profile(self) -> ProfileMapping:
        try:
            raw_profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProfileLoadError(
                f"Profile file does not exist: {self.profile_path}"
            ) from exc
        except OSError as exc:
            raise ProfileLoadError(
                f"Could not read profile file {self.profile_path}: {exc}"
            ) from exc
        except JSONDecodeError as exc:
            raise ProfileLoadError(
                f"Invalid JSON in profile file {self.profile_path} at line {exc.lineno}, column {exc.colno}"
            ) from exc

        if not isinstance(raw_profile, dict):
            raise ProfileValidationError("Profile root must be a JSON object")
        return cast(ProfileMapping, raw_profile)

    @classmethod
    def _validate_profile(cls, profile: ProfileMapping) -> None:
        missing_sections = [
            section for section in cls._REQUIRED_SECTIONS if section not in profile
        ]
        if missing_sections:
            missing = ", ".join(missing_sections)
            raise ProfileValidationError(f"Profile is missing sections: {missing}")

        for section, _entity_type in cls._COLLECTIONS:
            value = profile.get(section)
            if not isinstance(value, list):
                raise ProfileValidationError(
                    f"Profile section '{section}' must be a list"
                )

        all_ids: dict[str, str] = {}
        for section, _entity_type in cls._COLLECTIONS:
            entities = cast(list[object], profile[section])
            for index, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    raise ProfileValidationError(
                        f"Profile section '{section}' item {index} must be an object"
                    )
                entity_id = entity.get("id")
                if not isinstance(entity_id, str) or not entity_id.strip():
                    raise ProfileValidationError(
                        f"Profile section '{section}' item {index} requires a non-empty string id"
                    )
                previous_section = all_ids.get(entity_id)
                if previous_section is not None:
                    raise ProfileValidationError(
                        f"Duplicate entity id '{entity_id}' in '{section}' and '{previous_section}'"
                    )
                all_ids[entity_id] = section

    @classmethod
    def _build_indexes(
        cls, profile: ProfileMapping
    ) -> dict[str, dict[str, ProfileMapping]]:
        indexes: dict[str, dict[str, ProfileMapping]] = {}
        for section, entity_type in cls._COLLECTIONS:
            entities = cast(list[object], profile[section])
            indexes[entity_type] = {
                cast(str, cast(dict[str, object], entity)["id"]): cast(
                    ProfileMapping, entity
                )
                for entity in entities
            }
        return indexes

    def _get_entity(
        self,
        entity_type: str,
        entity_id: str,
        visibility: VisibilityPolicy,
    ) -> ProfileMapping:
        self._validate_policy(visibility)
        entity = self._indexes.get(entity_type, {}).get(entity_id)
        if entity is None or not self._is_visible(entity, visibility):
            raise EntityNotFoundError(entity_type, entity_id)
        return cast(ProfileMapping, self._copy_nested(entity, visibility))

    def _visible_entity(
        self, entity: ProfileMapping, visibility: VisibilityPolicy
    ) -> ProfileMapping | None:
        """Return the entity copy that is allowed to participate in a search."""

        if not self._is_visible(entity, visibility):
            return None
        copied = self._copy_nested(entity, visibility)
        if copied is _OMIT or not isinstance(copied, dict):
            return None
        return cast(ProfileMapping, copied)

    @staticmethod
    def _validate_policy(visibility: VisibilityPolicy) -> None:
        if visibility not in ("public", "internal_summary"):
            raise ValueError(
                "visibility must be 'public' or 'internal_summary'; 'do_not_expose' is never allowed"
            )

    @staticmethod
    def _is_visible(
        entity: ProfileMapping, visibility: VisibilityPolicy
    ) -> bool:
        actual_visibility = entity.get("visibility")
        if actual_visibility == "do_not_expose":
            return False
        if actual_visibility == "public":
            return True
        return actual_visibility == "internal_summary" and visibility == "internal_summary"

    def _filter_root(
        self, profile: ProfileMapping, visibility: VisibilityPolicy
    ) -> ProfileMapping:
        result: ProfileMapping = {}
        collection_sections = {section for section, _ in self._COLLECTIONS}

        for key, value in profile.items():
            if key in collection_sections:
                if not isinstance(value, list):
                    continue
                visible_items = [
                    self._copy_nested(item, visibility)
                    for item in value
                    if isinstance(item, dict)
                    and self._is_visible(cast(ProfileMapping, item), visibility)
                ]
                result[key] = cast(list[object], visible_items)
            elif key == "education" and isinstance(value, dict):
                filtered_education = self._filter_education(
                    cast(ProfileMapping, value), visibility
                )
                if filtered_education:
                    result[key] = filtered_education
            elif isinstance(value, dict):
                mapping = cast(ProfileMapping, value)
                if self._is_visible(mapping, visibility):
                    result[key] = cast(
                        ProfileMapping, self._copy_nested(mapping, visibility)
                    )
            elif isinstance(value, list):
                # Lists without explicitly visible entities are omitted
                # conservatively, e.g. agent_policies in the current profile.
                visible_items = [
                    self._copy_nested(item, visibility)
                    for item in value
                    if isinstance(item, dict)
                    and self._is_visible(cast(ProfileMapping, item), visibility)
                ]
                if visible_items:
                    result[key] = cast(list[object], visible_items)

        return result

    def _filter_education(
        self, education: ProfileMapping, visibility: VisibilityPolicy
    ) -> ProfileMapping:
        result: ProfileMapping = {}
        formal = education.get("formal")
        if isinstance(formal, list):
            result["formal"] = [
                self._copy_nested(item, visibility)
                for item in formal
                if isinstance(item, dict)
                and self._is_visible(cast(ProfileMapping, item), visibility)
            ]
        academic_origin = education.get("academic_origin")
        if isinstance(academic_origin, dict) and self._is_visible(
            cast(ProfileMapping, academic_origin), visibility
        ):
            result["academic_origin"] = self._copy_nested(
                academic_origin, visibility
            )
        return result

    def _copy_nested(
        self, value: object, visibility: VisibilityPolicy
    ) -> object:
        if isinstance(value, dict):
            mapping = cast(ProfileMapping, value)
            if "visibility" in mapping and not self._is_visible(mapping, visibility):
                return _OMIT
            copied: ProfileMapping = {}
            for key, child in mapping.items():
                filtered_child = self._copy_nested(child, visibility)
                if filtered_child is not _OMIT:
                    copied[key] = filtered_child
            return copied
        if isinstance(value, list):
            copied_list: list[object] = []
            for child in value:
                filtered_child = self._copy_nested(child, visibility)
                if filtered_child is not _OMIT:
                    copied_list.append(filtered_child)
            return copied_list
        return copy.deepcopy(value)

    def _iter_search_entities(
        self,
    ) -> Iterator[tuple[str, str, ProfileMapping]]:
        for entity_type, entities in self._indexes.items():
            for entity_id, entity in entities.items():
                yield entity_type, entity_id, entity
        for key, _title in self._ROOT_DOCUMENTS:
            value = self._profile.get(key)
            if isinstance(value, dict):
                yield "document", key, cast(ProfileMapping, value)

    def _build_relationship_terms(
        self, visibility: VisibilityPolicy
    ) -> dict[tuple[str, str], list[str]]:
        terms: dict[tuple[str, str], list[str]] = {}

        def add(target_id: object, term: object) -> None:
            if not isinstance(target_id, str) or not isinstance(term, str):
                return
            target_type = self._entity_type_for_id(target_id)
            if target_type is None:
                return
            key = (target_type, target_id)
            terms.setdefault(key, []).append(term)

        for entity_type, entity_id, entity in self._iter_search_entities():
            visible_entity = self._visible_entity(entity, visibility)
            if visible_entity is None:
                continue
            if entity_type == "experience":
                for project_id in self._string_list(
                    visible_entity.get("project_ids")
                ):
                    add(project_id, self._title_for(visible_entity, entity_id))
            if entity_type == "skill":
                for reference in self._string_list(visible_entity.get("evidence")):
                    add(reference, self._title_for(visible_entity, entity_id))
            if entity_type == "achievement":
                add(
                    visible_entity.get("project_id"),
                    self._title_for(visible_entity, entity_id),
                )

        return terms

    def _entity_type_for_id(self, entity_id: str) -> str | None:
        for entity_type, entities in self._indexes.items():
            if entity_id in entities:
                return entity_type
        return None

    def _score_entity(
        self,
        entity_type: str,
        entity_id: str,
        entity: ProfileMapping,
        normalized_query: str,
        relationship_terms: dict[tuple[str, str], list[str]],
    ) -> tuple[float, tuple[str, ...]]:
        buckets: list[tuple[str, list[str], float, float]] = [
            ("id", [entity_id], 100.0, 90.0),
            (
                "name/title",
                self._values_for_keys(entity, {"name", "title"}),
                85.0,
                80.0,
            ),
        ]

        context_values: list[tuple[str, list[str]]] = []
        body_values: list[tuple[str, list[str]]] = []
        for key, value in entity.items():
            if key in {"id", "name", "title", "visibility", "evidence_level"}:
                continue
            values = self._flatten_strings(value)
            if not values:
                continue
            target = context_values if key in self._CONTEXT_KEYS else body_values
            target.append((key, values))

        relation_terms = relationship_terms.get((entity_type, entity_id), [])
        if relation_terms:
            context_values.append(("related", relation_terms))

        for key, values in context_values:
            buckets.append((key, values, 65.0, 55.0))
        for key, values in body_values:
            buckets.append((key, values, 35.0, 25.0))

        score = 0.0
        matched: list[str] = []
        for label, values, exact_score, contains_score in buckets:
            bucket_score = self._score_values(
                values, normalized_query, exact_score, contains_score
            )
            if bucket_score > 0:
                score = max(score, bucket_score)
                matched.append(label)
        return score, tuple(matched)

    @staticmethod
    def _score_values(
        values: list[str],
        normalized_query: str,
        exact_score: float,
        contains_score: float,
    ) -> float:
        best = 0.0
        for value in values:
            normalized_value = ProfileService._normalize(value)
            if not normalized_value:
                continue
            if normalized_value == normalized_query:
                best = max(best, exact_score)
            elif normalized_query in normalized_value:
                best = max(best, contains_score)
        return best

    def _score_entity_tokens(
        self,
        entity_type: str,
        entity_id: str,
        entity: ProfileMapping,
        query_terms: list[str],
        relationship_terms: dict[tuple[str, str], list[str]],
    ) -> tuple[float, tuple[str, ...]]:
        score = 0.0
        matched: list[str] = []
        for term in query_terms:
            term_score, term_fields = self._score_entity(
                entity_type,
                entity_id,
                entity,
                term,
                relationship_terms,
            )
            score = max(score, term_score)
            for field in term_fields:
                if field not in matched:
                    matched.append(field)
        return score, tuple(matched)

    @classmethod
    def _meaningful_query_terms(cls, value: str) -> list[str]:
        normalized = cls._normalize(value)
        tokenized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
        return [
            token
            for token in tokenized.split()
            if token and token not in cls._QUERY_STOPWORDS
        ]

    @staticmethod
    def _normalize(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value)
        without_accents = "".join(
            character for character in decomposed if not unicodedata.combining(character)
        )
        return " ".join(without_accents.casefold().split())

    @staticmethod
    def _flatten_strings(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            values: list[str] = []
            for key, child in value.items():
                if key in {"visibility", "evidence_level"}:
                    continue
                values.extend(ProfileService._flatten_strings(child))
            return values
        if isinstance(value, list):
            values = []
            for child in value:
                values.extend(ProfileService._flatten_strings(child))
            return values
        return []

    @staticmethod
    def _values_for_keys(
        entity: ProfileMapping, keys: set[str]
    ) -> list[str]:
        values: list[str] = []
        for key in keys:
            value = entity.get(key)
            if isinstance(value, str):
                values.append(value)
        return values

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _title_for(entity: ProfileMapping, entity_id: str) -> str:
        for key in ("name", "title", "organization", "program"):
            value = entity.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return entity_id
