"""Small public models used by the deterministic profile retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VisibilityPolicy = Literal["public", "internal_summary"]


@dataclass(frozen=True)
class SearchResult:
    """A ranked, explainable result returned by lexical profile search."""

    entity_type: str
    entity_id: str
    title: str
    score: float
    matched_fields: tuple[str, ...]
    data: dict[str, object]
