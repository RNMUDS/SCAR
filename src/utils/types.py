"""Core data types for SCAR.

Designed for immutability: all containers are frozen dataclasses.
Operations on these structures must return new objects, not mutate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


ZONE_TYPES = ("meeting_room", "focus_booth", "lounge", "open_floor", "library")
ROLES = ("engineer", "designer", "pm", "exec", "intern")
DOC_KINDS = ("meeting_log", "spec_doc", "brainstorm_note", "code_review", "other")


@dataclass(frozen=True)
class Document:
    """A retrievable document in the virtual office."""
    doc_id: str
    text: str
    author: str                     # person id
    location_type: str              # one of ZONE_TYPES
    kind: str                       # one of DOC_KINDS
    created_day: int                # integer day index from corpus epoch
    mentions: tuple[str, ...] = ()  # person ids mentioned in text


@dataclass(frozen=True)
class SpatialContext:
    """The user's spatial state at query time."""
    user_id: str
    role: str                       # one of ROLES
    current_zone: str               # one of ZONE_TYPES
    current_day: int
    nearby_persons: tuple[str, ...] = ()   # within proximity threshold now
    gaze_target: Optional[str] = None      # person currently looked at, if any


@dataclass(frozen=True)
class Query:
    """A retrieval query."""
    query_id: str
    text: str
    context: SpatialContext


@dataclass(frozen=True)
class Relevance:
    """Ground-truth relevance grade for a (query, document) pair.

    Grades follow standard nDCG conventions:
        2 = highly relevant
        1 = partially relevant
        0 = not relevant
    """
    query_id: str
    doc_id: str
    grade: int


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved document with its scores."""
    doc_id: str
    final_score: float
    text_sim: float
    prox_bonus: float
    gaze_bonus: float
    loc_bonus: float
    time_decay: float


@dataclass(frozen=True)
class Corpus:
    """A collection of documents, queries, and relevance judgments."""
    name: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    qrels: tuple[Relevance, ...] = field(default_factory=tuple)

    def doc_by_id(self, doc_id: str) -> Document:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        raise KeyError(doc_id)

    def relevance_map(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self.qrels:
            out.setdefault(r.query_id, {})[r.doc_id] = r.grade
        return out
