"""Deterministic synthetic corpus generator for unit tests and smoke runs.

Designed so that:
- Many docs share text vocabulary (BM25 retrieves a large candidate set per query)
- Only a small number of docs are *truly* relevant (grade 2): spatial signal aligned
- A medium number of docs are *partially* relevant (grade 1): text-aligned only
- This forces BM25 to mis-order grade-2 vs grade-1 docs, giving SCAR room to win

This is NOT the LLM-simulated corpus from the paper. It exists only to:
1. validate that the scoring pipeline runs end-to-end
2. give LinUCB a reproducible playground for unit tests
3. provide a regression test corpus where SCAR > BM25 is provable
"""
from __future__ import annotations

import random
from typing import Sequence

from src.utils.types import (
    Corpus, Document, Query, Relevance, SpatialContext, ROLES, ZONE_TYPES, DOC_KINDS
)


def _person_id(i: int) -> str:
    return f"p{i:03d}"


# Each zone has a topic; query mentions the topic; only some docs include the topic word.
ZONE_TOPICS = {
    "meeting_room": "syncpoint",
    "focus_booth":  "specflow",
    "lounge":       "ideajam",
    "open_floor":   "crossteam",
    "library":      "litreview",
}

FILLER_VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi"]


def _make_doc_text(rng: random.Random, include_topic: bool, topic: str) -> str:
    body = " ".join(rng.choice(FILLER_VOCAB) for _ in range(12))
    if include_topic:
        # bury the topic word once, mid-text
        return f"{body[:30]} {topic} {body[30:]}"
    return body


def make_synth_corpus(
    n_persons: int = 40,
    n_docs: int = 200,
    n_queries: int = 30,
    seed: int = 42,
) -> Corpus:
    rng = random.Random(seed)
    persons = [_person_id(i) for i in range(n_persons)]

    documents: list[Document] = []
    for i in range(n_docs):
        author = rng.choice(persons)
        zone = rng.choice(ZONE_TYPES)
        kind = rng.choice(DOC_KINDS)
        day = rng.randint(0, 60)
        n_mentions = rng.randint(0, 2)
        mentions = tuple(
            rng.sample([p for p in persons if p != author], k=n_mentions)
        ) if n_mentions and n_persons > 1 else ()

        # 35% of docs include the zone-specific topic word; 65% are pure filler.
        # Of those that match a topic, the topic is the topic of THEIR own zone
        # (so a query in zone X retrieves docs from many zones via topic match
        # only if those docs happen to mention zone-X topic).
        include_topic = rng.random() < 0.35
        topic = ZONE_TOPICS[zone] if include_topic else ""
        text = _make_doc_text(rng, include_topic=include_topic, topic=topic)

        documents.append(
            Document(
                doc_id=f"d{i:03d}",
                text=text,
                author=author,
                location_type=zone,
                kind=kind,
                created_day=day,
                mentions=mentions,
            )
        )

    queries: list[Query] = []
    qrels: list[Relevance] = []
    today = 65

    for q_idx in range(n_queries):
        user = rng.choice(persons)
        role = rng.choice(ROLES)
        zone = rng.choice(ZONE_TYPES)
        topic = ZONE_TOPICS[zone]
        n_nearby = rng.randint(2, 5)
        nearby = tuple(rng.sample([p for p in persons if p != user], k=n_nearby))
        nearby_set = set(nearby)
        gaze_target = rng.choice(nearby) if rng.random() < 0.5 and nearby else None

        ctx = SpatialContext(
            user_id=user,
            role=role,
            current_zone=zone,
            current_day=today,
            nearby_persons=nearby,
            gaze_target=gaze_target,
        )
        # Query only includes the topic word — nothing else gives BM25 signal.
        query_text = f"please find {topic} now"
        query = Query(query_id=f"q{q_idx:03d}", text=query_text, context=ctx)
        queries.append(query)

        for d in documents:
            text_match = topic in d.text
            spatial_score = 0
            if d.author in nearby_set:
                spatial_score += 2
            if gaze_target is not None and (d.author == gaze_target or gaze_target in d.mentions):
                spatial_score += 2
            if d.location_type == zone:
                spatial_score += 1

            if text_match and spatial_score >= 2:
                grade = 2  # ideal: text + strong spatial alignment
            elif text_match:
                grade = 1  # partially relevant: text only
            elif spatial_score >= 3:
                # rare: spatial alignment is so strong that the doc is relevant
                # even without exact topic word (e.g. someone you're looking at
                # mentions a related person and you're in the right room).
                grade = 1
            else:
                grade = 0

            if grade > 0:
                qrels.append(Relevance(query_id=query.query_id, doc_id=d.doc_id, grade=grade))

    return Corpus(
        name="synth",
        documents=tuple(documents),
        queries=tuple(queries),
        qrels=tuple(qrels),
    )
