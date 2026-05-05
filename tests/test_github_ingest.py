"""Tests for src/corpus/github_ingest.py — uses fixture data, no network."""
from src.corpus.github_ingest import (
    CATEGORY_TO_ZONE, _extract_mentions, _zone_for_category, discussions_to_corpus,
)


def test_zone_mapping_known():
    assert _zone_for_category("Q&A") == "library"
    assert _zone_for_category("General") == "lounge"


def test_zone_mapping_unknown_falls_back():
    assert _zone_for_category("Random Garbage") == "open_floor"
    assert _zone_for_category(None) == "open_floor"


def test_extract_mentions():
    text = "Hi @alice and @bob, see @charlie-99 and not @"
    out = _extract_mentions(text)
    assert "alice" in out
    assert "bob" in out
    assert "charlie-99" in out


def _fixture_node(num: int, author: str, ans_author: str, comments: list[str], cat: str = "Q&A") -> dict:
    return {
        "id": f"DC_{num}",
        "number": num,
        "title": f"How to do thing {num}",
        "body": f"I want to do thing {num}. cc @teammate",
        "createdAt": f"2026-04-0{(num % 9) + 1}T00:00:00Z",
        "author": {"login": author},
        "category": {"name": cat, "slug": cat.lower()},
        "answer": {"id": f"DA_{num}", "author": {"login": ans_author}, "body": "Use foo()."},
        "comments": {
            "nodes": [
                {"id": f"DCM_{num}_{i}", "author": {"login": u}, "body": f"reply {i}", "createdAt": "2026-04-05T00:00:00Z"}
                for i, u in enumerate(comments)
            ]
        },
    }


def test_discussions_to_corpus_minimal():
    nodes = [
        _fixture_node(1, "alice", "bob", ["bob", "carol"], "Q&A"),
        _fixture_node(2, "carol", "alice", ["alice"], "General"),
    ]
    c = discussions_to_corpus(nodes, name="github_test")
    assert c.name == "github_test"
    assert len(c.documents) == 2
    assert len(c.queries) == 2
    # answered discussions: each query has at least one grade-2 qrel (the doc itself)
    assert any(r.grade == 2 for r in c.qrels)


def test_discussions_to_corpus_zone_mapped():
    nodes = [_fixture_node(1, "alice", "bob", ["bob"], "Announcements")]
    c = discussions_to_corpus(nodes, name="t")
    assert c.documents[0].location_type == "meeting_room"  # Announcements -> meeting_room


def test_empty_nodes_yields_empty_corpus():
    c = discussions_to_corpus([], name="t")
    assert c.documents == ()
    assert c.queries == ()
    assert c.qrels == ()
