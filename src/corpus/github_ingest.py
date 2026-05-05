"""GitHub Discussions corpus ingestion via the `gh` CLI.

Maps GitHub Discussions to SCAR's virtual-office abstractions:
    - Document       <- Discussion thread (title + body + top-3 comments)
    - location_type  <- Discussion category (mapped to virtual zone)
    - author         <- Discussion author login
    - mentions       <- @-mentioned users in the body
    - created_day    <- (created_at - corpus_epoch).days
    - Query          <- Title of an "answered" discussion (with answer doc removed
                       from candidate set; spatial context inferred)
    - qrels          <- Marked-as-answer comment author -> grade 2
                        Co-commenters of the answer thread -> grade 1
                        Other docs -> grade 0

We use the authenticated `gh api graphql` command rather than direct HTTPS so
we inherit gh's auth and rate-limit handling.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from src.utils.types import (
    Corpus, Document, DOC_KINDS, Query, Relevance, ROLES, SpatialContext, ZONE_TYPES,
)


# Map GitHub Discussion category names to SCAR zone types. Anything unknown
# falls back to "open_floor".
CATEGORY_TO_ZONE: dict[str, str] = {
    "Q&A": "library",
    "General": "lounge",
    "Ideas": "lounge",
    "Show and tell": "open_floor",
    "Announcements": "meeting_room",
    "Polls": "open_floor",
}

# Use answer state to assign role to the question author. Heuristic only.
ROLE_FALLBACK = "engineer"


GRAPHQL_QUERY = """
query ($owner: String!, $repo: String!, $cursor: String, $cat: ID) {
  repository(owner: $owner, name: $repo) {
    discussions(first: 50, after: $cursor, answered: true, categoryId: $cat) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        number
        title
        body
        createdAt
        author { login }
        category { name slug }
        answer { id author { login } body }
        comments(first: 10) {
          nodes {
            id
            author { login }
            body
            createdAt
          }
        }
      }
    }
  }
}
"""


def _gh_graphql(query: str, variables: dict) -> dict:
    """Run a GraphQL query via `gh api graphql -F`. Returns parsed JSON 'data'."""
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        cmd += ["-F", f"{k}={v}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api failed: {proc.stderr}")
    payload = json.loads(proc.stdout)
    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    return payload.get("data", {})


def fetch_answered_discussions(
    owner: str,
    repo: str,
    max_pages: int = 4,
) -> list[dict]:
    """Fetch up to max_pages * 50 answered discussions. Returns raw nodes."""
    all_nodes: list[dict] = []
    cursor: Optional[str] = None
    for _ in range(max_pages):
        data = _gh_graphql(GRAPHQL_QUERY, {"owner": owner, "repo": repo, "cursor": cursor})
        repo_data = data.get("repository") or {}
        disc = repo_data.get("discussions") or {}
        nodes = disc.get("nodes") or []
        all_nodes.extend(nodes)
        page_info = disc.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
    return all_nodes


def _zone_for_category(name: Optional[str]) -> str:
    if name is None:
        return "open_floor"
    return CATEGORY_TO_ZONE.get(name, "open_floor")


def _kind_for_thread(answer_present: bool) -> str:
    return "spec_doc" if answer_present else "brainstorm_note"


def _epoch_day(ts_iso: str, epoch_iso: str) -> int:
    """Return (ts - epoch).days. Uses simple ISO date math; no tz handling."""
    from datetime import datetime
    a = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    e = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00"))
    return (a - e).days


def discussions_to_corpus(
    nodes: Sequence[dict],
    name: str,
    queries_per_zone: int = 40,
) -> Corpus:
    """Convert GraphQL discussion nodes into a SCAR Corpus.

    Construction rules:
        - one Document per discussion (text = title + body + top-3 comments)
        - one Query per discussion that has an answer (its title becomes the query;
          answer author becomes the gaze target; co-commenters become nearby)
        - qrels: answer-author's discussion = grade 2; co-commenters' threads = grade 1
    """
    if not nodes:
        return Corpus(name=name, documents=(), queries=(), qrels=())

    # Use the earliest creation date as the epoch so all created_day values are non-negative
    epoch_iso = min(n["createdAt"] for n in nodes)

    documents: list[Document] = []
    author_threads: dict[str, list[str]] = defaultdict(list)  # author -> [doc_id]

    for n in nodes:
        author_login = ((n.get("author") or {}).get("login")) or "unknown"
        comments = (n.get("comments") or {}).get("nodes") or []
        body = (n.get("body") or "")
        title = (n.get("title") or "")
        merged_text = title + "\n" + body
        for c in comments[:3]:
            cb = c.get("body") or ""
            merged_text += "\n" + cb

        # extract @-mentions
        mentions = tuple(sorted(set(_extract_mentions(merged_text))))

        zone = _zone_for_category((n.get("category") or {}).get("name"))
        kind = _kind_for_thread(n.get("answer") is not None)
        created_day = max(0, _epoch_day(n["createdAt"], epoch_iso))

        doc_id = f"gh_{n.get('number', 'x')}"
        documents.append(Document(
            doc_id=doc_id,
            text=merged_text,
            author=author_login,
            location_type=zone,
            kind=kind,
            created_day=created_day,
            mentions=mentions,
        ))
        author_threads[author_login].append(doc_id)

    # Build queries from answered discussions
    queries: list[Query] = []
    qrels: list[Relevance] = []
    today = max(d.created_day for d in documents) + 1

    for n in nodes:
        ans = n.get("answer")
        if not ans:
            continue
        author_login = ((n.get("author") or {}).get("login")) or "unknown"
        ans_author = ((ans.get("author") or {}).get("login")) or "unknown"
        comments = (n.get("comments") or {}).get("nodes") or []
        commenter_logins = [
            (c.get("author") or {}).get("login")
            for c in comments
            if (c.get("author") or {}).get("login")
        ]
        # nearby = unique commenters (excluding the asker themselves)
        nearby = tuple(sorted({u for u in commenter_logins if u and u != author_login}))

        ctx = SpatialContext(
            user_id=author_login,
            role=ROLE_FALLBACK,
            current_zone=_zone_for_category((n.get("category") or {}).get("name")),
            current_day=today,
            nearby_persons=nearby[:5],  # cap to avoid explosion
            gaze_target=ans_author if ans_author and ans_author != author_login else None,
        )
        q = Query(
            query_id=f"gh_q_{n.get('number')}",
            text=(n.get("title") or "")[:200],
            context=ctx,
        )
        queries.append(q)

        # qrels: answer-author's other threads = grade 2 (they're an expert on this topic)
        # commenter authors' other threads = grade 1
        # the discussion itself: grade 2 (it has the answer)
        own_doc_id = f"gh_{n.get('number')}"
        qrels.append(Relevance(query_id=q.query_id, doc_id=own_doc_id, grade=2))

        for d_id in author_threads.get(ans_author, ()):
            if d_id == own_doc_id:
                continue
            qrels.append(Relevance(query_id=q.query_id, doc_id=d_id, grade=2))

        for commenter in nearby:
            for d_id in author_threads.get(commenter, ()):
                if d_id == own_doc_id:
                    continue
                qrels.append(Relevance(query_id=q.query_id, doc_id=d_id, grade=1))

    # cap queries to a reasonable size for evaluation speed
    if queries_per_zone:
        cap = queries_per_zone * len(ZONE_TYPES)
        queries = queries[:cap]

    return Corpus(
        name=name,
        documents=tuple(documents),
        queries=tuple(queries),
        qrels=tuple(qrels),
    )


def _extract_mentions(text: str) -> list[str]:
    """Cheap @login extraction. Returns logins without the @ prefix."""
    import re
    return [m.group(1) for m in re.finditer(r"@([A-Za-z][A-Za-z0-9-]{0,38})", text)]
