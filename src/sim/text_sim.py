"""Text similarity backend.

Default: BM25 (rank_bm25) over a tokenized corpus.
Future: replace with cosine over nomic-embed-text embeddings (Eq. 6 in paper).
The interface is the same so callers don't change.
"""
from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.utils.types import Corpus, Document


_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer. Keep alphanum tokens; split on punctuation/space."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class BM25Backend:
    """BM25 text similarity over a fixed document set."""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self._doc_tokens = [tokenize(d.text) for d in self.documents]
        self._bm25 = BM25Okapi(self._doc_tokens, k1=k1, b=b)

    def scores(self, query_text: str) -> dict[str, float]:
        """Return BM25 score per doc_id for the given query."""
        q_tokens = tokenize(query_text)
        raw = self._bm25.get_scores(q_tokens)
        # normalize to [0, 1] within this query so it composes fairly with the
        # spatial features (which are in [0, 1]).
        if len(raw) == 0:
            return {}
        max_s = float(max(raw)) or 1.0
        return {self.documents[i].doc_id: float(raw[i]) / max_s for i in range(len(raw))}

    @classmethod
    def from_corpus(cls, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> "BM25Backend":
        return cls(corpus.documents, k1=k1, b=b)
