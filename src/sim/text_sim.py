"""Text similarity backends.

Two interchangeable backends sharing the same `.scores(query_text) -> {doc_id: score}`
interface so the evaluator can swap them transparently:

  - `BM25Backend`     : sparse retrieval (BM25 over tokens via rank_bm25).
  - `DenseBackend`    : dense retrieval (cosine over nomic-embed-text 768-d
                        embeddings served by Ollama at /api/embeddings).

Both rescale scores to [0, 1] within a query so the spatial bonuses (also in
[0, 1]) compose fairly inside the SCAR scorer.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import requests
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


class DenseBackend:
    """Dense text similarity using nomic-embed-text via Ollama.

    Document embeddings are computed once at construction; query embeddings
    are computed on demand. Cosine similarities are rescaled from [-1, 1] to
    [0, 1] so they compose with spatial features on the same scale.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self.documents = list(documents)
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._doc_emb = self._embed_batch([d.text for d in self.documents])
        self._doc_emb = self._normalize(self._doc_emb)

    def _embed(self, text: str) -> np.ndarray:
        r = requests.post(
            f"{self.url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return np.asarray(r.json()["embedding"], dtype=np.float32)

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            n = float(np.linalg.norm(x))
            return x / n if n > 0 else x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return x / norms

    def scores(self, query_text: str) -> dict[str, float]:
        """Cosine sim per doc, min-max rescaled per query to [0, 1].

        Per-query min-max matches the effective dynamic range used by
        ``BM25Backend`` (which divides by max within a query), so that the
        downstream SCAR scorer composes text and spatial signals on the
        same scale regardless of which backend is in use.
        """
        if not self.documents:
            return {}
        q = self._normalize(self._embed(query_text))
        cos = self._doc_emb @ q  # in [-1, 1]
        c_min = float(cos.min())
        c_max = float(cos.max())
        if c_max - c_min > 1e-9:
            sims01 = (cos - c_min) / (c_max - c_min)
        else:
            sims01 = np.zeros_like(cos)
        return {self.documents[i].doc_id: float(sims01[i]) for i in range(len(self.documents))}

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> "DenseBackend":
        return cls(corpus.documents, model=model, url=url, timeout=timeout)
