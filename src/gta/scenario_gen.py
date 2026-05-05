"""Gaze Trajectory Attention (GTA) scenario generator.

Creates n=100-200 synthetic gaze trajectory scenarios with ground-truth target
documents. Each scenario specifies:
    - role x info_need x gaze_pattern (the diversity matrix from system_design_v2 §1.2)
    - a time-ordered sequence of gaze targets (person ids)
    - the ground-truth target document (the one a perfect GTA system should retrieve)
    - 4 distractor documents (sharing topic but wrong author/no spatial alignment)

Generation strategy:
    1. Procedurally enumerate the 5x5x4 = 100 (role, info_need, gaze_pattern) cells.
    2. For each cell, optionally call an LLM (qwen3.5 via Ollama) to generate
       a natural-language scenario description and a query phrasing.
    3. Construct deterministic ground-truth + distractors from a backing document
       pool so we can compute exact relevance grades.

The LLM call is wrapped behind a `LLMBackend` abstraction. For unit tests and
when Ollama is unavailable, a `TemplateBackend` produces reproducible scenarios
without network calls.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence


ROLES = ("engineer", "designer", "pm", "exec", "intern")
INFO_NEEDS = ("prior_art", "debug_help", "design_ref", "status_check", "onboarding")
GAZE_PATTERNS = ("single_fixation", "two_target_alt", "scan_zoom", "drift_return")

DIVERSITY_CELLS = tuple(
    (role, need, pattern)
    for role in ROLES
    for need in INFO_NEEDS
    for pattern in GAZE_PATTERNS
)
# 5 * 5 * 4 = 100 cells


@dataclass(frozen=True)
class GazeEvent:
    """One gaze observation: who or what the user looked at, for how long."""
    timestamp_ms: int
    target_person: str
    duration_ms: int


@dataclass(frozen=True)
class GTAScenario:
    """A complete GTA scenario."""
    scenario_id: str
    role: str
    info_need: str
    gaze_pattern: str
    description: str                       # LLM- or template-generated narrative
    query_text: str                        # the retrieval query
    gaze_trajectory: tuple[GazeEvent, ...]  # time-ordered gaze events
    ground_truth_doc_id: str
    distractor_doc_ids: tuple[str, ...]


class LLMBackend(Protocol):
    """Generate a (description, query) pair for a (role, info_need, gaze_pattern) cell."""

    def generate(self, role: str, info_need: str, gaze_pattern: str) -> tuple[str, str]: ...


@dataclass
class TemplateBackend:
    """Deterministic backend producing scripted descriptions. No LLM call."""

    def generate(self, role: str, info_need: str, gaze_pattern: str) -> tuple[str, str]:
        topic_keyword = {
            "prior_art": "previous research",
            "debug_help": "error trace",
            "design_ref": "design spec",
            "status_check": "weekly status",
            "onboarding": "newcomer guide",
        }[info_need]
        description = (
            f"A {role} engaged in {info_need} with a {gaze_pattern} gaze pattern, "
            f"requiring documents about {topic_keyword}."
        )
        query = f"find the {topic_keyword} relevant to my current work"
        return description, query


@dataclass
class OllamaBackend:
    """LLM-driven backend using a local Ollama model.

    Falls back gracefully if Ollama is unreachable: the caller can wrap with
    a try/except and swap in TemplateBackend.
    """
    model: str = "qwen3.5:latest"
    temperature: float = 0.6
    timeout_s: float = 30.0

    def generate(self, role: str, info_need: str, gaze_pattern: str) -> tuple[str, str]:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("ollama package not installed") from exc

        prompt = (
            "You are generating a scenario for a virtual office gaze-trajectory study.\n"
            f"Role: {role}\n"
            f"Information need: {info_need}\n"
            f"Gaze pattern: {gaze_pattern}\n"
            "\n"
            "Output two lines exactly:\n"
            "DESCRIPTION: <one sentence describing the user's situation>\n"
            "QUERY: <the natural-language search query the user would type, 4-10 words>\n"
        )
        client = ollama.Client()
        resp = client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature, "num_predict": 120},
        )
        text = resp.get("response", "")
        desc, query = "", ""
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("DESCRIPTION:"):
                desc = line.split(":", 1)[1].strip()
            elif line.upper().startswith("QUERY:"):
                query = line.split(":", 1)[1].strip()
        if not desc or not query:
            # Fallback to template if LLM output didn't parse
            return TemplateBackend().generate(role, info_need, gaze_pattern)
        return desc, query


def _build_gaze_trajectory(
    rng: random.Random,
    pattern: str,
    target_person: str,
    distractor_persons: Sequence[str],
) -> tuple[GazeEvent, ...]:
    """Procedurally produce a gaze trajectory matching the named pattern."""
    events: list[GazeEvent] = []
    if pattern == "single_fixation":
        # one long gaze on the target
        events.append(GazeEvent(timestamp_ms=0, target_person=target_person, duration_ms=2000))
    elif pattern == "two_target_alt":
        # alternate between target and one distractor
        d = rng.choice(distractor_persons) if distractor_persons else target_person
        for i in range(4):
            who = target_person if i % 2 == 0 else d
            events.append(GazeEvent(timestamp_ms=i * 500, target_person=who, duration_ms=400))
    elif pattern == "scan_zoom":
        # quick scan over distractors then settle on target
        for i, p in enumerate(distractor_persons[:3]):
            events.append(GazeEvent(timestamp_ms=i * 200, target_person=p, duration_ms=180))
        events.append(GazeEvent(timestamp_ms=600, target_person=target_person, duration_ms=1400))
    elif pattern == "drift_return":
        # look at target, drift away, then return
        events.append(GazeEvent(timestamp_ms=0, target_person=target_person, duration_ms=600))
        for i, p in enumerate(distractor_persons[:2]):
            events.append(GazeEvent(timestamp_ms=600 + i * 300, target_person=p, duration_ms=250))
        events.append(GazeEvent(timestamp_ms=1200, target_person=target_person, duration_ms=800))
    else:
        # unknown pattern: degrade to single fixation
        events.append(GazeEvent(timestamp_ms=0, target_person=target_person, duration_ms=1000))
    return tuple(events)


def generate_scenarios(
    n: int = 100,
    backend: Optional[LLMBackend] = None,
    seed: int = 42,
    backing_persons: Optional[Sequence[str]] = None,
    backing_docs_per_cell: int = 5,
) -> list[GTAScenario]:
    """Generate n GTA scenarios cycling through the diversity matrix.

    backing_persons: pool of person IDs used as gaze targets and document authors.
                     If None, defaults to a deterministic pool of 25.
    backing_docs_per_cell: how many candidate docs per cell exist (used to assign
                          a unique ground-truth doc ID; the actual document
                          objects come from the corpus the scenario lives in).
    """
    backend = backend or TemplateBackend()
    rng = random.Random(seed)
    if backing_persons is None:
        backing_persons = [f"p{i:03d}" for i in range(25)]

    scenarios: list[GTAScenario] = []
    cells = list(DIVERSITY_CELLS)
    if n > len(cells):
        # cycle through cells multiple times to reach n
        full, partial = divmod(n, len(cells))
        cells = cells * full + cells[:partial]
    else:
        cells = cells[:n]

    for idx, (role, need, pattern) in enumerate(cells):
        target = rng.choice(backing_persons)
        distractors = [p for p in backing_persons if p != target]
        rng.shuffle(distractors)
        distractors = distractors[:5]

        description, query = backend.generate(role, need, pattern)
        trajectory = _build_gaze_trajectory(rng, pattern, target, distractors)

        scenario = GTAScenario(
            scenario_id=f"gta{idx:04d}",
            role=role,
            info_need=need,
            gaze_pattern=pattern,
            description=description,
            query_text=query,
            gaze_trajectory=trajectory,
            ground_truth_doc_id=f"d_target_{target}",
            distractor_doc_ids=tuple(f"d_distract_{p}" for p in distractors[:4]),
        )
        scenarios.append(scenario)

    return scenarios


def scenarios_to_jsonable(scenarios: Sequence[GTAScenario]) -> list[dict]:
    """Serialize for JSON dump."""
    out: list[dict] = []
    for s in scenarios:
        out.append({
            "scenario_id": s.scenario_id,
            "role": s.role,
            "info_need": s.info_need,
            "gaze_pattern": s.gaze_pattern,
            "description": s.description,
            "query_text": s.query_text,
            "gaze_trajectory": [
                {"t_ms": e.timestamp_ms, "target": e.target_person, "dur_ms": e.duration_ms}
                for e in s.gaze_trajectory
            ],
            "ground_truth_doc_id": s.ground_truth_doc_id,
            "distractor_doc_ids": list(s.distractor_doc_ids),
        })
    return out
