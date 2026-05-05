"""Tests for src/gta/scenario_gen.py."""
from src.gta.scenario_gen import (
    DIVERSITY_CELLS, GTAScenario, TemplateBackend, generate_scenarios,
    scenarios_to_jsonable,
)


def test_diversity_matrix_size():
    assert len(DIVERSITY_CELLS) == 100  # 5 * 5 * 4


def test_template_backend_deterministic():
    b = TemplateBackend()
    a = b.generate("engineer", "debug_help", "single_fixation")
    c = b.generate("engineer", "debug_help", "single_fixation")
    assert a == c
    assert "debug_help" in a[0]


def test_generate_scenarios_size():
    sc = generate_scenarios(n=100, seed=42)
    assert len(sc) == 100
    # all scenarios are valid GTAScenario
    assert all(isinstance(s, GTAScenario) for s in sc)


def test_generate_scenarios_diverse():
    sc = generate_scenarios(n=100, seed=42)
    cells = {(s.role, s.info_need, s.gaze_pattern) for s in sc}
    assert len(cells) == 100  # every cell covered exactly once


def test_generate_scenarios_extension_cycles():
    sc = generate_scenarios(n=200, seed=42)
    assert len(sc) == 200


def test_gaze_trajectory_nonempty():
    sc = generate_scenarios(n=10, seed=42)
    for s in sc:
        assert len(s.gaze_trajectory) >= 1
        assert all(e.duration_ms > 0 for e in s.gaze_trajectory)


def test_distractors_disjoint_from_target():
    sc = generate_scenarios(n=20, seed=42)
    for s in sc:
        target_p = s.ground_truth_doc_id.replace("d_target_", "")
        for did in s.distractor_doc_ids:
            d_p = did.replace("d_distract_", "")
            assert d_p != target_p


def test_jsonable_roundtrip():
    sc = generate_scenarios(n=5, seed=42)
    j = scenarios_to_jsonable(sc)
    assert len(j) == 5
    assert all("scenario_id" in d for d in j)
    assert all("gaze_trajectory" in d for d in j)
