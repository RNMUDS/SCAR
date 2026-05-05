# SCAR System Design v2 — Brushup Architecture

**Date**: 2026-05-05
**Context**: Phase 0 deliverable. Designs the new system pieces needed for C案 (full re-experimentation + real corpus).
**Reference**: PRIMA `06_reviewer_response/runReviewerExperiments3.js` (LinUCB lines 263-450)

---

## 1. New Subsystems

### 1.1 LinUCB Online Adaptive Baseline (`src/baseline/linucb.py`)

**Framing**: Contextual bandit for **re-ranking strategy selection**, not document selection.

| | PRIMA | SCAR |
|---|---|---|
| Arms | 5 intervention styles | 6 re-ranking strategies |
| Context | 6-dim psych state + 5 personality one-hot = 11D | 4-dim spatial + 5 role one-hot + bias = 10D |
| Reward | accept_prob | nDCG@5 on chosen strategy for this query |
| Update | per intervention | per query |

**Arm definition** (6 strategies):
1. `bm25` — text only
2. `bm25+prox` — text + proximity feature
3. `bm25+gaze` — text + gaze feature
4. `bm25+loc` — text + location feature
5. `bm25+time` — text + time decay
6. `scar-full` — all 4 spatial features (= current SCAR fixed weights)

**Three variants** (mirrors PRIMA):
- `linucb-10d` — base
- `linucb-warm` — LLM warm-start: inject `warmStartN=20` synthetic obs per arm using prior expected nDCG (computed once on small held-out set)
- `scar+linucb` — SCAR's fixed weights provide initial θ; LinUCB only learns deviations

**Why this beats the current "6 LTR offline" baseline**: those models are trained offline on the entire query log. LinUCB is the natural online competitor — and the *honest* comparison for a system that claims context-awareness.

**Evaluation**: cumulative regret vs oracle (offline-trained best arm per role).

### 1.2 GTA Scenario Auto-Generator (`src/gta/scenario_gen.py`)

**Goal**: replace n=10 manual scenarios with n=100-200 LLM-generated, structurally diverse scenarios.

**Generator design**:
```
Diversity matrix: 5 roles × 5 information needs × 4 gaze patterns = 100 scenarios
Roles: {engineer, designer, PM, exec, intern}
Info needs: {prior-art lookup, debug help, design ref, status check, onboarding}
Gaze patterns: {single fixation, 2-target alternation, scan-and-zoom, drift-and-return}
```

**Per-scenario generation** (Mistral 7B on Ollama, local):
- Input: (role, info_need, gaze_pattern) triple
- Output: gaze trajectory time series + ground-truth document ID + distractor IDs
- Validation: at least 3 alternative-correct documents from the corpus, query maps to gaze pattern

**Quality control**: dedup via embedding similarity (>0.9 = duplicate, regenerate); reject scenarios where ground-truth doc has no overlap with gaze targets.

### 1.3 Weight Grid Search (`src/grid/lhs_search.py`)

5D weight space: (text, proximity, gaze, location, time). Constrain Σw=1, all w≥0 (simplex).

Latin Hypercube Sampling on the 4-simplex (project to 4D after dropping one coordinate), 81 points. Each point:
- evaluate on 3 corpora × 120 queries
- record nDCG@5, MRR, Recall@5

Output: `data/results/grid81.parquet` with one row per (point, corpus, query). Visualize as 2D barycentric projection heatmaps.

### 1.4 Noise Injection (`src/noise/inject.py`)

Add Gaussian noise:
- proximity: N(0, σ) on (x, y, z) per agent per timestep
- gaze: N(0, σ) on (yaw, pitch) per gaze event

Sweep σ ∈ {0, 0.1, 0.2, 0.3, 0.5}. Compare:
- BM25 (insensitive — sanity check)
- SCAR (should degrade gracefully)
- LinUCB (should adapt away from noisy features)

Expected outcome (paper claim): **SCAR with σ=0.5 ≥ BM25 ± SE**, and LinUCB with σ=0.5 → BM25-equivalent (auto-discounted noisy features).

### 1.5 GitHub Discussions Real Corpus (`src/corpus/github_ingest.py`)

**Target**: `kubernetes/kubernetes` Discussions (~5,000 threads, public, GraphQL accessible).

| SCAR concept | Mapping |
|---|---|
| Document | Discussion thread + top-3 comments concatenated |
| Room/Location | Discussion category (`general`, `feature-requests`, `q-and-a`, ...) |
| Proximity | Co-commenter graph (Jaccard over last 30 days) |
| Gaze (proxy) | Reactions + view bursts (where available via API) |
| Query | First sentence of new question OR explicit search-style header |
| Gold relevance | Marked-as-answer comment (grade 2), positive-reaction comments (grade 1), other (grade 0) |

**Ingest pipeline**:
1. GraphQL paginated fetch → raw JSON dump
2. Filter to closed/answered threads (need gold label)
3. Build spatial graph from co-commenter Jaccard
4. Embed with `nomic-embed-text` via Ollama (already on DGX)
5. Output `corpora/github_real/{docs.jsonl, queries.jsonl, qrels.txt, spatial_graph.json}`

Target scale: 2,000 docs / 200 queries / 5 categories.

---

## 2. Compute Plan on DGX Spark (GB10)

| Step | Workload | Est. time |
|---|---|---|
| LinUCB online learning (3 corpora × 120Q × 100 episodes × 3 variants) | bandit loop, GPU not used | ~30 min |
| GTA generation (n=200, Mistral 7B) | Ollama inference | ~2 hrs |
| Grid search 81 × eval | parallel CPU | ~3 hrs |
| Noise sweep (5 levels × 3 methods × full eval) | CPU | ~1.5 hrs |
| GitHub ingest + embedding (~2,000 docs, nomic-embed) | GPU | ~20 min |
| Real corpus replication | full pipeline | ~1 hr |
| LLM-as-Judge re-validation (Llama 3 70B class on judgment task) | GPU heavy | ~3 hrs |
| **Total** | | **~11 hrs** |

GB10 GPU is idle during bandit/grid runs — these are CPU-bound. We will saturate GPU only during GTA gen + embedding + LLM-as-Judge.

---

## 3. Determinism & Reproducibility

- All RNG seeds in `config/seeds.yaml`. Default seed 42 for main runs; sweep over 5 seeds for variance estimation.
- Each experiment writes a manifest: `{git_commit, config_hash, seed, env_capture, timestamp}`.
- Ollama model versions pinned by digest (recorded at experiment start).

---

## 4. Risk Hooks (early signals)

| Risk | Detection | Action |
|---|---|---|
| LinUCB beats SCAR | regret curve crosses zero | reframe paper around hybrid (SCAR-warm + online correction) |
| GTA n=100 still not significant | confidence interval excludes 0 at α=0.05 | grow to n=300, document negative result honestly |
| GitHub corpus too sparse | < 50 queries with gold answers | switch to `vercel/next.js` Discussions or extend Apache Camel JIRA |
| GB10 OOM during 70B judge | nvidia-smi alert | drop to 30B class or use FP4 quantization |

---

## 5. Phase 0 Deliverables Checklist

- [x] git tag v1.0-pre-brushup (local)
- [x] paper synced to DGX Spark (`~/SCAR/paper_v1/`)
- [x] PRIMA reference synced (`~/SCAR/reference/prima/`)
- [x] Python venv with torch+cu130, sklearn, sentence-transformers, ollama
- [x] system_info.txt captured
- [x] this design doc
- [ ] config/seeds.yaml
- [ ] tests/test_smoke.py (verify imports + GPU)

After this checklist → Phase 1.1 (LinUCB) starts.
