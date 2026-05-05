# SCAR Brushup Workspace (DGX Spark)

**Host**: `muds-spark@202.240.109.50` (NVIDIA GB10, CUDA 13.0, 119GB RAM, Ubuntu 24)
**Mirror of paper**: `~/SCAR/paper_v1/` (synced from local Mac, frozen tag `v1.0-pre-brushup`)
**Reference**: `~/SCAR/reference/prima/` (PRIMA repo subset for LinUCB / experiment design)

## Layout

```
~/SCAR/
├── README.md                    ← this file
├── paper_v1/                    ← frozen v1.0 SCAR paper (read-only reference)
├── reference/
│   └── prima/                   ← PRIMA reference (papers/, src/, 06_reviewer_response/)
├── src/
│   ├── baseline/                ← LinUCB and other adaptive baselines (Python)
│   ├── gta/                     ← Gaze Trajectory Attention scenario generator
│   ├── grid/                    ← weight grid search (LHS over 5D)
│   ├── noise/                   ← noise injection module
│   ├── corpus/                  ← real corpus (GitHub Discussions) ingestion
│   ├── sim/                     ← virtual office simulator (port from paper_v1)
│   └── utils/                   ← shared (RNG, metrics, IO, eval)
├── experiments/
│   ├── 01_linucb/               ← Phase 1.1 + 2.1
│   ├── 02_gta_n100/             ← Phase 1.2 + 2.2
│   ├── 03_grid81/               ← Phase 1.3 + 2.3
│   ├── 04_noise_sweep/          ← Phase 1.4 + 2.4
│   ├── 05_real_corpus/          ← Phase 1.5 + 2.5
│   └── 06_bias_verification/    ← Phase 2.6
├── data/
│   ├── corpora/                 ← generated and real corpora
│   ├── results/                 ← all experiment JSON outputs
│   └── figures/                 ← regenerated and new figures (PNG/PDF)
├── scripts/                     ← driver scripts (run_all, sync, build)
├── logs/                        ← stdout/stderr from runs
├── notebooks/                   ← exploratory analysis (.ipynb)
├── tests/                       ← pytest tests for src/
└── config/
    ├── system_info.txt          ← captured at Phase 0
    └── seeds.yaml               ← canonical RNG seeds for reproducibility
```

## How to run

```bash
cd ~/SCAR
source .venv/bin/activate
python -m experiments.01_linucb.run --seed 42 --config config/linucb_default.yaml
```

All experiments write to `data/results/<exp_id>_<timestamp>.json` and figures to `data/figures/`.

## Sync back to local Mac

```bash
# from local Mac
rsync -az --exclude='.venv' --exclude='reference' --exclude='paper_v1' \
  muds-spark@202.240.109.50:~/SCAR/ /Users/rn/Documents/Company/docs/research/scar_brushup/
```

Paper compilation stays on local Mac (DGX Spark has no LaTeX; we keep paper.tex authored locally and only ship figures/result tables back).

## Plan reference

Full plan: local Mac at `/Users/rn/Documents/Company/tasks/scar_brushup_plan.md`

Phase status: **Phase 0 in progress**.
