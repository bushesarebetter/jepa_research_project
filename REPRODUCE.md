# REPRODUCE.md — Exact reproduction commands

All result JSONs in `results/` are committed. To regenerate from scratch on a clean checkout:

## Prerequisites

```bash
pip install -r requirements.txt
# GPU strongly recommended for full-scale runs (see timing below)
# CPU is sufficient for smoke / verify
```

## 0. Verify the environment

```bash
python verify.py
# Expected: "verify.py PASS: ..." with all checks green
# Runtime: ~30s on CPU
# Device: CPU (no GPU needed)
```

## 1. SwitchColor p_repeat sweep (figures 1–5)

```bash
# Full training (produces results/*.json, updates results/all_results.json)
python run_experiment.py --config full
# Runtime: ~2–3h on GPU (30k steps × 3 seeds × 3 models × 4 p_values)
# Device: GPU recommended

# Smoke check (seconds on CPU)
python run_experiment.py --config quick
```

## 2. Quadrant matrix (figure 6) — objective × cell

```bash
python quadrant_experiment.py --full
# Writes: results/quadrant_matrix.json, results/quadrant/per-run JSONs
# Runtime: ~90min on GPU (4000 steps × 3 seeds × 7 objectives × 4 cells)
# Device: GPU recommended

# Smoke check
python quadrant_experiment.py
```

## 3. Cell-4 failure demo — both envs (figure 9, THE MAIN FIGURE)

```bash
python quadrant_experiment.py --config cell4_failure --full
# Writes: results/quadrant_matrix_cell4_failure_quadrant.json
# Runtime: ~20min on GPU (4000 steps × 3 seeds × 7 objectives × 1 cell)
# Device: GPU recommended

python quadrant_experiment.py --config cell4_failure --env gridworld --full
# Writes: results/quadrant_matrix_cell4_failure_gridworld.json
# Runtime: ~20min on GPU
# Device: GPU recommended

# Smoke checks
python quadrant_experiment.py --config cell4_failure
python quadrant_experiment.py --config cell4_failure --env gridworld
```

## 4. Minimum reward signal (figure 7)

```bash
python quadrant_experiment.py --config min_reward_signal --full
# Writes: results/min_reward_signal_quadrant.json
# Runtime: ~15min on GPU (4000 steps × 3 seeds × 7 fractions)
# Device: GPU recommended
```

## 5. Capacity sweep (figure 8)

```bash
python quadrant_experiment.py --config capacity_sweep --full
# Writes: results/capacity_sweep.json
# Runtime: ~60min on GPU (4000 steps × 3 seeds × 6 dims × 2 envs × 2 objectives)
# Device: GPU recommended
```

## 6. Observation 1 numbers (M5)

```bash
python proposition_numbers.py --full
# Writes: results/proposition_numbers_full.json, updates RESULTS.md M5 block
# Runtime: ~5min on GPU (4000 steps × 3 objectives)
# Device: GPU recommended

# Smoke (seconds on CPU)
python proposition_numbers.py
```

## 7. Generate all figures

```bash
python make_all_figures.py
# Reads all result JSONs, writes results/figures/*.{pdf,png}
# Runtime: <30s (no training)
# Device: CPU
```

## 8. Generate paper tables

```bash
python make_paper_tables.py
# Writes results/tables/table{1,2,3}.tex
# Runtime: <5s (no training)
# Device: CPU
```

## Full figure list (pdf + png for each)

| file | content | source JSON |
|---|---|---|
| figure1_main_result | SwitchColor probe acc vs p_repeat | all_results.json |
| figure2_bisimulation | centroid dist + control regret vs p_repeat | all_results.json |
| figure3_theory_vs_empirical | probe acc vs control regret scatter | all_results.json |
| figure4_ablations | ablation bars + effective rank | all_results.json |
| figure5_environment | environment frames (p=0.5 vs p=1.0) | — |
| figure6_quadrant_matrix | objective × cell heatmap | quadrant_matrix.json |
| figure7_min_reward_signal_quadrant | cell-4 acc vs reward_label_fraction | min_reward_signal_quadrant.json |
| figure8_capacity | cell-4 acc vs latent_dim | capacity_sweep.json |
| **figure9_cell4_failure_both_envs** | **main paper figure: probe acc + MI per objective, both envs** | quadrant_matrix_cell4_failure_{quadrant,gridworld}.json |

## Hardware and determinism notes

- **GPU used for training:** NVIDIA (Linux-6.6.122+, CUDA 12.8, torch 2.11.0+cu128)
- **CPU sufficient for:** verify.py, make_all_figures.py, make_paper_tables.py, smoke configs
- **Seeds:** all full runs use seeds [0, 1, 2]; eval seeds are train_seed + 100000 (disjoint)
- **Determinism:** PyTorch default seeding is used; CUDA nondeterminism means bit-exact
  reproduction requires the same GPU model. Means over 3 seeds are stable at 2 decimal places.
- **Where each number comes from:** all headline numbers in RESULTS.md cite the JSON file they
  were extracted from. The JSON `_provenance` block records git commit + library versions.

## M6 (future work — not run here)

`requirements-realmodel.txt` lists dependencies for the pretrained-encoder comparison
(V-JEPA 2-AC). This is explicitly out of scope for the current submission. Do not run
`python run_experiment.py --config realmodel` without the guarded download step.
