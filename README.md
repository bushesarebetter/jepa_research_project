# JEPA control-loss experiment

**Claim:** A JEPA-style latent self-prediction objective discards a feature that is
control-relevant but temporally unpredictable, even though the feature is trivially encodable.
A reward-grounded signal fixes this selectively.

## Results

Across two synthetic environments (QuadrantEnv and GridWorldHiddenRuleEnv) and six latent
sizes (16–1024), pure-prediction JEPA achieves **0.51 linear probe accuracy** on the
uncontrollable-but-reward-relevant cell-4 feature (chance = 0.50), while `jepa_reward` achieves
**1.00** — the gap is fully explained by the objective, not encoder capacity.

The bisimulation theory gap (Observation 1, empirical): the analytical bisimulation distance
for cell 4 is **1.0 > 0**, yet JEPA's latent class distance collapses to **0.105** — only 5%
of the oracle separation. The bisimulation error (oracle − JEPA) is **1.893**. This is an
empirical observation, not a proved theorem.


Labelling as few as **2% of transitions** with reward is sufficient to rescue the feature.

**Scope:** results are from controlled synthetic environments. Transfer to real pretrained
encoders (V-JEPA 2-AC) is future work (M6, out of scope for this submission).

## Quick start

```bash
pip install -r requirements.txt     # pinned versions used for training
python verify.py                    # all checks pass in ~30s on CPU
python make_all_figures.py          # regenerate all figures from result JSONs
python make_paper_tables.py         # generate LaTeX tables
```

## Phase → command table

| phase | command | what it produces |
|---|---|---|
| verify | `python verify.py` | all invariant checks + e2e smoke |
| SwitchColor sweep (figures 1–5) | `python run_experiment.py --config full` | all_results.json |
| Quadrant matrix (figure 6) | `python quadrant_experiment.py --full` | quadrant_matrix.json |
| Cell-4 failure demo (figure 9) | `python quadrant_experiment.py --config cell4_failure --full` | quadrant_matrix_cell4_failure_quadrant.json |
| Cell-4 failure, gridworld (figure 9) | `python quadrant_experiment.py --config cell4_failure --env gridworld --full` | quadrant_matrix_cell4_failure_gridworld.json |
| Min reward signal (figure 7) | `python quadrant_experiment.py --config min_reward_signal --full` | min_reward_signal_quadrant.json |
| Capacity sweep (figure 8) | `python quadrant_experiment.py --config capacity_sweep --full` | capacity_sweep.json |
| Observation 1 numbers (M5) | `python proposition_numbers.py --full` | proposition_numbers_full.json |
| All figures | `python make_all_figures.py` | results/figures/*.{pdf,png} |
| Paper tables | `python make_paper_tables.py` | results/tables/*.tex |

See `REPRODUCE.md` for exact commands, runtimes, and hardware notes.

## Optional: Weights & Biases

No W&B setup is required. The codebase does not import `wandb` and runs fully offline.
To enable W&B logging, wrap `train_objective` / `train_model` calls with your own logger.

## Scope

Synthetic environments only. The pretrained-encoder comparison (V-JEPA 2-AC) is future work
and is not run during build or smoke checks.

## TMLR reproducibility checklist

- **Seeds:** all full runs use seeds [0, 1, 2]; eval seeds are train_seed + 100000 (disjoint).
- **Hardware:** NVIDIA GPU, CUDA 12.8, torch 2.11.0+cu128, Linux. CPU sufficient for smoke/verify.
- **Determinism:** CUDA nondeterminism; means over 3 seeds are stable to 2 decimal places.
- **Where each number comes from:** RESULTS.md cites the JSON file; each JSON has `_provenance`
  (git commit + library versions).
- **All figures regenerate from result JSONs** via `python make_all_figures.py` (no retraining).
- **All tables regenerate** via `python make_paper_tables.py`.
