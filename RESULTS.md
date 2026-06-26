# RESULTS

**Verdict: GO (TMLR-ready)** — self-prediction drops the exo+rel feature; reward grounding fixes
it selectively; the pattern holds across 2 environments and 6 latent sizes.

---

## Headline numbers (full training, 3 seeds)

| metric | value |
|---|:---:|
| JEPA cell-4 probe acc (quadrant) | 0.51 ± 0.01 |
| JEPA+Reward cell-4 probe acc (quadrant) | **1.00 ± 0.00** |
| JEPA cell-4 probe acc (gridworld) | 0.51 ± 0.01 |
| JEPA+Reward cell-4 probe acc (gridworld) | **1.00 ± 0.00** |
| Observation 1: bisim error (oracle − JEPA class dist) | **1.893** |
| JEPA latent class distance (cell 4, p=0.5) | 0.105 |
| JEPA linear probe acc (cell 4, full training) | 0.493 |
| Minimum reward_label_fraction to rescue feature | **0.02** |

Chance accuracy = 0.50. Retain threshold = 0.75.

---

## M3 — quadrant matrix (objective × cell), full training (3 seeds, 4000 steps)

`python quadrant_experiment.py --full`

Value = mean linear-probe accuracy of the online encoder for the tagged feature on a disjoint
eval stream (chance = 0.5, retain threshold ≥ 0.75). Heatmap: `results/figures/figure6_quadrant_matrix.png`.

|             | cell 1 (ctrl+rel) | cell 2 (ctrl+irr) | cell 3 (exo+irr) | cell 4 (exo+rel) |
|-------------|:---:|:---:|:---:|:---:|
| recon       | 1.00 | 1.00 | 1.00 | 1.00 |
| jepa        | 0.51 | 0.51 | 0.52 | 0.52 |
| jepa_ac     | 0.51 | 0.51 | 0.52 | 0.52 |
| jepa_ctrl   | 0.67 | 0.67 | 0.52 | 0.52 |
| jepa_invdyn¹| 1.00 | 1.00 | 0.51 | 1.00 |
| jepa_reward | 1.00 | 0.51 | 0.52 | **1.00** |
| oracle      | 1.00 | 1.00 | 1.00 | 1.00 |

¹ `jepa_invdyn` is trained on **informative**-action data (a* with ε=0.2 exploration); all other
rows see the random-policy stream. Its cell-4 rescue tracks action informativeness, not a
reward-grounded signal — see the caveat below.

**Cell 4 (the hard case: uncontrollable + relevant):** only `jepa_reward` (and the upper-bound
`oracle` and reconstruction baseline `recon`) retain the feature. All pure-prediction objectives
(`jepa`, `jepa_ac`, `jepa_ctrl`) drop it. `jepa_invdyn` rescues it only because it was trained on
informative actions — under the random-policy stream it also drops it (see M4 / figure9).

---

## M4 — cell-4 failure both envs (full training, 3 seeds, random-action policy)

`python quadrant_experiment.py --config cell4_failure --full`
`python quadrant_experiment.py --config cell4_failure --env gridworld --full`

Figure: `results/figures/figure9_cell4_failure_both_envs.png`

### Quadrant

| objective | probe acc | MI (nats) |
|-----------|:---:|:---:|
| recon       | 1.00 ± 0.00 | 0.693 ± 0.000 |
| jepa        | 0.51 ± 0.01 | 0.000 ± 0.000 |
| jepa_ac     | 0.52 ± 0.01 | 0.000 ± 0.000 |
| jepa_ctrl   | 0.52 ± 0.01 | 0.000 ± 0.000 |
| jepa_invdyn | 0.52 ± 0.01 | 0.000 ± 0.000 |
| jepa_reward | **1.00 ± 0.00** | **0.693 ± 0.000** |
| oracle      | 1.00 ± 0.00 | 0.693 ± 0.000 |

### GridWorld

| objective | probe acc | MI (nats) |
|-----------|:---:|:---:|
| recon       | 1.00 ± 0.00 | 0.693 ± 0.000 |
| jepa        | 0.51 ± 0.01 | 0.019 ± 0.004 |
| jepa_ac     | 0.61 ± 0.07 | 0.159 ± 0.110 |
| jepa_ctrl   | 0.52 ± 0.06 | 0.111 ± 0.091 |
| jepa_invdyn | 0.60 ± 0.07 | 0.077 ± 0.067 |
| jepa_reward | **1.00 ± 0.00** | **0.693 ± 0.000** |
| oracle      | 1.00 ± 0.00 | 0.693 ± 0.000 |

The gridworld surface form causes non-zero std for `jepa_ac`, `jepa_ctrl`, `jepa_invdyn` —
the goal-cell patch provides incidental leakage on some seeds, confirming the ordering is
meaningful but noisier on a richer-texture environment.

### Minimum reward signal

`python quadrant_experiment.py --config min_reward_signal --full`

The smallest `reward_label_fraction` for which `jepa_reward` retains cell 4 (probe acc ≥ 0.75)
on Quadrant: **0.02** (mean acc 0.773 over 3 seeds). Labelling ~2% of transitions is sufficient.
Figure: `results/figures/figure7_min_reward_signal_quadrant.png`.

### Capacity is not the bottleneck

`python quadrant_experiment.py --config capacity_sweep --full`

`jepa` stays near chance and `jepa_reward` stays near perfect across `latent_dim ∈ {16, 64, 128,
256, 512, 1024}` on both Quadrant and SwitchColor. The cell-4 drop is an objective property, not
a capacity limitation. Figure: `results/figures/figure8_capacity.png`.

---

<!-- M5-PROP:START -->
## M5 — Observation 1 (empirical): bisimulation error > 0 at JEPA convergence

`python proposition_numbers.py --full` (full: 4000 steps, latent 128, 1 seed). Cell 4 (uncontrollable + relevant), p=0.5, gamma=0.99. Latent class distance = whitened (Mahalanobis) distance between the value=0 and value=1 centroids on a disjoint eval stream; ~0 means the feature is collapsed.

| representation | latent class distance | linear-probe acc |
|---|:---:|:---:|
| analytical bisim (closed form, reward units) | 1.000 | — |
| jepa (pure self-prediction) | 0.105 | 0.49 |
| recon / AE (pixels) | 1.998 | 1.00 |
| oracle (ground-truth bits) | 1.998 | 1.00 |

The cell-4 feature has analytical bisimulation distance **1.00 > 0** (immediate reward gap 1.0; the exogenous future term vanishes at p=0.5), so bisimulation theory keeps it. Only the reward-grounded oracle and the pixel-grounded AE realize that separation in latent space (class distance ~2.0); the JEPA latent realizes only **5%** of it (0.105). The **bisimulation error (oracle − JEPA = 1.893) stays > 0** empirically: latent self-prediction has no gradient toward an unpredictable-but-relevant feature and collapses it even though it is trivially encodable. (Full training, 1 seed; the residual JEPA class distance 0.105 is near-zero.)
<!-- M5-PROP:END -->

---

## Scope and limitations

- **Synthetic environments only.** All results are from controlled grid-world environments
  (`QuadrantEnv`, `SwitchColorEnv`, `GridWorldHiddenRuleEnv`) designed to isolate the
  controllability × relevance axes. Real-world transfer has not been tested.
- **Real-model transfer (V-JEPA 2-AC) is future work (M6, guarded download).** The harness for
  it exists in the codebase but is intentionally not run here; results should not be extrapolated
  beyond these synthetic settings.
- **Observation 1 is empirical, not a proved theorem.** The analytical bisimulation distance is
  a closed-form derivation (math is correct), but the claim that JEPA collapses the feature
  is measured from trained models on a finite budget. The pattern is stable across 3 seeds and
  2 environments but has not been proved in general.
- **`jepa_invdyn` caveat.** Inverse-dynamics rescue of cell 4 tracks action informativeness: it
  works when trained on informative-action data and fails under a random-policy stream. This is
  not a counterexample to the claim; it illustrates that a different signal (action labels) can
  also ground the feature.
- **`jepa_reward` leaks cell 3** (irrelevant distractor) slightly in smoke runs. Under full
  training (3 seeds, 4000 steps) the ordering is clean; the leak is a short-training artifact.
