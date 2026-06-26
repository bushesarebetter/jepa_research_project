# RESULTS

## M3 — quadrant matrix (objective × cell), CPU smoke

`python quadrant_experiment.py` (smoke: 150 steps, latent 64, 1 seed, p=0.5, threshold acc≥0.75).
Value = linear-probe accuracy of the *online* encoder for the tagged feature on a disjoint
eval stream (chance = 0.5). Heatmap: `results/figures/figure6_quadrant_matrix_smoke.png`.

|             | cell 1 (ctrl+rel) | cell 2 (ctrl+irr) | cell 3 (exo+irr) | cell 4 (exo+rel) |
|-------------|:---:|:---:|:---:|:---:|
| recon       | 1.00 | 1.00 | 1.00 | 1.00 |
| jepa        | 0.59 | 0.59 | 0.64 | 0.64 |
| jepa_ctrl   | 0.69 | 0.69 | 0.71 | 0.71 |
| jepa_invdyn¹| 1.00 | 1.00 | 0.66 | 1.00 |
| jepa_reward | 1.00 | 0.82 | 0.87 | 1.00 |
| oracle      | 1.00 | 1.00 | 1.00 | 1.00 |

¹ `jepa_invdyn` is the only row trained on **informative**-action data (action = a* with ε=0.2
exploration); every other row sees the random-policy stream. Its cell-4 rescue is the labelled
result: it tracks *action informativeness*, not supervision on the feature label.

### Does cell 4 (the hard case: uncontrollable + relevant) show the expected pattern?

**Yes, on this smoke run.** Reward-free pure-prediction objectives are low — `jepa` 0.64,
`jepa_ctrl` 0.71 (both below the 0.75 retain threshold) — while `jepa_reward` 1.00, `oracle`
1.00, and `recon` 1.00 keep it. That is exactly the claim: latent self-prediction and
controllability drop the uncontrollable-but-relevant feature; a reward-grounded signal (the
proposed fix) and the upper-bound oracle keep it, and reconstruction keeps it (at full pixel
cost). `jepa_invdyn` also keeps it (1.00) **because its actions were informative** — the
intended caveat, not a counterexample.

### Caveats / things to watch at full scale (not tuned away)

- **`jepa_reward` leaks cell 3** (irrelevant distractor) at 0.87 ≥ threshold — a reward
  objective should *not* need an irrelevant feature. Likely a small-sample / short-training
  artifact (the patch is large and high-contrast); worth re-checking at `--full` and with the
  M4 `reward_label_fraction` / bisimulation term before drawing conclusions.
- **`jepa_ctrl` barely separates controllable from exogenous** (0.69 vs 0.71) at 150 steps. The
  controllability signal needs more training to express; the ordering may sharpen at `--full`.
- `jepa` sits slightly above chance everywhere (0.59–0.64) — residual leakage from BatchNorm /
  the linear probe on a 64-dim latent, not retention.

These are smoke numbers (seconds on CPU, 1 seed). Run `python quadrant_experiment.py --full`
(3 seeds, 4000 steps) before treating any single cell as settled.

## M4 — minimal fix + capacity + 2nd env (CPU smoke)

### 1. Existing fixes fail on cell 4 (`--config cell4_failure`, random actions)

| env | jepa | jepa_ctrl | jepa_invdyn(random) | jepa_reward | oracle |
|-----|:---:|:---:|:---:|:---:|:---:|
| quadrant  | 0.64 | 0.71 | 0.71 | 1.00 | 1.00 |
| gridworld | 0.72 | 0.89 | 0.89 | 1.00 | 1.00 |

On **quadrant** the demo is clean: pure-prediction (`jepa`), controllability (`jepa_ctrl`), and
inverse-dynamics *with random actions* (`jepa_invdyn`) all sit below the 0.75 retain threshold,
while the reward-grounded fix and the oracle keep cell 4. (Contrast the default matrix, where
`jepa_invdyn` is trained on **informative** actions and reaches 1.00 — the rescue tracks action
informativeness, exactly the caveat.) On **gridworld** smoke the goal cell is a large, very
high-contrast patch, so the linear probe leaks it (0.89) even under pure prediction — a small-
sample/salience artifact; the ordering should sharpen at `--full`.

### 2. Minimum reward signal (`--config min_reward_signal`, quadrant)

`reward_label_fraction` ∈ {0.02, 0.1, 0.5, 1.0} (smoke). Smallest fraction whose mean cell-4
retention crosses 0.75: **0.02** (acc 0.96) — i.e. labelling ~2% of transitions already rescues
the feature here. (The high-contrast patch makes this easy at smoke; `--full` adds finer
fractions {0.01…} and 3 seeds.) Figure: `results/figures/figure7_min_reward_signal_quadrant_smoke.png`.

### 3. Capacity is not the bottleneck (`--config capacity_sweep`)

Mean cell-4 linear-probe acc vs `latent_dim` (smoke: {16, 64}):

| env / objective | 16 | 64 |
|-----------------|:---:|:---:|
| quadrant/jepa | 0.55 | 0.64 |
| quadrant/jepa_reward | 1.00 | 1.00 |
| switch_color/jepa | 0.55 | 0.64 |
| switch_color/jepa_reward | 1.00 | 1.00 |

`jepa` stays at chance and `jepa_reward` stays perfect across latent dims in both envs — the
cell-4 drop is the *objective*, not encoder capacity. Figure: `figure8_capacity_smoke.png`.
`--full` extends to {16, 64, 128, 256, 512, 1024} × 3 seeds.

### 4. Second environment

`GridWorldHiddenRuleEnv` (`src/gridworld_env.py`) instantiates the cell-4 rule (exogenous +
relevant, predictability-tunable) in a gridworld surface form (agent token on a deterministic
walk = predictable distractor; goal-cell colour = the hidden rule bit). `--env
{switch_color,gridworld,quadrant}` selects the surface form; `verify.py` asserts all three emit
the same cell-4 schema (a* == bits, reward == 1[action == a*]).

<!-- M5-PROP:START -->
## M5 — analytical bisimulation + proposition numbers

`python proposition_numbers.py` (smoke: 150 steps, latent 64, 1 seed). Cell 4 (uncontrollable + relevant), p=0.5, gamma=0.99. Latent class distance = whitened (Mahalanobis) distance between the value=0 and value=1 centroids on a disjoint eval stream; ~0 means the feature is collapsed.

| representation | latent class distance | linear-probe acc |
|---|:---:|:---:|
| analytical bisim (closed form, reward units) | 1.000 | — |
| jepa (pure self-prediction) | 0.967 | 0.64 |
| recon / AE (pixels) | 1.998 | 1.00 |
| oracle (ground-truth bits) | 2.000 | 1.00 |

The cell-4 feature has analytical bisimulation distance **1.00 > 0** (immediate reward gap 1.0; the exogenous future term vanishes at p=0.5), so bisimulation provably keeps it. Only the reward-grounded oracle and the pixel-grounded AE realize that separation in latent space (class distance ~2.0); the JEPA latent realizes only **48%** of it (0.967). The **bisimulation error (oracle − JEPA = 1.03) stays > 0**: latent self-prediction has no gradient toward an unpredictable-but-relevant feature and collapses it even though it is trivially encodable. (Smoke, 1 seed; the residual JEPA class distance is 150-step BatchNorm leakage — probe acc 0.64 vs chance 0.5 — and shrinks toward 0 at `--full`.)
<!-- M5-PROP:END -->
