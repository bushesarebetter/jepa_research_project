# PAPER_FACTS.md — every number the paper may cite, with its source file

**Purpose.** Single source of truth for the prose. Every number below was read directly from a
file in `results/` (or computed by re-aggregating the per-run records in those files). The
**file always wins** over the writing brief, RESULTS.md, or memory. Where a brief placeholder
disagrees with the file, the file value is written and the discrepancy is noted in §4.

**Conventions**
- All paths are relative to the repo root `jepa_final_results/`.
- `±` is **population std** (`numpy.std`, ddof=0) over the 3 seeds — this is exactly what
  `make_paper_tables.py` writes into the `.tex` tables (`fmt(np.mean, np.std)`). Per-seed values
  are listed so sample std can be recomputed if a reviewer wants ddof=1.
- Chance = 0.50. Retain threshold = 0.75. Full training = 4000 steps, latent_dim 128,
  mi_epochs 300, predictability p=0.5, seeds {0,1,2} (source: `config_params` in each JSON).
- "Probe acc" = linear probe accuracy of the **online** encoder on a disjoint eval stream.
- MI = InfoNCE estimate in nats; theoretical max for a 1-bit feature = log 2 ≈ 0.6931.

---

## 1. Numbers from each result JSON

### 1a. Quadrant matrix — `results/quadrant_matrix.json` (84 runs = 7 obj × 4 cells × 3 seeds, random-action policy)

`config_params`: n_steps 4000, train_n 20000, eval_n 4000, batch 128, lr 1e-3, latent_dim 128,
mi_epochs 300, predictability 0.5, seeds [0,1,2], img_size 12.

**Linear probe accuracy (mean ± popstd; per-seed [s0, s1, s2]):**

| objective | cell 1 (ctrl+rel) | cell 2 (ctrl+irr) | cell 3 (exo+irr) | cell 4 (exo+rel) |
|---|---|---|---|---|
| recon       | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| jepa        | 0.512 ± 0.008 [.5215,.5015,.5117] | 0.512 ± 0.009 [.5215,.4995,.5138] | 0.515 ± 0.016 [.4935,.5315,.5202] | 0.515 ± 0.016 [.4930,.5325,.5195] |
| jepa_ac     | 0.510 ± 0.009 [.5232,.5043,.5028] | 0.510 ± 0.008 [.5212,.5047,.5035] | 0.518 ± 0.026 [.4825,.5450,.5265] | 0.518 ± 0.025 [.4837,.5443,.5252] |
| jepa_ctrl   | **0.675 ± 0.230** [.5218,.5028,1.000] | **0.674 ± 0.231** [.5217,.5010,1.000] | 0.516 ± 0.021 [.4873,.5262,.5352] | 0.518 ± 0.022 [.4867,.5292,.5367] |
| jepa_invdyn¹| 1.00 ± 0.00 | 1.00 ± 0.00 | 0.509 ± 0.027 [.4715,.5315,.5242] | 1.00 ± 0.00 |
| jepa_reward | 1.00 ± 0.00 | 0.515 ± 0.007 [.5162,.5220,.5060] | 0.516 ± 0.017 [.4928,.5338,.5215] | **1.00 ± 0.00** |
| oracle      | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

¹ `jepa_invdyn` in this matrix is trained on **informative-action** data (a* with ε=0.2). Its
cell-4 rescue (1.00) tracks action informativeness, not reward grounding — under random actions it
drops (see §1b).

> ⚠️ `jepa_ctrl` cells 1/2 mean 0.67 is **bimodal**, not a stable retain: seeds 0,1 ≈ chance
> (0.52, 0.50), seed 2 = 1.00. popstd 0.23 (sample std 0.28). The `.tex` table reports
> `0.67 ± 0.23` faithfully; prose should not call this a clean "retain."

**MI (InfoNCE, nats) — mean ± popstd; notable per-seed:**

| objective | cell 1 | cell 2 | cell 3 | cell 4 |
|---|---|---|---|---|
| recon       | 0.693 ± 0.000 | 0.693 ± 0.000 | 0.693 ± 0.000 | 0.693 ± 0.000 |
| jepa        | 0.000 | 0.000 | 0.000 | 0.000 |
| jepa_ac     | 0.000 | 0.000 | 0.000 | 0.000 |
| jepa_ctrl   | 0.231 ± 0.327 [0,0,.693] | 0.231 ± 0.327 [0,0,.693] | 0.000 | 0.000 |
| jepa_invdyn | 0.693 ± 0.000 | 0.693 ± 0.000 | 0.000 | 0.693 ± 0.000 |
| jepa_reward | 0.693 ± 0.000 | 0.000 | 0.000 | 0.693 ± 0.000 |
| oracle      | 0.693 ± 0.000 | 0.693 ± 0.000 | 0.693 ± 0.000 | 0.693 ± 0.000 |

**Effective rank — mean ± popstd (collapse check; threshold 5):**

| objective | cell 1 | cell 2 | cell 3 | cell 4 |
|---|---|---|---|---|
| recon       | 11.17 ± 0.13 | 11.40 ± 0.33 | 11.93 ± 0.20 | 11.65 ± 0.38 |
| jepa        | 41.98 ± 4.20 | 41.98 ± 4.20 | 41.96 ± 3.44 | 41.96 ± 3.44 |
| jepa_ac     | 40.77 ± 5.74 | 40.77 ± 5.74 | 41.93 ± 4.34 | 41.93 ± 4.34 |
| jepa_ctrl   | 38.97 ± 5.18 | 38.97 ± 5.18 | 41.74 ± 3.81 | 41.74 ± 3.80 |
| jepa_invdyn | 37.15 ± 2.39 | 29.48 ± 1.01 | 34.44 ± 1.76 | 28.76 ± 0.91 |
| jepa_reward | 24.24 ± 1.94 | 61.95 ± 3.09 | 67.01 ± 5.65 | 24.07 ± 1.25 |
| oracle      | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

→ Dropping JEPA-family objectives sit at **eff_rank ≈ 38–42** on quadrant (NOT collapse; the
feature is dropped despite high rank). Oracle = 1.0 by construction (one-hot bits). Brief's
"quadrant dropping objectives 38–42" ✓.

The aggregated `matrix_linear_probe_acc` field in the JSON stores the same means used in table 1.

---

### 1b. Cell-4 failure, Quadrant — `results/quadrant_matrix_cell4_failure_quadrant.json` (21 runs, **random-action** policy, cell 4 only)

| objective | probe acc (mean ± popstd) | per-seed [s0,s1,s2] | MI (nats) | eff_rank |
|---|---|---|---|---|
| recon       | 1.00 ± 0.00 | [1,1,1] | 0.693 ± 0.000 | 11.75 ± 0.44 |
| jepa        | 0.514 ± 0.017 | [.4907,.5320,.5190] | 0.000 | 41.96 ± 3.44 |
| jepa_ac     | 0.518 ± 0.025 | [.4843,.5447,.5260] | 0.000 | 41.93 ± 4.34 |
| jepa_ctrl   | 0.516 ± 0.021 | [.4870,.5275,.5350] | 0.000 | 41.74 ± 3.81 |
| jepa_invdyn | 0.517 ± 0.021 | [.4867,.5285,.5347] | 0.000 | 41.74 ± 3.81 |
| jepa_reward | **1.00 ± 0.00** | [1,1,1] | **0.693 ± 0.000** | 24.07 ± 1.25 |
| oracle      | 1.00 ± 0.00 | [1,1,1] | 0.693 ± 0.000 | 1.00 ± 0.00 |

→ This is the config behind the headline "all reward-free objectives fail on cell 4 (quadrant)."
Here `jepa_invdyn` = 0.52 (random actions), unlike the informative-action matrix where it = 1.00.

### 1c. Cell-4 failure, GridWorld — `results/quadrant_matrix_cell4_failure_gridworld.json` (21 runs, **random-action** policy, cell 4 only)

| objective | probe acc (mean ± popstd) | per-seed [s0,s1,s2] | MI (nats, mean ± popstd) | eff_rank (mean ± popstd) |
|---|---|---|---|---|
| recon       | 1.00 ± 0.00 | [1,1,1] | 0.693 ± 0.000 | 3.30 ± 0.17 |
| jepa        | 0.508 ± 0.015 | [.4912,.5280,.5050] | 0.020 ± 0.004 | 3.01 ± 0.01 |
| jepa_ac     | 0.607 ± 0.078 | [.6190,.6955,.5050] | 0.159 ± 0.110 [.1925,.2738,.0106] | 3.00 ± 0.03 |
| jepa_ctrl   | 0.518 ± 0.028 | [.4912,.5043,.5570] | 0.111 ± 0.091 [.0906,.0103,.2312] | 2.93 ± 0.13 |
| **jepa_invdyn** | **0.603 ± 0.105** | **[.7512,.5280,.5308]** | 0.077 ± 0.067 [.1712,.0370,.0233] | 3.00 ± 0.01 |
| jepa_reward | **1.00 ± 0.00** | [1,1,1] | **0.693 ± 0.000** | 4.59 ± 0.01 |
| oracle      | 1.00 ± 0.00 | [1,1,1] | 0.693 ± 0.000 | 1.00 ± 0.00 |

> ⚠️ **`jepa_invdyn` gridworld std is large and meaningful (flagged by the brief).** popstd 0.105
> (sample std 0.128), driven by seed 0 = 0.7512 vs seeds 1,2 ≈ 0.53. The rescue tracks whether
> random actions happened to carry signal on a given seed. `jepa_ac` is also noisy
> (0.607 ± 0.078; seed-2 = 0.505 vs seeds 0,1 = 0.62/0.70). Report **mean ± std**, never mean
> alone, for these two.
>
> GridWorld eff_rank ≈ 3 for all objectives (intrinsic dimensionality of the gridworld skin, NOT
> collapse — `recon` retains the feature perfectly at eff_rank 3.30, seed-0 = 3.48). Range across
> all cell-4 gridworld runs: **2.75** (jepa_ctrl s0) → **4.61** (jepa_reward s2). Brief's
> "gridworld eff_rank 2.75–4.61" ✓.

---

### 1d. Capacity sweep — `results/capacity_sweep.json` (cell 4, latent_dim {16,64,128,256,512,1024}, 3 seeds; file stores 3-seed **means** only, no per-seed)

Cell-4 linear probe acc vs latent_dim:

| latent_dim | quadrant/jepa | quadrant/jepa_reward | switch_color/jepa | switch_color/jepa_reward |
|---|---|---|---|---|
| 16   | 0.5096 | 1.0000 | 0.5096 | 1.0000 |
| 64   | 0.5135 | 1.0000 | 0.5135 | 1.0000 |
| 128  | 0.5143 | 1.0000 | 0.5150 | 1.0000 |
| 256  | 0.5378 | 1.0000 | 0.5382 | 1.0000 |
| 512  | 0.5524 | 1.0000 | 0.5514 | 1.0000 |
| 1024 | 0.5577 | 0.9069 | 0.5590 | 0.9322 |

→ `jepa` flat-low 0.51→0.56 (never crosses 0.75); `jepa_reward` flat-high 1.00, dipping at
latent 1024 (0.907 quadrant / 0.932 switch — a training-budget artifact at the largest dim, not a
capacity floor). Brief's "0.510→0.558" ✓ (quadrant); "jepa_reward 0.907 at 1024" ✓.

---

### 1e. Minimum reward signal — `results/min_reward_signal_quadrant.json` (cell 4, quadrant, 3 seeds; file stores **means** only)

| reward_fraction | mean cell-4 probe acc |
|---|---|
| 0.01 | 0.7226 |
| 0.02 | 0.7733 |
| 0.05 | 1.0000 |
| 0.10 | 1.0000 |
| 0.25 | 1.0000 |
| 0.50 | 1.0000 |
| 1.00 | 1.0000 |

`min_fraction_passing` = **0.02** (smallest fraction with mean acc ≥ 0.75; mean acc 0.773).
At 1% the acc is 0.7226 (already above chance, just below threshold). Brief "2% / 1%→0.72" ✓.

---

### 1f. Proposition / Observation 1 — `results/proposition_numbers_full.json` (**FULL** training: 4000 steps, latent 128, **1 seed** [seed 0]; cell 4, p=0.5, γ=0.99)

Used the **full** file, not `proposition_numbers_smoke.json`.

| quantity | value | field |
|---|---|---|
| analytical bisim distance (cell 4, p=0.5) | **1.000** | `analytical_bisim_distance` |
| jepa latent class distance | **0.10523** | `latent_class_distance.jepa` |
| recon (AE) latent class distance | 1.99765 | `latent_class_distance.recon` |
| oracle latent class distance | 1.99823 | `latent_class_distance.oracle` |
| jepa linear probe acc | **0.49275** (chance) | `linear_probe_acc.jepa` |
| recon probe acc | 1.000 | `linear_probe_acc.recon` |
| oracle probe acc | 1.000 | `linear_probe_acc.oracle` |
| jepa retention vs oracle | 0.05266 (**≈5%**) | `jepa_retention_vs_oracle` |
| **bisim error (oracle − jepa class dist)** | **1.89300** | `bisim_error_vs_jepa` |

→ JEPA realizes ~5% of the oracle's class separation; bisim error 1.893 (out of max ≈ 2.0).
Single seed — report as a point estimate, not mean±std. RESULTS.md headline (1.893, 0.105, 0.493)
and the brief's "definitive" block (1.893, 0.105, 5%) both match this file ✓.

---

### 1g. Background / motivation — original switch-color sweep — `results/all_results.json` (36 runs = 3 models × 4 p_repeat × 3 seeds; per-seed files `results/p{p}_seed{s}_{jepa,ae,jepa_invdyn}_result.json`)

This is the **predictability-knob** demonstration (probe of the `switch_color` feature, 32×32 env).

**jepa vs ae probe acc at the two endpoints (mean ± popstd; per-seed):**

| model | p=0.5 (i.i.d., unpredictable) | p=1.0 (constant, predictable) |
|---|---|---|
| **jepa** | **0.614 ± 0.132** [.5132, .8012, .5284] | **1.000 ± 0.000** [1,1,1] |
| **ae**   | **1.000 ± 0.000** [1,1,1] | **1.000 ± 0.000** [1,1,1] |

Full sweep (all p, for figure 1):

| model | p=0.5 | p=0.65 | p=0.8 | p=1.0 |
|---|---|---|---|---|
| jepa        | 0.614 ± 0.132 | 0.587 ± 0.044 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ae          | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| jepa_invdyn | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

> ⚠️ Nuance: in the **switch-color** env, JEPA at p=0.5 = **0.614 (noisy, bimodal: seed 1 = 0.80)**,
> NOT the clean ~chance value. The clean ~0.51 chance result is the **quadrant** env (§1a/§1b).
> The paper's motivation figure (figure1) is switch-color; the main result (figure9) is quadrant.
> Do not conflate the two when quoting "JEPA drops the feature."

Other metrics from the same file (for figures 2–4), jepa vs ae:

| | jepa p=0.5 | jepa p=1.0 | ae p=0.5 | ae p=1.0 |
|---|---|---|---|---|
| control_regret | 0.404 ± 0.110 | 0.000 ± 0.000 | 0.000 | 0.000 |
| eff_rank | 8.77 ± 1.62 | 18.26 ± 3.87 | 3.09 ± 0.27 | 3.69 ± 0.30 |
| centroid_dist_norm | 0.321 ± 0.048 | 10.81 ± 2.53 | 7.46 ± 0.31 | 8.54 ± 0.81 |

(`control_regret` here is the probe-based regret stored per run; see §4-D5 on the regret-vs-MI figure.)

---

## 2. Figures — `results/figures/` (every listed figure has a `.pdf`; confirmed present)

Generator: `make_all_figures.py` → `src/figures.py`. Figures 1–5 use `all_results.json`
(switch-color sweep); 6–9 use the quadrant JSONs.

| file (.pdf ✓) | source data | what it plots |
|---|---|---|
| `figure1_main_result.pdf` | all_results.json | L: switch_color probe acc vs p_repeat (jepa/ae/jepa_invdyn, chance line). R: JEPA linear vs nonlinear probe acc vs p_repeat. |
| `figure2_bisimulation.pdf` | all_results.json | L: normalized centroid distance (c=0 vs c=1) vs p_repeat. R: control_regret vs p_repeat. |
| `figure3_theory_vs_empirical.pdf` | all_results.json | Scatter: control_regret (y) vs **linear_probe_acc** (x), with line `regret = 1 − acc`. **(probe-acc-based, not MI — see §4-D5.)** |
| `figure4_ablations.pdf` | all_results.json | L: bar of probe acc at p=0.5 per model. R: effective_rank vs p_repeat (jepa vs ae). |
| `figure5_environment.pdf` | SwitchColorEnv (rendered, no JSON) | Frame strips: p=0.5 (top, i.i.d.) vs p=1.0 (bottom, constant). |
| `figure6_quadrant_matrix.pdf` | quadrant_matrix.json | 7-objective × 4-cell heatmap of probe acc (RdYlGn, threshold 0.75). |
| `figure7_min_reward_signal_quadrant.pdf` | min_reward_signal_quadrant.json | Cell-4 acc vs reward_label_fraction (log x), min-passing 0.02 annotated. |
| `figure8_capacity.pdf` | capacity_sweep.json | Cell-4 acc vs latent_dim (log2 x), jepa vs jepa_reward, both envs. |
| `figure9_cell4_failure_both_envs.pdf` | cell4_failure_{quadrant,gridworld}.json | **MAIN FIGURE.** L: probe acc grouped bars (quadrant vs gridworld) per objective, threshold+chance lines. R: InfoNCE MI per objective, log-2 line. Error bars = std over 3 seeds. |

Also present (not for the main paper): `_smoke` variants of figures 6–9, and
`figure6_cell4_failure_{gridworld,quadrant}.pdf`. All have matching `.png`.

Brief's figure numbering (Fig 1 = figure9, Fig 2 = figure6, Fig 3 = figure8, Fig 4 = figure7,
Fig 5 = figure2, Fig 6 = figure3) is a **paper-order remap** of these filenames — keep the
mapping straight when writing captions.

---

## 3. Tables — `results/tables/` (every table has a `.tex`; confirmed present)

Generator: `make_paper_tables.py`. `\input` path = `results/tables/<name>` (drop `.tex`), relative
to repo root. All use `mean ± np.std` (population) over 3 seeds, booktabs.

| file (.tex ✓) | `\input{...}` | contents | source JSON |
|---|---|---|---|
| `table1_quadrant.tex` | `\input{results/tables/table1_quadrant}` | full 7×4 objective×cell probe-acc matrix (mean±std) | quadrant_matrix.json |
| `table2_cell4.tex` | `\input{results/tables/table2_cell4}` | cell-4 failure both envs: probe acc + MI (mean±std) per objective | cell4_failure_{quadrant,gridworld}.json |
| `table3_capacity_minreward.tex` | `\input{results/tables/table3_capacity_minreward}` | capacity sweep (means) + min-reward-fraction sub-table (min passing 0.02) | capacity_sweep.json, min_reward_signal_quadrant.json |

Values reproduced from the `.tex` exactly: table1 `jepa` cell3/4 = `0.52 ± 0.02`,
`jepa_ctrl` cell1/2 = `0.67 ± 0.23`; table2 quadrant `jepa` = `0.51 ± 0.02`, gridworld
`jepa_invdyn` = `0.60 ± 0.10`, `jepa_ac` = `0.61 ± 0.08`, `jepa_ctrl` = `0.52 ± 0.03`.

---

## 4. Cross-check vs the writing brief — file wins; discrepancies noted

Brief's "Numbers to pull from JSONs" list vs the files:

| brief placeholder | file value (write this) | source | verdict |
|---|---|---|---|
| jepa cell-4 probe, quadrant ~0.508 ± std | 0.515 ± 0.016 (matrix) / 0.514 ± 0.017 (cell4_failure) | quadrant_matrix.json / ..._failure_quadrant.json | **D3** brief low; use file |
| jepa cell-4 probe, gridworld ~0.508 ± std | 0.508 ± 0.015 | ..._failure_gridworld.json | ✓ matches |
| jepa_reward cell-4 probe, both 1.00 ± 0 | 1.00 ± 0.00 | both failure files | ✓ |
| InfoNCE MI jepa cell-4 ~0.0 | 0.000 (quadrant) / 0.020 ± 0.004 (gridworld) | failure files | ✓ (gridworld is ~0.02, not exactly 0) |
| InfoNCE MI jepa_reward cell-4 0.693 | 0.693 ± 0.000 | failure files | ✓ |
| min_fraction_passing 0.02 | 0.02 | min_reward_signal_quadrant.json | ✓ |
| bisim error 1.893 | 1.89300 | proposition_numbers_full.json | ✓ |
| JEPA class separation 0.105 (5% of oracle 1.998) | 0.10523; retention 0.05266; oracle 1.99823 | proposition_numbers_full.json | ✓ |
| eff_rank quadrant dropping objectives 38–42 | 38.97–41.98 | quadrant_matrix.json | ✓ |
| eff_rank gridworld 2.75–4.61 | 2.75–4.61 | ..._failure_gridworld.json | ✓ |
| jepa_invdyn gridworld report mean±std | 0.603 ± 0.105 | ..._failure_gridworld.json | ✓ (std large, as flagged) |

**Discrepancies (file wins):**

- **D1 — RESULTS.md M4 GridWorld std is STALE.** RESULTS.md prints `jepa_ac 0.61 ± 0.07`,
  `jepa_ctrl 0.52 ± 0.06`, `jepa_invdyn 0.60 ± 0.07`. The JSON (and `table2_cell4.tex`, which uses
  `np.std`) give `jepa_ac 0.61 ± 0.08`, `jepa_ctrl 0.52 ± 0.03`, `jepa_invdyn 0.60 ± 0.10`. **Use
  the `.tex`/JSON values.** (jepa_ctrl 0.06→0.03 and jepa_invdyn 0.07→0.10 are not rounding.)
- **D2 — "± 0.01" understates JEPA cell-4 std.** RESULTS.md headline & M4 quadrant say
  `0.51 ± 0.01`; JSON/`table` give `± 0.02` (popstd 0.016–0.017). Use ± 0.02.
- **D3 — brief placeholder 0.508 for jepa quadrant cell-4** is below the file (0.514–0.515). Minor;
  use file. Note the **two quadrant sources differ**: `quadrant_matrix.json` cell-4 = 0.515
  (table 1 rounds to **0.52**); `..._cell4_failure_quadrant.json` = 0.514 (table 2 rounds to
  **0.51**). They are different experiments (full matrix vs cell-4-only rerun) — cite the matching
  table, don't average them.
- **D4 — RESULTS.md M3 matrix shows `jepa_ac` as `0.51/0.51/0.52/0.52`** (it labels the row
  `jepa_ac`); fine, matches file. But the same RESULTS.md M3 table omits the **jepa_ctrl bimodal
  variance** (prints `0.67` with no ±). The `.tex` table correctly carries `± 0.23`. Prose must
  include the std or drop the "retain" framing for jepa_ctrl cells 1–2.
- **D5 — figure3 is the CIRCULAR version, contradicting STATUS.md/brief.** STATUS.md M2 and the
  brief (Fig 6) describe `figure3_theory_vs_empirical` as "x = estimated I(Z;c), y = rollout
  regret (de-circularized)." The actual `src/figures.py:figure3_theory_vs_empirical` plots
  **x = linear_probe_acc, y = control_regret** with the line `regret = 1 − acc` — i.e. regret vs
  probe accuracy on the switch-color sweep. There is **no MI-vs-regret figure** in `results/figures/`.
  Either regenerate the figure to match the claim or describe it as probe-acc-vs-regret. See §5.
- **D6 — proposition smoke ≠ full.** STATUS.md M5 narrates the **smoke** numbers (jepa class dist
  0.97, "48% of oracle"). The paper must use `proposition_numbers_full.json` (jepa 0.105, 5% of
  oracle, bisim error 1.893). Brief and RESULTS.md headline already use the full numbers.

---

## 5. GAPS — claims the brief wants but no file fully supports (do NOT cite without a source)

- **"Rollout regret (independent policy) vs InfoNCE MI" — non-circular theory figure.** The brief
  (Sec 5, Fig 6) and STATUS.md M2 promise a de-circularized MI-vs-regret plot. The shipped
  `figure3_theory_vs_empirical.pdf` is **probe-acc vs control_regret** (circular). The MI data
  (quadrant JSONs) and the regret data (switch-color JSONs) come from **different experiments** and
  are never jointly plotted. No file supports "regret tracks retained MI monotonically" as a single
  figure. → Regenerate or reframe; do not claim the MI-vs-regret relationship from current files.
- **Rollout regret on the quadrant / cell-4 experiments.** The quadrant matrix and cell-4 failure
  JSONs contain **no `control_regret` / rollout-regret field** (only probe acc, MI, eff_rank).
  Regret numbers exist **only** for the switch-color sweep (`all_results.json`). Any "control
  regret" statement about cell 4 / the objective zoo has no file backing.
- **V-JEPA 2-AC / real-model (M6) results.** None exist — M6 is explicitly skipped (STATUS.md M6,
  RESULTS.md scope). Frame strictly as future work; cite no real-model numbers.
- **"All six reward-free objectives fail on cell 4" needs a scoping caveat, not a number.** True
  only under the **random-action** cell-4 config (§1b: jepa_invdyn = 0.52). In the
  informative-action matrix (§1a) `jepa_invdyn` cell-4 = **1.00**, and on gridworld it is
  0.60 ± 0.10. The "all fail" claim is sourced only to `..._cell4_failure_quadrant.json`; state the
  policy condition.
- **Per-seed values for capacity and min-reward.** `capacity_sweep.json` and
  `min_reward_signal_quadrant.json` store **3-seed means only** — no per-seed array, so no honest
  std for those tables (table 3 correctly shows means without ±). Per-run files exist under
  `results/quadrant/` if a reviewer demands error bars, but the aggregate files alone cannot
  produce them.

---

## Provenance (from `_provenance` in the JSONs)
git_commit `520c76edeb39c64fd218cbeac7a01217765e4a3f`; python 3.12.13, torch 2.11.0+cu128,
numpy 2.0.2, sklearn 1.6.1, matplotlib 3.10.0; platform Linux (Colab GPU).
