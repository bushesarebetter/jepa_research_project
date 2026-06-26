# PLAN.md — JEPA control-loss experiment pipeline

This is the architecture + the 8-milestone plan with done-criteria. The claim, spine, and
5 scientific invariants live in **CLAUDE.md** (re-read it each prompt). Checkbox state lives in
**STATUS.md**.

---

## Architecture

A thin orchestrator over single-responsibility modules. Data flows one way; config and
provenance cut across.

```
config (YAML/CLI)  ─┐
                    ▼
   env.py  ──►  rollouts (.npz cache)  ──►  training.py  ──►  trained model
                                                  │ build_model()        │ get_latents()
                                                  ▼                      ▼
                                              models.py            evaluation.py  ──► metrics
                                          (shared Encoder)                              │
                                                                                       ▼
                                          provenance.py ─────────────►  result JSON  ──► figures.py
                                                                       (source of truth)   │
                                                                              │            ▼
                                                                              ▼        figures/*.png
                                                                       make_decision ──► decision.json
```

**Layering rules**
- `run_experiment.py` is the *only* orchestrator: it owns the sweep loop, the GO/NO-GO decision,
  and the figures entrypoint. No training/eval logic lives here.
- `src/*` modules are imported, never invoked as scripts. Each has one job.
- **Config layer** (M1): a YAML file resolved into the same `cfg` dict the CLI already builds, so
  legacy flags and config files coexist. Two canonical configs: `smoke` (CPU, seconds) and
  `full` (GPU). `--device` defaults to **auto**.
- **Provenance** (M0): every freshly-computed result JSON gets a `_provenance` block (git commit,
  full config, library versions). Additive and namespaced — invisible to existing readers.
- **Caching / idempotency**: rollouts cache to `.npz`; **skip-if-done keys on the result JSON**,
  never on checkpoints (invariant 5). Re-running a finished cell is a no-op.
- **No network during build**: large pretrained weights are M6-only, behind an explicit
  `--download-weights` guard; smoke/test never touch the network.

---

## Milestones & done-criteria

### M0 — Scaffolding + plan  *(this prompt)*
Set up the spine of the repo and the plan; change no experiment behavior.
**Done when:**
- [x] Every existing file read before any change.
- [x] `CLAUDE.md`, `PLAN.md`, `STATUS.md` written.
- [x] `configs/` (empty, `.gitkeep`), `results/SCHEMA.md`, `verify.py` stub (imports) exist.
- [x] `src/provenance.py` helper exists and is wired so new result JSONs carry git commit + full config + lib versions (additive; legacy schema unchanged).
- [x] `python verify.py` passes (imports + provenance smoke).
- [x] git initialized; committed `M0: scaffolding + plan`.

### M1 — Config/CLI layer + reproducibility spine
**Scope:** YAML config layer; `--config PATH`; `--device {auto,cpu,cuda}` (default auto); wire
`latent_dim`, `n_steps`, `batch_size`, `lr`, `p_values`, `seeds`, `model_types` through config.
Grow `verify.py` to run the smoke config end-to-end.
**Done when:**
- `configs/smoke.yaml` (CPU, finishes in seconds) and `configs/full.yaml` (GPU) exist.
- `python run_experiment.py --config configs/smoke.yaml` writes provenance-stamped JSONs on CPU in seconds.
- Legacy flags (`--quick`, `--p-values`, `--seeds`, `--n-steps`, `--only-figures`, `--only-decision`) still work; the prior 36-run full sweep reproduces with **identical scientific schema**.
- `verify.py` asserts invariants **1** (encoder `state_dict` keys+shapes identical across the 3 models), **4** (eval seed disjoint; probe reads `online_encoder`), **5** (`latent_dim` honored from config; skip-if-done on JSON).

### M2 — Invariant-compliant JEPA objective
**Scope:** bring JEPA loss to spec — **BYOL-style normalized (cosine) prediction loss + VICReg
variance AND covariance**. Make prediction-loss `{mse, cosine}` and anti-collapse `{var, var+cov}`
**config-selectable**, legacy (`mse`+`var`) reproducible; new default = `cosine`+`var+cov`.
**Done when:**
- Default JEPA/JEPA-InvDyn runs use cosine prediction + variance + covariance.
- `verify.py` smoke asserts invariant **2** (JEPA latent effective rank > 5) and invariant **3** (cosine loss + EMA path exercised).
- Legacy `mse`+`var` mode reproduces pre-M2 numbers; both paths covered by a test.

### M3 — Complete the 2×2 environment
**Scope:** add `GridWorldHiddenRuleEnv` (and/or extend `SwitchColorEnv`) so all four cells are
instantiable; controllability set by whether the action moves a feature; predictability knob
applies to relevant features; expose per-feature ground-truth labels for per-cell probing.
**Done when:**
- The env emits labels for all four cells (controllable±, relevant±).
- Smoke probes run per cell; a figure visualizes the 2×2 and the predictability knob.
- Existing `SwitchColorEnv` behavior and seeds unchanged.

### M4 — Objective zoo (shared encoder)
**Scope:** add the objectives the spine needs to discriminate cells — bisimulation (DBC-style),
reward-grounded JEPA (reward-prediction head), a controllability objective — alongside existing
pure-prediction (JEPA), reconstruction (AE), inverse-dynamics (JEPA-InvDyn). All reuse the
byte-identical `Encoder`.
**Done when:**
- Each objective trains on the smoke config and writes results.
- `verify.py` asserts invariant **1** across the *entire* zoo (all `state_dict` keys+shapes identical).
- Result schema extended additively; `build_model` covers every objective.

### M5 — Full evaluation matrix + decision
**Scope:** evaluate every objective × cell × predictability — per-cell linear/nonlinear probe,
per-cell control regret, centroid separation, effective rank. Generalize `make_decision` to the
2×2 predictions (cell-4 hard case decisive). Paper-grade figures.
**Done when:**
- The full matrix runs from one config; `decision.json` reports per-cell verdicts.
- Figures regenerate; a smoke variant runs in seconds.

### M6 — Pretrained-encoder comparison (GUARDED)
**Scope:** optional comparison against a large pretrained encoder. Weight download is behind an
explicit `--download-weights` flag and **never** triggered by build/test/smoke.
**Done when:**
- Code is unit-smokeable offline (tiny/stub encoder); real weights download only under the flag.
- A grep/inspection confirms no network call on the default/smoke path.

### M7 — Full reproduction + report
**Scope:** one-command full reproduction (GPU) and one-command smoke (CPU); final report tying
results to the 2×2 claim; `README.md`; all invariants asserted in `verify.py`.
**Done when:**
- `python verify.py` runs the smoke pipeline green asserting **all 5 invariants**.
- The full pipeline reproduces results + figures + report end-to-end from clean checkouts of code (datasets/checkpoints regenerate).

---

## Risks / things to watch
- **Schema drift**: any new metric is additive; never rename/remove the 11 scientific keys.
- **M2 vs reproducibility**: cosine+cov changes JEPA numbers — keep legacy mode selectable so the
  36-run sweep is still reproducible and comparable.
- **Encoder identity (invariant 1)** must survive every new objective; assert it in `verify.py` each milestone.
- **No-download rule** is easy to violate accidentally via a model hub import; gate it in M6.
