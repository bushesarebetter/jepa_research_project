"""All figure generation for the JEPA control-loss experiment."""
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style='ticks', context='paper')
except ImportError:
    plt.style.use('seaborn-v0_8-paper')

MODEL_COLORS = {'jepa': '#d62728', 'ae': '#1f77b4', 'jepa_invdyn': '#2ca02c'}
MODEL_LABELS = {'jepa': 'JEPA', 'ae': 'Autoencoder', 'jepa_invdyn': 'JEPA + InvDyn'}


def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)


def _agg(results, model_type, p, key):
    vals = [r[key] for r in results if r['model_type'] == model_type and r['p_repeat'] == p]
    return np.mean(vals), np.std(vals)


def figure1_main_result(results, p_values, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'linear_probe_acc')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='chance')
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Linear probe accuracy')
    ax.set_title('Switch color decodability vs predictability')
    ax.legend(fontsize=8)
    _clean(ax)

    ax = axes[1]
    for mt, key, label in [('jepa', 'linear_probe_acc', 'JEPA linear'),
                            ('jepa', 'nonlinear_probe_acc', 'JEPA nonlinear'),
                            ('ae', 'linear_probe_acc', 'AE linear (ref)')]:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, key)
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', label=label)
        ax.fill_between(p_values, means - stds, means + stds, alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Probe accuracy')
    ax.set_title('Linear vs nonlinear accessibility (JEPA)')
    ax.legend(fontsize=8)
    _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'figure1_main_result')


def figure2_bisimulation(results, p_values, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'centroid_dist_norm')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Normalized centroid distance')
    ax.set_title('Bisimulation proxy: c=0 vs c=1 separation')
    ax.legend(fontsize=8)
    _clean(ax)

    ax = axes[1]
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'control_regret')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='max regret (random)')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=1, label='zero regret')
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Control regret')
    ax.set_title('Control regret vs predictability')
    ax.legend(fontsize=8)
    _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'figure2_bisimulation')


def figure3_theory_vs_empirical(results, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5))
    for mt in ['jepa', 'ae', 'jepa_invdyn']:
        xs = [r['linear_probe_acc'] for r in results if r['model_type'] == mt]
        ys = [r['control_regret'] for r in results if r['model_type'] == mt]
        ax.scatter(xs, ys, color=MODEL_COLORS[mt], label=MODEL_LABELS[mt], alpha=0.8)

    grid = np.linspace(0.5, 1.0, 50)
    ax.plot(grid, 1 - grid, color='gray', linestyle='--', label='theory: regret = 1 - acc')
    ax.set_xlabel('Linear probe accuracy')
    ax.set_ylabel('Control regret')
    ax.set_title('Theory vs empirical alignment')
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    _save(fig, out_dir, 'figure3_theory_vs_empirical')


def figure4_ablations(results, p_values, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    model_types = ['jepa', 'ae', 'jepa_invdyn']
    means = []
    stds = []
    for mt in model_types:
        m, s = _agg(results, mt, 0.5, 'linear_probe_acc')
        means.append(m); stds.append(s)
    ax.bar(model_types, means, yerr=stds, color=[MODEL_COLORS[mt] for mt in model_types])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(model_types)))
    ax.set_xticklabels([MODEL_LABELS[mt] for mt in model_types])    
    ax.set_ylabel('Linear probe accuracy at p=0.5')
    ax.set_title('Ablation: model type at hardest setting')
    _clean(ax)

    ax = axes[1]
    for mt in ['jepa', 'ae']:
        means, stds = [], []
        for p in p_values:
            m, s = _agg(results, mt, p, 'effective_rank')
            means.append(m); stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(p_values, means, marker='o', color=MODEL_COLORS[mt], label=MODEL_LABELS[mt])
        ax.fill_between(p_values, means - stds, means + stds, color=MODEL_COLORS[mt], alpha=0.2)
    ax.set_xlabel('p_repeat')
    ax.set_ylabel('Effective rank')
    ax.set_title('Representation rank (capacity check)')
    ax.legend(fontsize=8)
    _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, 'figure4_ablations')


def figure5_environment(out_dir, img_size=32):
    from .env import SwitchColorEnv

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for row, p in enumerate([0.5, 1.0]):
        env = SwitchColorEnv(p_repeat=p, img_size=img_size, seed=42)
        obs = env.reset()
        for col in range(5):
            ax = axes[row, col]
            ax.imshow(np.transpose(obs, (1, 2, 0)))
            ax.set_title(f"t={env.t}  c={env.switch_color}", fontsize=9)
            ax.axis('off')
            obs, _, _ = env.step(env.switch_color)
    fig.suptitle('Environment frames: p=0.5 (top, i.i.d.) vs p=1.0 (bottom, constant)')
    fig.tight_layout()
    _save(fig, out_dir, 'figure5_environment')


def generate_all_figures(results, p_values, out_dir):
    figure1_main_result(results, p_values, out_dir)
    figure2_bisimulation(results, p_values, out_dir)
    figure3_theory_vs_empirical(results, out_dir)
    figure4_ablations(results, p_values, out_dir)
    figure5_environment(out_dir)
    print(f"Figures saved to {out_dir}")


# ── M3 / M4 quadrant figures ──────────────────────────────────────────────────

CELL_LABELS = {1: 'ctrl\n+rel', 2: 'ctrl\n+irr', 3: 'exo\n+irr', 4: 'exo\n+rel'}

OBJ_COLORS = {
    'recon':       '#1f77b4',
    'ae':          '#1f77b4',
    'jepa':        '#d62728',
    'jepa_ac':     '#ff7f0e',
    'jepa_ctrl':   '#9467bd',
    'jepa_invdyn': '#2ca02c',
    'jepa_reward': '#17becf',
    'oracle':      '#7f7f7f',
}


def quadrant_heatmap(matrix, objectives, cells, out_dir, *, threshold=0.75, name='figure6_quadrant_matrix'):
    """Heatmap of linear-probe accuracy for each (objective, cell) pair.

    matrix : 2-D numpy array, shape (len(objectives), len(cells))
    Green  = retained (>= threshold), Red = dropped (< threshold).
    Annotated with the raw accuracy value.
    """
    _disp = {'oracle': 'Supervised'}
    ylabels = [_disp.get(o, o) for o in objectives]
    n_obj, n_cell = len(objectives), len(cells)
    fig, ax = plt.subplots(figsize=(2.2 * n_cell + 1.2, 0.7 * n_obj + 1.8))

    try:
        import seaborn as sns
        cmap = 'RdYlGn'
        sns.heatmap(
            matrix, ax=ax,
            vmin=0.5, vmax=1.0, center=threshold,
            cmap=cmap, annot=True, fmt='.2f', linewidths=0.5,
            xticklabels=[CELL_LABELS.get(c, str(c)) for c in cells],
            yticklabels=ylabels,
        )
    except ImportError:
        im = ax.imshow(matrix, vmin=0.5, vmax=1.0, cmap='RdYlGn', aspect='auto')
        for i in range(n_obj):
            for j in range(n_cell):
                v = matrix[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if v < 0.62 or v > 0.88 else 'black')
        ax.set_xticks(range(n_cell))
        ax.set_xticklabels([CELL_LABELS.get(c, str(c)) for c in cells])
        ax.set_yticks(range(n_obj))
        ax.set_yticklabels(ylabels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel('Feature cell  (ctrl=controllable, exo=exogenous, rel=relevant, irr=irrelevant)')
    ax.set_ylabel('Objective')
    ax.set_title(f'Retained feature info by objective x cell\n'
                 f'(linear probe acc, chance=0.50, retain threshold={threshold})')
    fig.tight_layout()
    _save(fig, out_dir, name)


def figure7_min_reward_signal(fracs, means, out_dir, *,
                              threshold=0.75, min_frac=None,
                              env_name='quadrant',
                              name='figure7_min_reward_signal'):
    """Cell-4 retention vs reward_label_fraction for jepa_reward.

    fracs    : list of fractions swept (x-axis, log scale)
    means    : list of mean cell-4 linear-probe acc (y-axis, aligned with fracs)
    min_frac : the smallest fraction crossing threshold (annotated if not None)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    color = OBJ_COLORS.get('jepa_reward', '#17becf')
    ax.plot(fracs, means, marker='o', color=color, label='jepa_reward (cell 4)')
    ax.fill_between(fracs,
                    [max(0.5, m - 0.05) for m in means],
                    [min(1.0, m + 0.05) for m in means],
                    color=color, alpha=0.15)
    ax.axhline(threshold, color='gray', linestyle='--', linewidth=1,
               label=f'retain threshold ({threshold})')
    ax.axhline(0.5, color='lightgray', linestyle=':', linewidth=1, label='chance')
    if min_frac is not None:
        ax.axvline(min_frac, color='green', linestyle=':', linewidth=1.5,
                   label=f'min passing fraction = {min_frac:g}')
        matching = [m for f, m in zip(fracs, means) if f == min_frac]
        if matching:
            ax.annotate(f'  {min_frac:g}\n  ({matching[0]:.2f})',
                        xy=(min_frac, matching[0]), fontsize=8, color='green')
    ax.set_xscale('log')
    ax.set_xlabel('reward_label_fraction  (log scale)')
    ax.set_ylabel('Cell-4 linear probe accuracy')
    ax.set_title(f'Minimum reward signal to recover exo+rel feature\n(env={env_name})')
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=8)
    _clean(ax)
    fig.tight_layout()
    _save(fig, out_dir, name)


def figure9_cell4_failure_both_envs(data_q, data_g, out_dir, *, threshold=0.75, name='figure9_cell4_failure_both_envs'):
    """Main paper figure: cell-4 failure across both envs.

    Two panels:
      Left  — probe accuracy per objective (grouped bars: quadrant vs gridworld)
              with threshold=0.75 and chance=0.5 reference lines.
      Right — InfoNCE MI (nats) per objective, with log2≈0.693 reference line.
    Error bars = std over 3 seeds. Bars annotated with mean.
    """
    from collections import defaultdict

    obj_order = ['recon', 'jepa', 'jepa_ac', 'jepa_ctrl', 'jepa_invdyn', 'jepa_reward', 'oracle']
    obj_labels = {
        'recon': 'Recon', 'jepa': 'JEPA', 'jepa_ac': 'JEPA+AC',
        'jepa_ctrl': 'JEPA+Ctrl', 'jepa_invdyn': 'JEPA+InvDyn',
        'jepa_reward': 'JEPA+Reward', 'oracle': 'Supervised',
    }

    def _agg_runs(data):
        acc, mi = defaultdict(list), defaultdict(list)
        for r in data['results']:
            acc[r['objective']].append(r['linear_probe_acc'])
            mi[r['objective']].append(r.get('mi_infonce', 0.0))
        return acc, mi

    acc_q, mi_q = _agg_runs(data_q)
    acc_g, mi_g = _agg_runs(data_g)

    objs = [o for o in obj_order if o in acc_q]
    x = np.arange(len(objs))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.subplots_adjust(wspace=0.32)

    env_color = {'quadrant': '#2166ac', 'gridworld': '#d6604d'}
    log2 = float(np.log(2))

    for ax_idx, (metric_q, metric_g, ylabel, yref_val, yref_label, ylim) in enumerate([
        (acc_q, acc_g, 'Linear probe accuracy', threshold, f'retain threshold ({threshold})', (0.42, 1.18)),
        (mi_q, mi_g,   'InfoNCE MI (nats)',     log2,      f'log 2 ≈ {log2:.3f} nats',       (-0.02, 0.92)),
    ]):
        ax = axes[ax_idx]
        for i, (vals_q, vals_g) in enumerate(
            [(metric_q[o], metric_g[o]) for o in objs]
        ):
            mq, sq = float(np.mean(vals_q)), float(np.std(vals_q))
            mg, sg = float(np.mean(vals_g)), float(np.std(vals_g))
            bq = ax.bar(x[i] - w/2, mq, w, yerr=sq, color=env_color['quadrant'],
                        capsize=3, label='Quadrant' if i == 0 else '', zorder=3)
            bg = ax.bar(x[i] + w/2, mg, w, yerr=sg, color=env_color['gridworld'],
                        capsize=3, label='GridWorld' if i == 0 else '', zorder=3)
            # annotate above the whisker top (h + std) to avoid overlap
            for bar, val, std in [(bq[0], mq, sq), (bg[0], mg, sg)]:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + std + 0.025,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

        ax.axhline(yref_val, color='#444444', linestyle='--', linewidth=1.2,
                   label=yref_label, zorder=2)
        if ax_idx == 0:
            ax.axhline(0.5, color='#aaaaaa', linestyle=':', linewidth=1, label='chance (0.5)', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([obj_labels[o] for o in objs], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc='upper right' if ax_idx == 0 else 'lower right')
        ax.set_title(
            'Cell-4 probe accuracy (lower = dropped)' if ax_idx == 0
            else 'Cell-4 mutual information (nats)',
            fontsize=10
        )
        _clean(ax)

    fig.tight_layout()
    _save(fig, out_dir, name)


def figure8_capacity(series, dims, out_dir, *, threshold=0.75, name='figure8_capacity'):
    """Cell-4 retention vs latent_dim for jepa and jepa_reward.

    series : dict  {label -> list of mean acc aligned with dims}
             e.g. {'quadrant/jepa': [...], 'quadrant/jepa_reward': [...]}
    dims   : list of latent_dim values (x-axis, log2 scale)
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    linestyles = {'quadrant': '-', 'switch_color': '--', 'gridworld': '-.'}
    for label, accs in series.items():
        parts = label.split('/')
        env = parts[0] if len(parts) > 1 else 'unknown'
        obj = parts[1] if len(parts) > 1 else parts[0]
        color = OBJ_COLORS.get(obj, '#333333')
        ls = linestyles.get(env, ':')
        ax.plot(dims, accs, marker='o', color=color, linestyle=ls, label=label)
    ax.axhline(threshold, color='gray', linestyle='--', linewidth=1,
               label=f'retain threshold ({threshold})')
    ax.axhline(0.5, color='lightgray', linestyle=':', linewidth=1, label='chance')
    ax.set_xscale('log', base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel('latent_dim  (log2 scale)')
    ax.set_ylabel('Cell-4 linear probe accuracy')
    ax.set_title('Capacity is not the bottleneck: cell-4 drop persists as latent grows\n'
                 '(jepa flat-low, jepa_reward flat-high)')
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=8, ncol=2)
    _clean(ax)
    fig.tight_layout()
    _save(fig, out_dir, name)