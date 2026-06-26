"""InfoNCE lower-bound estimator of mutual information I(Z; c), in nats.

Trains a small separable critic g(z)·h(c) to maximize the InfoNCE / CPC bound

    I(Z; c) >= log K - CE(scores, diagonal)              (K = batch size)

and returns the bound in nats. Because the separable critic's score depends on c
only through its value, the bound saturates at ~log 2 (one bit) when a binary c is
fully recoverable from z, and at ~0 when c is independent noise (see verify.py).

This exists to de-circularize the regret-vs-information figure: x = estimated I(Z;c)
from this module, NOT 1 - probe_accuracy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in, d_out, hidden):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


def estimate_mi_infonce(Z, c, *, epochs=300, device='cpu', hidden=64, proj=32,
                        lr=1e-3, seed=0):
    """InfoNCE lower bound on I(Z; c) in nats.

    Z: (N, d) latents. c: (N,) or (N, k) feature (discrete or continuous).
    Returns a single float (nats). It is a *lower* bound, so it under-estimates.

    The critic trains on one half and the bound is read off the held-out half: an MLP
    critic memorizes pairings on its training set even when c is independent of Z, which
    would inflate the estimate -- the split keeps the bound honest (independent c -> ~0).
    The reported value is the best held-out bound over training (eval CE rises again once
    the critic starts overfitting the train half).
    """
    torch.manual_seed(seed)
    Z = np.asarray(Z, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    if c.ndim == 1:
        c = c[:, None]
    # standardize so the critic sees comparable scales on both sides
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    c = (c - c.mean(0)) / (c.std(0) + 1e-6)
    Zt = torch.from_numpy(Z).to(device)
    ct = torch.from_numpy(c).to(device)

    perm = torch.randperm(Zt.shape[0])
    n_tr = Zt.shape[0] // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    Z_tr, c_tr, Z_ev, c_ev = Zt[tr], ct[tr], Zt[ev], ct[ev]

    g = _mlp(Zt.shape[1], proj, hidden).to(device)
    h = _mlp(ct.shape[1], proj, hidden).to(device)
    opt = torch.optim.Adam(list(g.parameters()) + list(h.parameters()), lr=lr)

    y_tr = torch.arange(Z_tr.shape[0], device=device)
    y_ev = torch.arange(Z_ev.shape[0], device=device)
    logK_ev = float(np.log(Z_ev.shape[0]))
    best = 0.0
    for ep in range(epochs):
        scores = g(Z_tr) @ h(c_tr).t()      # (n_tr, n_tr): scores[i, j] = critic(z_i, c_j)
        loss = F.cross_entropy(scores, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                ev_loss = F.cross_entropy(g(Z_ev) @ h(c_ev).t(), y_ev).item()
            best = max(best, logK_ev - ev_loss)
    return max(best, 0.0)


if __name__ == '__main__':
    # demo / self-check: recoverable binary c -> ~log 2; independent noise -> ~0
    rng = np.random.default_rng(0)
    N, d = 2000, 16
    c = rng.integers(0, 2, size=N).astype(np.float32)
    Z = rng.normal(size=(N, d)).astype(np.float32)
    Z[:, 0] = 3.0 * c + 0.3 * rng.normal(size=N)
    mi_rec = estimate_mi_infonce(Z, c, epochs=400)
    mi_indep = estimate_mi_infonce(rng.normal(size=(N, d)).astype(np.float32),
                                   rng.integers(0, 2, size=N).astype(np.float32), epochs=400)
    print(f"recoverable={mi_rec:.3f} (~{np.log(2):.3f})  independent={mi_indep:.3f} (~0)")
