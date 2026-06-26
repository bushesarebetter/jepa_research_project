"""Training loops shared by JEPA, AE, and JEPA+InvDyn."""
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .models import build_model, OBJECTIVES


class PairDataset(Dataset):
    """obs_t, obs_t1, action_t -- used by jepa and jepa_invdyn."""
    def __init__(self, data):
        self.obs = data['obs']
        self.next_obs = data['next_obs']
        self.action = data['action']

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.tensor(self.action[idx], dtype=torch.long),
        )


class ObsDataset(Dataset):
    """obs_t only -- used by ae."""
    def __init__(self, data):
        self.obs = data['obs']

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.obs[idx])


def make_dataloader(data, model_type, batch_size):
    if model_type in ('jepa', 'jepa_invdyn'):
        ds = PairDataset(data)
    elif model_type == 'ae':
        ds = ObsDataset(data)
    else:
        raise ValueError(model_type)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def train_model(model_type, data_train, n_steps=30_000, lr=1e-3, batch_size=256,
                 device='cpu', grad_clip=1.0, weight_decay=1e-5, ema_decay=0.996,
                 log_every=500, verbose=True):
    model = build_model(model_type).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

    loader = make_dataloader(data_train, model_type, batch_size)
    steps_per_epoch = len(loader)
    n_epochs = math.ceil(n_steps / steps_per_epoch)

    model.train()
    step = 0
    loss_history = []
    for epoch in range(n_epochs):
        for batch in loader:
            if step >= n_steps:
                break
            if model_type in ('jepa', 'jepa_invdyn'):
                obs_t, obs_t1, action_t = batch
                obs_t, obs_t1, action_t = obs_t.to(device), obs_t1.to(device), action_t.to(device)
                if model_type == 'jepa_invdyn':
                    loss, info, _ = model(obs_t, obs_t1, action_t)
                else:
                    loss, info, _ = model(obs_t, obs_t1)
            else:
                obs_t = batch.to(device)
                loss, info, _ = model(obs_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            if model_type in ('jepa', 'jepa_invdyn'):
                model.ema_update(ema_decay)

            loss_history.append(loss.item())
            if verbose and step % log_every == 0:
                print(f"  [{model_type}] step {step}/{n_steps} loss={loss.item():.4f} {info}")
            step += 1
        if step >= n_steps:
            break

    model.eval()
    return model, loss_history


class TransitionDataset(Dataset):
    """Carries every signal (obs_t, obs_t1, action, reward); the trainer feeds each objective
    only the subset its forward() declares. Used by the M3 objective-registry trainer."""
    def __init__(self, data):
        self.obs, self.next_obs = data['obs'], data['next_obs']
        self.action, self.reward = data['action'], data['reward']
        # reward_mask defaults to all-ones (every transition labelled); M4's reward_label_fraction
        # sweep injects a sparser mask. All-ones is a no-op vs the pre-M4 plain-BCE reward loss.
        self.reward_mask = data.get('reward_mask', np.ones(len(self.obs), dtype=np.float32))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return {
            'obs_t': torch.from_numpy(self.obs[i]),
            'obs_t1': torch.from_numpy(self.next_obs[i]),
            'action_t': torch.tensor(self.action[i], dtype=torch.long),
            'reward_t': torch.tensor(self.reward[i], dtype=torch.float32),
            'reward_mask': torch.tensor(self.reward_mask[i], dtype=torch.float32),
        }


def train_objective(model_type, data, *, n_steps, img_size, latent_dim, n_actions=2,
                    device='cpu', lr=1e-3, batch_size=128, ema_decay=0.996, grad_clip=1.0,
                    weight_decay=1e-5, verbose=False):
    """Registry-driven trainer used by the quadrant experiment. Generic over objectives:
    builds the model, then feeds only the forward() kwargs that objective declares in
    OBJECTIVES[...]['inputs']. (run_experiment.py keeps using train_model unchanged.)"""
    model = build_model(model_type, latent_dim=latent_dim, img_size=img_size,
                        n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

    loader = DataLoader(TransitionDataset(data), batch_size=batch_size, shuffle=True, drop_last=True)
    needed = OBJECTIVES[model_type]['inputs']
    is_jepa = hasattr(model, 'ema_update')

    model.train()
    step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            inputs = {k: batch[k].to(device) for k in needed}
            loss, info, _ = model(**inputs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            if is_jepa:
                model.ema_update(ema_decay)
            if verbose and step % 50 == 0:
                print(f"  [{model_type}] step {step}/{n_steps} loss={loss.item():.4f} {info}")
            step += 1
    model.eval()
    return model


@torch.no_grad()
def get_latents(model, obs, device='cpu', batch_size=512):
    model.eval()
    latents = []
    for i in range(0, len(obs), batch_size):
        batch = torch.from_numpy(obs[i:i + batch_size]).to(device)
        z = model.get_latent(batch)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)
