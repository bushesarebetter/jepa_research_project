"""Models: a shared Encoder used identically by JEPA and AE, plus JEPA/AE/JEPA+InvDyn wrappers.

The encoder architecture MUST be identical across model types so that any
performance gap in downstream probes is attributable to the training
objective, not architectural capacity differences.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 256


def _conv_out_size(img_size):
    """Spatial size after the three stride-2 convs (the 4th is stride 1).

    img_size // 8 only happens to be right for sizes divisible by 8 (32 -> 4); 12 -> 2,
    not 1. This exact formula keeps 32 identical while supporting the 12x12 quadrant frame.
    """
    s = img_size
    for _ in range(3):
        s = (s - 1) // 2 + 1
    return s


class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        spatial = _conv_out_size(img_size)  # 32 -> 4, 12 -> 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial * spatial, 512), nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        return self.norm(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.spatial = _conv_out_size(img_size)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, 256 * self.spatial * self.spatial), nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, self.spatial, self.spatial)
        out = self.deconv(h)  # upsamples spatial*8; for 32 that is exactly img_size (no-op below)
        if out.shape[-1] != self.img_size:
            out = F.interpolate(out, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=32, n_actions=2):
        super().__init__()
        self.online_encoder = Encoder(latent_dim, img_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, latent_dim),
        )

    def get_latent(self, x):
        return self.online_encoder(x)

    def forward(self, obs_t, obs_t1):
        z_ctx = self.online_encoder(obs_t)
        z_pred = self.predictor(z_ctx)
        with torch.no_grad():
            z_tgt = self.target_encoder(obs_t1)
        pred_loss = F.mse_loss(z_pred, z_tgt)
        var_per_dim = z_pred.var(dim=0)
        var_loss = F.relu(1.0 - (var_per_dim + 1e-4).sqrt()).mean()
        total_loss = pred_loss + 0.1 * var_loss
        return total_loss, {'pred_loss': pred_loss.item(), 'var_loss': var_loss.item()}, z_ctx

    @torch.no_grad()
    def ema_update(self, decay=0.996):
        for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1 - decay)


class AutoencoderModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_size=32, n_actions=2):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)

    def get_latent(self, x):
        return self.encoder(x)

    def forward(self, obs_t):
        z = self.encoder(obs_t)
        recon = self.decoder(z)
        loss = F.mse_loss(recon, obs_t)
        return loss, {'recon_loss': loss.item()}, z


class JEPAInvDynModel(JEPAModel):
    """Inverse-dynamics / AC-State head: predict the TAKEN action from (z_t, z_{t+1}).

    Used for both `jepa_ctrl` (controllability: keeps action-moved features, drops exogenous)
    and `jepa_invdyn` (same mechanism; the rescue of an exogenous feature tracks how
    informative the taken actions are about it -- a property of the DATA, not this class).
    """
    def __init__(self, latent_dim=LATENT_DIM, img_size=32, n_actions=2):
        super().__init__(latent_dim, img_size, n_actions)
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.GELU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs_t, obs_t1, action_t=None):
        z_ctx = self.online_encoder(obs_t)
        z_pred = self.predictor(z_ctx)
        with torch.no_grad():
            z_tgt = self.target_encoder(obs_t1)
        pred_loss = F.mse_loss(z_pred, z_tgt)
        var_per_dim = z_pred.var(dim=0)
        var_loss = F.relu(1.0 - (var_per_dim + 1e-4).sqrt()).mean()

        z_t1_online = self.online_encoder(obs_t1)
        action_logits = self.action_head(torch.cat([z_ctx, z_t1_online], dim=-1))
        action_loss = F.cross_entropy(action_logits, action_t)

        total_loss = pred_loss + 0.1 * var_loss + 0.1 * action_loss
        info = {
            'pred_loss': pred_loss.item(),
            'var_loss': var_loss.item(),
            'action_loss': action_loss.item(),
        }
        return total_loss, info, z_ctx


class JEPARewardModel(JEPAModel):
    """JEPA + a reward head r_hat(z_t, a) -> the proposed FIX.

    Grounding the latent in reward forces it to keep RELEVANT features (reward depends on
    them) regardless of predictability/controllability. Trained on reward-bearing data with
    action variance (a random policy), so reward is not degenerate. A DBC-style bisimulation
    term is deferred. `reward_mask` (M4) implements reward_label_fraction: a 0/1 weight per
    transition so only a fraction of transitions carry the reward signal.
    """
    def __init__(self, latent_dim=LATENT_DIM, img_size=32, n_actions=2):
        super().__init__(latent_dim, img_size, n_actions)
        self.n_actions = n_actions
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + n_actions, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_t, obs_t1, action_t=None, reward_t=None, reward_mask=None):
        z_ctx = self.online_encoder(obs_t)
        z_pred = self.predictor(z_ctx)
        with torch.no_grad():
            z_tgt = self.target_encoder(obs_t1)
        pred_loss = F.mse_loss(z_pred, z_tgt)
        var_per_dim = z_pred.var(dim=0)
        var_loss = F.relu(1.0 - (var_per_dim + 1e-4).sqrt()).mean()

        a_oh = F.one_hot(action_t, self.n_actions).float()
        r_hat = self.reward_head(torch.cat([z_ctx, a_oh], dim=-1)).squeeze(-1)
        # reward_label_fraction (M4): only transitions with reward_mask==1 contribute the reward
        # signal; the rest are unlabelled. mask all-ones (the default) == plain BCE mean.
        bce = F.binary_cross_entropy_with_logits(r_hat, reward_t, reduction='none')
        if reward_mask is None:
            reward_loss = bce.mean()
        else:
            reward_loss = (bce * reward_mask).sum() / reward_mask.sum().clamp(min=1.0)

        total_loss = pred_loss + 0.1 * var_loss + reward_loss
        info = {'pred_loss': pred_loss.item(), 'var_loss': var_loss.item(),
                'reward_loss': reward_loss.item()}
        return total_loss, info, z_ctx


# Objective registry: encoder stays IDENTICAL across all of these (invariant 1). `signal`
# names what each objective consumes; `inputs` are the forward() kwargs the trainer feeds.
# 'oracle' has cls=None: it is not a pixel encoder, the orchestrator uses ground-truth bits
# as the representation (upper bound). 'ae' is kept as an alias of 'recon' for the legacy sweep.
OBJECTIVES = {
    'recon':       dict(cls=AutoencoderModel, signal='pixels', inputs=['obs_t']),
    'ae':          dict(cls=AutoencoderModel, signal='pixels', inputs=['obs_t']),
    'jepa':        dict(cls=JEPAModel,        signal='none',   inputs=['obs_t', 'obs_t1']),
    'jepa_ctrl':   dict(cls=JEPAInvDynModel,  signal='action', inputs=['obs_t', 'obs_t1', 'action_t']),
    'jepa_invdyn': dict(cls=JEPAInvDynModel,  signal='action', inputs=['obs_t', 'obs_t1', 'action_t']),
    'jepa_reward': dict(cls=JEPARewardModel,  signal='reward', inputs=['obs_t', 'obs_t1', 'action_t', 'reward_t', 'reward_mask']),
    'oracle':      dict(cls=None,             signal='state',  inputs=[]),
}


def build_model(model_type, latent_dim=LATENT_DIM, img_size=32, n_actions=2):
    spec = OBJECTIVES.get(model_type)
    if spec is None:
        raise ValueError(f"Unknown model_type: {model_type}")
    if spec['cls'] is None:
        raise ValueError(f"{model_type!r} has no encoder (handled directly by the orchestrator)")
    return spec['cls'](latent_dim, img_size, n_actions)


def get_encoder(model):
    """The shared Encoder submodule, whatever the objective wraps it in."""
    return getattr(model, 'online_encoder', None) or getattr(model, 'encoder')


def encoder_signature(model):
    """(key, shape) of every encoder param -- equal across objectives iff invariant 1 holds."""
    return tuple(sorted((k, tuple(v.shape)) for k, v in get_encoder(model).state_dict().items()))


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
