"""
LeWorldModel (LeWM) agent for JAXtari environments.

Based on: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
          Maes, Le Lidec, Scieur, LeCun, Balestriero (2026), arXiv:2603.19312
Code reference: https://github.com/lucas-maes/le-wm

Adapted for JAXtari by: Amirmohammad Raei (TU Darmstadt Praktikum, Topic 30)
Changes from original:
  - CNN encoder instead of ViT (faster for Atari pixel observations)
  - Discrete action embedding instead of continuous action encoder
  - Online data collection from JAXtari instead of offline HDF5 datasets
  - Uniform-random policy for data collection (no policy learning here, this
    script trains the world model only)

Usage
-----
    # one game
    python lewm_jaxatari.py --game pong --total_steps 10000 --outdir results/pong_run

    # all 15 required games
    python run_all_games.py --outdir results/full --total_steps 5000

Each run writes <outdir>/<game>/{curve.png, history.json, model.pt}. The learning
curve is refreshed every --plot_every steps, so an interrupted run still leaves a
usable plot behind.
"""

import argparse
import json
import os
import random
from pathlib import Path

# JAX (the environment) and PyTorch (the networks) share one GPU here. By default
# JAX preallocates ~75% of VRAM on first use, which leaves PyTorch unable to
# allocate and fails with a confusing OOM. Must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import jaxatari
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper

# Defaults for every knob `train()` understands. Both entry points (the CLI and
# run_all_games.py) build an argparse.Namespace, so anything missing is filled in
# from here, a caller that omits a key gets the documented default rather than
# an AttributeError.
DEFAULTS = dict(
    game="pong",
    emb_dim=256,
    seq_len=16,
    batch_size=32,
    lr=3e-4,
    total_steps=10_000,
    init_sequences=500,
    collect_every=100,
    collect_n=10,
    buffer_size=10_000,
    sigreg_weight=0.1,
    img_size=84,
    seed=42,
    log_every=100,
    eval_seq=32,
    eval_context=3,
    device="auto",
    outdir="results",
    plot=True,
    plot_every=500,
    stop_grad=False,
    ckpt_every=0,
    resume=None,
    max_windows_per_episode=4,
    max_episode_steps=500,
    action_cond="adaln",
)


# ---------------------------------------------------------------------------
# RTPT, required on the TU Darmstadt student pool
# ---------------------------------------------------------------------------

def make_rtpt(experiment: str, max_iterations: int, initials: str = "AR"):
    """Process-title reporter for the shared lab machines.

    The student pool asks every job to run under RTPT so other users can see
    whose experiment is on a GPU and how long it has left. Optional: if the
    package is not installed (e.g. running locally) this returns a no-op, so the
    same script works on a laptop and on the pool without branching.
    """
    try:
        from rtpt import RTPT
    except ImportError:
        class _NoRTPT:
            def start(self): pass
            def step(self, subtitle=None): pass
        return _NoRTPT()

    r = RTPT(name_initials=initials, experiment_name=experiment,
             max_iterations=max(1, max_iterations))
    r.start()
    return r


def fill_defaults(args):
    """Fill any option `train()` reads but the caller did not set."""
    for k, v in DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args

# ---------------------------------------------------------------------------
# Encoder: CNN that maps a single pixel frame -> embedding vector
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),  # (B, 32, 20, 20) for 84x84
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),            # (B, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),            # (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),                               # (B, 64*7*7 = 3136)
        )
        self.proj = nn.Sequential(
            nn.Linear(3136, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        """x: (B, C, H, W) pixel frame in [0, 1]"""
        return self.proj(self.net(x))


# ---------------------------------------------------------------------------
# SIGReg: Sketched Isotropic Gaussian Regularizer (copied from le-wm/module.py)
# Prevents representation collapse by enforcing Gaussian-distributed embeddings.
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Predictor: Transformer that predicts next embedding given past embeddings + actions
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """AdaLN modulation: scale and shift a normalised activation."""
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    """Causal transformer block with adaptive-LayerNorm action conditioning.

    This is how the paper injects actions (Sec. 3): instead of adding an action
    embedding to the token, the action produces per-token shift/scale/gate
    parameters for both sub-layers. The modulation head is **zero-initialised**,
    so at step 0 the gates are 0 and the block is exactly the identity, the
    predictor starts as a no-op and learns to use actions rather than having
    randomly-scaled action noise injected into the residual stream from the start.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)   # zero-init: block starts as identity
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, cond, attn_mask):
        """x, cond: (B, T, D); attn_mask: (T, T) causal float mask."""
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.modulation(cond).chunk(6, dim=-1)

        h = modulate(self.norm1(x), sh_a, sc_a)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False,
                         is_causal=True)
        x = x + g_a * a

        h = modulate(self.norm2(x), sh_m, sc_m)
        return x + g_m * self.mlp(h)


class Predictor(nn.Module):
    """Causal transformer over frame embeddings, conditioned on actions.

    `action_cond` selects how actions enter:
      "adaln": per-token AdaLN modulation, zero-init (what the paper does)
      "add":   action embedding added to the token (simpler; kept for ablation)

    The output projector mirrors the encoder's (Linear -> BatchNorm), as the paper
    specifies, so predictions live in the same normalised space as the targets
    they are compared against.
    """

    def __init__(self, emb_dim: int = 256, n_actions: int = 18,
                 n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1,
                 action_cond: str = "adaln"):
        super().__init__()
        if action_cond not in ("adaln", "add"):
            raise ValueError(f"action_cond must be 'adaln' or 'add', got {action_cond!r}")
        self.action_cond = action_cond
        self.act_emb = nn.Embedding(n_actions, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, emb_dim) * 0.02)

        if action_cond == "adaln":
            self.blocks = nn.ModuleList([
                AdaLNBlock(emb_dim, n_heads, dropout=dropout) for _ in range(n_layers)
            ])
            self.final_norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Projector matching the encoder's, per the paper.
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.out_bn = nn.BatchNorm1d(emb_dim)

    def forward(self, emb, actions):
        """
        emb:     (B, T, D)  sequence of frame embeddings
        actions: (B, T)     discrete actions taken at each step
        returns: (B, T, D)  predicted next embeddings
        """
        B, T, D = emb.shape
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=emb.device)
        act = self.act_emb(actions)

        if self.action_cond == "adaln":
            x = emb + self.pos_emb[:, :T]
            for blk in self.blocks:
                x = blk(x, act, mask)
            x = self.final_norm(x)
        else:
            x = emb + act + self.pos_emb[:, :T]
            x = self.transformer(x, mask=mask, is_causal=True)

        x = self.out_proj(x)
        # BatchNorm1d wants (N, C): fold batch and time together.
        return self.out_bn(x.reshape(B * T, D)).reshape(B, T, D)


# ---------------------------------------------------------------------------
# LeWM: full model (encoder + predictor + loss)
# ---------------------------------------------------------------------------

def effective_rank(emb_2d):
    """Entropy-based effective rank of an (N, D) embedding matrix.

    Ranges from ~1 (full dimensional collapse, embeddings lie on a line) up to
    D (embeddings fill the space). This is the right collapse metric here because
    BatchNorm forces unit per-dimension variance, so naive std is uninformative;
    collapse instead shows up as the embeddings occupying a low-dim subspace.
    """
    x = emb_2d.float().cpu()   # svdvals isn't implemented on MPS; tiny matrix anyway
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * (p + 1e-12).log()).sum()
    return float(torch.exp(entropy))


class LeWM(nn.Module):
    def __init__(self, n_actions: int, emb_dim: int = 256, frame_h: int = 84,
                 frame_w: int = 84, sigreg_weight: float = 0.1,
                 stop_grad: bool = False, action_cond: str = "adaln"):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, emb_dim=emb_dim)
        self.predictor = Predictor(emb_dim=emb_dim, n_actions=n_actions,
                                   action_cond=action_cond)
        self.sigreg = SIGReg()
        self.sigreg_weight = sigreg_weight
        # Faithful LeWM has NO stop-gradient (the paper's central claim is that
        # SIGReg makes this stable). stop_grad=True is only for A/B ablation.
        self.stop_grad = stop_grad

    def encode_sequence(self, obs_seq):
        """
        obs_seq: (B, T, C, H, W) float32 in [0, 1]
        returns: (B, T, D)
        """
        B, T, C, H, W = obs_seq.shape
        flat = rearrange(obs_seq, "b t c h w -> (b t) c h w")
        emb = self.encoder(flat)
        return rearrange(emb, "(b t) d -> b t d", b=B, t=T)

    def forward(self, obs_seq, actions):
        """
        obs_seq: (B, T, C, H, W)
        actions: (B, T) int64
        returns: dict with loss and components
        """
        emb = self.encode_sequence(obs_seq)          # (B, T, D)
        pred = self.predictor(emb[:, :-1], actions[:, :-1])  # predict t+1 from t

        target = emb[:, 1:]
        if self.stop_grad:                           # ablation only, NOT faithful LeWM
            target = target.detach()
        pred_loss = F.mse_loss(pred, target)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))       # (T, B, D)
        loss = pred_loss + self.sigreg_weight * sigreg_loss

        return {"loss": loss, "pred_loss": pred_loss,
                "sigreg_loss": sigreg_loss, "emb": emb.detach()}


# ---------------------------------------------------------------------------
# Replay buffer: stores sequences of (obs, action)
# ---------------------------------------------------------------------------

class SequenceBuffer:
    """Ring buffer of (obs, action) windows.

    Frames are kept as uint8 and scaled to [0, 1] only when a batch is sampled.
    At 84x84x3 a 17-frame window is 360 KB as uint8 against 1.4 MB as float32, so
    this is what makes a buffer of a few thousand sequences fit in RAM.
    """

    def __init__(self, capacity: int = 10_000, seq_len: int = 16):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_buf = []    # each entry: (seq_len+1, C, H, W) uint8
        self.act_buf = []    # each entry: (seq_len+1,) int64
        self._idx = 0

    def add(self, obs_seq, act_seq):
        """obs_seq: numpy (seq_len+1, C, H, W) uint8, act_seq: numpy (seq_len+1,)"""
        if len(self.obs_buf) < self.capacity:
            self.obs_buf.append(obs_seq)
            self.act_buf.append(act_seq)
        else:
            self.obs_buf[self._idx % self.capacity] = obs_seq
            self.act_buf[self._idx % self.capacity] = act_seq
        self._idx += 1

    def sample(self, batch_size: int):
        idx = random.sample(range(len(self.obs_buf)), batch_size)
        obs = np.stack([self.obs_buf[i] for i in idx])   # (B, T, C, H, W) uint8
        act = np.stack([self.act_buf[i] for i in idx])   # (B, T)
        return (torch.from_numpy(obs).float().div_(255.0),
                torch.from_numpy(act).long())

    def nbytes(self):
        """Approximate RAM held by the stored frames."""
        return sum(o.nbytes for o in self.obs_buf)

    def __len__(self):
        return len(self.obs_buf)


# ---------------------------------------------------------------------------
# Data collection: run JAXtari and fill the replay buffer
# ---------------------------------------------------------------------------

def make_env(game: str, img_size: int = 84):
    # sticky_actions=0.0 deliberately, overriding AtariWrapper's 0.25 default.
    # Under stickiness the wrapper executes the *previous* action with probability
    # 0.25 while the caller only sees the action it requested, so a world model
    # trained here would learn p(z'|z, a) from action labels that are wrong a
    # quarter of the time, which corrupts exactly what the action-conditioning
    # ablation is meant to measure. It also keeps the world model's dynamics
    # identical to the PPO environment in ppo_lewm.py, which sets 0.0 too.
    #
    # episodic_life=False for the same reason: it ends an "episode" at the first
    # lost life, which is a credit-assignment aid for RL and meaningless for a
    # reward-free world model. Left on, Breakout episodes end after ~22 random
    # steps, so the model would never observe a partly-cleared wall no matter how
    # long the step cap is.
    env = jaxatari.make(game)
    env = AtariWrapper(env, sticky_actions=0.0, episodic_life=False)
    env = PixelObsWrapper(
        env,
        do_pixel_resize=True,
        pixel_resize_shape=(img_size, img_size),
        grayscale=False,
        frame_stack_size=1,   # we handle our own sequence, no stacking needed
        frame_skip=4,
    )
    return env


def rollout_episode(env, key, n_actions: int, max_steps: int):
    """Play one episode under a uniform-random policy.

    Returns (obs, actions) as numpy arrays of the same length L <= max_steps:
    obs is (L, 3, H, W) uint8, actions is (L,) int64. actions[i] is the action
    taken *from* obs[i], so (obs[i], actions[i]) -> obs[i+1]. Scaling to [0, 1]
    happens at sampling time to keep the replay buffer small.
    """
    obs, state = env.reset(key)
    obs_list, act_list = [], []
    for _ in range(max_steps):
        # obs shape from PixelObsWrapper: (1, H, W, 3), 1 stacked frame
        frame = np.asarray(obs[0], dtype=np.uint8)
        obs_list.append(frame.transpose(2, 0, 1))       # (3, H, W)

        action = random.randint(0, n_actions - 1)
        act_list.append(action)

        obs, state, reward, done, truncated, info = env.step(state, action)
        if done or truncated:
            break

    return np.stack(obs_list), np.array(act_list, dtype=np.int64)


def episode_windows(obs, acts, seq_len: int, max_windows: int, rng=None):
    """Cut an episode into non-overlapping (seq_len+1)-frame windows.

    Windows never straddle a reset, so every sequence stays within one episode.

    When an episode yields more windows than `max_windows`, the kept ones are
    sampled uniformly from the whole episode rather than taken from the front.
    Taking the front is what an earlier version did, and combined with a short
    episode cap it meant the model only ever saw the opening seconds of a game,
    it never observed, say, a partly-cleared Breakout wall. That is a coverage
    limitation strong enough to be confused with a property of the objective, so
    it is worth avoiding rather than explaining away.
    """
    need = seq_len + 1
    starts = list(range(0, len(obs) - need + 1, need))
    if len(starts) > max_windows:
        rng = rng or np.random
        starts = sorted(rng.choice(starts, size=max_windows, replace=False))
    # .copy() so a stored window does not keep the whole episode array alive
    return [(obs[s:s + need].copy(), acts[s:s + need].copy()) for s in starts]


def collect_sequences(env, key, buffer: SequenceBuffer,
                      n_sequences: int, seq_len: int, n_actions: int,
                      max_windows_per_episode: int = 4,
                      max_episode_steps: int = 500):
    """Run a random policy and fill `buffer` with `n_sequences` windows."""
    collected = 0
    episodes = 0
    # Enough episodes to be sure the game genuinely cannot produce a full window,
    # not just that we were unlucky.
    max_episodes = n_sequences * 20 + 50
    while collected < n_sequences:
        if episodes > max_episodes:
            raise RuntimeError(
                f"collect_sequences: only gathered {collected}/{n_sequences} windows "
                f"from {episodes} episodes, no episode reached seq_len+1="
                f"{seq_len + 1} steps. Reduce --seq_len for this game."
            )
        episodes += 1
        key, rk = jax.random.split(key)
        obs, acts = rollout_episode(env, rk, n_actions, max_steps=max_episode_steps)
        for w_obs, w_act in episode_windows(obs, acts, seq_len, max_windows_per_episode):
            buffer.add(w_obs, w_act)
            collected += 1
            if collected >= n_sequences:
                break

    return key


# ---------------------------------------------------------------------------
# World-model evaluation: open-loop latent rollout error
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rollout(model, env, key, device, n_seq, seq_len, n_actions, context=3,
                     max_episode_steps=500):
    """Open-loop multi-step prediction error in latent space.

    Encode a fresh held-out trajectory, seed the predictor with `context` true
    embeddings, then autoregressively predict the remaining embeddings *feeding
    the model's own predictions back in* (no teacher forcing). Report MSE vs. the
    true embeddings per horizon step, alongside a 'frozen' baseline (assume the
    last observed embedding never changes) so the predictor's value is visible.

    Returns a dict, or None if not enough full sequences could be collected.

    Note: the model is evaluated with BatchNorm in *batch-statistics* mode (not
    running stats), to stay consistent with how it is trained. Dropout is still
    disabled.

    This cuts both ways for the predictor's output BatchNorm: during an
    autoregressive rollout the sequence grows and is increasingly self-generated,
    so batch statistics are computed partly over the model's own predictions.
    Measured on a trained Pong model, using fixed running statistics for the
    predictor's BN instead gives rollout MSE 0.182 vs 0.196, about 7%, i.e. real
    but small, and far from explaining architecture-level differences. Batch-stat
    mode is kept so that every number in these results is comparable; the 7% is
    the size of the methodological wobble underneath them.
    """
    model.eval()
    for mod in model.modules():           # BN back to batch-stat mode; dropout stays off
        if isinstance(mod, nn.BatchNorm1d):
            mod.train()
    obs_list, act_list = [], []
    collected, episodes = 0, 0
    while collected < n_seq and episodes < n_seq * 20 + 50:
        episodes += 1
        key, rk = jax.random.split(key)
        ep_obs, ep_act = rollout_episode(env, rk, n_actions, max_steps=max_episode_steps)
        for w_obs, w_act in episode_windows(ep_obs, ep_act, seq_len, max_windows=4):
            obs_list.append(w_obs)
            act_list.append(w_act)
            collected += 1
            if collected >= n_seq:
                break

    if collected == 0:
        model.train()
        return None

    obs = torch.from_numpy(np.stack(obs_list)).float().div_(255.0).to(device)
    act = torch.from_numpy(np.stack(act_list)).long().to(device)

    z = model.encode_sequence(obs)            # (N, T+1, D), ground-truth embeddings
    Tp1 = z.size(1)
    seq = z[:, :context, :].clone()           # seed with true context embeddings
    preds = []
    for L in range(context, Tp1):             # predict positions context .. T
        out = model.predictor(seq, act[:, :L])
        z_next = out[:, -1, :]                 # predicted next embedding
        preds.append(z_next)
        seq = torch.cat([seq, z_next.unsqueeze(1)], dim=1)

    preds = torch.stack(preds, dim=1)         # (N, H, D)
    targets = z[:, context:, :]               # (N, H, D)
    frozen = z[:, context - 1:context, :].expand_as(targets)  # last-seen, held constant

    rollout_mse = ((preds - targets) ** 2).mean(dim=(0, 2)).cpu().tolist()
    frozen_mse = ((frozen - targets) ** 2).mean(dim=(0, 2)).cpu().tolist()

    model.train()
    return {
        "context": context,
        "n_seq": collected,
        "rollout_mse_by_horizon": rollout_mse,   # index 0 = 1-step ahead
        "frozen_baseline_by_horizon": frozen_mse,
        "rollout_mse_mean": float(np.mean(rollout_mse)),
        "frozen_baseline_mean": float(np.mean(frozen_mse)),
    }


# ---------------------------------------------------------------------------
# Run artifacts: learning-curve plot + JSON history
# ---------------------------------------------------------------------------

def save_learning_curve(game: str, loss_history: list, path) -> bool:
    """Save the learning curve for one game as a PNG. Returns True if written.

    Left axis: total / prediction / SIGReg loss. Right axis: effective rank of
    the embeddings, the collapse diagnostic. A curve where the loss falls while
    the effective rank falls towards 1 is collapse, not learning, so the two
    belong on the same figure.
    """
    if len(loss_history) < 2:
        return False

    import matplotlib
    matplotlib.use("Agg")          # headless, no display needed
    import matplotlib.pyplot as plt

    steps = [e["step"] for e in loss_history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [e["loss"] for e in loss_history],
            label="total", linewidth=2, color="C0")
    ax.plot(steps, [e["pred_loss"] for e in loss_history],
            label="pred", linestyle="--", color="C1")
    ax.plot(steps, [e["sigreg_loss"] for e in loss_history],
            label="sigreg", linestyle=":", color="C2")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title(f"LeWM, {game}")
    ax.grid(True, alpha=0.3)

    ranks = [e.get("eff_rank") for e in loss_history]
    if any(r is not None for r in ranks):
        ax2 = ax.twinx()
        ax2.plot(steps, ranks, label="eff_rank", color="C3", marker=".", markersize=3)
        ax2.set_ylabel("Effective rank")
        ax2.set_ylim(bottom=0)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax.legend(fontsize=8)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_history(path, game, args, loss_history, eval_metrics=None):
    """Write the full per-game record as JSON (atomically)."""
    payload = {
        "game": game,
        "args": {k: v for k, v in vars(args).items()},
        "loss_history": loss_history,
        "eval_metrics": eval_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pick_device(requested: str = "auto") -> torch.device:
    """Resolve the torch device. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, step, loss_history, args):
    """Crash-safe checkpoint (atomic write). The replay buffer is NOT stored,
    it is re-collected on resume, since serializing it would be many GB."""
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_history": loss_history,
        "args": vars(args),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(args):
    args = fill_defaults(args)

    # Everything this run produces lives in one directory: <outdir>/<game>/
    game_dir = Path(args.outdir) / args.game
    game_dir.mkdir(parents=True, exist_ok=True)
    curve_path = game_dir / "curve.png"
    history_path = game_dir / "history.json"
    model_path = game_dir / "model.pt"
    ckpt_path = game_dir / "ckpt.pt"

    # Seed every RNG this run touches: the JAX key drives env resets, but the
    # behaviour policy's actions and the replay-buffer sampling come from Python's
    # `random`, and weight init from torch. Seeding only JAX would leave the run
    # unreproducible.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"Using device: {device}")
    print(f"Output dir:   {game_dir}")

    env = make_env(args.game, img_size=args.img_size)
    n_actions = env.action_space().n
    print(f"Game: {args.game} | Actions: {n_actions}")

    model = LeWM(
        n_actions=n_actions,
        emb_dim=args.emb_dim,
        sigreg_weight=args.sigreg_weight,
        stop_grad=args.stop_grad,
        action_cond=args.action_cond,
    ).to(device)
    print(f"Action conditioning: {args.action_cond}"
          f"{' (paper)' if args.action_cond == 'adaln' else ' (ABLATION, paper uses AdaLN)'}")
    if model.stop_grad:
        print("WARNING: stop_grad=True, this is the ablation, NOT faithful LeWM.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    buffer = SequenceBuffer(capacity=args.buffer_size, seq_len=args.seq_len)

    key = jax.random.PRNGKey(args.seed)

    # --- optional resume ---
    start_step = 0
    loss_history = []  # list of {"step", "loss", "pred_loss", "sigreg_loss", "eff_rank"}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        loss_history = ckpt.get("loss_history", [])
        print(f"Resumed from {args.resume} at step {start_step}")

    # --- initial data collection (always, buffer is not checkpointed) ---
    print(f"Collecting {args.init_sequences} initial sequences...")
    key = collect_sequences(env, key, buffer, args.init_sequences, args.seq_len,
                            n_actions, args.max_windows_per_episode,
                            args.max_episode_steps)
    print(f"Buffer size: {len(buffer)} sequences ({buffer.nbytes() / 1e6:.0f} MB)")

    if len(buffer) < args.batch_size:
        raise RuntimeError(
            f"buffer holds {len(buffer)} sequences but batch_size is {args.batch_size} "
            f"- raise --init_sequences or lower --batch_size."
        )

    print("Starting training...")
    rtpt = make_rtpt(f"LeWM-{args.game}", args.total_steps // max(1, args.log_every))
    for step in range(start_step + 1, args.total_steps + 1):
        # collect more data every N steps
        if step % args.collect_every == 0:
            key = collect_sequences(env, key, buffer, args.collect_n, args.seq_len,
                                    n_actions, args.max_windows_per_episode,
                                    args.max_episode_steps)

        # sample batch and train
        obs_batch, act_batch = buffer.sample(args.batch_size)
        obs_batch = obs_batch.to(device)
        act_batch = act_batch.to(device)

        model.train()
        optimizer.zero_grad()
        losses = model(obs_batch, act_batch)
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            emb = losses["emb"]                       # (B, T, D)
            rank = effective_rank(emb.reshape(-1, emb.size(-1)))
            entry = {
                "step": step,
                "loss": losses["loss"].item(),
                "pred_loss": losses["pred_loss"].item(),
                "sigreg_loss": losses["sigreg_loss"].item(),
                "eff_rank": rank,
            }
            loss_history.append(entry)
            print(
                f"step {step:6d} | "
                f"loss {entry['loss']:.4f} | "
                f"pred {entry['pred_loss']:.4f} | "
                f"sigreg {entry['sigreg_loss']:.4f} | "
                f"eff_rank {rank:6.1f}/{emb.size(-1)} | "
                f"buffer {len(buffer)}"
            )

        # Refresh the learning curve and history mid-run, so a long run that is
        # killed or crashes still leaves usable artifacts behind.
        if args.plot and args.plot_every and step % args.plot_every == 0:
            if save_learning_curve(args.game, loss_history, curve_path):
                save_history(history_path, args.game, args, loss_history)
                print(f"  curve -> {curve_path} (step {step})")

        if args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(str(ckpt_path), model, optimizer, step, loss_history, args)
            print(f"  checkpoint -> {ckpt_path} (step {step})")

    # --- evaluate: open-loop latent rollout error on held-out trajectories ---
    print("\nEvaluating (open-loop latent rollout)...")
    key, ek = jax.random.split(key)
    eval_metrics = evaluate_rollout(
        model, env, ek, device,
        n_seq=args.eval_seq,
        seq_len=args.seq_len, n_actions=n_actions,
        context=args.eval_context, max_episode_steps=args.max_episode_steps,
    )
    if eval_metrics is None:
        print("  (could not collect eval sequences, skipped)")
    else:
        roll = eval_metrics["rollout_mse_by_horizon"]
        base = eval_metrics["frozen_baseline_by_horizon"]
        print(f"  context={eval_metrics['context']}  n_seq={eval_metrics['n_seq']}")
        print(f"  {'horizon':>7} | {'rollout MSE':>12} | {'frozen base':>12}")
        for h, (r, b) in enumerate(zip(roll, base), start=1):
            print(f"  {h:>7} | {r:>12.4f} | {b:>12.4f}")
        print(f"  mean rollout MSE {eval_metrics['rollout_mse_mean']:.4f} "
              f"vs frozen {eval_metrics['frozen_baseline_mean']:.4f} "
              f"({'BEATS' if eval_metrics['rollout_mse_mean'] < eval_metrics['frozen_baseline_mean'] else 'WORSE THAN'} baseline)")

    # --- save artifacts: curve, history, weights ---
    if args.plot and save_learning_curve(args.game, loss_history, curve_path):
        print(f"Plot saved to    {curve_path}")
    save_history(history_path, args.game, args, loss_history, eval_metrics)
    print(f"History saved to {history_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to   {model_path}")

    return model, loss_history, eval_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="LeWM world model for JAXtari")
    d = DEFAULTS
    p.add_argument("--game", type=str, default=d["game"])
    p.add_argument("--emb_dim", type=int, default=d["emb_dim"])
    p.add_argument("--seq_len", type=int, default=d["seq_len"])
    p.add_argument("--batch_size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--total_steps", type=int, default=d["total_steps"])
    p.add_argument("--init_sequences", type=int, default=d["init_sequences"])
    p.add_argument("--collect_every", type=int, default=d["collect_every"])
    p.add_argument("--collect_n", type=int, default=d["collect_n"])
    p.add_argument("--buffer_size", type=int, default=d["buffer_size"])
    p.add_argument("--max_windows_per_episode", type=int, default=d["max_windows_per_episode"],
                   help="cap on sequences taken from a single episode")
    p.add_argument("--sigreg_weight", type=float, default=d["sigreg_weight"])
    p.add_argument("--img_size", type=int, default=d["img_size"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--log_every", type=int, default=d["log_every"])
    p.add_argument("--eval_seq", type=int, default=d["eval_seq"],
                   help="held-out sequences for the rollout evaluation")
    p.add_argument("--eval_context", type=int, default=d["eval_context"],
                   help="context frames seeding the open-loop rollout")
    p.add_argument("--device", type=str, default=d["device"],
                   help="auto | cpu | cuda | mps")
    p.add_argument("--outdir", type=str, default=d["outdir"],
                   help="artifacts go to <outdir>/<game>/{curve.png,history.json,model.pt}")
    p.add_argument("--no_plot", dest="plot", action="store_false",
                   help="skip the learning-curve PNG (written by default)")
    p.add_argument("--plot_every", type=int, default=d["plot_every"],
                   help="refresh the curve every N steps during training (0 = only at the end)")
    p.add_argument("--max_episode_steps", type=int, default=d["max_episode_steps"],
                   help="how far into an episode to play before resetting; windows "
                        "are then sampled across it, so coverage is not limited to "
                        "the opening seconds of a game")
    p.add_argument("--action_cond", choices=["adaln", "add"], default=d["action_cond"],
                   help="how actions enter the predictor: adaln = per-token AdaLN "
                        "modulation (what the paper does); add = additive action "
                        "embedding (simpler, kept for ablation)")
    p.add_argument("--stop_grad", action="store_true",
                   help="ABLATION ONLY: stop-gradient on the target. "
                        "Faithful LeWM does NOT use this.")
    p.add_argument("--ckpt_every", type=int, default=d["ckpt_every"],
                   help="save a resumable checkpoint every N steps (0 = off)")
    p.add_argument("--resume", type=str, default=d["resume"],
                   help="resume training from a checkpoint path")
    p.set_defaults(plot=d["plot"])
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
