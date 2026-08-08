"""
Turn a trained LeWM world model into a frozen feature extractor for `ppo_lewm.py`.

Only the encoder is reused; the predictor plays no part in acting. The one thing
that needs care is BatchNorm.

The encoder is trained online, so its BatchNorm *running* statistics chase a
moving feature distribution and end up stale — using them directly inflates the
embedding scale several-fold and hands the policy garbage features. (This is the
same pitfall documented in `evaluate_rollout`, which sidesteps it by evaluating in
batch-statistics mode.) A policy needs a genuinely fixed encoder, so instead we
recompute the running statistics over fresh frames from the game before freezing.
After that the encoder is deterministic and safe to put in eval mode.
"""

import argparse

import jax
import numpy as np
import torch

from lewm_jaxatari import CNNEncoder, make_env, rollout_episode


def load_encoder(model_path, device="cpu"):
    """Rebuild just the CNNEncoder from a saved LeWM state_dict."""
    state = torch.load(model_path, map_location="cpu")
    enc_state = {k[len("encoder."):]: v for k, v in state.items()
                 if k.startswith("encoder.")}
    if not enc_state:
        raise ValueError(f"{model_path} contains no 'encoder.*' weights")
    emb_dim = enc_state["proj.0.weight"].shape[0]
    encoder = CNNEncoder(in_channels=3, emb_dim=emb_dim)
    encoder.load_state_dict(enc_state)
    return encoder.to(device), emb_dim


@torch.no_grad()
def recalibrate_bn(encoder, game, n_frames=4096, batch_size=128, seed=0,
                   img_size=84, device="cpu"):
    """Recompute the encoder's BatchNorm running statistics on fresh frames.

    Returns the number of frames used. Leaves the encoder in eval mode.
    """
    env = make_env(game, img_size=img_size)
    n_actions = env.action_space().n
    key = jax.random.PRNGKey(seed)

    # momentum=None makes BatchNorm accumulate a cumulative average over every
    # batch it sees, rather than an exponential one weighted to the last few.
    for mod in encoder.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.reset_running_stats()
            mod.momentum = None

    encoder.train()                       # updates running stats, not the weights
    pending, used = np.zeros((0, 3, img_size, img_size), dtype=np.uint8), 0
    while used < n_frames:
        key, rk = jax.random.split(key)
        obs, _ = rollout_episode(env, rk, n_actions, max_steps=256)
        used += len(obs)
        pending = np.concatenate([pending, obs])
        while len(pending) >= batch_size:
            batch = torch.from_numpy(pending[:batch_size]).float().div_(255.0).to(device)
            encoder(batch)
            pending = pending[batch_size:]

    encoder.eval()
    return used


def load_frozen_encoder(model_path, game, device="cpu", recalibrate_frames=4096,
                        seed=0, img_size=84):
    """Load a LeWM encoder, refresh its BN statistics, and return it in eval mode.

    Returns (encoder, emb_dim). Freezing the parameters is left to the caller —
    `lewm_finetune` wants them trainable.
    """
    encoder, emb_dim = load_encoder(model_path, device=device)
    if recalibrate_frames > 0:
        used = recalibrate_bn(encoder, game, n_frames=recalibrate_frames,
                              seed=seed, img_size=img_size, device=device)
        print(f"  recalibrated BatchNorm over {used} frames of {game}")
    encoder.eval()
    return encoder, emb_dim


@torch.no_grad()
def embedding_stats(encoder, game, n_frames=512, seed=1, img_size=84, device="cpu"):
    """Sanity check: per-dimension mean/std of the frozen encoder's embeddings.

    A correctly recalibrated encoder gives roughly zero mean and unit std. A std
    far from 1 means the BatchNorm statistics do not match the data the policy
    will actually see.
    """
    env = make_env(game, img_size=img_size)
    n_actions = env.action_space().n
    key = jax.random.PRNGKey(seed)
    frames = []
    while sum(len(f) for f in frames) < n_frames:
        key, rk = jax.random.split(key)
        obs, _ = rollout_episode(env, rk, n_actions, max_steps=256)
        frames.append(obs)
    x = np.concatenate(frames)[:n_frames]
    z = encoder(torch.from_numpy(x).float().div_(255.0).to(device))
    return float(z.mean()), float(z.std())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect a frozen LeWM encoder")
    p.add_argument("--model", required=True, help="path to a LeWM model.pt")
    p.add_argument("--game", required=True, help="game the encoder was trained on")
    p.add_argument("--recalibrate_frames", type=int, default=4096)
    args = p.parse_args()

    enc, emb_dim = load_encoder(args.model)
    print(f"encoder loaded (emb_dim={emb_dim})")
    m, s = embedding_stats(enc, args.game)
    print(f"  before recalibration: mean {m:+.3f}  std {s:.3f}")
    recalibrate_bn(enc, args.game, n_frames=args.recalibrate_frames)
    m, s = embedding_stats(enc, args.game)
    print(f"  after  recalibration: mean {m:+.3f}  std {s:.3f}   (want ~0.0 / ~1.0)")
