"""
PPO on JAXtari with a pluggable feature extractor — the downstream evaluation of
the LeWM world model.

LeWM is trained reward-free (`lewm_jaxatari.py`), so on its own it produces no game
scores. This script closes that gap: it freezes the pretrained LeWM encoder, learns
a policy on top of its embeddings, and reports episodic return — numbers directly
comparable to the repo's PPO/PQN baselines and to Dreamer.

Three feature extractors, selected with `--features`:

    scratch        Nature-DQN CNN trained end-to-end with the policy (the control)
    lewm_frozen    pretrained LeWM encoder, weights frozen (representation quality)
    lewm_finetune  pretrained LeWM encoder, updated by the policy gradient

All three see identical observations, use identical policy/value heads, and run
under identical hyperparameters and step budgets, so the only thing that varies is
where the features come from. That is what makes the comparison mean anything.

Why PyTorch and not JAX
-----------------------
The environment stays JAXtari (JAX); only the networks are PyTorch. On Apple
silicon JAX has no GPU backend and runs on CPU, while PyTorch reaches the GPU
through MPS. Measured on the Nature-CNN trunk at batch 512, one forward+backward
costs 772 ms under JAX/CPU against 37 ms under PyTorch/MPS — a 20x gap on the
operation PPO spends nearly all of its time in. Keeping the networks in PyTorch
also means the LeWM encoder loads directly, with no cross-framework weight port.

PPO follows `scripts/benchmarks/ppo_jaxatari_scan.py` in this repo (itself
CleanRL's `ppo_atari_envpool_xla_jax_scan.py`, https://github.com/vwxyzjn/cleanrl,
adapted to JAXtari). Default hyperparameters match `config/alg/ppo_jaxatari_pixel.yaml`
so the baseline is comparable.

Usage
-----
    python ppo_lewm.py --game pong --features scratch     --total_timesteps 1000000
    python ppo_lewm.py --game pong --features lewm_frozen --total_timesteps 1000000 \
        --encoder results/full/pong/model.pt
"""

import argparse
import json
import os
import time
from pathlib import Path

# JAX (the environment) and PyTorch (the networks) share one GPU here. By default
# JAX preallocates ~75% of VRAM on first use, which leaves PyTorch unable to
# allocate and fails with a confusing OOM. Must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

import jaxatari
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper, LogWrapper

from lewm_jaxatari import make_rtpt, pick_device
from lewm_features import load_frozen_encoder


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_env(game: str, img_size: int = 84, frame_stack: int = 4, eval: bool = False):
    """RGB frame-stacked JAXtari env.

    Deliberately RGB rather than the grayscale the repo's PPO baseline uses,
    because the LeWM encoder was pretrained on RGB frames. Both arms of the
    comparison run on this same env, so the choice does not favour either.
    """
    env = jaxatari.make(game)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=not eval,
        first_fire=True,
        noop_max=30,
        full_action_space=False,
    )
    env = PixelObsWrapper(
        env,
        do_pixel_resize=True,
        pixel_resize_shape=(img_size, img_size),
        grayscale=False,
        frame_stack_size=frame_stack,
        frame_skip=4,
        max_pooling=True,
        clip_reward=not eval,
    )
    return LogWrapper(env)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def orthogonal_(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class NatureCNN(nn.Module):
    """The standard Atari trunk, trained from scratch — the control arm."""

    def __init__(self, in_channels: int, hidden: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            orthogonal_(nn.Conv2d(in_channels, 32, 8, 4)), nn.ReLU(),
            orthogonal_(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            orthogonal_(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(orthogonal_(nn.Linear(3136, hidden)), nn.ReLU())

    def forward(self, x):
        """x: (B, F, 3, H, W) float in [0,1] -> (B, hidden)"""
        B, F = x.shape[:2]
        return self.fc(self.conv(x.reshape(B, F * 3, *x.shape[3:])))


class LeWMTrunk(nn.Module):
    """Pretrained LeWM encoder applied per frame, then a learned projection.

    The world model encodes single frames — all temporal structure lives in its
    predictor, which the policy does not use. So the frame stack is encoded
    frame-by-frame with the *same* encoder and the embeddings are concatenated,
    which is what gives the policy access to motion.
    """

    def __init__(self, encoder, emb_dim: int, frame_stack: int,
                 hidden: int = 512, frozen: bool = True):
        super().__init__()
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.frozen = frozen
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.fc = nn.Sequential(
            orthogonal_(nn.Linear(frame_stack * emb_dim, hidden)), nn.ReLU()
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained encoder's BatchNorm always stays in inference mode, even
        # when fine-tuning: the weights still receive gradients, only the
        # normalisation statistics are held fixed.
        #
        # This is not cosmetic. PPO stores log-probs during the rollout and
        # recomputes them during the update to form an importance ratio. With BN
        # in training mode the two passes normalise by different statistics —
        # rollout batches are num_envs*frame_stack (64 frames here) while update
        # minibatches are 2048 — so the ratio compares two different functions,
        # explodes, and takes the policy and critic with it. Observed directly:
        # value loss reaching 1e15 and entropy collapsing to 0 within 25
        # iterations on every game.
        self.encoder.eval()
        return self

    def forward(self, x):
        """x: (B, F, 3, H, W) float in [0,1] -> (B, hidden)"""
        B, F = x.shape[:2]
        flat = x.reshape(B * F, *x.shape[2:])
        if self.frozen:
            with torch.no_grad():
                z = self.encoder(flat)
        else:
            z = self.encoder(flat)
        return self.fc(z.reshape(B, F * self.emb_dim))


class Agent(nn.Module):
    def __init__(self, trunk, hidden: int, n_actions: int):
        super().__init__()
        self.trunk = trunk
        self.actor = orthogonal_(nn.Linear(hidden, n_actions), gain=0.01)
        self.critic = orthogonal_(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    run_dir = Path(args.outdir) / f"{args.game}_{args.features}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    if num_iterations == 0:
        raise ValueError(
            f"total_timesteps={args.total_timesteps} is smaller than one batch "
            f"({batch_size}) — nothing would be trained."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    env = make_env(args.game, frame_stack=args.frame_stack)
    n_actions = env.action_space().n
    print(f"Run dir: {run_dir}")
    print(f"Game: {args.game} | actions: {n_actions} | features: {args.features} "
          f"| device: {device}")

    # --- feature extractor ---
    if args.features == "scratch":
        trunk = NatureCNN(in_channels=args.frame_stack * 3, hidden=args.hidden)
    else:
        if not args.encoder:
            raise ValueError(f"--features {args.features} requires --encoder <model.pt>")
        encoder, emb_dim = load_frozen_encoder(
            args.encoder, args.game, device=device,
            recalibrate_frames=args.recalibrate_frames, seed=args.seed,
        )
        trunk = LeWMTrunk(encoder, emb_dim, args.frame_stack, hidden=args.hidden,
                          frozen=args.features == "lewm_frozen")
        print(f"Loaded LeWM encoder from {args.encoder} (emb_dim={emb_dim}, "
              f"{'frozen' if trunk.frozen else 'fine-tuned'})")

    agent = Agent(trunk, args.hidden, n_actions).to(device)
    params = [p for p in agent.parameters() if p.requires_grad]

    # Fine-tuning uses a smaller step on the pretrained encoder than on the
    # freshly-initialised heads. With its BatchNorm held in inference mode the
    # encoder's output scale is no longer renormalised, so at the full learning
    # rate the embeddings drift in magnitude, the actor logits follow, and the
    # policy goes deterministic within a few iterations (entropy -> 0 by
    # iteration 8, measured). A smaller step keeps the encoder near the scale
    # its BatchNorm statistics were calibrated for.
    enc_params, head_params = [], []
    for name, p in agent.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("trunk.encoder.") else head_params).append(p)
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr * args.encoder_lr_scale})
        print(f"Fine-tuning encoder at lr x{args.encoder_lr_scale} "
              f"({len(enc_params)} tensors)")
    optimizer = torch.optim.Adam(groups, lr=args.lr, eps=1e-5)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    @jax.jit
    def vmap_reset(keys):
        return jax.vmap(env.reset)(keys)

    @jax.jit
    def vmap_step(state, action):
        obs, state, reward, term, trunc, info = jax.vmap(env.step)(state, action)
        return (obs, state, reward, jnp.logical_or(term, trunc),
                info["returned_episode"], info["returned_episode_returns"])

    key = jax.random.PRNGKey(args.seed)
    key, *reset_keys = jax.random.split(key, args.num_envs + 1)
    next_obs_j, env_state = vmap_reset(jnp.stack(reset_keys))
    next_done = torch.zeros(args.num_envs, device=device)

    def to_torch(obs_j):
        """(N, F, H, W, 3) uint8 jax -> (N, F, 3, H, W) float32 torch in [0,1]."""
        # np.array (not asarray) — JAX hands back a read-only buffer, and torch
        # refuses to own one safely.
        t = torch.from_numpy(np.array(obs_j)).to(device)
        return t.permute(0, 1, 4, 2, 3).float().div_(255.0)

    next_obs = to_torch(next_obs_j)

    obs_shape = (args.num_steps, args.num_envs, args.frame_stack, 3, 84, 84)
    b_obs = torch.zeros(obs_shape, device=device)
    b_actions = torch.zeros(args.num_steps, args.num_envs, dtype=torch.long, device=device)
    b_logprobs = torch.zeros(args.num_steps, args.num_envs, device=device)
    b_rewards = torch.zeros(args.num_steps, args.num_envs, device=device)
    b_dones = torch.zeros(args.num_steps, args.num_envs, device=device)
    b_values = torch.zeros(args.num_steps, args.num_envs, device=device)

    history, start, global_step = [], time.time(), 0
    print(f"Training for {num_iterations} iterations "
          f"({num_iterations * batch_size} steps)...")

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = frac * base

        ep_returns = []
        t_iter = time.time()

        # --- rollout ---
        agent.eval()
        for step in range(args.num_steps):
            global_step += args.num_envs
            b_obs[step] = next_obs
            b_dones[step] = next_done

            with torch.no_grad():
                logits, value = agent(next_obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                b_logprobs[step] = dist.log_prob(action)
            b_values[step] = value
            b_actions[step] = action

            a_j = jnp.asarray(action.cpu().numpy())
            next_obs_j, env_state, reward, done, fin, fin_ret = vmap_step(env_state, a_j)

            b_rewards[step] = torch.from_numpy(np.array(reward)).to(device)
            next_obs = to_torch(next_obs_j)
            next_done = torch.from_numpy(np.array(done, dtype=np.float32)).to(device)

            fin = np.asarray(fin)
            if fin.any():
                ep_returns.extend(np.asarray(fin_ret)[fin].tolist())
        t_rollout = time.time() - t_iter

        # --- GAE ---
        t_upd = time.time()
        with torch.no_grad():
            _, next_value = agent(next_obs)
        advantages = torch.zeros_like(b_rewards)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal, nextvalues = 1.0 - next_done, next_value
            else:
                nextnonterminal, nextvalues = 1.0 - b_dones[t + 1], b_values[t + 1]
            delta = b_rewards[t] + args.gamma * nextvalues * nextnonterminal - b_values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + b_values

        # --- PPO update ---
        agent.train()
        f_obs = b_obs.reshape(-1, *obs_shape[2:])
        f_actions = b_actions.reshape(-1)
        f_logprobs = b_logprobs.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)

        losses = []
        for _ in range(args.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for s in range(0, batch_size, minibatch_size):
                mb = perm[s:s + minibatch_size]
                logits, newvalue = agent(f_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                newlogprob = dist.log_prob(f_actions[mb])
                entropy = dist.entropy()

                logratio = newlogprob - f_logprobs[mb]
                ratio = logratio.exp()
                mb_adv = f_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue - f_returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                losses.append((loss.item(), v_loss.item(), ent_loss.item()))
        t_update = time.time() - t_upd

        arr = np.array(losses)
        entry = {
            "iteration": iteration,
            "global_step": global_step,
            "episodic_return_mean": float(np.mean(ep_returns)) if ep_returns else None,
            "episodes": len(ep_returns),
            "loss": float(arr[:, 0].mean()),
            "value_loss": float(arr[:, 1].mean()),
            "entropy": float(arr[:, 2].mean()),
            "sps": int(global_step / (time.time() - start)),
            "rollout_s": round(t_rollout, 2),
            "update_s": round(t_update, 2),
        }
        history.append(entry)

        rtpt.step()
        if iteration % args.log_every == 0 or iteration == num_iterations:
            ret = entry["episodic_return_mean"]
            print(f"iter {iteration:5d} | step {global_step:9d} | "
                  f"return {('%8.2f' % ret) if ret is not None else '     n/a'} "
                  f"({entry['episodes']:3d} eps) | v_loss {entry['value_loss']:7.3f} | "
                  f"ent {entry['entropy']:.3f} | {entry['sps']} sps "
                  f"| roll {entry['rollout_s']:.1f}s upd {entry['update_s']:.1f}s")
            save_run(run_dir, args, history)
            plot_returns(run_dir, args, history)

    save_run(run_dir, args, history)
    plot_returns(run_dir, args, history)
    torch.save(agent.state_dict(), run_dir / "agent.pt")
    print(f"Done in {(time.time() - start) / 60:.1f} min → {run_dir}")
    return history


def save_run(run_dir, args, history):
    payload = {"game": args.game, "features": args.features, "seed": args.seed,
               "args": vars(args), "history": history}
    tmp = Path(run_dir) / "history.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(Path(run_dir) / "history.json")


def plot_returns(run_dir, args, history):
    pts = [(e["global_step"], e["episodic_return_mean"])
           for e in history if e["episodic_return_mean"] is not None]
    if len(pts) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, rets = [p[0] for p in pts], [p[1] for p in pts]
    # Running mean — per-iteration returns average over few episodes and are noisy.
    w = max(1, len(rets) // 20)
    smooth = np.convolve(rets, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, rets, alpha=0.25, color="C0", linewidth=1)
    ax.plot(steps[w - 1:], smooth, color="C0", linewidth=2, label=args.features)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episodic return")
    ax.set_title(f"PPO on {args.game} — {args.features}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "returns.png", dpi=150)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(description="PPO on JAXtari with LeWM features")
    p.add_argument("--game", type=str, default="pong")
    p.add_argument("--features", choices=["scratch", "lewm_frozen", "lewm_finetune"],
                   default="scratch")
    p.add_argument("--encoder", type=str, default=None,
                   help="a LeWM model.pt (LeWM arms only)")
    p.add_argument("--encoder_lr_scale", type=float, default=0.1,
                   help="learning-rate multiplier for pretrained encoder weights "
                        "(lewm_finetune only)")
    p.add_argument("--recalibrate_frames", type=int, default=4096,
                   help="frames used to refresh the encoder's BatchNorm statistics")
    p.add_argument("--outdir", type=str, default="results/ppo")
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--frame_stack", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--anneal_lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--norm_adv", action="store_true", default=True)
    p.add_argument("--clip_coef", type=float, default=0.1)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
