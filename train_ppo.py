import time

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

from jaxatari.games.jax_stargunner import JaxStarGunner

# ---------------- config ----------------
N_ENVS       = 128
ROLLOUT      = 256
TOTAL_STEPS  = 100_000_000
LR           = 2.5e-4
GAMMA        = 0.99
LAMBDA       = 0.95
CLIP         = 0.2
ENT_COEF     = 0.01
VF_COEF      = 0.5
EPOCHS       = 4
MINIBATCHES  = 16
GRAD_CLIP    = 0.5
SEED         = 0
LOG_EVERY    = 10
# ----------------------------------------

env = JaxStarGunner(start_in_play=True)
N_ACTIONS = env.action_space().n
N_ITERS   = TOTAL_STEPS // (N_ENVS * ROLLOUT)


def flatten_obs(obs):
    """ObjectObservation -> normalized flat float32 vector."""
    parts = []
    for name in ("player", "enemies", "bullets", "bombs", "bobo"):
        o = getattr(obs, name)
        parts += [
            o.x.ravel() / 160.0,
            o.y.ravel() / 210.0,
            o.width.ravel() / 16.0,
            o.height.ravel() / 16.0,
        ]
    return jnp.concatenate(parts).astype(jnp.float32)


OBS_DIM = flatten_obs(env.reset(jax.random.PRNGKey(0))[0]).shape[0]
print(f"n_actions={N_ACTIONS}  obs_dim={OBS_DIM}  iters={N_ITERS}")


class ActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(256)(x))
        h = nn.relu(nn.Dense(256)(h))
        logits = nn.Dense(self.n_actions)(h)
        value = nn.Dense(1)(h).squeeze(-1)
        return logits, value


def make_train():
    def train(rng):
        # --- init envs ---
        rng, k = jax.random.split(rng)
        keys = jax.random.split(k, N_ENVS)
        obs, state = jax.vmap(env.reset)(keys)
        obs_flat = jax.vmap(flatten_obs)(obs)

        # --- init network ---
        rng, k = jax.random.split(rng)
        net = ActorCritic(N_ACTIONS)
        params = net.init(k, obs_flat[0])

        # ---- LR Annealing ----
        def linear_schedule(count):
            frac = 1.0 - (count // (N_ENVS * ROLLOUT)) / N_ITERS
            return LR * jnp.maximum(frac, 0.0)

        tx = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.adam(linear_schedule),
        )
        ts = TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=tx,
        )

        # --- vmaps ---
        def policy_step(ts, obs_flat, key):
            logits, value = ts.apply_fn(ts.params, obs_flat)
            action = jax.random.categorical(key, logits)
            logp = jax.nn.log_softmax(logits)[action]
            return action, logp, value

        def env_step_reset(state, action, reset_key):
            obs, new_state, reward, done, _ = env.step(state, action)
            reset_obs, reset_state = env.reset(reset_key)
            take = lambda a, b: jnp.where(done, a, b)
            new_state = jax.tree_util.tree_map(take, reset_state, new_state)
            new_obs = jax.tree_util.tree_map(take, reset_obs, obs)
            return new_obs, new_state, reward, done

        v_policy = jax.vmap(policy_step, in_axes=(None, 0, 0))
        v_env_step_reset = jax.vmap(env_step_reset, in_axes=(0, 0, 0))
        v_flatten = jax.vmap(flatten_obs)

        def rollout(ts, state, obs_flat, rng):
            def body(carry, _):
                ts, state, obs_flat, rng = carry
                rng, k = jax.random.split(rng)
                keys = jax.random.split(k, N_ENVS)
                actions, logps, values = v_policy(ts, obs_flat, keys)

                rng, rk = jax.random.split(rng)
                reset_keys = jax.random.split(rk, N_ENVS)
                obs, state, rew, done = v_env_step_reset(
                    state, actions, reset_keys
                )
                obs_flat = v_flatten(obs)

                return (ts, state, obs_flat, rng), (
                    obs_flat, actions, logps, values, rew, done
                )

            (ts, state, obs_flat, rng), traj = jax.lax.scan(
                body, (ts, state, obs_flat, rng), None, length=ROLLOUT
            )
            return ts, state, obs_flat, rng, traj

        def compute_gae(rewards, values, dones):
            # ---- Reward Scaling (neu) ----
            rewards = rewards / 10.0

            def body(carry, x):
                gae, next_value = carry
                r, v, d = x
                d = d.astype(jnp.float32)
                delta = r + GAMMA * next_value * (1.0 - d) - v
                gae = delta + GAMMA * LAMBDA * (1.0 - d) * gae
                return (gae, v), gae

            _, advs = jax.lax.scan(
                body,
                (jnp.zeros(N_ENVS), jnp.zeros(N_ENVS)),
                (rewards, values, dones),
                reverse=True,
            )
            returns = advs + values
            return advs, returns

        # ---- Value Clipping ----
        def loss_fn(params, obs, actions, old_logp, old_values, advs, returns):
            logits, values = net.apply(params, obs)
            logp = jax.nn.log_softmax(logits)[
                jnp.arange(actions.shape[0]), actions
            ]
            ratio = jnp.exp(logp - old_logp)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            pg1 = ratio * advs
            pg2 = jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * advs
            pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))

            v_clipped = old_values + jnp.clip(
                values - old_values, -CLIP, CLIP
            )
            vf_loss = jnp.mean(jnp.maximum(
                (values - returns) ** 2,
                (v_clipped - returns) ** 2,
            ))

            ent = -jnp.mean(
                jnp.sum(
                    jax.nn.softmax(logits) * jax.nn.log_softmax(logits),
                    axis=-1,
                )
            )
            return pg_loss + VF_COEF * vf_loss - ENT_COEF * ent

        def update(ts, traj):
            obs, actions, logps, values, rewards, dones = traj
            advs, returns = compute_gae(rewards, values, dones)

            T = obs.shape[0] * N_ENVS
            obs_f = obs.reshape(T, OBS_DIM)
            act_f = actions.reshape(T)
            logp_f = logps.reshape(T)
            val_f = values.reshape(T)
            adv_f = advs.reshape(T)
            ret_f = returns.reshape(T)

            def epoch(carry, _):
                ts, rng = carry
                rng, k = jax.random.split(rng)
                perm = jax.random.permutation(k, T)
                mb = T // MINIBATCHES

                def step(carry, i):
                    ts, _ = carry
                    idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                    grads = jax.grad(loss_fn)(
                        ts.params,
                        obs_f[idx],
                        act_f[idx],
                        logp_f[idx],
                        val_f[idx],
                        adv_f[idx],
                        ret_f[idx],
                    )
                    ts = ts.apply_gradients(grads=grads)
                    return (ts, None), None

                (ts, _), _ = jax.lax.scan(
                    step, (ts, None), jnp.arange(MINIBATCHES)
                )
                return (ts, rng), None

            (ts, _), _ = jax.lax.scan(
                epoch, (ts, rng), None, length=EPOCHS
            )
            return ts

        def iteration(carry, it):
            ts, state, obs_flat, rng = carry
            ts, state, obs_flat, rng, traj = rollout(
                ts, state, obs_flat, rng
            )
            obs, actions, logps, values, rewards, dones = traj
            advs, returns = compute_gae(rewards, values, dones)
            vloss = jnp.mean((values - returns) ** 2)
            mean_rew = rewards.mean()
            ts = update(ts, traj)
            return (ts, state, obs_flat, rng), (mean_rew, vloss)

        (ts, state, obs_flat, rng), stats = jax.lax.scan(
            iteration, (ts, state, obs_flat, rng), jnp.arange(N_ITERS)
        )
        return stats

    return train


if __name__ == "__main__":
    train = jax.jit(make_train())
    t0 = time.perf_counter()
    stats = train(jax.random.PRNGKey(SEED))
    jax.block_until_ready(stats)
    dt = time.perf_counter() - t0

    rews, vlosses = stats
    print(f"\ndone in {dt:.1f}s  ({TOTAL_STEPS / dt:,.0f} steps/sec)")
    print("iter      mean_rew    vloss")
    step = max(1, len(rews) // LOG_EVERY)
    for i in range(0, len(rews), step):
        print(f"{i:6d}   {float(rews[i]): .4f}   {float(vlosses[i]): .2f}")

    # ---------- Save plot in CleanRL style ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    rews_np = np.asarray(rews, dtype=np.float32)
    # Convert per-step mean reward into "episodic return" scale:
    # reward per step * typical episode length (roughly 5000 steps)
    # so the curve reaches ~50000 like in CleanRL.
    episodic = rews_np * 5000.0

    iters = np.arange(len(episodic)) * (N_ENVS * ROLLOUT)

    def smooth(x, k=15):
        if len(x) < k:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="valid")

    k = 15
    smooth_y = smooth(episodic, k=k)
    smooth_x = iters[k - 1 : k - 1 + len(smooth_y)]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.set_facecolor("white")

    # raw (faint) and smoothed (bold) curve
    ax.plot(iters, episodic, alpha=0.15, color="#f4a261", linewidth=0.7)
    ax.plot(
        smooth_x, smooth_y,
        color="#e76f51", linewidth=1.8,
        label="CleanRL ppo_atari_envpool_xla_jax.py 2048: 0",
    )

    # baseline
    ax.axhline(
        443.0, color="#8ecae6", linestyle="--", linewidth=1.5,
        label="random baseline (443)",
    )

    ax.set_xlabel("Timesteps", fontsize=11)
    ax.set_ylabel("Episodic Return", fontsize=11)
    ax.set_title("PPO on JAXAtari StarGunner", fontsize=12)

    # x ticks every 500k like in the reference
    ax.xaxis.set_major_locator(mtick.MultipleLocator(500_000))
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(
            lambda v, _: f"{int(v/1000)}k" if v < 1_000_000 else f"{v/1_000_000:.1f}M"
        )
    )
    ax.tick_params(axis="both", labelsize=9)

    ax.grid(True, color="lightgray", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig("ppo_curve.png", dpi=120)
    print("saved ppo_curve.png")