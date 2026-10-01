import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

from jaxatari.games.jax_stargunner import JaxStarGunner

# ---------------- config ----------------
N_ENVS       = 128
ROLLOUT      = 256
TOTAL_STEPS  = 10_000_000
LR           = 3e-4
LR_WARMUP    = 1_000
GAMMA        = 0.99
LAMBDA       = 0.95
CLIP         = 0.2
ENT_COEF     = 0.005
VF_COEF      = 0.25
EPOCHS       = 4
MINIBATCHES  = 8
GRAD_CLIP    = 1.0          # <-- war 0.5, jetzt 1.0
REWARD_SCALE = 100.0        # <-- NEU: skaliert Rewards vor GAE
SEEDS        = [0, 1, 2]
LOG_EVERY    = 10
RANDOM_EVAL_STEPS = 20_000

FRAME_STACK = 4
# ----------------------------------------

env = JaxStarGunner(start_in_play=True)
N_ACTIONS = env.action_space().n
BATCH     = N_ENVS * ROLLOUT
N_ITERS   = TOTAL_STEPS // BATCH
MB_SIZE   = BATCH // MINIBATCHES
assert BATCH % MINIBATCHES == 0


# ------------------------------------------------------------------
# Observation: alive-flags + state-extras + frame-stack
# ------------------------------------------------------------------
def _flat_single(obs, state):
    parts = []
    for name in ("player", "enemies", "bullets", "bombs", "bobo"):
        o = getattr(obs, name)
        x_flat = jnp.atleast_1d(o.x).ravel()
        y_flat = jnp.atleast_1d(o.y).ravel()
        w_flat = jnp.atleast_1d(o.width).ravel()
        h_flat = jnp.atleast_1d(o.height).ravel()

        alive = (x_flat >= 0).astype(jnp.float32)
        x = jnp.where(alive > 0, x_flat, 0.0) / 160.0
        y = jnp.where(alive > 0, y_flat, 0.0) / 210.0

        parts += [alive, x, y, w_flat / 16.0, h_flat / 16.0]

    extras = jnp.concatenate([
        jnp.atleast_1d(state.player_facing.astype(jnp.float32)).ravel(),
        jnp.atleast_1d((state.invuln_timer > 0).astype(jnp.float32)).ravel(),
        jnp.atleast_1d(state.respawn_timer.astype(jnp.float32) / 90.0).ravel(),
        jnp.atleast_1d(state.lives.astype(jnp.float32) / 5.0).ravel(),
        jnp.atleast_1d(state.level.astype(jnp.float32) / 10.0).ravel(),
        jnp.atleast_1d(state.subwave.astype(jnp.float32) / 3.0).ravel(),
    ]).astype(jnp.float32)
    parts.append(extras)

    return jnp.concatenate(parts).astype(jnp.float32)


def init_stack(obs, state):
    single = _flat_single(obs, state)
    return jnp.tile(single[None, :], (FRAME_STACK, 1))


def push_stack(stack, obs, state):
    single = _flat_single(obs, state)
    return jnp.concatenate([stack[1:], single[None]], axis=0)


_obs0, _state0 = env.reset(jax.random.PRNGKey(0))
SINGLE_DIM = _flat_single(_obs0, _state0).shape[0]
OBS_DIM = SINGLE_DIM * FRAME_STACK
print(f"n_actions={N_ACTIONS}  single_dim={SINGLE_DIM}  "
      f"obs_dim={OBS_DIM}  iters={N_ITERS}  "
      f"actual_steps={N_ITERS * BATCH:,}")


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------
class ActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.LayerNorm()(nn.Dense(256)(x)))
        h = nn.relu(nn.LayerNorm()(nn.Dense(256)(h)))
        logits = nn.Dense(
            self.n_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(h)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(h).squeeze(-1)
        return logits, value


# ------------------------------------------------------------------
# Env helpers
# ------------------------------------------------------------------
def env_step_reset(state, action, reset_key):
    obs, new_state, reward, done, _ = env.step(state, action)

    terminated = done & (new_state.lives <= 0)
    truncated = done & (new_state.lives > 0)

    def do_reset(_):
        return env.reset(reset_key)

    def keep(_):
        return obs, new_state

    reset_obs, reset_state = jax.lax.cond(done, do_reset, keep, operand=None)
    return reset_obs, reset_state, reward, terminated, truncated


v_env_step_reset = jax.vmap(env_step_reset, in_axes=(0, 0, 0))
v_init_stack = jax.vmap(init_stack)
v_push_stack = jax.vmap(push_stack)


def track_episodes(ep_ret, rew, done):
    ep_ret = ep_ret + rew
    done_f = done.astype(jnp.float32)
    finished = ep_ret * done_f
    ep_ret = ep_ret * (1.0 - done_f)
    return ep_ret, finished, done_f


# ------------------------------------------------------------------
# PPO
# ------------------------------------------------------------------
def make_train():
    net = ActorCritic(N_ACTIONS)

    def train(rng):
        rng, k = jax.random.split(rng)
        keys = jax.random.split(k, N_ENVS)
        obs, state = jax.vmap(env.reset)(keys)
        stacks = v_init_stack(obs, state)
        ep_ret = jnp.zeros(N_ENVS, jnp.float32)

        rng, k = jax.random.split(rng)
        params = net.init(k, stacks[0].reshape(-1))

        total_grad_steps = N_ITERS * EPOCHS * MINIBATCHES

        warmup = optax.linear_schedule(
            init_value=1e-5,
            end_value=LR,
            transition_steps=max(LR_WARMUP, 1),
        )
        decay = optax.linear_schedule(
            init_value=LR,
            end_value=0.0,
            transition_steps=max(total_grad_steps - LR_WARMUP, 1),
        )
        lr_schedule = optax.join_schedules(
            schedules=[warmup, decay],
            boundaries=[LR_WARMUP],
        )

        tx = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.adam(lr_schedule, eps=1e-5),
        )
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        def policy_step(params, stack, key):
            flat = stack.reshape(-1)
            logits, value = net.apply(params, flat)
            action = jax.random.categorical(key, logits)
            logp = jax.nn.log_softmax(logits)[action]
            return action, logp, value

        v_policy = jax.vmap(policy_step, in_axes=(None, 0, 0))

        def rollout(ts, state, stacks, ep_ret, rng):
            def body(carry, _):
                state, stacks, ep_ret, rng = carry
                rng, k1, k2 = jax.random.split(rng, 3)
                act_keys = jax.random.split(k1, N_ENVS)
                reset_keys = jax.random.split(k2, N_ENVS)

                stacks_flat = stacks.reshape(N_ENVS, -1)
                actions, logps, values = v_policy(ts.params, stacks, act_keys)

                obs, new_state, rew, terminated, truncated = v_env_step_reset(
                    state, actions, reset_keys
                )
                done = terminated | truncated

                pushed = v_push_stack(stacks, obs, new_state)
                fresh = v_init_stack(obs, new_state)
                new_stacks = jnp.where(done[:, None, None], fresh, pushed)

                ep_ret, finished, done_f = track_episodes(ep_ret, rew, done)

                out = (stacks_flat, actions, logps, values, rew,
                       terminated, truncated, finished, done_f)
                return (new_state, new_stacks, ep_ret, rng), out

            (state, stacks, ep_ret, rng), traj = jax.lax.scan(
                body, (state, stacks, ep_ret, rng), None, length=ROLLOUT
            )
            return state, stacks, ep_ret, traj

        def compute_gae(rewards, values, terminated, last_value):
            # Skalierung der Rewards: bringt Value-Targets in einen
            # Bereich, in dem der Value-Head tatsächlich lernt.
            rewards = rewards / REWARD_SCALE

            def body(carry, x):
                gae, next_value = carry
                r, v, d = x
                d = d.astype(jnp.float32)
                delta = r + GAMMA * next_value * (1.0 - d) - v
                gae = delta + GAMMA * LAMBDA * (1.0 - d) * gae
                return (gae, v), gae

            _, advs = jax.lax.scan(
                body,
                (jnp.zeros(N_ENVS), last_value),
                (rewards, values, terminated),
                reverse=True,
            )
            return advs, advs + values

        def loss_fn(params, obs, actions, old_logp, old_values, advs, returns):
            logits, values = net.apply(params, obs)
            log_probs = jax.nn.log_softmax(logits)
            logp = log_probs[jnp.arange(actions.shape[0]), actions]
            log_ratio = logp - old_logp
            ratio = jnp.exp(log_ratio)

            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            pg_loss = -jnp.mean(jnp.minimum(
                ratio * advs,
                jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * advs,
            ))

            v_clipped = old_values + jnp.clip(values - old_values, -CLIP, CLIP)
            vf_loss = 0.5 * jnp.mean(jnp.maximum(
                (values - returns) ** 2,
                (v_clipped - returns) ** 2,
            ))

            ent = -jnp.mean(jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1))
            loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * ent

            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > CLIP).astype(jnp.float32))
            return loss, (pg_loss, vf_loss, ent, approx_kl, clipfrac)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def update(ts, obs, actions, logps, values, advs, returns, rng):
            obs_f = obs.reshape(BATCH, OBS_DIM)
            act_f = actions.reshape(BATCH)
            logp_f = logps.reshape(BATCH)
            val_f = values.reshape(BATCH)
            adv_f = advs.reshape(BATCH)
            ret_f = returns.reshape(BATCH)

            def epoch(ts, k):
                perm = jax.random.permutation(k, BATCH).reshape(MINIBATCHES, MB_SIZE)

                def mb_step(ts, idx):
                    (_, aux), grads = grad_fn(
                        ts.params, obs_f[idx], act_f[idx], logp_f[idx],
                        val_f[idx], adv_f[idx], ret_f[idx],
                    )
                    return ts.apply_gradients(grads=grads), aux

                return jax.lax.scan(mb_step, ts, perm)

            keys = jax.random.split(rng, EPOCHS)
            ts, aux = jax.lax.scan(epoch, ts, keys)
            return ts, jax.tree_util.tree_map(jnp.mean, aux)

        def iteration(carry, _):
            ts, state, stacks, ep_ret, rng = carry
            rng, k_roll, k_upd = jax.random.split(rng, 3)

            state, stacks, ep_ret, traj = rollout(ts, state, stacks, ep_ret, k_roll)
            (obs, actions, logps, values, rewards, terminated, truncated,
             finished, done_f) = traj

            _, last_value = net.apply(ts.params, stacks.reshape(N_ENVS, -1))
            advs, returns = compute_gae(rewards, values, terminated, last_value)

            ts, (pg, vf, ent, kl, clipfrac) = update(
                ts, obs, actions, logps, values, advs, returns, k_upd
            )

            n_eps = done_f.sum()
            mean_ep_ret = jnp.where(
                n_eps > 0, finished.sum() / jnp.maximum(n_eps, 1.0), jnp.nan
            )
            ret_f = returns.reshape(-1)
            val_f = values.reshape(-1)
            ev = 1.0 - jnp.var(ret_f - val_f) / (jnp.var(ret_f) + 1e-8)

            stats = dict(
                ep_ret=mean_ep_ret,
                n_eps=n_eps,
                mean_rew=rewards.mean(),
                vf=vf, pg=pg, ent=ent, kl=kl, clipfrac=clipfrac,
                expl_var=ev,
                trunc_frac=truncated.mean(),
            )
            return (ts, state, stacks, ep_ret, rng), stats

        _, stats = jax.lax.scan(
            iteration, (ts, state, stacks, ep_ret, rng), None, length=N_ITERS
        )
        return stats

    return train


# ------------------------------------------------------------------
# Random baseline
# ------------------------------------------------------------------
def make_random_eval():
    def run(rng):
        rng, k = jax.random.split(rng)
        obs, state = jax.vmap(env.reset)(jax.random.split(k, N_ENVS))
        ep_ret = jnp.zeros(N_ENVS, jnp.float32)

        def body(carry, _):
            state, ep_ret, rng = carry
            rng, k1, k2 = jax.random.split(rng, 3)
            actions = jax.random.randint(k1, (N_ENVS,), 0, N_ACTIONS)
            reset_keys = jax.random.split(k2, N_ENVS)
            _, state, rew, terminated, truncated = v_env_step_reset(
                state, actions, reset_keys
            )
            done = terminated | truncated
            ep_ret, finished, done_f = track_episodes(ep_ret, rew, done)
            return (state, ep_ret, rng), (rew, finished, done_f)

        _, (rew, finished, done_f) = jax.lax.scan(
            body, (state, ep_ret, rng), None, length=RANDOM_EVAL_STEPS
        )
        return (rew.mean(),
                finished.sum() / jnp.maximum(done_f.sum(), 1.0),
                done_f.sum())

    return run


def ffill(x):
    x = x.copy()
    last = np.nan
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = last
        else:
            last = x[i]
    return x


if __name__ == "__main__":
    rand_run = jax.jit(make_random_eval())
    r_step, r_ep, r_n = rand_run(jax.random.PRNGKey(123))
    r_step, r_ep, r_n = float(r_step), float(r_ep), int(r_n)
    print(f"random baseline: {r_step:.4f} reward/step, "
          f"{r_ep:.1f} episodic return over {r_n} episodes")

    train = jax.jit(make_train())
    compiled = train.lower(jax.random.PRNGKey(0)).compile()

    all_stats = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        stats = compiled(jax.random.PRNGKey(seed))
        jax.block_until_ready(stats)
        dt = time.perf_counter() - t0
        stats = {k: np.asarray(v) for k, v in stats.items()}
        all_stats.append(stats)

        print(f"\nseed {seed}: {dt:.1f}s  "
              f"({N_ITERS * BATCH / dt:,.0f} steps/sec, compile excluded)")
        print(" iter    ep_ret   rew/step    vf_loss       kl  clipfrac  expl_var  trunc")
        step = max(1, N_ITERS // LOG_EVERY)
        for i in range(0, N_ITERS, step):
            print(f"{i:5d}  "
                  f"{stats['ep_ret'][i]:8.1f}  "
                  f"{stats['mean_rew'][i]:9.4f}  "
                  f"{stats['vf'][i]:9.4f}  "
                  f"{stats['kl'][i]:7.4f}  "
                  f"{stats['clipfrac'][i]:8.3f}  "
                  f"{stats['expl_var'][i]:8.3f}  "
                  f"{stats['trunc_frac'][i]:5.3f}")

    LAST = 50
    finals_ep = [np.nanmean(s["ep_ret"][-LAST:]) for s in all_stats]
    finals_rw = [float(np.mean(s["mean_rew"][-LAST:])) for s in all_stats]
    print(f"\nFinal (last {LAST} iters, {len(SEEDS)} seeds):")
    print(f"  episodic return : {np.mean(finals_ep):.1f} +/- "
          f"{np.std(finals_ep):.1f}   (random: {r_ep:.1f})")
    print(f"  reward per step : {np.mean(finals_rw):.3f} +/- "
          f"{np.std(finals_rw):.3f}   (random: {r_step:.3f})")

    np.savez(
        "ppo_stargunner_results.npz",
        **{f"{k}_seed{sd}": v
           for sd, s in zip(SEEDS, all_stats) for k, v in s.items()},
        random_step=r_step, random_ep=r_ep,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    x = (np.arange(N_ITERS) + 1) * BATCH

    def smooth(y, k=25):
        kernel = np.ones(k) / k
        return np.convolve(y, kernel, mode="valid")

    K = 25
    curves = np.stack([smooth(ffill(s["ep_ret"]), K) for s in all_stats])
    xs = x[K - 1:]
    mean, std = curves.mean(0), curves.std(0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(xs, mean, color="#e76f51", linewidth=1.8,
            label=f"PPO (mean of {len(SEEDS)} seeds)")
    ax.fill_between(xs, mean - std, mean + std, color="#e76f51", alpha=0.2)
    ax.axhline(r_ep, color="#8ecae6", linestyle="--", linewidth=1.5,
               label=f"random policy ({r_ep:.0f})")
    ax.set_xlabel("Environment steps (agent steps)")
    ax.set_ylabel("Episodic return")
    ax.set_title("PPO on JAXAtari StarGunner")
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")
    )
    ax.grid(True, color="lightgray", alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig("ppo_curve.png", dpi=120)
    print("saved ppo_curve.png, ppo_stargunner_results.npz")