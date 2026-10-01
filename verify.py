import ale_py
ale_py.register_v5_envs()

import jax, jax.numpy as jnp
import gymnasium as gym
from jaxatari.games.jax_stargunner import JaxStarGunner

# --- 1. single-env jit ---
env = JaxStarGunner(start_in_play=True)
step = jax.jit(env.step)
obs, s = env.reset(jax.random.PRNGKey(0))
key = jax.random.PRNGKey(1)
for i in range(2000):
    key, k = jax.random.split(key)
    a = jax.random.randint(k, (), 0, 18)
    obs, s, r, d, i_info = step(s, a)
    if bool(d):
        obs, s = env.reset(jax.random.PRNGKey(i))
for leaf in jax.tree_util.tree_leaves(s):
    if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
        assert not jnp.any(jnp.isnan(leaf)), "NaN in state"
print("[1] jit OK   score:", int(s.score), "lives:", int(s.lives))

# --- 2. vmap ---
B = 8
step_b = jax.jit(jax.vmap(env.step))
obs, s = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), B))
for _ in range(1000):
    obs, s, r, d, i_info = step_b(s, jnp.zeros(B, jnp.int32))
assert not jnp.any(jnp.isnan(s.score))
print("[2] vmap OK   scores:", s.score)

# --- 3. action space matches ALE ---
ale = gym.make("ALE/StarGunner-v5")
n_ale = ale.action_space.n
n_jax = env.action_space().n
assert n_ale == n_jax == 18, (n_ale, n_jax)
print("[3] action space OK   ALE:", n_ale, "JAX:", n_jax)

# --- 4. reward == score delta ---
env2 = JaxStarGunner(start_in_play=True)
obs, s = env2.reset(jax.random.PRNGKey(0))
total_r = 0.0
for _ in range(3000):
    obs, s, r, d, info = env2.step(s, jnp.array(1, jnp.int32))
    total_r += float(r)
    if bool(d):
        break
assert abs(total_r - float(s.score)) < 1e-3, (total_r, int(s.score))
print("[4] reward OK   total:", total_r, "score:", int(s.score))

# --- 5. forced hit gives positive reward ---
obs, s = env2.reset(jax.random.PRNGKey(0))
for _ in range(10):
    obs, s, r, d, info = env2.step(s, jnp.array(1, jnp.int32))
eidx = int(jnp.argmax(s.enemy_alive))
b0 = int(jnp.argmax(s.bullet_active))
s = s.replace(
    enemy_type=s.enemy_type.at[eidx].set(1),
    enemy_x=s.enemy_x.at[eidx].set(s.bullet_x[b0]),
    enemy_y=s.enemy_y.at[eidx].set(s.bullet_y[b0]),
    enemy_vy=s.enemy_vy.at[eidx].set(0.0),
    bullet_vx=s.bullet_vx.at[b0].set(0.0),
    bullet_vy=s.bullet_vy.at[b0].set(0.0),
)
obs, s2, r, d, info = env2.step(s, jnp.array(1, jnp.int32))
assert float(r) > 0, "forced hit gave no reward"
print("[5] forced hit OK   reward:", float(r), "score:", int(s2.score))

print()
print("ALL TESTS PASSED")
