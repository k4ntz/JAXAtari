from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import EnvState, JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    seed=1,
):
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id, seed, 1)()
    _Network, _Actor, _Critic = Model
    key = jax.random.key(seed)

    @jax.jit
    def wrapped_reset(key):
        """wrappes the reset function of the environment to correct the observation shape"""
        next_obs, state = env.reset(key)
        # NNs require shape (B, F, H, W), where B is the batch size and F is the frame stack size
        return next_obs.squeeze()[None, ...], state

    @jax.jit 
    def wrapped_step(state, action):
        """wrappes the step function of the environment to correct the observation shape"""
        next_obs, next_state, reward, done, info =  env.step(state, action.squeeze())
        # NNs require shape (B, F, H, W), where B is the batch size and F is the frame stack size
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    key, reset_key = jax.random.split(key)
    next_obs, handle = wrapped_reset(reset_key)
    network = _Network()
    actor = _Actor(action_dim=env.action_space().n)
    critic = _Critic()
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, network_key_2, actor_key_2, critic_key_2 = jax.random.split(key, 4)
    network_params = network.init(network_key, env.observation_space().sample(network_key_2).squeeze()[None, ...])
    actor_params = actor.init(actor_key, network.apply(network_params, env.observation_space().sample(actor_key_2).squeeze()[None, ...]))
    critic_params = critic.init(critic_key, network.apply(network_params, env.observation_space().sample(critic_key_2).squeeze()[None, ...]))
    # note: critic_params is not used in this script
    with open(model_path, "rb") as f:
        (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    @jax.jit
    def get_action_and_value(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    def step_fn(carry, input):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(get_action_and_value, in_axes=(None, None, 0, 0))(network_params, actor_params, next_obs, keys)
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))
        first_states = jax.tree.map(lambda x: x[0], env_state)
        # since the env is eval_env (without reward clipping and episodic life), we can just accumulate the rewards
        return (next_obs, env_state, keys), (first_states, done, reward) 

    # evaluate eval_episodes concurrently
    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (first_states, dones, rewards) = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=10_000)

    print("scanned rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))
    
    # obs shape: (time, eval_episodes, 1, H, W)
    first_done = jnp.argmax(dones, axis=0)  # shape: (eval_episodes,)
    # print("first dones: ", first_done)
    # reward_mask = jnp.arange(dones.shape[0])[:, None] <= first_done[None, :]  # shape: (time, eval_episodes)
    # rewards = rewards * reward_mask  # shape: (time, eval_episodes)
    # print("masked rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    # shift right by one timestep
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1,0),(0,0)), constant_values=0)
    # masked_rewards = rewards * (1 - mask_after_first_done)
    rewards = rewards * (1 - mask_after_first_done)
    print("filtered rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))

    episodic_returns = jnp.sum(rewards, axis=0)  # shape: (eval_episodes,)

    env_state = jax.tree_util.tree_flatten(first_states, is_leaf=lambda x: not (hasattr(x, "env_state") or hasattr(x, "atari_state")))[0][0]

    # first episode video capture
    # states_until_done = first_obs[:first_done[0] + 1, 0]  # shape: (time_until_done, 1, H, W)
    # env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], first_states.atari_state.atari_state.env_state)
    env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], env_state) 

    return episodic_returns, env_states_until_done