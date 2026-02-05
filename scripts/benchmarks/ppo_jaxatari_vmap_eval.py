from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment, JAXAtariAction
from jaxatari.wrappers import JaxatariWrapper

jaxatari_action_mapping = {
    JAXAtariAction.NOOP: "NOOP",
    JAXAtariAction.FIRE: "FIRE",
    JAXAtariAction.UP: "UP",
    JAXAtariAction.RIGHT: "RIGHT",
    JAXAtariAction.LEFT: "LEFT",
    JAXAtariAction.DOWN: "DOWN",
    JAXAtariAction.UPRIGHT: "UPRIGHT",
    JAXAtariAction.UPLEFT: "UPLEFT",
    JAXAtariAction.DOWNRIGHT: "DOWNRIGHT",
    JAXAtariAction.DOWNLEFT: "DOWNLEFT",
    JAXAtariAction.UPFIRE: "UPFIRE",
    JAXAtariAction.RIGHTFIRE: "RIGHTFIRE",
    JAXAtariAction.LEFTFIRE: "LEFTFIRE",
    JAXAtariAction.DOWNFIRE: "DOWNFIRE",
    JAXAtariAction.UPRIGHTFIRE: "UPRIGHTFIRE",
    JAXAtariAction.UPLEFTFIRE: "UPLEFTFIRE",
    JAXAtariAction.DOWNRIGHTFIRE: "DOWNRIGHTFIRE",
    JAXAtariAction.DOWNLEFTFIRE: "DOWNLEFTFIRE"
}

@flax.struct.dataclass
class EpisodeData:
    env_states: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    seed=1,
    data_path: str|None = None,
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
        next_obs, new_env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))

        # first_states = jax.tree.map(lambda x: x[0], env_state)
        # since the env is eval_env (without reward clipping and episodic life), we can just accumulate the rewards
        # return (next_obs, env_state, keys), (first_states, done, reward) 
        unpacked_states = jax.tree_util.tree_flatten(env_state, is_leaf=lambda x: not (hasattr(x, "env_state") or hasattr(x, "atari_state")))[0][0]
        episode_data = EpisodeData(
            env_states=unpacked_states, # old
            rewards=reward, # new
            dones=done, # new
            actions=actions, # old
        )
        return (next_obs, env_state, keys), episode_data#(first_states, done, reward) 

    # evaluate eval_episodes concurrently
    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    # _, (first_states, dones, rewards) = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=10_800)
    _, episode_data = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=10_800)

    rewards = episode_data.rewards  # shape: (time, eval_episodes)
    dones = episode_data.dones  # shape: (time, eval_episodes)

    # get first done index for each episode
    first_done = jnp.argmax(dones, axis=0)  # shape: (eval_episodes,)
    print("first done indices:", first_done)

    # Store results if wanted
    if data_path is not None:
        # get directory if it does not exist
        import os
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        import numpy as np
        action_str = [jaxatari_action_mapping[int(a)] for a in episode_data.actions.flatten()]
        action_str = np.array(action_str).reshape(episode_data.actions.shape)
        lives = episode_data.env_states.lives  # shape: (time, eval_episodes)
        #TODO: might need to vmap for rendering here
        # obs = env.render(episode_data.env_states)  # shape: (time, eval_episodes, H, W, C)
        obs = jax.vmap(jax.vmap(env.render))(episode_data.env_states)  # shape: (time, eval_episodes, H, W, C) 
        # need to flatten time * eval_episodes for saving

        # filter all the obs, rewards, dones, action_str, actions, lives after first done for each episode
        # loop over eval_episodes, store data until first done
        filtered_obs = []
        filtered_rewards = []
        filtered_dones = []
        filtered_action_str = []
        filtered_actions = []
        filtered_lives = []
        print("Before filtering: ", obs.shape)
        for i in range(eval_episodes):
            fd = first_done[i]
            filtered_obs.append(obs[:fd + 1, i, ...])
            print(i, first_done[i], obs[:fd + 1, i, ...].shape)
            filtered_rewards.append(rewards[:fd + 1, i])
            filtered_dones.append(dones[:fd + 1, i])
            filtered_action_str.append(action_str[:fd + 1, i])
            filtered_actions.append(episode_data.actions[:fd + 1, i])
            filtered_lives.append(lives[:fd + 1, i])

        np.savez_compressed(
            data_path,
            obs=np.array(jnp.concatenate(filtered_obs)).reshape((-1, ) + obs.shape[2:]),  # shape: (total_time, H, W, C)
            rewards=np.array(jnp.concatenate(filtered_rewards)).reshape((-1, )),
            dones=np.array(jnp.concatenate(filtered_dones)).reshape((-1, )),
            action_str=np.array(np.concatenate(filtered_action_str)).reshape((-1, )),
            actions=np.array(jnp.concatenate(filtered_actions)).reshape((-1, )),
            lives=np.array(jnp.concatenate(filtered_lives)).reshape((-1, )),
        )
        print(f"Saved evaluation data to {data_path}")
        print(f"obs shape: {np.array(jnp.concatenate(filtered_obs)).reshape((-1, ) + obs.shape[2:]).shape}")
    
    # obs shape: (time, eval_episodes, 1, H, W)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    # shift right by one timestep
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1,0),(0,0)), constant_values=0)
    rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(rewards, axis=0)  # shape: (eval_episodes,)

    first_env_state = jax.tree.map(lambda x: x[0], episode_data.env_states) 
    # first episode for video capture
    env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], first_env_state) 
    return episodic_returns, env_states_until_done