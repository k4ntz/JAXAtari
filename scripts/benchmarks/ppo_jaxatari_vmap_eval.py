import os
from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import numpy as np

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper

import cv2
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


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

    def wrapped_reset(key):
        """wrappes the reset function of the environment to correct the observation shape"""
        next_obs, state = env.reset(key)
        # NNs require shape (B, F, H, W), where B is the batch size and F is the frame stack size
        return next_obs.squeeze()[None, ...], state
    
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

    # a simple non-vectorized version

    episodic_returns = []
    for episode in range(eval_episodes):
        episodic_return = 0
        key, reset_key = jax.random.split(key)
        next_obs, handle = wrapped_reset(reset_key)
        terminated = False

        if capture_video:
            recorded_frames = []
            # conversion from grayscale into rgb
            recorded_frames.append(cv2.cvtColor(np.array(jax.device_get(next_obs[0][-1]))[..., None], cv2.COLOR_GRAY2RGB))
        while not terminated:
            actions, key = get_action_and_value(network_params, actor_params, next_obs, key)
            next_obs, handle, reward, done, infos = wrapped_step(handle, jnp.array(actions))
            episodic_return += reward
            terminated = done

            if capture_video and episode == 0:
                recorded_frames.append(cv2.cvtColor(np.array(jax.device_get(next_obs[0][-1]))[..., None], cv2.COLOR_GRAY2RGB))

            if terminated:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
                episodic_returns.append(episodic_return)
                if capture_video and episode == 0:
                    clip = ImageSequenceClip(recorded_frames, fps=24)
                    os.makedirs(f"videos/{run_name}", exist_ok=True)
                    clip.write_videofile(f"videos/{run_name}/{episode}.mp4", logger="bar")

    return episodic_returns