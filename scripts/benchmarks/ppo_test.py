"""
Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
"""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from functools import partial
from typing import Any, Sequence, NamedTuple
import os

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper, FlattenObservationWrapper, LogWrapper, ObjectCentricWrapper, NormalizeObservationWrapper
import hydra
from omegaconf import OmegaConf


import jaxatari
import wandb

from train_utils import video_callback, load_params


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_test(config, save_params):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env = jaxatari.make(config["ENV_NAME"].lower())
    mod_env = env
    if config.get("MOD_NAME", None) is not None:
        mod_env = jaxatari.modify(env, config.get("ENV_NAME", None).lower(), config.get("MOD_NAME", None).lower())
    renderer = jaxatari.make_renderer(config["ENV_NAME"].lower())

    def apply_wrappers(env):
        env = AtariWrapper(env, episodic_life=True, frame_skip=4, frame_stack_size=4, sticky_actions=True, max_pooling=True, clip_reward=True, noop_reset=30)
        if config.get("OBJECT_CENTRIC", False):
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
        else:
            env = PixelObsWrapper(env)
        env = NormalizeObservationWrapper(env)
        env = LogWrapper(env)
        return env

    env = apply_wrappers(env)
    mod_env = apply_wrappers(mod_env) 

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def test(rng):

        network = ActorCritic(
            env.action_space().n, activation=config["ACTIVATION"]
        )
        original_rng = rng
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        network_params = jax.tree.map(lambda x: x.squeeze(), save_params)
        # Not sure why this is necessary...
        for param in network_params:
            network_params[param]["Dense_5"]["kernel"] = jnp.expand_dims(network_params[param]["Dense_5"]["kernel"], axis=-1)
            network_params[param]["Dense_5"]["bias"] = jnp.expand_dims(network_params[param]["Dense_5"]["bias"], axis=-1)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        @partial(jax.jit, static_argnums=(1,))
        def get_test_metrics(train_state, mod, rng):

            def _env_step(carry, _):
                train_state, env_state, last_obs, rng = carry

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                if mod:
                    new_obs, new_env_state, reward, done, info = jax.vmap(mod_env.step)(env_state, action)
                else:
                    new_obs, new_env_state, reward, done, info = jax.vmap(env.step)(env_state, action)

                env_state_vid = jax.tree.map(lambda x: x[0], new_env_state)
                dones_vid = jax.tree.map(lambda x: x[0], done)
                carry = (train_state, new_env_state, new_obs, rng)
                return carry, (info, env_state_vid, dones_vid)

            rng, _rng = jax.random.split(rng)
            reset_keys = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            if mod:
                init_obs, env_state = jax.vmap(mod_env.reset)(reset_keys)
            else:
                init_obs, env_state = jax.vmap(env.reset)(reset_keys)

            _, output = jax.lax.scan(
                _env_step, (train_state, env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            infos, env_states, dones = output[0], output[1], output[2]

            if config.get("RECORD_VIDEO", False):
                jax.debug.callback(video_callback, env_states, dones, 0, renderer, mod=mod),

            # return mean of done infos
            done_infos = jax.tree_util.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x.squeeze(),
                        jnp.nan,
                    )
                ),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, False, _rng)

        mod_metrics = get_test_metrics(train_state, True, _rng) if config.get("MOD_NAME", None) is not None else {}

        return {"test_metrics": test_metrics, "mod_metrics": mod_metrics}

    return test

def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]
    oc = "oc" if config.get("OBJECT_CENTRIC", False) else "pixel"

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    save_dir = os.path.join(config["SAVE_PATH"], env_name)
    train_state_params = []
    for i, rng in enumerate(rngs):
        save_path = os.path.join(
            save_dir,
            f'{alg_name}_{env_name}_{oc}_seed{config["SEED"]}_vmap{i}.safetensors',
        )
        if not os.path.exists(save_path):
            raise ValueError(f"Save path {save_path} does not exist!")

        params = load_params(save_path)  # test loading
        train_state_params.append(params)

    train_state_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *train_state_params
    )

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_TEST'),
        config=config,
        mode=config["WANDB_MODE"],
    )


    t0 = time.time()

    train_vjit = jax.jit(jax.vmap(make_test(config, train_state_params)))
    train_vjit.lower(rngs).compile()
    compile_time = time.time()
    print(f"Compile time: {compile_time - t0} seconds.")
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Run time: {time.time() - compile_time} seconds.")
    print(f"Total: {time.time()-t0} seconds.")
    avg_return = outs["test_metrics"]["returned_episode_returns"]
    avg_return_mod = outs["mod_metrics"]["returned_episode_returns"]
    avg_len = outs["test_metrics"]["returned_episode_lengths"]
    avg_len_mod = outs["mod_metrics"]["returned_episode_lengths"]
    print(f"Average return of default env: {avg_return}, length: {avg_len}.")
    if config.get("MOD_NAME", None) is not None:
        print(f"Average return of modified env ({config['MOD_NAME']}): {avg_return_mod}, length: {avg_len_mod}.")

    


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()