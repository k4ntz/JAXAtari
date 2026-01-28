"""
Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
"""

import os
import time
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any
import os

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper, FlattenObservationWrapper, LogWrapper, ObjectCentricWrapper, NormalizeObservationWrapper
import hydra
from omegaconf import OmegaConf

import jaxatari
import wandb

from train_utils import video_callback, load_params

class CNN(nn.Module):

    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x

class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 128
    num_layers: int = 2
    norm_type: str = "layer_norm"
    norm_input: bool = False
    object_centric: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        if self.object_centric:
            # Use MLP for object centric observations
            if self.norm_type == "layer_norm":
                normalize = lambda x: nn.LayerNorm()(x)
            elif self.norm_type == "batch_norm":
                normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
            else:
                normalize = lambda x: x

            for l in range(self.num_layers):
                x = nn.Dense(self.hidden_size)(x)
                x = normalize(x)
                x = nn.relu(x)
        else:
            # Use CNN for pixel based observations
            x = CNN(norm_type=self.norm_type)(x, train)

        x = nn.Dense(self.action_dim)(x)

        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_test(config, save_params, batch_stats):

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
    renderer = mod_env.renderer

    def apply_wrappers(env):
        env = AtariWrapper(env, episodic_life=True, frame_skip=4, frame_stack_size=4, sticky_actions=True, max_pooling=True, clip_reward=True, noop_reset=30)
        if config.get("OBJECT_CENTRIC", False):
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
        else:
            grayscale = config.get("PIXEL_GRAYSCALE", False)
            do_resize = config.get("PIXEL_RESIZE", True)
            resize_shape = config.get("PIXEL_RESIZE_SHAPE", [84, 84])
            use_native_downscaling = config.get("USE_NATIVE_DOWNSCALING", False)
            env = PixelObsWrapper(env, do_pixel_resize=do_resize, pixel_resize_shape=resize_shape, grayscale=grayscale, use_native_downscaling=use_native_downscaling)
        env = NormalizeObservationWrapper(env)
        env = LogWrapper(env)
        return env

    env = apply_wrappers(env)
    mod_env = apply_wrappers(mod_env)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    def test(rng):

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space().n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            object_centric=config.get("OBJECT_CENTRIC", True),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space().shape))
            network_variables = network.init(rng, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            params = network_variables["params"] if save_params is None else save_params
            batch_sts = network_variables["batch_stats"] if batch_stats is None else batch_stats
            batch_sts = jax.tree.map(lambda x: x.squeeze(), batch_sts)  # remove vmap dim
            params = jax.tree.map(lambda x: x.squeeze(), params)  # remove vmap dim
            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=params,
                batch_stats=batch_sts,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)
        
        @partial(jax.jit, static_argnums=(1,))
        def get_test_metrics(train_state, mod, rng):

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                if mod:
                    new_obs, new_env_state, reward, done, info = jax.vmap(mod_env.step)(env_state, action)
                else:
                    new_obs, new_env_state, reward, done, info = jax.vmap(env.step)(env_state, action)
                env_state_vid = jax.tree.map(lambda x: x[0], new_env_state)
                dones_vid = jax.tree.map(lambda x: x[0], done)
                return (new_env_state, new_obs, rng), (info, env_state_vid, dones_vid)

            rng, _rng = jax.random.split(rng)
            reset_keys = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            if mod:
                init_obs, env_state = jax.vmap(mod_env.reset)(reset_keys)
            else:
                init_obs, env_state = jax.vmap(env.reset)(reset_keys)

            _, output = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
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
    batch_stats = []
    for i, rng in enumerate(rngs):
        save_path = os.path.join(
            save_dir,
            f'{alg_name}_{env_name}_{oc}_seed{config["SEED"]}_vmap{i}.safetensors',
        )
        if not os.path.exists(save_path):
            raise ValueError(f"Save path {save_path} does not exist!")

        params = load_params(save_path)  # test loading
        batch_sts = load_params(save_path.replace(".safetensors", "_bs.safetensors"))
        train_state_params.append(params)
        batch_stats.append(batch_sts)
    train_state_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *train_state_params
    )
    batch_stats = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *batch_stats
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

    train_vjit = jax.jit(jax.vmap(make_test(config, train_state_params, batch_stats)))
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