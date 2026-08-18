# Taken entirely from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
# Adapted to JaxAtari

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffers="
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import random
import time
from functools import partial
from typing import Sequence, NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxatari
import hydra
from omegaconf import OmegaConf
from jaxatari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxatari import spaces
from ppo_jaxatari_vmap_eval import evaluate

from rtpt import RTPT



def make_env(env_id, seed, num_envs, mods=[], pixel_based=True, native_downscaling=True, pixel_resize_shape=(84, 84), smooth_image=True, frame_stack=4, eval=False):
    def thunk():
        # For training (eval=False), avoid applying multiple potentially conflicting
        # mods at once. In that case, fall back to the base environment.
        # For evaluation (eval=True), we trust the caller to pass either a single
        # mod or an explicit list; this is used in the per-mod video generation.
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        # Normalize to None or list for jaxatari.make
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval, # only active during training 
                first_fire=True,
                noop_max=30,
                full_action_space=True, # use full action space (18 actions) to have consistent Actor output across tasks
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=pixel_resize_shape,
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=smooth_image,
                frame_stack_size=frame_stack,
                frame_skip=4,
                max_pooling=True,
                clip_reward=True, # only active during training
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=frame_stack,
                        frame_skip=4,
                        clip_reward=True,
                    )
                )
            )
        env = LogWrapper(env)
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.is_vector_env = True
        return env
    return thunk


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x

class MLP_Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        # 1. Hidden Layer
        x = nn.Dense(
            461,  # Hidden size H=461
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        # 2. Output Layer (matches the last layer of the CNN)
        x = nn.Dense(
            512,  # Output size
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)  # The CNN's last layer also has a ReLU
        return x

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class AgentParams(NamedTuple):
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def single_run(key: jax.random.PRNGKey, network: Network | MLP_Network, actor: Actor, critic: Critic, config: dict, task_id: str, agent_state: TrainState, ewc_fisher = None, ewc_params = None, max_D = None, global_step = None, global_iteration = None, global_start_time = None, seen_tasks = None, train_mods = [], eval_mods = []):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    config["TRAIN_MODS"] = train_mods
    config["EVAL_MODS"] = eval_mods

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    config["ENV_ID"] = task_id

    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'

    # env setup
    env = make_env(
            env_id=config["ENV_ID"], 
            seed=config["SEED"], 
            num_envs=config["NUM_ENVS"], 
            mods=list(config["TRAIN_MODS"]), 
            pixel_based=config["PIXEL_BASED"], 
            native_downscaling=config["NATIVE_DOWNSCALING"], 
            pixel_resize_shape=tuple(config["PIXEL_RESIZE_SHAPE"]), 
            smooth_image=config["SMOOTH_IMAGE"], 
            frame_stack=config["FRAME_STACK"]
        )()
    padding_width = max_D - env.observation_space().shape[-1] if not config["PIXEL_BASED"] else 0
    def pad_obs(obs):
        """Zero-pad object-centric observations"""
        return jnp.pad(obs, ((0, 0), (0, padding_width)))

    # vmap and squeeze observations in order to get (B, F, H, W, 1) -> (B, F, H, W),
    # where F is the frame stack which becomes the channel for the convolutions
    @jax.jit
    def wrapped_reset(key):
        obs, state = jax.vmap(env.reset)(key)
        
        if config["PIXEL_BASED"]:
            obs = obs.squeeze(-1)   # (B, F, H, W, 1) -> (B, F, H, W)
        else:
            # pad observations for object-centric environments to have consistent shape across tasks
            obs = pad_obs(obs)
        
        return obs, state
    
    @jax.jit
    def wrapped_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)

        if config["PIXEL_BASED"]:
            next_obs = next_obs.squeeze(-1)   # (B, F, H, W, 1) -> (B, F, H, W)
        else:
            # pad observations for object-centric environments to have consistent shape across tasks
            next_obs = pad_obs(next_obs)
        
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs, state, reward, next_done, info

    vmap_reset = wrapped_reset
    vmap_step = wrapped_step
    
    assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze(-1)
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"])

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze(-1)

        advantages = jnp.zeros((config["NUM_ENVS"],))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ewc_penalty(params, ewc_params, ewc_fisher):
        """
        Compute the EWC quadratic penalty.

        ewc_fisher pytree should be initialized with zeros for the first task.
        """

        def param_penalty(p, p_star, f):
            # returns the penalty (scalar) for one pytree leaf (array)
            return (f * (p - p_star) ** 2).sum()
        
        # params also includes the critic; we only want to restrict the network and actor params. 
        # no problem, because ewc_fisher values for all critic params are zero anyways
        penalties = jax.tree.map(param_penalty, params, ewc_params, ewc_fisher) # pytree with shape of params pytree, but leaves are scalars instead of arrays
        # TODO: (?) add normalization so the penalty does not grow with the number of parameters
        return 0.5 * sum(jax.tree.leaves(penalties))    # sum over scalar leaves to get a single scalar penalty

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, ewc_params, ewc_fisher):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if config["NORM_ADV"]:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - config["CLIP_COEF"], 1 + config["CLIP_COEF"])
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()

        ewc_loss = ewc_penalty(params, ewc_params, ewc_fisher)

        loss = (
            pg_loss
            - config["ENT_COEF"] * entropy_loss
            + v_loss * config["VF_COEF"]
            + config.get("EWC_LAMBDA", 0.0) * ewc_loss
        )
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
        ewc_params,
        ewc_fisher,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    ewc_params,
                    ewc_fisher,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=config["UPDATE_EPOCHS"]
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key
    
    def eval_and_vid(iteration, current_global_step):
        model_path = f'runs/{run_name}/{config["EXP_NAME"]}_{iteration}.cleanrl_model'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        config,
                        [
                            agent_state.params.network_params,
                            agent_state.params.actor_params,
                            agent_state.params.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

        for eval_task, eval_task_train_mods, eval_task_eval_mods in seen_tasks:
            # We only want to log videos for the CURRENT task to save time/space
            capture_video = config["CAPTURE_VIDEO"] and (eval_task == task_id) and (eval_task_train_mods == list(config["TRAIN_MODS"]))

            # Object centric: recalculate the padding for every eval task using the fully wrapped environment
            current_padding = 0
            if not config["PIXEL_BASED"]:
                temp_env = make_env(
                    env_id=eval_task, 
                    seed=config["SEED"], 
                    num_envs=1, 
                    mods=eval_task_eval_mods, 
                    pixel_based=config["PIXEL_BASED"], 
                    native_downscaling=config["NATIVE_DOWNSCALING"], 
                    pixel_resize_shape=tuple(config["PIXEL_RESIZE_SHAPE"]), 
                    smooth_image=config["SMOOTH_IMAGE"], 
                    frame_stack=config["FRAME_STACK"],
                    eval=True
                )()
                task_D = temp_env.observation_space().shape[0]
                current_padding = max_D - task_D

            episodic_returns, env_states = evaluate(
                model_path,
                partial(
                    make_env,
                    mods=list(eval_task_eval_mods),
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config["NATIVE_DOWNSCALING"],
                    smooth_image=config["SMOOTH_IMAGE"],
                    eval=True,
                ),
                eval_task,
                eval_episodes=10,
                run_name=f"{run_name}-{eval_task}-{str(list(eval_task_eval_mods))}-eval",
                Model=(Network, Actor, Critic) if config["PIXEL_BASED"] else (MLP_Network, Actor, Critic),
                seed=config["SEED"],
                object_centric=not config["PIXEL_BASED"],
                padding_width=current_padding,
                crl=True
            )
            wandb.log({f"{eval_task}+{str(list(eval_task_train_mods))}/eval/episodic_return_mod": np.mean(jax.device_get(episodic_returns))}, step=current_global_step)

            if capture_video: 
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxatari.make(eval_task).renderer
                # Mirror the pqn_agent final video behavior: log a video for the
                # current eval environment under a consistent wandb key.
                # env_state arrays have shape (N,)
                frames = jax.vmap(clean_renderer.render)(env_states)
                # shape: (N, H, W, C) -> (N, C, H, W)
                frames = jnp.transpose(frames, (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"{eval_task}+{str(list(eval_task_train_mods))}/eval/video": video,}, step=current_global_step)
                print(f"Video ({eval_task}+{str(list(eval_task_train_mods))}/eval) logged to wandb with {frames.shape[0]} frames.")

    def _generate_single_final_video(mods_config, video_label, video_index=0, current_global_step=None):
        """Generate a single video for the given mod configuration and log it to wandb.
        """
        if not config["CAPTURE_VIDEO"]:
            return None

        # Apply wrappers just for the agent's observations, like in pqn_agent.
        fake_env = jaxatari.make(config["ENV_ID"])
        renderer_local = fake_env.renderer
        env = make_env(
            env_id=config["ENV_ID"], 
            seed=config["SEED"], 
            num_envs=1, 
            mods=mods_config, 
            pixel_based=config["PIXEL_BASED"], 
            native_downscaling=config["NATIVE_DOWNSCALING"], 
            pixel_resize_shape=tuple(config["PIXEL_RESIZE_SHAPE"]),
            smooth_image=config["SMOOTH_IMAGE"], 
            eval=True)() 

        # Reset environment
        rng = jax.random.PRNGKey(config["SEED"] + video_index * 10000)
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        if config["PIXEL_BASED"]:
            obs = obs.squeeze(-1)  # pixel-based: (F, H, W, 1) -> (F, H, W)
        # object-centric: (D,) already, no need to squeeze

        frames = []
        total_reward = 0.0
        max_steps = 5000

        for step in range(max_steps):
            # Pixel-based: PPO network expects (B, F, H, W)
            policy_obs = obs[None, ...]
            if not config["PIXEL_BASED"]:
                policy_obs = jnp.pad(policy_obs, ((0, 0), (0, max_D - policy_obs.shape[-1])))
            hidden = network.apply(agent_state.params.network_params, policy_obs)
            logits = actor.apply(agent_state.params.actor_params, hidden)
            action = jnp.argmax(logits, axis=-1)[0]

            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, terminated, truncated, info = env.step(env_state, action)
            done = jnp.logical_or(terminated, truncated)
            if config["PIXEL_BASED"]:
                obs = obs.squeeze(-1)  # pixel-based: (F, H, W, 1) -> (F, H, W)
            # object-centric: (D,)
            total_reward += float(reward)

            # Render frame from the underlying base Atari state, using the original renderer.
            state_for_render = env_state
            while hasattr(state_for_render, "atari_state"):
                state_for_render = state_for_render.atari_state
            if hasattr(state_for_render, "env_state"):
                state_for_render = state_for_render.env_state

            frame = renderer_local.render(state_for_render)
            frames.append(np.array(frame, dtype=np.uint8))

            if bool(done):
                break

        print(f"Final video ({video_label}): {len(frames)} frames, total reward: {total_reward:.1f}")

        if len(frames) > 0:
            frames = np.stack(frames, axis=0)
            # (N, H, W, 3) -> (N, 3, H, W) for wandb
            frames = np.transpose(frames, (0, 3, 1, 2))
            video = wandb.Video(frames, fps=30, format="mp4")
            wandb.log(
                {
                    f'final_video_seed{config["SEED"]}_{video_label}': video,
                    f'final_return_seed{config["SEED"]}_{video_label}': total_reward,
                },
                step=current_global_step
            )
            print(f"Video '{video_label}' logged to wandb.")

        return total_reward

    def generate_final_video(current_global_step=None):
        """Generate videos of the trained agent: one for train env, one per eval mod.

        This mirrors pqn_agent.generate_final_video: first a video on the
        training environment (no mods), then one per entry in eval_mods (or mods).
        """
        if not config["CAPTURE_VIDEO"]:
            return

        print(f'Generating final videos for seed {config["SEED"]}...')

        video_configs = []
        # Always add train env (no mods)
        video_configs.append(([], "train"))

        # Prefer eval_mods, fall back to mods
        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        if len(eval_mods) > 0:
            mods_list = list(eval_mods)
            for mod in mods_list:
                mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
                mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
                video_configs.append((mods_config, mod_label))

        for video_index, (mods_config, video_label) in enumerate(video_configs):
            _generate_single_final_video(mods_config, video_label, video_index, current_global_step)

    # TRY NOT TO MODIFY: start the game
    key, reset_key = jax.random.split(key)
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, config["NUM_ENVS"]))
    next_done = jnp.zeros(config["NUM_ENVS"], dtype=jax.numpy.bool_)

    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py
    def step_once(carry, step, env_step_fn):
        agent_state, obs, done, key, env_state = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        next_obs, env_state, reward, next_done, info = env_step_fn(env_state, action)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return ((agent_state, next_obs, next_done, key, env_state), (storage, info))

    def rollout(agent_state, next_obs, next_done, key, env_state, step_once_fn, max_steps):
        (agent_state, next_obs, next_done, key, env_state), (storage, info) = jax.lax.scan(
            step_once_fn, (agent_state, next_obs, next_done, key, env_state), (), max_steps
        )
        return agent_state, next_obs, next_done, storage, key, env_state, info

    rollout = partial(rollout, step_once_fn=partial(step_once, env_step_fn=vmap_step), max_steps=config["NUM_STEPS"])

    rtpt = RTPT(name_initials='TR', experiment_name='PPOEWC_JAXAtari', max_iterations=config["NUM_ITERATIONS"])
    rtpt.start()
    task_start_time = time.time()
    compile_time = None
    for iteration in range(1, config["NUM_ITERATIONS"] + 1):
        global_iteration += 1
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and (iteration % config["EVAL_EVERY"] == 0 or iteration == 1):
           eval_and_vid(iteration, global_step) 

        iteration_time_start = time.time()
        agent_state, next_obs, next_done, storage, key, env_state, info = rollout(
            agent_state, next_obs, next_done, key, env_state
        )
        global_step += config["NUM_STEPS"] * config["NUM_ENVS"]
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
            ewc_params,
            ewc_fisher,
        )
        if compile_time is None:
            compile_time = time.time()
            print(f"Compile + first iteration time: {compile_time - task_start_time:.2f} seconds.")
        # print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        metrics = {
            # task specific metrics
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/charts/avg_episodic_return": info["returned_episode_returns"].mean(), 
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/charts/avg_episodic_length": info["returned_episode_lengths"].mean(),
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/losses/value_loss": v_loss[-1, -1].item(),
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/losses/policy_loss": pg_loss[-1, -1].item(),
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/losses/entropy": entropy_loss[-1, -1].item(),
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/losses/approx_kl": approx_kl[-1, -1].item(),
            f"{task_id}+{str(list(config["TRAIN_MODS"]))}/losses/loss": loss[-1, -1].item(),

            # global metrics
            "global/learning_rate": agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            "global/SPS_update": int(config["NUM_ENVS"] * config["NUM_STEPS"] / (time.time() - iteration_time_start)),
            "global/charts/SPS": int(global_step / (time.time() - global_start_time)),
            "global/charts/time": time.time() - global_start_time,
            "global/global_step": global_step,
            # "global/current_task_index": config["TASKS"].index(task_id),
        }
        # merge metrics and info (under charts/)
        wandb.log(metrics, step=global_step)
    end_time = time.time()
    print("Training done.")

    # Generate new rollout for fisher information compuation
    _, _, _, fisher_storage, _, _, _ = rollout(agent_state, next_obs, next_done, key, env_state)

    if compile_time is not None:
        print(f"Run time after first iteration: {end_time - compile_time:.2f} seconds.")
    print(f"Total train time: {end_time - task_start_time:.2f} seconds / {(end_time - task_start_time)/60:.2f} minutes.")
    generate_final_video(global_step)

    if config["SAVE_MODEL"]:
        eval_and_vid(iteration, global_step)
        # if config["UPLOAD_MODEL"]:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f'{config["ENV_ID"]}-{config["EXP_NAME"]}-seed{config["SEED"]}'
        #     repo_id = f'{config["HF_ENTITY"]}/{repo_name}' if config["HF_ENTITY"] else repo_name
        #     push_to_hub(config, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    # writer.close()
    
    return agent_state, fisher_storage, global_step, global_iteration

def continual_run(config: dict):
    if config["TRACK"]:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=f'CL_EWC_{config["TASKS"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'
        )

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]) gradient updates
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_ITERATIONS"]
        return config["LEARNING_RATE"] * frac
    
    @jax.jit
    def compute_logprob(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate logprob of supplied `action`"""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        return logprob
    
    @jax.jit
    def compute_fisher(agent_state, obs_batch, action_batch, ewc_fisher, ewc_decay, minibatch_size=256):
        """
        Approximate the diagonal Fisher Information Matrix via squared gradients
        of the log-probability of the actions taken under the current policy in batches.

        Call at the end of each task.

        Args:
            agent_state:  current TrainState after task training is complete.
            obs_batch:    a batch of observations collected during the final
                          rollout.
            action_batch: corresponding actions.
            ewc_fisher:   previous task's FIM estimate.
            ewc_decay:    decay factor for EWC.
            minibatch_size:   size of the minibatch.
        Returns:
            new_ewc_fisher: accumulated diagonal FIM.
        """
        
        def log_prob(params, obs, action):
            logprob = compute_logprob(params, obs[None, ...], action[None, ...])  # add batch dimension
            return logprob.squeeze()  # remove batch dimension

        per_sample_grad_fn = jax.vmap(jax.grad(log_prob), in_axes=(None, 0, 0))

        total_samples = obs_batch.shape[0]
        num_batches = total_samples // minibatch_size

        # Truncate to make perfectly sized batches
        obs_batch = obs_batch[:num_batches * minibatch_size].reshape((num_batches, minibatch_size) + obs_batch.shape[1:])
        action_batch = action_batch[:num_batches * minibatch_size].reshape((num_batches, minibatch_size) + action_batch.shape[1:])

        # Use jax.lax.scan to accumulate the Fisher information in batches to avoid memory spikes
        def scan_fn(carry_fisher, minibatch):
            obs_mini, action_mini = minibatch
            per_sample_grads_mini = per_sample_grad_fn(agent_state.params, obs_mini, action_mini)    # pytree with same structure as params

            fisher_sum_mini = jax.tree.map(lambda g: jnp.sum(g ** 2, axis=0), per_sample_grads_mini)

            new_carry = jax.tree.map(jnp.add, carry_fisher, fisher_sum_mini)
            return new_carry, None

        initial_fisher_sum = jax.tree.map(jnp.zeros_like, ewc_fisher)
        total_fisher_sum, _ = jax.lax.scan(scan_fn, initial_fisher_sum, (obs_batch, action_batch))

        # Average over the total number of samples processed
        new_fisher = jax.tree.map(lambda f: f / (num_batches * minibatch_size), total_fisher_sum)  # average over all samples

        # "Online EWC without decay, equivalent to "multi" mode from MEAL (https://github.com/TTomilin/MEAL/blob/main/experiments/continual/ewc.py)
        # new_fisher = jax.tree.map(jnp.add, ewc_fisher, new_fisher)

        # Online EWC with decay: https://arxiv.org/pdf/1805.06370 (slightly modified)
        new_fisher = jax.tree.map(
            lambda f_old, f_new: ewc_decay * f_old + (1. - ewc_decay) * f_new,
            ewc_fisher, new_fisher)

        # new_ewc_params = agent_state.params  # snapshot current weights as θ*
        return new_fisher

    # TRY NOT TO MODIFY: seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, obs_sample_key1, obs_sample_key2, obs_sample_key3 = jax.random.split(key, 4)

    config["BATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    config["MINIBATCH_SIZE"] = int(config["BATCH_SIZE"] // config["NUM_MINIBATCHES"])
    config["NUM_ITERATIONS"] = int(config["TIMESTEPS_PER_TASK"] // config["BATCH_SIZE"])
    all_train_mods = list(config["TRAIN_MODS"]) # list of lists of mods, one list of mods per task
    all_eval_mods = list(config["EVAL_MODS"]) # list of lists of mods, one list of mods per task

    # dummy observation to initialize the network; shape is (1, F, H, W) for pixel-based or (1, F, D) for object-centric
    # Problem: In object-centric, the observation space is different for each game
    # FIX: for object-centric, find maximum observation space dim and zero-pad all other observations to that size
    max_D = None
    observation_space_shape = None
    if not config["PIXEL_BASED"]:
        max_D = 0
        for i, task_id in enumerate(config["TASKS"]):
            env_ = make_env(
                env_id=task_id, 
                seed=config["SEED"], 
                num_envs=config["NUM_ENVS"], 
                mods=all_train_mods[i] if i < len(all_train_mods) else [], 
                pixel_based=config["PIXEL_BASED"], 
                native_downscaling=config["NATIVE_DOWNSCALING"], 
                pixel_resize_shape=tuple(config["PIXEL_RESIZE_SHAPE"]), 
                smooth_image=config["SMOOTH_IMAGE"], 
                frame_stack=config["FRAME_STACK"]
            )()
            D = env_.observation_space().shape[0]
            max_D = max(max_D, D)
        observation_space_shape = (config["FRAME_STACK"], max_D)    # (F, D) for object-centric
    else:
        observation_space_shape = (config["FRAME_STACK"], *config["PIXEL_RESIZE_SHAPE"])    # (F, H, W) for pixel-based
    dummy_obs = jnp.zeros((1, *observation_space_shape))  # (1, F, H, W) or (1, F, D)

    # initialize the network, actor, and critic
    network = Network() if config["PIXEL_BASED"] else MLP_Network()
    actor = Actor(action_dim=18)  # full action space (18 actions) for all Atari games
    critic = Critic()
    # network_params = network.init(network_key, env.observation_space().sample(obs_sample_key1).squeeze()[None, ...])
    # Init shape is (1,4,84,84) (which will be transposed inside the network to (1,84,84,4))
    network_params = network.init(network_key, dummy_obs)
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, network.apply(network_params, dummy_obs)),
            critic_params=critic.init(critic_key, network.apply(network_params, dummy_obs)),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LEARNING_RATE"], eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    ewc_fisher = jax.tree.map(jnp.zeros_like, agent_state.params)
    ewc_params = jax.tree.map(jnp.zeros_like, agent_state.params)

    global_step = 0 # tracks total number of environment steps across all tasks
    global_iteration = 0    # tracks total number of PPO updates
    global_start_time = time.time()
    seen_tasks = [] # tracks tasks that have been trained on so far, for evaluation

    for i, task_id in enumerate(config["TASKS"]):
        train_mods = all_train_mods[i] if i < len(all_train_mods) else []
        eval_mods = all_eval_mods[i] if i < len(all_eval_mods) else []
        seen_tasks.append((task_id, train_mods, eval_mods))

        # reset the optimizer state (step count & momentum) for new tasks
        if i > 0:
            agent_state = agent_state.replace(
                opt_state=agent_state.tx.init(agent_state.params)
            )

        # we get back the updated agent_state and the storage from the rollout
        agent_state, fisher_storage, global_step, global_iteration = single_run(
            key, network, actor, critic, config, task_id, agent_state, 
            ewc_fisher, ewc_params, max_D, global_step, global_iteration, 
            global_start_time, seen_tasks, train_mods, eval_mods
        )

        # Set EWC decay to 0 for the first task to ignore the initial zero-value pytree
        ewc_decay = 0.0 if i == 0 else config["EWC_DECAY"]

        # EWC: Compute Fisher Information Matrix and snapshot θ* at the end of the task
        # TODO: could the full storage be too large and cause memory issues?
        ewc_params = agent_state.params
        flat_obs = fisher_storage.obs.reshape((-1,) + fisher_storage.obs.shape[2:])
        flat_actions = fisher_storage.actions.reshape(-1)
        ewc_fisher = compute_fisher(agent_state, flat_obs, flat_actions, ewc_fisher, jnp.float32(ewc_decay))

    wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    continual_run(merged_config)

if __name__ == "__main__":
    main()
