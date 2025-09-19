import faulthandler

import jaxatari.core
faulthandler.enable()

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Callable
from collections import deque
from tqdm import tqdm
from ppo_agent import ActorCritic, create_ppo_train_state, ppo_loss_fn, ppo_update_minibatch 
import pygame
from functools import partial

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper, PixelAndObjectCentricWrapper
import jaxatari.games.jax_pong as jax_pong
import jaxatari.spaces as spaces


def normalize_observation_jaxatari(obs: jnp.ndarray, obs_space: spaces.Space) -> jnp.ndarray:
    # assuming the obs space is flattened at this point, get the max values for each feature
    max_values = obs_space.high.flatten()
    min_values = obs_space.low.flatten()

    num_features_per_frame = len(max_values)
    num_frames = obs.shape[-1] // num_features_per_frame
    max_values = jnp.tile(max_values, num_frames)
    min_values = jnp.tile(min_values, num_frames)

    return 2* ((obs - min_values)/(max_values-min_values)) - 1.0


def env_step(env_step_fn, state, action, agent_key):
    """Single environment step function (helper, not directly vmapped anymore)."""
    next_obs, curr_state, reward, terminated, _ = env_step_fn(agent_key, state, int(action))
    return next_obs, curr_state, reward, terminated

def collect_rollout_step_vmapped(
    train_state,
    current_obs_batched: jnp.ndarray,  # Already normalized and flat (num_envs, obs_dim)
    current_env_states_batched: Any,  # A PyTree of batched environment states
    agent_key: jnp.ndarray,  # A single key, will be split for vmap
    representative_env_step_fn: Callable,  # Single step function to vmap over
    num_envs: int,
    obs_space: spaces.Space
) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Function to collect a single step of rollout data using jax.vmap for env steps."""
    # 1. Get actions and values from the policy
    pi, value = train_state.apply_fn({'params': train_state.params}, current_obs_batched)

    if value.ndim == 2 and value.shape[1] == 1:
        value_squeezed = value.squeeze(axis=-1)
    else:
        value_squeezed = value

    actions = pi.sample(seed=agent_key)  # actions shape: (num_envs,)
    log_probs = pi.log_prob(actions)    # log_probs shape: (num_envs,)

    # 2. Vmap the environment step function
    vmapped_step_fn = jax.vmap(
        representative_env_step_fn, in_axes=(0, 0), out_axes=0
    )

    # current_env_states_batched is a PyTree where leaves have shape (num_envs, ...)
    # actions needs to be int32 for env step
    next_raw_obs_batched, next_env_states_batched, rewards_batched, dones_batched, _ = \
        vmapped_step_fn(current_env_states_batched, actions.astype(jnp.int32))

    # 3. Normalize observations
    next_obs_norm_batched = normalize_observation_jaxatari(next_raw_obs_batched, obs_space)
    # Ensure it's (num_envs, features_flat_after_norm)
    if next_obs_norm_batched.ndim == 1 and num_envs == 1:  # Special case for num_envs=1
        next_obs_norm_batched = next_obs_norm_batched.reshape(1, -1)
    elif next_obs_norm_batched.ndim == 2:  # Expected case (num_envs, features)
        pass
    else:  # Potentially (num_envs, H, W, C*F) if not flattened before normalization
        next_obs_norm_batched = next_obs_norm_batched.reshape(num_envs, -1)

    return (
        next_obs_norm_batched,
        next_env_states_batched,
        actions,
        log_probs,
        rewards_batched,
        dones_batched,
        value_squeezed
    )


# Create a JIT-compiled version with static arguments
collect_rollout_step_vmapped_jit = jax.jit(
    collect_rollout_step_vmapped,
    static_argnums=(4, 5, 6)  # Only mark representative_env_step_fn and num_envs as static
)

jax.jit
def compute_advantages(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    num_steps: int,
    gamma: float,
    gae_lambda: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled function to compute advantages and returns."""
    advantages = jnp.zeros_like(rewards)
    last_gae_lam = 0
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t+1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages = advantages.at[t].set(delta + gamma * gae_lambda * next_non_terminal * last_gae_lam)
        last_gae_lam = advantages[t]
    
    returns = advantages + values
    return advantages, returns


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12))
def update_minibatch_vmapped(
    train_state,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    log_probs_old: jnp.ndarray,
    values_old: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    num_minibatches: int,
    clip_eps: float,
    clip_vf_eps: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float
) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
    """JIT-compiled and vmapped function for PPO minibatch updates."""
    def single_update(carry, x):
        train_state = carry
        mb_indices = x
        
        train_state, loss, aux_info = ppo_update_minibatch(
            train_state,
            obs[mb_indices],
            actions[mb_indices],
            log_probs_old[mb_indices],
            values_old[mb_indices],
            advantages[mb_indices],
            returns[mb_indices],
            {
                "CLIP_EPS": clip_eps,
                "CLIP_VF_EPS": clip_vf_eps,
                "ENT_COEF": ent_coef,
                "VF_COEF": vf_coef,
                "MAX_GRAD_NORM": max_grad_norm
            }
        )
        return train_state, (loss, aux_info)
    
    # Create minibatch indices
    total_batch_size = obs.shape[0]
    minibatch_size = total_batch_size // num_minibatches
    indices = jnp.arange(total_batch_size)
    indices = jnp.reshape(indices, (num_minibatches, minibatch_size))
    
    # Vmap the updates
    final_state, (losses, aux_infos) = jax.lax.scan(single_update, train_state, indices)
    
    # Average the losses and aux_infos
    avg_loss = jax.tree.map(lambda x: jnp.mean(x, axis=0), losses)
    avg_aux_info = jax.tree.map(lambda x: jnp.mean(x, axis=0), aux_infos)
    
    return final_state, avg_loss, avg_aux_info


def train_ppo_with_jaxatari(config: Dict[str, Any]):
    np.random.seed(config["SEED"]) 
    main_rng = jax.random.PRNGKey(config["SEED"])

    game_name = config["ENV_NAME_JAXATARI"] 

    if game_name != "pong":
        # TODO: change the core to support other games
        raise ValueError(f"Game {game_name} is not supported for PPO training right now.")

    buffer_window = config["BUFFER_WINDOW"]
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE_CALCULATED"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    print(f"Using JaxAtari environment: {game_name}")
    # create all environments
    envs = []
    for i in range(config["NUM_ENVS"]):
        env = jaxatari.make(game_name)
        env: AtariWrapper = AtariWrapper(env, sticky_actions=True, frame_stack_size=buffer_window, frame_skip=config["FRAMESKIP"]) # get the atari wrapper to handle things like frame stacking, sticky actions, etc.
        env: ObjectCentricWrapper = ObjectCentricWrapper(env) # use the object centric wrapper to only return object centric observations
        env: FlattenObservationWrapper = FlattenObservationWrapper(env) # flatten the object centric observation to a single vector
        envs.append(env)

    # We will use envs[0].step as the representative_env_step_fn
    representative_env_step_fn = envs[0].step

    obs_list = []
    states_list_py = []  # Keep as Python list of individual PyTree states initially
    reset_keys = jax.random.split(main_rng, config["NUM_ENVS"] + 1)  # +1 for main_rng split later
    main_rng = reset_keys[0]

    obs_space = envs[0].observation_space()

    for i, env in enumerate(envs):
        # Each env.reset needs its own key
        obs, curr_state = env.reset(key=reset_keys[i+1])
        obs_list.append(obs)
        states_list_py.append(curr_state)

    current_raw_obs_stacked = jnp.stack(obs_list)  # (num_envs, *raw_obs_shape)
    obs_shape_flat = current_raw_obs_stacked.shape[1:]  # Shape of one flattened raw observation

    # Get the base environment directly TODO: maybe pass the action space through as well?
    possible_actions = envs[0]._env.action_space()

    # at this point print the possible actions
    print(f"Possible actions: {possible_actions}")

    agent_key, init_key = jax.random.split(main_rng)
    train_state = create_ppo_train_state(init_key, config, obs_shape_flat, possible_actions.n)

    # Initial normalization and state batching
    current_obs_stacked_norm_flat = normalize_observation_jaxatari(current_raw_obs_stacked, obs_space).reshape(config["NUM_ENVS"], -1)
    # Batch the list of PyTree states into a single PyTree with leading batch dimension
    current_batched_env_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states_list_py)

    rollout_obs_flat = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]) + obs_shape_flat, dtype=jnp.float32)
    rollout_actions = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.int32)
    rollout_log_probs = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32)
    rollout_rewards = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32)
    rollout_dones = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.bool_)
    rollout_values = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=jnp.float32)

    episode_rewards_deque = deque(maxlen=100 * config["NUM_ENVS"]) 
    all_episode_rewards_history = [] 
    all_mean_rewards_history, all_timesteps_history = [], []
    all_pg_loss_hist, all_vf_loss_hist, all_ent_hist = [], [], []
    current_episode_rewards_np = np.zeros(config["NUM_ENVS"]) 

    # Create progress bar for overall training
    pbar = tqdm(total=config["NUM_UPDATES"], desc="Training Progress", position=0)

    for update_idx in range(1, config["NUM_UPDATES"] + 1):
        agent_key, rollout_sample_key, update_perm_key = jax.random.split(agent_key, 3)

        # Create progress bar for rollout steps
        rollout_pbar = tqdm(total=config["NUM_STEPS"], desc="Rollout", position=1, leave=False)
        
        # Track rewards during rollout for immediate feedback
        rollout_rewards_sum = 0
        rollout_rewards_count = 0

        for step_idx in range(config["NUM_STEPS"]):
            total_env_steps_so_far = (update_idx - 1) * config["NUM_STEPS"] * config["NUM_ENVS"] + step_idx * config["NUM_ENVS"]

            # Use JIT-compiled rollout collection with vmapped step function
            next_obs_norm, next_batched_env_states, actions, log_probs, rewards, dones, value = collect_rollout_step_vmapped_jit(
                train_state,
                current_obs_stacked_norm_flat,  # Already normalized
                current_batched_env_states,     # Batched PyTree of env states
                rollout_sample_key,             # Key for this step
                representative_env_step_fn,     # Single step function to vmap over
                config["NUM_ENVS"],
                obs_space
            )

            # Split key for next iteration
            agent_key, rollout_sample_key = jax.random.split(agent_key)

            # Convert JAX arrays to NumPy for CPU-side logic
            rewards_np = np.array(rewards)
            dones_np = np.array(dones)

            # Update episode rewards and logging
            current_episode_rewards_np += rewards_np
            for i in range(config["NUM_ENVS"]):
                if dones_np[i]:
                    episode_rewards_deque.append(current_episode_rewards_np[i])
                    all_episode_rewards_history.append(current_episode_rewards_np[i])
                    current_episode_rewards_np[i] = 0.0
                    # Important: If an env is done, its state needs to be reset
                    # Get reset_obs and reset_state for done environments
                    reset_key, agent_key = jax.random.split(agent_key)
                    new_obs_done, new_state_done = envs[i].reset(reset_key)
                    # Update the corresponding slices in next_obs_norm and next_batched_env_states
                    next_obs_norm = next_obs_norm.at[i].set(normalize_observation_jaxatari(new_obs_done, obs_space).flatten())
                    # Update the batched state PyTree
                    next_batched_env_states = jax.tree_util.tree_map(
                        lambda leaf, new_state: leaf.at[i].set(new_state),
                        next_batched_env_states,
                        new_state_done
                    )

            # Update rollout progress bar metrics
            rollout_rewards_sum += jnp.sum(rewards).item()
            rollout_rewards_count += config["NUM_ENVS"]

            rollout_obs_flat = rollout_obs_flat.at[step_idx].set(current_obs_stacked_norm_flat)
            rollout_actions = rollout_actions.at[step_idx].set(actions)
            rollout_log_probs = rollout_log_probs.at[step_idx].set(log_probs)
            rollout_rewards = rollout_rewards.at[step_idx].set(rewards)
            rollout_dones = rollout_dones.at[step_idx].set(dones)
            
            current_obs_stacked_norm_flat = next_obs_norm
            current_batched_env_states = next_batched_env_states

            rollout_pbar.update(1)
            rollout_pbar.set_postfix({
                "env_steps": total_env_steps_so_far,
                "avg_reward": f"{rollout_rewards_sum / rollout_rewards_count:.2f}" if rollout_rewards_count > 0 else "N/A"
            })
    
        rollout_pbar.close()

        # Get final value for advantage calculation
        _, last_val = train_state.apply_fn({'params': train_state.params}, current_obs_stacked_norm_flat)
        
        # Use JIT-compiled advantage calculation
        advantages, returns = compute_advantages(
            rollout_rewards,
            rollout_values,
            rollout_dones,
            last_val,
            config["NUM_STEPS"],
            config["GAMMA"],
            config["GAE_LAMBDA"]
        )

        b_obs = rollout_obs_flat.reshape((-1,) + obs_shape_flat)
        b_actions = rollout_actions.reshape(-1)
        b_log_probs_old = rollout_log_probs.reshape(-1)
        b_values_old = rollout_values.reshape(-1) 
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Use vmapped minibatch updates
        for epoch in range(config["UPDATE_EPOCHS"]):
            # Ensure permutation is applied to all data consistently
            epoch_perm_key, agent_key = jax.random.split(agent_key)
            permutation = jax.random.permutation(epoch_perm_key, b_obs.shape[0])

            # Apply permutation to all batch elements
            shuffled_b_obs = b_obs[permutation]
            shuffled_b_actions = b_actions[permutation]
            shuffled_b_log_probs_old = b_log_probs_old[permutation]
            shuffled_b_values_old = b_values_old[permutation]
            shuffled_b_advantages = b_advantages[permutation]
            shuffled_b_returns = b_returns[permutation]

            train_state, loss, aux_info = update_minibatch_vmapped(
                train_state,
                shuffled_b_obs,
                shuffled_b_actions,
                shuffled_b_log_probs_old,
                shuffled_b_values_old,
                shuffled_b_advantages,
                shuffled_b_returns,
                config["NUM_MINIBATCHES"],
                config["CLIP_EPS"],
                config["CLIP_VF_EPS"],
                config["ENT_COEF"],
                config["VF_COEF"],
                config["MAX_GRAD_NORM"]
            )
            
        if update_idx % config.get("LOG_INTERVAL_UPDATES", 10) == 0:
            mean_all_rewards = np.mean(list(episode_rewards_deque))
            all_mean_rewards_history.append(mean_all_rewards)
            # Convert JAX arrays to Python values
            loss_float = float(loss)
            aux_info_dict = jax.tree_util.tree_map(lambda x: float(x), aux_info)
            all_pg_loss_hist.append(aux_info_dict["pg_loss"])
            all_vf_loss_hist.append(aux_info_dict["vf_loss"])
            all_ent_hist.append(aux_info_dict["entropy"])
            current_total_steps_log = update_idx * config["NUM_STEPS"] * config["NUM_ENVS"]
            all_timesteps_history.append(current_total_steps_log)

            pbar.set_postfix({
                "mean_all_rewards": f"{mean_all_rewards:.2f}",
                "pg_loss": f"{aux_info_dict['pg_loss']:.4f}",
                "vf_loss": f"{aux_info_dict['vf_loss']:.4f}",
                "entropy": f"{aux_info_dict['entropy']:.4f}"
            })

        pbar.update(1)

    pbar.close()
    print("Training finished.")

    training_results_metrics = {
        "timesteps": all_timesteps_history,
        "mean_rewards": all_mean_rewards_history,
        "pg_losses": all_pg_loss_hist,
        "vf_losses": all_vf_loss_hist,
        "ent_losses": all_ent_hist,
        "all_episode_rewards": all_episode_rewards_history
    }

    return train_state, training_results_metrics