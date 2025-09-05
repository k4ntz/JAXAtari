import faulthandler

import flax

import jaxatari.core
faulthandler.enable()
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# optional:
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import argparse
from datetime import datetime
import imageio
import shutil
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Callable
from collections import deque
from tqdm import tqdm
import pygame
from functools import partial
import matplotlib.pyplot as plt

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper, PixelAndObjectCentricWrapper
import jaxatari.games.jax_pong as jax_pong
import jaxatari.spaces as spaces
from ppo_agent import (ActorCritic, create_ppo_train_state,
                       ppo_update_minibatch)
from jaxatari.wrappers import (AtariWrapper, FlattenObservationWrapper,
                        ObjectCentricWrapper, PixelObsWrapper, NormalizeObservationWrapper)

# Default configuration
PPO_CONFIG = {
    "ENV_NAME_JAXATARI": "pong",
    "TOTAL_TIMESTEPS": 100_000_000,
    "LR": 5e-4,
    "NUM_ENVS": 128,
    "NUM_STEPS": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "NUM_MINIBATCHES": 16,
    "UPDATE_EPOCHS": 10,
    "CLIP_EPS": 0.2,
    "CLIP_VF_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "SEED": 42,
    "BUFFER_WINDOW": 4,
    "FRAMESKIP": 4,
    "LOG_INTERVAL_UPDATES": 20, # Log less frequently (since updates are less frequent)
}

def update_pygame(window, image_data, upscale_factor, width, height):
    """Updates the Pygame window with a new frame."""
    image_surface = pygame.surfarray.make_surface(np.transpose(image_data, (1, 0, 2)))
    upscaled_surface = pygame.transform.scale(
        image_surface, (width * upscale_factor, height * upscale_factor)
    )
    window.blit(upscaled_surface, (0, 0))
    pygame.display.flip()


def create_env(game_name: str, buffer_window: int, frameskip: int, use_pixels: bool):
    """Creates and wraps the JAXAtari environment based on config."""
    env = jaxatari.make(game_name)
    env = AtariWrapper(env, sticky_actions=True, frame_stack_size=buffer_window, frame_skip=frameskip, episodic_life=False)
    
    if use_pixels:
        print("Using Pixel-based observations.")
        # We use PixelObsWrapper to only output pixel observations.
        env = PixelObsWrapper(env)
    else:
        print("Using Object-centric observations.")
        env = ObjectCentricWrapper(env)
        
    env = FlattenObservationWrapper(env)
    env = NormalizeObservationWrapper(env, to_neg_one=True)
    return env


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

    # Observations are already normalized by the wrapper
    # Ensure it's (num_envs, features_flat_after_norm)
    if next_raw_obs_batched.ndim == 1 and num_envs == 1:  # Special case for num_envs=1
        next_raw_obs_batched = next_raw_obs_batched.reshape(1, -1)
    elif next_raw_obs_batched.ndim == 2:  # Expected case (num_envs, features)
        pass
    else:  # Potentially (num_envs, H, W, C*F) if not flattened before normalization
        next_raw_obs_batched = next_raw_obs_batched.reshape(num_envs, -1)

    return (
        next_raw_obs_batched,
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


@partial(jax.jit, static_argnums=(3, 4))
def collect_rollout_step(
    train_state,
    current_obs_batched: jnp.ndarray,
    current_env_states_batched: Any,
    vmapped_step_fn: Callable,
    obs_space: spaces.Space,
    agent_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Function to collect a single step of rollout data."""
    pi, value = train_state.apply_fn({'params': train_state.params}, current_obs_batched)
    if value.ndim > 1:
        value_squeezed = value.squeeze(axis=-1)
    else:
        value_squeezed = value
    actions = pi.sample(seed=agent_key)
    log_probs = pi.log_prob(actions)

    next_raw_obs_batched, next_env_states_batched, rewards_batched, dones_batched, _ = \
        vmapped_step_fn(current_env_states_batched, actions.astype(jnp.int32))

    # Observations are already normalized by the wrapper
    return (
        next_raw_obs_batched,
        next_raw_obs_batched,
        next_env_states_batched,
        actions,
        log_probs,
        rewards_batched,
        dones_batched,
        value_squeezed
    )


def train_ppo_with_jaxatari(config: Dict[str, Any]):
    """Main training loop for PPO on a JAXAtari environment."""
    np.random.seed(config["SEED"])
    main_rng = jax.random.PRNGKey(config["SEED"])

    game_name = config["ENV_NAME_JAXATARI"]
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    print(f"--- Starting Training on {game_name.upper()} ---")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num Updates: {config['NUM_UPDATES']:,}")
    print("-------------------------------------------------")

    # Vectorized environment creation using the new standalone function
    proto_env = create_env(game_name, config["BUFFER_WINDOW"], config["FRAMESKIP"], config["USE_PIXELS"])
    vmapped_reset_fn = jax.vmap(proto_env.reset)
    vmapped_step_fn = jax.vmap(proto_env.step)

    # Initial Reset
    reset_keys = jax.random.split(main_rng, config["NUM_ENVS"] + 1)
    main_rng, agent_key, init_key = jax.random.split(reset_keys[0], 3)
    current_raw_obs, current_batched_env_states = vmapped_reset_fn(reset_keys[1:])

    # Prepare for training
    obs_space = proto_env.observation_space()
    obs_shape_flat = current_raw_obs.shape[1:]
    train_state = create_ppo_train_state(init_key, config, obs_shape_flat, proto_env.action_space().n)
    
    # Logging
    episode_rewards_deque = deque(maxlen=100)
    all_mean_rewards_history, all_timesteps_history = [], []
    
    pbar = tqdm(total=config["TOTAL_TIMESTEPS"], desc="Training Progress")
    
    @jax.jit
    def collect_full_rollout(train_state, initial_obs_norm, initial_env_states, rollout_key):
        def _step_body(carry, _):
            obs_norm, env_states, key = carry
            key, step_key = jax.random.split(key)
            pi, value = train_state.apply_fn({'params': train_state.params}, obs_norm)
            actions = pi.sample(seed=step_key)
            log_probs = pi.log_prob(actions)
            next_raw_obs, next_env_states, rewards, dones, _ = vmapped_step_fn(env_states, actions.astype(jnp.int32))
            # Observations are already normalized by the wrapper
            transition = {"obs": obs_norm, "actions": actions, "log_probs": log_probs, "rewards": rewards, "dones": dones, "values": value if value.ndim == 1 else value.squeeze(-1)}
            return (next_raw_obs, next_env_states, key), transition
        (final_obs, final_states, _), collected_transitions = jax.lax.scan(_step_body, (initial_obs_norm, initial_env_states, rollout_key), None, length=config["NUM_STEPS"])
        _, last_val = train_state.apply_fn({'params': train_state.params}, final_obs)
        return final_obs, final_states, collected_transitions, last_val

    # JIT-compiled function for a SINGLE minibatch update.
    @jax.jit
    def _update_minibatch_jit(train_state, minibatch_data, ppo_hparams):
        obs_mb, act_mb, logp_mb, val_mb, adv_mb, ret_mb = minibatch_data
        new_train_state, _, _ = ppo_update_minibatch(
            train_state, obs_mb, act_mb, logp_mb, val_mb, adv_mb, ret_mb, ppo_hparams
        )
        return new_train_state

    # Simplified update loop using standard Python loops for clarity and robustness.
    def ppo_update_epochs(train_state, batch_data, update_key):
        b_obs, b_actions, b_log_probs, b_values, b_advantages, b_returns = batch_data
        ppo_hparams = {k: config[k] for k in ["CLIP_EPS", "CLIP_VF_EPS", "ENT_COEF", "VF_COEF", "MAX_GRAD_NORM"]}

        for _ in range(config["UPDATE_EPOCHS"]):
            update_key, perm_key = jax.random.split(update_key)
            total_batch_size = b_obs.shape[0]
            permutation = jax.random.permutation(perm_key, total_batch_size)
            shuffled_batch = jax.tree.map(lambda x: x[permutation], (b_obs, b_actions, b_log_probs, b_values, b_advantages, b_returns))

            num_minibatches = config["NUM_MINIBATCHES"]
            minibatch_size = total_batch_size // num_minibatches
            
            for i in range(num_minibatches):
                start, end = i * minibatch_size, (i + 1) * minibatch_size
                minibatch = jax.tree.map(lambda x: x[start:end], shuffled_batch)
                train_state = _update_minibatch_jit(train_state, minibatch, ppo_hparams)
                
        return train_state

    # Main training loop
    for update_idx in range(1, config["NUM_UPDATES"] + 1):
        agent_key, rollout_key, update_key = jax.random.split(agent_key, 3)
        current_raw_obs, current_batched_env_states, rollout_data, last_val = collect_full_rollout(train_state, current_raw_obs, current_batched_env_states, rollout_key)

        advantages, returns = compute_advantages(
            rollout_data["rewards"], rollout_data["values"], rollout_data["dones"],
            last_val if last_val.ndim == 1 else last_val.squeeze(), config["NUM_STEPS"], config["GAMMA"], config["GAE_LAMBDA"]
        )

        if update_idx % config["LOG_INTERVAL_UPDATES"] == 0:
            rewards_np = np.asarray(rollout_data["rewards"])
            mean_rollout_reward = np.mean(np.sum(rewards_np, axis=0))
            episode_rewards_deque.append(mean_rollout_reward)
            if len(episode_rewards_deque) > 0:
                mean_reward = np.mean(list(episode_rewards_deque))
                current_timesteps = update_idx * config["NUM_STEPS"] * config["NUM_ENVS"]
                all_mean_rewards_history.append(mean_reward)
                all_timesteps_history.append(current_timesteps)
                pbar.set_postfix({"mean_reward": f"{mean_reward:.2f}"})

        batch_data = (
            rollout_data["obs"].swapaxes(0, 1).reshape((-1,) + obs_shape_flat),
            rollout_data["actions"].swapaxes(0, 1).reshape(-1),
            rollout_data["log_probs"].swapaxes(0, 1).reshape(-1),
            rollout_data["values"].swapaxes(0, 1).reshape(-1),
            advantages.swapaxes(0, 1).reshape(-1),
            returns.swapaxes(0, 1).reshape(-1)
        )

        train_state = ppo_update_epochs(train_state, batch_data, update_key)
        pbar.update(config["NUM_STEPS"] * config["NUM_ENVS"])

    pbar.close()
    print("--- Training Finished ---")
    
    return train_state, {"timesteps": all_timesteps_history, "mean_rewards": all_mean_rewards_history}


def plot_training_rewards(metrics: Dict[str, Any], save_path: str):
    """Plots and saves the training reward curve."""
    if not metrics["timesteps"]:
        print("No metrics recorded, skipping plot generation.")
        return
        
    print(f"Generating reward plot at {save_path}...")
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["timesteps"], metrics["mean_rewards"])
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Plot saved successfully.")


def visualize_agent_gameplay(train_state, config: Dict[str, Any]):
    """Visualizes one episode of the trained agent's gameplay and saves it as an MP4."""
    print("\n--- Starting Visualization ---")
    
    UPSCALE_FACTOR = 4
    WIDTH, HEIGHT = 160, 210
    FPS = 30

    # Create a temporary directory for frames
    frames_dir = os.path.join(config["RESULTS_DIR"], "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)

    pygame.init()
    window = pygame.display.set_mode((WIDTH * UPSCALE_FACTOR, HEIGHT * UPSCALE_FACTOR))
    pygame.display.set_caption(f"PPO Agent Playing {config['ENV_NAME_JAXATARI'].upper()}")
    clock = pygame.time.Clock()

    # Create the environment with the correct wrapper (pixel or object)
    env = create_env(config["ENV_NAME_JAXATARI"], config["BUFFER_WINDOW"], config["FRAMESKIP"], config["USE_PIXELS"])
    
    vis_key = jax.random.PRNGKey(config["SEED"] + 1)
    obs, state = env.reset(key=vis_key)
    obs_space = env.observation_space()

    done, running = False, True
    total_reward, frame_count = 0.0, 0

    while not done and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Observations are already normalized by the wrapper
        pi, _ = train_state.apply_fn({'params': train_state.params}, jnp.expand_dims(obs, axis=0))
        action = pi.mode()[0]

        obs, state, reward, done, _ = env.step(state, action)
        total_reward += reward

        image = env._env.render(state.env_state)
        image_cpu = np.asarray(image)
        update_pygame(window, image_cpu, UPSCALE_FACTOR, WIDTH, HEIGHT)

        # Save the current frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.png")
        pygame.image.save(window, frame_path)
        frame_count += 1

        clock.tick(FPS)

    pygame.quit()
    print(f"--- Visualization Finished ---")
    print(f"Episode Score: {total_reward:.2f}")

    # --- Compile frames into a video ---
    video_path = os.path.join(config["RESULTS_DIR"], f"gameplay_{config['ENV_NAME_JAXATARI']}_{config['TIMESTAMP']}.mp4")
    print(f"\nCompiling {frame_count} frames into video: {video_path}")
    
    try:
        with imageio.get_writer(video_path, fps=FPS) as writer:
            for i in tqdm(range(frame_count), desc="Creating MP4"):
                frame_file = os.path.join(frames_dir, f"frame_{i:05d}.png")
                writer.append_data(imageio.imread(frame_file))
        print("Video saved successfully.")
    except Exception as e:
        print(f"Error creating video: {e}")
    finally:
        # Clean up temporary frames
        print("Cleaning up temporary frames...")
        shutil.rmtree(frames_dir)


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on a JAXAtari game, then visualize it.")
    parser.add_argument("--game", type=str, default="pong", help="Name of the JAXAtari game to play.")
    parser.add_argument("--timesteps", type=int, default=20_000_000, help="Total number of timesteps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments for training.")
    parser.add_argument("--use-pixels", action="store_true", default=False, help="Use pixel observations instead of object-centric ones.")

    args = parser.parse_args()

    config = PPO_CONFIG.copy()
    config["ENV_NAME_JAXATARI"] = args.game
    config["TOTAL_TIMESTEPS"] = args.timesteps
    config["SEED"] = args.seed
    config["NUM_ENVS"] = args.num_envs
    config["USE_PIXELS"] = args.use_pixels
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add results info to config for easy access
    config["RESULTS_DIR"] = results_dir
    config["TIMESTAMP"] = timestamp

    # 1. Train the agent
    trained_state, metrics = train_ppo_with_jaxatari(config)

    # 2. Plot the results
    plot_filename = f"reward_curve_{args.game}_{timestamp}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plot_training_rewards(metrics, plot_path)
    
    # 3. Save the trained agent's parameters
    print(f"\n--- Saving Agent Parameters ---")
    model_filename = f"ppo_agent_{args.game}_{timestamp}.npz"
    model_path = os.path.join(results_dir, model_filename)
    params_to_save = flax.serialization.to_state_dict(trained_state.params)
    np.savez(model_path, **params_to_save)
    print(f"Agent parameters saved to {model_path}")

    # 4. Visualize the trained agent
    visualize_agent_gameplay(trained_state, config)

if __name__ == "__main__":
    main()