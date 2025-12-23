import faulthandler

import jaxatari
from jaxatari.wrappers import ObjectCentricWrapper
faulthandler.enable()

import jax
import jax.numpy as jnp
import numpy as np
from ocatari.core import OCAtari

try:
    from train_ocatari_agent import (
        train_ppo_with_ocatari, 
        normalize_observation_ocatari 
    )

    from scripts.benchmarks.ppo_agent_old import (
        create_ppo_train_state,
        ActorCritic,
        TrainState
    )

    from train_jaxatari_agent import (
        train_ppo_with_jaxatari,
        normalize_observation_jaxatari
    )

    from jaxatari.games.jax_pong import JaxPong
    from jaxatari.games.jax_pong import PongRenderer as JaxPongRenderer
    from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper
    import jaxatari.rendering.jax_rendering_utils as jax_rendering_utils

except ImportError as e:
    print(f"Error importing modules: {e}")
    exit()

import matplotlib.pyplot as plt
import os
from datetime import datetime
import flax.serialization
import optax # For dummy optimizer if needed
import flax.core 
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
import pygame
from tqdm import tqdm

# Constants for visualization
SCALING_FACTOR = 3
WIDTH = 160  # Standard Atari game width
HEIGHT = 210  # Standard Atari game height

# Check if GPU is available and set it as the default device
if jax.devices('gpu'):
    jax.config.update('jax_platform_name', 'gpu')
    print("Using GPU for training")
else:
    print("No GPU found, using CPU")

# --- PPO Configuration (aligns with keys used in the new agent script) ---
ppo_config_distrax = {
    "ENV_NAME_OCATARI": "Pong", # Specific key for OCAtari env name
    "ENV_NAME_JAXATARI": "pong", # Specific key for JAXAtari env name
    "ENV_TYPE": "ocatari", # Can be "ocatari" or "jaxatari"
    "TOTAL_TIMESTEPS": 20_000_000,
    "TOTAL_TIMESTEPS_PER_EPOCH": 10_000,
    "LR": 5e-4,               # Learning rate
    "NUM_ENVS": 128,              # Number of parallel environments
    "NUM_STEPS": 256,           # Steps per environment per rollout (batch size for actor)
    "GAMMA": 0.99,              # Discount factor
    "GAE_LAMBDA": 0.95,         # GAE lambda
    "NUM_MINIBATCHES": 16,       # Number of minibatches for PPO update
    "UPDATE_EPOCHS": 10,         # Number of epochs for PPO update
    "CLIP_EPS": 0.2,            # PPO clip parameter
    "CLIP_VF_EPS": 0.2,         # PPO clip for value function (optional, if different from CLIP_EPS)
    "ENT_COEF": 0.01,           # Entropy coefficient
    "VF_COEF": 0.5,             # Value function coefficient
    "MAX_GRAD_NORM": 0.5,       # Max gradient norm for clipping
    "ACTIVATION": "relu",       # Activation function in network ("tanh" or "relu")
    "ANNEAL_LR": True,          # Whether to linearly anneal learning rate
    "SEED": 42,
    
    "BUFFER_WINDOW": 4,
    "FRAMESKIP": 4,
    "REPEAT_ACTION_PROBABILITY": 0.25,

    "LOG_INTERVAL_UPDATES": 20, # Log every 20 PPO updates
    "VISUALIZE_AFTER_TRAINING": False, 
    "VIZ_STEPS": 1000,
    "VIZ_FPS": 30, # FPS for visualization
    "SAVE_VIZ_VIDEO": True,
}

def train_ppo_agent_ocatari(config_dict: Dict[str, Any]) -> Tuple[TrainState, str, Dict[str, Any]]:
    env_type = config_dict.get("ENV_TYPE", "ocatari")
    env_name = config_dict["ENV_NAME_OCATARI"] if env_type == "ocatari" else "Pong"
    
    print(f"Training PPO agent (Distrax base) with {env_type.upper()} environment (Game: {env_name})...")
    
    trained_ppo_state, training_metrics = train_ppo_with_ocatari(config_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ppo_distrax_{env_type}_{env_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model parameters
    model_params_path = os.path.join(results_dir, "ppo_distrax_model_params.npz")
    params_dict_to_save = flax.serialization.to_state_dict(trained_ppo_state.params)
    np.savez(model_params_path, **params_dict_to_save)
    print(f"PPO (Distrax) model parameters saved to {model_params_path}")
    
    # Save metrics as both npz and csv
    metrics_path_npz = os.path.join(results_dir, "training_metrics_ppo_distrax.npz")
    np.savez(metrics_path_npz, **training_metrics)
    print(f"Training metrics saved to {metrics_path_npz}")
    
    # Save metrics as CSV for easier comparison
    metrics_df = pd.DataFrame({
        'timesteps': training_metrics['timesteps'],
        'mean_rewards': training_metrics['mean_rewards'],
        'pg_losses': training_metrics['pg_losses'],
        'vf_losses': training_metrics['vf_losses'],
        'ent_losses': training_metrics['ent_losses']
    })
    metrics_path_csv = os.path.join(results_dir, "training_metrics_ppo_distrax.csv")
    metrics_df.to_csv(metrics_path_csv, index=False)
    print(f"Training metrics saved to {metrics_path_csv}")

    return trained_ppo_state, results_dir, training_metrics


def train_ppo_agent_jaxatari(config_dict: Dict[str, Any]) -> Tuple[TrainState, str, Dict[str, Any]]:
    env_type = config_dict.get("ENV_TYPE", "jaxatari")
    env_name = config_dict["ENV_NAME_JAXATARI"] if env_type == "jaxatari" else "Pong"

    print(f"Training PPO agent (Distrax base) with {env_type.upper()} environment (Game: {env_name})...")
    
    trained_ppo_state, training_metrics = train_ppo_with_jaxatari(config_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ppo_distrax_{env_type}_{env_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model parameters
    model_params_path = os.path.join(results_dir, "ppo_distrax_model_params.npz")
    params_dict_to_save = flax.serialization.to_state_dict(trained_ppo_state.params)
    np.savez(model_params_path, **params_dict_to_save)
    print(f"PPO (Distrax) model parameters saved to {model_params_path}")
    
    # Save metrics as both npz and csv
    metrics_path_npz = os.path.join(results_dir, "training_metrics_ppo_distrax.npz")
    np.savez(metrics_path_npz, **training_metrics)
    print(f"Training metrics saved to {metrics_path_npz}")
    
    # Save metrics as CSV for easier comparison
    metrics_df = pd.DataFrame({
        'timesteps': training_metrics['timesteps'],
        'mean_rewards': training_metrics['mean_rewards'],
        'pg_losses': training_metrics['pg_losses'],
        'vf_losses': training_metrics['vf_losses'],
        'ent_losses': training_metrics['ent_losses']
    })
    metrics_path_csv = os.path.join(results_dir, "training_metrics_ppo_distrax.csv")
    metrics_df.to_csv(metrics_path_csv, index=False)
    print(f"Training metrics saved to {metrics_path_csv}")
    
    return trained_ppo_state, results_dir, training_metrics


def load_ppo_params_from_npz(model_path: str) -> flax.core.FrozenDict:
    loaded_np = np.load(model_path, allow_pickle=True)
    # Convert numpy arrays to JAX arrays while preserving the nested structure
    def convert_to_jax(x):
        if isinstance(x, dict):
            return {k: convert_to_jax(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            return jnp.array(x)
        return x
    
    # Load and convert the parameters
    loaded_params_dict = {key: convert_to_jax(loaded_np[key].item()) for key in loaded_np.files}
    return flax.core.freeze(loaded_params_dict)

def evaluate_ppo_agent(
    agent_representation: Union[str, TrainState], 
    config_dict: Dict[str, Any], 
    num_episodes: int = 10, 
    eval_seed: int = 123,
    eval_env_type: Optional[str] = None
) -> Tuple[float, float]:
    """
    Evaluate a PPO agent on either OCAtari or JAX environment.
    
    Args:
        agent_representation: Either path to saved params or TrainState object
        config_dict: Configuration dictionary
        num_episodes: Number of episodes to evaluate
        eval_seed: Random seed for evaluation
        eval_env_type: Override environment type for evaluation ("ocatari" or "jax")
    """
    env_type = eval_env_type if eval_env_type is not None else config_dict.get("ENV_TYPE", "ocatari")
    env_name = config_dict["ENV_NAME_OCATARI"] if env_type == "ocatari" else "Pong"
    
    if config_dict.get("BUFFER_WINDOW", None) is None: # fix for old configs
        config_dict["BUFFER_WINDOW"] = config_dict["OCATARI_BUFFER_WINDOW"] 

    if env_type == "ocatari":
        eval_env = OCAtari(
            env_name=env_name,
            mode="ram", 
            hud=False, 
            render_mode="rgb_array",
            obs_mode="obj", 
            buffer_window_size=config_dict["BUFFER_WINDOW"],
            frameskip=config_dict["FRAMESKIP"],
            repeat_action_probability=config_dict["REPEAT_ACTION_PROBABILITY"]
        )
    else:  # JAX environment
        eval_env_base = jaxatari.make(env_name.lower())
        eval_env = AtariWrapper(eval_env_base, sticky_actions=True, frame_stack_size=config_dict["BUFFER_WINDOW"], frame_skip=config_dict["FRAMESKIP"])
        eval_env = ObjectCentricWrapper(eval_env)
        eval_env = FlattenObservationWrapper(eval_env)

    episode_rewards = []
    
    # Determine obs_shape_flat and action_dim for initializing network if loading params
    if env_type == "ocatari":
        _obs_temp, _ = eval_env.reset(seed=eval_seed)
        obs_shape_flat_eval = (np.prod(_obs_temp.shape),)
        action_dim_eval = eval_env.action_space.n
    else:
        _obs_temp, _ = eval_env.reset(key=jax.random.PRNGKey(eval_seed))
        obs_shape_flat_eval = (np.prod(_obs_temp.shape),)
        action_dim_eval = eval_env._env.action_space().n

    if isinstance(agent_representation, str):
        print(f"Loading PPO (Distrax) model params from: {agent_representation}")
        loaded_params = load_ppo_params_from_npz(agent_representation)
        
        dummy_rng = jax.random.PRNGKey(0)
        eval_train_state = create_ppo_train_state(dummy_rng, config_dict, obs_shape_flat_eval, action_dim_eval)
        current_agent_state = eval_train_state.replace(params=loaded_params)
        print("PPO (Distrax) model params loaded for evaluation.")
    elif isinstance(agent_representation, TrainState):
        current_agent_state = agent_representation
        print("Using provided PPO TrainState for evaluation.")
    else:
        raise ValueError("agent_representation must be a path (str) or TrainState object.")

    eval_rng_key = jax.random.PRNGKey(eval_seed)

    for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="ep"):
        if env_type == "ocatari":
            obs_stacked, _ = eval_env.reset(seed=eval_seed + episode)
            obs_norm_flat = normalize_observation_ocatari(obs_stacked).reshape(1, -1)
        else:
            obs, state = eval_env.reset(key=jax.random.PRNGKey(eval_seed + episode))
            obs_norm_flat = normalize_observation_jaxatari(obs, eval_env.observation_space()).reshape(1, -1)
        
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            pi_eval, _ = current_agent_state.apply_fn({'params': current_agent_state.params}, jnp.array(obs_norm_flat))
            action_agent_jnp = pi_eval.mode()
            action_agent = int(action_agent_jnp[0])

            if env_type == "ocatari":
                next_obs_stacked, reward, terminated, truncated, _ = eval_env.step(action_agent)
                next_obs_norm_flat = normalize_observation_ocatari(next_obs_stacked).reshape(1, -1)
                done = terminated or truncated
            else:
                next_obs, state, reward, done, _ = eval_env.step(state, action_agent)
                next_obs_norm_flat = normalize_observation_jaxatari(next_obs, eval_env.observation_space()).reshape(1, -1)

            episode_reward += reward
            obs_norm_flat = next_obs_norm_flat
            step_count += 1
            if step_count > 20000:
                print(f"Warning: Eval episode {episode+1} exceeded max steps.")
                break
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    if env_type == "ocatari":
        eval_env.close()
        
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    return mean_reward, std_reward

def plot_training_metrics(metrics: Dict[str, Any], save_path: str, env_name: str) -> None:
    if 'timesteps' not in metrics or 'mean_rewards' not in metrics:
        print("Metrics for plotting mean rewards not found. Skipping reward plot.")
        return

    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Evaluation Rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['timesteps'], metrics['mean_rewards'], label='Training Reward', color='green')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f'PPO Training vs Evaluation Rewards - {env_name}')
    plt.legend()
    plt.grid(True)

    # Plot 2: Policy Loss
    if 'pg_losses' in metrics:
        plt.subplot(2, 2, 2)
        plt.plot(metrics['timesteps'], metrics['pg_losses'], label='Policy Loss', color='blue')
        plt.xlabel('Timesteps')
        plt.ylabel('Policy Loss')
        plt.title(f'PPO Policy Loss - {env_name}')
        plt.legend()
        plt.grid(True)

    # Plot 3: Value Loss
    if 'vf_losses' in metrics:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['timesteps'], metrics['vf_losses'], label='Value Loss', color='red')
        plt.xlabel('Timesteps')
        plt.ylabel('Value Loss')
        plt.title(f'PPO Value Loss - {env_name}')
        plt.legend()
        plt.grid(True)

    # Plot 4: Entropy
    if 'ent_losses' in metrics:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['timesteps'], metrics['ent_losses'], label='Entropy', color='purple')
        plt.xlabel('Timesteps')
        plt.ylabel('Entropy')
        plt.title(f'PPO Policy Entropy - {env_name}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")
    plt.close()

def compare_agents(agent_paths: Dict[str, str], config_dict: Dict[str, Any], num_episodes: int = 100) -> None:
    """
    Compare multiple trained agents by evaluating them on both environments.
    
    Args:
        agent_paths: Dictionary mapping agent names to their parameter file paths
        config_dict: Configuration dictionary
        num_episodes: Number of episodes for evaluation
    """
    results = {}
    
    for agent_name, agent_path in agent_paths.items():
        print(f"\nEvaluating {agent_name}...")
        
        # Evaluate on OCAtari
        ocatari_mean, ocatari_std = evaluate_ppo_agent(
            agent_path, config_dict, num_episodes=num_episodes, eval_env_type="ocatari"
        )
        
        # Evaluate on JAX
        jax_mean, jax_std = evaluate_ppo_agent(
            agent_path, config_dict, num_episodes=num_episodes, eval_env_type="jax"
        )
        
        results[agent_name] = {
            "ocatari_mean": ocatari_mean,
            "ocatari_std": ocatari_std,
            "jax_mean": jax_mean,
            "jax_std": jax_std
        }
    
    # Print comparison table
    print("\nAgent Comparison Results:")
    print("-" * 80)
    print(f"{'Agent Name':<20} {'OCAtari Mean':<15} {'OCAtari Std':<15} {'JAX Mean':<15} {'JAX Std':<15}")
    print("-" * 80)
    
    for agent_name, metrics in results.items():
        print(f"{agent_name:<20} {metrics['ocatari_mean']:<15.2f} {metrics['ocatari_std']:<15.2f} "
              f"{metrics['jax_mean']:<15.2f} {metrics['jax_std']:<15.2f}")


def visualize_agent(agent_path: str, config_dict: Dict[str, Any], num_episodes: int = 10) -> None:
    """
    Visualize a trained agent playing a game.
    
    Args:
        agent_path: Path to the agent's parameter file
        config_dict: Configuration dictionary
        num_episodes: Number of episodes to visualize
    """
    env_type = config_dict.get("ENV_TYPE", "ocatari")
    env_name = config_dict["ENV_NAME_OCATARI"] if env_type == "ocatari" else "Pong"
    
    if config_dict.get("BUFFER_WINDOW", None) is None: # fix for old configs
        config_dict["BUFFER_WINDOW"] = config_dict["OCATARI_BUFFER_WINDOW"] 
    
    # Initialize pygame for visualization
    pygame.init()
    pygame_screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    pygame.display.set_caption(f"Agent Visualization - {env_name}")
    clock = pygame.time.Clock()
    
    # Initialize video recording
    import cv2
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"agent_visualization_{env_name}_{timestamp}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, config_dict.get("VIZ_FPS", 30), 
                                     (WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    except Exception as e:
        print(f"Could not initialize video writer: {e}. Video will not be saved.")
        video_writer = None
    
    # Load the agent
    loaded_params = load_ppo_params_from_npz(agent_path)
    
    # Initialize environment
    if env_type == "ocatari":
        vis_env = OCAtari(
            env_name=env_name,
            mode="ram", 
            hud=False, 
            render_mode="rgb_array",
            obs_mode="obj", 
            buffer_window_size=config_dict["BUFFER_WINDOW"],
            frameskip=config_dict["FRAMESKIP"],
            repeat_action_probability=config_dict["REPEAT_ACTION_PROBABILITY"]
        )
        obs_shape_flat = (np.prod(vis_env.observation_space.shape),)
        action_dim = vis_env.action_space.n
    else:  # JAX environment
        vis_env_base = jaxatari.make(env_name.lower())
        vis_env = AtariWrapper(vis_env_base, sticky_actions=True, frame_stack_size=config_dict["BUFFER_WINDOW"], frame_skip=config_dict["FRAMESKIP"])
        vis_env = ObjectCentricWrapper(vis_env)
        vis_env = FlattenObservationWrapper(vis_env)
        obs_shape_flat = vis_env.reset(key=jax.random.PRNGKey(0))[0].shape
        action_dim = vis_env._env.action_space().n
    
    # Initialize agent
    dummy_rng = jax.random.PRNGKey(0)
    eval_train_state = create_ppo_train_state(dummy_rng, config_dict, obs_shape_flat, action_dim)
    current_agent_state = eval_train_state.replace(params=loaded_params)
    
    # Initialize visualization
    agent_key = jax.random.PRNGKey(config_dict["SEED"] + 777)
    
    if env_type == "ocatari":
        obs_viz, _ = vis_env.reset(seed=config_dict["SEED"])
        print(obs_viz)
        obs_viz_norm_flat = normalize_observation_ocatari(obs_viz).reshape(1, -1)
        current_frame = vis_env.render()
    else:
        print("Resetting environment...")
        vis_reset_key, agent_key = jax.random.split(agent_key)
        obs_viz_raw, state_viz = vis_env.reset(key=vis_reset_key)
        obs_viz_norm_flat = normalize_observation_jaxatari(obs_viz_raw, vis_env.observation_space()).reshape(1, -1)
        print("Environment reset complete")
    
    total_reward_viz = 0
    running = True
    episode_count = 0
        
    while running and episode_count < num_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        
        if not running:
            break
            
        # Get action from agent
        pi_viz, _ = current_agent_state.apply_fn({'params': current_agent_state.params}, obs_viz_norm_flat)
        action_viz = pi_viz.mode()  # Use mode for deterministic visualization
        
        # Step environment
        if env_type == "ocatari":
            next_obs_viz, reward_viz, terminated, truncated, _ = vis_env.step(int(action_viz[0]))
            next_obs_viz_norm_flat = normalize_observation_ocatari(next_obs_viz).reshape(1, -1)
            done_viz = terminated or truncated
            current_frame = vis_env.render()
            # Convert frame to pygame surface and display
            # Transpose frame from (H, W, C) to (W, H, C) for pygame
            frame_transposed = np.transpose(current_frame, (1, 0, 2))
            frame_surface = pygame.Surface(frame_transposed.shape[:2])
            pygame.pixelcopy.array_to_surface(frame_surface, frame_transposed)
            frame_surface_scaled = pygame.transform.scale(frame_surface, (WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
            pygame_screen.blit(frame_surface_scaled, (0, 0))
            pygame.display.flip()
            
            # Save frame to video
            if video_writer:
                view = pygame.surfarray.array3d(frame_surface_scaled)
                view = view.transpose([1, 0, 2])
                frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
        else:
            viz_step_key, agent_key = jax.random.split(agent_key)
            next_obs_viz_raw, state_viz, reward_viz, done_viz, _ = vis_env.step(
                state_viz, action_viz[0] if action_viz.ndim > 0 else action_viz
            )
            next_obs_viz_norm_flat = normalize_observation_jaxatari(next_obs_viz_raw, vis_env.observation_space()).reshape(1, -1)
            # Render the game state using JaxPongRenderer directly
            raster = vis_env_base.render(state_viz.env_state)
            # Update pygame display with the rendered frame 
            jax_rendering_utils.update_pygame(pygame_screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
            
            # Save frame to video
            if video_writer:
                view = pygame.surfarray.array3d(pygame_screen)
                view = view.transpose([1, 0, 2])
                frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
        
        total_reward_viz += reward_viz.item() if isinstance(reward_viz, jnp.ndarray) else reward_viz
        obs_viz_norm_flat = next_obs_viz_norm_flat
        
        if done_viz:
            print(f"Episode {episode_count + 1}/{num_episodes} finished with total reward {total_reward_viz:.2f}")
            episode_count += 1
            total_reward_viz = 0
            
            if env_type == "ocatari":
                obs_viz, _ = vis_env.reset(seed=config_dict["SEED"] + episode_count)
                obs_viz_norm_flat = normalize_observation_ocatari(obs_viz).reshape(1, -1)
                current_frame = vis_env.render()
            else:
                vis_reset_key, agent_key = jax.random.split(agent_key)
                obs_viz_raw, state_viz = vis_env.reset(key=vis_reset_key)
                obs_viz_norm_flat = normalize_observation_jaxatari(obs_viz_raw, vis_env.observation_space()).reshape(1, -1)
    
    # Release video writer if it exists
    if video_writer:
        video_writer.release()
        print(f"\nVideo saved as: {os.path.abspath(video_filename)}")
    
    if env_type == "ocatari":
        vis_env.close()
    pygame.quit()
    print("Visualization finished.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Script to compare PPO agents trained on either OCAtari or JAXAtari')
    parser.add_argument('--mode', type=str, choices=['train-ocatari', 'train-jaxatari', 'eval', 'visualize', 'compare'], required=True,
                      help='Mode: train-ocatari, train-jaxatari, eval, or compare agents')
    parser.add_argument('--env_type', type=str, choices=['ocatari', 'jax'], default='ocatari',
                      help='Environment type for training/evaluation')
    parser.add_argument('--model_path', type=str, help='Path to model parameters for evaluation')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--compare_paths', type=str, nargs='+', help='Paths to models for comparison')
    
    args = parser.parse_args()
    
    current_config = ppo_config_distrax.copy()
    current_config["ENV_TYPE"] = args.env_type
    
    if args.mode == "train-ocatari":
        print(f"--- Starting PPO (Distrax base) Agent Training ({current_config['ENV_TYPE']}) ---")
        print(f"Configuration: {current_config}")
        
        trained_ppo_state_obj, ppo_results_dir, training_metrics = train_ppo_agent_ocatari(current_config)
        
        plots_path = os.path.join(ppo_results_dir, "training_plots_ppo_distrax.png")
        plot_training_metrics(training_metrics, plots_path, current_config['ENV_NAME_OCATARI'])
        
        print("\n--- Evaluating trained agent ---")
        mean_reward, std_reward = evaluate_ppo_agent(
            trained_ppo_state_obj, current_config, num_episodes=args.num_episodes
        )
        print(f"Trained Agent - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif args.mode == "train-jaxatari":
        print(f"--- Starting PPO (Distrax base) Agent Training ({current_config['ENV_TYPE']}) ---")
        print(f"Configuration: {current_config}")
        
        trained_ppo_state_obj, ppo_results_dir, training_metrics = train_ppo_agent_jaxatari(current_config)

        plots_path = os.path.join(ppo_results_dir, "training_plots_ppo_distrax.png")
        plot_training_metrics(training_metrics, plots_path, current_config['ENV_NAME_OCATARI'])
        
        print("\n--- Evaluating trained agent ---")
        mean_reward, std_reward = evaluate_ppo_agent(
            trained_ppo_state_obj, current_config, num_episodes=args.num_episodes
        )
        print(f"Trained Agent - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif args.mode == "eval":
        if not args.model_path:
            raise ValueError("Model path must be provided for evaluation mode")
            
        print(f"--- Evaluating agent from {args.model_path} ---")
        mean_reward, std_reward = evaluate_ppo_agent(
            args.model_path, current_config, num_episodes=args.num_episodes
        )
        print(f"Agent Evaluation - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif args.mode == "compare":
        if not args.compare_paths:
            if args.model_path: 
                args.compare_paths = [args.model_path]
            else:
                raise ValueError("Model paths must be provided for comparison mode")
            
        agent_paths = {f"agent_{i}": path for i, path in enumerate(args.compare_paths)}
        compare_agents(agent_paths, current_config, num_episodes=args.num_episodes)

    elif args.mode == "visualize":
        if not args.model_path:
            raise ValueError("Model path must be provided for visualization mode")
            
        visualize_agent(args.model_path, current_config, num_episodes=args.num_episodes)

if __name__ == "__main__":
    main()