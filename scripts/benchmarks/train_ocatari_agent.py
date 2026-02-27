import faulthandler
faulthandler.enable()

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any
from collections import deque
import pygame
from tqdm import tqdm
from ocatari.core import OCAtari
from scripts.benchmarks.ppo_agent_old import ActorCritic, create_ppo_train_state, ppo_loss_fn, ppo_update_minibatch

# --- Observation Normalization (From previous versions) ---
def normalize_observation_ocatari(obs: np.ndarray) -> np.ndarray:
    single_frame_max_values = np.array([160, 210, 160, 210, 160, 210], dtype=np.float32)
    return 2 * (obs / single_frame_max_values) - 1.0

def train_ppo_with_ocatari(config: Dict[str, Any]):
    np.random.seed(config["SEED"]) 
    main_rng = jax.random.PRNGKey(config["SEED"])
    
    ocatari_game_name = config["ENV_NAME_OCATARI"] 
    ocatari_buffer_window = config["BUFFER_WINDOW"]
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE_CALCULATED"] = ( 
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    print(f"Using OCAtari environment: {ocatari_game_name} with obs_mode='obj'")
    envs = [OCAtari(env_name=ocatari_game_name,
                   mode="ram", hud=False, obs_mode="obj",
                   buffer_window_size=ocatari_buffer_window,
                   frameskip=config.get("FRAMESKIP", 4), 
                   repeat_action_probability=config.get("REPEAT_ACTION_PROBABILITY", 0.0),
                   max_episode_length=config.get("MAX_EPISODE_LENGTH", 1_000)
                   ) for _ in range(config["NUM_ENVS"])]

    obs_np_stacked_list = []
    for i, env in enumerate(envs):
        obs_np_stacked, _ = env.reset(seed=config["SEED"] + i) 
        obs_np_stacked_list.append(obs_np_stacked)
    
    current_obs_np_stacked = np.stack(obs_np_stacked_list)
    obs_shape_stacked = current_obs_np_stacked.shape[1:]
    obs_shape_flat = (np.prod(obs_shape_stacked),)

    action_dim = envs[0].action_space.n

    agent_key, init_key = jax.random.split(main_rng)
    train_state = create_ppo_train_state(init_key, config, obs_shape_flat, action_dim)

    print(f"OCAtari observation (stacked obj): {obs_shape_stacked}")
    print(f"Agent observation input (flattened): {obs_shape_flat}, Action dimension: {action_dim}")
    print(f"Total updates: {config['NUM_UPDATES']}, Minibatch size: {config['MINIBATCH_SIZE_CALCULATED']}")

    rollout_obs_flat = np.zeros((config["NUM_STEPS"], config["NUM_ENVS"]) + obs_shape_flat, dtype=np.float32)
    rollout_actions = np.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=np.int32)
    rollout_log_probs = np.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=np.float32)
    rollout_rewards = np.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=np.float32)
    rollout_dones = np.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=np.bool_)
    rollout_values = np.zeros((config["NUM_STEPS"], config["NUM_ENVS"]), dtype=np.float32)

    episode_rewards_deque = deque(maxlen=100 * config["NUM_ENVS"]) 
    all_episode_rewards_history = [] 
    all_mean_rewards_history, all_timesteps_history = [], []
    all_pg_loss_hist, all_vf_loss_hist, all_ent_hist = [], [], []
    current_episode_rewards_np = np.zeros(config["NUM_ENVS"]) 
    
    current_obs_np_norm_flat = normalize_observation_ocatari(current_obs_np_stacked).reshape(config["NUM_ENVS"], -1)

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
            
            rollout_obs_flat[step_idx] = current_obs_np_norm_flat 
            
            action_rng_key, rollout_sample_key = jax.random.split(rollout_sample_key)
            
            pi_agent, value_agent_jnp = train_state.apply_fn({'params': train_state.params}, 
                                                           jnp.array(current_obs_np_norm_flat))
            actions_agent_jnp = pi_agent.sample(seed=action_rng_key)
            log_probs_jnp = pi_agent.log_prob(actions_agent_jnp)
            
            actions_agent_np = np.array(actions_agent_jnp)
            log_probs_np = np.array(log_probs_jnp)
            values_np = np.array(value_agent_jnp)

            rollout_actions[step_idx] = actions_agent_np
            rollout_log_probs[step_idx] = log_probs_np
            rollout_values[step_idx] = values_np
            
            actions_env_np = actions_agent_np

            next_obs_np_stacked_list = [] 
            rewards_step_np = np.zeros(config["NUM_ENVS"])
            dones_step_np = np.zeros(config["NUM_ENVS"], dtype=bool)

            for env_idx in range(config["NUM_ENVS"]):
                env = envs[env_idx]
                action_env = actions_env_np[env_idx]
                
                next_obs_stacked_single, reward, truncated, terminated, _ = env.step(int(action_env))
                
                rewards_step_np[env_idx] = reward
                dones_step_np[env_idx] = terminated or truncated
                current_episode_rewards_np[env_idx] += reward

                if dones_step_np[env_idx]:
                    episode_rewards_deque.append(current_episode_rewards_np[env_idx])
                    all_episode_rewards_history.append(current_episode_rewards_np[env_idx])
                    current_episode_rewards_np[env_idx] = 0
                    obs_reset_stacked, _ = env.reset(seed=config["SEED"] + total_env_steps_so_far + env_idx) 
                    next_obs_np_stacked_list.append(obs_reset_stacked)
                else:
                    next_obs_np_stacked_list.append(next_obs_stacked_single)
            
            # Update rollout reward tracking
            rollout_rewards_sum += np.sum(rewards_step_np)
            rollout_rewards_count += config["NUM_ENVS"]
            current_avg_reward = rollout_rewards_sum / rollout_rewards_count if rollout_rewards_count > 0 else 0
            
            # Calculate average from recent episodes
            recent_avg_reward = np.mean(list(episode_rewards_deque)) if episode_rewards_deque else 0
            worst_score = -21.0
            best_score = 21.0
            distance_from_worst = ((recent_avg_reward - worst_score) / (best_score - worst_score)) * 100
            distance_from_worst = max(0, min(100, distance_from_worst))  # Clamp between 0 and 100
            
            rollout_rewards[step_idx] = rewards_step_np
            rollout_dones[step_idx] = dones_step_np
            
            current_obs_np_stacked = np.stack(next_obs_np_stacked_list)
            current_obs_np_norm_flat = normalize_observation_ocatari(current_obs_np_stacked).reshape(config["NUM_ENVS"], -1)

            rollout_pbar.update(1)
            rollout_pbar.set_postfix({
                "env_steps": total_env_steps_so_far,
                "avg_reward": f"{current_avg_reward:.2f}",
                "recent_avg": f"{recent_avg_reward:.2f}",
                "dist_from_worst": f"{distance_from_worst:.1f}%"
            })
        
        rollout_pbar.close()

        _, last_val_jnp = train_state.apply_fn({'params': train_state.params}, 
                                             jnp.array(current_obs_np_norm_flat))
        last_val_np = np.array(last_val_jnp)

        advantages_np = np.zeros_like(rollout_rewards)
        last_gae_lam = 0
        for t in reversed(range(config["NUM_STEPS"])):
            if t == config["NUM_STEPS"] - 1:
                next_non_terminal = 1.0 - rollout_dones[t] 
                next_values = last_val_np 
            else:
                next_non_terminal = 1.0 - rollout_dones[t] 
                next_values = rollout_values[t+1] 
            
            delta = rollout_rewards[t] + config["GAMMA"] * next_values * next_non_terminal - rollout_values[t]
            advantages_np[t] = last_gae_lam = delta + config["GAMMA"] * config["GAE_LAMBDA"] * next_non_terminal * last_gae_lam
        
        returns_np = advantages_np + rollout_values

        b_obs = rollout_obs_flat.reshape((-1,) + obs_shape_flat)
        b_actions = rollout_actions.reshape(-1)
        b_log_probs_old = rollout_log_probs.reshape(-1)
        b_values_old = rollout_values.reshape(-1) 
        b_advantages = advantages_np.reshape(-1)
        b_returns = returns_np.reshape(-1)

        total_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        
        for epoch in range(config["UPDATE_EPOCHS"]):
            permutation = jax.random.permutation(update_perm_key, total_batch_size)
            agent_key, update_perm_key = jax.random.split(agent_key) 
            
            for start_idx in range(0, total_batch_size, config["MINIBATCH_SIZE_CALCULATED"]):
                end_idx = start_idx + config["MINIBATCH_SIZE_CALCULATED"]
                mb_indices = permutation[start_idx:end_idx]

                train_state, loss, aux_info = ppo_update_minibatch(
                    train_state,
                    jnp.array(b_obs[mb_indices]),
                    jnp.array(b_actions[mb_indices]),
                    jnp.array(b_log_probs_old[mb_indices]),
                    jnp.array(b_values_old[mb_indices]), 
                    jnp.array(b_advantages[mb_indices]),
                    jnp.array(b_returns[mb_indices]),
                    config
                )
            
        if update_idx % config.get("LOG_INTERVAL_UPDATES", 10) == 0:
            mean_ep_reward = np.mean(list(episode_rewards_deque)) if episode_rewards_deque else 0
            all_mean_rewards_history.append(mean_ep_reward)
            all_pg_loss_hist.append(float(aux_info["pg_loss"]))
            all_vf_loss_hist.append(float(aux_info["vf_loss"]))
            all_ent_hist.append(float(aux_info["entropy"]))
            current_total_steps_log = update_idx * config["NUM_STEPS"] * config["NUM_ENVS"]
            all_timesteps_history.append(current_total_steps_log)

            # Update main progress bar with metrics
            pbar.set_postfix({
                "mean_reward": f"{mean_ep_reward:.2f}",
                "pg_loss": f"{aux_info['pg_loss']:.3f}",
                "vf_loss": f"{aux_info['vf_loss']:.3f}",
                "entropy": f"{aux_info['entropy']:.3f}"
            })

        pbar.update(1)

    pbar.close()
    for env in envs: env.close()
    print("Training finished.")

    if config.get("VISUALIZE_AFTER_TRAINING", False):
        print("\nStarting visualization...")
        viz_env = OCAtari(env_name=ocatari_game_name, mode="ram", hud=False,
                          obs_mode="obj", buffer_window_size=ocatari_buffer_window, 
                          render_mode="rgb_array",
                          frameskip=config.get("FRAMESKIP", 4),
                          repeat_action_probability=config.get("REPEAT_ACTION_PROBABILITY", 0.0))
        pygame.init()
        obs_viz_stacked, _ = viz_env.reset()
        frame = viz_env.render()
        
        screen = None
        video_writer = None
        if frame is not None:
            frame_shape_viz = frame.shape[:2] 
            screen_width, screen_height = frame_shape_viz[1] * 3, frame_shape_viz[0] * 3
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption(f"{ocatari_game_name} - Trained PPO Agent (Distrax)")

            # Add video recording setup
            if config.get("SAVE_VIZ_VIDEO", True):
                import cv2
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"ppo_trained_{ocatari_game_name}_{timestamp}.mp4"
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, config.get("VIZ_FPS", 30), (screen_width, screen_height))
                except Exception as e:
                    print(f"Could not initialize video writer: {e}. Video will not be saved.")
                    video_writer = None
        else:
            print("Warning: Frame rendering failed during viz setup.")

        clock = pygame.time.Clock()
        obs_viz_norm_flat = normalize_observation_ocatari(obs_viz_stacked).reshape(1, -1) 
        total_reward_viz = 0
        running = True
        
        input("Press Enter to start visualization...")
        agent_key, viz_action_key = jax.random.split(agent_key)

        for _ in range(config.get("VIZ_STEPS", 1000)): 
            if not running: break
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            if not running: break
            
            pi_viz, _ = train_state.apply_fn({'params': train_state.params}, 
                                             jnp.array(obs_viz_norm_flat))
            action_agent_viz = pi_viz.mode() 
            
            action_env_viz = action_agent_viz[0] 
            
            next_obs_viz_stacked, reward_viz, terminated_viz, truncated_viz, _ = viz_env.step(action_env_viz)
            total_reward_viz += reward_viz
            
            frame = viz_env.render()
            if frame is not None and screen is not None:
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                scaled_surface = pygame.transform.scale(frame_surface, (screen.get_width(), screen.get_height()))
                screen.blit(scaled_surface, (0, 0))
                pygame.display.flip()
                
                # Add video recording
                if video_writer:
                    view = pygame.surfarray.array3d(scaled_surface)
                    view = view.transpose([1, 0, 2])
                    frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
            
            obs_viz_norm_flat = normalize_observation_ocatari(next_obs_viz_stacked).reshape(1, -1)
            clock.tick(config.get("VIZ_FPS", 30))
            if terminated_viz or truncated_viz:
                print(f"Visualization Episode finished with total reward {total_reward_viz}")
                obs_viz_stacked, _ = viz_env.reset()
                obs_viz_norm_flat = normalize_observation_ocatari(obs_viz_stacked).reshape(1,-1)
                total_reward_viz = 0
        
        # Release video writer if it exists
        if video_writer:
            video_writer.release()
            print(f"\nVideo saved as: {os.path.abspath(video_filename)}")
        
        viz_env.close()
        if screen is not None:
            pygame.quit()
        print("Visualization finished.")
    
    training_results_metrics = {
        "timesteps": all_timesteps_history,
        "mean_rewards": all_mean_rewards_history,
        "pg_losses": all_pg_loss_hist,
        "vf_losses": all_vf_loss_hist,
        "ent_losses": all_ent_hist,
        "all_episode_rewards": all_episode_rewards_history
    }
    return train_state, training_results_metrics


# helper function
def visualize_random_actions(config: Dict[str, Any]):
    import cv2 
    import os
    from datetime import datetime

    ocatari_game_name = config["ENV_NAME_OCATARI"]
    seed = config["SEED"]
    ocatari_buffer_window = config["OCATARI_BUFFER_WINDOW"]
    frameskip = config.get("FRAMESKIP", 4)
    repeat_action_probability = config.get("REPEAT_ACTION_PROBABILITY", 0.0)

    viz_env = OCAtari(env_name=ocatari_game_name, mode="ram", hud=False,
                     obs_mode="obj", buffer_window_size=ocatari_buffer_window,
                     render_mode="rgb_array", frameskip=frameskip,
                     repeat_action_probability=repeat_action_probability)

    action_dim = viz_env.action_space.n

    pygame.init()
    obs_viz_stacked, _ = viz_env.reset(seed=seed)
    frame = viz_env.render()
    
    screen = None
    video_writer = None
    if frame is not None:
        frame_shape_viz = frame.shape[:2]
        screen_width, screen_height = frame_shape_viz[1] * 3, frame_shape_viz[0] * 3
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"{ocatari_game_name} - Random Actions (PPO Distrax Base)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"random_actions_{ocatari_game_name}_{timestamp}.mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(video_filename, fourcc, 1.0, (screen_width, screen_height)) 
        except Exception as e:
            print(f"Could not initialize video writer: {e}. Video will not be saved.")
            video_writer = None
    else:
        print("Warning: Frame rendering failed during viz setup. Cannot visualize or record.")
        return

    clock = pygame.time.Clock()
    running = True
    
    print("\nStarting visualization with random actions...")
    input("Press Enter to start...")

    for step in range(config.get("VIZ_STEPS_RANDOM", 50)):
        if not running: break
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        if not running: break

        action_env_viz = np.random.randint(0, action_dim)
        
        next_obs_viz_stacked, reward, terminated, truncated, _ = viz_env.step(action_env_viz)
        
        frame = viz_env.render()
        if frame is not None and screen is not None:
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(frame_surface, (screen.get_width(), screen.get_height()))
            screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            
            if video_writer:
                view = pygame.surfarray.array3d(scaled_surface)
                view = view.transpose([1, 0, 2])
                frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
        print(f"Step {step}: Agent Action {action_env_viz}, Reward: {reward}")
        
        clock.tick(1) 
        
        if terminated or truncated:
            print(f"Episode finished at step {step}. Total reward this ep: {viz_env.get_total_reward()}") 
            obs_viz_stacked, _ = viz_env.reset(seed=seed + step + 1) 
            
    if video_writer:
        video_writer.release()
        print(f"\nVideo saved as: {os.path.abspath(video_filename)}")
            
    viz_env.close()
    if screen is not None:
        pygame.quit()
    print("Random action visualization finished.")



if __name__ == '__main__':
    test_config_ppo_distrax = {
        "ENV_NAME_OCATARI": "Pong", 
        "OCATARI_BUFFER_WINDOW": 4,
        "SEED": 42,
        "FRAMESKIP": 4,
        "REPEAT_ACTION_PROBABILITY": 0.0,
        "VIZ_STEPS_RANDOM": 10, 
        "NUM_UPDATES": 1, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1, "LR": 0.001, "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 1.0, "ACTIVATION": "tanh", "NUM_STEPS":1, "NUM_ENVS":1, "TOTAL_TIMESTEPS":1 
    }
    print("Starting PPO (Distrax base) random action visualization...")
    try:
        visualize_random_actions(test_config_ppo_distrax)
        print("PPO (Distrax base) random action visualization completed.")
    except ImportError as e:
        print(f"Import Error: {e}. Make sure all dependencies (e.g., opencv-python) are installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()