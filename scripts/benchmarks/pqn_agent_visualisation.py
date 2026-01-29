"""
Visualization script for trained PQN agents.
Loads a trained agent and renders it playing in pygame.

Usage:
    python pqn_agent_visualisation.py --model ./models/Seaquest/pqn_Seaquest_oc_seed0_vmap0.safetensors
    python pqn_agent_visualisation.py --model ./models/Pong/pqn_Pong_pixel_seed0_vmap0.safetensors --fps 60
"""

import argparse
import sys
import os

import pygame
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper, PixelObsWrapper, LogWrapper, NormalizeObservationWrapper

# Import network and utilities from pqn_agent
from pqn_agent import QNetwork, CNN
from train_utils import load_params

UPSCALE_FACTOR = 4


def load_config(config_path: str = None, model_path: str = None):
    """Load agent configuration from YAML file."""
    # Try to find config file
    if config_path and os.path.exists(config_path):
        pass
    elif model_path:
        # Try to find config next to model file
        model_dir = os.path.dirname(model_path)
        # Look for config files in the same directory
        for f in os.listdir(model_dir):
            if f.endswith("_config.yaml"):
                config_path = os.path.join(model_dir, f)
                print(f"Found config file: {config_path}")
                break
    
    if config_path and os.path.exists(config_path):
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.load(config_path)
            return OmegaConf.to_container(config)
        except ImportError:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    return None


def update_pygame(window, image, upscale_factor, native_w, native_h):
    """Update pygame window with rendered image."""
    image = np.array(image, dtype=np.uint8)
    # Ensure image is in the correct format (H, W, 3)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    scaled_surface = pygame.transform.scale(
        surface, (native_w * upscale_factor, native_h * upscale_factor)
    )
    window.blit(scaled_surface, (0, 0))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained PQN agent playing a JAXAtari game."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (safetensors format).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to agent config YAML file (optional, will try to auto-detect from model path).",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Game name (optional, will try to detect from config or model path).",
    )
    parser.add_argument(
        "--mods",
        nargs='+',
        type=str,
        default=None,
        help="Mods to apply to the environment.",
    )
    parser.add_argument(
        "--object_centric",
        action="store_true",
        default=None,
        help="Force object-centric mode (default: auto-detect from config or filename).",
    )
    parser.add_argument(
        "--pixel",
        action="store_true",
        help="Force pixel-based mode (default: auto-detect from config or filename).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for visualization.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = infinite).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode - print actions and rewards.",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.model)
    
    # Determine game name
    game_name = args.game
    if game_name is None and config:
        game_name = config.get("ENV_NAME", None)
    if game_name is None:
        # Try to extract from model path
        model_basename = os.path.basename(args.model)
        # Format: pqn_GameName_oc_seed0_vmap0.safetensors
        parts = model_basename.split("_")
        if len(parts) >= 2:
            game_name = parts[1]
    if game_name is None:
        print("Error: Could not determine game name. Please specify with --game")
        sys.exit(1)
    
    print(f"Game: {game_name}")

    # Determine if object-centric
    object_centric = args.object_centric
    if args.pixel:
        object_centric = False
    elif object_centric is None:
        if config:
            object_centric = config.get("OBJECT_CENTRIC", False)
        elif "_oc_" in args.model:
            object_centric = True
        elif "_pixel_" in args.model:
            object_centric = False
        else:
            object_centric = True  # Default
    
    print(f"Object-centric: {object_centric}")

    # Load model parameters
    print(f"Loading model from: {args.model}")
    params = load_params(args.model)
    
    batch_stats_path = args.model.replace(".safetensors", "_bs.safetensors")
    if os.path.exists(batch_stats_path):
        batch_stats = load_params(batch_stats_path)
        print(f"Loaded batch stats from: {batch_stats_path}")
    else:
        print(f"Warning: Batch stats file not found at {batch_stats_path}")
        batch_stats = {}

    # Build mods config
    mods_config = args.mods if args.mods else []
    if config and config.get("MOD_NAME"):
        mods = config["MOD_NAME"]
        if mods and not args.mods:
            mods_config = mods if isinstance(mods, list) else [mods]

    # Create environment
    env = jaxatari.make(game_name.lower(), mods_config=mods_config)
    renderer = env.renderer

    # Apply wrappers (matching training setup)
    env = AtariWrapper(
        env, 
        episodic_life=False,  # Don't end on life loss for visualization
        frame_skip=4, 
        frame_stack_size=4, 
        sticky_actions=True, 
        max_pooling=True, 
        clip_reward=False,  # Don't clip rewards for accurate reporting
        noop_reset=30, 
        max_episode_length=18000
    )
    
    if object_centric:
        env = ObjectCentricWrapper(env)
        env = FlattenObservationWrapper(env)
    else:
        grayscale = config.get("PIXEL_GRAYSCALE", True) if config else True
        resize_shape = config.get("PIXEL_RESIZE_SHAPE", [84, 84]) if config else [84, 84]
        use_native = config.get("USE_NATIVE_DOWNSCALING", False) if config else False
        env = PixelObsWrapper(
            env, 
            do_pixel_resize=True, 
            pixel_resize_shape=resize_shape, 
            grayscale=grayscale, 
            use_native_downscaling=use_native
        )
    
    env = NormalizeObservationWrapper(env)
    env = LogWrapper(env)

    # Get network hyperparameters from config or use defaults
    hidden_size = config.get("HIDDEN_SIZE", 256) if config else 256
    num_layers = config.get("NUM_LAYERS", 3) if config else 3
    norm_type = config.get("NORM_TYPE", "layer_norm") if config else "layer_norm"
    norm_input = config.get("NORM_INPUT", False) if config else False

    # Create network
    network = QNetwork(
        action_dim=env.action_space().n,
        hidden_size=hidden_size,
        num_layers=num_layers,
        norm_type=norm_type,
        norm_input=norm_input,
        object_centric=object_centric,
    )
    
    print(f"Network: hidden_size={hidden_size}, num_layers={num_layers}, norm_type={norm_type}")

    # JIT-compile the forward pass
    @jax.jit
    def get_action(obs):
        q_vals = network.apply(
            {"params": params, "batch_stats": batch_stats},
            obs[None, ...],  # Add batch dimension
            train=False,
        )
        return jnp.argmax(q_vals, axis=-1)[0]

    # JIT environment functions
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    # Initialize pygame
    pygame.init()
    pygame.display.set_caption(f"PQN Agent - {game_name}")
    
    # Get render shape from a test render
    master_key = jrandom.PRNGKey(args.seed)
    reset_key = jrandom.fold_in(master_key, 0)
    obs, env_state = jitted_reset(reset_key)
    
    # Get underlying state for rendering
    state_for_render = env_state
    while hasattr(state_for_render, 'atari_state'):
        state_for_render = state_for_render.atari_state
    if hasattr(state_for_render, 'env_state'):
        state_for_render = state_for_render.env_state
    
    test_frame = renderer.render(state_for_render)
    render_shape = test_frame.shape[:2]
    
    window = pygame.display.set_mode(
        (render_shape[1] * UPSCALE_FACTOR, render_shape[0] * UPSCALE_FACTOR)
    )
    clock = pygame.time.Clock()

    # Main loop
    running = True
    pause = False
    frame_by_frame = False
    next_frame = False
    episode_count = 0
    reset_counter = 1
    total_reward = 0.0
    step_count = 0

    print("\nControls:")
    print("  P - Pause/Resume")
    print("  F - Toggle frame-by-frame mode")
    print("  N - Next frame (when in frame-by-frame mode)")
    print("  R - Reset episode")
    print("  ESC/Q - Quit")
    print("\nStarting visualization...\n")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    pause = not pause
                    print("Paused" if pause else "Resumed")
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    print(f"Frame-by-frame: {'ON' if frame_by_frame else 'OFF'}")
                elif event.key == pygame.K_n:
                    next_frame = True
                elif event.key == pygame.K_r:
                    reset_key = jrandom.fold_in(master_key, reset_counter)
                    obs, env_state = jitted_reset(reset_key)
                    reset_counter += 1
                    total_reward = 0.0
                    step_count = 0
                    print("Episode reset")

        # Skip if paused (unless next frame requested)
        if pause or (frame_by_frame and not next_frame):
            # Still render current frame
            state_for_render = env_state
            while hasattr(state_for_render, 'atari_state'):
                state_for_render = state_for_render.atari_state
            if hasattr(state_for_render, 'env_state'):
                state_for_render = state_for_render.env_state
            
            frame = renderer.render(state_for_render)
            update_pygame(window, frame, UPSCALE_FACTOR, render_shape[1], render_shape[0])
            clock.tick(args.fps)
            continue
        
        next_frame = False

        # Get action from agent
        action = get_action(obs)
        
        if args.verbose:
            print(f"Step {step_count}: action={int(action)}", end="")

        # Step environment
        obs, env_state, reward, done, info = jitted_step(env_state, action)
        total_reward += float(reward)
        step_count += 1
        
        if args.verbose:
            print(f", reward={float(reward):.1f}, total={total_reward:.1f}")

        # Render
        state_for_render = env_state
        while hasattr(state_for_render, 'atari_state'):
            state_for_render = state_for_render.atari_state
        if hasattr(state_for_render, 'env_state'):
            state_for_render = state_for_render.env_state
        
        frame = renderer.render(state_for_render)
        update_pygame(window, frame, UPSCALE_FACTOR, render_shape[1], render_shape[0])
        clock.tick(args.fps)

        # Handle episode end
        if done:
            episode_count += 1
            print(f"Episode {episode_count} finished: steps={step_count}, return={total_reward:.1f}")
            
            if args.num_episodes > 0 and episode_count >= args.num_episodes:
                print(f"\nCompleted {args.num_episodes} episodes.")
                running = False
            else:
                # Reset for next episode
                reset_key = jrandom.fold_in(master_key, reset_counter)
                obs, env_state = jitted_reset(reset_key)
                reset_counter += 1
                total_reward = 0.0
                step_count = 0

    pygame.quit()
    print("\nVisualization ended.")


if __name__ == "__main__":
    main()
