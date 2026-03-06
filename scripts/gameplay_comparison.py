#!/usr/bin/env python3
"""
A script to play and compare JAXAtari and ALE (Gymnasium) game versions.

Supports two modes:
1.  'parallel': Play both games side-by-side with mirrored input.
2.  'record_replay': Play and record JAXAtari, then replay the actions
    on both environments side-by-side for a deterministic comparison.

Prints a detailed action-mapping comparison on startup.

Controls:
-   Movement: Arrows / WASD
-   Fire: Space / Enter
-   Pause: P
-   Frame-by-Frame Toggle: F
-   Next Frame: N (when frame-by-frame is on)
-   Reset: R
-   Quit: ESCAPE (or Q in record mode)

Assumes `core.py` is in the same directory or Python path.
Requires `jaxatari`, `gymnasium[atari]`, `ale-py`, `pygame`, and `numpy`.
"""

import argparse
import sys
import os
import time
from typing import Tuple, Dict, Any, Optional, List

import pygame
import numpy as np
import gymnasium as gym
import ale_py  # Required for gym[atari]
import jax
import jax.random as jrandom
import jax.numpy as jnp

try:
    import jaxatari.core as core 
    from jaxatari.environment import JaxEnvironment, JAXAtariAction
    from jaxatari.renderers import JAXGameRenderer
except ImportError:
    print("Error: Could not import 'core' or 'jaxatari'.")
    print("Please ensure 'core.py' is in the same directory and 'jaxatari' is installed.")
    sys.exit(1)


# --- Constants ---
# Use a smaller upscale factor to fit 3 screens
UPSCALE_FACTOR = 3
# Standard Atari resolution
NATIVE_H, NATIVE_W = 210, 160
SCALED_W = NATIVE_W * UPSCALE_FACTOR
SCALED_H = NATIVE_H * UPSCALE_FACTOR

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_BG = (20, 20, 20)
COLOR_RECORD = (200, 0, 0)
COLOR_REPLAY = (0, 150, 200)
COLOR_PARALLEL = (0, 200, 0)
COLOR_PAUSE = (255, 255, 0)


# Define all possible semantic actions
ALL_SEMANTIC_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"
]


# --- Environment Setup ---

def setup_ale_env(game_name: str, seed: int) -> gym.Env:
    """Initializes the Gymnasium ALE environment."""
    print(f"Initializing ALE env: 'ALE/{game_name}-v5'")
    try:
        # Use frameskip=1 for 1-to-1 action comparison
        env = gym.make(
            f"ALE/{game_name}-v5",
            render_mode="rgb_array",
            frameskip=1,
            repeat_action_probability=0.0 # Deterministic
        )
        env.reset(seed=seed)
        print("ALE environment initialized.")
        return env
    except Exception as e:
        print(f"Error creating ALE environment: {e}")
        print("Ensure you have ROMs installed (e.g., `ale-import-roms .` or `pip install gymnasium[accept-rom-license]`)")
        sys.exit(1)

def setup_jax_env(game_name: str, seed: int) -> Dict[str, Any]:
    """Initializes the JAXAtari environment using core.py."""
    print(f"Initializing JAXAtari env: '{game_name}'")
    try:
        env, renderer = core.make(game_name), core.make_renderer(game_name)
        if env is None or renderer is None:
            raise ImportError(f"Could not load game or renderer for '{game_name}' from core.")
        
        master_key = jrandom.PRNGKey(seed)
        obs, state = env.reset(master_key)
        
        print("JAXAtari environment initialized and jitting functions...")
        return {
            "env": env,
            "renderer": renderer,
            "jitted_step": jax.jit(env.step),
            "jitted_render": jax.jit(env.render),
            "state": state,
            "key": master_key,
            "seed": seed,
        }
    except Exception as e:
        print(f"Error creating JAXAtari environment: {e}")
        sys.exit(1)


# --- Action Mapping ---

def build_jax_action_map() -> Dict[str, int]:
    """Builds a map of semantic action names to JAXAtariAction integers."""
    return {
        name: getattr(JAXAtariAction, name)
        for name in dir(JAXAtariAction)
        if not name.startswith("_") and isinstance(getattr(JAXAtariAction, name), int)
    }

def build_ale_action_map(env: gym.Env) -> Dict[str, int]:
    """Builds a map of semantic action names to ALE action integers."""
    try:
        meanings = env.unwrapped.get_action_meanings()
        # Default to 0 (NOOP)
        return {name: i for i, name in enumerate(meanings)}
    except Exception as e:
        print(f"Warning: Could not get ALE action meanings: {e}. Defaulting to NOOP.")
        return {"NOOP": 0}

def get_semantic_action_from_keys(pressed_keys: pygame.key.ScancodeWrapper) -> str:
    """
    Maps pygame keys to a single semantic action string.
    This handles combined inputs.
    """
    up = pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_w]
    down = pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_s]
    left = pressed_keys[pygame.K_LEFT] or pressed_keys[pygame.K_a]
    right = pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]
    fire = pressed_keys[pygame.K_SPACE] or pressed_keys[pygame.K_RETURN]

    # Check most complex combinations first
    if up and right and fire: return "UPRIGHTFIRE"
    if up and left and fire: return "UPLEFTFIRE"
    if down and right and fire: return "DOWNRIGHTFIRE"
    if down and left and fire: return "DOWNLEFTFIRE"
    
    if up and fire: return "UPFIRE"
    if down and fire: return "DOWNFIRE"
    if left and fire: return "LEFTFIRE"
    if right and fire: return "RIGHTFIRE"
    
    if up and right: return "UPRIGHT"
    if up and left: return "UPLEFT"
    if down and right: return "DOWNRIGHT"
    if down and left: return "DOWNLEFT"
    
    if fire: return "FIRE"
    if up: return "UP"
    if down: return "DOWN"
    if left: return "LEFT"
    if right: return "RIGHT"

    # Default: No action (log "NONE" as requested by user)
    return "NOOP"


# --- Pygame Rendering ---

def create_comparison_surface(
    jax_frame: np.ndarray,
    ale_frame: np.ndarray,
    font: pygame.font.Font,
    mode_text: str,
    text_color: Tuple[int, int, int]
) -> pygame.Surface:
    """
    Creates a single pygame surface with JAX, ALE, and Heatmap side-by-side.
    """
    total_width = SCALED_W * 3
    total_surface = pygame.Surface((total_width, SCALED_H))
    total_surface.fill(COLOR_BG)
    scaled_size_wh = (SCALED_W, SCALED_H)

    # --- 1. JAX Frame ---
    try:
        # JAX frame is (H, W, 3), transpose to (W, H, 3) for make_surface
        jax_surf = pygame.surfarray.make_surface(np.transpose(jax_frame, (1, 0, 2)))
        jax_surf_scaled = pygame.transform.scale(jax_surf, scaled_size_wh)
        total_surface.blit(jax_surf_scaled, (0, 0))
    except Exception as e:
        print(f"JAX render error: {e}")

    # --- 2. ALE Frame ---
    try:
        # ALE frame is (H, W, 3), transpose to (W, H, 3)
        ale_surf = pygame.surfarray.make_surface(np.transpose(ale_frame, (1, 0, 2)))
        ale_surf_scaled = pygame.transform.scale(ale_surf, scaled_size_wh)
        total_surface.blit(ale_surf_scaled, (SCALED_W, 0))
    except Exception as e:
        print(f"ALE render error: {e}")

    # --- 3. Heatmap Difference ---
    try:
        if jax_frame.shape == ale_frame.shape:
            # Calculate absolute difference
            diff = np.abs(jax_frame.astype(np.int16) - ale_frame.astype(np.int16))
            # Clip to 255 and cast to uint8
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            
            diff_surf = pygame.surfarray.make_surface(np.transpose(diff, (1, 0, 2)))
            diff_surf_scaled = pygame.transform.scale(diff_surf, scaled_size_wh)
            total_surface.blit(diff_surf_scaled, (SCALED_W * 2, 0))
        else:
            # Handle shape mismatch
            text = font.render("SHAPE MISMATCH", True, COLOR_WHITE)
            text_rect = text.get_rect(center=(SCALED_W * 2 + SCALED_W // 2, SCALED_H // 2))
            total_surface.blit(text, text_rect)
    except Exception as e:
        print(f"Diff render error: {e}")

    # --- Add Headers ---
    headers = ["JAXATARI", "ALE", "DIFFERENCE"]
    for i, header in enumerate(headers):
        text = font.render(header, True, COLOR_WHITE)
        text_rect = text.get_rect(center=(SCALED_W * i + SCALED_W // 2, 15))
        total_surface.blit(text, text_rect)

    # --- Add Mode Text ---
    mode_render = font.render(mode_text, True, text_color)
    mode_rect = mode_render.get_rect(center=(total_width // 2, SCALED_H - 20))
    total_surface.blit(mode_render, mode_rect)

    return total_surface

def render_single_frame(
    frame: np.ndarray,
    font: pygame.font.Font,
    header: str,
    text: str,
    text_color: Tuple[int, int, int]
) -> pygame.Surface:
    """Renders just one game screen, for the recording phase."""
    surface = pygame.Surface((SCALED_W, SCALED_H))
    surface.fill(COLOR_BG)
    scaled_size_wh = (SCALED_W, SCALED_H)
    
    try:
        frame_surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame_surf_scaled = pygame.transform.scale(frame_surf, scaled_size_wh)
        surface.blit(frame_surf_scaled, (0, 0))
    except Exception as e:
        print(f"Single frame render error: {e}")

    # Header
    header_render = font.render(header, True, COLOR_WHITE)
    header_rect = header_render.get_rect(center=(SCALED_W // 2, 15))
    surface.blit(header_render, header_rect)

    # Text
    text_render = font.render(text, True, text_color)
    text_rect = text_render.get_rect(center=(SCALED_W // 2, SCALED_H - 20))
    surface.blit(text_render, text_rect)
    
    return surface

# --- Game Loops ---

def run_parallel_mode(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    ale_env: gym.Env,
    jax_data: Dict[str, Any],
    jax_action_map: Dict[str, int],
    ale_action_map: Dict[str, int],
    fps: int,
    seed: int
):
    """
    Mode 1: Run both envs side-by-side, mirroring real-time input.
    """
    running = True
    pause = False
    frame_by_frame = False
    next_frame_asked = False

    jax_state = jax_data["state"]
    jitted_step = jax_data["jitted_step"]
    jitted_render = jax_data["jitted_render"]

    # Get initial frames
    jax_frame = np.array(jitted_render(jax_state))
    ale_frame = ale_env.render()

    while running:
        # --- Handle Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = not pause
                    print(f"Paused: {pause}")
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    pause = False  # Toggling f-b-f unpauses
                    print(f"Frame-by-frame: {frame_by_frame}")
                elif event.key == pygame.K_n:
                    next_frame_asked = True
                elif event.key == pygame.K_r:
                    print("Resetting environments by user request...")
                    ale_obs, ale_info = ale_env.reset(seed=seed)
                    jax_obs, jax_state = jax_data["env"].reset(jrandom.PRNGKey(seed))
                    # Get frames after reset
                    jax_frame = np.array(jitted_render(jax_state))
                    ale_frame = ale_env.render()

        # --- Pause/Frame-by-Frame Logic ---
        if pause or (frame_by_frame and not next_frame_asked):
            # Just re-render the current state
            mode_text = "PAUSED" if pause else "FRAME-BY-FRAME (Press N)"
            comparison_surface = create_comparison_surface(
                jax_frame, ale_frame, font, mode_text, COLOR_PAUSE
            )
            screen.blit(comparison_surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
            continue  # Skip the game step

        # --- Game Step Logic ---
        pressed_keys = pygame.key.get_pressed()
        semantic_action = get_semantic_action_from_keys(pressed_keys)

        # --- Map Actions ---
        jax_action = jax_action_map.get(semantic_action, JAXAtariAction.NOOP)
        ale_action = ale_action_map.get(semantic_action, 0)  # 0 is NOOP

        # --- Step Environments ---
        jax_obs, jax_state, jax_reward, jax_done, jax_info = jitted_step(jax_state, jax_action)
        ale_obs, ale_reward, ale_term, ale_trunc, ale_info = ale_env.step(ale_action)
        ale_done = ale_term or ale_trunc

        # --- Render Frames ---
        jax_frame = np.array(jitted_render(jax_state))
        ale_frame = ale_env.render()

        # --- Create Comparison and Blit ---
        comparison_surface = create_comparison_surface(
            jax_frame, ale_frame, font, "PARALLEL MODE", COLOR_PARALLEL
        )
        screen.blit(comparison_surface, (0, 0))
        pygame.display.flip()

        # --- Handle Reset ---
        # Reset both if either is done to keep them in sync
        if jax_done or ale_done:
            print("Resetting environments (end of episode)...")
            ale_obs, ale_info = ale_env.reset()
            # Use a new folded key for JAX's next episode
            jax_data["key"] = jrandom.fold_in(jax_data["key"], int(time.time()))
            jax_obs, jax_state = jax_data["env"].reset(jax_data["key"])
            # Get frames after reset
            jax_frame = np.array(jitted_render(jax_state))
            ale_frame = ale_env.render()

        # Reset frame-by-frame flag
        if next_frame_asked:
            next_frame_asked = False

        clock.tick(fps)

def run_record_replay_mode(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    ale_env: gym.Env,
    jax_data: Dict[str, Any],
    jax_action_map: Dict[str, int],
    ale_action_map: Dict[str, int],
    fps: int,
    seed: int
):
    """
    Mode 2: Record JAX, then replay both from the same seed.
    """
    
    # --- Part 1: Record JAX ---
    print("--- RECORDING PHASE ---")
    print("Playing JAXAtari. Press 'Q' or ESCAPE to stop recording and start replay.")
    
    # Resize window for single view
    screen = pygame.display.set_mode((SCALED_W, SCALED_H))
    
    recorded_actions: List[str] = []
    jax_state = jax_data["state"]
    jitted_step = jax_data["jitted_step"]
    jitted_render = jax_data["jitted_render"]
    jax_frame = np.array(jitted_render(jax_state))  # Initial frame
    
    recording = True
    pause = False
    frame_by_frame = False
    next_frame_asked = False
    
    while recording:
        # --- Handle Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q)):
                recording = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = not pause
                    print(f"Paused: {pause}")
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    pause = False
                    print(f"Frame-by-frame: {frame_by_frame}")
                elif event.key == pygame.K_n:
                    next_frame_asked = True
                elif event.key == pygame.K_r:
                    print("Resetting JAX environment (clearing actions)...")
                    jax_obs, jax_state = jax_data["env"].reset(jrandom.PRNGKey(seed))
                    jax_frame = np.array(jitted_render(jax_state))
                    recorded_actions = []
        
        # --- Pause/Frame-by-Frame Logic ---
        if pause or (frame_by_frame and not next_frame_asked):
            mode_text = "PAUSED (RECORDING)" if pause else "FRAME-BY-FRAME (Press N)"
            single_surface = render_single_frame(
                jax_frame, font, "JAXATARI (RECORDING)", mode_text, COLOR_PAUSE
            )
            screen.blit(single_surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
            continue

        # --- Game Step Logic ---
        pressed_keys = pygame.key.get_pressed()
        semantic_action = get_semantic_action_from_keys(pressed_keys)
        
        # Log this action every frame
        recorded_actions.append(semantic_action)
        
        # --- Map & Step JAX ---
        jax_action = jax_action_map.get(semantic_action, JAXAtariAction.NOOP)
        jax_obs, jax_state, jax_reward, jax_done, jax_info = jitted_step(jax_state, jax_action)
        
        # --- Render JAX ---
        jax_frame = np.array(jitted_render(jax_state))
        
        # --- Blit Single Frame ---
        single_surface = render_single_frame(
            jax_frame, font, "JAXATARI (RECORDING)", "Press Q or ESC to stop", COLOR_RECORD
        )
        screen.blit(single_surface, (0, 0))
        pygame.display.flip()
        
        if jax_done:
            print("JAX env done. Stopping recording.")
            recording = False
        
        # Reset frame-by-frame flag
        if next_frame_asked:
            next_frame_asked = False
            
        clock.tick(fps)
 
    print(f"\n--- REPLAY PHASE ---")
    print(f"Stopped recording. Replaying {len(recorded_actions)} actions...")
     
    # --- Part 2: Replay Both ---
     
    # Resize window for comparison view
    screen = pygame.display.set_mode((SCALED_W * 3, SCALED_H))
     
    # CRITICAL: Reset both environments to the *exact same* starting seed
    def reset_for_replay():
        print(f"Resetting both environments to seed {seed} for replay...")
        ale_obs, ale_info = ale_env.reset(seed=seed)
        jax_obs, jax_state = jax_data["env"].reset(jrandom.PRNGKey(seed))
        jax_frame = np.array(jitted_render(jax_state))
        ale_frame = ale_env.render()
        return ale_obs, jax_state, jax_frame, ale_frame

    ale_obs, jax_state, jax_frame, ale_frame = reset_for_replay()
     
    replay_idx = 0
    pause = False
    frame_by_frame = False
    next_frame_asked = False
     
    while replay_idx < len(recorded_actions):
        # Check for quit event during replay
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                print("Replay cancelled.")
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = not pause
                    print(f"Paused: {pause}")
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    pause = False
                    print(f"Frame-by-frame: {frame_by_frame}")
                elif event.key == pygame.K_n:
                    next_frame_asked = True
                elif event.key == pygame.K_r:
                    print("Restarting replay from beginning...")
                    ale_obs, jax_state, jax_frame, ale_frame = reset_for_replay()
                    replay_idx = 0
                    pause = False
                    frame_by_frame = False
                    next_frame_asked = False
                    continue
 
        # --- Pause/Frame-by-Frame Logic ---
        if pause or (frame_by_frame and not next_frame_asked):
            mode_text = "REPLAY PAUSED" if pause else "FRAME-BY-FRAME (Press N)"
            mode_text += f" (Frame {replay_idx+1}/{len(recorded_actions)})"
             
            comparison_surface = create_comparison_surface(
                jax_frame, ale_frame, font, mode_text, COLOR_PAUSE
            )
            screen.blit(comparison_surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
            continue
             
        # --- Replay Step Logic ---
        semantic_action = recorded_actions[replay_idx]
         
        # --- Map Actions ---
        jax_action = jax_action_map.get(semantic_action, JAXAtariAction.NOOP)
        ale_action = ale_action_map.get(semantic_action, 0)
         
        # --- Step Environments ---
        jax_obs, jax_state, jax_reward, jax_done, jax_info = jitted_step(jax_state, jax_action)
        ale_obs, ale_reward, ale_term, ale_trunc, ale_info = ale_env.step(ale_action)
        # --- Render Frames ---
        jax_frame = np.array(jitted_render(jax_state))
        ale_frame = ale_env.render()
         
        # --- Create Comparison and Blit ---
        replay_text = f"REPLAY MODE (Frame {replay_idx+1}/{len(recorded_actions)})"
        comparison_surface = create_comparison_surface(
            jax_frame, ale_frame, font, replay_text, COLOR_REPLAY
        )
        screen.blit(comparison_surface, (0, 0))
        pygame.display.flip()
         
        # Move to next frame
        replay_idx += 1
         
        # Reset frame-by-frame flag
        if next_frame_asked:
            next_frame_asked = False
             
        clock.tick(fps)
         
    print("Replay finished.")
    # Keep the final frame on screen for a moment
    time.sleep(5)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Compare JAXAtari and ALE game renders.")
    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        help="Name of the game (e.g., 'pong', 'breakout'). Must be supported by core.py and ALE."
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["parallel", "record_replay"],
        default="parallel",
        help="Comparison mode: 'parallel' (real-time mirrored input) or 'record_replay' (record jax, replay both)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for both environments."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for playback."
    )
    args = parser.parse_args()
    
    # Capitalize game name for ALE (e.g., 'pong' -> 'Pong')
    ale_game_name = args.game.capitalize()
    
    # --- Setup ---
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)
    
    # Set initial window size (will be resized by mode)
    window_width = SCALED_W * 3 if args.mode == "parallel" else SCALED_W
    screen = pygame.display.set_mode((window_width, SCALED_H))
    pygame.display.set_caption(f"JAXAtari vs ALE Comparison: {args.game}")
    clock = pygame.time.Clock()

    # Init environments
    ale_env = setup_ale_env(ale_game_name, args.seed)
    jax_data = setup_jax_env(args.game.lower(), args.seed)

    # Build universal action maps
    jax_action_map = build_jax_action_map()
    ale_action_map = build_ale_action_map(ale_env)
    
    print("\n" + "="*40)
    print("--- ACTION SPACE MAPPINGS ---")
    print(f"{'SEMANTIC':<18} | {'JAX INT':<8} | {'ALE INT':<8}")
    print("-"*44)
    
    for action_name in ALL_SEMANTIC_ACTIONS:
        jax_int = jax_action_map.get(action_name, 'N/A')
        ale_int = ale_action_map.get(action_name, 'N/A')
        
        # Only print actions that exist in at least one environment
        if jax_int != 'N/A' or ale_int != 'N/A':
            print(f"{action_name:<18} | {str(jax_int):<8} | {str(ale_int):<8}")
            
    print("="*44)
    print("--- CONTROLS ---")
    print("  Movement:  Arrows / WASD")
    print("  Fire:      Space / Enter")
    print("  Pause:     P")
    print("  Frame Step:F (Toggle), N (Next Frame)")
    print("  Reset:     R")
    print("  Quit:      ESCAPE (Q in Record mode)")
    print("="*44)
    print(f"Starting in '{args.mode}' mode in 3 seconds...\n")
    time.sleep(3)
    # --- *** END NEW SECTION *** ---

    # --- Run Mode ---
    try:
        if args.mode == "parallel":
            run_parallel_mode(
                screen, clock, font, ale_env, jax_data,
                jax_action_map, ale_action_map, args.fps, args.seed
            )
        elif args.mode == "record_replay":
            run_record_replay_mode(
                screen, clock, font, ale_env, jax_data,
                jax_action_map, ale_action_map, args.fps, args.seed
            )
    except Exception as e:
        print(f"\nAn error occurred during the game loop: {e}")
    finally:
        ale_env.close()
        pygame.quit()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()