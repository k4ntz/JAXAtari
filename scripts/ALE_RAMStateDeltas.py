import numpy as np
import pygame
import gymnasium as gym
import ale_py
from ale_py._ale_py import ALEInterface
from tqdm import tqdm
from collections import deque
from copy import deepcopy
import pickle as pkl
import atexit
import matplotlib.pyplot as plt
import sys
import os  # Import os module

# Constants for RAM panel layout (will be scaled)
RAM_N_COLS = 8
_BASE_SCALE = 4  # The scale factor these base dimensions were designed for
_BASE_RAM_RENDER_WIDTH = 1000
_BASE_RAM_CELL_WIDTH = 115
_BASE_RAM_CELL_HEIGHT = 45
_BASE_RAM_GRID_ANCHOR_PADDING = 28
_BASE_RAM_COL_SPACING = 120
_BASE_RAM_ROW_SPACING = 50
_BASE_ID_FONT_SIZE = 25
_BASE_VALUE_FONT_SIZE = 30


class Renderer:
    window: pygame.Surface
    clock: pygame.time.Clock
    env: gym.Env
    ale: ALEInterface

    # Add render_scale and screenshot_dir parameters
    def __init__(self, env_name, no_render=[], render_scale=4, screenshot_dir=None):
        self.render_scale = render_scale  # Store the desired scale factor
        self.screenshot_dir = screenshot_dir  # Store screenshot directory

        # Create screenshot directory if it doesn't exist
        if self.screenshot_dir:
            os.makedirs(self.screenshot_dir, exist_ok=True)
            start_frame_counter = 1  # Default starting number
            if os.path.isdir(self.screenshot_dir):
                try:
                    existing_files = os.listdir(self.screenshot_dir)
                    frame_numbers = [
                        int(filename[len("frame_"):-len(".npy")])
                        for filename in existing_files
                        if filename.startswith("frame_") and filename.endswith(".npy")
                    ]
                    if frame_numbers:
                        start_frame_counter = max(frame_numbers) + 1
                except Exception as e:
                    print(f"Error checking existing screenshots: {e}. Starting frame counter from 1.")
            self.frame_counter = start_frame_counter
        else:
            self.frame_counter = 1

        try:
            self.env = gym.make(
                f"ALE/{env_name}-v5",
                frameskip=1,
                render_mode="rgb_array",
            )
        except Exception as e:
            print(f"Error creating Gymnasium environment 'ALE/{env_name}-v5': {e}")
            print("Please ensure you have gymnasium[atari] and ROMs installed.")
            sys.exit(1)

        try:
            self.ale = self.env.unwrapped.ale
            print("Successfully accessed env.unwrapped.ale")
        except AttributeError:
            print("Error: Could not access the underlying ALE interface via env.unwrapped.ale.")
            self.ale = None

        self.initial_obs, self.info = self.env.reset(seed=42)
        self.current_frame = self.env.render()  # Get initial frame (likely native size)

        if self.current_frame is None:
            print("Error: env.render() returned None. Ensure render_mode='rgb_array' is set.")
            sys.exit(1)

        # Initialize pygame and calculate scaled sizes
        self._init_pygame(self.current_frame)
        self.paused = False

        self.current_keys_down = set()
        self.current_mouse_pos = None

        try:
            action_meanings = self.env.unwrapped.get_action_meanings()
            print(f"Action meanings: {action_meanings}")
            self.keys2actions = {(): 0}  # NOOP action
            if 'UP' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP]))] = action_meanings.index('UP')
            if 'DOWN' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_DOWN]))] = action_meanings.index('DOWN')
            if 'LEFT' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_LEFT]))] = action_meanings.index('LEFT')
            if 'RIGHT' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_RIGHT]))] = action_meanings.index('RIGHT')
            if 'FIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_SPACE]))] = action_meanings.index('FIRE')
            if 'LEFTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_LEFT, pygame.K_SPACE]))] = action_meanings.index('LEFTFIRE')
            if 'RIGHTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_RIGHT, pygame.K_SPACE]))] = action_meanings.index('RIGHTFIRE')
            if 'DOWNFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_DOWN, pygame.K_SPACE]))] = action_meanings.index('DOWNFIRE')
            if 'UPFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP, pygame.K_SPACE]))] = action_meanings.index('UPFIRE')
            if 'UPRIGHTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE]))] = action_meanings.index('UPRIGHTFIRE')
            if 'UPLEFTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE]))] = action_meanings.index('UPLEFTFIRE')
            if 'DOWNRIGHTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE]))] = action_meanings.index('DOWNRIGHTFIRE')
            if 'DOWNLEFTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE]))] = action_meanings.index('DOWNLEFTFIRE')
            if 'UPRIGHT' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP, pygame.K_RIGHT]))] = action_meanings.index('UPRIGHT')
            if 'UPLEFT' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP, pygame.K_LEFT]))] = action_meanings.index('UPLEFT')
            if 'DOWNRIGHT' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_DOWN, pygame.K_RIGHT]))] = action_meanings.index('DOWNRIGHT')
            if 'DOWNLEFT' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_DOWN, pygame.K_LEFT]))] = action_meanings.index('DOWNLEFT')
            print(f"Key to action mapping: {self.keys2actions}")
        except AttributeError:
            print("Warning: Could not get action meanings. Using default NOOP mapping.")
            self.keys2actions = {(): 0}

        self.active_cell_idx = None
        self.candidate_cell_ids = []
        self.current_active_cell_input: str = ""
        self.no_render = no_render
        self.red_render = []

        self.saved_frames = deque(maxlen=20)
        self.frame_by_frame = False
        self.next_frame = False

        self.past_ram = None
        self.ram = self._get_ram()
        self.delta_render = []

        self.clicked_cells = []
        self.recorded_ram_states = {}
        self.game_name = env_name

    def save_frame(self, frame: np.ndarray) -> None:
        if not self.screenshot_dir:
            print("Screenshot directory not set. Cannot save frame.")
            return
        if frame is None:
            print("Warning: Cannot save None frame."); return
        try:
            filepath = os.path.join(self.screenshot_dir, f"frame_{self.frame_counter}.npy")
            np.save(filepath, frame)
            print(f"Frame saved to {filepath}")
            self.frame_counter += 1
        except Exception as e:
            print(f"Error saving frame: {e}")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Gymnasium ALE RAM Explorer with Frame Saving")
    parser.add_argument("-g", "--game", type=str, default="VideoPinball", help="Name of the Atari game (e.g., 'Pong', 'Breakout').")
    parser.add_argument('--scale', type=int, default=4, help='Scale factor for the game display window')
    parser.add_argument("-ls", "--load_state", type=str, default=None, help="Path to a pickled ALE state file (.pkl) to load.")
    parser.add_argument("-nr", "--no_render", type=int, default=[], nargs="+", help="List of RAM cell indices (0-127) to hide.")
    parser.add_argument("-nra", "--no_render_all", action="store_true", help="Hide all RAM cells.")
    parser.add_argument('--screenshot-dir', type=str, default=None, help='Directory to save screenshots as .npy files (optional)')

    args = parser.parse_args()

    if args.screenshot_dir is None:
        args.screenshot_dir = f"../new_screenshots"

    if args.no_render_all:
        args.no_render = list(range(128))

    renderer = Renderer(env_name=args.game, no_render=args.no_render, render_scale=args.scale, screenshot_dir=args.screenshot_dir)

    if args.load_state:
        if renderer.ale:
            try:
                with open(args.load_state, "rb") as f:
                    state_to_load = pkl.load(f)
                    renderer._restore_state(state_to_load)
                    renderer.current_frame = renderer.env.render()  # Update visual frame
                    renderer.past_ram = None; renderer.delta_render = []
                    print(f"ALE state loaded from {args.load_state}")
            except FileNotFoundError: print(f"Error: Load state file not found: {args.load_state}")
            except Exception as e: print(f"Error loading state: {e}")
        else: print("Warning: Cannot load state, ALE interface not available.")

    def exit_handler():
        if 'renderer' in locals() and hasattr(renderer, 'no_render') and renderer.no_render:
            print("\nFinal hidden RAM cells (no_render list): ")
            print(*(sorted(renderer.no_render)))

    atexit.register(exit_handler)
    renderer.run()