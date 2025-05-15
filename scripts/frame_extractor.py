import pygame
import numpy as np
import os
import argparse
import gymnasium as gym
import ale_py
from typing import Dict, Tuple, List, Set
import time
import traceback

# UPSCALE_FACTOR = 4 # Example if needed elsewhere, but args.scale is used

def _build_dynamic_key_map(env: gym.Env) -> Dict[Tuple[int, ...], int]:
    """
    Builds a Pygame Key -> ALE Action Integer mapping based on the
    semantic meanings provided by the environment.

    Args:
        env: The Gymnasium Atari environment instance.

    Returns:
        A dictionary where keys are sorted tuples of pygame.K_* constants
        and values are the corresponding action integers for the loaded game.
    """
    # 1. Define the DESIRED semantic mappings (Key -> Meaning String)
    # Use sorted tuples for key combinations to ensure consistency.
    desired_semantic_map: Dict[Tuple[int, ...], str] = {
        # Single Keys
        (pygame.K_UP,):    'UP',
        (pygame.K_DOWN,):  'DOWN',
        (pygame.K_LEFT,):  'LEFT',
        (pygame.K_RIGHT,): 'RIGHT',
        (pygame.K_SPACE,): 'FIRE', # Primary action button

        # Combined Keys (Map to standard combined meanings)
        # Sorting ensures (UP, RIGHT) is treated the same as (RIGHT, UP)
        tuple(sorted((pygame.K_UP, pygame.K_RIGHT))): 'UPRIGHT',
        tuple(sorted((pygame.K_UP, pygame.K_LEFT))):  'UPLEFT',
        tuple(sorted((pygame.K_DOWN, pygame.K_RIGHT))):'DOWNRIGHT',
        tuple(sorted((pygame.K_DOWN, pygame.K_LEFT))): 'DOWNLEFT',
        tuple(sorted((pygame.K_UP, pygame.K_SPACE))):  'UPFIRE',
        tuple(sorted((pygame.K_DOWN, pygame.K_SPACE))):'DOWNFIRE',
        tuple(sorted((pygame.K_LEFT, pygame.K_SPACE))):'LEFTFIRE',
        tuple(sorted((pygame.K_RIGHT, pygame.K_SPACE))):'RIGHTFIRE',

        # Three-key combinations (less common, but possible)
        tuple(sorted((pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE))): 'UPRIGHTFIRE',
        tuple(sorted((pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE))):  'UPLEFTFIRE',
        tuple(sorted((pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE))):'DOWNRIGHTFIRE',
        tuple(sorted((pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE))): 'DOWNLEFTFIRE',
    }

    # 2. Get ACTUAL action meanings from the environment
    try:
        action_meanings: List[str] = env.unwrapped.get_action_meanings()
    except AttributeError:
        print("Warning: env.unwrapped.get_action_meanings() not found. Cannot build dynamic map.")
        # Fallback to a very basic map or raise error? For now, return empty.
        return {}
    except Exception as e:
        print(f"Warning: Error getting action meanings: {e}. Cannot build dynamic map.")
        return {}

    # 3. Create a reverse lookup: Meaning String -> Action Integer
    meaning_to_int: Dict[str, int] = {
        meaning: idx for idx, meaning in enumerate(action_meanings)
    }

    # 4. Build the final map: Pygame Keys Tuple -> Action Integer
    final_key_to_action_map: Dict[Tuple[int, ...], int] = {}
    mapped_keys: Set[int] = set() # Keep track of individual keys used

    print("\n--- Building Dynamic Key Map ---")
    for key_tuple, desired_meaning in desired_semantic_map.items():
        if desired_meaning in meaning_to_int:
            action_int = meaning_to_int[desired_meaning]
            final_key_to_action_map[key_tuple] = action_int
            # Add the individual keys from this tuple to our set
            mapped_keys.update(key_tuple)
            # Optional: print successful mappings
            # key_names = tuple(pygame.key.name(k) for k in key_tuple)
            # print(f"  Mapped {key_names} -> {desired_meaning} (Action {action_int})")
        # else:
            # Optional: print warnings for desired meanings not found in this game
            # key_names = tuple(pygame.key.name(k) for k in key_tuple)
            # print(f"  Info: Desired meaning '{desired_meaning}' for keys {key_names} not found in this game's actions.")

    # Add NOOP mapping (empty tuple for no keys pressed maps to action 0)
    # Note: The _get_action method already defaults to 0 if no match is found,
    # so explicitly adding this might be redundant depending on that logic,
    # but it's clearer. Check if action 0 is indeed NOOP.
    if 0 < len(action_meanings) and action_meanings[0] == 'NOOP':
         # Add mapping for "no relevant keys pressed" -> NOOP (Action 0)
         # This isn't strictly needed if _get_action() defaults to 0,
         # but helps conceptually. We won't add it directly to the map used
         # for key press lookups, but confirm NOOP=0 exists.
         print("  Confirmed: Action 0 is NOOP.")
         pass # NOOP handled by default in _get_action
    elif 0 in meaning_to_int.values():
        print(f"  Warning: Action 0 exists but is not 'NOOP' (it's '{action_meanings[0]}'). Default action might be unexpected.")
    else:
        print(f"  Warning: Action 0 not found in action space ({len(action_meanings)} actions total). Defaulting to 0 may error.")

    print("--- Dynamic Key Map Build Complete ---")

    # Store the set of relevant keys for faster checking in input handler
    final_key_to_action_map['_relevant_keys_'] = mapped_keys # Use a special key

    return final_key_to_action_map

class AtariPlayer:
    """
    A generalized Atari game player using Gymnasium and ALE-Py,
    correctly handling single and combined key inputs.
    """

    def __init__(self,
                 game_name: str = "Pong",
                 render_scale: int = 4,
                 screenshot_dir: str = None,
                 fps: int = 30):
        self.game_name = game_name
        self.render_scale = render_scale

        try:
            # Create Gymnasium ALE environment
            self.env = gym.make(f"ALE/{game_name}-v5", render_mode="rgb_array", frameskip=1)
            # No need to get self.ale explicitly unless using direct ALE calls not in Gym wrapper

            # Reset environment to get initial observation
            self.env.reset(seed=42)
            self.current_frame = self.env.render()
            if self.current_frame is None:
                raise RuntimeError("env.render() returned None. Check render_mode.")

            # Initialize pygame
            pygame.init()
            pygame.font.init() # Keep if drawing text overlays

            self.env_render_shape_h_w = self.current_frame.shape[:2]
            self.screen_width = int(self.env_render_shape_h_w[1] * self.render_scale)
            self.screen_height = int(self.env_render_shape_h_w[0] * self.render_scale)

            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(f"Gymnasium Atari Player: {game_name}")
            self.clock = pygame.time.Clock()
            self.fps = fps

            self.screenshot_dir = screenshot_dir
            if screenshot_dir:
                os.makedirs(screenshot_dir, exist_ok=True)
            self.frame_counter = 1

            self.current_keys_down = set() # Stores pygame.K_* constants currently pressed
            self.current_mouse_pos = None

            # --- CRITICAL CHANGE: Build dynamic key mappings ---
            # Call the new function after env is created
            self.keys2actions = _build_dynamic_key_map(self.env)
            # Extract the set of relevant keys for input handling
            self.relevant_keys = self.keys2actions.pop('_relevant_keys_', set())
            # ---

            self.paused = False
            self.frame_by_frame = False
            self.next_frame = False

            print(f"\nLoaded {game_name} environment")
            print(f"Raw observation shape (H, W, C): {self.current_frame.shape}")
            print(f"Scaled display size (W, H): {self.screen_width}x{self.screen_height}")
            print(f"Action space size: {self.env.action_space.n}")

            # Print the actual mappings being used (updated to use dynamic map)
            self._print_action_meanings_and_key_mappings()

        except Exception as e:
            pygame.quit()
            print(f"\nError during initialization: {e}")
            print("Ensure the game ROM is installed (e.g., `ale-import-roms /path/to/roms`)")
            print("Or try 'pip install gymnasium[accept-rom-license]'")
            traceback.print_exc()
            raise

    # --- REMOVE the old _get_valid_key_mappings ---
    # def _get_valid_key_mappings(self): ... # DELETE THIS FUNCTION

    # --- Helper method to print mappings (Modified slightly) ---
    def _print_action_meanings_and_key_mappings(self):
        """
        Print the available actions and their corresponding key mappings.
        """
        print("\n--- Action Meanings and Key Mappings (Dynamically Generated) ---")
        try:
            action_meanings = self.env.unwrapped.get_action_meanings()
            print(f"Available actions ({self.env.action_space.n}):")
            for i, meaning in enumerate(action_meanings):
                print(f"  Action {i}: {meaning}")

            if not self.keys2actions:  # Check the actual map now
                print("\nKey mappings: None generated (check warnings during build).")
                return

            print(f"\nKey mappings:")
            # Sort by action number for clarity
            sorted_mappings = sorted(self.keys2actions.items(), key=lambda item: item[1])

            for keys, action in sorted_mappings:
                # Ensure action is valid before accessing meanings list
                if 0 <= action < len(action_meanings):
                    key_names = [pygame.key.name(key).upper() for key in keys]
                    action_meaning = action_meanings[action]  # Use action index directly
                    print(f"  {' + '.join(key_names):<25} -> Action {action} ({action_meaning})")
                else:
                    # This case *shouldn't* happen with the dynamic builder, but good to check
                    key_names = [pygame.key.name(key).upper() for key in keys]
                    print(f"  {' + '.join(key_names):<25} -> Invalid Action Index ({action})")

        except AttributeError:
            print("Action meanings: env.unwrapped.get_action_meanings() not found.")
        except Exception as e:
            print(f"Warning: Could not print action meanings: {e}")
        print("-------------------------------------------------------------")

    def _get_valid_key_mappings(self):
        """
        Get the valid key to action mappings from the environment.
        This function ensures we only use actions that are valid for the current game.
        """
        # create a default mapping
        mapping = {
            (pygame.K_UP,): 2,
            (pygame.K_RIGHT,): 3,
            (pygame.K_LEFT,): 4,
            (pygame.K_DOWN,): 5,
            (pygame.K_SPACE,): 1,
            (pygame.K_UP, pygame.K_RIGHT): 6,
            (pygame.K_UP, pygame.K_LEFT): 7,
            (pygame.K_DOWN, pygame.K_RIGHT): 8,
            (pygame.K_DOWN, pygame.K_LEFT): 9,
            (pygame.K_UP, pygame.K_SPACE): 10,
            (pygame.K_RIGHT, pygame.K_SPACE): 11,
            (pygame.K_LEFT, pygame.K_SPACE): 12,
            (pygame.K_DOWN, pygame.K_SPACE): 13,
            (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,
            (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,
            (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,
            (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17,
        }

        # sort all keys in the mapping to ensure invariance to order of pressed keys
        temp = {}
        for keys in mapping.keys():
            sorted_keys = tuple(sorted(keys))
            temp[sorted_keys] = mapping[keys]

        # filter out invalid actions
        valid_keys = set()
        for keys, action in temp.items():
            if action < self.env.action_space.n:
                valid_keys.add(keys)

        # create a new mapping with only valid actions
        temp = {keys: action for keys, action in temp.items() if keys in valid_keys}

        return temp

    # --- Helper method to print mappings ---
    def _print_action_meanings_and_key_mappings(self):
        """
        Print the available actions and their corresponding key mappings.
        """
        print("\n--- Action Meanings and Key Mappings ---")
        try:
            action_meanings = self.env.unwrapped.get_action_meanings()
            print(f"Available actions ({self.env.action_space.n}): {action_meanings}")

            if not self.keys2actions:
                print("Key mappings: None found or defined.")
                return

            print(f"Key mappings:")
            # Sort by action number for clarity
            sorted_mappings = sorted(self.keys2actions.items(), key=lambda item: item[1])

            for keys, action in sorted_mappings:
                 # Ensure action is valid before accessing meanings list
                 if 0 <= action < len(action_meanings):
                     key_names = [pygame.key.name(key).upper() for key in keys]
                     action_meaning = action_meanings[action]
                     print(f"  {' + '.join(key_names):<20} -> {action_meaning} (Action {action})")
                 else:
                     # This case should have been filtered by _get_valid_key_mappings
                     key_names = [pygame.key.name(key).upper() for key in keys]
                     print(f"  {' + '.join(key_names):<20} -> Invalid Action Index ({action})")

        except AttributeError:
             print("Action meanings: env.unwrapped.get_action_meanings() not found.")
        except Exception as e:
            print(f"Warning: Could not print action meanings: {e}")
        print("----------------------------------------")

    # --- Frame Saving (Unchanged) ---
    def save_frame(self, frame: np.ndarray) -> None:
        if not self.screenshot_dir: return
        if frame is None: print("Warning: Cannot save None frame."); return
        try:
            filepath = os.path.join(self.screenshot_dir, f"frame_{self.frame_counter:05d}.npy")
            np.save(filepath, frame)
            self.frame_counter += 1
        except Exception as e: print(f"Error saving frame: {e}")

    # --- Action Determination (Unchanged - Relies on correct keys2actions) ---
    def _get_action(self) -> int:
        """
        Get the action based on the currently pressed keys by looking up
        the sorted tuple of keys in the dynamically generated keys2actions dictionary.
        Defaults to action 0 (assumed NOOP) if no matching combination is pressed.
        """
        # Get the set of *relevant* keys currently held down
        pressed_relevant_keys = self.current_keys_down.intersection(self.relevant_keys)

        # Sort the set of keys currently pressed to ensure consistent tuple representation
        # Use the intersection to only consider keys that are part of our map
        pressed_keys_tuple = tuple(sorted(list(pressed_relevant_keys)))

        # Look up this exact tuple in the dictionary
        # Default to NOOP (0) if combination not found or tuple is empty
        return self.keys2actions.get(pressed_keys_tuple, 0)

    # --- Input Handling (Logic remains correct, checks against fetched keys2actions) ---
    def _handle_user_input(self) -> bool:
        """
        Handle user input via keyboard and mouse.
        Updates self.current_keys_down based on keys defined as relevant
        by the dynamic mapping process. Handles non-gameplay keys.
        """
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())  # If needed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to stop the game loop

            # --- MOUSE (Keep existing logic if needed) ---
            # ...

            # --- KEY DOWN ---
            elif event.type == pygame.KEYDOWN:
                key = event.key
                # Non-action keys first
                if key == pygame.K_p:
                    self.paused = not self.paused; print(f"Game {'paused' if self.paused else 'resumed'}")
                elif key == pygame.K_f:
                    self.frame_by_frame = not self.frame_by_frame; self.next_frame = False; print(
                        f"Frame-by-frame {'enabled' if self.frame_by_frame else 'disabled'}")
                elif key == pygame.K_n and self.frame_by_frame:
                    self.next_frame = True; print("Next frame")
                elif key == pygame.K_r:
                    print("Resetting environment...");
                    try:
                        self.env.reset(seed=42);
                        self.current_frame = self._safe_render()
                        print("Game reset")
                    except Exception as e:
                        print(f"Error resetting environment: {e}")
                elif key == pygame.K_s:
                    print("Saving frame..."); self.save_frame(self.current_frame)

                # Check if the pressed key is one of the keys relevant to gameplay actions
                elif key in self.relevant_keys:
                    self.current_keys_down.add(key)  # Add to the set of currently held keys

            # --- KEY UP ---
            elif event.type == pygame.KEYUP:
                key = event.key
                # Check if the released key is relevant to gameplay actions
                if key in self.relevant_keys:
                    self.current_keys_down.discard(key)  # Remove from the set

        return True  # Continue the game loop

    # --- Rendering (Unchanged) ---
    def _safe_render(self):
        try:
            frame = self.env.render()
            if frame is None: print("Warning: env.render() returned None. Using previous."); return self.current_frame
            return frame
        except Exception as e: print(f"Warning: Rendering error ({e}). Using previous."); return self.current_frame

    def run(self) -> None:
        running = True
        total_reward = 0
        steps = 0
        # Print controls once at the start
        print("\n--- Game Controls ---")
        print("  P: Pause/resume | F: Frame-by-frame | N: Next frame | R: Reset | S: Save Frame")
        print("  Use keys listed in 'Key Mappings' above for gameplay.")
        print("--------------------\n")

        try:
            while running:
                running = self._handle_user_input()  # Updates self.current_keys_down
                if not running: break

                if not self.paused and not (self.frame_by_frame and not self.next_frame):
                    action = self._get_action()  # Gets action based on current self.current_keys_down

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    self.current_frame = self._safe_render()
                    self.next_frame = False

                    # Update Title (Score display logic unchanged)
                    score_display = f"{total_reward:.0f}"
                    try:
                        ram = self.ale.getRAM()  # Get fresh RAM state
                        # This score logic is game-dependent and might need adjustment per game
                        if 'player_score' in self.ram_positions and self.ram_positions['player_score'] < len(ram):
                            score_display = f"{ram[self.ram_positions['player_score']]}"
                        elif 'score_1' in self.ram_positions:  # Example: Breakout BCD score (simplistic representation)
                            s1, s2, s3 = ram[self.ram_positions['score_1']], ram[self.ram_positions['score_2']], ram[
                                self.ram_positions['score_3']]
                            score_display = f"{s1:X}{s2:X}{s3:X}"  # Display as hex BCD digits
                    except Exception:
                        pass
                    pygame.display.set_caption(f"{self.game_name} | Score: {score_display} | Steps: {steps}")

                    if terminated or truncated:
                        print(
                            f"--- Episode Finished (Score: {score_display}, Reward: {total_reward:.2f}, Steps: {steps}) ---")
                        print("Resetting...")
                        self.env.reset(seed=42);
                        self.current_frame = self.env.render()
                        total_reward = 0;
                        steps = 0
                        pygame.display.set_caption(f"{self.game_name} | Score: 0 | Steps: 0")

                # Rendering (Logic unchanged)
                if self.current_frame is not None:
                    try:
                        frame_surface = pygame.Surface((self.env_render_shape_h_w[1], self.env_render_shape_h_w[0]))
                        pygame.pixelcopy.array_to_surface(frame_surface, self.current_frame.swapaxes(0, 1))
                        scaled_surface = pygame.transform.scale(frame_surface, (self.screen_width, self.screen_height))
                        self.screen.blit(scaled_surface, (0, 0))
                        pygame.display.flip()
                    except Exception as e:
                        print(f"Error during pygame rendering: {e}")
                else:
                    self.screen.fill((0, 0, 0)); pygame.display.flip()

                self.clock.tick(self.fps)
        except Exception as e:
            print(f"\n--- Error during game run: {e} ---"); traceback.print_exc()
        finally:
            self.close()

    # --- Cleanup (Unchanged) ---
    def close(self):
        print("Closing environment and quitting pygame.")
        try: self.env.close()
        except Exception as e: print(f"Error closing environment: {e}")
        pygame.quit()

# --- Main Execution (Unchanged) ---
def main():
    parser = argparse.ArgumentParser(description='Play Atari games using Gymnasium')
    parser.add_argument('-g', '--game', type=str, default='WordZapper', help='Name of the Atari game ROM (e.g., Pong, Breakout)')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor for the display window')
    parser.add_argument('--fps', type=int, default=30, help='Target frames per second')
    parser.add_argument('--screenshot-dir', type=str, default=None, help='Directory to save screenshots as .npy files (optional)')
    args = parser.parse_args()

    # if the screenshot_dir is None, set it to {game_name}_screenshots
    if args.screenshot_dir is None:
        args.screenshot_dir = f"{args.game}_screenshots"

    player = None
    try:
        player = AtariPlayer(game_name=args.game, render_scale=args.scale, screenshot_dir=args.screenshot_dir, fps=args.fps)
        player.run()
    except KeyboardInterrupt: print("\nExiting by user request.")
    except Exception as e: pass # Error already handled within Player class
    # No finally block needed here as player.close() is called in player.run()'s finally

if __name__ == "__main__":
    main()