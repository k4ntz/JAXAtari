import pygame
import numpy as np
import os
import argparse
from typing import Dict, Tuple, List
import time
from ocatari.core import OCAtari, UPSCALE_FACTOR, AVAILABLE_GAMES


class OCAtariPlayer:
    """
    A generalized Atari game player using OCAtari, which handles ROM loading automatically.
    """

    def __init__(self,
                 game_name: str = "Boxing",
                 mode: str = "ram",
                 hud: bool = True,
                 obs_mode: str = "ori",
                 render_scale: int = None,
                 screenshot_dir: str = None,
                 fps: int = 30):
        """
        Initialize the OCAtari game player.

        Args:
            game_name: Name of the Atari game to load (e.g., 'Boxing', 'Breakout', 'Pong')
            mode: OCAtari mode ('ram', 'vision', or 'both')
            hud: Whether to include HUD elements in object detection
            obs_mode: Observation mode ('ori', 'dqn', or 'obj')
            render_scale: Scale factor for rendering (defaults to OCAtari's UPSCALE_FACTOR)
            screenshot_dir: Directory to save frame screenshots (None = no screenshots)
            fps: Target frames per second
        """
        # Create the OCAtari environment
        self.game_name = game_name

        try:
            # Create OCAtari environment
            self.env = OCAtari(
                game_name,
                mode=mode,
                hud=hud,
                render_mode="rgb_array",
                #render_oc_overlay=True,
                obs_mode=obs_mode,
                frameskip=1,
            )

            # Reset environment to get initial observation
            self.env.reset()
            self.current_frame = self.env.render()

            # Initialize rendering attributes
            self.render_scale = render_scale if render_scale is not None else UPSCALE_FACTOR

            # Initialize pygame
            pygame.init()
            self.env_render_shape = self.current_frame.shape[:2]
            self.screen_width = int(self.env_render_shape[1] * self.render_scale)  # Note the swap for typical Atari orientation
            self.screen_height = int(self.env_render_shape[0] * self.render_scale) # Note the swap for typical Atari orientation
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(f"OCAtari Player: {game_name}")
            self.clock = pygame.time.Clock()
            self.fps = fps

            # Initialize frame saving
            self.screenshot_dir = screenshot_dir
            if screenshot_dir:
                os.makedirs(screenshot_dir, exist_ok=True)
            self.frame_counter = 1

            # Input handling attributes
            self.current_keys_down = set()
            self.current_mouse_pos = None

            # Get action mappings and ensure they're valid
            self.keys2actions = self._get_valid_key_mappings()

            # Game state attributes
            self.paused = False
            self.frame_by_frame = False
            self.next_frame = False

            # Initialize RAM editing tools - track which RAM positions are important
            self.ram_positions = self._get_game_ram_positions()

            # Print information about the environment
            print(f"Loaded {game_name} environment")
            print(f"Observation shape: {self.env_render_shape}")
            print(f"Action space: {self.env.action_space}")
            self._print_action_meanings_and_key_mappings()

            # Print RAM manipulation instructions
            self._print_ram_instructions()

        except Exception as e:
            pygame.quit()
            raise e

    def _get_game_ram_positions(self):
        """
        Get the important RAM positions for the current game.
        This helps us know which RAM positions to modify for specific functionality.
        """
        # Default is an empty dictionary
        ram_positions = {}

        # Game-specific RAM positions
        if self.game_name.lower() == "pong":
            ram_positions = {
                "player_score": 14,  # RAM position for player score
                "enemy_score": 13,  # RAM position for enemy score
                "ball_x": 49,  # Ball X position
                "ball_y": 54,  # Ball Y position
                "player_y": 51,  # Player Y position
                "enemy_y": 50,  # Enemy Y position
            }
        # Add more games here as needed
        elif self.game_name.lower() == "breakout":
            ram_positions = {
                "score_1": 84,  # Score digit 1
                "score_2": 85,  # Score digit 2
                "score_3": 86,  # Score digit 3
                "lives": 73,  # Number of lives
                "ball_x": 99,  # Ball X position
                "ball_y": 101,  # Ball Y position
            }
        elif self.game_name.lower() == "boxing":
            ram_positions = {
                "player_score": 3,  # Player score
                "enemy_score": 2,  # Enemy score
            }
        elif self.game_name.lower() == "airraid":
            ram_positions = {
                # Score digit values - use these instead of the problematic score calculation
                "score_digit_a": 40,  # First score digit
                "score_digit_b": 41,  # Second score digit
                "score_digit_c": 42,  # Third score digit

                # Player position
                "player_x": 19,  # Player X position (minus 3 for actual position)

                # Buildings
                "building1_damage": 27,  # Building 1 damage level
                "building1_x": 20,      # Building 1 X position
                "building2_damage": 28,  # Building 2 damage level
                "building2_x": 21,      # Building 2 X position

                # Enemy positions and types
                "enemy1_y": 29,        # Enemy 1 Y position (195 - this value)
                "enemy2_y": 30,        # Enemy 2 Y position (195 - this value)
                "enemy3_y": 31,        # Enemy 3 Y position (195 - this value)
                "enemy1_type": 63,     # Enemy 1 type (0-1: 25pt, 2-3: 50pt, 4-5: 75pt, 6-7: 100pt)
                "enemy2_type": 64,     # Enemy 2 type
                "enemy3_type": 65,     # Enemy 3 type
                "enemy1_x": 66,        # Enemy 1 X position
                "enemy2_x": 67,        # Enemy 2 X position
                "enemy3_x": 68,        # Enemy 3 X position

                # Missiles
                "player_missile_y": 85,  # Player missile Y position (194 - this value)
                "player_missile_x": 86,  # Player missile X position (minus 5)
                "enemy_missile_y": 83,   # Enemy missile Y position (193 - this value)
                "enemy_missile_x": 84,   # Enemy missile X position (minus 5)

                # Lives
                "lives": 39,            # Number of lives (1 or 2)

                # Display mode
                "grayscale_mode": 84,   # 248 for color mode, otherwise grayscale
            }

        return ram_positions

    def _print_ram_instructions(self):
        """
        Print game-specific RAM manipulation instructions.
        """
        print("\nRAM Manipulation Controls:")
        print("  Middle Mouse Button: Increase player score by 1")

        # Game-specific instructions
        if self.game_name.lower() == "pong":
            print("  In Pong, this modifies RAM position 14")
        elif self.game_name.lower() == "breakout":
            print("  In Breakout, this modifies RAM positions 84-86")
        elif self.game_name.lower() == "boxing":
            print("  In Boxing, this modifies RAM position 3")
        elif self.game_name.lower() == "airraid":
            print("  In AirRaid, this modifies RAM positions 40-42 (score digits)")
            print("\nAdditional Air Raid RAM positions:")
            print("  Player: pos 19 (X)")
            print("  Enemies: pos 29-31 (Y), pos 63-65 (type), pos 66-68 (X)")
            print("  Buildings: pos 27-28 (damage), pos 20-21 (X)")

    def _get_valid_key_mappings(self):
        """
        Get the valid key to action mappings from the environment.
        This function ensures we only use actions that are valid for the current game.
        """
        try:
            keys2actions = self.env.unwrapped.get_keys_to_action()
            action_meanings = self.env.unwrapped.get_action_meanings()

            # Filter out any actions that are outside the valid range
            valid_keys2actions = {}
            for keys, action in keys2actions.items():
                if 0 <= action < len(action_meanings):
                    valid_keys2actions[keys] = action

            return valid_keys2actions
        except Exception as e:
            print(f"Warning: Could not get valid key mappings: {e}")
            # Return a default mapping if the original fails
            return {
                (pygame.K_UP,): 2,  # UP
                (pygame.K_DOWN,): 3,  # DOWN
                (pygame.K_LEFT,): 4,  # LEFT
                (pygame.K_RIGHT,): 5,  # RIGHT
                (pygame.K_SPACE,): 1,  # FIRE
            }

    def _print_action_meanings_and_key_mappings(self):
        """
        Print the available actions and their key mappings.
        This function handles the case where the action index is out of range.
        """
        try:
            action_meanings = self.env.unwrapped.get_action_meanings()
            print(f"Available actions: {action_meanings}")

            print(f"Key mappings:")
            for keys, action in self.keys2actions.items():
                if 0 <= action < len(action_meanings):
                    key_names = [pygame.key.name(key).upper() for key in keys]
                    action_meaning = action_meanings[action]
                    print(f"  {' + '.join(key_names)} -> {action_meaning} (action {action})")
                else:
                    key_names = [pygame.key.name(key).upper() for key in keys]
                    print(f"  {' + '.join(key_names)} -> Unknown action (index {action} out of range)")
        except Exception as e:
            print(f"Warning: Could not print action meanings: {e}")

    def save_frame(self, frame: np.ndarray) -> None:
        """
        Save the current frame as a NumPy array.

        Args:
            frame: Frame to save
        """
        if not self.screenshot_dir:
            return

        filepath = os.path.join(self.screenshot_dir, f"frame_{self.frame_counter}.npy")
        np.save(filepath, frame)
        print(f"Frame saved as NumPy array: {filepath}")
        self.frame_counter += 1

    def _get_action(self):
        """
        Get the action based on the currently pressed keys.
        """
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        """
        Handle user input via keyboard and mouse.
        """
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to stop the game loop

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Click handling logic can be added here
                    pass
                elif event.button == 2:
                    # TODO: here you can put ram manipulations to get the sprites faster (for example increasing the score manually)
                    self.env.set_ram(16, 10)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # Pause/resume
                    self.paused = not self.paused
                    print("Game " + ("paused" if self.paused else "resumed"))

                elif event.key == pygame.K_f:  # Frame by frame mode
                    self.frame_by_frame = not self.frame_by_frame
                    self.next_frame = False
                    print("Frame-by-frame mode " + ("enabled" if self.frame_by_frame else "disabled"))

                elif event.key == pygame.K_n:  # Next frame in frame-by-frame mode
                    self.next_frame = True
                    print("Next frame")

                elif event.key == pygame.K_r:  # Reset game
                    self.env.reset()
                    print("Game reset")

                elif event.key == pygame.K_s:  # Save screenshot
                    self.save_frame(self.current_frame)

                elif event.key == pygame.K_d:  # Debug RAM (print current RAM state)
                    ram = self.env.get_ram()
                    print(f"\nRAM state for important positions:")
                    for name, pos in self.ram_positions.items():
                        print(f"  {name}: {ram[pos]} (position {pos})")

                elif [x for x in self.keys2actions.keys() if event.key in x]:
                    # Environment action key pressed
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:
                if [x for x in self.keys2actions.keys() if event.key in x]:
                    # Environment action key released
                    if event.key in self.current_keys_down:
                        self.current_keys_down.remove(event.key)

        return True  # Continue the game loop

    def _safe_render(self):
        """
        Safely render the current frame, handling any errors that might occur.
        This addresses the overflow error that can happen during rendering.
        """
        try:
            return self.env.render()
        except OverflowError as e:
            print(f"Warning: Rendering error detected ({e}). Applying workaround...")

            # Try to fix by manually adjusting problematic objects in the environment
            for obj in self.env.objects:
                # Cap object positions to reasonable values (0-200)
                x, y = obj.xy
                w, h = obj.wh

                # Cap position values to prevent overflow
                if x > 200 or y > 200:
                    obj._xy = min(x, 200), min(y, 200)
                    print(f"Fixed object position: {x},{y} -> {obj.xy}")

                # Cap width/height values to prevent overflow
                if w > 100 or h > 100:
                    obj.wh = min(w, 100), min(h, 100)
                    print(f"Fixed object size: {w},{h} -> {obj.wh}")

            # Try rendering again with fixed objects
            try:
                return self.env.render()
            except Exception as e2:
                print(f"Still cannot render. Using previous frame. ({e2})")
                # Return the previous frame as a fallback
                return self.current_frame

    def run(self) -> None:
        """Main game loop."""
        running = True
        total_reward = 0
        steps = 0

        print("\nGame Controls:")
        print("  P: Pause/resume game")
        print("  F: Toggle frame-by-frame mode")
        print("  N: Next frame (when in frame-by-frame mode)")
        print("  R: Reset game")
        print("  S: Save current frame as NumPy array")
        print("  D: Debug RAM (print important RAM values)")
        print("  Game-specific controls are listed above\n")

        try:
            while running:
                # Handle user input
                running = self._handle_user_input()
                if not running:
                    break

                # Process game logic if not paused
                if not (self.frame_by_frame and not self.next_frame) and not self.paused:
                    # Get action based on pressed keys
                    action = self._get_action()

                    # Execute action
                    obs, reward, truncated, terminated, info = self.env.step(action)
                    total_reward += reward
                    steps += 1

                    # Update the current frame with safe rendering
                    self.current_frame = self._safe_render()

                    # Reset the next_frame flag
                    self.next_frame = False

                    # Update the window title with game information
                    score_text = f"{self.game_name} | Score: {total_reward:.1f} | Steps: {steps}"
                    pygame.display.set_caption(score_text)

                    # Check for game over
                    if terminated or truncated:
                        print(f"Game over! Final score: {total_reward}")
                        self.env.reset()
                        total_reward = 0
                        steps = 0

                # Render the current frame to the screen
                try:
                    frame_surface = pygame.Surface(self.env_render_shape)
                    pygame.pixelcopy.array_to_surface(frame_surface, self.current_frame)
                    scaled_surface = pygame.transform.scale(frame_surface, (self.screen_width, self.screen_height))
                    self.screen.blit(scaled_surface, (0, 0))
                    pygame.display.flip()
                except Exception as e:
                    print(self.env_render_shape)

                    print(f"Error rendering to screen: {e}")
                    # Try to recover by re-initializing the surface
                    try:
                        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                    except:
                        pass

                # Cap the framerate
                self.clock.tick(self.fps)
        except Exception as e:
            print(f"Error during game execution: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """Clean up resources."""
        try:
            self.env.close()
        except:
            pass
        pygame.quit()


def main():
    """Parse command line arguments and run the game."""
    parser = argparse.ArgumentParser(description='Play any Atari game using OCAtari')
    parser.add_argument('--game', type=str, default='Pong',
                        help='Name of the Atari game to play (e.g., Boxing, Breakout, Pong)')
    parser.add_argument('--mode', type=str, default='ram', choices=['ram', 'vision', 'both'],
                        help='OCAtari mode for object detection')
    parser.add_argument('--no-hud', action='store_true', default=False,  # Changed default to False for clarity
                        help='Exclude HUD elements in object detection')
    parser.add_argument('--obs-mode', type=str, default='ori', choices=['ori', 'dqn', 'obj'],
                        help='Observation mode')
    parser.add_argument('--scale', type=int, default=4,
                        help='Scale factor for the display window')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frames per second')
    parser.add_argument('--screenshot-dir', type=str, default="new_screenshots",
                        help='Directory to save screenshots (optional)')

    args = parser.parse_args()

    # Print available games if requested
    if args.game.lower() == 'list':
        print("Available Atari games in OCAtari:")
        for game in sorted(AVAILABLE_GAMES):
            print(f"  {game}")
        return

    try:
        player = OCAtariPlayer(
            game_name=args.game,
            mode=args.mode,
            hud=not args.no_hud,
            obs_mode=args.obs_mode,
            render_scale=args.scale,
            screenshot_dir=args.screenshot_dir,
            fps=args.fps
        )
        player.run()
    except KeyboardInterrupt:
        print("Exiting by user request")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'player' in locals():
            player.close()


if __name__ == "__main__":
    main()