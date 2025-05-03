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
import os # Import os module

# Constants for RAM panel layout (will be scaled)
RAM_N_COLS = 8
_BASE_SCALE = 4 # The scale factor these base dimensions were designed for
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
        self.render_scale = render_scale # Store the desired scale factor
        self.screenshot_dir = screenshot_dir # Store screenshot directory
        # self.frame_counter = 1 # Initialize frame counter to 1 initially

        # Create screenshot directory if it doesn't exist
        if self.screenshot_dir:
            os.makedirs(self.screenshot_dir, exist_ok=True)
            # --- Add logic to find the next frame counter ---
            start_frame_counter = 1 # Default starting number
            if os.path.isdir(self.screenshot_dir): # Check if the directory was successfully created
                try:
                    # List all files in the screenshot directory
                    existing_files = os.listdir(self.screenshot_dir)
                    # Filter files that match the pattern "frame_number.npy"
                    # Extract the number from the filename
                    frame_numbers = []
                    for filename in existing_files:
                        if filename.startswith("frame_") and filename.endswith(".npy"):
                            try:
                                # Remove "frame_" and ".npy" and convert the remaining part to an integer
                                number_str = filename[len("frame_"): -len(".npy")]
                                frame_number = int(number_str)
                                frame_numbers.append(frame_number)
                            except ValueError:
                                # Ignore files that match the pattern but have non-integer numbers
                                print(f"Warning: Ignoring file with non-integer frame number: {filename}")

                    if frame_numbers:
                        # Find the highest existing frame number and start from the next one
                        highest_frame_number = max(frame_numbers)
                        start_frame_counter = highest_frame_number + 1
                        print(f"Detected existing screenshots, starting frame counter from {start_frame_counter}")
                    else:
                        print("No existing screenshots found, starting frame counter from 1")

                except Exception as e:
                    # Handle potential errors during directory listing or file processing
                    print(f"Error checking existing screenshots: {e}. Starting frame counter from 1.")
                    start_frame_counter = 1

            self.frame_counter = start_frame_counter # Set the frame counter

            # --- End of logic to find the next frame counter ---
        else:
            self.frame_counter = 1 # If screenshot_dir is None or creation failed, start from 1


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
        self.current_frame = self.env.render() # Get initial frame (likely native size)

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
             if 'UPRIGHTFIRE' in action_meanings: self.keys2actions[tuple(sorted([pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE]))] = action_meanings.index('UPRIGHTFIRE')

             # Add more complex combos if needed, check action_meanings list first
             print(f"Key to action mapping: {self.keys2actions}")
        except AttributeError:
            print("Warning: Could not get action meanings. Using default NOOP mapping.")
            self.keys2actions = {(): 0}

        # RAM panel anchor/position is calculated in _init_pygame now

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

        # Create screenshot directory if it doesn't exist
        if self.screenshot_dir:
            os.makedirs(self.screenshot_dir, exist_ok=True)


    def _get_ram(self):
        if self.ale:
            return self.ale.getRAM()
        return None

    def _clone_state(self):
        if self.ale:
            return self.ale.cloneState()
        return None

    def _restore_state(self, state):
        if self.ale and state:
            self.ale.restoreState(state)
            self.ram = self._get_ram()
        else:
             print("Cannot restore state, ALE not available or state is None.")

    def _set_ram_value(self, index, value):
        if self.ale:
            if 0 <= index < 128 and 0 <= value < 256:
                self.ale.setRAM(index, value)
                self.ram = self._get_ram()
            else:
                 print(f"Error: Invalid RAM index {index} or value {value}")
        else:
             print("Cannot set RAM value, ALE not available.")


    def _init_pygame(self, sample_image):
        pygame.init()
        pygame.display.set_caption(f"Gymnasium Environment: ALE/{self.env.spec.name}")

        # Get NATIVE dimensions from the sample image (H, W, C)
        native_h, native_w, _ = sample_image.shape

        # Calculate TARGET scaled dimensions for the game display
        self.game_display_width = native_w * self.render_scale
        self.game_display_height = native_h * self.render_scale

        # Calculate effective RAM panel dimensions based on render_scale
        scale_ratio = self.render_scale / _BASE_SCALE
        self.effective_ram_render_width = round(_BASE_RAM_RENDER_WIDTH * scale_ratio)
        self.effective_ram_cell_width = round(_BASE_RAM_CELL_WIDTH * scale_ratio)
        self.effective_ram_cell_height = round(_BASE_RAM_CELL_HEIGHT * scale_ratio)
        self.effective_ram_grid_anchor_padding = round(_BASE_RAM_GRID_ANCHOR_PADDING * scale_ratio)
        self.effective_ram_col_spacing = round(_BASE_RAM_COL_SPACING * scale_ratio)
        self.effective_ram_row_spacing = round(_BASE_RAM_ROW_SPACING * scale_ratio)

        # Calculate total window size
        window_width = self.game_display_width + self.effective_ram_render_width
        # Make window height match the scaled game display height
        window_height = self.game_display_height
        window_size = (window_width, window_height)

        print(f"Native Game Size (W x H): {native_w} x {native_h}")
        print(f"Render Scale Factor: {self.render_scale}")
        print(f"Scaled Game Display Size (W x H): {self.game_display_width} x {self.game_display_height}")
        print(f"Calculated RAM Panel Width: {self.effective_ram_render_width}")
        print(f"Total Window Size (W x H): {window_size[0]} x {window_size[1]}")

        self.window = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()

        # Adjust font sizes based on scale_ratio
        self.ram_cell_id_font = pygame.font.SysFont(
            "Pixel12x10", round(_BASE_ID_FONT_SIZE * scale_ratio)
        )
        self.ram_cell_value_font = pygame.font.SysFont(
            "Pixel12x10", round(_BASE_VALUE_FONT_SIZE * scale_ratio)
        )

        # Update RAM grid anchor based on scaled game width and padding
        self.ram_grid_anchor_left = self.game_display_width + self.effective_ram_grid_anchor_padding
        self.ram_grid_anchor_top = self.effective_ram_grid_anchor_padding

    # --- Add the save_frame method from frame_extractor.py ---
    def save_frame(self, frame: np.ndarray) -> None:
        if not self.screenshot_dir:
            print("Screenshot directory not set. Cannot save frame.")
            return
        if frame is None:
            print("Warning: Cannot save None frame."); return
        try:
            # Use os.path.join for correct path construction
            filepath = os.path.join(self.screenshot_dir, f"frame_{self.frame_counter}.npy")
            np.save(filepath, frame)
            print(f"Frame saved to {filepath}") # Added confirmation print
            self.frame_counter += 1
        except Exception as e:
            print(f"Error saving frame: {e}")

    # --- End of save_frame method ---


    def run(self):
        self.running = True
        i = 0
        terminated = False
        truncated = False

        while self.running:
            self._handle_user_input()

            if not (self.frame_by_frame and not self.next_frame) and not self.paused:
                current_ram = deepcopy(self._get_ram())
                current_ale_state = self._clone_state()
                # Save the frame *before* stepping, used for stepping back
                current_display_frame = deepcopy(self.current_frame)

                if current_ram is not None and current_ale_state is not None:
                    self.saved_frames.append(
                        (current_ram, current_ale_state, current_display_frame)
                    )
                elif len(self.saved_frames.maxlen) > 0:
                    print("Warning: Could not save frame state (RAM or ALE state missing).")

                action = self._get_action()
                observation, reward, terminated, truncated, info = self.env.step(action)
                # Get the new frame data *after* stepping
                self.current_frame = self.env.render()

                self.past_ram = self.ram
                self.ram = self._get_ram()

                if self.past_ram is not None and self.ram is not None:
                    self.delta_render = []
                    min_len = min(len(self.ram), len(self.past_ram))
                    for j in range(min_len):
                        if self.ram[j] != self.past_ram[j]:
                            self.delta_render.append(j)
                    if len(self.ram) != len(self.past_ram):
                         print(f"Warning: RAM length changed from {len(self.past_ram)} to {len(self.ram)}")
                else:
                     self.delta_render = []

                if len(self.clicked_cells) > 0 and self.ram is not None:
                    selected_cell = self.clicked_cells[0]
                    if 0 <= selected_cell < len(self.ram):
                        current_value = int(self.ram[selected_cell])
                        if selected_cell in self.recorded_ram_states:
                            self.recorded_ram_states[selected_cell].append(current_value)
                    else:
                        print(f"Warning: Clicked cell index {selected_cell} out of bounds for RAM size {len(self.ram)}")
                        self.clicked_cells.remove(selected_cell)
                        if selected_cell in self.recorded_ram_states:
                           del self.recorded_ram_states[selected_cell]

                self.next_frame = False

                if terminated or truncated:
                    print(f"Episode finished (Terminated: {terminated}, Truncated: {truncated}). Resetting.")
                    self.initial_obs, self.info = self.env.reset()
                    self.current_frame = self.env.render()
                    self.past_ram = None
                    self.ram = self._get_ram()
                    self.delta_render = []
                    terminated, truncated = False, False


            self._render()
            self.clock.tick(30)

            i += 1
            # Removed the FPS print to avoid cluttering the terminal with RAM prints


        self.env.close()
        pygame.quit()
        plt.close('all')


    def _get_action(self):
        pressed_keys = tuple(sorted(list(self.current_keys_down)))
        print(f"Pressed keys: {[pygame.key.name(k) for k in pressed_keys]}")
        action = self.keys2actions.get(pressed_keys, 0)
        if self.env.action_space.contains(action):
            return action
        else:
            print(f"Invalid action {action}, defaulting to NOOP")
            return 0


    def _handle_user_input(self):
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check if click is within the scaled game display area
                    if self.current_mouse_pos[0] < self.game_display_width and self.current_mouse_pos[1] < self.game_display_height :
                        # Calculate click coordinates relative to the original ALE screen size
                        # Use self.env.observation_space.shape for native dims if it exists and is correct
                        try:
                            native_h, native_w, _ = self.env.observation_space.shape
                            scale_x = self.game_display_width / native_w
                            scale_y = self.game_display_height / native_h

                            if scale_x > 0 and scale_y > 0:
                                game_x = int(self.current_mouse_pos[0] / scale_x)
                                game_y = int(self.current_mouse_pos[1] / scale_y)
                                print(f"Clicked on game screen at approx original coords: ({game_x}, {game_y})")
                                # Removed find_causative_ram call as it depends on specific OCAtari analysis
                            else:
                                print("Could not determine scaling factor for game click coordinates.")
                        except AttributeError:
                             print("Could not get observation space shape for game click coordinates.")


                    # Click on RAM Panel
                    else:
                        cell_idx = self._get_cell_under_mouse()
                        if cell_idx is not None:
                            if cell_idx in self.clicked_cells:
                                self.clicked_cells.remove(cell_idx)
                                print(f"Stopped recording RAM values for cell {cell_idx}")
                                if cell_idx in self.recorded_ram_states:
                                    recorded_states = self.recorded_ram_states[cell_idx]
                                    if len(recorded_states) > 2:
                                        # Plotting logic remains, requires matplotlib
                                        try:
                                            first_derivative = [int(recorded_states[i+1]) - int(recorded_states[i]) for i in range(len(recorded_states)-1)]
                                            second_derivative = [int(first_derivative[i+1]) - int(first_derivative[i]) for i in range(len(first_derivative)-1)]

                                            print(f"Cell {cell_idx} Recorded States ({len(recorded_states)}): {recorded_states}")
                                            print(f"Cell {cell_idx} First Derivative ({len(first_derivative)}): {first_derivative}")
                                            print(f"Cell {cell_idx} Second Derivative ({len(second_derivative)}): {second_derivative}")

                                            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                                            ax1.plot(recorded_states, marker='o', linestyle='-', label="Recorded States")
                                            ax1.set_title(f"RAM Cell {cell_idx}: Recorded State & Derivatives")
                                            ax1.set_ylabel("State Value"); ax1.grid(True); ax1.legend()
                                            ax2.plot(first_derivative, marker='o', linestyle='-', color='orange', label="First Derivative")
                                            ax2.set_ylabel("1st Deriv."); ax2.grid(True); ax2.legend()
                                            ax3.plot(second_derivative, marker='o', linestyle='-', color='green', label="Second Derivative")
                                            ax3.set_xlabel("Time (Frames)"); ax3.set_ylabel("2nd Deriv."); ax3.grid(True); ax3.legend()
                                            fig.tight_layout(); plt.show()
                                        except Exception as e:
                                            print(f"Error plotting data for cell {cell_idx}: {e}")
                                    else:
                                        print(f"Not enough data ({len(recorded_states)}) for cell {cell_idx} to plot derivatives.")
                                    del self.recorded_ram_states[cell_idx]
                            else:
                                if len(self.clicked_cells) == 0: # Only allow recording one cell at a time
                                    self.clicked_cells.append(cell_idx)
                                    self.recorded_ram_states[cell_idx] = []
                                    print(f"Started recording RAM values for cell {cell_idx}")
                                else:
                                     print("Another cell is already selected for recording. Click it again to stop recording first.")


                elif event.button == 3: # Right Click -> Hide/Unhide
                    cell_idx = self._get_cell_under_mouse()
                    if cell_idx is not None:
                        if cell_idx in self.no_render: self.no_render.remove(cell_idx)
                        else: self.no_render.append(cell_idx)

                elif event.button == 2: # Middle Click -> Toggle Red
                    cell_idx = self._get_cell_under_mouse()
                    if cell_idx is not None:
                        if cell_idx in self.red_render: self.red_render.remove(cell_idx)
                        else: self.red_render.append(cell_idx)

                elif event.button == 4: # Wheel Up -> Increment RAM
                    cell_idx = self._get_cell_under_mouse()
                    if cell_idx is not None and self.ale: self._increment_ram_value_at(cell_idx)
                elif event.button == 5: # Wheel Down -> Decrement RAM
                    cell_idx = self._get_cell_under_mouse()
                    if cell_idx is not None and self.ale: self._decrement_ram_value_at(cell_idx)


            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p: self.paused = not self.paused; print(f"{'Paused' if self.paused else 'Resumed'}")
                elif event.key == pygame.K_f: self.frame_by_frame = not self.frame_by_frame; self.next_frame = False; print(f"{'Frame-by-frame' if self.frame_by_frame else 'Continuous'}")
                elif event.key == pygame.K_n and self.frame_by_frame: self.next_frame = True; print("Next Frame")
                elif event.key == pygame.K_b: # Step Back
                    if self.frame_by_frame and self.ale:
                        if len(self.saved_frames) > 0:
                            print("Going back one frame")
                            _, previous_state, previous_display_frame = self.saved_frames.pop()
                            self._restore_state(previous_state)
                            self.current_frame = previous_display_frame # Restore visual frame for render

                            if len(self.saved_frames) > 0: self.past_ram = self.saved_frames[-1][0]
                            else: self.past_ram = None

                            if self.past_ram is not None and self.ram is not None:
                                self.delta_render = []
                                min_len = min(len(self.ram), len(self.past_ram))
                                for j in range(min_len):
                                     if self.ram[j] != self.past_ram[j]: self.delta_render.append(j)
                            else: self.delta_render = []
                            self._render()
                        else: print("No previous frames saved.")
                    elif not self.ale: print("Cannot step back, ALE interface not available.")

                elif event.key == pygame.K_r: # Reset
                    print("Resetting environment")
                    self.initial_obs, self.info = self.env.reset()
                    self.current_frame = self.env.render()
                    self.past_ram = None; self.ram = self._get_ram(); self.delta_render = []
                    terminated, truncated = False, False

                elif event.key == pygame.K_c: # Clone State
                     if self.paused and self.ale:
                         state_to_save = self._clone_state()
                         if state_to_save:
                             save_path = f"gym_ale_state_{self.game_name}.pkl"
                             try:
                                 with open(save_path, "wb") as f: pkl.dump(state_to_save, f)
                                 print(f"ALE state saved to {save_path}")
                             except Exception as e: print(f"Error saving state: {e}")
                         else: print("Failed to clone state.")
                     elif not self.ale: print("Cannot save state, ALE not available.")
                     elif not self.paused: print("Pause the game ('P') before saving state ('C').")

                # --- Add Key Binding for Save Frame ('S') ---
                elif event.key == pygame.K_s:
                    print("Save frame key pressed. Calling save_frame...")
                    self.save_frame(self.current_frame)
                # --- End of Save Frame Key Binding ---


                elif event.key == pygame.K_ESCAPE:
                    if self.active_cell_idx is not None: self._unselect_active_cell()

                elif self.active_cell_idx is not None: # Handle RAM input
                    if pygame.K_0 <= event.key <= pygame.K_9: self.current_active_cell_input += str(event.key - pygame.K_0)
                    elif pygame.K_KP0 <= event.key <= pygame.K_KP9:
                         numpad_map = {pygame.K_KP0: '0', pygame.K_KP1: '1', pygame.K_KP2: '2', pygame.K_KP3: '3', pygame.K_KP4: '4', pygame.K_KP5: '5', pygame.K_KP6: '6', pygame.K_KP7: '7', pygame.K_KP8: '8', pygame.K_KP9: '9'}
                         char = numpad_map.get(event.key)
                         if char: self.current_active_cell_input += char
                    elif event.key == pygame.K_BACKSPACE: self.current_active_cell_input = self.current_active_cell_input[:-1]
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        if len(self.current_active_cell_input) > 0:
                            try:
                                new_value = int(self.current_active_cell_input)
                                self._set_ram_value(self.active_cell_idx, new_value)
                            except ValueError: print("Invalid input for RAM value.")
                        self._unselect_active_cell()
                else: # Handle Action keys
                    key_tuple = (event.key,)
                    is_action_key = False
                    # Check if the key is part of any action combination
                    for keys in self.keys2actions.keys():
                         if isinstance(keys, tuple) and event.key in keys: is_action_key = True; break
                    # If it's a single key mapping, check that
                    if not is_action_key and key_tuple in self.keys2actions: is_action_key = True

                    if is_action_key: self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:
                 if event.key in self.current_keys_down: self.current_keys_down.remove(event.key)


        return True  # Continue the game loop


    def _render(self):
        self.window.fill((0, 0, 0))
        self._render_game_screen() # Renders the scaled game screen
        self._render_ram()
        self._render_hover()
        pygame.display.flip()
        pygame.event.pump()


    def _render_game_screen(self):
        if self.current_frame is not None:
            try:
                # Create surface from the frame data (usually H, W, C)
                # Transpose to (W, H, C) for make_surface
                # frame_surface = pygame.surfarray.make_surface(np.transpose(self.current_frame, (1, 0, 2)))
                # Using pixelcopy might be slightly faster
                native_h, native_w, _ = self.current_frame.shape
                frame_surface = pygame.Surface((native_w, native_h))
                pygame.pixelcopy.array_to_surface(frame_surface, self.current_frame.swapaxes(0, 1))


                # Scale the surface to the target display size
                scaled_surface = pygame.transform.scale(frame_surface, (self.game_display_width, self.game_display_height))

                # Blit the scaled surface
                self.window.blit(scaled_surface, (0, 0))
            except Exception as e:
                print(f"Error rendering game screen: {e}")
                # Optionally draw a placeholder if rendering fails
                pygame.draw.rect(self.window, (50, 0, 0), (0, 0, self.game_display_width, self.game_display_height))


    def _render_ram(self):
        if self.ram is None:
            font = pygame.font.SysFont("Arial", 18) # Use a default font
            text = font.render("RAM not accessible", True, (255, 0, 0))
            # Center text in the RAM panel area
            ram_panel_rect = pygame.Rect(self.ram_grid_anchor_left, self.ram_grid_anchor_top,
                                         self.effective_ram_render_width, self.game_display_height - 2*self.effective_ram_grid_anchor_padding)
            text_rect = text.get_rect(center=ram_panel_rect.center)
            self.window.blit(text, text_rect)
            return

        for i, value in enumerate(self.ram):
             if i < 128: self._render_ram_cell(i, value)


    def _get_ram_value_at(self, idx: int):
        if self.ram is not None and 0 <= idx < len(self.ram): return self.ram[idx]
        return 0


    def _set_ram_value_at(self, idx: int, value: int):
        self._set_ram_value(idx, value)


    def _increment_ram_value_at(self, idx: int):
        if self.ale and self.ram is not None and 0 <= idx < len(self.ram):
            value = self._get_ram_value_at(idx)
            self._set_ram_value_at(idx, min(value + 1, 255))


    def _decrement_ram_value_at(self, idx: int):
        if self.ale and self.ram is not None and 0 <= idx < len(self.ram):
            value = self._get_ram_value_at(idx)
            self._set_ram_value_at(idx, max(value - 1, 0))


    def _render_ram_cell(self, cell_idx, value):
        is_active = cell_idx == self.active_cell_idx
        is_candidate = cell_idx in self.candidate_cell_ids
        is_red = cell_idx in self.red_render
        has_delta = cell_idx in self.delta_render
        is_clicked = cell_idx in self.clicked_cells

        x, y, w, h = self._get_ram_cell_rect(cell_idx)

        if is_active: color = (70, 70, 30)
        elif is_candidate: color = (15, 45, 100)
        elif is_clicked and has_delta: color = (238, 130, 238)
        elif is_clicked: color = (20, 150, 20)
        elif has_delta: color = (255, 165, 0)
        elif is_red: color = (150, 20, 20)
        else: color = (20, 20, 20)
        pygame.draw.rect(self.window, color, [x, y, w, h])

        if cell_idx in self.no_render: return

        id_color = (100, 150, 150)
        if is_active: id_color = (150, 150, 30)
        elif is_candidate: id_color = (20, 60, 200)
        text = self.ram_cell_id_font.render(str(cell_idx), True, id_color, None)
        text_rect = text.get_rect(); text_rect.topleft = (x + 3, y + 3)
        self.window.blit(text, text_rect)

        val_str = str(value); val_color = (200, 200, 200)
        if is_active: val_str = self.current_active_cell_input + "_"; val_color = (255, 255, 50)
        elif has_delta: val_color = (0,0,0)
        elif is_candidate: val_color = (30, 90, 255)
        text = self.ram_cell_value_font.render(val_str, True, val_color, None)
        text_rect = text.get_rect(); text_rect.bottomright = (x + w - 3, y + h - 3)
        self.window.blit(text, text_rect)

        if has_delta and self.past_ram is not None and cell_idx < len(self.past_ram):
            delta_color = (255, 0, 0)
            try:
                delta_value = int(self.ram[cell_idx]) - int(self.past_ram[cell_idx])
                delta_text = f"Δ{delta_value:+d}"
                text = self.ram_cell_value_font.render(delta_text, True, delta_color, None)
                text_rect = text.get_rect(); text_rect.bottomleft = (x + 3, y + h - 3)
                self.window.blit(text, text_rect)
            except (IndexError, TypeError): pass


    def _get_ram_cell_rect(self, idx: int):
        row = idx // RAM_N_COLS
        col = idx % RAM_N_COLS
        # Use the effective spacing/size calculated in _init_pygame
        x = round(self.ram_grid_anchor_left + col * self.effective_ram_col_spacing)
        y = round(self.ram_grid_anchor_top + row * self.effective_ram_row_spacing)
        w = self.effective_ram_cell_width
        h = self.effective_ram_cell_height
        return x, y, w, h


    def _render_hover(self):
        cell_idx = self._get_cell_under_mouse()
        if cell_idx is not None and cell_idx != self.active_cell_idx:
            x, y, w, h = self._get_ram_cell_rect(cell_idx)
            hover_surface = pygame.Surface((w, h))
            hover_surface.set_alpha(60)
            hover_surface.fill((255, 255, 255))
            self.window.blit(hover_surface, (x, y))


    def _get_cell_under_mouse(self):
        x, y = self.current_mouse_pos
        # Check bounds relative to the RAM panel area
        if x > self.ram_grid_anchor_left and y > self.ram_grid_anchor_top:
            # Use the effective spacing for calculation
            col_width = self.effective_ram_col_spacing
            row_height = self.effective_ram_row_spacing
            if col_width > 0 and row_height > 0:
                col = int((x - self.ram_grid_anchor_left) // col_width)
                row = int((y - self.ram_grid_anchor_top) // row_height)
                if 0 <= col < RAM_N_COLS and 0 <= row < 16:
                    idx = row * RAM_N_COLS + col
                    if 0 <= idx < 128: return idx
        return None


    def _unselect_active_cell(self):
        if self.active_cell_idx is not None:
            idx_to_redraw = self.active_cell_idx
            self.active_cell_idx = None
            self.current_active_cell_input = ""
            if self.ram is not None and 0 <= idx_to_redraw < len(self.ram):
                 value = self._get_ram_value_at(idx_to_redraw)
                 self._render_ram_cell(idx_to_redraw, value) # Redraw needed cell
        else:
             self.active_cell_idx = None
             self.current_active_cell_input = ""


    # find_causative_ram method removed as it relies on specific OCAtari analysis

# --- Main Execution ---
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Gymnasium ALE RAM Explorer with Frame Saving") # Updated description
    parser.add_argument("-g", "--game", type=str, default="VideoPinball", help="Name of the Atari game (e.g., 'Pong', 'Breakout').")
    # Add scale argument
    parser.add_argument('--scale', type=int, default=4, help='Scale factor for the game display window')
    parser.add_argument("-ls", "--load_state", type=str, default=None, help="Path to a pickled ALE state file (.pkl) to load.")
    parser.add_argument("-nr", "--no_render", type=int, default=[], nargs="+", help="List of RAM cell indices (0-127) to hide.")
    parser.add_argument("-nra", "--no_render_all", action="store_true", help="Hide all RAM cells.")
    # Add screenshot-dir argument
    parser.add_argument('--screenshot-dir', type=str, default=None, help='Directory to save screenshots as .npy files (optional)')

    args = parser.parse_args()

    # if the screenshot_dir is None, set it to {game_name}_screenshots
    if args.screenshot_dir is None:
        args.screenshot_dir = f"../new_screenshots"


    if args.no_render_all:
        args.no_render = list(range(128))

    # Pass scale and screenshot_dir arguments to Renderer
    renderer = Renderer(env_name=args.game, no_render=args.no_render, render_scale=args.scale, screenshot_dir=args.screenshot_dir)

    if args.load_state:
        if renderer.ale:
            try:
                with open(args.load_state, "rb") as f:
                    state_to_load = pkl.load(f)
                    renderer._restore_state(state_to_load)
                    renderer.current_frame = renderer.env.render() # Update visual frame
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