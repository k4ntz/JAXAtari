import uuid

import numpy as np
import pygame
from ocatari.core import OCAtari, UPSCALE_FACTOR
from tqdm import tqdm
from collections import deque
from copy import deepcopy
import pickle as pkl
import atexit
import matplotlib.pyplot as plt
import csv
import os

"""

This script is a modified version of the remgui.py script that comes with the OCAtari package.
The main difference is that this script highlights RAM cells which differ from the previous RAM state.
It also allows the user to record the RAM states of certaincells and observe the first and second 
derivatives of the recorded RAM states.

The script has been modified to include the following features:
- Highlight RAM cells which differ from the previous RAM state (deltas)
- The script allows the user to record the RAM states of certain cells and observe the first and second derivatives of the recorded RAM states.

"""

RAM_RENDER_WIDTH = round(1000 * (UPSCALE_FACTOR / 4))
RAM_N_COLS = 8
RAM_CELL_WIDTH = round(115 * (UPSCALE_FACTOR / 4))
RAM_CELL_HEIGHT = round(45 * (UPSCALE_FACTOR / 4))
Z_RESET_HEIGHT = 15

class Renderer:
    window: pygame.Surface
    clock: pygame.time.Clock
    env: OCAtari

    def __init__(self, env_name, mode="ram", no_render=[]):
        self.env = OCAtari(
            env_name,
            mode=mode,
            hud=True,
            render_mode="rgb_array",
            render_oc_overlay=True,
            frameskip=1,
            obs_mode="obj",
        )

        self.env.reset(seed=42)
        self.current_frame = self.env.render()
        self._init_pygame(self.current_frame)
        self.paused = False

        self.current_keys_down = set()
        self.current_mouse_pos = None
        self.keys2actions = self.env.unwrapped.get_keys_to_action()

        self.ram_grid_anchor_left = self.env_render_shape[0] + round(
            28 * (UPSCALE_FACTOR / 4)
        )
        self.ram_grid_anchor_top = round(28 * (UPSCALE_FACTOR / 4))

        self.active_cell_idx = None
        self.candidate_cell_ids = []
        self.current_active_cell_input: str = ""
        self.no_render = no_render
        self.red_render = []

        self.saved_frames = deque(maxlen=20)  # tuples of ram, state, image
        self.frame_by_frame = False
        self.next_frame = False

        self.past_ram = None
        self.ram = None
        self.delta_render = []

        self.clicked_cells = []
        self.recorded_ram_states = {}


    def _init_pygame(self, sample_image):
        pygame.init()
        pygame.display.set_caption("OCAtari Environment")
        self.env_render_shape = sample_image.shape[:2]
        window_size = (
            self.env_render_shape[0] + RAM_RENDER_WIDTH,
            self.env_render_shape[1],
        )

        # Print Shapes of environment
        print(f"Render-Shape (0) aka height: {self.env_render_shape[0]}")
        print(f"Render-Shape (1) aka width: {self.env_render_shape[1]}")

        print(f"Actual shape of environment: {self.env.image_size}")
        print(
            f"Scaling Factor of env -> rendering: {self.env_render_shape[0]/self.env.image_size[0]}"
        )

        self.window = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()

        print(self.clock.get_fps())

        self.ram_cell_id_font = pygame.font.SysFont(
            "Pixel12x10", round(25 * (UPSCALE_FACTOR / 4))
        )
        self.ram_cell_value_font = pygame.font.SysFont(
            "Pixel12x10", round(30 * (UPSCALE_FACTOR / 4))
        )

    def run(self):
        self.running = True
        i = 0
        while self.running:
            self._handle_user_input()
            if not (self.frame_by_frame and not (self.next_frame)) and not self.paused:
                self.saved_frames.append(
                    (
                        deepcopy(self.env.get_ram()),
                        self.env._ale.cloneState(),
                        self.current_frame,
                    )
                )  # ram, state, image (rgb)
                action = self._get_action()
                reward = self.env.step(action)[1]

                self.current_frame = self.env.render().copy()

                ale = self.env._ale
                self.past_ram = self.ram
                self.ram = ale.getRAM()

                if self.past_ram is not None and self.ram is not None:
                    self.delta_render = []
                    delta = self.ram - self.past_ram

                    for j, value in enumerate(delta):
                        if value != 0:
                            self.delta_render.append(j)

                if len(self.clicked_cells) > 0:
                    selected_cell = self.clicked_cells[0]
                    current_value = int(
                        self.ram[selected_cell]
                    )  # Convert to Python int
                    if selected_cell in self.recorded_ram_states:
                        self.recorded_ram_states[selected_cell].append(current_value)

                self._render()
                self.next_frame = False
                if i % 60 == 0:
                    print(f"Current FPS: {self.clock.get_fps()}")
            self._render()

            # Print FPS every 60 frames (optional)
            i += 1
        pygame.quit()

    def _get_action(self):
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                print(self.z_traces)
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:  # left mouse button pressed
                    x = self.current_mouse_pos[0] // UPSCALE_FACTOR
                    y = self.current_mouse_pos[1] // UPSCALE_FACTOR
                    if x < 160:
                        # Left Mouse Button Clicked on GAME
                        pass
                    else:
                        # Left Mouse Button Clicked on RAM
                        cell_idx = self._get_cell_under_mouse()
                        if cell_idx is not None:
                            # If the cell is already selected, unselect it and stop recording
                            if cell_idx in self.clicked_cells:
                                self.clicked_cells.remove(cell_idx)
                                if cell_idx in self.recorded_ram_states:

                                    recorded_states = self.recorded_ram_states[cell_idx]

                                    first_derivative = [
                                        recorded_states[i + 1] - recorded_states[i]
                                        for i in range(len(recorded_states) - 1)
                                    ]
                                    second_derivative = [
                                        first_derivative[i + 1] - first_derivative[i]
                                        for i in range(len(first_derivative) - 1)
                                    ]

                                    # Print results
                                    print(
                                        f"Recorded RAM states for cell {cell_idx}: {recorded_states}"

                                    )
                                    # Prepare CSV output
                                    os.makedirs("saved_states", exist_ok=True)
                                    csv_path = os.path.join("saved_states", f"cell_{cell_idx}{uuid.uuid4()}.csv")

                                    with open(csv_path, "w", newline="") as csvfile:
                                        writer = csv.writer(csvfile)
                                        for value in recorded_states:
                                            writer.writerow([value])  # Each value in its own row

                                    print(f"Saved RAM state values to {csv_path}")
                                    print(
                                        f"First derivative for cell {cell_idx}: {first_derivative}"
                                    )
                                    print(
                                        f"Second derivative for cell {cell_idx}: {second_derivative}"
                                    )

                                    # Create a figure with three subplots
                                    fig, (ax1, ax2, ax3) = plt.subplots(
                                        3, 1, figsize=(8, 8), sharex=True
                                    )

                                    # Plot recorded_states on the first subplot
                                    ax1.plot(
                                        recorded_states,
                                        marker="o",
                                        label="Recorded States",
                                    )
                                    ax1.set_title("Recorded State Over Time")
                                    ax1.set_ylabel("State Value")
                                    ax1.grid(True)
                                    ax1.legend()

                                    # Plot first_derivative on the second subplot
                                    ax2.plot(
                                        first_derivative,
                                        marker="o",
                                        color="orange",
                                        label="First Derivative",
                                    )
                                    ax2.set_title("First Derivative Over Time")
                                    ax2.set_ylabel("Derivative")
                                    ax2.grid(True)
                                    ax2.legend()

                                    # Plot second_derivative on the third subplot
                                    ax3.plot(
                                        second_derivative,
                                        marker="o",
                                        color="green",
                                        label="Second Derivative",
                                    )
                                    ax3.set_title("Second Derivative Over Time")
                                    ax3.set_xlabel("Time [Frames]")
                                    ax3.set_ylabel("Second Derivative")
                                    ax3.grid(True)
                                    ax3.legend()

                                    # Adjust layout so titles and labels don't overlap
                                    fig.tight_layout()

                                    # Show the combined figure
                                    plt.show()

                                    del self.recorded_ram_states[
                                        cell_idx
                                    ]  # Stop recording
                            else:
                                # If no cell is selected, start recording for the new cell
                                if (
                                    len(self.clicked_cells) == 0
                                ):  # Ensure no other cell is selected
                                    self.clicked_cells.append(cell_idx)
                                    self.recorded_ram_states[cell_idx] = (
                                        []
                                    )  # Start recording

                # Hiding of RAM Cells
                elif event.button == 3:  # right mouse button pressed
                    to_hide = self._get_cell_under_mouse()
                    if to_hide in self.no_render:
                        self.no_render.remove(to_hide)
                    else:
                        self.no_render.append(to_hide)

                # Red Highlighting of RAM Cells
                elif event.button == 2:  # middle mouse button pressed
                    toggle = self._get_cell_under_mouse()
                    if toggle in self.red_render:
                        self.red_render.remove(toggle)
                    else:
                        self.red_render.append(toggle)

                # Increment/Decrement RAM Cell Value
                elif event.button == 4:  # mousewheel up
                    cell_idx = self._get_cell_under_mouse()
                    if cell_idx is not None:
                        self._increment_ram_value_at(cell_idx)
                elif event.button == 5:  # mousewheel down
                    cell_idx = self._get_cell_under_mouse()
                    if cell_idx is not None:
                        self._decrement_ram_value_at(cell_idx)

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_f:  # Frame by frame
                    self.frame_by_frame = not (self.frame_by_frame)
                    self.next_frame = False

                elif event.key == pygame.K_n:  # next
                    print("Next Frame")
                    self.next_frame = True

                elif event.key == pygame.K_b:  # 'B': Backwards
                    if self.frame_by_frame:
                        if len(self.saved_frames) > 0:
                            previous = self.saved_frames.pop()
                            for i, ram_v in enumerate(previous[0]):
                                self.env.set_ram(i, ram_v)
                            for i, value in enumerate(previous[0]):
                                self._render_ram_cell(i, value)
                            self.env._ale.restoreState(previous[1])  # restore state
                            self.current_frame = previous[2].copy()
                            self._render_atari()
                            pygame.display.flip()
                            pygame.event.pump()
                        else:
                            print(
                                "There are no prior frames saved to go back to. Save more using the flag --previous_frames"
                            )

                if event.key == pygame.K_r:  # 'R': reset
                    self.env.reset()

                if event.key == pygame.K_c:  # 'C': clone
                    if self.paused:
                        statepkl = self.env._ale.cloneState()
                        with open(f"state_{self.env.game_name}.pkl", "wb") as f:
                            pkl.dump((statepkl, self.env._objects), f)
                            print(f"State cloned in state_{self.env.game_name}.pkl.")

                elif event.key == pygame.K_ESCAPE and self.active_cell_idx is not None:
                    self._unselect_active_cell()

                elif [
                    x for x in self.keys2actions.keys() if event.key in x
                ]:  # (event.key,) in self.keys2actions.keys() or [x for x in self.keys2actions.keys() if event.key in x]:  # env action
                    self.current_keys_down.add(event.key)

                elif pygame.K_0 <= event.key <= pygame.K_9:  # enter digit
                    char = str(event.key - pygame.K_0)
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input += char

                elif pygame.K_KP1 <= event.key <= pygame.K_KP0:  # enter digit
                    char = str((event.key - pygame.K_KP1 + 1) % 10)
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input += char

                elif event.key == pygame.K_BACKSPACE:  # remove character
                    if self.active_cell_idx is not None:
                        self.current_active_cell_input = self.current_active_cell_input[
                            :-1
                        ]

                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    if self.active_cell_idx is not None:
                        if len(self.current_active_cell_input) > 0:
                            new_cell_value = int(self.current_active_cell_input)
                            if new_cell_value < 256:
                                self._set_ram_value_at(
                                    self.active_cell_idx, new_cell_value
                                )
                        self._unselect_active_cell()

            elif event.type == pygame.KEYUP:  # keyboard key released
                if [
                    x for x in self.keys2actions.keys() if event.key in x
                ]:  # (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

    def _render(self, frame=None):
        self.window.fill((0, 0, 0))  # clear the entire window
        self._render_atari(frame)
        self._render_ram()
        self._render_hover()
        pygame.display.flip()
        pygame.event.pump()

    def _render_atari(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))
        self.clock.tick(1000)

    def _render_ram(self):
        ale = self.env._ale
        ram = ale.getRAM()

        for i, value in enumerate(ram):
            self._render_ram_cell(i, value)

    def _get_ram_value_at(self, idx: int):
        ale = self.env.unwrapped.ale
        ram = ale.getRAM()
        return ram[idx]

    def _set_ram_value_at(self, idx: int, value: int):
        ale = self.env.unwrapped.ale
        ale.setRAM(idx, value)
        # self.current_frame = self.env.render()
        # self._render()

    def _set_ram(self, values):
        ale = self.env.unwrapped.ale
        for k, value in enumerate(values):
            ale.setRAM(k, value)

    def _increment_ram_value_at(self, idx: int):
        value = self._get_ram_value_at(idx)
        self._set_ram_value_at(idx, min(value + 1, 255))

    def _decrement_ram_value_at(self, idx: int):
        value = self._get_ram_value_at(idx)
        self._set_ram_value_at(idx, max(value - 1, 0))

    def _render_ram_cell(self, cell_idx, value):
        is_active = cell_idx == self.active_cell_idx
        is_candidate = cell_idx in self.candidate_cell_ids
        is_red = cell_idx in self.red_render
        has_delta = cell_idx in self.delta_render
        is_clicked = cell_idx in self.clicked_cells

        x, y, w, h = self._get_ram_cell_rect(cell_idx)

        # Render cell background
        if is_active:
            color = (70, 70, 30)
        elif is_red:
            color = (150, 20, 20)
        elif is_candidate:
            color = (15, 45, 100)
        elif has_delta:
            color = (255, 165, 0)
        elif is_clicked:
            color = (20, 150, 20)
        else:
            color = (20, 20, 20)

        if is_clicked and has_delta:
            color = (238, 130, 238)

        pygame.draw.rect(self.window, color, [x, y, w, h])
        if cell_idx in self.no_render:
            return

        # Render cell ID label
        if is_active:
            color = (150, 150, 30)
        elif is_candidate:
            color = (20, 60, 200)
        else:
            color = (100, 150, 150)
        text = self.ram_cell_id_font.render(str(cell_idx), True, color, None)
        text_rect = text.get_rect()
        text_rect.topleft = (x + 2, y + 2)
        self.window.blit(text, text_rect)

        # Render cell value label
        if is_active:
            value = self.current_active_cell_input
        if value is not None:
            if is_active:
                color = (255, 255, 50)
            elif is_candidate:
                color = (30, 90, 255)
            elif has_delta:
                color = (0, 0, 0)
            else:
                color = (200, 200, 200)
            text = self.ram_cell_value_font.render(str(value), True, color, None)
            text_rect = text.get_rect()
            text_rect.bottomright = (x + w - 2, y + h - 2)
            self.window.blit(text, text_rect)

        # Render additional delta label
        if has_delta:
            color = (255, 0, 0)
            delta_value = int(self.ram[cell_idx]) - int(self.past_ram[cell_idx])
            # print(cell_idx, self.ram[cell_idx], self.past_ram[cell_idx], delta_value)
            delta_text = f"Δ{delta_value}"
            text = self.ram_cell_value_font.render(delta_text, True, color, None)
            text_rect = text.get_rect()
            # render bottom left
            text_rect.bottomleft = (x + 2, y + h - 2)
            self.window.blit(text, text_rect)

    def _get_ram_cell_rect(self, idx: int):
        row = idx // RAM_N_COLS
        col = idx % RAM_N_COLS
        x = round(self.ram_grid_anchor_left + col * 120 * (UPSCALE_FACTOR / 4))
        y = round(self.ram_grid_anchor_top + row * 50 * (UPSCALE_FACTOR / 4))
        w = RAM_CELL_WIDTH
        h = RAM_CELL_HEIGHT
        return x, y, w, h

    def _render_hover(self):
        cell_idx = self._get_cell_under_mouse()
        if cell_idx is not None and cell_idx != self.active_cell_idx:
            x, y, w, h = self._get_ram_cell_rect(cell_idx)
            hover_surface = pygame.Surface((w, h))
            hover_surface.set_alpha(60)
            hover_surface.set_colorkey((0, 0, 0))
            pygame.draw.rect(hover_surface, (255, 255, 255), [0, 0, w, h])
            self.window.blit(hover_surface, (x, y))

    def _get_cell_under_mouse(self):
        x, y = self.current_mouse_pos
        if x > self.ram_grid_anchor_left and y > self.ram_grid_anchor_top:
            col = int((x - self.ram_grid_anchor_left) // (120 * (UPSCALE_FACTOR / 4)))
            row = int((y - self.ram_grid_anchor_top) // (50 * (UPSCALE_FACTOR / 4)))
            if col < RAM_N_COLS and row < 16:
                return row * RAM_N_COLS + col
        return None

    def _unselect_active_cell(self):
        self.active_cell_idx = None
        self.current_active_cell_input = ""

    def find_causative_ram(self, x, y):
        """
        Goes over the entire RAM, manipulating it to observe possible changes
        in the current observation. Prints the RAM entry positions that are responsible
        for changes at pixel x, y.
        """
        ale = self.env.unwrapped.ale

        ram = ale.getRAM().copy()
        self.env.step(0)
        original_pixel = ale.getScreenRGB()[y, x]
        self._set_ram(ram)  # restore original RAM
        state = self.env._clone_state()

        self.candidate_cell_ids = []
        for i in tqdm(range(len(ram))):
            self.active_cell_idx = i
            for altered_value in [
                5
            ]:  # range(255):  # adding values != 0 causes Atari to crash
                self.current_active_cell_input = str(altered_value)
                self.env._restore_state(state)
                ale.setRAM(i, altered_value)
                self.env.step(0)
                new_frame = ale.getScreenRGB()
                self._render()
                new_pixel = new_frame[y, x]
                self._set_ram(ram)  # restore original RAM
                if np.any(new_pixel != original_pixel):
                    self.candidate_cell_ids.append(i)
                    break

        self.env._restore_state(state)
        self._unselect_active_cell()
        self._render()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="OCAtari remgui.py Argument Setter")

    parser.add_argument(
        "-g", "--game", type=str, default="Seaquest", help="Game to be run"
    )
    parser.add_argument(
        "-hu", "--human", action="store_true", help="Let user play the game."
    )
    parser.add_argument(
        "-sf",
        "--switch_frame",
        type=int,
        default=0,
        help="Swicht_modfis are applied to the game after this frame-threshold",
    )
    parser.add_argument(
        "-p",
        "--picture",
        type=int,
        default=0,
        help="Takes a picture after the number of steps provided.",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        default="",
        help="Path to the cleanrl trained agent to be loaded.",
    )
    parser.add_argument(
        "-nr",
        "--no_render",
        type=int,
        default=[],
        help="Cells to not render.",
        nargs="+",
    )
    parser.add_argument(
        "-nra", "--no_render_all", action="store_true", help="Not rendering any cell."
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="ram",
        choices=["ram", "vision"],
        help="Extraction mode.",
    )
    parser.add_argument(
        "-ls",
        "--load_state",
        type=str,
        default=None,
        help="Path to the state to be loaded.",
    )

    args = parser.parse_args()

    if args.no_render_all:
        args.no_render = list(range(128))

    renderer = Renderer(args.game, args.mode, args.no_render)

    if args.load_state:
        with open(args.load_state, "rb") as f:
            state, objects = pkl.load(f)
            renderer.env._ale.restoreState(state)
            renderer.env._objects = objects
            print(f"State loaded from {args.load_state}")

    def exit_handler():
        if renderer.no_render:
            print("\nno_render list: ")
            for i in sorted(renderer.no_render):
                print(i, end=" ")
            print("")

    atexit.register(exit_handler)

    renderer.run()
