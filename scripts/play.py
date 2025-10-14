import argparse
import sys
import os
import pygame

import jax
import jax.random as jrandom
import numpy as np

from jaxatari.environment import JAXAtariAction
from utils import get_human_action, load_game_environment, load_game_mod, update_pygame

UPSCALE_FACTOR = 4

# Map action names to their integer values
ACTION_NAMES = {
    v: k
    for k, v in vars(JAXAtariAction).items()
    if not k.startswith("_") and isinstance(v, int)
}


def main():
    parser = argparse.ArgumentParser(
        description="Play a JAXAtari game, record your actions or replay them."
    )
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        required=True,
        help="Name of the game to play (e.g. 'freeway', 'pong'). The game must be in the src/jaxatari/games directory.",
    )
    parser.add_argument(
        "-m", "--mod",
        type=str,
        required=False,
        help="Name of the mod class.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        help="Record your actions and save them to the specified file (e.g. actions.npy).",
    )
    mode_group.add_argument(
        "--replay",
        type=str,
        metavar="FILE",
        help="Replay recorded actions from the specified file (e.g. actions.npy).",
    )
    mode_group.add_argument(
        "--random",
        action="store_true",
        help="Play the game with random actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for JAX PRNGKey and random action generation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frame rate for the game.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode.",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default=None,
        help="Directory to save frames (npy/png). Defaults to '<game>_screenshots' on first save.",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["npy", "png"],
        default="npy",
        help="Format to save frames when pressing 'O' (npy or png).",
    )

    args = parser.parse_args()

    execute_without_rendering = False
    # Load the game environment
    try:
        env, renderer = load_game_environment(args.game)
        if args.mod is not None:
            mod = load_game_mod(args.game, args.mod)
            env = mod(env)

        if renderer is None:
            execute_without_rendering = True
            print("No renderer found, running without rendering.")

    except (FileNotFoundError, ImportError) as e:
        print(f"Error loading game: {e}")
        sys.exit(1)

    # Initialize the environment
    key = jrandom.PRNGKey(args.seed)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    # initialize the environment
    obs, state = jitted_reset(key)

    # setup pygame if we are rendering
    if not execute_without_rendering:
        pygame.init()
        pygame.display.set_caption(f"JAXAtari Game {args.game}")
        env_render_shape = jitted_render(state).shape[:2]
        window = pygame.display.set_mode(
            (env_render_shape[1] * UPSCALE_FACTOR, env_render_shape[0] * UPSCALE_FACTOR)
        )
        clock = pygame.time.Clock()

    # Frame saving setup: defer directory creation until first save request
    screenshot_dir = args.screenshot_dir

    frame_save_counter = 1

    def save_frame(image):
        nonlocal frame_save_counter, screenshot_dir
        if image is None:
            print("Warning: Cannot save None frame.")
            return
        # Create directory on first save if not provided
        if screenshot_dir is None:
            screenshot_dir = f"{args.game}_screenshots"
        try:
            os.makedirs(screenshot_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create screenshot directory '{screenshot_dir}': {e}")
            return

        try:
            if args.save_format == "npy":
                filepath = os.path.join(screenshot_dir, f"frame_{frame_save_counter:05d}.npy")
                np.save(filepath, np.asarray(image))
            else:  # png
                # Build a pygame surface from the RGB numpy array and save it
                # image expected shape: (H, W, 3) uint8
                arr = np.asarray(image)
                if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                    print("Warning: Unexpected image shape for PNG save; expected HxWx3 or HxWx4.")
                h, w = arr.shape[:2]
                surface = pygame.Surface((w, h))
                try:
                    pygame.pixelcopy.array_to_surface(surface, arr.swapaxes(0, 1))
                except Exception:
                    # Fallback: ensure contiguous uint8 RGB
                    arr = np.ascontiguousarray(arr[:, :, :3], dtype=np.uint8)
                    pygame.pixelcopy.array_to_surface(surface, arr.swapaxes(0, 1))
                filepath = os.path.join(screenshot_dir, f"frame_{frame_save_counter:05d}.png")
                pygame.image.save(surface, filepath)
            frame_save_counter += 1
            print(f"Saved frame to {filepath}")
        except Exception as e:
            print(f"Error saving frame: {e}")

    action_space = env.action_space()

    save_keys = {}
    running = True
    pause = False
    frame_by_frame = False
    frame_rate = args.fps
    next_frame_asked = False
    total_return = 0
    if args.replay:
        with open(args.replay, "rb") as f:
            # Load the saved data
            save_data = np.load(f, allow_pickle=True).item()

            # Extract saved data
            actions_array = save_data["actions"]
            seed = save_data["seed"]
            loaded_frame_rate = save_data["frame_rate"]

            frame_rate = loaded_frame_rate

            # Reset environment with the saved seed
            key = jrandom.PRNGKey(seed)
            obs, state = jitted_reset(key)

        # loop over all the actions and play the game
        for action in actions_array:
            # Convert numpy action to JAX array
            action = jax.numpy.array(action, dtype=jax.numpy.int32)
            if args.verbose:
                print(f"Action: {ACTION_NAMES[int(action)]} ({int(action)})")

            obs, state, reward, done, info = jitted_step(state, action)
            image = jitted_render(state)
            if not execute_without_rendering:
                update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
                clock.tick(frame_rate)

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_o: # save current frame (s is taken by navigation)
                    save_frame(image)

        pygame.quit()
        sys.exit(0)

    # display the first frame (reset frame) -> purely for aesthetics
    image = jitted_render(state)
    if not execute_without_rendering:
        update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
        clock.tick(frame_rate)

    frame_counter = 0

    # main game loop
    while running:
        # check for external actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # pause
                    pause = not pause
                elif event.key == pygame.K_r:  # reset
                    obs, state = jitted_reset(key)
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                elif event.key == pygame.K_n:
                    next_frame_asked = True
                elif event.key == pygame.K_o:  # save current frame
                    try:
                        current_image = jitted_render(state)
                    except Exception:
                        current_image = None
                    save_frame(current_image)
        if pause or (frame_by_frame and not next_frame_asked):
            image = jitted_render(state)
            if not execute_without_rendering:
                update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
                clock.tick(frame_rate)
            continue
        if args.random:
            # sample an action from the action space array
            action = action_space.sample(key)
            key, subkey = jax.random.split(key)
        else:
            # get the pressed keys
            action = get_human_action()

            # Save the action to the save_keys dictionary
            if args.record:
                # Save the action to the save_keys dictionary
                save_keys[len(save_keys)] = action

        if not frame_by_frame or next_frame_asked:
            action = get_human_action()
            obs, state, reward, done, info = jitted_step(state, action)
            total_return += reward
            if next_frame_asked:
                next_frame_asked = False

        if done:
            print(f"Done. Total return {total_return}, Frames this episode: {frame_counter}")
            total_return = 0
            frame_counter = 0
            obs, state = jitted_reset(key)

        # Render the environment
        if not execute_without_rendering:
            image = jitted_render(state)

            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)

            clock.tick(frame_rate)

        frame_counter += 1

    if args.record:
        # Convert dictionary to array of actions
        save_data = {
            "actions": np.array(
                [action for action in save_keys.values()], dtype=np.int32
            ),
            "seed": args.seed,  # The random seed used
            "frame_rate": frame_rate,  # The frame rate for consistent replay
        }
        with open(args.record, "wb") as f:
            np.save(f, save_data)

    pygame.quit()


if __name__ == "__main__":
    main()
