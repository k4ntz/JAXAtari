import argparse
import sys
import pygame

import jax
import jax.random as jrandom
import numpy as np

from jaxatari.environment import JAXAtariAction
from utils import get_human_action, update_pygame
from jaxatari.core import make as jaxatari_make

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
        "-m", "--mods",
        nargs='+',
        type=str,
        required=False,
        help="Name of the mods class.",
    )

    parser.add_argument(
        "--allow_conflicts",
        action="store_true",
        help="Allow loading conflicting mods (last mod in list takes priority).",
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
        default=30,
        help="Frame rate for the game.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode.",
    )

    args = parser.parse_args()

    execute_without_rendering = False
    # Load the game environment using the core.make() entry point
    try:
        env = jaxatari_make(
            game_name=args.game,
            mods_config=args.mods,
            allow_conflicts=args.allow_conflicts
        )

        if not hasattr(env, "renderer"):
            execute_without_rendering = True
            print("No renderer found, running without rendering.")

    except (FileNotFoundError, ImportError, ValueError, AttributeError, NotImplementedError) as e:
        print(f"Error loading game or mods: {e}")
        sys.exit(1)

    # Initialize the environment
    master_key = jrandom.PRNGKey(args.seed)
    reset_counter = 0
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env.render)

    # initialize the environment with the first reset key
    reset_key = jrandom.fold_in(master_key, reset_counter)
    obs, state = jitted_reset(reset_key)
    reset_counter += 1
    
    # For random actions, we need a separate key stream
    action_key = jrandom.fold_in(master_key, 1000000)  # Use a large offset to avoid collision

    # setup pygame if we are rendering
    if not execute_without_rendering:
        pygame.init()
        pygame.display.set_caption(f"JAXAtari Game {args.game}")
        env_render_shape = jitted_render(state).shape[:2]
        window = pygame.display.set_mode(
            (env_render_shape[1] * UPSCALE_FACTOR, env_render_shape[0] * UPSCALE_FACTOR)
        )
        clock = pygame.time.Clock()

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

            # Reset environment with the saved seed using the same approach
            master_key = jrandom.PRNGKey(seed)
            reset_key = jrandom.fold_in(master_key, 0)  # Use first reset key
            obs, state = jitted_reset(reset_key)

        # loop over all the actions and play the game
        for action in actions_array:
            # Convert numpy action to JAX array
            action = jax.numpy.array(action, dtype=jax.numpy.int32)
            if args.verbose:
                print(f"Action: {ACTION_NAMES[int(action)]} ({int(action)})")

            obs, state, reward, done, info = jitted_step(state, action)
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit(0)

        pygame.quit()
        sys.exit(0)

    # display the first frame (reset frame) -> purely for aesthetics
    image = jitted_render(state)
    update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
    clock.tick(frame_rate)

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
                    reset_key = jrandom.fold_in(master_key, reset_counter)
                    obs, state = jitted_reset(reset_key)
                    reset_counter += 1
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                elif event.key == pygame.K_n:
                    next_frame_asked = True
        if pause or (frame_by_frame and not next_frame_asked):
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)
            continue
        if args.random:
            # sample an action from the action space array
            action = action_space.sample(action_key)
            action_key, _ = jax.random.split(action_key)
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
            print(f"Done. Total return {total_return}")
            total_return = 0
            reset_key = jrandom.fold_in(master_key, reset_counter)
            obs, state = jitted_reset(reset_key)
            reset_counter += 1

        # Render the environment
        if not execute_without_rendering:
            image = jitted_render(state)

            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)

            clock.tick(frame_rate)

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
