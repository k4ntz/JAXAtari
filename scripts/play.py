import argparse
import sys
import pygame

import jax
import jax.random as jrandom
import jax.numpy as jnp
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
        "--logic-fps",
        type=int,
        default=None,
        help="Logic update rate (steps per second). Defaults to 2 for Surround.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode.",
    )

    args = parser.parse_args()

    logic_fps = args.logic_fps or (2 if args.game.lower() == "surround" else args.fps)
    render_fps = args.fps
    step_ms = 1000 // logic_fps
    acc_ms = 0
    latest_action = JAXAtariAction.NOOP

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
    jitted_render = jax.jit(renderer.render)

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

    action_space = env.action_space()

    save_keys = {}
    running = True
    pause = False
    frame_by_frame = False
    next_frame_asked = False
    total_return = 0
    if args.replay:
        with open(args.replay, "rb") as f:
            save_data = np.load(f, allow_pickle=True).item()
            actions_array = save_data["actions"]
            seed = save_data["seed"]
            render_fps = save_data.get("frame_rate", render_fps)
            logic_fps = save_data.get("logic_fps", logic_fps)
            step_ms = 1000 // logic_fps
            key = jrandom.PRNGKey(seed)
            obs, state = jitted_reset(key)

        for action in actions_array:
            action = jax.numpy.array(action, dtype=jax.numpy.int32)
            if args.verbose:
                print(f"Action: {ACTION_NAMES[int(action)]} ({int(action)})")

            obs, state, reward, done, info = jitted_step(state, action)
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(logic_fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit(0)

        pygame.quit()
        sys.exit(0)

    # main game loop
    while running:
        dt = clock.tick(render_fps)
        acc_ms += dt

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = not pause
                elif event.key == pygame.K_r:
                    obs, state = jitted_reset(key)
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                elif event.key == pygame.K_n:
                    next_frame_asked = True

        if args.random:
            key, subkey = jax.random.split(key)
            latest_action = action_space.sample(subkey)
        else:
            latest_action = get_human_action(events)

        if pause or (frame_by_frame and not next_frame_asked):
            continue

        if acc_ms >= step_ms and (not frame_by_frame or next_frame_asked):
            acc_ms -= step_ms
            act = latest_action
            if args.game.lower() == "surround":
                joint_action = jnp.array([act, JAXAtariAction.NOOP], dtype=jnp.int32)
            else:
                joint_action = act
            obs, state, reward, done, info = jitted_step(state, joint_action)
            total_return += reward
            if next_frame_asked:
                next_frame_asked = False
            if args.record:
                save_keys[len(save_keys)] = act
            if done:
                print(f"Done. Total return {total_return}")
                total_return = 0
                obs, state = jitted_reset(key)
                latest_action = JAXAtariAction.NOOP
                acc_ms = 0

        if not execute_without_rendering:
            image = jitted_render(state)
            update_pygame(window, image, UPSCALE_FACTOR, 160, 210)

    if args.record:
        # Convert dictionary to array of actions
        save_data = {
            "actions": np.array(
                [action for action in save_keys.values()], dtype=np.int32
            ),
            "seed": args.seed,  # The random seed used
            "frame_rate": render_fps,  # The frame rate for consistent replay
            "logic_fps": logic_fps,
        }
        with open(args.record, "wb") as f:
            np.save(f, save_data)

    pygame.quit()


if __name__ == "__main__":
    main()
