import argparse
import importlib.util
import inspect
import os
import sys
import pygame
from typing import Tuple

import jax
import jax.random as jrandom
import numpy as np

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
import jaxatari.rendering.atraJaxis as aj

UPSCALE_FACTOR = 4


def get_human_action() -> jax.numpy.ndarray:  # Or chex.Array if you use chex
    """
    Get human action from keyboard with support for diagonal movement and combined fire,
    using Action constants.
    Returns a JAX array containing a single integer action.
    """
    # Important: Process Pygame events to allow window to close, etc.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit("Pygame window closed by user.")
        # You could handle other events here if needed (e.g., KEYDOWN for one-shot actions)

    keys = pygame.key.get_pressed()

    # Consolidate key checks
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    action_to_take: int  # Explicitly declare the type for clarity

    # The order of these checks is crucial for prioritizing actions
    # (e.g., UPRIGHTFIRE before UPFIRE or UPRIGHT)

    # Diagonal movements with fire (3 keys)
    if up and right and fire:
        action_to_take = Action.UPRIGHTFIRE
    elif up and left and fire:
        action_to_take = Action.UPLEFTFIRE
    elif down and right and fire:
        action_to_take = Action.DOWNRIGHTFIRE
    elif down and left and fire:
        action_to_take = Action.DOWNLEFTFIRE

    # Cardinal directions with fire (2 keys)
    elif up and fire:
        action_to_take = Action.UPFIRE
    elif down and fire:
        action_to_take = Action.DOWNFIRE
    elif left and fire:
        action_to_take = Action.LEFTFIRE
    elif right and fire:
        action_to_take = Action.RIGHTFIRE

    # Diagonal movements (2 keys)
    elif up and right:
        action_to_take = Action.UPRIGHT
    elif up and left:
        action_to_take = Action.UPLEFT
    elif down and right:
        action_to_take = Action.DOWNRIGHT
    elif down and left:
        action_to_take = Action.DOWNLEFT

    # Cardinal directions (1 key for movement)
    elif up:
        action_to_take = Action.UP
    elif down:
        action_to_take = Action.DOWN
    elif left:
        action_to_take = Action.LEFT
    elif right:
        action_to_take = Action.RIGHT
    # Fire alone (1 key)
    elif fire:
        action_to_take = Action.FIRE
    # No relevant keys pressed
    else:
        action_to_take = Action.NOOP

    return jax.numpy.array(action_to_take, dtype=jax.numpy.int32)


def load_game_environment(game_file_path: str) -> Tuple[JaxEnvironment, AtraJaxisRenderer]:
    """
    Dynamically loads a game environment and the renderer from a .py file.
    It looks for a class that inherits from JaxEnvironment.
    """
    if not os.path.exists(game_file_path):
        raise FileNotFoundError(f"Game file not found: {game_file_path}")

    module_name = os.path.splitext(os.path.basename(game_file_path))[0]

    # Add the directory of the game file to sys.path to handle relative imports within the game file
    game_dir = os.path.dirname(os.path.abspath(game_file_path))
    if game_dir not in sys.path:
        sys.path.insert(0, game_dir)

    spec = importlib.util.spec_from_file_location(module_name, game_file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {game_file_path}")

    game_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(game_module)
    except Exception as e:
        if game_dir in sys.path and sys.path[0] == game_dir:  # Clean up sys.path if we added to it
            sys.path.pop(0)
        raise ImportError(f"Could not execute module {module_name}: {e}")

    if game_dir in sys.path and sys.path[0] == game_dir:  # Clean up sys.path if we added to it
        sys.path.pop(0)

    game = None
    renderer = None
    # Find the class that inherits from JaxEnvironment
    for name, obj in inspect.getmembers(game_module):
        if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
            print(f"Found game environment: {name}")
            game = obj()  # Instantiate and return

        if inspect.isclass(obj) and issubclass(obj, AtraJaxisRenderer) and obj is not AtraJaxisRenderer:
            print(f"Found renderer: {name}")
            renderer = obj()

    if game is None:
        raise ImportError(f"No class found in {game_file_path} that inherits from JaxEnvironment")

    return game, renderer


def main():
    parser = argparse.ArgumentParser(description="Play a JAXAtari game, record your actions or replay them.")
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="Path to the Python file containing the game environment class (e.g., ./games/JaxFreeway.py).",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--play",
        action="store_true",
        help="Play the game manually.",
    )
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

    args = parser.parse_args()

    # set the frame rate
    frame_rate = 30

    execute_without_rendering = False
    # Load the game environment
    try:
        env, renderer = load_game_environment(args.game)

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
    # TODO: for now, we do not test jitted rendering

    # initialize the environment
    obs, state = jitted_reset(key)

    # setup pygame if we are rendering
    if not execute_without_rendering:
        pygame.init()
        pygame.display.set_caption("JAXAtari Game")
        env_render_shape = renderer.render(state).shape[:2]
        window = pygame.display.set_mode((env_render_shape[0] * UPSCALE_FACTOR, env_render_shape[1] * UPSCALE_FACTOR))
        clock = pygame.time.Clock()

    # get the action space of the current game (i.e. which actions are available)
    action_space = env.get_action_space()

    save_keys = {}
    playing = True

    if args.replay:
        with open(args.replay, "rb") as f:
            # Load the saved data
            save_data = np.load(f, allow_pickle=True).item()

            # Extract saved data
            actions_array = save_data['actions']
            seed = save_data['seed']
            loaded_frame_rate = save_data['frame_rate']

            # Reset environment with the saved seed
            key = jrandom.PRNGKey(seed)
            obs, state = jitted_reset(key)

            # Use the saved frame rate
            clock.tick(loaded_frame_rate)

        # loop over all the actions and play the game
        for action in actions_array:
            # Convert numpy action to JAX array
            action = jax.numpy.array(action, dtype=jax.numpy.int32)
            obs, state, reward, done, info = jitted_step(state, action)
            image = renderer.render(state)
            aj.update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
            clock.tick(frame_rate)

            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit(0)

        pygame.quit()
        sys.exit(0)

    # main game loop
    while playing:
        action = Action.NOOP
        if args.play or args.record:
            # get the pressed keys
            action = get_human_action()

            # Check if the action is valid (otherwise send NOOP)
            if action not in action_space:
                action = Action.NOOP

            # Save the action to the save_keys dictionary
            if args.record:
                # Save the action to the save_keys dictionary
                save_keys[len(save_keys)] = action

        elif args.random:
            # sample an action from the action space array
            action = jax.random.choice(key, action_space)
            key, subkey = jax.random.split(key)

        else:
            print("Invalid mode. Use --play, --record, or --replay.")
            sys.exit(1)

        # Step the environment
        obs, state, reward, done, info = jitted_step(state, action)

        # Render the environment
        if not execute_without_rendering:
            image = renderer.render(state)

            aj.update_pygame(window, image, UPSCALE_FACTOR, 160, 210)

            clock.tick(frame_rate)

        # check for quit event (esc)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                playing = False

    if args.record:
        # Convert dictionary to array of actions
        save_data = {
            'actions': np.array([action for action in save_keys.values()], dtype=np.int32),
            'seed': args.seed,  # The random seed used
            'frame_rate': frame_rate  # The frame rate for consistent replay
        }
        with open(args.record, "wb") as f:
            np.save(f, save_data)

    pygame.quit()


if __name__ == "__main__":
    main()