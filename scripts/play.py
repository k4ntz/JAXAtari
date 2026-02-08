import argparse
import sys
import pygame

import jax
import jax.random as jrandom
import numpy as np

from jaxatari.environment import JAXAtariAction
from utils import get_human_action, update_pygame, load_game_environment, load_game_mods, print_observation_tree
from jaxatari.core import make as jaxatari_make

UPSCALE_FACTOR = 4


def _normalize_mods(mods):
    """Convert common CLI mods input into a list of mod name strings.

    Accepts:
      - None -> None
      - Space-separated (argparse nargs='+'): ['mod1', 'mod2']
      - Comma-separated: ['mod1,mod2'] or ['mod1, mod2'] -> ['mod1', 'mod2']
      - Single string (e.g. from config): 'mod1' or 'mod1,mod2' -> list
      - Mixed: ['mod1', 'mod2,mod3'] -> ['mod1', 'mod2', 'mod3']
    """
    if mods is None:
        return None
    if isinstance(mods, str):
        mods = [mods]
    result = []
    for item in mods:
        if not isinstance(item, str):
            item = str(item).strip()
        else:
            item = item.strip()
        if not item:
            continue
        # Split by comma so "mod1, mod2" and "mod1,mod2" both work
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result if result else None


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
        help="Mod name(s). Space-separated (e.g. -m ModA ModB) or comma-separated (e.g. -m ModA,ModB).",
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

    # Normalize mods so we accept space-separated, comma-separated, or mixed
    args.mods = _normalize_mods(args.mods)

    execute_without_rendering = False
    
    try:
        # 1. Try the registered path (core.make)
        env = jaxatari_make(
            game_name=args.game,
            mods=args.mods,
            allow_conflicts=args.allow_conflicts
        )
        print(f"Successfully loaded registered game: '{args.game}'")
    except (NotImplementedError, ImportError) as e:
        # 2. If not registered, try the dynamic path
        print(f"Game '{args.game}' not registered or import error ({e}). Trying dynamic load...")
        try:
            # 2a. Dynamically load the base game environment
            # We only need the 'game' object; it will have its own .renderer
            game_env, _ = load_game_environment(args.game)
            
            # 2b. Apply mods if requested
            if args.mods:
                print(f"Applying mods: {args.mods}")
                # Get the function that applies the full modding pipeline
                mod_applier = load_game_mods(
                    game_name=args.game,
                    mods_config=args.mods,
                    allow_conflicts=args.allow_conflicts
                )
                # Apply the mods to the base env
                env = mod_applier(game_env)
            else:
                # No mods, just use the dynamically loaded game
                env = game_env
            
            print(f"Successfully loaded unregistered game: '{args.game}'")
        except (FileNotFoundError, ImportError, ValueError, AttributeError) as e_dyn:
            # 3. If dynamic loading also fails, then we exit
            print(f"Error: Failed to load game '{args.game}' dynamically.")
            print(f"Details: {e_dyn}")
            sys.exit(1)
    
    except (FileNotFoundError, ValueError, AttributeError) as e_reg:
        # 4. Catch other errors from the registered path (e.g., mod conflict)
        print(f"Error loading registered game or mods: {e_reg}")
        sys.exit(1)

    if not hasattr(env, "renderer"):
        execute_without_rendering = True
        print("No renderer found, running without rendering.")

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

    def map_action_to_index(action_constant):
        """Convert Action constant to the specific index within the game's ACTION_SET."""
        if hasattr(env, 'ACTION_SET'):
            # Convert JAX array/constant to standard Python int for comparison
            action_set = np.array(env.ACTION_SET)
            action_int = int(action_constant)
            
            # Find where this constant lives in the current game's minimal set
            matches = np.where(action_set == action_int)[0]
            
            if len(matches) > 0:
                idx = int(matches[0])
                if args.verbose:
                    name = ACTION_NAMES.get(action_int, "UNKNOWN")
                    # Verification: "LEFT" (4) should map to Index 3 in Pong
                    print(f"[Action Debug] Input: {name} | Constant: {action_int} -> Env Index: {idx}")
                return jax.numpy.array(idx, dtype=jax.numpy.int32)
        
        # Fallback if no ACTION_SET is defined
        return jax.numpy.array(action_constant, dtype=jax.numpy.int32)

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
            if not execute_without_rendering:
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

        if not execute_without_rendering:
            pygame.quit()
        sys.exit(0)

    # display the first frame (reset frame) -> purely for aesthetics
    if not execute_without_rendering:
        image = jitted_render(state)
        update_pygame(window, image, UPSCALE_FACTOR, 160, 210)
        clock.tick(frame_rate)

    # main game loop
    while running:
        # check for external actions
        if not execute_without_rendering:
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
                        total_return = 0
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
            # get the pressed keys (returns Action constant) and map to action index
            action_constant = get_human_action()
            action = map_action_to_index(action_constant)
            # Save the action to the save_keys dictionary
            if args.record:
                # Save the action to the save_keys dictionary
                save_keys[len(save_keys)] = action

        if not frame_by_frame or next_frame_asked:
            obs, state, reward, done, info = jitted_step(state, action)
            total_return += reward
            if next_frame_asked:
                next_frame_asked = False
        else:
            # Need to get action to update event queue even if paused
            action_constant = get_human_action()
            action = map_action_to_index(action_constant)

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
        
        # Handle loop for no-rendering execution
        if execute_without_rendering:
            if done:
                running = False # Run for one episode if not rendering
            if pause or (frame_by_frame and not next_frame_asked):
                continue
            if frame_by_frame and next_frame_asked:
                next_frame_asked = False

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
    
    if not execute_without_rendering:
        pygame.quit()


if __name__ == "__main__":
    main()
