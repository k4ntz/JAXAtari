import pygame
import jax
import numpy as np
import importlib
import inspect
import os
import sys

from typing import Type, Tuple, Dict, Any, List, Callable

from functools import partial

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.wrappers import JaxatariWrapper
from jaxatari.renderers import JAXGameRenderer

def update_pygame(pygame_screen, raster, SCALING_FACTOR=3, WIDTH=400, HEIGHT=300):
    """Updates the Pygame display with the rendered raster.

    Args:
        pygame_screen: The Pygame screen surface.
        raster: JAX array of shape (Height, Width, 3/4) containing the image data.
        SCALING_FACTOR: Factor to scale the raster for display.
        WIDTH: Expected width of the input raster (used for scaling calculation).
        HEIGHT: Expected height of the input raster (used for scaling calculation).
    """
    pygame_screen.fill((0, 0, 0))

    # Convert JAX array (H, W, C) to NumPy (H, W, C)
    raster_np = np.array(raster)
    raster_np = raster_np.astype(np.uint8)

    if raster_np.ndim == 3 and raster_np.shape[2] == 1:
        raster_np = np.repeat(raster_np, 3, axis=2)

    # Pygame surface needs (W, H). make_surface expects (W, H, C) correctly.
    # Transpose from (H, W, C) to (W, H, C) for pygame
    frame_surface = pygame.surfarray.make_surface(raster_np.transpose(1, 0, 2))

    # Pygame scale expects target (width, height)
    # Note: raster_np is (H, W, C), so shape[1] is width and shape[0] is height
    target_width_px = int(raster_np.shape[1] * SCALING_FACTOR)
    target_height_px = int(raster_np.shape[0] * SCALING_FACTOR)


    frame_surface_scaled = pygame.transform.scale(
        frame_surface, (target_width_px, target_height_px)
    )

    pygame_screen.blit(frame_surface_scaled, (0, 0))
    pygame.display.flip()


def get_human_action() -> jax.numpy.ndarray: # Or chex.Array if you use chex
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

    action_to_take: int # Explicitly declare the type for clarity

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



def load_game_environment(game: str) -> Tuple[JaxEnvironment, JAXGameRenderer]:
    """
    Dynamically loads a game environment and the renderer from a .py file.
    It looks for a class that inherits from JaxEnvironment.
    """
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    game_file_path = os.path.join(project_root, "src", "jaxatari", "games", f"jax_{game.lower()}.py")
    
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    game_file_path = os.path.join(project_root, "src", "jaxatari", "games", f"jax_{game.lower()}.py")
    
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

        if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
            print(f"Found renderer: {name}")
            renderer = obj()

    if game is None:
        raise ImportError(f"No class found in {game_file_path} that inherits from JaxEnvironment")

    return game, renderer


def load_game_mods(game_name: str, mods_config: List[str]) -> Callable:
    """
    Dynamically loads and configures the single mod class (KangarooEnvMod) 
    using a list of mod strings.

    Args:
        game_name: The name of the game (e.g., 'kangaroo').
        mods_config: A list of mod strings (e.g., ['no_monkey', 'no_falling_coconut']).

    Returns:
        A callable function that takes the base environment and returns the 
        wrapped environment instance configured with the specified mods.

    Raises:
        ImportError: If the mod module or class cannot be found.
    """
    try:
        # 1. Dynamically import the game's mod file (e.g., 'jaxatari.games.mods.kangaroo_mods')
        module_path = f"jaxatari.games.mods.{game_name.lower()}_mods"
        mod_module = importlib.import_module(module_path)

        # 2. Assume the single mod class is named 'KangarooEnvMod'
        mod_class = getattr(mod_module, f"{game_name}EnvMod")

        # 3. Return a partial function to instantiate the class with the mods_config list.
        # This callable will be used in the main script: `env = mod_fn(env)`
        if mods_config:
            print(f"Loading mod configuration for {game_name}: {mods_config}")
        
        return partial(mod_class, mods_config=mods_config)

    except ImportError as e:
        # Check if the mod module exists but the class name is wrong, or if module failed to load.
        raise ImportError(
            f"Could not load mod module for game '{game_name}' or class 'KangarooEnvMod'. "
            f"Ensure the file {module_path}.py exists and contains the class. Details: {e}"
        )
    except AttributeError:
        # This handles the case where the module loaded but the class is missing.
        raise ImportError(
            f"Mod class 'KangarooEnvMod' not found in module {module_path}. "
            f"Please ensure the class name is correct."
        )

    except ImportError as e:
        raise ImportError(
            f"Could not load mod module for game '{game_name}' or mod '{mod_name}'. "
            f"Details: {e}"
        )


