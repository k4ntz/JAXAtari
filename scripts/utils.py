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
from jaxatari.modification import JaxAtariModController, JaxAtariModWrapper

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


def _dynamic_load_from_path(file_path: str, class_name: str):
    """Dynamically loads a class from a specific .py file."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Add dir to path for relative imports (e.g., pong_mods importing pong_mod_plugins)
    mod_dir = os.path.dirname(os.path.abspath(file_path))
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
        
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {file_path}")
        
    game_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(game_module)
    finally:
        if sys.path and sys.path[0] == mod_dir:
            sys.path.pop(0) # Clean up path
    return getattr(game_module, class_name)

# --- REWRITTEN FUNCTION ---
def load_game_mods(game_name: str, mods_config: List[str], allow_conflicts: bool = False) -> Callable:
    """
    Dynamically loads the modding pipeline for an unregistered game.
    
    This function re-implements the logic from core using dynamic
    file paths instead of the MOD_MODULES registry.
    Returns:
        A callable function that applies the full two-stage
        (Controller + Wrapper) pipeline to an environment.
    """
    
    # This is the function that will be returned by load_game_mods
    def apply_mods(env: JaxEnvironment) -> JaxEnvironment:
        """This closure captures the mod config and applies the full pipeline."""
        
        try:
            # 1. Dynamically find the paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            controller_path = os.path.join(
                project_root, "src", "jaxatari", "games", "mods", f"{game_name.lower()}_mods.py"
            )
            
            # 2. Load the Controller Class (e.g., PongEnvMod)
            # We must follow the naming convention: "Pong" -> "PongEnvMod"
            controller_class_name = f"{game_name.capitalize()}EnvMod"
            ControllerClass = _dynamic_load_from_path(controller_path, controller_class_name)
            
            # 3. --- PRE-SCAN FOR CONSTANT OVERRIDES (mirror core.make) ---
            registry = ControllerClass.REGISTRY
            const_overrides: Dict[str, Any] = {}
            for mod_key in mods_config:
                if mod_key not in registry:
                    raise ValueError(f"Mod '{mod_key}' not recognized.")
                plugin_class = registry[mod_key]
                if hasattr(plugin_class, "constants_overrides"):
                    const_overrides.update(plugin_class.constants_overrides)

            if const_overrides:
                # Recreate env with modded constants
                base_consts = env.consts
                modded_consts = base_consts._replace(**const_overrides)
                env = env.__class__(consts=modded_consts)

            # 4. BUILD STAGE 1 (Internal Controller)
            # Rely on the controller subclass to pass its own REGISTRY via super().__init__
            modded_env = ControllerClass(
                env=env,
                mods_config=mods_config,
                allow_conflicts=allow_conflicts
            )
            
            # 5. BUILD STAGE 2 (Post-Step Wrapper)
            # The wrapper gets the registry from the controller
            final_env = JaxAtariModWrapper(
                env=modded_env,
                mods_config=mods_config,
                allow_conflicts=allow_conflicts
            )
            
            print(f"Successfully loaded {len(mods_config)} mods for unregistered game '{game_name}'.")
            return final_env
            
        except (ImportError, AttributeError, FileNotFoundError) as e:
            print(f"Error loading mods for unregistered game '{game_name}': {e}")
            raise e
    
    # Return the callable 'apply_mods' function
    return apply_mods