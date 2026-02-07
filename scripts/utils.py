import pygame
import jax
import numpy as np
import importlib
import inspect
import os
import sys
import warnings

from typing import Type, Tuple, Dict, Any, List, Callable
from dataclasses import is_dataclass

from functools import partial

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
from jaxatari.wrappers import JaxatariWrapper
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import JaxAtariModController, JaxAtariModWrapper


def _warn_deprecated_obs_to_flat_array(env: JaxEnvironment) -> None:
    """Warn if legacy obs_to_flat_array is present on the environment."""
    if hasattr(env, "obs_to_flat_array") and callable(getattr(env, "obs_to_flat_array")):
        warnings.warn(
            "Environment exposes deprecated obs_to_flat_array(). "
            "Observations should now be flax.struct.dataclasses using ObjectObservation "
            "for objects or plain arrays for observations like lives, score, etc. "
            "Depending on legacy obs_to_flat_array might lead to unforseen issues with wrappers.",
            DeprecationWarning,
            stacklevel=2,
        )

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
    renderer_class = None
    # Find the class that inherits from JaxEnvironment
    for name, obj in inspect.getmembers(game_module):
        if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
            print(f"Found game environment: {name}")
            game = obj()  # Instantiate and return

        if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
            print(f"Found renderer: {name}")
            renderer_class = obj

    if game is None:
        raise ImportError(f"No class found in {game_file_path} that inherits from JaxEnvironment")

    # Instantiate renderer with constants from the game environment
    if renderer_class is not None:
        try:
            consts = game.consts if hasattr(game, 'consts') else None
            renderer = renderer_class(consts=consts)
        except Exception as e:
            print(f"Warning: Could not instantiate renderer with constants: {e}")
            # Fallback: try without constants (renderer will use defaults)
            renderer = renderer_class()

    _warn_deprecated_obs_to_flat_array(game)
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
                    err_msg = f"Mod '{mod_key}' not recognized. Available mods: \n"
                    err_msg += "".join([f" - {k}\n" for k in registry.keys()])
                    raise ValueError(err_msg)
                plugin_class = registry[mod_key]
                if hasattr(plugin_class, "constants_overrides"):
                    const_overrides.update(plugin_class.constants_overrides)

            if const_overrides:
                # Recreate env with modded constants
                base_consts = env.consts
                
                # 1. Modern (.replace)
                if hasattr(base_consts, 'replace'):
                    modded_consts = base_consts.replace(**const_overrides)
                
                # 2. Legacy (_replace)
                elif hasattr(base_consts, '_replace'):
                    warnings.warn(
                        f"Unregistered Game '{game_name}': Using legacy '_replace()' for constants. "
                        "Please migrate to 'flax.struct.PyTreeNode'.",
                        UserWarning
                    )
                    valid_fields = base_consts._fields
                    field_overrides = {k: v for k, v in const_overrides.items() if k in valid_fields}
                    modded_consts = base_consts._replace(**field_overrides)
                    
                    # Legacy attribute injection
                    remaining = {k: v for k, v in const_overrides.items() if k not in valid_fields}
                    if remaining:
                        for k, v in remaining.items():
                            setattr(type(base_consts), k, v)
                else:
                     raise TypeError(
                        f"Constants class {type(base_consts).__name__} must support .replace() or _replace()."
                    )
                
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



def print_observation_tree(observation, title="Observation", indent=0):
    """
    Recursively prints a JAXAtari observation tree with nice formatting.
    """
    prefix = "  " * indent
    
    # 1. Handle ObjectObservation (Special Pretty Print)
    if isinstance(observation, ObjectObservation):
        print(f"{prefix}{title}:")
        repr_lines = str(observation).split('\n')
        for line in repr_lines:
            print(f"{prefix}  {line}")
        return

    # 2. Handle JAX/Numpy Arrays (Prioritize this over generic objects!)
    if hasattr(observation, 'shape'):
        try:
            # Handle Scalar Arrays (0-dim or size 1)
            if observation.size == 1:
                # Try to convert to python scalar for clean output
                val = observation.item()
                if isinstance(val, (int, float)):
                     print(f"{prefix}{title}: {val}")
                else:
                     print(f"{prefix}{title}: {val} ({observation.dtype})")
            else:
                # Handle Vector/Grid Arrays
                shape_str = str(observation.shape)
                print(f"{prefix}{title}: Array {shape_str} {observation.dtype}")
                
                # Print small arrays fully for debugging
                if observation.size <= 16:
                    flat_vals = observation.flatten().tolist()
                    # Format list nicely
                    vals_str = ", ".join([str(v) for v in flat_vals])
                    print(f"{prefix}  Values: [{vals_str}]")
        except Exception:
            # Fallback for Tracers/JIT where .item() fails
            print(f"{prefix}{title}: {observation}")
        return

    # 3. Handle Dictionaries
    if isinstance(observation, dict):
        print(f"{prefix}{title}:")
        for key, value in observation.items():
            print_observation_tree(value, title=key, indent=indent + 1)
        return

    # 4. Handle Flax PyTreeNodes / NamedTuples / Generic Classes
    # (Moved below Array check to prevent JAX arrays getting caught here)
    if hasattr(observation, '__dict__') or hasattr(observation, '_fields'):
        name = type(observation).__name__
        print(f"{prefix}{title} ({name}):")
        
        # Get field names
        fields = getattr(observation, '_fields', None)
        if fields is None:
            fields = observation.__dict__.keys()
            
        for field in fields:
            # Skip internal attributes
            if field.startswith('_'): continue
            
            val = getattr(observation, field)
            print_observation_tree(val, title=field, indent=indent + 1)
        return

    # 5. Fallback (Basic Types)
    print(f"{prefix}{title}: {observation}")