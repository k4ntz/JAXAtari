import base64
import json
import pickle
import pygame
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import importlib
import inspect
import os
import re
import sys
import warnings

from jax.tree_util import tree_flatten, tree_unflatten
from typing import Type, Tuple, Dict, Any, List, Callable
from dataclasses import fields, is_dataclass

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


# ---------------------------------------------------------------------------
# Env state JSON (``scripts/play.py`` and other tooling)
# ---------------------------------------------------------------------------
#
# Typical workflow: save with ``save_env_state_json``, edit the file by hand
# (especially ``state_tree``: scalars, small arrays, nested ``{TypeName: {...}}``
# fields), then load by **merging** into a fresh ``reset()`` state — see
# ``reset_or_load_state`` or ``merge_env_state_from_json_dict``. That keeps
# runtime PyTree types correct while letting you override values from JSON.
#
# The editable **source of truth** is ``state_tree``. Flat ``leaves`` + pickled
# ``treedef`` are optional (``include_flat_leaves=True``): only needed for cold
# full restore without an env. Saving both alongside ``state_tree`` would
# duplicate the same state; after manual edits they could **disagree**, while
# merge always prefers ``state_tree`` when present — so the default save omits
# flat leaves to avoid that contradiction.
#
# Call sites should use the helpers in this section instead of raw
# ``open`` + ``json.load`` / ``json.dump`` on state files, so format checks and
# merge semantics stay in one place.
#
# Public surface (intended for imports):
#   * ``save_env_state_json``, ``env_state_to_json_dict`` — serialize to disk / dict
#   * ``read_env_state_json_file`` — parse a state JSON file to ``dict``
#   * ``merge_env_state_from_json_dict``, ``merge_env_state_from_json_path`` —
#     overlay saved data onto a reset state
#   * ``load_env_state_json``, ``env_state_from_json_dict`` — full rebuild from
#     flat ``leaves`` (no env reset); uncommon
#   * ``reset_or_load_state`` — reset + optional merge from path (used by play)
#   * ``warn_if_state_meta_mismatch`` — optional game/mods sanity check
#
# ---------------------------------------------------------------------------

STATE_JSON_FORMAT = "jaxatari_play_state_v1"

_ENV_STATE_JSON_TOP_KEYS = frozenset(
    {"format", "game", "mods", "treedef_b64", "leaves", "state_tree"}
)


def _numpy_scalar_to_python(arr: np.ndarray):
    """0-d numpy / JAX scalar -> plain Python bool / int / float."""
    v = arr.item()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _leaf_to_json_readable(x) -> Any:
    """Human-friendly leaf: scalars and small arrays inline; large arrays stay verbose."""
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        return _numpy_scalar_to_python(arr)
    if arr.size == 0:
        return []
    if arr.size <= 4096 and arr.ndim <= 3:
        return arr.tolist()
    return _leaf_to_json(x)


def _leaf_to_json(x):
    """Serialize a single PyTree leaf to a JSON-friendly object."""
    if x is None:
        return {"kind": "none"}
    arr = np.asarray(x)
    return {
        "kind": "ndarray",
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "data": arr.tolist(),
    }


def _leaf_from_json(d):
    """Restore a leaf from _leaf_to_json output."""
    if d.get("kind") == "none":
        return None
    if d.get("kind") != "ndarray":
        raise ValueError(f"Unknown leaf kind in state JSON: {d!r}")
    arr = np.array(d["data"], dtype=np.dtype(d["dtype"]))
    arr = arr.reshape(d["shape"])
    return jnp.array(arr)


def _leaf_shape_dtype_match(base, candidate) -> bool:
    """True if candidate can replace base in a PyTree (same shape and dtype)."""
    if base is None and candidate is None:
        return True
    if base is None or candidate is None:
        return False
    a = np.asarray(base)
    b = np.asarray(candidate)
    return a.shape == b.shape and a.dtype == b.dtype


def _pytree_to_readable_dict(obj: Any) -> Any:
    """Nested JSON-friendly view: NamedTuple/dataclass as {TypeName: {field: ...}}; arrays inlined when small."""
    if obj is None:
        return None
    if type(obj) is bool:
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, np.number):
        return _numpy_scalar_to_python(np.asarray(obj))
    if isinstance(obj, (jax.Array, np.ndarray)):
        return _leaf_to_json_readable(obj)
    if isinstance(obj, dict):
        return {str(k): _pytree_to_readable_dict(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            type(obj).__name__: {
                f.name: _pytree_to_readable_dict(getattr(obj, f.name))
                for f in fields(obj)
            }
        }
    if hasattr(obj, "_fields"):
        return {
            type(obj).__name__: {
                f: _pytree_to_readable_dict(getattr(obj, f)) for f in obj._fields
            }
        }
    if isinstance(obj, (list, tuple)):
        return [_pytree_to_readable_dict(x) for x in obj]
    warnings.warn(
        f"env state JSON: unsupported container type {type(obj)!r}; using repr.",
        UserWarning,
        stacklevel=3,
    )
    return repr(obj)


def _leaf_from_readable(node: Any, base):
    """Build a JAX leaf from an inline JSON value (number, list, or ndarray dict)."""
    if base is None:
        if node is None or (isinstance(node, dict) and node.get("kind") == "none"):
            return None
        warnings.warn(
            "merge_env_state: base leaf is None but JSON has a value; keeping None.",
            UserWarning,
            stacklevel=3,
        )
        return None
    base_np = np.asarray(base)
    if isinstance(node, dict) and node.get("kind") in ("ndarray", "none"):
        return _leaf_from_json(node)
    if base_np.ndim == 0:
        if type(node) is bool or isinstance(node, (int, float, np.number)):
            out = jnp.asarray(node, dtype=base_np.dtype)
            return out.reshape(())
        if node is None:
            return base
    arr = np.asarray(node, dtype=base_np.dtype)
    arr = arr.reshape(base_np.shape)
    return jnp.array(arr)


def _merge_state_tree_into_base(base, node: Any):
    """Overlay values from a nested ``state_tree`` onto ``base`` (same PyTree shape)."""
    if isinstance(base, (jax.Array, np.ndarray)) or base is None:
        try:
            cand = _leaf_from_readable(node, base)
        except Exception as e:
            warnings.warn(
                f"merge_env_state (state_tree): leaf ({e}); keeping reset value.",
                UserWarning,
                stacklevel=3,
            )
            return base
        if _leaf_shape_dtype_match(base, cand):
            return cand
        b_repr = (
            f"{np.asarray(base).shape}/{np.asarray(base).dtype}"
            if base is not None
            else "None"
        )
        c_repr = (
            f"{np.asarray(cand).shape}/{np.asarray(cand).dtype}"
            if cand is not None
            else "None"
        )
        warnings.warn(
            f"merge_env_state (state_tree): leaf shape/dtype mismatch "
            f"(base {b_repr}, saved {c_repr}); keeping reset value.",
            UserWarning,
            stacklevel=3,
        )
        return base

    if isinstance(base, dict):
        if not isinstance(node, dict):
            warnings.warn(
                "merge_env_state (state_tree): expected dict for dict base; keeping reset.",
                UserWarning,
                stacklevel=3,
            )
            return base
        out = dict(base)
        for k in node.keys():
            if k not in base:
                warnings.warn(
                    f"merge_env_state (state_tree): unknown key in JSON {k!r} (ignored).",
                    UserWarning,
                    stacklevel=3,
                )
        for k in base.keys():
            if k in node:
                out[k] = _merge_state_tree_into_base(base[k], node[k])
        return out

    if is_dataclass(base) and not isinstance(base, type):
        name = type(base).__name__
        if not isinstance(node, dict):
            warnings.warn(
                f"merge_env_state (state_tree): expected object for dataclass {name}; keeping reset.",
                UserWarning,
                stacklevel=3,
            )
            return base
        inner = node.get(name)
        if inner is None:
            if all(f.name in node for f in fields(base)):
                inner = {f.name: node[f.name] for f in fields(base)}
            else:
                warnings.warn(
                    f"merge_env_state (state_tree): missing wrapper {name!r} for dataclass; keeping reset.",
                    UserWarning,
                    stacklevel=3,
                )
                return base
        kwargs = {}
        for f in fields(base):
            if f.name not in inner:
                kwargs[f.name] = getattr(base, f.name)
            else:
                kwargs[f.name] = _merge_state_tree_into_base(
                    getattr(base, f.name), inner[f.name]
                )
        for k in inner.keys():
            if k not in {f.name for f in fields(base)}:
                warnings.warn(
                    f"merge_env_state (state_tree): unknown field {k!r} for dataclass {name!r} (ignored).",
                    UserWarning,
                    stacklevel=3,
                )
        return type(base)(**kwargs)

    if hasattr(base, "_fields"):
        name = type(base).__name__
        if not isinstance(node, dict):
            warnings.warn(
                f"merge_env_state (state_tree): expected object for NamedTuple {name}; keeping reset.",
                UserWarning,
                stacklevel=3,
            )
            return base
        inner = node.get(name)
        if inner is None:
            if all(f in node for f in base._fields):
                inner = {f: node[f] for f in base._fields}
            else:
                warnings.warn(
                    f"merge_env_state (state_tree): missing wrapper {name!r} for NamedTuple; keeping reset.",
                    UserWarning,
                    stacklevel=3,
                )
                return base
        kwargs = {}
        for f in base._fields:
            if f not in inner:
                kwargs[f] = getattr(base, f)
            else:
                kwargs[f] = _merge_state_tree_into_base(getattr(base, f), inner[f])
        for k in inner.keys():
            if k not in base._fields:
                warnings.warn(
                    f"merge_env_state (state_tree): unknown field {k!r} for {name!r} (ignored).",
                    UserWarning,
                    stacklevel=3,
                )
        return type(base)(**kwargs)

    if isinstance(base, (list, tuple)):
        if not isinstance(node, list) or len(node) != len(base):
            warnings.warn(
                "merge_env_state (state_tree): list/tuple length or type mismatch; keeping reset.",
                UserWarning,
                stacklevel=3,
            )
            return base
        merged = [_merge_state_tree_into_base(b, n) for b, n in zip(base, node)]
        return type(base)(merged) if isinstance(base, tuple) else merged

    warnings.warn(
        f"merge_env_state (state_tree): unsupported base type {type(base)!r}; keeping reset.",
        UserWarning,
        stacklevel=3,
    )
    return base


def env_state_to_json_dict(
    state,
    *,
    game: str,
    mods: list | None,
    include_flat_leaves: bool = False,
) -> dict:
    """Serialize env state.

    By default writes only ``state_tree`` (human-readable, single source of truth
    for manual edits). Set ``include_flat_leaves=True`` to also write pickled
    ``treedef_b64`` and flat ``leaves`` for ``env_state_from_json_dict`` / cold
    full restore — do not hand-edit those without updating the other copy.
    """
    leaves, treedef = tree_flatten(state)
    payload: Dict[str, Any] = {
        "format": STATE_JSON_FORMAT,
        "game": game,
        "mods": mods if mods is not None else [],
        "state_tree": _pytree_to_readable_dict(state),
    }
    if include_flat_leaves:
        payload["treedef_b64"] = base64.b64encode(
            pickle.dumps(treedef, protocol=4)
        ).decode("ascii")
        payload["leaves"] = [_leaf_to_json(x) for x in leaves]
    return payload


def env_state_from_json_dict(data: dict):
    """Rebuild env state from env_state_to_json_dict output (requires ``leaves`` + ``treedef_b64``)."""
    if data.get("format") != STATE_JSON_FORMAT:
        raise ValueError(
            f"Unsupported state JSON format (expected {STATE_JSON_FORMAT!r}, "
            f"got {data.get('format')!r})"
        )
    if "leaves" not in data or "treedef_b64" not in data:
        raise ValueError(
            "env_state_from_json_dict: missing 'leaves' and/or 'treedef_b64'; "
            "full restore needs save with include_flat_leaves=True, "
            "or use merge with reset() + state_tree only."
        )
    treedef = pickle.loads(base64.b64decode(data["treedef_b64"]))
    leaves = [_leaf_from_json(x) for x in data["leaves"]]
    return tree_unflatten(treedef, leaves)


def merge_env_state_from_json_dict(base_state, data: dict):
    """Overlay saved JSON onto ``base_state`` from a fresh reset.

    If ``state_tree`` is present, it is merged structurally (**only** this is used;
    any flat ``leaves`` in the same file are ignored). Otherwise ``treedef_b64`` +
    ``leaves`` are merged by leaf index (legacy files without ``state_tree``).

    Unknown top-level keys (besides format, game, mods, treedef_b64, leaves, state_tree)
    produce a warning.
    """
    if data.get("format") != STATE_JSON_FORMAT:
        raise ValueError(
            f"Unsupported state JSON format (expected {STATE_JSON_FORMAT!r}, "
            f"got {data.get('format')!r})"
        )
    extra_top = set(data.keys()) - _ENV_STATE_JSON_TOP_KEYS
    for k in sorted(extra_top):
        warnings.warn(
            f"merge_env_state: unknown top-level key in JSON {k!r} (ignored).",
            UserWarning,
            stacklevel=2,
        )

    if data.get("state_tree") is not None:
        return _merge_state_tree_into_base(base_state, data["state_tree"])

    base_leaves, base_treedef = tree_flatten(base_state)
    if "treedef_b64" not in data or "leaves" not in data:
        raise ValueError(
            "merge_env_state: JSON missing 'treedef_b64' and 'leaves' (and no 'state_tree')."
        )

    saved_treedef = pickle.loads(base64.b64decode(data["treedef_b64"]))
    if saved_treedef != base_treedef:
        warnings.warn(
            "merge_env_state: saved treedef differs from current reset state; "
            "merging leaves by index only.",
            UserWarning,
            stacklevel=2,
        )

    saved_leaves = data["leaves"]
    n_base = len(base_leaves)
    n_saved = len(saved_leaves)
    if n_saved > n_base:
        warnings.warn(
            f"merge_env_state: JSON has {n_saved - n_base} extra leaf entr(y/ies) "
            f"beyond current state ({n_base} leaves); ignored.",
            UserWarning,
            stacklevel=2,
        )

    merged_leaves = []
    for i, base_leaf in enumerate(base_leaves):
        if i >= n_saved:
            merged_leaves.append(base_leaf)
            continue
        try:
            cand = _leaf_from_json(saved_leaves[i])
        except Exception as e:
            warnings.warn(
                f"merge_env_state: leaf {i}: could not read JSON ({e}); "
                "keeping reset value.",
                UserWarning,
                stacklevel=2,
            )
            merged_leaves.append(base_leaf)
            continue
        if _leaf_shape_dtype_match(base_leaf, cand):
            merged_leaves.append(cand)
        else:
            b_repr = (
                f"{np.asarray(base_leaf).shape}/{np.asarray(base_leaf).dtype}"
                if base_leaf is not None
                else "None"
            )
            c_repr = (
                f"{np.asarray(cand).shape}/{np.asarray(cand).dtype}"
                if cand is not None
                else "None"
            )
            warnings.warn(
                f"merge_env_state: leaf {i}: shape/dtype mismatch "
                f"(base {b_repr}, saved {c_repr}); keeping reset value.",
                UserWarning,
                stacklevel=2,
            )
            merged_leaves.append(base_leaf)

    return tree_unflatten(base_treedef, merged_leaves)


def save_env_state_json(
    path: str,
    state,
    *,
    game: str,
    mods: list | None,
    include_flat_leaves: bool = False,
) -> None:
    """Write env state to JSON. Default: ``state_tree`` only (see module note on duplication)."""
    payload = env_state_to_json_dict(
        state, game=game, mods=mods, include_flat_leaves=include_flat_leaves
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _strip_trailing_commas_json(text: str) -> str:
    """Allow hand-edited files with trailing commas before ``}`` / ``]`` (invalid in strict JSON)."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r",(\s*[\]}])", r"\1", text)
    return text


def read_env_state_json_file(path: str) -> dict:
    """Parse a JAXAtari env state JSON file. Use this instead of raw ``json.load``.

    Tries strict JSON first, then a relaxed pass (strips trailing commas), which
    fixes common hand-edit mistakes that would otherwise skip merge entirely.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_err:
        relaxed = _strip_trailing_commas_json(text)
        try:
            out = json.loads(relaxed)
        except json.JSONDecodeError:
            raise first_err
        warnings.warn(
            "read_env_state_json_file: parsed after stripping trailing commas "
            "(strict JSON forbids commas before } or ]).",
            UserWarning,
            stacklevel=2,
        )
        return out


def merge_env_state_from_json_path(base_state, path: str):
    """Convenience: ``read_env_state_json_file`` + ``merge_env_state_from_json_dict``."""
    return merge_env_state_from_json_dict(base_state, read_env_state_json_file(path))


def load_env_state_json(path: str):
    """Load env state from a JSON file written by save_env_state_json (full restore)."""
    data = read_env_state_json_file(path)
    return env_state_from_json_dict(data), data


def warn_if_state_meta_mismatch(meta: dict, game: str, mods: list | None) -> None:
    """Warn when saved JSON game/mods metadata does not match the current session."""
    if meta.get("game") != game:
        print(
            f"Warning: state JSON is for game {meta.get('game')!r}, "
            f"current is {game!r} — load may fail or behave oddly."
        )
    saved = list(meta.get("mods") or [])
    cur = list(mods or [])
    if saved != cur:
        print(
            f"Warning: state JSON mods {saved} differ from current {cur} — "
            "load may behave oddly."
        )


def reset_or_load_state(
    *,
    load_path: str | None,
    game: str,
    mods: list | None,
    master_key,
    reset_counter: int,
    jitted_reset,
    label: str = "reset",
):
    """Return (obs, state, reset_counter_out).

    Always runs a jitted ``reset()`` first for a valid state, then merges in any
    JSON file at ``load_path`` when present (via ``read_env_state_json_file``).
    ``obs`` is None when a merge was applied (caller should derive display from ``state``).
    """
    reset_key = jrandom.fold_in(master_key, reset_counter)
    obs, base_state = jitted_reset(reset_key)
    reset_counter_out = reset_counter + 1
    if not load_path:
        return obs, base_state, reset_counter_out
    try:
        data = read_env_state_json_file(load_path)
        merged = merge_env_state_from_json_dict(base_state, data)
        warn_if_state_meta_mismatch(data, game, mods)
        return None, merged, reset_counter_out
    except Exception as e:
        print(f"{label}: could not load/merge state ({e}), using reset-only state.")
        return obs, base_state, reset_counter_out