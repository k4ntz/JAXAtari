import inspect
import importlib
import jax
import chex
from functools import partial
from collections import defaultdict
from abc import ABC, abstractmethod
from jaxatari.wrappers import JaxatariWrapper
from jaxatari.environment import JaxEnvironment


def _load_from_string(path: str):
    """Dynamically import an attribute from a module path string."""
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)

# --- 1. Plugin Base Classes ---
class JaxAtariInternalModPlugin(ABC):
    """
    Base class for an *internal* mod plugin.
    - Patches methods defined on the plugin.
    - Overrides constants and attributes.
    """
    # manually defined list of mods this mod may conflict with. Does not include functional conflicts (i.e. two mods modifying the same method) which are detected automatically.
    conflicts_with: list = []
    # For core.make to override constants at construction time
    constants_overrides: dict = {}
    # For overriding member attributes set in __init__
    attribute_overrides: dict = {}
    # Read by core.make to override assets via constants at construction time. This is a list of dicts, each with a 'name' key and a 'file' key.
    asset_overrides: dict = {}

class JaxAtariPostStepModPlugin(ABC):
    """
    Base class for a *post-step* mod plugin (runs after step).
    """
    # manually defined list of mods this mod may conflict with. Does not include functional conflicts (i.e. two mods modifying the same method) which are detected automatically.
    conflicts_with: list = []
    # For core.make to override constants at construction time
    constants_overrides: dict = {}
    # For overriding member attributes set in __init__
    attribute_overrides: dict = {}
    # Read by core.make to override assets via constants at construction time. This is a list of dicts, each with a 'name' key and a 'file' key.
    asset_overrides: dict = {}

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        Optional function called by the wrapper *after* the main step is complete.
        If implemented, this allows the plugin to modify state after each step.
        Access the environment via self._env (set by JaxAtariModWrapper).
        
        Args:
            prev_state: The state before the step was taken
            new_state: The state after the step was taken
            
        Returns:
            Modified state (or unchanged state)
        """
        # Default implementation: no-op, just return state as-is
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        """
        Optional function called by the wrapper *after* reset is complete.
        If implemented, this allows the plugin to modify the initial state.
        Access the environment via self._env (set by JaxAtariModWrapper).
        
        Args:
            obs: The observation returned by reset
            state: The initial state returned by reset
            
        Returns:
            Tuple of (obs, state) - can return modified versions
        """
        # Default implementation: no-op, just return as-is
        return obs, state

# --- 2. Shared Helper Function ---

def _check_gameplay_conflicts(
    active_plugins: dict, 
    allow_conflicts: bool
):
    """
    Checks for gameplay conflicts (using the 'conflicts_with' list)
    among a list of instantiated plugins.
    """
    active_keys = set(active_plugins.keys())
    all_conflicts = set()
    for plugin in active_plugins.values():
        all_conflicts.update(plugin.conflicts_with)
    
    conflicts_found = active_keys.intersection(all_conflicts)
    if conflicts_found:
        report = f"Two or more of the requested mods were marked as impacting the same gameplay element and conflict with each other: {conflicts_found}"
        if not allow_conflicts:
            raise ValueError(
                f"{report}\n(Pass allow_conflicts=True to ignore this warning.)"
            )
        else:
            print(f"WARNING: {report}. Conflicts ignored.")


def _build_modded_asset_config(base_consts, registry, expanded_mods_config):
    """Return a modded ASSET_CONFIG list based on plugin overrides, or None."""
    if not hasattr(base_consts, "ASSET_CONFIG"):
        return None

    all_asset_overrides = {}
    asset_conflicts = defaultdict(list)  # Track which mods override which assets
    
    for mod_key in expanded_mods_config:
        plugin_class = registry[mod_key]
        if hasattr(plugin_class, "asset_overrides"):
            for asset_name in plugin_class.asset_overrides:
                asset_conflicts[asset_name].append(mod_key)
            all_asset_overrides.update(plugin_class.asset_overrides)
    
    if not all_asset_overrides:
        return None

    original_assets_by_name = {}
    for asset in getattr(base_consts, "ASSET_CONFIG", ()):  # type: ignore[attr-defined]
        asset_dict = dict(asset)
        asset_name = asset_dict.get("name")
        if asset_name:
            original_assets_by_name[asset_name] = asset_dict

    modded_asset_map = original_assets_by_name.copy()

    for asset_name, override_value in all_asset_overrides.items():
        if override_value is None:
            modded_asset_map.pop(asset_name, None)
            continue

        if isinstance(override_value, str):
            replacement_name = override_value
            if replacement_name not in original_assets_by_name:
                raise ValueError(
                    "Asset override failed: mod wants to map "
                    f"'{asset_name}' to '{replacement_name}', but "
                    f"'{replacement_name}' does not exist in base assets."
                )
            new_config = dict(original_assets_by_name[replacement_name])
            new_config["name"] = asset_name
            modded_asset_map[asset_name] = new_config
            continue

        if isinstance(override_value, dict):
            override_dict = dict(override_value)
            if override_dict.get("name") != asset_name:
                raise ValueError(
                    f"Asset override for '{asset_name}' is invalid. "
                    "Override dictionaries must include a 'name' key "
                    f"set to '{asset_name}'."
                )
            modded_asset_map[asset_name] = override_dict
            continue

        raise TypeError(
            f"Asset override for '{asset_name}' has unsupported type "
            f"'{type(override_value).__name__}'."
        )

    return list(modded_asset_map.values()), asset_conflicts


def _collect_and_check_attribute_overrides(
    mods_config: list,
    registry: dict,
    env,
    allow_conflicts: bool = False
):
    """
    Helper function to collect attribute overrides from all plugins and check for conflicts.
    Returns a dict mapping mod_key to attribute_overrides dict, and reports conflicts.
    """
    all_attribute_overrides = {}
    attr_conflicts = defaultdict(list)
    
    # Collect attribute overrides from all plugins (both Internal and PostStep)
    for mod_key in mods_config:
        plugin_class = registry[mod_key]
        if hasattr(plugin_class, "attribute_overrides") and plugin_class.attribute_overrides:
            for attr_name in plugin_class.attribute_overrides:
                attr_conflicts[attr_name].append(mod_key)
            all_attribute_overrides[mod_key] = plugin_class.attribute_overrides
    
    # Check for attribute conflicts
    attr_conflicts_found = False
    attr_report_lines = []
    for attr_name, mods in attr_conflicts.items():
        if len(mods) > 1:
            attr_conflicts_found = True
            winner = mods[-1]
            attr_report_lines.append(
                f"  - The attribute '{attr_name}' is overridden by multiple mods: {mods}. "
                f"With allow_conflicts=True, the last mod in the list ('{winner}') would take priority."
            )
    
    # Report conflicts (but don't raise here - will be combined with other conflict types)
    return all_attribute_overrides, attr_conflicts_found, attr_report_lines


def _apply_attribute_overrides(env, attribute_overrides_dict: dict, mod_key: str):
    """
    Helper function to apply attribute overrides to an environment.
    
    Args:
        env: The environment to apply overrides to
        attribute_overrides_dict: Dictionary of attribute_name -> value
        mod_key: The mod key (for error messages)
    """
    for attr_name, value in attribute_overrides_dict.items():
        if not hasattr(env, attr_name):
            raise AttributeError(
                f"Mod '{mod_key}' tries to override attribute '{attr_name}', "
                f"but this attribute does not exist on the base environment."
            )
        setattr(env, attr_name, value)


class JaxAtariModController:
    """
    A generic Mod Controller that contains all the logic for:
    1. 'Internal' type plugin discovery and validation.
    2. Automated conflict detection and reporting.
    3. Applying 'internal' type patches (method replacement).
    This class is intended to be inherited by game-specific
    controllers (e.g., PongEnvMod, KangarooEnvMod).
    
    Conflict Detection:
    - Constants: Checked in pre_scan_for_overrides (all mods: Internal + PostStep)
    - Assets: Checked in pre_scan_for_overrides (all mods: Internal + PostStep)
    - Attributes: Checked in _collect_and_check_attribute_overrides (all mods: Internal + PostStep)
    - Methods: Checked in __init__ (Internal mods only, as PostStep don't patch methods)
    """
    @staticmethod
    def pre_scan_for_overrides(
        mods_config: list,
        registry: dict,
        base_consts,
        allow_conflicts: bool = False
    ) -> dict:
        """
        Scans all mods (both Internal and PostStep) for constant and asset overrides.
        Checks for conflicts across ALL mod types and reports them.
        """
        const_overrides = {}
        const_conflicts = defaultdict(list)  # Track which mods override which constants
        
        # Collect constants and detect conflicts
        for mod_key in mods_config:
            plugin_class = registry[mod_key]
            if hasattr(plugin_class, "constants_overrides"):
                for const_name in plugin_class.constants_overrides:
                    const_conflicts[const_name].append(mod_key)
                const_overrides.update(plugin_class.constants_overrides)

        # Check for constant conflicts
        constant_conflicts_found = False
        constant_report_lines = []
        for const_name, mods in const_conflicts.items():
            if len(mods) > 1:
                constant_conflicts_found = True
                winner = mods[-1]
                constant_report_lines.append(
                    f"  - The constant '{const_name}' is overridden by multiple mods: {mods}. "
                    f"With allow_conflicts=True, the last mod in the list ('{winner}') would take priority."
                )
        
        # Build asset config and get asset conflicts
        asset_result = _build_modded_asset_config(
            base_consts, registry, mods_config
        )
        
        if asset_result is not None:
            modded_asset_config, asset_conflicts = asset_result
            const_overrides["ASSET_CONFIG"] = modded_asset_config
            
            # Check for asset conflicts
            asset_conflicts_found = False
            asset_report_lines = []
            for asset_name, mods in asset_conflicts.items():
                if len(mods) > 1:
                    asset_conflicts_found = True
                    winner = mods[-1]
                    asset_report_lines.append(
                        f"  - The asset '{asset_name}' is overridden by multiple mods: {mods}. "
                        f"With allow_conflicts=True, the last mod in the list ('{winner}') would take priority."
                    )
        else:
            asset_conflicts_found = False
            asset_report_lines = []
        
        # Report all conflicts
        if (constant_conflicts_found or asset_conflicts_found) and not allow_conflicts:
            report_parts = []
            if constant_conflicts_found:
                report_parts.append("Constant conflicts detected:\n" + "\n".join(constant_report_lines))
            if asset_conflicts_found:
                report_parts.append("Asset conflicts detected:\n" + "\n".join(asset_report_lines))
            raise ValueError(
                "\n".join(report_parts) + "\n(Pass allow_conflicts=True to ignore this warning.)"
            )
        elif constant_conflicts_found or asset_conflicts_found:
            if constant_conflicts_found:
                print("WARNING: Constant conflicts detected:\n" + "\n".join(constant_report_lines))
            if asset_conflicts_found:
                print("WARNING: Asset conflicts detected:\n" + "\n".join(asset_report_lines))

        return const_overrides

    def __init__(self, 
                 env, 
                 mods_config: list,  # Full list of mod keys
                 registry: dict,     # Full game registry
                 allow_conflicts: bool = False
                 ):
        
        self._env = env
        self.registry = registry
        
        active_internal_plugins = {}
        patch_map = defaultdict(list)
        
        # 1. --- BUILD & FILTER PHASE ---
        internal_mod_keys = []
        for mod_key in mods_config:
            if mod_key not in self.registry:
                raise ValueError(f"Mod '{mod_key}' not recognized.")
            
            plugin_class = self.registry[mod_key]
            
            # Validate that the registry entry is a class (not a function, list, etc.)
            if isinstance(plugin_class, list):
                raise TypeError(
                    f"Mod '{mod_key}' in registry is a modpack (list), but modpacks should have been "
                    f"expanded before reaching this point. This indicates a bug in the mod expansion logic."
                )
            if not inspect.isclass(plugin_class):
                raise TypeError(
                    f"Mod '{mod_key}' in registry is not a class. "
                    f"Found type: {type(plugin_class).__name__}, value: {plugin_class}. "
                    f"Mod registry entries must be classes that inherit from "
                    f"JaxAtariInternalModPlugin or JaxAtariPostStepModPlugin."
                )
            
            # This class only cares about Internal plugins
            if issubclass(plugin_class, JaxAtariInternalModPlugin):
                try:
                    plugin_instance = plugin_class()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to instantiate mod '{mod_key}' (class: {plugin_class.__name__}). "
                        f"Error: {e}"
                    ) from e
                active_internal_plugins[mod_key] = plugin_instance
                internal_mod_keys.append(mod_key)
                
                # Attribute overrides will be collected and applied later via helper function
                # Build patch map for functional conflicts
                for fn_name, _ in inspect.getmembers(plugin_instance, predicate=inspect.ismethod):
                    if not fn_name.startswith("__"):
                        if not (hasattr(self._env, fn_name) or (hasattr(self._env, 'renderer') and hasattr(self._env.renderer, fn_name))):
                            raise AttributeError(
                                f"Mod '{mod_key}' tries to patch '{fn_name}', but neither env nor renderer define it."
                            )
                        patch_map[fn_name].append(mod_key)
        if not active_internal_plugins:
            return
        
        # 2. --- REPORTING (Gameplay Conflicts) ---
        _check_gameplay_conflicts(active_internal_plugins, allow_conflicts)
        
        # 2b. --- REPORTING (Functional Conflicts - Method Patches) ---
        # Check for conflicts in method patches (functional conflicts)
        functional_conflicts_found = False
        functional_report_lines = []
        for fn_name, mods in patch_map.items():
            if len(mods) > 1:
                functional_conflicts_found = True
                winner = mods[-1]
                functional_report_lines.append(
                    f"  - The function '{fn_name}' is modified by multiple mods: {mods}. "
                    f"With allow_conflicts=True, the last mod in the list ('{winner}') would take priority."
                )
        
        # Note: Attribute conflicts are checked and reported in the APPLY PHASE below
        # to ensure they're checked across ALL mods (both Internal and PostStep)
        
        if functional_conflicts_found and not allow_conflicts:
            raise ValueError(
                "Functional conflicts detected:\n" + "\n".join(functional_report_lines) + "\n(Pass allow_conflicts=True to ignore this warning.)"
            )
        elif functional_conflicts_found:
            print("WARNING: Functional conflicts detected:\n" + "\n".join(functional_report_lines))
        
        # 3. --- APPLY PHASE (Simplified) ---
        # Collect and check attribute overrides for ALL mods (both Internal and PostStep)
        # This ensures conflicts are detected across all plugin types
        all_attr_overrides, attr_conflicts_found, attr_report_lines = _collect_and_check_attribute_overrides(
            mods_config, self.registry, self._env, allow_conflicts
        )
        
        # Report attribute conflicts along with functional conflicts
        if attr_conflicts_found:
            if not allow_conflicts:
                raise ValueError(
                    "Attribute conflicts detected:\n" + "\n".join(attr_report_lines) + 
                    "\n(Pass allow_conflicts=True to ignore this warning.)"
                )
            else:
                print("WARNING: Attribute conflicts detected:\n" + "\n".join(attr_report_lines))
        
        for mod_key in internal_mod_keys:
            plugin = active_internal_plugins[mod_key]
            
            # Apply Attribute Overrides (to env) if this Internal mod has any
            if mod_key in all_attr_overrides:
                _apply_attribute_overrides(self._env, all_attr_overrides[mod_key], mod_key)
            
            # Provide env reference for plugin logic
            plugin._env = self._env

            # Apply Function Patches (to env OR renderer)
            for fn_name, fn_logic in inspect.getmembers(plugin, predicate=inspect.ismethod):
                if not fn_name.startswith("__"):
                    # Use the bound method directly; jit(static_argnums=(0,)) expects the instance as arg 0
                    
                    env_has_attr = hasattr(self._env, fn_name)
                    renderer_has_attr = hasattr(self._env, 'renderer') and hasattr(self._env.renderer, fn_name)
                    if env_has_attr and renderer_has_attr:
                        # This is ambiguous! Fail loudly.
                        raise AttributeError(
                            f"Mod '{mod_key}' tries to patch '{fn_name}', but this method exists on BOTH "
                            f"the base environment and the renderer. The modding system cannot determine which to override."
                        )
                    elif env_has_attr:
                        setattr(self._env, fn_name, fn_logic)
                    elif renderer_has_attr:
                        setattr(self._env.renderer, fn_name, fn_logic)
                    
                    # The 'else' case (does not exist anywhere) is already
                    # handled by the pre-check earlier in this method.
            
    def __getattr__(self, name):
        """Delegates all other attribute and method access to the wrapped environment."""
        return getattr(self._env, name)



# --- 4. ModWrapper for post-step plugins ---
class JaxAtariModWrapper(JaxatariWrapper):
    """
    A generic Mod Wrapper that handles all *post-step* logic.
    """
    def __init__(self, 
                 env, 
                 mods_config: list,  # Full list of mod keys
                 allow_conflicts: bool = False
                 ):
        
        super().__init__(env)
        
        try:
            registry = self._env.registry
        except AttributeError:
            raise TypeError(
                "JaxAtariModWrapper must be applied to a JaxAtariModController, "
                "but the environment provided does not have a .registry attribute."
            )

        # 1. --- BUILD & FILTER PHASE ---
        active_plugins = {} # {mod_key: instance}
        for mod_key in mods_config:
            if mod_key not in registry:
                raise ValueError(f"Mod '{mod_key}' not recognized.")
            
            plugin_class = registry[mod_key]
            
            # Validate that the registry entry is a class (not a function, list, etc.)
            if isinstance(plugin_class, list):
                raise TypeError(
                    f"Mod '{mod_key}' in registry is a modpack (list), but modpacks should have been "
                    f"expanded before reaching this point. This indicates a bug in the mod expansion logic."
                )
            if not inspect.isclass(plugin_class):
                raise TypeError(
                    f"Mod '{mod_key}' in registry is not a class. "
                    f"Found type: {type(plugin_class).__name__}, value: {plugin_class}. "
                    f"Mod registry entries must be classes that inherit from "
                    f"JaxAtariInternalModPlugin or JaxAtariPostStepModPlugin."
                )
            
            if issubclass(plugin_class, JaxAtariPostStepModPlugin):
                try:
                    active_plugins[mod_key] = plugin_class()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to instantiate mod '{mod_key}' (class: {plugin_class.__name__}). "
                        f"Error: {e}"
                    ) from e
        
        if not active_plugins:
            self.post_step_mods = []
            self.after_reset_mods = []
            return
        
        # 2. --- REPORT PHASE (Gameplay Conflicts) ---
        # Use the shared helper
        _check_gameplay_conflicts(active_plugins, allow_conflicts)
        
        # 3. --- APPLY PHASE ---
        # Note: Attribute overrides were already collected and checked in JaxAtariModController
        # We just need to apply them for PostStep plugins here
        # Get attribute overrides that were already collected (stored in controller's registry check)
        # We'll collect them again but skip conflict checking since it was already done
        all_attr_overrides = {}
        for mod_key in mods_config:
            plugin_class = registry[mod_key]
            if hasattr(plugin_class, "attribute_overrides") and plugin_class.attribute_overrides:
                all_attr_overrides[mod_key] = plugin_class.attribute_overrides
        
        # Store env reference on each plugin instance
        for mod_key, plugin in active_plugins.items():
            plugin._env = self._env._env
            
            # Apply Attribute Overrides (to base env) if this PostStep mod has any
            if mod_key in all_attr_overrides:
                _apply_attribute_overrides(self._env._env, all_attr_overrides[mod_key], mod_key)
        
        # Store run methods (only for plugins that implement it)
        # Check if the method is overridden (not just using the default implementation)
        self.post_step_mods = []
        for plugin in active_plugins.values():
            # Check if plugin class has overridden run by checking if it's in the class __dict__
            plugin_class = type(plugin)
            if 'run' in plugin_class.__dict__:
                self.post_step_mods.append(plugin.run)
        
        # Store after_reset methods (only for plugins that implement it)
        # Check if the method is overridden (not just using the default implementation)
        self.after_reset_mods = []
        for plugin in active_plugins.values():
            # Check if plugin class has overridden after_reset by checking if it's in the class __dict__
            plugin_class = type(plugin)
            if 'after_reset' in plugin_class.__dict__:
                self.after_reset_mods.append(plugin.after_reset)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        # 1. Run the base reset
        obs, state = self._env.reset(key)
        
        # 2. Run all after_reset mods in order
        for after_reset_fn in self.after_reset_mods:
            obs, state = after_reset_fn(obs, state)
        
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # 1. Run the (modded) step from the underlying env
        obs, new_state, reward, done, info = self._env.step(state, action)
        
        # 2. Run all post-step mods in order
        for mod_fn in self.post_step_mods:
            new_state = mod_fn(state, new_state)
            
        return obs, new_state, reward, done, info


def apply_modifications(
    game_name: str,
    mods_config: list,
    allow_conflicts: bool,
    base_consts,
    env_class,
    MOD_MODULES: dict
) -> JaxEnvironment:
    """
    Applies the full two-stage modding pipeline to a base environment class.

    This is the main entry point for the modding system, called by core.make().
    """

    if game_name not in MOD_MODULES:
        raise NotImplementedError(f"No mod module defined for '{game_name}'.")

    try:
        ControllerClass = _load_from_string(MOD_MODULES[game_name])
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to load mod controller for '{game_name}'. "
            f"Path: {MOD_MODULES[game_name]}. Error: {e}"
        ) from e
    
    if not hasattr(ControllerClass, 'REGISTRY'):
        raise AttributeError(
            f"Mod controller class '{ControllerClass.__name__}' does not have a REGISTRY attribute. "
            f"All mod controllers must define a REGISTRY class attribute."
        )
    
    registry = ControllerClass.REGISTRY
    if not isinstance(registry, dict):
        raise TypeError(
            f"Mod controller '{ControllerClass.__name__}' has a REGISTRY that is not a dictionary. "
            f"Found type: {type(registry).__name__}."
        )

    expanded_mods_config = []
    seen_mods = set()

    def expand_mods(mod_list, depth=0):
        if depth > 10:
            raise RecursionError("Circular dependency detected in modpacks.")
        for mod_key in mod_list:
            if mod_key in seen_mods:
                continue
            # Validate mod_key is a string
            if not isinstance(mod_key, str):
                raise TypeError(
                    f"Invalid mod key in modpack: expected string, got {type(mod_key).__name__}: {mod_key}. "
                    f"Modpacks must contain only string keys that reference other mods."
                )
            if mod_key not in registry:
                raise ValueError(
                    f"Mod '{mod_key}' not recognized. "
                    f"Available mods: {list(registry.keys())}"
                )
            plugin = registry[mod_key]
            if isinstance(plugin, list):
                expand_mods(plugin, depth + 1)
            else:
                expanded_mods_config.append(mod_key)
                seen_mods.add(mod_key)

    expand_mods(mods_config)

    const_overrides = ControllerClass.pre_scan_for_overrides(
        expanded_mods_config,
        registry,
        base_consts,
        allow_conflicts
    )

    modded_consts = base_consts._replace(**const_overrides)

    base_env = env_class(consts=modded_consts)

    modded_env = ControllerClass(
        env=base_env,
        mods_config=expanded_mods_config,
        allow_conflicts=allow_conflicts
    )

    final_env = JaxAtariModWrapper(
        env=modded_env,
        mods_config=expanded_mods_config,
        allow_conflicts=allow_conflicts
    )

    return final_env