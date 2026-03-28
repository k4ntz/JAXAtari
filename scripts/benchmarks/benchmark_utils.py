"""Utility helpers shared by RL benchmark scripts."""

from __future__ import annotations

import importlib
import re
from typing import Any

from jaxatari.core import MOD_MODULES


def _coerce_to_str_list(value: Any) -> list[str]:
    """Normalize config values like None/string/list[string] into a clean list."""
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, str):
                raise TypeError(
                    f"Mod names must be strings. Got element {item!r} of type {type(item)}."
                )
            stripped = item.strip()
            if stripped:
                items.append(stripped)
        return items

    raise TypeError(
        f"Unsupported mod config type: {type(value)}. Expected None, string, or list of strings."
    )


def _camel_to_snake(name: str) -> str:
    step_1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    step_2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step_1)
    return step_2.lower()


def _canonicalize_mod_name(raw_name: str) -> str:
    """
    Turn legacy names (e.g. LazyEnemyWrapper / LazyEnemyMod / lazy-enemy)
    into canonical snake_case (e.g. lazy_enemy).
    """
    name = raw_name.strip().replace("-", "_").replace(" ", "_")
    name = re.sub(r"(Wrapper|Mod)$", "", name, flags=re.IGNORECASE)
    name = _camel_to_snake(name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _get_registered_mods(env_name: str) -> list[str]:
    """
    Return all registered mod keys for the given environment.
    Returns an empty list if the environment has no mod controller.
    """
    controller_path = MOD_MODULES.get(env_name.lower())
    if controller_path is None:
        return []

    module_path, class_name = controller_path.rsplit(".", 1)
    controller_module = importlib.import_module(module_path)
    controller_class = getattr(controller_module, class_name)
    registry = getattr(controller_class, "REGISTRY", {})
    return list(registry.keys())


def resolve_mods_for_env(env_name: str, mods_config: Any) -> list[str]:
    """
    Resolve config-provided mod names to registered mod keys.
    Supports legacy names from older benchmark configs.
    """
    requested_mods = _coerce_to_str_list(mods_config)
    if not requested_mods:
        return []

    available_mods = _get_registered_mods(env_name)
    available_map = {m.replace("_", ""): m for m in available_mods}

    resolved: list[str] = []
    for raw_mod in requested_mods:
        canonical = _canonicalize_mod_name(raw_mod)

        if canonical in available_mods:
            mod_key = canonical
        elif canonical.replace("_", "") in available_map:
            mod_key = available_map[canonical.replace("_", "")]
        elif not available_mods:
            # Allow passthrough if env has no registered mods table.
            mod_key = canonical
        else:
            raise ValueError(
                f"Unknown mod '{raw_mod}' for env '{env_name}'. "
                f"Resolved to '{canonical}', available mods: {available_mods}"
            )

        if mod_key not in resolved:
            resolved.append(mod_key)

    return resolved


def get_eval_mods(config: dict[str, Any]) -> list[str]:
    """
    Prefer new-style EVAL_MODS; fall back to legacy MOD_NAME.
    """
    env_name = config["ENV_NAME"]
    if config.get("EVAL_MODS", None) is not None:
        return resolve_mods_for_env(env_name, config["EVAL_MODS"])
    return resolve_mods_for_env(env_name, config.get("MOD_NAME", None))


def get_train_mods(config: dict[str, Any]) -> list[str]:
    """Read optional TRAIN_MODS from config."""
    env_name = config["ENV_NAME"]
    return resolve_mods_for_env(env_name, config.get("TRAIN_MODS", None))
