"""Create / extend quirks-registry entries for games by exploring their observation structure.

Runs a short random trajectory per game, classifies every spatial observation field (entity, xy_pairs,
true_2d_grid, flat_per_row, slot_flags), maps names to the 7-class taxonomy via keyword rules and adds
entries for fields that are not yet in the registry (existing, hand-tuned entries are never changed).
Unmatched names get class_id "REVIEW" and need a manual decision.

    python train_detection/tools/build_registry.py pong breakout        # specific games
    python train_detection/tools/build_registry.py --all                # all games
    python train_detection/tools/build_registry.py pong --inspect       # only print the observation structure

Afterwards tune offsets / grid geometry of new entries by rendering labels:
    python train_detection/generate_dataset.py <game> --preview
"""
import argparse
import json
from pathlib import Path

import jax
import jaxtari
import jaxtari.core
from jaxtari.wrappers import AtariWrapper, PixelAndObjectObsWrapper

REGISTRY_PATH = Path(jaxtari.__file__).parent / "detection" / "quirks_registry.json"

# =============================================================================
#  GLOBAL CLASS MAP
#
#  0  player       — the character(s) the human controls
#  1  enemy        — anything that can kill/harm the player
#  2  projectile   — bullets, missiles, bombs (player or enemy)
#  3  collectible  — items that give points, power-ups, rewards
#  4  structure    — platforms, blocks, walls, static grid elements
#  5  neutral      — moving objects that are neither threat nor reward
# =============================================================================

GLOBAL_CLASSES = {
    0: "player",
    1: "enemy",
    2: "projectile",
    3: "collectible",
    4: "structure",
    5: "neutral",
    6: "enemy_projectile"
}

# ---------------------------------------------------------------------------
# Keyword rules for auto-assigning class IDs.
# Checked in order — first match wins.
# Extend this list freely if you find object names that don't resolve well.
# ---------------------------------------------------------------------------

CLASS_KEYWORDS = [
    # class 0 — player
    (0, ["player", "bailey", "chicken", "human", "hook", "active_piece"]),

    # class 1 — enemy
    (1, ["enemy", "enemies", "otto", "bear", "shark", "spider",
         "centipede", "flea", "scorpion", "alien", "kong", "monkeys",
         "mothership", "byte_bat", "rock_muncher", "radar_mortar",
         "kamikaze", "bouncer", "chasing", "ghost", "monster", "chaser", 
         "asteroids", "tentacles", "dragon", "level_boss"]),

    # class 2 — projectile
    (2, ["player_bullet", "player_missile", "player_shot", "player_spell", "player_projectile",
         "spell", "laser", "torpedo", "fireball", "detonator", "spear", "dynamite"]),

    # class 3 — collectible
    (3, ["collectible", "fruit", "child", "bell", "princess",
         "item", "coin", "rejuvenator", "energy_pod", "fish",
         "score_item", "kill_item", "prize", "chest", "key",
         "power", "pellet", "letter", "target_word", "oxygen_line", "fuel_tank", "divers"]),

    # class 4 — structure
    (4, ["block", "platform", "ladder", "wall", "grid", "path",
         "bank", "installation", "forcefield", "densepack",
         "mountain", "lane_blocker", "road", "ice", "obstacle",
         "board", "cube", "bumper", "flipper", "plunger",
         "rollover", "hole", "spinner", "door", "portal",
         "rope", "conveyor", "completed_rect", "walked", "barricade_health", 
         "house_tree", "dam_indicator", "star_base", "targets"]),

    # class 5 — neutral
    (5, ["truck", "car", "jet", "chopper", "cloud", "ufo",
         "falling_rock", "meteoroid", "debris", "disc", "boat", "ball", "next_piece", "ball"]),

    # class 6 - enemy_projectile
    (6, ["enemy_bullet", "enemy_bullets", "enemy_shot", "enemy_shots", "enemy_missile", "enemy_missiles",
         "plasma", "bomb", "bombs", "homing_missile", "radar_mortar_missile", "rock_muncher_missile",
         "coconut", "coconuts"])
]


def auto_assign_class(obj_name):
    """
    Walk CLASS_KEYWORDS and return the first matching class ID.
    Returns None if nothing matches —  'REVIEW' in the JSON.
    """
    name_lower = obj_name.lower()
    for class_id, keywords in CLASS_KEYWORDS:
        for kw in keywords:
            if kw in name_lower:
                return class_id
    return None


def detect_type(obj_data):
    """
    Returns {"detected_type": str, "raw_shape": list|None}
    """
    if obj_data is None:
        return {"detected_type": "non_spatial", "raw_shape": None}

    # 1. PRESERVE THIS: Standard JAXtari entities with explicit coordinates
    if hasattr(obj_data, 'x') or hasattr(obj_data, 'xy'):
        return {"detected_type": "entity", "raw_shape": None}

    # 2. THE NEW ARRAY CLASSIFIERS
    if hasattr(obj_data, 'shape'):
        shape = list(obj_data.shape)
        total = 1
        for s in shape:
            total *= s

        # Ignore non-spatial scalars or tiny boolean arrays
        if len(shape) == 0 or total <= 8:
            return {"detected_type": "non_spatial", "raw_shape": shape}

        # Pattern 1: slot_flags [4, N] (No spatial data, skip rendering)
        if len(shape) == 2 and shape[0] == 4 and shape[1] < 210:
            return {"detected_type": "slot_flags", "raw_shape": shape}

        # Pattern 2: xy_pairs [4, N, 2] (Raw coordinates)
        if len(shape) == 3 and shape[0] == 4 and shape[2] == 2:
            return {"detected_type": "xy_pairs", "raw_shape": shape}

        # Pattern 3: true_2d_grid [4, rows, cols]
        if len(shape) == 3 and shape[0] == 4 and shape[2] != 2:
            return {"detected_type": "true_2d_grid", "raw_shape": shape}

        # Pattern 4: flat_per_row [4, 210] (Scanlines)
        if len(shape) == 2 and shape[1] == 210:
            return {"detected_type": "flat_per_row", "raw_shape": shape}

        # Catch anything weird that survived
        return {"detected_type": "unknown_array", "raw_shape": shape}

    return {"detected_type": "non_spatial", "raw_shape": None}



def build_entry(obj_name, detected_type, raw_shape):
    class_id = auto_assign_class(obj_name)
    class_id_value = class_id if class_id is not None else "REVIEW"

    # XY pairs are entities, true_2d_grids are grids
    is_entity = detected_type in ["entity", "xy_pairs"]
    is_grid = detected_type == "true_2d_grid"

    # Infer grid_cols from shape where possible
    if is_grid and raw_shape:
        if len(raw_shape) == 1:
            grid_cols = "TODO"   # flat 1-D: total cells known, cols unknown
        else:
            grid_cols = raw_shape[-1]   # (rows, cols) or (frames, rows, cols)
    else:
        grid_cols = None

    return {
        "class_id":      class_id_value,   # int if matched, "REVIEW" if not
        "detected_type": detected_type,


        "x_offset": 0    if is_entity else None,
        "y_offset": 0    if is_entity else None,

        # ── grid fields (if array) ──────────────────────────────────
        # Tune grid_origin_* and cell_w/h with the debug overlay.
        # grid_cols is auto-inferred where possible, set manually for flat 1-D.
        "grid_origin_x": 0    if is_grid else None,
        "grid_origin_y": 0    if is_grid else None,
        "cell_w":        8    if is_grid else None,
        "cell_h":        8    if is_grid else None,
        "grid_cols":     grid_cols,
        "active_value":  1.0  if is_grid else None,

        "_raw_shape":    raw_shape,
    }


# ---------------------------------------------------------------------------
# Games list


def explore(game_name, steps, seed=42):
    """Random trajectory -> {field name: {detected_type, raw_shape}} of all spatial observation fields."""
    env = PixelAndObjectObsWrapper(AtariWrapper(jaxtari.make(game_name)))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    obs, state = reset(k)
    discovered = {}
    for t in range(steps):
        rng, k = jax.random.split(rng)
        obs, state, _, done, truncated, _ = step(state, jax.random.randint(k, (), 0, env.action_space().n))
        _, obs_stack = obs
        fields = obs_stack._asdict() if hasattr(obs_stack, "_asdict") else obs_stack.__dict__
        for name, data in fields.items():
            if name in discovered:
                continue
            meta = detect_type(data)
            if meta["detected_type"] == "non_spatial":
                continue
            discovered[name] = meta
            cls = auto_assign_class(name)
            label = f"class {cls} ({GLOBAL_CLASSES[cls]})" if cls is not None else "REVIEW"
            shape = f"  shape={meta['raw_shape']}" if meta["raw_shape"] else ""
            print(f"  [{t:>4}]  {meta['detected_type'].upper():14s}  {name:<30s}  -> {label}{shape}")
        if bool(done) or bool(truncated):
            rng, k = jax.random.split(rng)
            obs, state = reset(k)
    return discovered


def inspect(game_name):
    """Print the observation fields of one frame and their attributes."""
    env = PixelAndObjectObsWrapper(AtariWrapper(jaxtari.make(game_name)))
    (_, obs_stack), _ = env.reset(jax.random.PRNGKey(42))
    fields = obs_stack._asdict() if hasattr(obs_stack, "_asdict") else obs_stack.__dict__
    for name, data in fields.items():
        keys = list(data.keys()) if isinstance(data, dict) else [k for k in dir(data) if not k.startswith("_")]
        print(f"{name:30s} {detect_type(data)['detected_type']:14s} {keys if hasattr(data, 'x') else getattr(data, 'shape', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="*")
    ap.add_argument("--all", action="store_true", help="explore all games registered in jaxtari")
    ap.add_argument("--steps", type=int, default=300, help="exploration steps per game")
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--inspect", action="store_true", help="only print the observation structure")
    a = ap.parse_args()
    games = sorted(jaxtari.core.list_available_games()) if a.all else a.games
    if a.inspect:
        for g in games:
            print(f"== {g}")
            inspect(g)
        return
    path = Path(a.registry)
    registry = json.loads(path.read_text()) if path.exists() else {}
    for g in games:
        print(f"== {g}")
        try:
            discovered = explore(g, a.steps)
        except Exception as e:  # keep going with the other games
            print(f"  [FAILED] {type(e).__name__}: {e}")
            continue
        objects = registry.setdefault(g, {"objects": {}})["objects"]
        new = [n for n in discovered if n not in objects]
        for n in new:  # existing (hand-tuned) entries are kept as they are
            objects[n] = build_entry(n, discovered[n]["detected_type"], discovered[n]["raw_shape"])
        path.write_text(json.dumps(registry, indent=4))
        review = [n for n, e in objects.items() if e.get("class_id") == "REVIEW"]
        print(f"  {len(discovered)} spatial fields, {len(new)} new entries" + (f", REVIEW: {review}" if review else ""))


if __name__ == "__main__":
    main()
