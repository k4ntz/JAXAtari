#!/usr/bin/env python3
"""
Trajectory regression: record a baseline trajectory (states + actions, and optionally
pixel + object-centric observations) and later replay to verify refactors did not change behavior.

By default we save and compare: state, pixel observations, and object-centric observations.
So you can see whether a change affected only visuals (pixel_obs differed), only object
state (object_centric_obs differed), or full state. Use --no-obs for state-only (smaller
baselines, backward compatible).

Usage:
  # Record baseline (manual: play until you press 'q')
  python trajectory_regression.py -g pong --save-baseline pong_baseline.pkl --mode manual

  # Record baseline (automatic: N random steps; includes state + pixel_obs + oc_obs)
  python trajectory_regression.py -g pong --save-baseline pong_baseline.pkl --mode automatic --steps 500

  # Compare: reports which of state / pixel_obs / object_centric_obs differed at each checkpoint
  python trajectory_regression.py -g pong --compare-to pong_baseline.pkl

  # State-only (no pixel/oc): --no-obs
  python trajectory_regression.py -g pong --save-baseline pong.pkl --no-obs --mode automatic --steps 200

With mods:
  python trajectory_regression.py -g seaquest -m MyMod --save-baseline seaquest_mod.pkl --mode automatic --steps 300
  python trajectory_regression.py -g seaquest -m MyMod --compare-to seaquest_mod.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Allow running from project root or scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.name == "scripts" and str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from jax import tree_util


def _normalize_mods(mods):
    """Normalize mods to a list of strings (same as play.py)."""
    if mods is None:
        return None
    if isinstance(mods, str):
        mods = [mods]
    result = []
    for item in mods:
        item = str(item).strip() if item is not None else ""
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result if result else None


def _load_env(game_name: str, mods: list[str] | None, allow_conflicts: bool = False):
    """Load environment via core.make or scripts.utils fallback (same as play.py)."""
    try:
        from jaxatari.core import make as jaxatari_make

        env = jaxatari_make(
            game_name=game_name,
            mods=mods,
            allow_conflicts=allow_conflicts,
        )
        return env
    except (NotImplementedError, ImportError) as e:
        from utils import load_game_environment, load_game_mods

        game_env, _ = load_game_environment(game_name)
        if mods:
            mod_applier = load_game_mods(
                game_name=game_name,
                mods_config=mods,
                allow_conflicts=allow_conflicts,
            )
            return mod_applier(game_env)
        return game_env


def _load_env_with_obs(game_name: str, mods: list[str] | None, allow_conflicts: bool = False):
    """Load base env and wrap with AtariWrapper + PixelAndObjectCentricWrapper to get (pixel, object-centric) observations."""
    from jaxatari.wrappers import AtariWrapper, PixelAndObjectCentricWrapper

    base_env = _load_env(game_name, mods, allow_conflicts)
    return PixelAndObjectCentricWrapper(AtariWrapper(base_env, frame_skip=1))


# --- State serialization (PyTree: treedef + numpy leaves) ---


def _state_to_numpy_pickle(state) -> tuple:
    """Flatten state to (treedef, list of numpy arrays) for pickle."""
    leaves, treedef = tree_util.tree_flatten(state)
    numpy_leaves = [np.asarray(x) for x in leaves]
    return (treedef, numpy_leaves)


def _state_from_numpy_pickle(treedef, numpy_leaves):
    """Restore state from (treedef, list of numpy arrays)."""
    jax_leaves = [jnp.array(x) for x in numpy_leaves]
    return tree_util.tree_unflatten(treedef, jax_leaves)


def _trees_equal(a, b) -> bool:
    """Structural equality of two pytrees (element-wise)."""
    leaves_a = tree_util.tree_leaves(a)
    leaves_b = tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(bool(jnp.all(la == lb)) for la, lb in zip(leaves_a, leaves_b))


def _states_equal(state_a, state_b) -> bool:
    """Structural equality of two states (element-wise)."""
    return _trees_equal(state_a, state_b)


# --- Baseline file format ---


def save_baseline(
    path: str | Path,
    game_name: str,
    mods: list | None,
    seed: int,
    actions: np.ndarray,
    checkpoint_step_indices: np.ndarray,
    checkpoint_treedef,
    checkpoint_numpy_leaves_list: list,
    *,
    with_obs: bool = False,
    pixel_treedef=None,
    checkpoint_pixel_leaves_list: list | None = None,
    oc_treedef=None,
    checkpoint_oc_leaves_list: list | None = None,
):
    """Save baseline to a pickle file. If with_obs is True, pixel and object-centric obs are stored too."""
    data = {
        "game": game_name,
        "mods": mods if mods is not None else [],
        "seed": seed,
        "actions": np.asarray(actions, dtype=np.int32),
        "checkpoint_step_indices": np.asarray(checkpoint_step_indices, dtype=np.int64),
        "treedef": checkpoint_treedef,
        "checkpoint_leaves": checkpoint_numpy_leaves_list,
        "with_obs": with_obs,
    }
    if with_obs and checkpoint_pixel_leaves_list is not None and checkpoint_oc_leaves_list is not None:
        data["pixel_treedef"] = pixel_treedef
        data["checkpoint_pixel_leaves"] = checkpoint_pixel_leaves_list
        data["oc_treedef"] = oc_treedef
        data["checkpoint_oc_leaves"] = checkpoint_oc_leaves_list
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    obs_info = " (state + pixel_obs + oc_obs)" if with_obs else ""
    print(f"Saved baseline to {path} ({len(actions)} actions, {len(checkpoint_step_indices)} checkpoints){obs_info}")


def load_baseline(path: str | Path):
    """Load baseline from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# --- Recording ---


def _run_automatic(
    env,
    seed: int,
    num_steps: int,
    num_checkpoints: int,
    action_space,
    jitted_reset,
    jitted_step,
    capture_obs=None,
):
    """Run env with random actions for num_steps; record actions and state checkpoints.
    Checkpoints are stored as 'state after n steps' for n in checkpoint_step_indices (e.g. [0, 50, 100]).
    If capture_obs(obs, state) is provided, it should return (pixel_obs, oc_obs) and they are saved at each checkpoint.
    """
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = jitted_reset(reset_key)

    actions = []
    if num_steps <= 0:
        checkpoint_indices = [0]
    else:
        checkpoint_indices = np.linspace(
            0, num_steps, num=min(num_checkpoints, num_steps + 1), dtype=np.int64
        ).tolist()
        checkpoint_indices = sorted(set(int(x) for x in checkpoint_indices))

    checkpoints_treedef = None
    checkpoints_leaves = []
    pixel_treedef = None
    checkpoint_pixel_leaves = []
    oc_treedef = None
    checkpoint_oc_leaves = []
    next_checkpoint_idx = 0
    steps_done = 0

    def _save_checkpoint(s, current_obs):
        nonlocal checkpoints_treedef, pixel_treedef, oc_treedef
        td, le = _state_to_numpy_pickle(s)
        if checkpoints_treedef is None:
            checkpoints_treedef = td
        checkpoints_leaves.append(le)
        if capture_obs is not None and current_obs is not None:
            pixel_obs, oc_obs = capture_obs(current_obs, s)
            pt, pl = _state_to_numpy_pickle(pixel_obs)
            ot, ol = _state_to_numpy_pickle(oc_obs)
            if pixel_treedef is None:
                pixel_treedef, oc_treedef = pt, ot
            checkpoint_pixel_leaves.append(pl)
            checkpoint_oc_leaves.append(ol)

    # Checkpoint 0: state after 0 steps (initial state)
    if next_checkpoint_idx < len(checkpoint_indices) and checkpoint_indices[next_checkpoint_idx] == 0:
        _save_checkpoint(state, obs)
        next_checkpoint_idx += 1

    for step in range(num_steps):
        action = action_space.sample(key)
        key, _ = jax.random.split(key)
        actions.append(int(action))
        obs, state, reward, done, info = jitted_step(state, action)
        steps_done = step + 1
        if done:
            break
        if next_checkpoint_idx < len(checkpoint_indices) and steps_done == checkpoint_indices[next_checkpoint_idx]:
            _save_checkpoint(state, obs)
            next_checkpoint_idx += 1

    # Store final state if it falls on a checkpoint we haven't hit yet
    while next_checkpoint_idx < len(checkpoint_indices) and checkpoint_indices[next_checkpoint_idx] <= steps_done:
        _save_checkpoint(state, obs)
        next_checkpoint_idx += 1

    checkpoint_step_indices = np.array(checkpoint_indices[: len(checkpoints_leaves)], dtype=np.int64)
    result = (
        np.array(actions, dtype=np.int32),
        checkpoint_step_indices,
        checkpoints_treedef,
        checkpoints_leaves,
    )
    if capture_obs is not None and checkpoint_pixel_leaves:
        return result + (pixel_treedef, checkpoint_pixel_leaves, oc_treedef, checkpoint_oc_leaves)
    return result


def _run_manual(
    env,
    seed: int,
    num_checkpoints: int,
    jitted_reset,
    jitted_step,
    get_action_fn,
    quit_check_fn,
    after_step_callback=None,
    capture_obs=None,
):
    """Run env with human input until quit_check_fn() returns True. Record actions and state checkpoints.
    If capture_obs(obs, state) returns (pixel_obs, oc_obs), they are saved at each checkpoint.

    Manual mode records a checkpoint at step 0, then every 50 steps, then at quit. If that produces
    more than num_checkpoints checkpoints, we keep num_checkpoints evenly spaced ones (downsample).
    So --num-checks caps how many checkpoints are stored; we still buffer every 50 steps during play.
    """
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = jitted_reset(reset_key)
    if after_step_callback:
        after_step_callback(state)

    actions = []
    step = 0
    checkpoints_treedef = None
    checkpoints_leaves = []
    checkpoint_step_indices_list = []
    checkpoint_interval = 50  # record every N steps during play; later downsample to num_checkpoints
    pixel_treedef = None
    checkpoint_pixel_leaves = []
    oc_treedef = None
    checkpoint_oc_leaves = []

    def _save_cp(s, current_obs):
        nonlocal checkpoints_treedef, pixel_treedef, oc_treedef
        td, le = _state_to_numpy_pickle(s)
        if checkpoints_treedef is None:
            checkpoints_treedef = td
        checkpoints_leaves.append(le)
        if capture_obs is not None and current_obs is not None:
            pixel_obs, oc_obs = capture_obs(current_obs, s)
            pt, pl = _state_to_numpy_pickle(pixel_obs)
            ot, ol = _state_to_numpy_pickle(oc_obs)
            if pixel_treedef is None:
                pixel_treedef, oc_treedef = pt, ot
            checkpoint_pixel_leaves.append(pl)
            checkpoint_oc_leaves.append(ol)

    # Checkpoint at step 0
    _save_cp(state, obs)
    checkpoint_step_indices_list.append(0)

    while True:
        if quit_check_fn():
            break
        action = get_action_fn()
        if quit_check_fn():
            break
        actions.append(int(action))
        obs, state, reward, done, info = jitted_step(state, jnp.array(action, dtype=jnp.int32))
        if after_step_callback:
            after_step_callback(state)
        step += 1
        if done:
            key, reset_key = jax.random.split(key)
            obs, state = jitted_reset(reset_key)
            if after_step_callback:
                after_step_callback(state)
        if step > 0 and step % checkpoint_interval == 0:
            _save_cp(state, obs)
            checkpoint_step_indices_list.append(step)

    # Final state (after last action)
    _save_cp(state, obs)
    checkpoint_step_indices_list.append(step)

    # Downsample to num_checkpoints if we have too many
    n_cp = len(checkpoints_leaves)
    if n_cp > num_checkpoints:
        idx = np.linspace(0, n_cp - 1, num=num_checkpoints, dtype=np.int64)
        checkpoints_leaves = [checkpoints_leaves[i] for i in idx]
        checkpoint_step_indices_list = [checkpoint_step_indices_list[i] for i in idx]
        if checkpoint_pixel_leaves:
            checkpoint_pixel_leaves = [checkpoint_pixel_leaves[i] for i in idx]
            checkpoint_oc_leaves = [checkpoint_oc_leaves[i] for i in idx]
    checkpoint_step_indices = np.array(checkpoint_step_indices_list, dtype=np.int64)

    result = (
        np.array(actions, dtype=np.int32),
        checkpoint_step_indices,
        checkpoints_treedef,
        checkpoints_leaves,
    )
    if capture_obs is not None and checkpoint_pixel_leaves:
        return result + (pixel_treedef, checkpoint_pixel_leaves, oc_treedef, checkpoint_oc_leaves)
    return result


def run_save_baseline(
    game_name: str,
    mods: list | None,
    path: str | Path,
    mode: str,
    seed: int,
    steps: int,
    num_checkpoints: int,
    allow_conflicts: bool,
    with_obs: bool = True,
    fps: int = 15,
):
    """Record a baseline and save to path. If with_obs (default True), also save pixel and object-centric observations."""
    if with_obs:
        env = _load_env_with_obs(game_name, mods, allow_conflicts)
        # PixelAndObjectCentricWrapper returns obs = (image_stack, flat_obs)
        capture_obs = lambda obs, state: (obs[0], obs[1])
    else:
        env = _load_env(game_name, mods, allow_conflicts)
        capture_obs = None

    action_space = env.action_space()
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    if mode == "automatic":
        run_result = _run_automatic(
            env, seed, steps, num_checkpoints, action_space, jitted_reset, jitted_step, capture_obs=capture_obs
        )
        actions, cp_indices, treedef, cp_leaves = run_result[:4]
        pixel_treedef, cp_pixel, oc_treedef, cp_oc = (run_result[4:8] if len(run_result) > 4 else (None, None, None, None))
    else:
        # manual: need pygame and get_human_action, quit on 'q'
        import pygame
        from utils import get_human_action, update_pygame
        from jaxatari.environment import JAXAtariAction

        # With obs we use wrapped env: state is PixelAndObjectCentricState; use its image_stack for display.
        # env.render(state) would receive wrapped state and break (base renderer expects raw game state).
        if with_obs:
            jitted_render = jax.jit(lambda s: s.image_stack[-1])
        else:
            jitted_render = jax.jit(env.render)
        obs, state = jitted_reset(jax.random.PRNGKey(seed))
        frame = jitted_render(state)
        env_render_shape = frame.shape[:2]
        UPSCALE = 4
        pygame.init()
        window = pygame.display.set_mode(
            (env_render_shape[1] * UPSCALE, env_render_shape[0] * UPSCALE)
        )
        clock = pygame.time.Clock()
        pygame.display.set_caption(f"Trajectory baseline — {game_name} — press Q to quit and save")

        def map_action_to_index(action_constant):
            if hasattr(env, "ACTION_SET"):
                action_set = np.array(env.ACTION_SET)
                action_int = int(action_constant)
                matches = np.where(action_set == action_int)[0]
                if len(matches) > 0:
                    return int(matches[0])
            return int(action_constant)

        quit_requested = [False]  # list to allow closure to mutate

        def get_action():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested[0] = True
                    return jnp.array(0, dtype=jnp.int32)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    quit_requested[0] = True
                    return jnp.array(0, dtype=jnp.int32)
            ac = get_human_action()
            return jnp.array(map_action_to_index(ac), dtype=jnp.int32)

        def quit_check():
            return quit_requested[0]

        def render_callback(s):
            image = jitted_render(s)
            update_pygame(window, image, UPSCALE, 160, 210)
            clock.tick(fps)

        run_result = _run_manual(
            env,
            seed,
            num_checkpoints,
            jitted_reset,
            jitted_step,
            get_action_fn=get_action,
            quit_check_fn=quit_check,
            after_step_callback=render_callback,
            capture_obs=capture_obs,
        )
        actions, cp_indices, treedef, cp_leaves = run_result[:4]
        pixel_treedef, cp_pixel, oc_treedef, cp_oc = (run_result[4:8] if len(run_result) > 4 else (None, None, None, None))
        pygame.quit()

    save_baseline(
        path,
        game_name,
        mods,
        seed,
        actions,
        cp_indices,
        treedef,
        cp_leaves,
        with_obs=with_obs,
        pixel_treedef=pixel_treedef if with_obs else None,
        checkpoint_pixel_leaves_list=cp_pixel if with_obs else None,
        oc_treedef=oc_treedef if with_obs else None,
        checkpoint_oc_leaves_list=cp_oc if with_obs else None,
    )


def run_compare(
    game_name: str,
    mods: list | None,
    path: str | Path,
    allow_conflicts: bool,
):
    """Replay baseline and compare states (and optionally pixel/oc obs) at checkpoints; exit 0 if match, 1 if mismatch."""
    data = load_baseline(path)
    base_game = data["game"]
    base_mods = data.get("mods") or []
    seed = int(data["seed"])
    actions = data["actions"]
    checkpoint_step_indices = data["checkpoint_step_indices"]
    treedef = data["treedef"]
    checkpoint_leaves_list = data["checkpoint_leaves"]
    with_obs = data.get("with_obs", False)
    checkpoint_pixel_leaves = data.get("checkpoint_pixel_leaves")
    checkpoint_oc_leaves = data.get("checkpoint_oc_leaves")
    pixel_treedef = data.get("pixel_treedef")
    oc_treedef = data.get("oc_treedef")

    if base_game != game_name or base_mods != (mods or []):
        print(
            f"Warning: baseline was recorded with game={base_game} mods={base_mods}; "
            f"current run is game={game_name} mods={mods}. Proceeding anyway."
        )

    if with_obs and checkpoint_pixel_leaves is not None:
        env = _load_env_with_obs(game_name, mods, allow_conflicts)
    else:
        env = _load_env(game_name, mods, allow_conflicts)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = jitted_reset(reset_key)
    steps_done = 0
    action_idx = 0
    next_checkpoint_i = 0
    errors = []

    # Checkpoints are "state after n steps". Compare at each checkpoint.
    while next_checkpoint_i < len(checkpoint_step_indices):
        target_step = int(checkpoint_step_indices[next_checkpoint_i])
        while steps_done < target_step and action_idx < len(actions):
            action_jax = jnp.array(int(actions[action_idx]), dtype=jnp.int32)
            obs, state, reward, done, info = jitted_step(state, action_jax)
            action_idx += 1
            steps_done += 1
            if done:
                key, reset_key = jax.random.split(key)
                obs, state = jitted_reset(reset_key)
        if steps_done != target_step:
            errors.append(
                f"Checkpoint at step {target_step}: only {steps_done} actions in baseline (trajectory shorter than expected)."
            )
            next_checkpoint_i += 1
            continue

        # Compare state
        baseline_state = _state_from_numpy_pickle(treedef, checkpoint_leaves_list[next_checkpoint_i])
        state_ok = _trees_equal(state, baseline_state)
        diff_parts = []
        if not state_ok:
            diff_parts.append("state")

        # Compare pixel and object-centric obs if present
        if with_obs and checkpoint_pixel_leaves is not None and pixel_treedef is not None:
            baseline_pixel = _state_from_numpy_pickle(pixel_treedef, checkpoint_pixel_leaves[next_checkpoint_i])
            baseline_oc = _state_from_numpy_pickle(oc_treedef, checkpoint_oc_leaves[next_checkpoint_i])
            pixel_ok = _trees_equal(obs[0], baseline_pixel)
            oc_ok = _trees_equal(obs[1], baseline_oc)
            if not pixel_ok:
                diff_parts.append("pixel_obs")
            if not oc_ok:
                diff_parts.append("object_centric_obs")
        else:
            pixel_ok = oc_ok = True

        if diff_parts:
            errors.append(f"Checkpoint at step {target_step}: {' ,'.join(diff_parts)} differed.")
        next_checkpoint_i += 1

    if errors:
        for e in errors:
            print("FAIL:", e)
        print(f"Trajectory regression failed: {len(errors)} checkpoint(s) differ.")
        sys.exit(1)
    msg = f"OK: trajectory matches baseline ({len(checkpoint_step_indices)} checkpoints)"
    if with_obs and checkpoint_pixel_leaves is not None:
        msg += " (state, pixel_obs, object_centric_obs)"
    msg += "."
    print(msg)
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="Record or compare trajectory baselines for regression (same inputs → same states).",
    )
    ap.add_argument("-g", "--game", required=True, help="Game name (e.g. pong, seaquest).")
    ap.add_argument(
        "-m", "--mods",
        nargs="*",
        default=None,
        help="Mod name(s), space- or comma-separated.",
    )
    ap.add_argument("--allow_conflicts", action="store_true", help="Allow conflicting mods.")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--save-baseline",
        type=str,
        metavar="PATH",
        help="Record a baseline and save to PATH.",
    )
    group.add_argument(
        "--compare-to",
        type=str,
        metavar="PATH",
        help="Replay baseline from PATH and compare states at checkpoints.",
    )

    ap.add_argument(
        "--mode",
        choices=("automatic", "manual"),
        default="automatic",
        help="For --save-baseline: 'automatic' = random actions for N steps; 'manual' = play until Q.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=500,
        help="For --save-baseline --mode automatic: number of steps to run.",
    )
    ap.add_argument(
        "--num-checks",
        type=int,
        default=10,
        help="Number of checkpoints to keep. Automatic: N evenly spaced over --steps. Manual: record every 50 steps, then keep N evenly spaced (downsample).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX random seed for reset and (in automatic mode) actions.",
    )
    ap.add_argument(
        "--no-obs",
        action="store_true",
        help="Do not save or compare pixel/object-centric observations (state only).",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Manual mode: max steps per second (display/input rate). Default 30.",
    )

    args = ap.parse_args()
    args.mods = _normalize_mods(args.mods)

    if args.save_baseline:
        run_save_baseline(
            game_name=args.game,
            mods=args.mods,
            path=args.save_baseline,
            mode=args.mode,
            seed=args.seed,
            steps=args.steps,
            num_checkpoints=args.num_checks,
            allow_conflicts=args.allow_conflicts,
            with_obs=not args.no_obs,
            fps=args.fps,
        )
    else:
        run_compare(
            game_name=args.game,
            mods=args.mods,
            path=args.compare_to,
            allow_conflicts=args.allow_conflicts,
        )


if __name__ == "__main__":
    main()
