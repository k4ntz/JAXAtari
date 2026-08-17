"""Oracle lookahead: bake OC-future spawn/movement into JAX state at inject.

At sync frame ``t0``, scan the saved OC trajectory over ``[t0, t0+horizon)`` and
set latent spawn / facing fields so JAX fires the same events on the same
schedule. This is *not* PRNG inversion — it programs observable outcomes.

Enable via ``lockstep_eval.py --oracle-lookahead``.

INCLUDE (RNG spawn / movement): seaquest, enduro, asteroids, bankheist,
venture, beamrider, phoenix.

SKIP: kangaroo (deterministic FSM), breakout/pong/etc. (no enemy RNG).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oc_parity.translators.base import (
    collect_category,
    find_object,
    objects_as_dicts,
)

OracleFn = Callable[..., Any]

# Games where oracle lookahead is registered.
ORACLE_GAMES = (
    "seaquest",
    "enduro",
    "asteroids",
    "bankheist",
    "venture",
    "beamrider",
    "phoenix",
)


def supports_oracle(jax_key: str) -> bool:
    return jax_key in ORACLE_GAMES


def apply_oracle_lookahead(
    env,
    state: Any,
    traj: Mapping[str, Any],
    *,
    jax_key: str,
    t0: int,
    horizon: int,
) -> Any:
    """Return state with oracle adjustments, or unchanged if unsupported."""
    fn = _ORACLES.get(jax_key)
    if fn is None:
        return state
    T = len(traj.get("objects", []))
    if T <= t0:
        return state
    h = max(1, int(horizon))
    t1 = min(T, t0 + h)
    return fn(env, state, traj, t0=t0, t1=t1)


def _objs_at(traj: Mapping[str, Any], t: int) -> List[dict]:
    return objects_as_dicts(traj["objects"][t])


def _entity_signature(objs: Sequence[Mapping[str, Any]], category: str) -> List[Tuple[float, float]]:
    items = [
        (float(o.get("x", 0)), float(o.get("y", 0)))
        for o in collect_category(objs, category)
        if int(o.get("w", 0)) > 0
    ]
    return sorted(items)


# ----- Seaquest: schedule lane spawn timers from first future appear -----


def _oracle_seaquest(env, state, traj, *, t0: int, t1: int):
    from oc_parity.translators.seaquest import _nearest_lane, _entity_direction

    consts = env.consts
    lane_ys = np.asarray(consts.SPAWN_POSITIONS_Y, dtype=np.float64)
    sp = state.spawn_state
    timers = np.array(sp.spawn_timers, copy=True, dtype=np.int32)
    prev_sub = np.array(sp.prev_sub, copy=True)

    # For each lane, find first frame after t0 where a new Shark/Sub appears
    # that wasn't present at t0 (by nearest-lane bucket count increase).
    def _lane_counts(objs):
        sharks = collect_category(objs, "Shark")
        subs = collect_category(objs, "Submarine") or collect_category(objs, "Sub")
        counts = np.zeros(4, dtype=np.int32)
        kinds = [0] * 4  # 0 shark, 1 sub
        dirs = [1] * 4
        for o in sharks:
            lane = _nearest_lane(float(o.get("y", 0)), lane_ys)
            counts[lane] += 1
            dirs[lane] = _entity_direction(o, 1)
            kinds[lane] = 0
        for o in subs:
            lane = _nearest_lane(float(o.get("y", 0)), lane_ys)
            counts[lane] += 1
            dirs[lane] = _entity_direction(o, 1)
            kinds[lane] = 1
        return counts, kinds, dirs

    base_objs = _objs_at(traj, t0)
    base_counts, _, _ = _lane_counts(base_objs)
    first_kind = np.full(4, -1, dtype=np.int32)  # 0 shark, 1 sub
    for t in range(t0 + 1, t1):
        counts, kinds, dirs = _lane_counts(_objs_at(traj, t))
        for lane in range(4):
            if counts[lane] > base_counts[lane] and timers[lane] >= 9990:
                # First new spawn in this lane relative to inject.
                dt = t - t0
                timers[lane] = max(1, int(dt))
                prev_sub[lane] = 1 if kinds[lane] == 1 else 0
                first_kind[lane] = 1 if kinds[lane] == 1 else 0
        # Only schedule once per lane from first appearance.
        for lane in range(4):
            if counts[lane] > base_counts[lane]:
                base_counts[lane] = counts[lane]

    def _face_dir_for_lane_after(t_appear: int, lane: int) -> int:
        """Infer FACE dir from OC motion a few frames after appear (spawn dx often 0)."""
        xs = []
        for tt in range(t_appear, min(t_appear + 8, t1)):
            objs = _objs_at(traj, tt)
            for o in collect_category(objs, "Shark") + (
                collect_category(objs, "Submarine") or collect_category(objs, "Sub")
            ):
                if _nearest_lane(float(o.get("y", 0)), lane_ys) == lane:
                    xs.append(float(o.get("x", 0)))
                    dx = float(o.get("dx", 0))
                    if dx < 0:
                        return -1
                    if dx > 0:
                        return 1
                    break
        if len(xs) >= 2 and xs[-1] != xs[0]:
            return -1 if xs[-1] < xs[0] else 1
        return 1

    lane_dirs = np.array(getattr(sp, "lane_directions", np.ones(4)), copy=True)
    # Re-scan for direction of first future spawn per lane.
    # JAX lane_directions: 1 = moving left, 0 = moving right (NOT ±1 facing).
    base_counts, _, _ = _lane_counts(base_objs)
    diver_flags = np.full(4, 2, dtype=np.int32)  # 2 = disabled (see translator)
    for t in range(t0 + 1, t1):
        counts, kinds, dirs = _lane_counts(_objs_at(traj, t))
        # Also check nearby frames for escort divers (±5).
        divers_near = []
        for tt in range(max(t0, t - 2), min(t1, t + 6)):
            divers_near.extend(collect_category(_objs_at(traj, tt), "Diver"))
        for lane in range(4):
            if counts[lane] > base_counts[lane]:
                face = _face_dir_for_lane_after(t, lane)
                lane_dirs[lane] = 1 if face < 0 else 0
                for d in divers_near:
                    if _nearest_lane(float(d.get("y", 0)), lane_ys) == lane:
                        diver_flags[lane] = 1
                        break
                base_counts[lane] = counts[lane]

    survived = np.array(sp.survived, copy=True)
    # Force lanes whose first observed future spawn is Submarine to initialize
    # as subs (JAX initialize_new_spawn_cycle otherwise tends to emit shark first).
    for lane in range(4):
        if first_kind[lane] == 1:
            prev_sub[lane] = 0
            survived[lane * 3 : lane * 3 + 3] = np.array([0, 0, 1], dtype=survived.dtype)

    new_sp = sp.replace(
        spawn_timers=np.asarray(timers, dtype=np.dtype(np.asarray(sp.spawn_timers).dtype)),
        prev_sub=np.asarray(prev_sub, dtype=np.dtype(np.asarray(sp.prev_sub).dtype)),
        survived=np.asarray(survived, dtype=np.dtype(np.asarray(sp.survived).dtype)),
        diver_array=np.asarray(
            diver_flags, dtype=np.dtype(np.asarray(sp.diver_array).dtype)
        ),
    )
    if hasattr(sp, "lane_directions"):
        new_sp = new_sp.replace(
            lane_directions=np.asarray(
                lane_dirs, dtype=np.dtype(np.asarray(sp.lane_directions).dtype)
            )
        )
    return state.replace(spawn_state=new_sp)


# ----- Enduro: overlay future cars into visible slots / soften density -----


def _oracle_enduro(env, state, traj, *, t0: int, t1: int):
    # Peek a few frames ahead and merge unique cars (by rounded y-band) into slots.
    cars_all = []
    for t in range(t0, min(t1, t0 + 60)):
        objs = _objs_at(traj, t)
        for c in collect_category(objs, "Car"):
            if int(c.get("w", 0)) > 0:
                cars_all.append(c)
    if not cars_all:
        return state
    # Prefer most recent near cars.
    cars = sorted(cars_all, key=lambda o: -float(o.get("y", 0)))
    # Dedup by y bucket.
    seen = set()
    uniq = []
    for c in cars:
        key = int(round(float(c.get("y", 0)) / 4.0))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= 7:
            break
    vis = np.array(state.visible_opponent_positions, copy=True)
    vis[:, 0] = -1
    for i, c in enumerate(uniq[: vis.shape[0]]):
        vis[i, 0] = float(c["x"])
        vis[i, 1] = float(c["y"])
    # Freeze density advancement slightly so overlaid cars persist longer.
    return state.replace(
        visible_opponent_positions=vis.astype(
            np.dtype(np.asarray(state.visible_opponent_positions).dtype)
        ),
    )


# ----- Asteroids: when a new wave appears in lookahead, pack those rocks -----


def _oracle_asteroids(env, state, traj, *, t0: int, t1: int):
    from oc_parity.translators.asteroids import _size_from_wh, _color_from_rgb

    base = _entity_signature(_objs_at(traj, t0), "Asteroid")
    best = None
    for t in range(t0 + 1, t1):
        sig = _entity_signature(_objs_at(traj, t), "Asteroid")
        if len(sig) > len(base) + 1:
            best = (t, sig)
            break
    if best is None:
        return state
    t_wave, _ = best
    rocks = sorted(
        [
            a
            for a in collect_category(_objs_at(traj, t_wave), "Asteroid")
            if int(a.get("w", 0)) > 0
        ],
        key=lambda o: (float(o.get("y", 0)), float(o.get("x", 0))),
    )
    ast = np.array(state.asteroid_states, copy=True)
    # Keep current rocks; if wave is mostly new, replace inactive slots.
    inactive = [i for i in range(ast.shape[0]) if int(ast[i, 3]) == 0]
    for rock, idx in zip(rocks, inactive):
        ast[idx, 0] = int(rock["x"])
        ast[idx, 1] = int(rock["y"])
        ast[idx, 2] = 0
        ast[idx, 3] = _size_from_wh(rock.get("w", 0), rock.get("h", 0), env.consts)
        ast[idx, 4] = _color_from_rgb(rock.get("rgb"))
    return state.replace(asteroid_states=ast.astype(np.dtype(np.asarray(state.asteroid_states).dtype)))


# ----- BankHeist: set police directions from first future Δ -----


def _oracle_bankheist(env, state, traj, *, t0: int, t1: int):
    police = collect_category(_objs_at(traj, t0), "Police") or collect_category(
        _objs_at(traj, t0), "Enemy"
    )
    if not police:
        # Try Car-like police categories used by OC BankHeist.
        police = [
            o
            for o in _objs_at(traj, t0)
            if "police" in str(o.get("category", "")).lower()
            or str(o.get("category", "")) == "EnemyCar"
        ]
    # Directions already set by translator; lookahead: if a police turns soon, bake dir.
    ep = getattr(state, "enemy_positions", None)
    if ep is None:
        return state
    # enemy_positions may be a struct with position/direction arrays.
    try:
        dirs = np.array(ep.direction, copy=True)
        pos = np.array(ep.position, copy=True)
    except Exception:
        return state

    for t in range(t0 + 1, min(t1, t0 + 30)):
        objs = _objs_at(traj, t)
        foes = collect_category(objs, "Police") or collect_category(objs, "Enemy")
        for i, foe in enumerate(foes[: dirs.shape[0] if dirs.ndim else 0]):
            dx = float(foe.get("dx", 0))
            dy = float(foe.get("dy", 0))
            if dx == 0 and dy == 0:
                dx = float(foe.get("x", 0)) - float(foe.get("prev_x", foe.get("x", 0)))
                dy = float(foe.get("y", 0)) - float(foe.get("prev_y", foe.get("y", 0)))
            if abs(dx) + abs(dy) < 0.5:
                continue
            # Map to discrete dirs if array of ints.
            if abs(dx) >= abs(dy):
                d = 1 if dx > 0 else 3  # common RIGHT/LEFT encoding — leave if unknown
            else:
                d = 2 if dy > 0 else 0
            try:
                dirs[i] = d
            except Exception:
                pass
            break
    try:
        new_ep = ep.replace(direction=dirs.astype(np.dtype(np.asarray(ep.direction).dtype)))
        return state.replace(enemy_positions=new_ep)
    except Exception:
        return state


# ----- Venture: monster dx/dy from first future motion -----


def _oracle_venture(env, state, traj, *, t0: int, t1: int):
    monsters = getattr(state, "monsters", None)
    if monsters is None:
        return state
    try:
        dx = np.array(monsters.dx, copy=True)
        dy = np.array(monsters.dy, copy=True)
    except Exception:
        return state

    def _mons(objs):
        out = []
        for cat in ("Monster", "Hallmonster", "Goblin", "Enemy"):
            out.extend(collect_category(objs, cat))
        return [o for o in out if int(o.get("w", 0)) > 0]

    for t in range(t0, min(t1, t0 + 40)):
        mons = _mons(_objs_at(traj, t))
        changed = False
        for i, m in enumerate(mons[: dx.shape[0]]):
            mdx = float(m.get("dx", 0))
            mdy = float(m.get("dy", 0))
            if mdx == 0 and mdy == 0:
                mdx = float(m.get("x", 0)) - float(m.get("prev_x", m.get("x", 0)))
                mdy = float(m.get("y", 0)) - float(m.get("prev_y", m.get("y", 0)))
            if abs(mdx) + abs(mdy) < 0.5:
                continue
            n = (mdx ** 2 + mdy ** 2) ** 0.5 or 1.0
            dx[i] = mdx / n
            dy[i] = mdy / n
            changed = True
        if changed:
            break
    try:
        new_m = monsters.replace(
            dx=dx.astype(np.dtype(np.asarray(monsters.dx).dtype)),
            dy=dy.astype(np.dtype(np.asarray(monsters.dy).dtype)),
        )
        return state.replace(monsters=new_m)
    except Exception:
        return state


# ----- BeamRider: schedule white UFO from first future Saucer -----


def _oracle_beamrider(env, state, traj, *, t0: int, t1: int):
    level = state.level
    base = [
        s
        for s in collect_category(_objs_at(traj, t0), "Saucer")
        if int(s.get("w", 0)) >= 2
    ]
    appear_t = None
    saucer = None
    for t in range(t0 + 1, t1):
        saucers = [
            s
            for s in collect_category(_objs_at(traj, t), "Saucer")
            if int(s.get("w", 0)) >= 2
        ]
        if len(saucers) > len(base):
            appear_t = t
            saucer = saucers[0]
            break
        if not base and saucers:
            appear_t = t
            saucer = saucers[0]
            break
    if appear_t is None or saucer is None:
        return state
    dt = max(1, appear_t - t0)
    try:
        ufo = np.array(level.white_ufo_pos, copy=True)
        # Shape varies; try set xy.
        if ufo.ndim == 1 and ufo.shape[0] >= 2:
            ufo[0] = float(saucer["x"])
            ufo[1] = float(saucer["y"])
        elif ufo.ndim == 2:
            ufo[0, 0] = float(saucer["x"])
            ufo[0, 1] = float(saucer["y"])
        new_level = level.replace(
            white_ufo_spawn_delay=np.asarray(dt).astype(
                np.dtype(np.asarray(level.white_ufo_spawn_delay).dtype)
            )
            if hasattr(level, "white_ufo_spawn_delay")
            else level.white_ufo_spawn_delay,
            white_ufo_pos=ufo.astype(np.dtype(np.asarray(level.white_ufo_pos).dtype)),
        )
        return state.replace(level=new_level)
    except Exception:
        return state


# ----- Phoenix: arm dive when lookahead shows bird descending -----


def _oracle_phoenix(env, state, traj, *, t0: int, t1: int):
    # If a bird/enemy y increases substantially soon, arm attack flags.
    def _birds(objs):
        out = []
        for cat in ("Phoenix", "Bird", "Enemy", "Bat"):
            out.extend(collect_category(objs, cat))
        return [o for o in out if int(o.get("w", 0)) > 0]

    base = _birds(_objs_at(traj, t0))
    dive = None
    for t in range(t0 + 1, min(t1, t0 + 90)):
        cur = _birds(_objs_at(traj, t))
        for b in cur:
            by = float(b.get("y", 0))
            # Match to nearest base bird by x.
            if not base:
                if by > 80:
                    dive = b
                    break
                continue
            nearest = min(base, key=lambda o: abs(float(o.get("x", 0)) - float(b.get("x", 0))))
            if by - float(nearest.get("y", 0)) >= 8:
                dive = b
                break
        if dive is not None:
            break
    if dive is None:
        return state
    updates = {}
    if hasattr(state, "phoenix_do_attack"):
        try:
            pda = np.array(state.phoenix_do_attack, copy=True)
            pda[:] = True
            updates["phoenix_do_attack"] = pda.astype(
                np.dtype(np.asarray(state.phoenix_do_attack).dtype)
            )
        except Exception:
            pass
    if hasattr(state, "phoenix_attack_target_y"):
        try:
            pty = np.array(state.phoenix_attack_target_y, copy=True)
            pty[:] = float(dive.get("y", 100))
            updates["phoenix_attack_target_y"] = pty.astype(
                np.dtype(np.asarray(state.phoenix_attack_target_y).dtype)
            )
        except Exception:
            pass
    if hasattr(state, "bat_dive_timer"):
        try:
            # Low timer → allow dive soon.
            updates["bat_dive_timer"] = np.asarray(state.bat_dive_timer) * 0 + 1
            updates["bat_dive_timer"] = updates["bat_dive_timer"].astype(
                np.dtype(np.asarray(state.bat_dive_timer).dtype)
            )
        except Exception:
            pass
    if updates:
        return state.replace(**updates)
    return state


_ORACLES: Dict[str, OracleFn] = {
    "seaquest": _oracle_seaquest,
    "enduro": _oracle_enduro,
    "asteroids": _oracle_asteroids,
    "bankheist": _oracle_bankheist,
    "venture": _oracle_venture,
    "beamrider": _oracle_beamrider,
    "phoenix": _oracle_phoenix,
}
