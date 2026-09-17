"""Translator registry: OC game name ↔ JAX make-key, disclaimers, skipped titles."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

# JAX make() key → OCAtari / ALE PascalCase name (from docs comparison map).
JAX_TO_OC_GAME: Dict[str, str] = {
    "asteroids": "Asteroids",
    "bankheist": "BankHeist",
    "beamrider": "BeamRider",
    "breakout": "Breakout",
    "enduro": "Enduro",
    "freeway": "Freeway",
    "frostbite": "Frostbite",
    "gravitar": "Gravitar",
    "kangaroo": "Kangaroo",
    "montezumarevenge": "MontezumaRevenge",
    "mspacman": "MsPacman",
    "phoenix": "Phoenix",
    "pong": "Pong",
    "qbert": "Qbert",
    "seaquest": "Seaquest",
    "skiing": "Skiing",
    "tennis": "Tennis",
    "venture": "Venture",
}

OC_TO_JAX_GAME: Dict[str, str] = {v.lower(): k for k, v in JAX_TO_OC_GAME.items()}
# Also accept already-normalized jax keys.
OC_TO_JAX_GAME.update({k: k for k in JAX_TO_OC_GAME})

# OCAtari gaps — revisit later (wrapper fixes / coverage).
SKIPPED_OC_GAMES: Dict[str, str] = {
    "gravitar": "Not covered yet by OCAtari",
}

# Printed whenever a translator forces map/level (or similar) assumptions.
TRANSLATOR_DISCLAIMERS: Dict[str, List[str]] = {
    "bankheist": [
        "ASSUMPTION: always inject into JAX map_id=0 / level=0 / difficulty=0 "
        "(do not switch cities in OC if you want a fair visual transfer).",
        "ASSUMPTION: speed/timers/city_states/portals left at reset defaults; "
        "player_move_direction from multi-frame XY lookback (coasts on NOOP / "
        "half-speed frames); facing + latched_action from OC stick; "
        "fuel scaled from Gas_Tank height.",
    ],
    "pong": [
        "ASSUMPTION: player/enemy analog speeds from clipped Δy lookback; "
        "PRNG key from reset; step_counter≥60 when a live Ball is present.",
    ],
    "freeway": [
        "ASSUMPTION: P1 Chicken only (leftmost / x≈44); P2 ignored.",
        "ASSUMPTION: cars sorted by ascending y into lanes; overlay x only "
        "(JAX lane y kept). lane_time/cooldown/time RESET.",
    ],
    "skiing": [
        "WARNING: LOW COMPATIBILITY — Skiing transfer is best-effort soft survival only.",
        "ASSUMPTION: OC Score/Clock .value often stuck at 0; step_count=frame_index; "
        "speeds from Δ lookback; unused tree/mogul/flag slots keep reset leftovers.",
    ],
    "tennis": [
        "ASSUMPTION: JAX (ball_x,ball_y)=OC BallShadow; ball_z=shadow.y−ball.y. "
        "In-play iff ground velocity |dx|/|dy|≥0.5 (z-bounce alone = serve).",
        "ASSUMPTION: mid-rally move/vz from OC Δ; scores RESET when .value missing.",
    ],
    "seaquest": [
        "ASSUMPTION: sharks AND Submarine packed by nearest lane Y; facing from dx, "
        "else prev-frame Δx in-lane, else orientation.",
        "ASSUMPTION: CollectedDiver HUD count → divers_collected; just_surfaced=0 when "
        "player Y>52 (avoid init oxygen refill); diver_array cleared unless OC shows escort.",
        "ASSUMPTION: spawn timers softened / oracle-scheduled; oxygen from OxygenBar.value.",
        "ASSUMPTION: missing OC Player (death blink) → JAX player_x=-100, death_counter=45 "
        "(hide phase); compare extractors omit hidden player.",
    ],
    "kangaroo": [
        "ASSUMPTION: force current_level=1; platform/ladder geometry kept from JAX.",
        "ASSUMPTION: mid-jump from gap above platform (OC has no jumping flag; "
        "dy often 0 mid-arc) → is_jumping+jump_base_y/counter; "
        "monkeys prefer vertical FSM (climb) over lateral walk.",
    ],
    "venture": [
        "ASSUMPTION: level inferred from OC — Hallmonsters ⇒ main map (level 0); "
        "Goblin/large player ⇒ room 1–4 via nearest chest anchor. world=1.",
        "ASSUMPTION: hall player Y += 8; room player uses OC bbox center. "
        "Monsters keep OC top-left XY; velocity prefers OC dx/dy, else prevΔ, "
        "else a nonzero unit direction (OC often reports dx=dy=0).",
    ],
    "mspacman": [
        "ASSUMPTION: OC→JAX actor Y shifted by +3 (reset-calibrated); X 1:1. "
        "OC Pill grid_ij is (row,col) → JAX pellets[col,row].",
        "ASSUMPTION: ghost modes/timers RESET; HUD Fruit ignored; lives use OC value as-is "
        "(possible off-by-one vs JAX). last_horiz_dir uses act_to_dir indices "
        "(RIGHT=1, LEFT=2), not Action enums.",
    ],
    "frostbite": [
        "ASSUMPTION: ice white/blue from FloatingBlock rgb; ice_directions from Δx. "
        "Igloo NOT transferred (stays -1) to avoid instant build.",
        "ASSUMPTION: bird dir from OC dx, else opposite of ice row below "
        "(nearest-row fallback mis-aimed the upper bird); score/temp/lives RESET.",
    ],
    "montezumarevenge": [
        "ASSUMPTION: force INITIAL_ROOM_ID (start room); keep JAX room geometry.",
        "ASSUMPTION: OC_y = JAX_y + 47 applied on inject; only player/Skull/Key/HUD overlaid.",
        "ASSUMPTION: is_climbing from OC Ladder overlap using player feet + mid-x (not "
        "player top); sets last_ladder + snaps X to JAX ladder center when climbing.",
        "ASSUMPTION: trajectories should stay in room one (key collect OK).",
    ],
    "beamrider": [
        "ASSUMPTION: skip blue-line intro (blue_line_counter past INIT table) so "
        "enemies are visible; steps≥2001 so shooting unlocks. score/lives/sector RESET.",
        "ASSUMPTION: OC only exposes Player+Saucer (often 2×2); player snapped to "
        "BOTTOM_OF_LANES. Soft survival mostly player-lane + saucer XY.",
    ],
    "breakout": [
        "ASSUMPTION: Block runs expand into JAX (6,18) grid via BLOCK_START/SIZE; "
        "ball vel from dx/dy signs; small_paddle/hit counters RESET.",
        "ASSUMPTION: Lives category may be 'Live' (singular); score/lives only if .value set.",
    ],
    "enduro": [
        "ASSUMPTION: overlay Player + Cars into visible_opponent_positions (near→far by y); "
        "base_opponents / track / density / collision FSM RESET.",
        "ASSUMPTION: next JAX step may rewrite car X from lane+track — soft survival only.",
    ],
    "asteroids": [
        "ASSUMPTION: player screen XY → subpixel via (s//2)*256; rocks packed ≤17 by (y,x); "
        "size from w×h; missing Player → respawn_timer hide.",
        "ASSUMPTION: rng_key / side_step / missile speeds RESET; OC may list >17 asteroids.",
    ],
    "phoenix": [
        "ASSUMPTION: force level=1; boss/bat FSM left at RESET.",
        "ASSUMPTION: birds packed by nearest formation slot; group dir from OC dx "
        "(default −1). RNG dives frozen (high cooldown) unless OC bird is clearly "
        "below formation home — then dive FSM armed from observed y/dy.",
        "ASSUMPTION: Lives = Life icon count; score from .value.",
    ],
    "qbert": [
        "ASSUMPTION: force level/round=1; pyramid colors RESET.",
        "ASSUMPTION: snap only on-screen sprites within 28px of a walkable cell.",
        "ASSUMPTION: enemies soft-frozen ~180 frames (green_ball_freeze_step); "
        "first post-freeze Purple/Sam/ball hop biased via PRNG to match OC Δ "
        "(DOWN vs RIGHT). Coily chase is deterministic (not RNG).",
    ],
}


def normalize_game_name(game: str) -> Tuple[str, str]:
    """Return ``(jax_make_key, oc_game_name)`` for a user-supplied game string."""
    raw = game.strip()
    key = raw.lower().replace(" ", "").replace("'", "").replace("_", "")
    # Common aliases
    aliases = {
        "mspacman": "mspacman",
        "mspacmman": "mspacman",
        "montezuma": "montezumarevenge",
        "montezumarevenge": "montezumarevenge",
        "bankheist": "bankheist",
        "bank_heist": "bankheist",
    }
    key = aliases.get(key, key)
    if key in SKIPPED_OC_GAMES:
        raise SystemExit(
            f"Game '{raw}' is temporarily skipped for OC→JAX overlay: "
            f"{SKIPPED_OC_GAMES[key]}"
        )
    if key not in JAX_TO_OC_GAME:
        # Try OC PascalCase lookup
        if key in OC_TO_JAX_GAME:
            jax_key = OC_TO_JAX_GAME[key]
        else:
            supported = ", ".join(
                sorted(k for k in JAX_TO_OC_GAME if k not in SKIPPED_OC_GAMES)
            )
            raise SystemExit(
                f"Unknown/unsupported game '{raw}'. Supported make-keys: {supported}"
            )
    else:
        jax_key = key
    return jax_key, JAX_TO_OC_GAME[jax_key]


def print_disclaimers(jax_key: str) -> None:
    notes = TRANSLATOR_DISCLAIMERS.get(jax_key, [])
    if not notes:
        print(f"[oc_parity] No special assumptions listed for '{jax_key}'.")
        return
    print(f"[oc_parity] Translator assumptions for '{jax_key}':")
    for line in notes:
        print(f"  - {line}")


def print_skipped_games_note() -> None:
    if not SKIPPED_OC_GAMES:
        return
    print("[oc_parity] Skipped OC games (fix wrappers later):")
    for k, reason in sorted(SKIPPED_OC_GAMES.items()):
        print(f"  - {k}: {reason}")


TranslatorFn = Callable[..., object]

_IMPLEMENTED = [
    "pong",
    "bankheist",
    "freeway",
    "skiing",
    "tennis",
    "seaquest",
    "kangaroo",
    "venture",
    "mspacman",
    "frostbite",
    "montezumarevenge",
    "beamrider",
    "phoenix",
    "qbert",
    "breakout",
    "enduro",
    "asteroids",
]


def get_translator(jax_key: str) -> TranslatorFn:
    """Return ``oc_frame_to_*_state(env, objects, **kwargs)`` for ``jax_key``."""
    if jax_key == "pong":
        from .pong import oc_frame_to_pong_state

        return oc_frame_to_pong_state
    if jax_key == "bankheist":
        from .bankheist import oc_frame_to_bankheist_state

        return oc_frame_to_bankheist_state
    if jax_key == "freeway":
        from .freeway import oc_frame_to_freeway_state

        return oc_frame_to_freeway_state
    if jax_key == "skiing":
        from .skiing import oc_frame_to_skiing_state

        return oc_frame_to_skiing_state
    if jax_key == "tennis":
        from .tennis import oc_frame_to_tennis_state

        return oc_frame_to_tennis_state
    if jax_key == "seaquest":
        from .seaquest import oc_frame_to_seaquest_state

        return oc_frame_to_seaquest_state
    if jax_key == "kangaroo":
        from .kangaroo import oc_frame_to_kangaroo_state

        return oc_frame_to_kangaroo_state
    if jax_key == "venture":
        from .venture import oc_frame_to_venture_state

        return oc_frame_to_venture_state
    if jax_key == "mspacman":
        from .mspacman import oc_frame_to_mspacman_state

        return oc_frame_to_mspacman_state
    if jax_key == "frostbite":
        from .frostbite import oc_frame_to_frostbite_state

        return oc_frame_to_frostbite_state
    if jax_key == "montezumarevenge":
        from .montezumarevenge import oc_frame_to_montezumarevenge_state

        return oc_frame_to_montezumarevenge_state
    if jax_key == "beamrider":
        from .beamrider import oc_frame_to_beamrider_state

        return oc_frame_to_beamrider_state
    if jax_key == "phoenix":
        from .phoenix import oc_frame_to_phoenix_state

        return oc_frame_to_phoenix_state
    if jax_key == "qbert":
        from .qbert import oc_frame_to_qbert_state

        return oc_frame_to_qbert_state
    if jax_key == "breakout":
        from .breakout import oc_frame_to_breakout_state

        return oc_frame_to_breakout_state
    if jax_key == "enduro":
        from .enduro import oc_frame_to_enduro_state

        return oc_frame_to_enduro_state
    if jax_key == "asteroids":
        from .asteroids import oc_frame_to_asteroids_state

        return oc_frame_to_asteroids_state
    raise SystemExit(
        f"No OC→JAX translator implemented yet for '{jax_key}'. "
        f"Implemented: {', '.join(list_implemented_translators())}."
    )


def list_implemented_translators() -> List[str]:
    return list(_IMPLEMENTED)
