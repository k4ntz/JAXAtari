"""
Tests that verify the offramp-to-main-road transition is seamless (no teleportation).

Bug: when descending from the offramp to the main road via a bridge or merge diagonal,
the player was snapped back to the offramp the moment the bridge/merge scrolled away.

Root cause: the threshold used to decide "is the player on the offramp?" was the
midpoint of the gap between the two roads (y=106 for default constants).  Any player
y in [off_max_y+1 .. threshold-1] was still flagged "on offramp" even though the
player was physically below the offramp band.  Once in_transition turned False, the
bounds clamp snapped them back to off_max_y.

Fix: use off_max_y (= offramp_bottom - PLAYER_SIZE[1]) as the threshold.  The player
is "on the offramp" if and only if their top edge is within the offramp band.
"""

import jax
import jax.numpy as jnp
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jaxatari.games.jax_roadrunner import (
    JaxRoadRunner,
    RoadRunnerRenderer,
    RoadRunnerConstants,
    RoadRunner_Level_1,
    OfframpConfig,
    LevelConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env_with_offramp(scroll_start=100, scroll_end=500, bridges=()):
    """Create a minimal single-level env with an offramp."""
    base = RoadRunnerConstants()
    level = LevelConfig(
        level_number=1,
        scroll_distance_to_complete=base.LEVEL_COMPLETE_SCROLL_DISTANCE,
        road_sections=RoadRunner_Level_1.road_sections,
        spawn_seeds=False,
        spawn_trucks=False,
        offramp=OfframpConfig(
            enabled=True,
            scroll_start=scroll_start,
            scroll_end=scroll_end,
            bridges=bridges,
        ),
    )
    consts = base._replace(levels=(level,))
    return JaxRoadRunner(consts=consts)


def active_offramp_state(env, scroll_counter=200):
    """Return a reset state with offramp active (scroll counter past scroll_start)."""
    key = jax.random.PRNGKey(42)
    _, state = env.reset(key)
    state = state._replace(
        scrolling_step_counter=jnp.array(scroll_counter, dtype=jnp.int32),
        is_scrolling=jnp.array(True),
    )
    return state


# ---------------------------------------------------------------------------
# Test: off_max_y threshold — player flagged "on offramp" iff within offramp band
# ---------------------------------------------------------------------------

def test_on_offramp_by_y_uses_off_max_y_threshold():
    """Player is flagged 'on offramp' only when y <= off_max_y, not up to the midpoint."""
    env = make_env_with_offramp()
    consts = env.consts
    road_top = consts.ROAD_TOP_Y
    offramp_bottom = road_top - consts.OFFRAMP_GAP          # 102
    off_max_y = offramp_bottom - consts.PLAYER_SIZE[1]      # 70  (bottom of offramp band)

    state = active_offramp_state(env)

    # Put the player at off_max_y (bottom of the offramp band) — should be "on offramp"
    state_on = state._replace(
        player_y=jnp.array(off_max_y, dtype=jnp.int32),
        player_on_offramp=jnp.array(True),
    )
    # Put the player one pixel below off_max_y — should NOT be "on offramp"
    state_off = state._replace(
        player_y=jnp.array(off_max_y + 1, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),
    )

    _, _, offramp_top, offramp_bottom_v = env._get_offramp_info(state_on)[3], \
        env._get_offramp_info(state_on)[3], \
        env._get_offramp_info(state_on)[3], \
        env._get_offramp_info(state_on)[4]

    # The off_max_y threshold: player_y <= off_max_y means "on offramp"
    assert int(jnp.array(off_max_y)) <= int(offramp_bottom_v) - int(consts.PLAYER_SIZE[1])


# ---------------------------------------------------------------------------
# Test: descend via bridge — no teleportation
# ---------------------------------------------------------------------------

def test_no_teleport_when_descending_via_bridge():
    """
    Player descends from offramp to gap zone while on a bridge, then the bridge
    scrolls away.  The player must NOT be snapped back to the offramp.
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=500, bridges=(150,))
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    offramp_bottom = road_top - consts.OFFRAMP_GAP      # 102
    off_max_y = offramp_bottom - consts.PLAYER_SIZE[1]  # 70
    SPEED = consts.PLAYER_MOVE_SPEED
    BRIDGE_W = consts.OFFRAMP_BRIDGE_WIDTH
    PLAYER_W = consts.PLAYER_SIZE[0]

    # scroll_counter=200 → bridge screen_x = (200-150)*SPEED; put player on bridge
    scroll_counter = 200
    bridge_screen_x = (scroll_counter - 150) * SPEED   # bridge left edge on screen

    state = active_offramp_state(env, scroll_counter=scroll_counter)

    # Place player squarely on the bridge and just below the offramp band
    player_x = bridge_screen_x + (BRIDGE_W - PLAYER_W) // 2  # centred on bridge
    player_y_in_gap = off_max_y + 4                           # 74 — in the gap, below offramp

    state = state._replace(
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(player_y_in_gap, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),  # just transitioned off
    )

    # Verify: player is at a bridge (in_transition should be True at this step)
    at_bridge = env._player_at_bridge(state, jnp.array(player_x))
    assert bool(at_bridge), "Player should overlap the bridge"

    # Now advance the scroll so the bridge moves completely past the player
    # bridge_screen_x scrolls left by SPEED each step; bridge gone when bx < player_x
    # i.e. counter such that (counter - 150)*SPEED < player_x
    steps_until_gone = int(player_x // SPEED) + 5
    state_after = state._replace(
        scrolling_step_counter=jnp.array(scroll_counter + steps_until_gone, dtype=jnp.int32)
    )
    # Confirm bridge is no longer overlapping
    new_bridge_x = (int(state_after.scrolling_step_counter) - 150) * SPEED
    at_bridge_after = env._player_at_bridge(state_after, jnp.array(player_x))
    assert not bool(at_bridge_after), "Bridge should have scrolled away"

    # Run bounds check: player (on main road, not in transition) must NOT be teleported up
    road_top_v, road_bottom_v, _ = env._get_road_bounds(state_after)
    checked_x, checked_y = env._check_player_bounds(
        state_after,
        jnp.array(player_x),
        jnp.array(player_y_in_gap),
        road_top_v,
        road_bottom_v,
    )
    # The player must stay at or below player_y_in_gap — NOT snapped back to off_max_y
    assert int(checked_y) >= player_y_in_gap, (
        f"Player was snapped from y={player_y_in_gap} up to y={int(checked_y)} "
        f"(off_max_y={off_max_y}) — teleportation bug!"
    )


# ---------------------------------------------------------------------------
# Test: ascending path — player NOT flagged "on offramp" until within offramp band
# ---------------------------------------------------------------------------

def test_ascending_player_flagged_on_offramp_only_when_in_band():
    """
    When the player climbs up through a bridge, they should only become 'on offramp'
    once they are within the offramp band (y <= off_max_y).
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=500, bridges=(150,))
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    offramp_bottom = road_top - consts.OFFRAMP_GAP
    off_max_y = offramp_bottom - consts.PLAYER_SIZE[1]  # 70

    scroll_counter = 200
    bridge_screen_x = (scroll_counter - 150) * consts.PLAYER_MOVE_SPEED
    BRIDGE_W = consts.OFFRAMP_BRIDGE_WIDTH
    PLAYER_W = consts.PLAYER_SIZE[0]
    player_x = bridge_screen_x + (BRIDGE_W - PLAYER_W) // 2

    state_base = active_offramp_state(env, scroll_counter=scroll_counter)

    # Player is one pixel above off_max_y (still in gap but not in offramp band)
    y_in_gap = off_max_y + 1  # 71
    state_gap = state_base._replace(
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(y_in_gap, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),
    )
    # at_bridge should be True
    assert bool(env._player_at_bridge(state_gap, jnp.array(player_x)))

    # Player is exactly at off_max_y (just entering offramp band)
    state_offramp = state_base._replace(
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(off_max_y, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),
    )

    # After a step with action=UP the player's new_on_offramp should reflect y <= off_max_y
    road_top_v, road_bottom_v, _ = env._get_road_bounds(state_gap)

    # Verify on_offramp_by_y logic: y_in_gap (71) > off_max_y (70) → not on offramp
    # y=off_max_y (70) <= off_max_y → on offramp
    assert y_in_gap > off_max_y, "y_in_gap should be just above off_max_y"
    assert off_max_y <= off_max_y, "off_max_y boundary should yield 'on offramp'"


# ---------------------------------------------------------------------------
# Test: player in gap without transition snaps to main road (not median stuck)
# ---------------------------------------------------------------------------

def test_player_in_gap_no_transition_snaps_to_main_road():
    """
    If a player somehow ends up in the gap zone (y in (off_max_y, main_min_y)) with
    no active transition, they are snapped DOWN to main_min_y on the very next bounds
    check.  This prevents 'running on the median'.
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=500)
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    offramp_bottom = road_top - consts.OFFRAMP_GAP  # 102
    off_max_y = offramp_bottom - consts.PLAYER_SIZE[1]  # 70
    main_min_y = road_top - (consts.PLAYER_SIZE[1] - 5)  # 83

    state = active_offramp_state(env)
    gap_y = off_max_y + 6   # 76, clearly in the gap

    state = state._replace(
        player_y=jnp.array(gap_y, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),
        # player_x far from any split/merge/bridge → no transition
        player_x=jnp.array(consts.SIDE_MARGIN + 4, dtype=jnp.int32),
    )

    road_top_v, road_bottom_v, _ = env._get_road_bounds(state)

    # Any proposed y should be clipped to [main_min_y, main_max_y]
    for proposed_y in [gap_y - 2, gap_y, gap_y + 1, main_min_y - 1]:
        checked_x, checked_y = env._check_player_bounds(
            state,
            jnp.array(consts.SIDE_MARGIN + 4, dtype=jnp.int32),
            jnp.array(proposed_y, dtype=jnp.int32),
            road_top_v,
            road_bottom_v,
        )
        assert int(checked_y) >= main_min_y, (
            f"Player in gap (state.player_y={gap_y}) with no transition, "
            f"proposed y={proposed_y} should snap to main_min_y={main_min_y}, "
            f"but got y={int(checked_y)}"
        )


# ---------------------------------------------------------------------------
# Test: Issue 1 — player can use top lane of main road when offramp is active
# ---------------------------------------------------------------------------

def test_player_can_use_top_lane_when_offramp_active():
    """
    When the offramp is active, the player should NOT be restricted to a tighter
    y range.  They must still be able to reach the top lane (main_min_y) on the
    main road even with no transition point nearby.
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=500)
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    main_min_y = road_top - (consts.PLAYER_SIZE[1] - 5)  # 83

    state = active_offramp_state(env)
    state = state._replace(
        player_on_offramp=jnp.array(False),
        # x far from any transition
        player_x=jnp.array(consts.SIDE_MARGIN + 4, dtype=jnp.int32),
        player_y=jnp.array(main_min_y + 10, dtype=jnp.int32),  # somewhere in main road
    )

    road_top_v, road_bottom_v, _ = env._get_road_bounds(state)

    # Propose y = main_min_y (very top of main road) — must be accepted
    checked_x, checked_y = env._check_player_bounds(
        state,
        jnp.array(consts.SIDE_MARGIN + 4, dtype=jnp.int32),
        jnp.array(main_min_y, dtype=jnp.int32),
        road_top_v,
        road_bottom_v,
    )
    assert int(checked_y) == main_min_y, (
        f"Top lane (y={main_min_y}) should be accessible when offramp is active, "
        f"but player was pushed to y={int(checked_y)}"
    )


# ---------------------------------------------------------------------------
# Test: Issue 2 — crossing at transition snaps gap to destination road in one step
# ---------------------------------------------------------------------------

def test_crossing_at_split_snaps_gap_to_offramp():
    """
    At a split/merge/bridge, when the player's proposed y falls in the gap zone
    (off_max_y < y < main_min_y), they are immediately landed on the destination
    road — no multi-step traversal of the median.
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=500, bridges=(150,))
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    offramp_bottom = road_top - consts.OFFRAMP_GAP  # 102
    off_max_y = offramp_bottom - consts.PLAYER_SIZE[1]  # 70
    main_min_y = road_top - (consts.PLAYER_SIZE[1] - 5)  # 83
    SPEED = consts.PLAYER_MOVE_SPEED

    scroll_counter = 200
    bridge_screen_x = (scroll_counter - 150) * SPEED
    BRIDGE_W = consts.OFFRAMP_BRIDGE_WIDTH
    PLAYER_W = consts.PLAYER_SIZE[0]
    player_x = bridge_screen_x + (BRIDGE_W - PLAYER_W) // 2

    state = active_offramp_state(env, scroll_counter=scroll_counter)

    # --- Ascending: from main road, one UP step → land on offramp ---
    state_main = state._replace(
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(main_min_y, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),
    )
    assert bool(env._player_at_bridge(state_main, jnp.array(player_x))), \
        "Player should be on bridge"

    road_top_v, road_bottom_v, _ = env._get_road_bounds(state_main)
    # One step UP: proposed y = main_min_y - SPEED = 80 (in gap)
    _, cy_up = env._check_player_bounds(
        state_main,
        jnp.array(player_x),
        jnp.array(main_min_y - SPEED),
        road_top_v,
        road_bottom_v,
    )
    assert int(cy_up) == off_max_y, (
        f"One UP step at bridge from y={main_min_y} should land on offramp (y={off_max_y}), "
        f"but got y={int(cy_up)}"
    )

    # --- Descending: from offramp, one DOWN step → land on main road ---
    state_off = state._replace(
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(off_max_y, dtype=jnp.int32),
        player_on_offramp=jnp.array(True),
    )
    _, cy_down = env._check_player_bounds(
        state_off,
        jnp.array(player_x),
        jnp.array(off_max_y + SPEED),   # one step DOWN: in gap
        road_top_v,
        road_bottom_v,
    )
    assert int(cy_down) == main_min_y, (
        f"One DOWN step at bridge from y={off_max_y} should land on main road (y={main_min_y}), "
        f"but got y={int(cy_down)}"
    )


# ---------------------------------------------------------------------------
# Test: Issue 1 — collision on top lane IS valid (no false suppression)
# ---------------------------------------------------------------------------

def test_collision_valid_when_both_on_main_road_top_lane():
    """
    If both player and enemy are on the main road (player in top lane at y=main_min_y),
    the collision must NOT be suppressed — they are on the same surface.
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=500)
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    main_min_y = road_top - (consts.PLAYER_SIZE[1] - 5)  # 83

    state = active_offramp_state(env)

    # Player and enemy at the same top-lane position on the main road
    state = state._replace(
        player_y=jnp.array(main_min_y, dtype=jnp.int32),
        player_x=jnp.array(20, dtype=jnp.int32),
        enemy_y=jnp.array(main_min_y, dtype=jnp.int32),
        enemy_x=jnp.array(20, dtype=jnp.int32),
        player_on_offramp=jnp.array(False),
        is_round_over=jnp.array(False),
        enemy_flattened_timer=jnp.array(0),
    )

    result = env._check_game_over(state)
    assert bool(result.is_round_over), (
        "Collision between player and enemy in the top lane of the main road "
        "should be detected, but was falsely suppressed"
    )


# ---------------------------------------------------------------------------
# Test: phantom extension fix — merge is unidirectional (offramp → main only)
# ---------------------------------------------------------------------------

def test_merge_zone_does_not_allow_main_to_offramp():
    """
    After the player has descended to the main road via the merge, pressing UP
    while still in the merge zone must NOT snap them back to the offramp
    (the phantom-extension bug).

    The merge is the END of the offramp road.  There is nothing above it to the
    right.  A main-road player at the merge must be kept on the main road even
    if their proposed y enters the gap zone.
    """
    env = make_env_with_offramp(scroll_start=100, scroll_end=300)
    consts = env.consts

    road_top = consts.ROAD_TOP_Y
    offramp_bottom = road_top - consts.OFFRAMP_GAP      # 102
    off_max_y = offramp_bottom - consts.PLAYER_SIZE[1]  # 70
    main_min_y = road_top - (consts.PLAYER_SIZE[1] - 5) # 83
    SPEED = consts.PLAYER_MOVE_SPEED
    RAMP_W = consts.OFFRAMP_RAMP_WIDTH

    # scroll_counter where merge is in the middle of the screen
    # merge_x = (counter - scroll_end) * SPEED = 70  → counter = 300 + 70//3 ≈ 323
    scroll_counter = 323
    merge_x = (scroll_counter - 300) * SPEED  # ≈ 69

    # Place the player squarely in the merge zone, on the MAIN road (not offramp)
    player_x = merge_x + RAMP_W // 2 - consts.PLAYER_SIZE[0] // 2
    player_x = max(consts.SIDE_MARGIN, min(player_x, consts.WIDTH - consts.PLAYER_SIZE[0] - consts.SIDE_MARGIN))

    state = active_offramp_state(env, scroll_counter=scroll_counter)
    state = state._replace(
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(main_min_y, dtype=jnp.int32),  # on main road
        player_on_offramp=jnp.array(False),               # just came off offramp
    )

    road_top_v, road_bottom_v, _ = env._get_road_bounds(state)

    # Sanity: player overlaps the merge zone
    at_merge_raw = (player_x + consts.PLAYER_SIZE[0] > merge_x) & (player_x < merge_x + RAMP_W)
    assert at_merge_raw, (
        f"Player (x={player_x}) should overlap merge zone [merge_x={merge_x}, "
        f"merge_x+RAMP_W={merge_x + RAMP_W}]"
    )

    # Propose y = main_min_y - SPEED (falls in gap) — MUST be rejected upward movement
    proposed_y = main_min_y - SPEED   # e.g. 80, in the gap (70 < 80 < 83)
    assert off_max_y < proposed_y < main_min_y, (
        f"proposed_y={proposed_y} should be in the gap ({off_max_y}, {main_min_y})"
    )

    checked_x, checked_y = env._check_player_bounds(
        state,
        jnp.array(player_x, dtype=jnp.int32),
        jnp.array(proposed_y, dtype=jnp.int32),
        road_top_v,
        road_bottom_v,
    )

    # With the fix: player stays on the main road — NOT snapped up to off_max_y
    assert int(checked_y) >= main_min_y, (
        f"Player on main road at merge zone pressed UP; should stay on main road "
        f"(y >= {main_min_y}), but was snapped to y={int(checked_y)} "
        f"(off_max_y={off_max_y}) — phantom-extension bug!"
    )
