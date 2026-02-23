"""Tests for the offramp feature in the Road Runner game."""

import jax
import jax.numpy as jnp
import pytest

from jaxatari.games.jax_roadrunner import (
    JaxRoadRunner,
    JaxRoadRunner as Env,
    OfframpConfig,
    LevelConfig,
    RoadRunnerConstants,
    DEFAULT_LEVELS,
    Action,
    MAX_OFFRAMP_BRIDGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(offramp_enabled: bool = True) -> JaxRoadRunner:
    """Create a Road Runner environment, optionally with the offramp active."""
    if offramp_enabled:
        return JaxRoadRunner()  # Level 1 already has an offramp
    # Create a copy of Level 1 without an offramp for control tests
    base_level = DEFAULT_LEVELS[0]
    no_ramp_level = base_level._replace(offramp=OfframpConfig(enabled=False))
    consts = RoadRunnerConstants(levels=(no_ramp_level, DEFAULT_LEVELS[1]))
    return JaxRoadRunner(consts)


def fast_forward_to_split(env: JaxRoadRunner, state, scroll: int = 225):
    """Teleport the scroll counter to a position where the player (x=70) is within the split sprite.

    With scroll=225, RAMP_W=24:  split_x = (225-200)*3 = 75, sprite at x=[51, 75].
    Player at x=70, width=8: right edge 78 > 51 AND left 70 < 75 → player is at the split.
    """
    return state._replace(scrolling_step_counter=jnp.array(scroll, dtype=jnp.int32))


def move_player_onto_offramp(env: JaxRoadRunner, state):
    """Press UP repeatedly until the player is on the offramp."""
    for _ in range(15):
        _, state, _, _, _ = env.step(state, Action.UP)
        if state.player_on_offramp:
            break
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOfframpConfig:
    """Verify the OfframpConfig is stored and accessible in LevelConfig."""

    def test_default_offramp_disabled(self):
        """A fresh LevelConfig should have no offramp by default."""
        level = DEFAULT_LEVELS[1]  # Level 2 has no offramp
        assert not level.offramp.enabled

    def test_level1_offramp_enabled(self):
        """Level 1 should have the offramp enabled."""
        level = DEFAULT_LEVELS[0]
        assert level.offramp.enabled
        assert level.offramp.scroll_start < level.offramp.scroll_end


class TestOfframpState:
    """Verify the player_on_offramp state field."""

    def test_initial_state_not_on_offramp(self):
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        assert not bool(state.player_on_offramp)

    def test_player_enters_offramp_at_split(self):
        """Player can transition to the offramp in the split zone."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        state = fast_forward_to_split(env, state)
        state = move_player_onto_offramp(env, state)
        assert bool(state.player_on_offramp), "Player should be on offramp after moving UP at split"

    def test_player_stays_on_offramp_outside_transition(self):
        """Once on the offramp, the player stays there between transitions."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        state = fast_forward_to_split(env, state)
        state = move_player_onto_offramp(env, state)
        assert state.player_on_offramp

        # Advance scroll counter well past split (so we're not in any transition zone)
        state = state._replace(scrolling_step_counter=jnp.array(280, dtype=jnp.int32))

        # Several NOOPs - should stay on offramp
        for _ in range(5):
            _, state, _, _, _ = env.step(state, Action.NOOP)
        assert bool(state.player_on_offramp), "Player should remain on offramp between transitions"

    def test_player_exits_offramp_at_merge(self):
        """Player can return to the main road in the merge zone."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        state = fast_forward_to_split(env, state)
        state = move_player_onto_offramp(env, state)
        assert state.player_on_offramp

        # Advance to a scroll position where the merge sprite is at the player's X (x=70).
        # With RAMP_W=24: need merge_x in (46, 78). scroll=720 → merge_x = (720-700)*3 = 60.
        # Player at x=70: right edge (70+8=78) > merge_x (60) AND left edge (70) < merge_x+RAMP_W (84) → overlap ✓
        state = state._replace(scrolling_step_counter=jnp.array(720, dtype=jnp.int32))

        # Move DOWN to descend back to the main road
        for _ in range(20):
            _, state, _, _, _ = env.step(state, Action.DOWN)
            if not state.player_on_offramp:
                break

        assert not bool(state.player_on_offramp), "Player should exit offramp at merge"

    def test_player_on_offramp_reset_on_life_loss(self):
        """player_on_offramp should be reset to False after losing a life."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        state = fast_forward_to_split(env, state)
        state = move_player_onto_offramp(env, state)
        assert state.player_on_offramp

        # Force a life loss via _next_life_reset
        reset_state = env._next_life_reset(state)
        assert not bool(reset_state.player_on_offramp)


class TestOfframpBounds:
    """Verify the Y-position constraints for each road."""

    def test_player_bounded_to_offramp_y_range(self):
        """When on the offramp (no transition), the player cannot go below offramp_bottom - player_height."""
        env = make_env()
        consts = env.consts
        offramp_bottom = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP
        expected_max_y = offramp_bottom - consts.PLAYER_SIZE[1]  # 102 - 32 = 70

        _, state = env.reset(jax.random.PRNGKey(0))
        state = fast_forward_to_split(env, state)
        state = move_player_onto_offramp(env, state)

        # Advance past the split transition
        state = state._replace(scrolling_step_counter=jnp.array(280, dtype=jnp.int32))

        # Move DOWN aggressively
        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.DOWN)

        assert int(state.player_y) <= expected_max_y, (
            f"player_y={state.player_y} should be <= {expected_max_y} when on offramp"
        )

    def test_player_bounded_to_main_road_when_not_on_offramp(self):
        """When NOT on the offramp (and not in transition), the player cannot go above the main road top."""
        env = make_env()
        consts = env.consts
        road_top = consts.ROAD_TOP_Y
        expected_min_y = road_top - (consts.PLAYER_SIZE[1] - 5)  # 110 - 27 = 83

        _, state = env.reset(jax.random.PRNGKey(0))
        # Stay at scroll counter=0 (no offramp active), press UP many times
        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.UP)

        assert int(state.player_y) >= expected_min_y, (
            f"player_y={state.player_y} should be >= {expected_min_y} when on main road (no offramp)"
        )

    def test_player_cannot_penetrate_offramp_from_below(self):
        """When the offramp is active above the player, the player cannot go above road_top."""
        env = make_env()
        consts = env.consts
        road_top = consts.ROAD_TOP_Y  # 110 — player's top must not go above this

        _, state = env.reset(jax.random.PRNGKey(0))
        # Offramp is active (scroll=300) but split and merge diagonals are off screen.
        # No transition zone is reachable at the player's default X (~70).
        state = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))

        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.UP)

        assert int(state.player_y) >= road_top, (
            f"player_y={state.player_y} should be >= road_top={road_top}; "
            "player must not penetrate the median/offramp from below"
        )
        assert not bool(state.player_on_offramp), "Player should not be on offramp"

    def test_player_cannot_cross_when_away_from_diagonal(self):
        """Player stays on their current road when the offramp is active but they are away from a diagonal."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts
        road_top = consts.ROAD_TOP_Y  # blocked at road_top when offramp is active

        # Offramp active at scroll=300, but split_x ≫ WIDTH and merge hasn't started.
        state = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))

        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.UP)

        assert int(state.player_y) >= road_top, (
            f"player_y={state.player_y} should not cross to offramp when away from diagonal"
        )
        assert not bool(state.player_on_offramp), "Player should not be on offramp"


class TestOfframpInactive:
    """Verify the offramp is inactive outside its scroll range."""

    def test_offramp_inactive_before_scroll_start(self):
        """Before scroll_start, the offramp should not be accessible."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        # scroll_start = 200; ensure we're before it
        state = state._replace(scrolling_step_counter=jnp.array(50, dtype=jnp.int32))

        offramp_active, split_x, _, _, _ = env._get_offramp_info(state)
        assert not bool(offramp_active)
        assert int(split_x) < 0, "Split sprite should not have appeared yet"

    def test_offramp_inactive_after_scroll_end(self):
        """After the merge exits the screen, the offramp should be inactive."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        # scroll_end = 700; merge exits at ~700 + (160+24)//3 = ~761.
        # Use a value well past the exit.
        state = state._replace(scrolling_step_counter=jnp.array(800, dtype=jnp.int32))

        offramp_active, _, _, _, _ = env._get_offramp_info(state)
        assert not bool(offramp_active)

    def test_no_offramp_access_when_disabled(self):
        """With an env without an offramp, player_on_offramp never becomes True."""
        env = make_env(offramp_enabled=False)
        _, state = env.reset(jax.random.PRNGKey(0))

        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.UP)
        assert not bool(state.player_on_offramp)


class TestOfframpRendering:
    """Verify the offramp renders correctly in different scroll positions."""

    def test_no_offramp_pixels_before_scroll_start(self):
        """Before the offramp is active, its Y region should contain no road (black) pixels."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        # scroll=0, no offramp
        frame = env.render(state)
        consts = env.consts
        offramp_top = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP - consts.OFFRAMP_HEIGHT
        offramp_bottom = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP
        region = frame[offramp_top:offramp_bottom, consts.SIDE_MARGIN:consts.WIDTH - consts.SIDE_MARGIN]
        import numpy as np
        black_pixels = int(np.sum(np.all(np.array(region) == [0, 0, 0], axis=2)))
        assert black_pixels == 0, f"Expected 0 black pixels in offramp region before scroll_start, got {black_pixels}"

    def test_offramp_pixels_when_active(self):
        """When the offramp is fully active, its Y region should contain road (black) pixels."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        # Advance well past split (scroll=300, split_x=300; > WIDTH so fully revealed)
        state = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))
        frame = env.render(state)
        consts = env.consts
        offramp_top = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP - consts.OFFRAMP_HEIGHT
        offramp_bottom = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP
        region = frame[offramp_top:offramp_bottom, consts.SIDE_MARGIN:consts.WIDTH - consts.SIDE_MARGIN]
        import numpy as np
        black_pixels = int(np.sum(np.all(np.array(region) == [0, 0, 0], axis=2)))
        assert black_pixels > 0, "Expected black pixels in offramp region when active"

    def test_split_transition_partial_offramp(self):
        """During the split transition, only part of the offramp Y region is filled."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        # scroll=215: split_x=45, sprite at [21,45], offramp only visible in x=[8, 45]
        state_full = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))
        state_partial = state._replace(scrolling_step_counter=jnp.array(215, dtype=jnp.int32))
        frame_full = env.render(state_full)
        frame_partial = env.render(state_partial)
        consts = env.consts
        offramp_top = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP - consts.OFFRAMP_HEIGHT
        offramp_bottom = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP
        import numpy as np
        def count_black(frame):
            region = frame[offramp_top:offramp_bottom, consts.SIDE_MARGIN:consts.WIDTH - consts.SIDE_MARGIN]
            return int(np.sum(np.all(np.array(region) == [0, 0, 0], axis=2)))
        assert count_black(frame_partial) < count_black(frame_full), (
            "Partial split should have fewer offramp pixels than fully-open offramp"
        )

    def test_split_sprite_bottom_row_fully_connected(self):
        """The bottom row of the split sprite must be all road (fully connected to main road)."""
        import numpy as np
        env = make_env()
        consts = env.consts
        # Build the split sprite via the renderer helper (before palette conversion)
        split_sprite = np.array(env.renderer._create_offramp_split_sprite())
        OFFRAMP_H = consts.OFFRAMP_HEIGHT
        GAP_H = consts.OFFRAMP_GAP
        last_row = split_sprite[OFFRAMP_H + GAP_H - 1, :, :]
        # Last row should be black (road color = 0,0,0) across all columns.
        is_black = np.all(last_row[:, :3] == 0, axis=1)
        assert np.all(is_black), (
            "Split sprite bottom row should be all road (black) so it connects "
            "fully to the main road"
        )


class TestOfframpCollision:
    """Verify enemy-player collision is suppressed across the median."""

    def test_no_collision_when_player_on_offramp(self):
        """Enemy should not catch player when player is on offramp and enemy on main road."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))

        # Get player onto the offramp
        state = fast_forward_to_split(env, state)
        state = move_player_onto_offramp(env, state)
        assert bool(state.player_on_offramp), "Setup: player should be on offramp"

        # Advance past the split so no diagonal is near the player
        state = state._replace(scrolling_step_counter=jnp.array(280, dtype=jnp.int32))

        # Place enemy at the very top of the main road (closest possible to offramp)
        consts = env.consts
        enemy_y_top = consts.ROAD_TOP_Y - (consts.PLAYER_SIZE[1] // 3)  # top of main road enemy range
        state = state._replace(
            enemy_x=jnp.array(state.player_x, dtype=jnp.int32),
            enemy_y=jnp.array(enemy_y_top, dtype=jnp.int32),
            enemy_flattened_timer=jnp.array(0, dtype=jnp.int32),
        )

        # Step once — should NOT trigger game over
        _, state, _, _, _ = env.step(state, Action.NOOP)
        assert not bool(state.is_round_over), (
            "Enemy at top of main road must not catch player on offramp"
        )


class TestOfframpNoPositionJump:
    """Verify the offramp activating does not abruptly move the player."""

    def test_player_not_pushed_when_offramp_activates(self):
        """A player in the top lane must not be snapped downward when the offramp appears."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts
        # Move player to the top of the normal main-road range (y~83) before
        # the offramp exists (scroll=0).
        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.UP)
        y_before = int(state.player_y)
        assert y_before < consts.ROAD_TOP_Y, (
            f"Setup: player should be above road_top ({consts.ROAD_TOP_Y}), got y={y_before}"
        )

        # Teleport to the offramp-active zone (scroll=300) and take a single NOOP step.
        state = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))
        _, state, _, _, _ = env.step(state, Action.NOOP)
        y_after = int(state.player_y)

        # Player should NOT have been pushed down by more than the normal step size.
        assert abs(y_after - y_before) <= consts.PLAYER_MOVE_SPEED, (
            f"Offramp activating pushed player from y={y_before} to y={y_after}; "
            "should not cause a sudden large position change"
        )


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------

class TestOfframpBridge:
    """Verify bridge configuration, physics, and rendering."""

    def _make_env_with_bridge(self, bridge_scroll: int = 450) -> JaxRoadRunner:
        """Create an env where Level 1 has a single bridge at the given scroll step."""
        base_level = DEFAULT_LEVELS[0]
        bridge_level = base_level._replace(
            offramp=OfframpConfig(
                enabled=True,
                scroll_start=200,
                scroll_end=700,
                bridges=(bridge_scroll,),
            )
        )
        consts = RoadRunnerConstants(levels=(bridge_level, DEFAULT_LEVELS[1]))
        return JaxRoadRunner(consts)

    def test_bridge_config_stored(self):
        """Bridge scroll steps must be preserved in the OfframpConfig."""
        env = make_env()  # Level 1 already has bridges=(450,)
        level = env.consts.levels[0]
        assert level.offramp.enabled
        assert len(level.offramp.bridges) >= 1

    def test_bridge_data_in_offramp_array(self):
        """Bridge scroll positions must be stored in the precomputed offramp data array."""
        from jaxatari.games.jax_roadrunner import MAX_OFFRAMP_BRIDGES
        env = self._make_env_with_bridge(450)
        # Row 0 = [enabled, scroll_start, scroll_end, bridge_0, bridge_1, ...]
        row = env._offramp_data[0]
        assert int(row[0]) == 1, "enabled flag"
        assert int(row[3]) == 450, "first bridge scroll step"
        # Remaining slots should be -1 (unused)
        for i in range(4, 3 + MAX_OFFRAMP_BRIDGES):
            assert int(row[i]) == -1, f"unused bridge slot {i} should be -1"

    def test_player_can_cross_at_bridge(self):
        """Player on the main road can move onto the offramp when standing at a bridge."""
        env = self._make_env_with_bridge(450)
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts

        # Teleport scroll so the bridge is at the player's X (x=70, width=8).
        # bridge_x = (counter - 450) * 3; we want bridge_x in (62, 78).
        # counter = 450 + 23 = 473 → bridge_x = 69 ✓
        state = state._replace(scrolling_step_counter=jnp.array(473, dtype=jnp.int32))

        # Confirm bridge is detected at the player's X
        at_bridge = env._player_at_bridge(state, jnp.array(70, dtype=jnp.int32))
        assert bool(at_bridge), "Player should be detected at bridge"

        # Player should be able to move UP onto the offramp
        for _ in range(15):
            _, state, _, _, _ = env.step(state, Action.UP)
            if state.player_on_offramp:
                break

        assert bool(state.player_on_offramp), "Player should be able to cross at bridge"

    def test_player_cannot_cross_away_from_bridge(self):
        """Player cannot cross the median at a position away from any bridge."""
        env = self._make_env_with_bridge(450)
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts

        # Offramp active (scroll=300) but bridge_x = (300-450)*3 = -450 (off screen)
        state = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))
        at_bridge = env._player_at_bridge(state, jnp.array(70, dtype=jnp.int32))
        assert not bool(at_bridge), "Bridge should be off-screen at scroll=300"

        for _ in range(20):
            _, state, _, _, _ = env.step(state, Action.UP)

        assert not bool(state.player_on_offramp), (
            "Player should not cross to offramp when no bridge is at their position"
        )
        assert int(state.player_y) >= consts.ROAD_TOP_Y, (
            "Player should be blocked at road_top when away from bridge/diagonal"
        )

    def test_bridge_renders_black_pixels_in_gap(self):
        """The gap (median) rows should contain black pixels where a bridge is rendered."""
        import numpy as np
        env = self._make_env_with_bridge(450)
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts

        # Teleport so bridge is on screen (same as test_player_can_cross_at_bridge)
        state = state._replace(scrolling_step_counter=jnp.array(473, dtype=jnp.int32))
        frame = np.array(env.render(state))

        gap_top = consts.ROAD_TOP_Y - consts.OFFRAMP_GAP   # bottom of offramp = top of gap
        gap_bottom = consts.ROAD_TOP_Y                       # top of main road = bottom of gap
        # Bridge left edge ≈ 69 pixels, width = BRIDGE_WIDTH
        bridge_left = 69
        bridge_right = bridge_left + consts.OFFRAMP_BRIDGE_WIDTH

        region = frame[gap_top:gap_bottom, bridge_left:bridge_right]
        black_pixels = int(np.sum(np.all(region == [0, 0, 0], axis=2)))
        assert black_pixels > 0, (
            "Gap region at bridge position should contain black (road) pixels"
        )

    def test_bridge_sprite_dimensions(self):
        """Bridge sprite must be exactly GAP_H rows × BRIDGE_WIDTH columns."""
        import numpy as np
        env = make_env()
        consts = env.consts
        bridge_sprite = np.array(env.renderer._create_offramp_bridge_sprite())
        assert bridge_sprite.shape[0] == consts.OFFRAMP_GAP, "bridge height == OFFRAMP_GAP"
        assert bridge_sprite.shape[1] == consts.OFFRAMP_BRIDGE_WIDTH, "bridge width == OFFRAMP_BRIDGE_WIDTH"
        # Should be solid black
        assert np.all(bridge_sprite[:, :, :3] == 0), "bridge sprite should be solid black"

    def test_enemy_cannot_catch_player_on_bridge(self):
        """Enemy must not catch the player while the player is traversing a bridge."""
        env = self._make_env_with_bridge(450)
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts

        # Teleport scroll so the bridge is at the player's X (scroll=473 → bridge_x=69)
        state = state._replace(scrolling_step_counter=jnp.array(473, dtype=jnp.int32))

        # Confirm bridge is at player position
        at_bridge = env._player_at_bridge(state, jnp.array(state.player_x, dtype=jnp.int32))
        assert bool(at_bridge), "Setup: bridge should be at player X"

        # Move player up so they are in the gap area (above road_top, below offramp)
        # Push up a few steps — they won't be fully on the offramp yet, just crossing
        for _ in range(5):
            _, state, _, _, _ = env.step(state, Action.UP)

        # Place the enemy directly at the player's position (worst-case collision attempt)
        state = state._replace(
            enemy_x=jnp.array(state.player_x, dtype=jnp.int32),
            enemy_y=jnp.array(state.player_y, dtype=jnp.int32),
            enemy_flattened_timer=jnp.array(0, dtype=jnp.int32),
        )

        _, state, _, _, _ = env.step(state, Action.NOOP)
        assert not bool(state.is_round_over), (
            "Enemy must not catch player while player is crossing a bridge"
        )

