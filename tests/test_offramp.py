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
            f"player_y={state.player_y} should be >= {expected_min_y} when on main road"
        )

    def test_player_cannot_cross_when_away_from_diagonal(self):
        """Player stays on their current road when the offramp is active but they are away from a diagonal."""
        env = make_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        consts = env.consts
        main_min_y = consts.ROAD_TOP_Y - (consts.PLAYER_SIZE[1] - 5)  # 83

        # Offramp is active (scroll=300) but split_x=(300-200)*3=300 >> WIDTH=160 (off screen to the
        # right), and the merge hasn't started yet (counter=300 < scroll_end=700, merge_x negative).
        # Player should be blocked at main road bounds — no diagonal is reachable.
        state = state._replace(scrolling_step_counter=jnp.array(300, dtype=jnp.int32))

        for _ in range(30):
            _, state, _, _, _ = env.step(state, Action.UP)

        assert int(state.player_y) >= main_min_y, (
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
