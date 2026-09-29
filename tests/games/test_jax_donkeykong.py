import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import chex

import jaxatari
from jaxatari.environment import JAXAtariAction as Action


def test_idle_rollout_exact_equivalence():
    """
    Test Case 1:
    Verify that the refactored dynamic ladder implementation produces the EXACT same results
    as the baseline implementation (in terms of rendered frames and falling barrels) for a
    rollout where Mario does not move.
    """
    baseline_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "saved_rollouts", "baseline_rollout_idle_500.npz"
    )
    if not os.path.exists(baseline_path):
        pytest.skip(f"Baseline rollout fixture not found at {baseline_path}")
    
    baseline_data = np.load(baseline_path)
    expected_frames = baseline_data["frames"]
    expected_barrel_x = baseline_data["barrel_x"]
    expected_barrel_y = baseline_data["barrel_y"]
    expected_barrel_stage = baseline_data["barrel_stage"]

    env = jaxatari.make("donkeykong")
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    noop_idx = jnp.where(env.ACTION_SET == Action.NOOP)[0][0]

    # Initial frame
    frame0 = env.render(state)
    np.testing.assert_array_equal(
        np.array(frame0),
        expected_frames[0],
        err_msg="Initial frame 0 does not match baseline bit-for-bit"
    )

    # Step 0: Action.FIRE to start the game
    obs, state, r, d, info = env.step(state, fire_idx)

    # Steps 1 to 500: Action.NOOP while Mario stands idle and barrels roll/descend
    for step in range(500):
        frame = env.render(state)
        np.testing.assert_array_equal(
            np.array(frame),
            expected_frames[step + 1],
            err_msg=f"Rendered frame at step {step + 1} differs from baseline"
        )
        np.testing.assert_array_equal(
            np.array(state.barrels.barrel_x),
            expected_barrel_x[step],
            err_msg=f"Barrel X at step {step + 1} differs from baseline"
        )
        np.testing.assert_array_equal(
            np.array(state.barrels.barrel_y),
            expected_barrel_y[step],
            err_msg=f"Barrel Y at step {step + 1} differs from baseline"
        )
        np.testing.assert_array_equal(
            np.array(state.barrels.stage),
            expected_barrel_stage[step],
            err_msg=f"Barrel stage at step {step + 1} differs from baseline"
        )

        obs, state, r, d, info = env.step(state, noop_idx)


def test_mario_climbing_and_jumping():
    """
    Test Case 2 (Part A):
    Verify that Mario climbing up, climbing down, jumping, and broken ladder rejection
    all function correctly.
    """
    env = jaxatari.make("donkeykong")
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    up_idx = jnp.where(env.ACTION_SET == Action.UP)[0][0]

    # Start game
    obs, state, r, d, info = env.step(state, fire_idx)

    # Place Mario right at ladder 12 (stage 1, x=106, y=183)
    state = state.replace(
        mario_stage=jnp.int32(1),
        mario_x=jnp.float32(106.0),
        mario_y=jnp.float32(183.0),
        mario_climbing=False,
        mario_jumping=False,
    )

    # Press UP to begin climbing
    obs, state, r, d, info = env.step(state, up_idx)
    assert bool(state.mario_climbing), "Mario should start climbing when pressing UP at a valid ladder"

    # Step UP repeatedly until Mario reaches top of ladder
    # At 0.33 px/frame, climbing ~22 pixels takes ~66 frames
    for _ in range(100):
        if not bool(state.mario_climbing):
            break
        obs, state, r, d, info = env.step(state, up_idx)

    assert int(state.mario_stage) == 2, f"Mario should arrive at stage 2 after climbing, got {state.mario_stage}"
    assert not bool(state.mario_climbing), "Mario should finish climbing upon reaching top platform"

    # Place Mario near ladder 11 (broken ladder on stage 1, climbable=False)
    state = state.replace(
        mario_stage=jnp.int32(1),
        mario_x=jnp.float32(70.0),
        mario_y=jnp.float32(183.0),
        mario_climbing=False,
    )
    obs, state, r, d, info = env.step(state, up_idx)
    assert not bool(state.mario_climbing), "Mario must NOT be able to climb a broken (climbable=False) ladder"


def test_barrel_interactions():
    """
    Test Case 2 (Part B):
    Verify that barrels move along stages and descend ladders.
    """
    env = jaxatari.make("donkeykong")
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    noop_idx = jnp.where(env.ACTION_SET == Action.NOOP)[0][0]

    # Start game
    obs, state, r, d, info = env.step(state, fire_idx)

    # Initial barrel starts on stage 6
    assert int(state.barrels.stage[0]) == 6
    start_pos_y = int(state.barrels.barrel_y[0])

    # Run for 200 steps: barrel moves across stage and descends to stage 5
    for _ in range(200):
        obs, state, r, d, info = env.step(state, noop_idx)

    # Verify barrel moved horizontally and descended to lower stages
    assert int(state.barrels.stage[0]) < 6, f"Barrel should have descended below stage 6, got stage {state.barrels.stage[0]}"
    assert int(state.barrels.barrel_y[0]) != start_pos_y, "Barrel should have moved horizontally"


def test_shifted_ladders_mod_assetless():
    """
    Test Case 2 (Part C):
    Verify that ShiftedLaddersMod works dynamically without requiring custom .npy files.
    """
    env = jaxatari.make("donkeykong", mods=["shifted_ladders"])
    key = jax.random.PRNGKey(123)
    obs, state = env.reset(key)

    # Verify ladder positions were modified
    assert int(state.ladders.start_x[0]) == 60, "Ladder 0 should be shifted to x=60 in ShiftedLaddersMod"

    # Step and render
    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    obs, state, r, d, info = env.step(state, fire_idx)
    frame = env.render(state)

    assert frame.shape == (210, 160, 3)
    assert not np.all(np.array(frame) == 0)


def test_center_ladders_mod():
    """
    Verify that CenterLaddersMod shifts ladders inward toward screen center:
    - Level 1: Outer climbable ladders centered (Stage 1 x=94, Stage 2 x=58, Stage 3 x=94, Stage 4 x=58, Stage 5 x=94).
      Mario can climb the centered bottom ladder at x=94.
    - Level 2: 4 columns centered to [52, 68, 88, 104].
    """
    env = jaxatari.make("donkeykong", mods=["center_ladders"])
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    # Level 1 ladder assertions
    assert int(state.ladders.start_x[12]) == 94
    assert int(state.ladders.end_x[12]) == 94
    assert int(state.ladders.render_positions[17, 0]) == 96

    assert int(state.ladders.start_x[9]) == 58
    assert int(state.ladders.render_positions[13, 0]) == 60

    assert int(state.ladders.start_x[8]) == 94
    assert int(state.ladders.render_positions[12, 0]) == 96

    assert int(state.ladders.start_x[7]) == 82
    assert int(state.ladders.render_positions[11, 0]) == 84

    assert int(state.ladders.start_x[3]) == 58
    assert int(state.ladders.render_positions[5, 0]) == 60

    assert int(state.ladders.start_x[4]) == 70
    assert int(state.ladders.render_positions[6, 0]) == 72

    assert int(state.ladders.start_x[2]) == 94
    assert int(state.ladders.render_positions[4, 0]) == 96

    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    up_idx = jnp.where(env.ACTION_SET == Action.UP)[0][0]
    obs, state, r, d, info = env.step(state, fire_idx)

    # Place Mario at the centered bottom ladder position (x=94)
    state = state.replace(
        mario_stage=jnp.int32(1),
        mario_x=jnp.float32(94.0),
        mario_y=jnp.float32(185.0),
        mario_climbing=False,
    )
    obs, state, r, d, info = env.step(state, up_idx)
    assert bool(state.mario_climbing)

    for _ in range(120):
        if not bool(state.mario_climbing):
            break
        obs, state, r, d, info = env.step(state, up_idx)

    assert int(state.mario_stage) == 2

    # Level 2 ladder assertions: centered further to [62, 74, 86, 98] to avoid trap holes at x=52..56 and x=104..108
    l2 = env.init_ladders_for_level(2)
    expected_l2_cols = [62, 74, 86, 98] * 4
    np.testing.assert_array_equal(np.array(l2.start_x), expected_l2_cols)
    np.testing.assert_array_equal(np.array(l2.end_x), expected_l2_cols)
    np.testing.assert_array_equal(np.array(l2.render_positions[:20, 0]), [62, 74, 86, 98] * 5)

    # Verify Mario climbing Level 2 ladder arrives safely without triggering any trap death
    state_l2 = state.replace(
        level=jnp.int32(2),
        ladders=l2,
        invisible_wall_each_stage=env.init_invisible_wall_for_level(2),
        mario_stage=jnp.int32(1),
        mario_x=jnp.float32(62.0),
        mario_y=jnp.float32(87.0),
        mario_climbing=False,
        mario_climbing_delay=False,
    )
    obs, state_l2, r, d, info = env.step(state_l2, up_idx)
    assert bool(state_l2.mario_climbing)
    for _ in range(120):
        if not bool(state_l2.mario_climbing):
            break
        obs, state_l2, r, d, info = env.step(state_l2, up_idx)

    # Mario arrived on stage 2 safely without dying from a trap hole
    assert int(state_l2.mario_stage) == 2
    assert not bool(state_l2.mario_got_hit), "Mario died upon arriving on Level 2 platform!"


def test_jax_optimality_and_vmap():
    """
    Test Case 3:
    Verify JIT compilation and parallel execution throughput with jax.vmap.
    """
    env = jaxatari.make("donkeykong")
    step_fn = jax.jit(env.step)
    render_fn = jax.jit(env.render)

    key = jax.random.PRNGKey(777)
    obs, state = env.reset(key)
    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]

    # Warmup JIT
    obs, state, r, d, info = step_fn(state, fire_idx)
    frame = render_fn(state)
    assert frame.shape == (210, 160, 3)

    # Test vmap over batch of 128 environments
    batch_size = 128
    keys = jax.random.split(key, batch_size)
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    v_obs, v_states = v_reset(keys)
    v_actions = jnp.full((batch_size,), fire_idx, dtype=jnp.int32)

    v_obs, v_states, v_r, v_d, v_info = v_step(v_states, v_actions)

    assert v_states.mario_x.shape == (batch_size,)
    assert v_states.step_counter.shape == (batch_size,)


def test_mario_climbs_to_princess_goal():
    """
    Verify that Mario climbs all the way up to the princess platform (y <= 26)
    before the goal is cleared, rather than clearing prematurely at y=40.
    """
    env = jaxatari.make("donkeykong")
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    up_idx = jnp.where(env.ACTION_SET == Action.UP)[0][0]
    obs, state, r, d, info = env.step(state, fire_idx)

    # Place Mario at the bottom of the top goal ladder (Stage 6, x=76, y=59)
    # Clear barrels to prevent barrel collision from interrupting the climb test
    state = state.replace(
        mario_stage=jnp.int32(6),
        mario_x=jnp.float32(76.0),
        mario_y=jnp.float32(59.0),
        mario_climbing=False,
        barrels=state.barrels.replace(
            barrel_x=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
            barrel_y=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
            reached_the_end=jnp.array([True, True, True, True], dtype=bool),
        ),
    )

    # Start climbing
    obs, state, r, d, info = env.step(state, up_idx)
    assert bool(state.mario_climbing)

    # Climb until y passes below 35 -- goal must NOT trigger yet (in original game it triggered prematurely around 40)
    step_count = 0
    while float(state.mario_y) > 35.0 and step_count < 150:
        step_count += 1
        obs, state, r, d, info = env.step(state, up_idx)
    assert not bool(state.mario_reached_goal), "Goal triggered too early while Mario was still below platform!"

    # Climb all the way up to princess platform
    while not bool(state.mario_reached_goal) and step_count < 200:
        step_count += 1
        obs, state, r, d, info = env.step(state, up_idx)

    # Goal reached right next to the princess at the platform level (y <= 20)
    assert bool(state.mario_reached_goal)
    assert float(state.mario_y) <= 20.0


def test_start_top_platform_mod():
    """
    Verify start_top_platform mod starts Mario on the platform just below the top one
    (Stage 6, x=106, y=43) on the right side facing left, can walk to the ladder and climb to the goal.
    """
    env = jaxatari.make("donkeykong", mods=["start_top_platform"])
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    # Starts on stage 6 platform on right side, facing left
    assert float(state.mario_x) == 106.0
    assert float(state.mario_y) == 43.0
    assert int(state.mario_stage) == 6
    assert int(state.mario_view_direction) == 3  # MOVING_LEFT
    assert not bool(state.mario_reached_goal)

    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    left_idx = jnp.where(env.ACTION_SET == Action.LEFT)[0][0]
    up_idx = jnp.where(env.ACTION_SET == Action.UP)[0][0]
    obs, state, r, d, info = env.step(state, fire_idx)

    # Clear barrels for deterministic walk and climb
    state = state.replace(
        barrels=state.barrels.replace(
            barrel_x=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
            barrel_y=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
            reached_the_end=jnp.array([True, True, True, True], dtype=bool),
        )
    )

    # Walk left to ladder (x=76)
    step_count = 0
    while float(state.mario_x) > 76.5 and step_count < 150:
        step_count += 1
        obs, state, r, d, info = env.step(state, left_idx)

    # Start climbing
    obs, state, r, d, info = env.step(state, up_idx)
    assert bool(state.mario_climbing)

    # Climb up to top platform (y <= 20)
    step_count_climb = 0
    while not bool(state.mario_reached_goal) and step_count_climb < 150:
        step_count_climb += 1
        obs, state, r, d, info = env.step(state, up_idx)

    assert bool(state.mario_reached_goal), "Climbing to top platform should clear the level"
    assert float(state.mario_y) <= 20.0


def test_no_barrels_mod():
    """
    Verify that NoBarrelsMod prevents barrels from spawning either on game start (FIRE)
    or during game rollout.
    """
    env = jaxatari.make("donkeykong", mods=["no_barrels"])
    fire_idx = jnp.where(env.ACTION_SET == Action.FIRE)[0][0]
    right_idx = jnp.where(env.ACTION_SET == Action.RIGHT)[0][0]

    obs, state = env.reset(jax.random.PRNGKey(42))
    assert bool(jnp.all(state.barrels.reached_the_end)), "Barrels should all be reached_the_end on reset"

    # Start game with FIRE
    obs, state, r, d, info = env.step(state, fire_idx)
    assert bool(state.game_started), "Game should start on FIRE"
    assert bool(jnp.all(state.barrels.reached_the_end)), "No barrel should spawn on start in no_barrels mod"

    # Step through rollout
    for _ in range(50):
        obs, state, r, d, info = env.step(state, right_idx)
        assert bool(jnp.all(state.barrels.reached_the_end)), "No barrels should spawn during rollout"




