import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace

from jaxatari.games.jax_boxing import JaxBoxing, BoxingConstants, BoxingState, Action


class TestJaxBoxing:
    @pytest.fixture
    def env(self):
        return JaxBoxing()

    @pytest.fixture
    def initial_state(self, env):
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        return state

    def test_reset_positions_and_orientations(self, env, initial_state):
        # White boxer (P1) starts at top-left, Black boxer (P2) at bottom-right
        assert int(initial_state.pos[0, 0]) == env.consts.P1_START_X
        assert int(initial_state.pos[0, 1]) == env.consts.P1_START_Y
        assert int(initial_state.pos[1, 0]) == env.consts.P2_START_X
        assert int(initial_state.pos[1, 1]) == env.consts.P2_START_Y

        # P1 faces right (0), P2 faces left (1)
        assert int(initial_state.orientation[0]) == 0
        assert int(initial_state.orientation[1]) == 1

        assert int(initial_state.score[0]) == 0
        assert int(initial_state.score[1]) == 0
        assert int(initial_state.timer) == env.consts.TOTAL_TIME
        assert not bool(initial_state.done)

    def test_ring_boundaries(self, env, initial_state):
        # Try moving P1 past the top-left boundaries
        state = initial_state
        step_fn = jax.jit(env.step)
        for _ in range(50):
            _, state, _, _, _ = step_fn(state, Action.UPLEFT)

        assert int(state.pos[0, 0]) >= env.consts.XMIN
        assert int(state.pos[0, 1]) >= env.consts.YMIN

        # Try moving P1 past bottom-right boundaries
        for _ in range(150):
            _, state, _, _, _ = step_fn(state, Action.DOWNRIGHT)

        assert int(state.pos[0, 0]) <= env.consts.XMAX
        assert int(state.pos[0, 1]) <= env.consts.YMAX

    def test_ko_termination(self, env, initial_state):
        # State with score >= 100 ends match
        state = replace(initial_state, score=jnp.array([100, 50], dtype=jnp.int32))
        _, next_state, _, done, _ = env.step(state, Action.NOOP)
        assert bool(done)

    def test_timer_termination(self, env, initial_state):
        # State with timer <= 0 ends match
        state = replace(initial_state, timer=jnp.array(1, dtype=jnp.int32))
        _, next_state, _, done, _ = env.step(state, Action.NOOP)
        assert bool(done)
        assert int(next_state.timer) <= 0

    def test_cpu_opportunity_strike(self, env, initial_state):
        # Position P1 right in front of P2 with head aligned with P2's top arm
        # P2 top arm is at p2_y + TOP_ARM_Y (5)
        # P1 face is at [p1_y + 14, p1_y + 32]
        # Set p2 at (70, 70), p1 at (50, 60) -> p1 face [74, 92], p2 top arm 75 (aligned!)
        # horiz_dist = 20 <= 32
        p1_pos = jnp.array([50, 60], dtype=jnp.int32)
        p2_pos = jnp.array([70, 70], dtype=jnp.int32)
        state = replace(
            initial_state,
            pos=jnp.stack([p1_pos, p2_pos]),
            orientation=jnp.array([0, 1], dtype=jnp.int32),
            punch_state=jnp.array([0, 0], dtype=jnp.int32),
            punch_cooldown=jnp.array([0, 0], dtype=jnp.int32),
            stun_timer=jnp.array([0, 0], dtype=jnp.int32),
        )

        cpu_action = env._cpu_logic(state)
        # Action should include FIRE (e.g. Action.FIRE or a directional FIRE)
        is_fire = jnp.isin(
            cpu_action,
            jnp.array([
                Action.FIRE, Action.UPFIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE,
                Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
            ])
        )
        assert bool(is_fire)

    def test_jitted_step_and_render(self, env, initial_state):
        step_fn = jax.jit(env.step)
        render_fn = jax.jit(env.render)

        obs, state, r, d, info = step_fn(initial_state, Action.RIGHT)
        img = render_fn(state)

        assert img.shape == (210, 160, 3)
        assert img.dtype == jnp.uint8
