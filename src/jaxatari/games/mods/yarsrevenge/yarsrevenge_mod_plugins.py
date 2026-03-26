from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jaxatari.games.jax_yarsrevenge import (
    Direction,
    YarState,
    YarsRevengeGameState,
    YarsRevengeState,
)
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


class NoAnimationsMod(JaxAtariPostStepModPlugin):
    """
    Attaches post step logic to skip any animation-related game stage.
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(
        self, prev_state: YarsRevengeState, new_state: YarsRevengeState
    ) -> YarsRevengeState:
        not_playing = new_state.game_state != YarsRevengeGameState.PLAYING

        return_state = jax.lax.cond(
            not_playing,
            lambda: new_state.replace(
                game_state=jnp.where(
                    new_state.game_state == YarsRevengeGameState.SCOREBOARD,
                    YarsRevengeGameState.PLAYING,
                    new_state.game_state,
                ),  # Skip scoreboard
                game_state_timer=300,  # Large enough frame time to skip both animations
                yar=new_state.yar.replace(
                    x=jnp.array(10).astype(jnp.float32),
                    y=jnp.array(105).astype(jnp.float32),
                    direction=jnp.array(Direction.RIGHT).astype(jnp.int32),
                ),  # Default Yar entity on default positions
            ),
            lambda: new_state,
        )

        return return_state


class SpeedUpMod(JaxAtariInternalModPlugin):
    """
    Overrides the speed constants of everything to make it faster two times.
    """

    constants_overrides = {
        "SWIRL_PER_STEP": 500,  # steps between swirl spawns
        "SWIRL_FIRE_PER_STEP": 120,  # steps between swarm firing
        "QOTILE_SPEED": 1.0,  # slow vertical oscillation of qotile
        "YAR_SPEED": 4.0,  # horizontal move speed
        "YAR_DIAGONAL_SPEED": (
            2.0  # diagonal movement is slower to preserve overall speed
        ),
        "SWIRL_SPEED": 6.0,  # swirl target following speed
        "DESTROYER_SPEED": 0.25,  # very slow chase of Yar
        "ENERGY_MISSILE_SPEED": 8.0,  # fast missile from cannon or YAR
        "CANNON_SPEED": 4.0,  # horizontal cannon movement
        "SNAKE_FRAME": 2,  # snake shift applied every X steps
    }


class MoreSwirlsMod(JaxAtariInternalModPlugin):
    """
    Overrides the SWIRL_PER_STEP constant to make swirl projectiles more frequent.
    """

    constants_overrides = {"SWIRL_PER_STEP": 500}


class StaticEnergyShieldMod(JaxAtariInternalModPlugin):
    """
    Overrides the stage specific step function for preventing the snake shift function call.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _stage_specific_step(
        self, state: YarsRevengeState, energy_shield_state: jnp.ndarray
    ):
        return dict()


class OneShieldShapeMod(JaxAtariInternalModPlugin):
    """
    Overrides the INITIAL_ENERGY_SHIELD constant to only have one type of shield.
    """

    constants_overrides = {
        "INITIAL_ENERGY_SHIELD": jnp.array(
            [jnp.ones((16, 8), dtype=jnp.int32), jnp.ones((16, 8), dtype=jnp.int32)]
        )
    }


class ReversedSnakeMod(JaxAtariInternalModPlugin):
    """
    Attaches post step logic to skip any animation-related game stage.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _snake_shift(self, shield: jnp.ndarray) -> jnp.ndarray:
        """
        Shift the shield cells in a snake-like pattern used in stage 1.
        The algorithm creates an index mapping that flips every other row and then rolls by 1.
        It is implemented purely with JAX operations so it can be compiled.
        """
        n_rows, n_cols = shield.shape
        r = jnp.arange(n_rows).reshape(-1, 1)
        c = jnp.arange(n_cols).reshape(1, -1)
        idx_normal = r * n_cols + c
        idx_reversed = r * n_cols + (n_cols - 1 - c)
        snake_idx = jnp.where(r % 2 == 0, idx_normal, idx_reversed)

        s_flat_snake = shield.reshape(-1)[snake_idx.ravel()]
        shifted = jnp.roll(s_flat_snake, -1)
        new_snake = shifted.reshape(n_rows, n_cols)
        return jnp.where(r % 2 == 0, new_snake, new_snake[:, ::-1])


class VisualNoiseMod(JaxAtariInternalModPlugin):

    constants_overrides = {"RENDERER_VISUAL_NOISE": True}

class FireSpeedMod(JaxAtariInternalModPlugin):

    @partial(jax.jit, static_argnums=(0,))
    def _process_yar_movement(
        self,
        state: YarsRevengeState,
        direction_flags: Tuple[
            bool | jnp.ndarray,  # up flag
            bool | jnp.ndarray,  # down flag
            bool | jnp.ndarray,  # left flag
            bool | jnp.ndarray,  # right flag
        ],
        fire: bool,  # Needed for FireSpeedMod
    ):
        direction = Direction.from_flags(direction_flags)
        yar_moving = direction != Direction._CENTER
        new_yar_direction = jax.lax.select(yar_moving, direction, state.yar.direction)
        new_yar_state = jax.lax.select(yar_moving, YarState.MOVING, YarState.STEADY)

        # Diagonal speed handling
        yar_diagonal = (direction_flags[0] | direction_flags[1]) & (
            direction_flags[2] | direction_flags[3]
        )
        yar_speed = jax.lax.select(
            yar_diagonal,
            self._env.consts.YAR_DIAGONAL_SPEED,
            self._env.consts.YAR_SPEED,
        )
        yar_speed += fire.astype(int) * yar_speed

        delta_yar_x, delta_yar_y = Direction.to_delta(direction_flags, yar_speed)

        # New position
        new_yar_x, new_yar_y = self._env._move_entity(
            state.yar,
            delta_yar_x,
            delta_yar_y,
            wrap_y=True,
        )

        new_yar_entity = state.yar.replace(
            x=new_yar_x, y=new_yar_y, direction=new_yar_direction
        )

        return new_yar_state, new_yar_entity
