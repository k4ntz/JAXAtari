from functools import partial
import jax
import jax.numpy as jnp
from jaxatari.games.jax_yarsrevenge import (
    Direction,
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
