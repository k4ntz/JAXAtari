import jax
import jax.numpy as jnp
from functools import partial
import chex
from jaxatari.games.jax_defender import DefenderState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.environment import JAXAtariAction as Action

# --- Simple Mods ---


class InfiniteSmartBombsMod(JaxAtariPostStepModPlugin):
    """
    Prevents Smart Bomb count from decreasing, effectively giving infinite smart bombs.
    """

    constants_overrides = {
        "SPACE_SHIP_INIT_BOMBS": 99,  # Set initial bombs to
    }


class SlowerBulletsMod(JaxAtariInternalModPlugin):
    """
    Reduces the speed of enemy bullets.
    """

    constants_overrides = {
        "PLAYER_SIZE": (4, 4),
        "PLAYER_SIZE_SMALL": (4, 4),
    }


class InfiniteSmartBombsMod(JaxAtariInternalModPlugin):
    """
    Disables Lander's ability to pick up humans.
    """

    SPACE_SHIP_INIT_BOMBS


class StaticInvadersMod(JaxAtariInternalModPlugin):
    """
    Enemies do not move.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(
            enemy: chex.Array, state: DefenderState
        ) -> DefenderState:
            enemy_type = enemy[2]

            # Use original delete logic
            delete_state, score = self._env._delete_enemy(enemy)

            # Return enemy as is (static) unless deleted
            # Important: we must handle the case where delete returns a new state (dead/inactive)

            # If default enemy movement was called, it would return updated positions.
            # Here we just return `enemy` (no position change) for active logic.
            # But wait! jax.lax.switch requires all branches to return same type/shape.
            # `enemy` is [x, y, type, arg1, arg2].
            # `delete_state` is (new_enemy, score) tuple in `_delete_enemy` implementation!
            # Wait, `_delete_enemy` returns (new_enemy, score).
            # The original `_enemy_move_switch` implementations returns `enemy_state` (the array for the enemy).
            # So `_delete_enemy` logic in `_enemy_move_switch` likely UNPACKS it or handles it.

            # Let's check `_enemy_step` in `jax_defender.py` again.
            # delete_state, score = self._delete_enemy(enemy) -> returns (enemy_array, score)
            # lambda: delete_state -> returns enemy_array. Correct.

            return jax.lax.switch(
                jnp.array(enemy_type, int),
                [
                    lambda: enemy,  # Inactive - no change
                    lambda: enemy,  # Lander - static
                    lambda: enemy,  # Pod - static
                    lambda: enemy,  # Bomber - static
                    lambda: enemy,  # Swarmer - static
                    lambda: enemy,  # Mutant - static
                    lambda: enemy,  # Baiter - static
                    lambda: delete_state,  # Dead/Delete - must process
                ],
            )

        def _enemy_move_switch_wrapped(enemy: chex.Array) -> chex.Array:
            return _enemy_move_switch(enemy, state)

        enemy_states_updated = jax.vmap(_enemy_move_switch_wrapped, in_axes=(0,))(
            state.enemy_states
        )
        state = state._replace(enemy_states=enemy_states_updated)

        return state


class FasterLevelClearMod(JaxAtariInternalModPlugin):
    """
    Reduces the number of enemies required to clear a level.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _check_level_done(self, state: DefenderState) -> DefenderState:
        enemy_killed = state.enemy_killed
        needed_kills = jnp.asarray(
            [
                self._env.consts.LANDER_LEVEL_AMOUNT[state.level],
                self._env.consts.POD_LEVEL_AMOUNT[state.level],
                self._env.consts.BOMBER_LEVEL_AMOUNT[state.level],
            ]
        )

        # Reduced requirement: Max 1 kill needed per type (if amount > 0)
        reduced_goals = jnp.minimum(needed_kills, 1)

        is_done = jnp.all(enemy_killed >= reduced_goals)

        state = jax.lax.cond(
            is_done, lambda: self._env._end_level(state), lambda: state
        )
        return state


# --- Difficult Mods ---


class HardcoreStartMod(JaxAtariPostStepModPlugin):
    """
    Start with only 1 life.
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DefenderState, new_state: DefenderState) -> DefenderState:
        return jax.lax.cond(
            new_state.step_counter == 0,
            lambda: new_state._replace(space_ship_lives=jnp.array(1, dtype=jnp.int32)),
            lambda: new_state,
        )


class FasterInvadersMod(JaxAtariInternalModPlugin):
    """
    Increases enemy and bullet speeds.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _bullet_update(self, state: DefenderState) -> DefenderState:
        b_x = state.bullet_x
        b_y = state.bullet_y
        b_dir_x = state.bullet_dir_x
        b_dir_y = state.bullet_dir_y

        # MODIFIED: Increased speed (2x)
        speed_x = b_dir_x * self._env.consts.BULLET_SPEED * 2.0
        speed_y = b_dir_y * self._env.consts.BULLET_SPEED * 2.0

        b_x, b_y = self._env._move_with_space_ship(
            state,
            b_x,
            b_y,
            speed_x,
            speed_y,
            self._env.consts.BULLET_MOVE_WITH_SPACE_SHIP,
        )

        is_bomber = jnp.logical_and(b_dir_x == 0.0, b_dir_y == 0.0)

        def _bomber():
            mask = (state.enemy_states[:, 2] == self._env.consts.BOMBER) & (
                state.enemy_states[:, 3] > 0.0
            )
            match = jnp.nonzero(
                mask, size=self._env.consts.BOMBER_MAX_AMOUNT, fill_value=-1
            )[0]
            enemy = state.enemy_states[match[0]]
            new_ttl = enemy[3] - 1
            return self._env._update_enemy(state, match[0], arg1=new_ttl)

        state = jax.lax.cond(is_bomber, lambda: _bomber(), lambda: state)
        return state._replace(bullet_x=b_x, bullet_y=b_y)


class NoBrakesModActual(JaxAtariInternalModPlugin):
    """
    Ship cannot brake / stop.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _space_ship_step(
        self, state: DefenderState, action: chex.Array
    ) -> DefenderState:
        # Inputs
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        shoot = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.DOWNFIRE,
                    action == Action.UPFIRE,
                    action == Action.RIGHTFIRE,
                    action == Action.LEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )

        direction_x = jnp.where(left, -1, 0) + jnp.where(right, 1, 0)
        direction_y = jnp.where(up, -1, 0) + jnp.where(down, 1, 0)

        space_ship_facing_right = jax.lax.cond(
            direction_x != 0,
            lambda: direction_x > 0,
            lambda: state.space_ship_facing_right,
        )

        space_ship_speed = jax.lax.cond(
            direction_x != 0,
            lambda: state.space_ship_speed
            + direction_x * self._env.consts.SPACE_SHIP_ACCELERATION,
            # MODIFIED: NO BRAKES / NO FRICTION
            lambda: state.space_ship_speed,
        )

        space_ship_speed = jnp.clip(
            space_ship_speed,
            -self._env.consts.SPACE_SHIP_MAX_SPEED,
            self._env.consts.SPACE_SHIP_MAX_SPEED,
        )

        space_ship_x = state.space_ship_x
        space_ship_y = state.space_ship_y

        x_speed = space_ship_speed
        y_speed = direction_y
        space_ship_x, space_ship_y = self._env._move_and_clip(
            space_ship_x,
            space_ship_y,
            x_speed,
            y_speed,
            self._env.consts.SPACE_SHIP_HEIGHT,
        )

        # Decrease shooting cooldown
        shooting_cooldown = jax.lax.cond(
            state.shooting_cooldown > 0,
            lambda: state.shooting_cooldown - 1,
            lambda: 0,
        )

        state = state._replace(
            space_ship_speed=space_ship_speed,
            space_ship_x=space_ship_x,
            space_ship_y=space_ship_y,
            space_ship_facing_right=space_ship_facing_right,
            shooting_cooldown=shooting_cooldown,
        )

        # Shooting if the cooldown is down
        shoot = jnp.logical_and(shoot, shooting_cooldown <= 0)

        # Not be able to shoot in hyperspace
        hyperspace = space_ship_y < (2 - self._env.consts.SPACE_SHIP_HEIGHT)
        shoot_laser = jnp.logical_and(shoot, jnp.logical_not(hyperspace))

        # Shoot bomb if inside city
        in_city = (
            self._env.consts.WORLD_HEIGHT
            - self._env.consts.CITY_HEIGHT
            - self._env.consts.SPACE_SHIP_HEIGHT
        )
        shoot_smart_bomb = jnp.logical_and(
            shoot,
            space_ship_y > in_city,
        )

        # Shoot laser if not in hyperspace and city
        shoot_laser = jnp.logical_xor(shoot_laser, shoot_smart_bomb)

        # If smart bomb is the chosen shot, look up if it is available
        shoot_smart_bomb = jnp.logical_and(
            shoot_smart_bomb, state.smart_bomb_amount > 0
        )

        state = jax.lax.cond(
            shoot_laser, lambda: self._env._shoot_laser(state), lambda: state
        )
        state = jax.lax.cond(
            shoot_smart_bomb, lambda: self._env._shoot_smart_bomb(state), lambda: state
        )

        return state
