import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces
import jaxatari.rendering.jax_rendering_utils as jr
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Space

# Phoenix Game by: Florian Schmidt, Finn Keller
# new Constant class
class PhoenixConstants(NamedTuple):
    """Game constants for Phoenix."""
    PLAYER_POSITION: Tuple[int, int] = (76, 175)
    PLAYER_COLOR: Tuple[int, int, int] = (213, 130, 74)
    WIDTH: int = 160
    HEIGHT: int = 210
    WINDOW_WIDTH: int = 160 * 3
    WINDOW_HEIGHT: int = 210 * 3
    MAX_PLAYER: int = 1
    MAX_PLAYER_PROJECTILE: int = 1
    MAX_PHOENIX: int = 8
    MAX_BATS: int = 7
    MAX_BOSS: int = 1
    MAX_BOSS_BLOCK_GREEN: int = 2
    MAX_BOSS_BLOCK_BLUE: int = 24
    MAX_BOSS_BLOCK_RED: int = 104
    PROJECTILE_WIDTH: int = 2
    PROJECTILE_HEIGHT: int = 4
    ENEMY_WIDTH: int = 6
    ENEMY_HEIGHT:int = 5
    WING_WIDTH: int = 5
    BLOCK_WIDTH:int = 4
    BLOCK_HEIGHT:int = 4
    SCORE_COLOR: Tuple[int, int, int] = (210, 210, 64)
    PLAYER_BOUNDS: Tuple[int, int] = (0, 155)  # (left, right)
    ENEMY_POSITIONS_X_LIST = [
        lambda: jnp.array(
            [123 - 160 // 2, 123 - 160 // 2, 136 - 160 // 2, 136 - 160 // 2, 160 - 160 // 2, 160 - 160 // 2,
             174 - 160 // 2, 174 - 160 // 2]).astype(jnp.int32),
        lambda: jnp.array(
            [141 - 160 // 2, 155 - 160 // 2, 127 - 160 // 2, 169 - 160 // 2, 134 - 160 // 2, 162 - 160 // 2,
             120 - 160 // 2, 176 - 160 // 2]).astype(jnp.int32),
        lambda: jnp.array(
            [123 - 160 // 2, 170 - 160 // 2, 123 - 160 // 2, 180 - 160 // 2, 123 - 160 // 2, 170 - 160 // 2,
             123 - 160 // 2, -1]).astype(jnp.int32),
        lambda: jnp.array(
            [123 - 160 // 2, 180 - 160 // 2, 123 - 160 // 2, 170 - 160 // 2, 123 - 160 // 2, 180 - 160 // 2,
             123 - 160 // 2, -1]).astype(jnp.int32),
        lambda: jnp.array([78, -1, -1, -1, -1, -1, -1, -1]).astype(jnp.int32),
    ]
    ENEMY_POSITIONS_Y_LIST = [
        lambda: jnp.array(
            [210 - 135, 210 - 153, 210 - 117, 210 - 171, 210 - 117, 210 - 171, 210 - 135,
             210 - 153]).astype(jnp.int32),
        lambda: jnp.array(
            [210 - 171, 210 - 171, 210 - 135, 210 - 135, 210 - 153, 210 - 153, 210 - 117,
             210 - 117]).astype(jnp.int32),
        lambda: jnp.array(
            [210 - 99, 210 - 117, 210 - 135, 210 - 153, 210 - 171, 210 - 63, 210 - 81,
             210 + 20]).astype(jnp.int32),
        lambda: jnp.array(
            [210 - 63, 210 - 81, 210 - 99, 210 - 117, 210 - 135, 210 - 153, 210 - 171,
             210 + 20]).astype(jnp.int32),
        lambda: jnp.array([210 - 132, 210 + 20, 210 + 20, 210 + 20, 210 + 20, 210 + 20, 210 + 20,
                           210 + 20]).astype(jnp.int32),
    ]
    BLUE_BLOCK_X = jnp.linspace(PLAYER_BOUNDS[0] + 32, PLAYER_BOUNDS[1] - 32,
                                24).astype(jnp.int32)

    BLUE_BLOCK_Y_1 = jnp.full((24,), HEIGHT - 115, dtype=jnp.int32)
    BLUE_BLOCK_Y_2 = jnp.full((24,), HEIGHT - 117, dtype=jnp.int32)

    BLUE_BLOCK_POSITIONS = jnp.concatenate([
        jnp.stack((BLUE_BLOCK_X, BLUE_BLOCK_Y_1), axis=1),
        jnp.stack((BLUE_BLOCK_X, BLUE_BLOCK_Y_2), axis=1),
    ])

    # 1 Line with Blocks the same amount as Blue Blocks
    RED_BLOCK_X_1 = jnp.linspace(PLAYER_BOUNDS[0] + 32, PLAYER_BOUNDS[1] - 32, MAX_BOSS_BLOCK_BLUE).astype(jnp.int32)
    RED_BLOCK_X_2 = jnp.linspace(PLAYER_BOUNDS[0] + 36, PLAYER_BOUNDS[1] - 36, MAX_BOSS_BLOCK_BLUE - 2).astype(
        jnp.int32)
    RED_BLOCK_X_3 = jnp.linspace(PLAYER_BOUNDS[0] + 40, PLAYER_BOUNDS[1] - 40, MAX_BOSS_BLOCK_BLUE - 4).astype(
        jnp.int32)
    RED_BLOCK_X_4 = jnp.linspace(PLAYER_BOUNDS[0] + 44, PLAYER_BOUNDS[1] - 44, MAX_BOSS_BLOCK_BLUE - 6).astype(
        jnp.int32)
    RED_BLOCK_X_5 = jnp.linspace(PLAYER_BOUNDS[0] + 48, PLAYER_BOUNDS[1] - 48, MAX_BOSS_BLOCK_BLUE - 8).astype(
        jnp.int32)
    RED_BLOCK_X_6 = jnp.linspace(PLAYER_BOUNDS[0] + 52, PLAYER_BOUNDS[1] - 52, MAX_BOSS_BLOCK_BLUE - 10).astype(
        jnp.int32)
    RED_BLOCK_X_7 = jnp.linspace(PLAYER_BOUNDS[0] + 56, PLAYER_BOUNDS[1] - 56, MAX_BOSS_BLOCK_BLUE - 12).astype(
        jnp.int32)
    RED_BLOCK_POSITIONS = jnp.concatenate(
        [
            jnp.stack((RED_BLOCK_X_1, jnp.full((MAX_BOSS_BLOCK_BLUE,), HEIGHT - 111, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_2, jnp.full((MAX_BOSS_BLOCK_BLUE - 2,), HEIGHT - 108, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_3, jnp.full((MAX_BOSS_BLOCK_BLUE - 4,), HEIGHT - 105, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_4, jnp.full((MAX_BOSS_BLOCK_BLUE - 6,), HEIGHT - 102, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_5, jnp.full((MAX_BOSS_BLOCK_BLUE - 8,), HEIGHT - 99, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_6, jnp.full((MAX_BOSS_BLOCK_BLUE - 10,), HEIGHT - 96, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_7, jnp.full((MAX_BOSS_BLOCK_BLUE - 12,), HEIGHT - 93, dtype=jnp.int32)), axis=1)
        ],
        axis=0
    )

    GREEN_BLOCK_Y_1 = jnp.linspace(HEIGHT - 120, HEIGHT - 128, 5).astype(jnp.int32)
    GREEN_BLOCK_X_1 = jnp.full((5,), WIDTH // 2 + 8, dtype=jnp.int32)

    GREEN_BLOCK_X_2 = jnp.full((4,), WIDTH // 2 + 12, dtype=jnp.int32)
    GREEN_BLOCK_Y_2 = jnp.linspace(HEIGHT - 120, HEIGHT - 126, 4).astype(jnp.int32)

    GREEN_BLOCK_X_3 = jnp.full((3,), WIDTH // 2 + 16, dtype=jnp.int32)
    GREEN_BLOCK_Y_3 = jnp.linspace(HEIGHT - 120, HEIGHT - 124, 3).astype(jnp.int32)

    GREEN_BLOCK_X_4 = jnp.full((2,), WIDTH // 2 + 20, dtype=jnp.int32)
    GREEN_BLOCK_Y_4 = jnp.linspace(HEIGHT - 120, HEIGHT - 122, 2).astype(jnp.int32)

    GREEN_BLOCK_X_5 = jnp.full((1,), WIDTH // 2 + 24, dtype=jnp.int32)
    GREEN_BLOCK_Y_5 = jnp.linspace(HEIGHT - 120, HEIGHT - 120, 1).astype(jnp.int32)

    # mirror the blocks to the left side
    GREEN_BLOCK_Y_6 = jnp.linspace(HEIGHT - 120, HEIGHT - 128, 5).astype(jnp.int32)
    GREEN_BLOCK_X_6 = jnp.full((5,), WIDTH // 2 - 8, dtype=jnp.int32)

    GREEN_BLOCK_X_7 = jnp.full((4,), WIDTH // 2 - 12, dtype=jnp.int32)
    GREEN_BLOCK_Y_7 = jnp.linspace(HEIGHT - 120, HEIGHT - 126, 4).astype(jnp.int32)

    GREEN_BLOCK_X_8 = jnp.full((3,), WIDTH // 2 - 16, dtype=jnp.int32)
    GREEN_BLOCK_Y_8 = jnp.linspace(HEIGHT - 120, HEIGHT - 124, 3).astype(jnp.int32)

    GREEN_BLOCK_X_9 = jnp.full((2,), WIDTH // 2 - 20, dtype=jnp.int32)
    GREEN_BLOCK_Y_9 = jnp.linspace(HEIGHT - 120, HEIGHT - 122, 2).astype(jnp.int32)

    GREEN_BLOCK_X_10 = jnp.full((1,), WIDTH // 2 - 24, dtype=jnp.int32)
    GREEN_BLOCK_Y_10 = jnp.linspace(HEIGHT - 120, HEIGHT - 120, 1).astype(jnp.int32)

    GREEN_BLOCK_POSITIONS = jnp.concatenate(
        [
            jnp.stack((GREEN_BLOCK_X_1, GREEN_BLOCK_Y_1), axis=1),
            jnp.stack((GREEN_BLOCK_X_2, GREEN_BLOCK_Y_2), axis=1),
            jnp.stack((GREEN_BLOCK_X_3, GREEN_BLOCK_Y_3), axis=1),
            jnp.stack((GREEN_BLOCK_X_4, GREEN_BLOCK_Y_4), axis=1),
            jnp.stack((GREEN_BLOCK_X_5, GREEN_BLOCK_Y_5), axis=1),
            jnp.stack((GREEN_BLOCK_X_6, GREEN_BLOCK_Y_1), axis=1),
            jnp.stack((GREEN_BLOCK_X_7, GREEN_BLOCK_Y_2), axis=1),
            jnp.stack((GREEN_BLOCK_X_8, GREEN_BLOCK_Y_3), axis=1),
            jnp.stack((GREEN_BLOCK_X_9, GREEN_BLOCK_Y_4), axis=1),
            jnp.stack((GREEN_BLOCK_X_10, GREEN_BLOCK_Y_5), axis=1)

        ]
    )

# === GAME STATE ===
class PhoenixState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    step_counter: chex.Array
    enemies_x: chex.Array # Gegner X-Positionen
    enemies_y: chex.Array
    enemy_direction: chex.Array
    vertical_direction: chex.Array
    phoenix_cooldown: chex.Array
    blue_blocks: chex.Array
    red_blocks: chex.Array
    green_blocks: chex.Array
    invincibility: chex.Array
    invincibility_timer: chex.Array
    bat_wings: chex.Array
    projectile_x: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil
    projectile_y: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil # Gegner Y-Positionen
    enemy_projectile_x: chex.Array = jnp.full((8,), -1) # Enemy projectile X-Positionen
    enemy_projectile_y: chex.Array = jnp.full((8,), -1) # Enemy projectile Y-Positionen

    score: chex.Array = jnp.array(0)  # Score
    lives: chex.Array = jnp.array(5) # Lives
    player_respawn_timer: chex.Array = 0 # Invincibility timer
    level: chex.Array = jnp.array(1)  # Level, starts at 1



class PhoenixOberservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_score: chex.Array
    lives: chex.Array

class PhoenixInfo(NamedTuple):
    step_counter: jnp.ndarray

class CarryState(NamedTuple):
    score: chex.Array

class EntityPosition(NamedTuple):## not sure
    x: chex.Array
    y: chex.Array

class JaxPhoenix(JaxEnvironment[PhoenixState, PhoenixOberservation, PhoenixInfo, None]):
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PhoenixState) -> PhoenixOberservation:
        player = EntityPosition(x=state.player_x, y=state.player_y)
        return PhoenixOberservation(
            player_x = player[0],
            player_y= player[1],
            player_score = state.score,
            lives= state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PhoenixState, all_rewards: jnp.ndarray) -> PhoenixInfo:
        return PhoenixInfo(
            step_counter=0,
        )

    def action_space(self) -> Space:
        return self._get_observation(PhoenixState)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PhoenixState) -> Tuple[bool, PhoenixState]:
        return jnp.less_equal(state.lives,0)
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    def __init__(self, consts: PhoenixConstants = None, reward_funcs: list[callable]=None):
        consts = consts or PhoenixConstants()
        super().__init__(consts)
        self.renderer = PhoenixRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.step_counter = 0
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]# Add step counter tracking

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: PhoenixState, action: chex.Array) -> tuple[chex.Array]:
        step_size = 2  # Größerer Wert = schnellerer Schritt
        # left action
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        # right action
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )
        # Ability : it holds on for ... amount; it can only be reactivated when an enemy is hitted
        invinsibility = jnp.any(jnp.array([action == Action.DOWN]))
        new_invinsibility = jnp.where(invinsibility & (state.invincibility_timer == 0), True, state.invincibility)
        new_timer = jnp.where(invinsibility & (state.invincibility_timer == 0), 200, state.invincibility_timer)
        new_timer = jnp.where(new_timer > 0, new_timer - 1, 0)
        new_invinsibility = jnp.where(new_timer == 0, False, new_invinsibility)
        # movement right
        player_x = jnp.where(
            right & jnp.logical_not(new_invinsibility), state.player_x + step_size, jnp.where(left & jnp.logical_not(new_invinsibility), state.player_x - step_size, state.player_x)
        )
        # movement left
        player_x = jnp.where(
            player_x < self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[0],
            jnp.where(player_x > self.consts.PLAYER_BOUNDS[1], self.consts.PLAYER_BOUNDS[1], player_x)
        )
        state = state._replace(player_x= player_x.astype(jnp.float32),
                               invincibility=new_invinsibility,
                               invincibility_timer=new_timer)

        return state

    def phoenix_step(self, state):
        enemy_step_size = 0.4
        vertical_step_size = 0.3

        active_enemies = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10)

        # Prüfen, ob ein Gegner die linke oder rechte Grenze erreicht hat
        at_left_boundary = jnp.any(jnp.logical_and(state.enemies_x <= self.consts.PLAYER_BOUNDS[0], active_enemies))
        at_right_boundary = jnp.any(
            jnp.logical_and(state.enemies_x >= self.consts.PLAYER_BOUNDS[1] - self.consts.ENEMY_WIDTH / 2,
                            active_enemies))
        # Richtung ändern, wenn eine Grenze erreicht wird
        new_direction = jax.lax.cond(
            at_left_boundary,
            lambda: jnp.full_like(state.enemy_direction, 1.0, dtype=jnp.float32),
            lambda: jax.lax.cond(
                at_right_boundary,
                lambda: jnp.full_like(state.enemy_direction, -1.0, dtype=jnp.float32),
                lambda: state.enemy_direction.astype(jnp.float32),
            ),
        )
        # Choose the two most bottom phoenix enemies to move
        # jax.debug.print("Enemy Y positions: {}", state.enemies_y)
        # jax.debug.print("ENEMY_Y: {}", ENEMY_POSITIONS_Y)
        enemy_indices = jnp.argsort(state.enemies_y, axis=0)[-2:]
        hit_bottom_mask = (state.enemies_y >= 159) & jnp.isin(jnp.arange(state.enemies_y.shape[0]), enemy_indices)
        new_vertical_direction = jnp.where(
            hit_bottom_mask,
            -state.vertical_direction,
            state.vertical_direction
        )
        new_enemies_y = jnp.where(
            jnp.isin(jnp.arange(state.enemies_y.shape[0]), enemy_indices),
            state.enemies_y + (new_vertical_direction * vertical_step_size),
            state.enemies_y
        )

        # Gegner basierend auf der Richtung bewegen, nur aktive Gegner
        new_enemies_x = jnp.where(active_enemies, state.enemies_x + (new_direction * enemy_step_size), state.enemies_x)

        # Begrenzung der Positionen innerhalb des Spielfelds
        new_enemies_x = jnp.clip(new_enemies_x, self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[1])

        # jax.debug.print("dir :{}", new_vertical_direction)
        state = state._replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            enemy_direction=new_direction.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
            vertical_direction=new_vertical_direction
        )

        # Aktualisierten Zustand zurückgeben
        return state

    def bat_step(self, state):
        bat_step_size = 0.5
        active_bats = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10)
        proj_pos = jnp.array([state.projectile_x, state.projectile_y])

        # Initialisiere neue Richtungen für jede Fledermaus
        new_directions = jnp.where(
            jnp.logical_and(state.enemies_x <= self.consts.PLAYER_BOUNDS[0] + 3, active_bats),
            jnp.ones(state.enemy_direction.shape, dtype=jnp.float32),  # Force array shape
            jnp.where(
                jnp.logical_and(state.enemies_x >= self.consts.PLAYER_BOUNDS[1] - self.consts.ENEMY_WIDTH / 2,
                                active_bats),
                jnp.ones(state.enemy_direction.shape, dtype=jnp.float32) * -1,  # Force array shape
                state.enemy_direction.astype(jnp.float32)  # Ensure consistency
            )
        )

        # Bewege Fledermäuse basierend auf ihrer individuellen Richtung
        new_enemies_x = jnp.where(active_bats, state.enemies_x + (new_directions * bat_step_size), state.enemies_x)
        enemy_pos = jnp.stack([new_enemies_x, state.enemies_y], axis=1)
        new_enemies_x = jnp.clip(new_enemies_x, self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[1])
        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            proj_x, proj_y = projectile_pos
            wing_left_x = enemy_x - 5
            wing_y = enemy_y + 2
            wing_right_x = enemy_x + 5
            collision_x_left = (proj_x + self.consts.PROJECTILE_WIDTH > wing_left_x) & (
                    proj_x < wing_left_x + self.consts.WING_WIDTH)
            collision_y = (proj_y + self.consts.PROJECTILE_HEIGHT > wing_y) & (
                    proj_y < enemy_y + 2)
            collision_x_right = (proj_x + self.consts.PROJECTILE_WIDTH > wing_right_x) & (
                    proj_x < wing_right_x + self.consts.WING_WIDTH)

            return collision_x_left & collision_y, collision_x_right & collision_y

        left_wing_collision, right_wing_collision = jax.vmap(lambda entity_pos: check_collision(entity_pos, proj_pos))(enemy_pos)
        left_hit_valid = left_wing_collision & ((state.bat_wings == 2) | (state.bat_wings == -1))
        right_hit_valid = right_wing_collision & ((state.bat_wings == 2) | (state.bat_wings == 1))

        # Only remove the projectile if any valid hit occurred
        any_valid_hit = jnp.any(left_hit_valid | right_hit_valid)
        new_proj_y = jnp.where(any_valid_hit, -1, state.projectile_y)
        def update_wing_state(current_state, left_hit, right_hit):
            # current_state: int (-1,0,1,2), left_hit & right_hit: bool

            # First handle left wing hit
            updated = jnp.where(
                left_hit,
                jnp.where(current_state == 2, 1,  # both wings → right wing only
                          jnp.where(current_state == -1, 0, current_state)),  # right only → none, else unchanged
                current_state
            )

            # Then handle right wing hit
            updated = jnp.where(
                right_hit,
                jnp.where(updated == 2, -1,  # both wings → left wing only
                          jnp.where(updated == 1, 0, updated)),  # left only → none, else unchanged
                updated
            )

            return updated

        new_bat_wings = jax.vmap(update_wing_state)(state.bat_wings, left_wing_collision, right_wing_collision)
        jax.debug.print("bat_wings: {}", new_bat_wings)


        state = state._replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            enemies_y=state.enemies_y.astype(jnp.float32),
            enemy_direction=new_directions.astype(jnp.float32),
            projectile_y=new_proj_y,
            bat_wings= new_bat_wings,
        )

        return state

    def boss_step(self, state):
        step_size = 0.05
        step_count = state.step_counter

        condition = (state.enemies_y[0] <= 100) & ((step_count % 30) == 0)

        def move_blocks(blocks):
            not_removed = blocks[:, 0] > -99  # Filter out removed blocks
            move_mask = condition & not_removed
            return blocks.at[:, 1].set(
                jnp.where(move_mask, blocks[:, 1] + step_size, blocks[:, 1])
            )

        new_green_blocks = move_blocks(state.green_blocks)
        new_red_blocks = move_blocks(state.red_blocks)
        new_blue_blocks = move_blocks(state.blue_blocks)

        new_enemy_y = jnp.where(condition, state.enemies_y + step_size, state.enemies_y.astype(jnp.float32))

        projectile_active = (state.projectile_x >= 0) & (state.projectile_y >= 0)
        projectile_pos = jnp.array([state.projectile_x, state.projectile_y])

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            projectile_x, projectile_y = projectile_pos

            collision_x = (projectile_x + self.consts.PROJECTILE_WIDTH > enemy_x) & (
                        projectile_x < enemy_x + self.consts.BLOCK_WIDTH)
            collision_y = (projectile_y + self.consts.PROJECTILE_HEIGHT > enemy_y) & (
                        projectile_y < enemy_y + self.consts.BLOCK_HEIGHT)
            return collision_x & collision_y

        def process_collisions(_):
            # Check collisions for each block group
            green_block_collisions = jax.vmap(lambda entity_pos: check_collision(entity_pos, projectile_pos))(
                new_green_blocks)
            red_block_collisions = jax.vmap(lambda entity_pos: check_collision(entity_pos, projectile_pos))(
                new_red_blocks)
            blue_block_collisions = jax.vmap(lambda entity_pos: check_collision(entity_pos, projectile_pos))(
                new_blue_blocks)

            def remove_first_hit(blocks, collisions):
                hit_indices = jnp.where(collisions, size=1, fill_value=-1)[0]
                first_hit_index = hit_indices[0]  # scalar

                def remove_block(i, arr):
                    # Create a mask with True only at index i
                    mask = jnp.arange(arr.shape[0]) == i  # shape (50,)
                    # Broadcast mask to shape (50, 1) to align with arr shape (50, 2)
                    mask = mask[:, None]  # shape (50, 1)

                    # Create an array of -100 with the same shape as one block (2,)
                    replacement = jnp.full(arr.shape[1:], -100)  # shape (2,)

                    # Use jnp.where to replace the entire row at index i with -100 vector
                    return jnp.where(mask, replacement, arr)

                return jax.lax.cond(
                    first_hit_index >= 0,
                    lambda: remove_block(first_hit_index, blocks),
                    lambda: blocks
                ), first_hit_index

            new_green, _ = remove_first_hit(new_green_blocks, green_block_collisions)
            new_red, _ = remove_first_hit(new_red_blocks, red_block_collisions)
            new_blue, first_hit_idx = remove_first_hit(new_blue_blocks, blue_block_collisions)
            hit_any = jnp.any(green_block_collisions) | jnp.any(red_block_collisions) | jnp.any(blue_block_collisions)
            return new_green, new_red, new_blue, hit_any

        def skip_collisions(_):
            return (new_green_blocks, new_red_blocks, new_blue_blocks, False)

        # Use lax.cond to select between processing or skipping collision
        new_green_blocks, new_red_blocks, new_blue_blocks, projectile_hit_detected = jax.lax.cond(
            projectile_active,
            process_collisions,
            skip_collisions,
            operand=None
        )

        def rotate(arr):
            return jnp.stack([jnp.roll(arr[:, 0], 1), arr[:, 1]], axis=1)

        new_blue_blocks = jax.lax.cond(
            jnp.logical_and(jnp.any(new_blue_blocks <= -100), step_count % 20 == 0),
            lambda: rotate(new_blue_blocks),
            lambda: new_blue_blocks,
        )
        projectile_x = jnp.where(projectile_hit_detected, -1, state.projectile_x)
        projectile_y = jnp.where(projectile_hit_detected, -1, state.projectile_y)

        state = state._replace(
            enemies_y=new_enemy_y.astype(jnp.float32),
            blue_blocks=new_blue_blocks,
            red_blocks=new_red_blocks,
            green_blocks=new_green_blocks,
            projectile_x=projectile_x,
            projectile_y=projectile_y,
            enemies_x = state.enemies_x.astype(jnp.float32),
        )
        return state

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PhoenixOberservation, PhoenixState]:

        return_state = PhoenixState(
            player_x=jnp.array(self.consts.PLAYER_POSITION[0]),
            player_y=jnp.array(self.consts.PLAYER_POSITION[1]),
            step_counter=jnp.array(0),
            enemies_x = self.consts.ENEMY_POSITIONS_X_LIST[0](),
            enemies_y = self.consts.ENEMY_POSITIONS_Y_LIST[0](),
            enemy_direction =  jnp.full((8,), -1.0),
            enemy_projectile_x=jnp.full((8,), -1),
            enemy_projectile_y=jnp.full((8,), -1),
            projectile_x=jnp.array(-1),  # Standardwert: kein Projektil
            score = jnp.array(0), # Standardwert: Score=0
            lives=jnp.array(5), # Standardwert: 5 Leben
            player_respawn_timer=jnp.array(5),
            level=jnp.array(1),
            phoenix_cooldown=jnp.array(30),
            vertical_direction=jnp.full((8,),1.0),
            invincibility=jnp.array(False),
            invincibility_timer=jnp.array(0),
            bat_wings=jnp.full((8,), 2),

            blue_blocks=self.consts.BLUE_BLOCK_POSITIONS.astype(jnp.float32),
            red_blocks=self.consts.RED_BLOCK_POSITIONS.astype(jnp.float32),
            green_blocks = self.consts.GREEN_BLOCK_POSITIONS.astype(jnp.float32),            
        )

        initial_obs = self._get_observation(return_state)
        return initial_obs, return_state


    def step(self,state, action: Action) -> Tuple[PhoenixOberservation, PhoenixState, float, bool, PhoenixInfo]:
        state = self.player_step(state, action)
        #jax.debug.print("invinsiblity:{}", state.invincibility)
        #jax.debug.print("timer:{}", state.invincibility_timer)

        projectile_active = state.projectile_y >= 0

        # Can fire only if inactive
        can_fire = ~projectile_active
        firing = (action == Action.FIRE) & can_fire

        state = jax.lax.cond(
            jnp.logical_or((state.level % 5) == 1, (state.level % 5) == 2),
            lambda: self.phoenix_step(state),
            lambda: jax.lax.cond(
                jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4),
                lambda: self.bat_step(state),
                lambda: self.boss_step(state),

            )
        )
        projectile_x = jnp.where(firing,
                                 state.player_x + 2,


                                 state.projectile_x).astype(jnp.int32)

        projectile_y = jnp.where(firing,
                                 state.player_y - 1,
                                 jnp.where(projectile_active,
                                           state.projectile_y - 3,  # move up if active
                                           state.projectile_y))  # stay
        projectile_y = jnp.where(projectile_y < 0, -6, projectile_y)
        # use step_counter for randomness
        def generate_fire_key_and_chance(step_counter: int, fire_chance: float) -> Tuple[jax.random.PRNGKey, float]:
            key = jax.random.PRNGKey(step_counter)
            return key, fire_chance

        key, fire_chance = generate_fire_key_and_chance(state.step_counter, 0.0075)  # 2% chance per enemy per frame

        # Random decision: should each enemy fire?
        enemy_should_fire = jax.random.uniform(key, (8,)) < fire_chance

        # Fire only from active enemies
        can_fire = (state.enemy_projectile_y < 0) & (state.enemies_x > -1)
        enemy_fire_mask = enemy_should_fire & can_fire

        # Fire from current enemy positions
        enemy_projectile_x = jnp.where(enemy_fire_mask, state.enemies_x + self.consts.ENEMY_WIDTH // 2,
                                           state.enemy_projectile_x)
        enemy_projectile_y = jnp.where(enemy_fire_mask, state.enemies_y + self.consts.ENEMY_HEIGHT, state.enemy_projectile_y)

        # Move enemy projectiles downwards
        enemy_projectile_y = jnp.where(state.enemy_projectile_y >= 0, state.enemy_projectile_y + 4, # +4 regelt enemy projectile speed
                                           enemy_projectile_y)

        # Remove enemy projectile if off-screen
        enemy_projectile_y = jnp.where(enemy_projectile_y > 185 - self.consts.PROJECTILE_HEIGHT, -1, enemy_projectile_y)


        projectile_pos = jnp.array([projectile_x, projectile_y])
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            projectile_x, projectile_y = projectile_pos

            collision_x = (projectile_x + self.consts.PROJECTILE_WIDTH > enemy_x) & (projectile_x < enemy_x + self.consts.ENEMY_WIDTH)
            collision_y = (projectile_y + self.consts.PROJECTILE_HEIGHT > enemy_y) & (projectile_y < enemy_y + self.consts.ENEMY_HEIGHT)
            return collision_x & collision_y


        # Kollisionsprüfung Gegner
        enemy_collisions = jax.vmap(lambda enemy_pos: check_collision(enemy_pos, projectile_pos))(enemy_positions)
        enemy_hit_detected = jnp.any(enemy_collisions)


        # Gegner und Projektil entfernen wenn eine Kollision erkannt wurde
        enemies_x = jnp.where(enemy_collisions, -1, state.enemies_x).astype(jnp.float32)
        enemies_y = jnp.where(enemy_collisions, self.consts.HEIGHT+20, state.enemies_y).astype(jnp.float32)
        projectile_x = jnp.where(enemy_hit_detected, -1, projectile_x)
        projectile_y = jnp.where(enemy_hit_detected, -1, projectile_y)
        score = jnp.where(enemy_hit_detected, state.score + 20, state.score)

        # Checken ob alle Gegner getroffen wurden
        all_enemies_hit = jnp.all(enemies_y >= self.consts.HEIGHT + 10)
        new_level = jnp.where(all_enemies_hit, (state.level % 5) + 1, state.level)
        new_enemies_x = jax.lax.cond(
            all_enemies_hit,
            lambda: jax.lax.switch((new_level -1 )% 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32),
            lambda: state.enemies_x.astype(jnp.float32)
        )
        new_enemies_y = jax.lax.cond(
            all_enemies_hit,
            lambda: jax.lax.switch((new_level -1 )% 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32),
            lambda: enemies_y.astype(jnp.float32)
        )
        enemies_x = new_enemies_x
        enemies_y = new_enemies_y
        level = new_level

        #jax.debug.print("Level: {}",level )
        def check_player_hit(projectile_xs, projectile_ys, player_x, player_y):
            def is_hit(px, py):
                hit_x = (px + self.consts.PROJECTILE_WIDTH > player_x) & (px < player_x + 5)
                hit_y = (py + self.consts.PROJECTILE_HEIGHT > player_y) & (py < player_y + self.consts.PROJECTILE_HEIGHT)
                return hit_x & hit_y

            hits = jax.vmap(is_hit)(projectile_xs, projectile_ys)
            return jnp.any(hits)



        # Kollisionsüberprüfung Spieler
        # Remaining lives updaten und Spieler neu Spawnen
        is_vulnerable = state.player_respawn_timer <= 0
        player_hit_detected = jnp.where(jnp.logical_and(is_vulnerable,state.invincibility == jnp.array(False)), check_player_hit(state.enemy_projectile_x, enemy_projectile_y, state.player_x, state.player_y), False)
        lives = jnp.where(player_hit_detected, state.lives - 1, state.lives)
        player_x = jnp.where(player_hit_detected, self.consts.PLAYER_POSITION[0], state.player_x)
        player_respawn_timer = jnp.where(
            player_hit_detected,
            5,
            jnp.maximum(state.player_respawn_timer - 1, 0)
        )
        # Respawn remaining enemies
        enemy_respawn_x = jax.lax.switch((level -1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32)
        enemy_respawn_y = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.int32)

        enemy_respawn_mask = jnp.logical_and(player_hit_detected, (enemies_x > 0) & (enemies_y < self.consts.HEIGHT + 10))
        enemies_x = jnp.where(enemy_respawn_mask, enemy_respawn_x, enemies_x)
        enemies_y = jnp.where(enemy_respawn_mask, enemy_respawn_y, enemies_y)



        # Enemy Projectile entfernen wenn eine Kollision mit dem Spieler erkannt wurde
        enemy_projectile_x = jnp.where(player_hit_detected, -1, enemy_projectile_x)
        enemy_projectile_y = jnp.where(player_hit_detected, -1, enemy_projectile_y)

        return_state = PhoenixState(
            player_x = player_x,
            player_y = state.player_y,
            step_counter = state.step_counter + 1,
            projectile_x = projectile_x,
            projectile_y = projectile_y,
            enemies_x = enemies_x,
            enemies_y = enemies_y,
            enemy_direction = state.enemy_direction,
            score= score,
            enemy_projectile_x=enemy_projectile_x,
            enemy_projectile_y=enemy_projectile_y,
            lives=lives,
            player_respawn_timer = player_respawn_timer,
            level = level,
            phoenix_cooldown = state.phoenix_cooldown,
            vertical_direction=state.vertical_direction,
            blue_blocks=state.blue_blocks,
            red_blocks=state.red_blocks,
            green_blocks=state.green_blocks,
            invincibility=state.invincibility,
            invincibility_timer=state.invincibility_timer,
            bat_wings=state.bat_wings

        )
        observation = self._get_observation(return_state)
        env_reward = jnp.where(enemy_hit_detected, 1.0, 0.0)
        done = self._get_done(return_state)
        info = self._get_info(return_state, env_reward)
        return observation, return_state, env_reward, done, info

    def render(self, state:PhoenixState) -> jnp.ndarray:
        return self.renderer.render(state)
from jaxatari.renderers import JAXGameRenderer

class PhoenixRenderer(JAXGameRenderer):
    def __init__(self, consts: PhoenixConstants = None):
        super().__init__()
        self.consts = consts or PhoenixConstants()
        (
            self.SPRITE_PLAYER,
            self.BG_SPRITE,
            self.SPRITE_PLAYER_PROJECTILE,
            self.SPRITE_FLOOR,
            self.SPRITE_ENEMY1,
            self.SPRITE_ENEMY2,
            self.SPRITE_BAT_HIGH_WING,
            self.SPRITE_BAT_LOW_WING,
            self.SPRITE_BAT_2_HIGH_WING,
            self.SPRITE_BAT_2_LOW_WING,
            self.SPRITE_BOSS,
            self.SPRITE_ENEMY_PROJECTILE,
            self.DIGITS,
            self.LIFE_INDICATOR,
            self.SPRITE_RED_BLOCK,
            self.SPRITE_BLUE_BLOCK,
            self.SPRITE_GREEN_BLOCK,
            self.SPRITE_ABILITY,
            self.SPRITE_MAIN_BAT_1,
            self.SPRITE_LEFT_WING_BAT_1,
            self.SPRITE_RIGHT_WING_BAT_1,
            self.SPRITE_MAIN_BAT_2,
            self.SPRITE_LEFT_WING_BAT_2,
            self.SPRITE_RIGHT_WING_BAT_2
        ) = self.load_sprites()
    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load individual sprite frames
        player_sprites = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player.npy"))
        bg_sprites = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/pong/background.npy"))
        floor_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/floor.npy"))
        player_projectile = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player_projectile.npy"))
        bat_high_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_high_wings.npy"))
        bat_low_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_low_wings.npy"))
        bat_2_high_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_high_wings.npy"))
        bat_2_low_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_low_wings.npy"))
        enemy1_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix.npy"))
        enemy2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix_2.npy"))
        boss_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/boss.npy"))
        enemy_projectile = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_projectile.npy"))
        boss_block_red = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/red_block.npy"))
        boss_block_blue = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/blue_block.npy"))
        boss_block_green = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/green_block.npy"))
        ability = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/ability.npy"))
        main_bat_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_1_main.npy"))
        left_wing_bat_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_1_wing_left.npy"))
        right_wing_bat_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_1_wing_right.npy"))
        main_bat_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_main.npy"))
        left_wing_bat_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_wing_left.npy"))
        right_wing_bat_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_wing_right.npy"))
        SPRITE_ABILITY = ability

        SPRITE_PLAYER = jnp.expand_dims(player_sprites, axis=0)
        BG_SPRITE = jnp.expand_dims(np.zeros_like(bg_sprites), axis=0)
        SPRITE_FLOOR = jnp.expand_dims(floor_sprite, axis=0)
        SPRITE_PLAYER_PROJECTILE = jnp.expand_dims(player_projectile, axis=0)
        SPRITE_ENEMY1 = jnp.expand_dims(enemy1_sprite, axis=0)
        SPRITE_ENEMY2 = jnp.expand_dims(enemy2_sprite, axis=0)
        SPRITE_BAT_HIGH_WING = jnp.expand_dims(bat_high_wings_sprite, axis=0)
        SPRITE_BAT_LOW_WING = jnp.expand_dims(bat_low_wings_sprite, axis=0)
        SPRITE_BAT_2_HIGH_WING = jnp.expand_dims(bat_2_high_wings_sprite, axis=0)
        SPRITE_BAT_2_LOW_WING = jnp.expand_dims(bat_2_low_wings_sprite, axis=0)
        SPRITE_BOSS = jnp.expand_dims(boss_sprite, axis=0)
        SPRITE_ENEMY_PROJECTILE = jnp.expand_dims(enemy_projectile, axis=0)
        SPRITE_BLUE_BLOCK = boss_block_blue
        SPRITE_RED_BLOCK = boss_block_red
        SPRITE_GREEN_BLOCK = boss_block_green
        DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/phoenix/digits/{}.npy"))
        LIFE_INDICATOR = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/life_indicator.npy"))
        SPRITE_MAIN_BAT_1 = jnp.expand_dims(main_bat_1, axis=0)
        SPRITE_LEFT_WING_BAT_1 = jnp.expand_dims(left_wing_bat_1, axis=0)
        SPRITE_RIGHT_WING_BAT_1 = jnp.expand_dims(right_wing_bat_1, axis=0)
        SPRITE_MAIN_BAT_2 = jnp.expand_dims(main_bat_2, axis=0)
        SPRITE_LEFT_WING_BAT_2 = jnp.expand_dims(left_wing_bat_2, axis=0)
        SPRITE_RIGHT_WING_BAT_2 = jnp.expand_dims(right_wing_bat_2, axis=0)
        return (
            SPRITE_PLAYER,
            BG_SPRITE,
            SPRITE_PLAYER_PROJECTILE,
            SPRITE_FLOOR,
            SPRITE_ENEMY1,
            SPRITE_ENEMY2,
            SPRITE_BAT_HIGH_WING,
            SPRITE_BAT_LOW_WING,
            SPRITE_BAT_2_HIGH_WING,
            SPRITE_BAT_2_LOW_WING,
            SPRITE_BOSS,
            SPRITE_ENEMY_PROJECTILE,
            DIGITS,
            LIFE_INDICATOR,
            SPRITE_RED_BLOCK,
            SPRITE_BLUE_BLOCK,
            SPRITE_GREEN_BLOCK,
            SPRITE_ABILITY,
            SPRITE_MAIN_BAT_1,
            SPRITE_LEFT_WING_BAT_1,
            SPRITE_RIGHT_WING_BAT_1,
            SPRITE_MAIN_BAT_2,
            SPRITE_LEFT_WING_BAT_2,
            SPRITE_RIGHT_WING_BAT_2,
        )

    # load sprites on module layer

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        # Render background
        frame_bg = jr.get_sprite_frame(self.BG_SPRITE, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)
        # Render floor
        frame_floor = jr.get_sprite_frame(self.SPRITE_FLOOR, 0)
        raster = jr.render_at(raster, 0, 185, frame_floor)
        # Render player
        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = jr.render_at(raster, state.player_x, state.player_y, frame_player)
        # Render projectile
        frame_projectile = jr.get_sprite_frame(self.SPRITE_PLAYER_PROJECTILE, 0)
        # Render enemies
        frame_enemy_1 = jr.get_sprite_frame(self.SPRITE_ENEMY1, 0)
        frame_enemy_2 = jr.get_sprite_frame(self.SPRITE_ENEMY2, 0)
        frame_bat_high_wings = jr.get_sprite_frame(self.SPRITE_BAT_HIGH_WING, 0)
        frame_bat_low_wings = jr.get_sprite_frame(self.SPRITE_BAT_LOW_WING, 0)
        frame_bat_2_high_wings = jr.get_sprite_frame(self.SPRITE_BAT_2_HIGH_WING, 0)
        frame_bat_2_low_wings = jr.get_sprite_frame(self.SPRITE_BAT_2_LOW_WING, 0)
        frame_enemy_projectile = jr.get_sprite_frame(self.SPRITE_ENEMY_PROJECTILE, 0)
        frame_boss = jr.get_sprite_frame(self.SPRITE_BOSS, 0)

        frame_main_bat = jr.get_sprite_frame(self.SPRITE_MAIN_BAT_1, 0)
        frame_left_wing_bat_1 = jr.get_sprite_frame(self.SPRITE_LEFT_WING_BAT_1, 0)
        frame_right_wing_bat_1 = jr.get_sprite_frame(self.SPRITE_RIGHT_WING_BAT_1, 0)

        frame_main_bat_2 = jr.get_sprite_frame(self.SPRITE_MAIN_BAT_2, 0)
        frame_left_wing_bat_2 = jr.get_sprite_frame(self.SPRITE_LEFT_WING_BAT_2, 0)
        frame_right_wing_bat_2 = jr.get_sprite_frame(self.SPRITE_RIGHT_WING_BAT_2, 0)



        def render_enemy(raster, input):
            enemy_pos, wings = input
            x, y = enemy_pos

            def render_level1(r):
                return jr.render_at(r, x, y, frame_enemy_1)

            def render_level2(r):
                return jr.render_at(r, x, y, frame_enemy_2)
            def render_level3(r):
                r = jr.render_at(r, x, y, frame_main_bat)

                def no_wings(r):
                    return r

                def left_wing_only(r):
                    return jr.render_at(r, x - 5, y+2, frame_left_wing_bat_1)

                def right_wing_only(r):
                    return jr.render_at(r, x + 4, y+2, frame_right_wing_bat_1)

                def both_wings(r):
                    r = jr.render_at(r, x - 5, y+2, frame_left_wing_bat_1)
                    r = jr.render_at(r, x + 4, y+2, frame_right_wing_bat_1)
                    return r
                wing_idx = wings + 1
                r = jax.lax.switch(
                    wing_idx,
                    [
                        left_wing_only,  # 0: no wings
                        no_wings,  # 1: left wing only
                        right_wing_only,  # 2: right wing only
                        both_wings,  # 3: both wings
                    ], r
                )
                return r
            def render_level4(r):
                r = jr.render_at(r, x, y, frame_main_bat_2)

                def no_wings(r):
                    return r

                def left_wing_only(r):
                    return jr.render_at(r, x - 5, y + 2, frame_left_wing_bat_2)

                def right_wing_only(r):
                    return jr.render_at(r, x + 5, y + 2, frame_right_wing_bat_2)

                def both_wings(r):
                    r = jr.render_at(r, x - 5, y + 2, frame_left_wing_bat_2)
                    r = jr.render_at(r, x + 5, y + 2, frame_right_wing_bat_2)
                    return r

                wing_idx = wings + 1
                r = jax.lax.switch(
                    wing_idx,
                    [
                        left_wing_only,  # 0: no wings
                        no_wings,  # 1: left wing only
                        right_wing_only,  # 2: right wing only
                        both_wings,  # 3: both wings
                    ], r
                )
                return r
            def render_level5(r):
                return jr.render_at(r, x, y, frame_boss)

            def render_if_active(r):
                return jax.lax.switch(
                    state.level - 1,
                    [
                        render_level1,
                        render_level2,
                        render_level3,
                        render_level4,
                        render_level5,
                    ],
                    r
                )

            raster = jax.lax.cond(x > -1, render_if_active, lambda r: r, raster)

            return raster, None
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)
        wings_array = jnp.full((enemy_positions.shape[0],), state.bat_wings)
        inputs = (enemy_positions, wings_array)
        raster, _ = jax.lax.scan(render_enemy, raster, inputs)

        # Render player projectiles
        def render_player_projectile(r):
            return jr.render_at(r, state.projectile_x, state.projectile_y, frame_projectile)

        raster = jax.lax.cond(
            state.projectile_x > -1,
            render_player_projectile,
            lambda r: r,
            raster
        )

        def render_ability(r):
            return jax.lax.cond(
                state.invincibility,  # condition must be a scalar bool (e.g., jnp.bool_)
                lambda _: jr.render_at(r, state.player_x - 5, state.player_y - 4, self.SPRITE_ABILITY),
                lambda _: r,
                operand=None  # no operand needed
            )

        raster = jax.lax.cond(
            state.invincibility,
            render_ability,
            lambda r: r,
            raster
        )
        def render_enemy_projectile(raster, projectile_pos):
            x, y = projectile_pos
            return jax.lax.cond(
                y > -1,
                lambda r: jr.render_at(r, x, y, frame_enemy_projectile),
                lambda r: r,
                raster
            ), None
        def render_boss_block_blue(raster, block_pos):
            x,y = block_pos
            return jax.lax.cond(
                state.level% 5 == 0,
                lambda r: jr.render_at(r, x, y, self.SPRITE_BLUE_BLOCK),
                lambda r:r,
                raster
            ), None
        def render_boss_block_red(raster, block_pos):
            x,y = block_pos
            return jax.lax.cond(
                state.level% 5 == 0,
                lambda r: jr.render_at(r, x, y, self.SPRITE_RED_BLOCK),
                lambda r:r,
                raster
            ), None

        def render_boss_block_green(raster, block_pos):
            x, y = block_pos
            return jax.lax.cond(
                state.level % 5 == 0,
                lambda r: jr.render_at(r, x, y, self.SPRITE_GREEN_BLOCK),
                lambda r: r,
                raster
            ), None

        blue_block_positions = state.blue_blocks
        raster, _ = jax.lax.scan(render_boss_block_blue, raster, blue_block_positions)
        red_block_positions = state.red_blocks
        raster, _ = jax.lax.scan(render_boss_block_red, raster, red_block_positions)
        green_block_positions = state.green_blocks
        raster, _ = jax.lax.scan(render_boss_block_green, raster, green_block_positions)
        enemy_proj_positions = jnp.stack((state.enemy_projectile_x, state.enemy_projectile_y), axis=1)
        raster, _ = jax.lax.scan(render_enemy_projectile, raster, enemy_proj_positions)
        # render score
        score_array = jr.int_to_digits(state.score, max_digits=5)  # 5 for now
        raster = jr.render_label(raster, 60, 10, score_array, self.DIGITS, spacing=8)
        # render lives
        lives_value = jnp.sum(jr.int_to_digits(state.lives, max_digits=2))
        raster = jr.render_indicator(raster, 70, 20, lives_value, self.LIFE_INDICATOR, spacing=4)

        return raster




