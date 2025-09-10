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

from jaxatari.rendering.jax_rendering_utils import pad_to_match


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
    BAT_REGEN: int = 250
    BLOCK_WIDTH:int = 4
    BLOCK_HEIGHT:int = 4
    SCORE_COLOR: Tuple[int, int, int] = (210, 210, 64)
    PLAYER_BOUNDS: Tuple[int, int] = (0, 155)  # (left, right)
    ENEMY_DEATH_DURATION: int = 30 # ca. 0,5 Sekunden bei 60 FPS
    PLAYER_DEATH_DURATION: int = 90 # ca. 1,5 Sekunden bei 60 FPS
    ENEMY_PROJECTILE_SPEED: int = 2
    PLAYER_RESPAWN_DURATION: int = 360 # ca. 6 Sekunden bei 60 FPS
    ABILITY_COOLDOWN: int = 600 # ca. 10 sekunden bei 60FPS
    FIRE_CHANCE: float = 0.005
    LEVEL_TRANSITION_DURATION: int = 240 # ca. 4 Sekunden bei 60 FPS
    ENEMY_ANIMATION_SPEED: int = 30  # ca. 0,5 Sekunden bei 60 FPS
    PLAYER_ANIMATION_SPEED: int = 6  # ca. 0,1 Sekunden bei 60 FPS
    PLAYER_LIVES: int = 4 # Anzahl der Leben
    ENEMY_POSITIONS_X_LIST = [
        lambda: jnp.array(
            [123 - 160 // 2, 123 - 160 // 2, 136 - 160 // 2, 136 - 160 // 2, 160 - 160 // 2, 160 - 160 // 2,
             174 - 160 // 2, 174 - 160 // 2]).astype(jnp.float32),
        lambda: jnp.array(
            [141 - 160 // 2, 155 - 160 // 2, 127 - 160 // 2, 169 - 160 // 2, 134 - 160 // 2, 162 - 160 // 2,
             120 - 160 // 2, 176 - 160 // 2]).astype(jnp.float32),
        lambda: jnp.array(
            [123 - 160 // 2, 170 - 160 // 2, 123 - 160 // 2, 180 - 160 // 2, 123 - 160 // 2, 170 - 160 // 2,
             123 - 160 // 2, -1]).astype(jnp.float32),
        lambda: jnp.array(
            [123 - 160 // 2, 180 - 160 // 2, 123 - 160 // 2, 170 - 160 // 2, 123 - 160 // 2, 180 - 160 // 2,
             123 - 160 // 2, -1]).astype(jnp.float32),
        lambda: jnp.array([78, -1, -1, -1, -1, -1, -1, -1]).astype(jnp.float32),
    ]
    ENEMY_POSITIONS_Y_LIST = [
        lambda: jnp.array(
            [210 - 135, 210 - 153, 210 - 117, 210 - 171, 210 - 117, 210 - 171, 210 - 135,
             210 - 153]).astype(jnp.float32),
        lambda: jnp.array(
            [210 - 171, 210 - 171, 210 - 135, 210 - 135, 210 - 153, 210 - 153, 210 - 117,
             210 - 117]).astype(jnp.float32),
        lambda: jnp.array(
            [210 - 99, 210 - 117, 210 - 135, 210 - 153, 210 - 171, 210 - 63, 210 - 81,
             210 + 20]).astype(jnp.float32),
        lambda: jnp.array(
            [210 - 63, 210 - 81, 210 - 99, 210 - 117, 210 - 135, 210 - 153, 210 - 171,
             210 + 20]).astype(jnp.float32),
        lambda: jnp.array([210 - 132, 210 + 20, 210 + 20, 210 + 20, 210 + 20, 210 + 20, 210 + 20,
                           210 + 20]).astype(jnp.float32),
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
    horizontal_direction_enemies: chex.Array
    vertical_direction_enemies: chex.Array
    blue_blocks: chex.Array
    red_blocks: chex.Array
    green_blocks: chex.Array
    invincibility: chex.Array
    invincibility_timer: chex.Array
    ability_cooldown: chex.Array

    bat_wings: chex.Array
    bat_dying: chex.Array # Bat dying status, (8,), bool
    bat_death_timer: chex.Array # Timer for Bat death animation, (8,), int
    bat_wing_regen_timer: chex.Array

    phoenix_do_attack: chex.Array  # Phoenix attack state
    phoenix_attack_target_y: chex.Array  # Target Y position for Phoenix attack
    phoenix_original_y: chex.Array  # Original Y position of the Phoenix
    phoenix_cooldown: chex.Array
    phoenix_drift: chex.Array
    phoenix_returning: chex.Array # Returning status of the Phoenix
    phoenix_dying: chex.Array # Dying status of the Phoenix, (8,), bool
    phoenix_death_timer: chex.Array # Timer for Phoenix death animation, (8,), int

    player_dying: chex.Array = jnp.array(False)  # Player dying status, bool
    player_death_timer: chex.Array = jnp.array(0)  # Timer for player death animation, int
    player_moving: chex.Array = jnp.array(False) # Player moving status, bool

    projectile_x: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil
    projectile_y: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil # Gegner Y-Positionen
    enemy_projectile_x: chex.Array = jnp.full((8,), -1) # Enemy projectile X-Positionen
    enemy_projectile_y: chex.Array = jnp.full((8,), -1) # Enemy projectile Y-Positionen

    score: chex.Array = jnp.array(0)  # Score
    lives: chex.Array = jnp.array(5) # Lives
    player_respawn_timer: chex.Array = 0 # Invincibility timer
    level: chex.Array = jnp.array(1)  # Level, starts at 1
    level_transition_timer: chex.Array = jnp.array(0) # Timer for level transition

class PhoenixObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_score: chex.Array
    lives: chex.Array

class PhoenixInfo(NamedTuple):
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

class CarryState(NamedTuple):
    score: chex.Array

class EntityPosition(NamedTuple):## not sure
    x: chex.Array
    y: chex.Array

class JaxPhoenix(JaxEnvironment[PhoenixState, PhoenixObservation, PhoenixInfo, None]):
    def __init__(self, consts: PhoenixConstants = None, reward_funcs: list[callable]=None):
        consts = consts or PhoenixConstants()
        super().__init__(consts)
        self.renderer = PhoenixRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
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
    def _get_observation(self, state: PhoenixState) -> PhoenixObservation:
        player = EntityPosition(x=state.player_x, y=state.player_y)
        return PhoenixObservation(
            player_x=player[0],
            player_y=player[1],
            player_score=state.score,
            lives=state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: PhoenixState, state: PhoenixState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PhoenixState, all_rewards: jnp.ndarray) -> PhoenixInfo:
        return PhoenixInfo(
            step_counter=0,
            all_rewards=all_rewards
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PhoenixState) -> Tuple[bool, PhoenixState]:
        return jnp.less_equal(state.lives, 0)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> Space:
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=self.consts.WIDTH - 1, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=self.consts.HEIGHT - 1, shape=(), dtype=jnp.int32),
            "player_score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: PhoenixObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_x.flatten(),
            obs.player_y.flatten(),
            obs.player_score.flatten(),
            obs.lives.flatten()
        ]
        )
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
         #Ability : it holds on for ... amount
        invinsibility = jnp.any(jnp.array([action == Action.DOWN])) & (state.ability_cooldown == 0) & (state.invincibility_timer == 0)

        new_invinsibility = jnp.where(invinsibility, True, state.invincibility)
        new_timer = jnp.where(invinsibility & (state.invincibility_timer == 0), 200, state.invincibility_timer)

        new_timer = jnp.where(new_timer > 0, new_timer - 1, 0)
        new_invinsibility = jnp.where(new_timer == 0, False, new_invinsibility)

        new_cooldown = jnp.where(new_timer == 2, self.consts.ABILITY_COOLDOWN, state.ability_cooldown)
        new_cooldown = jnp.where(new_cooldown > 0, new_cooldown - 1, 0)
        # movement right
        player_x = jnp.where(
            right & jnp.logical_not(new_invinsibility), state.player_x + step_size, jnp.where(left & jnp.logical_not(new_invinsibility), state.player_x - step_size, state.player_x)
        )
        # movement left
        player_x = jnp.where(
            player_x < self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[0],
            jnp.where(player_x > self.consts.PLAYER_BOUNDS[1], self.consts.PLAYER_BOUNDS[1], player_x)
        )

        # Did the player move?
        player_moved = jnp.not_equal(player_x.astype(jnp.int32), state.player_x.astype(jnp.int32))
        new_player_moving = player_moved.astype(jnp.bool_)

        state = state._replace(player_x= player_x.astype(jnp.int32),
                               invincibility=new_invinsibility,
                               invincibility_timer=new_timer,
                                player_moving=new_player_moving,
                                 ability_cooldown=new_cooldown
        )

        return state

    def phoenix_step(self, state):
        enemy_step_size = 0.4
        attack_speed = 1#0.4
        tolerance = 0.5 #TODO Kann evtl entfernt werden

        # Nur Gegner mit gültiger Position im Spielfeld bewegen
        active_enemies = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10) & (~state.phoenix_dying)

        # Unterste aktive Phoenixe (zum Starten eines Angriffs)
        masked_enemies_y = jnp.where(active_enemies, state.enemies_y, -jnp.inf)
        max_y = jnp.max(masked_enemies_y)
        lowest_mask = (state.enemies_y == max_y) & active_enemies

        # Angriff starten nur wenn nicht bereits am Angreifen/Zurückkehren und kein Cooldown
        can_attack = (
                lowest_mask
                & (~state.phoenix_do_attack)
                & (~state.phoenix_returning)
                & (state.phoenix_cooldown == 0)
                & ((state.phoenix_original_y == -1) | (state.enemies_y == state.phoenix_original_y))
        )
        key = jax.random.PRNGKey(state.step_counter)
        attack_chance = jax.random.uniform(key, shape=()) < 0.005
        attack_trigger = lowest_mask & jnp.any(can_attack & attack_chance)

        # Zielbereich für den Angriff
        min_attack_y = jnp.max(jnp.where(active_enemies & (~state.phoenix_do_attack), state.enemies_y, -jnp.inf)) + 20
        max_attack_y = jnp.minimum(state.player_y - 10, self.consts.HEIGHT - 50)
        common_target_y = jax.random.randint(key, (), minval=min_attack_y, maxval=max_attack_y)

        new_phoenix_do_attack = jnp.where(attack_trigger, True, state.phoenix_do_attack)
        new_phoenix_attack_target_y = jnp.where(
            attack_trigger,
            jnp.full_like(state.phoenix_attack_target_y, common_target_y),
            state.phoenix_attack_target_y
        ).astype(jnp.float32)
        new_phoenix_original_y = jnp.where(attack_trigger, state.enemies_y, state.phoenix_original_y).astype(
            jnp.float32)

        # Drift nur beim Abtauchen/Anflug
        drift_prob = 0.6
        drift_max = 0.6#0.35
        num = state.enemies_x.shape[0]
        drift_key = jax.random.PRNGKey(state.step_counter + 999)
        dir_key, mag_key, on_key = jax.random.split(drift_key, 3)
        dir_sign = jnp.where(jax.random.uniform(dir_key, (num,)) < 0.5, -1.0, 1.0)
        magnitude = jax.random.uniform(mag_key, (num,)) * drift_max
        apply = jax.random.uniform(on_key, (num,)) < drift_prob
        drift_sample = jnp.where(apply, dir_sign * magnitude, 0.0).astype(jnp.float32)
        new_phoenix_drift = jnp.where(attack_trigger, drift_sample, state.phoenix_drift)

        # Angriffsbewegung (runter/hoch zum Ziel)
        going_down = new_phoenix_do_attack & (state.enemies_y < new_phoenix_attack_target_y - tolerance)
        going_up = new_phoenix_do_attack & (state.enemies_y > new_phoenix_attack_target_y + tolerance)

        # WICHTIG: Y-Bewegung nur für aktive Gegner
        new_enemies_y = jnp.where(active_enemies & going_down, state.enemies_y + attack_speed, state.enemies_y)
        new_enemies_y = jnp.where(active_enemies & going_up, new_enemies_y - attack_speed, new_enemies_y)

        # Seiten-Drift nur während des Abtauchens/an Zielanflug
        lateral_drift = jnp.where(going_down | going_up, new_phoenix_drift, 0.0).astype(jnp.float32)

        # Ziel erreicht? -> gemeinsamen "unten bleiben"-Cooldown starten
        target_reached = (~going_down) & (~going_up) & new_phoenix_do_attack
        key_delay = jax.random.PRNGKey(state.step_counter + 123)
        common_delay = jax.random.randint(key_delay, (), 30, 120)
        any_reached_target = jnp.any(target_reached & (state.phoenix_cooldown == 0))

        new_phoenix_cooldown = jnp.where(
            any_reached_target,
            jnp.full_like(state.phoenix_cooldown, common_delay),
            state.phoenix_cooldown
        )

        # Rückflug-Start wenn Cooldown abgelaufen
        start_return = target_reached & (new_phoenix_cooldown == 1)
        new_phoenix_returning = jnp.where(start_return, True, state.phoenix_returning)
        new_phoenix_do_attack = jnp.where(start_return, False, new_phoenix_do_attack)
        new_phoenix_attack_target_y = jnp.where(start_return, -1, new_phoenix_attack_target_y)

        # Rückflug: gleiches Tempo wie Angriff (nur aktive Gegner)
        returning_active = new_phoenix_returning
        dy = new_phoenix_original_y - new_enemies_y
        step = jnp.clip(dy, -attack_speed, attack_speed)
        new_enemies_y = jnp.where(active_enemies & returning_active, new_enemies_y + step, new_enemies_y)

        arrived = active_enemies & returning_active & (jnp.abs(new_enemies_y - new_phoenix_original_y) <= tolerance)
        new_enemies_y = jnp.where(arrived, new_phoenix_original_y, new_enemies_y)
        new_phoenix_returning = jnp.where(arrived, False, new_phoenix_returning)
        new_phoenix_original_y = jnp.where(arrived, -1, new_phoenix_original_y)
        new_phoenix_cooldown = jnp.where(arrived, 30, new_phoenix_cooldown)

        # Gruppenbewegung: nur während des Abtauchens ausnehmen
        group_mask = active_enemies & (~going_down)

        # Richtungswechsel nur anhand der oberen Formation
        direction_mask = active_enemies & (new_phoenix_original_y == -1)

        at_left_boundary = jnp.any(jnp.logical_and(state.enemies_x <= self.consts.PLAYER_BOUNDS[0], direction_mask))
        at_right_boundary = jnp.any(
            jnp.logical_and(
                state.enemies_x >= self.consts.PLAYER_BOUNDS[1] - self.consts.ENEMY_WIDTH / 2,
                direction_mask
            )
        )
        new_direction = jax.lax.cond(
            at_left_boundary,
            lambda: jnp.full_like(state.horizontal_direction_enemies, 1.0, dtype=jnp.float32),
            lambda: jax.lax.cond(
                at_right_boundary,
                lambda: jnp.full_like(state.horizontal_direction_enemies, -1.0, dtype=jnp.float32),
                lambda: state.horizontal_direction_enemies.astype(jnp.float32),
            ),
        )

        # Horizontale Bewegung anwenden
        group_step = jnp.where(group_mask, new_direction * enemy_step_size, 0.0).astype(jnp.float32)
        new_enemies_x = jnp.where(
            active_enemies,
            state.enemies_x + group_step + lateral_drift,
            state.enemies_x
        )
        # WICHTIG: Clipping nur für aktive Gegner, damit Tote (-1) nicht auf 0 geclippt werden
        clipped_x = jnp.clip(new_enemies_x, self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[1])
        new_enemies_x = jnp.where(active_enemies, clipped_x, state.enemies_x)

        # Cooldown am Ende einmal dekrementieren
        new_phoenix_cooldown = jnp.where(new_phoenix_cooldown > 0, new_phoenix_cooldown - 1, 0)


        state = state._replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            horizontal_direction_enemies=new_direction.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
            vertical_direction_enemies=state.vertical_direction_enemies.astype(jnp.float32),
            phoenix_do_attack=new_phoenix_do_attack,
            phoenix_attack_target_y=new_phoenix_attack_target_y.astype(jnp.float32),
            phoenix_original_y=new_phoenix_original_y.astype(jnp.float32),
            phoenix_cooldown=new_phoenix_cooldown.astype(jnp.int32),
            phoenix_drift=new_phoenix_drift.astype(jnp.float32),
            phoenix_returning=new_phoenix_returning.astype(jnp.bool_),
            phoenix_dying=state.phoenix_dying.astype(jnp.bool_),
            phoenix_death_timer=state.phoenix_death_timer.astype(jnp.int32),
            player_dying=state.player_dying.astype(jnp.bool_),
            player_death_timer=state.player_death_timer.astype(jnp.int32),
        )
        return state

    def bat_step(self, state):
        bat_step_size = 0.5
        active_bats = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10) & (~state.bat_dying)
        proj_pos = jnp.array([state.projectile_x, state.projectile_y])

        # Initialisiere neue Richtungen für jede Fledermaus
        new_directions = jnp.where(
            jnp.logical_and(state.enemies_x <= self.consts.PLAYER_BOUNDS[0] + 3, active_bats),
            jnp.ones(state.horizontal_direction_enemies.shape, dtype=jnp.float32),  # Force array shape
            jnp.where(
                jnp.logical_and(state.enemies_x >= self.consts.PLAYER_BOUNDS[1] - self.consts.ENEMY_WIDTH / 2,
                                active_bats),
                jnp.ones(state.horizontal_direction_enemies.shape, dtype=jnp.float32) * -1,  # Force array shape
                state.horizontal_direction_enemies.astype(jnp.float32)  # Ensure consistency
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

        no_wings = (new_bat_wings == 0) & active_bats
        new_regen_timer = jnp.where(no_wings, state.bat_wing_regen_timer + 1, 0)
        regenerated = (new_regen_timer >= self.consts.BAT_REGEN)
        new_bat_wings = jnp.where(regenerated, 2, new_bat_wings)
        new_regen_timer = jnp.where(regenerated, 0, new_regen_timer)

        state = state._replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            enemies_y=state.enemies_y.astype(jnp.float32),
            horizontal_direction_enemies=new_directions.astype(jnp.float32),
            projectile_y=new_proj_y,
            bat_wings= new_bat_wings,
            bat_wing_regen_timer=new_regen_timer,
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
            blue_blocks=new_blue_blocks.astype(jnp.int32),
            red_blocks=new_red_blocks.astype(jnp.int32),
            green_blocks=new_green_blocks.astype(jnp.int32),
            projectile_x=projectile_x.astype(jnp.int32),
            projectile_y=projectile_y.astype(jnp.int32),
            enemies_x = state.enemies_x.astype(jnp.float32),
        )
        return state

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PhoenixObservation, PhoenixState]:

        return_state = PhoenixState(
            player_x=jnp.array(self.consts.PLAYER_POSITION[0], dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_POSITION[1], dtype=jnp.int32),
            step_counter=jnp.array(0),
            enemies_x = self.consts.ENEMY_POSITIONS_X_LIST[0](),
            enemies_y = self.consts.ENEMY_POSITIONS_Y_LIST[0](),
            horizontal_direction_enemies = jnp.full((8,), -1.0),
            vertical_direction_enemies = jnp.full((8,), 1.0),
            enemy_projectile_x=jnp.full((8,), -1),
            enemy_projectile_y=jnp.full((8,), -1),
            projectile_x=jnp.array(-1),  # Standardwert: kein Projektil
            score = jnp.array(0), # Standardwert: Score=0
            lives=jnp.array(self.consts.PLAYER_LIVES), # Standardwert: 4 Leben
            player_respawn_timer=jnp.array(5),
            level=jnp.array(1),
            level_transition_timer=jnp.array(0),  # Timer for level transition, starts at 0

            invincibility=jnp.array(False),
            invincibility_timer=jnp.array(0),
            ability_cooldown=jnp.array(0),

            bat_wings=jnp.full((8,), 2),
            bat_dying=jnp.full((8,), False, dtype=jnp.bool), # Bat dying status, (8,), bool
            bat_death_timer=jnp.full((8,), 0, dtype=jnp.int32), # Timer for Bat death animation, (8,), int
            bat_wing_regen_timer=jnp.full((8,), 0, dtype=jnp.int32),
            phoenix_do_attack = jnp.full((8,), 0, dtype=jnp.bool),  # Phoenix attack state
            phoenix_attack_target_y = jnp.full((8,), -1, dtype=jnp.float32),  # Target Y position for Phoenix attack
            phoenix_original_y = jnp.full((8,), -1, dtype=jnp.float32),  # Original Y position of the Phoenix
            phoenix_cooldown=jnp.full((8,), 0),  # Cooldown für Phoenix-Angriff
            phoenix_drift=jnp.full((8,), 0.0, dtype=jnp.float32),  # Drift-Werte für Phoenix
            phoenix_returning=jnp.full((8,), False, dtype=jnp.bool),  # Returning status of the Phoenix
            phoenix_dying=jnp.full((8,), False, dtype=jnp.bool),  # Dying status of the Phoenix
            phoenix_death_timer=jnp.full((8,), 0, dtype=jnp.int32),  # Timer for Phoenix death animation

            player_dying=jnp.array(False, dtype = jnp.bool),  # Player dying status, bool
            player_death_timer=jnp.array(0, dtype = jnp.int32),  # Timer for player death animation, int
            player_moving=jnp.array(False, dtype = jnp.bool), # Player moving status, bool

            # Initialierung der Blockpositionen
            blue_blocks=self.consts.BLUE_BLOCK_POSITIONS.astype(jnp.int32),
            red_blocks=self.consts.RED_BLOCK_POSITIONS.astype(jnp.int32),
            green_blocks = self.consts.GREEN_BLOCK_POSITIONS.astype(jnp.int32),
        )

        initial_obs = self._get_observation(return_state)
        return initial_obs, return_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self,state, action: Action) -> Tuple[PhoenixObservation, PhoenixState, float, bool, PhoenixInfo]:
        new_respawn_timer = jnp.where(state.player_respawn_timer > 0, state.player_respawn_timer - 1, 0)
        respawn_ended = (state.player_respawn_timer > 0) & (new_respawn_timer == 0)

        state = state._replace(player_respawn_timer=new_respawn_timer.astype(jnp.int32))

        state = jax.lax.cond(
            jnp.logical_or(state.player_dying, state.player_respawn_timer > 0),
            lambda s: s,
            lambda s: self.player_step(s, action),
            state
        ) # Player_step only if not dying

        projectile_active = state.projectile_y >= 0

        # Can fire only if inactive
        can_fire = (~projectile_active) & (~state.player_dying) & (state.player_respawn_timer <= 0)
        fire_actions = jnp.array([
            action == Action.FIRE,
            action == Action.LEFTFIRE,
            action == Action.RIGHTFIRE,
            action == Action.UPFIRE,
            action == Action.DOWNFIRE,
            action == Action.UPLEFTFIRE,
            action == Action.UPRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.DOWNRIGHTFIRE,
        ])
        firing = jnp.any(fire_actions) & can_fire

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

        key, fire_chance = generate_fire_key_and_chance(state.step_counter, self.consts.FIRE_CHANCE)

        # Random decision: should each enemy fire?
        enemy_should_fire = jax.random.uniform(key, (8,)) < fire_chance

        # Fire only from active enemies
        can_fire = (state.enemy_projectile_y < 0) & (state.enemies_x > -1)
        not_attacking = jnp.logical_not(jnp.logical_or(state.phoenix_do_attack, state.phoenix_returning))
        not_attacking = not_attacking & (~state.phoenix_returning)
        enemy_fire_mask = enemy_should_fire & can_fire & not_attacking


        # Fire from current enemy positions
        enemy_projectile_x = jnp.where(enemy_fire_mask, state.enemies_x + self.consts.ENEMY_WIDTH // 2,
                                           state.enemy_projectile_x)
        enemy_projectile_y = jnp.where(enemy_fire_mask, state.enemies_y + self.consts.ENEMY_HEIGHT, state.enemy_projectile_y)

        # Move enemy projectiles downwards
        enemy_projectile_y = jnp.where(state.enemy_projectile_y >= 0, state.enemy_projectile_y + self.consts.ENEMY_PROJECTILE_SPEED,
                                           enemy_projectile_y)

        # Remove enemy projectile if off-screen
        enemy_projectile_y = jnp.where(enemy_projectile_y > 185 - self.consts.PROJECTILE_HEIGHT, -1, enemy_projectile_y) # TODO 185 durch Konstante ersetzen, die global geändert werden kann.



        projectile_pos = jnp.array([projectile_x, projectile_y])
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            projectile_x, projectile_y = projectile_pos

            collision_x = (projectile_x + self.consts.PROJECTILE_WIDTH > enemy_x) & (projectile_x < enemy_x + self.consts.ENEMY_WIDTH)
            collision_y = (projectile_y + self.consts.PROJECTILE_HEIGHT > enemy_y) & (projectile_y < enemy_y + self.consts.ENEMY_HEIGHT)
            return collision_x & collision_y


        # Kollisionsprüfung Gegner
        enemy_collisions_raw = jax.vmap(lambda enemy_pos: check_collision(enemy_pos, projectile_pos))(enemy_positions)
        is_bat_level = jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4)
        dying_mask = jnp.where(is_bat_level, state.bat_dying, state.phoenix_dying)
        enemy_collisions = enemy_collisions_raw & (~dying_mask)
        enemy_hit_detected = jnp.any(enemy_collisions)

        # Phoenix-Death-Animation starten (nur Phoenix-Levels)
        p_hit_mask = enemy_collisions & (~is_bat_level)
        new_phoenix_dying = jnp.where(p_hit_mask, True, state.phoenix_dying)
        new_phoenix_death_timer = jnp.where(
            p_hit_mask, self.consts.ENEMY_DEATH_DURATION, state.phoenix_death_timer
        )
        p_dec_timer = jnp.where(
            new_phoenix_dying & (new_phoenix_death_timer > 0),
            new_phoenix_death_timer - 1,
            new_phoenix_death_timer,
        )
        p_death_done = new_phoenix_dying & (p_dec_timer == 0)
        new_phoenix_dying = jnp.where(p_death_done, False, new_phoenix_dying)
        p_dec_timer = jnp.where(p_death_done, 0, p_dec_timer)

        # Bat-Death-Animation starten (nur Bat-Levels)
        b_hit_mask = enemy_collisions & is_bat_level
        new_bat_dying = jnp.where(b_hit_mask, True, state.bat_dying)
        new_bat_death_timer = jnp.where(
            b_hit_mask, self.consts.ENEMY_DEATH_DURATION, state.bat_death_timer
        )
        b_dec_timer = jnp.where(
            new_bat_dying & (new_bat_death_timer > 0),
            new_bat_death_timer - 1,
            new_bat_death_timer,
        )
        b_death_done = new_bat_dying & (b_dec_timer == 0)
        new_bat_dying = jnp.where(b_death_done, False, new_bat_dying)
        b_dec_timer = jnp.where(b_death_done, 0, b_dec_timer)

        # Phoenix-Angriffsstatus nur in Phoenix-Levels zurücksetzen
        phoenix_do_attack = jnp.where(p_hit_mask, False, state.phoenix_do_attack)
        phoenix_attack_target_y = jnp.where(p_hit_mask, -1, state.phoenix_attack_target_y)
        phoenix_original_y = jnp.where(p_hit_mask, -1, state.phoenix_original_y)

        # Projektil/Score bei jedem Treffer (egal welches Level)
        projectile_x = jnp.where(enemy_hit_detected, -1, projectile_x)
        projectile_y = jnp.where(enemy_hit_detected, -1, projectile_y)
        score = jnp.where(enemy_hit_detected, state.score + 20, state.score)

        # Gegner entfernen nach Ablauf der jeweiligen Death-Animation
        death_done_any = jnp.where(is_bat_level, b_death_done, p_death_done)
        enemies_x = jnp.where(death_done_any, -1, state.enemies_x)
        enemies_y = jnp.where(death_done_any, self.consts.HEIGHT + 20, state.enemies_y)

        score = jnp.where(enemy_hit_detected, state.score + 20, state.score)


        # Checken ob alle Gegner getroffen wurden
        #all_enemies_hit = jnp.all(enemies_y >= self.consts.HEIGHT + 10)
        #new_level = jnp.where(all_enemies_hit, (state.level % 5) + 1, state.level)
        #new_enemies_x = jax.lax.cond(
        #    all_enemies_hit,
        #    lambda: jax.lax.switch((new_level -1 )% 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32),
        #    lambda: state.enemies_x.astype(jnp.float32)
        #)
        #new_enemies_y = jax.lax.cond(
        #    all_enemies_hit,
        #    lambda: jax.lax.switch((new_level -1 )% 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32),
        #    lambda: enemies_y.astype(jnp.float32)
        #)
        #enemies_x = new_enemies_x
        #enemies_y = new_enemies_y
        #level = new_level

        # 1) Level-Übergangstimer starten/fortschreiben
        all_enemies_cleared = jnp.all(enemies_y >= self.consts.HEIGHT + 10)
        start_transition = all_enemies_cleared & (state.level_transition_timer == 0)

        new_level_transition_timer = jnp.where(
            start_transition,
            self.consts.LEVEL_TRANSITION_DURATION,
            state.level_transition_timer
        )
        new_level_transition_timer = jnp.where(new_level_transition_timer > 0, new_level_transition_timer - 1, 0)

        transition_ended = (state.level_transition_timer > 0) & (new_level_transition_timer == 0)

        # 2) Nächstes Level vormerken und erst bei Timerende aktivieren
        pending_next_level = (state.level % 5) + 1
        level = jnp.where(transition_ended, pending_next_level, state.level)

        # 3) Gegner-Formationen nur bei Timerende spawnen
        next_enemies_x = jax.lax.switch((pending_next_level - 1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(
            jnp.float32)
        next_enemies_y = jax.lax.switch((pending_next_level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(
            jnp.float32)


        reset_mask = transition_ended
        enemies_x = jnp.where(reset_mask, next_enemies_x.astype(jnp.float32), enemies_x)
        enemies_y = jnp.where(reset_mask, next_enemies_y.astype(jnp.float32), enemies_y)

        # Richtungen der Formation zurücksetzen
        new_horizontal_direction_enemies = jnp.where(
            reset_mask, jnp.full((8,), -1.0, dtype=jnp.float32), state.horizontal_direction_enemies
        )
        new_vertical_direction_enemies = jnp.where(
            reset_mask, jnp.full((8,), 1.0, dtype=jnp.float32), state.vertical_direction_enemies
        )

        # Death-/Timer-/Flügel-Status zurücksetzen
        new_phoenix_dying = jnp.where(reset_mask, jnp.full((8,), False), new_phoenix_dying)
        p_dec_timer = jnp.where(reset_mask, jnp.full((8,), 0), p_dec_timer)

        new_bat_dying = jnp.where(reset_mask, jnp.full((8,), False), new_bat_dying)
        b_dec_timer = jnp.where(reset_mask, jnp.full((8,), 0), b_dec_timer)
        new_bat_wings = jnp.where(reset_mask, jnp.full((8,), 2, dtype=jnp.int32), state.bat_wings)

        # Boss-Blöcke nur beim Eintritt in das Boss-Level neu initialisieren
        enter_boss_next = ((pending_next_level % 5) == 0)
        reset_blocks = reset_mask & enter_boss_next
        blue_blocks = jnp.where(reset_blocks, self.consts.BLUE_BLOCK_POSITIONS.astype(jnp.float32), state.blue_blocks)
        red_blocks = jnp.where(reset_blocks, self.consts.RED_BLOCK_POSITIONS.astype(jnp.float32), state.red_blocks)
        green_blocks = jnp.where(reset_blocks, self.consts.GREEN_BLOCK_POSITIONS.astype(jnp.float32), state.green_blocks)

        # Gegner-Respawn nach Spieler-Respawn nur, wenn kein Level-Übergang läuft
        enemy_respawn_x = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32)
        enemy_respawn_y = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32)

        enemy_respawn_mask = respawn_ended & (new_level_transition_timer == 0)
        enemy_alive_mask = (enemies_x > -1) & (enemies_y < self.consts.HEIGHT + 10)
        enemies_x = jnp.where(enemy_respawn_mask & enemy_alive_mask, enemy_respawn_x, enemies_x)
        enemies_y = jnp.where(enemy_respawn_mask & enemy_alive_mask, enemy_respawn_y, enemies_y)



        is_vulnerable = (new_respawn_timer <= 0) & (~state.player_dying) & (~state.invincibility)

        def check_player_hit(projectile_xs, projectile_ys, player_x, player_y):
            def is_hit(px, py):
                hit_x = (px + self.consts.PROJECTILE_WIDTH > player_x) & (px < player_x + 5) # TODO 5 durch Konstante ersetzen, die global geändert werden kann.
                hit_y = (py + self.consts.PROJECTILE_HEIGHT > player_y) & (py < player_y + self.consts.PROJECTILE_HEIGHT)
                return hit_x & hit_y

            hits = jax.vmap(is_hit)(projectile_xs, projectile_ys)
            return jnp.any(hits)



        # Kollisionsüberprüfung Spieler
        player_hit_detected = jnp.where(
            is_vulnerable & (state.invincibility == jnp.array(False)),
            check_player_hit(enemy_projectile_x, enemy_projectile_y, state.player_x, state.player_y),
            False
        )

        # Bei Treffer: Spieler-Dying-Status setzen und Timer starten
        player_death_duration = self.consts.PLAYER_DEATH_DURATION
        new_player_dying = jnp.where(player_hit_detected, True, state.player_dying)
        player_death_timer_start = jnp.where(player_hit_detected, player_death_duration, state.player_death_timer)

        lives = jnp.where(player_hit_detected, state.lives - 1, state.lives)


        # Enemy Projectile entfernen wenn eine Kollision mit dem Spieler erkannt wurde
        enemy_projectile_x = jnp.where(player_hit_detected, -1, enemy_projectile_x)
        enemy_projectile_y = jnp.where(player_hit_detected, -1, enemy_projectile_y)

        # Player-Death-Teimer herunterzählen
        dec_player_timer = jnp.where(
            new_player_dying & (player_death_timer_start > 0),
            player_death_timer_start - 1,
            player_death_timer_start
        )
        player_death_done = new_player_dying & (dec_player_timer == 0) & (player_death_timer_start > 0)

        player_x = jnp.where(player_death_done, self.consts.PLAYER_POSITION[0], state.player_x)
        player_respawn_timer = jnp.where(
            player_death_done,
            self.consts.PLAYER_RESPAWN_DURATION,
            new_respawn_timer
        )

        new_player_moving = jnp.where(
            jnp.logical_or(new_player_dying, player_respawn_timer > 0),
            jnp.array(False, dtype=jnp.bool_),
            state.player_moving
        )

        #enemy_respawn_x = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32)
        #enemy_respawn_y = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32)

        #enemies_x = jnp.where(respawn_ended, enemy_respawn_x, enemies_x)
        #enemies_y = jnp.where(respawn_ended, enemy_respawn_y, enemies_y)

        new_player_dying = jnp.where(player_death_done, False, new_player_dying).astype(jnp.bool_)
        new_player_death_timer = jnp.where(player_death_done, 0, dec_player_timer).astype(jnp.int32)

        formation_reset = transition_ended | (respawn_ended & (new_level_transition_timer == 0))
        new_phoenix_do_attack = jnp.where(formation_reset, jnp.full((8,), False), state.phoenix_do_attack)
        new_phoenix_returning = jnp.where(formation_reset, jnp.full((8,), False), state.phoenix_returning)
        new_phoenix_attack_target = jnp.where(formation_reset, jnp.full((8,), -1.0), state.phoenix_attack_target_y)
        new_phoenix_cooldown = jnp.where(formation_reset, jnp.full((8,), 0), state.phoenix_cooldown)
        new_phoenix_drift = jnp.where(formation_reset, jnp.full((8,), 0.0), state.phoenix_drift)
        new_phoenix_original_y = jnp.where(formation_reset, jnp.full((8,), -1.0), state.phoenix_original_y)

        return_state = PhoenixState(
            player_x = player_x,
            player_y = state.player_y,
            step_counter = state.step_counter + 1,
            projectile_x = projectile_x,
            projectile_y = projectile_y,
            enemies_x = enemies_x,
            enemies_y = enemies_y,
            horizontal_direction_enemies = new_horizontal_direction_enemies,
            score= score,
            enemy_projectile_x=enemy_projectile_x.astype(jnp.int32),
            enemy_projectile_y=enemy_projectile_y.astype(jnp.int32),
            lives=lives,
            player_respawn_timer = player_respawn_timer,
            level = level,
            vertical_direction_enemies=new_vertical_direction_enemies,
            blue_blocks=blue_blocks.astype(jnp.int32),
            red_blocks=red_blocks.astype(jnp.int32),
            green_blocks=green_blocks.astype(jnp.int32),
            invincibility=state.invincibility,
            invincibility_timer=state.invincibility_timer,
            bat_wings=new_bat_wings,
            bat_dying=new_bat_dying,
            bat_death_timer=b_dec_timer,
            phoenix_do_attack=new_phoenix_do_attack,
            phoenix_attack_target_y=new_phoenix_attack_target,
            phoenix_original_y=new_phoenix_original_y,
            phoenix_cooldown=new_phoenix_cooldown,
            phoenix_drift=new_phoenix_drift,
            phoenix_returning=new_phoenix_returning,
            phoenix_dying=new_phoenix_dying,
            phoenix_death_timer=p_dec_timer,
            player_dying=new_player_dying,
            player_death_timer=new_player_death_timer,
            player_moving=new_player_moving,
            level_transition_timer=new_level_transition_timer,
            ability_cooldown=state.ability_cooldown,
            bat_wing_regen_timer=state.bat_wing_regen_timer,

        )
        observation = self._get_observation(return_state)
        env_reward = jnp.where(enemy_hit_detected, 1.0, 0.0)
        done = self._get_done(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)
        return observation, return_state, env_reward, done, info

    def render(self, state:PhoenixState) -> jnp.ndarray:
        return self.renderer.render(state)
from jaxatari.renderers import JAXGameRenderer

class PhoenixRenderer(JAXGameRenderer):
    def __init__(self, consts: PhoenixConstants = None):
        super().__init__()
        self.consts = consts or PhoenixConstants()
        (
            # --- FIELD SPRITES ---
            self.BG_SPRITE,
            self.SPRITE_FLOOR,
            # --- UI SPRITES ---
            self.DIGITS,
            self.LIFE_INDICATOR,
            # --- PLAYER SPRITES ---
            self.SPRITE_PLAYER,
            self.SPRITE_PLAYER_DEATH_1,
            self.SPRITE_PLAYER_DEATH_2,
            self.SPRITE_PLAYER_DEATH_3,
            self.SPRITE_PLAYER_MOVE,
            self.SPRITE_PLAYER_ABILITY,
            # --- PROJECTILE SPRITES ---
            self.SPRITE_PLAYER_PROJECTILE,
            self.SPRITE_ENEMY_PROJECTILE,
            # --- PHOENIX SPRITES ---
            self.SPRITE_PHOENIX_1,
            self.SPRITE_PHOENIX_2,
            self.SPRITE_PHOENIX_ATTACK,
            self.SPRITE_PHOENIX_DEATH_1,
            self.SPRITE_PHOENIX_DEATH_2,
            # --- BAT BLUE SPRITES ---
            self.SPRITE_BAT_BLUE_MAIN,
            self.SPRITE_BAT_BLUE_LEFT_WING_MIDDLE,
            self.SPRITE_BAT_BLUE_RIGHT_WING_MIDDLE,
            self.SPRITE_BAT_BLUE_LEFT_WING_UP,
            self.SPRITE_BAT_BLUE_RIGHT_WING_UP,
            self.SPRITE_BAT_BLUE_LEFT_WING_DOWN,
            self.SPRITE_BAT_BLUE_RIGHT_WING_DOWN,
            self.SPRITE_BAT_BLUE_LEFT_WING_DOWN_2,
            self.SPRITE_BAT_BLUE_RIGHT_WING_DOWN_2,
            self.SPRITE_BAT_BLUE_DEATH_1,
            self.SPRITE_BAT_BLUE_DEATH_2,
            self.SPRITE_BAT_BLUE_DEATH_3,
            # --- BAT RED SPRITES ---
            self.SPRITE_BAT_RED_MAIN,
            self.SPRITE_BAT_RED_LEFT_WING_MIDDLE,
            self.SPRITE_BAT_RED_RIGHT_WING_MIDDLE,
            self.SPRITE_BAT_RED_LEFT_WING_UP,
            self.SPRITE_BAT_RED_RIGHT_WING_UP,
            self.SPRITE_BAT_RED_LEFT_WING_DOWN,
            self.SPRITE_BAT_RED_RIGHT_WING_DOWN,
            self.SPRITE_BAT_RED_LEFT_WING_DOWN_2,
            self.SPRITE_BAT_RED_RIGHT_WING_DOWN_2,
            self.SPRITE_BAT_RED_DEATH_1,
            self.SPRITE_BAT_RED_DEATH_2,
            self.SPRITE_BAT_RED_DEATH_3,
            # --- OLD BAT SPRITES -- TODO: remove
            self.SPRITE_MAIN_BAT_1,
            self.SPRITE_LEFT_WING_BAT_1,
            self.SPRITE_RIGHT_WING_BAT_1,
            self.SPRITE_MAIN_BAT_2,
            self.SPRITE_LEFT_WING_BAT_2,
            self.SPRITE_RIGHT_WING_BAT_2,
            # --- BOSS SPRITES ---
            self.SPRITE_BOSS,
            self.SPRITE_RED_BLOCK,
            self.SPRITE_BLUE_BLOCK,
            self.SPRITE_GREEN_BLOCK,

        ) = self.load_sprites()
    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load individual sprite frames
        # --- LOAD PLAYER SPRITES ---
        player_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player/player.npy"))
        player_move_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player/player_move.npy"))
        player_death_1_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player/player_death_1.npy"))
        player_death_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player/player_death_2.npy")) # TODO Testen ob konkatinieren der zusammengehören Sprites zu einem funktioniert
        player_ability = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/ability.npy"))
        player_death_3_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player/player_death_3.npy"))
        # --- LOAD FIELD SPRITES ---
        bg_sprites = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/pong/background.npy"))
        floor_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/floor.npy"))
        # --- LOAD PROJECTILE SPRITES ---
        player_projectile = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/projectiles/player_projectile.npy"))
        enemy_projectile = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/projectiles/enemy_projectile.npy"))
        # --- LOAD PHOENIX SPRITES ---
        enemy_phoenix_1_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix/enemy_phoenix.npy"))
        enemy_phoenix_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix/enemy_phoenix_2.npy"))
        enemy_phoenix_attack = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix/enemy_phoenix_attack.npy"))
        enemy_phoenix_death_sprite_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix/enemy_phoenix_death_1.npy"))
        enemy_phoenix_death_sprite_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix/enemy_phoenix_death_2.npy"))
        # --- LOAD BLUE BATS SPRITES ---
        bat_blue_main_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_main.npy"))
        bat_blue_left_wing_middle_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_left_wing_middle.npy"))
        bat_blue_right_wing_middle_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_right_wing_middle.npy"))
        bat_blue_left_wing_up_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_left_wing_up.npy"))
        bat_blue_right_wing_up_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_right_wing_up.npy"))
        bat_blue_left_wing_down_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_left_wing_down.npy"))
        bat_blue_right_wing_down_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_right_wing_down.npy"))
        bat_blue_left_wing_down_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_left_wing_down_2.npy"))
        bat_blue_right_wing_down_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_right_wing_down_2.npy"))
        bat_blue_death_1_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_death_1.npy"))
        bat_blue_death_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_death_2.npy"))
        bat_blue_death_3_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_blue/bat_blue_death_3.npy"))
        # --- LOAD RED BATS SPRITES ---
        bat_red_main_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_main.npy"))
        bat_red_left_wing_middle_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_left_wing_middle.npy"))
        bat_red_right_wing_middle_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_right_wing_middle.npy"))
        bat_red_left_wing_up_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_left_wing_up.npy"))
        bat_red_right_wing_up_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_right_wing_up.npy"))
        bat_red_left_wing_down_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_left_wing_down.npy"))
        bat_red_right_wing_down_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_right_wing_down.npy"))
        bat_red_left_wing_down_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_left_wing_down_2.npy"))
        bat_red_right_wing_down_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_right_wing_down_2.npy"))
        bat_red_death_1_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_death_1.npy"))
        bat_red_death_2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_death_2.npy"))
        bat_red_death_3_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_red/bat_red_death_3.npy"))

        # --- OLD BAT SPRITES --- TODO: Remove
        main_bat_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_1_main.npy"))
        left_wing_bat_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_1_wing_left.npy"))
        right_wing_bat_1 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_1_wing_right.npy"))
        main_bat_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_2_main.npy"))
        left_wing_bat_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_2_wing_left.npy"))
        right_wing_bat_2 = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_bats/bats_2_wing_right.npy"))

        # --- LOAD BOSS SPRITES ---
        boss_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/boss/boss.npy"))
        boss_block_red = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/boss/red_block.npy"))
        boss_block_blue = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/boss/blue_block.npy"))
        boss_block_green = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/boss/green_block.npy"))
        # --- PADDED ANIMATION SPRITES ---
        phoenix_sprites_to_pad = [enemy_phoenix_1_sprite, enemy_phoenix_2_sprite, enemy_phoenix_attack, enemy_phoenix_death_sprite_1, enemy_phoenix_death_sprite_2]
        padded_phoenix_sprites, _ = pad_to_match(phoenix_sprites_to_pad)
        enemy_phoenix_1_sprite, enemy_phoenix_2_sprite, enemy_phoenix_attack, enemy_phoenix_death_sprite_1, enemy_phoenix_death_sprite_2 = padded_phoenix_sprites

        player_sprites_to_pad = [player_sprite, player_death_1_sprite, player_death_2_sprite, player_death_3_sprite, player_move_sprite]
        padded_player_sprites, _ = pad_to_match(player_sprites_to_pad)
        player_sprite, player_death_1_sprite, player_death_2_sprite, player_death_3_sprite, player_move_sprite = padded_player_sprites

        bat_blue_wing_sprites_to_pad = [
            bat_blue_left_wing_middle_sprite, bat_blue_right_wing_middle_sprite,
            bat_blue_left_wing_up_sprite, bat_blue_right_wing_up_sprite,
            bat_blue_left_wing_down_sprite, bat_blue_right_wing_down_sprite,
            bat_blue_left_wing_down_2_sprite, bat_blue_right_wing_down_2_sprite
        ]
        padded_bat_blue_wings, _ = pad_to_match(bat_blue_wing_sprites_to_pad)
        bat_blue_left_wing_middle_sprite, bat_blue_right_wing_middle_sprite, bat_blue_left_wing_up_sprite, bat_blue_right_wing_up_sprite, bat_blue_left_wing_down_sprite, bat_blue_right_wing_down_sprite, bat_blue_left_wing_down_2_sprite, bat_blue_right_wing_down_2_sprite = padded_bat_blue_wings

        bat_blue_sprites_to_pad = [bat_blue_main_sprite, bat_blue_death_1_sprite, bat_blue_death_2_sprite, bat_blue_death_3_sprite]
        padded_bat_blue_sprites, _ = pad_to_match(bat_blue_sprites_to_pad)
        bat_blue_main_sprite, bat_blue_death_1_sprite, bat_blue_death_2_sprite, bat_blue_death_3_sprite = padded_bat_blue_sprites

        bat_red_wing_sprites_to_pad = [
            bat_red_left_wing_middle_sprite, bat_red_right_wing_middle_sprite,
            bat_red_left_wing_up_sprite, bat_red_right_wing_up_sprite,
            bat_red_left_wing_down_sprite, bat_red_right_wing_down_sprite,
            bat_red_left_wing_down_2_sprite, bat_red_right_wing_down_2_sprite
        ]
        padded_bat_red_wings, _ = pad_to_match(bat_red_wing_sprites_to_pad)
        bat_red_left_wing_middle_sprite, bat_red_right_wing_middle_sprite, bat_red_left_wing_up_sprite, bat_red_right_wing_up_sprite, bat_red_left_wing_down_sprite, bat_red_right_wing_down_sprite, bat_red_left_wing_down_2_sprite, bat_red_right_wing_down_2_sprite = padded_bat_red_wings

        bat_red_wing_sprites_to_pad = [bat_red_main_sprite, bat_red_death_1_sprite, bat_red_death_2_sprite, bat_red_death_3_sprite]
        padded_bat_red_sprites, _ = pad_to_match(bat_red_wing_sprites_to_pad)
        bat_red_main_sprite, bat_red_death_1_sprite, bat_red_death_2_sprite, bat_red_death_3_sprite = padded_bat_red_sprites

        # --- PLAYER SPRITES ---
        SPRITE_PLAYER = jnp.expand_dims(player_sprite, axis=0)
        SPRITE_PLAYER_DEATH_1 = jnp.expand_dims(player_death_1_sprite, axis=0)
        SPRITE_PLAYER_DEATH_2 = jnp.expand_dims(player_death_2_sprite, axis=0)
        SPRITE_PLAYER_DEATH_3 = jnp.expand_dims(player_death_3_sprite, axis=0)
        SPRITE_PLAYER_MOVE = jnp.expand_dims(player_move_sprite, axis=0)
        SPRITE_PLAYER_ABILITY = jnp.expand_dims(player_ability, axis=0)
        # --- FIELD SPRITES ---
        BG_SPRITE = jnp.expand_dims(np.zeros_like(bg_sprites), axis=0)
        SPRITE_FLOOR = jnp.expand_dims(floor_sprite, axis=0)
        # --- UI SPRITES ---
        DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/phoenix/digits/{}.npy"))
        LIFE_INDICATOR = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/life_indicator.npy"))
        # --- PROJECTILE SPRITES ---
        SPRITE_PLAYER_PROJECTILE = jnp.expand_dims(player_projectile, axis=0)
        SPRITE_ENEMY_PROJECTILE = jnp.expand_dims(enemy_projectile, axis=0)
        # --- PHOENIX SPRITES ---
        SPRITE_PHOENIX_1 = jnp.expand_dims(enemy_phoenix_1_sprite, axis=0)
        SPRITE_PHOENIX_2 = jnp.expand_dims(enemy_phoenix_2_sprite, axis=0)
        SPRITE_PHOENIX_ATTACK = jnp.expand_dims(enemy_phoenix_attack, axis=0)
        SPRITE_PHOENIX_DEATH_1 = jnp.expand_dims(enemy_phoenix_death_sprite_1, axis=0)
        SPRITE_PHOENIX_DEATH_2 = jnp.expand_dims(enemy_phoenix_death_sprite_2, axis=0)
        # --- BAT BLUE SPRITES ---
        SPRITE_BAT_BLUE_MAIN = jnp.expand_dims(bat_blue_main_sprite, axis=0)
        SPRITE_BAT_BLUE_LEFT_WING_MIDDLE = jnp.expand_dims(bat_blue_left_wing_middle_sprite, axis=0)
        SPRITE_BAT_BLUE_RIGHT_WING_MIDDLE = jnp.expand_dims(bat_blue_right_wing_middle_sprite, axis=0)
        SPRITE_BAT_BLUE_LEFT_WING_UP = jnp.expand_dims(bat_blue_left_wing_up_sprite, axis=0)
        SPRITE_BAT_BLUE_RIGHT_WING_UP = jnp.expand_dims(bat_blue_right_wing_up_sprite, axis=0)
        SPRITE_BAT_BLUE_LEFT_WING_DOWN = jnp.expand_dims(bat_blue_left_wing_down_sprite, axis=0)
        SPRITE_BAT_BLUE_RIGHT_WING_DOWN = jnp.expand_dims(bat_blue_right_wing_down_sprite, axis=0)
        SPRITE_BAT_BLUE_LEFT_WING_DOWN_2 = jnp.expand_dims(bat_blue_left_wing_down_2_sprite, axis=0)
        SPRITE_BAT_BLUE_RIGHT_WING_DOWN_2 = jnp.expand_dims(bat_blue_right_wing_down_2_sprite, axis=0)
        SPRITE_BAT_BLUE_DEATH_1 = jnp.expand_dims(bat_blue_death_1_sprite, axis=0)
        SPRITE_BAT_BLUE_DEATH_2 = jnp.expand_dims(bat_blue_death_2_sprite, axis=0)
        SPRITE_BAT_BLUE_DEATH_3 = jnp.expand_dims(bat_blue_death_3_sprite, axis=0)
        # --- BAT RED SPRITES ---
        SPRITE_BAT_RED_MAIN = jnp.expand_dims(bat_red_main_sprite, axis=0)
        SPRITE_BAT_RED_LEFT_WING_MIDDLE = jnp.expand_dims(bat_red_left_wing_middle_sprite, axis=0)
        SPRITE_BAT_RED_RIGHT_WING_MIDDLE = jnp.expand_dims(bat_red_right_wing_middle_sprite, axis=0)
        SPRITE_BAT_RED_LEFT_WING_UP = jnp.expand_dims(bat_red_left_wing_up_sprite, axis=0)
        SPRITE_BAT_RED_RIGHT_WING_UP = jnp.expand_dims(bat_red_right_wing_up_sprite, axis=0)
        SPRITE_BAT_RED_LEFT_WING_DOWN = jnp.expand_dims(bat_red_left_wing_down_sprite, axis=0)
        SPRITE_BAT_RED_RIGHT_WING_DOWN = jnp.expand_dims(bat_red_right_wing_down_sprite, axis=0)
        SPRITE_BAT_RED_LEFT_WING_DOWN_2 = jnp.expand_dims(bat_red_left_wing_down_2_sprite, axis=0)
        SPRITE_BAT_RED_RIGHT_WING_DOWN_2 = jnp.expand_dims(bat_red_right_wing_down_2_sprite, axis=0)
        SPRITE_BAT_RED_DEATH_1 = jnp.expand_dims(bat_red_death_1_sprite, axis=0)
        SPRITE_BAT_RED_DEATH_2 = jnp.expand_dims(bat_red_death_2_sprite, axis=0)
        SPRITE_BAT_RED_DEATH_3 = jnp.expand_dims(bat_red_death_3_sprite, axis=0)

        # --- OLD BATS SPRITES ---
        SPRITE_MAIN_BAT_1 = jnp.expand_dims(main_bat_1, axis=0)
        SPRITE_LEFT_WING_BAT_1 = jnp.expand_dims(left_wing_bat_1, axis=0)
        SPRITE_RIGHT_WING_BAT_1 = jnp.expand_dims(right_wing_bat_1, axis=0)
        SPRITE_MAIN_BAT_2 = jnp.expand_dims(main_bat_2, axis=0)
        SPRITE_LEFT_WING_BAT_2 = jnp.expand_dims(left_wing_bat_2, axis=0)
        SPRITE_RIGHT_WING_BAT_2 = jnp.expand_dims(right_wing_bat_2, axis=0)
        # --- BOSS SPRITES ---
        SPRITE_BOSS = jnp.expand_dims(boss_sprite, axis=0)
        SPRITE_BLUE_BLOCK = boss_block_blue
        SPRITE_RED_BLOCK = boss_block_red
        SPRITE_GREEN_BLOCK = boss_block_green

        return (
            # --- FIELD SPRITES ---
            BG_SPRITE,
            SPRITE_FLOOR,
            # --- UI SPRITES ---
            DIGITS,
            LIFE_INDICATOR,
            # --- PLAYER SPRITES ---
            SPRITE_PLAYER,
            SPRITE_PLAYER_DEATH_1,
            SPRITE_PLAYER_DEATH_2,
            SPRITE_PLAYER_DEATH_3,
            SPRITE_PLAYER_MOVE,
            SPRITE_PLAYER_ABILITY,
            # --- PROJECTILE SPRITES ---
            SPRITE_PLAYER_PROJECTILE,
            SPRITE_ENEMY_PROJECTILE,
            # --- PHOENIX SPRITES ---
            SPRITE_PHOENIX_1,
            SPRITE_PHOENIX_2,
            SPRITE_PHOENIX_ATTACK,
            SPRITE_PHOENIX_DEATH_1,
            SPRITE_PHOENIX_DEATH_2,
            # --- BAT BLUE SPRITES ---
            SPRITE_BAT_BLUE_MAIN,
            SPRITE_BAT_BLUE_LEFT_WING_MIDDLE,
            SPRITE_BAT_BLUE_RIGHT_WING_MIDDLE,
            SPRITE_BAT_BLUE_LEFT_WING_UP,
            SPRITE_BAT_BLUE_RIGHT_WING_UP,
            SPRITE_BAT_BLUE_LEFT_WING_DOWN,
            SPRITE_BAT_BLUE_RIGHT_WING_DOWN,
            SPRITE_BAT_BLUE_LEFT_WING_DOWN_2,
            SPRITE_BAT_BLUE_RIGHT_WING_DOWN_2,
            SPRITE_BAT_BLUE_DEATH_1,
            SPRITE_BAT_BLUE_DEATH_2,
            SPRITE_BAT_BLUE_DEATH_3,
            # --- BAT RED SPRITES ---
            SPRITE_BAT_RED_MAIN,
            SPRITE_BAT_RED_LEFT_WING_MIDDLE,
            SPRITE_BAT_RED_RIGHT_WING_MIDDLE,
            SPRITE_BAT_RED_LEFT_WING_UP,
            SPRITE_BAT_RED_RIGHT_WING_UP,
            SPRITE_BAT_RED_LEFT_WING_DOWN,
            SPRITE_BAT_RED_RIGHT_WING_DOWN,
            SPRITE_BAT_RED_LEFT_WING_DOWN_2,
            SPRITE_BAT_RED_RIGHT_WING_DOWN_2,
            SPRITE_BAT_RED_DEATH_1,
            SPRITE_BAT_RED_DEATH_2,
            SPRITE_BAT_RED_DEATH_3,
            # --- OLD BAT SPRITES --- # TODO remove
            SPRITE_MAIN_BAT_1,
            SPRITE_LEFT_WING_BAT_1,
            SPRITE_RIGHT_WING_BAT_1,
            SPRITE_MAIN_BAT_2,
            SPRITE_LEFT_WING_BAT_2,
            SPRITE_RIGHT_WING_BAT_2,
            # --- BOSS SPRITES ---
            SPRITE_BOSS,
            SPRITE_RED_BLOCK,
            SPRITE_BLUE_BLOCK,
            SPRITE_GREEN_BLOCK,

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
        frame_player_death_1 = jr.get_sprite_frame(self.SPRITE_PLAYER_DEATH_1, 0)
        frame_player_death_2 = jr.get_sprite_frame(self.SPRITE_PLAYER_DEATH_2, 0)
        frame_player_death_3 = jr.get_sprite_frame(self.SPRITE_PLAYER_DEATH_3, 0)
        frame_player_move = jr.get_sprite_frame(self.SPRITE_PLAYER_MOVE, 0)
        frame_player_ability = jr.get_sprite_frame(self.SPRITE_PLAYER_ABILITY, 0)
        #raster = jr.render_at(raster, state.player_x, state.player_y, frame_player)

        # Render projectiles
        frame_projectile = jr.get_sprite_frame(self.SPRITE_PLAYER_PROJECTILE, 0)
        frame_enemy_projectile = jr.get_sprite_frame(self.SPRITE_ENEMY_PROJECTILE, 0)

        # Render enemy phoenix
        frame_phoenix_1 = jr.get_sprite_frame(self.SPRITE_PHOENIX_1, 0)
        frame_phoenix_2 = jr.get_sprite_frame(self.SPRITE_PHOENIX_2, 0)
        frame_phoenix_attack = jr.get_sprite_frame(self.SPRITE_PHOENIX_ATTACK, 0)
        frame_phoenix_death_1 = jr.get_sprite_frame(self.SPRITE_PHOENIX_DEATH_1, 0)
        frame_phoenix_death_2 = jr.get_sprite_frame(self.SPRITE_PHOENIX_DEATH_2, 0)

        # Render enemy bats --- OLD --- TODO: remove
        frame_main_bat = jr.get_sprite_frame(self.SPRITE_MAIN_BAT_1, 0)
        frame_left_wing_bat_1 = jr.get_sprite_frame(self.SPRITE_LEFT_WING_BAT_1, 0)
        frame_right_wing_bat_1 = jr.get_sprite_frame(self.SPRITE_RIGHT_WING_BAT_1, 0)
        frame_main_bat_2 = jr.get_sprite_frame(self.SPRITE_MAIN_BAT_2, 0)
        frame_left_wing_bat_2 = jr.get_sprite_frame(self.SPRITE_LEFT_WING_BAT_2, 0)
        frame_right_wing_bat_2 = jr.get_sprite_frame(self.SPRITE_RIGHT_WING_BAT_2, 0)

        # Render enemy bats blue
        frame_bat_blue_main = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_MAIN, 0)
        frame_bat_blue_left_wing_middle = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_LEFT_WING_MIDDLE, 0)
        frame_bat_blue_right_wing_middle = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_RIGHT_WING_MIDDLE, 0)
        frame_bat_blue_left_wing_up = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_LEFT_WING_UP, 0)
        frame_bat_blue_right_wing_up = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_RIGHT_WING_UP, 0)
        frame_bat_blue_left_wing_down = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_LEFT_WING_DOWN, 0)
        frame_bat_blue_right_wing_down = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_RIGHT_WING_DOWN, 0)
        frame_bat_blue_left_wing_down_2 = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_LEFT_WING_DOWN_2, 0)
        frame_bat_blue_right_wing_down_2 = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_RIGHT_WING_DOWN_2, 0)
        frame_bat_blue_death_1 = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_DEATH_1, 0)
        frame_bat_blue_death_2 = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_DEATH_2, 0)
        frame_bat_blue_death_3 = jr.get_sprite_frame(self.SPRITE_BAT_BLUE_DEATH_3, 0)

        # Render enemy bats red
        frame_bat_red_main = jr.get_sprite_frame(self.SPRITE_BAT_RED_MAIN, 0)
        frame_bat_red_left_wing_middle = jr.get_sprite_frame(self.SPRITE_BAT_RED_LEFT_WING_MIDDLE, 0)
        frame_bat_red_right_wing_middle = jr.get_sprite_frame(self.SPRITE_BAT_RED_RIGHT_WING_MIDDLE, 0)
        frame_bat_red_left_wing_up = jr.get_sprite_frame(self.SPRITE_BAT_RED_LEFT_WING_UP, 0)
        frame_bat_red_right_wing_up = jr.get_sprite_frame(self.SPRITE_BAT_RED_RIGHT_WING_UP, 0)
        frame_bat_red_left_wing_down = jr.get_sprite_frame(self.SPRITE_BAT_RED_LEFT_WING_DOWN, 0)
        frame_bat_red_right_wing_down = jr.get_sprite_frame(self.SPRITE_BAT_RED_RIGHT_WING_DOWN, 0)
        frame_bat_red_left_wing_down_2 = jr.get_sprite_frame(self.SPRITE_BAT_RED_LEFT_WING_DOWN_2, 0)
        frame_bat_red_right_wing_down_2 = jr.get_sprite_frame(self.SPRITE_BAT_RED_RIGHT_WING_DOWN_2, 0)
        frame_bat_red_death_1 = jr.get_sprite_frame(self.SPRITE_BAT_RED_DEATH_1, 0)
        frame_bat_red_death_2 = jr.get_sprite_frame(self.SPRITE_BAT_RED_DEATH_2, 0)
        frame_bat_red_death_3 = jr.get_sprite_frame(self.SPRITE_BAT_RED_DEATH_3, 0)

        # Render boss
        frame_boss = jr.get_sprite_frame(self.SPRITE_BOSS, 0)

        player_death_sprite_duration = self.consts.PLAYER_DEATH_DURATION // 3  # Duration for each player death sprite frame

        # Render player death animation
        def pick_player_death_sprite():
            return jax.lax.cond(
                state.player_death_timer >= 2 * player_death_sprite_duration,
                lambda: frame_player_death_1,
                lambda: jax.lax.cond(
                    state.player_death_timer >= player_death_sprite_duration,
                    lambda: frame_player_death_2,
                    lambda: frame_player_death_3
                )
            )

        def pick_player_alive_sprite():
            anim_toggle = (((state.step_counter // self.consts.PLAYER_ANIMATION_SPEED) % 2) == 0)
            return jax.lax.cond(
                state.invincibility,  # Fähigkeit aktiv -> immer Move-Sprite
                lambda: frame_player_move,
                lambda: jax.lax.cond(
                    state.player_moving & anim_toggle,
                    lambda: frame_player_move,
                    lambda: frame_player
                )
            )

        frame_player_used = jax.lax.cond(
            state.player_dying,
            pick_player_death_sprite,
            pick_player_alive_sprite
        )

        raster = jax.lax.cond(
            jnp.logical_or(state.player_dying, state.player_respawn_timer <= 0),
            lambda r: jr.render_at(r, state.player_x, state.player_y, frame_player_used),
            lambda r: r,
            operand=raster
        )


        # Phoenix attack rendering logic
        tol = 0.5
        going_down = state.phoenix_do_attack & (state.enemies_y < state.phoenix_attack_target_y - tol)
        going_up = state.phoenix_do_attack & (state.enemies_y > state.phoenix_attack_target_y + tol)
        returning_moving = state.phoenix_returning & (jnp.abs(state.enemies_y - state.phoenix_original_y) > tol)
        is_moving_vert = going_down | going_up | returning_moving

        # Phoenix death rendering logic
        death_flags = state.phoenix_dying
        death_phase = (state.phoenix_death_timer <= self.consts.ENEMY_DEATH_DURATION // 2).astype(jnp.int32)  # 0 for first half, 1 for second half


        def render_enemy(raster, input):
            enemy_pos, wings, moving_vert, enemy_dying, enemy_phase = input
            x, y = enemy_pos
            anim_toggle = ((state.step_counter // self.consts.ENEMY_ANIMATION_SPEED) % 2) == 0  # toggle every 32 steps

            def pick_phoenix_death_sprite():
                return jax.lax.cond(
                    enemy_phase == 0,
                    lambda: frame_phoenix_death_1,
                    lambda: frame_phoenix_death_2
                )

            def pick_phoenix_alive_sprite():
                return jax.lax.cond(
                    moving_vert,
                    lambda: frame_phoenix_attack,
                    lambda: jax.lax.cond(
                        anim_toggle,
                        lambda: frame_phoenix_1,
                        lambda: frame_phoenix_2
                    )
                )

            def render_level1(r):
                #phoenix_anim = jax.lax.select(anim_toggle, frame_phoenix_1, frame_phoenix_2)
                #phoenix_frame = jax.lax.select(moving_vert, frame_phoenix_attack, phoenix_anim)
                phoenix_frame = jax.lax.cond(
                    enemy_dying,
                    pick_phoenix_death_sprite,
                    pick_phoenix_alive_sprite
                )
                return jr.render_at(r, x, y, phoenix_frame)
                #return jr.render_at(r, x, y, frame_phoenix_1) # OLD OLD OLD

            def render_level2(r):
                #phoenix_anim = jax.lax.select(anim_toggle, frame_phoenix_1, frame_phoenix_2)
                #phoenix_frame = jax.lax.select(moving_vert, frame_phoenix_attack, phoenix_anim)
                phoenix_frame = jax.lax.cond(
                    enemy_dying,
                    pick_phoenix_death_sprite,
                    pick_phoenix_alive_sprite
                )
                return jr.render_at(r, x, y, phoenix_frame)
                #return jr.render_at(r, x, y, frame_phoenix_1) # OLD OLD OLD
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

        # Update the raster
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)
        wings_array = jnp.full((enemy_positions.shape[0],), state.bat_wings)
        moving_flags = is_moving_vert
        death_flags = jnp.full((enemy_positions.shape[0],), death_flags)
        death_phase = jnp.full((enemy_positions.shape[0],), death_phase)
        inputs = (enemy_positions, wings_array, moving_flags, death_flags, death_phase)
        def draw_phoenix(rr):
            return jax.lax.scan(render_enemy, rr, inputs)[0]

        # Render player projectiles
        def render_player_projectile(r):
            return jr.render_at(r, state.projectile_x, state.projectile_y, frame_projectile)

        raster = jax.lax.cond(
            state.projectile_x > -1,
            render_player_projectile,
            lambda r: r,
            raster
        )

        # Render enemy bats
        is_blue_level = (state.level % 5) == 3
        is_red_level = (state.level % 5) == 4
        is_bat_level = is_blue_level | is_red_level

        raster = jax.lax.cond(jnp.logical_not(is_bat_level), draw_phoenix, lambda rr: rr, raster)

        seg = jnp.maximum(1, self.consts.ENEMY_DEATH_DURATION // 3)

        def pick_death_frame(t: jnp.int32):
            def blue():
                return jax.lax.cond(
                    t > 2 * seg,
                    lambda: frame_bat_blue_death_1,
                    lambda: jax.lax.cond(
                        t > seg,
                        lambda: frame_bat_blue_death_2,
                        lambda: frame_bat_blue_death_3
                    )
                )

            def red():
                return jax.lax.cond(
                    t > 2 * seg,
                    lambda: frame_bat_red_death_1,
                    lambda: jax.lax.cond(
                        t > seg,
                        lambda: frame_bat_red_death_2,
                        lambda: frame_bat_red_death_3
                    )
                )

            return jax.lax.cond(is_blue_level, blue, red)

        def pick_alive_body():
            return jax.lax.cond(is_blue_level, lambda: frame_bat_blue_main, lambda: frame_bat_red_main)

        def pick_middle_wings():
            def blue():
                return frame_bat_blue_left_wing_middle, frame_bat_blue_right_wing_middle

            def red():
                return frame_bat_red_left_wing_middle, frame_bat_red_right_wing_middle

            return jax.lax.cond(is_blue_level, blue, red)

        def body(i, r):
            x = state.enemies_x[i].astype(jnp.int32)
            y = state.enemies_y[i].astype(jnp.int32)
            active = (x > -1) & (y < self.consts.HEIGHT + 10) & is_bat_level
            dying = state.bat_dying[i] & is_bat_level
            t = state.bat_death_timer[i].astype(jnp.int32)

            def draw_death(rr):
                # Death-Sprite zentriert auf den Körper-Anker ausrichten
                frame = pick_death_frame(t)
                body_frame = pick_alive_body()
                bh, bw = body_frame.shape[:2]
                dh, dw = frame.shape[:2]
                ox = x + (bw - dw) // 2 - 5
                oy = y + (bh - dh) // 2
                return jr.render_at(rr, ox, oy, frame)

            def draw_alive(rr):
                # Körper
                rr = jr.render_at(rr, x, y, pick_alive_body())

                # Flügel (mittlere) mit korrekt horizontalem Versatz und vertikal +1px
                left_frame, right_frame = pick_middle_wings()
                wing_state = state.bat_wings[i].astype(jnp.int32)
                draw_left = (wing_state == 2) | (wing_state == -1)
                draw_right = (wing_state == 2) | (wing_state == 1)

                x_left = x - self.consts.WING_WIDTH
                x_right = x + self.consts.ENEMY_WIDTH - 1  # rechter Flügel 1px weiter links
                y_wings = y + 2  # beide Flügel 1px tiefer

                rr = jax.lax.cond(
                    draw_left,
                    lambda r2: jr.render_at(r2, x_left, y_wings, left_frame),
                    lambda r2: r2,
                    rr
                )
                rr = jax.lax.cond(
                    draw_right,
                    lambda r2: jr.render_at(r2, x_right, y_wings, right_frame),
                    lambda r2: r2,
                    rr
                )
                return rr

            def draw_one(rr):
                return jax.lax.cond(dying, draw_death, draw_alive, rr)

            return jax.lax.cond(active, draw_one, lambda rr: rr, r)

        raster = jax.lax.fori_loop(0, state.enemies_x.shape[0], body, raster)

        def render_ability(r):
            ah, aw = frame_player_ability.shape[:2]
            ph, pw = frame_player_used.shape[:2]
            ax = state.player_x + (pw - aw) // 2
            ay = state.player_y + (ph - ah) // 2
            return jr.render_at(r, ax, ay, frame_player_ability)

        ability_visible = state.invincibility & ((state.step_counter % 4) == 0) # Zeige ability nur jeden vierten Frame

        raster = jax.lax.cond(
            ability_visible,
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