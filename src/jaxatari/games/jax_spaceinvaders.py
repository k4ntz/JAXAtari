#
# JAX SpaceInvaders
#
# Simulates the Space Invaders game using JAX
#
# Authors:
# - Luca Philippi 
# - Julian Bayer 
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for SpaceInvaders.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        # Background
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        
        # Player (0=visible, 1=invisible)
        {'name': 'player', 'type': 'group', 'files': ['player.npy', 'player_invisible.npy']},
        
        # Score
        {'name': 'digits_green', 'type': 'digits', 'pattern': 'numbers/{}_green.npy'},
        {'name': 'zero_yellow', 'type': 'single', 'file': 'numbers/0_yellow.npy'},
        
        # Defense
        {'name': 'defense', 'type': 'single', 'file': 'defense.npy'},
        
        # Enemies (A/B animation frames)
        {'name': 'opponent_1', 'type': 'group', 'files': ['opponents/opponent_1_a.npy', 'opponents/opponent_1_b.npy']},
        {'name': 'opponent_2', 'type': 'group', 'files': ['opponents/opponent_2_a.npy', 'opponents/opponent_2_b.npy']},
        {'name': 'opponent_3', 'type': 'group', 'files': ['opponents/opponent_3_a.npy', 'opponents/opponent_3_b.npy']},
        {'name': 'opponent_4', 'type': 'group', 'files': ['opponents/opponent_4_a.npy', 'opponents/opponent_4_b.npy']},
        # Opponent 5 doesn't change, so we duplicate its file for the A/B structure
        {'name': 'opponent_5', 'type': 'group', 'files': ['opponents/opponent_5.npy', 'opponents/opponent_5.npy']},
        {'name': 'opponent_6', 'type': 'group', 'files': ['opponents/opponent_6_a.npy', 'opponents/opponent_6_b.npy']},
        {'name': 'ufo', 'type': 'single', 'file': 'opponents/ufo.npy'},
        
        # Bullet
        {'name': 'bullet', 'type': 'single', 'file': 'bullet.npy'},
        
        # Explosions
        {'name': 'enemy_explosion', 'type': 'group', 'files': [
            'explosions/explosion_1.npy', 'explosions/explosion_2.npy', 
            'explosions/explosion_3.npy', 'explosions/explosion_4.npy'
        ]},
        {'name': 'ufo_explosion', 'type': 'group', 'files': ['explosions/exp_ufo_1.npy', 'explosions/exp_ufo_2.npy']},
        {'name': 'player_explosion', 'type': 'group', 'files': ['explosions/exp_player_1.npy', 'explosions/exp_player_2.npy']},
        
        # Lives
        {'name': 'lives', 'type': 'group', 'files': ['lifes/one.npy', 'lifes/two.npy', 'lifes/three.npy']},
    )

class SpaceInvadersConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    SCALING_FACTOR: int = struct.field(pytree_node=False, default=4)
    WINDOW_WIDTH: int = struct.field(pytree_node=False, default=None)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default=None)

    NUMBER_YELLOW_OFFSET: int = struct.field(pytree_node=False, default=83)

    WALL_LEFT_X: int = struct.field(pytree_node=False, default=34)
    WALL_RIGHT_X: int = struct.field(pytree_node=False, default=123)

    OPPONENT_LIMIT_X: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (22, 136))
    OPPONENT_LIMIT_Y: Tuple[int, None] = struct.field(pytree_node=False, default_factory=lambda: (31, None))

    MAX_SPEED: int = struct.field(pytree_node=False, default=1)
    ACCELERATION: int = struct.field(pytree_node=False, default=1)
    BULLET_SPEED: int = struct.field(pytree_node=False, default=1)
    ENEMY_BULLET_SPEED: int = struct.field(pytree_node=False, default=1)

    PATH_SPRITES: str = struct.field(pytree_node=False, default_factory=lambda: os.path.join(render_utils.get_base_sprite_dir(), "spaceinvaders"))
    ENEMY_ROWS: int = struct.field(pytree_node=False, default=6)
    ENEMY_COLS: int = struct.field(pytree_node=False, default=6)
    MAX_ENEMY_BULLETS: int = struct.field(pytree_node=False, default=3)

    EXPLOSION_FRAMES: jnp.array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([3, 11, 19, 27], dtype=jnp.int32))
    PLAYER_EXPLOSION_FRAMES: int = struct.field(pytree_node=False, default=4)
    PLAYER_EXPLOSION_DURATION: int = struct.field(pytree_node=False, default=126)
    PLAYER_RESET_DURATION: int = struct.field(pytree_node=False, default=64)
    PLAYER_RESET_FRAMES: int = struct.field(pytree_node=False, default=8)
    FULL_PAUSE_DURATION: int = struct.field(pytree_node=False, default=None) # +1 for the frame between the end of the explosion and the start of the blinking animation

    POSITION_LIFE_X: int = struct.field(pytree_node=False, default=83)

    # Thresholds: [15, 29, 33, 34, 35] (Total destroyed count)
    MOVEMENT_THRESHOLDS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([15, 29, 33, 34, 35], dtype=jnp.int32))
    # Rates: Delays in frames. 
    # 0-14 dead -> 32 frames delay
    # 15-28 dead -> 16 frames delay
    # ...
    # 35 dead (1 left) -> 1 frame delay (fastest)
    MOVEMENT_RATES: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([32, 16, 8, 4, 2, 1], dtype=jnp.int32))
    ENEMY_FIRE_RATE: int = struct.field(pytree_node=False, default=60)

    INITIAL_LIVES: int = struct.field(pytree_node=False, default=3)
    INITIAL_PLAYER_X: int = struct.field(pytree_node=False, default=41)
    INITIAL_BULLET_POS: int = struct.field(pytree_node=False, default=78)
    INITIAL_OPPONENT_DIRECTION: int = struct.field(pytree_node=False, default=1)
    WIN_SCORE: int = struct.field(pytree_node=False, default=1000) # actually infinite 

    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(7, 10))
    BULLET_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(1, 10))
    WALL_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(2, 4))
    BACKGROUND_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(WIDTH, 15))
    NUMBER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(12, 9))
    OPPONENT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 10))
    OPPONENT_VERTICAL_STEP_SIZE : int = struct.field(pytree_node=False, default=10) # vertical moving distance of opponents
    OFFSET_OPPONENT: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 8))

    UFO_SCORE: int = struct.field(pytree_node=False, default=200)
    UFO_Y: int = struct.field(pytree_node=False, default=11)
    UFO_START: int = struct.field(pytree_node=False, default=960)
    UFO_MOVEMENT_RATE: int = struct.field(pytree_node=False, default=4)
    UFO_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(7, 8))

    BARRICADE_POS: Tuple[jnp.array, int] = struct.field(pytree_node=False, default=None)

    PLAYER_Y: int = struct.field(pytree_node=False, default=None)

    BULLET_DAMAGE: int = struct.field(pytree_node=False, default=6)

    # --- NEW BARRICADE CONSTANTS (balanced model) ---
    # Health grid shape per barricade (rows, cols)
    BARRICADE_GRID_SHAPE: Tuple[int, int] = struct.field(pytree_node=False, default=(6, 8))
    # Size of the original barricade sprite mask (height, width)
    BARRICADE_SPRITE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(18, 8))
    # Chunk size in pixels (height, width), as floats
    BARRICADE_CHUNK_SIZE: Tuple[float, float] = struct.field(pytree_node=False, default=None)
    # Initial health per chunk and damage per bullet
    BARRICADE_HEALTH_INITIAL: int = struct.field(pytree_node=False, default=6)
    BARRICADE_BULLET_DAMAGE: int = struct.field(pytree_node=False, default=6)
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default=_get_default_asset_config())

    def compute_derived(self):
        return {
            "WINDOW_HEIGHT": self.HEIGHT * self.SCALING_FACTOR,
            "WINDOW_WIDTH": self.WIDTH * self.SCALING_FACTOR,
            "FULL_PAUSE_DURATION": self.PLAYER_EXPLOSION_DURATION + 1 + self.PLAYER_RESET_DURATION,
            "PLAYER_Y": self.HEIGHT - self.PLAYER_SIZE[1] - self.BACKGROUND_SIZE[1],
            "BARRICADE_POS": (jnp.array([41, 73, 105], dtype=jnp.int32), self.HEIGHT - 53),
            "BARRICADE_CHUNK_SIZE": (
                self.BARRICADE_SPRITE_SIZE[0] / self.BARRICADE_GRID_SHAPE[0],
                self.BARRICADE_SPRITE_SIZE[1] / self.BARRICADE_GRID_SHAPE[1]
            )
        }

@struct.dataclass
class SpaceInvadersState:
    player_x: chex.Array
    player_speed: chex.Array
    player_dead: int
    step_counter: chex.Array
    player_score: chex.Array
    player_lives: chex.Array
    destroyed: chex.Array
    opponent_current_x: int
    opponent_current_y: int
    opponent_bounding_rect: NamedTuple
    opponent_direction: int
    ufo_x: int
    ufo_state: int # 0 = not visible, 1 = alive, 2-UFO_EXPLOSION_DURATION = explosion, UFO_EXPLOSION_DURATION + 1 = dead
    ufo_dir: int
    bullet_active: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_bullets_active: chex.Array
    enemy_bullets_x: chex.Array
    enemy_bullets_y: chex.Array
    enemy_fire_cooldown: chex.Array
    barricade_health: chex.Array
    active_barricade: int # 1 = barricades active, 0 = barricades inactive (deactivated when enemies reach barricade line)
    enemy_flip: chex.Array 
    rng: jax.random.PRNGKey

@struct.dataclass
class SpaceInvadersObservation:
    player: ObjectObservation
    enemies: ObjectObservation  # n=36 (6x6 grid)
    player_bullet: ObjectObservation
    enemy_bullets: ObjectObservation  # n=3
    ufo: ObjectObservation
    
    # Dense features
    barricade_health: chex.Array  # (3, 6, 8) grid of healths
    
    score_player: jnp.ndarray
    lives: jnp.ndarray

@struct.dataclass
class SpaceInvadersInfo:
    time: jnp.ndarray

class JaxSpaceInvaders(JaxEnvironment[SpaceInvadersState, SpaceInvadersObservation, SpaceInvadersInfo, SpaceInvadersConstants]):
    # Minimal ALE action set for Space Invaders:
    # 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )

    def __init__(self, consts: SpaceInvadersConstants = None):
        consts = consts or SpaceInvadersConstants()
        super().__init__(consts)
        self.renderer = SpaceInvadersRenderer(self.consts)
        self.obs_size = 3 * 4 + 1 + 1
        self.renderer = SpaceInvadersRenderer(consts)

    def render(self, state):
        return self.renderer.render(state)

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_enemy_position(self, opponent_current_x, opponent_current_y, row, col):
        x = opponent_current_x + col * (self.consts.OFFSET_OPPONENT[0] + self.consts.OPPONENT_SIZE[0])
        y = opponent_current_y + row * (self.consts.OFFSET_OPPONENT[1] + self.consts.OPPONENT_SIZE[1])
        return x, y

    @partial(jax.jit, static_argnums=(0,))
    def _check_collision(self, bullet_x, bullet_y, target_x, target_y, target_width, target_height):
        bullet_right = bullet_x + self.consts.BULLET_SIZE[0]
        bullet_bottom = bullet_y + self.consts.BULLET_SIZE[1]
        target_right = target_x + target_width
        target_bottom = target_y + target_height

        collision = (bullet_x < target_right) & (bullet_right > target_x) & (bullet_y < target_bottom) & (
                    bullet_bottom > target_y)
        return collision

    @partial(jax.jit, static_argnums=(0,))
    def _check_bullet_enemy_collisions(self, state: SpaceInvadersState):
        def check_single_enemy(i):
            row = i // self.consts.ENEMY_COLS
            col = i % self.consts.ENEMY_COLS

            enemy_x, enemy_y = self._get_enemy_position(
                state.opponent_current_x,
                state.opponent_current_y,
                row,
                col
            )

            enemy_alive = jnp.logical_not(state.destroyed[i])
            collision = jnp.logical_and(
                jnp.logical_and(enemy_alive, state.bullet_active),
                self._check_collision(state.bullet_x, state.bullet_y, enemy_x, enemy_y,
                                self.consts.OPPONENT_SIZE[0], self.consts.OPPONENT_SIZE[1])
            )

            new_destroyed = jnp.where((state.destroyed[i] == 0) & collision, 1, state.destroyed[i])

            score = (self.consts.ENEMY_ROWS - row) * 5
            score_contrib = jnp.where(collision, score, 0)

            bullet_hit = collision

            # Returns if destroyed, the contribution to the score and if the bullet hit
            return new_destroyed, score_contrib, bullet_hit

        destroyed_incremented = jnp.where(state.destroyed != 0, jnp.minimum(state.destroyed + 1, 29), 0)

        def check_all_enemies():
            # vmap over all enemy indices
            enemy_indices = jnp.arange(self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS)
            destroyed_vals, score_contribs, bullet_hits = jax.vmap(check_single_enemy)(enemy_indices)

            # Merge new hits with the incremented explosions
            final_destroyed = jnp.where(destroyed_vals > state.destroyed, destroyed_vals, destroyed_incremented)

            total_score = state.player_score + jnp.sum(score_contribs)

            # check possible ufo collision
            ufo_collision = self._check_collision(state.bullet_x, state.bullet_y, state.ufo_x, self.consts.UFO_Y, self.consts.UFO_SIZE[0], self.consts.UFO_SIZE[1])
            total_score, new_ufo_state = jax.lax.cond(
                ufo_collision,
                lambda: (total_score + self.consts.UFO_SCORE, 2),
                lambda: (total_score, state.ufo_state)
            )

            bullet_active = jnp.logical_not(jnp.logical_or(jnp.any(bullet_hits), ufo_collision))

            return final_destroyed, total_score, bullet_active, new_ufo_state

        def no_bullet():
            return destroyed_incremented, state.player_score, state.bullet_active, state.ufo_state

        return jax.lax.cond(
            state.bullet_active,
            check_all_enemies,
            no_bullet
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_enemy_bullet_player_collisions(self, state: SpaceInvadersState):
        
        # 1. Define a function to check a single bullet
        #    (We vmap this function)
        def check_single_bullet(bullet_x, bullet_y, bullet_active):
            collision = jnp.logical_and(
                bullet_active,
                self._check_collision(
                    bullet_x,
                    bullet_y,
                    state.player_x - self.consts.PLAYER_SIZE[0],
                    self.consts.PLAYER_Y,
                    self.consts.PLAYER_SIZE[0],
                    self.consts.PLAYER_SIZE[1]
                )
            )
            return collision

        # 2. Vmap the check over all bullets
        all_collisions = jax.vmap(check_single_bullet)(
            state.enemy_bullets_x,
            state.enemy_bullets_y,
            state.enemy_bullets_active
        )

        # 3. Reduce the results
        num_hits = jnp.sum(all_collisions)
        new_lives = state.player_lives - num_hits
        
        # 4. Deactivate bullets that hit
        #    (A bullet is now active if it WAS active AND it did NOT collide)
        new_enemy_bullets_active = jnp.logical_and(
            state.enemy_bullets_active, 
            jnp.logical_not(all_collisions)
        )

        return new_lives, new_enemy_bullets_active


    @partial(jax.jit, static_argnums=(0,))
    def _check_barricade_collision(self, index, input):
        """
        Calculates the collision on the barricades and masks pixels depending on the hit position.
        is_player_bullet is used to differentiate between player and enemy bullets.
        """
        active, bullets_x, bullets_y, masks, is_player_bullet = input

        mask = masks[index]
        H, W = mask.shape
        start_x = self.consts.BARRICADE_POS[0][index]
        start_y = self.consts.BARRICADE_POS[1]
        bullet_w = self.consts.BULLET_SIZE[0]

        def check_bullet(i, carry):
            mask_a, active_a = carry

            bx = bullets_x[i]
            by = bullets_y[i]
            bullet_active = active_a[i]

            # simplified collision detection with rectangle
            collision = (
                (bx >= start_x) &
                (bx < (start_x + W)) &
                (by >= start_y) &
                (by < (start_y + H))
            )

            pot_hit_x = jnp.floor(bx + bullet_w / 2.0) - start_x
            mask_x = jnp.clip(pot_hit_x.astype(jnp.int32), 0, W - 1)

            collumn = mask[:, mask_x]
            col_collision = jnp.any(collumn != 0)

            def on_collision(carry_b):
                mask_b, active_b = carry_b
                final_active = active_b.at[i].set(False)

                # find and mask first n visible Pixel
                idxs = jnp.arange(H)
                valid_idxs = jnp.where(collumn != 0, idxs, H)
                sorted_idxs = jax.lax.cond(is_player_bullet, lambda s: s[::-1], lambda s: s, jnp.sort(valid_idxs))
                remove = sorted_idxs[:self.consts.BULLET_DAMAGE]

                def remove_pixel(j, carry_c):
                    mask_c = carry_c
                    y = remove[j]
                    return jax.lax.cond(y < H, lambda: mask_c.at[y, mask_x].set(0), lambda: mask_c)
                
                masked = jax.lax.fori_loop(0, self.consts.BULLET_DAMAGE, remove_pixel, mask_b)

                return masked, final_active

            return jax.lax.cond(
                jnp.all(jnp.array([bullet_active, collision, col_collision]), axis=0),
                on_collision,
                lambda c: c,
                (mask_a, active_a)
            )

        # looping all bullets
        mask_final, active_final = jax.lax.fori_loop(0, active.size, check_bullet, (mask, active))

        # updating barricade mask
        new_masks = masks.at[index].set(mask_final)

        return active_final, bullets_x, bullets_y, new_masks, is_player_bullet

    @partial(jax.jit, static_argnums=(0,))
    def _get_barricade_hit_indices(self, bx: chex.Array, by: chex.Array, b_active: chex.Array):
        barricade_xs = self.consts.BARRICADE_POS[0]
        barricade_y = self.consts.BARRICADE_POS[1]

        full_h = self.consts.BARRICADE_SPRITE_SIZE[0]
        full_w = self.consts.BARRICADE_SPRITE_SIZE[1]
        chunk_h = self.consts.BARRICADE_CHUNK_SIZE[0]
        chunk_w = self.consts.BARRICADE_CHUNK_SIZE[1]

        y_hit = (by >= barricade_y) & (by < (barricade_y + full_h))
        x_hit_per = (bx >= barricade_xs) & (bx < (barricade_xs + full_w))
        hit_mask = b_active & y_hit & x_hit_per
        did_hit = jnp.any(hit_mask)

        barricade_idx = jnp.argmax(hit_mask)
        hit_x_start = barricade_xs[barricade_idx]
        rel_x = bx - hit_x_start
        rel_y = by - barricade_y
        chunk_col = (rel_x / chunk_w).astype(jnp.int32)
        chunk_row = (rel_y / chunk_h).astype(jnp.int32)

        indices = jnp.array([
            barricade_idx,
            jnp.clip(chunk_row, 0, self.consts.BARRICADE_GRID_SHAPE[0] - 1),
            jnp.clip(chunk_col, 0, self.consts.BARRICADE_GRID_SHAPE[1] - 1)
        ])
        no_hit = jnp.array([-1, -1, -1])
        final_indices = jnp.where(did_hit, indices, no_hit)
        return did_hit, final_indices

    @partial(jax.jit, static_argnums=(0,))
    def _update_barricade_health(
        self,
        barricade_health: chex.Array,
        bullets_x: chex.Array,
        bullets_y: chex.Array,
        bullets_active: chex.Array,
    ):
        def check_fn(x, y, a):
            return self._get_barricade_hit_indices(x, y, a)

        all_did_hit, all_indices = jax.vmap(check_fn)(bullets_x, bullets_y, bullets_active)

        safe_indices = jnp.where(all_did_hit[:, None], all_indices, 0)
        hit_chunk_healths = barricade_health[
            safe_indices[:, 0],
            safe_indices[:, 1],
            safe_indices[:, 2]
        ]
        real_hit = jnp.logical_and(all_did_hit, hit_chunk_healths > 0)

        # Deactivate only bullets that had a real hit
        new_bullets_active = jnp.logical_and(bullets_active, jnp.logical_not(real_hit))

        # Apply damage only for real hits
        damage = jnp.where(real_hit, self.consts.BARRICADE_BULLET_DAMAGE, 0)
        safe_indices = jnp.where(real_hit[:, None], all_indices, 0)
        idxs = (safe_indices[:, 0], safe_indices[:, 1], safe_indices[:, 2])
        new_health = barricade_health.at[idxs].add(-damage)
        new_health = jnp.clip(new_health, 0, self.consts.BARRICADE_HEALTH_INITIAL)
        return new_bullets_active, new_health

    @partial(jax.jit, static_argnums=(0,))
    def _get_bottom_enemies(self, state: SpaceInvadersState):
        enemies_alive = jnp.logical_not(state.destroyed).reshape(self.consts.ENEMY_ROWS, self.consts.ENEMY_COLS)
        row_indices = jnp.arange(self.consts.ENEMY_ROWS, dtype=jnp.int32)[:, None]
        indices_if_alive = jnp.where(enemies_alive, row_indices, -1)
        bottom_rows = jnp.max(indices_if_alive, axis=0)
        return bottom_rows

    @partial(jax.jit, static_argnums=(0,))
    def _update_enemy_bullets(self, state: SpaceInvadersState, key):
        new_y = state.enemy_bullets_y + self.consts.ENEMY_BULLET_SPEED
        new_active = jnp.where(new_y >= self.consts.HEIGHT, False, state.enemy_bullets_active)

        new_cooldown = jnp.maximum(0, state.enemy_fire_cooldown - 1)
        should_fire = (new_cooldown == 0) & (jnp.sum(new_active) < self.consts.MAX_ENEMY_BULLETS)

        def spawn_bullet():
            bottom_enemies = self._get_bottom_enemies(state)

            valid_columns = jnp.where(bottom_enemies >= 0, jnp.arange(self.consts.ENEMY_COLS), -1)
            mask = valid_columns >= 0
            indices = jnp.nonzero(mask, size=self.consts.ENEMY_COLS)[0]
            valid_columns = valid_columns[indices]
            num_valid = valid_columns.shape[0]

            def fire_bullet():
                col_idx = jax.random.randint(key, (), 0, num_valid)
                firing_col = valid_columns[col_idx]
                firing_row = bottom_enemies[firing_col]

                enemy_x, enemy_y = self._get_enemy_position(
                    state.opponent_current_x,
                    state.opponent_current_y,
                    firing_row,
                    firing_col
                )

                def find_slot(i, slot):
                    return jnp.where((slot == -1) & (new_active[i] == False), i, slot)

                bullet_slot = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_BULLETS, find_slot, -1)

                spawn_active = jnp.where(
                    bullet_slot >= 0,
                    new_active.at[bullet_slot].set(True),
                    new_active
                )
                spawn_x = jnp.where(
                    bullet_slot >= 0,
                    state.enemy_bullets_x.at[bullet_slot].set(enemy_x + self.consts.OPPONENT_SIZE[0] // 2),
                    state.enemy_bullets_x
                )
                spawn_y = jnp.where(
                    bullet_slot >= 0,
                    new_y.at[bullet_slot].set(enemy_y + self.consts.OPPONENT_SIZE[1]),
                    new_y
                )

                return spawn_active, spawn_x, spawn_y, self.consts.ENEMY_FIRE_RATE

            def no_fire():
                return new_active, state.enemy_bullets_x, new_y, new_cooldown

            return jax.lax.cond(num_valid > 0, fire_bullet, no_fire)

        def no_spawn():
            return new_active, state.enemy_bullets_x, new_y, new_cooldown

        return jax.lax.cond(should_fire, spawn_bullet, no_spawn)

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state_player_x, state_player_speed, action: chex.Array):
        left = (action == Action.LEFT) | (action == Action.LEFTFIRE)
        right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)

        bounds_left = self.consts.WALL_LEFT_X + self.consts.PLAYER_SIZE[0]
        bounds_right = self.consts.WALL_RIGHT_X

        touches_wall_left = state_player_x <= bounds_left
        touches_wall_right = state_player_x >= bounds_right
        touches_wall = jnp.logical_or(touches_wall_left, touches_wall_right)

        player_speed = jax.lax.cond(
            jnp.logical_not(jnp.logical_or(left, right)) | touches_wall,
            lambda s: 0,
            lambda s: s,
            operand=state_player_speed,
        )

        player_speed = jax.lax.cond(
            right,
            lambda s: jnp.minimum(s + self.consts.ACCELERATION, self.consts.MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        player_speed = jax.lax.cond(
            left,
            lambda s: jnp.maximum(s - self.consts.ACCELERATION, -self.consts.MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        player_x = jnp.clip(
            state_player_x + player_speed,
            bounds_left,
            bounds_right
        )

        return player_x, player_speed

    @partial(jax.jit, static_argnums=(0,))
    def _player_bullet_step(self, state: SpaceInvadersState, action: chex.Array):
        fired = (action == Action.FIRE) | (action == Action.RIGHTFIRE) | (action == Action.LEFTFIRE)

        new_bullet_active, new_bullet_x, new_bullet_y = jax.lax.cond(
            state.bullet_active,
            lambda: jax.lax.cond(
                (state.bullet_y - self.consts.BULLET_SPEED) < 0,
                lambda: (False, state.bullet_x, state.bullet_y),
                lambda: (True, state.bullet_x, state.bullet_y - self.consts.BULLET_SPEED)
            ),
            lambda: jax.lax.cond(
                fired,
                lambda: (
                    True,
                    state.player_x - (self.consts.PLAYER_SIZE[0] // 2),
                    self.consts.PLAYER_Y - self.consts.PLAYER_SIZE[1]
                ),
                lambda: (False, state.bullet_x, state.bullet_y)
            )
        )

        return new_bullet_active, new_bullet_x, new_bullet_y

    def reset(self, key=None) -> Tuple[SpaceInvadersObservation, SpaceInvadersState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        opponent_rect_width = self.consts.OPPONENT_SIZE[0] * self.consts.ENEMY_COLS + self.consts.OFFSET_OPPONENT[0] * (self.consts.ENEMY_COLS - 1)
        opponent_rect_height = self.consts.OPPONENT_SIZE[1] * self.consts.ENEMY_ROWS + self.consts.OFFSET_OPPONENT[1] * (self.consts.ENEMY_ROWS - 1)

        state = SpaceInvadersState(
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            player_dead=self.consts.PLAYER_EXPLOSION_DURATION,
            step_counter=jnp.array(0).astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(self.consts.INITIAL_LIVES).astype(jnp.int32),
            destroyed=jnp.zeros((self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS,), dtype=jnp.int32), # If 0 its alive, after it counts up to 28 each frame showing a different animation state depending on this value. Starting with 29 its gone
            opponent_current_x=self.consts.OPPONENT_LIMIT_X[0],
            opponent_current_y=self.consts.OPPONENT_LIMIT_Y[0],
            opponent_bounding_rect=(opponent_rect_width, opponent_rect_height),
            opponent_direction=self.consts.INITIAL_OPPONENT_DIRECTION,
            ufo_x = 0,
            ufo_state = 0,
            ufo_dir = 1,
            bullet_active=jnp.array(0).astype(jnp.int32),
            bullet_x=jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            bullet_y=jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            enemy_bullets_active=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.bool),
            enemy_bullets_x=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_bullets_y=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_fire_cooldown=jnp.array(self.consts.ENEMY_FIRE_RATE).astype(jnp.int32),
            barricade_health=jnp.full(
                (3, self.consts.BARRICADE_GRID_SHAPE[0], self.consts.BARRICADE_GRID_SHAPE[1]),
                self.consts.BARRICADE_HEALTH_INITIAL,
                dtype=jnp.int32
            ),
            active_barricade=jnp.array(1, dtype=jnp.int32),
            enemy_flip=jnp.array(0, dtype=jnp.int32),
            rng=key
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    def reset_level(self, state: SpaceInvadersState) -> SpaceInvadersState:
        # new_state = SpaceInvadersState(
        state = state.replace(
            player_x = jnp.array(self.consts.INITIAL_PLAYER_X).astype(jnp.int32),
            player_speed = jnp.array(0.0).astype(jnp.int32),
            player_dead = state.player_dead,
            step_counter = 0,
            player_score = state.player_score,
            player_lives = state.player_lives,
            destroyed = state.destroyed,
            opponent_current_x = self.consts.OPPONENT_LIMIT_X[0],
            opponent_current_y = self.consts.OPPONENT_LIMIT_Y[0],
            opponent_bounding_rect = state.opponent_bounding_rect,
            opponent_direction = self.consts.INITIAL_OPPONENT_DIRECTION,
            ufo_x = 0,
            ufo_state = 0,
            ufo_dir = 1,
            bullet_active = jnp.array(0).astype(jnp.int32),
            bullet_x = jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            bullet_y = jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            enemy_bullets_active = jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.bool),
            enemy_bullets_x = jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_bullets_y = jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_fire_cooldown = jnp.array(self.consts.ENEMY_FIRE_RATE).astype(jnp.int32),
            barricade_health=jnp.full(
                (3, self.consts.BARRICADE_GRID_SHAPE[0], self.consts.BARRICADE_GRID_SHAPE[1]),
                self.consts.BARRICADE_HEALTH_INITIAL,
                dtype=jnp.int32
            ),
            active_barricade=jnp.array(1, dtype=jnp.int32)
        )

        return state

    def step_paused(self, state: SpaceInvadersState) -> SpaceInvadersState:
        new_player_dead = jax.lax.cond(
            state.player_dead + 1 > self.consts.FULL_PAUSE_DURATION,
            lambda _: 0,
            lambda _: state.player_dead + 1,
            None
        )

        # Resets the level only once during the pause animation
        state = jax.lax.cond(
            state.player_dead == self.consts.PLAYER_EXPLOSION_DURATION + 1,
            lambda: self.reset_level(state),
            lambda: state
        )
        state = state.replace(player_dead = new_player_dead)

        return state

    def step_running(self, state: SpaceInvadersState, action: chex.Array, key) -> SpaceInvadersState:
        new_player_x, new_player_speed = self._player_step(state.player_x, state.player_speed, action)

        new_player_x, new_player_speed = jax.lax.cond(
            jnp.logical_and(state.step_counter % 2 == 0, state.player_dead == 0),
            lambda _: (new_player_x, new_player_speed),
            lambda _: (state.player_x, state.player_speed),
            operand=None,
        )

        new_bullet_active, new_bullet_x, new_bullet_y = self._player_bullet_step(state, action)

        new_bullet_state = state.replace(
            bullet_active=new_bullet_active,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y
        )
        
        new_destroyed, new_score, final_bullet_active, new_ufo_state = self._check_bullet_enemy_collisions(new_bullet_state)

        def find_bounds(arr, axis, limit):
            # A position is occupied if the value is not 29 (29 = fully dead/gone)
            # 0 = alive, 1..28 = exploding. Both count as "occupying space" for movement bounds.
            mask = arr < 29
            
            # Check which rows/cols have at least one occupied cell
            has_occupied = jnp.any(mask, axis=axis)
            
            # Identify first and last indices
            first = jnp.argmax(has_occupied)
            last = limit - 1 - jnp.argmax(has_occupied[::-1])
            
            # Handle case where everything is dead (has_occupied are all False)
            any_occupied = jnp.any(has_occupied)
            first = jnp.where(any_occupied, first, -1)
            last = jnp.where(any_occupied, last, -1)

            return first, last

        dest = new_destroyed.reshape(self.consts.ENEMY_ROWS, self.consts.ENEMY_COLS)
        row_top, row_bottom = find_bounds(dest, 1, self.consts.ENEMY_ROWS)
        col_left, col_right = find_bounds(dest, 0, self.consts.ENEMY_COLS)

        # FIX 1: Calculate count correctly (inclusive range)
        col_count = jnp.maximum(0, col_right - col_left + 1)
        row_count = jnp.maximum(0, row_bottom - row_top + 1)

        new_rect_width = self.consts.OPPONENT_SIZE[0] * col_count + self.consts.OFFSET_OPPONENT[0] * jnp.maximum(0, col_count - 1)
        new_rect_height = self.consts.OPPONENT_SIZE[1] * row_count + self.consts.OFFSET_OPPONENT[1] * jnp.maximum(0, row_count - 1)

        # --- ENEMY BULLETS LOGIC (Keep existing) ---
        enemy_bullet_key, ufo_key = jax.random.split(key, 2)
        enemy_bullets_active, enemy_bullets_x, enemy_bullets_y, enemy_fire_cooldown = self._update_enemy_bullets(
            state.replace(
                destroyed=new_destroyed,
                enemy_bullets_active=state.enemy_bullets_active,
                enemy_bullets_x=state.enemy_bullets_x,
                enemy_bullets_y=state.enemy_bullets_y,
                enemy_fire_cooldown=state.enemy_fire_cooldown,
            ),
            enemy_bullet_key
        )

        new_lives, final_enemy_bullets_active = self._check_enemy_bullet_player_collisions(
            state.replace(
                player_x=new_player_x,
                enemy_bullets_active=enemy_bullets_active,
                enemy_bullets_x=enemy_bullets_x,
                enemy_bullets_y=enemy_bullets_y
            )
        )
        new_player_dead = jnp.where(new_lives != state.player_lives, 1, 0)

        # --- NEW SPEED CALCULATION ---
        # 1. Count total dead enemies (non-zero entries in destroyed array)
        num_destroyed = jnp.sum(new_destroyed != 0)

        # 2. Determine Speed Level
        # Compare current destroyed count against thresholds [15, 29, 33, 34, 35]
        # Summing the boolean results gives the index for the rates array.
        # e.g., if 0 destroyed: sum([F,F,F,F,F]) = 0 -> rate 32
        # e.g., if 15 destroyed: sum([T,F,F,F,F]) = 1 -> rate 16
        speed_level = jnp.sum(num_destroyed >= self.consts.MOVEMENT_THRESHOLDS)
        
        # 3. Get the dynamic rate
        current_rate = self.consts.MOVEMENT_RATES[speed_level]
        # -----------------------------

        # --- MOVEMENT LOGIC (Updated to use current_rate) ---
        def get_opponent_position():
            # ... (Keep your corrected bounds logic from the previous answer) ...
            # (Just reusing the logic here for context, no changes inside this function needed)
            stride_x = self.consts.OPPONENT_SIZE[0] + self.consts.OFFSET_OPPONENT[0]
            visual_left_x = state.opponent_current_x + (col_left * stride_x)
            visual_right_x = state.opponent_current_x + (col_right * stride_x) + self.consts.OPPONENT_SIZE[0]
            
            direction = jax.lax.cond(
                state.opponent_direction < 0,
                lambda: jax.lax.cond(visual_left_x <= self.consts.OPPONENT_LIMIT_X[0], lambda: 1, lambda: -1),
                lambda: jax.lax.cond(visual_right_x >= self.consts.OPPONENT_LIMIT_X[1], lambda: -1, lambda: 1)
            )

            new_position_y = jax.lax.cond(
                direction != state.opponent_direction,
                lambda y: y + self.consts.OPPONENT_VERTICAL_STEP_SIZE,
                lambda y: y,
                state.opponent_current_y
            )

            new_position_x = state.opponent_current_x + direction
            return (direction, new_position_x, new_position_y)

        # USE THE DYNAMIC RATE HERE
        is_opponent_step = state.step_counter % current_rate == 0
        
        new_enemy_flip = jax.lax.cond(
            is_opponent_step,
            lambda: 1 - state.enemy_flip,
            lambda: state.enemy_flip
        )
        
        all_dead = col_left == -1

        (direction, new_position_x, new_position_y) = jax.lax.cond(
            jnp.logical_and(is_opponent_step, jnp.logical_not(all_dead)),
            lambda: get_opponent_position(),
            lambda: (state.opponent_direction, state.opponent_current_x, state.opponent_current_y)
        )

        # If enemies overlap barricade line, deactivate barricades
        barricade_state = new_position_y + state.opponent_bounding_rect[1] > self.consts.BARRICADE_POS[1]
        new_active_barricade = jax.lax.cond(
            barricade_state,
            lambda: jnp.array(0, dtype=jnp.int32),
            lambda: state.active_barricade
        )
        new_barricade_health = state.barricade_health

        # Apply enemy bullet damage (only if barricade is active)
        final_enemy_bullets_active, new_barricade_health = jax.lax.cond(
            state.active_barricade == 1,
            lambda: self._update_barricade_health(
                new_barricade_health,
                enemy_bullets_x,
                enemy_bullets_y,
                final_enemy_bullets_active
            ),
            lambda: (final_enemy_bullets_active, new_barricade_health)
        )

        # Apply player bullet damage (only if barricade is active)
        p_active_arr = jnp.array([final_bullet_active])
        p_x_arr = jnp.array([new_bullet_x])
        p_y_arr = jnp.array([new_bullet_y])
        p_active_arr, new_barricade_health = jax.lax.cond(
            state.active_barricade == 1,
            lambda: self._update_barricade_health(
                new_barricade_health,
                p_x_arr,
                p_y_arr,
                p_active_arr
            ),
            lambda: (p_active_arr, new_barricade_health)
        )
        final_bullet_active = p_active_arr[0]

        # Ufo Controlling
        dir_random = jax.random.choice(ufo_key, jnp.array([1, -1]))

        new_ufo_state, new_ufo_dir, new_ufo_x = jax.lax.cond(
            state.step_counter % self.consts.UFO_START == 0,
            lambda: (1, dir_random, jnp.argmax(jnp.array([0, dir_random * (-1)])) * self.consts.WIDTH),
            lambda: (
                jax.lax.cond(
                    new_ufo_state > 1,
                    lambda: new_ufo_state + 1,
                    lambda: new_ufo_state
                ),
                state.ufo_dir,
                state.ufo_x
            )
        )

        new_ufo_x, new_ufo_state = jax.lax.cond(
            jnp.any(jnp.array([new_ufo_x < 0, new_ufo_x > self.consts.WIDTH, new_ufo_state > self.consts.EXPLOSION_FRAMES[3]])),
            lambda: (0, 0),
            lambda: (new_ufo_x, new_ufo_state)
        )

        new_ufo_x = jax.lax.cond(
            jnp.logical_and(
                new_ufo_state == 1,
                state.step_counter % self.consts.UFO_MOVEMENT_RATE == 0
            ),
            lambda x: x + new_ufo_dir,
            lambda x: x,
            new_ufo_x
        )

        # State Update
        new_state = SpaceInvadersState(
            player_x=new_player_x,
            player_speed=new_player_speed,
            player_dead=new_player_dead,
            step_counter=state.step_counter,
            player_score=new_score,
            player_lives=new_lives,
            destroyed=new_destroyed,
            opponent_current_x=new_position_x,
            opponent_current_y=new_position_y,
            opponent_bounding_rect=(new_rect_width, new_rect_height),
            opponent_direction=direction,
            ufo_x = new_ufo_x,
            ufo_state = new_ufo_state,
            ufo_dir = new_ufo_dir,
            bullet_active=final_bullet_active.astype(jnp.int32),
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y,
            enemy_bullets_active=final_enemy_bullets_active,
            enemy_bullets_x=enemy_bullets_x,
            enemy_bullets_y=enemy_bullets_y,
            enemy_fire_cooldown=enemy_fire_cooldown,
            barricade_health=new_barricade_health,
            active_barricade=new_active_barricade,
            enemy_flip=new_enemy_flip,
            rng=state.rng
        )

        wave_cleared = jnp.all(new_state.destroyed >= 29)

        final_state = jax.lax.cond(
            wave_cleared,
            lambda: self._next_wave(new_state), # Start new wave
            lambda: new_state                   # Continue current wave
        )

        return final_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SpaceInvadersState, action: chex.Array) -> Tuple[SpaceInvadersObservation, SpaceInvadersState, float, bool, SpaceInvadersInfo]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        new_step_counter = state.step_counter + 1
        state = state.replace(step_counter = new_step_counter)

        # Split the key for this step to avoid reuse
        step_key, next_rng = jax.random.split(state.rng)

        new_state: SpaceInvadersState = jax.lax.cond(
            state.player_dead > 0,
            lambda: self.step_paused(state),
            lambda: self.step_running(state, atari_action, step_key)
        )

        # Persist next RNG regardless of branch path
        new_state = new_state.replace(rng=next_rng)

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def observation_space(self) -> spaces.Dict:
        # Use self.consts directly (cast to int for concrete values)
        h = int(self.consts.HEIGHT)
        w = int(self.consts.WIDTH)
        screen_size = (h, w)
        
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        
        # Enemies: 6 rows * 6 cols = 36
        num_enemies = int(self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS)
        
        return spaces.Dict({
            "player": single_obj,
            "enemies": spaces.get_object_space(n=num_enemies, screen_size=screen_size),
            "player_bullet": single_obj,
            "enemy_bullets": spaces.get_object_space(n=self.consts.MAX_ENEMY_BULLETS, screen_size=screen_size),
            "ufo": single_obj,
            
            # Barricades are dense grids (health 0-6)
            "barricade_health": spaces.Box(
                low=0, 
                high=self.consts.BARRICADE_HEALTH_INITIAL, 
                shape=(3, self.consts.BARRICADE_GRID_SHAPE[0], self.consts.BARRICADE_GRID_SHAPE[1]), 
                dtype=jnp.int32
            ),
            
            "score_player": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
        })

    def render(self, state: SpaceInvadersState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SpaceInvadersState) -> SpaceInvadersObservation:
        c = self.consts
        w, h = int(c.WIDTH), int(c.HEIGHT)

        # --- Player ---
        player = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.player_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(c.PLAYER_Y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(c.PLAYER_SIZE[1], dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32)
        )

        # --- Enemies ---
        total_enemies = c.ENEMY_ROWS * c.ENEMY_COLS
        idx = jnp.arange(total_enemies)
        rows = idx // c.ENEMY_COLS
        cols = idx % c.ENEMY_COLS

        # Vectorized position calculation
        def get_pos(r, c):
            return self._get_enemy_position(state.opponent_current_x, state.opponent_current_y, r, c)
        
        ex, ey = jax.vmap(get_pos)(rows, cols)
        e_active = (state.destroyed == 0).astype(jnp.int32) # 0 means alive in 'destroyed' array
        
        # Determine visual ID based on row (3 types usually)
        # Top 2 rows = 2, Middle 2 = 1, Bottom 2 = 0
        e_vid = jnp.where(rows < 2, 2, jnp.where(rows < 4, 1, 0)).astype(jnp.int32)

        enemies = ObjectObservation.create(
            x=jnp.clip(ex.astype(jnp.int32), 0, w),
            y=jnp.clip(ey.astype(jnp.int32), 0, h),
            width=jnp.full((total_enemies,), c.OPPONENT_SIZE[0], dtype=jnp.int32),
            height=jnp.full((total_enemies,), c.OPPONENT_SIZE[1], dtype=jnp.int32),
            active=e_active,
            visual_id=e_vid
        )

        # --- Player Bullet ---
        pb_active = state.bullet_active.astype(jnp.int32)
        player_bullet = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.bullet_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(state.bullet_y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.BULLET_SIZE[0], dtype=jnp.int32),
            height=jnp.array(c.BULLET_SIZE[1], dtype=jnp.int32),
            active=pb_active
        )

        # --- Enemy Bullets ---
        eb_active = state.enemy_bullets_active.astype(jnp.int32)
        enemy_bullets = ObjectObservation.create(
            x=jnp.clip(state.enemy_bullets_x, 0, w),
            y=jnp.clip(state.enemy_bullets_y, 0, h),
            width=jnp.full((c.MAX_ENEMY_BULLETS,), c.BULLET_SIZE[0], dtype=jnp.int32),
            height=jnp.full((c.MAX_ENEMY_BULLETS,), c.BULLET_SIZE[1], dtype=jnp.int32),
            active=eb_active
        )

        # --- UFO ---
        # ufo_state: 1=alive
        ufo_active = (state.ufo_state == 1).astype(jnp.int32)
        ufo = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.ufo_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(c.UFO_Y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.UFO_SIZE[0], dtype=jnp.int32),
            height=jnp.array(c.UFO_SIZE[1], dtype=jnp.int32),
            active=ufo_active
        )

        return SpaceInvadersObservation(
            player=player,
            enemies=enemies,
            player_bullet=player_bullet,
            enemy_bullets=enemy_bullets,
            ufo=ufo,
            barricade_health=state.barricade_health,
            score_player=state.player_score,
            lives=state.player_lives
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )
    
    def _next_wave(self, state: SpaceInvadersState) -> SpaceInvadersState:
            """
            Prepares the state for the next wave:
            1. Resets Enemy Positions (this 'reactivates' shields by moving enemies away from them).
            2. Resets Enemies to Alive.
            3. Resets Bullets.
            4. PRESERVES Barricade Health (does not repair them).
            """
            return state.replace(
                # Reset Player Position (Optional, matches original game feel)
                player_x=jnp.array(self.consts.INITIAL_PLAYER_X).astype(jnp.int32),
                player_speed=jnp.array(0.0).astype(jnp.int32),
                
                # Revive all enemies
                destroyed=jnp.zeros((self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS,), dtype=jnp.int32),
                
                # Reset Enemy Positions (moves them back to top)
                opponent_current_x=self.consts.OPPONENT_LIMIT_X[0],
                opponent_current_y=self.consts.OPPONENT_LIMIT_Y[0],
                opponent_direction=self.consts.INITIAL_OPPONENT_DIRECTION,
                
                # Reset UFO
                ufo_x=0,
                ufo_state=0,
                ufo_dir=1,
                
                # Reset Bullets
                bullet_active=jnp.array(0).astype(jnp.int32),
                bullet_x=jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
                bullet_y=jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
                enemy_bullets_active=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.bool),
                enemy_bullets_x=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
                enemy_bullets_y=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
                enemy_fire_cooldown=jnp.array(self.consts.ENEMY_FIRE_RATE).astype(jnp.int32),
                
                # --- CRITICAL: Keep current barricade health ---
                # Resetting enemy positions above stops the 'wipe' logic in step_running,
                # effectively reactivating the shields with their existing damage.
                barricade_health=state.barricade_health,
                active_barricade=jnp.array(1, dtype=jnp.int32)
            )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SpaceInvadersState) -> SpaceInvadersInfo:
        return SpaceInvadersInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SpaceInvadersState, state: SpaceInvadersState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SpaceInvadersState) -> bool:
        # The game only ends if the player runs out of lives.
        return state.player_lives <= 0

class SpaceInvadersRenderer(JAXGameRenderer):
    def __init__(self, consts: SpaceInvadersConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or SpaceInvadersConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160), # H, W
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Define sprite path
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "spaceinvaders")
        
        # 3. Use asset config from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 4. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # Replace opaque black (ID 0) pixels in player sprites with TRANSPARENT_ID
        player_masks = self.SHAPE_MASKS['player']
        transparent_player_masks = jnp.where(
            player_masks == 0,
            self.jr.TRANSPARENT_ID,
            player_masks
        )
        self.SHAPE_MASKS['player'] = transparent_player_masks
        
        # The BACKGROUND sprite is only (15, 160) for the score area.
        # We need to create a full-size raster to match the target rendering dimensions.
        # Determine target dimensions (same logic as renderer: downscale if set, else game_dimensions)
        if self.config.downscale:
            target_h, target_w = self.config.downscale[0], self.config.downscale[1]
        else:
            target_h, target_w = self.config.game_dimensions[0], self.config.game_dimensions[1]
        
        bg_h, bg_w = self.BACKGROUND.shape
        if bg_h != target_h or bg_w != target_w:
            # Create full-size raster filled with a proper background color ID (prefer black if available)
            try:
                black_id = self.COLOR_TO_ID[(0, 0, 0)]
            except KeyError:
                # Fallback: use the first palette entry (usually background-like)
                black_id = jnp.array(0, dtype=self.BACKGROUND.dtype)
            full_background = jnp.full((target_h, target_w), black_id, dtype=self.BACKGROUND.dtype)
            # Place the small background at the bottom (score area)
            full_background = full_background.at[target_h - bg_h:target_h, :bg_w].set(self.BACKGROUND)
            self.BACKGROUND = full_background
        
        # 5. Pre-compute/cache values for rendering
        # Determine a valid barricade color ID (non-transparent) from the defense mask
        _def_mask = self.SHAPE_MASKS['defense']
        _transparent = self.jr.TRANSPARENT_ID
        # Pick the max non-transparent ID as a stable color id
        self.barricade_color_id = int(jnp.max(jnp.where(_def_mask != _transparent, _def_mask, 0)))
        
        # Cache the stacked animation frames for enemies
        self.opponent_sprites_a = jnp.stack([
            self.SHAPE_MASKS['opponent_1'][0],
            self.SHAPE_MASKS['opponent_2'][0],
            self.SHAPE_MASKS['opponent_3'][0],
            self.SHAPE_MASKS['opponent_4'][0],
            self.SHAPE_MASKS['opponent_5'][0], # index 4
            self.SHAPE_MASKS['opponent_6'][0]  # index 5
        ])
        self.opponent_sprites_b = jnp.stack([
            self.SHAPE_MASKS['opponent_1'][1],
            self.SHAPE_MASKS['opponent_2'][1],
            self.SHAPE_MASKS['opponent_3'][1],
            self.SHAPE_MASKS['opponent_4'][1],
            self.SHAPE_MASKS['opponent_5'][1], # index 4
            self.SHAPE_MASKS['opponent_6'][1]  # index 5
        ])
        
        # Cache barricade pixel coordinate grids (legacy per-pixel utils if needed)
        self._barricade_xx, self._barricade_yy = self._precompute_barricade_coords()

        # Precompute barricade pixel -> chunk lookup for health-grid masking
        self._precompute_barricade_pixel_map()


    def _precompute_barricade_coords(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Creates global coordinate grids for all 4 barricades.
        This is used by the optimized `render_defense` function.
        """
        # Infer barricade mask shape (H, W) from the defense sprite mask
        h, w = self.SHAPE_MASKS['defense'].shape
        barricade_shape = (h, w)
        
        # Create a single barricade's local grid
        xx, yy = jnp.meshgrid(jnp.arange(barricade_shape[1]), jnp.arange(barricade_shape[0]))
        
        # Get the origin (top-left) x,y for each of the 4 barricades
        origins_x = jnp.array(self.consts.BARRICADE_POS[0])
        origin_y = self.consts.BARRICADE_POS[1]
        
        # Tile the local grids and add the origins
        # Shape becomes (num_barricades, H, W)
        num_barricades = origins_x.shape[0]
        all_xx = jnp.tile(xx[None, ...], (num_barricades, 1, 1)) + origins_x[:, None, None]
        all_yy = jnp.tile(yy[None, ...], (num_barricades, 1, 1)) + origin_y
        
        return all_xx, all_yy


    def get_player_sprite(self, state: SpaceInvadersState):
        return jax.lax.cond(
            state.player_dead > self.consts.PLAYER_EXPLOSION_DURATION,
            lambda _: jnp.where(jnp.floor((state.player_dead - 1) / self.consts.PLAYER_RESET_FRAMES) % 2 == 0, self.SHAPE_MASKS['player'][0], self.SHAPE_MASKS['player'][1]),
            lambda _: jnp.where(jnp.floor(state.player_dead / self.consts.PLAYER_EXPLOSION_FRAMES) % 2 == 0, self.SHAPE_MASKS['player_explosion'][0], self.SHAPE_MASKS['player_explosion'][1]),
            None
        )

    def _precompute_barricade_chunks(self):
        # Choose simple logical chunk sprite size for rendering (approximate)
        chunk_h, chunk_w = (4, 2)
        barricade_color_id = self.barricade_color_id
        self.barricade_chunk_full = jnp.full((chunk_h, chunk_w), barricade_color_id, dtype=jnp.uint8)
        # Damaged variant: stripe rows set to TRANSPARENT_ID to let background show through
        self.barricade_chunk_damaged = self.barricade_chunk_full.at[::2, :].set(self.jr.TRANSPARENT_ID)
        self.barricade_damage_threshold = self.consts.BARRICADE_HEALTH_INITIAL // 2

        # Precompute positions for each chunk in each barricade
        base_xs = self.consts.BARRICADE_POS[0]
        base_y = self.consts.BARRICADE_POS[1]
        grid_h, grid_w = self.consts.BARRICADE_GRID_SHAPE
        spacing_x, spacing_y = (1, 1)
        y_offsets = jnp.arange(grid_h) * (chunk_h + spacing_y)
        x_offsets = jnp.arange(grid_w) * (chunk_w + spacing_x)
        # Shapes: (3, 4, 3)
        self.chunk_y_pos = (base_y + y_offsets[None, :, None]).astype(jnp.int32)
        self.chunk_x_pos = (base_xs[:, None, None] + x_offsets[None, None, :]).astype(jnp.int32)

    def _precompute_barricade_pixel_map(self):
        # Build pixel->chunk lookup using the ACTUAL (possibly downscaled) defense sprite size
        base_sprite = self.SHAPE_MASKS['defense']
        sprite_h, sprite_w = base_sprite.shape
        grid_h, grid_w = self.consts.BARRICADE_GRID_SHAPE
        yy, xx = jnp.meshgrid(jnp.arange(sprite_h), jnp.arange(sprite_w), indexing='ij')
        # Map pixel coordinates into chunk indices proportionally
        row_map = (yy / sprite_h * grid_h).astype(jnp.int32)
        col_map = (xx / sprite_w * grid_w).astype(jnp.int32)
        self.barricade_pixel_to_chunk_row = jnp.clip(row_map, 0, grid_h - 1)
        self.barricade_pixel_to_chunk_col = jnp.clip(col_map, 0, grid_w - 1)

    def render_life(self, state: SpaceInvadersState, raster):
        life_mask = self.SHAPE_MASKS['lives'][state.player_lives - 1]
        raster = self.jr.render_at(raster, self.consts.POSITION_LIFE_X, self.consts.PLAYER_Y, life_mask)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render_score(self, state: SpaceInvadersState, raster):
        digits = self.jr.int_to_digits(state.player_score, max_digits=4)
        score_pos_x = jnp.array([3, 6, 9, 12], dtype=jnp.int32)
        score_pos_y = jnp.array([9, 10, 10, 9], dtype=jnp.int32)
        yellow_zero_mask = self.SHAPE_MASKS['zero_yellow']

        def render_digit_and_zero(i, r):
            xg = score_pos_x[i] + i * self.consts.NUMBER_SIZE[0]
            yg = score_pos_y[i]
            digit_mask = self.SHAPE_MASKS['digits_green'][digits[i]]
            r = self.jr.render_at(r, xg, yg, digit_mask)
            xy = self.consts.NUMBER_YELLOW_OFFSET + score_pos_x[i] + i * self.consts.NUMBER_SIZE[0]
            yy = score_pos_y[i]
            return self.jr.render_at(r, xy, yy, yellow_zero_mask)

        return jax.lax.fori_loop(0, 4, render_digit_and_zero, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_defense(self, state: SpaceInvadersState, raster):
        """
        Render the original (18x8) defense sprite masked by the barricade health grid
        via the precomputed pixel->chunk map.
        Only render if active_barricade is true.
        """
        base_sprite = self.SHAPE_MASKS['defense']

        def render_single_barricade(i, current_raster):
            health_grid = state.barricade_health[i]
            pixel_health = health_grid[self.barricade_pixel_to_chunk_row, self.barricade_pixel_to_chunk_col]
            pixel_mask = (pixel_health > 0)
            # Use TRANSPARENT_ID for invisible pixels to avoid black artifacts
            masked_sprite = jnp.where(pixel_mask, base_sprite, jnp.asarray(self.jr.TRANSPARENT_ID, base_sprite.dtype))
            x_pos = self.consts.BARRICADE_POS[0][i]
            y_pos = self.consts.BARRICADE_POS[1]
            return self.jr.render_at(current_raster, x_pos, y_pos, masked_sprite)

        # Only render barricades if active_barricade is 1
        return jax.lax.cond(
            state.active_barricade == 1,
            lambda: jax.lax.fori_loop(0, 3, render_single_barricade, raster),
            lambda: raster
        )

    def render_explosion(self, state: SpaceInvadersState, raster, x, y, frame_id):
        sprite_id = jnp.searchsorted(self.consts.EXPLOSION_FRAMES, frame_id)
        sprite_mask = self.SHAPE_MASKS['enemy_explosion'][sprite_id]
        return self.jr.render_at_clipped(raster, x, y, sprite_mask)

    @partial(jax.jit, static_argnums=(0,))
    def render_enemies(self, state: SpaceInvadersState, raster):
        # 1) Determine scaled sprite and step sizes to avoid drift and mis-centering
        flip = jax.lax.cond(
            state.player_dead != 0,
            lambda: False,
            lambda: state.enemy_flip == 1 # Simple check against the state
        )
        
        sprites_to_render = jax.lax.cond(
            flip,
            lambda: self.opponent_sprites_b,
            lambda: self.opponent_sprites_a
        )

        # Use the actual scaled sprite size for layout
        sample_sprite = sprites_to_render[0]
        sprite_h = sample_sprite.shape[0]
        sprite_w = sample_sprite.shape[1]

        # Calculate scaled step sizes (gap between top-left corners)
        step_x = sprite_w + int(round(self.consts.OFFSET_OPPONENT[0] * float(self.config.width_scaling)))
        step_y = sprite_h + int(round(self.consts.OFFSET_OPPONENT[1] * float(self.config.height_scaling)))
        # Calculate canvas size
        canvas_h = (self.consts.ENEMY_ROWS - 1) * step_y + sprite_h
        canvas_w = (self.consts.ENEMY_COLS - 1) * step_x + sprite_w
        enemy_canvas = jnp.full((canvas_h, canvas_w), self.jr.TRANSPARENT_ID, dtype=jnp.uint8)

        total = self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS
        indices = jnp.arange(total)
        row_indices = indices // self.consts.ENEMY_COLS
        col_indices = indices % self.consts.ENEMY_COLS

        # Use step sizes computed above for consistent placement
        flat_x_pos = (col_indices * step_x).astype(jnp.int32)
        flat_y_pos = (row_indices * step_y).astype(jnp.int32)

        destroyed_timers = state.destroyed
        is_alive = destroyed_timers == 0
        is_exploding = (destroyed_timers > 0) & (destroyed_timers < 29)
        alive_sprites = sprites_to_render[row_indices]
        explosion_sprite_ids = jnp.searchsorted(self.consts.EXPLOSION_FRAMES, destroyed_timers)
        explosion_sprites = self.SHAPE_MASKS['enemy_explosion'][explosion_sprite_ids]

        def render_single_enemy_to_canvas(i, current_canvas):
            x = flat_x_pos[i]  # This is the final SCALED x
            y = flat_y_pos[i]  # This is the final SCALED y

            # Select the correct scaled sprite
            sprite_mask = jnp.where(
                is_alive[i, None, None],
                alive_sprites[i],
                jnp.where(is_exploding[i, None, None], explosion_sprites[i], 0)
            )
            should_render = is_alive[i] | is_exploding[i]
            do_flip = (row_indices[i] == 4) & flip
            # This is an unscaled blit operation
            def stamp_sprite(canvas):
                # Apply flip
                flipped_mask = jax.lax.cond(
                    do_flip, lambda m: jnp.flip(m, axis=1), lambda m: m, sprite_mask
                )

                # Get the target slice from the canvas at the SCALED (y, x)
                target_slice = jax.lax.dynamic_slice(canvas, (y, x), flipped_mask.shape)
                # Merge the sprite over the slice
                updated_slice = jnp.where(flipped_mask != self.jr.TRANSPARENT_ID, flipped_mask, target_slice)
                # Write the merged slice back to the canvas
                return jax.lax.dynamic_update_slice(canvas, updated_slice, (y, x))

            # Only stamp if the enemy is alive or exploding
            return jax.lax.cond(
                should_render,
                stamp_sprite,
                lambda r: r,
                current_canvas
            )

        final_enemy_canvas = jax.lax.fori_loop(0, total, render_single_enemy_to_canvas, enemy_canvas)

        # 4) One render to the main raster using native game coordinates
        return self.jr.render_at_clipped(
            raster,
            state.opponent_current_x,
            state.opponent_current_y,
            final_enemy_canvas
        )

    def render_ufo(self, state: SpaceInvadersState, raster):
        def draw_alive(r):
            return self.jr.render_at_clipped(r, state.ufo_x, self.consts.UFO_Y, self.SHAPE_MASKS['ufo'])
        def draw_explosion(r):
            return self.render_explosion(state, r, state.ufo_x, self.consts.UFO_Y, state.ufo_state)
        return jax.lax.cond(state.ufo_state == 1, draw_alive, draw_explosion, raster)

    def render_bullets(self, state: SpaceInvadersState, raster):
        bullet_mask = self.SHAPE_MASKS['bullet']

        # Player Bullet (blink on even frames)
        def draw_player_bullet(r):
            return self.jr.render_at_clipped(r, state.bullet_x, state.bullet_y, bullet_mask)
        raster = jax.lax.cond(
            (state.step_counter % 2 == 0) & state.bullet_active,
            draw_player_bullet,
            lambda r: r,
            raster
        )

        # Enemy Bullets
        def render_enemy_bullet(i, current_raster):
            should_render = state.enemy_bullets_active[i] & (state.step_counter % 2 == 0)
            return jax.lax.cond(
                should_render,
                lambda r: self.jr.render_at_clipped(r, state.enemy_bullets_x[i], state.enemy_bullets_y[i], bullet_mask),
                lambda r: r,
                current_raster
            )
        return jax.lax.fori_loop(0, self.consts.MAX_ENEMY_BULLETS, render_enemy_bullet, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SpaceInvadersState) -> chex.Array:
        # Start with a raster matching renderer resolution
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render Player
        player_mask = jax.lax.cond(
            state.player_dead == 0,
            lambda: self.SHAPE_MASKS['player'][0],
            lambda: self.get_player_sprite(state)
        )
        raster = self.jr.render_at(raster, state.player_x - self.consts.PLAYER_SIZE[0], self.consts.PLAYER_Y, player_mask)

        # Render Life Counter
        raster = jax.lax.cond(
            state.player_dead > self.consts.PLAYER_EXPLOSION_DURATION + 1,
            lambda r: self.render_life(state, r),
            lambda r: r,
            raster
        )

        # Render Bullets (Player and Enemy)
        raster = self.render_bullets(state, raster)

        # Render Score
        raster = jax.lax.cond(state.ufo_state == 0, lambda: self.render_score(state, raster), lambda: raster)

        # Render Defense Object
        raster = self.render_defense(state, raster)

        raster = self.render_enemies(state, raster)

        # Render UFO (if active)
        raster = jax.lax.cond(state.ufo_state != 0, lambda: self.render_ufo(state, raster), lambda: raster)

        # Final conversion from palette IDs to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)

