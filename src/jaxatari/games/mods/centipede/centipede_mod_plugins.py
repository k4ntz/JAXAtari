import random
from functools import partial
import time
from typing import Any

import chex
import jax
import jax.numpy as jnp

from jaxatari.games.jax_centipede import CentipedeState
from jaxatari.games.jax_centipede import JaxCentipede
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.wrappers import JaxatariWrapper

class SlowSpellMod(JaxAtariInternalModPlugin):
    """Player spells have a third the speed."""
    conflicts_with = ["fast_spell"]
    constants_overrides = {
        "PLAYER_SPELL_SPEED": 3,
    }

class FastSpellMod(JaxAtariInternalModPlugin):
    """Player spells have double the speed."""
    conflicts_with = ["slow_spell"]
    constants_overrides = {
        "PLAYER_SPELL_SPEED": 18,
    }

class MaxLivesResetMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "PLAYER_LIVES_RESET": 6,
    }

class RandomMushroomsMod(JaxAtariPostStepModPlugin):
    """Initialize mushroom positions randomly."""
    """def __init__(self, env):
        super().__init__(env)
        self._env = env
        # Overrides initialize_mushroom_positions from env
        self._env.initialize_mushroom_positions = self.initialize_mushroom_positions.__get__(self._env)"""

    @partial(jax.jit, static_argnums=(0,))
    def spawn_mushrooms(self, key, centipede, p: jnp.ndarray = jnp.array(0.0888)) -> chex.Array:
        # Overrides the default function from the env
        rows = jnp.arange(self._env.consts.MUSHROOM_NUMBER_OF_ROWS) # 19
        cols = jnp.arange(self._env.consts.MUSHROOM_NUMBER_OF_COLS) # 16

        spawn = jax.random.bernoulli(key, p, (19,16))

        # --- Per-cell computation ---
        def cell_fn(row, col):
            row_is_even = (row % 2) == 0
            column_start = jnp.where(
                row_is_even,
                self._env.consts.MUSHROOM_COLUMN_START_EVEN,
                self._env.consts.MUSHROOM_COLUMN_START_ODD,
            )
            x = column_start + self._env.consts.MUSHROOM_X_SPACING * col
            y = row * self._env.consts.MUSHROOM_Y_SPACING + 7
            lives = jnp.where(spawn[row, col] != 0, 3, 0)
            return jnp.array([x, y, 0, lives], dtype=jnp.int32)

        # Vectorize across grid with nested vmaps
        grid = jax.vmap(lambda r: jax.vmap(lambda c: cell_fn(r, c))(cols))(rows)

        # Flatten to (N*M, 4)
        mushrooms = grid.reshape(-1, 4)

        # Remove mushrooms that would spawn overlapping any centipede segment.
        # centipede: (9,5) -> use first two columns as positions, and column 3 as 'alive' flag
        centipede_pos = centipede[:, :2]
        centipede_alive = centipede[:, 3]

        # single mushroom vs single segment collision check
        def check_against_segment(m_pos, seg_pos, seg_alive):
            return jnp.where(
                seg_alive != 0,
                self._env.check_collision_single(
                    pos1=m_pos,
                    size1=self._env.consts.MUSHROOM_SIZE,
                    pos2=seg_pos,
                    size2=self._env.consts.SEGMENT_SIZE,
                ),
                False,
            )

        # for a single mushroom, check collision against all segments and any collision -> True
        def mushroom_collides(mush):
            m_pos = mush[:2]
            collisions = jax.vmap(lambda seg_pos, seg_alive: check_against_segment(m_pos, seg_pos, seg_alive))(centipede_pos, centipede_alive)
            return jnp.any(collisions)

        colliding_mask = jax.vmap(mushroom_collides)(mushrooms)

        # Set lives to 0 for mushrooms that collide with the centipede (i.e., remove them)
        new_lives = jnp.where(colliding_mask, 0, mushrooms[:, 3])
        mushrooms = mushrooms.at[:, 3].set(new_lives.astype(jnp.int32))

        return mushrooms

    def run(self, prev_state, new_state):
        """
        if prev_state is None:
            p = jnp.array(0.0888)   # ~ 27 / 304 -> roughly same number of mushrooms on screen
            new_mushroom_positions = self.spawn_mushrooms(p=p)
            return new_state.replace(mushroom_positions=new_mushroom_positions)
        """

        num_mushrooms = jnp.sum(new_state.mushroom_positions[:, 3] > 0)
        p = num_mushrooms / 304  # Adjust probability based on current number of mushrooms
        mush_key, rng_key = jax.random.split(new_state.rng_key)
        new_mushroom_positions = self.spawn_mushrooms(mush_key, new_state.centipede_position, p=p)

        cond = jnp.logical_or(
            jnp.equal(prev_state.step_counter, 0),
            jnp.logical_or(
                jnp.invert(jnp.all(jnp.equal(prev_state.wave, new_state.wave))),
                jnp.not_equal(prev_state.lives, new_state.lives)
            )
        )

        return jax.lax.cond(
            cond,
            lambda: new_state.replace(mushroom_positions=new_mushroom_positions, rng_key=rng_key),
            lambda: new_state,
        )

class RandomPlayerMovementMod(JaxAtariPostStepModPlugin):
    def run(self, prev_state, new_state):
        act_key, ber_key, rng_key = jax.random.split(new_state.rng_key, 3)
        new_action = jax.random.choice(act_key, 17)
        new_player_x, _, new_vel_x = self._env.player_step(
            prev_state.player_x,
            prev_state.player_y,
            prev_state.player_velocity_x,
            new_action
        )
        return jax.lax.cond(
            jax.random.bernoulli(ber_key, p=0.5),
            lambda: new_state,
            lambda: new_state.replace(player_x=new_player_x, player_velocity_x=new_vel_x)
        )

class DeadlyMushroomsMod(JaxAtariPostStepModPlugin):
    """Mushrooms are deadly to the player on contact, instead of just being obstacles."""
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: CentipedeState, new_state: CentipedeState) -> CentipedeState:
        player_rect = jnp.array(
            [
                new_state.player_x,
                new_state.player_y,
                self._env.consts.PLAYER_SIZE[0],
                self._env.consts.PLAYER_SIZE[1],
            ],
            dtype=jnp.int32,
        )

        mushroom_x = new_state.mushroom_positions[:, 0]
        mushroom_y = new_state.mushroom_positions[:, 1]
        mushroom_lives = new_state.mushroom_positions[:, 3]

        mushroom_rects = jnp.stack(
            [
                mushroom_x,
                mushroom_y,
                jnp.full_like(mushroom_x, self._env.consts.MUSHROOM_SIZE[0]),
                jnp.full_like(mushroom_y, self._env.consts.MUSHROOM_SIZE[1]),
            ],
            axis=1,
        )

        collision_x = jnp.logical_and(
            player_rect[0] < mushroom_rects[:, 0] + mushroom_rects[:, 2],
            player_rect[0] + player_rect[2] > mushroom_rects[:, 0],
        )
        collision_y = jnp.logical_and(
            player_rect[1] < mushroom_rects[:, 1] + mushroom_rects[:, 3],
            player_rect[1] + player_rect[3] > mushroom_rects[:, 1],
        )
        collision = jnp.logical_and(collision_x, collision_y)
        deadly_collision = jnp.logical_and(collision, mushroom_lives > 0)

        was_alive_and_playing = prev_state.death_counter == 0
        should_trigger = jnp.logical_and(jnp.any(deadly_collision), was_alive_and_playing)

        new_death_counter = jnp.where(should_trigger, jnp.array(-1), new_state.death_counter)

        return new_state.replace(death_counter=new_death_counter)

class InvincibleMobsMod(JaxAtariInternalModPlugin):
    """Mobs (centipede excluded) are invincible to the player."""

    ## -------- Spider Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_spider_collision(
            self,
            spell_state: chex.Array,
            spider_position: chex.Array,
            score: chex.Array,
            player_y: chex.Array,
            spider_points: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:

        # Check if spell is still active
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        # Check if spider is still active
        spider_x, spider_y, spider_dir = spider_position
        spider_alive = spider_dir != 0

        # Default return (no collision, no sprite)
        def no_collision():
            return spell_state, spider_position, score, spider_points

        def check_hit():
            collision = self._env.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self._env.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([spider_x + 2, spider_y - 2]),
                size2=self._env.consts.SPIDER_SIZE,
            )

            def on_hit():
                new_spell = spell_state.at[2].set(0)
                return new_spell, spider_position, score, spider_points

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, spider_alive),
            check_hit,
            no_collision,
        )

    ## -------- Flea Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_flea_collision(
            self,
            spell_state: chex.Array,
            flea_position: chex.Array,
            flea_spawn_counter: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        # Spell info
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        flea_x, flea_y, flea_lives = flea_position
        flea_alive = flea_lives != 0

        # Default: no collision
        def no_collision():
            return spell_state, flea_position, flea_spawn_counter, score

        def check_hit():
            # Collision check
            collision = self._env.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self._env.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([flea_x, flea_y]),
                size2=self._env.consts.FLEA_SIZE,
            )

            def on_hit():
                new_spell = spell_state.at[2].set(0)
                return new_spell, flea_position, flea_spawn_counter, score

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, flea_alive),
            check_hit,
            no_collision
        )

    ## -------- Scorpion Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_scorpion_collision(
            self,
            spell_state: chex.Array,
            scorpion_position: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        # Spell info
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        # Scorpion info
        scorpion_x, scorpion_y, scorpion_dir, scorpion_speed = scorpion_position
        scorpion_alive = scorpion_dir != 0

        # Default: no collision
        def no_collision():
            return spell_state, scorpion_position, score, jnp.array(0, dtype=jnp.int32)

        def check_hit():
            # Collision check
            collision = self._env.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self._env.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([scorpion_x, scorpion_y]),
                size2=self._env.consts.SCORPION_SIZE,
            )

            def on_hit():
                new_spell = spell_state.at[2].set(0)
                return new_spell, scorpion_position, score, jnp.array(0, dtype=jnp.int32)

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, scorpion_alive),
            check_hit,
            no_collision
        )

class FriendlyMobsMod(JaxAtariInternalModPlugin):
    """Spiders are friendly to the player and do not harm them."""
    ## -------- Player Enemy Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_player_enemy_collision(
            self,
            player_x,
            player_y,
            centipede_position,
            spider_position,
            flea_position,
    ) -> chex.Array:
        # Get centipede params
        centipede_is_alive = jnp.any(centipede_position[:, 3] != 0)

        # Default: no collision
        def no_collision():
            return jnp.array(0)

        def check_hit():
            # Check Centipede Player collision
            def single_collision(c_xy, active):
                return jnp.where(
                    active != 0,
                    self._env.check_collision_single(
                        pos1=jnp.array([player_x, player_y + 1]),
                        size1=(4, 8),
                        pos2=c_xy,
                        size2=self._env.consts.SEGMENT_SIZE,
                    ),
                    False
                )

            centipede_collision = jax.vmap(single_collision)(
                centipede_position[:, :2],
                centipede_position[:, 3]
            )

            collision = jnp.any(centipede_collision)

            def on_hit():
                return jnp.array(-1)

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            centipede_is_alive,
            check_hit,
            no_collision
        )

class RandomCentipedeMod(JaxAtariInternalModPlugin):
    def __init__(self):
        super().__init__()
    ## -------- Centipede Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def initialize_centipede_positions(self, wave: chex.Array) -> chex.Array:
        # Generate key fresh each time using current time to ensure randomness across runs
        # (JAX JIT doesn't allow mutable state updates to propagate to Python level)
        key = jax.random.PRNGKey(time.time_ns() % (2 ** 32))
        key = jax.random.fold_in(key, wave[0].astype(jnp.int32))
        key, key_x, key_y = jax.random.split(key, 3)
        base_x = (16 + 4 * jax.random.randint(key_x, (1,), 0, 23))[0]  # min: 16, max: 108
        base_y = (5 + 9 * jax.random.randint(key_y, (1,), 0, 2))[0]  # min: 5, max: 23

        jax.debug.print("key: {}", key)

        wave = wave[0]
        slow_wave = wave < 0
        num_heads = jnp.abs(wave)
        main_segments = self._env.consts.MAX_SEGMENTS - num_heads

        def spawn_segment(i):
            def main_body():
                is_head = i == 0
                return jnp.where(
                    slow_wave,
                    jnp.where(
                        is_head,
                        jnp.array([base_x + 4 * i, base_y, -1, 1, 2]),
                        jnp.array([base_x + 4 * i, base_y, -1, 1, 1]),
                    ),
                    jnp.where(
                        is_head,
                        jnp.array([base_x + 4 * i, base_y, -2, 1, 2]),
                        jnp.array([base_x + 4 * i, base_y, -2, 1, 1]),
                    )
                )

            def single_head():      # May not be 100% accurate (1-2px offset, varying per round)
                nonlocal key
                key, key_x, key_y = jax.random.split(key, 3)
                x = (16 + 4 * jax.random.randint(key_x, (1,), 0, 23))[0]  # min: 16, max: 108
                y = (5 + 9 * jax.random.randint(key_y, (1,), 0, 2))[0]  # min: 5, max: 23
                return jnp.array([x, y, -2, 1, 2])

            return jax.lax.cond(
                i < main_segments,
                main_body,
                single_head,
            )

        carry = jnp.arange(0, 9)
        return jax.vmap(spawn_segment)(carry).astype(jnp.float32)

class FastCentipedeMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_centipede_collision(
            self,
            spell_state: chex.Array,
            centipede_position: chex.Array,
            mushroom_positions: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        spell_active = spell_state[2] != 0

        def no_hit():
            return (
                jnp.repeat(spell_active, centipede_position.shape[0]),
                centipede_position,
                jnp.repeat(0, centipede_position.shape[0]),
                jnp.repeat(0, centipede_position.shape[0]),
                jnp.repeat(-1, centipede_position.shape[0])
            )

        def check_single_segment(is_alive, seg):
            seg_pos = seg[:2]

            collision = self._env.check_collision_single(
                pos1=jnp.array([spell_state[0], spell_state[1]]),
                size1=self._env.consts.PLAYER_SPELL_SIZE,
                pos2=seg_pos,
                size2=self._env.consts.SEGMENT_SIZE,
            )

            def on_hit():
                mush_y = seg[1] + 2
                odd_mush_row = seg[1] % 2 == 0
                mush_x = jnp.where(
                    odd_mush_row,
                    jnp.where(
                        seg[2] > 0,
                        jnp.ceil(seg[0] / 8) * 8,
                        jnp.floor(seg[0] / 8) * 8,
                    ),
                    jnp.where(
                        seg[2] > 0,
                        jnp.ceil(seg[0] / 8) * 8 + 4,
                        jnp.floor(seg[0] / 8) * 8 + 4,
                    )
                )
                out_of_border = jnp.where(
                    odd_mush_row,
                    jnp.logical_or(
                        jnp.logical_and(seg[2] > 0, mush_x > 136),
                        jnp.logical_and(seg[2] < 0, mush_x < 16)
                    ),
                    jnp.logical_or(
                        jnp.logical_and(seg[2] > 0, mush_x > 140),
                        jnp.logical_and(seg[2] < 0, mush_x < 20)
                    )
                )
                idx = jnp.where(out_of_border, -1, self._env.get_mushroom_index(jnp.array([mush_x, mush_y])))
                return (
                    False,
                    jnp.zeros_like(seg),
                    jnp.where(seg[4] == 2, 100, 10),
                    jnp.array(1),
                    jnp.array(idx, dtype=jnp.int32)
                )

            return jax.lax.cond(collision, on_hit, lambda: (is_alive, seg, 0, jnp.array(0), jnp.array(-1)))

        check = jax.vmap(lambda s: check_single_segment(spell_active, s), in_axes=0)

        (
            spell_active,
            new_centipede_position,
            new_score,
            segment_hit,
            mush_idx
        ) = jax.lax.cond(spell_active != 0, lambda: check(centipede_position), no_hit)
        spell_active = jnp.invert(jnp.any(jnp.invert(spell_active)))

        new_score = jnp.sum(new_score)
        new_heads = jnp.roll(segment_hit, 1)
        mush_idx = jnp.max(mush_idx)
        new_mushroom_positions = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    mush_idx >= 0,
                    mush_idx < self._env.consts.MAX_MUSHROOMS
                ),
                mushroom_positions[mush_idx, 3] == 0
            ),
            mushroom_positions.at[mush_idx, 3].set(3),
            mushroom_positions
        )

        def speed_up(seg):
            return jnp.where(
                jnp.not_equal(seg[4], 0), # jnp.logical_and(seg[4] != 0, seg[4] != 2),
                seg.at[2].set(seg[2] * 1.25),
                seg
            )

        new_centipede_position = jnp.where(
            jnp.all(jnp.equal(centipede_position, new_centipede_position)),
            new_centipede_position,
            jax.vmap(speed_up)(new_centipede_position),
        )

        def set_new_status(seg, new):  # change value of hit following segment to head
            return jnp.where(jnp.logical_and(new == 1, seg[4] != 0), seg.at[4].set(2), seg)

        return (
            spell_state.at[2].set(jnp.where(spell_active, 1, 0)),
            jax.vmap(set_new_status)(new_centipede_position, new_heads),
            new_mushroom_positions,
            score + new_score
        )