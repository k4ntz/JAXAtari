import os
import time
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_tutankham import TutankhamState
from jaxatari.games.jax_tutankham import can_walk_to


# Shared by render + step mods (must stay in sync).
_TIME_CYCLE = 4000
_NIGHT_FRAMES = 250 # Duration of the night phase. Cant be greater than 499 because the first initialization of the night_timer with 3500 

class NightModeMod(JaxAtariInternalModPlugin):
    """
    Draws the night_cover while the night_timer is in the active phase, which means it is currently night time.
    """
    asset_overrides = {
        "night_cover": {
            "name": "night_cover",
            "type": "single",
            "file": "night_cover.npy",
        }
    }

    def _render_hook_night_mode(self, raster, state, camera_offset):
        jr = self._env.renderer.jr
        night_cover_mask = self._env.renderer.SHAPE_MASKS["night_cover"]
        day_time = _TIME_CYCLE - _NIGHT_FRAMES
        night = state.night_timer > day_time # True for the first _NIGHT_FRAMES frames of the cycle

        def _draw_night(r):
            return jr.render_at_clipped(
                r,
                0,
                36,
                night_cover_mask,
                flip_offset=self._env.consts.ZERO_FLIP,
            )

        return jax.lax.cond(night, _draw_night, lambda r: r, raster)


class NightModeStepMod(JaxAtariPostStepModPlugin):
    """
    Update step for the day/night cycle.
    Based on the trigger probability, which is increased by the day_time, duration. 
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        night_time_left = new_state.night_timer - 1

        day_time = _TIME_CYCLE - _NIGHT_FRAMES

        trigger_prob = jnp.clip((day_time - night_time_left) / jnp.float32(day_time), 0.0, 1.0)
        jax.debug.print("trigger_prob={}", trigger_prob)

        _, subkey = jax.random.split(jax.random.PRNGKey(int(time.time())))
        triggered = jax.random.bernoulli(subkey, p=trigger_prob) # Probability of triggering is higher the longer the day phase is.

        in_off_phase = night_time_left < day_time
        new_timer = jnp.where(
            (night_time_left <= 0) | (in_off_phase & triggered),
            jnp.int32(_TIME_CYCLE),
            night_time_left,
        )

        # Force night off on death or map transition.
        should_turn_off = (new_state.lives < prev_state.lives) | (new_state.level != prev_state.level)
        new_timer = jnp.where(should_turn_off, day_time, new_timer)

        return new_state.replace(night_timer=new_timer)


class MimicMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "mimic": {
            "name": "mimic",
            "type": "group",
            "files": ["mimic_01.npy", "mimic_00.npy"],
        }
    }
    def _render_hook_mimic(self, raster, state, camera_offset):
        jr = self._env.renderer.jr
        mimic_mask = self._env.renderer.SHAPE_MASKS["mimic"]
        frame_idx = (state.step_counter // self._env.consts.ANIMATION_SPEED) % 2
        mimic_x = state.mimic_state[0]
        mimic_y = state.mimic_state[1]
        mimic_active = state.mimic_state[2]
        mimic_direction = state.mimic_state[3]

        def _draw_mimic(r):
            return jr.render_at_clipped(
                r,
                mimic_x,
                mimic_y - camera_offset + frame_idx,
                mimic_mask[frame_idx],
                flip_offset=self._env.consts.ZERO_FLIP,
                flip_horizontal=(mimic_direction == 1)
            )

        return jax.lax.cond(mimic_active == 1, _draw_mimic, lambda r: r, raster)


class MimicStepMod(JaxAtariPostStepModPlugin):
    # Player has to get this close to the selected item.
    TRIGGER_RANGE_X: int = 30
    TRIGGER_RANGE_Y: int = 30
    INVALID_TARGET_IDX: int = -1
    MIMIC_SIZE: jnp.ndarray = jnp.array([8, 8], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _pick_target_idx(self, item_states, key):
        # Never allow the key (index 0) to be selected as mimic.
        active_mask = item_states[:, 3] == self._env.consts.ACTIVE
        candidate_mask = active_mask & (jnp.arange(item_states.shape[0]) != 0)
        has_active_items = jnp.any(candidate_mask)
        weights = candidate_mask.astype(jnp.float32)
        idx = jax.random.choice(key, item_states.shape[0], shape=(), p=weights)
        return jnp.where(has_active_items, idx.astype(jnp.int32), jnp.int32(self.INVALID_TARGET_IDX))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        # Choose exactly one candidate item at level start.
        rng, subkey = jax.random.split(state.rng_key)
        target_idx = self._pick_target_idx(state.item_states, subkey)
        mimic_state = jnp.array([0, 0, 0, 0, target_idx], dtype=jnp.int32)
        return obs, state.replace(mimic_state=mimic_state, rng_key=rng)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: TutankhamState, new_state: TutankhamState) -> TutankhamState:
        rng, subkey = jax.random.split(new_state.rng_key)

        mimic_state = new_state.mimic_state
        mimic_x = mimic_state[0]
        mimic_y = mimic_state[1]
        mimic_active = mimic_state[2]
        mimic_direction = mimic_state[3]
        mimic_target_idx = mimic_state[4]

        level_changed = new_state.level != prev_state.level
        life_lost = new_state.lives < prev_state.lives

        # One random target per level: only re-pick on level transition.
        repicked_target_idx = self._pick_target_idx(new_state.item_states, subkey)
        target_idx = jnp.where(level_changed, repicked_target_idx, mimic_target_idx)

        # If player "dies", deactivate mimics visual but keep target fixed for this level.
        mimic_active = jnp.where(life_lost | level_changed, 0, mimic_active)
        mimic_x = jnp.where(level_changed, 0, mimic_x)
        mimic_y = jnp.where(level_changed, 0, mimic_y)
        mimic_direction = jnp.where(level_changed, 0, mimic_direction)

        target_item = new_state.item_states[target_idx]
        target_x = target_item[0]
        target_y = target_item[1]
        target_is_active = (target_item[3] == self._env.consts.ACTIVE)

        in_range = (
            (jnp.abs(new_state.player_x - target_x) <= self.TRIGGER_RANGE_X) &
            (jnp.abs(new_state.player_y - target_y) <= self.TRIGGER_RANGE_Y)
        )
        trigger_now = (mimic_active == 0) & target_is_active & in_range

        new_item_states = jax.lax.cond(
            trigger_now,
            lambda items: items.at[target_idx, 3].set(self._env.consts.INACTIVE),
            lambda items: items,
            new_state.item_states,
        )

        mimic_state = jnp.array(
            [
                jnp.where(trigger_now, target_x, mimic_x),
                jnp.where(trigger_now, target_y, mimic_y),
                jnp.where(trigger_now, 1, mimic_active),
                mimic_direction,
                target_idx,
            ],
            dtype=jnp.int32,
        )

        return new_state.replace(item_states=new_item_states, mimic_state=mimic_state)

class GhostMod(JaxAtariInternalModPlugin):    
    asset_overrides = {
        "ghost": {
            "name": "ghost",
            "type": "group",
            "files": ["ghost_00.npy", "ghost_01.npy"],
        }
    }
    def _render_hook_ghost(self, raster, state, camera_offset):
        jr = self._env.renderer.jr
        ghost_mask = self._env.renderer.SHAPE_MASKS["ghost"]
        frame_idx = (state.step_counter // self._env.consts.ANIMATION_SPEED) % 2

        def _draw_ghost():
            return jr.render_at_clipped(
                raster,
                100, #TODO SET X
                100 - camera_offset, #TODO SET Y
                ghost_mask[frame_idx],
                flip_offset=self._env.consts.ZERO_FLIP,
                flip_horizontal= False # TODO Direction Flip
            )

        return _draw_ghost()


class UpsideDownMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "MAP_CHECKPOINTS": jnp.array([
            # MAP 1
            [
                [588, 800, 18, 684],
                [405, 587, 80, 586],
                [201, 404, 12, 403],
                [0,   200, 78, 199],
            ],
            # MAP 2
            [
                [573, 800, 19, 634],
                [425, 572, 24, 572],
                [261, 426, 78, 426],
                [0,   260, 78, 259],
            ],
            # MAP 3
            [
                [553, 800,  107, 715],
                [401, 552,  98,  550],
                [269, 400,  78,  396],
                [0,   268,  39,  248],
            ],
            # MAP 4
            [
                [531, 800, 77,  719],
                [391, 532, 119, 531],
                [204, 392, 18,  391],
                [0 ,  203, 30,  203],
            ],
        ], dtype=jnp.int32),

        "MAP_GOAL_POSITIONS": jnp.array([
            [[134, 61]],  # MAP 1
            [[136, 60]],  # MAP 2
            [[16,  93]],  # MAP 3
            [[82,  95]]   # MAP 4
        ], dtype=jnp.int32)
    }


class MovingItemsMod(JaxAtariInternalModPlugin):
    
    # add item direction to the initial item states (to calculate movement)
    constants_overrides = {
        "MAP_ITEMS": jnp.array([
            # Level 1 (MAP 1)
            [
                [51, 87, 0, 1, 0],   # KEY_MAP1=0       [x, y, item_type, active, direction]
                [99, 183, 5, 1, 0],  # CROWN_02_MAP1=5
                [68, 262, 2, 1, 0],  # RING_MAP1=2
                [7, 311, 3, 1, 0],   # RUBY_MAP1=3
                [93, 382, 4, 1, 0],  # CHALICE_MAP1=4
                [18, 494, 1, 1, 0],  # CROWN_01_MAP1=1
                [0, 0, 0, 0, 0],     # Padding
            ],
            # Level 2 (MAP 2)
            [
                [21, 272, 6, 1, 0],  # KEY_MAP2=6
                [44, 155, 8, 1, 0],  # CROWN_MAP2=8
                [128, 98, 7, 1, 0],  # RING_MAP2=7
                [37, 406, 9, 1, 0],  # EMERALD_MAP2=9
                [91, 482, 10, 1, 0], # GOBLET_MAP2=10
                [23, 547, 11, 1, 0], # BUST_MAP2=11
                [0, 0, 0, 0, 0],     # Padding
            ],
            # Level 3 (MAP 3)
            [
                [22, 411, 12, 1, 0], # KEY_MAP3=12
                [15, 173, 14, 1, 0], # RING_MAP3=14
                [128, 98, 13, 1, 0], # TRIDENT_MAP3=13
                [17, 278, 15, 1, 0], # HERB_MAP3=15
                [108, 323, 16, 1, 0],# DIAMOND_MAP3=16
                [27, 656, 17, 1, 0], # CANDELABRA_MAP3=17
                [0, 0, 0, 0, 0],     # Padding
            ],
            # Level 4 (MAP 4)
            [
                [144, 110, 18, 1, 0], # KEY_MAP4=18
                [125, 221, 19, 1, 0], # RING_MAP4=19
                [117, 269, 20, 1, 0], # AMULET_MAP4=20
                [19, 326, 21, 1, 0],  # FAN_MAP4=21
                [55, 510, 23, 1, 0],  # ZIRCON_MAP4=23
                [110, 401, 22, 1, 0], # CRYSTAL_MAP4=22
                [66, 607, 24, 1, 0],  # DAGGER_MAP4=24
            ],
        ], dtype=jnp.int32),
    }

    @partial(jax.jit, static_argnums=(0,))
    def item_step(self, item_states, level, rng_key):
        """Moves active items using random-walk pathing"""

        def move_item(item, rng_key):
            item_x, item_y, item_type, active, direction = item
            
            # Natural patrol: randomly change direction --------------------------------
            change_probability = 0.08
            possible_directions = jnp.array([-1, -2, 1, 2]) # right, left, down, up

            rng_key, subkey_01, subkey_02 = jax.random.split(rng_key, 3)
            random_dir = jax.random.choice(subkey_01, possible_directions)
            should_change = jax.random.bernoulli(subkey_02, p=change_probability)

            new_direction = jnp.where(should_change, random_dir, direction)
            new_direction = jnp.where(direction == 0, random_dir, new_direction)

            # Indices: 0, 1(Down), 2(Up), -1(Right), -2(Left)
            # mapping array where the index matches the direction value
            lookup_x = jnp.array([0, 0,  0, -1, 1])
            lookup_y = jnp.array([0, 1, -1, 0,  0])
            x_direction = lookup_x[new_direction]
            y_direction = lookup_y[new_direction]

            # move creature
            new_x = item_x +  x_direction * active
            new_y = item_y +  y_direction * active           
            item_x, item_y, is_walkable = can_walk_to(self._env.consts.ITEM_SIZES[item_type], new_x, new_y, item_x, item_y, self._env.consts.VALID_POS_MAPS[level%4])
    
            #--------------------------------------------------------------------------

            return jnp.array([item_x, item_y, item_type, active, new_direction], dtype=jnp.int32), rng_key

        keys = jax.random.split(rng_key, len(item_states) + 1)
        rng_key = keys[0]
        new_item_states, _ = jax.vmap(move_item)(item_states, keys[1:])

        
        return new_item_states, rng_key

    