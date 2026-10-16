import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple
import os
import sys

from jaxtari.environment import JaxEnvironment, JaxtariAction as Action, ObjectObservation
import jaxtari.spaces as spaces
from jaxtari.rendering import jax_rendering_utils as render_utils

from jaxtari.games.montezuma_revenge.core import (
    enemy_bounce_offsets,
    rope_widths,
    MontezumaRevengeConstants,
    MontezumaRevengeState,
    MontezumaRevengeObservation,
    MontezumaRevengeInfo,
    get_room_idx,
    check_platform,
)
from jaxtari.games.montezuma_revenge.renderer import MontezumaRevengeRenderer
from jaxtari.games.montezuma_revenge.rooms import load_room, initialize_level, LEVEL_CONTENTS, ALE_ROOM_IDS, NATIVE_EXIT_MASKS, NATIVE_ROOM_HISTORY
from jaxtari.games.mods.montezuma_revenge.custom_rooms import load_custom_room


class JaxMontezumaRevenge(JaxEnvironment[MontezumaRevengeState, MontezumaRevengeObservation, MontezumaRevengeInfo, MontezumaRevengeConstants]):
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
    ], dtype=jnp.int32)

    def __init__(self, consts: MontezumaRevengeConstants = None):
        consts = consts or MontezumaRevengeConstants()
        super().__init__(consts)
        self.renderer = MontezumaRevengeRenderer(self.consts)
        transparent = self.renderer.jr.TRANSPARENT_ID
        self._key_collision_mask = self.renderer.SHAPE_MASKS['key'] != transparent
        item_shapes = [self.renderer.SHAPE_MASKS[name] for name in ('key', 'gem', 'amulet', 'sword')]
        item_shapes.append(self.renderer.SHAPE_MASKS['torch'][0])
        self._item_collision_masks = jnp.stack([
            jnp.pad(mask != transparent, ((0, 20-mask.shape[0]), (0, 8-mask.shape[1])))
            for mask in item_shapes])
        self._key_player_stand_mask = self.renderer.SHAPE_MASKS['player'][0] != transparent
        self._key_player_jump_mask = self.renderer.SHAPE_MASKS['player'][7] != transparent
        self._player_collision_masks = self.renderer.SHAPE_MASKS['player'] != transparent
        self._original_walk_1_collision_mask = self.renderer.ORIGINAL_WALK_1 != transparent
        self._spider_collision_mask = self.renderer.SHAPE_MASKS['spider'] != transparent
        self._snake_collision_masks = self.renderer.SHAPE_MASKS['snake'] != transparent
        self._skull_collision_masks = self.renderer.SHAPE_MASKS['skull'] != transparent
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "montezuma")
        
        sprite_path_0 = os.path.join(sprite_path, "backgrounds", "base_collision_map.npy")
        col_map_0 = jnp.load(sprite_path_0)[:149, :, 0]
        
        # New 3: Leftmost
        room_col_0_3 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_0_3 = room_col_0_3.at[6:48, 0:4].set(1) # Left wall
        room_col_0_3 = room_col_0_3.at[147:149, 72:88].set(0) # Hole for ladder down

        sprite_path_1 = os.path.join(sprite_path, "backgrounds", "mid_room_collision_level_0.npy")
        col_map_1 = jnp.load(sprite_path_1)[:149, :, 0] # (149, 160)
        # New 4: Middle
        room_col_0_4 = jnp.where(col_map_1 > 0, 1, 0).astype(jnp.int32)
        room_col_0_4 = room_col_0_4.at[147:149, 72:88].set(0) # Hole for ladder down
        # No side walls for room_0_4

        # New 5: Rightmost
        room_col_0_5 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_0_5 = room_col_0_5.at[6:48, 156:160].set(1) # Right wall
        room_col_0_5 = room_col_0_5.at[147:149, 72:88].set(0) # Hole for ladder down
        
        room_col_1_3 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_1_3 = room_col_1_3.at[147:149, 72:88].set(0) # Hole for ladder down
        room_col_1_2 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_1_2 = room_col_1_2.at[6:48, 0:4].set(1)
        room_col_1_2 = room_col_1_2.at[147:149, 72:88].set(0) # Hole for ladder down to room 18
        
        sprite_path_2 = os.path.join(sprite_path, "backgrounds", "mid_room_collision_level_1.npy")
        col_map_2 = jnp.load(sprite_path_2)[:149, :, 0]
        room_col_1_4 = jnp.where(col_map_2 > 0, 1, 0).astype(jnp.int32)
        room_col_1_4 = room_col_1_4.at[147:149, 72:88].set(0) # Hole for ladder
        room_col_1_4 = room_col_1_4.at[6:46, 124:126].set(0) # Fix pillar 2 and rope collision (Right)
        
        room_col_1_5 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_1_5 = room_col_1_5.at[147:149, 72:88].set(0) # Hole for ladder down
        # room_1_5 is no longer the rightmost room on level 1
        
        # New 14: Rightmost on level 1 (corresponds to ROOM_1_4 in M1)
        room_col_1_6 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_1_6 = room_col_1_6.at[6:48, 156:160].set(1) # Right wall
        room_col_1_6 = room_col_1_6.at[147:149, 72:88].set(0) # Hole for ladder down to room 22
        
        # New 18: Level 2, col 2 (corresponds to ROOM_2_1 in M1)
        room_col_2_2 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_2_2 = room_col_2_2.at[6:48, 156:160].set(1) # Right wall
        
        sprite_path_3 = os.path.join(sprite_path, "backgrounds", "room_0_collision_level_2.npy")
        col_map_3 = jnp.load(sprite_path_3)[:149, :, 0]
        room_col_2_1 = jnp.where(col_map_3 > 0, 1, 0).astype(jnp.int32)
        room_col_2_1 = room_col_2_1.at[6:, 0:4].set(1) # Left wall

        sprite_path_4 = os.path.join(sprite_path, "backgrounds", "pitroom_collision_map.npy")
        col_map_4 = jnp.load(sprite_path_4)[:149, :, 0]
        room_col_2_3 = jnp.where(col_map_4 > 0, 1, 0).astype(jnp.int32)
        room_col_2_3 = room_col_2_3.at[6:48, 0:4].set(1) # Left wall
        # Measured ALE pit ledges meet the dropout floor at x36/124, y46.
        room_col_2_3 = room_col_2_3.at[46:52, :36].set(1)
        room_col_2_3 = room_col_2_3.at[46:52, 124:].set(1)
        room_col_2_3 = room_col_2_3.at[46:52, 36:124].set(0)

        # New 20: Level 2, col 4 (corresponds to ROOM_2_3 in M1)
        room_col_2_4 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_2_4 = room_col_2_4.at[147:149, 72:88].set(0) # Hole for ladder down

        # New 21: Level 2, col 5 (corresponds to ROOM_2_4 in M1)
        room_col_2_5 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        
        # New 22: Level 2, col 6 (corresponds to ROOM_2_5 in M1)
        room_col_2_6 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_2_6 = room_col_2_6.at[147:149, 72:88].set(0) # Hole for ladder down

        # New 23: Level 2, col 7 (corresponds to ROOM_2_6 in M1)
        sprite_path_5 = os.path.join(sprite_path, "backgrounds", "room_6_collision_level_2.npy")
        col_map_5 = jnp.load(sprite_path_5)[:147, :, 0]
        room_col_2_7 = jnp.zeros((149, 160), dtype=jnp.int32)
        room_col_2_7 = room_col_2_7.at[:147, :].set(jnp.where(col_map_5 > 0, 1, 0))
        room_col_2_7 = room_col_2_7.at[6:, 156:160].set(1) # Right wall
        room_col_2_7 = room_col_2_7.at[147:149, 72:88].set(0) # Hole for ladder down

        # New 31: Level 3, col 7 (corresponds to ROOM_3_7 in M1)
        # Using pitroom_collision_map.npy as specified for ROOM_3_7
        sprite_path_6 = os.path.join(sprite_path, "backgrounds", "pitroom_collision_map.npy")
        col_map_6 = jnp.load(sprite_path_6)[:149, :, 0]
        room_col_3_7 = jnp.where(col_map_6 > 0, 1, 0).astype(jnp.int32)
        # No side walls as requested

        # New 32: Level 3, col 8 (corresponds to ROOM_3_8 in M1)
        room_col_3_8 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_3_8 = room_col_3_8.at[6:48, 156:160].set(1) # Right wall

        # New 30: Level 3, col 6 (corresponds to ROOM_3_6 in M1)
        room_col_3_6 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_3_6 = room_col_3_6.at[6:48, 0:4].set(1) # Left wall

        # New 28: Level 3, col 4 (corresponds to ROOM_3_4 in M1)
        room_col_3_4 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        # No side walls

        # New 27: Level 3, col 3 (corresponds to ROOM_3_3 in M1)
        # Using pitroom_collision_map.npy as specified for ROOM_3_3
        sprite_path_7 = os.path.join(sprite_path, "backgrounds", "pitroom_collision_map.npy")
        col_map_7 = jnp.load(sprite_path_7)[:149, :, 0]
        room_col_3_3 = jnp.where(col_map_7 > 0, 1, 0).astype(jnp.int32)
        # No left wall (open to ROOM_3_2)

        # New 29: Level 3, col 5 (corresponds to ROOM_3_5 in M1)
        # Using pitroom_collision_map.npy as specified for ROOM_3_5
        sprite_path_8 = os.path.join(sprite_path, "backgrounds", "pitroom_collision_map.npy")
        col_map_8 = jnp.load(sprite_path_8)[:149, :, 0]
        room_col_3_5 = jnp.where(col_map_8 > 0, 1, 0).astype(jnp.int32)
        room_col_3_5 = room_col_3_5.at[6:48, 156:160].set(1) # Right wall

        # New 25: Level 3, col 1 (corresponds to ROOM_3_1 in M1)
        room_col_3_1 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)

        # New 26: Level 3, col 2 (corresponds to ROOM_3_2 in M1)
        room_col_3_2 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)

        # New 24: Bonus Room (corresponds to ROOM_3_0 in M1)
        sprite_path_9 = os.path.join(sprite_path, "backgrounds", "bonus_room_collision_map.npy")
        col_map_9 = jnp.load(sprite_path_9)[:149, :, 0]
        room_col_3_0 = jnp.zeros((149, 160), dtype=jnp.int32)
        room_col_3_0 = room_col_3_0.at[:col_map_9.shape[0], :].set(jnp.where(col_map_9 > 0, 1, 0))
        room_col_3_0 = room_col_3_0.at[6:148, 0:4].set(1) # Left wall
        room_col_3_0 = room_col_3_0.at[6:148, 156:160].set(1) # Right wall
        room_col_3_0 = room_col_3_0.at[47:50, :].set(1) # Thin invisible horizontal platform at Y=47

        room_col_3_7 = room_col_3_7.at[46:52, :36].set(1)
        room_col_3_7 = room_col_3_7.at[46:52, 124:].set(1)
        room_col_3_7 = room_col_3_7.at[46:52, 36:124].set(0)
        room_col_3_3 = room_col_3_3.at[46:52, :36].set(1)
        room_col_3_3 = room_col_3_3.at[46:52, 124:].set(1)
        room_col_3_3 = room_col_3_3.at[46:52, 36:124].set(0)
        room_col_3_5 = room_col_3_5.at[46:52, :36].set(1)
        room_col_3_5 = room_col_3_5.at[46:52, 124:].set(1)
        room_col_3_5 = room_col_3_5.at[46:52, 36:124].set(0)
        self.ROOM_COLLISION_MAPS = jnp.stack([room_col_0_3, room_col_0_4, room_col_0_5, room_col_1_3, room_col_1_2, room_col_1_4, room_col_1_5, room_col_1_6, room_col_2_2, room_col_2_1, room_col_2_3, room_col_2_4, room_col_2_5, room_col_2_6, room_col_2_7, room_col_3_7, room_col_3_8, room_col_3_6, room_col_3_4, room_col_3_3, room_col_3_5, room_col_3_1, room_col_3_2, room_col_3_0])
        # ALE top-floor contact is one pixel higher in these measured rooms.
        # Keep the shared asset files unchanged.
        for room_id in (3, 5, 10, 11, 13, 14, 18, 20, 21, 22, 24, 25, 26, 28, 30, 32):
            idx = get_room_idx(jnp.int32(room_id))
            self.ROOM_COLLISION_MAPS = self.ROOM_COLLISION_MAPS.at[idx, 46, 4:156].set(1)



    def reset(self, key: jrandom.PRNGKey) -> Tuple[MontezumaRevengeObservation, MontezumaRevengeState]:
        state = MontezumaRevengeState(
            room_id=jnp.array(self.consts.INITIAL_ROOM_ID, dtype=jnp.int32),
            lives=jnp.array(5, dtype=jnp.int32),
            score=jnp.array([0], dtype=jnp.int32),
            frame_count=jnp.array(0, dtype=jnp.int32),
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.INITIAL_PLAYER_Y, dtype=jnp.int32),
            player_vx=jnp.array(0, dtype=jnp.int32),
            player_vy=jnp.array(0, dtype=jnp.int32),
            player_dir=jnp.array(-1, dtype=jnp.int32),
            entry_x=jnp.array(self.consts.INITIAL_PLAYER_X, dtype=jnp.int32),
            entry_y=jnp.array(self.consts.INITIAL_PLAYER_Y, dtype=jnp.int32),
            entry_is_climbing=jnp.array(0, dtype=jnp.int32),
            entry_last_ladder=jnp.array(-1, dtype=jnp.int32),
            is_jumping=jnp.array(0, dtype=jnp.int32),
            is_falling=jnp.array(0, dtype=jnp.int32),
            fall_after_jump=jnp.array(0, dtype=jnp.int32),
            fall_distance=jnp.array(0, dtype=jnp.int32),
            jump_counter=jnp.array(0, dtype=jnp.int32),
            is_climbing=jnp.array(0, dtype=jnp.int32),
            out_of_ladder_delay=jnp.array(0, dtype=jnp.int32),
            last_rope=jnp.array(-1, dtype=jnp.int32),
            last_ladder=jnp.array(-1, dtype=jnp.int32),
            prev_is_fire=jnp.array(False),
            enemies_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_y=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_direction=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_type=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_min_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_max_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_bouncing=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            ladders_x=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_top=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_bottom=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_active=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ropes_x=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_top=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_bottom=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_active=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            items_x=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_y=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_active=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_type=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            doors_x=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            doors_y=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            doors_active=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            conveyors_x=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_y=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_active=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_direction=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            lasers_x=jnp.zeros(self.consts.MAX_LASERS_PER_ROOM, dtype=jnp.int32),
            lasers_active=jnp.zeros(self.consts.MAX_LASERS_PER_ROOM, dtype=jnp.int32),
            laser_cycle=jnp.array(31, dtype=jnp.int32),
            platforms_x=jnp.zeros(self.consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32),
            platforms_y=jnp.zeros(self.consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32),
            platforms_width=jnp.full(self.consts.MAX_PLATFORMS_PER_ROOM, 12, dtype=jnp.int32),
            platforms_active=jnp.zeros(self.consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32),
            platform_cycle=jnp.array(31, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            death_type=jnp.array(0, dtype=jnp.int32),
            inventory=jnp.array([0, 0, 0, 0], dtype=jnp.int32), # keys, sword, torch, amulet
            amulet_time=jnp.array(0, dtype=jnp.int32),
            bonus_room_timer=jnp.array(0, dtype=jnp.int32),
            first_gem_pickup=jnp.array(0, dtype=jnp.int32),
            global_enemies_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ENEMIES_PER_ROOM), dtype=jnp.int32),
            global_enemies_type=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ENEMIES_PER_ROOM), dtype=jnp.int32),
            global_items_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ITEMS_PER_ROOM), dtype=jnp.int32),
            global_items_type=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ITEMS_PER_ROOM), dtype=jnp.int32),
            global_doors_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_DOORS_PER_ROOM), dtype=jnp.int32),
            key=key
        )
        
        gia = state.global_items_active
        gia = gia.at[3, 0].set(1) # New 3 (Left)
        gia = gia.at[4, 0].set(1) # New 4 (Mid)
        gia = gia.at[12, 0].set(1)
        gia = gia.at[13, 0].set(1)
        gia = gia.at[14, 0].set(1) # Key in Room 14
        gia = gia.at[17, 0].set(1) # Key in Room 17
        gia = gia.at[19, 0].set(1) # Gem in Room 19
        gia = gia.at[23, 0].set(1) # Key in Room 23
        gia = gia.at[32, 0].set(1) # 3 Gems in Room 32
        gia = gia.at[32, 1].set(1)
        gia = gia.at[32, 2].set(1)
        gia = gia.at[28, 0].set(1) # Amulet in Room 28
        gia = gia.at[29, 0].set(1) # Gem in Room 29
        gia = gia.at[29, 1].set(1) # Gem in Room 29
        gia = gia.at[24, 0].set(1) # Gem in Room 24 (Bonus Room)

        gda = state.global_doors_active
        gda = gda.at[4, 0].set(1) # New 4 (Mid)
        gda = gda.at[4, 1].set(1)
        gda = gda.at[12, 0].set(1)
        gda = gda.at[12, 1].set(1)
        gda = gda.at[26, 0].set(1) # Room 26 (ROOM_3_2)
        gda = gda.at[26, 1].set(1)
        
        gea = state.global_enemies_active
        gea = gea.at[4, 0].set(1) # New 4 (Mid)
        gea = gea.at[5, 0].set(1) # New 5 (Right)
        gea = gea.at[5, 1].set(1)
        gea = gea.at[11, 0].set(1)
        gea = gea.at[10, 0].set(1)
        gea = gea.at[10, 1].set(1)
        gea = gea.at[12, 0].set(1)
        gea = gea.at[18, 0].set(1)
        gea = gea.at[18, 1].set(1)
        gea = gea.at[20, 0].set(1)
        gea = gea.at[20, 1].set(1)
        gea = gea.at[22, 0].set(1)
        gea = gea.at[31, 0].set(1) # Snake in Room 31
        gea = gea.at[30, 0].set(1) # Spider in Room 30
        gea = gea.at[27, 0].set(1) # Skull in Room 27 (ROOM_3_3)

        gety = state.global_enemies_type
        # ROLL_SKULL = 1, BOUNCE_SKULL = 2, SPIDER = 3, SNAKE = 4
        gety = gety.at[4, 0].set(1)
        gety = gety.at[5, 0].set(1)
        gety = gety.at[5, 1].set(1)
        gety = gety.at[11, 0].set(3)
        gety = gety.at[10, 0].set(1)
        gety = gety.at[10, 1].set(1)
        gety = gety.at[12, 0].set(1)
        # Assuming snakes for ROOM 18 as requested
        gety = gety.at[18, 0].set(4)
        gety = gety.at[18, 1].set(4)
        # Snakes for ROOM 20
        gety = gety.at[20, 0].set(4)
        gety = gety.at[20, 1].set(4)
        gety = gety.at[22, 0].set(3)
        gety = gety.at[31, 0].set(4) # Snake in Room 31
        gety = gety.at[30, 0].set(3) # Spider in Room 30
        gety = gety.at[27, 0].set(1) # Skull in Room 27 (ROOM_3_3)

        giy = state.global_items_type
        giy = giy.at[3, 0].set(1) # Gem in room 3
        # Torch in room 12
        giy = giy.at[12, 0].set(4)
        # Sword in room 13
        giy = giy.at[13, 0].set(3)
        # Gem in room 19
        giy = giy.at[19, 0].set(1)
        # Gems in room 32
        giy = giy.at[32, 0].set(1)
        giy = giy.at[32, 1].set(1)
        giy = giy.at[32, 2].set(1)
        giy = giy.at[28, 0].set(2) # Amulet in room 28
        giy = giy.at[29, 0].set(1) # Gem in room 29
        giy = giy.at[29, 1].set(1) # Gem in room 29
        giy = giy.at[24, 0].set(1) # Gem in room 24 (Bonus Room)
        
        state = state.replace(global_items_active=gia, global_doors_active=gda, global_enemies_active=gea,
            global_enemies_type=gety, global_items_type=giy, native_room_history=NATIVE_ROOM_HISTORY[0])
        
        if self.consts.CUSTOM_ROOMS:
            state = load_custom_room(jnp.array(self.consts.INITIAL_ROOM_ID, dtype=jnp.int32), state, self.consts)
        else:
            state = load_room(jnp.array(self.consts.INITIAL_ROOM_ID, dtype=jnp.int32), state, self.consts)
        obs = self._get_observation(state)
        return obs, state
    
    def _begin_next_level(self, state):
        # Mode1 initializes the next temple on the frame after the floor exit.
        # Preserve the free-running clocks/RNG and cumulative score and lives.
        _, fresh = self.reset(state.key)
        # At the hardest layout the ROM restarts it without another bonus.
        bonus = jnp.where(state.level < 2, 10000, 0)
        fresh = fresh.replace(level=jnp.minimum(state.level + 1, 2),
            score=state.score + bonus, lives=state.lives + (bonus > 0).astype(jnp.int32),
            frame_count=state.frame_count + 1, native_frame_count=state.native_frame_count,
            native_rng_a=state.native_rng_a, native_rng_b=state.native_rng_b,
            native_rng_carry=jnp.int32(0), rolling_skull_direction=jnp.int32(1),
            rolling_skull_phase=state.rolling_skull_phase,
            native_rolling_direction=state.native_rolling_direction,
            laser_cycle=state.laser_cycle, platform_cycle=state.platform_cycle)
        fresh = load_room(jnp.int32(4), initialize_level(fresh, self.consts), self.consts)
        return (self._get_observation(fresh), fresh, self._get_reward(state, fresh),
                self._get_done(fresh), self._get_info(fresh))

    def step(self, state: MontezumaRevengeState, action: int) -> Tuple[MontezumaRevengeObservation, MontezumaRevengeState, float, bool, MontezumaRevengeInfo]:
        # Native counters advance before dispatch, including frozen poses.
        native_cycle = jnp.mod(state.native_frame_count + 1 + 31, 128)
        platform_cycle = jnp.where(native_cycle == 0, 0,
            jnp.where(native_cycle == 92, 92, state.platform_cycle))
        native_beams = jnp.where(native_cycle == 0, (state.lasers_x > 0).astype(jnp.int32),
            jnp.where(native_cycle == 92, 0, state.lasers_active))
        state = state.replace(
            lasers_active=jnp.where(jnp.logical_and(state.level == 2, not self.consts.CUSTOM_ROOMS),
                native_beams, state.lasers_active),
            laser_cycle=jnp.where(self.consts.CUSTOM_ROOMS, state.laser_cycle, native_cycle),
            platform_cycle=jnp.where(self.consts.CUSTOM_ROOMS, state.platform_cycle, platform_cycle))
        amulet_time = jnp.maximum(state.amulet_time - 1, 0)
        bonus_time = jnp.where(state.room_id == 24,
            jnp.where(state.transition_pending > 0,
                jnp.mod(state.native_frame_count + 1 + 67, 8),
                state.bonus_room_timer + 1), 0)
        state = state.replace(amulet_time=amulet_time,
            inventory=state.inventory.at[3].set(jnp.where(amulet_time > 0, 1, 0)),
            bonus_room_timer=bonus_time)
        def hold_pickup():
            held = state.replace(pickup_timer=state.pickup_timer - 1,
                                 native_mode=jnp.where(state.pickup_timer > 2, 6, jnp.where(state.pickup_timer == 2, 3, 4)),
                                 frame_count=state.frame_count + 1)
            return (self._get_observation(held), held,
                    self._get_reward(state, held), self._get_done(held), self._get_info(held))

        def relocate():
            held = state.replace(player_x=state.transition_x, player_y=state.transition_y,
                native_mode=jnp.int32(4),
                transition_pending=jnp.int32(0), room5_skull_spawn_pending=jnp.bool_(False),
                frame_count=state.frame_count + 1,
                is_jumping=jnp.where(state.transition_pending == 1, 0, state.is_jumping),
                is_falling=jnp.where(state.transition_pending == 1, 0, state.is_falling),
                fall_after_jump=jnp.where(state.transition_pending == 1, 0, state.fall_after_jump),
                native_jump_momentum=jnp.where(state.transition_pending == 1, False, state.native_jump_momentum),
                player_vx=jnp.where(state.transition_pending == 1, 0, state.player_vx),
                jump_counter=jnp.where(state.transition_pending == 1, 0, state.jump_counter),
                enemies_y=jnp.where(jnp.logical_and(state.room_id == 12, state.enemies_type == 1), 69, state.enemies_y),
                laser_cycle=jnp.mod(state.laser_cycle + 1, 128))
            return (self._get_observation(held), held,
                    self._get_reward(state, held), self._get_done(held), self._get_info(held))

        def floor_exit():
            held = state.replace(native_mode=jnp.int32(1), native_rng_carry=jnp.int32(0),
                                 frame_count=state.frame_count + 1)
            return (self._get_observation(held), held, jnp.float32(0),
                    self._get_done(held), self._get_info(held))

        obs, next_state, reward, done, info = jax.lax.cond(
            jnp.logical_and(state.native_mode == 1, not self.consts.CUSTOM_ROOMS), lambda: self._begin_next_level(state),
            lambda: jax.lax.cond(jnp.logical_and(not self.consts.CUSTOM_ROOMS,
                jnp.logical_and(state.room_id == 24, state.player_y >= 126)), floor_exit,
                lambda: jax.lax.cond(state.transition_pending > 0, relocate,
                    lambda: jax.lax.cond(state.pickup_timer > 0, hold_pickup,
                                        lambda: self._step_active(state, action)))))
        # ALE's RAM0 clock continues through pickup, relocation and death.
        # Retained runtime initializes RAM0 at67; our visible-first convention
        # is the same128-tick phase shifted by92: (native tick +31) %128.
        native_cycle = jnp.mod(state.native_frame_count + 1 + 31, 128)
        # The ROM installs a visible floor on room entry. Its pointer then
        # changes only at the two clock edges, including during death holds.
        platform_cycle = jnp.where(native_cycle == 0, 0,
            jnp.where(native_cycle == 92, 92, state.platform_cycle))
        platform_cycle = jnp.where(next_state.room_id != state.room_id, 0, platform_cycle)
        platform_cycle = jnp.where(jnp.logical_and(state.level == 2, state.native_mode == 2), 0, platform_cycle)
        next_state = next_state.replace(native_frame_count=state.native_frame_count + 1,
            laser_cycle=jnp.where(self.consts.CUSTOM_ROOMS, next_state.laser_cycle, native_cycle),
            platform_cycle=jnp.where(self.consts.CUSTOM_ROOMS, next_state.platform_cycle, platform_cycle))
        # Metadata cleanup is the last executed ROM branch in modes2/3.
        ale_room = state.room_id - jnp.where(state.room_id <= 5, 3,
            jnp.where(state.room_id <= 14, 7, 9))
        item_room = jnp.logical_or(jnp.isin(ale_room, jnp.array([2,3])),
            LEVEL_CONTENTS[jnp.minimum(state.level, 2), ale_room, 3] > 0)
        native_type = LEVEL_CONTENTS[jnp.minimum(state.level, 2), ale_room, 0]
        packed_mask = jnp.sum(state.enemies_active * (1 << jnp.arange(self.consts.MAX_ENEMIES_PER_ROOM)))
        packed_mask = jnp.where(state.level == 2,
            jnp.sum(next_state.enemies_active * (1 << jnp.arange(self.consts.MAX_ENEMIES_PER_ROOM))), packed_mask)
        metadata_carry = jnp.where(item_room,
            jnp.where(native_type == 5, jnp.where(packed_mask == 1, 1, state.enemies_x[0] + 17 > 255), native_type >= 5),
            ale_room >= 2).astype(jnp.int32)
        carry = jnp.where(jnp.logical_or(state.native_mode == 2, state.native_mode == 3),
            metadata_carry, jnp.where(state.native_mode == 6, 0, next_state.native_rng_carry))
        a, b = state.native_rng_a, state.native_rng_b
        next_state = next_state.replace(
            native_rng_a=jnp.bitwise_xor(jnp.mod(2*a + carry + 195 + (b >> 7), 256), a),
            native_rng_b=jnp.bitwise_xor(b, jnp.bitwise_and(2*b | (a >> 7), 255)),
            native_rng_carry=carry)
        walking_pose = jnp.logical_and(jnp.where(state.level == 2, state.native_walk_pose, state.player_vx != 0),
            jnp.logical_and(state.is_climbing == 0, jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)))
        airborne_pose = jnp.logical_or(jnp.logical_and(state.is_jumping == 1, state.jump_counter > 0), state.is_falling == 1)
        pose = jnp.where(walking_pose, 1 + ((state.native_frame_count + 68) // 4) % 2, 0)
        pose = jnp.where(state.is_climbing == 1, 3 + (state.player_y // 4) % 2, pose)
        native_climb_pose = jnp.where(state.last_rope >= 0, 5, 3) + jnp.where(
            state.native_climb_moving, ((state.native_frame_count + 68) // 4) % 2, 0)
        pose = jnp.where(jnp.logical_and(state.level == 2, state.is_climbing == 1),
            native_climb_pose, pose)
        pose = jnp.where(airborne_pose, 7, pose)
        next_state = next_state.replace(collision_player_x=state.player_x, collision_player_y=state.player_y,
            collision_player_pose=pose, collision_player_dir=state.player_dir)
        # Room entry clears the TIA collision latch before play resumes.
        clear_contact = jnp.logical_and(state.level == 2,
            jnp.logical_and(state.death_timer == 1, next_state.death_timer == 0))
        next_state = next_state.replace(
            collision_player_x=jnp.where(clear_contact, next_state.player_x, next_state.collision_player_x),
            collision_player_y=jnp.where(clear_contact, next_state.player_y, next_state.collision_player_y),
            collision_player_pose=jnp.where(clear_contact, 0, next_state.collision_player_pose),
            collision_player_dir=jnp.where(clear_contact, next_state.player_dir, next_state.collision_player_dir))
        # Dispatch observations precede the final clock and room-entry updates.
        # Expose the same hazard state that callers receive and render.
        return self._get_observation(next_state), next_state, reward, done, info

    def _step_active(self, state: MontezumaRevengeState, action: int) -> Tuple[MontezumaRevengeObservation, MontezumaRevengeState, float, bool, MontezumaRevengeInfo]:
        previous_x, previous_y = state.player_x, state.player_y
        is_active = state.death_timer == 0
        room_idx = get_room_idx(state.room_id)
        room_col_map = self.ROOM_COLLISION_MAPS[room_idx]
        
        is_bonus_room = state.room_id == 24
        bonus_collapse = jnp.logical_and(is_bonus_room, state.bonus_room_timer == 632)
        floor_dropped = jnp.logical_and(is_bonus_room, state.bonus_room_timer >= 632)
        room_col_map = jax.lax.select(floor_dropped, room_col_map.at[46:50, 4:156].set(0), room_col_map)
        
        platform_active_now = jnp.less(state.platform_cycle, self.consts.PLATFORM_ACTIVE_DURATION)
        previous_score = state.score

        # Amulet counter
        new_amulet_time = state.amulet_time
        is_amulet_active = new_amulet_time > 0

        is_up = jnp.logical_or(action == Action.UP, jnp.logical_or(action == Action.UPRIGHT, action == Action.UPLEFT))
        is_up = jnp.logical_or(is_up, jnp.logical_or(action == Action.UPFIRE, jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE)))
        is_down = jnp.logical_or(action == Action.DOWN, jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNLEFT))
        is_down = jnp.logical_or(is_down, jnp.logical_or(action == Action.DOWNFIRE, jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE)))
        is_right = jnp.logical_or(action == Action.RIGHT, jnp.logical_or(action == Action.UPRIGHT, action == Action.DOWNRIGHT))
        is_right = jnp.logical_or(is_right, jnp.logical_or(action == Action.RIGHTFIRE, jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE)))
        is_left = jnp.logical_or(action == Action.LEFT, jnp.logical_or(action == Action.UPLEFT, action == Action.DOWNLEFT))
        is_left = jnp.logical_or(is_left, jnp.logical_or(action == Action.LEFTFIRE, jnp.logical_or(action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE)))
        is_fire = jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.UPFIRE, jnp.logical_or(action == Action.DOWNFIRE, jnp.logical_or(action == Action.RIGHTFIRE, action == Action.LEFTFIRE))))
        is_fire = jnp.logical_or(is_fire, jnp.logical_or(action == Action.UPRIGHTFIRE, jnp.logical_or(action == Action.UPLEFTFIRE, jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE))))
        
        # Player Velocity is locked during jump (or falls after jumps), not during actual falls
        retained_drift = jnp.logical_and(state.level == 2, state.native_jump_momentum)
        drifting_fall = jnp.logical_and(state.is_falling == 1, retained_drift)
        is_in_air = jnp.logical_or(drifting_fall,
            jnp.logical_or(state.is_jumping == 1, state.fall_after_jump == 1))
        new_vx = jax.lax.select(is_right, self.consts.PLAYER_SPEED, 0)
        new_vx = jax.lax.select(is_left, -self.consts.PLAYER_SPEED, new_vx)

        air_vx = jnp.where(drifting_fall, state.player_dir, state.player_vx)
        air_clock = jnp.where(state.level > 0, state.native_frame_count, state.frame_count)
        air_dx = jnp.where(jnp.logical_and(is_in_air, air_clock % 2 == 0), air_vx, 0)
        air_dx = jnp.where(state.is_jumping, state.player_vx, air_dx) # Allow horizontal control during jump, but not during fall after jump
        dx = jnp.where(is_in_air, air_dx, new_vx)
        # x: move only every other frame (when in air)
        # just falling: no x velocity
        dx = jnp.where(jnp.logical_and(state.is_falling == 1,
            jnp.logical_and(state.fall_after_jump == 0, ~retained_drift)), 0, dx)
        # Bonus exit mode8 drops vertically, independently of the jump bit.
        dx = jnp.where(jnp.logical_and(is_bonus_room, state.native_mode == 8), 0, dx)
        new_player_dir = jnp.where(is_right, 1, jnp.where(is_left, -1, state.player_dir))
        new_player_dir = jnp.where(jnp.logical_and(state.level == 2, is_in_air),
            state.player_dir, new_player_dir)

        player_mid_x = state.player_x + self.consts.PLAYER_WIDTH // 2
        player_feet_y = state.player_y + self.consts.PLAYER_HEIGHT - 1
        
        # 0. Ladder and Rope Climbing Logic
        new_out_of_ladder_delay = jnp.where(state.out_of_ladder_delay > 0, state.out_of_ladder_delay - 1, 0)

        # Vectorized check_ladder
        l_x = state.ladders_x
        ladder_mid_x = l_x + 8
        l_top = state.ladders_top
        l_bottom = state.ladders_bottom
        # B3BF accepts ten native x positions, four left and five right of
        # the centered standing anchor, before snapping onto the ladder.
        ladder_delta = state.player_x + self.consts.LADDER_PLAYER_X_OFFSET - ladder_mid_x
        is_aligned_ladder = jnp.logical_and(state.ladders_active == 1,
            jnp.logical_and(ladder_delta >= -4,
                ladder_delta <= jnp.where(self.consts.CUSTOM_ROOMS, 4, 5)))
        is_aligned_ladder = jnp.logical_and(is_aligned_ladder, new_out_of_ladder_delay == 0)
        # Horizontal joystick input takes priority over entering or moving on ladders.
        ladder_down = jnp.logical_and(is_down, jnp.logical_not(jnp.logical_or(is_left, is_right)))
        ladder_up = jnp.logical_and(is_up, jnp.logical_not(jnp.logical_or(is_left, is_right)))
        get_on_top_ladder = jnp.logical_and(is_aligned_ladder, jnp.logical_and(ladder_down, jnp.abs(player_feet_y - l_top) <= 5))
        get_on_bottom_ladder = jnp.logical_and(is_aligned_ladder, jnp.logical_and(ladder_up, jnp.abs(player_feet_y - l_bottom) <= 5))
        is_airborne = jnp.logical_or(state.is_jumping == 1, jnp.logical_or(state.is_falling == 1, state.fall_after_jump == 1))
        can_grab_ladder = jnp.logical_or(
            get_on_top_ladder,
            jnp.logical_and(get_on_bottom_ladder, jnp.logical_not(is_airborne))
        )
        ladder_bottom_bound = jnp.where(l_bottom >= 148, 170, l_bottom + 1)
        # At the pit-room endpoint, held DOWN starts a two-pixel fall on
        # the tick after y73 if the floor is absent (native ALE clone probe).
        ladder_bottom_bound = jnp.where(
            jnp.logical_and(state.room_id == 19, ladder_down), l_bottom, ladder_bottom_bound)
        ladder_top_bound = jnp.where(l_top <= 6, 0, l_top - 4)
        in_ladder_zone = jnp.logical_and(is_aligned_ladder, jnp.logical_and(player_feet_y >= ladder_top_bound, player_feet_y <= ladder_bottom_bound))
        on_this_ladder = jnp.where(state.is_climbing == 1, jnp.logical_and(in_ladder_zone, jnp.logical_or(state.last_ladder == jnp.arange(self.consts.MAX_LADDERS_PER_ROOM), state.last_ladder == -1)), can_grab_ladder)
        ladder_handoff = jnp.logical_and(jnp.logical_or(get_on_top_ladder, jnp.logical_and(get_on_bottom_ladder, player_feet_y <= l_bottom - 1)),
            jnp.logical_and(state.is_climbing == 1,
                jnp.logical_and(state.last_ladder >= 0,
                    state.last_ladder != jnp.arange(self.consts.MAX_LADDERS_PER_ROOM))))
        on_this_ladder = jnp.logical_or(on_this_ladder, ladder_handoff)
        can_ladder = jnp.any(on_this_ladder)
        ladder_idx = jnp.where(jnp.any(ladder_handoff),
            jnp.argmax(ladder_handoff.astype(jnp.int32)),
            jnp.where(can_ladder, jnp.argmax(on_this_ladder.astype(jnp.int32)), -1))

        # Vectorized check_rope
        r_x = state.ropes_x
        r_width = rope_widths(state)
        r_top = state.ropes_top
        r_bottom = state.ropes_bottom
        # Rope contact requires overlap with the eight-pixel player pose.
        # A symmetric radius around the seven-pixel collision center catches
        # one pixel too early when approaching from the right.
        is_aligned_rope = jnp.logical_and(state.ropes_active == 1,
            jnp.logical_and(r_x + r_width > state.player_x,
                            r_x < state.player_x + 2 * self.consts.ROPE_PLAYER_X_OFFSET))
        player_top_y = state.player_y
        intersect_y_rope = jnp.logical_and(player_feet_y >= r_top, player_top_y <= r_bottom)
        catch_rope = jnp.logical_and(state.is_climbing == 0, jnp.logical_and(is_aligned_rope, jnp.logical_and(intersect_y_rope, state.last_rope != jnp.arange(self.consts.MAX_ROPES_PER_ROOM))))
        # B26A skips rope contact while the remaining jump timer is >=12.
        # It starts at19 and is decremented before the contact probe.
        catch_rope = jnp.logical_and(catch_rope, jnp.logical_or(state.level < 2,
            jnp.logical_or(state.is_jumping == 0, state.jump_counter >= 7)))
        # The falling pose first contacts the upper rope at its visible head.
        catch_rope = jnp.logical_and(catch_rope, jnp.logical_or(state.room_id != 17,
            state.player_y >= r_top - jnp.where(jnp.logical_and(state.level == 2, state.player_dir == 1), 15, 13)))
        # The right-hand rope's bottom catch uses the raised-arm pose.
        catch_rope = jnp.logical_and(catch_rope, jnp.logical_or(
            jnp.logical_or(state.room_id != 12, jnp.arange(r_x.shape[0]) != 1),
            state.player_y <= r_bottom - 12))
        get_on_top_rope = jnp.logical_and(is_aligned_rope, jnp.logical_and(is_down, jnp.abs(player_feet_y - r_top) <= 5))
        get_on_bottom_rope = jnp.logical_and(is_aligned_rope, jnp.logical_and(is_up, jnp.abs(player_feet_y - r_bottom) <= 5))
        can_climb_above = jnp.logical_or(jnp.logical_and(state.room_id == 12, jnp.arange(self.consts.MAX_ROPES_PER_ROOM) == 0), jnp.logical_and(state.room_id == 17, jnp.arange(self.consts.MAX_ROPES_PER_ROOM) == 0))
        top_bound_rope = jnp.where(can_climb_above, r_top - 5, r_top)
        in_rope_zone = jnp.logical_and(is_aligned_rope, jnp.logical_and(player_feet_y >= top_bound_rope, player_top_y <= r_bottom + 2))
        on_this_rope = jnp.where(state.is_climbing == 1, jnp.logical_and(in_rope_zone, jnp.logical_or(state.last_rope == jnp.arange(self.consts.MAX_ROPES_PER_ROOM), state.last_rope == -1)), jnp.logical_and(in_rope_zone, jnp.logical_or(get_on_top_rope, jnp.logical_or(get_on_bottom_rope, catch_rope))))
        top_entry_rope = jnp.logical_and(jnp.isin(state.room_id, jnp.array([17,23])),
            jnp.logical_and(state.is_climbing == 0,
                jnp.logical_and(ladder_down, jnp.logical_and(is_aligned_rope,
                    jnp.abs(player_feet_y + 1 - (r_top - 3)) <= 1))))
        # B41A preserves the walking bitmap during vertical input in the
        # stair chamber. At its upper rope the feet, rather than a bounding
        # rectangle, must reach the rope before B2AD can grab it.
        rope_pose = self._player_collision_masks[state.collision_player_pose]
        rope_pose = jax.lax.cond(state.collision_player_dir == 1,
            lambda: jnp.flip(rope_pose, axis=1), lambda: rope_pose)
        ry, rx = jnp.indices(rope_pose.shape)
        rope_pixel_contact = jnp.any(jnp.logical_and(rope_pose[None, :, :],
            jnp.logical_and(
                jnp.logical_and(state.player_x + rx >= r_x[:, None, None],
                    state.player_x + rx < (r_x + r_width)[:, None, None]),
                jnp.logical_and(state.player_y + ry >= r_top[:, None, None],
                    state.player_y + ry <= r_bottom[:, None, None]))), axis=(1, 2))
        stair_rope_contact = jnp.where(jnp.logical_and(state.level == 2, state.native_walk_pose),
            rope_pixel_contact,
            state.player_y >= r_top - jnp.where(jnp.logical_and(state.level == 2, state.player_dir == 1), 15, 13))
        on_this_rope = jnp.logical_and(on_this_rope, jnp.logical_or(state.room_id != 17,
            jnp.logical_or(state.is_climbing == 1, stair_rope_contact)))
        on_this_rope = jnp.logical_or(on_this_rope, jnp.logical_and(
            jnp.logical_and(state.level == 2, state.room_id == 17),
            jnp.logical_and(state.native_walk_pose, jnp.logical_and(in_rope_zone, rope_pixel_contact))))
        on_this_rope = jnp.logical_or(on_this_rope, top_entry_rope)
        can_rope = jnp.any(on_this_rope)
        rope_idx = jnp.where(can_rope, jnp.argmax(on_this_rope.astype(jnp.int32)), -1)

        raw_new_x_check = state.player_x + dx
        new_left_x_check = jnp.clip(raw_new_x_check, 0, self.consts.WIDTH - 1)
        new_right_x_check = jnp.clip(raw_new_x_check + self.consts.PLAYER_WIDTH - 1, 0, self.consts.WIDTH - 1)
        front_x_check = jnp.where(dx > 0, new_right_x_check, new_left_x_check)
        
        check_y_top_check = jnp.clip(state.player_y, 0, 148)
        check_y_mid_check = jnp.clip(state.player_y + self.consts.PLAYER_HEIGHT // 2, 0, 148)
        check_y_bot_check = jnp.clip(player_feet_y, 0, 148)
        
        hit_wall_check = jnp.logical_or(
            room_col_map[check_y_top_check, front_x_check] == 1,
            jnp.logical_or(
                room_col_map[check_y_mid_check, front_x_check] == 1,
                room_col_map[check_y_bot_check, front_x_check] == 1
            )
        )

        can_move_off = jnp.logical_and(jnp.logical_or(is_left, is_right), jnp.logical_not(hit_wall_check))

        # Ladders are vertical-only: horizontal input does not disengage climbing.
        abort_ladder = jnp.logical_and(can_ladder,
            jnp.logical_and(state.is_climbing == 1,
                jnp.logical_and(jnp.logical_or(is_left, is_right),
                    jnp.logical_and(l_bottom[jnp.maximum(ladder_idx, 0)] <= 47,
                        player_feet_y >= l_bottom[jnp.maximum(ladder_idx, 0)] - 6))))

        abort_ladder_top = jnp.logical_and(can_ladder,
            jnp.logical_and(state.is_climbing == 1,
                jnp.logical_and(jnp.logical_or(is_left, is_right),
                    jnp.logical_and(l_top[jnp.maximum(ladder_idx, 0)] >= 46,
                        player_feet_y <= l_top[jnp.maximum(ladder_idx, 0)] - 3))))
        abort_ladder = jnp.logical_or(abort_ladder, abort_ladder_top)

        is_jumping_off_rope = jnp.logical_and(can_rope, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(jnp.logical_and(is_fire, jnp.logical_not(state.prev_is_fire)), can_move_off)))
        abort_rope = is_jumping_off_rope

        is_climbing_ladder = jnp.logical_and(can_ladder, jnp.logical_not(abort_ladder))
        # ALE holds the final bottom anchor for one tick, then changes
        # from rope pose to the falling/standing anchor on the next tick.
        rope_release = jnp.logical_and(jnp.isin(state.room_id, jnp.array([17, 23])),
            jnp.logical_and(state.is_climbing == 1,
                jnp.logical_and(ladder_down, jnp.logical_and(rope_idx >= 0,
                    state.player_y >= r_bottom[jnp.maximum(rope_idx, 0)]))))
        rope_top_exit = jnp.logical_and(jnp.isin(state.room_id, jnp.array([17,23])),
            jnp.logical_and(state.is_climbing == 1,
                jnp.logical_and(ladder_up, jnp.logical_and(jnp.logical_or(state.room_id == 23, rope_idx == 0),
                    state.player_y <= r_top[jnp.maximum(rope_idx, 0)] - 7))))
        rope_top_drop = jnp.logical_and(state.room_id == 17,
            jnp.logical_and(state.is_climbing == 0,
                jnp.logical_and(ladder_down, jnp.logical_and(rope_idx == 0,
                    state.player_y + self.consts.PLAYER_HEIGHT <= r_top[0]))))
        is_climbing_rope = jnp.logical_and(can_rope, jnp.logical_not(jnp.logical_or(jnp.logical_or(abort_rope, rope_release), jnp.logical_or(rope_top_exit, rope_top_drop))))

        is_climbing = jnp.where(jnp.logical_or(is_climbing_ladder, is_climbing_rope), 1, 0)
        
        # If we are aborting the ladder, or simply falling off, start the delay
        # But only if we were previously climbing!
        started_delay = jnp.logical_and(state.is_climbing == 1, is_climbing == 0)
        new_out_of_ladder_delay = jnp.where(started_delay, self.consts.OUT_OF_LADDER_DELAY, new_out_of_ladder_delay)

        target_climb_x = state.player_x
        target_climb_x = jnp.where(ladder_idx != -1, state.ladders_x[ladder_idx] + 8 - self.consts.LADDER_PLAYER_X_OFFSET, target_climb_x)
        target_climb_x = jnp.where(rope_idx != -1, state.ropes_x[rope_idx] + r_width[rope_idx] // 2 - self.consts.ROPE_PLAYER_X_OFFSET, target_climb_x)
        target_climb_x = jnp.where(jnp.logical_and(state.room_id == 12, rope_idx == 1), 123, target_climb_x)
        
        current_x = jnp.where(is_climbing == 1, target_climb_x, state.player_x)
        current_x = jnp.where(rope_top_drop, target_climb_x, current_x)
        
        def check_platform_local(y, x):
            # Standing/landing contact uses the player's collision footprint.
            # A center-only probe misses a ledge reached by the leading foot,
            # e.g. the top-right platform after the first-room return jump.
            footprint_x = jnp.clip(x + jnp.arange(self.consts.PLAYER_WIDTH) - self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)
            contact = jnp.where(jnp.isin(state.room_id, jnp.array([12,29])),
                room_col_map[y, jnp.clip(x, 0, self.consts.WIDTH-1)] == 1,
                jnp.any(room_col_map[y, footprint_x] == 1))
            pit = jnp.isin(state.room_id, jnp.array([19,27,29,31]))
            foot = x + jnp.where(pit, (state.player_dir < 0).astype(jnp.int32), 2)
            contact = jnp.where(jnp.logical_and(state.level == 2,
                jnp.logical_or(pit, state.is_falling == 1)),
                room_col_map[y, jnp.clip(foot, 0, self.consts.WIDTH-1)] == 1, contact)
            # B630 accepts native X48..103 on the torch-room middle shelf.
            # Its visible right edge extends past the landing interval.
            brown_floor = jnp.logical_and(jnp.logical_and(state.level == 2, state.room_id == 12),
                jnp.logical_and(y >= 83, y < 86))
            brown_anchor = jnp.logical_and(state.player_x >= 47, state.player_x < 103)
            return jnp.logical_and(contact, jnp.logical_or(~brown_floor, brown_anchor))
        
        # 1. Check if strictly on ground
        safe_x = jnp.clip(current_x + self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)
        safe_y = jnp.clip(player_feet_y + 1, 0, 148)
        on_ground = check_platform_local(safe_y, safe_x)
        
        # Vectorized conveyor check
        is_on_conveyor_ground = jnp.logical_and(
            state.conveyors_active == 1,
            jnp.logical_and(player_feet_y == state.conveyors_y - 1, jnp.logical_and(player_mid_x >= state.conveyors_x - jnp.where(state.level == 2, 1, 2), player_mid_x < state.conveyors_x + 42))
        )
        on_ground = jnp.logical_or(on_ground, jnp.any(is_on_conveyor_ground))

        # Vectorized platform check
        platform_contact_x = jnp.logical_and(safe_x >= state.platforms_x,
            safe_x < state.platforms_x + state.platforms_width)
        # DB10 tests the standing anchor, not the visible step width.
        # Sprite contact can still permit FIRE just inside either edge.
        stair_side_anchor = jnp.logical_or(current_x <= 8, current_x >= 144)
        platform_support_x = jnp.logical_and(platform_contact_x,
            jnp.logical_or(state.room_id != 17, stair_side_anchor))
        jump_on_ground = jnp.logical_or(on_ground, jnp.any(jnp.logical_and(
            jnp.logical_and(state.platforms_active == 1, platform_active_now),
            jnp.logical_and(player_feet_y == state.platforms_y - 1,
                jnp.where(state.room_id == 17,
                    jnp.logical_and(current_x + self.consts.PLAYER_WIDTH > state.platforms_x,
                        current_x < state.platforms_x + state.platforms_width), platform_contact_x)))))
        is_on_plat_ground = jnp.logical_and(
            jnp.logical_and(state.platforms_active == 1, platform_active_now),
            jnp.logical_and(player_feet_y == state.platforms_y - 1, platform_support_x)
        )
        on_ground = jnp.logical_or(on_ground, jnp.any(is_on_plat_ground))
        native_pit_upper = jnp.logical_and(state.level == 2,
            jnp.logical_and(jnp.isin(state.room_id, jnp.array([19,27,29,31])), state.player_y == 26))
        on_ground = jnp.where(native_pit_upper,
            jnp.logical_or(platform_active_now,
                jnp.logical_or(state.player_x < 33, state.player_x >= 120)), on_ground)
        jump_on_ground = jnp.where(native_pit_upper, on_ground, jump_on_ground)

        dropout_collapse = jnp.logical_and(state.level == 2,
            jnp.logical_and((state.native_frame_count + 1 + 31) % 128 == 92,
                jnp.logical_and(~check_platform_local(safe_y, safe_x),
                    jnp.any(jnp.logical_and(state.platforms_active == 1,
                        jnp.logical_and(player_feet_y == state.platforms_y - 1,
                            jnp.logical_and(safe_x >= state.platforms_x,
                                safe_x < state.platforms_x + state.platforms_width)))))))


        # Update last_rope and last_ladder
        new_last_rope = jnp.where(on_ground, -1, state.last_rope)
        new_last_rope = jnp.where(is_climbing_rope, rope_idx, new_last_rope)
        new_last_rope = jnp.where(is_jumping_off_rope, rope_idx, new_last_rope)

        new_last_ladder = jnp.where(on_ground, -1, state.last_ladder)
        new_last_ladder = jnp.where(is_climbing == 1, ladder_idx, new_last_ladder)

        # 2. Process Jump Initiation
        was_on_ladder = jnp.logical_and(state.is_climbing == 1, state.last_ladder != -1)
        start_jump_normal = jnp.logical_and(
            # B1C8 rejects a purely downward joystick before preparing a hop.
            # FIRE is still sampled; diagonal horizontal input takes priority.
            jnp.logical_and(jnp.logical_and(is_fire, jnp.logical_not(state.prev_is_fire)),
                jnp.logical_or(self.consts.CUSTOM_ROOMS, ~ladder_down)),
            jnp.logical_and(
                jump_on_ground,
                jnp.logical_and(
                    state.is_jumping == 0,
                    jnp.logical_and(is_climbing == 0, jnp.logical_not(was_on_ladder)),
                ),
            ),
        )
        start_jump = jnp.logical_or(start_jump_normal, is_jumping_off_rope)
        new_player_dir = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(state.is_climbing == 1, ~is_jumping_off_rope)),
            state.player_dir, new_player_dir)
        # ROM short-jump variant for stationary hops on the room8 side steps.
        stair_jump = jnp.logical_and(state.room_id == 17,
            jnp.logical_and(state.player_y > 26,
                jnp.logical_and(jnp.logical_not(jnp.logical_or(is_left, is_right)),
                    jnp.logical_or(state.player_x <= 8, state.player_x >= 144))))
        jump_type = jnp.where(start_jump, stair_jump.astype(jnp.int32), state.jump_type)
        is_jumping = jnp.where(start_jump, 1, state.is_jumping)
        is_jumping = jnp.where(is_climbing == 1, 0, is_jumping) # cancel jump
        jump_counter = jnp.where(start_jump, 0, state.jump_counter)

        # Keep latched airborne momentum even on frames where dx is intentionally zero.
        current_vx = jnp.where(is_in_air, air_vx, dx)
        # ALE latches jump momentum now, but starts position changes next frame.
        # The final jump descent frame holds x before the post-jump horizontal cadence.
        last_jump_frame = jnp.logical_and(state.is_jumping == 1, state.jump_counter == self.consts.JUMP_Y_OFFSETS.shape[0] - 1)
        dx = jnp.where(jnp.logical_or(start_jump, last_jump_frame), 0, dx)

        # 3. Calculate DY
        def get_jump_dy():
            dy_jump = jnp.where(start_jump, 0, -self.consts.JUMP_Y_OFFSETS[jump_counter])
            return dy_jump, jnp.where(start_jump, 0, jump_counter + 1), 1
            
        def get_fall_dy():
            pixel_1_below = check_platform_local(safe_y, safe_x)
            pixel_2_below = check_platform_local(jnp.clip(player_feet_y + 2, 0, 148), safe_x)
            
            is_on_c_pixel2 = jnp.logical_and(
                state.conveyors_active == 1,
                jnp.logical_and(player_feet_y + 1 == state.conveyors_y - 1, jnp.logical_and(safe_x >= state.conveyors_x - jnp.where(state.level == 2, 1, 2), safe_x < state.conveyors_x + 42))
            )
            pixel_2_below = jnp.logical_or(pixel_2_below, jnp.any(is_on_c_pixel2))

            is_on_p_pixel2 = jnp.logical_and(
                jnp.logical_and(state.platforms_active == 1, platform_active_now),
                jnp.logical_and(player_feet_y + 1 == state.platforms_y - 1, platform_support_x)
            )
            pixel_2_below = jnp.logical_or(pixel_2_below, jnp.any(is_on_p_pixel2))


            fall_dist = jnp.where(on_ground, 0, jnp.where(pixel_2_below, 1, self.consts.GRAVITY))
            return fall_dist, 0, 0
            
        def get_climb_dy():
            # Horizontal joystick input wins on ropes as on ladders.
            # In ALE, held UPRIGHT catches the room23 rope but does not
            # climb until the input becomes purely vertical (native4723).
            climb_dist = jnp.where(ladder_down, self.consts.PLAYER_SPEED,
                                   jnp.where(ladder_up, -self.consts.PLAYER_SPEED, 0))
            # Switching from standing to the top ladder pose moves the player anchor.
            entered_ladder_top = jnp.logical_and(jnp.logical_or(state.is_climbing == 0, jnp.any(ladder_handoff)), jnp.logical_and(ladder_idx != -1, get_on_top_ladder[jnp.maximum(ladder_idx, 0)]))
            entry_y = state.ladders_top[jnp.maximum(ladder_idx, 0)] - self.consts.PLAYER_HEIGHT + 5
            climb_dist = jnp.where(entered_ladder_top, entry_y - state.player_y, climb_dist)
            bottom_handoff = jnp.logical_and(jnp.any(ladder_handoff),
                get_on_bottom_ladder[jnp.maximum(ladder_idx, 0)])
            entered_ladder_bottom = jnp.logical_and(
                jnp.logical_or(bottom_handoff, jnp.logical_or(jnp.isin(state.room_id, jnp.array([4, 13, 19, 20, 18, 30, 31])),
                    jnp.logical_and(state.level == 2, state.room_id == 28))),
                jnp.logical_and(jnp.logical_or(state.is_climbing == 0, bottom_handoff),
                    jnp.logical_and(ladder_idx != -1, get_on_bottom_ladder[jnp.maximum(ladder_idx, 0)])))
            # The measured bottom-entry pose changes the anchor by six pixels
            # in rooms4/20 and five in rooms13/19; attached motion stays one per tick.
            bottom_entry_y = state.player_y - jnp.where(jnp.logical_or(jnp.isin(state.room_id, jnp.array([4,20])), bottom_handoff), 6, 5)
            climb_dist = jnp.where(entered_ladder_bottom, bottom_entry_y - state.player_y, climb_dist)
            # Rope attachment occurs after this frame's airborne movement.
            # Cancelling it early loses the final descent pixels at the catch.
            just_caught_rope = jnp.logical_and(state.is_climbing == 0, rope_idx != -1)
            catch_dy = jnp.where(
                state.is_jumping == 1,
                -self.consts.JUMP_Y_OFFSETS[jnp.minimum(state.jump_counter, self.consts.JUMP_Y_OFFSETS.shape[0] - 1)],
                jnp.where(jnp.logical_and(state.level == 2, state.native_mode == 4),
                    climb_dist, jnp.int32(0)),
            )
            catch_dy = catch_dy - jnp.logical_and(state.room_id == 12, rope_idx == 1).astype(jnp.int32)
            climb_dist = jnp.where(just_caught_rope, catch_dy, climb_dist)
            climb_dist = jnp.where(jnp.logical_and(state.room_id == 23, jnp.any(top_entry_rope)),
                r_top[jnp.maximum(rope_idx, 0)] - 14 - state.player_y, climb_dist)
            return climb_dist, 0, 0

        dy, new_jump_counter, new_is_jumping = jax.lax.cond(
            is_climbing == 1,
            get_climb_dy,
            lambda: jax.lax.cond(
                is_jumping == 1,
                get_jump_dy,
                get_fall_dy
            )
        )
        
        short_jump_end = jnp.logical_and(state.is_jumping == 1,
            jnp.logical_and(state.jump_type == 1, state.jump_counter == 15))
        dy = jnp.where(short_jump_end, dy + 1, dy)
        new_jump_counter = jnp.where(short_jump_end, self.consts.JUMP_Y_OFFSETS.shape[0], new_jump_counter)
        after_rope_release = jnp.logical_and(state.room_id == 23,
            jnp.logical_and(state.last_rope >= 0,
                jnp.logical_and(state.is_climbing == 0,
                    jnp.logical_and(state.is_jumping == 0,
                        state.out_of_ladder_delay == self.consts.OUT_OF_LADDER_DELAY))))
        stair_gap = jnp.logical_and(state.room_id == 17,
            jnp.logical_and(~stair_side_anchor,
                jnp.logical_and(state.player_y > 26, state.player_y < 106)))
        native_support_loss = jnp.logical_and(jnp.logical_or(state.level == 2,
            jnp.logical_and(stair_gap, ~state.native_walk_pose)),
            jnp.logical_and(state.native_mode == 4, jnp.logical_and(~on_ground,
                jnp.logical_and(state.player_x >= 4, jnp.logical_and(state.player_x <= 148,
                    jnp.logical_and(state.is_falling == 0, jnp.logical_and(state.is_jumping == 0,
                        jnp.logical_and(is_climbing == 0, ~start_jump))))))))
        native_support_loss = jnp.logical_and(native_support_loss, ~abort_ladder)
        dy = jnp.where(jnp.logical_or(native_support_loss, jnp.logical_or(bonus_collapse, dropout_collapse)), 0, dy)
        dy = jnp.where(rope_release, 0, jnp.where(after_rope_release, 4, dy))
        dy = jnp.where(rope_top_exit, 46 - self.consts.PLAYER_HEIGHT - state.player_y,
            jnp.where(rope_top_drop, 8, dy))
        dy = jnp.where(abort_ladder,
            jnp.where(abort_ladder_top, l_top[jnp.maximum(ladder_idx, 0)] - 4, jnp.where(l_bottom[jnp.maximum(ladder_idx, 0)] <= 47, 46, l_bottom[jnp.maximum(ladder_idx, 0)])) - self.consts.PLAYER_HEIGHT - state.player_y, dy)
        dx = jnp.where(abort_ladder, 0, dx)
        new_is_jumping = jnp.where(new_jump_counter >= self.consts.JUMP_Y_OFFSETS.shape[0], 0, new_is_jumping)
        new_is_jumping = jnp.where(bonus_collapse, 0, new_is_jumping)
        
        # 4. Resolve Vertical Collision
        new_y = state.player_y + dy

        # Fall distance only increments when below y position of jump start. Important for jumps from ropes to platforms.
        new_fall_distance = jnp.where(new_is_jumping == 1, state.fall_distance - dy, state.fall_distance)

        # Allow climbing slightly above the rope top when pressing UP to reach platforms
        can_climb_above_rope = jnp.logical_or(
            jnp.logical_and(state.room_id == 12, rope_idx == 0),
            jnp.logical_or(state.room_id == 23, jnp.logical_and(state.room_id == 17, rope_idx == 0))
        )
        top_extension = jnp.where(can_climb_above_rope, 25, 0)
        top_extension = jnp.where(jnp.logical_and(is_down, can_climb_above_rope), 15, top_extension)
        rope_top_limit = state.ropes_top[rope_idx] - top_extension
        new_y = jnp.where(jnp.logical_and(state.is_climbing == 1,
            jnp.logical_and(jnp.logical_or(ladder_up, ladder_down),
                jnp.logical_and(is_climbing == 1, rope_idx != -1))), jnp.maximum(new_y, rope_top_limit), new_y)
        new_feet_y = new_y + self.consts.PLAYER_HEIGHT - 1

        # Convert the last upward climbing pose to the supported standing
        # anchor on this frame, so FIRE on the next frame can prepare a jump.
        # The start-room central ladder reaches its standing anchor smoothly;
        # the two side ladders change pose five pixels before that anchor.
        ladder_top_exit = state.ladders_top[jnp.maximum(ladder_idx, 0)] - jnp.where(ladder_idx == 0, 5, 0)
        left_ladder_top = jnp.logical_and(
            jnp.isin(state.room_id, jnp.array([4, 5, 13])),
            jnp.logical_and(jnp.logical_and(is_climbing_ladder, ladder_up), new_feet_y <= ladder_top_exit),
        )
        top_exit_y = state.ladders_top[jnp.maximum(ladder_idx, 0)] - self.consts.PLAYER_HEIGHT - 4
        new_y = jnp.where(left_ladder_top, top_exit_y, new_y)
        new_feet_y = new_y + self.consts.PLAYER_HEIGHT - 1
        is_climbing = jnp.where(left_ladder_top, 0, is_climbing)
        new_last_ladder = jnp.where(left_ladder_top, -1, new_last_ladder)
        new_out_of_ladder_delay = jnp.where(left_ladder_top, self.consts.OUT_OF_LADDER_DELAY, new_out_of_ladder_delay)

        # Calculate if we are near the top of what we are climbing
        climb_top = jnp.where(ladder_idx != -1, state.ladders_top[ladder_idx], 0)
        climb_top = jnp.where(rope_idx != -1, state.ropes_top[rope_idx], climb_top)
        is_near_top = jnp.logical_and(is_climbing == 1, player_feet_y <= climb_top + 5)

        # All platforms (and ceilings) are permeable from below. 
        # We only hit floors when moving downwards.
        hit_ceiling = 0
        new_y = jnp.where(hit_ceiling, state.player_y, new_y)
        new_is_jumping = jnp.where(hit_ceiling, 0, new_is_jumping)
        
        # Improved static platform collision: check if we crossed any solid pixel that acts as a top surface
        y_check_offsets = jnp.arange(5)
        y_checks = jnp.clip(player_feet_y + 1 + y_check_offsets, 0, 148)
        
        # A pixel y_check is a top surface if it's solid AND the pixel above it is empty
        def is_solid_func(y):
            return check_platform_local(y, safe_x)
        
        is_solid = jax.vmap(is_solid_func)(y_checks)
        is_solid_above = jax.vmap(is_solid_func)(jnp.clip(y_checks - 1, 0, 148))
        is_top_surface = jnp.logical_and(is_solid, jnp.logical_not(is_solid_above))
        
        # Contact is reached when the feet are one pixel above the surface,
        # matching snapped_y_rm and the standing on_ground test. Waiting for
        # penetration leaves a completed jump falsely falling for one frame.
        # Descending jump contact converts the airborne anchor before standing feet
        # reach the ledge; native ALE snaps the final two pixels on this tick.
        converted_fall = jnp.logical_and(state.is_falling == 1,
            jnp.logical_or(jnp.isin(state.room_id, jnp.array([12,17])),
                jnp.logical_or(jnp.logical_and(state.level > 0, state.room_id == 4),
                    jnp.logical_and(state.level == 2, state.room_id == 23))))
        landing_reach = dy + jnp.where(jnp.logical_or(state.is_jumping == 1, converted_fall), 2, 0)
        is_hit_rm = jnp.logical_and(jnp.logical_not(is_near_top), jnp.logical_and(dy > 0, jnp.logical_and(landing_reach >= y_check_offsets, is_top_surface)))
        hit_floor_rm = jnp.logical_and(jnp.any(is_hit_rm), jnp.logical_not(rope_top_drop))
        # Standing dispatch keeps a running jump permeable to the fixed
        # shelves. The stair chamber also has a permanent middle shelf.
        fixed_shelf_y = y_checks[jnp.argmax(is_hit_rm.astype(jnp.int32))] - self.consts.PLAYER_HEIGHT
        hit_floor_rm = jnp.logical_and(hit_floor_rm, ~jnp.logical_and(
            jnp.isin(state.room_id, jnp.array([17,23])),
            jnp.logical_and(new_is_jumping == 1,
                jnp.logical_or(fixed_shelf_y == 26,
                    jnp.logical_and(state.room_id == 17, fixed_shelf_y == 76)))))
        snapped_y_rm = jnp.where(hit_floor_rm, y_checks[jnp.argmax(is_hit_rm.astype(jnp.int32))] - self.consts.PLAYER_HEIGHT, new_y)

        # Conveyor hit floor
        crossed_c = jnp.logical_and(player_feet_y <= state.conveyors_y - 1, new_feet_y + jnp.where(jnp.logical_or(state.is_jumping == 1, converted_fall), 2, 0) >= state.conveyors_y - 1)
        is_hit_c = jnp.logical_and(
            jnp.logical_not(is_near_top),
            jnp.logical_and(
                state.conveyors_active == 1,
                jnp.logical_and(dy > 0, jnp.logical_and(crossed_c, jnp.logical_and(safe_x >= state.conveyors_x - jnp.where(state.level == 2, 1, 2), safe_x < state.conveyors_x + 42)))
            )
        )
        # At a ladder-to-belt exit ALE changes from the climbing pose to
        # the standing anchor four pixels before the normal feet contact.
        # Restrict this conversion to a ladder ending at an active belt.
        ladder_to_conveyor = jnp.logical_and(
            jnp.logical_and(is_climbing_ladder, ladder_down),
            jnp.logical_and(
                state.conveyors_active == 1,
                jnp.logical_and(
                    jnp.abs(state.ladders_bottom[jnp.maximum(ladder_idx, 0)] - state.conveyors_y) <= 1,
                    jnp.logical_and(
                        new_feet_y >= state.conveyors_y - 5,
                        jnp.logical_and(safe_x >= state.conveyors_x - jnp.where(state.level == 2, 1, 2), safe_x < state.conveyors_x + 42),
                    ),
                ),
            ),
        )
        is_hit_c = jnp.logical_or(is_hit_c, ladder_to_conveyor)
        hit_floor_c = jnp.any(is_hit_c)
        snapped_y_c = jnp.where(hit_floor_c, state.conveyors_y[jnp.argmax(is_hit_c.astype(jnp.int32))] - 1 - self.consts.PLAYER_HEIGHT + 1, snapped_y_rm)

        # Platform hit floor
        crossed_p = jnp.logical_and(player_feet_y <= state.platforms_y - 1, new_feet_y >= state.platforms_y - 1)
        is_hit_p = jnp.logical_and(
            jnp.logical_not(is_near_top),
            jnp.logical_and(
                jnp.logical_and(state.platforms_active == 1, platform_active_now),
                jnp.logical_and(dy > 0, jnp.logical_and(crossed_p, platform_support_x))
            )
        )
        hit_floor_p = jnp.any(is_hit_p)
        snapped_y = jnp.where(hit_floor_p, state.platforms_y[jnp.argmax(is_hit_p.astype(jnp.int32))] - 1 - self.consts.PLAYER_HEIGHT + 1, snapped_y_c)
        hit_floor = jnp.logical_or(hit_floor_rm, jnp.logical_or(hit_floor_c, hit_floor_p))

        new_y = jnp.where(hit_floor, snapped_y, new_y)
        dx = jnp.where(jnp.logical_and(jnp.logical_or(state.is_jumping == 1,
            jnp.logical_and(state.room_id == 12, state.is_falling == 1)), hit_floor), 0, dx)

        # Stop climbing if we hit a floor (e.g. landing on a conveyor belt)
        is_climbing = jnp.where(hit_floor, 0, is_climbing)
        new_last_ladder = jnp.where(hit_floor, -1, new_last_ladder)
        new_is_jumping = jnp.where(hit_floor, 0, new_is_jumping)

        # Set is_falling state
        new_is_falling = jnp.where(jnp.logical_and(new_is_jumping == 0, hit_floor == False), jnp.where(dy > 0, 1, 0), 0)
        new_is_falling = jnp.where(is_climbing == 1, 0, new_is_falling)
        new_is_falling = jnp.where(jnp.logical_or(rope_release,
            jnp.logical_or(native_support_loss, jnp.logical_or(bonus_collapse, dropout_collapse))), 1, new_is_falling)
        # The stair-room top entry centers x and lowers the anchor by eight.
        # A retained walking pose stays in standing dispatch for this tick.
        stair_walk_hold = jnp.logical_and(state.room_id == 17,
            jnp.logical_and(jnp.logical_or(jnp.logical_or(is_up, is_down), jnp.logical_or(is_left, is_right)),
                jnp.logical_and(state.native_walk_pose,
                    jnp.logical_and(state.native_mode == 4,
                        jnp.logical_and(state.is_jumping == 0, jnp.logical_and(is_climbing == 0,
                            jnp.logical_or(rope_top_drop,
                                jnp.logical_and(state.player_y > 26, state.player_y < 76))))))))
        new_is_falling = jnp.where(stair_walk_hold, 0, new_is_falling)
        new_is_falling = jnp.where(jnp.logical_or(rope_top_exit, abort_ladder), 0, new_is_falling)
        dx = jnp.where(stair_walk_hold, 0, dx)
        
        # 5. Resolve Horizontal with Wall Collision
        # Once falling, the pit ignores new horizontal input. Standing
        # dispatch still completes its movement on the floor-collapse tick.
        dx = jnp.where(jnp.logical_and(state.room_id == 29,
            jnp.logical_and(state.native_mode == 7, jnp.logical_and(jnp.logical_not(on_ground),
                jnp.logical_and(is_climbing == 0, jnp.logical_and(state.is_jumping == 0, state.fall_after_jump == 0))))), 0, dx)
        raw_new_x = current_x + dx
        transition_left = jnp.logical_and(dx < 0, raw_new_x < 4)
        transition_right = jnp.logical_and(dx > 0, raw_new_x > 148)
        # FC76/FCE0 hold side exits below the upper-floor anchor, even
        # while the jump keeps pushing against a lower boundary wall.
        side_exit_height = jnp.logical_or(self.consts.CUSTOM_ROOMS, new_y <= 26)
        transition_left = jnp.logical_and(transition_left, side_exit_height)
        transition_right = jnp.logical_and(transition_right, side_exit_height)
        # B20C skips horizontal dispatch for stationary hops. A new joystick
        # direction during such a hop cannot trigger a side-room transition.
        transition_down = jnp.logical_and(dy > 0, jnp.logical_and(new_y >= self.consts.ROOM_EXIT_Y_BOTTOM, jnp.logical_not(state.room_id == 24)))
        transition_up = jnp.logical_and(dy < 0, new_y <= self.consts.ROOM_EXIT_Y_TOP)
        exits = NATIVE_EXIT_MASKS[jnp.argmax(ALE_ROOM_IDS == state.room_id)]
        transition_left = jnp.logical_and(transition_left, jnp.logical_or(state.level < 2, (exits & 4) != 0))
        transition_right = jnp.logical_and(transition_right, jnp.logical_or(state.level < 2, (exits & 8) != 0))
        transition_down = jnp.logical_and(transition_down, jnp.logical_or(state.level < 2, (exits & 2) != 0))
        transition_up = jnp.logical_and(transition_up, jnp.logical_or(state.level < 2, (exits & 1) != 0))

        new_x = jnp.clip(raw_new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        new_x = jnp.where(jnp.logical_and(not self.consts.CUSTOM_ROOMS, ~side_exit_height),
            jnp.clip(new_x, 4, 148), new_x)
        # Walking off the upper start-room ledge prepares a stationary fall
        # at the new horizontal anchor before the first downward tick.
        ledge_fall = jnp.logical_and(jnp.logical_and(state.level > 0, state.room_id == 4),
            jnp.logical_and(state.player_y == 26, jnp.logical_and(on_ground,
                jnp.logical_and(is_climbing == 0, jnp.logical_and(new_is_jumping == 0,
                    ~check_platform_local(safe_y, new_x + self.consts.PLAYER_WIDTH // 2))))))
        pit = jnp.isin(state.room_id, jnp.array([19,27,29,31]))
        support_x = new_x + jnp.where(pit, jnp.where(dx < 0, 4, 3), jnp.where(dx < 0, 5, 3))
        new_support = room_col_map[safe_y, jnp.clip(support_x, 0, self.consts.WIDTH - 1)] == 1
        new_support = jnp.logical_or(new_support, jnp.any(jnp.logical_and(
            jnp.logical_and(state.platforms_active == 1, platform_active_now),
            jnp.logical_and(player_feet_y == state.platforms_y - 1,
                jnp.logical_and(support_x >= state.platforms_x,
                    support_x < state.platforms_x + state.platforms_width)))))
        new_support = jnp.logical_or(new_support, jnp.any(jnp.logical_and(state.conveyors_active == 1,
            jnp.logical_and(player_feet_y == state.conveyors_y - 1,
                jnp.logical_and(support_x >= state.conveyors_x - jnp.where(state.level == 2, 1, 2), support_x < state.conveyors_x + 42)))))
        # DA74 tests native X34..120 at y73. The generic leading-foot
        # footprint wrongly opens a gap while returning to the left bank.
        new_support = jnp.where(jnp.logical_and(pit, state.player_y == 26),
            jnp.logical_or(platform_active_now, jnp.logical_or(new_x < 33, new_x >= 120)), new_support)
        native_ledge_fall = jnp.logical_and(jnp.logical_and(state.level == 2,
            jnp.logical_and(state.player_x >= 4, state.player_x <= 148)),
            jnp.logical_and(on_ground, jnp.logical_and(~new_support,
                jnp.logical_and(dy == 0, jnp.logical_and(~start_jump,
                    jnp.logical_and(is_climbing == 0, new_is_jumping == 0))))))
        ledge_fall = jnp.logical_or(ledge_fall, native_ledge_fall)
        new_is_falling = jnp.where(ledge_fall, 1, new_is_falling)
        current_vx = jnp.where(ledge_fall, jnp.where(jnp.logical_and(native_ledge_fall,
            state.native_jump_momentum), current_vx, 0), current_vx)
        new_left_x = jnp.clip(new_x, 0, self.consts.WIDTH - 1)
        new_right_x = jnp.clip(new_x + self.consts.PLAYER_WIDTH - 1, 0, self.consts.WIDTH - 1)
        
        front_x = jnp.where(dx > 0, new_right_x, new_left_x)
        
        check_y_top = jnp.clip(new_y, 0, 148)
        check_y_mid = jnp.clip(new_y + self.consts.PLAYER_HEIGHT // 2, 0, 148)
        check_y_bot = jnp.clip(new_y + self.consts.PLAYER_HEIGHT - 1, 0, 148)
        def is_wall(y, x):
            # A horizontal platform is permeable from below.
            # We ignore horizontal wall collisions if we are in the air
            # and there is empty space within 15 pixels above (indicating a horizontal platform/surface).
            # This allows jumping through thick platforms while still being blocked by vertical walls.
            is_perm_platform = jnp.any(room_col_map[jnp.clip(y - jnp.arange(1, 16), 0, 148), x] == 0)
            stair_airborne = jnp.logical_and(jnp.isin(state.room_id, jnp.array([17,23])),
                state.is_jumping == 1)
            ignore = jnp.logical_and(jnp.logical_or(on_ground == 0, stair_airborne), is_perm_platform)
            ignore = jnp.where(jnp.logical_and(state.level == 2,
                jnp.logical_and(state.room_id == 12, on_ground == 0)), y < 40, ignore)
            return jnp.logical_and(room_col_map[y, x] == 1, jnp.logical_not(ignore))
        
        # Upper torch-room pillars test the rendered left edge before walking
        # updates x; the first step from x128 to127 is still unobstructed.
        torch_upper_walk = jnp.logical_and(state.room_id == 12,
            jnp.logical_and(state.player_y == 26,
                jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)))
        wall_front_x = jnp.where(jnp.logical_and(torch_upper_walk, dx < 0),
            state.player_x, front_x)
        hit_wall = jnp.logical_or(
            is_wall(check_y_top, wall_front_x),
            jnp.logical_or(
                is_wall(check_y_mid, wall_front_x),
                is_wall(check_y_bot, wall_front_x)
            )
        )
        
        exit_wall_left = jnp.logical_or(is_wall(check_y_top, 3),
            jnp.logical_or(is_wall(check_y_mid, 3), is_wall(check_y_bot, 3)))
        exit_wall_right = jnp.logical_or(is_wall(check_y_top, 156),
            jnp.logical_or(is_wall(check_y_mid, 156), is_wall(check_y_bot, 156)))
        transition_left = jnp.logical_and(transition_left, jnp.logical_not(exit_wall_left))
        transition_right = jnp.logical_and(transition_right, jnp.logical_not(exit_wall_right))
        new_x = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(exit_wall_right, jnp.logical_and(dx > 0, new_x > 148))), 148, new_x)
        new_x = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(exit_wall_left, jnp.logical_and(dx < 0, new_x < 4))), 4, new_x)

        # 5.5 Item Collection
        overlap_x_item = jnp.logical_and(new_left_x < state.items_x + 6, new_right_x >= state.items_x)
        overlap_y_item = jnp.logical_and(check_y_top < state.items_y + 8, check_y_bot >= state.items_y)
        # Atari reads the previous rendered frame's collision result.
        # The key's transparent outline cannot collide like a solid6x8 box.
        airborne = jnp.logical_or(new_is_jumping == 1, new_is_falling == 1)
        retained_pose = jnp.logical_and(jnp.logical_and(state.level == 1, state.room_id != 12), state.collision_player_x >= 0)
        # Room0's lower-floor kernel retains the preceding pose for item
        # contact. The upper-floor kernels test the current native position.
        retained_item_pose = jnp.logical_or(jnp.logical_and(retained_pose, state.room_id == 3),
            jnp.logical_and(state.level == 2,
                jnp.logical_and(state.room_id == 14, state.collision_player_x >= 0)))
        collision_airborne = jnp.where(retained_item_pose, state.collision_player_pose == 7, airborne)
        collision_dir = jnp.where(retained_item_pose, state.collision_player_dir, new_player_dir)
        player_mask = jax.lax.select(collision_airborne, self._key_player_jump_mask, self._key_player_stand_mask)
        flip_player = jnp.where(collision_airborne, collision_dir == 1, collision_dir == -1)
        player_mask = jax.lax.cond(flip_player, lambda: jnp.flip(player_mask, axis=1), lambda: player_mask)
        contact_x = jnp.where(retained_pose, state.collision_player_x, state.player_x)
        contact_y = jnp.where(retained_pose, state.collision_player_y, state.player_y)
        contact_y = contact_y + jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(state.is_climbing == 1, state.last_ladder >= 0)), 2, 0)
        item_contact_x = jnp.where(retained_item_pose, state.collision_player_x, state.player_x)
        item_contact_y = jnp.where(retained_item_pose, state.collision_player_y, state.player_y)
        py, px = jnp.indices(player_mask.shape)
        key_px = item_contact_x + px[None, :, :] - state.items_x[:, None, None]
        key_py = item_contact_y + py[None, :, :] - state.items_y[:, None, None]
        inside_key = jnp.logical_and(jnp.logical_and(key_px >= 0, key_px < self._item_collision_masks.shape[2]),
                                     jnp.logical_and(key_py >= 0, key_py < self._item_collision_masks.shape[1]))
        key_pixels = self._item_collision_masks[state.items_type[:, None, None],
            jnp.clip(key_py, 0, self._item_collision_masks.shape[1] - 1),
            jnp.clip(key_px, 0, self._item_collision_masks.shape[2] - 1)]
        key_overlap = jnp.any(jnp.logical_and(player_mask[None, :, :], jnp.logical_and(inside_key, key_pixels)), axis=(1, 2))
        start_room_key = jnp.logical_and(state.room_id == 4, state.items_type == 0)
        item_overlap = key_overlap
        pending_pickup = jnp.arange(self.consts.MAX_ITEMS_PER_ROOM) == state.pickup_item_idx
        collect_item_mask = jnp.logical_and(is_active, jnp.logical_and(
            jnp.logical_and(state.items_active == 1, jnp.logical_not(pending_pickup)), item_overlap))
        # ROM RAM52 pickup bit survives bonus respawns until the jump ends.
        bonus_pose_reset = jnp.logical_or(hit_floor,
            jnp.logical_and(state.is_jumping == 1, new_is_jumping == 0))
        bonus_pickup_blocked = jnp.logical_and(is_bonus_room,
            jnp.logical_and(state.bonus_pickup_latched, jnp.logical_not(bonus_pose_reset)))
        collect_item_mask = jnp.logical_and(collect_item_mask, jnp.logical_not(bonus_pickup_blocked))
        first_key_pickup = jnp.any(jnp.logical_and(start_room_key, collect_item_mask))
        any_item_pickup = jnp.any(collect_item_mask)

        # Measured ALE rewards: key100, ruby1000, amulet200, sword100.
        # Native type6 adds BCD29 plus carry, i.e.3000 points for a torch.
        item_scores = jnp.array([100, 1000, 200, 100, 3000], dtype=jnp.int32)[state.items_type]
        new_score = state.score + jnp.sum(jnp.where(collect_item_mask, item_scores, 0))
        # ALE keeps collected items visible through the pickup hold;
        # remove it on the first resumed movement frame, without scoring twice.
        new_items_active = jnp.where(pending_pickup, 0, state.items_active)

        # Inventory updates
        keys_collected = jnp.sum(jnp.where(jnp.logical_and(collect_item_mask, state.items_type == 0), 1, 0))
        swords_collected = jnp.sum(jnp.where(jnp.logical_and(collect_item_mask, state.items_type == 3), 1, 0))
        torch_collected = jnp.any(jnp.logical_and(collect_item_mask, state.items_type == 4))
        amulet_collected = jnp.any(jnp.logical_and(collect_item_mask, state.items_type == 2))

        current_inventory = state.inventory
        current_inventory = current_inventory.at[0].add(keys_collected)
        current_inventory = current_inventory.at[1].set(jnp.minimum(current_inventory[1] + swords_collected, 3))
        current_inventory = current_inventory.at[2].set(jnp.where(torch_collected, 1, current_inventory[2]))
        
        # Handle amulet time reset and disappearance
        amulet_duration = self.consts.AMULET_DURATION - jnp.where(self.consts.CUSTOM_ROOMS, 0,
            jnp.mod(state.native_frame_count + 1 + 67, 8))
        new_amulet_time = jnp.where(amulet_collected, amulet_duration, new_amulet_time)
        current_inventory = current_inventory.at[3].set(jnp.where(new_amulet_time > 0, 1, 0))

        # Door checks
        measured_start_door = state.room_id == 4
        door_contact_x = jnp.where(jnp.logical_or(measured_start_door, state.room_id == 26),
            jnp.where(dx > 0, state.player_x + 7, state.player_x), front_x)
        door_collision_x = state.doors_x + jnp.where(measured_start_door,
            jnp.where(state.doors_x < 80, 4, -4), 0)
        in_x_door = jnp.logical_and(door_contact_x >= door_collision_x, door_contact_x < door_collision_x + 4)
        in_y_door_top = jnp.logical_and(check_y_top >= state.doors_y, check_y_top < state.doors_y + 38)
        in_y_door_mid = jnp.logical_and(check_y_mid >= state.doors_y, check_y_mid < state.doors_y + 38)
        in_y_door_bot = jnp.logical_and(check_y_bot >= state.doors_y, check_y_bot < state.doors_y + 38)
        in_y_door = jnp.logical_or(in_y_door_top, jnp.logical_or(in_y_door_mid, in_y_door_bot))
        # B228 calls B6F6 before either horizontal jump direction. A barrier
        # touching the rendered sprite blocks that tick even when jumping away.
        door_px = state.player_x + px[None, :, :] - door_collision_x[:, None, None]
        door_py = state.player_y + py[None, :, :] - state.doors_y[:, None, None]
        jump_door_contact = jnp.any(jnp.logical_and(player_mask[None, :, :],
            jnp.logical_and(jnp.logical_and(door_px >= 0, door_px < 4),
                jnp.logical_and(door_py >= 0, door_py < 37))), axis=(1, 2))
        jump_door_contact = jnp.logical_and(jump_door_contact,
            jnp.logical_and(not self.consts.CUSTOM_ROOMS, state.is_jumping == 1))
        hit_door_mask = jnp.logical_and(state.doors_active == 1,
            jnp.logical_or(jnp.logical_and(in_x_door, in_y_door), jump_door_contact))
        
        hit_door_order = jnp.cumsum(hit_door_mask.astype(jnp.int32))
        open_door_mask = jnp.logical_and(hit_door_mask, hit_door_order <= current_inventory[0])
        
        num_opened = jnp.sum(open_door_mask.astype(jnp.int32))
        current_inventory = current_inventory.at[0].add(-num_opened)
        new_doors_active = jnp.where(open_door_mask, 0, state.doors_active)
        new_score = new_score + num_opened * 300
        
        hit_door_as_wall = jnp.logical_and(hit_door_mask, jnp.logical_not(open_door_mask))
        hit_wall = jnp.logical_or(hit_wall, jnp.any(hit_door_as_wall))
        # The torch-room upper pillars also feed B6F6's rendered contact.
        # Probe the old sprite: its leading arm reaches the left pillar before
        # the bounding-box wall probe, while the opposite approach enters x127.
        torch_px, torch_py = state.player_x + px, state.player_y + py
        torch_pillars = jnp.logical_and(torch_py >= 6, torch_py < 46)
        torch_pillars = jnp.logical_and(torch_pillars,
            jnp.logical_or(jnp.logical_and(torch_px >= 32, torch_px < 36),
                jnp.logical_and(torch_px >= 124, torch_px < 128)))
        torch_wall_contact = jnp.logical_and(state.room_id == 12,
            jnp.logical_and(new_y <= 26, jnp.logical_and(state.is_jumping == 1,
                jnp.any(jnp.logical_and(player_mask, torch_pillars)))))
        hit_wall = jnp.logical_or(hit_wall,
            jnp.logical_and(not self.consts.CUSTOM_ROOMS, torch_wall_contact))

        # Bonus gems respawn after pickup cleanup using the free-running ROM
        # byte. Rejected positions are retried on the next active tick.
        is_bonus_gem_collected = jnp.logical_and(is_bonus_room, collect_item_mask[0])
        bonus_spawn_attempt = jnp.logical_and(is_bonus_room,
            jnp.logical_and(state.bonus_room_timer < 632,
                jnp.logical_and(new_items_active[0] == 0, is_active)))
        bonus_spawn = jnp.logical_and(bonus_spawn_attempt,
            jnp.logical_and(state.native_rng_a >= 16, state.native_rng_a < 136))
        new_items_x = state.items_x.at[0].set(jnp.where(bonus_spawn,
            state.native_rng_a - 1, state.items_x[0]))
        new_items_active = new_items_active.at[0].set(jnp.where(bonus_spawn, 1, new_items_active[0]))
        new_first_gem_pickup = jnp.where(is_bonus_gem_collected, 1, state.first_gem_pickup)
        new_bonus_room_timer = state.bonus_room_timer
        new_first_gem_pickup = jax.lax.select(is_bonus_room, new_first_gem_pickup, 0)
        new_rng_key = state.key

        new_x = jnp.where(jnp.logical_or(hit_wall, is_climbing == 1), current_x, new_x)
        
        new_mid_x = new_x + self.consts.PLAYER_WIDTH // 2
        new_feet_y_after = new_y + self.consts.PLAYER_HEIGHT - 1
        
        # Vectorized conveyor physics
        is_on_conveyor_physics = jnp.logical_and(
            state.conveyors_active == 1,
            jnp.logical_and(jnp.logical_or(new_feet_y_after == state.conveyors_y - 1,
                jnp.logical_and(jnp.logical_and(state.level < 2,
                    jnp.logical_and(state.is_jumping == 1, dy > 0)),
                    jnp.logical_and(new_feet_y_after >= state.conveyors_y - 4,
                                    new_feet_y_after <= state.conveyors_y - 1))), jnp.logical_and(new_mid_x >= state.conveyors_x - 2, new_mid_x < state.conveyors_x + 42))
        )
        # The measured start-room belt phase differs from the other rooms.
        conveyor_phase = jnp.where(state.level > 0, state.native_frame_count + 1,
            state.frame_count + jnp.where(state.room_id == 4, 1, 0))
        conveyor_velocities = jnp.mod(conveyor_phase, 2) * state.conveyors_direction
        total_conveyor_velocity = jnp.sum(jnp.where(jnp.logical_and(is_on_conveyor_physics, is_climbing == 0), conveyor_velocities, 0))
        new_x = new_x + total_conveyor_velocity
        new_x = jnp.clip(new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)

        # DAE7 tests the upper stair-room gaps even during a rope jump.
        # Crossing that anchor prepares mode7 and suspends the jump timer.
        gap_left = jnp.where(state.room_id == 17, 45, 37)
        gap_right = jnp.where(state.room_id == 17, 81, 89)
        stair_jump_gap = jnp.logical_and(jnp.isin(state.room_id, jnp.array([17,23])),
            jnp.logical_and(new_is_jumping == 1, jnp.logical_and(~start_jump,
                jnp.logical_and(new_y == 26,
                    jnp.logical_or(jnp.logical_and(new_x >= gap_left, new_x < gap_left + 26),
                        jnp.logical_and(new_x >= gap_right, new_x < gap_right + 26))))))
        new_is_jumping = jnp.where(stair_jump_gap, 0, new_is_jumping)
        new_is_falling = jnp.where(stair_jump_gap, 1, new_is_falling)
        
        current_vx = jnp.where(jnp.logical_or(jnp.logical_and(hit_wall,
            jnp.logical_not(jnp.logical_and(not self.consts.CUSTOM_ROOMS,
                jnp.logical_or(start_jump, state.is_jumping == 1)))), hit_floor), 0, current_vx)
        
        # 6. Enemy Movement
        # Native enemy cadence doubles with each of the three temple layouts.
        speed_enemy = jnp.where(jnp.logical_and(is_active, jnp.mod(state.frame_count, 4) == 0), 1, 0)
        # Room5 initialization occupies the first frame after a room change.
        # ALE's free-running clock ticks when the *previous* native frame is 0 mod4.
        measured_room5 = jnp.logical_and(jnp.logical_or(jnp.logical_and(state.level > 0, jnp.any(state.enemies_bouncing)), jnp.isin(state.room_id, jnp.array([5, 10]))), not self.consts.CUSTOM_ROOMS)
        room5_tick = jnp.logical_and(is_active, jnp.logical_and(
            jnp.mod(state.native_frame_count, jnp.right_shift(jnp.int32(4), state.level)) == 0,
            jnp.logical_not(state.room5_skull_spawn_pending)))
        # Rolling skulls and spiders share the free-running enemy clock.
        measured_quarter_clock = jnp.logical_or(measured_room5, jnp.logical_or(jnp.isin(state.room_id, jnp.array([4,12])), jnp.any(state.enemies_type == 3)))
        measured_quarter_clock = jnp.logical_or(measured_quarter_clock,
            jnp.logical_and(state.level == 2, state.room_id == 27))
        speed_enemy = jnp.where(jnp.logical_and(measured_quarter_clock, not self.consts.CUSTOM_ROOMS), room5_tick.astype(jnp.int32), speed_enemy)
        new_room5_phase = jnp.where(jnp.logical_and(measured_room5, room5_tick),
            jnp.mod(state.room5_skull_phase + 1, 30), state.room5_skull_phase)
        raw_new_enemies_x = state.enemies_x + state.enemies_direction * speed_enemy
        
        # Bounce off walls
        hit_wall_left_enemy = jnp.logical_and(state.enemies_direction < 0, room_col_map[state.enemies_y + 8, jnp.clip(raw_new_enemies_x, 0, self.consts.WIDTH - 1)] == 1)
        hit_wall_right_enemy = jnp.logical_and(state.enemies_direction > 0, room_col_map[state.enemies_y + 8, jnp.clip(raw_new_enemies_x + 8, 0, self.consts.WIDTH - 1)] == 1)
        
        hit_left_enemy = jnp.logical_and(state.enemies_direction < 0, raw_new_enemies_x <= state.enemies_min_x)
        hit_right_enemy = jnp.logical_and(state.enemies_direction > 0, raw_new_enemies_x >= state.enemies_max_x)
        
        bounce_enemy = jnp.logical_and(is_active, jnp.logical_or(hit_left_enemy, jnp.logical_or(hit_right_enemy, jnp.logical_or(hit_wall_left_enemy, hit_wall_right_enemy))))
        
        # Synchronize bouncing: if ANY active enemy in the room hits a wall, ALL of them flip direction
        bounce_any = jnp.any(jnp.logical_and(state.enemies_active == 1, bounce_enemy))
        new_enemies_direction = jnp.where(bounce_any, -state.enemies_direction, state.enemies_direction)
        new_enemies_x = jnp.where(bounce_any, state.enemies_x, raw_new_enemies_x)
        native_room_index = jnp.argmax(ALE_ROOM_IDS == state.room_id)
        history = state.native_room_history[native_room_index]
        rolling_room = jnp.isin(state.room_id, jnp.array([4,12,27]))
        rolling_live = jnp.logical_and(is_active,
            jnp.logical_and(jnp.isin(state.native_mode, jnp.array([4,7])), (history & 2) != 0))
        rolling_move = jnp.logical_and(rolling_live,
            ((state.native_frame_count + 68) & (jnp.right_shift(4, state.level) - 1)) == 0)
        rolling_bound = jnp.where(state.native_rolling_direction < 0,
            jnp.where(native_room_index < 6, 22, 34), jnp.where(native_room_index < 6, 72, 122))
        rolling_turn = jnp.logical_and(rolling_move, state.native_rolling_x == rolling_bound)
        rolling_direction = jnp.where(rolling_turn, -state.native_rolling_direction, state.native_rolling_direction)
        rolling_x = state.native_rolling_x + jnp.where(jnp.logical_and(rolling_move, ~rolling_turn), state.native_rolling_direction, 0)
        rolling_phase = jnp.where(rolling_turn, state.rolling_skull_phase ^ 15, state.rolling_skull_phase)
        rotation_tick = jnp.logical_and(rolling_live,
            ((state.native_frame_count + 68) & (jnp.right_shift(8, state.level) - 1)) == 0)
        rolling_phase = (rolling_phase + rotation_tick.astype(jnp.int32)) & 15
        native_rolling_enemy = jnp.logical_and(state.level == 2, rolling_room)
        # RAM47 is a kernel-local position: the barrier chambers add their
        # horizontal origin, whereas the pit kernel uses the screen origin.
        rolling_screen_x = rolling_x + jnp.where(state.room_id == 27, -1, 32)
        new_enemies_x = new_enemies_x.at[0].set(jnp.where(native_rolling_enemy, rolling_screen_x, new_enemies_x[0]))
        new_enemies_direction = new_enemies_direction.at[0].set(jnp.where(native_rolling_enemy,
            rolling_direction, new_enemies_direction[0]))
        
        # 7. Dying Mechanism (Fall Damage & Enemy Collision)
        fall_stopped = jnp.logical_and(state.is_falling == 1, new_is_falling == 0)
        # B65B kills on landing after eight falling ticks. The final four
        # pixels of stair anchor conversion are excluded from that count.
        native_stair_fall = jnp.logical_and(not self.consts.CUSTOM_ROOMS,
            state.room_id == 17)
        died_from_fall = jnp.logical_and(
            jnp.logical_and(fall_stopped, is_climbing == 0),
            jnp.where(native_stair_fall, new_fall_distance >= self.consts.MAX_FALL_DISTANCE - 4,
                new_fall_distance > self.consts.MAX_FALL_DISTANCE)
        )

        new_fall_distance = jnp.where(
            dy > 0,
            jnp.where(is_climbing == 0, new_fall_distance + dy, 0),
            0
        )
        # The jump timer's final descent is not a B6A5 falling tick.
        new_fall_distance = jnp.where(jnp.logical_and(native_stair_fall,
            jnp.logical_and(state.is_jumping == 1, new_is_jumping == 0)), 0, new_fall_distance)
        
        # Vectorized enemy collision
        e_bounce_offset = enemy_bounce_offsets(state, self.consts)
        e_y_col = state.enemies_y - e_bounce_offset
        overlap_x_enemy = jnp.logical_and(new_left_x < new_enemies_x + 7, new_right_x >= new_enemies_x + 1)
        overlap_y_enemy = jnp.logical_and(check_y_top < e_y_col + 15, check_y_bot >= e_y_col + 1)
        # Native room20 snake jumps clear the transparent sprite outlines.
        # TIA reports contact from the previous rendered player/enemy pixels.
        walking = jnp.logical_and(jnp.where(state.level == 2, state.native_walk_pose, state.player_vx != 0), jnp.logical_and(state.is_climbing == 0,
            jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)))
        airborne = jnp.logical_or(jnp.logical_and(state.is_jumping == 1, state.jump_counter > 0),
            jnp.logical_and(state.is_falling == 1, jnp.logical_or(state.level < 2, state.fall_after_jump == 1)))
        landed_last_tick = jnp.logical_and(jnp.logical_or(state.room_id == 18, state.level == 2),
            jnp.logical_and(state.jump_counter >= self.consts.JUMP_Y_OFFSETS.shape[0],
                jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)))
        # ROM draw pointers retain the jump sprite for the landing tick.
        airborne = jnp.logical_or(airborne, landed_last_tick)
        player_idx = jnp.where(walking, 1 + ((state.native_frame_count + jnp.where(state.level == 2, 67, 68)) // 4) % 2, 0)
        # Preserve the previously matching room20 snake contact while the
        # newly measured walking phase/bitmap is used for room10 skulls.
        # This separate bitmap/phase also reproduces the ALE death at3745.
        player_idx = jnp.where(jnp.logical_and(state.room_id == 20, walking),
            1 + (state.frame_count // 4) % 2, player_idx)
        player_idx = jnp.where(state.is_climbing == 1, 3 + (state.player_y // 4) % 2, player_idx)
        # ROM 9243 selects a fixed hanging pose; moving climbs alternate on
        # RAM0 bit2, independently of height. REFP0 keeps the last facing bit.
        climb_pose = jnp.where(state.last_rope >= 0, 5, 3) + jnp.where(
            state.native_climb_moving, ((state.native_frame_count + 68) // 4) % 2, 0)
        player_idx = jnp.where(jnp.logical_and(state.level == 2, state.is_climbing == 1),
            climb_pose, player_idx)
        player_idx = jnp.where(airborne, 7, player_idx)
        player_idx = jnp.where(retained_pose, state.collision_player_pose, player_idx)
        player_pixels = self._player_collision_masks[player_idx]
        player_pixels = jnp.where(jnp.logical_and(jnp.logical_or(state.room_id == 20, state.level == 2), player_idx == 2),
            self._original_walk_1_collision_mask, player_pixels)
        flip_pixels = jnp.where(jnp.logical_or(walking, airborne), state.player_dir == 1, state.player_dir == -1)
        retained_direction = jnp.where(retained_pose, state.collision_player_dir, state.player_dir)
        flip_pixels = jnp.where(jnp.logical_and(retained_pose, jnp.isin(player_idx, jnp.array([1,2,7]))), retained_direction == 1,
            jnp.where(retained_pose, retained_direction == -1, flip_pixels))
        player_pixels = jax.lax.cond(flip_pixels, lambda: jnp.flip(player_pixels, axis=1), lambda: player_pixels)
        enemy_animation = jnp.where(state.level == 2,
            (state.native_frame_count + 66) // 8, state.frame_count // 7) % 2
        snake_pixels = self._snake_collision_masks[enemy_animation]
        snake_pixels = jnp.where((state.enemies_direction == -1)[:, None, None],
            jnp.flip(snake_pixels, axis=1)[None, :, :], snake_pixels[None, :, :])
        sy, sx = jnp.indices(player_pixels.shape)
        snake_x = contact_x + sx[None, :, :] - state.enemies_x[:, None, None]
        snake_y = contact_y + sy[None, :, :] - state.enemies_y[:, None, None]
        inside_snake = jnp.logical_and(jnp.logical_and(snake_x >= 0, snake_x < snake_pixels.shape[2]),
            jnp.logical_and(snake_y >= 0, snake_y < snake_pixels.shape[1]))
        snake_contact_pixels = snake_pixels[jnp.arange(self.consts.MAX_ENEMIES_PER_ROOM)[:, None, None],
            jnp.clip(snake_y, 0, snake_pixels.shape[1]-1), jnp.clip(snake_x, 0, snake_pixels.shape[2]-1)]
        snake_contact = jnp.any(jnp.logical_and(player_pixels[None, :, :],
            jnp.logical_and(inside_snake, snake_contact_pixels)), axis=(1, 2))
        spider_pixels = jax.lax.cond(enemy_animation == 1,
            lambda: jnp.flip(self._spider_collision_mask, axis=1), lambda: self._spider_collision_mask)
        spider_contact_x = jnp.where(state.level == 2, new_enemies_x, state.enemies_x)
        spider_x = contact_x + sx[None, :, :] - spider_contact_x[:, None, None]
        spider_y = contact_y + sy[None, :, :] - state.enemies_y[:, None, None]
        inside_spider = jnp.logical_and(jnp.logical_and(spider_x >= 0, spider_x < spider_pixels.shape[1]),
            jnp.logical_and(spider_y >= 0, spider_y < spider_pixels.shape[0]))
        spider_contact = jnp.any(jnp.logical_and(player_pixels[None, :, :],
            jnp.logical_and(inside_spider, spider_pixels[jnp.clip(spider_y, 0, spider_pixels.shape[0]-1),
                jnp.clip(spider_x, 0, spider_pixels.shape[1]-1)])), axis=(1, 2))
        skull_pixels = self._skull_collision_masks[jnp.where(
            jnp.logical_and(state.level == 2, rolling_room), rolling_phase, 0)]
        # Room0's bouncing-object kernel omits the bitmap's final row.
        # Keeping that extra row creates a false hit on a hanging player.
        skull_pixels = jnp.where(jnp.logical_and(state.level == 2, state.room_id == 3),
            skull_pixels.at[-1, :].set(False), skull_pixels)
        skull_pixels = jax.lax.cond(jnp.logical_and(native_rolling_enemy, rolling_direction < 0),
            lambda: jnp.flip(skull_pixels, axis=1), lambda: skull_pixels)
        next_bounce = enemy_bounce_offsets(state.replace(room5_skull_phase=new_room5_phase), self.consts)
        skull_x = contact_x + sx[None, :, :] - new_enemies_x[:, None, None]
        skull_y = contact_y + sy[None, :, :] - (state.enemies_y - next_bounce)[:, None, None]
        inside_skull = jnp.logical_and(jnp.logical_and(skull_x >= 0, skull_x < skull_pixels.shape[1]),
            jnp.logical_and(skull_y >= 0, skull_y < skull_pixels.shape[0]))
        skull_contact = jnp.any(jnp.logical_and(player_pixels[None, :, :],
            jnp.logical_and(inside_skull, skull_pixels[jnp.clip(skull_y, 0, skull_pixels.shape[0]-1),
                jnp.clip(skull_x, 0, skull_pixels.shape[1]-1)])), axis=(1, 2))
        enemy_contact = jnp.where(jnp.logical_and(jnp.logical_or(state.level > 0, jnp.isin(state.room_id, jnp.array([18, 20]))), state.enemies_type == 4),
            snake_contact, jnp.logical_and(overlap_x_enemy, overlap_y_enemy))
        enemy_contact = jnp.where(jnp.logical_and(state.room_id == 10, state.enemies_type == 1),
            skull_contact, enemy_contact)
        enemy_contact = jnp.where(jnp.logical_and(state.level > 0, state.enemies_type == 3), spider_contact, enemy_contact)
        enemy_contact = jnp.where(jnp.logical_and(state.level > 0, state.enemies_bouncing == 1), skull_contact, enemy_contact)
        enemy_contact = jnp.where(jnp.logical_and(state.room_id == 12, state.enemies_type == 1), skull_contact, enemy_contact)
        # The rolling-object kernel tests both updated positions.
        rolling_dx = new_x + sx - new_enemies_x[0]
        # The pit's rolling-object kernel draws the player one scanline lower.
        rolling_dy = new_y + (state.room_id == 27).astype(jnp.int32) + sy - state.enemies_y[0]
        rolling_inside = jnp.logical_and(jnp.logical_and(rolling_dx >= 0, rolling_dx < skull_pixels.shape[1]),
            jnp.logical_and(rolling_dy >= 0, rolling_dy < skull_pixels.shape[0]))
        rolling_contact = jnp.any(jnp.logical_and(player_pixels,
            jnp.logical_and(rolling_inside, skull_pixels[jnp.clip(rolling_dy, 0, skull_pixels.shape[0]-1),
                jnp.clip(rolling_dx, 0, skull_pixels.shape[1]-1)])))
        enemy_contact = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(rolling_room, jnp.arange(self.consts.MAX_ENEMIES_PER_ROOM) == 0)),
            rolling_contact, enemy_contact)
        this_hit_enemy = jnp.logical_and(state.enemies_active == 1, enemy_contact)
        
        # Neutralize enemy collision if amulet is active
        this_hit_enemy = jnp.logical_and(is_active, jnp.logical_and(this_hit_enemy, jnp.logical_not(is_amulet_active)))
        # Mode7 falls bypass the standing handler's TIA enemy contact read.
        this_hit_enemy = jnp.logical_and(this_hit_enemy,
            jnp.logical_or(state.level < 2, state.native_mode == 4))

        has_sword = current_inventory[1] > 0
        # Native swords kill skulls and spiders; snakes remain lethal.
        sword_vulnerable = jnp.logical_or(state.enemies_type == 1, state.enemies_type == 3)
        kill_enemy_mask = jnp.logical_and(this_hit_enemy, jnp.logical_and(has_sword, sword_vulnerable))
        kill_order = jnp.cumsum(kill_enemy_mask.astype(jnp.int32))
        actually_killed_mask = jnp.logical_and(kill_enemy_mask, kill_order <= current_inventory[1])
        
        died_from_enemy = jnp.any(jnp.logical_and(this_hit_enemy, jnp.logical_not(actually_killed_mask)))
        # Sword kills consume the enemy; fatal-contact cleanup is room-specific.
        new_enemies_active = jnp.where(actually_killed_mask, 0, state.enemies_active)
        # The rolling skull's room bit is cleared on fatal contact as well.
        new_enemies_active = jnp.where(jnp.logical_and(jnp.logical_or(state.room_id == 4,
            jnp.logical_and(state.level == 2, rolling_room)),
            jnp.logical_and(died_from_enemy, this_hit_enemy)), 0, new_enemies_active)
        # Room20 snakes stay visible throughout death and respawn, then
        # the touched snake disappears on the first resumed native frame.
        # This is death cleanup, with no sword use or kill reward.
        cleanup_enemy = jnp.logical_and(state.room_id == 20, jnp.logical_and(
            state.death_timer == 0, state.respawn_enemy_idx >= 0))
        cleanup_mask = jnp.logical_and(cleanup_enemy,
            jnp.arange(self.consts.MAX_ENEMIES_PER_ROOM) == state.respawn_enemy_idx)
        new_enemies_active = jnp.where(cleanup_mask, 0, new_enemies_active)
        swords_used = jnp.sum(actually_killed_mask.astype(jnp.int32))
        current_inventory = current_inventory.at[1].add(-swords_used)
        # Keep the explicit uniform-reward override used by reward mods.
        if self.consts.KILL_ENEMY_REWARD is None:
            enemy_scores = jnp.where(state.enemies_type == 3,
                                     self.consts.KILL_SPIDER_REWARD,
                                     self.consts.KILL_SKULL_REWARD)
        else:
            enemy_scores = self.consts.KILL_ENEMY_REWARD
        new_score = new_score + jnp.sum(jnp.where(actually_killed_mask, enemy_scores, 0))

        # Laser Collision
        # Update laser and platform cycles
        new_laser_cycle = jnp.where(is_active, jnp.mod(state.laser_cycle + 1, 128), state.laser_cycle)
        new_platform_cycle = jnp.where(is_active, jnp.mod(state.platform_cycle + 1, self.consts.PLATFORM_CYCLE_LENGTH), state.platform_cycle)
        laser_active_now = jnp.logical_and(jnp.greater_equal(state.laser_cycle, 0), jnp.less(state.laser_cycle, 92))
        platform_active_now = jnp.less(state.platform_cycle, self.consts.PLATFORM_ACTIVE_DURATION)

        l_active_las = jnp.logical_and(state.lasers_active == 1, laser_active_now)
        overlap_x_las = jnp.logical_and(new_left_x < state.lasers_x + 4, new_right_x >= state.lasers_x)
        overlap_y_las = jnp.logical_and(check_y_top < 46, check_y_bot >= 7)
        died_from_laser = jnp.logical_and(~is_amulet_active,
            jnp.any(jnp.logical_and(l_active_las, jnp.logical_and(overlap_x_las, overlap_y_las))))
        # The beam's native contact span is three pixels. Room metadata
        # restores visible beams after a death even during the dark phase.
        laser_pose = jnp.where(walking,
            1 + ((state.native_frame_count + 68) // 4) % 2, player_idx)
        laser_player = jnp.where(laser_pose == 2,
            self._original_walk_1_collision_mask, self._player_collision_masks[laser_pose])
        laser_flip = jnp.where(jnp.isin(laser_pose, jnp.array([1,2,7])),
            state.player_dir == 1, state.player_dir == -1)
        laser_player = jax.lax.cond(laser_flip, lambda: jnp.flip(laser_player, axis=1), lambda: laser_player)
        laser_pixel_x = state.player_x + sx[None, :, :]
        laser_pixel_y = state.player_y + sy[None, :, :]
        native_lasers = state.lasers_active == 1
        pixel_laser_contact = jnp.any(jnp.logical_and(laser_player[None, :, :],
            jnp.logical_and(native_lasers[:, None, None],
                jnp.logical_and(laser_pixel_x >= state.lasers_x[:, None, None],
                    jnp.logical_and(laser_pixel_x < state.lasers_x[:, None, None] + 3,
                        jnp.logical_and(laser_pixel_y >= 7, laser_pixel_y < 47))))))
        died_from_laser = jnp.where(state.level == 2,
            jnp.logical_and(~is_amulet_active, pixel_laser_contact), died_from_laser)
        
        is_pit_room = jnp.any(state.room_id == jnp.array([19, 27, 29, 31]))
        died_from_pit = jnp.logical_and(is_pit_room,
            jnp.logical_and(state.is_falling == 1, state.player_y + 47 > 100))

        player_died = jnp.logical_or(died_from_fall, jnp.logical_or(died_from_enemy, jnp.logical_or(died_from_laser, died_from_pit)))
        
        start_death = jnp.logical_and(state.death_timer == 0, player_died)
        # ALE resumes after at least50 ticks, on a fixed four-frame cadence.
        death_wait = self.consts.DEATH_TIMER_FRAMES + jnp.mod(-(state.native_frame_count + 1), 4)
        new_death_timer = jnp.where(start_death, death_wait,
                                    jnp.where(state.death_timer > 0, state.death_timer - 1, 0))
        
        death_type = jnp.where(died_from_fall, 1, jnp.where(died_from_enemy, 2, jnp.where(died_from_laser, 3, jnp.where(died_from_pit, 1, 0))))
        new_death_type = jnp.where(start_death, death_type, jnp.where(new_death_timer == 0, 0, state.death_type))

        final_death = jnp.logical_and(jnp.logical_or(state.game_over_pending,
            jnp.logical_and(start_death, state.lives == 0)), not self.consts.CUSTOM_ROOMS)
        respawn_now = jnp.logical_and(jnp.logical_and(state.death_timer == 1, new_death_timer == 0), ~final_death)
        
        spawn_x = state.entry_x
        spawn_y = state.entry_y
        
        extra_lives = jnp.sum(new_score // 10000 - state.score // 10000)
        # ALE ends the last life after the death hold, without respawning.
        lose_life = jnp.logical_or(jnp.logical_and(start_death, ~final_death),
            jnp.logical_and(final_death, state.death_timer == 1))
        new_lives = jnp.where(lose_life, state.lives - 1, state.lives) + extra_lives
        final_x = jnp.where(respawn_now, spawn_x, jnp.where(state.death_timer > 0, state.player_x, new_x))
        final_x = jnp.where(jnp.logical_and(not self.consts.CUSTOM_ROOMS,
            jnp.logical_and(start_death, jnp.logical_or(died_from_fall,
                jnp.logical_and(state.level == 2, died_from_pit)))), state.player_x, final_x)
        final_y = jnp.where(respawn_now, spawn_y, jnp.where(state.death_timer > 0, state.player_y, new_y))
        final_y = jnp.where(jnp.logical_and(start_death, died_from_pit), 49, final_y)
        final_vx = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, current_vx)
        final_vy = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, dy)
        final_player_dir = jnp.where(respawn_now, jnp.where(state.level == 2, -1, 1), jnp.where(new_death_timer > 0, state.player_dir, new_player_dir))
        final_is_jumping = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_is_jumping)
        final_is_falling = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_is_falling)
        final_is_climbing = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), jnp.where(respawn_now, state.entry_is_climbing, 0), is_climbing)
        final_last_ladder = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), jnp.where(respawn_now, state.entry_last_ladder, -1), new_last_ladder)
        final_jump_counter = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_jump_counter)
        final_fall_distance = jnp.where(respawn_now, 0, jnp.where(new_death_timer > 0, state.fall_distance, new_fall_distance))
        final_prev_is_fire = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), jnp.where(respawn_now, False, state.prev_is_fire), is_fire)
        # The ROM samples the button latch in standing/rope dispatch only.
        # Jumping, falling and death retain it, including across respawn.
        sample_fire = jnp.logical_and(is_active, jnp.logical_and(state.native_mode == 4,
            jnp.logical_and(state.is_jumping == 0, jnp.logical_and(state.is_falling == 0,
                jnp.logical_or(state.is_climbing == 0, state.last_rope >= 0)))))
        final_prev_is_fire = jnp.where(state.level == 2,
            jnp.where(jnp.logical_or(sample_fire, state.native_mode == 5), is_fire, state.prev_is_fire), final_prev_is_fire)
        # The ladder-exit dispatch has not handled FIRE yet.
        final_prev_is_fire = jnp.where(abort_ladder, False, final_prev_is_fire)

        # if jumped before
        # fall_after_jump -> (prev_jumped, not anymore , or prev fall_after_jump) and falling
        fall_after_jump = jnp.where(jnp.logical_or(state.fall_after_jump == 1, jnp.logical_and(state.is_jumping == 1, new_is_jumping == 0)), 1, 0)
        fall_after_jump = jnp.where(jnp.logical_and(native_ledge_fall, state.native_jump_momentum), 1, fall_after_jump)
        fall_after_jump = jnp.where(retained_drift, 1, fall_after_jump)
        fall_after_jump = jnp.where(final_is_falling == 1, fall_after_jump, 0)
        fall_after_jump = jnp.where(is_bonus_room, 0, fall_after_jump)
        native_jump_momentum = jnp.logical_or(state.native_jump_momentum,
            jnp.logical_and(start_jump, jnp.logical_or(is_left, is_right)))
        # FCA4 clears bit2 on a rope catch; ladder entry preserves it.
        native_jump_momentum = jnp.where(jnp.logical_or(respawn_now,
            jnp.logical_and(is_climbing_rope, jnp.logical_and(~is_jumping_off_rope,
                ~jnp.logical_and(state.room_id == 23, jnp.any(top_entry_rope))))), False, native_jump_momentum)
        native_jump_momentum = jnp.where(jnp.logical_or(is_active, respawn_now),
            native_jump_momentum, state.native_jump_momentum)

        transition_any = jnp.logical_or(jnp.logical_or(transition_left, transition_right), jnp.logical_or(transition_down, transition_up))
        carry = (state.room_id >= 24).astype(jnp.int32)
        carry = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(jnp.any(hit_door_as_wall),
                jnp.logical_and(state.player_vx != 0,
                    jnp.logical_and(state.is_jumping == 1, ~last_jump_frame)))), 0, carry)
        carry = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(state.room_id == 12, jnp.logical_and(hit_wall,
                jnp.logical_and(state.player_vx != 0,
                    jnp.logical_and(~last_jump_frame, jnp.logical_and(state.is_jumping == 1,
                        jnp.logical_or(dy > 0, torch_wall_contact))))))), 1, carry)
        carry = jnp.where(ledge_fall, jnp.where(native_ledge_fall,
            jnp.logical_or(pit, state.room_id == 4).astype(jnp.int32), 1), carry)
        carry = jnp.where(jnp.logical_or(native_support_loss, dropout_collapse),
            jnp.logical_or(pit, state.room_id == 4).astype(jnp.int32), carry)
        carry = jnp.where(bonus_collapse, 1, carry)
        carry = jnp.where(jnp.logical_and(is_bonus_room, bonus_spawn_attempt), state.native_rng_a >= 136, carry)
        rope_path = jnp.logical_and(state.last_rope >= 0,
            jnp.logical_and(state.is_climbing == 1, jnp.logical_or(ladder_down, jnp.logical_not(jnp.logical_or(is_up, jnp.logical_or(is_down, jnp.logical_or(is_left, is_right)))))))
        rope_path = jnp.logical_or(rope_path, jnp.logical_and(is_climbing_rope, ladder_down))
        carry = jnp.where(jnp.logical_or(rope_path, rope_top_exit), 1, carry)
        # The other stair chamber enters through B410's ladder dispatch;
        # its first top-entry tick has not executed the rope DOWN handler.
        carry = jnp.where(jnp.logical_and(state.room_id == 23,
            jnp.any(top_entry_rope)), 0, carry)
        carry = jnp.where(jnp.logical_and(is_climbing_rope,
            jnp.logical_and(ladder_up, state.player_y <= rope_top_limit)), 1, carry)
        carry = jnp.where(jnp.logical_and(state.room_id == 12,
            jnp.logical_and(last_jump_frame, final_is_falling == 1)), 1, carry)
        carry = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_and(last_jump_frame, final_is_falling == 1)),
            (state.room_id != 23).astype(jnp.int32), carry)
        # DB10's middle-band stair-gap comparison leaves carry clear when
        # a completed rope jump enters a fall between the outer steps.
        stair_gap_carry = jnp.logical_and(state.room_id == 17,
            jnp.logical_and(final_y > 26, jnp.logical_and(final_y <= 105,
                jnp.logical_and(final_y != 76,
                    jnp.logical_and(final_x >= 9, final_x <= 143)))))
        carry = jnp.where(jnp.logical_and(stair_gap_carry,
            jnp.logical_and(last_jump_frame, final_is_falling == 1)), 0, carry)
        carry = jnp.where(jnp.logical_and(last_jump_frame, jnp.logical_and(ladder_up,
            jnp.any(jnp.logical_and(state.ropes_active == 1,
                jnp.logical_and(r_x + r_width >= state.player_x,
                    jnp.logical_and(r_x < state.player_x + 8, intersect_y_rope)))))), 1, carry)
        fall_carry = jnp.where(jnp.logical_or(jnp.logical_or(hit_floor, start_death), is_climbing_rope), 0, 1)
        right_air_drift = jnp.logical_and(jnp.logical_and(~is_bonus_room, air_vx > 0),
            jnp.logical_and(jnp.logical_or(state.fall_after_jump == 1, drifting_fall),
                (state.native_frame_count + 1 + 67) % 2 == 0))
        fall_carry = jnp.where(right_air_drift, new_x >= self.consts.WIDTH - 9, fall_carry)
        # Standing dispatch can prepare mode7 before any falling motion.
        # Its next contact still follows the standing path, including carry.
        carry = jnp.where(jnp.logical_and(jnp.isin(state.native_mode, jnp.array([7, 8])),
            state.is_falling == 1), fall_carry, carry)
        pickup_carry = jnp.where(jnp.logical_or(amulet_collected, torch_collected), 1,
            jnp.where(keys_collected > 0, state.inventory[0] == 0, 0))
        carry = jnp.where(any_item_pickup, pickup_carry, carry)
        carry = jnp.where(jnp.logical_or(num_opened > 0, jnp.any(actually_killed_mask)), 0, carry)
        native_room = state.room_id - jnp.where(state.room_id <= 5, 3,
            jnp.where(state.room_id <= 14, 7, 9))
        contact_death_carry = jnp.where(jnp.logical_and(state.level == 2,
            jnp.logical_or(died_from_enemy, died_from_laser)),
            jnp.where(died_from_laser, 0, native_room % 2), 1)
        carry = jnp.where(start_death, jnp.where(died_from_pit, 0, contact_death_carry), carry)
        carry = jnp.where(transition_any, jnp.logical_or(transition_left, transition_up), carry)
        original_duplicates = LEVEL_CONTENTS[jnp.minimum(state.level, 2), native_room_index, 3]
        history_weights = jnp.array([1,0,4], dtype=jnp.int32).at[1].set(
            jnp.where((original_duplicates & 1) != 0, 2, 8))
        removed_enemies = jnp.logical_or(actually_killed_mask,
            jnp.logical_and(start_death, jnp.logical_and(died_from_enemy, this_hit_enemy)))
        removed = jnp.sum(jnp.where(collect_item_mask, history_weights, 0))
        removed = removed | jnp.where(jnp.logical_and(rolling_room, jnp.any(removed_enemies)),
            2, jnp.sum(jnp.where(removed_enemies, history_weights, 0)))
        removed = removed | jnp.sum(jnp.where(open_door_mask, jnp.where(state.doors_x < 80, 8, 4), 0))
        new_history = state.native_room_history.at[native_room_index].set(history & (15 ^ removed))
        # Bonus respawn reinstalls the ruby bit after pickup cleared it.
        new_history = new_history.at[native_room_index].set(jnp.where(
            jnp.logical_and(is_bonus_room, bonus_spawn), history | 1,
            new_history[native_room_index]))
        # The standing handler tests support at the new horizontal anchor.
        # The pit edge therefore enters mode7 one tick before y advances.
        pit_edge_dispatch = jnp.logical_and(state.level < 2, jnp.logical_and(state.room_id == 29,
            jnp.logical_and(~platform_active_now, jnp.logical_and(state.player_y == 26,
                jnp.logical_and(on_ground,
                    ~check_platform_local(safe_y, new_x + self.consts.PLAYER_WIDTH // 2))))))
        # DB10 rejects standing anchors between the side steps. It also
        # tests the dropout-floor pointer throughout this middle band.
        # Falling dispatch can find the permanent middle step next tick,
        # alternating 7/4 without moving the visible anchor.
        clock = (state.native_frame_count + 1 + 31) % 128
        floor_missing = jnp.where(clock == 0, False,
            jnp.logical_or(clock == 92, state.platform_cycle >= 92))
        stair_dispatch = jnp.logical_and(state.room_id == 17,
            jnp.logical_and(jnp.logical_or(floor_missing,
                jnp.logical_and(final_x >= 9, final_x <= 143)), jnp.logical_and(state.native_mode == 4,
                jnp.logical_and(final_y > 26, final_y < 106))))
        prepared_fall = jnp.logical_and(jnp.logical_or(pit_edge_dispatch,
            jnp.logical_and(stair_dispatch, ~stair_walk_hold)),
            jnp.logical_and(final_is_climbing == 0, final_is_jumping == 0))
        fall_mode = jnp.where(is_bonus_room, 8, 7)
        next_mode = jnp.where(transition_any, 2,
            jnp.where(jnp.logical_or(new_death_timer > 0,
                jnp.logical_or(any_item_pickup, jnp.logical_or(num_opened > 0, jnp.any(actually_killed_mask)))), 6,
                jnp.where(jnp.logical_or(final_is_falling == 1, prepared_fall), fall_mode, 4)))
        next_mode = jnp.where(new_death_timer == 1, 2, next_mode)
        next_mode = jnp.where(jnp.logical_and(final_death,
            jnp.logical_and(state.death_timer > 0, new_death_timer <= 1)), 5, next_mode)
        # B75E samples FIRE in game-over dispatch too; a new press prepares
        # mode0 on the terminal frame without running another episode.
        next_mode = jnp.where(jnp.logical_and(state.native_mode == 5,
            jnp.logical_and(is_fire, ~state.prev_is_fire)), 0, next_mode)
        state = state.replace(
            rolling_skull_phase=rolling_phase, native_room_history=new_history,
            native_rolling_x=rolling_x, native_rolling_direction=rolling_direction,
            native_mode=next_mode.astype(jnp.int32), native_rng_carry=carry.astype(jnp.int32),
            game_over_pending=final_death,
            lives=new_lives,
            score=new_score,
            respawn_enemy_idx=jnp.where(
                jnp.logical_and(start_death, jnp.logical_and(died_from_enemy,
                    jnp.logical_or(state.room_id == 20,
                        jnp.logical_and(state.level == 2, ~jnp.isin(state.room_id, jnp.array([4,12])))))),
                jnp.argmax(this_hit_enemy.astype(jnp.int32)),
                jnp.where(cleanup_enemy, -1, state.respawn_enemy_idx)),
            pickup_timer=jnp.where(jnp.logical_or(jnp.logical_or(any_item_pickup, num_opened > 0), jnp.any(actually_killed_mask)),
                2 + jnp.mod(-(state.native_frame_count + 1), 4), 0),
            pickup_item_idx=jnp.where(any_item_pickup, jnp.argmax(collect_item_mask.astype(jnp.int32)), -1),
            player_x=jnp.where(num_opened > 0, state.player_x, final_x),
            player_y=jnp.where(num_opened > 0, state.player_y, final_y),
            player_vx=final_vx,
            player_vy=final_vy,
            player_dir=final_player_dir,
            is_jumping=final_is_jumping,
            fall_after_jump=fall_after_jump,
            native_jump_momentum=jnp.where(state.level == 2, native_jump_momentum, state.native_jump_momentum),
            native_climb_moving=jnp.where(is_active,
                jnp.logical_and(final_is_climbing == 1, jnp.logical_or(ladder_up, ladder_down)),
                state.native_climb_moving),
            native_walk_pose=jnp.where(is_active,
                jnp.logical_and(final_is_climbing == 0,
                    jnp.logical_and(final_is_jumping == 0, jnp.logical_and(final_is_falling == 0,
                        jnp.logical_or(abort_ladder, jnp.logical_or(stair_walk_hold, final_vx != 0))))), state.native_walk_pose),
            jump_counter=final_jump_counter,
            jump_type=jump_type,
            is_climbing=final_is_climbing,
            out_of_ladder_delay=jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_out_of_ladder_delay),
            last_rope=new_last_rope,
            last_ladder=final_last_ladder,
            prev_is_fire=final_prev_is_fire,
            is_falling=final_is_falling,
            fall_distance=final_fall_distance,
            frame_count=jnp.where(is_active, state.frame_count + 1, state.frame_count),
            room5_skull_phase=new_room5_phase,
            room5_skull_spawn_pending=jnp.where(is_active, False, state.room5_skull_spawn_pending),
            enemies_x=new_enemies_x,
            enemies_active=new_enemies_active,
            enemies_direction=new_enemies_direction,
            rolling_skull_direction=jnp.where(jnp.any(jnp.logical_and(is_active, state.enemies_type == 1)),
                new_enemies_direction[jnp.argmax(jnp.logical_and(is_active, state.enemies_type == 1))], state.rolling_skull_direction),
            inventory=current_inventory,
            amulet_time=new_amulet_time,
            items_x=new_items_x,
            items_active=new_items_active,
            doors_active=new_doors_active,
            laser_cycle=new_laser_cycle,
            platform_cycle=new_platform_cycle,
            death_timer=new_death_timer,
            death_type=new_death_type,
            bonus_room_timer=new_bonus_room_timer,
            first_gem_pickup=new_first_gem_pickup,
            bonus_pickup_latched=jnp.logical_and(is_bonus_room,
                jnp.logical_and(jnp.logical_or(state.bonus_pickup_latched, is_bonus_gem_collected),
                    jnp.logical_not(bonus_pose_reset))),
            key=new_rng_key
        )

        def reload_after_death(st):
            active = st.enemies_active.at[jnp.maximum(st.respawn_enemy_idx, 0)].set(
                jnp.where(st.respawn_enemy_idx >= 0, 0, st.enemies_active[jnp.maximum(st.respawn_enemy_idx, 0)]))
            st = st.replace(global_enemies_active=st.global_enemies_active.at[st.room_id].set(active),
                global_items_active=st.global_items_active.at[st.room_id].set(st.items_active),
                global_doors_active=st.global_doors_active.at[st.room_id].set(st.doors_active),
                respawn_enemy_idx=jnp.int32(-1))
            return load_room(st.room_id, st.replace(platform_cycle=jnp.int32(0)), self.consts)
        state = jax.lax.cond(jnp.logical_and(respawn_now, state.level == 2),
            lambda: reload_after_death(state), lambda: state)

        transition_any = jnp.logical_or(jnp.logical_or(transition_left, transition_right), jnp.logical_or(transition_down, transition_up))
        new_room_id = jnp.where(transition_left, state.room_id - 1, jnp.where(transition_right, state.room_id + 1, jnp.where(transition_up, state.room_id - 8, jnp.where(transition_down, state.room_id + 8, state.room_id))))

        def transition_fn(state_in):
            # room_idx = get_room_idx(new_room_id)
            st = state_in.replace(
                global_doors_active=state_in.global_doors_active.at[state_in.room_id].set(state_in.doors_active),
                global_items_active=state_in.global_items_active.at[state_in.room_id].set(state_in.items_active),
                global_enemies_active=state_in.global_enemies_active.at[state_in.room_id].set(state_in.enemies_active)
            )
            if self.consts.CUSTOM_ROOMS:
                st = load_custom_room(new_room_id, st, self.consts)
            else:
                st = load_room(new_room_id, st, self.consts)
            new_px = jnp.where(transition_left, self.consts.ROOM_ENTRY_X_RIGHT, jnp.where(transition_right, self.consts.ROOM_ENTRY_X_LEFT, new_x))
            temp_py = jnp.where(transition_down, self.consts.ROOM_ENTRY_Y_TOP, jnp.where(transition_up, self.consts.ROOM_ENTRY_Y_BOTTOM, new_y))

            # Prevent landing below floor: 
            # If feet are currently inside a floor, push up until they are just above it.
            new_room_idx = get_room_idx(new_room_id)
            new_room_col_map = self.ROOM_COLLISION_MAPS[new_room_idx]
            safe_px_trans = jnp.clip(new_px + self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)

            def is_inside(py):
                fy = jnp.clip(py + self.consts.PLAYER_HEIGHT - 1, 0, 148)
                in_col = check_platform(new_room_col_map, fy, safe_px_trans, self.consts.WIDTH)
                
                in_this_p = jnp.logical_and(st.platforms_active == 1, 
                        jnp.logical_and(jnp.logical_and(fy >= st.platforms_y, fy < st.platforms_y + 4),
                                        jnp.logical_and(safe_px_trans >= st.platforms_x, safe_px_trans < st.platforms_x + st.platforms_width)))
                in_plat = jnp.any(in_this_p)
                
                in_this_c = jnp.logical_and(st.conveyors_active == 1,
                        jnp.logical_and(jnp.logical_and(fy >= st.conveyors_y, fy < st.conveyors_y + 5),
                                        jnp.logical_and(safe_px_trans >= st.conveyors_x - 2, safe_px_trans < st.conveyors_x + 42)))
                in_conv = jnp.any(in_this_c)
                
                return jnp.logical_or(in_col, jnp.logical_or(in_plat, in_conv))

            py_offsets = jnp.arange(40)
            py_candidates = temp_py - py_offsets
            inside_mask = jax.vmap(is_inside)(py_candidates)
            num_to_push = jnp.sum(jnp.cumprod(inside_mask.astype(jnp.int32)))
            new_py = jnp.where(jnp.logical_or(transition_up, transition_down), temp_py, temp_py - num_to_push)
            # Upper corridor entry converts an airborne pose to the standing anchor.
            new_py = jnp.where(jnp.logical_and(jnp.logical_or(transition_left, transition_right),
                new_y <= 26), 26, new_py)

            return st.replace(
                player_x=jnp.where(jnp.logical_or(transition_left, transition_right), previous_x, new_px),
                player_y=jnp.where(jnp.logical_or(transition_up, transition_down), previous_y, new_y),
                transition_pending=jnp.where(jnp.logical_or(transition_left, transition_right), 1, 2),
                transition_x=new_px,
                transition_y=new_py,
                last_ladder=jnp.array(-1, dtype=jnp.int32),
                last_rope=jnp.array(-1, dtype=jnp.int32),
                # Horizontal death respawns use the boundary, not the inward
                # movement offset used while entering a room.
                entry_x=jnp.where(transition_right, 0, jnp.where(transition_left, self.consts.WIDTH - 9, new_px)),
                entry_y=new_py,
                entry_is_climbing=is_climbing,
                entry_last_ladder=jnp.array(-1, dtype=jnp.int32)
            )

        state = jax.lax.cond(transition_any, transition_fn, lambda x: x, state)

        obs = self._get_observation(state)
        reward = self._get_reward_from_scores(previous_score, state.score)
        done = self._get_done(state)
        info = self._get_info(state)

        return obs, state, reward, done, info
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        screen_size = (self.consts.HEIGHT, self.consts.WIDTH)
        return spaces.Dict({
            "player": spaces.get_object_space(n=1, screen_size=screen_size),
            "enemies": spaces.get_object_space(n=self.consts.MAX_ENEMIES_PER_ROOM, screen_size=screen_size),
            "items": spaces.get_object_space(n=self.consts.MAX_ITEMS_PER_ROOM, screen_size=screen_size),
            "conveyors": spaces.get_object_space(n=self.consts.MAX_CONVEYORS_PER_ROOM, screen_size=screen_size),
            "doors": spaces.get_object_space(n=self.consts.MAX_DOORS_PER_ROOM, screen_size=screen_size),
            "ropes": spaces.get_object_space(n=self.consts.MAX_ROPES_PER_ROOM, screen_size=screen_size),
            "platforms": spaces.get_object_space(n=self.consts.MAX_PLATFORMS_PER_ROOM, screen_size=screen_size),
            "ladders": spaces.get_object_space(n=self.consts.MAX_LADDERS_PER_ROOM, screen_size=screen_size),
            "lasers": spaces.get_object_space(n=self.consts.MAX_LASERS_PER_ROOM, screen_size=screen_size),
        })
        
    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def render(self, state: MontezumaRevengeState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: MontezumaRevengeState) -> MontezumaRevengeObservation:
        room_y = 47
        is_dark = jnp.logical_and(jnp.isin(state.room_id, jnp.array([25, 26, 27, 28, 29, 30, 31, 32])), state.inventory[2] != 1)
        lit = jnp.logical_not(is_dark)

        climbing = jnp.logical_and(
            jnp.logical_and(state.is_climbing == 1, state.death_timer == 0),
            jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)
        )
        player_obs = ObjectObservation.create(
            x=jnp.array([state.player_x]),
            y=jnp.array([state.player_y + room_y]),
            width=jnp.array([jnp.where(climbing, 7, 8)]),
            height=jnp.array([jnp.where(jnp.logical_and(climbing, state.last_rope == -1), 19, 20)]),
            active=jnp.array([1])
        )

        bounce = enemy_bounce_offsets(state, self.consts)
        enemies_obs = ObjectObservation.create(
            x=state.enemies_x,
            y=state.enemies_y + room_y - bounce,
            width=jnp.where(state.enemies_type == 4, 7, 8),
            height=jnp.where(state.enemies_type == 3, 11, 13),
            active=jnp.where(state.transition_pending > 0, 0, state.enemies_active)
        )

        hidden_gem = jnp.logical_and(state.items_type == 1, is_dark)
        items_obs = ObjectObservation.create(
            x=state.items_x,
            y=state.items_y + room_y,
            width=jnp.array([7, 7, 6, 6, 7])[state.items_type],
            height=jnp.array([15, 12, 15, 15, 13])[state.items_type],
            active=jnp.logical_and(state.items_active == 1, jnp.logical_not(hidden_gem)).astype(jnp.int32)
        )

        conveyors_obs = ObjectObservation.create(
            x=state.conveyors_x,
            y=state.conveyors_y + room_y,
            width=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 40),
            height=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 5),
            active=state.conveyors_active
        )

        doors_obs = ObjectObservation.create(
            x=state.doors_x,
            y=state.doors_y + room_y,
            width=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 4),
            height=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 37),
            active=jnp.logical_and(state.doors_active == 1, lit).astype(jnp.int32)
        )

        ropes_obs = ObjectObservation.create(
            x=state.ropes_x,
            y=state.ropes_top + room_y,
            width=rope_widths(state),
            height=state.ropes_bottom - state.ropes_top + 1,
            active=jnp.logical_and(state.ropes_active == 1, lit).astype(jnp.int32)
        )

        is_pit_room = jnp.isin(state.room_id, jnp.array([19, 27, 29, 31]))
        platforms_obs = ObjectObservation.create(
            x=state.platforms_x,
            y=state.platforms_y + room_y,
            width=jnp.where(is_pit_room, state.platforms_width // 8 * 8, state.platforms_width // 12 * 12),
            height=jnp.full(self.consts.MAX_PLATFORMS_PER_ROOM, jnp.where(is_pit_room, 7, 4)),
            active=jnp.logical_and(state.platforms_active == 1, state.platform_cycle < self.consts.PLATFORM_ACTIVE_DURATION).astype(jnp.int32)
        )

        long_ladder = jnp.logical_and(jnp.isin(state.room_id, jnp.array([3, 5, 10, 11, 12, 14])), jnp.arange(self.consts.MAX_LADDERS_PER_ROOM) == 0)
        ladders_top = state.ladders_top + room_y - jnp.where(long_ladder, 1, 0)
        ladders_bottom = state.ladders_bottom + room_y
        ladders_bottom = jnp.where(jnp.logical_and(state.room_id == 4, state.ladders_bottom == 130), ladders_bottom + 3, ladders_bottom)
        ladders_bottom = jnp.where(jnp.logical_and(state.room_id == 23, state.ladders_bottom == 150), ladders_bottom - 3, ladders_bottom)
        ladders_obs = ObjectObservation.create(
            x=state.ladders_x - jnp.where(long_ladder, 4, 0),
            y=ladders_top,
            width=jnp.where(long_ladder, 24, 16),
            height=ladders_bottom - ladders_top,
            active=jnp.logical_and(state.ladders_active == 1, lit).astype(jnp.int32)
        )

        laser_on = jnp.logical_or(state.level == 2, jnp.logical_and(state.laser_cycle >= 0, state.laser_cycle < 92))
        lasers_obs = ObjectObservation.create(
            x=state.lasers_x,
            y=jnp.full(self.consts.MAX_LASERS_PER_ROOM, 54),
            width=jnp.full(self.consts.MAX_LASERS_PER_ROOM, 4),
            height=jnp.full(self.consts.MAX_LASERS_PER_ROOM, 40),
            active=jnp.logical_and(state.lasers_active == 1, laser_on).astype(jnp.int32)
        )

        return MontezumaRevengeObservation(player=player_obs, enemies=enemies_obs, items=items_obs, conveyors=conveyors_obs, doors=doors_obs, ropes=ropes_obs, platforms=platforms_obs, ladders=ladders_obs, lasers=lasers_obs)
    
    def _get_info(self, state: MontezumaRevengeState) -> MontezumaRevengeInfo:
        return MontezumaRevengeInfo(lives=state.lives, room_id=state.room_id)

    def _get_reward_from_scores(self, previous_score: jnp.ndarray, score: jnp.ndarray) -> float:
        return jnp.sum(score - previous_score).astype(jnp.float32)

    def _get_reward(self, previous_state: MontezumaRevengeState, state: MontezumaRevengeState) -> float:
        """Mod-pipeline contract: reward is computed from (previous_state, state)."""
        return self._get_reward_from_scores(previous_state.score, state.score)

    def _get_done(self, state: MontezumaRevengeState) -> bool:
        if self.consts.CUSTOM_ROOMS:
            return jnp.logical_or(state.lives < 0,
                jnp.logical_and(state.room_id == 24, state.player_y >= 148))
        return state.lives < 0
