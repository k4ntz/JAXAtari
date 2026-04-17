import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple
import os

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
import jaxatari.spaces as spaces
from jaxatari.rendering import jax_rendering_utils as render_utils

from jaxatari.games.montezuma_revenge.core import (
    MontezumaRevengeConstants,
    MontezumaRevengeState,
    MontezumaRevengeObservation,
    MontezumaRevengeInfo,
    get_room_idx,
    check_platform,
)
from jaxatari.games.montezuma_revenge.renderer import MontezumaRevengeRenderer
from jaxatari.games.montezuma_revenge.rooms import load_room


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

        self.ROOM_COLLISION_MAPS = jnp.stack([room_col_0_3, room_col_0_4, room_col_0_5, room_col_1_3, room_col_1_2, room_col_1_4, room_col_1_5, room_col_1_6, room_col_2_2, room_col_2_1, room_col_2_3, room_col_2_4, room_col_2_5, room_col_2_6, room_col_2_7, room_col_3_7, room_col_3_8, room_col_3_6, room_col_3_4, room_col_3_3, room_col_3_5, room_col_3_1, room_col_3_2, room_col_3_0])

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
            player_dir=jnp.array(1, dtype=jnp.int32),
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
            laser_cycle=jnp.array(0, dtype=jnp.int32),
            platforms_x=jnp.zeros(self.consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32),
            platforms_y=jnp.zeros(self.consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32),
            platforms_width=jnp.full(self.consts.MAX_PLATFORMS_PER_ROOM, 12, dtype=jnp.int32),
            platforms_active=jnp.zeros(self.consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32),
            platform_cycle=jnp.array(0, dtype=jnp.int32),
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
        
        state = state.replace(global_items_active=gia, global_doors_active=gda, global_enemies_active=gea, global_enemies_type=gety, global_items_type=giy)
        
        state = load_room(jnp.array(self.consts.INITIAL_ROOM_ID, dtype=jnp.int32), state, self.consts)
        obs = self._get_observation(state)
        return obs, state
    
    def step(self, state: MontezumaRevengeState, action: int) -> Tuple[MontezumaRevengeObservation, MontezumaRevengeState, float, bool, MontezumaRevengeInfo]:
        is_active = state.death_timer == 0
        room_idx = get_room_idx(state.room_id)
        room_col_map = self.ROOM_COLLISION_MAPS[room_idx]
        
        is_bonus_room = state.room_id == 24
        floor_dropped = jnp.logical_and(is_bonus_room, state.bonus_room_timer >= 640)
        room_col_map = jax.lax.select(floor_dropped, room_col_map.at[47:50, 4:156].set(0), room_col_map)
        
        platform_active_now = jnp.less(state.platform_cycle, self.consts.PLATFORM_ACTIVE_DURATION)
        previous_score = state.score

        # Amulet counter
        new_amulet_time = jnp.where(is_active, jnp.maximum(state.amulet_time - 1, 0), state.amulet_time)
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
        is_in_air = jnp.logical_or(state.is_jumping == 1, state.fall_after_jump == 1)
        new_vx = jax.lax.select(is_right, self.consts.PLAYER_SPEED, 0)
        new_vx = jax.lax.select(is_left, -self.consts.PLAYER_SPEED, new_vx)

        air_vx = state.player_vx
        air_dx = jnp.where(jnp.logical_and(is_in_air, state.frame_count % 2 == 0), air_vx, 0)
        air_dx = jnp.where(state.is_jumping, state.player_vx, air_dx) # Allow horizontal control during jump, but not during fall after jump
        dx = jnp.where(is_in_air, air_dx, new_vx)
        # x: move only every other frame (when in air)
        # just falling: no x velocity
        dx = jnp.where(jnp.logical_and(state.is_falling == 1, state.fall_after_jump == 0), 0, dx)
        new_player_dir = jnp.where(is_right, 1, jnp.where(is_left, -1, state.player_dir))

        player_mid_x = state.player_x + self.consts.PLAYER_WIDTH // 2
        player_feet_y = state.player_y + self.consts.PLAYER_HEIGHT - 1
        
        # 0. Ladder and Rope Climbing Logic
        new_out_of_ladder_delay = jnp.where(state.out_of_ladder_delay > 0, state.out_of_ladder_delay - 1, 0)

        # Vectorized check_ladder
        l_x = state.ladders_x
        ladder_mid_x = l_x + 8
        l_top = state.ladders_top
        l_bottom = state.ladders_bottom
        is_aligned_ladder = jnp.logical_and(state.ladders_active == 1, jnp.abs(player_mid_x - ladder_mid_x) <= 4)
        is_aligned_ladder = jnp.logical_and(is_aligned_ladder, new_out_of_ladder_delay == 0)
        get_on_top_ladder = jnp.logical_and(is_aligned_ladder, jnp.logical_and(is_down, jnp.abs(player_feet_y - l_top) <= 5))
        get_on_bottom_ladder = jnp.logical_and(is_aligned_ladder, jnp.logical_and(is_up, jnp.abs(player_feet_y - l_bottom) <= 5))
        ladder_bottom_bound = jnp.where(l_bottom >= 148, 170, l_bottom + 1)
        ladder_top_bound = jnp.where(l_top <= 6, 0, l_top - 4)
        in_ladder_zone = jnp.logical_and(is_aligned_ladder, jnp.logical_and(player_feet_y >= ladder_top_bound, player_feet_y <= ladder_bottom_bound))
        on_this_ladder = jnp.where(state.is_climbing == 1, jnp.logical_and(in_ladder_zone, jnp.logical_or(state.last_ladder == jnp.arange(self.consts.MAX_LADDERS_PER_ROOM), state.last_ladder == -1)), jnp.logical_or(get_on_top_ladder, get_on_bottom_ladder))
        can_ladder = jnp.any(on_this_ladder)
        ladder_idx = jnp.where(can_ladder, jnp.argmax(on_this_ladder.astype(jnp.int32)), -1)

        # Vectorized check_rope
        r_x = state.ropes_x
        r_top = state.ropes_top
        r_bottom = state.ropes_bottom
        is_aligned_rope = jnp.logical_and(state.ropes_active == 1, jnp.abs(player_mid_x - r_x) <= 4)
        player_top_y = state.player_y
        intersect_y_rope = jnp.logical_and(player_feet_y >= r_top, player_top_y <= r_bottom)
        catch_rope = jnp.logical_and(state.is_climbing == 0, jnp.logical_and(is_aligned_rope, jnp.logical_and(intersect_y_rope, state.last_rope != jnp.arange(self.consts.MAX_ROPES_PER_ROOM))))
        get_on_top_rope = jnp.logical_and(is_aligned_rope, jnp.logical_and(is_down, jnp.abs(player_feet_y - r_top) <= 5))
        get_on_bottom_rope = jnp.logical_and(is_aligned_rope, jnp.logical_and(is_up, jnp.abs(player_feet_y - r_bottom) <= 5))
        can_climb_above = jnp.logical_or(jnp.logical_and(state.room_id == 12, jnp.arange(self.consts.MAX_ROPES_PER_ROOM) == 0), jnp.logical_and(state.room_id == 17, jnp.arange(self.consts.MAX_ROPES_PER_ROOM) == 0))
        top_bound_rope = jnp.where(can_climb_above, r_top - 5, r_top)
        in_rope_zone = jnp.logical_and(is_aligned_rope, jnp.logical_and(player_feet_y >= top_bound_rope, player_top_y <= r_bottom + 2))
        on_this_rope = jnp.where(state.is_climbing == 1, jnp.logical_and(in_rope_zone, jnp.logical_or(state.last_rope == jnp.arange(self.consts.MAX_ROPES_PER_ROOM), state.last_rope == -1)), jnp.logical_and(catch_rope, jnp.logical_or(get_on_top_rope, jnp.logical_or(get_on_bottom_rope, in_rope_zone))))
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

        is_jumping_off_ladder = jnp.logical_and(can_ladder, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(is_fire, can_move_off)))
        is_moving_off_ladder = jnp.logical_and(can_ladder, jnp.logical_and(state.is_climbing == 1, can_move_off))
        abort_ladder = jnp.logical_or(is_jumping_off_ladder, is_moving_off_ladder)

        is_jumping_off_rope = jnp.logical_and(can_rope, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(is_fire, can_move_off)))
        abort_rope = is_jumping_off_rope

        is_climbing_ladder = jnp.logical_and(can_ladder, jnp.logical_not(abort_ladder))
        is_climbing_rope = jnp.logical_and(can_rope, jnp.logical_not(abort_rope))

        is_climbing = jnp.where(jnp.logical_or(is_climbing_ladder, is_climbing_rope), 1, 0)
        
        # If we are aborting the ladder, or simply falling off, start the delay
        # But only if we were previously climbing!
        started_delay = jnp.logical_and(state.is_climbing == 1, is_climbing == 0)
        new_out_of_ladder_delay = jnp.where(started_delay, self.consts.OUT_OF_LADDER_DELAY, new_out_of_ladder_delay)

        target_climb_x = state.player_x
        target_climb_x = jnp.where(ladder_idx != -1, state.ladders_x[ladder_idx] + 8 - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        target_climb_x = jnp.where(rope_idx != -1, state.ropes_x[rope_idx] - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        
        current_x = jnp.where(is_climbing == 1, target_climb_x, state.player_x)
        
        def check_platform_local(y, x):
            return check_platform(room_col_map, y, x, self.consts.WIDTH)
        
        # 1. Check if strictly on ground
        safe_x = jnp.clip(current_x + self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)
        safe_y = jnp.clip(player_feet_y + 1, 0, 148)
        on_ground = check_platform_local(safe_y, safe_x)
        
        # Vectorized conveyor check
        is_on_conveyor_ground = jnp.logical_and(
            state.conveyors_active == 1,
            jnp.logical_and(player_feet_y == state.conveyors_y - 1, jnp.logical_and(player_mid_x >= state.conveyors_x - 2, player_mid_x < state.conveyors_x + 42))
        )
        on_ground = jnp.logical_or(on_ground, jnp.any(is_on_conveyor_ground))

        # Vectorized platform check
        is_on_plat_ground = jnp.logical_and(
            jnp.logical_and(state.platforms_active == 1, platform_active_now),
            jnp.logical_and(player_feet_y == state.platforms_y - 1, jnp.logical_and(safe_x >= state.platforms_x, safe_x < state.platforms_x + state.platforms_width))
        )
        on_ground = jnp.logical_or(on_ground, jnp.any(is_on_plat_ground))


        # Update last_rope and last_ladder
        new_last_rope = jnp.where(on_ground, -1, state.last_rope)
        new_last_rope = jnp.where(is_climbing_rope, rope_idx, new_last_rope)
        new_last_rope = jnp.where(is_jumping_off_rope, rope_idx, new_last_rope)

        new_last_ladder = jnp.where(on_ground, -1, state.last_ladder)
        new_last_ladder = jnp.where(is_climbing == 1, ladder_idx, new_last_ladder)

        # 2. Process Jump Initiation
        start_jump_normal = jnp.logical_and(is_fire, jnp.logical_and(on_ground, jnp.logical_and(state.is_jumping == 0, is_climbing == 0)))
        start_jump = jnp.logical_or(start_jump_normal, is_jumping_off_rope)
        is_jumping = jnp.where(start_jump, 1, state.is_jumping)
        is_jumping = jnp.where(is_climbing == 1, 0, is_jumping) # cancel jump
        jump_counter = jnp.where(start_jump, 0, state.jump_counter)

        # Keep latched airborne momentum even on frames where dx is intentionally zero.
        current_vx = jnp.where(is_in_air, air_vx, dx)

        # 3. Calculate DY
        def get_jump_dy():
            dy_jump = -self.consts.JUMP_Y_OFFSETS[jump_counter]
            return dy_jump, jump_counter + 1, 1
            
        def get_fall_dy():
            pixel_1_below = check_platform_local(safe_y, safe_x)
            pixel_2_below = check_platform_local(jnp.clip(player_feet_y + 2, 0, 148), safe_x)
            
            is_on_c_pixel2 = jnp.logical_and(
                state.conveyors_active == 1,
                jnp.logical_and(player_feet_y + 1 == state.conveyors_y - 1, jnp.logical_and(safe_x >= state.conveyors_x - 2, safe_x < state.conveyors_x + 42))
            )
            pixel_2_below = jnp.logical_or(pixel_2_below, jnp.any(is_on_c_pixel2))

            is_on_p_pixel2 = jnp.logical_and(
                jnp.logical_and(state.platforms_active == 1, platform_active_now),
                jnp.logical_and(player_feet_y + 1 == state.platforms_y - 1, jnp.logical_and(safe_x >= state.platforms_x, safe_x < state.platforms_x + state.platforms_width))
            )
            pixel_2_below = jnp.logical_or(pixel_2_below, jnp.any(is_on_p_pixel2))


            fall_dist = jnp.where(on_ground, 0, jnp.where(pixel_2_below, 1, self.consts.GRAVITY))
            return fall_dist, 0, 0
            
        def get_climb_dy():
            climb_dist = jnp.where(is_down, self.consts.PLAYER_SPEED, jnp.where(is_up, -self.consts.PLAYER_SPEED, 0))
            # Zero out vertical speed on the frame we catch the rope
            just_caught_rope = jnp.logical_and(state.is_climbing == 0, rope_idx != -1)
            climb_dist = jnp.where(just_caught_rope, 0, climb_dist)
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
        
        new_is_jumping = jnp.where(new_jump_counter >= self.consts.JUMP_Y_OFFSETS.shape[0], 0, new_is_jumping)
        
        # 4. Resolve Vertical Collision
        new_y = state.player_y + dy
        # Allow climbing slightly above the rope top when pressing UP to reach platforms
        can_climb_above_rope = jnp.logical_or(
            jnp.logical_and(state.room_id == 12, rope_idx == 0),
            jnp.logical_and(state.room_id == 17, rope_idx == 0)
        )
        top_extension = jnp.where(jnp.logical_and(is_up, can_climb_above_rope), 25, 0)
        rope_top_limit = state.ropes_top[rope_idx] - top_extension
        new_y = jnp.where(jnp.logical_and(is_climbing == 1, rope_idx != -1), jnp.maximum(new_y, rope_top_limit), new_y)
        new_feet_y = new_y + self.consts.PLAYER_HEIGHT - 1
        
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
        
        is_hit_rm = jnp.logical_and(jnp.logical_not(is_near_top), jnp.logical_and(dy >= y_check_offsets + 1, is_top_surface))
        hit_floor_rm = jnp.any(is_hit_rm)
        snapped_y_rm = jnp.where(hit_floor_rm, y_checks[jnp.argmax(is_hit_rm.astype(jnp.int32))] - self.consts.PLAYER_HEIGHT, new_y)

        # Conveyor hit floor
        crossed_c = jnp.logical_and(player_feet_y <= state.conveyors_y - 1, new_feet_y >= state.conveyors_y - 1)
        is_hit_c = jnp.logical_and(
            jnp.logical_not(is_near_top),
            jnp.logical_and(
                state.conveyors_active == 1,
                jnp.logical_and(dy > 0, jnp.logical_and(crossed_c, jnp.logical_and(safe_x >= state.conveyors_x - 2, safe_x < state.conveyors_x + 42)))
            )
        )
        hit_floor_c = jnp.any(is_hit_c)
        snapped_y_c = jnp.where(hit_floor_c, state.conveyors_y[jnp.argmax(is_hit_c.astype(jnp.int32))] - 1 - self.consts.PLAYER_HEIGHT + 1, snapped_y_rm)

        # Platform hit floor
        crossed_p = jnp.logical_and(player_feet_y <= state.platforms_y - 1, new_feet_y >= state.platforms_y - 1)
        is_hit_p = jnp.logical_and(
            jnp.logical_not(is_near_top),
            jnp.logical_and(
                jnp.logical_and(state.platforms_active == 1, platform_active_now),
                jnp.logical_and(dy > 0, jnp.logical_and(crossed_p, jnp.logical_and(safe_x >= state.platforms_x, safe_x < state.platforms_x + state.platforms_width)))
            )
        )
        hit_floor_p = jnp.any(is_hit_p)
        snapped_y = jnp.where(hit_floor_p, state.platforms_y[jnp.argmax(is_hit_p.astype(jnp.int32))] - 1 - self.consts.PLAYER_HEIGHT + 1, snapped_y_c)
        hit_floor = jnp.logical_or(hit_floor_rm, jnp.logical_or(hit_floor_c, hit_floor_p))

        new_y = jnp.where(hit_floor, snapped_y, new_y)

        # Stop climbing if we hit a floor (e.g. landing on a conveyor belt)
        is_climbing = jnp.where(hit_floor, 0, is_climbing)
        new_is_jumping = jnp.where(hit_floor, 0, new_is_jumping)

        # Set is_falling state
        new_is_falling = jnp.where(jnp.logical_and(new_is_jumping == 0, hit_floor == False), jnp.where(dy > 0, 1, 0), 0)
        new_is_falling = jnp.where(is_climbing == 1, 0, new_is_falling)
        
        # 5. Resolve Horizontal with Wall Collision
        raw_new_x = current_x + dx
        transition_left = jnp.logical_and(raw_new_x < 0, jnp.isin(state.room_id, jnp.array([4, 5, 12, 11, 13, 14, 18, 17, 19, 20, 21, 22, 23, 31, 32, 28, 29, 26, 27, 25])))
        transition_right = jnp.logical_and(raw_new_x + self.consts.PLAYER_WIDTH > self.consts.WIDTH, jnp.isin(state.room_id, jnp.array([3, 4, 12, 10, 11, 13, 17, 18, 19, 20, 21, 22, 30, 31, 28, 27, 25, 26, 24])))
        transition_down = jnp.logical_and(new_y >= self.consts.ROOM_EXIT_Y_BOTTOM, jnp.isin(state.room_id, jnp.array([3, 4, 5, 10, 11, 12, 13, 14, 22, 23, 20])))
        transition_up = jnp.logical_and(new_y <= self.consts.ROOM_EXIT_Y_TOP, jnp.isin(state.room_id, jnp.array([11, 12, 13, 18, 19, 20, 21, 22, 30, 31, 28])))

        new_x = jnp.clip(raw_new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
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
            ignore = jnp.logical_and(on_ground == 0, is_perm_platform)
            return jnp.logical_and(room_col_map[y, x] == 1, jnp.logical_not(ignore))
        
        hit_wall = jnp.logical_or(
            is_wall(check_y_top, front_x),
            jnp.logical_or(
                is_wall(check_y_mid, front_x),
                is_wall(check_y_bot, front_x)
            )
        )
        
        # 5.5 Item Collection
        overlap_x_item = jnp.logical_and(new_left_x < state.items_x + 6, new_right_x >= state.items_x)
        overlap_y_item = jnp.logical_and(check_y_top < state.items_y + 8, check_y_bot >= state.items_y)
        collect_item_mask = jnp.logical_and(state.items_active == 1, jnp.logical_and(overlap_x_item, overlap_y_item))

        item_scores = jnp.where(state.items_type == 0, 100, 
                        jnp.where(state.items_type == 2, 100, 1000)) # Amulet score is 100
        new_score = state.score + jnp.sum(jnp.where(collect_item_mask, item_scores, 0))
        new_items_active = jnp.where(collect_item_mask, 0, state.items_active)

        # Inventory updates
        keys_collected = jnp.sum(jnp.where(jnp.logical_and(collect_item_mask, state.items_type == 0), 1, 0))
        sword_collected = jnp.any(jnp.logical_and(collect_item_mask, state.items_type == 3))
        torch_collected = jnp.any(jnp.logical_and(collect_item_mask, state.items_type == 4))
        amulet_collected = jnp.any(jnp.logical_and(collect_item_mask, state.items_type == 2))

        current_inventory = state.inventory
        current_inventory = current_inventory.at[0].add(keys_collected)
        current_inventory = current_inventory.at[1].set(jnp.where(sword_collected, 1, current_inventory[1]))
        current_inventory = current_inventory.at[2].set(jnp.where(torch_collected, 1, current_inventory[2]))
        
        # Handle amulet time reset and disappearance
        new_amulet_time = jnp.where(amulet_collected, self.consts.AMULET_DURATION, new_amulet_time)
        current_inventory = current_inventory.at[3].set(jnp.where(new_amulet_time > 0, 1, 0))

        # Door checks
        in_x_door = jnp.logical_and(front_x >= state.doors_x, front_x < state.doors_x + 4)
        in_y_door_top = jnp.logical_and(check_y_top >= state.doors_y, check_y_top < state.doors_y + 38)
        in_y_door_mid = jnp.logical_and(check_y_mid >= state.doors_y, check_y_mid < state.doors_y + 38)
        in_y_door_bot = jnp.logical_and(check_y_bot >= state.doors_y, check_y_bot < state.doors_y + 38)
        in_y_door = jnp.logical_or(in_y_door_top, jnp.logical_or(in_y_door_mid, in_y_door_bot))
        hit_door_mask = jnp.logical_and(state.doors_active == 1, jnp.logical_and(in_x_door, in_y_door))
        
        hit_door_order = jnp.cumsum(hit_door_mask.astype(jnp.int32))
        open_door_mask = jnp.logical_and(hit_door_mask, hit_door_order <= current_inventory[0])
        
        num_opened = jnp.sum(open_door_mask.astype(jnp.int32))
        current_inventory = current_inventory.at[0].add(-num_opened)
        new_doors_active = jnp.where(open_door_mask, 0, state.doors_active)
        new_score = new_score + num_opened * 300
        
        hit_door_as_wall = jnp.logical_and(hit_door_mask, jnp.logical_not(open_door_mask))
        hit_wall = jnp.logical_or(hit_wall, jnp.any(hit_door_as_wall))

        # Bonus Room Gem Spawning
        is_bonus_gem_collected = jnp.logical_and(is_bonus_room, collect_item_mask[0])
        new_rng_key, subkey = jax.random.split(state.key)
        random_x = jax.random.uniform(subkey, (), minval=19, maxval=140)
        
        new_items_x = jax.lax.select(is_bonus_gem_collected, state.items_x.at[0].set(random_x.astype(jnp.int32)), state.items_x)
        new_items_active = jax.lax.select(is_bonus_gem_collected, new_items_active.at[0].set(1), new_items_active)
        new_first_gem_pickup = jnp.where(is_bonus_gem_collected, 1, state.first_gem_pickup)
        # Increment timer every frame we are in the bonus room
        new_bonus_room_timer = jax.lax.select(is_bonus_room, jnp.where(is_active, state.bonus_room_timer + 1, state.bonus_room_timer), 0)
        new_first_gem_pickup = jax.lax.select(is_bonus_room, new_first_gem_pickup, 0)

        new_x = jnp.where(jnp.logical_or(hit_wall, is_climbing == 1), current_x, new_x)
        
        new_mid_x = new_x + self.consts.PLAYER_WIDTH // 2
        new_feet_y_after = new_y + self.consts.PLAYER_HEIGHT - 1
        
        # Vectorized conveyor physics
        is_on_conveyor_physics = jnp.logical_and(
            state.conveyors_active == 1,
            jnp.logical_and(new_feet_y_after == state.conveyors_y - 1, jnp.logical_and(new_mid_x >= state.conveyors_x - 2, new_mid_x < state.conveyors_x + 42))
        )
        conveyor_velocities = jnp.mod(state.frame_count, 2) * state.conveyors_direction
        total_conveyor_velocity = jnp.sum(jnp.where(jnp.logical_and(is_on_conveyor_physics, is_climbing == 0), conveyor_velocities, 0))
        new_x = new_x + total_conveyor_velocity
        new_x = jnp.clip(new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        
        current_vx = jnp.where(jnp.logical_or(hit_wall, hit_floor), 0, current_vx)
        
        # 6. Enemy Movement
        # Move 1 pixel every 4 frames
        speed_enemy = jnp.where(jnp.logical_and(is_active, jnp.mod(state.frame_count, 4) == 0), 1, 0)
        raw_new_enemies_x = state.enemies_x + state.enemies_direction * speed_enemy
        
        # Bounce off walls
        hit_wall_left_enemy = jnp.logical_and(state.enemies_direction < 0, room_col_map[state.enemies_y + 8, jnp.clip(raw_new_enemies_x, 0, self.consts.WIDTH - 1)] == 1)
        hit_wall_right_enemy = jnp.logical_and(state.enemies_direction > 0, room_col_map[state.enemies_y + 8, jnp.clip(raw_new_enemies_x + 8, 0, self.consts.WIDTH - 1)] == 1)
        
        hit_left_enemy = raw_new_enemies_x <= state.enemies_min_x
        hit_right_enemy = raw_new_enemies_x >= state.enemies_max_x
        
        bounce_enemy = jnp.logical_and(is_active, jnp.logical_or(hit_left_enemy, jnp.logical_or(hit_right_enemy, jnp.logical_or(hit_wall_left_enemy, hit_wall_right_enemy))))
        
        # Synchronize bouncing: if ANY active enemy in the room hits a wall, ALL of them flip direction
        bounce_any = jnp.any(jnp.logical_and(state.enemies_active == 1, bounce_enemy))
        new_enemies_direction = jnp.where(bounce_any, -state.enemies_direction, state.enemies_direction)
        new_enemies_x = jnp.where(bounce_any, state.enemies_x, raw_new_enemies_x)
        
        # 7. Dying Mechanism (Fall Damage & Enemy Collision)
        fall_stopped = jnp.logical_and(state.is_falling == 1, new_is_falling == 0)
        died_from_fall = jnp.logical_and(
            jnp.logical_and(fall_stopped, is_climbing == 0),
            state.fall_distance > self.consts.MAX_FALL_DISTANCE
        )

        new_fall_distance = jnp.where(
            dy > 0,
            state.fall_distance + dy,
            0
        )
        
        # Vectorized enemy collision
        e_bounce_offset = jnp.where(state.enemies_bouncing == 1, self.consts.BOUNCE_OFFSETS[jnp.mod(state.frame_count // 4, 22)], 0)
        e_y_col = state.enemies_y - e_bounce_offset
        overlap_x_enemy = jnp.logical_and(new_left_x < new_enemies_x + 7, new_right_x >= new_enemies_x + 1)
        overlap_y_enemy = jnp.logical_and(check_y_top < e_y_col + 15, check_y_bot >= e_y_col + 1)
        this_hit_enemy = jnp.logical_and(state.enemies_active == 1, jnp.logical_and(overlap_x_enemy, overlap_y_enemy))
        
        # Neutralize enemy collision if amulet is active
        this_hit_enemy = jnp.logical_and(this_hit_enemy, jnp.logical_not(is_amulet_active))

        has_sword = current_inventory[1] == 1
        kill_enemy_mask = jnp.logical_and(this_hit_enemy, has_sword)
        kill_order = jnp.cumsum(kill_enemy_mask.astype(jnp.int32))
        actually_killed_mask = jnp.logical_and(kill_enemy_mask, kill_order <= 1)
        
        died_from_enemy = jnp.any(jnp.logical_and(this_hit_enemy, jnp.logical_not(actually_killed_mask)))
        new_enemies_active = jnp.where(this_hit_enemy, 0, state.enemies_active)
        sword_used = jnp.any(actually_killed_mask)
        current_inventory = current_inventory.at[1].set(jnp.where(sword_used, 0, current_inventory[1]))
        new_score = new_score + jnp.sum(jnp.where(actually_killed_mask, 100, 0))

        # Laser Collision
        # Update laser and platform cycles
        new_laser_cycle = jnp.where(is_active, jnp.mod(state.laser_cycle + 1, 128), state.laser_cycle)
        new_platform_cycle = jnp.where(is_active, jnp.mod(state.platform_cycle + 1, self.consts.PLATFORM_CYCLE_LENGTH), state.platform_cycle)
        laser_active_now = jnp.logical_and(jnp.greater_equal(state.laser_cycle, 0), jnp.less(state.laser_cycle, 92))
        platform_active_now = jnp.less(state.platform_cycle, self.consts.PLATFORM_ACTIVE_DURATION)

        l_active_las = jnp.logical_and(state.lasers_active == 1, laser_active_now)
        overlap_x_las = jnp.logical_and(new_left_x < state.lasers_x + 4, new_right_x >= state.lasers_x)
        overlap_y_las = jnp.logical_and(check_y_top < 46, check_y_bot >= 7)
        died_from_laser = jnp.any(jnp.logical_and(l_active_las, jnp.logical_and(overlap_x_las, overlap_y_las)))
        
        is_pit_room = jnp.any(state.room_id == jnp.array([19, 27, 29, 31]))
        died_from_pit = jnp.logical_and(is_pit_room, jnp.logical_and(player_feet_y >= 110, jnp.logical_not(on_ground)))

        player_died = jnp.logical_or(died_from_fall, jnp.logical_or(died_from_enemy, jnp.logical_or(died_from_laser, died_from_pit)))
        
        start_death = jnp.logical_and(state.death_timer == 0, player_died)
        new_death_timer = jnp.where(start_death, self.consts.DEATH_TIMER_FRAMES, 
                                    jnp.where(state.death_timer > 0, state.death_timer - 1, 0))
        
        death_type = jnp.where(died_from_fall, 1, jnp.where(died_from_enemy, 2, jnp.where(died_from_laser, 3, jnp.where(died_from_pit, 1, 0))))
        new_death_type = jnp.where(start_death, death_type, jnp.where(new_death_timer == 0, 0, state.death_type))

        respawn_now = jnp.logical_and(state.death_timer == 1, new_death_timer == 0)
        
        spawn_x = state.entry_x
        spawn_y = state.entry_y
        
        new_lives = jnp.where(start_death, state.lives - 1, state.lives)
        final_x = jnp.where(respawn_now, spawn_x, jnp.where(new_death_timer > 0, state.player_x, new_x))
        final_y = jnp.where(respawn_now, spawn_y, jnp.where(new_death_timer > 0, state.player_y, new_y))
        final_vx = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, current_vx)
        final_vy = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, dy)
        final_player_dir = jnp.where(respawn_now, 1, jnp.where(new_death_timer > 0, state.player_dir, new_player_dir))
        final_is_jumping = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_is_jumping)
        final_is_falling = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_is_falling)
        final_is_climbing = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), jnp.where(respawn_now, state.entry_is_climbing, 0), is_climbing)
        final_last_ladder = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), jnp.where(respawn_now, state.entry_last_ladder, -1), new_last_ladder)
        final_jump_counter = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_jump_counter)
        final_fall_distance = jnp.where(respawn_now, 0, jnp.where(new_death_timer > 0, state.fall_distance, new_fall_distance))

        # if jumped before
        # fall_after_jump -> (prev_jumped, not anymore , or prev fall_after_jump) and falling
        fall_after_jump = jnp.where(jnp.logical_or(state.fall_after_jump == 1, jnp.logical_and(state.is_jumping == 1, new_is_jumping == 0)), 1, 0)
        fall_after_jump = jnp.where(final_is_falling == 1, fall_after_jump, 0)
        
        state = state.replace(
            lives=new_lives,
            score=new_score,
            player_x=final_x,
            player_y=final_y,
            player_vx=final_vx,
            player_vy=final_vy,
            player_dir=final_player_dir,
            is_jumping=final_is_jumping,
            fall_after_jump=fall_after_jump,
            jump_counter=final_jump_counter,
            is_climbing=final_is_climbing,
            out_of_ladder_delay=jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_out_of_ladder_delay),
            last_rope=new_last_rope,
            last_ladder=final_last_ladder,
            is_falling=final_is_falling,
            fall_distance=final_fall_distance,
            frame_count=jnp.where(is_active, state.frame_count + 1, state.frame_count),
            enemies_x=new_enemies_x,
            enemies_active=new_enemies_active,
            enemies_direction=new_enemies_direction,
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
            key=new_rng_key
        )

        transition_any = jnp.logical_or(jnp.logical_or(transition_left, transition_right), jnp.logical_or(transition_down, transition_up))
        new_room_id = jnp.where(transition_left, 
                                jnp.where(state.room_id == 5, 4, jnp.where(state.room_id == 4, 3, jnp.where(state.room_id == 11, 10, jnp.where(state.room_id == 12, 11, jnp.where(state.room_id == 13, 12, jnp.where(state.room_id == 14, 13, jnp.where(state.room_id == 18, 17, jnp.where(state.room_id == 20, 19, jnp.where(state.room_id == 21, 20, jnp.where(state.room_id == 22, 21, jnp.where(state.room_id == 23, 22, jnp.where(state.room_id == 32, 31, jnp.where(state.room_id == 31, 30, jnp.where(state.room_id == 29, 28, jnp.where(state.room_id == 28, 27, jnp.where(state.room_id == 27, 26, jnp.where(state.room_id == 26, 25, jnp.where(state.room_id == 25, 24, state.room_id)))))))))))))))))),
                                jnp.where(transition_right, 
                                          jnp.where(state.room_id == 3, 4, jnp.where(state.room_id == 4, 5, jnp.where(state.room_id == 10, 11, jnp.where(state.room_id == 11, 12, jnp.where(state.room_id == 12, 13, jnp.where(state.room_id == 13, 14, jnp.where(state.room_id == 17, 18, jnp.where(state.room_id == 19, 20, jnp.where(state.room_id == 20, 21, jnp.where(state.room_id == 21, 22, jnp.where(state.room_id == 22, 23, jnp.where(state.room_id == 31, 32, jnp.where(state.room_id == 30, 31, jnp.where(state.room_id == 28, 29, jnp.where(state.room_id == 27, 28, jnp.where(state.room_id == 25, 26, jnp.where(state.room_id == 26, 27, jnp.where(state.room_id == 24, 25, state.room_id)))))))))))))))))),
                                jnp.where(transition_down, 
                                          jnp.where(state.room_id == 3, 11, jnp.where(state.room_id == 4, 12, jnp.where(state.room_id == 5, 13, jnp.where(state.room_id == 10, 18, jnp.where(state.room_id == 11, 19, jnp.where(state.room_id == 12, 20, jnp.where(state.room_id == 13, 21, jnp.where(state.room_id == 14, 22, jnp.where(state.room_id == 23, 31, jnp.where(state.room_id == 22, 30, jnp.where(state.room_id == 20, 28, state.room_id))))))))))),
                                          jnp.where(transition_up, 
                                                    jnp.where(state.room_id == 11, 3, jnp.where(state.room_id == 12, 4, jnp.where(state.room_id == 13, 5, jnp.where(state.room_id == 18, 10, jnp.where(state.room_id == 19, 11, jnp.where(state.room_id == 20, 12, jnp.where(state.room_id == 21, 13, jnp.where(state.room_id == 22, 14, jnp.where(state.room_id == 31, 23, jnp.where(state.room_id == 30, 22, jnp.where(state.room_id == 28, 20, state.room_id))))))))))), 
                                                    state.room_id))))
        def transition_fn(state_in):
            # room_idx = get_room_idx(new_room_id)
            st = state_in.replace(
                global_doors_active=state_in.global_doors_active.at[state_in.room_id].set(state_in.doors_active),
                global_items_active=state_in.global_items_active.at[state_in.room_id].set(state_in.items_active),
                global_enemies_active=state_in.global_enemies_active.at[state_in.room_id].set(state_in.enemies_active)
            )
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
            new_py = temp_py - num_to_push

            return st.replace(
                player_x=new_px,
                player_y=new_py,
                last_ladder=jnp.array(-1, dtype=jnp.int32),
                last_rope=jnp.array(-1, dtype=jnp.int32),
                entry_x=new_px,
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
        })
        
    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def render(self, state: MontezumaRevengeState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: MontezumaRevengeState) -> MontezumaRevengeObservation:
        player_obs = ObjectObservation.create(
            x=jnp.array([state.player_x]),
            y=jnp.array([state.player_y]),
            width=jnp.array([self.consts.PLAYER_WIDTH]),
            height=jnp.array([self.consts.PLAYER_HEIGHT]),
            active=jnp.array([1])
        )
        
        enemies_obs = ObjectObservation.create(
            x=state.enemies_x + 1,
            y=state.enemies_y + 1 - jnp.where(state.enemies_bouncing == 1, self.consts.BOUNCE_OFFSETS[jnp.mod(state.frame_count // 4, 22)], 0),
            width=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 6),
            height=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 14),
            active=state.enemies_active
        )
        
        items_obs = ObjectObservation.create(
            x=state.items_x,
            y=state.items_y,
            width=jnp.full(self.consts.MAX_ITEMS_PER_ROOM, 6),
            height=jnp.full(self.consts.MAX_ITEMS_PER_ROOM, 8),
            active=state.items_active
        )
        
        conveyors_obs = ObjectObservation.create(
            x=state.conveyors_x,
            y=state.conveyors_y,
            width=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 40),
            height=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 5),
            active=state.conveyors_active
        )

        doors_obs = ObjectObservation.create(
            x=state.doors_x,
            y=state.doors_y,
            width=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 4),
            height=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 38),
            active=state.doors_active
        )

        ropes_obs = ObjectObservation.create(
            x=state.ropes_x,
            y=state.ropes_top,
            width=jnp.full(self.consts.MAX_ROPES_PER_ROOM, 1),
            height=state.ropes_bottom - state.ropes_top,
            active=state.ropes_active
        )
        
        platforms_obs = ObjectObservation.create(
            x=state.platforms_x,
            y=state.platforms_y,
            width=state.platforms_width,
            height=jnp.full(self.consts.MAX_PLATFORMS_PER_ROOM, 4),
            active=state.platforms_active
        )

        return MontezumaRevengeObservation(player=player_obs, enemies=enemies_obs, items=items_obs, conveyors=conveyors_obs, doors=doors_obs, ropes=ropes_obs, platforms=platforms_obs)
    
    def _get_info(self, state: MontezumaRevengeState) -> MontezumaRevengeInfo:
        return MontezumaRevengeInfo(lives=state.lives, room_id=state.room_id)

    def _get_reward_from_scores(self, previous_score: jnp.ndarray, score: jnp.ndarray) -> float:
        return jnp.sum(score - previous_score).astype(jnp.float32)

    def _get_reward(self, previous_state: MontezumaRevengeState, state: MontezumaRevengeState) -> float:
        """Mod-pipeline contract: reward is computed from (previous_state, state)."""
        return self._get_reward_from_scores(previous_state.score, state.score)

    def _get_done(self, state: MontezumaRevengeState) -> bool:
        fell_off_bonus = jnp.logical_and(state.room_id == 24, state.player_y >= 148)
        return jnp.logical_or(state.lives < 0, fell_off_bonus)
