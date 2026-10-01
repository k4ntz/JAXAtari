import jax
import jax.numpy as jnp
from functools import partial
from flax import struct

import os

from jaxatari.rendering import jax_rendering_utils as render_utils

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.montezuma_revenge.core import MontezumaRevengeState

# --- Gameplay & Ability Mods ---

class InfiniteAmuletMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to keep the amulet active forever.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        return new_state.replace(
            amulet_time=jnp.array(660, dtype=jnp.int32),
            inventory=new_state.inventory.at[3].set(1)
        )

class SuperJumpMod(JaxAtariInternalModPlugin):
    """
    Internal mod to increase jump height.
    """
    constants_overrides = {
        "JUMP_Y_OFFSETS": jnp.array([3, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3, -3], dtype=jnp.int32)
    }

class FastPlayerMod(JaxAtariInternalModPlugin):
    """
    Internal mod to increase player speed.
    """
    constants_overrides = {
        "PLAYER_SPEED": 2
    }

class NoFallDamageMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable fall damage.
    """
    constants_overrides = {
        "MAX_FALL_DISTANCE": 255
    }

# --- Utility & Visual Mods ---

class RevealMapMod(JaxAtariInternalModPlugin):
    """
    Internal mod to make dark rooms always visible, by granting the player 
    the torch effect during rendering.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_pre_render(self, state: MontezumaRevengeState) -> MontezumaRevengeState:
        return state.replace(
            inventory=state.inventory.at[2].set(1)
        )

class DebugHudMod(JaxAtariInternalModPlugin):
    """
    Internal mod to display debug info (Room ID, X, Y) on the HUD.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster: jnp.ndarray, state: MontezumaRevengeState) -> jnp.ndarray:
        jr = self._env.renderer.jr
        masks = self._env.renderer.digit_masks
        
        # In MontezumaRevenge, masks[0] is 'digit_none', masks[1] is '0', etc.
        # So we add 1 to the digits to get the correct sprite index.
        
        # Room ID
        room_digits = jr.int_to_digits(state.room_id, max_digits=2) + 1
        raster = jr.render_label(raster, 10, 10, room_digits, masks, 7, 2)
        
        # Player X
        x_digits = jr.int_to_digits(state.player_x, max_digits=3) + 1
        raster = jr.render_label(raster, 10, 20, x_digits, masks, 7, 3)
        
        # Player Y
        y_digits = jr.int_to_digits(state.player_y, max_digits=3) + 1
        raster = jr.render_label(raster, 10, 30, y_digits, masks, 7, 3)
        
        return raster

class NoEnemiesMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to immediately remove any enemies in the current room.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        return new_state.replace(
            enemies_active=jnp.zeros_like(new_state.enemies_active)
        )

class CenterBouncingSkullMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make the rolling skull (type 1) jump vertically at the center of the screen.
    Forces its X position to 77, enables bouncing, and removes horizontal direction.
    Only applies if there is exactly one skull of type 1 in the room.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_skull_type_1 = new_state.enemies_type == 1
        num_skulls = jnp.sum(jnp.logical_and(new_state.enemies_active == 1, is_skull_type_1))
        is_single_skull_room = num_skulls == 1
        
        is_target = jnp.logical_and(is_single_skull_room, is_skull_type_1)
        
        # Center X is approximately 77 (160 / 2 - 6 / 2)
        new_enemies_x = jnp.where(is_target, 77, new_state.enemies_x)
        
        # Enable bouncing for vertical jumping
        new_enemies_bouncing = jnp.where(is_target, 1, new_state.enemies_bouncing)
        
        # Disable horizontal movement
        new_enemies_direction = jnp.where(is_target, 0, new_state.enemies_direction)
        
        return new_state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing,
            enemies_direction=new_enemies_direction
        )

class RollingSkullsMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make the two skulls that usually bump (bounce) roll instead,
    and slightly augment the space between them.
    Only applies if there are exactly two skulls of type 1 in the room.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_skull_type_1 = new_state.enemies_type == 1
        num_skulls = jnp.sum(jnp.logical_and(new_state.enemies_active == 1, is_skull_type_1))
        is_double_skull_room = num_skulls == 2
        
        is_target = jnp.logical_and(is_double_skull_room, is_skull_type_1)
        
        # Apply initial position change when entering a room with 2 skulls
        just_entered_room = new_state.room_id != prev_state.room_id
        should_shift = jnp.logical_and(just_entered_room, is_double_skull_room)
        
        # General shift logic: move them further apart by 8 pixels each
        x0 = new_state.enemies_x[0]
        x1 = new_state.enemies_x[1]
        new_x0 = jnp.where(x0 < x1, x0 - 8, x0 + 8)
        new_x1 = jnp.where(x1 < x0, x1 - 8, x1 + 8)
        
        # We assume the skulls are at index 0 and 1 (standard in M2)
        shifted_x = new_state.enemies_x.at[0].set(new_x0).at[1].set(new_x1)
        new_enemies_x = jnp.where(should_shift, shifted_x, new_state.enemies_x)
        
        # Disable bouncing for rolling animation and no vertical jump
        new_enemies_bouncing = jnp.where(is_target, 0, new_state.enemies_bouncing)
        
        return new_state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        is_skull_type_1 = state.enemies_type == 1
        num_skulls = jnp.sum(jnp.logical_and(state.enemies_active == 1, is_skull_type_1))
        is_double_skull_room = num_skulls == 2
        
        is_target = jnp.logical_and(is_double_skull_room, is_skull_type_1)
        
        x0 = state.enemies_x[0]
        x1 = state.enemies_x[1]
        new_x0 = jnp.where(x0 < x1, x0 - 8, x0 + 8)
        new_x1 = jnp.where(x1 < x0, x1 - 8, x1 + 8)
        
        new_enemies_x = jnp.where(is_double_skull_room, 
                                  state.enemies_x.at[0].set(new_x0).at[1].set(new_x1), 
                                  state.enemies_x)
        
        new_enemies_bouncing = jnp.where(is_target, 0, state.enemies_bouncing)
        
        state = state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing
        )
        return obs, state


class MovingSnakesMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make snakes move horizontally in all rooms.
    They will move on the x axis from 20 to 140.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_snake = new_state.enemies_type == 4
        
        new_enemies_direction = jnp.where(is_snake,
                                          jnp.where(new_state.enemies_direction == 0, 1, new_state.enemies_direction),
                                          new_state.enemies_direction)
        
        new_eminx = jnp.where(is_snake, 20, new_state.enemies_min_x)
        new_emaxx = jnp.where(is_snake, 140, new_state.enemies_max_x)
        
        # Ensure their X position is strictly within bounds so they don't get stuck bouncing on speed=0 frames
        new_enemies_x = jnp.where(is_snake, 
                                  jnp.clip(new_state.enemies_x, 21, 139), 
                                  new_state.enemies_x)

        return new_state.replace(
            enemies_direction=new_enemies_direction,
            enemies_min_x=new_eminx,
            enemies_max_x=new_emaxx,
            enemies_x=new_enemies_x
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        is_snake = state.enemies_type == 4
        
        new_enemies_direction = jnp.where(is_snake,
                                          jnp.where(state.enemies_direction == 0, 1, state.enemies_direction),
                                          state.enemies_direction)
        
        new_eminx = jnp.where(is_snake, 20, state.enemies_min_x)
        new_emaxx = jnp.where(is_snake, 140, state.enemies_max_x)

        new_enemies_x = jnp.where(is_snake, 
                                  jnp.clip(state.enemies_x, 21, 139), 
                                  state.enemies_x)

        state = state.replace(
            enemies_direction=new_enemies_direction,
            enemies_min_x=new_eminx,
            enemies_max_x=new_emaxx,
            enemies_x=new_enemies_x
        )
        return obs, state


class JumpingSpidersMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make all spiders (type 3) jump (bounce).
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_spider = new_state.enemies_type == 3
        
        new_enemies_bouncing = jnp.where(is_spider, 1, new_state.enemies_bouncing)
        
        return new_state.replace(
            enemies_bouncing=new_enemies_bouncing
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        is_spider = state.enemies_type == 3
        
        new_enemies_bouncing = jnp.where(is_spider, 1, state.enemies_bouncing)
        
        state = state.replace(
            enemies_bouncing=new_enemies_bouncing
        )
        return obs, state

class SwordKillBonusMod(JaxAtariInternalModPlugin):
    """
    Internal mod to provide bonus points if the player kills an enemy with the sword.
    It overrides the KILL_ENEMY_REWARD constant to 300.
    """
    constants_overrides = {
        "KILL_ENEMY_REWARD": 300
    }

class ThreeSwordsMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to start the player with 3 swords.
    Since the base game now natively supports up to 3 swords in the inventory,
    we simply set it at reset.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        # Give 3 swords natively
        new_inventory = state.inventory.at[1].set(3)
        
        state = state.replace(
            inventory=new_inventory
        )
        return obs, state

class DifferentStart(JaxAtariInternalModPlugin):
    """
    Internal mod to provide bonus points if the player kills an enemy with the sword.
    It overrides the KILL_ENEMY_REWARD constant to 300.
    """
    constants_overrides = {
        "CUSTOM_ROOMS": True,
        "INITIAL_ROOM_ID": 5
    }

class ChangeEnemyActivity(JaxAtariPostStepModPlugin):
    """
    Post-step mod to immediately remove any enemies in the current room.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):

        # remove enemies
        gea = jnp.zeros_like(state.global_enemies_active)
        gea = gea.at[4, 0].set(0) # New 4 (Mid)
        gea = gea.at[5, 0].set(0) # New 5 (Right)
        gea = gea.at[5, 1].set(0)
        gea = gea.at[11, 0].set(1)
        gea = gea.at[10, 0].set(0)
        gea = gea.at[10, 1].set(0)
        gea = gea.at[12, 0].set(0)
        gea = gea.at[18, 0].set(0)
        gea = gea.at[18, 1].set(0)
        gea = gea.at[20, 0].set(0)
        gea = gea.at[20, 1].set(0)
        gea = gea.at[22, 0].set(0)
        gea = gea.at[31, 0].set(0) # Snake in Room 31
        gea = gea.at[30, 0].set(0) # Spider in Room 30
        gea = gea.at[27, 0].set(0) # Skull in Room 27 (ROOM_3_3)

        # add and remove doores
        gda = state.global_doors_active
        gda = gda.at[4, 1].set(0)
        gda = gda.at[5, 0].set(1)
        gda = gda.at[26, 1].set(0)

        # add items
        gia= state.global_items_active
        gia = gia.at[5, 0].set(1)
        gia = gia.at[12, 0].set(0)

        # change item types
        git = state.global_items_type
        git = git.at[3, 0].set(3)
        git = git.at[19, 0].set(4)

        # overwrite game state
        new_state = state.replace(
            global_doors_active=gda,
            doors_active=gda[state.room_id],
            global_enemies_active=gea,
            enemies_active=gea[state.room_id],
            global_items_active=gia,
            items_active=gia[state.room_id],
            global_items_type=git,
        )

        return obs, new_state


class ChangeCollision(JaxAtariInternalModPlugin):
    """
    Internal mod to .
    """
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
    
    # New 18: Level 2, col 2 (corresponds to ROOM_2_1 in M1)
    room_col_2_2 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
    # room_col_2_2 = room_col_2_2.at[6:48, 156:160].set(1) # Right wall removed
    
    sprite_path_3 = os.path.join(sprite_path, "backgrounds", "room_0_collision_level_2.npy")
    col_map_3 = jnp.load(sprite_path_3)[:149, :, 0]
    room_col_2_1 = jnp.where(col_map_3 > 0, 1, 0).astype(jnp.int32)
    room_col_2_1 = room_col_2_1.at[6:, 0:4].set(1) # Left wall

    sprite_path_4 = os.path.join(sprite_path, "backgrounds", "pitroom_collision_map.npy")
    col_map_4 = jnp.load(sprite_path_4)[:149, :, 0]
    room_col_2_3 = jnp.where(col_map_4 > 0, 1, 0).astype(jnp.int32)
    # room_col_2_3 = room_col_2_3.at[6:48, 0:4].set(1) # Left wall
    room_col_2_3 = room_col_2_3.at[6:48, 156:160].set(1) # Right wall

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

    # New 25: Level 3, col 1 (corresponds to ROOM_3_1 in M1)
    room_col_3_1 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)

    # New 26: Level 3, col 2 (corresponds to ROOM_3_2 in M1)
    room_col_3_2 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
    room_col_3_2 = room_col_3_2.at[6:48, 156:160].set(1) # Right wall

    # New 24: Bonus Room (corresponds to ROOM_3_0 in M1)
    sprite_path_9 = os.path.join(sprite_path, "backgrounds", "bonus_room_collision_map.npy")
    col_map_9 = jnp.load(sprite_path_9)[:149, :, 0]
    room_col_3_0 = jnp.zeros((149, 160), dtype=jnp.int32)
    room_col_3_0 = room_col_3_0.at[:col_map_9.shape[0], :].set(jnp.where(col_map_9 > 0, 1, 0))
    room_col_3_0 = room_col_3_0.at[6:148, 0:4].set(1) # Left wall
    room_col_3_0 = room_col_3_0.at[6:148, 156:160].set(1) # Right wall
    room_col_3_0 = room_col_3_0.at[47:50, :].set(1) # Thin invisible horizontal platform at Y=47

    attribute_overrides = {
        "ROOM_COLLISION_MAPS": jnp.stack([room_col_0_3, room_col_0_4, room_col_0_5, \
                                          room_col_1_3, room_col_1_2, room_col_1_4, jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), \
                                          room_col_2_2, room_col_2_1, room_col_2_3, jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), \
                                          jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), jnp.zeros_like(room_col_0_3), room_col_3_3, jnp.zeros_like(room_col_0_3), room_col_3_1, room_col_3_2, room_col_3_0])
    }


class LadderPits(JaxAtariInternalModPlugin):
    """
    Fill or Remove pits for the ladders.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster: jnp.ndarray, state: MontezumaRevengeState) -> jnp.ndarray:
        renderer = self._env.renderer

        room_y = 47

        # fill pit in room five
        raster = jax.lax.cond(
            state.room_id == 5,
            # lambda r: stamp_room(r, jnp.concatenate([renderer.SHAPE_MASKS["room_bg_0"][:48], renderer.SHAPE_MASKS["room_bg_level2_base"][48:]], axis=0)),
            lambda r: jnp.concatenate([raster[:(room_y+48)], renderer.SHAPE_MASKS["room_bg_level2_base"][48:], renderer.SHAPE_MASKS["room_bg_level2_base"][135:]], axis=0),
            lambda r: r,
            raster,
        )

        # jax.debug.print("{x}", x=raster[80, 80])

        mask_l2 = jnp.where(renderer.SHAPE_MASKS["room_bg_level2_base"] == 1, renderer.LEVEL2_PLATFORM_ID, renderer.SHAPE_MASKS["room_bg_level2_base"])

        def draw_ladder(r_in):
            return jnp.where(raster == 20, raster, r_in)

        def redraw_player(r_in):
            return jnp.where(jnp.logical_and(raster > 8, raster < 20), raster, r_in)

        def clear_hole(r_in, y0, y1, x0, x1):
            pos = jnp.array([[x0, room_y + y0]])
            size = jnp.array([[x1 - x0, y1 - y0]])
            return renderer.jr.draw_rects(r_in, pos, size, jnp.uint8(0))
        
        def stamp_room(r_in, mask):
            return renderer.jr.render_at(r_in, 0, room_y, mask)

        def remove_wall(r_in, x_min, x_max):
            return r_in.at[53:94, x_min:x_max].set(jnp.uint8(0))

        def add_wall(r_in):
            return r_in.at[53:95, 156:160].set(jnp.uint8(20))
        
        raster = jax.lax.cond(
            state.room_id == 18,
            lambda r: redraw_player(remove_wall(draw_ladder(clear_hole(stamp_room(r, mask_l2), 48, 149, 72, 88)), 156, 160)),
            lambda r: r,
            raster,
        )
        
        raster = jax.lax.cond(
            state.room_id == 19,
            lambda r: redraw_player(remove_wall(add_wall(raster), 0, 6)),
            lambda r: r,
            raster,
        )
        
        raster = jax.lax.cond(
            state.room_id == 26,
            lambda r: add_wall(raster),
            lambda r: r,
            raster,
        )

        return raster
