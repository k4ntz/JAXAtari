import jax
import jax.image as jim
import jax.numpy as jnp
from functools import partial
import os

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.games.montezuma_revenge.core import MontezumaRevengeConstants, MontezumaRevengeState

class MontezumaRevengeRenderer(JAXGameRenderer):
    def __init__(self, consts: MontezumaRevengeConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or MontezumaRevengeConstants()
        
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        # Keep rendering logic in native 210x160 RGB space, then optionally apply
        # final-frame downscaling / grayscale conversion from self.config.
        internal_config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
            downscale=None,
        )
        self.jr = render_utils.JaxRenderingUtils(internal_config)
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "montezuma")
        
        # Transparent background base for the 210x160 raster
        bg_data = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 4), dtype=jnp.uint8)
        bg_data = bg_data.at[:, :, 3].set(jnp.uint8(255)) # Opaque black
        
        final_asset_config = [
            {'name': 'bg', 'type': 'background', 'data': bg_data},
            {'name': 'room_bg_0', 'type': 'single', 'file': 'backgrounds/base_sprite_level_0.npy', 'transpose': False},
            {'name': 'room_bg_1', 'type': 'single', 'file': 'backgrounds/mid_room_level_0.npy', 'transpose': False},
            {'name': 'room_bg_2', 'type': 'single', 'file': 'backgrounds/base_sprite_level_1.npy', 'transpose': False},
            {'name': 'room_bg_3', 'type': 'single', 'file': 'backgrounds/mid_room_level_1.npy', 'transpose': False},
            {'name': 'room_bg_4', 'type': 'single', 'file': 'backgrounds/base_sprite_level_3.npy', 'transpose': False},
            {'name': 'room_bg_level2_base', 'type': 'single', 'file': 'backgrounds/base_sprite_level_2.npy', 'transpose': False},
            {'name': 'room_bg_level2_room0', 'type': 'single', 'file': 'backgrounds/room_0_level_2.npy', 'transpose': False},
            {'name': 'room_bg_level2_room6', 'type': 'single', 'file': 'backgrounds/room_6_level_2.npy', 'transpose': False},
            {'name': 'room_bg_level2_pit', 'type': 'single', 'file': 'backgrounds/pitroom_level_2.npy', 'transpose': False},
            {'name': 'room_bg_pit_original', 'type': 'single', 'file': 'backgrounds/pitroom.npy', 'transpose': False},
            {'name': 'room_bg_bonus', 'type': 'single', 'file': 'backgrounds/bonus_room_sprite.npy', 'transpose': False},
            {
                'name': 'player', 'type': 'group',
                'files': [
                    'player/player_sprite.npy',
                    'player/walking_0.npy',
                    'player/walking_1.npy',
                    'player/ladder_climb1.npy',
                    'player/ladder_climb2.npy',
                    'player/rope_climb_0.npy',
                    'player/rope_climb_1.npy',
                    'player/player_jump.npy',
                    'player/player_splosh_0.npy',
                    'player/player_splosh_1.npy',
                    'player/splutter_0.npy',
                    'player/splutter_1.npy'
                ]
            },
            {
                'name': 'skull', 'type': 'group',
                'files': [f'enemies/skull_cycle/skull_{i}.npy' for i in range(1, 17)]
            },
            {
                'name': 'spider', 'type': 'single', 'file': 'enemies/spidder.npy', 'transpose': False
            },
            {
                'name': 'snake', 'type': 'group',
                'files': ['enemies/snake_0.npy', 'enemies/snake_1.npy']
            },
            {'name': 'key', 'type': 'single', 'file': 'items/key.npy', 'transpose': False},
            {'name': 'gem', 'type': 'single', 'file': 'items/gem.npy', 'transpose': False},
            {'name': 'amulet', 'type': 'single', 'file': 'items/amulet.npy', 'transpose': False},
            {'name': 'sword', 'type': 'single', 'file': 'items/sword.npy', 'transpose': False},
            {
                'name': 'torch', 'type': 'group',
                'files': ['items/torch_1.npy', 'items/torch_2.npy']
            },
            {'name': 'door', 'type': 'single', 'file': 'door.npy', 'transpose': False},
            {'name': 'conveyor', 'type': 'single', 'file': 'conveyor_belt.npy', 'transpose': False},
            {
                'name': 'dropout_floor',
                'type': 'group',
                'files': ['other_dropout_floor.npy', 'other_dropout_floor2.npy']
            },
            {
                'name': 'pitroom_dropout_floor',
                'type': 'group',
                'files': ['pitroom_dropout_floor.npy', 'pitroom_dropout_floor2.npy']
            },
            {'name': 'life', 'type': 'single', 'file': 'life_sprite.npy', 'transpose': False},
            {'name': 'digit_0', 'type': 'single', 'file': 'digits/digit_0.npy', 'transpose': False},
            {'name': 'digit_1', 'type': 'single', 'file': 'digits/digit_1.npy', 'transpose': False},
            {'name': 'digit_2', 'type': 'single', 'file': 'digits/digit_2.npy', 'transpose': False},
            {'name': 'digit_3', 'type': 'single', 'file': 'digits/digit_3.npy', 'transpose': False},
            {'name': 'digit_4', 'type': 'single', 'file': 'digits/digit_4.npy', 'transpose': False},
            {'name': 'digit_5', 'type': 'single', 'file': 'digits/digit_5.npy', 'transpose': False},
            {'name': 'digit_6', 'type': 'single', 'file': 'digits/digit_6.npy', 'transpose': False},
            {'name': 'digit_7', 'type': 'single', 'file': 'digits/digit_7.npy', 'transpose': False},
            {'name': 'digit_8', 'type': 'single', 'file': 'digits/digit_8.npy', 'transpose': False},
            {'name': 'digit_9', 'type': 'single', 'file': 'digits/digit_9.npy', 'transpose': False},
            {'name': 'digit_none', 'type': 'single', 'file': 'digits/digit_none.npy', 'transpose': False},
        ]
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        # Accurate ladder color for Difficulty 1, Layer 1
        self.LADDER_COLOR = jnp.array([66, 158, 130], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LADDER_COLOR[None, :]], axis=0)
        self.LADDER_ID = self.PALETTE.shape[0] - 1

        # Accurate ladder color for Difficulty 1, Layer 2 (purple)
        self.LADDER_COLOR_L2 = jnp.array([104, 25, 154], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LADDER_COLOR_L2[None, :]], axis=0)
        self.LADDER_ID_L2 = self.PALETTE.shape[0] - 1
        
        # Blue ladder color for long ladders
        self.BLUE_LADDER_COLOR = jnp.array([24, 59, 157], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.BLUE_LADDER_COLOR[None, :]], axis=0)
        self.BLUE_LADDER_ID = self.PALETTE.shape[0] - 1

        # Yellow ladder color for long ladders
        self.YELLOW_LADDER_COLOR = jnp.array([204, 216, 110], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.YELLOW_LADDER_COLOR[None, :]], axis=0)
        self.YELLOW_LADDER_ID = self.PALETTE.shape[0] - 1

        # Accurate door color
        self.DOOR_COLOR = jnp.array([232, 204, 99], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.DOOR_COLOR[None, :]], axis=0)
        self.DOOR_ID = self.PALETTE.shape[0] - 1
        
        # Laser color
        self.LASER_COLOR = jnp.array([101, 111, 228], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LASER_COLOR[None, :]], axis=0)
        self.LASER_ID = self.PALETTE.shape[0] - 1
        
        # Level 2 platform blue color
        self.LEVEL2_PLATFORM_COLOR = jnp.array([45, 87, 176], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LEVEL2_PLATFORM_COLOR[None, :]], axis=0)
        self.LEVEL2_PLATFORM_ID = self.PALETTE.shape[0] - 1

        # Room 31 (ROOM_3_7) specific colors
        self.ORANGE_LADDER_COLOR = jnp.array([213, 130, 74], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.ORANGE_LADDER_COLOR[None, :]], axis=0)
        self.ORANGE_LADDER_ID = self.PALETTE.shape[0] - 1

        self.DEEP_BLUE_PLATFORM_COLOR = jnp.array([24, 26, 167], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.DEEP_BLUE_PLATFORM_COLOR[None, :]], axis=0)
        self.DEEP_BLUE_PLATFORM_ID = self.PALETTE.shape[0] - 1

        # Gray color for neutralized enemies
        self.GRAY_COLOR = jnp.array([142, 142, 142], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.GRAY_COLOR[None, :]], axis=0)
        self.GRAY_ID = self.PALETTE.shape[0] - 1

        # Sarlacc pit colors for ROOM_2_3
        self.PIT_RGB_BASE = jnp.array([210, 164, 74], dtype=jnp.uint8)
        self.PIT_PATTERN = jnp.array([
            [2, 1, 0, 1, 2, 1, 0, 1, 2],
            [1, 0, 1, 2, 1, 0, 1, 2, 3],
            [0, 1, 2, 1, 0, 1, 2, 3, 2],
            [1, 2, 1, 0, 1, 2, 3, 2, 1]
        ], dtype=jnp.int32)
        
        self.PIT_COLORS = []
        for i in range(8):
            color_index = i
            r = int(-(0.65*color_index**2)-(14*color_index)+210)
            r = max(r, 0)
            g = int(-(color_index**2)-(19*color_index)+164)
            g = max(g, 0)
            b = int(-(0.88*color_index**2)-(11.25*color_index)+74)
            b = max(b, 0)
            self.PIT_COLORS.append([r, g, b])
        
        self.PIT_COLOR_IDS = []
        for c in self.PIT_COLORS:
            self.PALETTE = jnp.concatenate([self.PALETTE, jnp.array(c, dtype=jnp.uint8)[None, :]], axis=0)
            self.PIT_COLOR_IDS.append(self.PALETTE.shape[0] - 1)
        self.PIT_COLOR_IDS = jnp.array(self.PIT_COLOR_IDS)
        
        door_mask = self.SHAPE_MASKS["door"]
        self.SHAPE_MASKS["door"] = jnp.where(door_mask != self.jr.TRANSPARENT_ID, self.DOOR_ID, self.jr.TRANSPARENT_ID)

        self.digit_masks = jnp.stack([
            self.SHAPE_MASKS["digit_none"],
            self.SHAPE_MASKS["digit_0"],
            self.SHAPE_MASKS["digit_1"],
            self.SHAPE_MASKS["digit_2"],
            self.SHAPE_MASKS["digit_3"],
            self.SHAPE_MASKS["digit_4"],
            self.SHAPE_MASKS["digit_5"],
            self.SHAPE_MASKS["digit_6"],
            self.SHAPE_MASKS["digit_7"],
            self.SHAPE_MASKS["digit_8"],
            self.SHAPE_MASKS["digit_9"],
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_pre_render(self, state: MontezumaRevengeState) -> MontezumaRevengeState:
        """Hook called at the very beginning of render() to allow state modification."""
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster: jnp.ndarray, state: MontezumaRevengeState) -> jnp.ndarray:
        """Hook called at the very end of render() to allow drawing on top of the final frame."""
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: MontezumaRevengeState) -> jnp.ndarray:
        # Apply pre-render hook
        state = self._render_hook_pre_render(state)

        # Start with solid black background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Draw Room Background
        mask_0 = self.SHAPE_MASKS["room_bg_0"][:149, ...]
        mask_1 = self.SHAPE_MASKS["room_bg_1"][:149, ...]
        mask_2 = self.SHAPE_MASKS["room_bg_2"][:149, ...]
        mask_3 = self.SHAPE_MASKS["room_bg_3"][:149, ...]
        mask_4 = self.SHAPE_MASKS["room_bg_4"][:149, ...]
        
        mask_0_modified = mask_0.at[147:149, 72:88].set(jnp.uint8(0))
        room_bg_mask = jnp.where(state.room_id == 4, mask_1, mask_0_modified)
        
        mask_3_modified = mask_3.at[147:149, 72:88].set(jnp.uint8(0))
        room_bg_mask = jnp.where(state.room_id == 12, mask_3_modified, room_bg_mask)
        is_mask4_room = jnp.isin(state.room_id, jnp.array([25, 26, 28, 30, 32]))
        room_bg_mask = jnp.where(is_mask4_room, mask_4, room_bg_mask)
        mask_2_modified = mask_2.at[48:149, 72:88].set(jnp.uint8(0))
        room_bg_mask = jnp.where(jnp.logical_or(state.room_id == 11, jnp.logical_or(state.room_id == 10, state.room_id == 14)), mask_2_modified, room_bg_mask)
        room_bg_mask = jnp.where(state.room_id == 13, mask_2, room_bg_mask)

        # Level 2 rooms
        mask_l2 = self.SHAPE_MASKS["room_bg_level2_base"][:149, ...]
        mask_l2 = jnp.where(mask_l2 == 1, self.LEVEL2_PLATFORM_ID, mask_l2)
        mask_l2_room0 = self.SHAPE_MASKS["room_bg_level2_room0"][:149, ...]
        mask_l2_room0 = jnp.where(mask_l2_room0 == 1, self.LEVEL2_PLATFORM_ID, mask_l2_room0)
        mask_l2_room6 = self.SHAPE_MASKS["room_bg_level2_room6"]
        # Pad to 149 if height is 147
        padding = 149 - mask_l2_room6.shape[0]
        mask_l2_room6 = jnp.pad(mask_l2_room6, ((0, padding), (0, 0)), mode='constant', constant_values=0)
        mask_l2_room6 = jnp.where(mask_l2_room6 == 1, self.LEVEL2_PLATFORM_ID, mask_l2_room6)
        mask_l2_pit = self.SHAPE_MASKS["room_bg_level2_pit"][:149, ...]
        mask_l2_pit = jnp.where(mask_l2_pit == 1, self.LEVEL2_PLATFORM_ID, mask_l2_pit)
        mask_pit_original = self.SHAPE_MASKS["room_bg_pit_original"][:149, ...]
        
        mask_l2_hole = mask_l2.at[48:, 72:88].set(jnp.uint8(0))
        
        room_bg_mask = jnp.where(state.room_id == 18, mask_l2, room_bg_mask)
        room_bg_mask = jnp.where(state.room_id == 17, mask_l2_room0, room_bg_mask)
        room_bg_mask = jnp.where(state.room_id == 19, mask_l2_pit, room_bg_mask)
        is_pit_room = jnp.any(state.room_id == jnp.array([31, 27, 29]))
        room_bg_mask = jnp.where(is_pit_room, mask_pit_original, room_bg_mask)
        room_bg_mask = jnp.where(jnp.logical_or(state.room_id == 20, state.room_id == 22), mask_l2_hole, room_bg_mask)
        room_bg_mask = jnp.where(state.room_id == 21, mask_l2, room_bg_mask)
        room_bg_mask = jnp.where(state.room_id == 23, mask_l2_room6, room_bg_mask)

        # Bonus Room (ROOM_3_0)
        mask_bonus = self.SHAPE_MASKS["room_bg_bonus"]
        padding_b = 149 - mask_bonus.shape[0]
        mask_bonus = jnp.pad(mask_bonus, ((0, padding_b), (0, 0)), mode='constant', constant_values=0)
        room_bg_mask = jnp.where(state.room_id == 24, mask_bonus, room_bg_mask)

        # DARK ROOM LOGIC
        is_dark_room = jnp.isin(state.room_id, jnp.array([25, 26, 27, 28, 29, 30, 31, 32]))
        has_torch = state.inventory[2] == 1
        is_rendered_dark = jnp.logical_and(is_dark_room, jnp.logical_not(has_torch))
        room_bg_mask = jnp.where(is_rendered_dark, jnp.zeros_like(room_bg_mask), room_bg_mask)

        # Add lava rendering for ROOM_2_3 (room_id 19) and ROOM_3_7 (room_id 31)
        lava_y_start = 76
        lava_y_end = 124 # gap ends at 123
        anim_frame = jnp.mod(state.frame_count // 8, 4)
        row_indices = jnp.arange(lava_y_end - lava_y_start)
        band_indices = row_indices // 2
        color_indices = (band_indices // 9) + self.PIT_PATTERN[anim_frame][jnp.mod(band_indices, 9)]
        band_color_ids = self.PIT_COLOR_IDS[color_indices]
        lava_mask = jnp.tile(band_color_ids[:, None], (1, 160))
        
        # Apply lava only to room 19 and 31, and only in the empty (black) areas between lava_y_start and lava_y_end
        lava_region = room_bg_mask[lava_y_start:lava_y_end, :]
        is_black = jnp.all(self.PALETTE[lava_region] == 0, axis=-1)
        new_lava_region = jnp.where(is_black, lava_mask, lava_region)
        room_bg_mask_with_lava = room_bg_mask.at[lava_y_start:lava_y_end, :].set(new_lava_region.astype(jnp.uint8))
        is_lava_room = jnp.any(state.room_id == jnp.array([19, 31, 27, 29]))
        room_bg_mask = jnp.where(is_lava_room, room_bg_mask_with_lava, room_bg_mask)
        
        # Add walls for side rooms Level 0 and Level 1 and Level 2
        # Use LEVEL2_PLATFORM_ID for Level 2 walls (rooms 17), LADDER_ID (green) for room 19, and ORANGE_LADDER_ID for room 30
        left_wall_color = jnp.where(state.room_id == 19, self.LADDER_ID,
                                    jnp.where(state.room_id == 30, self.ORANGE_LADDER_ID,
                                              jnp.where(state.room_id == 17, self.LEVEL2_PLATFORM_ID, 1)))
        # Room 3, 10, 19, 30 and 25 walls should only be on top (above floor)
        is_side_room_left = jnp.isin(state.room_id, jnp.array([3, 10, 19, 30]))
        is_side_room_left = jnp.logical_and(is_side_room_left, jnp.logical_not(is_rendered_dark))
        room_bg_mask = jnp.where(is_side_room_left, room_bg_mask.at[6:48, 0:4].set(left_wall_color.astype(jnp.uint8)), room_bg_mask)
        # Other rooms left wall
        is_left_wall_room = jnp.logical_and(state.room_id == 17, jnp.logical_not(is_rendered_dark))
        room_bg_mask = jnp.where(is_left_wall_room, room_bg_mask.at[6:149, 0:4].set(left_wall_color.astype(jnp.uint8)), room_bg_mask)
        
        right_wall_color = jnp.where(state.room_id == 18, self.LADDER_ID,
                                     jnp.where(state.room_id == 29, self.DEEP_BLUE_PLATFORM_ID,
                                               jnp.where(state.room_id == 23, self.LEVEL2_PLATFORM_ID, 1)))
        # Room 5, 14, 18, 32, and 29 walls should only be on top (above floor)
        is_side_room_right = jnp.isin(state.room_id, jnp.array([5, 14, 18, 32, 29]))
        is_side_room_right = jnp.logical_and(is_side_room_right, jnp.logical_not(is_rendered_dark))
        room_bg_mask = jnp.where(is_side_room_right, room_bg_mask.at[6:48, 156:160].set(right_wall_color.astype(jnp.uint8)), room_bg_mask)
        # Other rooms right wall
        is_right_wall_room = jnp.logical_and(state.room_id == 23, jnp.logical_not(is_rendered_dark))
        room_bg_mask = jnp.where(is_right_wall_room, room_bg_mask.at[6:149, 156:160].set(right_wall_color.astype(jnp.uint8)), room_bg_mask)
        
        raster = self.jr.render_at(raster, 0, 47, room_bg_mask)
        
        # Draw Ladders (Vertical Rails + Horizontal Rungs)
        def draw_ladder_accurate(i, r):
            x, top, bottom = state.ladders_x[i], state.ladders_top[i] + 47, state.ladders_bottom[i] + 47
            bottom = jnp.where(jnp.logical_and(state.room_id == 4, state.ladders_bottom[i] == 130), bottom + 3, bottom)
            bottom = jnp.where(jnp.logical_and(state.room_id == 23, state.ladders_bottom[i] == 150), bottom - 3, bottom)
            active = state.ladders_active[i]

            def _draw(raster_in):
                def render_long_ladder(r_in, l_color, bg_color):
                    long_top = top - 1
                    long_height = bottom - long_top
                    
                    bg_pos = jnp.array([[x - 4, long_top]])
                    bg_size = jnp.array([[24, long_height]])
                    r_in = self.jr.draw_rects(r_in, bg_pos, bg_size, bg_color)
                    
                    new_rail_pos = jnp.array([[x, long_top], [x + 16 - 4, long_top]])
                    new_rail_size = jnp.array([[4, long_height], [4, long_height]])
                    new_rung_pos = jnp.array([[x, long_top + 4]])
                    new_rung_size = jnp.array([[16, long_height - 4]])
                    
                    r_in = self.jr.draw_rects(r_in, new_rail_pos, new_rail_size, l_color)
                    return self.jr.draw_ladders(r_in, new_rung_pos, new_rung_size, 2, 5, l_color)

                ladder_width = 16
                # Vertical Rails (4 pixels wide)
                rail_pos = jnp.array([[x, top], [x + ladder_width - 4, top]])
                rail_size = jnp.array([[4, bottom - top], [4, bottom - top]])

                # Horizontal Rungs (2 pixels high, 5 pixels gap)
                # First rung starts at top + 4
                rung_pos = jnp.array([[x, top + 4]])
                rung_size = jnp.array([[ladder_width, bottom - top - 4]])
                
                def draw_l1(r_in):
                    r_in = self.jr.draw_rects(r_in, rail_pos, rail_size, self.LADDER_ID)
                    return self.jr.draw_ladders(r_in, rung_pos, rung_size, 2, 5, self.LADDER_ID)

                def draw_l2(r_in):
                    r_in = self.jr.draw_rects(r_in, rail_pos, rail_size, self.LADDER_ID_L2)
                    return self.jr.draw_ladders(r_in, rung_pos, rung_size, 2, 5, self.LADDER_ID_L2)
                    
                def draw_yellow_l2(r_in):
                    r_in = self.jr.draw_rects(r_in, rail_pos, rail_size, self.YELLOW_LADDER_ID)
                    return self.jr.draw_ladders(r_in, rung_pos, rung_size, 2, 5, self.YELLOW_LADDER_ID)

                def draw_orange(r_in):
                    r_in = self.jr.draw_rects(r_in, rail_pos, rail_size, self.ORANGE_LADDER_ID)
                    return self.jr.draw_ladders(r_in, rung_pos, rung_size, 2, 5, self.ORANGE_LADDER_ID)

                def draw_long(r_in):
                    return render_long_ladder(r_in, self.BLUE_LADDER_ID, self.LADDER_ID)

                def draw_long_l2(r_in):
                    return render_long_ladder(r_in, self.YELLOW_LADDER_ID, self.LADDER_ID_L2)

                is_layer_2 = jnp.logical_or(state.room_id == 11, jnp.logical_or(state.room_id == 10, jnp.logical_or(state.room_id == 12, state.room_id == 14)))
                is_long_ladder = jnp.logical_and(jnp.logical_or(state.room_id == 3, state.room_id == 5), i == 0)
                is_long_ladder_l2 = jnp.logical_and(
                    jnp.logical_or(state.room_id == 12, 
                        jnp.logical_or(state.room_id == 11, 
                            jnp.logical_or(state.room_id == 10, state.room_id == 14)
                        )
                    ), 
                    i == 0
                )
                is_small_yellow = jnp.logical_or(
                    jnp.logical_and(state.room_id == 11, i == 1),
                    jnp.logical_and(state.room_id == 13, i == 0)
                )
                is_room_orange_ladder = jnp.isin(state.room_id, jnp.array([31, 30, 28]))
                
                r_out = jax.lax.cond(is_layer_2, draw_l2, draw_l1, raster_in)
                r_out = jax.lax.cond(is_small_yellow, draw_yellow_l2, lambda r: r_out, r_out)
                r_out = jax.lax.cond(is_long_ladder_l2, draw_long_l2, lambda r: r_out, r_out)
                r_out = jax.lax.cond(is_long_ladder, draw_long, lambda r: r_out, r_out)
                return jax.lax.cond(is_room_orange_ladder, draw_orange, lambda r: r_out, r_out)

            return jax.lax.cond(active == 1, _draw, lambda r_in: r_in, r)
        raster = jax.lax.fori_loop(0, self.consts.MAX_LADDERS_PER_ROOM, draw_ladder_accurate, raster)

        # Draw Ropes
        def draw_rope(i, r):
            x, top, bottom = state.ropes_x[i], state.ropes_top[i] + 47, state.ropes_bottom[i] + 47
            active = state.ropes_active[i]
            
            def _draw(raster_in):
                rail_pos = jnp.array([[x, top]])
                rail_size = jnp.array([[1, bottom - top + 1]])
                return self.jr.draw_rects(raster_in, rail_pos, rail_size, self.DOOR_ID)
                
            return jax.lax.cond(active == 1, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_ROPES_PER_ROOM, draw_rope, raster)
        
        # Draw Lasers
        laser_active_now = jnp.logical_and(jnp.greater_equal(state.laser_cycle, 0), jnp.less(state.laser_cycle, 92))
        laser_offset = jnp.mod(state.laser_cycle, 4)
        def draw_laser(i, r):
            x = state.lasers_x[i]
            active = jnp.logical_and(state.lasers_active[i] == 1, laser_active_now)
            
            def _draw(raster_in):
                # 40 pixels high. We batch 11 stripes max to handle offset properly.
                start_j = jnp.mod(4 - laser_offset, 4) - 4
                k_idx = jnp.arange(11)
                j_vals = start_j + k_idx * 4
                
                valid = jnp.logical_and(j_vals >= 0, j_vals < 40)
                pos_x = jnp.where(valid, x, -1)
                pos_y = jnp.where(valid, 54 + j_vals, -1)
                
                sizes = jnp.where(valid, 4, 1)
                
                pos = jnp.stack([pos_x, pos_y], axis=-1)
                size = jnp.stack([sizes, jnp.ones_like(sizes)], axis=-1)
                
                return self.jr.draw_rects(raster_in, pos, size, self.LASER_ID)
                
            return jax.lax.cond(active, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_LASERS_PER_ROOM, draw_laser, raster)
        

        # Draw Platforms
        platform_active_now = jnp.less(state.platform_cycle, self.consts.PLATFORM_ACTIVE_DURATION)
        def render_platform(i, raster):
            is_pit_room = jnp.isin(state.room_id, jnp.array([19, 27, 29, 31]))
            is_active = jnp.logical_and(state.platforms_active[i] == 1, platform_active_now)

            # Color remap: Use LEVEL2_PLATFORM_ID for Level 2 rooms (17, 18, 19), otherwise LADDER_ID_L2
            is_layer_2_room = jnp.logical_or(state.room_id == 17, jnp.logical_or(state.room_id == 18, state.room_id == 19))
            p_color = jax.lax.select(is_layer_2_room, self.LEVEL2_PLATFORM_ID, self.LADDER_ID_L2)
            is_deep_blue_room = jnp.any(state.room_id == jnp.array([31, 27, 29]))
            p_color = jax.lax.select(is_deep_blue_room, self.DEEP_BLUE_PLATFORM_ID, p_color)

            def _draw_pit(r):
                anim_idx = (state.frame_count // 8) % 2
                mask = self.SHAPE_MASKS["pitroom_dropout_floor"][anim_idx]
                mask = jnp.concatenate([mask, mask[0:1, :]], axis=0) # 7x8
                mask = jnp.where(mask != self.jr.TRANSPARENT_ID, p_color, self.jr.TRANSPARENT_ID)
                num_tiles = state.platforms_width[i] // 8
                def _tile_fn(j, r_in):
                    return self.jr.render_at(r_in, state.platforms_x[i] + j * 8, state.platforms_y[i] + 47, mask)
                return jax.lax.fori_loop(0, num_tiles, _tile_fn, r)

            def _draw_other(r):
                anim_idx = (state.frame_count // 8) % 2
                mask = self.SHAPE_MASKS["dropout_floor"][anim_idx]
                mask = jnp.where(mask != self.jr.TRANSPARENT_ID, p_color, self.jr.TRANSPARENT_ID)
                num_tiles = state.platforms_width[i] // 12
                def _tile_fn(j, r_in):
                    return self.jr.render_at(r_in, state.platforms_x[i] + j * 12, state.platforms_y[i] + 47, mask)
                return jax.lax.fori_loop(0, num_tiles, _tile_fn, r)

            def _draw_active(r):
                return jax.lax.cond(is_pit_room, _draw_pit, _draw_other, r)

            return jax.lax.cond(
                is_active,
                _draw_active,
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_PLATFORMS_PER_ROOM, render_platform, raster)

        # Draw Conveyors
        anim_idx = jnp.less(jnp.mod(state.frame_count, 16), 8)
        def render_conveyor(i, raster):
            mask = self.SHAPE_MASKS["conveyor"]
            # Color remap: Use LADDER_ID_L2 (purple) for Layer 2 rooms, otherwise LADDER_ID (green)
            is_layer_2 = jnp.isin(state.room_id, jnp.array([10, 11, 12, 14]))
            c_color = jax.lax.select(is_layer_2, self.LADDER_ID_L2, self.LADDER_ID)
            mask = jnp.where(mask != self.jr.TRANSPARENT_ID, c_color.astype(jnp.uint8), self.jr.TRANSPARENT_ID)

            return jax.lax.cond(
                state.conveyors_active[i] == 1,
                lambda r: self.jr.render_at(r, state.conveyors_x[i], state.conveyors_y[i] + 47, mask, flip_vertical=anim_idx),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, render_conveyor, raster)
        
        # Draw Items
        def render_item(i, raster):
            x = state.items_x[i]
            y = state.items_y[i] + 47
            
            is_bonus_gem = jnp.logical_and(state.room_id == 24, state.items_type[i] == 1)
            is_flicker_off = jnp.logical_and(is_bonus_gem, jnp.mod(state.frame_count, 2) == 0)
            should_render = jnp.logical_and(state.items_active[i] == 1, jnp.logical_not(is_flicker_off))
            
            # Hide gems/coins (type 1) if in a dark room without a torch
            is_hidden_gem = jnp.logical_and(state.items_type[i] == 1, is_rendered_dark)
            should_render = jnp.logical_and(should_render, jnp.logical_not(is_hidden_gem))
            
            def render_key(r): return self.jr.render_at(r, x, y, self.SHAPE_MASKS['key'])
            def render_gem(r): return self.jr.render_at(r, x, y, self.SHAPE_MASKS['gem'])
            def render_amulet(r): return self.jr.render_at(r, x, y, self.SHAPE_MASKS['amulet'])
            def render_sword(r): return self.jr.render_at(r, x, y, self.SHAPE_MASKS['sword'])
            def render_torch(r): 
                # Animate torch with two sprites, alternating every 8 frames
                anim_idx = jnp.mod(state.frame_count // 8, 2)
                return self.jr.render_at(r, x, y, self.SHAPE_MASKS['torch'][anim_idx])
            
            return jax.lax.cond(
                should_render,
                lambda r: jax.lax.switch(state.items_type[i], [
                    render_key, render_gem, render_amulet, render_sword, render_torch
                ], r),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ITEMS_PER_ROOM, render_item, raster)

        # Draw Doors
        def render_door(i, raster):
            mask = self.SHAPE_MASKS["door"]
            is_active = jnp.logical_and(state.doors_active[i] == 1, jnp.logical_not(is_rendered_dark))
            return jax.lax.cond(
                is_active,
                lambda r: self.jr.render_at(r, state.doors_x[i], state.doors_y[i] + 47, mask),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_DOORS_PER_ROOM, render_door, raster)
        
        # Draw Enemies
        def render_enemy(i, raster):
            anim_idx = jax.lax.select(state.enemies_bouncing[i] == 1, 0, jnp.mod(state.enemies_x[i], 16))
            bounce_offset = jax.lax.select(state.enemies_bouncing[i] == 1, self.consts.BOUNCE_OFFSETS[jnp.mod(state.frame_count // 4, 22)], 0)

            spider_anim = jnp.mod(jnp.floor_divide(state.frame_count, 7), 2)
            spider_mask = jax.lax.cond(
                spider_anim == 1,
                lambda _: jnp.flip(self.SHAPE_MASKS["spider"], axis=1),
                lambda _: self.SHAPE_MASKS["spider"],
                None
            )

            snake_anim = jnp.mod(jnp.floor_divide(state.frame_count, 7), 2)
            base_snake_mask = self.SHAPE_MASKS["snake"][snake_anim]
            snake_mask = jax.lax.cond(
                state.enemies_direction[i] == -1,
                lambda _: jnp.flip(base_snake_mask, axis=1),
                lambda _: base_snake_mask,
                None
            )
            
            skull_mask = self.SHAPE_MASKS["skull"][anim_idx]
            
            # Neutralize (gray out) enemies if amulet is active
            is_neutralized = state.amulet_time > 0
            
            def gray_out(mask):
                return jnp.where(mask != self.jr.TRANSPARENT_ID, self.GRAY_ID, self.jr.TRANSPARENT_ID)
            
            spider_mask = jax.lax.select(is_neutralized, gray_out(spider_mask).astype(jnp.uint8), spider_mask)
            snake_mask = jax.lax.select(is_neutralized, gray_out(snake_mask).astype(jnp.uint8), snake_mask)
            skull_mask = jax.lax.select(is_neutralized, gray_out(skull_mask).astype(jnp.uint8), skull_mask)

            def _render_active(r):
                return jax.lax.cond(
                    state.enemies_type[i] == 3,
                    lambda r_in: self.jr.render_at(r_in, state.enemies_x[i], state.enemies_y[i] + 47 - bounce_offset, spider_mask),
                    lambda r_in: jax.lax.cond(
                        state.enemies_type[i] == 4,
                        lambda rr: self.jr.render_at(rr, state.enemies_x[i], state.enemies_y[i] + 47 - bounce_offset, snake_mask),
                        lambda rr: self.jr.render_at(rr, state.enemies_x[i], state.enemies_y[i] + 47 - bounce_offset, skull_mask),
                        r_in
                    ),
                    r
                )
            return jax.lax.cond(
                state.enemies_active[i] == 1,
                _render_active,
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES_PER_ROOM, render_enemy, raster)
        
        # Draw Player
        is_walking = jnp.logical_and(state.player_vx != 0, jnp.logical_and(state.is_climbing == 0, jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)))
        is_laddering = jnp.logical_and(state.is_climbing == 1, state.last_rope == -1)
        is_roping = jnp.logical_and(state.is_climbing == 1, state.last_rope != -1)
        is_in_air = jnp.logical_or(state.is_jumping == 1, state.is_falling == 1)

        walk_anim = jnp.mod(jnp.floor_divide(state.frame_count, 4), 2)
        ladder_anim = jnp.mod(jnp.floor_divide(state.player_y, 4), 2)
        rope_anim = jnp.mod(jnp.floor_divide(state.player_y, 4), 2)

        player_sprite_idx = jnp.array(0)
        player_sprite_idx = jnp.where(is_walking, 1 + walk_anim, player_sprite_idx)
        player_sprite_idx = jnp.where(is_laddering, 3 + ladder_anim, player_sprite_idx)
        player_sprite_idx = jnp.where(is_roping, 5 + rope_anim, player_sprite_idx)
        player_sprite_idx = jnp.where(is_in_air, 7, player_sprite_idx)
        
        is_dying = state.death_timer > 0
        death_anim_frame = jnp.mod(jnp.floor_divide(state.death_timer, 8), 2)
        death_base_idx = jnp.where(state.death_type == 1, 8, 10)
        player_sprite_idx = jnp.where(is_dying, death_base_idx + death_anim_frame, player_sprite_idx)
        
        # Standing sprite (0) faces right, flip if facing left.
        # Walking (1, 2) and jumping (7) sprites face left natively, flip if facing right.
        flip_player = jax.lax.select(
            jnp.logical_or(jnp.logical_or(player_sprite_idx == 1, player_sprite_idx == 2), player_sprite_idx == 7),
            state.player_dir == 1,
            jax.lax.select(
                player_sprite_idx == 0,
                state.player_dir == -1,
                False
            )
        )
        
        player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
        raster = self.jr.render_at(raster, state.player_x, state.player_y + 47, player_mask, flip_horizontal=flip_player)

        # Render Score
        score = state.score

        k_100_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([100000])), 10) + 1
        k_10_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([10000])), 10) + 1
        thousands_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([1000])), 10) + 1
        hundreds_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([100])), 10) + 1
        tens_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([10])), 10) + 1
        ones_sprite_index = jnp.add(jnp.mod(score, jnp.array([10])), 1)

        # Remove leading zeroes
        leading_zeros = jnp.array([
            k_100_sprite_index == 1,
            k_10_sprite_index == 1,
            thousands_sprite_index == 1,
            hundreds_sprite_index == 1,
            tens_sprite_index == 1
        ])
        mask = jnp.cumprod(leading_zeros)

        k_100_sprite_index = jnp.where(mask[0], 0, k_100_sprite_index)[0]
        k_10_sprite_index = jnp.where(mask[1], 0, k_10_sprite_index)[0]
        thousands_sprite_index = jnp.where(mask[2], 0, thousands_sprite_index)[0]
        hundreds_sprite_index = jnp.where(mask[3], 0, hundreds_sprite_index)[0]
        tens_sprite_index = jnp.where(mask[4], 0, tens_sprite_index)[0]
        ones_sprite_index = ones_sprite_index[0]

        def render_digit(raster, index, digit_idx):
            mask = self.digit_masks[digit_idx]
            x = self.consts.SCORE_X + index * (self.consts.DIGIT_WIDTH + self.consts.DIGIT_OFFSET)
            y = self.consts.SCORE_Y
            return self.jr.render_at(raster, x, y, mask)

        raster = render_digit(raster, 0, k_100_sprite_index)
        raster = render_digit(raster, 1, k_10_sprite_index)
        raster = render_digit(raster, 2, thousands_sprite_index)
        raster = render_digit(raster, 3, hundreds_sprite_index)
        raster = render_digit(raster, 4, tens_sprite_index)
        raster = render_digit(raster, 5, ones_sprite_index)

        # Render Lives
        def render_life(i, raster):
            mask = self.SHAPE_MASKS["life"]
            x = self.consts.ITEMBAR_LIFES_STARTING_X + i * (mask.shape[1] + 1)
            y = self.consts.LIFES_STARTING_Y
            return self.jr.render_at(raster, x, y, mask)

        raster = jax.lax.fori_loop(0, state.lives, render_life, raster)

        # Render Inventory (Keys)
        def render_key(i, raster):
            mask = self.SHAPE_MASKS["key"]
            x = self.consts.ITEMBAR_LIFES_STARTING_X + i * 8
            y = self.consts.ITEMBAR_STARTING_Y
            return self.jr.render_at(raster, x, y, mask)
            
        raster = jax.lax.fori_loop(0, state.inventory[0], render_key, raster)

        # Render Sword
        def render_sword(raster_in):
             offset = state.inventory[0]
             mask = self.SHAPE_MASKS["sword"]
             x = self.consts.ITEMBAR_LIFES_STARTING_X + offset * 8
             y = self.consts.ITEMBAR_STARTING_Y
             return self.jr.render_at(raster_in, x, y, mask)
        
        raster = jax.lax.cond(state.inventory[1] == 1, render_sword, lambda r: r, raster)
        
        # Render Torch
        def render_torch(raster_in):
             offset = state.inventory[0] + state.inventory[1]
             # Use only the first sprite (torch_1) for HUD display
             mask = self.SHAPE_MASKS["torch"][0]
             x = self.consts.ITEMBAR_LIFES_STARTING_X + offset * 8
             y = self.consts.ITEMBAR_STARTING_Y
             return self.jr.render_at(raster_in, x, y, mask)
             
        raster = jax.lax.cond(state.inventory[2] == 1, render_torch, lambda r: r, raster)

        # Render Amulet
        def render_amulet(raster_in):
             offset = state.inventory[0] + state.inventory[1] + state.inventory[2]
             mask = self.SHAPE_MASKS["amulet"]
             x = self.consts.ITEMBAR_LIFES_STARTING_X + offset * 8
             y = self.consts.ITEMBAR_STARTING_Y
             return self.jr.render_at(raster_in, x, y, mask)
             
        raster = jax.lax.cond(state.inventory[3] == 1, render_amulet, lambda r: r, raster)

        # Apply post-ui hook
        raster = self._render_hook_post_ui(raster, state)

        frame = self.PALETTE[raster]

        if self.config.downscale is not None:
            target_h, target_w = self.config.downscale
            frame = jim.resize(
                frame.astype(jnp.float32),
                (target_h, target_w, frame.shape[-1]),
                method="bilinear",
            )

        if self.config.channels == 1:
            frame = jnp.dot(
                frame.astype(jnp.float32),
                jnp.array([0.2989, 0.5870, 0.1140], dtype=jnp.float32),
            )[..., None]

        return frame.astype(jnp.uint8)
