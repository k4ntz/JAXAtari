import jax
import jax.numpy as jnp
from functools import partial
import os

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.games.montezuma_revenge.core import MontezumaRevengeConstants, MontezumaRevengeState
from jaxatari.games.montezuma_revenge.rooms import load_room

class MontezumaRevengeRenderer(JAXGameRenderer):
    def __init__(self, consts: MontezumaRevengeConstants = None, config: render_utils.RendererConfig = None, pre_load_rooms: bool = False):
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

        self.pre_load_rooms = pre_load_rooms

        self.jr = render_utils.JaxRenderingUtils(self.config)
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
        self.PALETTE, self.LADDER_ID = self.jr.add_palette_color(self.PALETTE, self.LADDER_COLOR)

        # Accurate ladder color for Difficulty 1, Layer 2 (purple)
        self.LADDER_COLOR_L2 = jnp.array([104, 25, 154], dtype=jnp.uint8)
        self.PALETTE, self.LADDER_ID_L2 = self.jr.add_palette_color(self.PALETTE, self.LADDER_COLOR_L2)
        
        # Blue ladder color for long ladders
        self.BLUE_LADDER_COLOR = jnp.array([24, 59, 157], dtype=jnp.uint8)
        self.PALETTE, self.BLUE_LADDER_ID = self.jr.add_palette_color(self.PALETTE, self.BLUE_LADDER_COLOR)

        # Yellow ladder color for long ladders
        self.YELLOW_LADDER_COLOR = jnp.array([204, 216, 110], dtype=jnp.uint8)
        self.PALETTE, self.YELLOW_LADDER_ID = self.jr.add_palette_color(self.PALETTE, self.YELLOW_LADDER_COLOR)

        # Accurate door color
        self.DOOR_COLOR = jnp.array([232, 204, 99], dtype=jnp.uint8)
        self.PALETTE, self.DOOR_ID = self.jr.add_palette_color(self.PALETTE, self.DOOR_COLOR)
        
        # Laser color
        self.LASER_COLOR = jnp.array([101, 111, 228], dtype=jnp.uint8)
        self.PALETTE, self.LASER_ID = self.jr.add_palette_color(self.PALETTE, self.LASER_COLOR)
        
        # Level 2 platform blue color
        self.LEVEL2_PLATFORM_COLOR = jnp.array([45, 87, 176], dtype=jnp.uint8)
        self.PALETTE, self.LEVEL2_PLATFORM_ID = self.jr.add_palette_color(self.PALETTE, self.LEVEL2_PLATFORM_COLOR)

        # Room 31 (ROOM_3_7) specific colors
        self.ORANGE_LADDER_COLOR = jnp.array([213, 130, 74], dtype=jnp.uint8)
        self.PALETTE, self.ORANGE_LADDER_ID = self.jr.add_palette_color(self.PALETTE, self.ORANGE_LADDER_COLOR)

        self.DEEP_BLUE_PLATFORM_COLOR = jnp.array([24, 26, 167], dtype=jnp.uint8)
        self.PALETTE, self.DEEP_BLUE_PLATFORM_ID = self.jr.add_palette_color(self.PALETTE, self.DEEP_BLUE_PLATFORM_COLOR)

        # Gray color for neutralized enemies
        self.GRAY_COLOR = jnp.array([142, 142, 142], dtype=jnp.uint8)
        self.PALETTE, self.GRAY_ID = self.jr.add_palette_color(self.PALETTE, self.GRAY_COLOR)

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
        
        self.PALETTE, self.PIT_COLOR_IDS = self.jr.add_palette_colors(self.PALETTE, self.PIT_COLORS)
        
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
        if pre_load_rooms:
            room_template_state = self._create_room_geometry_template_state()
            self.room_backgrounds = jnp.stack([
                self._build_room_background(room_id, self._load_room_geometry(room_id, room_template_state))
                for room_id in range(self.consts.MAX_ROOMS)
            ])

    def _create_room_geometry_template_state(self) -> MontezumaRevengeState:
        return MontezumaRevengeState(
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
            inventory=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
            amulet_time=jnp.array(0, dtype=jnp.int32),
            bonus_room_timer=jnp.array(0, dtype=jnp.int32),
            first_gem_pickup=jnp.array(0, dtype=jnp.int32),
            global_enemies_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ENEMIES_PER_ROOM), dtype=jnp.int32),
            global_enemies_type=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ENEMIES_PER_ROOM), dtype=jnp.int32),
            global_items_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ITEMS_PER_ROOM), dtype=jnp.int32),
            global_items_type=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ITEMS_PER_ROOM), dtype=jnp.int32),
            global_doors_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_DOORS_PER_ROOM), dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
        )

    def _load_room_geometry(self, room_id: int, template_state: MontezumaRevengeState) -> MontezumaRevengeState:
        return load_room(jnp.array(room_id, dtype=jnp.int32), template_state, self.consts)
    
    def _build_room_background(self, room_id: int, room_state: MontezumaRevengeState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        room_y = 47
        room_h = 149
        room_w = 160

        def clear_room(r_in):
            pos = jnp.array([[0, room_y]])
            size = jnp.array([[room_w, room_h]])
            return self.jr.draw_rects(r_in, pos, size, jnp.uint8(0))

        def clear_hole(r_in, y0, y1, x0, x1):
            pos = jnp.array([[x0, room_y + y0]])
            size = jnp.array([[x1 - x0, y1 - y0]])
            return self.jr.draw_rects(r_in, pos, size, jnp.uint8(0))

        def stamp_room(r_in, mask):
            rr = clear_room(r_in)
            return self.jr.render_at(rr, 0, room_y, mask)

        def draw_wall(r_in, x, y0, y1, color):
            pos = jnp.array([[x, room_y + y0]])
            size = jnp.array([[4, y1 - y0]])
            return self.jr.draw_rects(r_in, pos, size, color.astype(jnp.uint8))

        raster = clear_room(raster)
        raster = self.jr.render_at(raster, 0, room_y, self.SHAPE_MASKS["room_bg_0"])
        raster = clear_hole(raster, 147, 149, 72, 88)

        raster = jax.lax.cond(
            room_id == 4,
            lambda r: stamp_room(r, self.SHAPE_MASKS["room_bg_1"]),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            room_id == 12,
            lambda r: clear_hole(stamp_room(r, self.SHAPE_MASKS["room_bg_3"]), 147, 149, 72, 88),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            jnp.isin(room_id, jnp.array([25, 26, 28, 30, 32])),
            lambda r: stamp_room(r, self.SHAPE_MASKS["room_bg_4"]),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            jnp.isin(room_id, jnp.array([10, 11, 14])),
            lambda r: clear_hole(stamp_room(r, self.SHAPE_MASKS["room_bg_2"]), 48, 149, 72, 88),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            room_id == 13,
            lambda r: stamp_room(r, self.SHAPE_MASKS["room_bg_2"]),
            lambda r: r,
            raster,
        )

        mask_l2 = jnp.where(self.SHAPE_MASKS["room_bg_level2_base"] == 1, self.LEVEL2_PLATFORM_ID, self.SHAPE_MASKS["room_bg_level2_base"])
        mask_l2_room0 = jnp.where(self.SHAPE_MASKS["room_bg_level2_room0"] == 1, self.LEVEL2_PLATFORM_ID, self.SHAPE_MASKS["room_bg_level2_room0"])
        mask_l2_room6 = jnp.where(self.SHAPE_MASKS["room_bg_level2_room6"] == 1, self.LEVEL2_PLATFORM_ID, self.SHAPE_MASKS["room_bg_level2_room6"])
        mask_l2_pit = jnp.where(self.SHAPE_MASKS["room_bg_level2_pit"] == 1, self.LEVEL2_PLATFORM_ID, self.SHAPE_MASKS["room_bg_level2_pit"])

        raster = jax.lax.cond(room_id == 18, lambda r: stamp_room(r, mask_l2), lambda r: r, raster)
        raster = jax.lax.cond(room_id == 17, lambda r: stamp_room(r, mask_l2_room0), lambda r: r, raster)
        raster = jax.lax.cond(room_id == 19, lambda r: stamp_room(r, mask_l2_pit), lambda r: r, raster)
        raster = jax.lax.cond(
            jnp.isin(room_id, jnp.array([27, 29, 31])),
            lambda r: stamp_room(r, self.SHAPE_MASKS["room_bg_pit_original"]),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            jnp.isin(room_id, jnp.array([20, 22])),
            lambda r: clear_hole(stamp_room(r, mask_l2), 48, 149, 72, 88),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(room_id == 21, lambda r: stamp_room(r, mask_l2), lambda r: r, raster)

        raster = jax.lax.cond(room_id == 23, lambda r: stamp_room(r, mask_l2_room6), lambda r: r, raster)

        raster = jax.lax.cond(room_id == 24, lambda r: stamp_room(r, self.SHAPE_MASKS["room_bg_bonus"]), lambda r: r, raster)

        left_wall_color = jnp.where(room_id == 19, self.LADDER_ID,
                                    jnp.where(room_id == 30, self.ORANGE_LADDER_ID,
                                              jnp.where(room_id == 17, self.LEVEL2_PLATFORM_ID, 1)))
        raster = jax.lax.cond(
            jnp.isin(room_id, jnp.array([3, 10, 19, 30])),
            lambda r: draw_wall(r, 0, 6, 48, left_wall_color),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(room_id == 17, lambda r: draw_wall(r, 0, 6, 149, left_wall_color), lambda r: r, raster)

        right_wall_color = jnp.where(room_id == 18, self.LADDER_ID,
                                     jnp.where(room_id == 29, self.DEEP_BLUE_PLATFORM_ID,
                                               jnp.where(room_id == 23, self.LEVEL2_PLATFORM_ID, 1)))
        raster = jax.lax.cond(
            jnp.isin(room_id, jnp.array([5, 14, 18, 32, 29])),
            lambda r: draw_wall(r, 156, 6, 48, right_wall_color),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(room_id == 23, lambda r: draw_wall(r, 156, 6, 149, right_wall_color), lambda r: r, raster)

        # Draw ladders as static room geometry.
        def draw_ladder_accurate(i, r):
            x = room_state.ladders_x[i]
            top = room_state.ladders_top[i] + 47
            bottom = room_state.ladders_bottom[i] + 47
            bottom = jnp.where(jnp.logical_and(room_id == 4, room_state.ladders_bottom[i] == 130), bottom + 3, bottom)
            bottom = jnp.where(jnp.logical_and(room_id == 23, room_state.ladders_bottom[i] == 150), bottom - 3, bottom)
            active = room_state.ladders_active[i]

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
                rail_pos = jnp.array([[x, top], [x + ladder_width - 4, top]])
                rail_size = jnp.array([[4, bottom - top], [4, bottom - top]])
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

                is_layer_2 = jnp.isin(room_id, jnp.array([10, 11, 12, 14]))
                is_long_ladder = jnp.logical_and(jnp.isin(room_id, jnp.array([3, 5])), i == 0)
                is_long_ladder_l2 = jnp.logical_and(jnp.isin(room_id, jnp.array([10, 11, 12, 14])), i == 0)
                is_small_yellow = jnp.logical_or(
                    jnp.logical_and(room_id == 11, i == 1),
                    jnp.logical_and(room_id == 13, i == 0)
                )
                is_room_orange_ladder = jnp.isin(room_id, jnp.array([28, 30, 31]))

                ladder_draw_case = jnp.where(is_layer_2, 1, 0)
                ladder_draw_case = jnp.where(is_small_yellow, 2, ladder_draw_case)
                ladder_draw_case = jnp.where(is_long_ladder_l2, 3, ladder_draw_case)
                ladder_draw_case = jnp.where(is_long_ladder, 4, ladder_draw_case)
                ladder_draw_case = jnp.where(is_room_orange_ladder, 5, ladder_draw_case)

                return jax.lax.switch(
                    ladder_draw_case,
                    [draw_l1, draw_l2, draw_yellow_l2, draw_long_l2, draw_long, draw_orange],
                    raster_in,
                )

            return jax.lax.cond(active == 1, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_LADDERS_PER_ROOM, draw_ladder_accurate, raster)

        # Draw ropes as static room geometry.
        def draw_rope(i, r):
            x = room_state.ropes_x[i]
            top = room_state.ropes_top[i] + 47
            bottom = room_state.ropes_bottom[i] + 47
            active = room_state.ropes_active[i]

            def _draw(raster_in):
                rail_pos = jnp.array([[x, top]])
                rail_size = jnp.array([[1, bottom - top + 1]])
                return self.jr.draw_rects(raster_in, rail_pos, rail_size, self.DOOR_ID)

            return jax.lax.cond(active == 1, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_ROPES_PER_ROOM, draw_rope, raster)

        return raster

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

        # Select the precomputed static room background.
        if self.pre_load_rooms:
            background_raster = self.room_backgrounds[state.room_id]
        else:
            background_raster = self._build_room_background(state.room_id, state) 
        background_before_dark = background_raster
        room_y = 47


        def clear_room(r_in):
            room_h = 149
            room_w = 160
            pos = jnp.array([[0, room_y]])
            size = jnp.array([[room_w, room_h]])
            return self.jr.draw_rects(r_in, pos, size, jnp.uint8(0))

        # DARK ROOM LOGIC
        is_dark_room = jnp.isin(state.room_id, jnp.array([25, 26, 27, 28, 29, 30, 31, 32]))
        has_torch = state.inventory[2] == 1
        is_rendered_dark = jnp.logical_and(is_dark_room, jnp.logical_not(has_torch))
        background_raster = jax.lax.cond(is_rendered_dark, clear_room, lambda rr: rr, background_raster)

        # Add lava rendering for ROOM_2_3 (room_id 19) and ROOM_3_7 (room_id 31)
        def _render_lava(background_raster):
            lava_y_start = 76
            lava_y_end = 124 # gap ends at 123
            # scale the coords
            lava_y_start_scaled = int(self.jr.config.height_scaling * lava_y_start)
            lava_y_end_scaled = int(self.jr.config.height_scaling * lava_y_end)
            anim_frame = jnp.mod(state.frame_count // 8, 4)
            row_indices = jnp.arange(lava_y_end_scaled - lava_y_start_scaled)
            band_indices = row_indices // 2
            color_indices = (band_indices // 9) + self.PIT_PATTERN[anim_frame][jnp.mod(band_indices, 9)]
            band_color_ids = self.PIT_COLOR_IDS[color_indices]
            lava_width = int(self.jr.config.width_scaling * 160)
            lava_mask = jnp.tile(band_color_ids[:, None], (1, lava_width))
            
            lava_raster = self.jr.render_at(background_raster, 0, room_y + lava_y_start, lava_mask)
            lava_raster = jnp.where(background_before_dark == 0, lava_raster, background_raster)
            is_lava_room = jnp.any(state.room_id == jnp.array([19, 31, 27, 29]))
            lava_cond = jnp.logical_and(is_lava_room, jnp.logical_not(is_rendered_dark))  # Only show lava if it's a lava room and not dark
            background_raster = jax.lax.cond(lava_cond, lambda r: lava_raster, lambda r: r, background_raster)
            return background_raster

        raster = jax.lax.cond(jnp.any(state.room_id == jnp.array([19, 27, 29, 31])), _render_lava, lambda r: r, background_raster) 

        def _draw_lasers(raster):
            # Draw Lasers - vectorized to draw all lasers at once
            laser_active_now = jnp.logical_and(jnp.greater_equal(state.laser_cycle, 0), jnp.less(state.laser_cycle, 92))
            laser_offset = jnp.mod(state.laser_cycle, 4)
            
            # Compute stripe j values (40 pixels high, 4-pixel spacing with offset)
            start_j = jnp.mod(4 - laser_offset, 4) - 4
            k_idx = jnp.arange(11)
            j_vals = start_j + k_idx * 4  # shape: (11,)
            
            # Broadcast laser positions and active states across all lasers and stripes
            # Shape: (MAX_LASERS_PER_ROOM, 11)
            x_per_laser = jnp.tile(state.lasers_x[:, None], (1, 11))
            j_vals_per_laser = jnp.tile(j_vals[None, :], (self.consts.MAX_LASERS_PER_ROOM, 1))
            active_per_laser = jnp.tile(
                jnp.logical_and(state.lasers_active == 1, laser_active_now)[:, None],
                (1, 11)
            )
            
            # Check validity: stripe within [0, 40) range AND laser is active
            valid = jnp.logical_and(
                jnp.logical_and(j_vals_per_laser >= 0, j_vals_per_laser < 40),
                active_per_laser
            )
            
            # Build positions and sizes
            pos_x = jnp.where(valid, x_per_laser, -1)
            pos_y = jnp.where(valid, 54 + j_vals_per_laser, -1)
            sizes = jnp.where(valid, 4, 1)
            
            # Flatten all positions and draw in one call
            pos_flat = jnp.stack([pos_x.flatten(), pos_y.flatten()], axis=-1)
            size_flat = jnp.stack([sizes.flatten(), jnp.ones_like(sizes.flatten())], axis=-1)

            raster = self.jr.draw_rects(raster, pos_flat, size_flat, self.LASER_ID)
            return raster

        raster = jax.lax.cond(jnp.any(state.lasers_active == 1), lambda r: _draw_lasers(r), lambda r: r, raster)

        # Draw Platforms
        # Pre-compute room state checks (same for all platforms in this frame)
        platform_active_now = jnp.less(state.platform_cycle, self.consts.PLATFORM_ACTIVE_DURATION)
        is_pit_room = jnp.isin(state.room_id, jnp.array([19, 27, 29, 31]))
        is_layer_2_room = jnp.logical_or(state.room_id == 17, jnp.logical_or(state.room_id == 18, state.room_id == 19))
        is_deep_blue_room = jnp.any(state.room_id == jnp.array([31, 27, 29]))
        p_color = jax.lax.select(is_layer_2_room, self.LEVEL2_PLATFORM_ID, self.LADDER_ID_L2)
        p_color = jax.lax.select(is_deep_blue_room, self.DEEP_BLUE_PLATFORM_ID, p_color)
        anim_idx_platform = (state.frame_count // 8) % 2
        
        def render_platform(i, raster):
            is_active = jnp.logical_and(state.platforms_active[i] == 1, platform_active_now)

            def _draw_pit(r):
                # 1. Prepare and scale the base tile
                mask = self.SHAPE_MASKS["pitroom_dropout_floor"][anim_idx_platform]
                mask = jnp.concatenate([mask, mask[0:1, :]], axis=0)
                mask = jnp.where(mask != self.jr.TRANSPARENT_ID, p_color, self.jr.TRANSPARENT_ID)

                # 2. Get the dimensions AFTER scaling
                # These are concrete values to JAX during tracing if MAX_TILES is an int
                MAX_TILES = 12
                tile_h, tile_w = mask.shape 

                # 3. Tile horizontally
                total_mask = jnp.tile(mask, (1, MAX_TILES))
                
                # 4. Calculate dynamic visibility based on current mask width
                # We find how many original 8px tiles fit, then multiply by current tile_w
                num_tiles = state.platforms_width[i] // 8
                actual_width_px = num_tiles * tile_w
                
                # Use total_mask.shape[1] to guarantee the indices match the array width
                x_indices = jnp.arange(total_mask.shape[1])
                is_visible = x_indices < actual_width_px
                
                # 5. Mask and Render
                full_mask = jnp.where(is_visible[None, :], total_mask, self.jr.TRANSPARENT_ID)

                return self.jr.render_at(r, state.platforms_x[i], state.platforms_y[i] + 47, full_mask)

            def _draw_other(r):
                # 1. Prepare and scale the base tile
                mask = self.SHAPE_MASKS["dropout_floor"][anim_idx_platform]
                mask = jnp.where(mask != self.jr.TRANSPARENT_ID, p_color, self.jr.TRANSPARENT_ID)

                # 2. Setup Static Dimensions
                # Based on your previous code, max width seems to be 12 tiles
                MAX_TILES = 12 
                tile_h, tile_w = mask.shape

                # 3. Create the tiled "Super Mask" (Horizontal)
                # Shape: (tile_h, tile_w * 12)
                total_mask = jnp.tile(mask, (1, MAX_TILES))

                # 4. Calculate Dynamic Visibility
                # Original tile width was 12. We find how many tiles, then map to scaled width.
                num_tiles = state.platforms_width[i] // 12
                actual_width_px = num_tiles * tile_w
                
                # Create the x-axis index mask
                x_indices = jnp.arange(total_mask.shape[1])
                is_visible = x_indices < actual_width_px
                
                # Mask out the inactive tiles
                full_mask = jnp.where(is_visible[None, :], total_mask, self.jr.TRANSPARENT_ID)

                # 5. Render in one shot
                return self.jr.render_at(r, state.platforms_x[i], state.platforms_y[i] + 47, full_mask)

            def _draw_active(r):
                return jax.lax.cond(is_pit_room, _draw_pit, _draw_other, r)

            return jax.lax.cond(
                is_active,
                _draw_active,
                lambda r: r,
                raster
            )
        
        raster = jax.lax.cond(
            jnp.any(state.platforms_active == 1),
            lambda r: jax.lax.fori_loop(0, self.consts.MAX_PLATFORMS_PER_ROOM, render_platform, r),
            lambda r: r,
            raster
        )

        # Draw Conveyors
        # Pre-compute room state check and animation (same for all conveyors in this frame)
        anim_idx = jnp.less(jnp.mod(state.frame_count, 16), 8)
        is_layer_2 = jnp.isin(state.room_id, jnp.array([10, 11, 12, 14]))
        c_color = jax.lax.select(is_layer_2, self.LADDER_ID_L2, self.LADDER_ID)
        base_mask = self.SHAPE_MASKS["conveyor"]
        conveyor_mask = jnp.where(base_mask != self.jr.TRANSPARENT_ID, c_color.astype(jnp.uint8), self.jr.TRANSPARENT_ID)
        
        def render_conveyor_body(i, raster):
            return jax.lax.cond(
                state.conveyors_active[i] == 1,
                lambda r: self.jr.render_at(r, state.conveyors_x[i], state.conveyors_y[i] + 47, conveyor_mask, flip_vertical=anim_idx),
                lambda r: r,
                raster
            )
        
        raster = jax.lax.cond(
            jnp.any(state.conveyors_active == 1),
            # _render_conveyor,
            lambda r: jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, render_conveyor_body, raster),
            lambda r: r,
            raster
        )
        
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
            
            spider_mask = jax.lax.select(is_neutralized, gray_out(spider_mask).astype(spider_mask.dtype), spider_mask)
            snake_mask = jax.lax.select(is_neutralized, gray_out(snake_mask).astype(snake_mask.dtype), snake_mask)
            skull_mask = jax.lax.select(is_neutralized, gray_out(skull_mask).astype(skull_mask.dtype), skull_mask)

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
        def render_sword(i, raster_in):
             offset = state.inventory[0] + i
             mask = self.SHAPE_MASKS["sword"]
             x = self.consts.ITEMBAR_LIFES_STARTING_X + offset * 8
             y = self.consts.ITEMBAR_STARTING_Y
             return self.jr.render_at(raster_in, x, y, mask)
        
        raster = jax.lax.fori_loop(0, state.inventory[1], render_sword, raster)
        
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

        return self.jr.render_from_palette(raster, self.PALETTE) 
