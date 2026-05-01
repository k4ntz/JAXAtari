import os
from functools import partial
import jax
import jax.numpy as jnp
import jax.image
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.games.venture.core import VentureConstants, GameState

def _get_venture_asset_config() -> list[dict]:
    return [
        {'name': 'background', 'type': 'background', 'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)},
        {'name': 'map_w1', 'type': 'single', 'file': 'map.npy'},
        {'name': 'room1_w1', 'type': 'single', 'file': 'room1.npy'},
        {'name': 'room2_w1', 'type': 'single', 'file': 'room2.npy'},
        {'name': 'room3_w1', 'type': 'single', 'file': 'room3.npy'},
        {'name': 'room4_w1', 'type': 'single', 'file': 'room4.npy'},
        {'name': 'map_w2', 'type': 'single', 'file': 'map2.npy'},
        {'name': 'room1_w2', 'type': 'single', 'file': 'room21.npy'},
        {'name': 'room2_w2', 'type': 'single', 'file': 'room22.npy'},
        {'name': 'room3_w2', 'type': 'single', 'file': 'room23.npy'},
        {'name': 'room4_w2', 'type': 'single', 'file': 'room24.npy'},
        {'name': 'player_dot_w1', 'type': 'single', 'file': 'player_dot.npy'},
        {'name': 'player_dot_w2', 'type': 'single', 'file': 'player_dot2.npy'},
        {'name': 'player_detailed', 'type': 'single', 'file': 'player_detailed.npy'},
        {'name': 'monster_map_w1', 'type': 'single', 'file': 'main_map_monster.npy'},
        {'name': 'monster_r2_w1', 'type': 'single', 'file': 'monster2.npy'},
        {'name': 'monster_r3_w1', 'type': 'single', 'file': 'monster3.npy'},
        {'name': 'monster_r4_w1', 'type': 'single', 'file': 'monster4.npy'},
        {'name': 'monster_map_w2', 'type': 'single', 'file': 'main_map_monster2.npy'},
        {'name': 'monster_r1_w2', 'type': 'single', 'file': 'monster21.npy'},
        {'name': 'monster_r2_w2', 'type': 'single', 'file': 'monster22.npy'},
        {'name': 'monster_r3_w2', 'type': 'single', 'file': 'monster23.npy'},
        {'name': 'monster_r4_w2', 'type': 'single', 'file': 'monster24.npy'},
        {'name': 'monster_dead_map_w1', 'type': 'single', 'file': 'monster2_dead.npy'},
        {'name': 'monster_dead_r2_w1', 'type': 'single', 'file': 'monster2_dead.npy'},
        {'name': 'monster_dead_r3_w1', 'type': 'single', 'file': 'monster3_dead.npy'},
        {'name': 'monster_dead_r4_w1', 'type': 'single', 'file': 'monster4_dead.npy'},
        {'name': 'monster_dead_map_w2', 'type': 'single', 'file': 'monster21_dead.npy'},
        {'name': 'monster_dead_r1_w2', 'type': 'single', 'file': 'monster21_dead.npy'},
        {'name': 'monster_dead_r2_w2', 'type': 'single', 'file': 'monster22_dead.npy'},
        {'name': 'monster_dead_r3_w2', 'type': 'single', 'file': 'monster23_dead.npy'},
        {'name': 'monster_dead_r4_w2', 'type': 'single', 'file': 'monster24_dead.npy'},
        {'name': 'reward1_w1', 'type': 'single', 'file': 'reward1.npy'},
        {'name': 'reward2_w1', 'type': 'single', 'file': 'reward2.npy'},
        {'name': 'reward3_w1', 'type': 'single', 'file': 'reward3.npy'},
        {'name': 'reward4_w1', 'type': 'single', 'file': 'reward4.npy'},
        {'name': 'reward1_w2', 'type': 'single', 'file': 'reward21.npy'},
        {'name': 'reward2_w2', 'type': 'single', 'file': 'reward22.npy'},
        {'name': 'reward3_w2', 'type': 'single', 'file': 'reward23.npy'},
        {'name': 'reward4_w2', 'type': 'single', 'file': 'reward24.npy'},
        {'name': 'health_w1', 'type': 'single', 'file': 'health.npy'},
        {'name': 'health_w2', 'type': 'single', 'file': 'health2.npy'},
        {'name': 'chaser', 'type': 'single', 'file': 'chaser.npy'},
        {'name': 'laser_ho', 'type': 'single', 'file': 'laser_wall_ho.npy'},
        {'name': 'laser_ve', 'type': 'single', 'file': 'laser_wall_ve.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': '{}.npy'},
    ]

class VentureRenderer(JAXGameRenderer):
    def __init__(self, consts: VentureConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or VentureConstants()
        self.config = config or render_utils.RendererConfig(game_dimensions=(210, 160), channels=3, downscale=None)
        self.jr = render_utils.JaxRenderingUtils(self.config)
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "venture")
        asset_config = _get_venture_asset_config()
        procedural_assets = self._create_procedural_assets(sprite_path)
        asset_config.extend(procedural_assets)
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, self.COLOR_TO_ID, self.FLIP_OFFSETS) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        
        def stack_and_pad(masks):
            max_h = max(m.shape[0] for m in masks)
            max_w = max(m.shape[1] for m in masks)
            padded = [jnp.pad(m, ((0, max_h - m.shape[0]), (0, max_w - m.shape[1])), constant_values=self.jr.TRANSPARENT_ID) for m in masks]
            return jnp.stack(padded)

        all_wall_masks = jnp.stack([
            stack_and_pad([self.SHAPE_MASKS['map_w1'], self.SHAPE_MASKS['room1_w1'], self.SHAPE_MASKS['room2_w1'], self.SHAPE_MASKS['room3_w1'], self.SHAPE_MASKS['room4_w1']]),
            stack_and_pad([self.SHAPE_MASKS['map_w2'], self.SHAPE_MASKS['room1_w2'], self.SHAPE_MASKS['room2_w2'], self.SHAPE_MASKS['room3_w2'], self.SHAPE_MASKS['room4_w2']])
        ])
        base_raster = self.jr.create_object_raster(self.BACKGROUND)
        self.all_background_rasters = jax.vmap(jax.vmap(lambda m: self.jr.render_at(base_raster, 0, 0, m)))(all_wall_masks)
        self.all_monster_masks = jnp.stack([
            stack_and_pad([self.SHAPE_MASKS['monster_map_w1'], self.SHAPE_MASKS['monster_map_w1'], self.SHAPE_MASKS['monster_r2_w1'], self.SHAPE_MASKS['monster_r3_w1'], self.SHAPE_MASKS['monster_r4_w1']]),
            stack_and_pad([self.SHAPE_MASKS['monster_map_w2'], self.SHAPE_MASKS['monster_r1_w2'], self.SHAPE_MASKS['monster_r2_w2'], self.SHAPE_MASKS['monster_r3_w2'], self.SHAPE_MASKS['monster_r4_w2']])
        ])
        self.all_dead_monster_masks = jnp.stack([
            stack_and_pad([self.SHAPE_MASKS['monster_dead_map_w1'], self.SHAPE_MASKS['monster_dead_map_w1'], self.SHAPE_MASKS['monster_dead_r2_w1'], self.SHAPE_MASKS['monster_dead_r3_w1'], self.SHAPE_MASKS['monster_dead_r4_w1']]),
            stack_and_pad([self.SHAPE_MASKS['monster_dead_map_w2'], self.SHAPE_MASKS['monster_dead_r1_w2'], self.SHAPE_MASKS['monster_dead_r2_w2'], self.SHAPE_MASKS['monster_dead_r3_w2'], self.SHAPE_MASKS['monster_dead_r4_w2']])
        ])
        self.all_chest_masks = jnp.stack([
            stack_and_pad([self.SHAPE_MASKS['reward1_w1'], self.SHAPE_MASKS['reward2_w1'], self.SHAPE_MASKS['reward3_w1'], self.SHAPE_MASKS['reward4_w1']]),
            stack_and_pad([self.SHAPE_MASKS['reward1_w2'], self.SHAPE_MASKS['reward2_w2'], self.SHAPE_MASKS['reward3_w2'], self.SHAPE_MASKS['reward4_w2']])
        ])
        self.all_life_masks = stack_and_pad([self.SHAPE_MASKS['health_w1'], self.SHAPE_MASKS['health_w2']])
        self.all_player_dot_masks = stack_and_pad([self.SHAPE_MASKS['player_dot_w1'], self.SHAPE_MASKS['player_dot_w2']])
        self.monster_offsets = jnp.array([self.consts.MONSTER_RENDER_WIDTH / 2, self.consts.MONSTER_RENDER_HEIGHT / 2], dtype=jnp.int32)
        self.chest_offsets = jnp.array([self.consts.CHEST_WIDTH / 2, self.consts.CHEST_HEIGHT / 2], dtype=jnp.int32)
        self.player_dot_offsets = jnp.array([self.consts.PLAYER_DOT_RENDER_WIDTH / 2, self.consts.PLAYER_DOT_RENDER_HEIGHT / 2], dtype=jnp.int32)
        self.player_detailed_offsets = jnp.array([self.consts.PLAYER_DETAILED_RENDER_WIDTH / 2, self.consts.PLAYER_DETAILED_RENDER_HEIGHT / 2], dtype=jnp.int32)
        self.chaser_offsets = jnp.array([self.consts.CHASER_RENDER_WIDTH / 2, self.consts.CHASER_RENDER_HEIGHT / 2], dtype=jnp.int32)

    def _create_procedural_assets(self, sprite_path: str) -> list[dict]:
        assets = []
        def load_resize(filename, target_shape, name):
            path = os.path.join(sprite_path, filename)
            frame = self.jr.loadFrame(path)
            resized = jax.image.resize(frame, target_shape, method='nearest').astype(jnp.uint8)
            if resized.shape[-1] == 3: resized = jnp.concatenate([resized, jnp.full(resized.shape[:2] + (1,), 255, dtype=jnp.uint8)], axis=-1)
            return {'name': name, 'type': 'procedural', 'data': resized}
        proj_size = int(self.consts.PROJECTILE_RADIUS * 2)
        assets.append(load_resize('player_dot.npy', (proj_size, proj_size, 4), 'projectile_resized'))
        x_span_start, x_span_end, y_span_start, y_span_end = self.consts.LASER_ROOM_SPAN
        room_h, room_w, thickness = int(y_span_end - y_span_start), int(x_span_end - x_span_start), int(self.consts.LASER_THICKNESS)
        assets.append(load_resize('laser_wall_ve.npy', (room_h, thickness, 4), 'laser_ve_stretched'))
        assets.append(load_resize('laser_wall_ho.npy', (thickness, room_w, 4), 'laser_ho_stretched'))
        return assets

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState):
        world_idx = state.world_level - 1
        level_idx = state.current_level
        is_in_room = level_idx > 0
        canvas = self.all_background_rasters[world_idx, level_idx]
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        canvas = self.jr.render_label(canvas, 8, 10, score_digits, self.SHAPE_MASKS['digits'], spacing=6, max_digits=6)
        canvas = self.jr.render_indicator(canvas, 120, 10, state.lives - 1, self.all_life_masks[world_idx], spacing=10, max_value=3)

        def draw_chests(c):
            chest_idx = level_idx - 1
            global_idx = world_idx * 5 + level_idx
            is_active = state.chests_active[chest_idx] & (state.collected_chest_in_current_visit != chest_idx)
            top_left = (self.consts.CHEST_POSITIONS[global_idx] - self.chest_offsets).astype(jnp.int32)
            return jax.lax.cond(is_active, lambda _c: self.jr.render_at(_c, top_left[0], top_left[1], self.all_chest_masks[world_idx, chest_idx]), lambda _c: _c, c)
        canvas = jax.lax.cond(is_in_room, draw_chests, lambda c: c, canvas)

        global_level = world_idx * 5 + level_idx
        start_idx, end_idx = self.consts.LEVEL_OFFSETS[global_level], self.consts.LEVEL_OFFSETS[global_level + 1]
        monster_mask = self.all_monster_masks[world_idx, level_idx]
        def draw_single_monster(i, _c):
            mx, my = (state.monsters.x[i] - self.monster_offsets[0]).astype(jnp.int32), (state.monsters.y[i] - self.monster_offsets[1]).astype(jnp.int32)
            return jax.lax.cond(state.monsters.active[i], lambda __c: self.jr.render_at(__c, mx, my, monster_mask), lambda __c: __c, _c)
        canvas = jax.lax.fori_loop(start_idx, end_idx, draw_single_monster, canvas)

        dead_monster_mask = self.all_dead_monster_masks[world_idx, level_idx]
        def draw_single_dead_monster(i, _c):
            mx, my = (state.monsters.x[i] - self.monster_offsets[0]).astype(jnp.int32), (state.monsters.y[i] - self.monster_offsets[1]).astype(jnp.int32)
            return jax.lax.cond(state.monsters.dead_for[i] > 0, lambda __c: self.jr.render_at(__c, mx, my, dead_monster_mask), lambda __c: __c, _c)
        canvas = jax.lax.fori_loop(start_idx, end_idx, draw_single_dead_monster, canvas)

        chaser_tl = (jnp.array([state.chaser.x, state.chaser.y]) - self.chaser_offsets).astype(jnp.int32)
        canvas = jax.lax.cond(state.chaser.active, lambda c: self.jr.render_at(c, chaser_tl[0], chaser_tl[1], self.SHAPE_MASKS['chaser']), lambda c: c, canvas)

        def draw_lasers(c):
            x_span_start, _, y_span_start, _ = self.consts.LASER_ROOM_SPAN
            thick_h = self.consts.LASER_THICKNESS / 2
            c = self.jr.render_at(c, (state.lasers.positions[0] - thick_h).astype(jnp.int32), y_span_start.astype(jnp.int32), self.SHAPE_MASKS['laser_ve_stretched'])
            c = self.jr.render_at(c, (state.lasers.positions[1] - thick_h).astype(jnp.int32), y_span_start.astype(jnp.int32), self.SHAPE_MASKS['laser_ve_stretched'])
            c = self.jr.render_at(c, x_span_start.astype(jnp.int32), (state.lasers.positions[2] - thick_h).astype(jnp.int32), self.SHAPE_MASKS['laser_ho_stretched'])
            c = self.jr.render_at(c, x_span_start.astype(jnp.int32), (state.lasers.positions[3] - thick_h).astype(jnp.int32), self.SHAPE_MASKS['laser_ho_stretched'])
            return c
        canvas = jax.lax.cond((level_idx == 1) & (state.world_level == 1), draw_lasers, lambda c: c, canvas)

        def draw_player(c):
            def _room(_c):
                px, py = (state.player.x - self.player_detailed_offsets[0]).astype(jnp.int32), (state.player.y - self.player_detailed_offsets[1]).astype(jnp.int32)
                return self.jr.render_at(_c, px, py, self.SHAPE_MASKS['player_detailed'])
            def _map(_c):
                px, py = (state.player.x - self.player_dot_offsets[0]).astype(jnp.int32), (state.player.y - self.player_dot_offsets[1]).astype(jnp.int32)
                return self.jr.render_at(_c, px, py, self.all_player_dot_masks[world_idx])
            return jax.lax.cond(is_in_room, _room, _map, c)
        canvas = draw_player(canvas)

        def draw_aiming_dot(c):
            dot_x = state.player.x + state.player.last_dx * (self.consts.PLAYER_DETAILED_RENDER_WIDTH / 2 + self.consts.AIMING_DOT_OFFSET)
            dot_y = state.player.y + state.player.last_dy * (self.consts.PLAYER_DETAILED_RENDER_HEIGHT / 2 + self.consts.AIMING_DOT_OFFSET)
            return self.jr.render_at(c, (dot_x - self.player_dot_offsets[0]).astype(jnp.int32), (dot_y - self.player_dot_offsets[1]).astype(jnp.int32), self.all_player_dot_masks[world_idx])
        def draw_projectile(c):
             return self.jr.render_at(c, (state.projectile.x - self.consts.PROJECTILE_RADIUS).astype(jnp.int32), (state.projectile.y - self.consts.PROJECTILE_RADIUS).astype(jnp.int32), self.SHAPE_MASKS['projectile_resized'])

        canvas = jax.lax.cond(state.projectile.active, draw_projectile, lambda c: jax.lax.cond(is_in_room, draw_aiming_dot, lambda _c: _c, c), canvas)
        return self.jr.render_from_palette(canvas, self.PALETTE)
