import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Any
from jaxatari.games.venture.core import (
    VentureConstants, GameState, PlayerState, MonsterState, ChaserState, LaserState, ProjectileState,
    UPACTIONS, DOWNACTIONS, LEFTACTIONS, RIGHTACTIONS, ROOM_PORTAL_PADDING
)

def _update_dead_for(dead_for: chex.Array) -> chex.Array:
    new_dead_for = jnp.where(dead_for > 0, dead_for - 1, dead_for)
    return jnp.where(new_dead_for == 0, -1, new_dead_for)

def _update_chaser(chaser_state: ChaserState, player_state: PlayerState, chaser_speed: float) -> ChaserState:
    dx, dy = player_state.x - chaser_state.x, player_state.y - chaser_state.y
    norm = jnp.sqrt(dx ** 2 + dy ** 2)
    safe_norm = jnp.where(norm == 0, 1.0, norm)
    new_x = chaser_state.x + (dx / safe_norm) * chaser_speed
    new_y = chaser_state.y + (dy / safe_norm) * chaser_speed
    return chaser_state.replace(x=new_x, y=new_y)

def _update_lasers(laser_state: LaserState, laser_speed: float, laser_bounds: chex.Array) -> LaserState:
    new_positions = laser_state.positions + laser_state.directions * laser_speed
    min_bounds, max_bounds = jnp.minimum(laser_bounds[:, 0], laser_bounds[:, 1]), jnp.maximum(laser_bounds[:, 0], laser_bounds[:, 1])
    should_flip = (new_positions < min_bounds) | (new_positions > max_bounds)
    new_directions = jnp.where(should_flip, -laser_state.directions, laser_state.directions)
    return LaserState(positions=jnp.clip(new_positions, min_bounds, max_bounds), directions=new_directions)

def _update_player(player_state: PlayerState, action: int, is_in_collision: bool, wall_map: chex.Array,
                   is_in_room: bool, on_main_map: bool, portal_rects: chex.Array, should_fill: chex.Array,
                   consts: VentureConstants) -> Tuple[PlayerState, chex.Array]:
    player_hw = jnp.where(is_in_room, consts.PLAYER_DETAILED_RENDER_WIDTH / 2.0, consts.PLAYER_DOT_RENDER_WIDTH / 2.0)
    player_hh = jnp.where(is_in_room, consts.PLAYER_DETAILED_RENDER_HEIGHT / 2.0, consts.PLAYER_DOT_RENDER_HEIGHT / 2.0)
    player_radius = float(consts.PLAYER_ROOM_RADIUS)

    def check_collision_rect(pos_x, pos_y):
        corners_x = jnp.array([pos_x - player_hw, pos_x + player_hw - 1, pos_x - player_hw, pos_x + player_hw - 1], dtype=jnp.int32)
        corners_y = jnp.array([pos_y - player_hh, pos_y - player_hh, pos_y + player_hh - 1, pos_y + player_hh - 1], dtype=jnp.int32)
        base_collision = jnp.any(wall_map[corners_y, corners_x] == 1)
        def check_portals():
            def check_single_corner(cx, cy):
                def check_single_rect(rect, fill):
                    rx, ry, rw, rh = rect
                    return ((cx >= rx) & (cx < rx + rw) & (cy >= ry) & (cy < ry + rh)) & fill
                return jnp.any(jax.vmap(check_single_rect)(portal_rects, should_fill))
            return jnp.any(jax.vmap(check_single_corner)(corners_x.astype(jnp.float32), corners_y.astype(jnp.float32)))
        return base_collision | jax.lax.cond(on_main_map, check_portals, lambda: False)

    def bounce_back():
        return PlayerState(x=player_state.last_valid_x, y=player_state.last_valid_y, last_valid_x=player_state.last_valid_x, last_valid_y=player_state.last_valid_y, last_dx=player_state.last_dx, last_dy=player_state.last_dy), jnp.array(False)

    def normal_move():
        dx, dy = 0.0, 0.0
        dy = jnp.where(jnp.isin(action, UPACTIONS), -consts.PLAYER_SPEED, dy)
        dy = jnp.where(jnp.isin(action, DOWNACTIONS), consts.PLAYER_SPEED, dy)
        dx = jnp.where(jnp.isin(action, LEFTACTIONS), -consts.PLAYER_SPEED, dx)
        dx = jnp.where(jnp.isin(action, RIGHTACTIONS), consts.PLAYER_SPEED, dx)
        is_moving = (dx != 0.0) | (dy != 0.0)
        norm = jnp.sqrt(dx**2 + dy**2)
        safe_norm = jnp.where(norm == 0, 1.0, norm)
        new_last_dx, new_last_dy = jnp.where(is_moving, dx / safe_norm, player_state.last_dx), jnp.where(is_moving, dy / safe_norm, player_state.last_dy)
        proposed_x, proposed_y = player_state.x + dx/2, player_state.y + dy
        min_x_clip = jnp.where(is_in_room, player_radius, player_hw)
        max_x_clip = consts.SCREEN_WIDTH - min_x_clip
        min_y_clip = consts.PLAY_AREA_Y_START + jnp.where(is_in_room, player_radius, player_hh)
        max_y_clip = consts.PLAY_AREA_Y_END - jnp.where(is_in_room, player_radius, player_hh)
        proposed_x, proposed_y = jnp.clip(proposed_x, min_x_clip, max_x_clip), jnp.clip(proposed_y, min_y_clip, max_y_clip)
        is_colliding_now = check_collision_rect(proposed_x, proposed_y)
        return PlayerState(x=proposed_x, y=proposed_y, last_valid_x=jnp.where(is_colliding_now, player_state.x, proposed_x), last_valid_y=jnp.where(is_colliding_now, player_state.y, proposed_y), last_dx=new_last_dx, last_dy=new_last_dy), is_colliding_now

    return jax.lax.cond(is_in_collision, bounce_back, normal_move)

def _update_monsters_optimized(monster_state: MonsterState, key: jax.random.PRNGKey, wall_map: chex.Array,
                               monster_speed: float, level_idx: int, world_idx: int, on_main_map: bool,
                               portal_rects: chex.Array, should_fill: chex.Array, consts: VentureConstants) -> MonsterState:
    global_level = world_idx * 5 + level_idx
    start_idx = consts.LEVEL_OFFSETS[global_level]
    MAX_MONSTERS_PER_LEVEL = 6

    def update_slice(m_x, m_y, m_dx, m_dy, m_active, k):
        def single_monster_update(x, y, dx, dy, active, skey):
            def _do_update():
                skey1, skey2, skey3 = jax.random.split(skey, 3)
                change_dir = jax.random.uniform(skey2) < consts.MONSTER_CHANGE_DIR_PROB
                angle = jax.random.uniform(skey3, minval=0, maxval=2 * jnp.pi)
                new_dx, new_dy = jnp.where(change_dir, jnp.cos(angle), dx), jnp.where(change_dir, jnp.sin(angle), dy)
                proposed_x, proposed_y = x + new_dx * monster_speed, y + new_dy * monster_speed
                hw, hh = consts.MONSTER_RENDER_WIDTH / 2.0, consts.MONSTER_RENDER_HEIGHT / 2.0
                c_x = jnp.array([proposed_x - hw, proposed_x + hw - 1, proposed_x - hw, proposed_x + hw - 1])
                c_y = jnp.array([proposed_y - hh, proposed_y - hh, proposed_y + hh - 1, proposed_y + hh - 1])
                ic_x, ic_y = jnp.clip(c_x.astype(jnp.int32), 0, consts.SCREEN_WIDTH - 1), jnp.clip(c_y.astype(jnp.int32), 0, consts.SCREEN_HEIGHT - 1)
                base_coll = jnp.any(wall_map[ic_y, ic_x] == 1)
                def check_portals():
                    def check_single_rect(rect, fill):
                        rx, ry, rw, rh = rect
                        return jnp.any(((c_x >= rx) & (c_x < rx + rw) & (c_y >= ry) & (c_y < ry + rh)) & fill)
                    return jnp.any(jax.vmap(check_single_rect)(portal_rects, should_fill))
                is_colliding = base_coll | jax.lax.cond(on_main_map, check_portals, lambda: False) | (c_x[0] < 0) | (c_x[1] >= consts.SCREEN_WIDTH) | (c_y[0] < consts.PLAY_AREA_Y_START) | (c_y[3] >= consts.PLAY_AREA_Y_END)
                return jnp.where(is_colliding, x, proposed_x), jnp.where(is_colliding, y, proposed_y), jnp.where(is_colliding, -new_dx, new_dx), jnp.where(is_colliding, -new_dy, new_dy)
            return jax.lax.cond(active, _do_update, lambda: (x, y, dx, dy))
        skeys = jax.random.split(k, MAX_MONSTERS_PER_LEVEL)
        return jax.vmap(single_monster_update)(m_x, m_y, m_dx, m_dy, m_active, skeys)

    slice_x = jax.lax.dynamic_slice(monster_state.x, (start_idx,), (MAX_MONSTERS_PER_LEVEL,))
    slice_y = jax.lax.dynamic_slice(monster_state.y, (start_idx,), (MAX_MONSTERS_PER_LEVEL,))
    slice_dx = jax.lax.dynamic_slice(monster_state.dx, (start_idx,), (MAX_MONSTERS_PER_LEVEL,))
    slice_dy = jax.lax.dynamic_slice(monster_state.dy, (start_idx,), (MAX_MONSTERS_PER_LEVEL,))
    slice_active = jax.lax.dynamic_slice(monster_state.active, (start_idx,), (MAX_MONSTERS_PER_LEVEL,))
    new_x, new_y, new_dx, new_dy = update_slice(slice_x, slice_y, slice_dx, slice_dy, slice_active, key)
    return monster_state.replace(x=jax.lax.dynamic_update_slice(monster_state.x, new_x, (start_idx,)), y=jax.lax.dynamic_update_slice(monster_state.y, new_y, (start_idx,)), dx=jax.lax.dynamic_update_slice(monster_state.dx, new_dx, (start_idx,)), dy=jax.lax.dynamic_update_slice(monster_state.dy, new_dy, (start_idx,)))

def _check_player_hazard_collision(player_state: PlayerState, monster_state: MonsterState, chaser_state: ChaserState, laser_state: LaserState, current_level: int, world_level: int, consts: VentureConstants) -> chex.Array:
    is_in_room = current_level != 0
    px_hw = jnp.where(is_in_room, consts.PLAYER_DETAILED_RENDER_WIDTH / 2.0, consts.PLAYER_DOT_RENDER_WIDTH / 2.0)
    py_hh = jnp.where(is_in_room, consts.PLAYER_DETAILED_RENDER_HEIGHT / 2.0, consts.PLAYER_DOT_RENDER_HEIGHT / 2.0)
    def monster_collision_logic(mx, my, active, dead_for):
        coll_x = (jnp.abs(player_state.x - mx) < (px_hw + consts.MONSTER_RENDER_WIDTH / 2.0))
        coll_y = (jnp.abs(player_state.y - my) < (py_hh + consts.MONSTER_RENDER_HEIGHT / 2.0))
        return (active | (dead_for > 0)) & coll_x & coll_y
    any_monster_collision = jnp.any(jax.vmap(monster_collision_logic)(monster_state.x, monster_state.y, monster_state.active, monster_state.dead_for))
    any_chaser_collision = jax.lax.cond(chaser_state.active, lambda: (jnp.abs(player_state.x - chaser_state.x) < (px_hw + consts.CHASER_RENDER_WIDTH / 2.0)) & (jnp.abs(player_state.y - chaser_state.y) < (py_hh + consts.CHASER_RENDER_HEIGHT / 2.0)), lambda: False)
    def check_laser_collision():
        x_span_start, x_span_end, y_span_start, y_span_end = consts.LASER_ROOM_SPAN
        thick_h = consts.LASER_THICKNESS / 2.0
        rect_x = jnp.array([laser_state.positions[0] - thick_h, laser_state.positions[1] - thick_h, x_span_start, x_span_start])
        rect_y = jnp.array([y_span_start, y_span_start, laser_state.positions[2] - thick_h, laser_state.positions[3] - thick_h])
        rect_w = jnp.array([consts.LASER_THICKNESS, consts.LASER_THICKNESS, x_span_end - x_span_start, x_span_end - x_span_start])
        rect_h = jnp.array([y_span_end - y_span_start, y_span_end - y_span_start, consts.LASER_THICKNESS, consts.LASER_THICKNESS])
        return jnp.any((jnp.abs(player_state.x - (rect_x + rect_w / 2.0)) < (px_hw + rect_w / 2.0)) & (jnp.abs(player_state.y - (rect_y + rect_h / 2.0)) < (py_hh + rect_h / 2.0)))
    any_laser_collision = jax.lax.cond((current_level == 1) & (world_level == 1), check_laser_collision, lambda: False)
    return any_monster_collision | any_chaser_collision | any_laser_collision

def _handle_level_transitions(state: GameState, consts: VentureConstants) -> GameState:
    px, py = state.player.x, state.player.y
    world_idx = state.world_level - 1
    level_transitions = consts.JAX_TRANSITIONS[world_idx, state.current_level]
    def _is_candidate(flat):
        rect, to_level, spawn = flat[0:4], flat[4].astype(jnp.int32), flat[5:7]
        in_portal = (px > rect[0]) & (px < rect[0] + rect[2]) & (py > rect[1]) & (py < rect[1] + rect[3])
        return (flat[2] > 0) & in_portal & ~((to_level > 0) & ~state.chests_active[to_level - 1])
    mask = jax.vmap(_is_candidate)(level_transitions)
    def transition(idx):
        flat = level_transitions[idx]
        to_level, spawn_pos = flat[4].astype(jnp.int32), flat[5:7]
        pending = state.collected_chest_in_current_visit
        new_chests = jax.lax.cond((state.current_level > 0) & (to_level == 0) & (pending != -1), lambda: state.chests_active.at[pending].set(False), lambda: state.chests_active)
        target_global_idx = (state.world_level - 1) * 5 + to_level
        start, end = consts.LEVEL_OFFSETS[target_global_idx], consts.LEVEL_OFFSETS[target_global_idx + 1]
        new_active = (jnp.arange(consts.TOTAL_MONSTERS) >= start) & (jnp.arange(consts.TOTAL_MONSTERS) < end)
        return state.replace(current_level=to_level, player=state.player.replace(x=spawn_pos[0], y=spawn_pos[1], last_valid_x=spawn_pos[0], last_valid_y=spawn_pos[1]), monsters=state.monsters.replace(active=new_active, x=consts.ALL_MONSTER_SPAWNS[:, 0], y=consts.ALL_MONSTER_SPAWNS[:, 1], dead_for=jnp.full_like(state.monsters.dead_for, -1)), level_timer=0, chaser=state.chaser.replace(active=False), lasers=LaserState(positions=consts.LASER_INITIAL_POSITIONS, directions=consts.LASER_INITIAL_DIRECTIONS), collected_chest_in_current_visit=-1, is_in_collision=False, chests_active=new_chests)
    return jax.lax.cond(jnp.any(mask), lambda: transition(jnp.argmax(mask)), lambda: state, )
