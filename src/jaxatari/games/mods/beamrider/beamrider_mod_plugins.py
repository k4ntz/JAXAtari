from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_beamrider import (
    BLUE_LINE_INIT_TABLE,
    LaneBlockerState,
    WhiteUFOUpdate,
    WhiteUFOPattern,
    _get_index_bullet,
    _get_index_falling_rock,
    _get_index_kamikaze,
    _get_index_lane_blocker,
    _get_index_rejuvenator,
    _get_index_ufo,
    _get_player_shot_screen_x,
    _get_ufo_alignment,
)


def _get_white_ufo_mask(renderer, y_pos):
    white_ufo_masks = renderer.SHAPE_MASKS["white_ufo"]
    sprite_idx = jnp.clip(_get_index_ufo(y_pos) - 1, 0, white_ufo_masks.shape[0] - 1)
    return white_ufo_masks[sprite_idx]


def _render_enemy_explosion(renderer, raster, explosion_frame, explosion_pos):
    sprite_idx, y_offset = renderer._get_enemy_explosion_visuals(explosion_frame)
    sprite = renderer.SHAPE_MASKS["enemy_explosion"][sprite_idx]
    x_pos = explosion_pos[0] + _get_ufo_alignment(explosion_pos[1])
    y_pos = explosion_pos[1] + y_offset
    return renderer.jr.render_at_clipped(raster, x_pos, y_pos, sprite)


def _render_ufo_like_enemy(renderer, raster, x_pos, y_pos, active, align=True, mask=None):
    mask = _get_white_ufo_mask(renderer, y_pos) if mask is None else mask
    render_y = jnp.where(active, y_pos, 500.0)
    aligned_x = x_pos + _get_ufo_alignment(render_y) if align else x_pos
    render_x = jnp.where(active, aligned_x, 500.0)
    return renderer.jr.render_at_clipped(raster, render_x, render_y, mask)


def _render_mask_when_active(renderer, raster, x_pos, y_pos, active, mask, align=True):
    render_y = jnp.where(active, y_pos, 500.0)
    aligned_x = x_pos + _get_ufo_alignment(render_y) if align else x_pos
    render_x = jnp.where(active, aligned_x, 500.0)
    return renderer.jr.render_at_clipped(raster, render_x, render_y, mask)


MOTHERSHIP_LASER_SEGMENT_COUNT = 20
MOTHERSHIP_LASER_SEGMENT_SPACING = 1.0
MOTHERSHIP_LASER_SPEED = 0.7
MOTHERSHIP_LASER_DURATION = 200
MOTHERSHIP_LASER_SPRITE_IDX = 1
MOTHERSHIP_LASER_PHASE_MASK = 0xFF
MOTHERSHIP_LASER_FIRING_BIT = 1 << 8
MOTHERSHIP_LASER_POST_FIRE_BIT = 1 << 9
MOTHERSHIP_LASER_LANE_SHIFT = 10

TELEPORT_UFO_PATTERN_ID = 10
TELEPORT_UFO_PATTERN_DURATION = 42
TELEPORT_UFO_PATTERN_WEIGHT = 0.2
TELEPORT_UFO_TIMER_MASK = 0xFF
TELEPORT_UFO_USED_BIT = 1 << 8
TELEPORT_UFO_LANE_OFFSETS = jnp.array([-2, -1, 1, 2], dtype=jnp.int32)

THREE_LANE_TOP_IDS = jnp.array([2, 3, 4], dtype=jnp.int32)
THREE_LANE_TOP_LANES = jnp.array([71.0, 71.0, 71.0, 81.0, 91.0, 91.0, 91.0], dtype=jnp.float32)
THREE_LANE_BOTTOM_LANES = jnp.array([0.0, 52.0, 77.0, 102.0, 154.0], dtype=jnp.float32)
THREE_LANE_TOP_TO_BOTTOM = jnp.array(
    [
        (-0.52, 4.0),
        (-0.52, 4.0),
        (-0.52, 4.0),
        (0.0, 4.0),
        (0.52, 4.0),
        (0.52, 4.0),
        (0.52, 4.0),
    ],
    dtype=jnp.float32,
)
THREE_LANE_BOTTOM_TO_TOP = jnp.array(
    [
        (-0.52, 4.0),
        (-0.52, 4.0),
        (0.0, 4.0),
        (0.52, 4.0),
        (0.52, 4.0),
    ],
    dtype=jnp.float32,
)
THREE_LANE_LEFT_BOUND = 71.0
THREE_LANE_RIGHT_BOUND = 91.0
THREE_LANE_BACKGROUND_BLUE_RGB = (45, 109, 152)
THREE_LANE_BACKGROUND_HORIZON_Y = 45
THREE_LANE_GUIDE_MARKER_SIZE = jnp.array([[2, 1]], dtype=jnp.int32)
# Preserve the stock center-three background guide markers exactly; the mod
# only removes the outer lanes visually.
THREE_LANE_GUIDE_POSITIONS = jnp.array(
    [
        (72, 53),
        (70, 67),
        (68, 81),
        (66, 95),
        (65, 109),
        (63, 119),
        (62, 129),
        (61, 139),
        (60, 149),
        (58, 159),
        (83, 55),
        (83, 69),
        (83, 83),
        (83, 97),
        (83, 111),
        (83, 121),
        (83, 131),
        (83, 141),
        (83, 151),
        (83, 161),
        (94, 51),
        (96, 65),
        (98, 79),
        (99, 93),
        (101, 107),
        (102, 117),
        (104, 127),
        (105, 137),
        (106, 147),
        (107, 157),
    ],
    dtype=jnp.int32,
)
THREE_LANE_GUIDE_SIZES = jnp.tile(THREE_LANE_GUIDE_MARKER_SIZE, (THREE_LANE_GUIDE_POSITIONS.shape[0], 1))

DOUBLE_ENEMY_SPEED_BOUNCER_PATTERN = jnp.array((0, 6, 0, 6), dtype=jnp.int32)
DOUBLE_ENEMY_SPEED_METEOROID_CYCLE_DX = jnp.array((4, 0, 2, 0, 4, 0, 4, 0), dtype=jnp.float32)
DOUBLE_ENEMY_SPEED_METEOROID_CYCLE_DY = jnp.array((2, 0, 0, 0, 2, 0, 0, 0), dtype=jnp.float32)
DOUBLE_ENEMY_SPEED_REJUV_STAGE3_DY = jnp.array((2.0, 2.0, 4.0), dtype=jnp.float32)
DOUBLE_ENEMY_SPEED_REJUV_STAGE4_DY = jnp.array((4.0, 6.0), dtype=jnp.float32)
DOUBLE_ENEMY_SPEED_STAGE1_ACCEL = 0.008
DOUBLE_ENEMY_SPEED_STAGE2_ACCEL = 0.016
DOUBLE_ENEMY_SPEED_WHITE_UFO_ATTACK_VX = 1.0
DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY = 0.5


def _get_lane_x(env, lane, y_pos):
    return env.top_lanes_x[lane] + env.lane_dx_over_dy[lane] * (y_pos - float(env.consts.TOP_CLIP))


def _is_offscreen(pos, offscreen):
    return jnp.all(pos == offscreen, axis=0)


def _double_enemy_stage_accel(y_pos, late_accel):
    return jnp.where(
        y_pos < 64,
        DOUBLE_ENEMY_SPEED_STAGE1_ACCEL,
        jnp.where(y_pos < 85, DOUBLE_ENEMY_SPEED_STAGE2_ACCEL, late_accel),
    )


def _swept_hit_mask(
    prev_x,
    prev_y,
    prev_active,
    next_x,
    next_y,
    next_active,
    prev_sizes,
    next_sizes,
    shot_x,
    shot_y,
    bullet_size,
    *,
    top_extension=0.0,
    teleported=None,
):
    if teleported is None:
        teleported = jnp.zeros_like(prev_active, dtype=jnp.bool_)

    start_x = jnp.where(prev_active, prev_x, next_x)
    start_y = jnp.where(prev_active, prev_y, next_y)
    use_next = next_active & (~teleported)
    end_x = jnp.where(use_next, next_x, start_x)
    end_y = jnp.where(use_next, next_y, start_y)

    prev_active_exp = jnp.expand_dims(prev_active, axis=-1)
    sizes = jnp.where(prev_active_exp, jnp.maximum(prev_sizes, next_sizes), next_sizes)
    height = sizes[..., 0]
    width = sizes[..., 1]

    left = jnp.minimum(start_x, end_x)
    right = jnp.maximum(start_x + width, end_x + width)
    top = jnp.minimum(start_y - top_extension, end_y - top_extension)
    bottom = jnp.maximum(start_y + height, end_y + height)

    active = prev_active | next_active
    return (
        active
        & (left < shot_x + bullet_size[1])
        & (shot_x < right)
        & (top < shot_y + bullet_size[0])
        & (shot_y < bottom)
    )


def _double_enemy_white_ufo_top_lane_substep(env, white_ufo_pos, white_ufo_vel_x, pattern_id, key):
    hold_position = jnp.logical_or(
        pattern_id == int(WhiteUFOPattern.SHOOT),
        white_ufo_pos[1] > float(env.consts.TOP_CLIP),
    )
    min_speed = float(env.consts.WHITE_UFO_TOP_LANE_MIN_SPEED) * 0.5
    turn_speed = float(env.consts.WHITE_UFO_TOP_LANE_TURN_SPEED) * 0.5

    vx = jnp.where(hold_position, 0.0, white_ufo_vel_x)
    need_boost = jnp.logical_and(jnp.logical_not(hold_position), jnp.abs(vx) < min_speed)
    random_sign = jnp.where(jax.random.uniform(key) < 0.5, -1.0, 1.0)
    direction = jnp.where(vx == 0.0, random_sign, jnp.sign(vx))
    vx = jnp.where(need_boost, direction * min_speed, vx)

    do_bounce = jnp.logical_not(hold_position)
    vx = jnp.where(
        jnp.logical_and(do_bounce, white_ufo_pos[0] >= env.consts.RIGHT_CLIP_PLAYER),
        -turn_speed,
        vx,
    )
    vx = jnp.where(
        jnp.logical_and(do_bounce, white_ufo_pos[0] <= env.consts.LEFT_CLIP_PLAYER),
        turn_speed,
        vx,
    )
    return vx, 0.0


def _double_enemy_white_ufo_normal_substep(env, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y, pattern_id, already_left):
    speed_factor = env.consts.WHITE_UFO_SPEED_FACTOR * 0.5
    retreat_mult = env.consts.WHITE_UFO_RETREAT_SPEED_MULT
    x, y = white_ufo_pos[0], white_ufo_pos[1]

    lane_x_at_y = env.top_lanes_x + env.lane_dx_over_dy * (y - float(env.consts.TOP_CLIP))

    closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_y - x))

    lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_RIGHT), 1, 0)
    lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_LEFT), -1, lane_offset)
    lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, lane_offset)
    lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)

    in_restricted_stage = y >= 86.0
    min_lane = jnp.where(in_restricted_stage, 1, 0)
    max_lane = jnp.where(in_restricted_stage, 5, 6)
    target_lane_id = jnp.clip(closest_lane_id + lane_offset, min_lane, max_lane)

    lane_vector = env.lane_vectors_t2b[target_lane_id]
    target_lane_x = lane_x_at_y[target_lane_id]

    is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
    is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
    is_kamikaze = pattern_id == int(WhiteUFOPattern.KAMIKAZE)
    is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))

    cross_track = target_lane_x - x
    distance_to_lane = jnp.abs(cross_track)
    direction = jnp.sign(cross_track)

    def seek_lane(_):
        attack_vx = jnp.where(direction == 0, 0.0, direction * (DOUBLE_ENEMY_SPEED_WHITE_UFO_ATTACK_VX * 0.5))
        retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)

        new_vx = jnp.where(is_retreat | is_kamikaze | is_triple, retreat_vx, attack_vx)

        normal_vy = DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY * 0.5
        retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
        move_back_vy = -lane_vector[1] * speed_factor
        kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult
        triple_vy = DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY * 0.5

        new_vy = jnp.where(is_retreat, retreat_vy, normal_vy)
        new_vy = jnp.where(is_move_back, move_back_vy, new_vy)
        new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
        new_vy = jnp.where(is_triple, triple_vy, new_vy)

        return new_vx, new_vy

    def follow_lane(_):
        normal_vx = lane_vector[0] * speed_factor
        normal_vy = lane_vector[1] * speed_factor

        retreat_vx = -lane_vector[0] * speed_factor * retreat_mult
        retreat_vy = -lane_vector[1] * speed_factor * retreat_mult

        move_back_vx = -lane_vector[0] * speed_factor
        move_back_vy = -lane_vector[1] * speed_factor

        kamikaze_vx = lane_vector[0] * speed_factor * retreat_mult
        kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult

        triple_vy = DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY * 0.5

        new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_move_back, move_back_vx, normal_vx))
        new_vx = jnp.where(is_kamikaze, kamikaze_vx, new_vx)

        new_vy = jnp.where(is_retreat, retreat_vy, jnp.where(is_move_back, move_back_vy, normal_vy))
        new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
        new_vy = jnp.where(is_triple, triple_vy, new_vy)

        return new_vx, new_vy

    return jax.lax.cond(
        distance_to_lane <= 0.25,
        follow_lane,
        seek_lane,
        operand=None,
    )


def _canonical_three_lane_ufo_lane(lane: jnp.ndarray) -> jnp.ndarray:
    lane = lane.astype(jnp.int32)
    return jnp.where(lane <= 2, 2, jnp.where(lane >= 4, 4, 3))


def _is_three_lane_shootable(lane: jnp.ndarray) -> jnp.ndarray:
    lane = lane.astype(jnp.int32)
    return (lane >= 2) & (lane <= 4)


def _get_mothership_y(env):
    return float(env.consts.MOTHERSHIP_EMERGE_Y - env.consts.MOTHERSHIP_HEIGHT)


def _get_mothership_stop_x(env, lane):
    head_y = jnp.array(float(env.consts.MOTHERSHIP_EMERGE_Y) + 1.0, dtype=jnp.float32)
    laser_left_x = _get_lane_x(env, lane, head_y) + _get_ufo_alignment(head_y)
    ship_half_width = env.consts.MOTHERSHIP_SPRITE_SIZE[1] / 2.0
    laser_half_width = env.consts.ENEMY_SHOT_SPRITE_SIZES[MOTHERSHIP_LASER_SPRITE_IDX][1] / 2.0
    return jnp.array(laser_left_x + laser_half_width - ship_half_width, dtype=jnp.float32)


def _pack_mothership_laser_timer(lane, phase_timer, firing, post_fire):
    lane = lane.astype(jnp.int32)
    phase_timer = phase_timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_PHASE_MASK)
    firing_bits = jnp.where(firing, jnp.int32(MOTHERSHIP_LASER_FIRING_BIT), jnp.int32(0))
    post_fire_bits = jnp.where(post_fire, jnp.int32(MOTHERSHIP_LASER_POST_FIRE_BIT), jnp.int32(0))
    return (lane << MOTHERSHIP_LASER_LANE_SHIFT) | post_fire_bits | firing_bits | phase_timer


def _get_mothership_laser_phase_timer(timer):
    return timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_PHASE_MASK)


def _get_mothership_laser_lane_from_timer(timer):
    lane = timer.astype(jnp.int32) >> MOTHERSHIP_LASER_LANE_SHIFT
    return jnp.clip(lane, 1, 5)


def _is_mothership_laser_firing(timer):
    return (timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_FIRING_BIT)) != 0


def _has_mothership_laser_post_fire(timer):
    return (timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_POST_FIRE_BIT)) != 0


def _get_mothership_laser_lane(env, state):
    return _get_mothership_laser_lane_from_timer(state.level.mothership_timer)


def _get_mothership_laser_segments(env, state):
    timer = state.level.mothership_timer.astype(jnp.int32)
    phase_timer = _get_mothership_laser_phase_timer(timer)
    laser_active = (state.level.mothership_stage.astype(jnp.int32) == 2) & _is_mothership_laser_firing(timer) & (
        phase_timer < MOTHERSHIP_LASER_DURATION
    )
    lane = _get_mothership_laser_lane(env, state)
    segment_idx = jnp.arange(MOTHERSHIP_LASER_SEGMENT_COUNT, dtype=jnp.float32)
    head_y = jnp.array(float(env.consts.MOTHERSHIP_EMERGE_Y) + 1.0, dtype=jnp.float32)
    head_y = head_y + phase_timer.astype(jnp.float32) * MOTHERSHIP_LASER_SPEED
    seg_y = head_y + (segment_idx * MOTHERSHIP_LASER_SEGMENT_SPACING)
    seg_x = _get_lane_x(env, lane, seg_y) + _get_ufo_alignment(seg_y)
    visible = laser_active & (seg_y <= float(env.consts.BOTTOM_CLIP)) & (seg_y >= float(env.consts.TOP_CLIP) - 8.0)
    return seg_x, seg_y, visible


def _laser_hits_box(env, state, left_x, top_y, width, height):
    seg_x, seg_y, visible = _get_mothership_laser_segments(env, state)
    seg_size = env.enemy_shot_sprite_sizes[MOTHERSHIP_LASER_SPRITE_IDX]
    seg_h = seg_size[0]
    seg_w = seg_size[1]
    hits = visible & (
        (seg_x < left_x + width)
        & (left_x < seg_x + seg_w)
        & (seg_y - env.consts.ENEMY_HITBOX_TOP_EXTENSION < top_y + height)
        & (top_y < seg_y + seg_h)
    )
    return jnp.any(hits)


def _is_teleport_ufo_pattern(pattern_id):
    return pattern_id.astype(jnp.int32) == jnp.int32(TELEPORT_UFO_PATTERN_ID)


def _get_teleport_ufo_remaining(timer):
    return timer.astype(jnp.int32) & jnp.int32(TELEPORT_UFO_TIMER_MASK)


def _has_teleport_ufo_been_used(timer):
    return (timer.astype(jnp.int32) & jnp.int32(TELEPORT_UFO_USED_BIT)) != 0


def _pack_teleport_ufo_timer(remaining, used):
    remaining = remaining.astype(jnp.int32) & jnp.int32(TELEPORT_UFO_TIMER_MASK)
    used_bits = jnp.where(used, jnp.int32(TELEPORT_UFO_USED_BIT), jnp.int32(0))
    return remaining | used_bits


def _get_teleport_ufo_lane_bounds_from_stage(stage):
    in_restricted_stage = stage.astype(jnp.int32) >= 6
    min_lane = jnp.where(in_restricted_stage, 1, 0)
    max_lane = jnp.where(in_restricted_stage, 5, 6)
    return min_lane.astype(jnp.int32), max_lane.astype(jnp.int32)


def _get_teleport_ufo_lane_bounds_from_y(y_pos):
    in_restricted_stage = y_pos.astype(jnp.float32) >= 86.0
    min_lane = jnp.where(in_restricted_stage, 1, 0)
    max_lane = jnp.where(in_restricted_stage, 5, 6)
    return min_lane.astype(jnp.int32), max_lane.astype(jnp.int32)


def _get_teleport_ufo_candidate_lanes(current_lane, min_lane, max_lane):
    current_lane = current_lane.astype(jnp.int32)
    candidates = current_lane + TELEPORT_UFO_LANE_OFFSETS
    valid = (candidates >= min_lane) & (candidates <= max_lane)
    return candidates.astype(jnp.int32), valid


def _init_white_ufo_pattern_timer(pattern, duration):
    is_triple = (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
    is_teleport = _is_teleport_ufo_pattern(pattern)
    timer = jnp.where(is_triple, duration | (15 << 3), duration)
    return jnp.where(is_teleport, _pack_teleport_ufo_timer(duration, jnp.array(False)), timer)


def _get_fog_of_war_top_y(state, consts) -> jnp.ndarray:
    active_lines = state.level.line_positions >= 0
    safe_line_positions = jnp.where(
        active_lines,
        state.level.line_positions.astype(jnp.int32),
        jnp.array(consts.BLUE_LINE_OFFSCREEN_Y, dtype=jnp.int32),
    )
    top_line_y = jnp.where(
        jnp.any(active_lines),
        jnp.min(safe_line_positions),
        jnp.array(consts.MIN_BLUE_LINE_POS, dtype=jnp.int32),
    )
    top_dot_ufo_visible_limit = jnp.array(48, dtype=jnp.int32)
    return jnp.maximum(top_line_y - 1, top_dot_ufo_visible_limit)


def _get_fog_of_war_cutoff_y(fog_top, consts) -> jnp.ndarray:
    player_y = jnp.array(consts.PLAYER_POS_Y, dtype=jnp.int32)
    visible_band_height = jnp.maximum(player_y - fog_top, 0)
    return player_y - jnp.floor_divide(visible_band_height, 3)


def _triangle_wave(values, period):
    phase = jnp.mod(values, period)
    half_period = period // 2
    return jnp.abs(phase - half_period)


def _apply_fog_of_war(renderer, raster, state):
    fog_top = _get_fog_of_war_top_y(state, renderer.consts)
    fog_cutoff = _get_fog_of_war_cutoff_y(fog_top, renderer.consts)
    fog_height = jnp.maximum(fog_cutoff - fog_top, 1)
    fog_left = jnp.array(8, dtype=jnp.int32)
    fog_right = fog_left + jnp.array(renderer.SHAPE_MASKS["blue_line"].shape[1], dtype=jnp.int32)
    xx = renderer.jr._xx
    yy = renderer.jr._yy
    rel_y = jnp.clip(yy - fog_top, 0, fog_height)
    in_fog_band = (yy >= fog_top) & (yy < fog_cutoff) & (xx >= fog_left) & (xx < fog_right)

    black_id = jnp.array(renderer.COLOR_TO_ID[(0, 0, 0)], dtype=raster.dtype)
    shadow_id = jnp.array(renderer.COLOR_TO_ID[(80, 0, 132)], dtype=raster.dtype)
    body_id = jnp.array(renderer.COLOR_TO_ID[(104, 25, 154)], dtype=raster.dtype)
    edge_id = jnp.array(renderer.COLOR_TO_ID[(45, 109, 152)], dtype=raster.dtype)

    local_x = xx - fog_left
    phase_fast = state.steps // 16
    phase_slow = state.steps // 27
    wave_a = _triangle_wave(local_x + phase_fast, 42) // 7
    wave_b = _triangle_wave((local_x * 3) + phase_slow, 64) // 11
    wave_c = _triangle_wave((local_x * 5) + phase_fast, 96) // 16
    boundary_offset = jnp.clip(wave_a - wave_b + wave_c - 2, -3, 4)
    local_cutoff = fog_cutoff + boundary_offset
    depth_to_edge = local_cutoff - yy

    in_fog_band = in_fog_band & (yy < local_cutoff)

    wave_scroll = state.steps // 18
    wave_band = jnp.mod(depth_to_edge + wave_scroll, 14)
    fill_id = jnp.where(
        depth_to_edge <= 3,
        edge_id,
        jnp.where(
            wave_band < 3,
            edge_id,
            jnp.where(wave_band < 7, body_id, jnp.where(wave_band < 10, shadow_id, black_id)),
        ),
    )

    boundary_noise = jnp.mod((local_x // 6) + (yy // 3) + (state.steps // 19), 7)
    edge_shadow = (boundary_noise <= 1) & (depth_to_edge <= 4) & (depth_to_edge >= 2)
    fill_id = jnp.where(edge_shadow, shadow_id, fill_id)

    return jnp.where(in_fog_band, fill_id, raster)


class FogOfWarMod(JaxAtariInternalModPlugin):
    """Hide the upper two thirds of the playfield behind opaque fog."""

    @partial(jax.jit, static_argnums=(0,))
    def _render_mothership(self, raster, state):
        renderer = self._env.renderer
        raster = type(renderer)._render_mothership(renderer, raster, state)
        return _apply_fog_of_war(renderer, raster, state)


class HardcoreMod(JaxAtariInternalModPlugin):
    """Start with one life and never allow extra lives."""

    constants_overrides = {
        "STARTING_LIVES": 1,
        "MAX_LIVES": 1,
    }


class DoubleEnemySpeedMod(JaxAtariInternalModPlugin):
    """Double Beamrider enemy movement speed and preserve projectile hits."""

    constants_overrides = {
        "WHITE_UFO_SPEED_FACTOR": 0.2,
        "WHITE_UFO_SHOT_SPEED_FACTOR": 1.6,
        "WHITE_UFO_TOP_LANE_MIN_SPEED": 0.6,
        "WHITE_UFO_TOP_LANE_TURN_SPEED": 1.0,
        "BOUNCER_SPEED_PATTERN": (0, 6, 0, 6),
        "BOUNCER_SWITCH_SPEED_X": 2.4,
        "BOUNCER_SWITCH_SPEED_Y": 2.0,
        "CHASING_METEOROID_LANE_SPEED": 1.8,
        "CHASING_METEOROID_ACCEL": 0.09,
        "CHASING_METEOROID_CYCLE_DX": (4, 0, 2, 0, 4, 0, 4, 0),
        "CHASING_METEOROID_CYCLE_DY": (2, 0, 0, 0, 2, 0, 0, 0),
        "CHASING_METEOROID_LANE_ALIGN_THRESHOLD": 3.0,
        "FALLING_ROCK_INIT_VEL": 0.14,
        "FALLING_ROCK_ACCEL": 0.042,
        "LANE_BLOCKER_INIT_VEL": 0.04,
        "LANE_BLOCKER_SINK_INTERVAL": 1,
        "COIN_SPEED_Y": 1.0,
        "COIN_SPEED_X": 3.875,
    }

    attribute_overrides = {
        "bouncer_speed_pattern": DOUBLE_ENEMY_SPEED_BOUNCER_PATTERN,
        "chasing_meteoroid_cycle_dx": DOUBLE_ENEMY_SPEED_METEOROID_CYCLE_DX,
        "chasing_meteoroid_cycle_dy": DOUBLE_ENEMY_SPEED_METEOROID_CYCLE_DY,
    }

    @partial(jax.jit, static_argnums=(0,))
    def _collision_handler(
        self,
        state,
        new_white_ufo_pos,
        new_shot_pos,
        new_bullet_type,
        current_patterns,
        current_timers,
        current_spawn_delays,
        key,
        shot_x,
    ):
        env = self._env
        enemies_raw = new_white_ufo_pos.T

        prev_ufo_pos = state.level.white_ufo_pos
        prev_ufo_x = prev_ufo_pos[0] + _get_ufo_alignment(prev_ufo_pos[1])
        prev_ufo_y = prev_ufo_pos[1]
        prev_ufo_indices = jnp.clip(_get_index_ufo(prev_ufo_y) - 1, 0, len(env.consts.UFO_SPRITE_SIZES) - 1)
        prev_ufo_sizes = jnp.take(env.ufo_sprite_sizes, prev_ufo_indices, axis=0)
        prev_active = ~_is_offscreen(prev_ufo_pos, env.enemy_offscreen_ufo)

        ufo_x = new_white_ufo_pos[0, :] + _get_ufo_alignment(new_white_ufo_pos[1, :])
        ufo_y = new_white_ufo_pos[1, :]
        ufo_indices = jnp.clip(_get_index_ufo(ufo_y) - 1, 0, len(env.consts.UFO_SPRITE_SIZES) - 1)
        ufo_sizes = jnp.take(env.ufo_sprite_sizes, ufo_indices, axis=0)
        next_active = ~_is_offscreen(new_white_ufo_pos, env.enemy_offscreen_ufo)
        teleported = prev_active & next_active & ((ufo_y < prev_ufo_y) | (jnp.abs(ufo_x - prev_ufo_x) > 12.0))

        shot_y = new_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, new_bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)

        hit_mask_ufo = _swept_hit_mask(
            prev_ufo_x,
            prev_ufo_y,
            prev_active,
            ufo_x,
            ufo_y,
            next_active,
            prev_ufo_sizes,
            ufo_sizes,
            shot_x,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
            teleported=teleported,
        )

        hit_index = jnp.argmax(hit_mask_ufo)
        hit_exists_ufo = jnp.any(hit_mask_ufo)

        should_respawn = state.level.white_ufo_left > 3
        respawn_pos = env.white_ufo_respawn_pos
        offscreen_pos = env.enemy_offscreen
        target_pos = jnp.where(should_respawn, respawn_pos, offscreen_pos)

        enemy_pos_after_hit = enemies_raw.at[hit_index].set(target_pos).T

        new_patterns = jnp.where(hit_mask_ufo, int(WhiteUFOPattern.IDLE), current_patterns)
        new_timers = jnp.where(hit_mask_ufo, 0, current_timers)
        new_spawn_delays = jnp.where(hit_mask_ufo, jax.random.randint(key, (3,), 1, 301), current_spawn_delays)

        player_shot_pos = jnp.where(hit_exists_ufo, env.bullet_offscreen, new_shot_pos)
        enemy_pos = jnp.where(hit_exists_ufo, enemy_pos_after_hit, new_white_ufo_pos)
        white_ufo_left = jnp.where(
            hit_exists_ufo,
            jnp.maximum(state.level.white_ufo_left - 1, 0),
            state.level.white_ufo_left,
        )
        clamped_sector = jnp.minimum(state.sector, 89)
        ufo_score = 40 + 4 * clamped_sector
        score = jnp.where(hit_exists_ufo, state.score + ufo_score, state.score)
        return (
            enemy_pos,
            player_shot_pos,
            new_patterns,
            new_timers,
            new_spawn_delays,
            white_ufo_left,
            score,
            hit_mask_ufo,
            hit_exists_ufo,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _collisions_step(
        self,
        state,
        player_x,
        vel_x,
        player_shot_pos,
        player_shot_vel,
        player_shot_frame,
        torpedos_left,
        bullet_type,
        shooting_cooldown,
        shooting_delay,
        shot_type_pending,
        enemy_updates,
        key,
    ):
        env = self._env

        ufo_update = enemy_updates["ufo"]
        (
            bouncer_pos,
            bouncer_vel,
            bouncer_state,
            bouncer_timer,
            bouncer_active,
            bouncer_lane,
            bouncer_step_index,
        ) = enemy_updates["bouncer"]
        (
            chasing_meteoroid_pos,
            chasing_meteoroid_active,
            chasing_meteoroid_vel_y,
            chasing_meteoroid_phase,
            chasing_meteoroid_frame,
            chasing_meteoroid_lane,
            chasing_meteoroid_side,
            chasing_meteoroid_spawn_timer,
            chasing_meteoroid_remaining,
            chasing_meteoroid_wave_active,
        ) = enemy_updates["meteoroid"]
        (falling_rock_pos, falling_rock_active, falling_rock_lane, falling_rock_vel_y) = enemy_updates["rock"]
        (
            lane_blocker_pos,
            lane_blocker_active,
            lane_blocker_lane,
            lane_blocker_vel_y,
            lane_blocker_phase,
            lane_blocker_timer,
        ) = enemy_updates["blocker"]
        (kamikaze_pos, kamikaze_active, kamikaze_lane, kamikaze_vel_y, kamikaze_tracking, kamikaze_spawn_timer) = enemy_updates["kamikaze"]
        (coin_pos, coin_active, coin_timer, coin_side, coin_spawn_count) = enemy_updates["coin"]
        (rejuv_pos, rejuv_active, rejuv_dead, rejuv_frame, rejuv_lane) = enemy_updates["rejuv"]
        (enemy_shot_pos, enemy_shot_lane, enemy_shot_timer, shot_hit_count) = enemy_updates["shots"]

        shot_x_screen = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            env.consts.LASER_ID,
        )

        (
            white_ufo_pos,
            player_shot_pos,
            white_ufo_pattern_id,
            white_ufo_pattern_timer,
            white_ufo_spawn_delay,
            white_ufo_left,
            score,
            hit_mask_ufo,
            hit_exists_ufo,
        ) = self._collision_handler(
            state,
            ufo_update.pos,
            player_shot_pos,
            bullet_type,
            ufo_update.pattern_id,
            ufo_update.pattern_timer,
            ufo_update.spawn_delay,
            key,
            shot_x_screen,
        )

        pre_collision_bouncer_pos = bouncer_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_bouncer_y = state.level.bouncer_pos[1]
        prev_bouncer_x = state.level.bouncer_pos[0] + _get_ufo_alignment(prev_bouncer_y)
        next_bouncer_y = bouncer_pos[1]
        next_bouncer_x = bouncer_pos[0] + _get_ufo_alignment(next_bouncer_y)
        bouncer_hit = _swept_hit_mask(
            prev_bouncer_x,
            prev_bouncer_y,
            state.level.bouncer_active,
            next_bouncer_x,
            next_bouncer_y,
            bouncer_active,
            env.bouncer_sprite_size,
            env.bouncer_sprite_size,
            shot_x_screen,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
        )
        bullet_type_is_laser = bullet_type == env.consts.LASER_ID
        bouncer_destroyed = bouncer_hit & jnp.logical_not(bullet_type_is_laser)
        bouncer_pos = jnp.where(bouncer_destroyed, env.enemy_offscreen, bouncer_pos)
        bouncer_active = jnp.where(bouncer_destroyed, False, bouncer_active)
        player_shot_pos = jnp.where(bouncer_hit, env.bullet_offscreen, player_shot_pos)
        score = jnp.where(bouncer_destroyed, score + 80, score)

        pre_collision_meteoroid_pos = chasing_meteoroid_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_meteoroid_y = state.level.chasing_meteoroid_pos[1]
        prev_meteoroid_x = state.level.chasing_meteoroid_pos[0] + _get_ufo_alignment(prev_meteoroid_y).astype(
            state.level.chasing_meteoroid_pos.dtype
        )
        next_meteoroid_y = chasing_meteoroid_pos[1]
        next_meteoroid_x = chasing_meteoroid_pos[0] + _get_ufo_alignment(next_meteoroid_y).astype(chasing_meteoroid_pos.dtype)
        meteoroid_collision_mask = _swept_hit_mask(
            prev_meteoroid_x,
            prev_meteoroid_y,
            state.level.chasing_meteoroid_active,
            next_meteoroid_x,
            next_meteoroid_y,
            chasing_meteoroid_active,
            env.meteoroid_sprite_size,
            env.meteoroid_sprite_size,
            shot_x_screen,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
        )
        chasing_meteoroid_hit_mask = meteoroid_collision_mask & (bullet_type == env.consts.TORPEDO_ID)
        hit_exists_meteoroid = jnp.any(chasing_meteoroid_hit_mask)
        collision_exists_m = jnp.any(meteoroid_collision_mask)
        hit_index = jnp.argmax(chasing_meteoroid_hit_mask)
        hit_one_hot = jax.nn.one_hot(hit_index, env.consts.CHASING_METEOROID_MAX, dtype=chasing_meteoroid_pos.dtype)
        hit_one_hot_bool = hit_one_hot.astype(jnp.bool_)
        chasing_meteoroid_pos = jnp.where(
            hit_exists_meteoroid,
            chasing_meteoroid_pos + (env.enemy_offscreen[:, None] - chasing_meteoroid_pos) * hit_one_hot[None, :],
            chasing_meteoroid_pos,
        )
        chasing_meteoroid_active = jnp.where(hit_exists_meteoroid, jnp.where(hit_one_hot_bool, False, chasing_meteoroid_active), chasing_meteoroid_active)
        chasing_meteoroid_vel_y = jnp.where(hit_exists_meteoroid, jnp.where(hit_one_hot_bool, 0.0, chasing_meteoroid_vel_y), chasing_meteoroid_vel_y)
        chasing_meteoroid_phase = jnp.where(hit_exists_meteoroid, jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_phase), chasing_meteoroid_phase)
        chasing_meteoroid_frame = jnp.where(hit_exists_meteoroid, jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_frame), chasing_meteoroid_frame)
        chasing_meteoroid_lane = jnp.where(hit_exists_meteoroid, jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_lane), chasing_meteoroid_lane)
        chasing_meteoroid_side = jnp.where(hit_exists_meteoroid, jnp.where(hit_one_hot_bool, 1, chasing_meteoroid_side), chasing_meteoroid_side)
        player_shot_pos = jnp.where(collision_exists_m, env.bullet_offscreen, player_shot_pos)

        pre_collision_rock_pos = falling_rock_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_rock_y = state.level.falling_rock_pos[1]
        prev_rock_x = state.level.falling_rock_pos[0] + _get_ufo_alignment(prev_rock_y).astype(state.level.falling_rock_pos.dtype)
        next_rock_y = falling_rock_pos[1]
        next_rock_x = falling_rock_pos[0] + _get_ufo_alignment(next_rock_y).astype(falling_rock_pos.dtype)
        prev_rock_indices = jnp.clip(_get_index_falling_rock(prev_rock_y) - 1, 0, len(env.consts.FALLING_ROCK_SPRITE_SIZES) - 1)
        next_rock_indices = jnp.clip(_get_index_falling_rock(next_rock_y) - 1, 0, len(env.consts.FALLING_ROCK_SPRITE_SIZES) - 1)
        prev_rock_sizes = jnp.take(env.falling_rock_sprite_sizes, prev_rock_indices, axis=0)
        next_rock_sizes = jnp.take(env.falling_rock_sprite_sizes, next_rock_indices, axis=0)
        rock_hit_mask = _swept_hit_mask(
            prev_rock_x,
            prev_rock_y,
            state.level.falling_rock_active,
            next_rock_x,
            next_rock_y,
            falling_rock_active,
            prev_rock_sizes,
            next_rock_sizes,
            shot_x_screen,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
        )
        hit_exists_rock = jnp.any(rock_hit_mask)
        falling_rock_hit_mask = rock_hit_mask & (bullet_type == env.consts.TORPEDO_ID)
        falling_rock_pos = jnp.where(falling_rock_hit_mask[None, :], env.enemy_offscreen_falling, falling_rock_pos)
        falling_rock_active = jnp.where(falling_rock_hit_mask, False, falling_rock_active)
        player_shot_pos = jnp.where(hit_exists_rock, env.bullet_offscreen, player_shot_pos)

        pre_collision_lane_blocker_pos = lane_blocker_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_blocker_y = state.level.lane_blocker_pos[1]
        prev_blocker_x = state.level.lane_blocker_pos[0] + _get_ufo_alignment(prev_blocker_y).astype(state.level.lane_blocker_pos.dtype)
        next_blocker_y = lane_blocker_pos[1]
        next_blocker_x = lane_blocker_pos[0] + _get_ufo_alignment(next_blocker_y).astype(lane_blocker_pos.dtype)
        prev_blocker_indices = jnp.clip(_get_index_lane_blocker(prev_blocker_y) - 1, 0, len(env.consts.LANE_BLOCKER_SPRITE_SIZES) - 1)
        next_blocker_indices = jnp.clip(_get_index_lane_blocker(next_blocker_y) - 1, 0, len(env.consts.LANE_BLOCKER_SPRITE_SIZES) - 1)
        prev_blocker_indices = jnp.where(prev_blocker_indices == 2, 3, prev_blocker_indices)
        next_blocker_indices = jnp.where(next_blocker_indices == 2, 3, next_blocker_indices)
        prev_blocker_sizes = jnp.take(env.lane_blocker_sprite_sizes, prev_blocker_indices, axis=0)
        next_blocker_sizes = jnp.take(env.lane_blocker_sprite_sizes, next_blocker_indices, axis=0)
        lane_blocker_hit_mask = _swept_hit_mask(
            prev_blocker_x,
            prev_blocker_y,
            state.level.lane_blocker_active,
            next_blocker_x,
            next_blocker_y,
            lane_blocker_active,
            prev_blocker_sizes,
            next_blocker_sizes,
            shot_x_screen,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
        )
        hit_exists_lane_blocker = jnp.any(lane_blocker_hit_mask)
        blocker_destroyed = lane_blocker_hit_mask & (bullet_type == env.consts.TORPEDO_ID)
        blocker_retreat = lane_blocker_hit_mask & (bullet_type != env.consts.TORPEDO_ID)
        lane_blocker_pos = jnp.where(blocker_destroyed[None, :], env.enemy_offscreen_lane_blocker, lane_blocker_pos)
        lane_blocker_active = jnp.where(blocker_destroyed, False, lane_blocker_active)
        lane_blocker_phase = jnp.where(blocker_retreat, int(LaneBlockerState.RETREAT), lane_blocker_phase)
        lane_blocker_phase = jnp.where(blocker_destroyed, int(LaneBlockerState.DESCEND), lane_blocker_phase)
        lane_blocker_timer = jnp.where(blocker_retreat | blocker_destroyed, 0, lane_blocker_timer)
        lane_blocker_vel_y = jnp.where(blocker_retreat | blocker_destroyed, 0.0, lane_blocker_vel_y)
        player_shot_pos = jnp.where(hit_exists_lane_blocker, env.bullet_offscreen, player_shot_pos)

        pre_collision_kamikaze_pos = kamikaze_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_kamikaze_y = state.level.kamikaze_pos[1, 0]
        prev_kamikaze_x = state.level.kamikaze_pos[0, 0] + _get_ufo_alignment(prev_kamikaze_y).astype(state.level.kamikaze_pos.dtype)
        next_kamikaze_y = kamikaze_pos[1, 0]
        next_kamikaze_x = kamikaze_pos[0, 0] + _get_ufo_alignment(next_kamikaze_y).astype(kamikaze_pos.dtype)
        prev_kamikaze_indices = jnp.clip(_get_index_kamikaze(prev_kamikaze_y) - 1, 0, 3)
        next_kamikaze_indices = jnp.clip(_get_index_kamikaze(next_kamikaze_y) - 1, 0, 3)
        prev_kamikaze_sizes = jnp.take(env.lane_blocker_sprite_sizes, prev_kamikaze_indices, axis=0)
        next_kamikaze_sizes = jnp.take(env.lane_blocker_sprite_sizes, next_kamikaze_indices, axis=0)
        kamikaze_hit = _swept_hit_mask(
            prev_kamikaze_x,
            prev_kamikaze_y,
            state.level.kamikaze_active[0],
            next_kamikaze_x,
            next_kamikaze_y,
            kamikaze_active[0],
            prev_kamikaze_sizes,
            next_kamikaze_sizes,
            shot_x_screen,
            shot_y,
            bullet_size,
        )
        kamikaze_destroyed = jnp.array([kamikaze_hit & (bullet_type == env.consts.TORPEDO_ID)])
        kamikaze_pos = jnp.where(kamikaze_destroyed[0], env.kamikaze_offscreen, kamikaze_pos)
        kamikaze_active = jnp.array([jnp.where(kamikaze_destroyed[0], False, kamikaze_active[0])])
        player_shot_pos = jnp.where(kamikaze_hit, env.bullet_offscreen, player_shot_pos)
        hit_exists_kamikaze = jnp.array([kamikaze_hit])

        pre_collision_coin_pos = coin_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_coin_y = state.level.coin_pos[1]
        prev_coin_x = state.level.coin_pos[0] + _get_ufo_alignment(prev_coin_y).astype(state.level.coin_pos.dtype)
        next_coin_y = coin_pos[1]
        next_coin_x = coin_pos[0] + _get_ufo_alignment(next_coin_y).astype(coin_pos.dtype)
        hit_mask_coin = _swept_hit_mask(
            prev_coin_x,
            prev_coin_y,
            state.level.coin_active,
            next_coin_x,
            next_coin_y,
            coin_active,
            env.coin_sprite_size,
            env.coin_sprite_size,
            shot_x_screen,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
        )
        hit_exists_coin = jnp.any(hit_mask_coin)
        coin_pos = jnp.where(hit_mask_coin[None, :], env.enemy_offscreen_col, coin_pos)
        coin_active = jnp.where(hit_mask_coin, False, coin_active)
        player_shot_pos = jnp.where(hit_exists_coin, env.bullet_offscreen, player_shot_pos)
        clamped_sector = jnp.minimum(state.sector, 89)
        score = jnp.where(
            hit_exists_coin,
            score + 300 + 30 * clamped_sector + jnp.maximum(state.lives - 1, 0) * (100 + 10 * clamped_sector),
            score,
        )

        pre_collision_rejuv_pos = rejuv_pos
        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, env.consts.LASER_ID)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        prev_rejuv_y = state.level.rejuvenator_pos[1]
        prev_rejuv_x = state.level.rejuvenator_pos[0] + _get_ufo_alignment(prev_rejuv_y)
        next_rejuv_y = rejuv_pos[1]
        next_rejuv_x = rejuv_pos[0] + _get_ufo_alignment(next_rejuv_y)
        prev_rejuv_indices = jnp.where(
            state.level.rejuvenator_dead,
            4,
            jnp.clip(_get_index_rejuvenator(prev_rejuv_y) - 1, 0, 3),
        )
        next_rejuv_indices = jnp.where(
            rejuv_dead,
            4,
            jnp.clip(_get_index_rejuvenator(next_rejuv_y) - 1, 0, 3),
        )
        prev_rejuv_sizes = jnp.take(env.rejuvenator_sprite_sizes, prev_rejuv_indices, axis=0)
        next_rejuv_sizes = jnp.take(env.rejuvenator_sprite_sizes, next_rejuv_indices, axis=0)
        rejuv_hit = _swept_hit_mask(
            prev_rejuv_x,
            prev_rejuv_y,
            state.level.rejuvenator_active,
            next_rejuv_x,
            next_rejuv_y,
            rejuv_active,
            prev_rejuv_sizes,
            next_rejuv_sizes,
            shot_x_screen,
            shot_y,
            bullet_size,
            top_extension=float(env.consts.ENEMY_HITBOX_TOP_EXTENSION),
        )
        rejuv_hit = rejuv_hit & (shot_y < env.consts.BOTTOM_CLIP)
        rejuv_destroyed = rejuv_hit & rejuv_dead & (bullet_type == env.consts.TORPEDO_ID)
        rejuv_dead = jnp.logical_or(rejuv_dead, rejuv_hit)
        rejuv_active = jnp.where(rejuv_destroyed, False, rejuv_active)
        rejuv_pos = jnp.where(rejuv_destroyed, env.enemy_offscreen, rejuv_pos)
        player_shot_pos = jnp.where(rejuv_hit, env.bullet_offscreen, player_shot_pos)
        score = jnp.where(rejuv_destroyed, score + 150, score)
        bullet_size = jnp.take(env.bullet_sprite_sizes, bullet_idx, axis=0)
        is_torpedo = bullet_type == env.consts.TORPEDO_ID

        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_active = player_shot_pos[1] < env.consts.BOTTOM_CLIP
        hit_mothership = env._mothership_bullet_collision(
            state.level.mothership_stage,
            state.level.mothership_position,
            player_shot_pos,
            shot_x_screen,
            shot_active,
            bullet_size,
            is_torpedo
        )
        player_shot_pos = jnp.where(hit_mothership, env.bullet_offscreen, player_shot_pos)
        score = jnp.where(hit_mothership, score + 300 + 30 * clamped_sector + jnp.maximum(state.lives - 1, 0) * (100 + 10 * clamped_sector), score)

        shot_x_screen = _get_player_shot_screen_x(player_shot_pos, player_shot_vel, bullet_type, env.consts.LASER_ID)
        shot_active = player_shot_pos[1] < env.consts.BOTTOM_CLIP
        hit_mask_shot, hit_exists_shot = env._enemy_shot_bullet_collision(
            enemy_shot_pos,
            enemy_shot_timer,
            player_shot_pos,
            shot_x_screen,
            shot_active,
            bullet_size,
            is_torpedo
        )
        enemy_shot_pos_pre_collision = enemy_shot_pos
        enemy_shot_pos = jnp.where(hit_mask_shot[None, :], env.bullet_offscreen_shots, enemy_shot_pos)
        enemy_shot_timer = jnp.where(hit_mask_shot, 0, enemy_shot_timer)
        player_shot_pos = jnp.where(hit_exists_shot, env.bullet_offscreen, player_shot_pos)

        bullet_hit_any = (
            hit_exists_ufo
            | bouncer_hit
            | hit_exists_meteoroid
            | hit_exists_rock
            | hit_exists_lane_blocker
            | rejuv_hit
            | hit_mothership
            | hit_exists_shot
            | hit_exists_coin
        )
        player_shot_frame, shooting_cooldown = env._resolve_player_projectile(
            state,
            player_shot_frame,
            shooting_cooldown,
            bullet_hit_any,
        )

        any_explosion_triggered = (
            jnp.any(hit_mask_ufo)
            | bouncer_destroyed
            | jnp.any(chasing_meteoroid_hit_mask)
            | jnp.any(falling_rock_hit_mask)
            | jnp.any(blocker_destroyed)
            | jnp.any(kamikaze_destroyed)
            | jnp.any(hit_mask_coin)
            | jnp.any(hit_mask_shot)
            | rejuv_destroyed
        )

        ufo_explosion_frame, ufo_explosion_pos = env._update_enemy_explosions(
            state.level.ufo_explosion_frame,
            state.level.ufo_explosion_pos,
            hit_mask_ufo,
            ufo_update.pos,
            env.enemy_offscreen_ufo,
        )
        bouncer_explosion_frame, bouncer_explosion_pos = env._update_enemy_explosions(
            state.level.bouncer_explosion_frame[None],
            state.level.bouncer_explosion_pos[:, None],
            bouncer_destroyed[None],
            pre_collision_bouncer_pos[:, None],
            env.enemy_offscreen_col,
        )
        chasing_meteoroid_explosion_frame, chasing_meteoroid_explosion_pos = env._update_enemy_explosions(
            state.level.chasing_meteoroid_explosion_frame,
            state.level.chasing_meteoroid_explosion_pos,
            chasing_meteoroid_hit_mask,
            pre_collision_meteoroid_pos,
            env.enemy_offscreen_meteoroids,
        )
        falling_rock_explosion_frame, falling_rock_explosion_pos = env._update_enemy_explosions(
            state.level.falling_rock_explosion_frame,
            state.level.falling_rock_explosion_pos,
            falling_rock_hit_mask,
            pre_collision_rock_pos,
            env.enemy_offscreen_falling,
        )
        lane_blocker_explosion_frame, lane_blocker_explosion_pos = env._update_enemy_explosions(
            state.level.lane_blocker_explosion_frame,
            state.level.lane_blocker_explosion_pos,
            blocker_destroyed,
            pre_collision_lane_blocker_pos,
            env.enemy_offscreen_lane_blocker,
        )
        enemy_shot_explosion_frame, enemy_shot_explosion_pos = env._update_enemy_explosions(
            state.level.enemy_shot_explosion_frame,
            state.level.enemy_shot_explosion_pos,
            hit_mask_shot,
            enemy_shot_pos_pre_collision,
            env.enemy_offscreen_shots,
        )
        coin_explosion_frame, coin_explosion_pos = env._update_enemy_explosions(
            state.level.coin_explosion_frame,
            state.level.coin_explosion_pos,
            hit_mask_coin,
            pre_collision_coin_pos,
            env.enemy_offscreen_coins,
        )
        kamikaze_explosion_frame, kamikaze_explosion_pos = env._update_enemy_explosions(
            state.level.kamikaze_explosion_frame,
            state.level.kamikaze_explosion_pos,
            kamikaze_destroyed,
            pre_collision_kamikaze_pos,
            env.kamikaze_offscreen,
        )
        rejuv_explosion_frame, rejuv_explosion_pos = env._update_enemy_explosions(
            state.level.rejuvenator_explosion_frame[None],
            state.level.rejuvenator_explosion_pos[:, None],
            rejuv_destroyed[None],
            pre_collision_rejuv_pos[:, None],
            env.enemy_offscreen_col,
        )

        (
            total_hit_count,
            bouncer_hits_p,
            chasing_meteoroid_hits_p,
            rejuv_hit_player,
            gain_life,
            lose_life_rejuv,
            kamikaze_hits_player,
        ) = env._player_collision_check(
            state,
            player_x,
            white_ufo_pos,
            bouncer_pos,
            bouncer_active,
            chasing_meteoroid_pos,
            chasing_meteoroid_active,
            rejuv_pos,
            rejuv_active,
            rejuv_dead,
            falling_rock_pos,
            falling_rock_active,
            lane_blocker_pos,
            lane_blocker_active,
            lane_blocker_lane,
            lane_blocker_phase,
            kamikaze_pos,
            kamikaze_active,
            shot_hit_count,
        )

        bouncer_active = jnp.where(bouncer_hits_p, False, bouncer_active)
        bouncer_pos = jnp.where(bouncer_hits_p, env.enemy_offscreen, bouncer_pos)
        rejuv_active = jnp.where(rejuv_hit_player, False, rejuv_active)
        rejuv_pos = jnp.where(rejuv_hit_player, env.enemy_offscreen, rejuv_pos)
        kamikaze_active = jnp.where(kamikaze_hits_player, jnp.array([False]), kamikaze_active)
        kamikaze_pos = jnp.where(kamikaze_hits_player, env.kamikaze_offscreen, kamikaze_pos)
        chasing_meteoroid_offscreen = env.enemy_offscreen_meteoroids
        chasing_meteoroid_active = jnp.where(chasing_meteoroid_hits_p, False, chasing_meteoroid_active)
        chasing_meteoroid_pos = jnp.where(chasing_meteoroid_hits_p[None, :], chasing_meteoroid_offscreen, chasing_meteoroid_pos)
        reached_player = jnp.logical_and(chasing_meteoroid_active, chasing_meteoroid_pos[1] >= float(env.consts.PLAYER_POS_Y))
        chasing_meteoroid_active = jnp.where(reached_player, False, chasing_meteoroid_active)
        chasing_meteoroid_pos = jnp.where(reached_player[None, :], chasing_meteoroid_offscreen, chasing_meteoroid_pos)

        return {
            "player": (
                player_x,
                vel_x,
                player_shot_pos,
                player_shot_vel,
                player_shot_frame,
                torpedos_left,
                bullet_type,
                shooting_cooldown,
                shooting_delay,
                shot_type_pending,
            ),
            "ufo": (
                white_ufo_pos,
                white_ufo_pattern_id,
                white_ufo_pattern_timer,
                white_ufo_left,
                ufo_update.vel,
                ufo_update.time_on_lane,
                ufo_update.attack_time,
                hit_mask_ufo,
                ufo_update.already_left,
                white_ufo_spawn_delay,
                ufo_update.rngs,
            ),
            "bouncer": (bouncer_pos, bouncer_vel, bouncer_state, bouncer_timer, bouncer_active, bouncer_lane, bouncer_step_index),
            "meteoroid": (
                chasing_meteoroid_pos,
                chasing_meteoroid_active,
                chasing_meteoroid_vel_y,
                chasing_meteoroid_phase,
                chasing_meteoroid_frame,
                chasing_meteoroid_lane,
                chasing_meteoroid_side,
                chasing_meteoroid_spawn_timer,
                chasing_meteoroid_remaining,
                chasing_meteoroid_wave_active,
            ),
            "rock": (falling_rock_pos, falling_rock_active, falling_rock_lane, falling_rock_vel_y),
            "blocker": (lane_blocker_pos, lane_blocker_active, lane_blocker_lane, lane_blocker_vel_y, lane_blocker_phase, lane_blocker_timer),
            "kamikaze": (kamikaze_pos, kamikaze_active, kamikaze_lane, kamikaze_vel_y, kamikaze_tracking, kamikaze_spawn_timer),
            "coin": (coin_pos, coin_active, coin_timer, coin_side, coin_spawn_count),
            "rejuv": (rejuv_pos, rejuv_active, rejuv_dead, rejuv_frame, rejuv_lane, gain_life, rejuv_hit),
            "shots": (enemy_shot_pos, enemy_shot_lane, enemy_shot_timer),
            "score": score,
            "total_hit_count": total_hit_count,
            "hit_mothership": hit_mothership,
            "any_explosion_triggered": any_explosion_triggered,
            "explosions": (
                ufo_explosion_frame,
                ufo_explosion_pos,
                bouncer_explosion_frame[0],
                bouncer_explosion_pos[:, 0],
                chasing_meteoroid_explosion_frame,
                chasing_meteoroid_explosion_pos,
                falling_rock_explosion_frame,
                falling_rock_explosion_pos,
                lane_blocker_explosion_frame,
                lane_blocker_explosion_pos,
                enemy_shot_explosion_frame,
                enemy_shot_explosion_pos,
                coin_explosion_frame,
                coin_explosion_pos,
                kamikaze_explosion_frame,
                kamikaze_explosion_pos,
                rejuv_explosion_frame[0],
                rejuv_explosion_pos[:, 0],
            ),
        }

    @partial(jax.jit, static_argnums=(0,))
    def _rejuvenator_step(self, state, key):
        env = self._env
        pos = state.level.rejuvenator_pos
        active = state.level.rejuvenator_active
        dead = state.level.rejuvenator_dead
        frame = state.level.rejuvenator_frame
        lane = state.level.rejuvenator_lane
        mothership_active = state.level.mothership_stage > 0

        key_spawn, key_lane = jax.random.split(key)
        spawn_roll = jax.random.uniform(key_spawn)
        should_spawn = jnp.logical_and.reduce(
            jnp.array(
                [
                    jnp.logical_not(active),
                    jnp.logical_not(mothership_active),
                    spawn_roll < env.consts.REJUVENATOR_SPAWN_PROB,
                    state.level.white_ufo_left > 0,
                ]
            )
        )

        spawn_lane = jax.random.randint(key_lane, (), 1, 6)
        spawn_x = env.top_lanes_x[spawn_lane]
        spawn_y = float(env.consts.TOP_CLIP)

        pos = jnp.where(should_spawn, jnp.array([spawn_x, spawn_y]), pos)
        lane = jnp.where(should_spawn, spawn_lane, lane)
        active = jnp.where(should_spawn, True, active)
        dead = jnp.where(should_spawn, False, dead)
        frame = jnp.where(should_spawn, 0, frame)

        y = pos[1]
        stage = _get_index_rejuvenator(y)
        should_move_normal = jax.lax.switch(
            jnp.clip(stage - 1, 0, 3),
            [
                lambda: (state.steps % 6 == 0) | (state.steps % 6 == 2),
                lambda: (state.steps % 2) == 0,
                lambda: (state.steps % 2) == 0,
                lambda: (state.steps % 2) == 0,
            ],
        )
        should_move = jnp.logical_and(active, jnp.where(dead, True, should_move_normal))

        dy = jax.lax.switch(
            jnp.clip(stage - 1, 0, 3),
            [
                lambda: 2.0,
                lambda: 2.0,
                lambda: jnp.take(DOUBLE_ENEMY_SPEED_REJUV_STAGE3_DY, frame % 3),
                lambda: jnp.take(DOUBLE_ENEMY_SPEED_REJUV_STAGE4_DY, frame % 2),
            ],
        )

        new_y = y + jnp.where(should_move, dy, 0.0)
        new_x = jnp.take(env.top_lanes_x, lane) + jnp.take(env.lane_dx_over_dy, lane) * (new_y - float(env.consts.TOP_CLIP))

        pos = jnp.where(active, jnp.array([new_x, new_y]), pos)
        frame = jnp.where(should_move, frame + 1, frame)

        off_screen = new_y > env.consts.PLAYER_POS_Y + 1.0
        active = jnp.where(off_screen, False, active)
        pos = jnp.where(active, jnp.array([new_x, new_y]), env.enemy_offscreen)

        return pos, active, dead, frame, lane

    @partial(jax.jit, static_argnums=(0,))
    def _falling_rock_step(self, state, key):
        env = self._env
        pos = state.level.falling_rock_pos
        active = state.level.falling_rock_active
        lane = state.level.falling_rock_lane
        vel_y = state.level.falling_rock_vel_y
        mothership_active = state.level.mothership_stage > 0

        key_spawn, key_lane = jax.random.split(key)

        is_level_2_plus = state.sector >= 2
        spawn_roll = jax.random.uniform(key_spawn)
        can_spawn = jnp.logical_and.reduce(
            jnp.array(
                [
                    is_level_2_plus,
                    jnp.sum(active, dtype=jnp.int32) < env.consts.FALLING_ROCK_MAX,
                    jnp.logical_not(mothership_active),
                    state.level.white_ufo_left > 0,
                ]
            )
        )
        should_spawn = jnp.logical_and(can_spawn, spawn_roll < env.consts.FALLING_ROCK_SPAWN_PROB)

        spawn_lane = jax.random.randint(key_lane, (), 1, 6)
        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask)
        one_hot = jax.nn.one_hot(slot, env.consts.FALLING_ROCK_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        spawn_x = env.top_lanes_x[spawn_lane]
        spawn_y = float(env.consts.FALLING_ROCK_SPAWN_Y)
        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)

        pos = jnp.where(should_spawn, pos + (spawn_pos[:, None] - pos) * one_hot[None, :], pos)
        active = jnp.where(should_spawn, jnp.where(one_hot_bool, True, active), active)
        lane = jnp.where(should_spawn, jnp.where(one_hot_bool, spawn_lane, lane), lane)
        vel_y = jnp.where(should_spawn, jnp.where(one_hot_bool, env.consts.FALLING_ROCK_INIT_VEL, vel_y), vel_y)

        y = pos[1]
        accel = _double_enemy_stage_accel(y, env.consts.FALLING_ROCK_ACCEL)
        new_vel_y = vel_y + accel
        new_y = y + new_vel_y

        target_lane_dx_over_dy = jnp.take(env.lane_dx_over_dy, lane)
        target_lanes_top_x = jnp.take(env.top_lanes_x, lane)
        new_x = target_lanes_top_x + target_lane_dx_over_dy * (new_y - float(env.consts.TOP_CLIP))

        pos = jnp.where(active[None, :], jnp.stack([new_x, new_y]), pos)
        vel_y = jnp.where(active, new_vel_y, vel_y)

        off_screen = new_y > env.consts.FALLING_ROCK_BOTTOM_CLIP
        active = jnp.where(off_screen, False, active)
        pos = jnp.where(active[None, :], pos, env.enemy_offscreen_falling)
        vel_y = jnp.where(active, vel_y, 0.0)

        return pos, active, lane, vel_y

    @partial(jax.jit, static_argnums=(0,))
    def _lane_blocker_step(self, state, key):
        env = self._env
        pos = state.level.lane_blocker_pos
        active = state.level.lane_blocker_active
        lane = state.level.lane_blocker_lane
        vel_y = state.level.lane_blocker_vel_y
        phase = state.level.lane_blocker_phase
        hold_timer = state.level.lane_blocker_timer
        mothership_active = state.level.mothership_stage > 0

        key_spawn, key_lane = jax.random.split(key)

        is_level_10_plus = state.sector >= env.consts.LANE_BLOCKER_START_LEVEL
        spawn_roll = jax.random.uniform(key_spawn)
        can_spawn = jnp.logical_and.reduce(
            jnp.array(
                [
                    is_level_10_plus,
                    jnp.sum(active, dtype=jnp.int32) < env.consts.LANE_BLOCKER_MAX,
                    jnp.logical_not(mothership_active),
                    state.level.white_ufo_left > 0,
                ]
            )
        )
        should_spawn = jnp.logical_and(can_spawn, spawn_roll < env.consts.LANE_BLOCKER_SPAWN_PROB)

        spawn_lane = jax.random.choice(key_lane, env.middle_lane_spawn)
        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask)
        one_hot = jax.nn.one_hot(slot, env.consts.LANE_BLOCKER_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        spawn_x = env.top_lanes_x[spawn_lane]
        spawn_y = float(env.consts.LANE_BLOCKER_SPAWN_Y)
        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)

        pos = jnp.where(should_spawn, pos + (spawn_pos[:, None] - pos) * one_hot[None, :], pos)
        active = jnp.where(should_spawn, jnp.where(one_hot_bool, True, active), active)
        lane = jnp.where(should_spawn, jnp.where(one_hot_bool, spawn_lane, lane), lane)
        vel_y = jnp.where(should_spawn, jnp.where(one_hot_bool, env.consts.LANE_BLOCKER_INIT_VEL, vel_y), vel_y)
        phase = jnp.where(should_spawn, jnp.where(one_hot_bool, int(LaneBlockerState.DESCEND), phase), phase)
        hold_timer = jnp.where(should_spawn, jnp.where(one_hot_bool, 0, hold_timer), hold_timer)

        y = pos[1]
        bottom_y = float(env.consts.LANE_BLOCKER_BOTTOM_Y)

        descend = active & (phase == int(LaneBlockerState.DESCEND))
        hold = active & (phase == int(LaneBlockerState.HOLD))
        sink = active & (phase == int(LaneBlockerState.SINK))
        retreat = active & (phase == int(LaneBlockerState.RETREAT))

        accel = _double_enemy_stage_accel(y, env.consts.FALLING_ROCK_ACCEL)
        falling_vel_y = vel_y + accel
        new_y_descend = y + falling_vel_y
        reached_bottom = new_y_descend >= bottom_y
        y_descend = jnp.where(reached_bottom, bottom_y, new_y_descend)
        vel_descend = jnp.where(reached_bottom, 0.0, falling_vel_y)
        phase_descend = jnp.where(reached_bottom, int(LaneBlockerState.HOLD), int(LaneBlockerState.DESCEND))
        hold_timer_descend = jnp.where(reached_bottom, env.consts.LANE_BLOCKER_HOLD_FRAMES, hold_timer)

        hold_timer_next = jnp.maximum(hold_timer - 1, 0)
        start_sink = hold_timer_next == 0
        phase_hold = jnp.where(start_sink, int(LaneBlockerState.SINK), int(LaneBlockerState.HOLD))

        sink_step = (state.steps % env.consts.LANE_BLOCKER_SINK_INTERVAL) == 0
        sink_dy = jnp.where(sink_step, 1.0, 0.0)
        y_sink = y + sink_dy

        retreat_speed = env.lane_vectors_t2b[:, 1] * env.consts.WHITE_UFO_SPEED_FACTOR * env.consts.WHITE_UFO_RETREAT_SPEED_MULT
        retreat_speed = retreat_speed * env.consts.LANE_BLOCKER_RETREAT_SPEED_MULT
        lane_retreat_speed = jnp.take(retreat_speed, lane)
        y_retreat = y - lane_retreat_speed

        new_y = jnp.where(descend, y_descend, y)
        new_y = jnp.where(hold, bottom_y, new_y)
        new_y = jnp.where(sink, y_sink, new_y)
        new_y = jnp.where(retreat, y_retreat, new_y)

        new_vel_y = vel_y
        new_vel_y = jnp.where(descend, vel_descend, new_vel_y)
        new_vel_y = jnp.where(hold | sink | retreat, 0.0, new_vel_y)

        new_phase = phase
        new_phase = jnp.where(descend, phase_descend, new_phase)
        new_phase = jnp.where(hold, phase_hold, new_phase)
        new_phase = jnp.where(sink, int(LaneBlockerState.SINK), new_phase)
        new_phase = jnp.where(retreat, int(LaneBlockerState.RETREAT), new_phase)

        new_timer = hold_timer
        new_timer = jnp.where(descend, hold_timer_descend, new_timer)
        new_timer = jnp.where(hold, hold_timer_next, new_timer)
        new_timer = jnp.where(sink | retreat, 0, new_timer)

        target_lane_dx_over_dy = jnp.take(env.lane_dx_over_dy, lane)
        target_lanes_top_x = jnp.take(env.top_lanes_x, lane)
        new_x = target_lanes_top_x + target_lane_dx_over_dy * (new_y - float(env.consts.TOP_CLIP))

        retreat_done = retreat & (new_y <= float(env.consts.LANE_BLOCKER_SPAWN_Y))
        sink_done = sink & ((new_y - bottom_y) >= float(env.consts.LANE_BLOCKER_HEIGHT))
        done = retreat_done | sink_done

        active = jnp.where(done, False, active)
        new_phase = jnp.where(done, int(LaneBlockerState.DESCEND), new_phase)
        new_timer = jnp.where(done, 0, new_timer)
        new_vel_y = jnp.where(done, 0.0, new_vel_y)

        pos = jnp.where(active[None, :], jnp.stack([new_x, new_y]), env.enemy_offscreen_lane_blocker)
        new_vel_y = jnp.where(active, new_vel_y, 0.0)

        return pos, active, lane, new_vel_y, new_phase, new_timer

    @partial(jax.jit, static_argnums=(0,))
    def _kamikaze_step(self, state, key):
        env = self._env
        pos = state.level.kamikaze_pos
        active = state.level.kamikaze_active[0]
        lane = state.level.kamikaze_lane[0]
        vel_y = state.level.kamikaze_vel_y[0]
        tracking = state.level.kamikaze_tracking[0]
        spawn_timer = state.level.kamikaze_spawn_timer[0]
        mothership_active = state.level.mothership_stage > 0

        is_level_12_plus = state.sector >= env.consts.KAMIKAZE_START_SECTOR
        spawn_timer = jnp.where(is_level_12_plus & jnp.logical_not(active), spawn_timer + 1, 0)

        should_spawn = is_level_12_plus & jnp.logical_not(active) & (spawn_timer >= env.consts.KAMIKAZE_SPAWN_INTERVAL)
        should_spawn = jnp.logical_and(should_spawn, jnp.logical_not(mothership_active))

        spawn_lane = jax.random.randint(key, (), 1, 6)
        spawn_x = env.top_lanes_x[spawn_lane]
        spawn_y = float(env.consts.KAMIKAZE_START_Y)

        pos = jnp.where(should_spawn, jnp.array([[spawn_x], [spawn_y]]), pos)
        active = jnp.where(should_spawn, True, active)
        lane = jnp.where(should_spawn, spawn_lane, lane)
        vel_y = jnp.where(should_spawn, env.consts.LANE_BLOCKER_INIT_VEL, vel_y)
        tracking = jnp.where(should_spawn, True, tracking)
        spawn_timer = jnp.where(should_spawn, 0, spawn_timer)

        y = pos[1, 0]
        accel = _double_enemy_stage_accel(y, env.consts.FALLING_ROCK_ACCEL)
        new_vel_y = vel_y + accel
        new_y = y + new_vel_y

        player_x = state.level.player_pos
        player_lane = jnp.argmin(jnp.abs(env.bottom_lanes - player_x)) + 1
        should_track = active & (new_y >= env.consts.KAMIKAZE_TRACK_Y) & tracking

        current_lane_x = env.top_lanes_x[lane] + env.lane_dx_over_dy[lane] * (new_y - float(env.consts.TOP_CLIP))
        target_lane_x = env.top_lanes_x[player_lane] + env.lane_dx_over_dy[player_lane] * (new_y - float(env.consts.TOP_CLIP))

        lateral_dist = target_lane_x - pos[0, 0]
        dx_dir = jnp.sign(lateral_dist)
        dx_lateral = 2.0 * new_vel_y * dx_dir
        reached_lane = jnp.abs(lateral_dist) <= jnp.abs(dx_lateral)

        new_x = jnp.where(
            should_track,
            jnp.where(reached_lane, target_lane_x, pos[0, 0] + dx_lateral),
            current_lane_x,
        )

        lane = jnp.where(should_track & reached_lane, player_lane, lane)
        tracking = jnp.where(should_track & reached_lane, False, tracking)

        off_screen = new_y > env.consts.PLAYER_POS_Y + 5.0
        active = jnp.where(off_screen, False, active)
        final_pos = jnp.where(active, jnp.array([[new_x], [new_y]]), env.kamikaze_offscreen)

        return (
            final_pos,
            jnp.array([active]),
            jnp.array([lane]),
            jnp.array([new_vel_y]),
            jnp.array([tracking]),
            jnp.array([spawn_timer]),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_normal(self, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y, pattern_id, already_left):
        env = self._env
        speed_factor = env.consts.WHITE_UFO_SPEED_FACTOR
        retreat_mult = env.consts.WHITE_UFO_RETREAT_SPEED_MULT
        x, y = white_ufo_pos[0], white_ufo_pos[1]

        lane_x_at_y = env.top_lanes_x + env.lane_dx_over_dy * (y - float(env.consts.TOP_CLIP))

        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_y - x))

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_LEFT), -1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)

        in_restricted_stage = y >= 86.0
        min_lane = jnp.where(in_restricted_stage, 1, 0)
        max_lane = jnp.where(in_restricted_stage, 5, 6)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, min_lane, max_lane)

        lane_vector = env.lane_vectors_t2b[target_lane_id]
        target_lane_x = lane_x_at_y[target_lane_id]

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        is_kamikaze = pattern_id == int(WhiteUFOPattern.KAMIKAZE)
        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))

        cross_track = target_lane_x - x
        distance_to_lane = jnp.abs(cross_track)
        direction = jnp.sign(cross_track)

        def seek_lane(_):
            attack_vx = jnp.where(direction == 0, 0.0, direction * DOUBLE_ENEMY_SPEED_WHITE_UFO_ATTACK_VX)
            retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)

            new_vx = jnp.where(is_retreat | is_kamikaze | is_triple, retreat_vx, attack_vx)

            normal_vy = DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
            move_back_vy = -lane_vector[1] * speed_factor
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult
            triple_vy = DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY

            new_vy = jnp.where(is_retreat, retreat_vy, normal_vy)
            new_vy = jnp.where(is_move_back, move_back_vy, new_vy)
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)

            return new_vx, new_vy

        def follow_lane(_):
            normal_vx = lane_vector[0] * speed_factor
            normal_vy = lane_vector[1] * speed_factor

            retreat_vx = -lane_vector[0] * speed_factor * retreat_mult
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult

            move_back_vx = -lane_vector[0] * speed_factor
            move_back_vy = -lane_vector[1] * speed_factor

            kamikaze_vx = lane_vector[0] * speed_factor * retreat_mult
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult

            triple_vy = DOUBLE_ENEMY_SPEED_WHITE_UFO_NORMAL_VY

            new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_move_back, move_back_vx, normal_vx))
            new_vx = jnp.where(is_kamikaze, kamikaze_vx, new_vx)

            new_vy = jnp.where(is_retreat, retreat_vy, jnp.where(is_move_back, move_back_vy, normal_vy))
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)

            return new_vx, new_vy

        return jax.lax.cond(
            distance_to_lane <= 0.25,
            follow_lane,
            seek_lane,
            operand=None,
        )

    def _white_ufo_step(
        self,
        sector,
        white_ufo_position,
        white_ufo_vel,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        key,
    ):
        env = self._env
        white_ufo_vel_x = white_ufo_vel[0]
        white_ufo_vel_y = white_ufo_vel[1]

        offscreen_pos = env.enemy_offscreen
        is_offscreen = jnp.all(white_ufo_position == offscreen_pos)

        new_key, key_motion_1, key_motion_2, choice_key1, choice_key2, float_key = jax.random.split(key, 6)
        float_rolls = jax.random.uniform(float_key, shape=(4,))

        spawn_delay_roll = float_rolls[0]
        retreat_roll = float_rolls[2]
        start_roll = float_rolls[3]

        spawn_delay = jnp.maximum(spawn_delay - 1, 0)

        pattern_id, pattern_timer, time_on_lane, attack_time = env._white_ufo_update_pattern_state(
            sector,
            white_ufo_position,
            time_on_lane,
            attack_time,
            already_left,
            spawn_delay,
            pattern_id,
            pattern_timer,
            retreat_roll,
            start_roll,
            choice_key1,
            choice_key2,
        )

        requires_lane_motion = env._white_ufo_pattern_requires_lane_motion(pattern_id)

        def substep(position, vel_x, vel_y, already_left_now, key_motion):
            on_top_lane_now = position[1] <= env.consts.TOP_CLIP
            already_left_now = already_left_now | jnp.logical_not(on_top_lane_now)

            def follow_lane(_):
                return _double_enemy_white_ufo_normal_substep(env, position, vel_x, vel_y, pattern_id, already_left_now)

            def stay_on_top(_):
                return _double_enemy_white_ufo_top_lane_substep(env, position, vel_x, pattern_id, key_motion)

            sub_vel_x, sub_vel_y = jax.lax.cond(
                requires_lane_motion,
                follow_lane,
                stay_on_top,
                operand=None,
            )

            new_x = position[0] + sub_vel_x
            new_y = position[1] + sub_vel_y

            on_top_lane_next = new_y <= env.consts.TOP_CLIP
            clipped_x = jnp.clip(new_x, env.consts.LEFT_CLIP_PLAYER, env.consts.RIGHT_CLIP_PLAYER)
            new_x = jnp.where(on_top_lane_next, clipped_x, new_x)
            new_y = jnp.clip(new_y, env.consts.TOP_CLIP, env.consts.PLAYER_POS_Y + 1.0)

            return jnp.array([new_x, new_y]), sub_vel_x, sub_vel_y, already_left_now

        pos_1, vel_x_1, vel_y_1, already_left = substep(
            white_ufo_position,
            white_ufo_vel_x,
            white_ufo_vel_y,
            already_left,
            key_motion_1,
        )
        pos_2, white_ufo_vel_x, white_ufo_vel_y, already_left = substep(
            pos_1,
            vel_x_1,
            vel_y_1,
            already_left,
            key_motion_2,
        )

        new_x = pos_2[0]
        new_y = pos_2[1]

        should_respawn = jnp.logical_and(
            jnp.logical_not(is_offscreen),
            jnp.logical_or(
                new_x < 0,
                jnp.logical_or(
                    new_x > env.consts.SCREEN_WIDTH,
                    new_y > env.consts.PLAYER_POS_Y
                )
            )
        )

        white_ufo_position = jnp.where(should_respawn, jnp.array([81.0, 43.0]), jnp.array([new_x, new_y]))
        white_ufo_vel_x = jnp.where(should_respawn, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(should_respawn, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(should_respawn, 0, time_on_lane)
        attack_time = jnp.where(should_respawn, 0, attack_time)
        new_spawn_delay_val = jnp.floor(spawn_delay_roll * 300.0).astype(jnp.int32) + 1
        spawn_delay = jnp.where(should_respawn, new_spawn_delay_val, spawn_delay)
        pattern_id = jnp.where(should_respawn, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(should_respawn, 0, pattern_timer)

        white_ufo_position = jnp.where(is_offscreen, offscreen_pos, white_ufo_position)
        white_ufo_vel_x = jnp.where(is_offscreen, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(is_offscreen, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(is_offscreen, 0, time_on_lane)
        attack_time = jnp.where(is_offscreen, 0, attack_time)
        spawn_delay = jnp.where(is_offscreen, 0, spawn_delay)
        pattern_id = jnp.where(is_offscreen, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(is_offscreen, 0, pattern_timer)

        return (
            white_ufo_position,
            white_ufo_vel_x,
            white_ufo_vel_y,
            time_on_lane,
            attack_time,
            already_left,
            spawn_delay,
            pattern_id,
            pattern_timer,
            new_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _chasing_meteoroid_step(
        self,
        state,
        player_x,
        player_vel,
        white_ufo_left,
        key,
    ):
        env = self._env
        pos = state.level.chasing_meteoroid_pos
        active = state.level.chasing_meteoroid_active
        vel_y = state.level.chasing_meteoroid_vel_y
        phase = state.level.chasing_meteoroid_phase
        frame = state.level.chasing_meteoroid_frame
        lane = state.level.chasing_meteoroid_lane
        side = state.level.chasing_meteoroid_side
        spawn_timer = state.level.chasing_meteoroid_spawn_timer
        remaining = state.level.chasing_meteoroid_remaining
        wave_active = state.level.chasing_meteoroid_wave_active

        is_sector_6_plus = state.sector >= 6
        key_start, key_wave, key_interval, key_side = jax.random.split(key, 4)

        ms_stage = state.level.mothership_stage
        ms_pos = state.level.mothership_position
        is_ltr = (state.sector % 2) != 0
        ms_too_far = jnp.where(is_ltr, ms_pos > 100, ms_pos < 60)

        can_spawn_in_ms = (white_ufo_left == 0) & (ms_stage < 3) & jnp.logical_not(ms_too_far)

        # The faster mothership halves the usable spawn window, so the mothership
        # phase compensates by starting waves twice as often.
        start_chance = jnp.where(can_spawn_in_ms, 0.1, jnp.where(is_sector_6_plus & (white_ufo_left > 0), 0.0021, 0.0))

        start_wave = jnp.logical_and(
            jax.random.uniform(key_start) < start_chance,
            jnp.logical_not(wave_active)
        )

        wave_active = jnp.where(start_wave, True, wave_active)

        min_w = jnp.where(white_ufo_left == 0, env.consts.CHASING_METEOROID_WAVE_MIN, 1)
        max_w = jnp.where(white_ufo_left == 0, env.consts.CHASING_METEOROID_WAVE_MAX, 3)

        wave_count = jax.random.randint(
            key_wave,
            (),
            min_w,
            max_w + 1,
        )
        remaining = jnp.where(start_wave, wave_count, remaining)
        spawn_timer = jnp.where(start_wave, 0, spawn_timer)

        should_cancel = jnp.logical_and(jnp.logical_not(is_sector_6_plus), white_ufo_left > 0)
        should_cancel = jnp.logical_or(should_cancel, (white_ufo_left == 0) & (ms_stage >= 3))
        should_cancel = jnp.logical_or(should_cancel, (white_ufo_left == 0) & ms_too_far)

        wave_finished = wave_active & (remaining == 0) & jnp.all(jnp.logical_not(active))

        wave_active = jnp.where(should_cancel | wave_finished, False, wave_active)
        remaining = jnp.where(should_cancel, 0, remaining)
        spawn_timer = jnp.where(should_cancel, 0, spawn_timer)

        spawn_timer = jnp.where(
            jnp.logical_and(wave_active, remaining > 0),
            jnp.maximum(spawn_timer - 1, 0),
            spawn_timer,
        )

        should_spawn = jnp.logical_and.reduce(jnp.array([
            wave_active,
            remaining > 0,
            spawn_timer == 0,
        ]))
        has_slot = jnp.any(jnp.logical_not(active))
        should_spawn = jnp.logical_and(should_spawn, has_slot)

        spawn_interval = jax.random.randint(
            key_interval,
            (),
            env.consts.CHASING_METEOROID_SPAWN_INTERVAL_MIN,
            env.consts.CHASING_METEOROID_SPAWN_INTERVAL_MAX + 1,
        )
        spawn_interval = jnp.where(
            can_spawn_in_ms,
            jnp.maximum((spawn_interval + 1) // 2, 1),
            spawn_interval,
        )
        spawn_side = jnp.where(jax.random.uniform(key_side) < 0.5, 1, -1)
        spawn_x = jnp.where(
            spawn_side == 1,
            float(env.consts.LEFT_CLIP_PLAYER),
            float(env.consts.RIGHT_CLIP_PLAYER),
        )
        spawn_y = float(env.consts.CHASING_METEOROID_SPAWN_Y)

        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask)
        one_hot = jax.nn.one_hot(slot, env.consts.CHASING_METEOROID_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)
        pos_spawned = pos + (spawn_pos[:, None] - pos) * one_hot[None, :]
        active_spawned = jnp.where(one_hot_bool, True, active)
        vel_y_spawned = jnp.where(one_hot_bool, 0.0, vel_y)
        phase_spawned = jnp.where(one_hot_bool, 0, phase)
        frame_spawned = jnp.where(one_hot_bool, 0, frame)
        lane_spawned = jnp.where(one_hot_bool, 0, lane)
        side_spawned = jnp.where(one_hot_bool, spawn_side, side)

        pos = jnp.where(should_spawn, pos_spawned, pos)
        active = jnp.where(should_spawn, active_spawned, active)
        vel_y = jnp.where(should_spawn, vel_y_spawned, vel_y)
        phase = jnp.where(should_spawn, phase_spawned, phase)
        frame = jnp.where(should_spawn, frame_spawned, frame)
        lane = jnp.where(should_spawn, lane_spawned, lane)
        side = jnp.where(should_spawn, side_spawned, side)
        remaining = jnp.where(should_spawn, remaining - 1, remaining)
        spawn_timer = jnp.where(should_spawn, spawn_interval, spawn_timer)

        cycle_dx = env.chasing_meteoroid_cycle_dx
        cycle_dy = env.chasing_meteoroid_cycle_dy
        dy_a = jnp.take(cycle_dy, frame)
        new_y_a = pos[1] + dy_a

        lane_vectors = env.lane_vectors_t2b
        lanes_top_x = env.top_lanes_x
        lane_dx_over_dy = env.lane_dx_over_dy

        lane_x_at_current_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            new_y_a[None, :] - float(env.consts.TOP_CLIP)
        )
        dx_dir = side.astype(pos.dtype)
        dx = jnp.take(cycle_dx, frame) * dx_dir
        new_x_a = pos[0] + dx

        new_vel_y = vel_y + env.consts.CHASING_METEOROID_ACCEL
        new_y_b = pos[1] + new_vel_y
        lane_x_at_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            new_y_b[None, :] - float(env.consts.TOP_CLIP)
        )
        lane_x_b = jnp.take_along_axis(lane_x_at_y, lane[None, :], axis=0).squeeze(0)
        new_x_b = lane_x_b

        phase_descend = phase == 2
        phase_horizontal = jnp.logical_not(phase_descend)
        new_x = jnp.where(phase_descend, new_x_b, new_x_a)
        new_y = jnp.where(phase_descend, new_y_b, new_y_a)

        player_left = player_x
        player_right = player_x + float(env.consts.PLAYER_SPRITE_SIZE[1])
        align_x = _get_ufo_alignment(new_y_a).astype(new_x_a.dtype)
        aligned_x_a = new_x_a + align_x
        chasing_meteoroid_left = aligned_x_a
        chasing_meteoroid_right = aligned_x_a + float(env.consts.METEOROID_SPRITE_SIZE[1])
        hits_x = jnp.logical_and(chasing_meteoroid_right >= player_left, chasing_meteoroid_left <= player_right)
        player_center = player_x + float(env.consts.PLAYER_SPRITE_SIZE[1]) / 2.0
        bottom_lanes = env.bottom_lanes
        nearest_lane_idx = jnp.argmin(jnp.abs(bottom_lanes - player_center)).astype(jnp.int32)

        playable_lanes_x = lane_x_at_current_y[1:6]

        dist_to_lanes = jnp.abs(playable_lanes_x - new_x_a[None, :])
        is_aligned = dist_to_lanes <= float(env.consts.CHASING_METEOROID_LANE_ALIGN_THRESHOLD)

        lane_indices = jnp.arange(5)[:, None]

        side_broad = side[None, :]
        player_idx_broad = nearest_lane_idx

        should_drop_right = (side_broad > 0) & (lane_indices >= player_idx_broad)
        should_drop_left = (side_broad < 0) & (lane_indices <= player_idx_broad)

        is_valid_drop_lane = should_drop_right | should_drop_left

        trigger_mask = is_aligned & is_valid_drop_lane

        should_descend = jnp.any(trigger_mask, axis=0)

        target_lane_idx_0_4 = jnp.argmax(trigger_mask, axis=0).astype(jnp.int32)

        chosen_lane = target_lane_idx_0_4 + 1

        start_descend_now = active & (phase == 0) & should_descend

        new_phase = jnp.where(start_descend_now, 2, phase)
        new_lane = jnp.where(start_descend_now, chosen_lane, lane)

        current_lane_idx = jnp.clip(lane - 1, 0, 4)
        current_lane_x = playable_lanes_x[current_lane_idx, jnp.arange(env.consts.CHASING_METEOROID_MAX)]
        is_on_current_lane = jnp.abs(current_lane_x - new_x_a) <= float(env.consts.CHASING_METEOROID_LANE_ALIGN_THRESHOLD)

        start_descend = start_descend_now | ((phase == 1) & is_on_current_lane)
        new_phase = jnp.where(start_descend, 2, new_phase)

        new_vel_y = jnp.where(start_descend, float(env.consts.CHASING_METEOROID_LANE_SPEED), vel_y)
        new_vel_y = jnp.where(phase_descend, vel_y + env.consts.CHASING_METEOROID_ACCEL, new_vel_y)

        new_frame = jnp.where(phase_horizontal, (frame + 1) % 8, frame)
        new_frame = jnp.where(start_descend, 0, new_frame)

        new_pos = jnp.stack([new_x, new_y])
        offscreen = env.enemy_offscreen
        new_pos = jnp.where(active[None, :], new_pos, offscreen[:, None])

        out_of_bounds = jnp.logical_and(
            active,
            jnp.logical_or(new_x < 0.0, new_x > float(env.consts.SCREEN_WIDTH)),
        )
        new_active = jnp.where(out_of_bounds, False, active)
        new_pos = jnp.where(new_active[None, :], new_pos, offscreen[:, None])
        new_vel_y = jnp.where(new_active, new_vel_y, 0.0)
        new_phase = jnp.where(new_active, new_phase, 0)
        new_frame = jnp.where(new_active, new_frame, 0)
        new_lane = jnp.where(new_active, new_lane, 0)
        new_side = jnp.where(new_active, side, 1)

        return (
            new_pos,
            new_active,
            new_vel_y,
            new_phase,
            new_frame,
            new_lane,
            new_side,
            spawn_timer,
            remaining,
            wave_active,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _mothership_step(self, state, white_ufo_left, enemy_explosion_frame, is_hit):
        env = self._env
        base_step = type(env)._mothership_step

        pos_1, timer_1, stage_1, sector_advance_1 = base_step(env, state, white_ufo_left, enemy_explosion_frame, is_hit)
        level_1 = state.level.replace(
            mothership_position=pos_1,
            mothership_timer=timer_1,
            mothership_stage=stage_1,
        )
        state_1 = state.replace(level=level_1)
        can_advance_twice = (stage_1 != 0) & (stage_1 != 4) & (stage_1 != 5)

        pos_2, timer_2, stage_2, sector_advance_2 = base_step(env, state_1, white_ufo_left, enemy_explosion_frame, is_hit)

        final_pos = jnp.where(can_advance_twice, pos_2, pos_1)
        final_timer = jnp.where(can_advance_twice, timer_2, timer_1)
        final_stage = jnp.where(can_advance_twice, stage_2, stage_1)
        final_sector_advance = sector_advance_1 | (can_advance_twice & sector_advance_2)

        return final_pos, final_timer, final_stage, final_sector_advance


class ThreeLanesMod(JaxAtariInternalModPlugin):
    """Collapse Beamrider's track to the center three lanes."""

    constants_overrides = {
        "LEFT_CLIP_PLAYER": 52,
        "RIGHT_CLIP_PLAYER": 117,
    }

    attribute_overrides = {
        "bottom_lanes": THREE_LANE_BOTTOM_LANES,
        "top_lanes_x": THREE_LANE_TOP_LANES,
        "lane_vectors_t2b": THREE_LANE_TOP_TO_BOTTOM,
        "lane_vectors_b2t": THREE_LANE_BOTTOM_TO_TOP,
        "lane_dx_over_dy": THREE_LANE_TOP_TO_BOTTOM[:, 0] / THREE_LANE_TOP_TO_BOTTOM[:, 1],
        "middle_lane_spawn": THREE_LANE_TOP_IDS,
    }

    @partial(jax.jit, static_argnums=(0,))
    def _render_colored_background(self, raster, state):
        renderer = self._env.renderer
        raster = type(renderer)._render_colored_background(renderer, raster, state)

        blue_id = renderer.COLOR_TO_ID[THREE_LANE_BACKGROUND_BLUE_RGB]
        clear_mask = (renderer.jr._yy > THREE_LANE_BACKGROUND_HORIZON_Y) & (raster == blue_id)
        row_background = raster[:, :1]
        raster = jnp.where(clear_mask, row_background, raster)

        return renderer.jr.draw_rects(
            raster,
            THREE_LANE_GUIDE_POSITIONS,
            THREE_LANE_GUIDE_SIZES,
            blue_id,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_choose_pattern(
        self,
        key,
        *,
        allow_shoot,
        prev_pattern,
        is_kamikaze_zone,
        sector,
        stage,
        lane,
        is_on_lane,
    ):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.MOVE_BACK),
                int(WhiteUFOPattern.KAMIKAZE),
                int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT),
                int(WhiteUFOPattern.TRIPLE_SHOT_LEFT),
            ],
            dtype=jnp.int32,
        )
        pattern_probs = self._env.ufo_pattern_probs

        is_move_back = prev_pattern == int(WhiteUFOPattern.MOVE_BACK)
        chain_mask = jnp.ones_like(pattern_probs).at[0].set(jnp.where(is_move_back, 0.0, 1.0))
        pattern_probs = pattern_probs * chain_mask

        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)

        kamikaze_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(is_kamikaze_zone, pattern_probs, pattern_probs * kamikaze_mask)

        move_back_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(stage >= 4, pattern_probs, pattern_probs * move_back_mask)

        can_triple = (sector >= 7) & (stage >= 4) & (stage <= 6) & is_on_lane
        can_triple_right = can_triple & (lane >= 2) & (lane <= 3)
        can_triple_left = can_triple & (lane >= 3) & (lane <= 4)

        triple_right_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
        triple_left_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        pattern_probs = jnp.where(can_triple_right, pattern_probs, pattern_probs * triple_right_mask)
        pattern_probs = jnp.where(can_triple_left, pattern_probs, pattern_probs * triple_left_mask)

        prob_sum = jnp.sum(pattern_probs)
        pattern_probs = jnp.where(prob_sum > 0, pattern_probs / prob_sum, pattern_probs)

        pattern = jax.random.choice(key, pattern_choices, shape=(), p=pattern_probs)
        duration = self._env.ufo_pattern_durations[pattern]
        return pattern, duration

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_update_pattern_state(
        self,
        sector,
        position,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        retreat_roll,
        start_roll,
        key_chain_choice,
        key_start_choice
    ):
        on_top_lane = position[1] <= self._env.consts.TOP_CLIP
        time_on_lane = jnp.where(on_top_lane, time_on_lane + 1, 0)
        attack_time = jnp.where(on_top_lane, 0, attack_time)

        ufo_x = position[0].astype(jnp.float32)
        ufo_y = position[1].astype(jnp.float32)
        lane_x_at_ufo_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (ufo_y - float(self._env.consts.TOP_CLIP))
        raw_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)
        closest_lane_id = _canonical_three_lane_ufo_lane(raw_lane_id)
        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25

        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))

        shots_left = pattern_timer & 7
        last_lane = (pattern_timer >> 3) & 15
        shoot_now = (pattern_timer >> 7) & 1

        def update_triple(_):
            can_shoot = (shots_left > 0) & is_on_lane & (closest_lane_id != last_lane)
            new_shoot_now = jnp.where(shoot_now == 1, 0, jnp.where(can_shoot, 1, 0))
            new_shots_left = jnp.where(can_shoot, shots_left - 1, shots_left)
            new_last_lane = jnp.where(can_shoot, closest_lane_id, last_lane)
            return (new_shoot_now << 7) | (new_last_lane << 3) | new_shots_left

        pattern_timer = jnp.where(
            is_triple,
            update_triple(None),
            jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer)),
        )

        allow_shoot = (~on_top_lane) & _is_three_lane_shootable(closest_lane_id)

        is_drop_pattern = (
            (pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT))
            | (pattern_id == int(WhiteUFOPattern.DROP_LEFT))
            | (pattern_id == int(WhiteUFOPattern.DROP_RIGHT))
            | (pattern_id == int(WhiteUFOPattern.MOVE_BACK))
        )
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = is_drop_pattern | is_shoot_pattern | is_triple
        attack_time = jnp.where((~on_top_lane) & is_engagement_pattern, attack_time + 1, attack_time)

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        movement_finished = (is_retreat | is_move_back) & on_top_lane
        pattern_id = jnp.where(movement_finished, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(movement_finished, 0, pattern_timer)
        attack_time = jnp.where(movement_finished, 0, attack_time)

        triple_finished = is_triple & ((pattern_timer & 7) == 0) & jnp.logical_not((pattern_timer >> 7) & 1) & is_on_lane

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, 2, 4)

        triple_stuck = is_triple & is_on_lane & (shots_left > 0) & (target_lane_id == closest_lane_id) & (closest_lane_id == last_lane)
        triple_finished = triple_finished | triple_stuck

        pattern_finished_off_top = (~on_top_lane) & is_engagement_pattern & jnp.where(is_triple, triple_finished, pattern_timer == 0) & is_on_lane

        retreat_prob = self._env._white_ufo_retreat_prob(attack_time)
        retreat_now = pattern_finished_off_top & (retreat_roll < retreat_prob)
        pattern_id = jnp.where(retreat_now, int(WhiteUFOPattern.RETREAT), pattern_id)
        pattern_timer = jnp.where(retreat_now, self._env.consts.WHITE_UFO_RETREAT_DURATION, pattern_timer)
        attack_time = jnp.where(retreat_now, 0, attack_time)

        chain_next = pattern_finished_off_top & (~retreat_now)
        ufo_stage = _get_index_ufo(position[1])

        def choose_chain_pattern(_):
            is_kamikaze_zone = position[1] >= self._env.consts.KAMIKAZE_Y_THRESHOLD
            pattern, duration = self._white_ufo_choose_pattern(
                key_chain_choice,
                allow_shoot=allow_shoot,
                prev_pattern=pattern_id,
                is_kamikaze_zone=is_kamikaze_zone,
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        pattern_id, pattern_timer = jax.lax.cond(
            chain_next,
            choose_chain_pattern,
            lambda _: (pattern_id, pattern_timer),
            operand=None,
        )

        should_choose_new = on_top_lane & (pattern_id == int(WhiteUFOPattern.IDLE)) & (pattern_timer == 0) & (spawn_delay == 0)
        p_start = type(self._env).entropy_heat_prob_static(
            jnp.where(already_left, time_on_lane * 10, time_on_lane),
            alpha=self._env.consts.WHITE_UFO_ATTACK_ALPHA,
            p_min=jnp.where(already_left, 0.1, self._env.consts.WHITE_UFO_ATTACK_P_MIN),
            p_max=self._env.consts.WHITE_UFO_ATTACK_P_MAX,
        )
        start_attack = should_choose_new & (start_roll < p_start)

        def choose_new_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(
                key_start_choice,
                allow_shoot=jnp.array(False),
                prev_pattern=pattern_id,
                is_kamikaze_zone=jnp.array(False),
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        pattern_id, pattern_timer = jax.lax.cond(
            start_attack,
            choose_new_pattern,
            lambda _: (pattern_id, pattern_timer),
            operand=None,
        )

        return pattern_id, pattern_timer, time_on_lane, attack_time

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_top_lane(self, white_ufo_pos, white_ufo_vel_x, pattern_id, motion_roll):
        hold_position = (pattern_id == int(WhiteUFOPattern.SHOOT)) | (white_ufo_pos[1] > float(self._env.consts.TOP_CLIP))
        min_speed = float(self._env.consts.WHITE_UFO_TOP_LANE_MIN_SPEED)
        turn_speed = float(self._env.consts.WHITE_UFO_TOP_LANE_TURN_SPEED)

        vx = jnp.where(hold_position, 0.0, white_ufo_vel_x)
        need_boost = (~hold_position) & (jnp.abs(vx) < min_speed)
        random_sign = jnp.where(motion_roll < 0.5, -1.0, 1.0)
        direction = jnp.where(vx == 0.0, random_sign, jnp.sign(vx))
        vx = jnp.where(need_boost, direction * min_speed, vx)

        do_bounce = ~hold_position
        vx = jnp.where(do_bounce & (white_ufo_pos[0] >= THREE_LANE_RIGHT_BOUND), -turn_speed, vx)
        vx = jnp.where(do_bounce & (white_ufo_pos[0] <= THREE_LANE_LEFT_BOUND), turn_speed, vx)
        return vx, 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_normal(self, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y, pattern_id, already_left):
        speed_factor = self._env.consts.WHITE_UFO_SPEED_FACTOR
        retreat_mult = self._env.consts.WHITE_UFO_RETREAT_SPEED_MULT
        x, y = white_ufo_pos[0], white_ufo_pos[1]

        lane_x_at_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (y - float(self._env.consts.TOP_CLIP))
        raw_lane_id = jnp.argmin(jnp.abs(lane_x_at_y - x))
        closest_lane_id = _canonical_three_lane_ufo_lane(raw_lane_id)

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_LEFT), -1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, 2, 4)

        lane_vector = self._env.lane_vectors_t2b[target_lane_id]
        target_lane_x = lane_x_at_y[target_lane_id]

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        is_kamikaze = pattern_id == int(WhiteUFOPattern.KAMIKAZE)
        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))

        cross_track = target_lane_x - x
        distance_to_lane = jnp.abs(cross_track)
        direction = jnp.sign(cross_track)

        def seek_lane(_):
            attack_vx = jnp.where(direction == 0, 0.0, direction * 0.5)
            retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)
            new_vx = jnp.where(is_retreat | is_kamikaze | is_triple, retreat_vx, attack_vx)

            normal_vy = 0.25
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
            move_back_vy = -lane_vector[1] * speed_factor
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult
            triple_vy = 0.25

            new_vy = jnp.where(is_retreat, retreat_vy, normal_vy)
            new_vy = jnp.where(is_move_back, move_back_vy, new_vy)
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)
            return new_vx, new_vy

        def follow_lane(_):
            normal_vx = lane_vector[0] * speed_factor
            normal_vy = lane_vector[1] * speed_factor

            retreat_vx = -lane_vector[0] * speed_factor * retreat_mult
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult

            move_back_vx = -lane_vector[0] * speed_factor
            move_back_vy = -lane_vector[1] * speed_factor

            kamikaze_vx = lane_vector[0] * speed_factor * retreat_mult
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult

            triple_vy = 0.25

            new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_move_back, move_back_vx, normal_vx))
            new_vx = jnp.where(is_kamikaze, kamikaze_vx, new_vx)

            new_vy = jnp.where(is_retreat, retreat_vy, jnp.where(is_move_back, move_back_vy, normal_vy))
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)
            return new_vx, new_vy

        return jax.lax.cond(distance_to_lane <= 0.25, follow_lane, seek_lane, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_shot_step(self, state, white_ufo_pos, white_ufo_pattern_id, white_ufo_pattern_timer):
        lane_vectors = self._env.lane_vectors_t2b
        lanes_top_x = self._env.top_lanes_x
        lane_dx_over_dy = self._env.lane_dx_over_dy

        offscreen = self._env.bullet_offscreen_shots

        shot_pos = state.level.enemy_shot_pos.astype(jnp.float32)
        shot_lane = state.level.enemy_shot_vel.astype(jnp.int32)
        shot_timer = state.level.enemy_shot_timer.astype(jnp.int32)

        shot_active = shot_pos[1] <= float(self._env.consts.BOTTOM_CLIP)
        shot_timer = jnp.where(shot_active, shot_timer + 1, 0)

        shoot_duration = self._env.ufo_pattern_durations[int(WhiteUFOPattern.SHOOT)]
        is_triple = (white_ufo_pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (white_ufo_pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        shoot_now_triple = (white_ufo_pattern_timer >> 7) & 1

        wants_spawn = (white_ufo_pattern_id == int(WhiteUFOPattern.SHOOT)) & (white_ufo_pattern_timer == shoot_duration)
        wants_spawn = wants_spawn | (is_triple & (shoot_now_triple == 1))

        ufo_on_screen = white_ufo_pos[1] <= float(self._env.consts.BOTTOM_CLIP)
        ufo_not_on_top_lane = white_ufo_pos[1] > float(self._env.consts.TOP_CLIP)
        ufo_x = white_ufo_pos[0].astype(jnp.float32)
        ufo_y = white_ufo_pos[1].astype(jnp.float32)

        lane_x_at_ufo_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (ufo_y[None, :] - float(self._env.consts.TOP_CLIP))
        raw_closest_lane = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x[None, :]), axis=0).astype(jnp.int32)
        closest_lane = _canonical_three_lane_ufo_lane(raw_closest_lane)
        allowed_shot_lane = _is_three_lane_shootable(closest_lane)

        ufo_shot_active = jnp.reshape(shot_active, (3, 3))
        first_inactive_slot = jnp.argmax(jnp.logical_not(ufo_shot_active), axis=1)
        has_inactive_slot = jnp.any(jnp.logical_not(ufo_shot_active), axis=1)

        can_shoot = (state.steps > 2000) | state.ufo_killed
        spawn = jnp.logical_and.reduce(
            jnp.array(
                [
                    wants_spawn,
                    ufo_on_screen,
                    ufo_not_on_top_lane,
                    allowed_shot_lane,
                    has_inactive_slot,
                ]
            )
        ) & can_shoot

        spawn_y = jnp.clip(ufo_y + 4.0, float(self._env.consts.TOP_CLIP), float(self._env.consts.BOTTOM_CLIP))
        spawn_x = jnp.take(lanes_top_x, closest_lane) + jnp.take(lane_dx_over_dy, closest_lane) * (
            spawn_y - float(self._env.consts.TOP_CLIP)
        )

        spawn_slots = jnp.arange(3) * 3 + first_inactive_slot
        spawn_mask = (jax.nn.one_hot(spawn_slots, 9) * spawn[:, None]).sum(axis=0).astype(jnp.bool_)

        spawn_x_expanded = jnp.repeat(spawn_x, 3)
        spawn_y_expanded = jnp.repeat(spawn_y, 3)
        spawn_pos_expanded = jnp.stack([spawn_x_expanded, spawn_y_expanded])

        shot_pos = jnp.where(spawn_mask[None, :], spawn_pos_expanded, shot_pos)

        closest_lane_expanded = jnp.repeat(closest_lane, 3)
        shot_lane = jnp.where(spawn_mask, closest_lane_expanded, shot_lane)
        shot_timer = jnp.where(spawn_mask, 0, shot_timer)
        shot_active = jnp.logical_or(shot_active, spawn_mask)

        should_move = shot_active & ((shot_timer % 4) == 2)
        speed = float(self._env.consts.WHITE_UFO_SHOT_SPEED_FACTOR)
        lane_dy = jnp.take(lane_vectors[:, 1], shot_lane)
        y_after = shot_pos[1] + jnp.where(should_move, lane_dy * speed, 0.0)
        x_after = jnp.take(lanes_top_x, shot_lane) + jnp.take(lane_dx_over_dy, shot_lane) * (
            y_after - float(self._env.consts.TOP_CLIP)
        )
        shot_pos = jnp.where(shot_active, jnp.stack([x_after, y_after]), shot_pos)

        moved_offscreen = shot_pos[1] > float(self._env.consts.BOTTOM_CLIP)
        shot_pos = jnp.where(moved_offscreen, offscreen, shot_pos)
        shot_timer = jnp.where(moved_offscreen, 0, shot_timer)
        shot_active = shot_active & (~moved_offscreen)

        player_left = state.level.player_pos.astype(jnp.float32)
        player_y = float(self._env.consts.PLAYER_POS_Y)
        player_size = self._env.player_sprite_size

        shot_x = shot_pos[0] + _get_ufo_alignment(shot_pos[1])
        shot_y = shot_pos[1]

        sprite_idx = (jnp.floor_divide(shot_timer, 4) % 2).astype(jnp.int32)
        shot_sizes = jnp.take(self._env.enemy_shot_sprite_sizes, sprite_idx, axis=0)

        hits = (
            shot_active
            & (shot_x < player_left + player_size[1])
            & (player_left < shot_x + shot_sizes[:, 1])
            & (shot_y < player_y + player_size[0])
            & (player_y < shot_y + shot_sizes[:, 0])
        )

        hit_count = jnp.sum(hits, dtype=jnp.int32)
        shot_pos = jnp.where(hits[None, :], offscreen, shot_pos)
        shot_timer = jnp.where(hits, 0, shot_timer)
        return shot_pos, shot_lane, shot_timer, hit_count


class SameEnemiesMod(JaxAtariInternalModPlugin):
    """Render all Beamrider enemy types using the white UFO visuals."""

    @partial(jax.jit, static_argnums=(0,))
    def _render_bouncer(self, raster, state):
        renderer = self._env.renderer
        explosion_frame = state.level.bouncer_explosion_frame

        def render_explosion(r_in):
            return _render_enemy_explosion(
                renderer,
                r_in,
                explosion_frame,
                state.level.bouncer_explosion_pos,
            )

        def render_active_bouncer(r_in):
            return _render_ufo_like_enemy(
                renderer,
                r_in,
                state.level.bouncer_pos[0],
                state.level.bouncer_pos[1],
                state.level.bouncer_active,
                align=True,
            )

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_active_bouncer,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_chasing_meteoroids(self, raster, state):
        renderer = self._env.renderer

        def body_fun(raster, idx):
            explosion_frame = state.level.chasing_meteoroid_explosion_frame[idx]

            def render_explosion(r_in):
                return _render_enemy_explosion(
                    renderer,
                    r_in,
                    explosion_frame,
                    state.level.chasing_meteoroid_explosion_pos[:, idx],
                )

            def render_enemy(r_in):
                return _render_ufo_like_enemy(
                    renderer,
                    r_in,
                    state.level.chasing_meteoroid_pos[0][idx],
                    state.level.chasing_meteoroid_pos[1][idx],
                    state.level.chasing_meteoroid_active[idx],
                    align=True,
                )

            new_raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_enemy,
                raster,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.CHASING_METEOROID_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_falling_rocks(self, raster, state):
        renderer = self._env.renderer

        def body_fun(raster, idx):
            explosion_frame = state.level.falling_rock_explosion_frame[idx]

            def render_explosion(r_in):
                return _render_enemy_explosion(
                    renderer,
                    r_in,
                    explosion_frame,
                    state.level.falling_rock_explosion_pos[:, idx],
                )

            def render_enemy(r_in):
                return _render_ufo_like_enemy(
                    renderer,
                    r_in,
                    state.level.falling_rock_pos[0][idx],
                    state.level.falling_rock_pos[1][idx],
                    state.level.falling_rock_active[idx],
                    align=True,
                )

            new_raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_enemy,
                raster,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.FALLING_ROCK_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lane_blockers(self, raster, state):
        renderer = self._env.renderer

        def body_fun(raster, idx):
            explosion_frame = state.level.lane_blocker_explosion_frame[idx]

            def render_explosion(r_in):
                return _render_enemy_explosion(
                    renderer,
                    r_in,
                    explosion_frame,
                    state.level.lane_blocker_explosion_pos[:, idx],
                )

            def render_enemy(r_in):
                y_pos = state.level.lane_blocker_pos[1][idx]
                mask = _get_white_ufo_mask(renderer, y_pos)

                clip_rows = jnp.maximum(
                    0,
                    jnp.round(y_pos - self._env.consts.LANE_BLOCKER_BOTTOM_Y).astype(jnp.int32),
                )
                is_sinking = state.level.lane_blocker_phase[idx] == int(LaneBlockerState.SINK)

                def clip_mask(base_mask):
                    height = base_mask.shape[0]
                    visible_rows = jnp.maximum(0, height - clip_rows)
                    row_idx = jnp.arange(height)
                    row_mask = row_idx < visible_rows
                    transparent = jnp.array(renderer.jr.TRANSPARENT_ID, dtype=base_mask.dtype)
                    return jnp.where(row_mask[:, None], base_mask, transparent)

                mask = jax.lax.cond(is_sinking, clip_mask, lambda m: m, mask)
                return _render_ufo_like_enemy(
                    renderer,
                    r_in,
                    state.level.lane_blocker_pos[0][idx],
                    y_pos,
                    state.level.lane_blocker_active[idx],
                    align=True,
                    mask=mask,
                )

            new_raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_enemy,
                raster,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.LANE_BLOCKER_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_rejuvenator(self, raster, state):
        renderer = self._env.renderer
        explosion_frame = state.level.rejuvenator_explosion_frame

        def render_explosion(r_in):
            return _render_enemy_explosion(
                renderer,
                r_in,
                explosion_frame,
                state.level.rejuvenator_explosion_pos,
            )

        def render_enemy(r_in):
            return _render_ufo_like_enemy(
                renderer,
                r_in,
                state.level.rejuvenator_pos[0],
                state.level.rejuvenator_pos[1],
                state.level.rejuvenator_active,
                align=False,
            )

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_enemy,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_kamikaze(self, raster, state):
        renderer = self._env.renderer
        explosion_frame = state.level.kamikaze_explosion_frame[0]

        def render_explosion(r_in):
            return _render_enemy_explosion(
                renderer,
                r_in,
                explosion_frame,
                state.level.kamikaze_explosion_pos[:, 0],
            )

        def render_enemy(r_in):
            return _render_ufo_like_enemy(
                renderer,
                r_in,
                state.level.kamikaze_pos[0][0],
                state.level.kamikaze_pos[1][0],
                state.level.kamikaze_active[0],
                align=True,
            )

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_enemy,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_mothership(self, raster, state):
        renderer = self._env.renderer
        stage = state.level.mothership_stage.astype(jnp.int32)
        timer = state.level.mothership_timer
        pos_x = state.level.mothership_position
        pos_y = jnp.array(
            self._env.consts.MOTHERSHIP_EMERGE_Y - self._env.consts.MOTHERSHIP_HEIGHT,
            dtype=jnp.float32,
        )

        def render_none(r):
            return r

        def render_ship(r):
            mask = _get_white_ufo_mask(renderer, pos_y)
            active = (stage > 0) & (stage < 4)
            return _render_ufo_like_enemy(
                renderer,
                r,
                pos_x,
                pos_y,
                active,
                align=False,
                mask=mask,
            )

        def render_exploding(r):
            explosion_masks = renderer.SHAPE_MASKS["mothership_explosion"]
            step_duration = self._env.consts.MOTHERSHIP_EXPLOSION_STEP_DURATION
            step_idx = jnp.clip(timer // step_duration, 0, 8)
            sprite_idx = renderer._mothership_explosion_seq[step_idx]
            exp_mask = explosion_masks[sprite_idx]
            return renderer.jr.render_at_clipped(r, pos_x, pos_y, exp_mask)

        return jax.lax.switch(
            stage,
            [render_none, render_ship, render_ship, render_ship, render_none, render_exploding],
            raster,
        )


class ToasterMod(JaxAtariInternalModPlugin):
    """Disable Beamrider's heavier visual effects while keeping gameplay unchanged."""

    conflicts_with = ["same_enemies"]

    @partial(jax.jit, static_argnums=(0,))
    def _render_colored_background(self, raster, state):
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_blue_lines(self, raster, state):
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lives(self, raster, state):
        renderer = self._env.renderer
        hp_mask = renderer.SHAPE_MASKS["live"]
        max_visible_lives = max(self._env.consts.MAX_LIVES - 1, 0)

        def body_fun(r_in, idx):
            is_visible = (state.lives - 1) > idx
            pos_x = jnp.where(is_visible, 32 + (idx * 9), -100)
            new_raster = renderer.jr.render_at_clipped(r_in, pos_x, 183, hp_mask)
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(max_visible_lives))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_player_and_bullet(self, raster, state):
        renderer = self._env.renderer
        player_masks = renderer.SHAPE_MASKS["player_sprite"]
        dead_player_mask = renderer.SHAPE_MASKS["dead_player"]

        raster = jax.lax.cond(
            state.level.death_timer > 0,
            lambda r: renderer.jr.render_at(r, state.level.player_pos, self._env.consts.PLAYER_POS_Y, dead_player_mask),
            lambda r: renderer.jr.render_at(r, state.level.player_pos, self._env.consts.PLAYER_POS_Y, player_masks[9]),
            raster,
        )

        bullet_mask = renderer.SHAPE_MASKS["bullet_sprite"][
            _get_index_bullet(state.level.player_shot_pos[1], state.level.bullet_type, self._env.consts.LASER_ID)
        ]
        shot_x_screen = _get_player_shot_screen_x(
            state.level.player_shot_pos,
            state.level.player_shot_vel,
            state.level.bullet_type,
            self._env.consts.LASER_ID,
        )
        return renderer.jr.render_at_clipped(
            raster,
            shot_x_screen,
            state.level.player_shot_pos[1],
            bullet_mask,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemy_shots(self, raster, state):
        renderer = self._env.renderer
        shot_mask = renderer.SHAPE_MASKS["enemy_shot"][0]

        def body_fun(r_in, idx):
            visible = (state.level.enemy_shot_explosion_frame[idx] == 0) & (
                state.level.enemy_shot_pos[1][idx] <= self._env.consts.BOTTOM_CLIP
            )
            y_pos = jnp.where(visible, state.level.enemy_shot_pos[1][idx], 500.0)
            x_pos = state.level.enemy_shot_pos[0][idx] + _get_ufo_alignment(y_pos)
            new_raster = renderer.jr.render_at_clipped(r_in, x_pos, y_pos, shot_mask)
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(9))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_white_ufos(self, raster, state):
        renderer = self._env.renderer
        hide_all = state.level.blue_line_counter < len(BLUE_LINE_INIT_TABLE)

        def body_fun(r_in, idx):
            y_pos = state.level.white_ufo_pos[1][idx]
            mask = _get_white_ufo_mask(renderer, y_pos)
            visible = (state.level.ufo_explosion_frame[idx] == 0) & jnp.logical_not(hide_all)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.white_ufo_pos[0][idx],
                y_pos,
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(3))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_bouncer(self, raster, state):
        renderer = self._env.renderer
        visible = state.level.bouncer_active & (state.level.bouncer_explosion_frame == 0)
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.bouncer_pos[0],
            state.level.bouncer_pos[1],
            visible,
            renderer.SHAPE_MASKS["bouncer"],
            align=True,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_chasing_meteoroids(self, raster, state):
        renderer = self._env.renderer
        mask = renderer.SHAPE_MASKS["chasing_meteoroid"]

        def body_fun(r_in, idx):
            visible = state.level.chasing_meteoroid_active[idx] & (
                state.level.chasing_meteoroid_explosion_frame[idx] == 0
            )
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.chasing_meteoroid_pos[0][idx],
                state.level.chasing_meteoroid_pos[1][idx],
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.CHASING_METEOROID_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_falling_rocks(self, raster, state):
        renderer = self._env.renderer
        rock_masks = renderer.SHAPE_MASKS["falling_rocks"]

        def body_fun(r_in, idx):
            y_pos = state.level.falling_rock_pos[1][idx]
            sprite_idx = _get_index_falling_rock(y_pos) - 1
            mask = rock_masks[sprite_idx]
            visible = state.level.falling_rock_active[idx] & (state.level.falling_rock_explosion_frame[idx] == 0)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.falling_rock_pos[0][idx],
                y_pos,
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.FALLING_ROCK_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lane_blockers(self, raster, state):
        renderer = self._env.renderer
        blocker_masks = renderer.SHAPE_MASKS["lane_blocker"]

        def body_fun(r_in, idx):
            y_pos = state.level.lane_blocker_pos[1][idx]
            sprite_idx = _get_index_lane_blocker(y_pos) - 1
            sprite_idx = jnp.clip(sprite_idx, 0, blocker_masks.shape[0] - 1)
            mask = blocker_masks[sprite_idx]

            clip_rows = jnp.maximum(
                0,
                jnp.round(y_pos - self._env.consts.LANE_BLOCKER_BOTTOM_Y).astype(jnp.int32),
            )
            is_sinking = state.level.lane_blocker_phase[idx] == int(LaneBlockerState.SINK)

            def clip_mask(base_mask):
                height = base_mask.shape[0]
                visible_rows = jnp.maximum(0, height - clip_rows)
                row_idx = jnp.arange(height)
                row_mask = row_idx < visible_rows
                transparent = jnp.array(renderer.jr.TRANSPARENT_ID, dtype=base_mask.dtype)
                return jnp.where(row_mask[:, None], base_mask, transparent)

            mask = jax.lax.cond(is_sinking, clip_mask, lambda m: m, mask)
            visible = state.level.lane_blocker_active[idx] & (state.level.lane_blocker_explosion_frame[idx] == 0)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.lane_blocker_pos[0][idx],
                y_pos,
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.LANE_BLOCKER_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_rejuvenator(self, raster, state):
        renderer = self._env.renderer
        rejuv_masks = renderer.SHAPE_MASKS["rejuvenator"]
        stage = _get_index_rejuvenator(state.level.rejuvenator_pos[1])
        sprite_idx = jnp.where(state.level.rejuvenator_dead, 4, jnp.clip(stage - 1, 0, 3))
        mask = rejuv_masks[sprite_idx]
        visible = state.level.rejuvenator_active & (state.level.rejuvenator_explosion_frame == 0)
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.rejuvenator_pos[0],
            state.level.rejuvenator_pos[1],
            visible,
            mask,
            align=False,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_kamikaze(self, raster, state):
        renderer = self._env.renderer
        y_pos = state.level.kamikaze_pos[1][0]
        sprite_idx = _get_index_kamikaze(y_pos) - 1
        sprite_idx = jnp.clip(sprite_idx, 0, 3)
        mask = renderer.SHAPE_MASKS["kamikaze"][sprite_idx]
        visible = state.level.kamikaze_active[0] & (state.level.kamikaze_explosion_frame[0] == 0)
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.kamikaze_pos[0][0],
            y_pos,
            visible,
            mask,
            align=True,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_coins(self, raster, state):
        renderer = self._env.renderer
        coin_masks = renderer.SHAPE_MASKS["coin"]
        static_mask = coin_masks[renderer._coin_anim_seq[0]]

        def body_fun(r_in, idx):
            visible = state.level.coin_active[idx] & (state.level.coin_explosion_frame[idx] == 0)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.coin_pos[0][idx],
                state.level.coin_pos[1][idx],
                visible,
                static_mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.COIN_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_mothership(self, raster, state):
        renderer = self._env.renderer
        visible = state.level.mothership_stage.astype(jnp.int32) == 2
        y_pos = jnp.array(
            self._env.consts.MOTHERSHIP_EMERGE_Y - self._env.consts.MOTHERSHIP_HEIGHT,
            dtype=jnp.float32,
        )
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.mothership_position,
            y_pos,
            visible,
            renderer.SHAPE_MASKS["mothership"],
            align=False,
        )


class MothershipLaserMod(JaxAtariInternalModPlugin):
    """Turn the mothership sequence into a stationary lane laser attack."""

    conflicts_with = ["toaster"]

    @partial(jax.jit, static_argnums=(0,))
    def _mothership_step(self, state, white_ufo_left, enemy_explosion_frame, is_hit):
        stage = state.level.mothership_stage.astype(jnp.int32)
        timer = state.level.mothership_timer.astype(jnp.int32)
        pos_x = state.level.mothership_position
        sector = state.sector

        is_ltr = (sector % 2) != 0

        def idle_logic():
            explosions_finished = jnp.all(enemy_explosion_frame == 0)
            start = (white_ufo_left == 0) & explosions_finished
            return jnp.where(start, 1, 0), jnp.where(start, 1, 0), pos_x.astype(jnp.float32)

        def emergence_logic():
            next_timer = timer + 1
            finished = next_timer > 15
            s = jnp.clip((timer - 1) // 2, 0, 6)
            rel_x = jnp.take(self._env.mothership_anim_x, s)
            travel_x = jnp.where(
                is_ltr,
                rel_x.astype(jnp.float32),
                (160 - 16 - rel_x + 8).astype(jnp.float32),
            )
            random_lane = jax.random.randint(state.rng, (), 1, 6, dtype=jnp.int32)
            packed_timer = _pack_mothership_laser_timer(
                random_lane,
                jnp.array(1, dtype=jnp.int32),
                jnp.array(False),
                jnp.array(False),
            )
            next_pos = travel_x
            next_stage = jnp.where(finished, 2, 1)
            next_timer = jnp.where(finished, packed_timer, next_timer)
            return next_stage, next_timer, next_pos

        def moving_or_firing_logic():
            target_lane = _get_mothership_laser_lane_from_timer(timer)
            stop_x = _get_mothership_stop_x(self._env, target_lane)
            exit_x_val = self._env.mothership_anim_x[6]
            exit_x = jnp.where(
                is_ltr,
                160.0 - 16.0 - exit_x_val + 8.0,
                exit_x_val,
            )
            phase_timer = _get_mothership_laser_phase_timer(timer)
            firing = _is_mothership_laser_firing(timer)
            post_fire = _has_mothership_laser_post_fire(timer)

            def moving_logic():
                move_target_x = jnp.where(post_fire, exit_x, stop_x)
                should_move = (phase_timer % 6 == 2) | (phase_timer % 6 == 0)
                dx = jnp.where(is_ltr, 1.0, -1.0)
                moved_x = pos_x + jnp.where(should_move, dx, 0.0)
                reached = jnp.where(is_ltr, moved_x >= move_target_x, moved_x <= move_target_x)
                next_pos = jnp.where(reached, move_target_x, moved_x)
                continue_timer = _pack_mothership_laser_timer(
                    target_lane,
                    phase_timer + 1,
                    jnp.array(False),
                    post_fire,
                )
                firing_timer = _pack_mothership_laser_timer(
                    target_lane,
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(True),
                    jnp.array(False),
                )
                next_stage = jnp.where(post_fire & reached, 3, 2)
                next_timer = jnp.where(
                    post_fire & reached,
                    jnp.array(1, dtype=jnp.int32),
                    jnp.where(reached, firing_timer, continue_timer),
                )
                final_stage = jnp.where(is_hit, 5, next_stage)
                final_timer = jnp.where(is_hit, 0, next_timer)
                return final_stage, final_timer, next_pos

            def firing_logic():
                next_phase_timer = phase_timer + 1
                finished = next_phase_timer >= MOTHERSHIP_LASER_DURATION
                continue_timer = _pack_mothership_laser_timer(
                    target_lane,
                    jnp.array(1, dtype=jnp.int32),
                    jnp.array(False),
                    jnp.array(True),
                )
                next_stage = jnp.where(finished, 2, 2)
                next_timer = jnp.where(
                    finished,
                    continue_timer,
                    _pack_mothership_laser_timer(
                        target_lane,
                        next_phase_timer,
                        jnp.array(True),
                        jnp.array(False),
                    ),
                )
                final_stage = jnp.where(is_hit, 5, next_stage)
                final_timer = jnp.where(is_hit, 0, next_timer)
                return final_stage, final_timer, stop_x

            return jax.lax.cond(firing, firing_logic, moving_logic)

        def descending_logic():
            next_timer = timer + 1
            finished = next_timer > 15
            s = jnp.clip(6 - (timer - 1) // 2, 0, 6)
            rel_x = jnp.take(self._env.mothership_anim_x, s)
            calculated_pos = jnp.where(
                is_ltr,
                (160 - 16 - rel_x + 8).astype(jnp.float32),
                rel_x.astype(jnp.float32),
            )
            return jnp.where(finished, 4, 3), jnp.where(finished, 0, next_timer), calculated_pos

        def done_logic():
            return 0, 0, self._env.mothership_offscreen

        def exploding_logic():
            duration = 9 * self._env.consts.MOTHERSHIP_EXPLOSION_STEP_DURATION
            finished = timer >= duration
            return jnp.where(finished, 4, 5), timer + 1, pos_x

        new_stage, new_timer, new_pos = jax.lax.switch(
            stage,
            [idle_logic, emergence_logic, moving_or_firing_logic, descending_logic, done_logic, exploding_logic],
        )
        sector_advance = new_stage == 4
        return new_pos, new_timer, new_stage, sector_advance

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_shot_step(self, state, white_ufo_pos, white_ufo_pattern_id, white_ufo_pattern_timer):
        shot_pos, shot_lane, shot_timer, hit_count = type(self._env)._enemy_shot_step(
            self._env,
            state,
            white_ufo_pos,
            white_ufo_pattern_id,
            white_ufo_pattern_timer,
        )

        player_left = state.level.player_pos.astype(jnp.float32)
        player_top = float(self._env.consts.PLAYER_POS_Y)
        player_size = self._env.player_sprite_size
        laser_hit = _laser_hits_box(
            self._env,
            state,
            player_left,
            player_top,
            player_size[1],
            player_size[0],
        ).astype(jnp.int32)
        return shot_pos, shot_lane, shot_timer, hit_count + laser_hit

    @partial(jax.jit, static_argnums=(0,))
    def _collisions_step(
        self,
        state,
        player_x,
        vel_x,
        player_shot_pos,
        player_shot_vel,
        player_shot_frame,
        torpedos_left,
        bullet_type,
        shooting_cooldown,
        shooting_delay,
        shot_type_pending,
        enemy_updates,
        key,
    ):
        collision_results = type(self._env)._collisions_step(
            self._env,
            state,
            player_x,
            vel_x,
            player_shot_pos,
            player_shot_vel,
            player_shot_frame,
            torpedos_left,
            bullet_type,
            shooting_cooldown,
            shooting_delay,
            shot_type_pending,
            enemy_updates,
            key,
        )

        (
            player_x,
            vel_x,
            player_shot_pos,
            player_shot_vel,
            player_shot_frame,
            torpedos_left,
            bullet_type,
            shooting_cooldown,
            shooting_delay,
            shot_type_pending,
        ) = collision_results["player"]

        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self._env.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, self._env.consts.LASER_ID)
        bullet_size = jnp.take(self._env.bullet_sprite_sizes, bullet_idx, axis=0)
        shot_active = shot_y < float(self._env.consts.BOTTOM_CLIP)
        laser_hit = _laser_hits_box(
            self._env,
            state,
            shot_x,
            shot_y,
            bullet_size[1],
            bullet_size[0],
        )
        laser_hit = shot_active & laser_hit

        player_shot_pos = jnp.where(laser_hit, self._env.bullet_offscreen, player_shot_pos)
        player_shot_frame = jnp.where(laser_hit, jnp.array(-1, dtype=player_shot_frame.dtype), player_shot_frame)
        shooting_cooldown = jnp.where(laser_hit, self._env.consts.PLAYER_SHOT_RECOVERY, shooting_cooldown)

        return collision_results | {
            "player": (
                player_x,
                vel_x,
                player_shot_pos,
                player_shot_vel,
                player_shot_frame,
                torpedos_left,
                bullet_type,
                shooting_cooldown,
                shooting_delay,
                shot_type_pending,
            )
        }

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemy_shots(self, raster, state):
        renderer = self._env.renderer
        raster = type(renderer)._render_enemy_shots(renderer, raster, state)
        shot_mask = renderer.SHAPE_MASKS["enemy_shot"][MOTHERSHIP_LASER_SPRITE_IDX]
        seg_x, seg_y, visible = _get_mothership_laser_segments(self._env, state)

        def body_fun(r_in, idx):
            draw_y = jnp.where(visible[idx], seg_y[idx], 500.0)
            draw_x = jnp.where(visible[idx], seg_x[idx], 500.0)
            new_raster = renderer.jr.render_at_clipped(r_in, draw_x, draw_y, shot_mask)
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(MOTHERSHIP_LASER_SEGMENT_COUNT))
        return raster


class TeleportUFOsMod(JaxAtariInternalModPlugin):
    """Adds a rare White-UFO pattern that snaps once to a nearby lane."""

    def _advance_white_ufos(self, state):
        # Beamrider caches a vmapped White-UFO step function at env init time.
        # Rebuild it here so this mod uses the patched _white_ufo_step logic.
        results = jax.vmap(
            self._white_ufo_step,
            in_axes=(None, 1, 1, 0, 0, 0, 0, 0, 0, 0),
        )(
            state.sector,
            state.level.white_ufo_pos,
            state.level.white_ufo_vel,
            state.level.white_ufo_time_on_lane,
            state.level.white_ufo_attack_time,
            state.level.white_ufo_already_left,
            state.level.white_ufo_spawn_delay,
            state.level.white_ufo_pattern_id,
            state.level.white_ufo_pattern_timer,
            state.level.white_ufo_rngs,
        )

        positions, vel_x, vel_y, time_on_lane, attack_time, already_left, spawn_delay, pattern_id, pattern_timer, new_keys = results
        return WhiteUFOUpdate(
            pos=positions.T,
            vel=jnp.stack([vel_x, vel_y]),
            time_on_lane=time_on_lane,
            attack_time=attack_time,
            already_left=already_left,
            spawn_delay=spawn_delay,
            pattern_id=pattern_id.astype(jnp.int32),
            pattern_timer=pattern_timer.astype(jnp.int32),
            rngs=new_keys,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_pattern_requires_lane_motion(self, pattern_id):
        return jnp.isin(pattern_id, self._env._lane_motion_patterns) | _is_teleport_ufo_pattern(pattern_id)

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_choose_pattern(
        self,
        key,
        *,
        allow_shoot,
        prev_pattern,
        is_kamikaze_zone,
        sector,
        stage,
        lane,
        is_on_lane,
    ):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.MOVE_BACK),
                int(WhiteUFOPattern.KAMIKAZE),
                int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT),
                int(WhiteUFOPattern.TRIPLE_SHOT_LEFT),
                TELEPORT_UFO_PATTERN_ID,
            ],
            dtype=jnp.int32,
        )
        pattern_probs = jnp.concatenate(
            [
                self._env.ufo_pattern_probs,
                jnp.array([TELEPORT_UFO_PATTERN_WEIGHT], dtype=jnp.float32),
            ]
        )

        is_move_back = prev_pattern == int(WhiteUFOPattern.MOVE_BACK)
        chain_mask = jnp.ones_like(pattern_probs).at[0].set(jnp.where(is_move_back, 0.0, 1.0))
        pattern_probs = pattern_probs * chain_mask

        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)

        kamikaze_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(is_kamikaze_zone, pattern_probs, pattern_probs * kamikaze_mask)

        move_back_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(stage >= 4, pattern_probs, pattern_probs * move_back_mask)

        can_triple = (sector >= 7) & (stage >= 4) & (stage <= 6) & is_on_lane
        can_triple_right = can_triple & (lane >= 1) & (lane <= 3)
        can_triple_left = can_triple & (lane >= 3) & (lane <= 5)

        triple_right_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        triple_left_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(can_triple_right, pattern_probs, pattern_probs * triple_right_mask)
        pattern_probs = jnp.where(can_triple_left, pattern_probs, pattern_probs * triple_left_mask)

        min_lane, max_lane = _get_teleport_ufo_lane_bounds_from_stage(stage)
        _, teleport_valid_mask = _get_teleport_ufo_candidate_lanes(lane, min_lane, max_lane)
        can_teleport = is_on_lane & jnp.any(teleport_valid_mask)
        teleport_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        pattern_probs = jnp.where(can_teleport, pattern_probs, pattern_probs * teleport_mask)

        prob_sum = jnp.sum(pattern_probs)
        pattern_probs = jnp.where(prob_sum > 0, pattern_probs / prob_sum, pattern_probs)

        duration_table = jnp.concatenate(
            [
                self._env.ufo_pattern_durations,
                jnp.array([TELEPORT_UFO_PATTERN_DURATION], dtype=jnp.int32),
            ]
        )
        pattern = jax.random.choice(key, pattern_choices, shape=(), p=pattern_probs)
        duration = duration_table[pattern]
        return pattern, duration

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_update_pattern_state(
        self,
        sector,
        position,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        retreat_roll,
        start_roll,
        key_chain_choice,
        key_start_choice
    ):
        on_top_lane = position[1] <= self._env.consts.TOP_CLIP
        time_on_lane = jnp.where(on_top_lane, time_on_lane + 1, 0)
        attack_time = jnp.where(on_top_lane, 0, attack_time)

        ufo_x = position[0].astype(jnp.float32)
        ufo_y = position[1].astype(jnp.float32)
        lane_x_at_ufo_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (ufo_y - float(self._env.consts.TOP_CLIP))
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)
        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25

        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        is_teleport = _is_teleport_ufo_pattern(pattern_id)

        shots_left = pattern_timer & 7
        last_lane = (pattern_timer >> 3) & 15
        shoot_now = (pattern_timer >> 7) & 1

        def update_triple():
            can_shoot = (shots_left > 0) & is_on_lane & (closest_lane_id != last_lane)
            new_shoot_now = jnp.where(shoot_now == 1, 0, jnp.where(can_shoot, 1, 0))
            new_shots_left = jnp.where(can_shoot, shots_left - 1, shots_left)
            new_last_lane = jnp.where(can_shoot, closest_lane_id, last_lane)
            return (new_shoot_now << 7) | (new_last_lane << 3) | new_shots_left

        plain_timer = jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer))
        teleport_timer = _pack_teleport_ufo_timer(
            jnp.maximum(_get_teleport_ufo_remaining(pattern_timer) - 1, 0),
            _has_teleport_ufo_been_used(pattern_timer),
        )
        pattern_timer = jnp.where(is_triple, update_triple(), jnp.where(is_teleport, teleport_timer, plain_timer))

        shootable_lane = (closest_lane_id > 0) & (closest_lane_id < 6)
        allow_shoot = (~on_top_lane) & shootable_lane

        is_drop_pattern = (
            (pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT))
            | (pattern_id == int(WhiteUFOPattern.DROP_LEFT))
            | (pattern_id == int(WhiteUFOPattern.DROP_RIGHT))
            | (pattern_id == int(WhiteUFOPattern.MOVE_BACK))
            | is_teleport
        )
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = is_drop_pattern | is_shoot_pattern | is_triple
        attack_time = jnp.where((~on_top_lane) & is_engagement_pattern, attack_time + 1, attack_time)

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        movement_finished = (is_retreat | is_move_back) & on_top_lane
        pattern_id = jnp.where(movement_finished, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(movement_finished, 0, pattern_timer)
        attack_time = jnp.where(movement_finished, 0, attack_time)

        triple_finished = is_triple & ((pattern_timer & 7) == 0) & jnp.logical_not((pattern_timer >> 7) & 1) & is_on_lane

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        in_restricted_stage = position[1] >= 86.0
        min_lane = jnp.where(in_restricted_stage, 1, 0)
        max_lane = jnp.where(in_restricted_stage, 5, 6)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, min_lane, max_lane)

        triple_stuck = is_triple & is_on_lane & (shots_left > 0) & (target_lane_id == closest_lane_id) & (closest_lane_id == last_lane)
        triple_finished = triple_finished | triple_stuck

        teleport_finished = is_teleport & (_get_teleport_ufo_remaining(pattern_timer) == 0)
        pattern_finished = jnp.where(is_triple, triple_finished, jnp.where(is_teleport, teleport_finished, pattern_timer == 0))
        pattern_finished_off_top = (~on_top_lane) & is_engagement_pattern & pattern_finished & is_on_lane

        retreat_prob = self._env._white_ufo_retreat_prob(attack_time)
        retreat_now = pattern_finished_off_top & (retreat_roll < retreat_prob)
        pattern_id = jnp.where(retreat_now, int(WhiteUFOPattern.RETREAT), pattern_id)
        pattern_timer = jnp.where(retreat_now, self._env.consts.WHITE_UFO_RETREAT_DURATION, pattern_timer)
        attack_time = jnp.where(retreat_now, 0, attack_time)

        chain_next = pattern_finished_off_top & (~retreat_now)
        ufo_stage = _get_index_ufo(position[1])

        def choose_chain_pattern(_):
            is_kamikaze_zone = position[1] >= self._env.consts.KAMIKAZE_Y_THRESHOLD
            pattern, duration = self._white_ufo_choose_pattern(
                key_chain_choice,
                allow_shoot=allow_shoot,
                prev_pattern=pattern_id,
                is_kamikaze_zone=is_kamikaze_zone,
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        def keep_after_chain(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(chain_next, choose_chain_pattern, keep_after_chain, operand=None)

        should_choose_new = on_top_lane & (pattern_id == int(WhiteUFOPattern.IDLE)) & (pattern_timer == 0) & (spawn_delay == 0)
        p_start = type(self._env).entropy_heat_prob_static(
            jnp.where(already_left, time_on_lane * 10, time_on_lane),
            alpha=self._env.consts.WHITE_UFO_ATTACK_ALPHA,
            p_min=jnp.where(already_left, 0.1, self._env.consts.WHITE_UFO_ATTACK_P_MIN),
            p_max=self._env.consts.WHITE_UFO_ATTACK_P_MAX,
        )
        start_attack = should_choose_new & (start_roll < p_start)

        def choose_new_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(
                key_start_choice,
                allow_shoot=jnp.array(False),
                prev_pattern=pattern_id,
                is_kamikaze_zone=jnp.array(False),
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        def keep_pattern(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(start_attack, choose_new_pattern, keep_pattern, operand=None)
        return pattern_id, pattern_timer, time_on_lane, attack_time

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_step(
        self,
        sector,
        white_ufo_position,
        white_ufo_vel,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        key,
    ):
        white_ufo_vel_x = white_ufo_vel[0]
        white_ufo_vel_y = white_ufo_vel[1]

        offscreen_pos = self._env.enemy_offscreen
        is_offscreen = jnp.all(white_ufo_position == offscreen_pos)

        new_key, key_motion, key_chain_choice, key_start_choice, float_key = jax.random.split(key, 5)
        float_rolls = jax.random.uniform(float_key, shape=(4,))

        spawn_delay_roll = float_rolls[0]
        motion_roll = float_rolls[1]
        retreat_roll = float_rolls[2]
        start_roll = float_rolls[3]

        spawn_delay = jnp.maximum(spawn_delay - 1, 0)

        pattern_id, pattern_timer, time_on_lane, attack_time = self._white_ufo_update_pattern_state(
            sector,
            white_ufo_position,
            time_on_lane,
            attack_time,
            already_left,
            spawn_delay,
            pattern_id,
            pattern_timer,
            retreat_roll,
            start_roll,
            key_chain_choice,
            key_start_choice
        )

        ufo_x = white_ufo_position[0].astype(jnp.float32)
        ufo_y = white_ufo_position[1].astype(jnp.float32)
        lane_x_at_ufo_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (ufo_y - float(self._env.consts.TOP_CLIP))
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)
        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25
        min_lane, max_lane = _get_teleport_ufo_lane_bounds_from_y(ufo_y)
        candidate_lanes, candidate_valid = _get_teleport_ufo_candidate_lanes(closest_lane_id, min_lane, max_lane)
        has_candidate_lane = jnp.any(candidate_valid)

        def choose_target_lane(_):
            candidate_probs = candidate_valid.astype(jnp.float32)
            candidate_probs = candidate_probs / jnp.sum(candidate_probs)
            return jax.random.choice(key_motion, candidate_lanes, shape=(), p=candidate_probs)

        target_lane = jax.lax.cond(has_candidate_lane, choose_target_lane, lambda _: closest_lane_id, operand=None)
        can_teleport_now = (
            (~is_offscreen)
            & _is_teleport_ufo_pattern(pattern_id)
            & (~_has_teleport_ufo_been_used(pattern_timer))
            & is_on_lane
            & has_candidate_lane
        )
        teleported_position = jnp.array([lane_x_at_ufo_y[target_lane], ufo_y], dtype=jnp.float32)
        white_ufo_position = jnp.where(can_teleport_now, teleported_position, white_ufo_position)
        pattern_timer = jnp.where(
            can_teleport_now,
            _pack_teleport_ufo_timer(_get_teleport_ufo_remaining(pattern_timer), jnp.array(True)),
            pattern_timer,
        )

        requires_lane_motion = self._white_ufo_pattern_requires_lane_motion(pattern_id)
        on_top_lane = white_ufo_position[1] <= self._env.consts.TOP_CLIP
        already_left = already_left | jnp.logical_not(on_top_lane)

        def follow_lane(_):
            return self._env._white_ufo_normal(white_ufo_position, white_ufo_vel_x, white_ufo_vel_y, pattern_id, already_left)

        def stay_on_top(_):
            return self._env._white_ufo_top_lane(white_ufo_position, white_ufo_vel_x, pattern_id, motion_roll)

        white_ufo_vel_x, white_ufo_vel_y = jax.lax.cond(requires_lane_motion, follow_lane, stay_on_top, operand=None)

        new_x = white_ufo_position[0] + white_ufo_vel_x
        new_y = white_ufo_position[1] + white_ufo_vel_y

        on_top_lane = new_y <= self._env.consts.TOP_CLIP
        clipped_x = jnp.clip(new_x, self._env.consts.LEFT_CLIP_PLAYER, self._env.consts.RIGHT_CLIP_PLAYER)
        new_x = jnp.where(on_top_lane, clipped_x, new_x)
        new_y = jnp.clip(new_y, self._env.consts.TOP_CLIP, self._env.consts.PLAYER_POS_Y + 1.0)

        should_respawn = (~is_offscreen) & ((new_x < 0) | (new_x > self._env.consts.SCREEN_WIDTH) | (new_y > self._env.consts.PLAYER_POS_Y))

        white_ufo_position = jnp.where(should_respawn, jnp.array([81.0, 43.0]), jnp.array([new_x, new_y]))
        white_ufo_vel_x = jnp.where(should_respawn, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(should_respawn, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(should_respawn, 0, time_on_lane)
        attack_time = jnp.where(should_respawn, 0, attack_time)
        new_spawn_delay_val = jnp.floor(spawn_delay_roll * 300.0).astype(jnp.int32) + 1
        spawn_delay = jnp.where(should_respawn, new_spawn_delay_val, spawn_delay)
        pattern_id = jnp.where(should_respawn, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(should_respawn, 0, pattern_timer)

        white_ufo_position = jnp.where(is_offscreen, offscreen_pos, white_ufo_position)
        white_ufo_vel_x = jnp.where(is_offscreen, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(is_offscreen, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(is_offscreen, 0, time_on_lane)
        attack_time = jnp.where(is_offscreen, 0, attack_time)
        spawn_delay = jnp.where(is_offscreen, 0, spawn_delay)
        pattern_id = jnp.where(is_offscreen, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(is_offscreen, 0, pattern_timer)

        return (
            white_ufo_position,
            white_ufo_vel_x,
            white_ufo_vel_y,
            time_on_lane,
            attack_time,
            already_left,
            spawn_delay,
            pattern_id,
            pattern_timer,
            new_key,
        )
