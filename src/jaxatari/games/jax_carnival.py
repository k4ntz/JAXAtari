from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action




class CarnivalConstants(NamedTuple):
    # Screen
    w: int
    h: int

    # Actions
    n_actions: int
    action_left: jax.Array
    action_right: jax.Array
    action_fire: jax.Array

    # Episode
    max_steps: int
    sticky_action_p: float

    # Frame-skip
    frameskip_min: int
    frameskip_max: int

    # Speeds (pixels per internal step)
    player_speed_px_per_step: int
    bullet_speed_px_per_step: int
    target_speed_px_per_step: int
    fall_speed_px_per_step: int

    # Player
    player_w: int
    player_h: int
    player_y: int

    # Bullet
    bullet_w: int
    bullet_h: int

    # Lanes
    n_lanes: int
    lane_dir: jax.Array
    lane_spawn_mul: jax.Array

    # Targets (lane movers)
    max_targets: int
    target_w: int
    target_h: int

    # Spawn spacing (pixels)
    min_gap_px: int
    max_gap_px: int

    # Random spawn offset (still off-screen)
    spawn_extra_max_px: int

    # Target types (ONLY 3: yellow, blue, red)
    n_types: int
    type_spawn_weights: jax.Array
    type_scores: jax.Array
    type_color_idx: jax.Array

    # Red plus/minus behavior (red is pm_type_id)
    pm_type_id: int
    pm_bonus_value_score: int
    pm_display_value: int

    # Falling target spawns
    fall_spawn_min_steps: int
    fall_spawn_max_steps: int

    # Ammo
    initial_ammo: int
    max_ammo: int

    # Renderer palette
    palette_rgb: jax.Array

    @staticmethod
    def default() -> "CarnivalConstants":
        w, h = 160, 210
        n_actions = 18

        action_left = jnp.zeros((n_actions,), dtype=jnp.bool_)
        action_right = jnp.zeros((n_actions,), dtype=jnp.bool_)
        action_fire = jnp.zeros((n_actions,), dtype=jnp.bool_)

        # Atari-like action ids
        left_ids = jnp.array([4, 13, 16], dtype=jnp.int32)
        right_ids = jnp.array([3, 12, 15], dtype=jnp.int32)
        fire_ids = jnp.array([1, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32)

        action_left = action_left.at[left_ids].set(True)
        action_right = action_right.at[right_ids].set(True)
        action_fire = action_fire.at[fire_ids].set(True)

        # Episode settings
        max_steps = 60 * 60
        sticky_action_p = 0.25

        # Frame-skip
        frameskip_min = 1
        frameskip_max = 1

        # Speeds are explicit and controllable
        player_speed_px_per_step = 3
        bullet_speed_px_per_step = 2
        target_speed_px_per_step = 1
        fall_speed_px_per_step = 1

        player_w, player_h, player_y = 16, 8, 190
        bullet_w, bullet_h = 2, 4

        n_lanes = 3
        # Lane directions: lane 0 and 2 right, lane 1 left
        lane_dir = jnp.array([1, -1, 1], dtype=jnp.int32)
        lane_spawn_mul = jnp.array([1, 1, 3], dtype=jnp.int32)

        max_targets = 24
        target_w, target_h = 14, 8

        min_gap_px = target_w + 6
        max_gap_px = target_w + 40
        spawn_extra_max_px = 80

        # Types: 0 yellow (10), 1 blue (20), 2 red (+/- 15)
        n_types = 3
        # Mostly yellow, less blue, few red
        type_spawn_weights = jnp.array([0.78, 0.16, 0.06], dtype=jnp.float32)
        type_spawn_weights = type_spawn_weights / jnp.sum(type_spawn_weights)

        type_scores = jnp.array([10, 20, 15], dtype=jnp.int32)

        # Palette indices for each target type
        # 1 = yellow, 4 = blue, 5 = red
        type_color_idx = jnp.array([1, 4, 5], dtype=jnp.uint8)

        # Red is the only +/- type
        pm_type_id = 2
        pm_bonus_value_score = 15
        pm_display_value = 15

        # Falling target spawn cooldown
        fall_spawn_min_steps = 25
        fall_spawn_max_steps = 80

        max_ammo = 40
        initial_ammo = 40

        palette_rgb = jnp.array(
            [
                [0, 0, 0],        # 0 background
                [255, 255, 0],    # 1 yellow
                [255, 255, 255],  # 2 white digits
                [0, 255, 0],      # 3 green (player, bullet)
                [0, 160, 255],    # 4 blue
                [255, 0, 0],      # 5 red
            ],
            dtype=jnp.uint8,
        )

        return CarnivalConstants(
            w=w,
            h=h,
            n_actions=n_actions,
            action_left=action_left,
            action_right=action_right,
            action_fire=action_fire,
            max_steps=max_steps,
            sticky_action_p=sticky_action_p,
            frameskip_min=frameskip_min,
            frameskip_max=frameskip_max,
            player_speed_px_per_step=player_speed_px_per_step,
            bullet_speed_px_per_step=bullet_speed_px_per_step,
            target_speed_px_per_step=target_speed_px_per_step,
            fall_speed_px_per_step=fall_speed_px_per_step,
            player_w=player_w,
            player_h=player_h,
            player_y=player_y,
            bullet_w=bullet_w,
            bullet_h=bullet_h,
            n_lanes=n_lanes,
            lane_dir=lane_dir,
            lane_spawn_mul=lane_spawn_mul,
            max_targets=max_targets,
            target_w=target_w,
            target_h=target_h,
            min_gap_px=min_gap_px,
            max_gap_px=max_gap_px,
            spawn_extra_max_px=spawn_extra_max_px,
            n_types=n_types,
            type_spawn_weights=type_spawn_weights,
            type_scores=type_scores,
            type_color_idx=type_color_idx,
            pm_type_id=pm_type_id,
            pm_bonus_value_score=pm_bonus_value_score,
            pm_display_value=pm_display_value,
            fall_spawn_min_steps=fall_spawn_min_steps,
            fall_spawn_max_steps=fall_spawn_max_steps,
            initial_ammo=initial_ammo,
            max_ammo=max_ammo,
            palette_rgb=palette_rgb,
        )


class CarnivalState(NamedTuple):
    rng: jax.Array
    t: jax.Array
    score: jax.Array

    # Lane y positions, randomized once per episode
    lane_y: jax.Array

    player_x: jax.Array

    bullet_active: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array

    ammo: jax.Array

    # Lane targets (fixed-shape pool)
    target_active: jax.Array
    target_x: jax.Array
    target_lane: jax.Array
    target_type: jax.Array
    target_sign: jax.Array

    spawn_timer: jax.Array

    # Falling target (single)
    fall_active: jax.Array
    fall_x: jax.Array
    fall_y: jax.Array
    fall_type: jax.Array
    fall_sign: jax.Array
    fall_timer: jax.Array

    # HUD indicator for +/- 15 (in red), updated when a red target spawns
    hud_pm_sign: jax.Array

    last_action: jax.Array

    last_reward: jax.Array
    last_hit: jax.Array
    last_hit_index: jax.Array


class CarnivalInfo(NamedTuple):
    t: jax.Array
    score: jax.Array
    ammo: jax.Array
    last_reward: jax.Array
    hit: jax.Array
    hit_index: jax.Array
    pm_sign: jax.Array
    last_action: jax.Array


class CarnivalObs(NamedTuple):
    # IMPORTANT: alphabetical field order to match spaces.Dict leaf traversal reliably
    ammo: jax.Array
    bullet_active: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    fall_active: jax.Array
    fall_sign: jax.Array
    fall_type: jax.Array
    fall_x: jax.Array
    fall_y: jax.Array
    player_x: jax.Array
    target_active: jax.Array
    target_lane: jax.Array
    target_sign: jax.Array
    target_type: jax.Array
    target_x: jax.Array


class _CarnivalRenderer:
    def __init__(self, c: CarnivalConstants):
        self.c = c

        self._pad_x = int(c.target_w)
        self._pad_y = int(c.target_h)

        self._player_patch = jnp.full((c.player_h, c.player_w), jnp.uint8(3), dtype=jnp.uint8)
        self._bullet_patch = jnp.full((c.bullet_h, c.bullet_w), jnp.uint8(3), dtype=jnp.uint8)

        # Digit masks (5x3)
        d = []
        d.append(jnp.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8))  # 0
        d.append(jnp.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=jnp.uint8))  # 1
        d.append(jnp.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=jnp.uint8))  # 2
        d.append(jnp.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8))  # 3
        d.append(jnp.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=jnp.uint8))  # 4
        d.append(jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8))  # 5
        d.append(jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8))  # 6
        d.append(jnp.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=jnp.uint8))  # 7
        d.append(jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8))  # 8
        d.append(jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8))  # 9

        self._digit_masks = jnp.stack(d, axis=0).astype(jnp.uint8)
        self._digit_patches = (self._digit_masks * jnp.uint8(2)).astype(jnp.uint8)

        # Minimal glyphs for labels
        self._blank = jnp.zeros((5, 3), dtype=jnp.uint8)
        self._glyph_A = jnp.array([[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=jnp.uint8) * 2
        self._glyph_M = jnp.array([[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=jnp.uint8) * 2
        self._glyph_O = d[0] * 2
        self._glyph_S = jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8) * 2
        self._glyph_C = jnp.array([[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=jnp.uint8) * 2
        self._glyph_R = jnp.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=jnp.uint8) * 2
        self._glyph_E = jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=jnp.uint8) * 2
        self._glyph_colon = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.uint8) * 2

        self._label_score = self._make_text_patch("SCORE:")
        self._label_ammo = self._make_text_patch("AMMO:")

        # Plus and minus masks (5x3)
        self._plus_mask = jnp.array(
            [[0, 1, 0],
             [0, 1, 0],
             [1, 1, 1],
             [0, 1, 0],
             [0, 1, 0]],
            dtype=jnp.uint8,
        )
        self._minus_mask = jnp.array(
            [[0, 0, 0],
             [0, 0, 0],
             [1, 1, 1],
             [0, 0, 0],
             [0, 0, 0]],
            dtype=jnp.uint8,
        )

        self._target_patches = self._build_target_patches()
        self._pm_hud_patches = self._build_pm_hud_patches()

    def _make_text_patch(self, text: str) -> jax.Array:
        cols = len(text) * 4 - 1
        patch = jnp.zeros((5, cols), dtype=jnp.uint8)
        for i, ch in enumerate(text):
            if ch == "A":
                g = self._glyph_A
            elif ch == "M":
                g = self._glyph_M
            elif ch == "O":
                g = self._glyph_O
            elif ch == "S":
                g = self._glyph_S
            elif ch == "C":
                g = self._glyph_C
            elif ch == "R":
                g = self._glyph_R
            elif ch == "E":
                g = self._glyph_E
            elif ch == ":":
                g = self._glyph_colon
            else:
                g = self._blank
            x0 = i * 4
            patch = patch.at[:, x0:x0 + 3].set(g.astype(jnp.uint8))
        return patch

    def _blit_patch_padded(self, raster_pad: jax.Array, patch: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        # JIT-safe blit with dynamic indices
        xp = (x + jnp.int32(self._pad_x)).astype(jnp.int32)
        yp = (y + jnp.int32(self._pad_y)).astype(jnp.int32)

        H, W = raster_pad.shape
        h, w = patch.shape

        xp = jnp.clip(xp, jnp.int32(0), jnp.int32(W - w)).astype(jnp.int32)
        yp = jnp.clip(yp, jnp.int32(0), jnp.int32(H - h)).astype(jnp.int32)

        sub = jax.lax.dynamic_slice(raster_pad, (yp, xp), (h, w))
        merged = jnp.maximum(sub, patch)
        return jax.lax.dynamic_update_slice(raster_pad, merged, (yp, xp))

    @staticmethod
    def _overlay_mask(base: jax.Array, mask: jax.Array, x0: int, y0: int, color: jnp.uint8) -> jax.Array:
        h, w = mask.shape
        glyph = (mask.astype(jnp.uint8) * color).astype(jnp.uint8)
        sub = base[y0:y0 + h, x0:x0 + w]
        return base.at[y0:y0 + h, x0:x0 + w].set(jnp.maximum(sub, glyph))

    def _build_target_patches(self) -> jax.Array:
        c = self.c
        patches = []
        y0 = 1

        for t in range(int(c.n_types)):
            base_color = int(c.type_color_idx[t])
            base = jnp.full((c.target_h, c.target_w), jnp.uint8(base_color), dtype=jnp.uint8)

            variants = []
            for sign_index in [0, 1]:
                patch = base

                if t == int(c.pm_type_id):
                    # Red target shows +/- and "15" in WHITE for contrast
                    is_minus = sign_index == 0
                    sign_mask = self._minus_mask if is_minus else self._plus_mask
                    patch = self._overlay_mask(patch, sign_mask, x0=1, y0=y0, color=jnp.uint8(2))

                    digits = [1, 5]  # Always show 15
                    patch = self._overlay_mask(patch, self._digit_masks[digits[0]], x0=6, y0=y0, color=jnp.uint8(2))
                    patch = self._overlay_mask(patch, self._digit_masks[digits[1]], x0=9, y0=y0, color=jnp.uint8(2))
                else:
                    # Yellow type (t == 0) should have no digits drawn on the box
                    if t != 0:
                        val = int(c.type_scores[t])
                        tens = (val // 10) % 10
                        ones = val % 10
                        patch = self._overlay_mask(patch, self._digit_masks[tens], x0=4, y0=y0, color=jnp.uint8(2))
                        patch = self._overlay_mask(patch, self._digit_masks[ones], x0=7, y0=y0, color=jnp.uint8(2))


                variants.append(patch)

            patches.append(jnp.stack(variants, axis=0))

        return jnp.stack(patches, axis=0).astype(jnp.uint8)

    def _build_pm_hud_patches(self) -> jax.Array:
        # Top HUD indicator: "+15" or "-15" in RED (palette id 5)
        # Patch size: 5 x 10
        base = jnp.zeros((5, 10), dtype=jnp.uint8)

        def build(sign_index: int) -> jax.Array:
            patch = base
            is_minus = sign_index == 0
            sign_mask = self._minus_mask if is_minus else self._plus_mask
            patch = self._overlay_mask(patch, sign_mask, x0=0, y0=0, color=jnp.uint8(5))
            patch = self._overlay_mask(patch, self._digit_masks[1], x0=4, y0=0, color=jnp.uint8(5))
            patch = self._overlay_mask(patch, self._digit_masks[5], x0=7, y0=0, color=jnp.uint8(5))
            return patch

        minus_patch = build(0)
        plus_patch = build(1)
        return jnp.stack([minus_patch, plus_patch], axis=0).astype(jnp.uint8)

    def _draw_digits(self, raster_pad: jax.Array, value: jax.Array, n_digits: int, x: jax.Array, y: jax.Array) -> jax.Array:
        v = jnp.asarray(value, dtype=jnp.int32)
        v = jnp.clip(v, 0, 9999)

        d0 = (v // 1000) % 10
        d1 = (v // 100) % 10
        d2 = (v // 10) % 10
        d3 = v % 10
        digits4 = jnp.stack([d0, d1, d2, d3], axis=0).astype(jnp.int32)

        def body(i: jax.Array, rast: jax.Array) -> jax.Array:
            di = digits4[4 - n_digits + i]
            patch = jax.lax.dynamic_index_in_dim(self._digit_patches, di, axis=0, keepdims=False)
            return self._blit_patch_padded(rast, patch, x + i * jnp.int32(4), y)

        return jax.lax.fori_loop(0, n_digits, body, raster_pad)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CarnivalState) -> jax.Array:
        c = self.c
        raster_pad = jnp.zeros((c.h + 2 * self._pad_y, c.w + 2 * self._pad_x), dtype=jnp.uint8)

        def draw_target(i: jax.Array, rast: jax.Array) -> jax.Array:
            active = state.target_active[i]
            x = state.target_x[i]
            lane = state.target_lane[i]
            y = state.lane_y[lane]

            ttype = state.target_type[i]
            is_pm = ttype == jnp.int32(c.pm_type_id)
            is_minus = (state.target_sign[i] < 0) & is_pm
            sign_index = jnp.where(is_minus, jnp.int32(0), jnp.int32(1))

            patch2 = self._target_patches[ttype, sign_index]

            return jax.lax.cond(
                active,
                lambda r: self._blit_patch_padded(r, patch2, x, y),
                lambda r: r,
                rast,
            )

        raster_pad = jax.lax.fori_loop(0, c.max_targets, draw_target, raster_pad)

        # Falling target
        fall_is_pm = state.fall_type == jnp.int32(c.pm_type_id)
        fall_is_minus = (state.fall_sign < 0) & fall_is_pm
        fall_sign_index = jnp.where(fall_is_minus, jnp.int32(0), jnp.int32(1))
        fall_patch = self._target_patches[state.fall_type, fall_sign_index]

        raster_pad = jax.lax.cond(
            state.fall_active,
            lambda r: self._blit_patch_padded(r, fall_patch, state.fall_x, state.fall_y),
            lambda r: r,
            raster_pad,
        )

        # Player
        raster_pad = self._blit_patch_padded(raster_pad, self._player_patch, state.player_x, jnp.int32(c.player_y))

        # Bullet
        raster_pad = jax.lax.cond(
            state.bullet_active,
            lambda r: self._blit_patch_padded(r, self._bullet_patch, state.bullet_x, state.bullet_y),
            lambda r: r,
            raster_pad,
        )

        # HUD: Score left, Ammo right
        margin = jnp.int32(2)
        gap = jnp.int32(2)
        y0 = jnp.int32(2)

        score_x0 = margin
        raster_pad = self._blit_patch_padded(raster_pad, self._label_score, score_x0, y0)
        raster_pad = self._draw_digits(
            raster_pad, state.score, 4, score_x0 + jnp.int32(self._label_score.shape[1]) + gap, y0
        )

        ammo_digits_w = jnp.int32(2 * 4)
        ammo_total_w = jnp.int32(self._label_ammo.shape[1]) + gap + ammo_digits_w
        ammo_x0 = jnp.int32(c.w) - ammo_total_w - margin

        raster_pad = self._blit_patch_padded(raster_pad, self._label_ammo, ammo_x0, y0)
        raster_pad = self._draw_digits(
            raster_pad, state.ammo, 2, ammo_x0 + jnp.int32(self._label_ammo.shape[1]) + gap, y0
        )

        # Top indicator: +15 / -15 in red
        hud_sign_index = jnp.where(state.hud_pm_sign < 0, jnp.int32(0), jnp.int32(1))
        pm_patch = self._pm_hud_patches[hud_sign_index]
        pm_x0 = (jnp.int32(c.w) - jnp.int32(pm_patch.shape[1])) // jnp.int32(2)
        raster_pad = self._blit_patch_padded(raster_pad, pm_patch, pm_x0, y0)

        raster = raster_pad[self._pad_y:self._pad_y + c.h, self._pad_x:self._pad_x + c.w]
        return c.palette_rgb[raster]


class JaxCarnival(JaxEnvironment):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, consts: Optional[CarnivalConstants] = None):
        super().__init__()

        self.metadata = dict(JaxCarnival.metadata)

        self.consts = CarnivalConstants.default() if consts is None else consts

        c = self.consts
        

        self._renderer = _CarnivalRenderer(c)
        self.renderer = self._renderer

        self._action_space = spaces.Discrete(c.n_actions)

        # IMPORTANT: keys in the same deterministic order as CarnivalObs fields (alphabetical)
        self._observation_space = spaces.Dict(
            {
                "ammo": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(c.max_ammo, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "bullet_active": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(1, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "bullet_x": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(c.w, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "bullet_y": spaces.Box(
                    low=jnp.array(-c.bullet_h, dtype=jnp.int32),
                    high=jnp.array(c.h, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "fall_active": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(1, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "fall_sign": spaces.Box(
                    low=jnp.array(-1, dtype=jnp.int32),
                    high=jnp.array(1, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "fall_type": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(c.n_types - 1, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "fall_x": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(c.w, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "fall_y": spaces.Box(
                    low=jnp.array(-c.target_h, dtype=jnp.int32),
                    high=jnp.array(c.h, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "player_x": spaces.Box(
                    low=jnp.array(0, dtype=jnp.int32),
                    high=jnp.array(c.w - c.player_w, dtype=jnp.int32),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "target_active": spaces.Box(
                    low=jnp.zeros((c.max_targets,), dtype=jnp.int32),
                    high=jnp.ones((c.max_targets,), dtype=jnp.int32),
                    shape=(c.max_targets,),
                    dtype=jnp.int32,
                ),
                "target_lane": spaces.Box(
                    low=jnp.zeros((c.max_targets,), dtype=jnp.int32),
                    high=jnp.full((c.max_targets,), c.n_lanes - 1, dtype=jnp.int32),
                    shape=(c.max_targets,),
                    dtype=jnp.int32,
                ),
                "target_sign": spaces.Box(
                    low=jnp.full((c.max_targets,), -1, dtype=jnp.int32),
                    high=jnp.full((c.max_targets,), 1, dtype=jnp.int32),
                    shape=(c.max_targets,),
                    dtype=jnp.int32,
                ),
                "target_type": spaces.Box(
                    low=jnp.zeros((c.max_targets,), dtype=jnp.int32),
                    high=jnp.full((c.max_targets,), c.n_types - 1, dtype=jnp.int32),
                    shape=(c.max_targets,),
                    dtype=jnp.int32,
                ),
                "target_x": spaces.Box(
                    low=jnp.full((c.max_targets,), -c.target_w - c.spawn_extra_max_px, dtype=jnp.int32),
                    high=jnp.full((c.max_targets,), c.w + c.spawn_extra_max_px, dtype=jnp.int32),
                    shape=(c.max_targets,),
                    dtype=jnp.int32,
                ),
            }
        )

        self._image_space = spaces.Box(
            low=jnp.zeros((c.h, c.w, 3), dtype=jnp.uint8),
            high=jnp.ones((c.h, c.w, 3), dtype=jnp.uint8) * jnp.uint8(255),
            shape=(c.h, c.w, 3),
            dtype=jnp.uint8,
        )

    def action_space(self) -> spaces.Space:
        return self._action_space

    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def image_space(self) -> spaces.Space:
        return self._image_space

    def get_action_meanings(self) -> Tuple[str, ...]:
        return (
            "NOOP",
            "FIRE",
            "UP",
            "RIGHT",
            "LEFT",
            "DOWN",
            "UPRIGHT",
            "UPLEFT",
            "DOWNRIGHT",
            "DOWNLEFT",
            "UPFIRE",
            "RIGHTFIRE",
            "LEFTFIRE",
            "DOWNFIRE",
            "UPRIGHTFIRE",
            "UPLEFTFIRE",
            "DOWNRIGHTFIRE",
            "DOWNLEFTFIRE",
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> Tuple[CarnivalObs, CarnivalState]:
        c = self.consts
        rng, k_lane, k_timers, k_fall = jax.random.split(key, 4)

        # Randomize lane y positions (top-to-bottom), small jitter
        base = jnp.array([30, 60, 90], dtype=jnp.int32)
        jitter = jax.random.randint(k_lane, (c.n_lanes,), jnp.int32(-6), jnp.int32(7), dtype=jnp.int32)
        lane_y = jnp.clip(base + jitter, jnp.int32(18), jnp.int32(120)).astype(jnp.int32)

        player_x0 = jnp.int32((c.w - c.player_w) // 2)

        fall_timer0 = jax.random.randint(
            k_fall,
            (),
            jnp.int32(c.fall_spawn_min_steps),
            jnp.int32(c.fall_spawn_max_steps + 1),
            dtype=jnp.int32,
        )

        state = CarnivalState(
            rng=rng,
            t=jnp.int32(0),
            score=jnp.int32(0),
            lane_y=lane_y,
            player_x=player_x0,
            bullet_active=jnp.array(False),
            bullet_x=jnp.int32(0),
            bullet_y=jnp.int32(0),
            ammo=jnp.int32(c.initial_ammo),
            target_active=jnp.zeros((c.max_targets,), dtype=jnp.bool_),
            target_x=jnp.zeros((c.max_targets,), dtype=jnp.int32),
            target_lane=jnp.zeros((c.max_targets,), dtype=jnp.int32),
            target_type=jnp.zeros((c.max_targets,), dtype=jnp.int32),
            target_sign=jnp.ones((c.max_targets,), dtype=jnp.int32),
            spawn_timer=jax.random.randint(k_timers, (c.n_lanes,), jnp.int32(0), jnp.int32(20), dtype=jnp.int32),
            fall_active=jnp.array(False),
            fall_x=jnp.int32(0),
            fall_y=jnp.int32(-c.target_h),
            fall_type=jnp.int32(0),
            fall_sign=jnp.int32(1),
            fall_timer=fall_timer0,
            hud_pm_sign=jnp.int32(1),
            last_action=jnp.int32(0),
            last_reward=jnp.float32(0.0),
            last_hit=jnp.array(False),
            last_hit_index=jnp.int32(-1),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: CarnivalState, action: Action
    ) -> Tuple[CarnivalObs, CarnivalState, jax.Array, jax.Array, CarnivalInfo]:
        c = self.consts

        a0 = self._sanitize_action(action)
        rng1, a = self._apply_sticky_action(state.rng, state.last_action, a0)

        move_left = c.action_left[a]
        move_right = c.action_right[a]
        fire = c.action_fire[a]

        rng_fs, rng2 = jax.random.split(rng1, 2)
        fs = jax.random.randint(
            rng_fs,
            (),
            jnp.int32(c.frameskip_min),
            jnp.int32(c.frameskip_max + 1),
            dtype=jnp.int32,
        )

        def one_frame(st: CarnivalState) -> Tuple[CarnivalState, jax.Array]:
            player_x1 = self._player_step(st.player_x, move_left, move_right)

            bullet_active1, bullet_x1, bullet_y1, ammo1 = self._bullet_fire_step(
                st.bullet_active, st.bullet_x, st.bullet_y, st.ammo, player_x1, fire
            )
            bullet_active2, bullet_y2 = self._bullet_move_step(bullet_active1, bullet_y1)

            target_active2, target_x2 = self._targets_move_and_deactivate(
                st.target_active, st.target_x, st.target_lane
            )

            (
                rng3,
                target_active3,
                target_x3,
                target_lane3,
                target_type3,
                target_sign3,
                spawn_timer2,
                hud_pm_sign2,
            ) = self._spawn_step(
                st.rng,
                target_active2,
                target_x2,
                st.target_lane,
                st.target_type,
                st.target_sign,
                st.spawn_timer,
                st.hud_pm_sign,
            )

            # Keep red target signs, force others to +1
            target_sign4 = self._pm_sign_step(target_active3, target_type3, target_sign3)

            (
                rng4,
                fall_active2,
                fall_x2,
                fall_y2,
                fall_type2,
                fall_sign2,
                fall_timer2,
                hud_pm_sign3,
            ) = self._fall_step(
                rng3,
                st.fall_active,
                st.fall_x,
                st.fall_y,
                st.fall_type,
                st.fall_sign,
                st.fall_timer,
                hud_pm_sign2,
            )

            (
                bullet_active3,
                target_active4,
                fall_active3,
                reward_i32,
                hit,
                hit_idx,
            ) = self._bullet_target_collision(
                bullet_active2,
                bullet_x1,
                bullet_y2,
                st.lane_y,
                target_active3,
                target_x3,
                target_lane3,
                target_type3,
                target_sign4,
                fall_active2,
                fall_x2,
                fall_y2,
                fall_type2,
                fall_sign2,
            )

            ammo2 = jnp.clip(ammo1, jnp.int32(0), jnp.int32(c.max_ammo)).astype(jnp.int32)
            score2 = jnp.maximum(st.score + reward_i32, jnp.int32(0)).astype(jnp.int32)

            st2 = st._replace(
                rng=rng4,
                t=(st.t + jnp.int32(1)).astype(jnp.int32),
                score=score2,
                lane_y=st.lane_y,
                player_x=player_x1,
                bullet_active=bullet_active3,
                bullet_x=bullet_x1,
                bullet_y=bullet_y2,
                ammo=ammo2,
                target_active=target_active4,
                target_x=target_x3,
                target_lane=target_lane3,
                target_type=target_type3,
                target_sign=target_sign4,
                spawn_timer=spawn_timer2,
                fall_active=fall_active3,
                fall_x=fall_x2,
                fall_y=fall_y2,
                fall_type=fall_type2,
                fall_sign=fall_sign2,
                fall_timer=fall_timer2,
                hud_pm_sign=hud_pm_sign3,
                last_action=a,
                last_reward=reward_i32.astype(jnp.float32),
                last_hit=hit,
                last_hit_index=hit_idx,
            )
            return st2, reward_i32.astype(jnp.int32)

        def fs_body(i: jax.Array, carry):
            st, rsum = carry
            st2, r_i32 = one_frame(st)
            return (st2, (rsum + r_i32).astype(jnp.int32))

        carry0 = (state._replace(rng=rng2), jnp.int32(0))
        state_after, rsum_i32 = jax.lax.fori_loop(jnp.int32(0), fs, fs_body, carry0)

        reward = rsum_i32.astype(jnp.float32)
        state_after = state_after._replace(last_reward=reward)

        obs = self._get_observation(state_after)
        done = self._get_done(state_after)
        info = self._get_info(state_after)
        return obs, state_after, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CarnivalState) -> jax.Array:
        return self._renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: CarnivalState) -> CarnivalObs:
        return CarnivalObs(
            ammo=state.ammo.astype(jnp.int32),
            bullet_active=state.bullet_active.astype(jnp.int32),
            bullet_x=state.bullet_x.astype(jnp.int32),
            bullet_y=state.bullet_y.astype(jnp.int32),
            fall_active=state.fall_active.astype(jnp.int32),
            fall_sign=state.fall_sign.astype(jnp.int32),
            fall_type=state.fall_type.astype(jnp.int32),
            fall_x=state.fall_x.astype(jnp.int32),
            fall_y=state.fall_y.astype(jnp.int32),
            player_x=state.player_x.astype(jnp.int32),
            target_active=state.target_active.astype(jnp.int32),
            target_lane=state.target_lane.astype(jnp.int32),
            target_sign=state.target_sign.astype(jnp.int32),
            target_type=state.target_type.astype(jnp.int32),
            target_x=state.target_x.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CarnivalObs) -> jnp.ndarray:
        """
        Flatten object-centric observation into a 1D feature vector per frame.

        IMPORTANT: The order MUST match the pytree leaf order of observation_space().
        With our spaces.Dict keys (alphabetical), the order is:
        ammo, bullet_active, bullet_x, bullet_y,
        fall_active, fall_sign, fall_type, fall_x, fall_y,
        player_x,
        target_active, target_lane, target_sign, target_type, target_x
        """
        scalars = jnp.stack(
            [
                obs.ammo.astype(jnp.int32),
                obs.bullet_active.astype(jnp.int32),
                obs.bullet_x.astype(jnp.int32),
                obs.bullet_y.astype(jnp.int32),
                obs.fall_active.astype(jnp.int32),
                obs.fall_sign.astype(jnp.int32),
                obs.fall_type.astype(jnp.int32),
                obs.fall_x.astype(jnp.int32),
                obs.fall_y.astype(jnp.int32),
                obs.player_x.astype(jnp.int32),
            ],
            axis=0,
        )

        tail = jnp.concatenate(
            [
                obs.target_active.astype(jnp.int32),
                obs.target_lane.astype(jnp.int32),
                obs.target_sign.astype(jnp.int32),
                obs.target_type.astype(jnp.int32),
                obs.target_x.astype(jnp.int32),
            ],
            axis=0,
        )

        return jnp.concatenate([scalars, tail], axis=0).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CarnivalState) -> CarnivalInfo:
        return CarnivalInfo(
            t=jnp.asarray(state.t, dtype=jnp.int32),
            score=jnp.asarray(state.score, dtype=jnp.int32),
            ammo=jnp.asarray(state.ammo, dtype=jnp.int32),
            last_reward=jnp.asarray(state.last_reward, dtype=jnp.float32),
            hit=jnp.asarray(state.last_hit, dtype=jnp.bool_),
            hit_index=jnp.asarray(state.last_hit_index, dtype=jnp.int32),
            pm_sign=jnp.asarray(state.hud_pm_sign, dtype=jnp.int32),
            last_action=jnp.asarray(state.last_action, dtype=jnp.int32),
        )

    # Gymnasium functional wrapper hooks
    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: CarnivalState) -> CarnivalObs:
        return self._get_observation(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: CarnivalState, state: CarnivalState) -> jax.Array:
        return state.last_reward.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_terminated(self, state: CarnivalState) -> jax.Array:
        # Natural termination: no ammo and no bullet in flight
        return (state.ammo <= 0) & (~state.bullet_active)

    @partial(jax.jit, static_argnums=(0,))
    def _get_truncated(self, state: CarnivalState) -> jax.Array:
        return state.t >= jnp.int32(self.consts.max_steps)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: CarnivalState) -> jax.Array:
        return self._get_terminated(state) | self._get_truncated(state)

    def _sanitize_action(self, action: Action) -> jax.Array:
        a = jnp.asarray(action, dtype=jnp.int32)
        return jnp.clip(a, jnp.int32(0), jnp.int32(self.consts.n_actions - 1)).astype(jnp.int32)

    def _apply_sticky_action(self, rng: jax.Array, last_action: jax.Array, action: jax.Array) -> Tuple[jax.Array, jax.Array]:
        c = self.consts
        rng2, k = jax.random.split(rng, 2)
        use_last = jax.random.bernoulli(k, p=jnp.float32(c.sticky_action_p))
        a = jnp.where(use_last, last_action, action).astype(jnp.int32)
        return rng2, a

    def _player_step(self, player_x: jax.Array, move_left: jax.Array, move_right: jax.Array) -> jax.Array:
        c = self.consts
        dx = jnp.int32(0)
        dx = jnp.where(move_left, dx - jnp.int32(c.player_speed_px_per_step), dx)
        dx = jnp.where(move_right, dx + jnp.int32(c.player_speed_px_per_step), dx)
        x = (player_x + dx).astype(jnp.int32)
        return jnp.clip(x, jnp.int32(0), jnp.int32(c.w - c.player_w)).astype(jnp.int32)

    def _bullet_fire_step(
        self,
        bullet_active: jax.Array,
        bullet_x: jax.Array,
        bullet_y: jax.Array,
        ammo: jax.Array,
        player_x: jax.Array,
        fire: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        c = self.consts
        can_fire = (~bullet_active) & (ammo > 0) & fire

        def do_fire(_: None):
            bx = (player_x + jnp.int32(c.player_w // 2) - jnp.int32(c.bullet_w // 2)).astype(jnp.int32)
            by = jnp.int32(c.player_y - c.bullet_h)
            return jnp.array(True), bx, by, (ammo - jnp.int32(1)).astype(jnp.int32)

        def no_fire(_: None):
            return bullet_active, bullet_x, bullet_y, ammo

        return jax.lax.cond(can_fire, do_fire, no_fire, operand=None)

    def _bullet_move_step(self, bullet_active: jax.Array, bullet_y: jax.Array) -> Tuple[jax.Array, jax.Array]:
        c = self.consts
        y = (bullet_y - jnp.int32(c.bullet_speed_px_per_step)).astype(jnp.int32)
        still = bullet_active & (y >= -jnp.int32(c.bullet_h))
        y2 = jnp.where(still, y, bullet_y).astype(jnp.int32)
        return still, y2

    def _targets_move_and_deactivate(
        self,
        target_active: jax.Array,
        target_x: jax.Array,
        target_lane: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        c = self.consts
        dir_lane = c.lane_dir[target_lane].astype(jnp.int32)
        x2 = (target_x + dir_lane * jnp.int32(c.target_speed_px_per_step)).astype(jnp.int32)

        off_right = (dir_lane > 0) & (x2 >= jnp.int32(c.w))
        off_left = (dir_lane < 0) & ((x2 + jnp.int32(c.target_w)) <= jnp.int32(0))
        still_active = target_active & (~(off_right | off_left))
        return still_active, x2

    def _pm_sign_step(
        self,
        target_active: jax.Array,
        target_type: jax.Array,
        target_sign: jax.Array,
    ) -> jax.Array:
        # Keep the per-target sign for red targets, force others to +1
        c = self.consts
        is_pm = target_active & (target_type == jnp.int32(c.pm_type_id))
        return jnp.where(is_pm, target_sign, jnp.int32(1)).astype(jnp.int32)

    def _spawn_step(
        self,
        rng: jax.Array,
        target_active: jax.Array,
        target_x: jax.Array,
        target_lane: jax.Array,
        target_type: jax.Array,
        target_sign: jax.Array,
        spawn_timer: jax.Array,
        hud_pm_sign: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        c = self.consts

        timer2 = jnp.maximum(spawn_timer - jnp.int32(1), jnp.int32(0)).astype(jnp.int32)
        eligible = (timer2 == 0)
        can_attempt = (jnp.any(eligible)) & (jnp.any(~target_active))

        def attempt_spawn(r: jax.Array):
            r1, k_lane, k_type, k_gap, k_spawnx, k_sign = jax.random.split(r, 6)

            # JIT-safe lane selection
            lane_logits = jnp.where(eligible, jnp.float32(0.0), jnp.float32(-1e9))
            lane = jax.random.categorical(k_lane, logits=lane_logits, shape=()).astype(jnp.int32)

            dir_lane = jax.lax.dynamic_index_in_dim(c.lane_dir, lane, axis=0, keepdims=False).astype(jnp.int32)

            extra = jax.random.randint(
                k_spawnx,
                (),
                jnp.int32(0),
                jnp.int32(c.spawn_extra_max_px + 1),
                dtype=jnp.int32,
            )

            spawn_x = jax.lax.cond(
                dir_lane > 0,
                lambda _: (-jnp.int32(c.target_w) - extra).astype(jnp.int32),
                lambda _: (jnp.int32(c.w) + extra).astype(jnp.int32),
                operand=None,
            )

            same_lane = target_active & (target_lane == lane)

            big_pos = jnp.int32(c.w + 999)
            big_neg = jnp.int32(-c.w - 999)

            min_x = jnp.min(jnp.where(same_lane, target_x, big_pos))
            max_x = jnp.max(jnp.where(same_lane, target_x, big_neg))

            clear_right = (~jnp.any(same_lane)) | (min_x >= (spawn_x + jnp.int32(c.min_gap_px)))
            clear_left = (~jnp.any(same_lane)) | (max_x <= (spawn_x - jnp.int32(c.min_gap_px)))
            entry_clear = jax.lax.cond(dir_lane > 0, lambda _: clear_right, lambda _: clear_left, operand=None)

            def do_spawn(_: None):
                slot = jnp.argmax((~target_active).astype(jnp.int32)).astype(jnp.int32)

                logits = jnp.log(c.type_spawn_weights)
                ttype = jax.random.categorical(k_type, logits=logits, shape=()).astype(jnp.int32)

                is_red = ttype == jnp.int32(c.pm_type_id)
                sign_bit = jax.random.bernoulli(k_sign, p=jnp.float32(0.5))
                sign = jnp.where(is_red, jnp.where(sign_bit, jnp.int32(1), jnp.int32(-1)), jnp.int32(1)).astype(jnp.int32)

                gap_px = jax.random.randint(
                    k_gap,
                    (),
                    jnp.int32(c.min_gap_px),
                    jnp.int32(c.max_gap_px + 1),
                    dtype=jnp.int32,
                )
                speed = jnp.int32(c.target_speed_px_per_step)
                delay_steps = jnp.maximum(jnp.int32(1), (gap_px + speed - 1) // speed).astype(jnp.int32)

                a2 = target_active.at[slot].set(True)
                x2 = target_x.at[slot].set(spawn_x)
                l2 = target_lane.at[slot].set(lane)
                t2 = target_type.at[slot].set(ttype)
                s2 = target_sign.at[slot].set(sign)

                mul = jax.lax.dynamic_index_in_dim(c.lane_spawn_mul, lane, axis=0, keepdims=False).astype(jnp.int32)
                timer3 = timer2.at[lane].set((delay_steps * mul).astype(jnp.int32))

                hud2 = jnp.where(is_red, sign, hud_pm_sign).astype(jnp.int32)

                return r1, a2, x2, l2, t2, s2, timer3, hud2

            def no_spawn(_: None):
                return r1, target_active, target_x, target_lane, target_type, target_sign, timer2, hud_pm_sign

            return jax.lax.cond(entry_clear, do_spawn, no_spawn, operand=None)

        def no_attempt(r: jax.Array):
            return r, target_active, target_x, target_lane, target_type, target_sign, timer2, hud_pm_sign

        return jax.lax.cond(can_attempt, attempt_spawn, no_attempt, rng)

    def _fall_step(
        self,
        rng: jax.Array,
        fall_active: jax.Array,
        fall_x: jax.Array,
        fall_y: jax.Array,
        fall_type: jax.Array,
        fall_sign: jax.Array,
        fall_timer: jax.Array,
        hud_pm_sign: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        c = self.consts

        rng2, k_cool, k_spawnx, k_type, k_sign, k_timer = jax.random.split(rng, 6)

        # Move down if active
        y2 = jnp.where(
            fall_active,
            (fall_y + jnp.int32(c.fall_speed_px_per_step)).astype(jnp.int32),
            fall_y.astype(jnp.int32),
        )

        left_screen = fall_active & (y2 >= jnp.int32(c.h))
        active2 = fall_active & (~left_screen)

        # Cooldown timer: if we just left the screen, reset to a random cooldown
        new_cool = jax.random.randint(
            k_cool,
            (),
            jnp.int32(c.fall_spawn_min_steps),
            jnp.int32(c.fall_spawn_max_steps + 1),
            dtype=jnp.int32,
        )

        timer2 = jnp.where(left_screen, new_cool, fall_timer).astype(jnp.int32)
        timer2 = jnp.where(~active2, jnp.maximum(timer2 - jnp.int32(1), jnp.int32(0)), timer2).astype(jnp.int32)

        can_spawn = (~active2) & (timer2 == 0)

        def do_spawn(_: None):
            # Spawn at top, falling down
            x = jax.random.randint(
                k_spawnx,
                (),
                jnp.int32(0),
                jnp.int32(c.w - c.target_w + 1),
                dtype=jnp.int32,
            )
            logits = jnp.log(c.type_spawn_weights)
            ttype = jax.random.categorical(k_type, logits=logits, shape=()).astype(jnp.int32)

            is_red = ttype == jnp.int32(c.pm_type_id)
            sign_bit = jax.random.bernoulli(k_sign, p=jnp.float32(0.5))
            sign = jnp.where(is_red, jnp.where(sign_bit, jnp.int32(1), jnp.int32(-1)), jnp.int32(1)).astype(jnp.int32)

            y = (-jnp.int32(c.target_h)).astype(jnp.int32)

            next_timer = jax.random.randint(
                k_timer,
                (),
                jnp.int32(c.fall_spawn_min_steps),
                jnp.int32(c.fall_spawn_max_steps + 1),
                dtype=jnp.int32,
            )

            hud2 = jnp.where(is_red, sign, hud_pm_sign).astype(jnp.int32)

            return rng2, jnp.array(True), x, y, ttype, sign, next_timer, hud2

        def no_spawn(_: None):
            return rng2, active2, fall_x, y2, fall_type, fall_sign, timer2, hud_pm_sign

        return jax.lax.cond(can_spawn, do_spawn, no_spawn, operand=None)

    def _bullet_target_collision(
        self,
        bullet_active: jax.Array,
        bullet_x: jax.Array,
        bullet_y: jax.Array,
        lane_y: jax.Array,
        target_active: jax.Array,
        target_x: jax.Array,
        target_lane: jax.Array,
        target_type: jax.Array,
        target_sign: jax.Array,
        fall_active: jax.Array,
        fall_x: jax.Array,
        fall_y: jax.Array,
        fall_type: jax.Array,
        fall_sign: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        c = self.consts

        bx0 = bullet_x
        bx1 = (bullet_x + jnp.int32(c.bullet_w)).astype(jnp.int32)
        by0 = bullet_y
        by1 = (bullet_y + jnp.int32(c.bullet_h)).astype(jnp.int32)

        # Lane targets collision
        ty = lane_y[target_lane].astype(jnp.int32)
        tx0 = target_x
        tx1 = (target_x + jnp.int32(c.target_w)).astype(jnp.int32)
        ty0 = ty
        ty1 = (ty + jnp.int32(c.target_h)).astype(jnp.int32)

        overlap_x = (bx0 < tx1) & (bx1 > tx0)
        overlap_y = (by0 < ty1) & (by1 > ty0)
        hit_mask = target_active & bullet_active & overlap_x & overlap_y

        hit_any_lane = jnp.any(hit_mask)
        hit_idx_lane = jnp.argmax(hit_mask.astype(jnp.int32)).astype(jnp.int32)
        hit_idx_lane = jnp.where(hit_any_lane, hit_idx_lane, jnp.int32(-1)).astype(jnp.int32)

        # Falling target collision
        fx0 = fall_x
        fx1 = (fall_x + jnp.int32(c.target_w)).astype(jnp.int32)
        fy0 = fall_y
        fy1 = (fall_y + jnp.int32(c.target_h)).astype(jnp.int32)

        fall_overlap_x = (bx0 < fx1) & (bx1 > fx0)
        fall_overlap_y = (by0 < fy1) & (by1 > fy0)
        hit_fall = fall_active & bullet_active & fall_overlap_x & fall_overlap_y

        # Priority: lane targets first, then falling target
        hit_any = hit_any_lane | ((~hit_any_lane) & hit_fall)
        hit_is_fall = (~hit_any_lane) & hit_fall

        # Determine hit type and sign
        ttype = jnp.where(hit_is_fall, fall_type, jnp.where(hit_any_lane, target_type[hit_idx_lane], jnp.int32(0))).astype(jnp.int32)
        sign = jnp.where(hit_is_fall, fall_sign, jnp.where(hit_any_lane, target_sign[hit_idx_lane], jnp.int32(1))).astype(jnp.int32)

        base_points = jnp.take(c.type_scores, ttype, mode="clip").astype(jnp.int32)
        is_red = ttype == jnp.int32(c.pm_type_id)
        score_delta = jnp.where(is_red, sign * jnp.int32(c.pm_bonus_value_score), base_points).astype(jnp.int32)

        reward_i32 = jnp.where(hit_any, score_delta, jnp.int32(0)).astype(jnp.int32)

        # Deactivate hit objects
        target_active2 = jax.lax.cond(
            hit_any_lane,
            lambda a: a.at[hit_idx_lane].set(False),
            lambda a: a,
            target_active,
        )
        fall_active2 = jnp.where(hit_is_fall, jnp.array(False), fall_active)

        bullet_active2 = jnp.where(hit_any, jnp.array(False), bullet_active)

        # hit_idx: lane index, or -2 for falling target, or -1 for no hit
        hit_idx = jnp.where(hit_is_fall, jnp.int32(-2), hit_idx_lane).astype(jnp.int32)

        return bullet_active2, target_active2, fall_active2, reward_i32, hit_any, hit_idx
