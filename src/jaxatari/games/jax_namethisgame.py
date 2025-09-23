"""JAX + pygame implementation of the Atari-like game *Name This Game*.

Structure
---------
- `NameThisGameConfig`: Tunable constants and UI layout.
- `NameThisGameState`: Pure, JIT-friendly game state (NamedTuple of arrays).
- `Renderer_NameThisGame`: Sprite/solid renderer producing RGBA frames with JAX.
- `JaxNameThisGame`: Environment with reset/step and object-centric observations.

Design notes
------------
- All state transitions are functional and JIT-able. Avoid Python side effects.
- Most integers are pixels; timers are in frames.
- Oxygen logic uses HUD *pixels*. `oxygen_frames_remaining` mirrors `oxy_bar_px`.
- The "REST" phase refills HUD bars and pauses hazards until the player fires.

Controls
--------
- Left/Right: `A`/`D` or arrow keys
- Fire: `Space`
- Toggle frame-by-frame: `F`
- Step when frame-by-frame is on: `N`
"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Optional, NamedTuple, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.lax
import chex

import pygame
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs
import jaxatari.spaces as spaces

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass(frozen=True)
class NameThisGameConfig:
    """All tunable parameters for *Name This Game*.

    Units & conventions
    - Positions/sizes are in pixels.
    - Timers are in frames.
    - Oxygen logic uses HUD *pixels* (not time). `oxygen_full` remains for the
      observation space upper-bound but runtime oxygen mirrors `oxy_bar_px`.

    This config is treated as static for JIT; avoid mutating or capturing large
    dynamic arrays inside it.
    """
    # Screen & scaling
    screen_width: int = 160
    screen_height: int = 250
    scaling_factor: int = 3

    # HUD bars (both centered at bottom)
    hud_bar_initial_px: int = 128            # initial width in pixels
    hud_bar_step_frames: int = 250           # shrink cadence (frames per step)
    hud_bar_shrink_px_per_step_total: int = 8  # px removed per tick (4 each side visually)
    bar_green_height: int = 4
    bar_orange_height: int = 12
    bars_gap_px: int = 0
    bars_bottom_margin_px: int = 25

    # Kraken sprite (background deco)
    kraken_x: int = 20
    kraken_y: int = 63

    # Boat (surface)
    boat_width: int = 16
    boat_speed_px: int = 1
    boat_move_every_n_frames: int = 4  # motion cadence

    # Diver (player)
    diver_width: int = 16
    diver_height: int = 13
    diver_y_floor: int = 173
    diver_speed_px: int = 1

    # Spear
    spear_width: int = 1
    spear_height: int = 1
    spear_dy: int = -3
    spear_ceiling_y: int = 63

    # Shark
    shark_lanes_y: jnp.ndarray = field(
        default_factory=lambda: jnp.array([69, 83, 97, 111, 123, 137, 151], dtype=jnp.int32)
    )
    shark_base_speed: int = 1
    shark_width: int = 15
    shark_height: int = 12
    shark_points: jnp.ndarray = field(
        default_factory=lambda: jnp.array([10, 20, 30, 40, 50, 80, 100], dtype=jnp.int32)
    )

    # Tentacles (octopus arms)
    max_tentacles: int = 8
    tentacle_base_x: jnp.ndarray = field(
        default_factory=lambda: jnp.array([16, 32, 48, 64, 80, 96, 112, 128], dtype=jnp.int32)
    )
    tentacle_ys: jnp.ndarray = field(
        default_factory=lambda: jnp.array([97, 104, 111, 118, 125, 132, 139, 146, 153, 160], dtype=jnp.int32)
    )
    tentacle_num_cols: int = 4
    tentacle_col_width: int = 4
    tentacle_square_w: int = 4
    tentacle_square_h: int = 6
    tentacle_width: int = 4  # for obs space compatibility (narrow bboxes)
    tentacle_base_growth_p: float = 0.01
    tentacle_destroy_points: int = 50

    # Oxygen line (drops from boat)
    oxygen_full: int = 1200                     # kept for obs-space high
    oxygen_pickup_radius: int = 4               # half-width tolerance
    oxygen_drop_min_interval: int = 240
    oxygen_drop_max_interval: int = 480
    oxygen_line_width: int = 1
    oxygen_y: int = 57
    oxygen_line_ttl_frames: int = 100
    oxygen_contact_every_n_frames: int = 8
    oxygen_contact_points: int = 10

    # Round progression & difficulty
    speed_progression_start_lane_for_2: int = 4
    round_clear_shark_resets: int = 3
    oxy_frames_speedup_per_round: int = 30
    oxy_min_shrink_interval: int = 20
    tentacle_growth_round_coeff: float = 0.005

    # Lives & score UI
    lives_max: int = 3
    treasure_ui_x: int = 72
    treasure_ui_y: int = 197
    max_digits_for_score: int = 9


# ------------------------------------------------------------
# Small utility types
# ------------------------------------------------------------

class EntityPosition(NamedTuple):
    """Axis-aligned bounding boxes for N entities.

    All fields are `int32` arrays of shape `(n,)`.
    `alive` is 1 or 0 (int32) for compatibility with common spaces.
    """
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    alive: jnp.ndarray


class NameThisGameState(NamedTuple):
    """Complete, immutable game state used by the environment and renderer.

    Notes
    -----
    - Booleans are stored as `jnp.bool_` for JAX; observations cast to `int32`.
    - `oxygen_frames_remaining` is the integer mirror of `oxy_bar_px`.
    - `resting=True` freezes hazards and refills bars; it ends on a fire press.
    - RNG: carry a single key (`rng`) and split inside update functions to keep
      determinism and make step order explicit.
    """
    # Score & wave
    score: chex.Array
    reward: chex.Array
    round: chex.Array
    shark_resets_this_round: chex.Array

    # HUD / timers
    oxy_bar_px: chex.Array
    wave_bar_px: chex.Array
    bar_frame_counter: chex.Array
    oxy_frame_counter: chex.Array
    resting: chex.Array

    # Boat
    boat_x: chex.Array
    boat_dx: chex.Array
    boat_move_counter: chex.Array

    # Diver
    diver_x: chex.Array
    diver_y: chex.Array
    diver_alive: chex.Array
    diver_dir: chex.Array
    fire_button_prev: chex.Array

    # Shark
    shark_x: chex.Array
    shark_y: chex.Array
    shark_dx: chex.Array
    shark_lane: chex.Array
    shark_alive: chex.Array

    # Tentacles
    tentacle_base_x: chex.Array
    tentacle_len: chex.Array
    tentacle_cols: chex.Array
    tentacle_dir: chex.Array
    tentacle_edge_wait: chex.Array
    tentacle_active: chex.Array
    tentacle_turn: chex.Array

    # Spear
    spear: chex.Array  # [x, y, dx, dy]
    spear_alive: chex.Array

    # Oxygen system (mirrors orange bar)
    oxygen_frames_remaining: chex.Array
    oxygen_line_active: chex.Array
    oxygen_line_x: chex.Array
    oxygen_drop_timer: chex.Array
    oxygen_line_ttl: chex.Array
    oxygen_contact_counter: chex.Array

    # Lives
    lives_remaining: chex.Array

    # RNG
    rng: chex.Array


class NameThisGameObservation(NamedTuple):
    """Object-centric observation returned to agents.

    Contains scalar score/round and per-entity AABBs (diver, shark, spear,
    tentacles) with narrow tentacle boxes for compatibility.
    """
    score: jnp.ndarray
    diver: EntityPosition
    shark: EntityPosition
    spear: EntityPosition
    tentacles: EntityPosition
    oxygen_frames_remaining: jnp.ndarray
    oxygen_line_active: jnp.ndarray
    oxygen_line_x: jnp.ndarray
    round_idx: jnp.ndarray


class NameThisGameInfo(NamedTuple):
    """Diagnostics and shaping hooks returned via the `info` dict.

    `all_rewards` can aggregate auxiliary reward terms if provided at init time.
    """
    score: jnp.ndarray
    round: jnp.ndarray
    shark_lane: jnp.ndarray
    shark_alive: jnp.ndarray
    spear_alive: jnp.ndarray
    tentacles_alive: jnp.ndarray
    oxygen_frames_remaining: jnp.ndarray
    oxygen_line_active: jnp.ndarray
    diver_alive: jnp.ndarray
    lives_remaining: jnp.ndarray
    all_rewards: jnp.ndarray


class NameThisGameConstants(NamedTuple):
    """Kept for interface compatibility; currently unused."""
    pass


# ------------------------------------------------------------
# Rendering
# ------------------------------------------------------------

class Renderer_NameThisGame(JAXGameRenderer):
    """Sprite-based renderer with solid-color fallbacks.

    Loads sprites from `<repo>/sprites/namethisgame`. If a sprite is missing, a
    solid RGBA block is drawn instead so rendering remains robust under JIT.
    """

    sprites: Dict[str, Any]

    def __init__(self, config: NameThisGameConfig = None):
        super().__init__()
        self.config = config or NameThisGameConfig()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/namethisgame"
        self.sprites = self._load_sprites()
        self.score_digit_sprites = self.sprites.get("score_digit_sprites")

    def _load_sprites(self) -> Dict[str, Any]:
        """Load per-object sprites from disk and return a name->array dict.

        Each sprite is a `uint8` RGBA or RGB array. Score digits are loaded as a packed
        sprite sheet and padded for fixed-width rendering.
        """
        sprites: Dict[str, Any] = {}

        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f"{name}.npy")
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)
            return None

        for name in [
            "diver",
            "shark",
            "tentacle",
            "oxygen_line",
            "background",
            "kraken",
            "boat",
            "treasure1",
            "treasure2",
            "treasure3",
        ]:
            spr = _load_sprite_frame(name)
            if spr is not None:
                sprites[name] = spr

        sprites["score_digit_sprites"] = aj.load_and_pad_digits(
            os.path.join(self.sprite_path, "{}_sprite.npy"), num_chars=10
        )
        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: NameThisGameState) -> chex.Array:
        """Render the current state to an HxWx3 uint8 RGB image.

        Order: background → HUD bars (centered shrink) → kraken → boat → diver → shark
        → spear → tentacles → oxygen line → lives → score.
        """
        cfg = self.config
        W, H = cfg.screen_width, cfg.screen_height

        def _solid_rgba(w: int, h: int, rgb: tuple[int, int, int]) -> chex.Array:
            rgb_arr = jnp.broadcast_to(jnp.array(rgb, jnp.uint8), (h, w, 3))
            a = jnp.full((h, w, 1), 255, jnp.uint8)
            return jnp.concatenate([rgb_arr, a], axis=-1)

        raster = jnp.zeros((H, W, 3), jnp.uint8)
        if "background" in self.sprites:
            raster = aj.render_at(raster, 0, 0, self.sprites["background"])

        # HUD bars -----------------------------------------------------------
        # Draw a centered horizontal bar by rendering a full-width sprite and masking
        # its right side to leave exactly `visible_px` columns; then offset so the
        # visible portion remains horizontally centered.
        def _draw_hbar(ras, visible_px: chex.Array, h_px: int, y_px: int, rgb: tuple[int, int, int]):
            max_w = cfg.hud_bar_initial_px
            w = jnp.clip(visible_px, 0, jnp.array(max_w, jnp.int32))
            cols = jnp.arange(max_w, dtype=jnp.int32)
            alpha_row = jnp.where(cols < w, 255, 0).astype(jnp.uint8)
            alpha = jnp.broadcast_to(alpha_row[None, :, None], (h_px, max_w, 1))
            spr = jnp.concatenate(
                [jnp.broadcast_to(jnp.array(rgb, jnp.uint8), (h_px, max_w, 3)), alpha],
                axis=-1,
            )
            x_left = (cfg.screen_width - w) // 2
            return aj.render_at(ras, x_left, y_px, spr)

        orange_y = cfg.screen_height - cfg.bars_bottom_margin_px - cfg.bar_orange_height
        green_y = orange_y - cfg.bars_gap_px - cfg.bar_green_height
        raster = _draw_hbar(raster, state.oxy_bar_px, cfg.bar_orange_height, orange_y, (195, 102, 52))
        raster = _draw_hbar(raster, state.wave_bar_px, cfg.bar_green_height, green_y, (27, 121, 38))

        # Kraken (deco)
        if "kraken" in self.sprites:
            raster = aj.render_at(raster, cfg.kraken_x, cfg.kraken_y, self.sprites["kraken"])

        # Boat ---------------------------------------------------------------
        # Attach the boat to the top of the oxygen line baseline and clamp to screen.
        if "boat" in self.sprites:
            boat_sprite = self.sprites["boat"]
            boat_h = int(boat_sprite.shape[0])
        else:
            boat_h = 8
            boat_sprite = _solid_rgba(cfg.boat_width, boat_h, (200, 200, 200))

        boat_y = jnp.maximum(0, cfg.oxygen_y - boat_h)
        raster = aj.render_at(
            raster,
            state.boat_x,
            boat_y,
            boat_sprite,
            flip_horizontal=(state.boat_dx < 0),
        )

        # Diver --------------------------------------------------------------
        diver_sprite = self.sprites.get("diver", _solid_rgba(cfg.diver_width, cfg.diver_height, (0, 255, 0)))
        raster = jax.lax.cond(
            state.diver_alive,
            lambda r: aj.render_at(r, state.diver_x, state.diver_y, diver_sprite, flip_horizontal=(state.diver_dir > 0)),
            lambda r: r,
            raster,
        )

        # Shark --------------------------------------------------------------
        shark_sprite = self.sprites.get("shark", _solid_rgba(cfg.shark_width, cfg.shark_height, (150, 150, 150)))
        raster = jax.lax.cond(
            state.shark_alive,
            lambda r: aj.render_at(r, state.shark_x, state.shark_y, shark_sprite, flip_horizontal=(state.shark_dx < 0)),
            lambda r: r,
            raster,
        )

        # Spear --------------------------------------------------------------
        spear_sprite = _solid_rgba(cfg.spear_width, cfg.spear_height, (255, 255, 255))
        raster = jax.lax.cond(
            state.spear_alive,
            lambda r: aj.render_at(r, state.spear[0], state.spear[1], spear_sprite),
            lambda r: r,
            raster,
        )

        # Tentacles -----------------------------------------------------------
        # Tentacles are drawn as a vertical stack of small squares; lateral position
        # comes from per-row column indices in `state.tentacle_cols[i, k]`.
        T = self.config.max_tentacles
        L = int(self.config.tentacle_ys.shape[0])
        col_w = self.config.tentacle_col_width
        sq_w = self.config.tentacle_square_w
        sq_h = self.config.tentacle_square_h
        tent_color = (0, 0, 0)
        square_rgba = _solid_rgba(sq_w, sq_h, tent_color)

        def _draw_one_tentacle(i, ras):
            length = state.tentacle_len[i]
            base_x = state.tentacle_base_x[i]

            def _draw_k(k, r2):
                def _place(rr):
                    col = state.tentacle_cols[i, k]
                    x = base_x + col * col_w
                    y = self.config.tentacle_ys[k]
                    return aj.render_at(rr, x, y, square_rgba)

                return jax.lax.cond(k < length, _place, lambda rr: rr, r2)

            return jax.lax.fori_loop(0, L, _draw_k, ras)

        raster = jax.lax.fori_loop(0, T, _draw_one_tentacle, raster)

        # Oxygen line --------------------------------------------------------
        oxy_sprite = self.sprites.get(
            "oxygen_line", _solid_rgba(cfg.oxygen_line_width, cfg.diver_y_floor, (255, 255, 255))
        )
        raster = jax.lax.cond(
            state.oxygen_line_active,
            lambda r: aj.render_at(r, state.oxygen_line_x, cfg.oxygen_y, oxy_sprite),
            lambda r: r,
            raster,
        )

        # Lives (treasure) ---------------------------------------------------
        if all(k in self.sprites for k in ("treasure1", "treasure2", "treasure3")):
            idx = jnp.clip(state.lives_remaining, 0, 3).astype(jnp.int32)

            def draw_none(r):
                return r

            def draw_t1(r):
                return aj.render_at(r, cfg.treasure_ui_x, cfg.treasure_ui_y, self.sprites["treasure1"])

            def draw_t2(r):
                return aj.render_at(r, cfg.treasure_ui_x, cfg.treasure_ui_y, self.sprites["treasure2"])

            def draw_t3(r):
                return aj.render_at(r, cfg.treasure_ui_x, cfg.treasure_ui_y, self.sprites["treasure3"])

            raster = jax.lax.switch(idx, (draw_none, draw_t1, draw_t2, draw_t3), raster)
        else:
            lives = jnp.clip(state.lives_remaining, 0, 3)
            sq = jnp.concatenate([jnp.full((6, 6, 3), 255, jnp.uint8), jnp.full((6, 6, 1), 255, jnp.uint8)], axis=-1)

            def maybe_draw(i, ras):
                def draw(rr):
                    return aj.render_at(rr, cfg.treasure_ui_x + 8 * i, cfg.treasure_ui_y, sq)

                return jax.lax.cond(lives > i, draw, lambda rr: rr, ras)

            raster = jax.lax.fori_loop(0, 3, maybe_draw, raster)

        # Score --------------------------------------------------------------
        if self.score_digit_sprites is not None:
            max_digits = cfg.max_digits_for_score
            num_digits = jnp.where(state.score > 0, jnp.ceil(jnp.log10(state.score.astype(jnp.float32) + 1.0)).astype(jnp.int32), 1)
            score_digits = aj.int_to_digits(state.score, max_digits=max_digits)
            digit_w = 8
            total_w = digit_w * num_digits
            score_x = (cfg.screen_width - total_w) // 2
            score_y = 215
            raster = aj.render_label_selective(
                raster,
                score_x,
                score_y,
                score_digits,
                self.score_digit_sprites,
                max_digits - num_digits,
                num_digits,
                spacing=digit_w,
            )

        return raster


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

class JaxNameThisGame(
    JaxEnvironment[NameThisGameState, NameThisGameObservation, NameThisGameInfo, NameThisGameConstants]
):
    """JAX implementation of *Name This Game* with object-centric observations.

    Highlights
    ---------
    - Functional, JIT-friendly state updates (no Python side effects).
    - Round-based difficulty: faster oxygen decay & shark lanes.
    - REST state: HUD refills; resume on fire press.
    - Oxygen line under the boat: contact refills bar, awards points when full.
    - Tentacles update round-robin (grow or shuffle laterally with adjacency).
    """

    def __init__(self, frameskip: int = 1, reward_funcs: list = None, config: NameThisGameConfig = None):
        super().__init__()
        self.config = config or NameThisGameConfig()
        self.frameskip = frameskip
        self.frame_stack_size = 4
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None
        self.renderer = Renderer_NameThisGame(config=self.config)
        self.action_set = [Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE]

    # -------------------------- Spaces -------------------------------------
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        cfg = self.config

        def entity_space(n: int, w_max: int, h_max: int) -> spaces.Dict:
            return spaces.Dict(
                {
                    "x": spaces.Box(low=-w_max, high=cfg.screen_width, shape=(n,), dtype=jnp.int32),
                    "y": spaces.Box(low=-h_max, high=cfg.screen_height, shape=(n,), dtype=jnp.int32),
                    "width": spaces.Box(low=0, high=w_max, shape=(n,), dtype=jnp.int32),
                    "height": spaces.Box(low=0, high=h_max, shape=(n,), dtype=jnp.int32),
                    "alive": spaces.Box(low=0, high=1, shape=(n,), dtype=jnp.int32),
                }
            )

        tentacle_h_max = int(cfg.tentacle_ys[-1] - cfg.tentacle_ys[0] + cfg.tentacle_square_h)
        return spaces.Dict(
            {
                "score": spaces.Box(low=0, high=(10 ** cfg.max_digits_for_score) - 1, shape=(), dtype=jnp.int32),
                "diver": entity_space(1, cfg.diver_width, cfg.diver_height),
                "shark": entity_space(1, cfg.shark_width, cfg.shark_height),
                "spear": entity_space(1, cfg.spear_width, cfg.spear_height),
                "tentacles": entity_space(cfg.max_tentacles, cfg.tentacle_width, tentacle_h_max),
                "oxygen_frames_remaining": spaces.Box(low=0, high=cfg.oxygen_full, shape=(), dtype=jnp.int32),
                "oxygen_line_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                "oxygen_line_x": spaces.Box(low=-cfg.oxygen_line_width, high=cfg.screen_width, shape=(), dtype=jnp.int32),
                "round_idx": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        cfg = self.config
        return spaces.Box(low=0, high=255, shape=(cfg.screen_height, cfg.screen_width, 3), dtype=jnp.uint8)

    # -------------------------- Reset --------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Tuple[NameThisGameObservation, NameThisGameState]:
        """Create an initial state and observation.

        - Starts in REST with full bars and hazards reset.
        - Shark direction is randomized; oxygen drop timer is randomized in range.
        - Returns `(obs, state)` to match the environment interface.
        """
        cfg = self.config
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])

        rng, rng_dir, rng_oxy = jax.random.split(key, 3)

        # Boat
        init_boat_x = jnp.array(cfg.screen_width // 2 - cfg.boat_width // 2, jnp.int32)
        init_boat_dx = jnp.array(1, jnp.int32)
        init_boat_counter = jnp.array(0, jnp.int32)

        # Diver
        init_diver_x = jnp.array(cfg.screen_width // 2, jnp.int32)
        init_diver_y = jnp.array(cfg.diver_y_floor, jnp.int32)

        # Shark
        go_left = jax.random.bernoulli(rng_dir)
        init_shark_lane = jnp.array(0, jnp.int32)
        init_shark_y = cfg.shark_lanes_y[init_shark_lane]
        init_shark_x = jnp.where(go_left, -cfg.shark_width, cfg.screen_width)
        init_shark_speed = self._shark_speed_for_lane(jnp.array(0, jnp.int32), init_shark_lane)
        init_shark_dx = jnp.where(go_left, init_shark_speed, -init_shark_speed).astype(jnp.int32)

        # Tentacles start empty
        tentacle_len = jnp.zeros((T,), jnp.int32)
        tentacle_cols = jnp.zeros((T, L), jnp.int32)
        tentacle_dir = jnp.ones((T,), jnp.int32)
        tentacle_edge_wait = jnp.zeros((T,), jnp.int32)
        tentacle_active = (tentacle_len > 0)

        # Spear
        empty_spear = jnp.array([0, 0, 0, 0], jnp.int32)

        # Oxygen (start REST with full bars in *pixels*)
        init_oxygen_bar = jnp.array(cfg.hud_bar_initial_px, jnp.int32)
        min_int = int(cfg.oxygen_drop_min_interval)
        max_int = int(cfg.oxygen_drop_max_interval)
        init_drop_timer = jax.random.randint(rng_oxy, (), min_int, max_int + 1, dtype=jnp.int32)

        state = NameThisGameState(
            score=jnp.array(0, jnp.int32),
            reward=jnp.array(0.0, jnp.float32),
            round=jnp.array(0, jnp.int32),
            shark_resets_this_round=jnp.array(0, jnp.int32),

            oxy_bar_px=init_oxygen_bar,
            wave_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            bar_frame_counter=jnp.array(0, jnp.int32),
            oxy_frame_counter=jnp.array(0, jnp.int32),
            resting=jnp.array(True, jnp.bool_),

            boat_x=init_boat_x,
            boat_dx=init_boat_dx,
            boat_move_counter=init_boat_counter,

            diver_x=init_diver_x,
            diver_y=init_diver_y,
            diver_alive=jnp.array(True, jnp.bool_),
            diver_dir=jnp.array(1, jnp.int32),
            fire_button_prev=jnp.array(False, jnp.bool_),

            shark_x=init_shark_x.astype(jnp.int32),
            shark_y=init_shark_y.astype(jnp.int32),
            shark_dx=init_shark_dx,
            shark_lane=init_shark_lane,
            shark_alive=jnp.array(True, jnp.bool_),

            tentacle_base_x=cfg.tentacle_base_x,
            tentacle_len=tentacle_len,
            tentacle_cols=tentacle_cols,
            tentacle_dir=tentacle_dir,
            tentacle_edge_wait=tentacle_edge_wait,
            tentacle_active=tentacle_active,
            tentacle_turn=jnp.array(0, jnp.int32),

            spear=empty_spear,
            spear_alive=jnp.array(False, jnp.bool_),

            oxygen_frames_remaining=init_oxygen_bar,  # mirror orange bar
            oxygen_line_active=jnp.array(False, jnp.bool_),
            oxygen_line_x=jnp.array(-1, jnp.int32),
            oxygen_drop_timer=init_drop_timer,
            oxygen_line_ttl=jnp.array(0, jnp.int32),
            oxygen_contact_counter=jnp.array(0, jnp.int32),

            lives_remaining=jnp.array(cfg.lives_max, jnp.int32),

            rng=rng,
        )
        obs = self._get_observation(state)
        return obs, state

    # -------------------------- Obs / Info ---------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: EnvObs) -> jnp.ndarray:
        """Flatten a structured observation into a single int32 vector.

        Useful for simple agents or logging. Field order matches the concatenation
        order below.
        """
        def _flat(ep: EntityPosition) -> jnp.ndarray:
            return jnp.concatenate(
                [
                    jnp.ravel(ep.x).astype(jnp.int32),
                    jnp.ravel(ep.y).astype(jnp.int32),
                    jnp.ravel(ep.width).astype(jnp.int32),
                    jnp.ravel(ep.height).astype(jnp.int32),
                    jnp.ravel(ep.alive).astype(jnp.int32),
                ],
                axis=0,
            )

        return jnp.concatenate(
            [
                jnp.atleast_1d(obs.score).astype(jnp.int32),
                _flat(obs.diver),
                _flat(obs.shark),
                _flat(obs.spear),
                _flat(obs.tentacles),
                jnp.atleast_1d(obs.oxygen_frames_remaining).astype(jnp.int32),
                jnp.atleast_1d(obs.oxygen_line_active).astype(jnp.int32),
                jnp.atleast_1d(obs.oxygen_line_x).astype(jnp.int32),
                jnp.atleast_1d(obs.round_idx).astype(jnp.int32),
            ],
            axis=0,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: NameThisGameState) -> NameThisGameObservation:
        """Build the object-centric observation from the current state.

        Tentacle boxes are narrow one-column rectangles that span from the top segment
        to the current tip, for compatibility with older consumers.
        """

        cfg = self.config
        diver_pos = EntityPosition(
            x=jnp.atleast_1d(state.diver_x),
            y=jnp.atleast_1d(state.diver_y),
            width=jnp.atleast_1d(jnp.array(cfg.diver_width, jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.diver_height, jnp.int32)),
            alive=jnp.atleast_1d(state.diver_alive.astype(jnp.int32)),
        )
        shark_pos = EntityPosition(
            x=jnp.atleast_1d(state.shark_x),
            y=jnp.atleast_1d(state.shark_y),
            width=jnp.atleast_1d(jnp.array(cfg.shark_width, jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.shark_height, jnp.int32)),
            alive=jnp.atleast_1d(state.shark_alive.astype(jnp.int32)),
        )
        spear_pos = EntityPosition(
            x=jnp.atleast_1d(state.spear[0]),
            y=jnp.atleast_1d(state.spear[1]),
            width=jnp.atleast_1d(jnp.array(cfg.spear_width, jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.spear_height, jnp.int32)),
            alive=jnp.atleast_1d(state.spear_alive.astype(jnp.int32)),
        )

        # Tentacle bboxes (narrow, one-column wide for obs compatibility)
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])
        len_vec = state.tentacle_len
        alive_vec = (len_vec > 0).astype(jnp.int32)
        top_cols = jnp.where(len_vec > 0, state.tentacle_cols[:, 0], jnp.array(0, jnp.int32))
        tentacle_left_x = jnp.where(
            len_vec > 0, state.tentacle_base_x + top_cols * cfg.tentacle_col_width, jnp.array(-1, jnp.int32)
        )
        tentacle_y_top = jnp.where(len_vec > 0, jnp.full((T,), cfg.tentacle_ys[0], jnp.int32), jnp.array(-1, jnp.int32))
        last_y = jnp.take_along_axis(
            cfg.tentacle_ys[None, :].repeat(T, axis=0), jnp.clip((len_vec - 1)[:, None], 0, L - 1), axis=1
        ).squeeze(1)
        tentacle_height_px = jnp.where(len_vec > 0, last_y - cfg.tentacle_ys[0] + cfg.tentacle_square_h, jnp.array(0, jnp.int32))
        tentacle_width_px = jnp.where(len_vec > 0, jnp.full((T,), cfg.tentacle_width, jnp.int32), jnp.array(0, jnp.int32))

        tentacle_pos = EntityPosition(
            x=tentacle_left_x,
            y=tentacle_y_top,
            width=tentacle_width_px,
            height=tentacle_height_px,
            alive=alive_vec,
        )

        return NameThisGameObservation(
            score=state.score,
            diver=diver_pos,
            shark=shark_pos,
            spear=spear_pos,
            tentacles=tentacle_pos,
            oxygen_frames_remaining=state.oxygen_frames_remaining,
            oxygen_line_active=state.oxygen_line_active.astype(jnp.int32),
            oxygen_line_x=state.oxygen_line_x,
            round_idx=state.round,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: NameThisGameState, all_rewards: chex.Array = None) -> NameThisGameInfo:
        """Assemble auxiliary info (diagnostics and optional shaping rewards)."""

        return NameThisGameInfo(
            score=state.score,
            round=state.round,
            shark_lane=state.shark_lane,
            shark_alive=state.shark_alive.astype(jnp.int32),
            spear_alive=state.spear_alive.astype(jnp.int32),
            tentacles_alive=state.tentacle_active.astype(jnp.int32),
            oxygen_frames_remaining=state.oxygen_frames_remaining,
            oxygen_line_active=state.oxygen_line_active.astype(jnp.int32),
            diver_alive=state.diver_alive.astype(jnp.int32),
            lives_remaining=state.lives_remaining,
            all_rewards=(all_rewards if all_rewards is not None else jnp.zeros(1, jnp.float32)),
        )

    # -------------------------- Core helpers -------------------------------

    @staticmethod
    @jax.jit
    def _rects_overlap(l1, r1, t1, b1, l2, r2, t2, b2):
        return (l1 < r2) & (r1 > l2) & (t1 < b2) & (b1 > t2)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, prev_state: NameThisGameState, state: NameThisGameState) -> chex.Array:
        """Compute auxiliary reward components.

        Notes
        -----
        `self.reward_funcs` must be a static tuple of callables for JIT stability.
        Each callable has signature `(prev_state, state) -> float`.
        """

        if self.reward_funcs is None:
            return jnp.zeros((1,), jnp.float32)
        # Python-side list comp is fine under jit if reward_funcs is static
        rewards = jnp.array([func(prev_state, state) for func in self.reward_funcs], jnp.float32)
        return rewards

    # -------------------------- Update systems -----------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _update_bars_and_rest(self, state: NameThisGameState, just_pressed: chex.Array) -> NameThisGameState:
        """Update wave/oxygen bars and handle REST transitions.

        - While active: both bars tick down on separate cadences.
        - When the wave bar hits zero: enter REST, increment `round`, clear hazards,
          refill bars, and reset certain shark fields.
        - REST ends on a new fire press.
        """

        cfg = self.config

        # Exit REST on fire
        def _exit_rest(s: NameThisGameState) -> NameThisGameState:
            return s._replace(
                resting=jnp.array(False, jnp.bool_),
                wave_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
                bar_frame_counter=jnp.array(0, jnp.int32),
            )

        state = jax.lax.cond(state.resting & just_pressed, _exit_rest, lambda q: q, state)

        # Freeze during REST
        def _if_rest(s: NameThisGameState) -> NameThisGameState:
            return s

        def _if_active(s: NameThisGameState) -> NameThisGameState:
            wave_cnt = s.bar_frame_counter + 1
            wave_tick = wave_cnt >= jnp.array(cfg.hud_bar_step_frames, jnp.int32)

            # Oxygen bar shrinks faster each round, but never below `oxy_min_shrink_interval`.
            r_eff = jnp.maximum(s.round, jnp.array(1, jnp.int32))
            oxy_interval = jnp.array(cfg.hud_bar_step_frames, jnp.int32) - jnp.array(cfg.oxy_frames_speedup_per_round, jnp.int32) * (r_eff - 1)
            oxy_interval = jnp.maximum(oxy_interval, jnp.array(cfg.oxy_min_shrink_interval, jnp.int32))
            oxy_cnt = s.oxy_frame_counter + 1
            oxy_tick = oxy_cnt >= oxy_interval

            dec = jnp.array(cfg.hud_bar_shrink_px_per_step_total, jnp.int32)
            new_wave = jnp.maximum(s.wave_bar_px - jnp.where(wave_tick, dec, 0), 0)
            new_oxy = jnp.maximum(s.oxy_bar_px - jnp.where(oxy_tick, dec, 0), 0)

            s2 = s._replace(
                wave_bar_px=new_wave,
                oxy_bar_px=new_oxy,
                bar_frame_counter=jnp.where(wave_tick, jnp.array(0, jnp.int32), wave_cnt),
                oxy_frame_counter=jnp.where(oxy_tick, jnp.array(0, jnp.int32), oxy_cnt),
            )

            # Wave bar empty -> enter REST, increment round, clear hazards
            def _enter_rest(st: NameThisGameState) -> NameThisGameState:
                zeros_T = jnp.zeros_like(st.tentacle_len)
                return st._replace(
                    resting=jnp.array(True, jnp.bool_),
                    round=st.round + jnp.array(1, jnp.int32),
                    tentacle_len=zeros_T,
                    tentacle_active=zeros_T.astype(jnp.bool_),
                    oxygen_line_active=jnp.array(False, jnp.bool_),
                    oxygen_line_x=jnp.array(-1, jnp.int32),
                    oxygen_line_ttl=jnp.array(0, jnp.int32),
                    oxy_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
                    oxygen_frames_remaining=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
                    shark_lane=jnp.array(0, jnp.int32),
                    shark_y=cfg.shark_lanes_y[0],
                    shark_alive=jnp.array(True, jnp.bool_),
                )

            return jax.lax.cond(s2.wave_bar_px <= 0, _enter_rest, lambda z: z, s2)

        return jax.lax.cond(state.resting, _if_rest, _if_active, state)

    @partial(jax.jit, static_argnums=(0,))
    def _interpret_action(self, state: NameThisGameState, action: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Decode discrete action into `(move_dir, fire_pressed)`.

        `move_dir` ∈ {-1, 0, +1}. `fire_pressed` is a boolean.
        """

        move_dir = jnp.where((action == Action.LEFT) | (action == Action.LEFTFIRE), -1, jnp.where((action == Action.RIGHT) | (action == Action.RIGHTFIRE), 1, 0))
        fire_pressed = (action == Action.FIRE) | (action == Action.LEFTFIRE) | (action == Action.RIGHTFIRE)
        return move_dir, fire_pressed

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_spear(self, state: NameThisGameState) -> NameThisGameState:
        """Spawn a spear from the diver tip if no spear is currently alive.

        The spear travels vertically with fixed dy.
        """

        cfg = self.config

        def _spawn(s):
            spawn_x = s.diver_x + (cfg.diver_width // 2)
            spawn_y = s.diver_y
            new = jnp.array([spawn_x, spawn_y, 0, cfg.spear_dy], jnp.int32)
            return s._replace(spear=new, spear_alive=jnp.array(True, jnp.bool_))

        return jax.lax.cond(~state.spear_alive, _spawn, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def _move_diver(self, state: NameThisGameState, move_dir: chex.Array) -> NameThisGameState:
        """Move the diver horizontally, clamp to screen, and update facing."""

        cfg = self.config
        new_x = jnp.clip(state.diver_x + move_dir * cfg.diver_speed_px, 0, cfg.screen_width - cfg.diver_width)
        new_dir = jnp.where(move_dir != 0, move_dir.astype(jnp.int32), state.diver_dir)
        return state._replace(diver_x=new_x, diver_dir=new_dir)

    @partial(jax.jit, static_argnums=(0,))
    def _move_spear(self, state: NameThisGameState) -> NameThisGameState:
        """Advance the spear by its velocity; despawn when it crosses the ceiling."""
        cfg = self.config

        def _step(s):
            # Compute next position (x, y) and write it back.
            pos_xy = s.spear[:2] + s.spear[2:4]
            s2 = s._replace(spear=s.spear.at[:2].set(pos_xy))
            x, y = pos_xy[0], pos_xy[1]

            # Alive if on-screen and still below the ceiling line.
            in_x = (x >= 0) & (x < cfg.screen_width)
            in_y = (y >= 0) & (y < cfg.screen_height)
            below_ceiling = y > cfg.spear_ceiling_y   # despawn at/above the ceiling (<= becomes dead)
            alive_next = jnp.logical_and(s.spear_alive, in_x & in_y & below_ceiling)

            return s2._replace(spear_alive=alive_next)

        return jax.lax.cond(state.spear_alive, _step, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def _move_shark(self, state: NameThisGameState) -> NameThisGameState:
        """Move shark horizontally; when fully off-screen, drop one lane.

        - In REST: shark paces and bounces within the screen.
        - Active: moving off the left/right edge respawns at the opposite side one
          lane lower; when no more lanes remain, the shark is marked not alive.
        """

        cfg = self.config
        x_next = state.shark_x + state.shark_dx

        def _resting_move(s: NameThisGameState) -> NameThisGameState:
            hit_left = x_next <= 0
            hit_right = (x_next + cfg.shark_width) >= cfg.screen_width
            hit_edge = hit_left | hit_right
            new_dx = jnp.where(hit_edge, -s.shark_dx, s.shark_dx)
            clamped_x = jnp.where(hit_left, 0, jnp.where(hit_right, cfg.screen_width - cfg.shark_width, x_next))
            return s._replace(shark_x=clamped_x, shark_dx=new_dx, shark_lane=jnp.array(0, jnp.int32), shark_y=cfg.shark_lanes_y[0], shark_alive=jnp.array(True, jnp.bool_))

        def _normal_move(s: NameThisGameState) -> NameThisGameState:
            new_x = x_next
            dx = s.shark_dx
            # When the shark fully exits the screen, respawn one lane lower from the opposite side.
            off_right = (dx > 0) & (new_x >= cfg.screen_width)
            off_left = (dx < 0) & ((new_x + cfg.shark_width) <= 0)

            def _drop_lane(st: NameThisGameState, going_left: bool) -> NameThisGameState:
                new_lane = st.shark_lane + 1
                last_idx = cfg.shark_lanes_y.shape[0] - 1

                def _lane_exists(tt: NameThisGameState) -> NameThisGameState:
                    safe_idx = jnp.clip(new_lane, 0, last_idx)
                    new_y = jnp.take(cfg.shark_lanes_y, safe_idx, mode="clip")
                    speed_abs = self._shark_speed_for_lane(tt.round, safe_idx)
                    new_dx_val = jnp.where(going_left, speed_abs, -speed_abs)
                    new_x_val = jnp.where(going_left, -cfg.shark_width, cfg.screen_width)
                    return tt._replace(shark_x=new_x_val, shark_y=new_y, shark_dx=new_dx_val, shark_lane=new_lane)

                def _no_lane(tt: NameThisGameState) -> NameThisGameState:
                    return tt._replace(shark_alive=jnp.array(False, jnp.bool_))

                has_lane = new_lane < (last_idx + 1)
                return jax.lax.cond(has_lane, _lane_exists, _no_lane, st)

            st = s._replace(shark_x=new_x)
            st = jax.lax.cond(off_right, lambda u: _drop_lane(u, going_left=False), lambda u: u, st)
            st = jax.lax.cond(off_left, lambda u: _drop_lane(u, going_left=True), lambda u: u, st)
            return st

        return jax.lax.cond(state.resting, _resting_move, _normal_move, state)

    @partial(jax.jit, static_argnums=(0,))
    def _update_one_tentacle(self, state: NameThisGameState) -> NameThisGameState:
        """Update exactly one tentacle per frame in round-robin order.

        Two behaviors:
        - GROW: Increase length by one, shifting previous columns downward.
        - MOVE: Slide laterally by `dir` unless blocked; if the tip is blocked and the
          body is stacked, wait one tick then flip direction and move.

        `tentacle_edge_wait` implements the one-tick hesitation at edges.
        """

        cfg = self.config
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])
        max_col = cfg.tentacle_num_cols - 1

        def _no_update(s: NameThisGameState) -> NameThisGameState:
            return s

        def _active_update(s0: NameThisGameState) -> NameThisGameState:
            i = s0.tentacle_turn % T
            rng_choice, rng_after = jax.random.split(s0.rng)

            r_float = s0.round.astype(jnp.float32)
            p_grow = jnp.clip(cfg.tentacle_base_growth_p + cfg.tentacle_growth_round_coeff * r_float, 0.0, 0.95)
            do_grow = jax.random.bernoulli(rng_choice, p_grow)

            def row(a):
                return a[i]

            def set_row(a, v):
                return a.at[i].set(v)

            def set_row2d(a, v):
                return a.at[i, :].set(v)

            cols_i = row(s0.tentacle_cols)
            len_i = row(s0.tentacle_len)
            dir_i = row(s0.tentacle_dir)
            wait_i = row(s0.tentacle_edge_wait)

            # -------------------- GROW --------------------
            def _grow(s: NameThisGameState) -> NameThisGameState:
                l = len_i
                cols = cols_i

                def _when_empty():
                    start_col = jnp.array(1, jnp.int32)
                    new_cols = cols.at[:].set(0)
                    new_cols = new_cols.at[0].set(start_col)
                    new_len = jnp.minimum(l + 1, L)
                    return new_cols, new_len

                def _when_non_empty():
                    prev_top = cols[0]
                    idx = jnp.arange(L, dtype=jnp.int32)
                    prevs = jnp.concatenate([jnp.array([prev_top], jnp.int32), cols[:-1]])
                    l_clamped = jnp.minimum(l, jnp.array(L - 1, jnp.int32))
                    mask = idx <= l_clamped
                    new_cols = jnp.where(mask, prevs, cols)
                    new_len = jnp.minimum(l + 1, L)
                    return new_cols, new_len

                new_cols, new_len = jax.lax.cond(len_i == 0, _when_empty, _when_non_empty)
                s = s._replace(
                    tentacle_cols=set_row2d(s0.tentacle_cols, new_cols),
                    tentacle_len=set_row(s0.tentacle_len, new_len),
                    tentacle_active=set_row(s0.tentacle_active, new_len > 0),
                )
                return s

            # -------------------- MOVE --------------------
            def _move(s: NameThisGameState) -> NameThisGameState:
                cols = cols_i
                l = len_i
                d = dir_i
                w = wait_i

                def _nothing(st):
                    return st

                def _do_move(st: NameThisGameState) -> NameThisGameState:
                    at_left = (cols[0] == 0)
                    at_right = (cols[0] == max_col)
                    blocked = jnp.where(d < 0, at_left, at_right)
                    stacked = jnp.where(l <= 1, True, cols[1] == cols[0])

                    def _perform_move(cur_cols, direction):
                        tip_new = jnp.clip(cur_cols[0] + direction, 0, max_col)

                        def body(k, acc):
                            def first(a):
                                # set new tip
                                return a.at[0].set(tip_new)

                            def rest(a):
                                return a.at[k].set(cur_cols[k - 1])

                            return jax.lax.cond(k == 0, first, rest, acc)

                        return jax.lax.fori_loop(0, L,
                                                 lambda k, a: jax.lax.cond(k < l, lambda x: body(k, x), lambda x: x, a),
                                                 cur_cols)

                    def _edge_logic(st2: NameThisGameState) -> NameThisGameState:
                        def _first_wait():
                            return st2._replace(tentacle_edge_wait=set_row(s0.tentacle_edge_wait, jnp.array(1, jnp.int32)))

                        def _flip_and_move():
                            new_dir = -d
                            moved = _perform_move(cols, new_dir)
                            return st2._replace(
                                tentacle_dir=set_row(s0.tentacle_dir, new_dir),
                                tentacle_cols=set_row2d(s0.tentacle_cols, moved),
                                tentacle_edge_wait=set_row(s0.tentacle_edge_wait, jnp.array(0, jnp.int32)),
                            )

                        return jax.lax.cond((w == 0), _first_wait, _flip_and_move)

                    def _normal_move(st2: NameThisGameState) -> NameThisGameState:
                        moved = _perform_move(cols, d)
                        return st2._replace(
                            tentacle_cols=set_row2d(s0.tentacle_cols, moved),
                            tentacle_edge_wait=set_row(s0.tentacle_edge_wait, jnp.array(0, jnp.int32)),
                        )

                    return jax.lax.cond((blocked & stacked), _edge_logic, _normal_move, st)

                return jax.lax.cond(l == 0, _nothing, _do_move, s)

            s1 = jax.lax.cond(do_grow, _grow, _move, s0)

            next_turn = (s1.tentacle_turn + 1) % T
            s1 = s1._replace(tentacle_turn=next_turn, rng=rng_after)
            s1 = s1._replace(tentacle_active=(s1.tentacle_len > 0))
            return s1

        return jax.lax.cond(state.resting, _no_update, _active_update, state)

    @partial(jax.jit, static_argnums=(0,))
    def _check_spear_shark_collision(self, state: NameThisGameState) -> NameThisGameState:
        """Award points and reset the shark when hit by the spear.

        Points depend on the current lane. The shark respawns at lane 0 heading in a
        random horizontal direction; the spear is consumed.
        """

        cfg = self.config
        rng_side, rng_after = jax.random.split(state.rng)

        spear_l = state.spear[0]
        spear_r = state.spear[0] + cfg.spear_width
        spear_t = state.spear[1]
        spear_b = state.spear[1] + cfg.spear_height

        shark_l = state.shark_x
        shark_r = state.shark_x + cfg.shark_width
        shark_t = state.shark_y
        shark_b = state.shark_y + cfg.shark_height

        hit = state.spear_alive & state.shark_alive & self._rects_overlap(
            spear_l, spear_r, spear_t, spear_b, shark_l, shark_r, shark_t, shark_b
        )

        def _on_hit(s):
            lane_points = cfg.shark_points
            idx = jnp.clip(s.shark_lane, 0, lane_points.shape[0] - 1)
            points = jnp.take(lane_points, idx, mode="clip")
            go_left = jax.random.bernoulli(rng_side)
            reset_lane = jnp.array(0, jnp.int32)
            reset_y = cfg.shark_lanes_y[reset_lane]
            reset_speed = self._shark_speed_for_lane(s.round, reset_lane)
            reset_dx = jnp.where(go_left, reset_speed, -reset_speed).astype(jnp.int32)
            reset_x = jnp.where(go_left, -cfg.shark_width, cfg.screen_width)
            return s._replace(
                score=s.score + points,
                shark_x=reset_x.astype(jnp.int32),
                shark_y=jnp.array(reset_y, jnp.int32),
                shark_dx=reset_dx,
                shark_lane=reset_lane,
                shark_alive=jnp.array(True, jnp.bool_),
                shark_resets_this_round=s.shark_resets_this_round + 1,
                spear_alive=jnp.array(False, jnp.bool_),
                rng=rng_after,
            )

        return jax.lax.cond(hit, _on_hit, lambda s: s._replace(rng=rng_after), state)

    @partial(jax.jit, static_argnums=(0,))
    def _check_spear_tentacle_collision(self, state: NameThisGameState) -> NameThisGameState:
        """Destroy tentacle tips hit by the spear and award points.

        If any tentacle is hit this frame, the spear is consumed; otherwise it continues
        moving. Multiple tips can be hit simultaneously.
        """

        cfg = self.config
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])

        sl = state.spear[0]
        sr = state.spear[0] + cfg.spear_width
        st = state.spear[1]
        sb = state.spear[1] + cfg.spear_height

        lens = state.tentacle_len
        has_tip = lens > 0
        tip_idx = jnp.maximum(lens - 1, 0)
        tip_cols = jnp.take_along_axis(state.tentacle_cols, tip_idx[:, None], axis=1).squeeze(1)

        x_lefts = state.tentacle_base_x + tip_cols * cfg.tentacle_col_width
        x_rights = x_lefts + cfg.tentacle_square_w
        y_tops = jnp.take(cfg.tentacle_ys, tip_idx)
        y_bottoms = y_tops + cfg.tentacle_square_h

        over_x = (sl < x_rights) & (sr > x_lefts)
        over_y = (st < y_bottoms) & (sb > y_tops)
        hits = state.spear_alive & has_tip & over_x & over_y

        num_hits = jnp.sum(hits.astype(jnp.int32))
        gained = num_hits * jnp.array(cfg.tentacle_destroy_points, jnp.int32)

        new_len = jnp.where(hits, jnp.array(0, jnp.int32), state.tentacle_len)
        new_dir = jnp.where(hits, jnp.array(1, jnp.int32), state.tentacle_dir)
        new_wait = jnp.where(hits, jnp.array(0, jnp.int32), state.tentacle_edge_wait)
        spear_alive_next = jnp.logical_and(state.spear_alive, jnp.logical_not(jnp.any(hits)))

        return state._replace(
            score=state.score + gained,
            tentacle_len=new_len,
            tentacle_cols=state.tentacle_cols,
            tentacle_dir=new_dir,
            tentacle_edge_wait=new_wait,
            tentacle_active=(new_len > 0),
            spear_alive=spear_alive_next,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_diver_hazard(self, state: NameThisGameState) -> NameThisGameState:
        """Kill the diver if any tentacle has reached the bottom row."""

        L = int(self.config.tentacle_ys.shape[0])
        reached = jnp.any(state.tentacle_len >= L)
        return state._replace(diver_alive=jnp.where(reached, jnp.array(False, jnp.bool_), state.diver_alive))

    @partial(jax.jit, static_argnums=(0,))
    def _move_boat(self, state: NameThisGameState) -> NameThisGameState:
        """Move the boat with a fixed cadence and bounce at screen edges."""

        cfg = self.config
        next_counter = state.boat_move_counter + 1
        move_now = (state.boat_move_counter % cfg.boat_move_every_n_frames) == 0

        def _do_move(s: NameThisGameState) -> NameThisGameState:
            new_x = s.boat_x + s.boat_dx * cfg.boat_speed_px
            hit_left = new_x <= 0
            hit_right = (new_x + cfg.boat_width) >= cfg.screen_width
            hit_edge = hit_left | hit_right
            new_dx = jnp.where(hit_edge, -s.boat_dx, s.boat_dx)
            clamped_x = jnp.where(hit_left, 0, jnp.where(hit_right, cfg.screen_width - cfg.boat_width, new_x))
            return s._replace(boat_x=clamped_x, boat_dx=new_dx, boat_move_counter=next_counter)

        # if not time to move, only update the counter
        return jax.lax.cond(move_now, _do_move, lambda s: s._replace(boat_move_counter=next_counter), state)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_or_update_oxygen_line(self, state: NameThisGameState) -> NameThisGameState:
        """Handle oxygen line spawning, following the boat while active, and expiry.

        - When inactive: a countdown spawns a new line centered under the boat.
        - While active: the line follows the boat and decrements a TTL.
        - On expiry: disable the line and schedule the next drop with a random delay.
        """

        cfg = self.config

        def _rest(s: NameThisGameState) -> NameThisGameState:
            # Frozen in REST; line already disabled on entering REST.
            return s

        def _active(s: NameThisGameState) -> NameThisGameState:
            rng_line, rng_interval, rng_after = jax.random.split(s.rng, 3)

            def _no_line(st: NameThisGameState) -> NameThisGameState:
                new_timer = jnp.maximum(st.oxygen_drop_timer - 1, 0)

                def _spawn_line(ss: NameThisGameState) -> NameThisGameState:
                    new_x = st.boat_x + (cfg.boat_width - cfg.oxygen_line_width) // 2
                    new_x = jnp.clip(new_x, 0, cfg.screen_width - cfg.oxygen_line_width).astype(jnp.int32)
                    return ss._replace(
                        oxygen_line_active=jnp.array(True, jnp.bool_),
                        oxygen_line_x=new_x,
                        oxygen_drop_timer=jnp.array(0, jnp.int32),
                        oxygen_line_ttl=jnp.array(cfg.oxygen_line_ttl_frames, jnp.int32),
                        oxygen_contact_counter=jnp.array(0, jnp.int32),
                    )

                def _no_spawn(ss: NameThisGameState) -> NameThisGameState:
                    return ss._replace(oxygen_drop_timer=new_timer)

                return jax.lax.cond(new_timer <= 0, _spawn_line, _no_spawn, st)

            def _line_active(st: NameThisGameState) -> NameThisGameState:
                new_x = st.boat_x + (cfg.boat_width - cfg.oxygen_line_width) // 2
                new_x = jnp.clip(new_x, 0, cfg.screen_width - cfg.oxygen_line_width).astype(jnp.int32)
                new_ttl = jnp.maximum(st.oxygen_line_ttl - 1, 0)

                def _expire(ss: NameThisGameState) -> NameThisGameState:
                    next_timer = jax.random.randint(
                        rng_interval, (), cfg.oxygen_drop_min_interval, cfg.oxygen_drop_max_interval + 1, dtype=jnp.int32
                    )
                    return ss._replace(
                        oxygen_line_active=jnp.array(False, jnp.bool_),
                        oxygen_line_x=jnp.array(-1, jnp.int32),
                        oxygen_drop_timer=next_timer,
                        oxygen_line_ttl=jnp.array(0, jnp.int32),
                        oxygen_contact_counter=jnp.array(0, jnp.int32),
                    )

                def _still(ss: NameThisGameState) -> NameThisGameState:
                    return ss._replace(oxygen_line_x=new_x, oxygen_line_ttl=new_ttl)

                return jax.lax.cond(new_ttl <= 0, _expire, _still, st)

            out = jax.lax.cond(s.oxygen_line_active, _line_active, _no_line, s)
            return out._replace(rng=rng_after)

        return jax.lax.cond(state.resting, _rest, _active, state)

    @partial(jax.jit, static_argnums=(0,))
    def _update_oxygen(self, state: NameThisGameState) -> NameThisGameState:
        """Drain/refill the orange oxygen bar and drip points when full.

        - When the diver is horizontally under the line, every K frames we either:
          * refill the bar by a fixed pixel amount until full, OR
          * if already full, award small points at the same cadence.
        - `oxygen_frames_remaining` mirrors `oxy_bar_px` for integer-safe logic.
        """
        cfg = self.config

        def _rest(s: NameThisGameState) -> NameThisGameState:
            # keep the integer mirror coherent
            return s._replace(oxygen_frames_remaining=s.oxy_bar_px)

        def _active(s: NameThisGameState) -> NameThisGameState:
            diver_center = s.diver_x + (cfg.diver_width // 2)
            line_center = s.oxygen_line_x + (cfg.oxygen_line_width // 2)
            under_line = s.oxygen_line_active & (jnp.abs(diver_center - line_center) <= cfg.oxygen_pickup_radius)

            cnt_next = jnp.where(under_line, s.oxygen_contact_counter + 1, jnp.array(0, jnp.int32))
            # Only count contact every K frames to create a steady refill/drip cadence.
            k = jnp.array(cfg.oxygen_contact_every_n_frames, jnp.int32)
            tick = under_line & (cnt_next % k == 0) & (cnt_next > 0)

            max_px = jnp.array(cfg.hud_bar_initial_px, jnp.int32)
            full_before = s.oxy_bar_px >= max_px

            inc_bar = jnp.where(tick & (~full_before), jnp.array(cfg.hud_bar_shrink_px_per_step_total, jnp.int32), 0)
            new_bar = jnp.minimum(s.oxy_bar_px + inc_bar, max_px)

            add_points = jnp.where(tick & full_before, jnp.array(cfg.oxygen_contact_points, jnp.int32), 0)
            new_score = s.score + add_points

            return s._replace(
                oxy_bar_px=new_bar,
                oxygen_contact_counter=cnt_next,
                oxygen_frames_remaining=new_bar,
                score=new_score,
            )

        return jax.lax.cond(state.resting, _rest, _active, state)

    @partial(jax.jit, static_argnums=(0,))
    def _life_loss_reset(self, state: NameThisGameState) -> NameThisGameState:
        """Consume one life and soft-reset into REST (score/round are preserved)."""
        cfg = self.config
        rng1, rng2 = jax.random.split(state.rng)
        next_timer = jax.random.randint(
            rng2, (), cfg.oxygen_drop_min_interval, cfg.oxygen_drop_max_interval + 1, dtype=jnp.int32
        )
        zeros_T = jnp.zeros_like(state.tentacle_len)

        return state._replace(
            resting=jnp.array(True, jnp.bool_),
            wave_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            oxy_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            bar_frame_counter=jnp.array(0, jnp.int32),
            oxy_frame_counter=jnp.array(0, jnp.int32),

            diver_x=jnp.array(cfg.screen_width // 2, jnp.int32),
            diver_dir=jnp.array(1, jnp.int32),
            diver_alive=jnp.array(True, jnp.bool_),
            spear_alive=jnp.array(False, jnp.bool_),
            spear=jnp.array([0, 0, 0, 0], jnp.int32),

            tentacle_len=zeros_T,
            tentacle_active=zeros_T.astype(jnp.bool_),
            tentacle_edge_wait=zeros_T,
            tentacle_dir=jnp.ones_like(zeros_T),

            oxygen_frames_remaining=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            oxygen_line_active=jnp.array(False, jnp.bool_),
            oxygen_line_x=jnp.array(-1, jnp.int32),
            oxygen_drop_timer=next_timer,
            oxygen_line_ttl=jnp.array(0, jnp.int32),
            oxygen_contact_counter=jnp.array(0, jnp.int32),

            shark_lane=jnp.array(0, jnp.int32),
            shark_y=self.config.shark_lanes_y[0],
            shark_alive=jnp.array(True, jnp.bool_),

            rng=rng1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_and_consume_life(self, state: NameThisGameState) -> Tuple[NameThisGameState, chex.Array]:
        """Detect death conditions and decrement lives.

        Death occurs if oxygen is empty, the diver is dead, or the shark 'reaches' the
        diver (represented as `shark_alive == False`). When lives reach zero, the
        episode is done; otherwise perform a soft reset.
        """
        oxygen_out = state.oxygen_frames_remaining <= 0
        diver_dead = ~state.diver_alive
        shark_reached_diver = ~state.shark_alive
        death_now = oxygen_out | diver_dead | shark_reached_diver

        def on_death(s):
            remaining = s.lives_remaining - jnp.array(1, jnp.int32)

            def game_over(st):
                return st._replace(lives_remaining=jnp.array(0, jnp.int32)), jnp.array(True, jnp.bool_)

            def lose_life(st):
                st2 = self._life_loss_reset(st)
                return st2._replace(lives_remaining=remaining), jnp.array(False, jnp.bool_)

            return jax.lax.cond(remaining <= 0, game_over, lose_life, s)

        def no_death(s):
            return s, jnp.array(False, jnp.bool_)

        return jax.lax.cond(death_now, on_death, no_death, state)

    @partial(jax.jit, static_argnums=(0,))
    def _shark_speed_for_lane(self, round_idx: chex.Array, lane: chex.Array) -> chex.Array:
        """Absolute shark speed (px/frame) as a function of round and lane.

        Rules
        -----
        - Round 0: all lanes move at 1 px/frame.
        - From round 1: a 2 px/frame boundary rises one lane per wave until all lanes
          are ≥ 2.
        - After all lanes are at least 2, bottom-up promotions add +1 every `nlanes`
          rounds repeatedly (i.e., a staircase by lane rank).

        Example
        -------
        If there are 7 lanes (0=top, 6=bottom) and `speed_progression_start_lane_for_2`
        is 4, the bottom lanes reach 2 px/frame first and the faster band grows upward
        each round.
        """

        cfg = self.config
        nlanes_i = jnp.array(cfg.shark_lanes_y.shape[0], jnp.int32)
        r = jnp.maximum(round_idx + jnp.array(1, jnp.int32), jnp.array(0, jnp.int32))
        lane_i = lane.astype(jnp.int32)

        # rank from bottom (0=bottom)
        b = (nlanes_i - jnp.array(1, jnp.int32)) - lane_i

        start2 = jnp.array(cfg.speed_progression_start_lane_for_2, jnp.int32)
        k2 = jnp.where(
            r == 0,
            jnp.array(0, jnp.int32),
            jnp.clip(nlanes_i - start2 + (r - jnp.array(1, jnp.int32)), 0, nlanes_i),
        )
        is_ge2 = (b < k2).astype(jnp.int32)

        r_all2 = jnp.array(1, jnp.int32) + start2
        t = r - r_all2  # waves since all lanes reached 2
        extra = jnp.maximum(
            jnp.array(0, jnp.int32),
            jnp.floor_divide(t - jnp.array(1, jnp.int32) - b, nlanes_i) + jnp.array(1, jnp.int32),
        )

        mult = jnp.array(1, jnp.int32) + is_ge2 + extra
        return (mult * jnp.array(cfg.shark_base_speed, jnp.int32)).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: NameThisGameState) -> jnp.bool_:
        """Episode ends only when `lives_remaining` reaches zero."""
        return state.lives_remaining <= 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        previous_state: NameThisGameState,
        state: NameThisGameState,
    ) -> jnp.float32:
        """Gym wrapper hook: scalar reward for the transition.

        We define reward as the change in score across the step, which matches
        what `_step_once` computes internally. Returning it here ensures the
        Gymnasium functional wrapper (which computes reward separately) agrees.
        """
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: NameThisGameState) -> chex.Array:
        """Render the given state to an HxWx3 uint8 RGB image.

        Notes
        -----
        - Delegates to `Renderer_NameThisGame`.
        - Marked as JIT with a static `self` arg so it can be composed in wrappers.
        """
        return self.renderer.render(state)


    # -------------------------- Step API -----------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _step_once(
        self, state: NameThisGameState, action: chex.Array
    ) -> Tuple[NameThisGameObservation, NameThisGameState, jnp.float32, jnp.bool_, NameThisGameInfo]:
        """Single-frame update: interpret input, update systems, collisions, scoring.

        Returns `(obs, state, reward, done, info)`. Reward is the score delta.
        """

        prev_state = state

        # Inputs -> intents
        move_dir, fire_pressed = self._interpret_action(state, action)
        # Diver
        state = self._move_diver(state, move_dir)
        # Fire
        just_pressed = fire_pressed & (~state.fire_button_prev)
        can_shoot = just_pressed & state.diver_alive
        state = jax.lax.cond(can_shoot, lambda s: self._spawn_spear(s), lambda s: s, state)

        # Systems
        state = self._move_boat(state)
        state = self._update_bars_and_rest(state, just_pressed)
        state = self._spawn_or_update_oxygen_line(state)
        state = self._move_spear(state)
        state = self._move_shark(state)
        state = self._update_one_tentacle(state)

        # Collisions / hazards
        state = self._check_spear_shark_collision(state)
        state = self._check_spear_tentacle_collision(state)
        state = self._check_diver_hazard(state)

        # Oxygen
        state = self._update_oxygen(state)


        # Lives / soft reset on death
        state, _ = self._check_and_consume_life(state)

        # Reward = score delta
        step_reward = (state.score - prev_state.score).astype(jnp.float32)
        state = state._replace(reward=step_reward, fire_button_prev=fire_pressed)

        # Outputs
        observation = self._get_observation(state)
        done = self._get_done(state)
        all_rewards = self._get_all_reward(prev_state, state)
        info = self._get_info(state, all_rewards)
        return observation, state, step_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: NameThisGameState, action: chex.Array
    ) -> Tuple[NameThisGameObservation, NameThisGameState, jnp.float32, jnp.bool_, NameThisGameInfo]:
        """Frameskip wrapper over `_step_once` that accumulates reward within a macro-step."""

        def body(i, carry):
            st, total_r, done_flag = carry

            def do_step(c):
                st0, tr0, df0 = c
                _obs_i, st1, r_i, done_i, _info_i = self._step_once(st0, action)
                return (st1, tr0 + r_i, jnp.logical_or(df0, done_i))

            return jax.lax.cond(done_flag, lambda c: c, do_step, (st, total_r, done_flag))

        init_carry = (state, jnp.array(0.0, jnp.float32), jnp.array(False, jnp.bool_))
        state_fs, total_reward, _ = jax.lax.fori_loop(0, int(self.frameskip), body, init_carry)

        obs_final = self._get_observation(state_fs)
        done_final = self._get_done(state_fs)
        state_fs = state_fs._replace(reward=total_reward)
        all_rewards = self._get_all_reward(state, state_fs)
        info_final = self._get_info(state_fs, all_rewards)
        return obs_final, state_fs, total_reward, done_final, info_final


# -------------------------- Human control (optional) -----------------------

def get_human_action() -> chex.Array:
    """Poll keyboard and return a discrete action as an int array."""

    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if fire:
        return jnp.array(Action.FIRE)
    if right:
        return jnp.array(Action.RIGHT)
    if left:
        return jnp.array(Action.LEFT)
    return jnp.array(Action.NOOP)


def main():
    """Run a human-playable loop with pygame.

    Controls:
    - A/Left ←, D/Right →, Space = Fire
    - F toggles frame-by-frame mode; when on, press N to step one frame.
    """

    config = NameThisGameConfig()
    pygame.init()
    screen = pygame.display.set_mode(
        (config.screen_width * config.scaling_factor, config.screen_height * config.scaling_factor)
    )
    pygame.display.set_caption("NameThisGame")
    clock = pygame.time.Clock()
    game = JaxNameThisGame(config=config)
    renderer = Renderer_NameThisGame(config=config)

    obs, state = game.reset(jax.random.PRNGKey(0))
    jitted_step = game.step

    running = True
    frame_by_frame = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                frame_by_frame = not frame_by_frame
            elif frame_by_frame and event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                action = get_human_action()
                obs, state, reward, done, info = jitted_step(state, action)
                if bool(jax.device_get(done)):
                    running = False

        if not frame_by_frame:
            action = get_human_action()
            obs, state, reward, done, info = jitted_step(state, action)
            if bool(jax.device_get(done)):
                running = False

        raster = renderer.render(state)
        frame_np = np.array(jax.device_get(raster), dtype=np.uint8)
        frame_np = np.transpose(frame_np, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame_np)
        if config.scaling_factor != 1:
            surface = pygame.transform.scale(
                surface, (config.screen_width * config.scaling_factor, config.screen_height * config.scaling_factor)
            )
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
