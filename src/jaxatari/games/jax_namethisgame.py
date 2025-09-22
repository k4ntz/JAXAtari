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

@dataclass(frozen=True)
class NameThisGameConfig:
    """Game configuration parameters for NameThisGame."""
    # Screen & scaling
    screen_width: int = 160
    screen_height: int = 250
    scaling_factor: int = 3
    # HUD bars
    hud_bar_initial_px: int = 128               # initial length of both bars
    hud_bar_step_frames: int = 250              # shrink cadence for both bars
    hud_bar_shrink_px_per_step_total: int = 8   # 4 from each side => width - 8
    bar_green_height: int = 4
    bar_orange_height: int = 12
    bars_gap_px: int = 0
    bars_bottom_margin_px: int = 25
    # Kraken location
    kraken_x: int = 16
    kraken_y: int = 63
    # Boat (at the surface)
    boat_width: int = 16             # used for bounds and oxygen drop alignment
    boat_speed_px: int = 1            # pixels per movement
    boat_move_every_n_frames: int = 4 # move once every N frames
    # Diver (player)
    diver_width: int = 16
    diver_height: int = 13
    diver_y_floor: int = 173  # fixed y-coordinate for diver (sea floor)
    diver_speed_px: int = 1
    # Spear properties
    spear_width: int = 1
    spear_height: int = 1
    spear_dy: int = -3  # vertical speed (negative = upward)
    # Shark (enemy) lanes and speed
    shark_lanes_y: jnp.ndarray = field(default_factory=lambda: jnp.array([69, 83, 97, 111, 123, 137, 151], dtype=jnp.int32))
    shark_base_speed: int = 1
    shark_width: int = 15
    shark_height: int = 12
    # Tentacles
    max_tentacles: int = 8
    tentacle_base_x: jnp.ndarray = field(default_factory=lambda: jnp.array([24, 38, 54, 70, 86, 102, 118, 134], dtype=jnp.int32))
    tentacle_ys: jnp.ndarray = field(default_factory=lambda: jnp.array([97, 104, 111, 118, 125, 132, 139, 146, 153, 160], dtype=jnp.int32))
    tentacle_num_cols: int = 4
    tentacle_col_width: int = 4
    tentacle_square_w: int = 4
    tentacle_square_h: int = 6
    tentacle_width: int = 4  # kept for obs space compatibility
    tentacle_base_growth_p: float = 0.01       # baseline prob to GROW (vs move)
    tentacle_destroy_points: int = 50
    # Oxygen supply line
    oxygen_full: int = 1200                 # full oxygen frames (~20 seconds at 60 FPS)
    oxygen_pickup_radius: int = 4           # horizontal radius for diver to grab oxygen line (px)
    oxygen_drop_min_interval: int = 240     # minimum frames between oxygen line drops
    oxygen_drop_max_interval: int = 480     # maximum frames between oxygen line drops
    oxygen_line_width: int = 1
    oxygen_y: int = 57
    oxygen_line_ttl_frames: int = 100
    oxygen_contact_every_n_frames: int = 3
    oxygen_contact_points: int = 10
    # Round progression
    speed_progression_start_lane_for_2: int = 3  # wave 1: lanes >=3 are 2px/frame
    round_clear_shark_resets: int = 3
    oxy_frames_speedup_per_round: int = 30
    oxy_min_shrink_interval: int = 20
    tentacle_growth_round_coeff: float = 0.01
    # Lives / treasure
    lives_max: int = 3
    treasure_ui_x: int = 72
    treasure_ui_y: int = 197
    # Score display
    max_digits_for_score: int = 9  # maximum digits to display for score (like Atari scoreboard)

class EntityPosition(NamedTuple):
    """Positions and sizes for entity groups (for observation output)."""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    alive: jnp.ndarray

class NameThisGameState(NamedTuple):
    """Complete game state for NameThisGame (JAX-friendly, immutable)."""
    score: chex.Array             # current score (int32)
    reward: chex.Array            # reward earned in last step (int32 or float)
    round: chex.Array             # current round index (int32)
    shark_resets_this_round: chex.Array  # how many times shark reset (shot) in the current round (int32)

    # HUD bars
    oxy_bar_px: chex.Array        # current orange bar width (px) [0..hud_bar_initial_px]
    wave_bar_px: chex.Array       # current green bar width (px) [0..hud_bar_initial_px]
    bar_frame_counter: chex.Array # frames since the last shrink tick
    oxy_frame_counter: chex.Array

    # Rest state flag
    resting: chex.Array           # bool; when True, tentacles freeze, no oxygen decay/drop; boat+shark still move (lane 0)

    # Boat
    boat_x: chex.Array           # boat's left x position (int32)
    boat_dx: chex.Array          # boat's horizontal direction/speed sign (-1 or +1) (int32)
    boat_move_counter: chex.Array  # counts frames; boat moves when counter % N == 0 (int32)

    # Diver (player)
    diver_x: chex.Array           # diver's x position (int32)
    diver_y: chex.Array           # diver's y position (int32, constant = floor)
    diver_alive: chex.Array       # diver alive flag (bool)
    diver_dir: chex.Array         # Diver facing: -1 = left, +1 = right
    fire_button_prev: chex.Array  # whether fire was pressed in previous frame (bool)

    # Shark (enemy)
    shark_x: chex.Array           # shark's x position (int32)
    shark_y: chex.Array           # shark's y position (int32)
    shark_dx: chex.Array          # shark's horizontal velocity (int32)
    shark_lane: chex.Array        # current lane index (int32, 0=top)
    shark_alive: chex.Array       # shark alive/active flag (bool)

    # Tentacles (octopus arms)
    tentacle_base_x: chex.Array  # (T,)
    tentacle_len: chex.Array  # (T,) number of blocks currently present
    tentacle_cols: chex.Array  # (T, L) per-depth column index (0..3); only first len rows valid
    tentacle_dir: chex.Array  # (T,) -1 or +1 (shared across the whole stack)
    tentacle_edge_wait: chex.Array  # (T,) 0 or 1; 1 = waited one move at edge, next move flips dir
    tentacle_active: chex.Array
    tentacle_turn: chex.Array                   # scalar index 0..T-1
    # Spear
    spear: chex.Array  # shape (4,) -> [x, y, dx, dy] int32
    spear_alive: chex.Array  # bool

    # Oxygen system
    oxygen_frames_remaining: chex.Array  # frames of oxygen left (int32)
    oxygen_line_active: chex.Array       # whether oxygen line is currently dropped (bool)
    oxygen_line_x: chex.Array            # x position of the dropped oxygen line (int32)
    oxygen_drop_timer: chex.Array       # frames until next oxygen line drop (int32)
    oxygen_line_ttl: chex.Array         # frames remaining while the line is active (int32)
    oxygen_contact_counter: chex.Array  # counts consecutive contact frames with diver (int32)

    # Lives
    lives_remaining: chex.Array  # int32 in [0..lives_max]

    rng: chex.Array               # PRNG state (JAX key, shape (2,))

class NameThisGameObservation(NamedTuple):
    """Observation of the current game state, with object-centric fields."""
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
    """Additional info returned by the environment (debug/training use)."""
    score: jnp.ndarray
    round: jnp.ndarray
    shark_lane: jnp.ndarray
    shark_alive: jnp.ndarray
    spear_alive: jnp.ndarray
    tentacles_alive: jnp.ndarray     # boolean array of tentacle alive flags
    oxygen_frames_remaining: jnp.ndarray
    oxygen_line_active: jnp.ndarray
    diver_alive: jnp.ndarray
    lives_remaining: jnp.ndarray
    all_rewards: jnp.ndarray         # vector of all rewards if using custom reward functions

class NameThisGameConstants(NamedTuple):
    """No dynamic constants for this environment (placeholder for interface)."""
    pass

class Renderer_NameThisGame(JAXGameRenderer):
    """Renderer for NameThisGame, handling sprite loading and fallback shapes."""
    sprites: Dict[str, Any]

    def __init__(self, config: NameThisGameConfig = None):
        super().__init__()
        self.config = config or NameThisGameConfig()
        # Set up sprite directory path
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/namethisgame"
        self.sprites = self._load_sprites()
        self.score_digit_sprites = self.sprites.get("score_digit_sprites")

    def _load_sprites(self) -> Dict[str, Any]:
        """Loads game sprites from .npy files if available, otherwise returns dict of available sprites."""
        sprites: Dict[str, Any] = {}
        # Helper to load a single sprite from .npy (returns jnp.ndarray RGBA or None if not found)
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f"{name}.npy")
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)
            return None

        # Attempt to load relevant sprites
        sprite_names = ["diver", "shark", "tentacle", "oxygen_line", "background", "kraken", "boat",  "treasure1", "treasure2", "treasure3"]
        for name in sprite_names:
            spr = _load_sprite_frame(name)
            if spr is not None:
                sprites[name] = spr
        # Load score digit sprites 0-9 if available
        sprites["score_digit_sprites"] = aj.load_and_pad_digits(os.path.join(self.sprite_path, "{}_sprite.npy"), num_chars=10)
        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: NameThisGameState) -> chex.Array:
        """Render the game state into an RGB image (uint8)."""
        # Utility: create a solid-color RGBA sprite
        def _solid_sprite(width: int, height: int, rgb: tuple[int, int, int]) -> chex.Array:
            rgb_arr = jnp.broadcast_to(jnp.array(rgb, dtype=jnp.uint8), (height, width, 3))
            alpha = jnp.full((height, width, 1), 255, dtype=jnp.uint8)
            return jnp.concatenate([rgb_arr, alpha], axis=-1)  # shape (height, width, 4)

        cfg = self.config
        W, H = cfg.screen_width, cfg.screen_height

        # Background
        raster = jnp.zeros((H, W, 3), dtype=jnp.uint8)
        if "background" in self.sprites:
            raster = aj.render_at(raster, 0, 0, self.sprites["background"])

        # Bottom HUD bars
        def _draw_hbar(ras, width_px: chex.Array, height_px: int, y_px: int, rgb: tuple[int, int, int]):
            # clamp desired visible width to [0, max]
            w = jnp.clip(width_px, 0, jnp.array(cfg.hud_bar_initial_px, jnp.int32))

            # Build a fixed-size RGBA sprite (height x max_width) and mask columns >= w
            rgb_arr = jnp.broadcast_to(
                jnp.array(rgb, dtype=jnp.uint8),
                (height_px, cfg.hud_bar_initial_px, 3)
            )
            cols = jnp.arange(cfg.hud_bar_initial_px, dtype=jnp.int32)  # (max_w,)
            alpha_row = jnp.where(cols < w, 255, 0).astype(jnp.uint8)  # (max_w,)
            alpha = jnp.broadcast_to(alpha_row[None, :, None],
                                     (height_px, cfg.hud_bar_initial_px, 1))
            spr = jnp.concatenate([rgb_arr, alpha], axis=-1)  # (h, max_w, 4)

            # center the *visible* part
            x_left = (cfg.screen_width - w) // 2
            return aj.render_at(ras, x_left, y_px, spr)

        orange_y = cfg.screen_height - cfg.bars_bottom_margin_px - cfg.bar_orange_height
        green_y = orange_y - cfg.bars_gap_px - cfg.bar_green_height
        raster = _draw_hbar(raster, state.oxy_bar_px, cfg.bar_orange_height, orange_y, (195, 102, 52))
        raster = _draw_hbar(raster, state.wave_bar_px, cfg.bar_green_height, green_y, (27, 121, 38))

        # draw kraken (octopus)
        if "kraken" in self.sprites:
            raster = aj.render_at(raster, cfg.kraken_x, cfg.kraken_y, self.sprites["kraken"])

        # draw boat at the surface (bottom of boat aligned to oxygen_y)
        if "boat" in self.sprites:
            boat_sprite = self.sprites["boat"]
            boat_h = int(boat_sprite.shape[0])  # height in pixels
        else:
            boat_h = 8
            boat_sprite = _solid_sprite(cfg.boat_width, boat_h, (200, 200, 200))

        # pre-flip boat sprite based on direction; default sprite assumed facing right
        def _flip(spr):  # horizontal mirror
            return jnp.flip(spr, axis=1)

        boat_y = jnp.maximum(0, cfg.oxygen_y - boat_h)
        boat_flipped = jax.lax.cond(state.boat_dx < 0, _flip, lambda spr: spr, boat_sprite)
        raster = aj.render_at(raster, state.boat_x, boat_y, boat_flipped)

        # Draw diver (only if alive)
        if "diver" in self.sprites:
            diver_sprite = self.sprites["diver"]
        else:
            diver_sprite = _solid_sprite(cfg.diver_width, cfg.diver_height, (0, 255, 0))
        diver_to_draw = jax.lax.cond(state.diver_dir > 0, _flip, lambda spr: spr, diver_sprite)
        raster = jax.lax.cond(
            state.diver_alive,
            lambda r: aj.render_at(r, state.diver_x, state.diver_y, diver_to_draw),
            lambda r: r,
            raster,
        )

        # Draw shark (if alive)
        if "shark" in self.sprites:
            shark_sprite = self.sprites["shark"]
        else:
            shark_sprite = _solid_sprite(cfg.shark_width, cfg.shark_height, (150, 150, 150))  # gray shark
        raster = jax.lax.cond(
            state.shark_alive,
            lambda r: aj.render_at(r, state.shark_x, state.shark_y, shark_sprite, flip_horizontal=(state.shark_dx < 0)),
            lambda r: r,
            raster,
        )

        # Draw spear (white)
        spear_sprite = _solid_sprite(cfg.spear_width, cfg.spear_height, (255, 255, 255))
        raster = jax.lax.cond(
            state.spear_alive,
            lambda r: aj.render_at(r, state.spear[0], state.spear[1], spear_sprite),
            lambda r: r,
            raster,
        )

        # Draw tentacles: stacks of 4x6 purple squares at discrete y's and 4 columns
        T = self.config.max_tentacles
        L = int(self.config.tentacle_ys.shape[0])
        col_w = self.config.tentacle_col_width
        sq_w = self.config.tentacle_square_w
        sq_h = self.config.tentacle_square_h
        tent_color = (0, 0, 0)

        square_rgba = _solid_sprite(sq_w, sq_h, tent_color)
        def _draw_one_tentacle(i, ras):
            length = state.tentacle_len[i]
            base_x = state.tentacle_base_x[i]
            def _draw_k(k, r2):
                # draw only while k < length
                def _place(rr):
                    col = state.tentacle_cols[i, k]
                    x = base_x + col * col_w
                    y = self.config.tentacle_ys[k]
                    return aj.render_at(rr, x, y, square_rgba)
                return jax.lax.cond(k < length, _place, lambda rr: rr, r2)
            return jax.lax.fori_loop(0, L, _draw_k, ras)
        raster = jax.lax.fori_loop(0, T, _draw_one_tentacle, raster)

        # Draw oxygen line (if active)
        if "oxygen_line" in self.sprites:
            oxy_sprite = self.sprites["oxygen_line"]
        else:
            # Draw a vertical white line from surface to diver_y_floor
            oxy_sprite = _solid_sprite(cfg.oxygen_line_width, cfg.diver_y_floor, (255, 255, 255))
        raster = jax.lax.cond(
            state.oxygen_line_active,
            lambda r: aj.render_at(r, state.oxygen_line_x, cfg.oxygen_y, oxy_sprite),
            lambda r: r,
            raster,
        )

        if all(k in self.sprites for k in ("treasure1", "treasure2", "treasure3")):
            idx = jnp.clip(state.lives_remaining, 0, 3).astype(jnp.int32)

            def draw_none(r):
                return r

            def draw_t1(r):
                return aj.render_at(r, self.config.treasure_ui_x, self.config.treasure_ui_y, self.sprites["treasure1"])

            def draw_t2(r):
                return aj.render_at(r, self.config.treasure_ui_x, self.config.treasure_ui_y, self.sprites["treasure2"])

            def draw_t3(r):
                return aj.render_at(r, self.config.treasure_ui_x, self.config.treasure_ui_y, self.sprites["treasure3"])

            raster = jax.lax.switch(idx, (draw_none, draw_t1, draw_t2, draw_t3), raster)
        else:
            # Fallback: draw up to 3 tiny white squares
            lives = jnp.clip(state.lives_remaining, 0, 3)
            sq = jnp.concatenate([
                jnp.full((6, 6, 3), 255, jnp.uint8), jnp.full((6, 6, 1), 255, jnp.uint8)
            ], axis=-1)

            def maybe_draw(i, ras):
                def draw(rr): return aj.render_at(rr, self.config.treasure_ui_x + 8 * i, self.config.treasure_ui_y, sq)

                return jax.lax.cond(lives > i, draw, lambda rr: rr, ras)

            raster = jax.lax.fori_loop(0, 3, maybe_draw, raster)

        # Render score (centered at top)
        if self.score_digit_sprites is not None:
            max_digits = cfg.max_digits_for_score
            num_digits = jnp.where(
                state.score > 0,
                jnp.ceil(jnp.log10(state.score.astype(jnp.float32) + 1.0)).astype(jnp.int32),
                1,
            )
            score_digits = aj.int_to_digits(state.score, max_digits=max_digits)
            digit_w = 8  # width of each score digit sprite
            total_w = digit_w * num_digits
            score_x = (cfg.screen_width - total_w) // 2
            score_y = 5
            raster = aj.render_label_selective(
                raster,
                score_x,
                score_y,
                score_digits,
                self.score_digit_sprites,
                max_digits - num_digits,  # skip leading zeros
                num_digits,
                spacing=digit_w,
            )

        return raster

class JaxNameThisGame(JaxEnvironment[NameThisGameState, NameThisGameObservation, NameThisGameInfo, NameThisGameConstants]):
    """
    JAX implementation of the Atari 2600 game 'Name This Game'.

    Game Description:
    - You control a diver on the sea floor who can move left or right and fire a spear upward.
    - A shark repeatedly swims across the screen in horizontal lanes, dropping one lane deeper each time it exits the screen. Shooting the shark causes it to reset to the top lane (earning points).
    - Above, an octopus extends tentacles downward. You must shoot the tip of each tentacle to destroy it (earn points) before it reaches you.
    - The diver has a limited oxygen supply (represented by a decreasing counter). Periodically, an oxygen line drops from a boat at the surface; moving under it replenishes oxygen to full.
    - The game is organized in rounds: once you destroy all tentacles and have shot the shark a required number of times, a new round begins with increased difficulty (shark moves faster, tentacles extend faster, oxygen lines appear less frequently).
    - Game ends (done) if you run out of oxygen, if a tentacle tip touches the diver, or if the shark reaches the diver lane at the bottom.

    Technical Features:
    - Uses JAX for vectorized, functional style updates.
    - The environment state is a NamedTuple of JAX arrays, and all operations are JIT-compiled for performance.
    - Supports batched operations (vmap) and deterministic behavior under a given PRNG key.
    """
    def __init__(self, frameskip: int = 1, reward_funcs: list = None, config: NameThisGameConfig = None):
        super().__init__()
        self.config = config or NameThisGameConfig()
        self.frameskip = frameskip
        self.frame_stack_size = 4  # standard frame stack length for Atari
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.renderer = Renderer_NameThisGame(config=self.config)
        # Define action set: 0: NOOP, 1: LEFT, 2: RIGHT, 3: FIRE, 4: LEFTFIRE, 5: RIGHTFIRE
        self.action_set = [
            Action.NOOP,
            Action.LEFT,
            Action.RIGHT,
            Action.FIRE,
            Action.LEFTFIRE,
            Action.RIGHTFIRE,
        ]

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Tuple[NameThisGameObservation, NameThisGameState]:
        cfg = self.config
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])

        # PRNG splits
        key, sub_key_dir, sub_key_oxy = jax.random.split(key, 3)

        # Boat
        init_boat_x = jnp.array(cfg.screen_width // 2 - cfg.boat_width // 2, dtype=jnp.int32)
        init_boat_dx = jnp.array(1, dtype=jnp.int32)
        init_boat_counter = jnp.array(0, dtype=jnp.int32)

        # Diver
        init_diver_x = jnp.array(cfg.screen_width // 2, dtype=jnp.int32)
        init_diver_y = jnp.array(cfg.diver_y_floor, dtype=jnp.int32)

        # Shark
        go_left = jax.random.bernoulli(sub_key_dir)
        init_shark_lane = jnp.array(0, dtype=jnp.int32)
        init_shark_y = cfg.shark_lanes_y[init_shark_lane]
        init_shark_x = jnp.where(go_left, -cfg.shark_width, cfg.screen_width)
        init_shark_speed = self._shark_speed_for_lane(jnp.array(0, jnp.int32), init_shark_lane)
        init_shark_dx = jnp.where(go_left, init_shark_speed, -init_shark_speed).astype(jnp.int32)

        # Tentacles: empty stacks to start
        tentacle_len = jnp.zeros((T,), dtype=jnp.int32)
        tentacle_cols = jnp.zeros((T, L), dtype=jnp.int32)  # values don't matter until len>0
        tentacle_dir = jnp.ones((T,), dtype=jnp.int32)  # start moving right
        tentacle_edge_wait = jnp.zeros((T,), dtype=jnp.int32)
        tentacle_active = (tentacle_len > 0)

        # Spear
        empty_spear = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

        # Oxygen
        init_oxygen = jnp.array(cfg.oxygen_full, dtype=jnp.int32)
        init_oxygen_line_active = jnp.array(False, dtype=jnp.bool_)
        init_oxygen_line_x = jnp.array(-1, dtype=jnp.int32)
        min_int = int(cfg.oxygen_drop_min_interval)
        max_int = int(cfg.oxygen_drop_max_interval)
        init_drop_timer = jax.random.randint(sub_key_oxy, (), min_int, max_int + 1, dtype=jnp.int32)

        state = NameThisGameState(
            score=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(0, dtype=jnp.int32),
            round=jnp.array(0, dtype=jnp.int32),
            shark_resets_this_round=jnp.array(0, dtype=jnp.int32),

            oxy_bar_px=jnp.array(cfg.hud_bar_initial_px, dtype=jnp.int32),
            wave_bar_px=jnp.array(cfg.hud_bar_initial_px, dtype=jnp.int32),
            bar_frame_counter=jnp.array(0, jnp.int32),
            oxy_frame_counter=jnp.array(0, jnp.int32),

            resting=jnp.array(True, dtype=jnp.bool_),

            boat_x=init_boat_x,
            boat_dx=init_boat_dx,
            boat_move_counter=init_boat_counter,

            diver_x=init_diver_x,
            diver_y=init_diver_y,
            diver_alive=jnp.array(True, dtype=jnp.bool_),
            diver_dir=jnp.array(1, dtype=jnp.int32),
            fire_button_prev=jnp.array(False, dtype=jnp.bool_),

            shark_x=init_shark_x.astype(jnp.int32),
            shark_y=init_shark_y.astype(jnp.int32),
            shark_dx=init_shark_dx,
            shark_lane=init_shark_lane,
            shark_alive=jnp.array(True, dtype=jnp.bool_),

            tentacle_base_x=cfg.tentacle_base_x,
            tentacle_len=tentacle_len,
            tentacle_cols=tentacle_cols,
            tentacle_dir=tentacle_dir,
            tentacle_edge_wait=tentacle_edge_wait,
            tentacle_active=tentacle_active,

            spear=empty_spear,
            spear_alive=jnp.array(False, dtype=jnp.bool_),

            oxygen_frames_remaining=init_oxygen,
            oxygen_line_active=init_oxygen_line_active,
            oxygen_line_x=init_oxygen_line_x,
            oxygen_drop_timer=init_drop_timer,
            oxygen_line_ttl=jnp.array(0, dtype=jnp.int32),
            oxygen_contact_counter=jnp.array(0, dtype=jnp.int32),

            tentacle_turn=jnp.array(0, dtype=jnp.int32),

            lives_remaining=jnp.array(cfg.lives_max, dtype=jnp.int32),

            rng=key,
        )
        obs = self._get_observation(state)
        return obs, state

    def action_space(self) -> spaces.Discrete:
        """Action space: 6 discrete actions (NOOP, LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE)."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """Observation space specification as a dictionary of subspaces."""
        cfg = self.config
        # Helper for entity subspace
        def entity_space(n: int, w_max: int, h_max: int) -> spaces.Dict:
            return spaces.Dict({
                "x": spaces.Box(low=-w_max, high=cfg.screen_width, shape=(n,), dtype=jnp.int32),
                "y": spaces.Box(low=-h_max, high=cfg.screen_height, shape=(n,), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=w_max, shape=(n,), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=h_max, shape=(n,), dtype=jnp.int32),
                "alive": spaces.Box(low=0, high=1, shape=(n,), dtype=jnp.int32),
            })
        tentacle_h_max = int(cfg.tentacle_ys[-1] - cfg.tentacle_ys[0] + cfg.tentacle_square_h)
        return spaces.Dict({
            "score": spaces.Box(low=0, high=(10**cfg.max_digits_for_score) - 1, shape=(), dtype=jnp.int32),
            "diver": entity_space(n=1, w_max=cfg.diver_width, h_max=cfg.diver_height),
            "shark": entity_space(n=1, w_max=cfg.shark_width, h_max=cfg.shark_height),
            "spear": entity_space(n=1, w_max=cfg.spear_width, h_max=cfg.spear_height),
            "tentacles": entity_space(n=cfg.max_tentacles, w_max=cfg.tentacle_width, h_max=tentacle_h_max),
            "oxygen_frames_remaining": spaces.Box(low=0, high=cfg.oxygen_full, shape=(), dtype=jnp.int32),
            "oxygen_line_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "oxygen_line_x": spaces.Box(low=-cfg.oxygen_line_width, high=cfg.screen_width, shape=(), dtype=jnp.int32),
            "round_idx": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Observation image (render) space: RGB image of shape (H, W, 3)."""
        cfg = self.config
        return spaces.Box(low=0, high=255, shape=(cfg.screen_height, cfg.screen_width, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: EnvObs) -> jnp.ndarray:
        """Convert an observation to a flat 1D array (for agents)."""
        def _flat(ep: EntityPosition) -> jnp.ndarray:
            return jnp.concatenate([
                jnp.ravel(ep.x).astype(jnp.int32),
                jnp.ravel(ep.y).astype(jnp.int32),
                jnp.ravel(ep.width).astype(jnp.int32),
                jnp.ravel(ep.height).astype(jnp.int32),
                jnp.ravel(ep.alive).astype(jnp.int32),
            ], axis=0)
        return jnp.concatenate([
            jnp.atleast_1d(obs.score).astype(jnp.int32),
            _flat(obs.diver),
            _flat(obs.shark),
            _flat(obs.spear),
            _flat(obs.tentacles),
            jnp.atleast_1d(obs.oxygen_frames_remaining).astype(jnp.int32),
            jnp.atleast_1d(obs.oxygen_line_active).astype(jnp.int32),
            jnp.atleast_1d(obs.oxygen_line_x).astype(jnp.int32),
            jnp.atleast_1d(obs.round_idx).astype(jnp.int32),
        ], axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: NameThisGameState) -> NameThisGameObservation:
        cfg = self.config
        # Diver
        diver_alive = state.diver_alive.astype(jnp.int32)
        diver_pos = EntityPosition(
            x=jnp.atleast_1d(state.diver_x),
            y=jnp.atleast_1d(state.diver_y),
            width=jnp.atleast_1d(jnp.array(cfg.diver_width, dtype=jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.diver_height, dtype=jnp.int32)),
            alive=jnp.atleast_1d(diver_alive),
        )
        # Shark
        shark_alive = state.shark_alive.astype(jnp.int32)
        shark_pos = EntityPosition(
            x=jnp.atleast_1d(state.shark_x),
            y=jnp.atleast_1d(state.shark_y),
            width=jnp.atleast_1d(jnp.array(cfg.shark_width, dtype=jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.shark_height, dtype=jnp.int32)),
            alive=jnp.atleast_1d(shark_alive),
        )
        # Spear
        spear_alive = state.spear_alive.astype(jnp.int32)
        spear_pos = EntityPosition(
            x=jnp.atleast_1d(state.spear[0]),
            y=jnp.atleast_1d(state.spear[1]),
            width=jnp.atleast_1d(jnp.array(cfg.spear_width, dtype=jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.spear_height, dtype=jnp.int32)),
            alive=jnp.atleast_1d(spear_alive),
        )
        # Tentacles => bounding boxes for each stack (keeps your original Obs shape)
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])
        len_vec = state.tentacle_len  # (T,)
        alive_vec = (len_vec > 0).astype(jnp.int32)

        # leftmost col across existing blocks; if none -> -1
        # For simplicity, use the top block's column as left; stacks are narrow anyway.
        top_cols = jnp.where(len_vec > 0, state.tentacle_cols[:, 0], jnp.array(0, jnp.int32))
        tentacle_left_x = jnp.where(
            len_vec > 0,
            state.tentacle_base_x + top_cols * cfg.tentacle_col_width,
            jnp.array(-1, jnp.int32)
        )
        tentacle_y_top = jnp.where(len_vec > 0, jnp.full((T,), cfg.tentacle_ys[0], dtype=jnp.int32),
                                   jnp.array(-1, jnp.int32))
        # height in pixels: from y[0] to y[len-1] + block_h
        last_y = jnp.take_along_axis(cfg.tentacle_ys[None, :].repeat(T, axis=0),
                                     jnp.clip((len_vec - 1)[:, None], 0, L - 1), axis=1).squeeze(1)
        tentacle_height_px = jnp.where(len_vec > 0, last_y - cfg.tentacle_ys[0] + cfg.tentacle_square_h,
                                       jnp.array(0, jnp.int32))
        tentacle_width_px = jnp.where(len_vec > 0, jnp.full((T,), cfg.tentacle_width, dtype=jnp.int32),
                                      jnp.array(0, jnp.int32))

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
        """Construct the Info NamedTuple from the state and reward components."""
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
            lives_remaining=state.lives_remaining,   # NEW
            all_rewards=(all_rewards if all_rewards is not None else jnp.zeros(1, dtype=jnp.float32)),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, prev_state: NameThisGameState, state: NameThisGameState) -> chex.Array:
        """Compute vector of all reward function outputs, if any custom reward functions are provided."""
        if self.reward_funcs is None:
            return jnp.zeros((1,), dtype=jnp.float32)
        rewards = jnp.array([func(prev_state, state) for func in self.reward_funcs], dtype=jnp.float32)
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _update_bars_and_rest(self, state: NameThisGameState, just_pressed: chex.Array) -> NameThisGameState:
        cfg = self.config

        # Exit rest when the player fires a spear
        def _exit_rest(s: NameThisGameState) -> NameThisGameState:
            return s._replace(
                resting=jnp.array(False, jnp.bool_),
                wave_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
                bar_frame_counter=jnp.array(0, jnp.int32),
                # Optional: bump difficulty using your built-in round scaling.
                round=s.round + 1,
            )

        state = jax.lax.cond(state.resting & just_pressed, _exit_rest, lambda q: q, state)

        # While resting => freeze counters/bars
        def _if_rest(s: NameThisGameState) -> NameThisGameState:
            return s

        def _if_active(s: NameThisGameState) -> NameThisGameState:
            # --- wave bar cadence (constant) ---
            wave_cnt = s.bar_frame_counter + 1
            wave_tick = wave_cnt >= jnp.array(cfg.hud_bar_step_frames, jnp.int32)

            # --- oxygen bar cadence (speeds up per wave) ---
            r_eff = jnp.maximum(s.round, jnp.array(1, jnp.int32))  # wave 1 => 250, wave 2 => 220, ...
            oxy_interval = (
                    jnp.array(cfg.hud_bar_step_frames, jnp.int32)
                    - jnp.array(cfg.oxy_frames_speedup_per_round, jnp.int32) * (r_eff - 1)
            )
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

            # If wave bar hits 0 -> enter REST (as before)
            def _enter_rest(st: NameThisGameState) -> NameThisGameState:
                zeros_T = jnp.zeros_like(st.tentacle_len)
                return st._replace(
                    resting=jnp.array(True, jnp.bool_),
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
        """Decode the input action into movement direction and fire button press."""
        # Determine horizontal movement: -1 for left, +1 for right, 0 for none
        move_dir = jnp.where(
            (action == Action.LEFT) | (action == Action.LEFTFIRE),
            -1,
            jnp.where((action == Action.RIGHT) | (action == Action.RIGHTFIRE), 1, 0)
        )
        # Determine if fire is pressed
        fire_pressed = (action == Action.FIRE) | (action == Action.LEFTFIRE) | (action == Action.RIGHTFIRE)
        return move_dir, fire_pressed

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_spear(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config

        # only spawn if not already alive
        def _spawn(s):
            spawn_x = s.diver_x + (cfg.diver_width // 2)
            spawn_y = s.diver_y
            new = jnp.array([spawn_x, spawn_y, 0, cfg.spear_dy], dtype=jnp.int32)
            return s._replace(spear=new, spear_alive=jnp.array(True, jnp.bool_))
        return jax.lax.cond(~state.spear_alive, _spawn, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def _move_diver(self, state: NameThisGameState, move_dir: chex.Array) -> NameThisGameState:
        cfg = self.config
        new_x = jnp.clip(state.diver_x + move_dir * cfg.diver_speed_px, 0, cfg.screen_width - cfg.diver_width)
        new_dir = jnp.where(move_dir != 0, move_dir.astype(jnp.int32), state.diver_dir)
        return state._replace(diver_x=new_x, diver_dir=new_dir)

    @partial(jax.jit, static_argnums=(0,))
    def _move_spear(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config
        def _step(s):
            # (2,) position = xy + dxy
            pos_xy = s.spear[:2] + s.spear[2:4]
            s2 = s._replace(spear=s.spear.at[:2].set(pos_xy))
            x, y = pos_xy[0], pos_xy[1]
            in_bounds = (x >= 0) & (x < cfg.screen_width) & (y >= 0) & (y < cfg.screen_height)
            return s2._replace(spear_alive=jnp.logical_and(s.spear_alive, in_bounds))
        return jax.lax.cond(state.spear_alive, _step, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def _move_shark(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config
        x = state.shark_x + state.shark_dx

        def _resting_move(s: NameThisGameState) -> NameThisGameState:
            # Keep shark on lane 0, wrap horizontally instead of dropping lanes
            off_right = (s.shark_dx > 0) & (x >= cfg.screen_width)
            off_left  = (s.shark_dx < 0) & ((x + cfg.shark_width) <= 0)

            def _wrap_right(st):  # moving right, wrap to left
                return st._replace(shark_x=jnp.array(-cfg.shark_width, jnp.int32))
            def _wrap_left(st):   # moving left, wrap to right
                return st._replace(shark_x=jnp.array(cfg.screen_width, jnp.int32))

            st = s._replace(
                shark_x=x,
                shark_lane=jnp.array(0, jnp.int32),
                shark_y=cfg.shark_lanes_y[0],
                shark_alive=jnp.array(True, jnp.bool_),
            )
            st = jax.lax.cond(off_right, _wrap_right, lambda q: q, st)
            st = jax.lax.cond(off_left,  _wrap_left,  lambda q: q, st)
            return st

        def _normal_move(s: NameThisGameState) -> NameThisGameState:
            new_x = x
            dx = s.shark_dx
            off_right = (dx > 0) & (new_x >= cfg.screen_width)
            off_left  = (dx < 0) & ((new_x + cfg.shark_width) <= 0)

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
                    return tt._replace(shark_alive=jnp.array(False, dtype=jnp.bool_))

                has_lane = new_lane < (last_idx + 1)
                return jax.lax.cond(has_lane, _lane_exists, _no_lane, st)

            st = s._replace(shark_x=new_x)
            st = jax.lax.cond(off_right, lambda u: _drop_lane(u, going_left=False), lambda u: u, st)
            st = jax.lax.cond(off_left,  lambda u: _drop_lane(u, going_left=True),  lambda u: u, st)
            return st

        return jax.lax.cond(state.resting, _resting_move, _normal_move, state)

    @partial(jax.jit, static_argnums=(0,))
    def _update_one_tentacle(self, state: NameThisGameState) -> NameThisGameState:
        """Round-robin: on each frame update exactly one tentacle: MOVE (more likely) or GROW.
        JAX-safe: no dynamic slicing and loops have static bounds."""
        cfg = self.config
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])
        max_col = cfg.tentacle_num_cols - 1

        # Freeze updates during rest
        def _no_update(s: NameThisGameState) -> NameThisGameState:
            return s

        def _active_update(s0: NameThisGameState) -> NameThisGameState:
            i = s0.tentacle_turn % T
            rng_choice, rng_after = jax.random.split(s0.rng)

            # growth probability increases with round
            r_float = s0.round.astype(jnp.float32)
            p_grow = jnp.clip(cfg.tentacle_base_growth_p + cfg.tentacle_growth_round_coeff * r_float, 0.0, 0.95)
            do_grow = jax.random.bernoulli(rng_choice, p_grow)

            # Helpers to read/write the i-th tentacle
            def get_row(arr): return arr[i]

            def set_row(arr, val): return arr.at[i].set(val)

            def set_row2d(arr, val): return arr.at[i, :].set(val)

            cols_i = get_row(s0.tentacle_cols)
            len_i = get_row(s0.tentacle_len)
            dir_i = get_row(s0.tentacle_dir)
            wait_i = get_row(s0.tentacle_edge_wait)

            # ----- GROW -----
            def _grow(s: NameThisGameState) -> NameThisGameState:
                l = len_i
                cols = cols_i

                def _when_empty():
                    start_col = jnp.array(1, dtype=jnp.int32)
                    new_cols = cols.at[:].set(0)
                    new_cols = new_cols.at[0].set(start_col)
                    new_len = jnp.minimum(l + 1, L)
                    return new_cols, new_len

                def _when_non_empty():
                    prev_top = cols[0]
                    # Static-length "shift down by 1" for indices 0..l
                    idx = jnp.arange(L, dtype=jnp.int32)
                    prevs = jnp.concatenate([jnp.array([prev_top], jnp.int32), cols[:-1]])
                    # mask for positions we update: 0..l
                    # (clamp l to L-1 so mask is well-formed)
                    l_clamped = jnp.minimum(l, jnp.array(L - 1, jnp.int32))
                    mask = idx <= l_clamped
                    new_cols = jnp.where(mask, prevs, cols)
                    new_len = jnp.minimum(l + 1, L)
                    return new_cols, new_len

                new_cols, new_len = jax.lax.cond(l == 0, _when_empty, _when_non_empty)
                s = s._replace(
                    tentacle_cols=set_row2d(s0.tentacle_cols, new_cols),
                    tentacle_len=set_row(s0.tentacle_len, new_len),
                    tentacle_active=set_row(s0.tentacle_active, new_len > 0),
                )
                return s

            # ----- MOVE (all blocks advance together, adjacency <= 1, edge wait + flip) -----
            def _move(s: NameThisGameState) -> NameThisGameState:
                cols = cols_i
                l = len_i
                d = dir_i
                w = wait_i

                def _nothing(st):  # empty stack
                    return st

                def _do_move(st: NameThisGameState) -> NameThisGameState:
                    at_left = (cols[0] == 0)
                    at_right = (cols[0] == max_col)
                    blocked = jnp.where(d < 0, at_left, at_right)
                    idx = jnp.arange(L, dtype=jnp.int32)
                    stacked = jnp.all((cols == cols[0]) | (idx >= l))
                    def _perform_move(cur_cols, direction):
                        top_new = jnp.clip(cur_cols[0] + direction, 0, max_col)

                        def body(k, acc):
                            # only update rows < l
                            def do_step(a):
                                # k == 0 => set top
                                def first(a2):
                                    return a2.at[0].set(top_new)

                                # k > 0 => constrain to adjacency of previous new row
                                def rest(a2):
                                    prev_new = a2[k - 1]
                                    desired = jnp.clip(cur_cols[k] + direction, 0, max_col)
                                    low = jnp.maximum(prev_new - 1, 0)
                                    high = jnp.minimum(prev_new + 1, max_col)
                                    newk = jnp.clip(desired, low, high)
                                    return a2.at[k].set(newk)

                                return jax.lax.cond(k == 0, first, rest, a)

                            return jax.lax.cond(k < l, do_step, lambda a: a, acc)

                        out = jax.lax.fori_loop(0, L, body, cur_cols)
                        return out

                        # Note: L is static, so the loop bound is static. We guard with k < l.

                    def _edge_logic(st2: NameThisGameState) -> NameThisGameState:
                        # If blocked & stacked: first wait one move, next time flip and move.
                        def _first_wait():
                            return st2._replace(
                                tentacle_edge_wait=set_row(s0.tentacle_edge_wait, jnp.array(1, jnp.int32)))

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

            # Advance round-robin index and RNG
            next_turn = (s1.tentacle_turn + 1) % T
            s1 = s1._replace(tentacle_turn=next_turn, rng=rng_after)
            # Keep tentacle_active coherent with len>0
            s1 = s1._replace(tentacle_active=(s1.tentacle_len > 0))
            return s1

        return jax.lax.cond(state.resting, _no_update, _active_update, state)

    @partial(jax.jit, static_argnums=(0,))
    def _check_spear_shark_collision(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config
        rng_side, rng_after = jax.random.split(state.rng)

        spear_left = state.spear[0]
        spear_right = state.spear[0] + cfg.spear_width
        spear_top = state.spear[1]
        spear_bottom = state.spear[1] + cfg.spear_height

        shark_left = state.shark_x
        shark_right = state.shark_x + cfg.shark_width
        shark_top = state.shark_y
        shark_bottom = state.shark_y + cfg.shark_height

        hit = state.spear_alive & state.shark_alive & \
              (spear_left < shark_right) & (spear_right > shark_left) & \
              (spear_top < shark_bottom) & (spear_bottom > shark_top)

        def _on_hit(s):
            points = jnp.array(100, jnp.int32) * (s.shark_lane + 1)
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
                spear_alive=jnp.array(False, jnp.bool_),  # spear consumed
                rng=rng_after
            )

        return jax.lax.cond(hit, _on_hit, lambda s: s._replace(rng=rng_after), state)

    @partial(jax.jit, static_argnums=(0,))
    def _check_spear_tentacle_collision(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config
        T = cfg.max_tentacles
        L = int(cfg.tentacle_ys.shape[0])

        # spear rect
        sl = state.spear[0]
        sr = state.spear[0] + cfg.spear_width
        st = state.spear[1]
        sb = state.spear[1] + cfg.spear_height

        lens = state.tentacle_len  # (T,)
        has_tip = lens > 0
        tip_idx = jnp.maximum(lens - 1, 0)  # (T,)

        # gather per-tentacle tip column
        tip_cols = jnp.take_along_axis(state.tentacle_cols, tip_idx[:, None], axis=1).squeeze(axis=1)  # (T,)

        x_lefts = state.tentacle_base_x + tip_cols * cfg.tentacle_col_width
        x_rights = x_lefts + cfg.tentacle_square_w
        y_tops = jnp.take(cfg.tentacle_ys, tip_idx)  # (T,)
        y_bottoms = y_tops + cfg.tentacle_square_h

        over_x = (sl < x_rights) & (sr > x_lefts)
        over_y = (st < y_bottoms) & (sb > y_tops)
        hits = state.spear_alive & has_tip & over_x & over_y  # (T,)

        num_hits = jnp.sum(hits.astype(jnp.int32))
        gained = num_hits * jnp.array(cfg.tentacle_destroy_points, jnp.int32)

        # For hit tentacles: zero their stacks; reset dir to +1, wait=0
        new_len = jnp.where(hits, jnp.array(0, jnp.int32), state.tentacle_len)
        new_cols = state.tentacle_cols  # contents irrelevant if len=0
        new_dir = jnp.where(hits, jnp.array(1, jnp.int32), state.tentacle_dir)
        new_wait = jnp.where(hits, jnp.array(0, jnp.int32), state.tentacle_edge_wait)

        spear_alive_next = jnp.logical_and(state.spear_alive, jnp.logical_not(jnp.any(hits)))

        return state._replace(
            score=state.score + gained,
            tentacle_len=new_len,
            tentacle_cols=new_cols,
            tentacle_dir=new_dir,
            tentacle_edge_wait=new_wait,
            tentacle_active=(new_len > 0),
            spear_alive=spear_alive_next,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_diver_hazard(self, state: NameThisGameState) -> NameThisGameState:
        """Diver dies if any tentacle has a block at the last y position (i.e., length == L)."""
        L = int(self.config.tentacle_ys.shape[0])
        reached = jnp.any(state.tentacle_len >= L)
        return state._replace(diver_alive=jnp.where(reached, jnp.array(False, dtype=jnp.bool_), state.diver_alive))

    @partial(jax.jit, static_argnums=(0,))
    def _move_boat(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config
        counter = state.boat_move_counter + 1
        move_now = (state.boat_move_counter % cfg.boat_move_every_n_frames) == 0

        def _do_move(s: NameThisGameState) -> NameThisGameState:
            new_x = s.boat_x + s.boat_dx * cfg.boat_speed_px

            hit_left = new_x <= 0
            hit_right = (new_x + cfg.boat_width) >= cfg.screen_width
            hit_edge = hit_left | hit_right

            # reflect direction on edge hit
            new_dx = jnp.where(hit_edge, -s.boat_dx, s.boat_dx)

            # clamp x inside bounds after move
            clamped_x = jnp.where(
                hit_left, 0,
                jnp.where(hit_right, cfg.screen_width - cfg.boat_width, new_x)
            )

            return s._replace(boat_x=clamped_x, boat_dx=new_dx, boat_move_counter=counter)

        # if not time to move, just bump the counter
        return jax.lax.cond(move_now, _do_move, lambda s: s._replace(boat_move_counter=counter), state)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_or_update_oxygen_line(self, state: NameThisGameState) -> NameThisGameState:
        """Spawn under the boat when timer hits 0; keep it for TTL frames, following the boat;
        when TTL expires, despawn and schedule the next drop."""
        cfg = self.config
        rng_line, rng_interval, rng_after = jax.random.split(state.rng, 3)
        def _no_line(s: NameThisGameState) -> NameThisGameState:
            new_timer = jnp.maximum(s.oxygen_drop_timer - 1, 0)
            def _spawn_line(st: NameThisGameState) -> NameThisGameState:
                new_x = s.boat_x + (cfg.boat_width - cfg.oxygen_line_width) // 2
                new_x = jnp.clip(new_x, 0, cfg.screen_width - cfg.oxygen_line_width).astype(jnp.int32)
                return st._replace(
                    oxygen_line_active=jnp.array(True, dtype=jnp.bool_),
                    oxygen_line_x=new_x,
                    oxygen_drop_timer=jnp.array(0, dtype=jnp.int32),
                    oxygen_line_ttl=jnp.array(cfg.oxygen_line_ttl_frames, dtype=jnp.int32),
                    oxygen_contact_counter=jnp.array(0, dtype=jnp.int32),
                )
            def _no_spawn(st: NameThisGameState) -> NameThisGameState:
                return st._replace(oxygen_drop_timer=new_timer)
            return jax.lax.cond(new_timer <= 0, _spawn_line, _no_spawn, s)
        def _line_active(s: NameThisGameState) -> NameThisGameState:
            # Follow boat horizontally and age TTL
            new_x = s.boat_x + (cfg.boat_width - cfg.oxygen_line_width) // 2
            new_x = jnp.clip(new_x, 0, cfg.screen_width - cfg.oxygen_line_width).astype(jnp.int32)
            new_ttl = jnp.maximum(s.oxygen_line_ttl - 1, 0)
            def _expire(st: NameThisGameState) -> NameThisGameState:
                # Schedule next random interval on expiry
                min_int = cfg.oxygen_drop_min_interval
                max_int = cfg.oxygen_drop_max_interval
                next_timer = jax.random.randint(rng_interval, (), min_int, max_int + 1, dtype=jnp.int32)
                return st._replace(
                    oxygen_line_active=jnp.array(False, dtype=jnp.bool_),
                    oxygen_line_x=jnp.array(-1, dtype=jnp.int32),
                    oxygen_drop_timer=next_timer,
                    oxygen_line_ttl=jnp.array(0, dtype=jnp.int32),
                    oxygen_contact_counter=jnp.array(0, dtype=jnp.int32),
                )
            def _still(st: NameThisGameState) -> NameThisGameState:
                return st._replace(oxygen_line_x=new_x, oxygen_line_ttl=new_ttl)
            return jax.lax.cond(new_ttl <= 0, _expire, _still, s)
        state = jax.lax.cond(state.oxygen_line_active, _line_active, _no_line, state)
        return state._replace(rng=rng_after)

    @partial(jax.jit, static_argnums=(0,))
    def _update_oxygen(self, state: NameThisGameState) -> NameThisGameState:
        cfg = self.config

        # Freeze in rest state (no drain, no refill)
        def _rest(s: NameThisGameState) -> NameThisGameState:
            return s._replace(oxygen_frames_remaining=s.oxy_bar_px)  # keep coherent

        def _active(s: NameThisGameState) -> NameThisGameState:
            # Are we under the oxygen line?
            diver_center = s.diver_x + (cfg.diver_width // 2)
            line_center  = s.oxygen_line_x + (cfg.oxygen_line_width // 2)
            under = s.oxygen_line_active & (jnp.abs(diver_center - line_center) <= cfg.oxygen_pickup_radius)

            # Count contact frames and apply +4 px every K frames
            contact_frames_needed = cfg.oxygen_contact_every_n_frames
            cnt_next = jnp.where(under, s.oxygen_contact_counter + 1, jnp.array(0, jnp.int32))
            tick = under & (cnt_next % contact_frames_needed == 0) & (cnt_next > 0)

            inc = jnp.where(tick, jnp.array(cfg.hud_bar_shrink_px_per_step_total, jnp.int32), jnp.array(0, jnp.int32))
            new_oxy_bar = jnp.minimum(s.oxy_bar_px + inc, jnp.array(cfg.hud_bar_initial_px, jnp.int32))

            return s._replace(
                oxy_bar_px=new_oxy_bar,
                oxygen_contact_counter=cnt_next,
                oxygen_frames_remaining=new_oxy_bar,  # keep a simple proxy in 'frames' slot
            )

        return jax.lax.cond(state.resting, _rest, _active, state)

    @partial(jax.jit, static_argnums=(0,))
    def _life_loss_reset(self, state: NameThisGameState) -> NameThisGameState:
        """Soft reset after a death: go to REST, refill bars/oxygen, clear hazards, keep score/round."""
        cfg = self.config
        # Re-seed next oxygen drop timer for current round difficulty
        rng1, rng2 = jax.random.split(state.rng)
        min_int = cfg.oxygen_drop_min_interval
        max_int = cfg.oxygen_drop_max_interval
        next_timer = jax.random.randint(rng2, (), min_int, max_int + 1, dtype=jnp.int32)

        zeros_T = jnp.zeros_like(state.tentacle_len)
        return state._replace(
            resting=jnp.array(True, jnp.bool_),
            wave_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            oxy_bar_px=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            bar_frame_counter=jnp.array(0, jnp.int32),
            oxy_frame_counter=jnp.array(0, jnp.int32),

            # Diver back to center, alive, spear cleared
            diver_x=jnp.array(cfg.screen_width // 2, jnp.int32),
            diver_dir=jnp.array(1, jnp.int32),
            diver_alive=jnp.array(True, jnp.bool_),
            spear_alive=jnp.array(False, jnp.bool_),
            spear=jnp.array([0, 0, 0, 0], jnp.int32),

            # Tentacles cleared/frozen
            tentacle_len=zeros_T,
            tentacle_active=zeros_T.astype(jnp.bool_),
            tentacle_edge_wait=zeros_T,
            tentacle_dir=jnp.ones_like(zeros_T),

            # Oxygen system reset & disabled during rest
            oxygen_frames_remaining=jnp.array(cfg.hud_bar_initial_px, jnp.int32),
            oxygen_line_active=jnp.array(False, jnp.bool_),
            oxygen_line_x=jnp.array(-1, jnp.int32),
            oxygen_drop_timer=next_timer,
            oxygen_line_ttl=jnp.array(0, jnp.int32),
            oxygen_contact_counter=jnp.array(0, jnp.int32),

            # Shark reset to top lane and alive
            shark_lane=jnp.array(0, jnp.int32),
            shark_y=self.config.shark_lanes_y[0],
            shark_alive=jnp.array(True, jnp.bool_),

            rng=rng1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_and_consume_life(self, state: NameThisGameState) -> Tuple[NameThisGameState, chex.Array]:
        """If a death condition happened this frame, consume a life.
           If lives remain, soft reset to REST; else it's game over.
           Returns (possibly-updated state, game_over_flag)."""
        # Original death conditions (before lives)
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
    def _update_round(self, state: NameThisGameState) -> NameThisGameState:
        """Check conditions to advance to the next round and apply difficulty scaling."""
        cfg = self.config
        # Round clears when all tentacles are destroyed and shark has been reset required times
        round_clear = (~jnp.any(state.tentacle_active)) & (state.shark_resets_this_round >= cfg.round_clear_shark_resets)
        def _new_round(s: NameThisGameState) -> NameThisGameState:
            new_round_idx = s.round + 1
            T = self.config.max_tentacles
            zeros_T = jnp.zeros((T,), dtype=jnp.int32)
            # keep dirs positive; stacks empty
            return s._replace(
                round=new_round_idx,
                shark_resets_this_round=jnp.array(0, dtype=jnp.int32),
                tentacle_len=zeros_T,
                tentacle_active=zeros_T.astype(jnp.bool_),
                tentacle_edge_wait=zeros_T,
                tentacle_dir=jnp.ones_like(zeros_T),
            )
        state = jax.lax.cond(round_clear, _new_round, lambda s: s, state)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _shark_speed_for_lane(self, round_idx: chex.Array, lane: chex.Array) -> chex.Array:
        """Absolute shark speed (px/frame) per wave & lane.

        Pattern:
          - r = 0: all lanes 1 px/frame.
          - r >= 1: lanes at/under `speed_progression_start_lane_for_2` (and deeper) are 2 px.
            Each next wave shifts that 2 px boundary one lane earlier (toward the surface).
          - After all lanes are 2 px (at r = 1 + start_lane_for_2), every wave promotes
            one additional *bottom-most* lane to the next tier (3, then 4, then 5, ...).
            After `nlanes` waves, *all* lanes have that higher tier, then the process repeats
            for the next tier, with no upper cap.
        """
        cfg = self.config
        # ints for JAX
        nlanes_i = jnp.array(cfg.shark_lanes_y.shape[0], jnp.int32)
        r = jnp.maximum(round_idx, jnp.array(0, jnp.int32))
        lane_i = lane.astype(jnp.int32)

        # Rank from bottom: b=0 bottom lane, b=nlanes-1 top lane
        b = (nlanes_i - jnp.array(1, jnp.int32)) - lane_i

        # --- Phase 1: reach 2 px everywhere ---
        # How many bottom lanes are already at least 2 px in this wave?
        # r=0 -> 0 lanes; r=1 -> (nlanes - start2) lanes; then +1 lane per wave until all lanes.
        start2 = jnp.array(cfg.speed_progression_start_lane_for_2, jnp.int32)
        k2 = jnp.where(
            r == 0,
            jnp.array(0, jnp.int32),
            jnp.clip(nlanes_i - start2 + (r - jnp.array(1, jnp.int32)), 0, nlanes_i),
        )
        is_ge2 = (b < k2).astype(jnp.int32)

        # Wave at which all lanes reached 2 px
        r_all2 = jnp.array(1, jnp.int32) + start2  # e.g., start2=3 -> r_all2=4

        # --- Phase 2+: add tiers above 2, bottom-up, one lane per wave, repeating every nlanes ---
        # Number of waves since all lanes reached 2
        t = r - r_all2  # can be negative
        # For a given lane rank b, number of *extra* tiers above 2 it has received:
        # extra = max(0, floor((t - 1 - b)/nlanes) + 1)
        extra = jnp.maximum(
            jnp.array(0, jnp.int32),
            jnp.floor_divide(t - jnp.array(1, jnp.int32) - b, nlanes_i) + jnp.array(1, jnp.int32)
        )

        # Final multiplier: 1 + (1 if >=2 else 0) + extra tiers above 2
        mult = jnp.array(1, jnp.int32) + is_ge2 + extra

        return (mult * jnp.array(cfg.shark_base_speed, jnp.int32)).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: NameThisGameState) -> jnp.bool_:
        """Game ends only when you have no lives left."""
        return state.lives_remaining <= 0

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: NameThisGameState, action: chex.Array) -> Tuple[NameThisGameObservation, NameThisGameState, float, bool, NameThisGameInfo]:
        """Run one time-step of the game given an action. Returns (observation, new_state, reward, done, info)."""
        prev_state = state
        # Interpret action into move direction and fire press
        move_dir, fire_pressed = self._interpret_action(state, action)
        # Move diver
        state = self._move_diver(state, move_dir)
        # Check if we can shoot this frame: just pressed fire
        just_pressed = fire_pressed & (~state.fire_button_prev)
        can_shoot = just_pressed & state.diver_alive
        # Spawn spear if allowed
        state = jax.lax.cond(can_shoot, lambda s: self._spawn_spear(s), lambda s: s, state)
        # Move boat (every N frames, bounce at edges)
        state = self._move_boat(state)
        # Update the HUD bars
        state = self._update_bars_and_rest(state, just_pressed)
        # Handle oxygen line spawns
        state = self._spawn_or_update_oxygen_line(state)
        # Move spear
        state = self._move_spear(state)
        # Move shark and possibly drop lane or mark escaped
        state = self._move_shark(state)
        # Move tentacles (discrete update: grow or move exactly one tentacle)
        state = self._update_one_tentacle(state)
        # Handle collisions
        state = self._check_spear_shark_collision(state)
        state = self._check_spear_tentacle_collision(state)
        state = self._check_diver_hazard(state)
        # Update oxygen status (decrement or refill if diver picks line)
        state = self._update_oxygen(state)
        # Check round progression and update difficulty if needed
        state = self._update_round(state)
        # Handle life loss & soft reset if we died this frame
        state, game_over = self._check_and_consume_life(state)
        # Compute observation, reward, done, and info
        observation = self._get_observation(state)
        done = game_over
        # Compute observation, reward, done, and info
        observation = self._get_observation(state)
        done = self._get_done(state)
        # Calculate reward as score gain in this step
        new_reward = (state.score - prev_state.score).astype(jnp.float32)
        state = state._replace(reward=new_reward)
        all_rewards = self._get_all_reward(prev_state, state)
        info = self._get_info(state, all_rewards)
        return observation, state, state.reward.astype(jnp.float32), done, info

# Keyboard control for manual play (optional utility functions)
def get_human_action() -> chex.Array:
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

# Optional: main loop for playing manually using Pygame
def main():
    config = NameThisGameConfig()
    pygame.init()
    screen = pygame.display.set_mode((config.screen_width * config.scaling_factor, config.screen_height * config.scaling_factor))
    pygame.display.set_caption("NameThisGame")
    clock = pygame.time.Clock()
    game = JaxNameThisGame(config=config)
    renderer = Renderer_NameThisGame(config=config)
    jitted_reset = jax.jit(game.reset)
    jitted_step = jax.jit(game.step)
    obs, state = jitted_reset()
    running = True
    frame_by_frame = False
    counter = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                frame_by_frame = not frame_by_frame
            elif frame_by_frame and event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                # advance one frame in frame-by-frame mode
                action = get_human_action()
                obs, state, reward, done, info = jitted_step(state, action)
                if bool(jax.device_get(done)):
                    running = False
        if not frame_by_frame:
            action = get_human_action()
            obs, state, reward, done, info = jitted_step(state, action)
            if bool(jax.device_get(done)):
                running = False
        # Render game state to screen
        raster = renderer.render(state)
        frame_np = np.array(jax.device_get(raster), dtype=np.uint8)  # (H,W,3)
        frame_np = np.transpose(frame_np, (1, 0, 2))  # (W,H,3) for pygame
        surface = pygame.surfarray.make_surface(frame_np)
        if config.scaling_factor != 1:
            surface = pygame.transform.scale(surface, (config.screen_width * config.scaling_factor, config.screen_height * config.scaling_factor))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
        counter += 1
    pygame.quit()

if __name__ == "__main__":
    main()
