import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Optional, NamedTuple, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.lax
import chex

import pygame  # used only for human control in main loop
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
    diver_speed_px: int = 2
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
    tentacle_ys: jnp.ndarray = field(default_factory = lambda: jnp.array([97, 104, 111, 118, 125, 132, 139, 146, 153, 160],dtype=jnp.int32))
    tentacle_width: int = 4
    tentacle_amplitude: int = 12        # horizontal swing amplitude (px)
    tentacle_phase_speed: float = 0.2  # phase increment (rad/frame)
    tentacle_extend_speed: int = 1     # extension speed (px/frame)
    tentacle_tip_hitbox_h: int = 4     # height of tip region for collisions
    tentacle_max_length_px: int = 110  # maximum extension length (px)
    tentacle_destroy_points: int = 500
    # Oxygen supply line
    oxygen_full: int = 1200                 # full oxygen frames (~20 seconds at 60 FPS)
    oxygen_pickup_radius: int = 4           # horizontal radius for diver to grab oxygen line (px)
    oxygen_drop_min_interval: int = 240     # minimum frames between oxygen line drops
    oxygen_drop_max_interval: int = 480     # maximum frames between oxygen line drops
    oxygen_line_width: int = 1
    oxygen_y: int = 57
    oxygen_line_ttl_frames: int = 100
    oxygen_contact_every_n_frames: int = 3
    oxygen_contact_points: int = 25
    # Round progression
    round_clear_shark_resets: int = 3
    speed_increase_per_round_shark: int = 1
    speed_increase_per_round_tentacle: int = 1
    oxygen_interval_multiplier_per_round: float = 1.2
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

    # Boat
    boat_x: chex.Array           # boat's left x position (int32)
    boat_dx: chex.Array          # boat's horizontal direction/speed sign (-1 or +1) (int32)
    boat_move_counter: chex.Array  # counts frames; boat moves when counter % N == 0 (int32)

    # Diver (player)
    diver_x: chex.Array           # diver's x position (int32)
    diver_y: chex.Array           # diver's y position (int32, constant = floor)
    diver_alive: chex.Array       # diver alive flag (bool)
    fire_button_prev: chex.Array  # whether fire was pressed in previous frame (bool)

    # Shark (enemy)
    shark_x: chex.Array           # shark's x position (int32)
    shark_y: chex.Array           # shark's y position (int32)
    shark_dx: chex.Array          # shark's horizontal velocity (int32)
    shark_lane: chex.Array        # current lane index (int32, 0=top)
    shark_alive: chex.Array       # shark alive/active flag (bool)

    # Tentacles (octopus arms)
    tentacle_base_x: chex.Array   # base x positions for tentacles (int32, shape (max_tentacles,))
    tentacle_height: chex.Array   # current extension length of each tentacle (int32, shape (max_tentacles,))
    tentacle_phase: chex.Array    # current phase angle for each tentacle's oscillation (float32, shape (max_tentacles,))
    tentacle_active: chex.Array   # tentacle active (not destroyed) flags (bool, shape (max_tentacles,))

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
        sprite_names = ["diver", "shark", "tentacle", "oxygen_line", "background", "kraken", "boat"]
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
            diver_sprite = _solid_sprite(cfg.diver_width, cfg.diver_height, (0, 255, 0))  # green diver
        raster = jax.lax.cond(
            state.diver_alive,
            lambda r: aj.render_at(r, state.diver_x, state.diver_y, diver_sprite),
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

        # Draw tentacles (vertical pillars for each active tentacle)
        # Use purple color for tentacles
        tent_color = (180, 0, 255)
        # Pre-compute a full-length sprite template for tentacles
        full_tentacle_rgb = jnp.broadcast_to(jnp.array(tent_color, dtype=jnp.uint8), (cfg.tentacle_max_length_px, cfg.tentacle_width, 3))
        idx_array = jnp.arange(cfg.tentacle_max_length_px)[:, None]  # shape (max_length, 1)
        def _draw_tentacle(i, ras):
            active = state.tentacle_active[i]
            # Compute current tentacle X (left coordinate) from base + sinusoidal offset
            base = state.tentacle_base_x[i]
            phase = state.tentacle_phase[i]
            x_center = base + jnp.rint(jnp.sin(phase) * cfg.tentacle_amplitude).astype(jnp.int32)
            left_x = x_center - (cfg.tentacle_width // 2)
            tip_length = state.tentacle_height[i].astype(jnp.int32)

            def _do_draw(r):
                alpha_mask = (idx_array < tip_length)  # (max_len, 1)
                # broadcast across tentacle width and add channel dim
                alpha_mask = jnp.broadcast_to(alpha_mask,
                                              (cfg.tentacle_max_length_px, cfg.tentacle_width))
                alpha = jnp.where(alpha_mask, 255, 0).astype(jnp.uint8)[..., None]  # (max_len, width, 1)
                tent_rgba = jnp.concatenate([full_tentacle_rgb, alpha], axis=-1)  # (max_len, width, 4)
                return aj.render_at(r, left_x, cfg.tentacle_ys[0], tent_rgba)
            return jax.lax.cond(active, _do_draw, lambda r: r, ras)
        raster = jax.lax.fori_loop(0, cfg.max_tentacles, _draw_tentacle, raster)

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
        """Start a new game. Returns initial observation and state."""
        cfg = self.config
        # Split PRNG for different initial randomizations
        key, sub_key_dir, sub_key_oxy, sub_key_phase = jax.random.split(key, 4)
        # Boat starts centered, moving right
        init_boat_x = jnp.array(cfg.screen_width // 2 - cfg.boat_width // 2, dtype=jnp.int32)
        init_boat_dx = jnp.array(1, dtype=jnp.int32)   # +1 = right, -1 = left
        init_boat_counter = jnp.array(0, dtype=jnp.int32)
        # Initial diver position
        init_diver_x = jnp.array(cfg.screen_width//2, dtype=jnp.int32)
        init_diver_y = jnp.array(cfg.diver_y_floor, dtype=jnp.int32)
        # Choose initial shark direction randomly (True = from left to right, False = from right to left)
        go_left = jax.random.bernoulli(sub_key_dir)  # bool
        init_shark_lane = jnp.array(0, dtype=jnp.int32)
        init_shark_y = cfg.shark_lanes_y[init_shark_lane]
        init_shark_x = jnp.where(go_left, -cfg.shark_width, cfg.screen_width)
        init_shark_speed = cfg.shark_base_speed  # base speed for lane 0
        init_shark_dx = jnp.where(go_left, init_shark_speed, -init_shark_speed)
        # Tentacles: start all active, no extension
        init_tent_height = jnp.zeros((cfg.max_tentacles,), dtype=jnp.int32)
        # Random initial phase for oscillation of each tentacle
        init_tent_phase = jax.random.uniform(sub_key_phase, (cfg.max_tentacles,), minval=0.0, maxval=2*jnp.pi, dtype=jnp.float32)
        # Base X positions (could randomize slightly if desired, but use defaults for initial round)
        init_tent_base_x = cfg.tentacle_base_x
        init_tent_active = jnp.ones((cfg.max_tentacles,), dtype=jnp.bool_)
        empty_spear = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        # Oxygen: full at start, no line active
        init_oxygen = jnp.array(cfg.oxygen_full, dtype=jnp.int32)
        init_oxygen_line_active = jnp.array(False, dtype=jnp.bool_)
        init_oxygen_line_x = jnp.array(-1, dtype=jnp.int32)
        # Schedule first oxygen drop after random interval
        # Draw random frames until next drop
        oxygen_interval_factor = cfg.oxygen_interval_multiplier_per_round ** 0  # round 0 factor = 1.0
        min_int = int(cfg.oxygen_drop_min_interval * oxygen_interval_factor)
        max_int = int(cfg.oxygen_drop_max_interval * oxygen_interval_factor)
        init_drop_timer = jax.random.randint(sub_key_oxy, (), min_int, max_int + 1, dtype=jnp.int32)

        # Assemble initial state
        state = NameThisGameState(
            score=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(0, dtype=jnp.int32),
            round=jnp.array(0, dtype=jnp.int32),
            shark_resets_this_round=jnp.array(0, dtype=jnp.int32),
            boat_x=init_boat_x,
            boat_dx=init_boat_dx,
            boat_move_counter=init_boat_counter,
            diver_x=init_diver_x,
            diver_y=init_diver_y,
            diver_alive=jnp.array(True, dtype=jnp.bool_),
            fire_button_prev=jnp.array(False, dtype=jnp.bool_),
            shark_x=init_shark_x.astype(jnp.int32),
            shark_y=init_shark_y.astype(jnp.int32),
            shark_dx=jnp.array(init_shark_dx, dtype=jnp.int32),
            shark_lane=init_shark_lane,
            shark_alive=jnp.array(True, dtype=jnp.bool_),
            tentacle_base_x=init_tent_base_x,
            tentacle_height=init_tent_height,
            tentacle_phase=init_tent_phase,
            tentacle_active=init_tent_active,
            spear=empty_spear,
            spear_alive=jnp.array(False, dtype=jnp.bool_),
            oxygen_frames_remaining=init_oxygen,
            oxygen_line_active=init_oxygen_line_active,
            oxygen_line_x=init_oxygen_line_x,
            oxygen_drop_timer=init_drop_timer,
            oxygen_line_ttl=jnp.array(0, dtype=jnp.int32),  # <—
            oxygen_contact_counter=jnp.array(0, dtype=jnp.int32),  # <—
            rng=key,  # carry the remaining RNG key
        )
        # Create initial observation from state
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
        return spaces.Dict({
            "score": spaces.Box(low=0, high=(10**cfg.max_digits_for_score) - 1, shape=(), dtype=jnp.int32),
            "diver": entity_space(n=1, w_max=cfg.diver_width, h_max=cfg.diver_height),
            "shark": entity_space(n=1, w_max=cfg.shark_width, h_max=cfg.shark_height),
            "spear": entity_space(n=1, w_max=cfg.spear_width, h_max=cfg.spear_height),
            "tentacles": entity_space(n=cfg.max_tentacles, w_max=cfg.tentacle_width, h_max=cfg.tentacle_max_length_px),
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
        """Construct an observation NamedTuple from the game state."""
        cfg = self.config
        # Diver entity (single)
        diver_alive = state.diver_alive.astype(jnp.int32)
        diver_pos = EntityPosition(
            x=jnp.atleast_1d(state.diver_x),
            y=jnp.atleast_1d(state.diver_y),
            width=jnp.atleast_1d(jnp.array(cfg.diver_width, dtype=jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.diver_height, dtype=jnp.int32)),
            alive=jnp.atleast_1d(diver_alive),
        )
        # Shark entity (single)
        shark_alive = state.shark_alive.astype(jnp.int32)
        shark_pos = EntityPosition(
            x=jnp.atleast_1d(state.shark_x),
            y=jnp.atleast_1d(state.shark_y),
            width=jnp.atleast_1d(jnp.array(cfg.shark_width, dtype=jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.shark_height, dtype=jnp.int32)),
            alive=jnp.atleast_1d(shark_alive),
        )
        # Spear entity (single)
        spear_alive = state.spear_alive.astype(jnp.int32)
        spear_pos = EntityPosition(
            x=jnp.atleast_1d(state.spear[0]),
            y=jnp.atleast_1d(state.spear[1]),
            width=jnp.atleast_1d(jnp.array(cfg.spear_width, dtype=jnp.int32)),
            height=jnp.atleast_1d(jnp.array(cfg.spear_height, dtype=jnp.int32)),
            alive=jnp.atleast_1d(spear_alive),
        )
        # Tentacles entities
        tent_alive = state.tentacle_active.astype(jnp.int32)
        # Tentacle x position (left coordinate) and dimensions for observation
        # Represent each tentacle as the column from top (y=0) down to current tip
        tentacle_x_center = state.tentacle_base_x + jnp.rint(jnp.sin(state.tentacle_phase) * cfg.tentacle_amplitude).astype(jnp.int32)
        tentacle_left = tentacle_x_center - 1  # left edge given width=3
        tentacle_y = jnp.where(state.tentacle_active, jnp.zeros_like(state.tentacle_height), -1)  # top of tentacle (0 if active, -1 if not)
        tentacle_w = jnp.where(state.tentacle_active, jnp.full((cfg.max_tentacles,), cfg.tentacle_width, dtype=jnp.int32), 0)
        tentacle_h = jnp.where(state.tentacle_active, state.tentacle_height, 0)
        tentacle_pos = EntityPosition(tentacle_left, tentacle_y, tentacle_w, tentacle_h, tent_alive)
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
        """Update diver's horizontal position based on move_dir (-1, 0, +1)."""
        cfg = self.config
        new_x = jnp.clip(state.diver_x + move_dir * cfg.diver_speed_px, 0, cfg.screen_width - cfg.diver_width)
        return state._replace(diver_x=new_x)

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
        x = state.shark_x
        dx = state.shark_dx
        new_x = x + dx

        off_right = (dx > 0) & (new_x >= cfg.screen_width)
        off_left = (dx < 0) & ((new_x + cfg.shark_width) <= 0)

        def _drop_lane(s: NameThisGameState, going_left: bool) -> NameThisGameState:
            new_lane = s.shark_lane + 1
            last_idx = cfg.shark_lanes_y.shape[0] - 1

            def _lane_exists(st: NameThisGameState) -> NameThisGameState:
                # Safe index even if compiled with out-of-range values
                safe_idx = jnp.clip(new_lane, 0, last_idx)
                new_y = jnp.take(cfg.shark_lanes_y, safe_idx, mode="clip")
                base_speed = cfg.shark_base_speed + st.round * cfg.speed_increase_per_round_shark
                speed = jnp.where(s.shark_lane>2, base_speed*2, base_speed)
                new_dx_val = jnp.where(going_left, jnp.array(speed, jnp.int32), -jnp.array(speed, jnp.int32))
                new_x_val = jnp.where(going_left, -cfg.shark_width, cfg.screen_width)
                return st._replace(shark_x=new_x_val, shark_y=new_y, shark_dx=new_dx_val, shark_lane=new_lane)

            def _no_lane(st: NameThisGameState) -> NameThisGameState:
                return st._replace(shark_alive=jnp.array(False, dtype=jnp.bool_))

            has_lane = new_lane < (last_idx + 1)
            return jax.lax.cond(has_lane, _lane_exists, _no_lane, s)

        state = state._replace(shark_x=new_x)
        state = jax.lax.cond(off_right, lambda s: _drop_lane(s, going_left=False), lambda s: s, state)
        state = jax.lax.cond(off_left, lambda s: _drop_lane(s, going_left=True), lambda s: s, state)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _move_tentacles(self, state: NameThisGameState) -> NameThisGameState:
        """Extend tentacles downward and oscillate their horizontal position."""
        cfg = self.config
        # Compute current extension speed based on round (increases by 1 each round)
        extend_speed = cfg.tentacle_extend_speed + state.round * cfg.speed_increase_per_round_tentacle
        # Update phase for all tentacles (even if inactive, phase can freeze but it won't matter)
        new_phase = (state.tentacle_phase + cfg.tentacle_phase_speed) % (2 * jnp.pi)
        # Increase height for active tentacles, capped at max length
        current_height = state.tentacle_height
        target_height = jnp.minimum(current_height + extend_speed, jnp.array(cfg.tentacle_max_length_px, dtype=jnp.int32))
        new_height = jnp.where(state.tentacle_active, target_height, current_height)
        return state._replace(tentacle_phase=new_phase, tentacle_height=new_height)

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
            base_speed = cfg.shark_base_speed + s.round * cfg.speed_increase_per_round_shark
            reset_dx = jnp.where(go_left, base_speed, -base_speed).astype(jnp.int32)
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

        # spear rect (scalars)
        sl = state.spear[0]
        sr = state.spear[0] + cfg.spear_width
        st = state.spear[1]
        sb = state.spear[1] + cfg.spear_height

        # tentacle tip regions (vectors length max_tentacles)
        x_center = state.tentacle_base_x + jnp.rint(jnp.sin(state.tentacle_phase) * cfg.tentacle_amplitude).astype(
            jnp.int32)
        tl = x_center - 1
        tr = tl + cfg.tentacle_width
        tip_y = state.tentacle_height
        tt = tip_y - cfg.tentacle_tip_hitbox_h
        tb = tip_y

        # vectorized overlap against all tentacles
        over_x = (sl < tr) & (sr > tl)
        over_y = (st < tb) & (sb > tt)
        hit_vec = state.spear_alive & state.tentacle_active & over_x & over_y  # (max_tentacles,)

        # per-tentacle deactivation + points
        points = jnp.sum(jnp.where(hit_vec, jnp.array(cfg.tentacle_destroy_points, jnp.int32), 0))
        new_score = state.score + points
        new_tent_active = jnp.logical_and(state.tentacle_active, jnp.logical_not(hit_vec))

        # spear is consumed if it hit any tentacle
        spear_hit_any = jnp.any(hit_vec)
        new_spear_alive = jnp.logical_and(state.spear_alive, jnp.logical_not(spear_hit_any))

        return state._replace(
            score=new_score,
            tentacle_active=new_tent_active,
            spear_alive=new_spear_alive,
        )
    @partial(jax.jit, static_argnums=(0,))
    def _check_diver_hazard(self, state: NameThisGameState) -> NameThisGameState:
        """Check if diver is hit by a tentacle tip. If so, mark diver as dead."""
        cfg = self.config
        # Compute horizontal overlap of diver with each tentacle tip
        diver_left = state.diver_x
        diver_right = state.diver_x + cfg.diver_width
        tentacle_x_center = state.tentacle_base_x + jnp.rint(jnp.sin(state.tentacle_phase) * cfg.tentacle_amplitude).astype(jnp.int32)
        tentacle_left = tentacle_x_center - 1
        tentacle_right = tentacle_left + cfg.tentacle_width
        overlap_x = (diver_left < tentacle_right) & (diver_right > tentacle_left)
        # Check if any active tentacle tip has reached diver's level or beyond and overlaps horizontally
        tip_reached = state.tentacle_height >= cfg.diver_y_floor
        hazard = jnp.any(state.tentacle_active & tip_reached & overlap_x)
        # If hazard, diver dies
        return state._replace(diver_alive=jnp.where(hazard, jnp.array(False, dtype=jnp.bool_), state.diver_alive))

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
                factor = cfg.oxygen_interval_multiplier_per_round ** (st.round.astype(jnp.float32))
                min_int = (cfg.oxygen_drop_min_interval * factor).astype(jnp.int32)
                max_int = (cfg.oxygen_drop_max_interval * factor).astype(jnp.int32)
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
        """While under the line: every N contact frames either refill to full, or if already full, add points.
        Oxygen does not drain while under the line; otherwise it drains by 1 per frame."""
        cfg = self.config
        full_oxy = jnp.array(cfg.oxygen_full, dtype=jnp.int32)

        # Is diver under the active oxygen line?
        diver_center = state.diver_x + (cfg.diver_width // 2)
        line_center = state.oxygen_line_x + (cfg.oxygen_line_width // 2)
        diver_under_line = state.oxygen_line_active & (jnp.abs(diver_center - line_center) <= cfg.oxygen_pickup_radius)

        # Count consecutive contact frames; apply effect every K frames
        cnt_next = jnp.where(diver_under_line, state.oxygen_contact_counter + 1, jnp.array(0, dtype=jnp.int32))
        apply_effect = diver_under_line & ((cnt_next % cfg.oxygen_contact_every_n_frames) == 0)

        oxygen_not_full = state.oxygen_frames_remaining < full_oxy
        # Refill to full iff effect triggers and oxygen wasn't full
        oxy_after_effect = jnp.where(apply_effect & oxygen_not_full, full_oxy, state.oxygen_frames_remaining)

        # Award points iff effect triggers and oxygen already full
        add_points = jnp.where(
            apply_effect & (~oxygen_not_full),
            jnp.array(cfg.oxygen_contact_points, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        new_score = state.score + add_points

        # No drain while under the line; otherwise drain by 1
        new_oxy = jnp.where(
            diver_under_line,
            oxy_after_effect,
            jnp.maximum(oxy_after_effect - 1, 0),
        )

        return state._replace(
            oxygen_frames_remaining=new_oxy,
            oxygen_contact_counter=cnt_next,
            score=new_score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_round(self, state: NameThisGameState) -> NameThisGameState:
        """Check conditions to advance to the next round and apply difficulty scaling."""
        cfg = self.config
        # Round clears when all tentacles are destroyed and shark has been reset required times
        round_clear = (~jnp.any(state.tentacle_active)) & (state.shark_resets_this_round >= cfg.round_clear_shark_resets)
        def _new_round(s: NameThisGameState) -> NameThisGameState:
            rng_off, rng_phase = jax.random.split(s.rng)
            new_round_idx = s.round + 1
            # Reset tentacles for new round (all active, height 0, randomize phase and possibly base positions)
            new_active = jnp.ones_like(s.tentacle_active)
            new_height = jnp.zeros_like(s.tentacle_height)
            # Optionally randomize base X positions within a small range
            offset = jax.random.randint(rng_off, s.tentacle_base_x.shape, -5, 6, dtype=jnp.int32)
            new_base_x = jnp.clip(cfg.tentacle_base_x + offset, 1, cfg.screen_width - cfg.tentacle_width - 1)
            # Update shark speed for new round (current dx sign * increased base speed)
            current_sign = jnp.where(s.shark_dx < 0, -1, 1)
            new_shark_speed = cfg.shark_base_speed + new_round_idx * cfg.speed_increase_per_round_shark
            new_shark_dx = current_sign * jnp.array(new_shark_speed, dtype=jnp.int32)
            return s._replace(
                round=new_round_idx,
                shark_resets_this_round=jnp.array(0, dtype=jnp.int32),
                tentacle_active=new_active,
                tentacle_height=new_height,
                tentacle_phase=jax.random.uniform(s.rng, s.tentacle_phase.shape, minval=0.0, maxval=2*jnp.pi, dtype=jnp.float32),
                tentacle_base_x=new_base_x,
                shark_dx=new_shark_dx
            )
        state = jax.lax.cond(round_clear, _new_round, lambda s: s, state)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: NameThisGameState) -> jnp.bool_:
        """Determine if the game is over (terminal state)."""
        cfg = self.config
        # Condition 1: Oxygen depleted
        oxygen_out = state.oxygen_frames_remaining <= 0
        # Condition 2: Diver dead (tentacle got him)
        diver_dead = ~state.diver_alive
        # Condition 3: Shark reached the diver (left bottom without being shot)
        shark_reached_diver = ~state.shark_alive
        done = oxygen_out | diver_dead | shark_reached_diver
        return done

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
        # Handle oxygen line spawns
        state = self._spawn_or_update_oxygen_line(state)
        # Move spear
        state = self._move_spear(state)
        # Move shark and possibly drop lane or mark escaped
        state = self._move_shark(state)
        # Move tentacles (extend and wave)
        state = self._move_tentacles(state)
        # Handle collisions
        state = self._check_spear_shark_collision(state)
        state = self._check_spear_tentacle_collision(state)
        state = self._check_diver_hazard(state)
        # Update oxygen status (decrement or refill if diver picks line)
        state = self._update_oxygen(state)
        # Check round progression and update difficulty if needed
        state = self._update_round(state)
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
