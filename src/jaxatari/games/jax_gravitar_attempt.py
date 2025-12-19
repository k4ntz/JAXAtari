from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Dict as SpaceDict, Box
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering.jax_rendering_utils import RendererConfig


class GravitarConstants(NamedTuple):
    """
    Game constants following ALE Gravitar specifications.
    Based on: https://ale.farama.org/environments/gravitar/
    Manual: https://atariage.com/manual_html_page.php?SoftwareLabelID=223
    """
    # Screen and rendering
    SCREEN_HEIGHT: int = 210
    SCREEN_WIDTH: int = 160
    CHANNELS: int = 3

    # Game difficulty levels (0-4, from ALE)
    # Level 0 (mode 1): 6 ships, hardest
    # Level 1 (mode 2): 15 ships
    # Level 2 (mode 3): 6 ships, enemies don't fire
    # Level 3 (mode 4): 100 ships, practice mode
    # Level 4 (mode 5): 25 ships, no gravity
    STARTING_LIVES_BY_LEVEL: tuple = (6, 15, 6, 100, 25)
    
    # Physics constants (tuned to approximate ALE behavior)
    GRAVITY_STRENGTH: float = 0.08  # Planet gravity pull
    SUN_GRAVITY: float = 0.04  # Solar system sun gravity
    SHIP_THRUST_ACCEL: float = 0.18  # Acceleration from thrust
    SHIP_ROTATION_SPEED: float = jnp.deg2rad(6.0)  # Radians per frame
    SHIP_MAX_SPEED: float = 3.5
    SHIP_DRAG: float = 0.98  # Velocity decay factor

    # Ship dimensions and collision
    SHIP_RADIUS: float = 3.0
    SHIP_COLLISION_RADIUS: float = 4.0
    
    # Bullet parameters
    BULLET_SPEED: float = 4.0
    BULLET_LIFETIME: int = 60  # frames
    MAX_PLAYER_BULLETS: int = 4
    MAX_ENEMY_BULLETS: int = 16
    BULLET_COOLDOWN_STEPS: int = 8
    
    # Enemy parameters
    MAX_BUNKERS_PER_PLANET: int = 8
    MAX_RAMMERS_PER_PLANET: int = 4
    MAX_FUEL_DEPOTS_PER_PLANET: int = 4
    BUNKER_RADIUS: float = 4.0
    RAMMER_RADIUS: float = 3.0
    RAMMER_SPEED: float = 1.5
    BUNKER_FIRE_RATE: int = 60  # frames between shots
    BUNKER_BULLET_SPEED: float = 2.0
    
    # Saucer (solar system enemy)
    SAUCER_SPEED: float = 1.0
    SAUCER_FIRE_RATE: int = 40
    SAUCER_RADIUS: float = 4.0
    
    # Fuel system
    STARTING_FUEL: int = 10000
    LOW_FUEL_WARNING: int = 2000
    FUEL_COST_THRUST: int = 1  # per frame
    FUEL_COST_SHIELD: int = 1  # per frame
    FUEL_COST_TRACTOR: int = 1  # per frame
    FUEL_FROM_DEPOT: int = 5000
    
    # Scoring (from manual)
    SCORE_SAUCER: int = 100
    SCORE_RAMMER: int = 100
    SCORE_BUNKER: int = 250
    BONUS_SHIP_EVERY: int = 10000
    MAX_SCORE: int = 999950
    
    # Solar system completion bonuses
    BONUS_FUEL: int = 7000
    BONUS_SHIPS: int = 2
    BONUS_POINTS: int = 4000
    
    # Galaxy and solar system structure
    NUM_GALAXIES: int = 4
    SOLAR_SYSTEMS_PER_GALAXY: int = 3
    NUM_PLANETS_PER_SYSTEM: int = 4  # 3-4 regular planets + reactor base
    
    # Reactor base
    REACTOR_COUNTDOWN_START: int = 60  # seconds (frames)
    REACTOR_COUNTDOWN_MIN: int = 25  # minimum countdown in later systems
    
    # Sun parameters
    SUN_RADIUS: float = 8.0
    SUN_KILL_RADIUS: float = 12.0
    
    # Planet parameters
    PLANET_RADIUS: float = 15.0  # Distance to enter
    PLANET_GRAVITY_RANGE: float = 100.0


# ============================================================================
# State Component Classes
# ============================================================================

class BulletPoolState(NamedTuple):
    """Pool of bullets for both player and enemies."""
    x: chex.Array  # shape (max_bullets,)
    y: chex.Array
    vx: chex.Array
    vy: chex.Array
    active: chex.Array  # boolean
    lifetime: chex.Array  # frames remaining
    owner: chex.Array  # 0=player, 1=enemy


class BunkerState(NamedTuple):
    """Enemy bunkers on planets."""
    x: chex.Array  # shape (max_bunkers,)
    y: chex.Array
    active: chex.Array
    fire_timer: chex.Array


class RammerState(NamedTuple):
    """Rammer enemies that patrol planets."""
    x: chex.Array  # shape (max_rammers,)
    y: chex.Array
    vx: chex.Array
    vy: chex.Array
    active: chex.Array


class FuelDepotState(NamedTuple):
    """Fuel depots on planets."""
    x: chex.Array  # shape (max_depots,)
    y: chex.Array
    fuel_amount: chex.Array
    active: chex.Array


class PlanetState(NamedTuple):
    """State for a single planet within a solar system."""
    planet_type: chex.Array  # 0-3 for different planet types
    x: chex.Array  # Position in solar system view
    y: chex.Array
    completed: chex.Array  # All bunkers destroyed
    visited: chex.Array
    bunkers: BunkerState
    rammers: RammerState
    fuel_depots: FuelDepotState


class SaucerState(NamedTuple):
    """Enemy saucer in solar system view."""
    x: chex.Array
    y: chex.Array
    vx: chex.Array
    vy: chex.Array
    active: chex.Array
    fire_timer: chex.Array


class ReactorState(NamedTuple):
    """Reactor base state."""
    activated: chex.Array  # boolean
    countdown: chex.Array  # frames remaining
    exploding: chex.Array  # boolean


class SolarSystemState(NamedTuple):
    """State for one complete solar system."""
    system_id: chex.Array  # 0-11
    completed: chex.Array
    sun_x: chex.Array  # Sun position
    sun_y: chex.Array
    # Note: planets would be an array, but for now we'll handle current planet separately
    reactor: ReactorState
    saucer: SaucerState


class GravitarState(NamedTuple):
    """
    Complete Gravitar game state.
    
    The game has multiple phases:
    - Solar system view (location=0): Navigate between planets, avoid sun
    - Planet surface (location=1-4): Destroy bunkers, collect fuel
    - Reactor base (location=5): Navigate to core and escape
    """
    # Ship state
    ship_x: chex.Array
    ship_y: chex.Array
    ship_vx: chex.Array
    ship_vy: chex.Array
    ship_theta: chex.Array  # radians, 0=right, π/2=up
    ship_alive: chex.Array  # boolean
    ship_death_timer: chex.Array  # animation frames
    fuel: chex.Array

    # Bullets (shared pool)
    bullets: BulletPoolState
    bullet_cooldown: chex.Array

    # Current location and progress
    current_galaxy: chex.Array  # 0-3
    current_system: chex.Array  # 0-2 within galaxy
    current_location: chex.Array  # 0=solar system, 1-4=planet, 5=reactor
    
    # Current area state (changes based on location)
    solar_system: SolarSystemState
    current_planet: PlanetState  # Active when on planet surface
    
    # Game state
    score: chex.Array
    lives: chex.Array
    game_level: chex.Array  # Difficulty 0-4
    step_count: chex.Array
    
    # RNG
    rng_key: chex.PRNGKey


class GravitarObservation(NamedTuple):
    """
    Observation provided to the agent.
    Contains only observable game elements (not internal state).
    """
    # Ship state
    ship_x: chex.Array
    ship_y: chex.Array
    ship_vx: chex.Array
    ship_vy: chex.Array
    ship_theta: chex.Array
    ship_alive: chex.Array
    fuel: chex.Array
    
    # Visible enemies/objects (context-dependent on location)
    bunkers_x: chex.Array  # (max_bunkers,)
    bunkers_y: chex.Array
    bunkers_active: chex.Array
    
    rammers_x: chex.Array  # (max_rammers,)
    rammers_y: chex.Array
    rammers_active: chex.Array
    
    saucer_x: chex.Array  # scalar
    saucer_y: chex.Array
    saucer_active: chex.Array
    
    fuel_depots_x: chex.Array  # (max_depots,)
    fuel_depots_y: chex.Array
    fuel_depots_active: chex.Array
    
    # Bullets
    player_bullets_x: chex.Array
    player_bullets_y: chex.Array
    player_bullets_active: chex.Array
    
    enemy_bullets_x: chex.Array
    enemy_bullets_y: chex.Array
    enemy_bullets_active: chex.Array
    
    # Game info
    score: chex.Array
    lives: chex.Array
    current_location: chex.Array  # 0=solar system, 1-4=planet, 5=reactor


class GravitarInfo(NamedTuple):
    """Additional game information not directly observable."""
    current_galaxy: chex.Array
    current_system: chex.Array
    step_count: chex.Array
    all_rewards: chex.Array

class GravitarSprites(IntEnum):
    # Ship & bullets
    SHIP_IDLE = 0  # spaceship.npy
    SHIP_THRUST = 1  # ship_thrust.npy
    SHIP_BULLET = 2  # ship_bullet.npy

    # Enemy bullets
    ENEMY_BULLET = 3  # enemy_bullet.npy
    ENEMY_GREEN_BULLET = 4  # enemy_green_bullet.npy

    # Enemies
    ENEMY_ORANGE = 5  # enemy_orange.npy
    ENEMY_GREEN = 6  # enemy_green.npy
    ENEMY_SAUCER = 7  # saucer.npy
    ENEMY_UFO = 8  # UFO.npy

    # Explosions / crashes
    ENEMY_CRASH = 9  # enemy_crash.npy
    SAUCER_CRASH = 10  # saucer_crash.npy
    SHIP_CRASH = 11  # ship_crash.npy

    # World objects
    FUEL_TANK = 12  # fuel_tank.npy
    OBSTACLE = 13  # obstacle.npy
    SPAWN_LOC = 14  # spawn_location.npy

    # Reactor & terrain
    REACTOR = 15  # reactor.npy
    REACTOR_TERR = 16  # reactor_terrant.npy
    TERRANT1 = 17  # terrant1.npy
    TERRANT2 = 18  # terrant2.npy
    TERRANT3 = 19  # terrant_3.npy
    TERRANT4 = 20  # terrant_4.npy

    # Planets & UI
    PLANET1 = 21  # planet1.npy
    PLANET2 = 22  # planet2.npy
    PLANET3 = 23  # planet3.npy
    PLANET4 = 24  # planet4.npy
    REACTOR_DEST = 25  # reactor_destination.npy
    SCORE_UI = 26  # score.npy
    HP_UI = 27  # HP.npy
    SHIP_THRUST_BACK = 28
    # Score digits
    DIGIT_0 = 29
    DIGIT_1 = 30
    DIGIT_2 = 31
    DIGIT_3 = 32
    DIGIT_4 = 33
    DIGIT_5 = 34
    DIGIT_6 = 35
    DIGIT_7 = 36
    DIGIT_8 = 37
    DIGIT_9 = 38
    ENEMY_ORANGE_FLIPPED = 39
    SHIELD = 40

LEVEL_LAYOUTS = {
    # Level 0 (Planet 1)
    0: [
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (37, 44)},  # 158
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (82, 32)},  # 146     114
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (152, -3)},  # 112
        {'type': GravitarSprites.ENEMY_GREEN, 'coords': (22, 71)},
        {'type': GravitarSprites.FUEL_TANK, 'coords': (104, 60)},  # 174 114
    ],
    # Level 1 (Planet 2)
    1: [
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (93, 19)},
        {'type': GravitarSprites.ENEMY_ORANGE_FLIPPED, 'coords': (52, 77)},
        {'type': GravitarSprites.ENEMY_ORANGE_FLIPPED, 'coords': (8, 36)},
        {'type': GravitarSprites.ENEMY_GREEN, 'coords': (11, 60)},
        {'type': GravitarSprites.FUEL_TANK, 'coords': (29, 0)},
    ],
    # Level 2 (Planet 3)
    2: [
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (24, 38)},
        {'type': GravitarSprites.ENEMY_GREEN, 'coords': (43, 82)},
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (60, -2)},
        {'type': GravitarSprites.ENEMY_GREEN, 'coords': (108, 22)},
        {'type': GravitarSprites.FUEL_TANK, 'coords': (135, 68)},
    ],
    # Level 3 (Planet 4)
    3: [
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (88, 93 - 114 + 48)},
        {'type': GravitarSprites.ENEMY_ORANGE_FLIPPED, 'coords': (116, 73 - 114 + 51)},
        {'type': GravitarSprites.ENEMY_ORANGE, 'coords': (122, 180 - 114 + 47)},
        {'type': GravitarSprites.ENEMY_GREEN, 'coords': (76, 126 - 114 + 47)},
        {'type': GravitarSprites.FUEL_TANK, 'coords': (19, 162 - 114 + 47)},
    ],
    # Level 4 (Reactor)
    4: [],
}


class JaxGravitar(JaxEnvironment[GravitarState, GravitarObservation, GravitarInfo, GravitarConstants]):
    def __init__(self, consts: GravitarConstants | None = None, game_level: int = 0):
        self.consts = consts or GravitarConstants()
        self.game_level = jnp.clip(game_level, 0, 4)
        self.renderer = GravitarRenderer(
            RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=self.consts.CHANNELS,
                downscale=None,
            )
        )
    
    # ========================================================================
    # Helper Methods - State Initialization
    # ========================================================================
    
    def _create_empty_bullet_pool(self) -> BulletPoolState:
        """Create an empty bullet pool for player and enemy bullets."""
        max_bullets = self.consts.MAX_PLAYER_BULLETS + self.consts.MAX_ENEMY_BULLETS
        return BulletPoolState(
            x=jnp.zeros(max_bullets, dtype=jnp.float32),
            y=jnp.zeros(max_bullets, dtype=jnp.float32),
            vx=jnp.zeros(max_bullets, dtype=jnp.float32),
            vy=jnp.zeros(max_bullets, dtype=jnp.float32),
            active=jnp.zeros(max_bullets, dtype=jnp.bool_),
            lifetime=jnp.zeros(max_bullets, dtype=jnp.int32),
            owner=jnp.zeros(max_bullets, dtype=jnp.int32)
        )
    
    def _create_empty_bunkers(self) -> BunkerState:
        """Create empty bunker state."""
        return BunkerState(
            x=jnp.zeros(self.consts.MAX_BUNKERS_PER_PLANET, dtype=jnp.float32),
            y=jnp.zeros(self.consts.MAX_BUNKERS_PER_PLANET, dtype=jnp.float32),
            active=jnp.zeros(self.consts.MAX_BUNKERS_PER_PLANET, dtype=jnp.bool_),
            fire_timer=jnp.zeros(self.consts.MAX_BUNKERS_PER_PLANET, dtype=jnp.int32)
        )
    
    def _create_empty_rammers(self) -> RammerState:
        """Create empty rammer state."""
        return RammerState(
            x=jnp.zeros(self.consts.MAX_RAMMERS_PER_PLANET, dtype=jnp.float32),
            y=jnp.zeros(self.consts.MAX_RAMMERS_PER_PLANET, dtype=jnp.float32),
            vx=jnp.zeros(self.consts.MAX_RAMMERS_PER_PLANET, dtype=jnp.float32),
            vy=jnp.zeros(self.consts.MAX_RAMMERS_PER_PLANET, dtype=jnp.float32),
            active=jnp.zeros(self.consts.MAX_RAMMERS_PER_PLANET, dtype=jnp.bool_)
        )
    
    def _create_empty_fuel_depots(self) -> FuelDepotState:
        """Create empty fuel depot state."""
        return FuelDepotState(
            x=jnp.zeros(self.consts.MAX_FUEL_DEPOTS_PER_PLANET, dtype=jnp.float32),
            y=jnp.zeros(self.consts.MAX_FUEL_DEPOTS_PER_PLANET, dtype=jnp.float32),
            fuel_amount=jnp.full(self.consts.MAX_FUEL_DEPOTS_PER_PLANET, self.consts.FUEL_FROM_DEPOT, dtype=jnp.int32),
            active=jnp.zeros(self.consts.MAX_FUEL_DEPOTS_PER_PLANET, dtype=jnp.bool_)
        )
    
    def _create_empty_planet(self) -> PlanetState:
        """Create an empty planet state."""
        return PlanetState(
            planet_type=jnp.int32(0),
            x=jnp.float32(80.0),
            y=jnp.float32(80.0),
            completed=jnp.bool_(False),
            visited=jnp.bool_(False),
            bunkers=self._create_empty_bunkers(),
            rammers=self._create_empty_rammers(),
            fuel_depots=self._create_empty_fuel_depots()
        )
    
    def _create_empty_saucer(self) -> SaucerState:
        """Create an inactive saucer."""
        return SaucerState(
            x=jnp.float32(0.0),
            y=jnp.float32(0.0),
            vx=jnp.float32(0.0),
            vy=jnp.float32(0.0),
            active=jnp.bool_(False),
            fire_timer=jnp.int32(0)
        )
    
    def _create_empty_reactor(self) -> ReactorState:
        """Create an inactive reactor."""
        return ReactorState(
            activated=jnp.bool_(False),
            countdown=jnp.int32(self.consts.REACTOR_COUNTDOWN_START),
            exploding=jnp.bool_(False)
        )
    
    def _create_initial_solar_system(self, key: jax.Array) -> SolarSystemState:
        """Create the initial solar system state."""
        c = self.consts
        
        # Saucer spawns active in the solar system
        saucer = SaucerState(
            x=jnp.float32(c.SCREEN_WIDTH / 4),
            y=jnp.float32(c.SCREEN_HEIGHT / 4),
            vx=jnp.float32(0.0),
            vy=jnp.float32(0.0),
            active=jnp.bool_(True),
            fire_timer=jnp.int32(c.SAUCER_FIRE_RATE)
        )
        
        return SolarSystemState(
            system_id=jnp.int32(0),
            completed=jnp.bool_(False),
            sun_x=jnp.float32(self.consts.SCREEN_WIDTH / 2),
            sun_y=jnp.float32(self.consts.SCREEN_HEIGHT / 2),
            reactor=self._create_empty_reactor(),
            saucer=saucer
        )

    def action_space(self) -> Discrete:
        # Use the standard 18-Discrete action space per JAXAtariAction
        return Discrete(18)

    def observation_space(self) -> SpaceDict:
        """Return the observation space structure."""
        c = self.consts
        return SpaceDict(
            {
                "ship_x": Box(0.0, float(c.SCREEN_WIDTH), shape=()),
                "ship_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=()),
                "ship_vx": Box(-c.SHIP_MAX_SPEED, c.SHIP_MAX_SPEED, shape=()),
                "ship_vy": Box(-c.SHIP_MAX_SPEED, c.SHIP_MAX_SPEED, shape=()),
                "ship_theta": Box(-jnp.pi, jnp.pi, shape=()),
                "ship_alive": Box(0, 1, shape=(), dtype=jnp.bool_),
                "fuel": Box(0, c.STARTING_FUEL * 2, shape=(), dtype=jnp.float32),
                "bunkers_x": Box(0.0, float(c.SCREEN_WIDTH), shape=(c.MAX_BUNKERS_PER_PLANET,)),
                "bunkers_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=(c.MAX_BUNKERS_PER_PLANET,)),
                "bunkers_active": Box(0, 1, shape=(c.MAX_BUNKERS_PER_PLANET,), dtype=jnp.bool_),
                "rammers_x": Box(0.0, float(c.SCREEN_WIDTH), shape=(c.MAX_RAMMERS_PER_PLANET,)),
                "rammers_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=(c.MAX_RAMMERS_PER_PLANET,)),
                "rammers_active": Box(0, 1, shape=(c.MAX_RAMMERS_PER_PLANET,), dtype=jnp.bool_),
                "saucer_x": Box(0.0, float(c.SCREEN_WIDTH), shape=()),
                "saucer_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=()),
                "saucer_active": Box(0, 1, shape=(), dtype=jnp.bool_),
                "fuel_depots_x": Box(0.0, float(c.SCREEN_WIDTH), shape=(c.MAX_FUEL_DEPOTS_PER_PLANET,)),
                "fuel_depots_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=(c.MAX_FUEL_DEPOTS_PER_PLANET,)),
                "fuel_depots_active": Box(0, 1, shape=(c.MAX_FUEL_DEPOTS_PER_PLANET,), dtype=jnp.bool_),
                "player_bullets_x": Box(0.0, float(c.SCREEN_WIDTH), shape=(c.MAX_PLAYER_BULLETS,)),
                "player_bullets_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=(c.MAX_PLAYER_BULLETS,)),
                "player_bullets_active": Box(0, 1, shape=(c.MAX_PLAYER_BULLETS,), dtype=jnp.bool_),
                "enemy_bullets_x": Box(0.0, float(c.SCREEN_WIDTH), shape=(c.MAX_ENEMY_BULLETS,)),
                "enemy_bullets_y": Box(0.0, float(c.SCREEN_HEIGHT), shape=(c.MAX_ENEMY_BULLETS,)),
                "enemy_bullets_active": Box(0, 1, shape=(c.MAX_ENEMY_BULLETS,), dtype=jnp.bool_),
                "score": Box(0, c.MAX_SCORE, shape=(), dtype=jnp.int32),
                "lives": Box(0, 100, shape=(), dtype=jnp.int32),
                "current_location": Box(0, 5, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> Box:
        return Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, self.consts.CHANNELS),
            dtype=jnp.uint8,
        )

    def _get_observation(self, state: GravitarState) -> GravitarObservation:
        """Extract observation from state."""
        c = self.consts
        
        # Get current planet state (if on planet)
        planet = state.current_planet
        
        # Separate player and enemy bullets
        player_mask = (state.bullets.owner == 0) & state.bullets.active
        enemy_mask = (state.bullets.owner == 1) & state.bullets.active
        
        # Extract player bullets (first MAX_PLAYER_BULLETS slots)
        player_bullets_x = jnp.where(player_mask[:c.MAX_PLAYER_BULLETS], 
                                     state.bullets.x[:c.MAX_PLAYER_BULLETS], 
                                     jnp.float32(0.0))
        player_bullets_y = jnp.where(player_mask[:c.MAX_PLAYER_BULLETS], 
                                     state.bullets.y[:c.MAX_PLAYER_BULLETS], 
                                     jnp.float32(0.0))
        player_bullets_active = player_mask[:c.MAX_PLAYER_BULLETS]
        
        # Extract enemy bullets (remaining slots)
        enemy_bullets_x = jnp.where(enemy_mask[c.MAX_PLAYER_BULLETS:], 
                                    state.bullets.x[c.MAX_PLAYER_BULLETS:], 
                                    jnp.float32(0.0))
        enemy_bullets_y = jnp.where(enemy_mask[c.MAX_PLAYER_BULLETS:], 
                                    state.bullets.y[c.MAX_PLAYER_BULLETS:], 
                                    jnp.float32(0.0))
        enemy_bullets_active = enemy_mask[c.MAX_PLAYER_BULLETS:]
        
        return GravitarObservation(
            ship_x=state.ship_x,
            ship_y=state.ship_y,
            ship_vx=state.ship_vx,
            ship_vy=state.ship_vy,
            ship_theta=state.ship_theta,
            ship_alive=state.ship_alive,
            fuel=state.fuel,
            bunkers_x=planet.bunkers.x,
            bunkers_y=planet.bunkers.y,
            bunkers_active=planet.bunkers.active,
            rammers_x=planet.rammers.x,
            rammers_y=planet.rammers.y,
            rammers_active=planet.rammers.active,
            saucer_x=state.solar_system.saucer.x,
            saucer_y=state.solar_system.saucer.y,
            saucer_active=state.solar_system.saucer.active,
            fuel_depots_x=planet.fuel_depots.x,
            fuel_depots_y=planet.fuel_depots.y,
            fuel_depots_active=planet.fuel_depots.active,
            player_bullets_x=player_bullets_x,
            player_bullets_y=player_bullets_y,
            player_bullets_active=player_bullets_active,
            enemy_bullets_x=enemy_bullets_x,
            enemy_bullets_y=enemy_bullets_y,
            enemy_bullets_active=enemy_bullets_active,
            score=state.score,
            lives=state.lives,
            current_location=state.current_location
        )

    def _get_info(self, state: GravitarState, all_rewards: jnp.array | None = None) -> GravitarInfo:
        """Extract additional info from state."""
        return GravitarInfo(
            current_galaxy=state.current_galaxy,
            current_system=state.current_system,
            step_count=state.step_count,
            all_rewards=jnp.asarray(0.0 if all_rewards is None else all_rewards)
        )

    def render(self, state: GravitarState) -> jnp.ndarray:
        """Render the game state to an image."""
        return self.renderer.render(state)

    # ========================================================================
    # Utility Functions
    # ========================================================================
    
    @staticmethod
    @jax.jit
    def _distance(x1: chex.Array, y1: chex.Array, x2: chex.Array, y2: chex.Array) -> chex.Array:
        """Calculate Euclidean distance between two points."""
        dx = x2 - x1
        dy = y2 - y1
        return jnp.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    @jax.jit
    def _circles_collide(x1: chex.Array, y1: chex.Array, r1: float,
                        x2: chex.Array, y2: chex.Array, r2: float) -> chex.Array:
        """Check if two circles collide."""
        dist = JaxGravitar._distance(x1, y1, x2, y2)
        return dist < (r1 + r2)
    
    # ========================================================================
    # Core Game Logic
    # ========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def _fire_bullet(self, state: GravitarState) -> GravitarState:
        """Fire a bullet from the ship if cooldown allows."""
        c = self.consts
        bullets = state.bullets
        
        # Find first inactive player bullet slot (0 to MAX_PLAYER_BULLETS-1)
        player_slots = jnp.arange(c.MAX_PLAYER_BULLETS)
        inactive_mask = ~bullets.active[player_slots]
        slot_idx = jnp.argmax(inactive_mask)  # First inactive slot
        can_fire = inactive_mask[slot_idx]  # True if we found an inactive slot
        
        # Calculate bullet velocity based on ship heading
        bullet_vx = jnp.cos(state.ship_theta) * c.BULLET_SPEED
        bullet_vy = jnp.sin(state.ship_theta) * c.BULLET_SPEED
        
        # Calculate spawn position (slightly ahead of ship)
        spawn_x = state.ship_x + jnp.cos(state.ship_theta) * (c.SHIP_RADIUS + 2)
        spawn_y = state.ship_y + jnp.sin(state.ship_theta) * (c.SHIP_RADIUS + 2)
        
        # Update bullet at slot
        def update_bullet_slot(bullets, idx):
            return bullets._replace(
                x=bullets.x.at[idx].set(jnp.where(can_fire, spawn_x, bullets.x[idx])),
                y=bullets.y.at[idx].set(jnp.where(can_fire, spawn_y, bullets.y[idx])),
                vx=bullets.vx.at[idx].set(jnp.where(can_fire, bullet_vx, bullets.vx[idx])),
                vy=bullets.vy.at[idx].set(jnp.where(can_fire, bullet_vy, bullets.vy[idx])),
                active=bullets.active.at[idx].set(jnp.where(can_fire, True, bullets.active[idx])),
                lifetime=bullets.lifetime.at[idx].set(jnp.where(can_fire, c.BULLET_LIFETIME, bullets.lifetime[idx])),
                owner=bullets.owner.at[idx].set(jnp.where(can_fire, 0, bullets.owner[idx]))  # 0 = player
            )
        
        new_bullets = update_bullet_slot(bullets, slot_idx)
        new_cooldown = jnp.where(can_fire, c.BULLET_COOLDOWN_STEPS, state.bullet_cooldown)
        
        return state._replace(bullets=new_bullets, bullet_cooldown=new_cooldown)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_bullets(self, state: GravitarState) -> GravitarState:
        """Update all bullet positions and lifetimes."""
        c = self.consts
        bullets = state.bullets
        
        # Update positions for active bullets
        new_x = bullets.x + bullets.vx * bullets.active
        new_y = bullets.y + bullets.vy * bullets.active
        
        # Decrement lifetime
        new_lifetime = jnp.maximum(0, bullets.lifetime - bullets.active.astype(jnp.int32))
        
        # Check if bullets are still in bounds and have lifetime
        in_bounds = (new_x >= 0) & (new_x < c.SCREEN_WIDTH) & \
                   (new_y >= 0) & (new_y < c.SCREEN_HEIGHT)
        has_lifetime = new_lifetime > 0
        new_active = bullets.active & in_bounds & has_lifetime
        
        new_bullets = bullets._replace(
            x=new_x,
            y=new_y,
            lifetime=new_lifetime,
            active=new_active
        )
        
        return state._replace(bullets=new_bullets)
    
    @partial(jax.jit, static_argnums=(0,))
    def _apply_sun_gravity(self, state: GravitarState) -> GravitarState:
        """Apply gravitational pull from the sun (solar system view only)."""
        c = self.consts
        sun = state.solar_system
        
        # Calculate vector from ship to sun
        dx = sun.sun_x - state.ship_x
        dy = sun.sun_y - state.ship_y
        dist = jnp.maximum(self._distance(state.ship_x, state.ship_y, sun.sun_x, sun.sun_y), 1.0)
        
        # Apply inverse-square gravity (simplified)
        gravity_strength = c.SUN_GRAVITY / (dist * dist) * 100.0  # Scale factor for playability
        gx = (dx / dist) * gravity_strength
        gy = (dy / dist) * gravity_strength
        
        # Update velocity
        new_vx = state.ship_vx + gx
        new_vy = state.ship_vy + gy
        
        return state._replace(ship_vx=new_vx, ship_vy=new_vy)
    
    @partial(jax.jit, static_argnums=(0,))
    def _apply_planet_gravity(self, state: GravitarState) -> GravitarState:
        """Apply gravitational pull when on a planet."""
        c = self.consts
        
        # Simple downward gravity (or upward for reverse gravity galaxies)
        galaxy = state.current_galaxy
        is_reverse = jnp.isin(galaxy, jnp.array([1, 3]))  # Galaxies 2 and 4 have reverse gravity
        
        # Check if game level 4 (no gravity mode)
        no_gravity = state.game_level == 4
        
        gravity_y = jnp.where(is_reverse, -c.GRAVITY_STRENGTH, c.GRAVITY_STRENGTH)
        gravity_y = jnp.where(no_gravity, 0.0, gravity_y)
        
        new_vy = state.ship_vy + gravity_y
        
        return state._replace(ship_vy=new_vy)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_saucer(self, state: GravitarState) -> GravitarState:
        """Update enemy saucer in solar system view."""
        c = self.consts
        saucer = state.solar_system.saucer
        
        # Simple AI: Move toward player
        dx = state.ship_x - saucer.x
        dy = state.ship_y - saucer.y
        dist = jnp.maximum(self._distance(saucer.x, saucer.y, state.ship_x, state.ship_y), 1.0)
        
        # Normalize and apply speed
        move_x = (dx / dist) * c.SAUCER_SPEED * saucer.active
        move_y = (dy / dist) * c.SAUCER_SPEED * saucer.active
        
        new_x = saucer.x + move_x
        new_y = saucer.y + move_y
        
        # Wrap around screen
        new_x = new_x % c.SCREEN_WIDTH
        new_y = jnp.clip(new_y, 0.0, float(c.SCREEN_HEIGHT - 1))
        
        # Update fire timer
        new_fire_timer = jnp.maximum(0, saucer.fire_timer - saucer.active.astype(jnp.int32))
        
        # Fire at player if timer expired and saucer is active
        # (Actual firing will be handled in collision/combat system)
        
        new_saucer = saucer._replace(
            x=new_x,
            y=new_y,
            vx=move_x,
            vy=move_y,
            fire_timer=new_fire_timer
        )
        
        new_solar_system = state.solar_system._replace(saucer=new_saucer)
        return state._replace(solar_system=new_solar_system)
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_sun_collision(self, state: GravitarState) -> GravitarState:
        """Check if ship collided with the sun."""
        c = self.consts
        sun = state.solar_system
        
        # Check distance to sun
        collided = self._circles_collide(
            state.ship_x, state.ship_y, c.SHIP_COLLISION_RADIUS,
            sun.sun_x, sun.sun_y, c.SUN_KILL_RADIUS
        )
        
        # If collided and ship is alive, kill ship
        ship_died = jnp.logical_and(collided, state.ship_alive)
        
        new_lives = jnp.where(ship_died, state.lives - 1, state.lives)
        new_alive = jnp.where(ship_died, False, state.ship_alive)
        new_death_timer = jnp.where(ship_died, 60, state.ship_death_timer)  # 60 frame death animation
        
        return state._replace(
            lives=new_lives,
            ship_alive=new_alive,
            ship_death_timer=new_death_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_saucer_collision(self, state: GravitarState) -> GravitarState:
        """Check ship-saucer collision and bullet-saucer collisions."""
        c = self.consts
        saucer = state.solar_system.saucer
        
        # Skip if saucer not active
        if_active = saucer.active
        
        # Check ship-saucer collision
        ship_collided = self._circles_collide(
            state.ship_x, state.ship_y, c.SHIP_COLLISION_RADIUS,
            saucer.x, saucer.y, c.SAUCER_RADIUS
        )
        ship_hit = jnp.logical_and(jnp.logical_and(ship_collided, if_active), state.ship_alive)
        
        # Check bullet-saucer collisions
        player_bullets_mask = (state.bullets.owner == 0) & state.bullets.active
        
        # Vectorized collision check for all player bullets
        bullet_hits = jax.vmap(
            lambda bx, by, active: jnp.logical_and(
                self._circles_collide(bx, by, 1.0, saucer.x, saucer.y, c.SAUCER_RADIUS),
                active
            )
        )(state.bullets.x[:c.MAX_PLAYER_BULLETS], 
          state.bullets.y[:c.MAX_PLAYER_BULLETS],
          player_bullets_mask[:c.MAX_PLAYER_BULLETS])
        
        any_bullet_hit = jnp.any(bullet_hits) & if_active
        
        # Deactivate hit bullets
        new_bullet_active = jnp.where(
            bullet_hits,
            False,
            state.bullets.active[:c.MAX_PLAYER_BULLETS]
        )
        bullets = state.bullets._replace(
            active=state.bullets.active.at[:c.MAX_PLAYER_BULLETS].set(new_bullet_active)
        )
        
        # Update saucer (deactivate if hit)
        new_saucer_active = jnp.where(any_bullet_hit, False, saucer.active)
        new_saucer = saucer._replace(active=new_saucer_active)
        
        # Update score (100 points for destroying saucer)
        score_gain = jnp.where(any_bullet_hit, c.SCORE_SAUCER, 0)
        new_score = state.score + score_gain
        
        # Update ship (kill if collided with saucer)
        new_lives = jnp.where(ship_hit, state.lives - 1, state.lives)
        new_ship_alive = jnp.where(ship_hit, False, state.ship_alive)
        new_death_timer = jnp.where(ship_hit, 60, state.ship_death_timer)
        
        new_solar_system = state.solar_system._replace(saucer=new_saucer)
        
        return state._replace(
            bullets=bullets,
            solar_system=new_solar_system,
            score=new_score,
            lives=new_lives,
            ship_alive=new_ship_alive,
            ship_death_timer=new_death_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_planet_entry(self, state: GravitarState) -> GravitarState:
        """Check if ship is entering a planet."""
        c = self.consts
        
        # For now, use simple position-based triggers
        # Top half: planets, bottom half: reactor
        in_top_half = state.ship_y < (c.SCREEN_HEIGHT / 2)
        in_left_half = state.ship_x < (c.SCREEN_WIDTH / 2)
        
        # Determine which planet (1-4) or reactor (5)
        # This is a simplified version; proper implementation would check actual planet positions
        enter_planet_1 = jnp.logical_and(in_top_half, in_left_half) & (state.ship_y < 40)
        enter_planet_2 = jnp.logical_and(in_top_half, ~in_left_half) & (state.ship_y < 40)
        enter_planet_3 = jnp.logical_and(~in_top_half, in_left_half) & (state.ship_y > 170)
        enter_planet_4 = jnp.logical_and(~in_top_half, ~in_left_half) & (state.ship_y > 170)
        
        # For now, enter planet 1 if at top-left corner
        should_enter = enter_planet_1
        new_location = jnp.where(should_enter, 1, state.current_location)
        
        # Reset ship position when entering planet
        new_ship_x = jnp.where(should_enter, c.SCREEN_WIDTH / 2, state.ship_x)
        new_ship_y = jnp.where(should_enter, c.SCREEN_HEIGHT - 30, state.ship_y)
        new_ship_vx = jnp.where(should_enter, 0.0, state.ship_vx)
        new_ship_vy = jnp.where(should_enter, 0.0, state.ship_vy)
        
        return state._replace(
            current_location=new_location,
            ship_x=new_ship_x,
            ship_y=new_ship_y,
            ship_vx=new_ship_vx,
            ship_vy=new_ship_vy
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_planet_exit(self, state: GravitarState) -> GravitarState:
        """Check if ship is exiting a planet (flying off top of screen)."""
        c = self.consts
        
        # Exit when ship reaches top of screen
        should_exit = state.ship_y < 10
        new_location = jnp.where(should_exit, 0, state.current_location)  # 0 = solar system
        
        # Reset ship position when exiting to solar system
        new_ship_x = jnp.where(should_exit, c.SCREEN_WIDTH / 2, state.ship_x)
        new_ship_y = jnp.where(should_exit, c.SCREEN_HEIGHT - 30, state.ship_y)
        new_ship_vx = jnp.where(should_exit, 0.0, state.ship_vx)
        new_ship_vy = jnp.where(should_exit, 0.0, state.ship_vy)
        
        return state._replace(
            current_location=new_location,
            ship_x=new_ship_x,
            ship_y=new_ship_y,
            ship_vx=new_ship_vx,
            ship_vy=new_ship_vy
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _step_solar_system(self, state: GravitarState) -> GravitarState:
        """Update state when in solar system view."""
        # Apply sun gravity
        state = self._apply_sun_gravity(state)
        
        # Update saucer
        state = self._update_saucer(state)
        
        # Check collisions
        state = self._check_sun_collision(state)
        state = self._check_saucer_collision(state)
        
        # Check for planet entry
        state = self._check_planet_entry(state)
        
        return state
    
    @partial(jax.jit, static_argnums=(0,))
    def _step_planet(self, state: GravitarState) -> GravitarState:
        """Update state when on a planet surface."""
        # Apply planet gravity (unless in no-gravity mode)
        state = self._apply_planet_gravity(state)
        
        # Check for planet exit
        state = self._check_planet_exit(state)
        
        # TODO: Update enemies (bunkers, rammers)
        # TODO: Check collisions with terrain and enemies
        # TODO: Check fuel depot collection
        
        return state
    
    @partial(jax.jit, static_argnums=(0,))
    def _apply_action(self, state: GravitarState, action: chex.Array) -> GravitarState:
        """Apply player action to update ship state."""
        c = self.consts

        # Decode action into booleans
        a = action.astype(jnp.int32)
        is_up = jnp.isin(a, jnp.array([Action.UP, Action.UPLEFT, Action.UPRIGHT, 
                                       Action.UPFIRE, Action.UPLEFTFIRE, Action.UPRIGHTFIRE]))
        is_down = jnp.isin(a, jnp.array([Action.DOWN, Action.DOWNLEFT, Action.DOWNRIGHT,
                                         Action.DOWNFIRE, Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE]))
        is_left = jnp.isin(a, jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, 
                                         Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE]))
        is_right = jnp.isin(a, jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, 
                                          Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE]))
        is_fire = jnp.isin(a, jnp.array([Action.FIRE, Action.UPFIRE, Action.LEFTFIRE, Action.RIGHTFIRE, 
                                         Action.DOWNFIRE, Action.UPLEFTFIRE, Action.UPRIGHTFIRE, 
                                         Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE]))

        # Rotation
        dtheta = jnp.where(is_left, -c.SHIP_ROTATION_SPEED, 0.0) + \
                 jnp.where(is_right, c.SHIP_ROTATION_SPEED, 0.0)
        theta = state.ship_theta + dtheta
        # Normalize angle to [-π, π]
        theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))

        # Thrust (accelerate along heading)
        # Note: DOWN pulls back on joystick = shields/tractor beam (handled separately)
        thrust_x = jnp.where(is_up, c.SHIP_THRUST_ACCEL * jnp.cos(theta), 0.0)
        thrust_y = jnp.where(is_up, c.SHIP_THRUST_ACCEL * jnp.sin(theta), 0.0)
        
        # Update velocity with thrust
        vx = state.ship_vx + thrust_x
        vy = state.ship_vy + thrust_y
        
        # Apply drag
        vx = vx * c.SHIP_DRAG
        vy = vy * c.SHIP_DRAG
        
        # Clamp to max speed
        speed = jnp.sqrt(vx*vx + vy*vy)
        scale = jnp.minimum(1.0, c.SHIP_MAX_SPEED / jnp.maximum(speed, 1e-6))
        vx = vx * scale
        vy = vy * scale

        # Integrate position
        x = state.ship_x + vx
        y = state.ship_y + vy

        # Screen wrapping (horizontal) and clamping (vertical)
        x = x % c.SCREEN_WIDTH
        y = jnp.clip(y, 0.0, float(c.SCREEN_HEIGHT - 1))

        # Fuel consumption
        fuel_cost = jnp.where(is_up, c.FUEL_COST_THRUST, 0)
        fuel_cost = jnp.where(is_down, fuel_cost + c.FUEL_COST_TRACTOR, fuel_cost)  # Down = tractor/shields
        new_fuel = jnp.maximum(0, state.fuel - fuel_cost)
        
        # Update ship state
        state = state._replace(
            ship_x=x,
            ship_y=y,
            ship_vx=vx,
            ship_vy=vy,
            ship_theta=theta,
            fuel=new_fuel,
            bullet_cooldown=jnp.maximum(0, state.bullet_cooldown - 1)
        )
        
        # Handle firing
        state = jax.lax.cond(
            jnp.logical_and(is_fire, state.bullet_cooldown == 0),
            lambda s: self._fire_bullet(s),
            lambda s: s,
            state
        )
        
        # Update bullets
        state = self._update_bullets(state)
        
        # Apply location-specific updates
        state = jax.lax.cond(
            state.current_location == 0,  # Solar system view
            lambda s: self._step_solar_system(s),
            lambda s: self._step_planet(s),  # Planet or reactor
            state
        )
        
        return state

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array | None = None) -> Tuple[GravitarObservation, GravitarState]:
        """Reset the environment to initial state."""
        rng = key if key is not None else jax.random.PRNGKey(0)
        
        # Initialize ship at starting position in solar system view
        state = GravitarState(
            ship_x=jnp.float32(self.consts.SCREEN_WIDTH / 2.0),
            ship_y=jnp.float32(self.consts.SCREEN_HEIGHT - 30.0),  # Bottom of screen
            ship_vx=jnp.float32(0.0),
            ship_vy=jnp.float32(0.0),
            ship_theta=jnp.float32(jnp.pi / 2),  # Pointing up
            ship_alive=jnp.bool_(True),
            ship_death_timer=jnp.int32(0),
            fuel=jnp.float32(self.consts.STARTING_FUEL),
            bullets=self._create_empty_bullet_pool(),
            bullet_cooldown=jnp.int32(0),
            current_galaxy=jnp.int32(0),
            current_system=jnp.int32(0),
            current_location=jnp.int32(0),  # Start in solar system view
            solar_system=self._create_initial_solar_system(rng),
            current_planet=self._create_empty_planet(),
            score=jnp.int32(0),
            lives=jnp.int32(self.consts.STARTING_LIVES_BY_LEVEL[self.game_level]),
            game_level=jnp.int32(self.game_level),
            step_count=jnp.int32(0),
            rng_key=rng,
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GravitarState) -> chex.Array:
        """Check if episode is terminated."""
        # Game over when out of lives
        no_lives = state.lives <= 0
        # Or when ship is dead with no fuel (can't respawn)
        no_fuel_dead = jnp.logical_and(~state.ship_alive, state.fuel <= 0)
        return jnp.logical_or(no_lives, no_fuel_dead)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: GravitarState, state: GravitarState) -> chex.Array:
        """Calculate reward based on score change."""
        # Reward is the change in score
        score_delta = state.score - prev.score
        return score_delta.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GravitarState, action) -> Tuple[GravitarObservation, GravitarState, float, bool, GravitarInfo]:
        """Execute one environment step."""
        # Ensure action is an array for JIT compatibility
        a = jnp.asarray(action, dtype=jnp.int32)
        
        # Apply action (handles ship control, firing, bullet updates)
        new_state = self._apply_action(state, a)
        
        # Increment step counter
        new_state = new_state._replace(step_count=new_state.step_count + 1)

        # Calculate reward and check if done
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        
        # Get observation and info
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, reward)
        
        return obs, new_state, reward, done, info


class GravitarRenderer(JAXGameRenderer):
    def __init__(self, config: RendererConfig | None = None):
        super().__init__(config=config)
        self._load_sprites()
    
    def _load_sprites(self):
        """Load sprite assets from disk."""
        import os
        import numpy as np
        
        sprite_dir = os.path.join(os.path.dirname(__file__), "sprites", "gravitar")
        
        def load_sprite(name: str) -> jnp.ndarray | None:
            """Load a sprite from .npy file."""
            path = os.path.join(sprite_dir, f"{name}.npy")
            if not os.path.exists(path):
                return None
            try:
                arr = np.load(path, allow_pickle=False)
                # Transpose if needed (channel-first to channel-last)
                if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] != arr.shape[-1]:
                    arr = np.transpose(arr, (1, 2, 0))
                # Normalize to uint8
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                elif arr.dtype == np.uint8 and arr.max() <= 1:
                    arr = arr * 255
                # Ensure RGBA
                if arr.shape[-1] == 3:
                    rgb = arr
                    alpha = (rgb.max(axis=-1) >= 1).astype(np.uint8) * 255
                    arr = np.dstack([rgb, alpha])
                return jnp.array(arr, dtype=jnp.uint8)
            except Exception:
                return None
        
        # Load sprites
        self.sprite_ship = load_sprite("spaceship")
        self.sprite_ship_thrust = load_sprite("ship_thrust")
        self.sprite_bullet = load_sprite("ship_bullet")
        self.sprite_enemy_bullet = load_sprite("enemy_bullet")
        self.sprite_saucer = load_sprite("saucer")
        self.sprite_saucer_crash = load_sprite("saucer_crash")
        self.sprite_ship_crash = load_sprite("ship_crash")
        
        # Load digit sprites for score display
        self.sprite_digits = [load_sprite(f"score_{i}") for i in range(10)]
    
    @staticmethod
    @jax.jit
    def _rotate_sprite(sprite: chex.Array, angle_deg: chex.Array) -> chex.Array:
        """Rotate sprite using JAX operations."""
        from jax.scipy import ndimage
        
        angle_rad = jnp.deg2rad(angle_deg)
        height, width = sprite.shape[:2]
        center_y, center_x = height / 2, width / 2
        
        y_coords, x_coords = jnp.mgrid[0:height, 0:width]
        y_centered, x_centered = y_coords - center_y, x_coords - center_x
        
        cos_angle, sin_angle = jnp.cos(-angle_rad), jnp.sin(-angle_rad)
        source_x = center_x + x_centered * cos_angle - y_centered * sin_angle
        source_y = center_y + x_centered * sin_angle + y_centered * cos_angle
        source_coords = jnp.stack([source_y, source_x])
        
        rotated_channels = []
        for i in range(sprite.shape[2]):
            rotated_channel = ndimage.map_coordinates(
                sprite[..., i], source_coords, order=1, mode='constant', cval=0
            )
            rotated_channels.append(rotated_channel)
        
        return jnp.stack(rotated_channels, axis=-1).astype(sprite.dtype)
    
    @staticmethod
    def _blit_sprite_static(canvas: chex.Array, sprite: chex.Array, x: chex.Array, y: chex.Array) -> chex.Array:
        """Blit sprite onto canvas with alpha blending (simplified, no JIT for now)."""
        if sprite is None:
            return canvas
        
        H, W = canvas.shape[:2]
        sh, sw = sprite.shape[:2]
        
        # Ensure x, y are integers
        x = int(x) if not isinstance(x, int) else x
        y = int(y) if not isinstance(y, int) else y
        
        # Calculate blit region (centered on x, y)
        x0 = x - sw // 2
        y0 = y - sh // 2
        
        # Check if sprite is completely off-screen
        if x0 >= W or y0 >= H or x0 + sw < 0 or y0 + sh < 0:
            return canvas
        
        # Simple approach: blend pixel by pixel where sprite overlaps canvas
        for sy in range(sh):
            cy = y0 + sy
            if 0 <= cy < H:
                for sx in range(sw):
                    cx = x0 + sx
                    if 0 <= cx < W:
                        alpha = float(sprite[sy, sx, 3]) / 255.0
                        if alpha > 0:
                            for c in range(3):
                                canvas = canvas.at[cy, cx, c].set(
                                    jnp.uint8(
                                        sprite[sy, sx, c] * alpha +
                                        canvas[cy, cx, c] * (1.0 - alpha)
                                    )
                                )
        
        return canvas
    
    @staticmethod
    @jax.jit
    def _blit_sprite(canvas: chex.Array, sprite: chex.Array, x: chex.Array, y: chex.Array) -> chex.Array:
        """Blit sprite onto canvas - JIT-compatible version (just draw a simple marker for now)."""
        if sprite is None:
            return canvas
        
        # For JIT compatibility, just draw a simple colored square where the sprite would be
        # This is a placeholder until we implement proper sprite rendering
        H, W = canvas.shape[:2]
        
        x = x.astype(jnp.int32) if hasattr(x, 'astype') else jnp.int32(x)
        y = y.astype(jnp.int32) if hasattr(y, 'astype') else jnp.int32(y)
        
        # Draw a 5x5 colored square
        color = jnp.array([200, 200, 200], dtype=jnp.uint8)
        
        # Vectorized approach using masks
        yy, xx = jnp.mgrid[0:H, 0:W]
        mask = (jnp.abs(xx - x) <= 2) & (jnp.abs(yy - y) <= 2)
        
        for c in range(3):
            canvas = canvas.at[:, :, c].set(
                jnp.where(mask, color[c], canvas[:, :, c])
            )
        
        return canvas
    
    @staticmethod
    @jax.jit
    def _draw_circle(canvas: chex.Array, cx: chex.Array, cy: chex.Array, radius: int, color: chex.Array) -> chex.Array:
        """Draw a filled circle on canvas."""
        H, W = canvas.shape[:2]
        y, x = jnp.mgrid[0:H, 0:W]
        # Ensure cx, cy are proper types for comparison
        cx = cx.astype(jnp.float32) if hasattr(cx, 'astype') else jnp.float32(cx)
        cy = cy.astype(jnp.float32) if hasattr(cy, 'astype') else jnp.float32(cy)
        mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
        
        for i in range(3):
            canvas = canvas.at[:, :, i].set(
                jnp.where(mask, color[i], canvas[:, :, i])
            )
        
        return canvas
    
    @staticmethod
    @jax.jit
    def _draw_text_digit(canvas: chex.Array, digit_sprite: chex.Array, x: chex.Array, y: chex.Array) -> chex.Array:
        """Draw a digit sprite at position."""
        return GravitarRenderer._blit_sprite(canvas, digit_sprite, x, y)
    
    def render(self, state: GravitarState) -> jnp.ndarray:
        """Render the game state to an image."""
        H, W = self.config.game_dimensions
        
        # Create background based on location
        is_solar_system = state.current_location == 0
        bg_color = jax.lax.select(
            is_solar_system,
            jnp.array([10, 10, 30], dtype=jnp.uint8),  # Dark blue for space
            jnp.array([0, 0, 0], dtype=jnp.uint8)      # Black for planets
        )
        
        canvas = jnp.broadcast_to(bg_color.reshape(1, 1, 3), (H, W, 3)).copy()
        
        # Draw sun in solar system view
        canvas = jax.lax.cond(
            is_solar_system,
            lambda c: self._draw_circle(
                c,
                state.solar_system.sun_x,
                state.solar_system.sun_y,
                8,
                jnp.array([255, 200, 50], dtype=jnp.uint8)
            ),
            lambda c: c,
            canvas
        )
        
        # Draw saucer in solar system view
        should_draw_saucer = is_solar_system & state.solar_system.saucer.active
        canvas = jax.lax.cond(
            should_draw_saucer,
            lambda c: self._blit_sprite(
                c,
                self.sprite_saucer if self.sprite_saucer is not None else jnp.zeros((8, 8, 4), dtype=jnp.uint8),
                state.solar_system.saucer.x,
                state.solar_system.saucer.y
            ),
            lambda c: c,
            canvas
        )
        
        # Draw bullets
        def draw_one_bullet(i, c):
            is_player_bullet = state.bullets.active[i] & (state.bullets.owner[i] == 0)
            x = jnp.clip(state.bullets.x[i], 0, W - 1).astype(jnp.int32)
            y = jnp.clip(state.bullets.y[i], 0, H - 1).astype(jnp.int32)
            bullet_color = jnp.array([255, 255, 0], dtype=jnp.uint8)
            return jax.lax.cond(
                is_player_bullet,
                lambda c: c.at[y, x, :].set(bullet_color),
                lambda c: c,
                c
            )
        canvas = jax.lax.fori_loop(0, state.bullets.x.shape[0], draw_one_bullet, canvas)
        
        # Draw ship
        canvas = jax.lax.cond(
            state.ship_alive,
            lambda c: self._draw_ship(c, state),
            lambda c: c,
            canvas
        )
        
        # Draw HUD (score, lives, fuel)
        canvas = self._draw_hud(canvas, state)
        
        return canvas
    
    @staticmethod
    @jax.jit
    def _draw_ship(canvas: chex.Array, state: GravitarState) -> chex.Array:
        """Draw the ship with simple marker (sprites not JIT-compatible)."""
        # Simple fallback: draw marker at ship position
        x = state.ship_x.astype(jnp.int32)
        y = state.ship_y.astype(jnp.int32)
        
        H, W = canvas.shape[:2]
        color = jnp.array([200, 200, 200], dtype=jnp.uint8)
        
        # Draw 3x3 square
        yy, xx = jnp.mgrid[0:H, 0:W]
        mask = (jnp.abs(xx - x) <= 1) & (jnp.abs(yy - y) <= 1)
        
        for c in range(3):
            canvas = canvas.at[:, :, c].set(
                jnp.where(mask, color[c], canvas[:, :, c])
            )
        
        return canvas
    
    @staticmethod
    @jax.jit
    def _draw_hud(canvas: chex.Array, state: GravitarState) -> chex.Array:
        """Draw HUD elements (score, lives, fuel)."""
        H, W = canvas.shape[:2]
        lives_to_show = jnp.minimum(state.lives, 10)
        
        # Draw lives indicator using masks
        y = 20
        lives_color = jnp.array([255, 0, 0], dtype=jnp.uint8)
        
        def draw_life_dot(i, c):
            should_draw = i < lives_to_show
            x = 10 + i * 6
            yy, xx = jnp.mgrid[0:H, 0:W]
            mask = (xx >= x) & (xx < x + 2) & (yy >= y) & (yy < y + 2)
            
            for ch in range(3):
                c = c.at[:, :, ch].set(
                    jnp.where(should_draw & mask, lives_color[ch], c[:, :, ch])
                )
            return c
        
        canvas = jax.lax.fori_loop(0, 10, draw_life_dot, canvas)
        
        # Draw fuel bar
        fuel_pct = jnp.clip(state.fuel / 10000.0, 0.0, 1.0)
        fuel_width = (fuel_pct * 50).astype(jnp.int32)
        
        yy, xx = jnp.mgrid[0:H, 0:W]
        fuel_mask = (yy >= 25) & (yy < 27) & (xx >= 10) & (xx < 10 + fuel_width)
        fuel_color = jnp.array([0, 255, 0], dtype=jnp.uint8)
        
        for ch in range(3):
            canvas = canvas.at[:, :, ch].set(
                jnp.where(fuel_mask, fuel_color[ch], canvas[:, :, ch])
            )
        
        return canvas
