import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, List, Optional

import jax
import jax.numpy as jnp
import chex
from jax.image import resize

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari import spaces

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

class SirLancelotConstants(NamedTuple):
    # Screen dimensions
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 250
    
    # Play area boundaries
    TOP_BLACK_BAR_HEIGHT: int = 24
    PLAY_AREA_TOP: int = 24
    PLAY_AREA_BOTTOM: int = 193  # Ground level for Sir Lancelot (3 pixels down from 190)
    
    # Player constants
    PLAYER_WIDTH: int = 8
    PLAYER_HEIGHT: int = 12
    PLAYER_START_X: int = 141  # Sir Lancelot spawn position
    PLAYER_START_Y: int = 180  # On ground level (PLAY_AREA_BOTTOM - PLAYER_HEIGHT = 193 - 13)
    
    # =========================================================================
    # FLYING PHYSICS CONFIGURATION - TWEAK ALL FLYING PARAMETERS HERE
    # =========================================================================
    
    # VERTICAL PHYSICS (Up/Down movement)
    GRAVITY: float = 0.008              # Slower fall for moon-like physics
    FLAP_IMPULSE: float = -0.30         # Upward boost per flap (adds to velocity)
    FLAP_COOLDOWN: int = 5              # Frames to wait between flaps (60 FPS)
    MAX_VEL_Y_UP: float = -0.96         # Reduced by 50% from -1.92 (60% total from original)
    MAX_VEL_Y_DOWN: float = 0.48        # Reduced by 50% from 0.96 (60% total from original)
    
    # HORIZONTAL PHYSICS (Left/Right movement)
    HORIZONTAL_FLAP_IMPULSE: float = 0.25  # Stronger side-kick per LEFTFIRE/RIGHTFIRE
    HORIZONTAL_DECAY: float = 0.995        # Less decay, retains momentum longer
    MAX_VEL_X: float = 2.0                 # Reduced by 50% from 4.0 (60% total from original)
    
    # ANIMATION
    FLAP_ANIM_FRAMES: int = 5              # Longer flap animation (~83ms)
    
    # LANDING PHYSICS
    GROUND_STICK: bool = True           # Stick to ground when landing
    LANDING_VELOCITY_KILL: float = 0.8  # Horizontal momentum kept on landing (0-1)
    
    # =========================================================================
    # END OF FLYING PHYSICS CONSTANTS
    # =========================================================================
    
    # Enemy constants (Quest 1: Basic Flyers)
    ENEMY_WIDTH: int = 12
    ENEMY_HEIGHT: int = 10
    NUM_ENEMIES: int = 4  # Quest 1 has 4 beasts as requested
    ENEMY_HORIZONTAL_SPEED: float = 0.6  # Slower horizontal movement
    ENEMY_VERTICAL_OSCILLATION: float = 0.3  # Gentler vertical drift
    ENEMY_SPAWN_Y: chex.Array = jnp.array([50, 80, 110, 140], dtype=jnp.float32)
    ENEMY_MIN_Y: int = 140  # Lowest beast position from Level 1 (ground safety)
    ENEMY_ANIMATION_SPEED: int = 60  # Animation switches every second (60 frames at 60 FPS)
    
    # Quest/Level progression
    INITIAL_QUEST: int = 1
    MAX_QUEST: int = 4
    INITIAL_LEVEL: int = 1  # Start at level 1
    MAX_LEVEL: int = 8  # 8 levels total (4 beast levels, 4 dragon levels)
    
    # Scoring per level - Beast levels (odd levels)
    POINTS_LEVEL_1_BEAST: int = 250      # Flying Snakes
    POINTS_LEVEL_3_BEAST: int = 750      # Monster Bees
    POINTS_LEVEL_5_BEAST: int = 1500     # Killer Dragonflies
    POINTS_LEVEL_7_BEAST: int = 3000     # Invisible Invincibles
    
    # Quick kill bonuses for beast levels
    # Level 1 (Flying Snakes)
    QUICK_KILL_BONUS_1_L1: int = 1000    # 1st quick kill
    QUICK_KILL_BONUS_2_L1: int = 2000    # 2nd quick kill  
    QUICK_KILL_BONUS_3_L1: int = 5000    # 3rd quick kill
    # Level 3 (Monster Bees)
    QUICK_KILL_BONUS_1_L3: int = 1750    # 1st quick kill
    QUICK_KILL_BONUS_2_L3: int = 4000    # 2nd quick kill  
    QUICK_KILL_BONUS_3_L3: int = 10000   # 3rd quick kill
    # Level 5 (Killer Dragonflies)
    QUICK_KILL_BONUS_1_L5: int = 3500    # 1st quick kill
    QUICK_KILL_BONUS_2_L5: int = 6000    # 2nd quick kill  
    QUICK_KILL_BONUS_3_L5: int = 15000   # 3rd quick kill
    # Level 7 (Invisible Invincibles)
    QUICK_KILL_BONUS_1_L7: int = 5000    # 1st quick kill
    QUICK_KILL_BONUS_2_L7: int = 8000    # 2nd quick kill  
    QUICK_KILL_BONUS_3_L7: int = 20000   # 3rd quick kill
    
    # Dragon level scoring
    # Level 2 (Old Dragon)
    POINTS_DRAGON_SURVIVE_L2: int = 10    # Per 1.5 seconds
    POINTS_DRAGON_KILL_L2: int = 2500     # For killing dragon
    POINTS_DRAGON_SAVE_L2: int = 10000    # For saving Carolyn
    # Level 4 (Young Grok)
    POINTS_DRAGON_SURVIVE_L4: int = 20    # Per 1.5 seconds
    POINTS_DRAGON_KILL_L4: int = 5000     # For killing dragon
    POINTS_DRAGON_SAVE_L4: int = 20000    # For saving Sarah
    # Level 6 (Jarek the Speedy)
    POINTS_DRAGON_SURVIVE_L6: int = 30    # Per 1.5 seconds
    POINTS_DRAGON_KILL_L6: int = 7500     # For killing dragon
    POINTS_DRAGON_SAVE_L6: int = 40000    # For saving Lauren
    # Level 8 (Hanek the Horrible)
    POINTS_DRAGON_SURVIVE_L8: int = 40    # Per 1.5 seconds
    POINTS_DRAGON_KILL_L8: int = 10000    # For killing dragon
    POINTS_DRAGON_SAVE_L8: int = 80000    # For saving Elisabeth
    
    EXTRA_LIFE_POINTS: int = 100000  # Get extra life every 100k points
    QUICK_KILL_TIME_LIMIT: int = 60  # Frames to qualify for quick kill
    
    # Game mechanics
    INITIAL_LIVES: int = 3
    MAX_LIVES: int = 6  # Maximum total lives (not 3 + 6)
    DEATH_ANIMATION_FRAMES: int = 60
    STUN_DURATION_FRAMES: int = 180  # 3 seconds stun from fireball
    # No flap cooldown needed - using continuous thrust model
    
    # Enemy mechanics per level
    ENEMY_BASE_SPEED: float = 0.4  # Base horizontal speed
    ENEMY_SPEED_VARIATION: float = 0.15  # Speed variation for randomized speeds
    ENEMY_ANIMATION_VARIATION: int = 15  # Random animation timing variation (±15 frames)
    ENEMY_VERTICAL_SPEED: float = 0.3  # For level 3,5,7
    ENEMY_RANDOM_SPEED_RANGE: float = 0.6  # For level 5,7 random movement
    ENEMY_SPEED_QUEST_MULTIPLIER: float = 0.25  # Speed increase per quest
    
    # Level 7 invisibility
    INVISIBILITY_DURATION: int = 90  # 1.5 seconds at 60 FPS
    VISIBILITY_DURATION: int = 120  # 2 seconds visible
    INVISIBILITY_QUEST_MULTIPLIER: float = 0.5  # Invisibility lasts 50% longer per quest
    
    # Quick kill timing (frames)
    QUICK_KILL_TIME_LIMIT_1: int = 60  # Within 1 second for 1st bonus
    QUICK_KILL_TIME_LIMIT_2: int = 40  # Within 2/3 second for 2nd bonus  
    QUICK_KILL_TIME_LIMIT_3: int = 20  # Within 1/3 second for 3rd bonus
    
    # Player states
    PLAYER_STATE_IDLE: int = 0
    PLAYER_STATE_FLYING: int = 1
    PLAYER_STATE_STUNNED: int = 2
    PLAYER_STATE_DEATH: int = 3
    
    # HUD - positioned at bottom like the real cartridge
    SCORE_X: int = 58   # Score starting X position
    SCORE_Y: int = 200  # Score Y position
    LIVES_X: int = 57   # Lives starting X position (1 pixel to the left of score)
    LIVES_Y: int = 213  # Lives Y position (4 pixels from bottom of numbers to top of lives)
    
    # Altitude tick marks on left edge
    TICK_Y_START: int = 32  # First altitude mark
    TICK_Y_STEP: int = 38   # Distance between marks
    TICK_LEN: int = 24      # Length of each mark
    
    # Colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 128)  # Dark blue for sky
    
    # Dragon constants (for all dragon levels: 2, 4, 6, 8)
    DRAGON_WIDTH: int = 24
    DRAGON_HEIGHT: int = 20
    DRAGON_START_X: int = 80
    DRAGON_Y: int = 28  # Move down 3 pixels (25 + 3)
    DRAGON_SPEED: float = 0.4  # Half speed
    DRAGON_MIN_X: int = 10
    DRAGON_MAX_X: int = 130
    DRAGON_WING_CYCLE_FRAMES: int = 30  # 0.5 seconds per wing position (30 frames at 60 FPS)
    
    # Fireball constants (dragons only shoot on levels 2 and 8)
    MAX_FIREBALLS: int = 3  # Maximum simultaneous fireballs on screen
    FIREBALL_WIDTH: int = 4
    FIREBALL_HEIGHT: int = 8
    FIREBALL_SPEED: float = 0.8  # Slower falling speed
    FIREBALL_SHOOT_COOLDOWN: int = 240  # 4 seconds at 60 FPS
    FIREBALL_SHOOT_ZONE: int = 40  # Dragon shoots when player is within this X distance
    
    # Lava constants (appears in castle levels)
    LAVA_START_Y: int = 169  # Fixed lava position (moved down 1 pixel)
    LAVA_RISE_SPEED: float = 0.0  # Lava doesn't rise
    
    # Magic wall (barrier in castle levels)
    MAGIC_WALL_Y: int = 46  # Y position of magic wall barrier
    
    # Castle ground layout
    GROUND_Y: int = 157  # Where player can stand (moved down one more pixel)
    LEFT_WALL_WIDTH: int = 4  # Left wall thickness
    RIGHT_WALL_WIDTH: int = 4  # Right wall thickness
    GROUND_LEFT_EDGE: int = 17  # Ground starts 17 pixels from left
    GROUND_RIGHT_EDGE: int = 143  # Ground ends 17 pixels from right (160-17)
    DRAGON_DEFEAT_RADIUS: int = 15  # Dragon must be within this radius when player hits wall
    
    # Dragon difficulty scaling
    DRAGON_SPEED_LEVEL_4: float = 1.0  # 2.5x speed
    DRAGON_SPEED_LEVEL_6: float = 1.6  # 4x speed
    DRAGON_SPEED_LEVEL_8: float = 2.4  # 6x speed (Hanek the Horrible)
    DRAGON_COOLDOWN_LEVEL_4: int = 120  # Half cooldown
    DRAGON_COOLDOWN_LEVEL_6: int = 80  # 1/3 cooldown
    DRAGON_COOLDOWN_LEVEL_8: int = 60  # 1/4 cooldown (fastest)
    DRAGON_TARGET_ZONE_LEVEL_4: int = 30  # Closer targeting
    DRAGON_TARGET_ZONE_LEVEL_6: int = 20  # Even closer
    DRAGON_TARGET_ZONE_LEVEL_8: int = 15  # Tightest targeting
    
    # Dragon survival scoring
    DRAGON_SURVIVAL_TIMER: int = 90  # 1.5 seconds at 60 FPS
    
    # Fireball animation
    FIREBALL_ANIMATION_SPEED: int = 7  # Change frame every 0.5 seconds (30 frames at 60 FPS)
    
    # Castle levels - Player start position (applies to all castle levels: 2, 4, 6, 8)
    PLAYER_START_X_CASTLE: int = 152  # Right side, just inside wall (160 - 4 - 4)
    PLAYER_START_Y_CASTLE: int = 145  # Standing on ground (157 - 12)

    # Testing helpers (turn off when done)
    DEBUG_EASY_VERTICAL_BOUNCE: bool = False  # Makes vertical bounces much easier to trigger for testing

    # Starting level
    START_LEVEL: int = 1  # Start at level 1


# -----------------------------------------------------------------------------
# STATE REPRESENTATIONS
# -----------------------------------------------------------------------------

class PlayerState(NamedTuple):
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array  # Horizontal velocity
    vel_y: chex.Array  # Vertical velocity
    facing_left: chex.Array  # True if facing left
    state: chex.Array  # IDLE, FLYING, STUNNED, DEATH
    state_timer: chex.Array  # Timer for current state
    is_flapping: chex.Array  # True when showing flapping animation
    fire_was_pressed: chex.Array  # Track if fire was pressed last frame (edge detection)
    flap_cooldown: chex.Array  # Cooldown counter for wing flaps
    flap_anim_timer: chex.Array  # Animation timer (0 = neutral sprite)
    death_timer: chex.Array  # Timer for death animation
    tie_cooldown: chex.Array  # Brief invulnerability after ties to prevent double-hits


class EnemyState(NamedTuple):
    # Shape (NUM_ENEMIES, 2) for x,y positions
    positions: chex.Array
    # Shape (NUM_ENEMIES,) for velocities
    vel_x: chex.Array  # Horizontal velocity
    vel_y: chex.Array  # Vertical velocity (for levels 3,5,7)
    # Shape (NUM_ENEMIES,) for active flags
    active: chex.Array
    # Shape (NUM_ENEMIES,) for sprite animation frame
    animation_frame: chex.Array
    # Shape (NUM_ENEMIES,) for facing direction (True = left)
    facing_left: chex.Array
    # Level 7 invisibility
    invisibility_timer: chex.Array  # Countdown for current state
    is_invisible: chex.Array  # True when invisible
    # Animation timing variation
    animation_timer: chex.Array  # Individual animation timers for each enemy
    animation_speed: chex.Array  # Individual animation speeds (frames per change)
    # Vertical bounce animation (for smooth movement)
    bounce_target_y: chex.Array  # Target Y position after bounce
    bounce_timer: chex.Array  # Timer for smooth transition (0 = no bounce)


class DragonState(NamedTuple):
    x: chex.Array
    vel_x: chex.Array  # +1 = moving right, -1 = moving left
    facing_left: chex.Array  # True if dragon faces left
    wing_frame: chex.Array  # 0=up, 1=middle_up, 2=down, 3=middle_down
    animation_timer: chex.Array  # Timer for wing animation
    shoot_cooldown: chex.Array  # Cooldown for shooting fireballs
    is_active: chex.Array  # Dragon active in castle levels (2, 4, 6, 8)


class FireballState(NamedTuple):
    # Shape (MAX_FIREBALLS, 2) for x,y positions
    positions: chex.Array
    # Shape (MAX_FIREBALLS,) for active flags
    active: chex.Array
    # Shape (MAX_FIREBALLS,) for animation frames
    animation_frame: chex.Array
    # Shape (MAX_FIREBALLS,) for animation timer
    animation_timer: chex.Array


class SirLancelotState(NamedTuple):
    player: PlayerState
    enemies: EnemyState
    dragon: DragonState
    fireballs: FireballState
    score: chex.Array
    lives: chex.Array
    quest: chex.Array  # Current quest number (1-4)
    stage: chex.Array  # Current stage (1=beast, 2=dragon)
    level: chex.Array  # Current level (1-7)
    time: chex.Array
    game_over: chex.Array
    stage_complete: chex.Array
    last_extra_life_score: chex.Array  # Track when to give next life
    # Quick kill tracking
    last_kill_time: chex.Array
    consecutive_quick_kills: chex.Array
    # Castle level specific (applies to levels 2, 4, 6, 8)
    lava_y: chex.Array  # Y position of rising lava
    # Dragon survival scoring
    dragon_survival_timer: chex.Array  # Timer for survival scoring (dragon levels)
    # Random key for any randomness
    key: jax.random.PRNGKey


# -----------------------------------------------------------------------------
# OBSERVATION / INFO
# -----------------------------------------------------------------------------

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class SirLancelotObservation(NamedTuple):
    player: EntityPosition
    enemies: jnp.ndarray  # Shape (NUM_ENEMIES, 5) for x,y,w,h,active
    dragon: jnp.ndarray  # Shape (5,) for x,y,w,h,active
    fireballs: jnp.ndarray  # Shape (MAX_FIREBALLS, 5) for x,y,w,h,active
    score: jnp.ndarray
    lives: jnp.ndarray
    stage: jnp.ndarray  # Current stage (1=aerial, 2=dragon)


class SirLancelotInfo(NamedTuple):
    time: jnp.ndarray
    level_complete: jnp.ndarray
    consecutive_kills: jnp.ndarray


# -----------------------------------------------------------------------------
# ENVIRONMENT IMPLEMENTATION
# -----------------------------------------------------------------------------

class JaxSirLancelot(JaxEnvironment[SirLancelotState, SirLancelotObservation, SirLancelotInfo, SirLancelotConstants]):
    """Sir Lancelot JAX implementation.
    
    An 8-level action game where players control Sir Lancelot:
    - Odd levels (1,3,5,7): Outdoor beast combat
    - Even levels (2,4,6,8): Castle dragon battles
    
    Controls:
        - Arrow keys: Move left/right
        - Space + Arrow: Fly in direction
        - Combat: Face enemies to defeat them
        
    Game mechanics:
        - Combat based on facing direction and height
        - Quick kill bonuses for fast consecutive defeats
        - 3 starting lives, max 6 total lives
        - Extra life every 100k points (up to max)
    """
    
    def __init__(self, consts: SirLancelotConstants = None, reward_funcs: list[callable] = None):
        consts = consts or SirLancelotConstants()
        super().__init__(consts)
        self.action_set = Action.get_all_values()
        self.renderer = SirLancelotRenderer(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        
    def reset(self, key: jax.random.PRNGKey = None):
        """Reset the game to initial state.
        
        Args:
            key: JAX random key (optional, defaults to key 0)
            
        Returns:
            Tuple of (initial_observation, initial_state)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Start at the specified level
        starting_level = self.consts.START_LEVEL
        # Derive stage from level (odd levels = stage 1, even levels = stage 2)
        starting_stage = 2 if starting_level % 2 == 0 else 1
        
        # Initialize player position based on stage
        player_x = jnp.where(
            starting_stage == 2,
            self.consts.PLAYER_START_X_CASTLE,
            self.consts.PLAYER_START_X
        ).astype(jnp.float32)
        player_y = jnp.where(
            starting_stage == 2,
            self.consts.PLAYER_START_Y_CASTLE,
            self.consts.PLAYER_START_Y
        ).astype(jnp.float32)
        
        # Initialize player
        player = PlayerState(
            x=player_x,
            y=player_y,
            vel_x=jnp.array(0.0, dtype=jnp.float32),
            vel_y=jnp.array(0.0, dtype=jnp.float32),
            facing_left=jnp.array(True),  # Start facing left
            state=jnp.array(self.consts.PLAYER_STATE_IDLE, dtype=jnp.int32),
            state_timer=jnp.array(0, dtype=jnp.int32),
            is_flapping=jnp.array(False),
            fire_was_pressed=jnp.array(False),
            flap_cooldown=jnp.array(0, dtype=jnp.int32),
            flap_anim_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            tie_cooldown=jnp.array(0, dtype=jnp.int32)
        )
        
        # Initialize enemies based on starting stage
        # Stage 1 (beast levels): Initialize enemies at spawn positions  
        # Stage 2 (dragon levels): No enemies
        enemy_x = jnp.where(
            starting_stage == 1,
            jnp.full(self.consts.NUM_ENEMIES, self.consts.SCREEN_WIDTH, dtype=jnp.float32),
            jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.float32)
        )
        enemy_positions = jnp.stack([enemy_x, self.consts.ENEMY_SPAWN_Y], axis=1)
        
        # Enemies are active in outdoor levels (odd), inactive in castle levels (even)
        enemies_active = jnp.where(
            starting_stage == 1,
            jnp.ones(self.consts.NUM_ENEMIES, dtype=bool),
            jnp.zeros(self.consts.NUM_ENEMIES, dtype=bool)
        )
        
        # Initialize velocities based on level
        # For level 1, generate random speeds for each beast
        rng_key = key
        keys = jax.random.split(rng_key, 2)
        random_speeds = jax.random.uniform(keys[0], shape=(self.consts.NUM_ENEMIES,), 
                                          minval=self.consts.ENEMY_BASE_SPEED - self.consts.ENEMY_SPEED_VARIATION,
                                          maxval=self.consts.ENEMY_BASE_SPEED + self.consts.ENEMY_SPEED_VARIATION)
        enemy_vel_x = jnp.where(
            starting_stage == 1,  # Beast stage
            -random_speeds,  # Beast stages: move left with random speeds
            jnp.zeros(self.consts.NUM_ENEMIES)
        )
        enemy_vel_y = jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.float32)
        
        # Generate random animation speeds for each enemy
        anim_keys = jax.random.split(keys[1], self.consts.NUM_ENEMIES)
        random_anim_speeds = jax.vmap(lambda k: jax.random.randint(
            k, shape=(), 
            minval=self.consts.ENEMY_ANIMATION_SPEED - self.consts.ENEMY_ANIMATION_VARIATION,
            maxval=self.consts.ENEMY_ANIMATION_SPEED + self.consts.ENEMY_ANIMATION_VARIATION + 1
        ))(anim_keys)
        
        enemies = EnemyState(
            positions=enemy_positions,
            vel_x=enemy_vel_x,
            vel_y=enemy_vel_y,
            active=enemies_active,
            animation_frame=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32),
            facing_left=jnp.ones(self.consts.NUM_ENEMIES, dtype=bool),  # Start facing left
            invisibility_timer=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32),
            is_invisible=jnp.zeros(self.consts.NUM_ENEMIES, dtype=bool),
            animation_timer=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32),
            animation_speed=random_anim_speeds,
            bounce_target_y=enemy_positions[:, 1],  # Initialize with current positions
            bounce_timer=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32)
        )
        
        # Initialize dragon for castle levels (2, 4, 6, 8)
        dragon = DragonState(
            x=jnp.array(self.consts.DRAGON_START_X, dtype=jnp.float32),
            vel_x=jnp.array(self.consts.DRAGON_SPEED, dtype=jnp.float32),  # Start moving right
            facing_left=jnp.array(False),  # Start facing right
            wing_frame=jnp.array(0, dtype=jnp.int32),  # Start with wings up
            animation_timer=jnp.array(0, dtype=jnp.int32),
            shoot_cooldown=jnp.array(self.consts.FIREBALL_SHOOT_COOLDOWN, dtype=jnp.int32),
            is_active=jnp.array(starting_stage == 2, dtype=bool)
        )
        
        # Initialize fireballs
        fireball_positions = jnp.zeros((self.consts.MAX_FIREBALLS, 2), dtype=jnp.float32)
        fireballs = FireballState(
            positions=fireball_positions,
            active=jnp.zeros(self.consts.MAX_FIREBALLS, dtype=bool),
            animation_frame=jnp.zeros(self.consts.MAX_FIREBALLS, dtype=jnp.int32),
            animation_timer=jnp.zeros(self.consts.MAX_FIREBALLS, dtype=jnp.int32)
        )
        
        # Initialize game state
        state = SirLancelotState(
            player=player,
            enemies=enemies,
            dragon=dragon,
            fireballs=fireballs,
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.INITIAL_LIVES, dtype=jnp.int32),
            quest=jnp.array(self.consts.INITIAL_QUEST, dtype=jnp.int32),
            stage=jnp.array(starting_stage, dtype=jnp.int32),
            level=jnp.array(starting_level, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False),
            stage_complete=jnp.array(False),
            last_extra_life_score=jnp.array(0, dtype=jnp.int32),
            last_kill_time=jnp.array(-999, dtype=jnp.int32),
            consecutive_quick_kills=jnp.array(0, dtype=jnp.int32),
            lava_y=jnp.array(self.consts.LAVA_START_Y, dtype=jnp.float32),
            dragon_survival_timer=jnp.array(0, dtype=jnp.int32),
            key=key
        )
        
        return self._get_observation(state), state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SirLancelotState, action: int):
        """Execute one game step.
        
        Updates all game state including player, enemies, physics, and scoring.
        Handles level progression and game over conditions.
        
        Args:
            state: Current game state
            action: Player action from action set
            
        Returns:
            Tuple of (observation, new_state, reward, done, info)
        """
        new_time = state.time + 1
        
        # Update player based on current state
        new_player = self._update_player_state_machine(state.player, action)
        
        # Apply bouncy wall constraints for castle levels (even levels: 2, 4, 6, 8)
        is_castle_level = (state.level % 2) == 0
        new_player = jax.lax.cond(
            is_castle_level,
            lambda p: p._replace(
                x=jnp.clip(
                    p.x,
                    self.consts.LEFT_WALL_WIDTH,
                    self.consts.SCREEN_WIDTH - self.consts.RIGHT_WALL_WIDTH - self.consts.PLAYER_WIDTH
                ),
                vel_x=jnp.where(
                    p.x <= self.consts.LEFT_WALL_WIDTH,
                    jnp.abs(p.vel_x),  # Bounce right
                    jnp.where(
                        p.x >= self.consts.SCREEN_WIDTH - self.consts.RIGHT_WALL_WIDTH - self.consts.PLAYER_WIDTH,
                        -jnp.abs(p.vel_x),  # Bounce left
                        p.vel_x
                    )
                )
            ),
            lambda p: p,
            new_player
        )
        
        # Branch based on stage
        def update_stage_1():
            # Update enemies based on level (1, 3, 5, or 7)
            updated_enemies = self._update_beasts(state.enemies, new_time, state.level, state.quest)
            
            # Check for combat collisions
            combat_results = self._resolve_combat(new_player, updated_enemies, state.score, 
                                                state.last_kill_time, state.consecutive_quick_kills, new_time, state.level)
            updated_player, updated_enemies, updated_score, updated_last_kill_time, updated_consecutive_kills = combat_results
            
            # No dragon/fireballs in stage 1
            return (updated_player, updated_enemies, state.dragon, state.fireballs, 
                    updated_score, updated_last_kill_time, updated_consecutive_kills, state.lava_y, state.lives, state.dragon_survival_timer)
        
        def update_stage_2():
            # Wall constraints already applied for castle levels above
            wall_constrained_player = new_player
            
            # Ground does NOT exist from x=17 to x=143
            player_center_x = wall_constrained_player.x + self.consts.PLAYER_WIDTH // 2
            player_NOT_on_ground_area = jnp.logical_and(
                player_center_x >= self.consts.GROUND_LEFT_EDGE,
                player_center_x <= self.consts.GROUND_RIGHT_EDGE
            )
            
            # Only apply ground collision if player is OUTSIDE the no-ground area
            can_stand_on_ground = jnp.logical_and(
                jnp.logical_not(player_NOT_on_ground_area),  # Reversed
                wall_constrained_player.y >= self.consts.GROUND_Y - self.consts.PLAYER_HEIGHT
            )
            
            ground_constrained_player = wall_constrained_player._replace(
                y=jnp.where(
                    can_stand_on_ground,
                    self.consts.GROUND_Y - self.consts.PLAYER_HEIGHT,
                    wall_constrained_player.y
                ),
                vel_x=jnp.where(
                    can_stand_on_ground,
                    wall_constrained_player.vel_x * 0.7,  # Apply friction to horizontal velocity
                    wall_constrained_player.vel_x
                ),
                vel_y=jnp.where(
                    can_stand_on_ground,
                    0.0,  # Stop vertical movement on ground
                    wall_constrained_player.vel_y
                )
            )
            
            # Apply magic wall constraint - bounce when hitting magic wall
            magic_wall_player = ground_constrained_player._replace(
                y=jnp.maximum(ground_constrained_player.y, self.consts.MAGIC_WALL_Y),
                vel_y=jnp.where(
                    jnp.logical_and(ground_constrained_player.y <= self.consts.MAGIC_WALL_Y, ground_constrained_player.vel_y < 0),
                    jnp.abs(ground_constrained_player.vel_y) * 0.8,  # Bounce down with 80% force
                    ground_constrained_player.vel_y
                )
            )
            
            # Update dragon based on level difficulty
            updated_dragon = self._update_dragon(state.dragon, magic_wall_player.x, state.level, new_time)
            
            # Check if dragon just shot (cooldown went from 0 to something)
            dragon_just_shot = jnp.logical_and(
                state.dragon.shoot_cooldown == 0,
                updated_dragon.shoot_cooldown > 0
            )
            
            # Update fireballs
            updated_fireballs = self._update_fireballs(state.fireballs, updated_dragon, 
                                                       magic_wall_player.x, state.quest, dragon_just_shot)
            
            # Check fireball collisions
            updated_player, updated_fireballs = self._check_fireball_collision(magic_wall_player, updated_fireballs)
            
            # Lava is fixed at Y=168
            updated_lava_y = jnp.array(self.consts.LAVA_START_Y, dtype=jnp.float32)
            
            # Check if player falls into lava - instant death and respawn
            player_in_lava = updated_player.y + self.consts.PLAYER_HEIGHT >= updated_lava_y
            
            # Instant respawn when hitting lava
            lava_player = jax.lax.cond(
                player_in_lava,
                lambda p: PlayerState(
                    x=jnp.array(self.consts.PLAYER_START_X_CASTLE, dtype=jnp.float32),
                    y=jnp.array(self.consts.PLAYER_START_Y_CASTLE, dtype=jnp.float32),
                    vel_x=jnp.array(0.0, dtype=jnp.float32),
                    vel_y=jnp.array(0.0, dtype=jnp.float32),
                    facing_left=jnp.array(True),
                    state=jnp.array(self.consts.PLAYER_STATE_IDLE, dtype=jnp.int32),
                    state_timer=jnp.array(0, dtype=jnp.int32),
                    is_flapping=jnp.array(False),
                    fire_was_pressed=jnp.array(False),
                    flap_cooldown=jnp.array(0, dtype=jnp.int32),
                    flap_anim_timer=jnp.array(0, dtype=jnp.int32),
                    death_timer=jnp.array(0, dtype=jnp.int32),
                    tie_cooldown=jnp.array(0, dtype=jnp.int32)
                ),
                lambda p: p,
                updated_player
            )
            
            # Deduct a life if hit lava
            lava_lives = jnp.where(player_in_lava, state.lives - 1, state.lives)
            lava_score = state.score  # Keep score the same
            
            # Check for dragon defeat - player hits magic wall while dragon is within radius
            player_at_barrier = magic_wall_player.y <= self.consts.MAGIC_WALL_Y
            dragon_distance = jnp.abs(magic_wall_player.x - (updated_dragon.x + self.consts.DRAGON_WIDTH // 2))
            dragon_in_range = dragon_distance <= self.consts.DRAGON_DEFEAT_RADIUS
            dragon_defeated = jnp.logical_and(
                jnp.logical_and(player_at_barrier, dragon_in_range),
                updated_dragon.is_active
            )
            
            # Deactivate dragon if defeated
            final_dragon = updated_dragon._replace(
                is_active=jnp.logical_and(
                    updated_dragon.is_active,
                    jnp.logical_not(dragon_defeated)
                )
            )
            
            # Dragon survival scoring - award points every 1.5 seconds
            new_survival_timer = state.dragon_survival_timer + 1
            survival_points_earned = new_survival_timer >= self.consts.DRAGON_SURVIVAL_TIMER
            
            # Get survival points based on level
            survival_points = jax.lax.switch(
                (state.level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
                [lambda: self.consts.POINTS_DRAGON_SURVIVE_L2, 
                 lambda: self.consts.POINTS_DRAGON_SURVIVE_L4,
                 lambda: self.consts.POINTS_DRAGON_SURVIVE_L6,
                 lambda: self.consts.POINTS_DRAGON_SURVIVE_L8]
            )
            
            # Add survival points if timer reached
            score_with_survival = jnp.where(
                jnp.logical_and(survival_points_earned, updated_dragon.is_active),
                state.score + survival_points,
                state.score
            )
            
            # Reset timer if points earned
            final_survival_timer = jnp.where(
                survival_points_earned,
                0,
                new_survival_timer
            )
            
            # Get dragon defeat points based on level
            dragon_kill_points = jax.lax.switch(
                (state.level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
                [lambda: self.consts.POINTS_DRAGON_KILL_L2, 
                 lambda: self.consts.POINTS_DRAGON_KILL_L4,
                 lambda: self.consts.POINTS_DRAGON_KILL_L6,
                 lambda: self.consts.POINTS_DRAGON_KILL_L8]
            )
            
            # Get maiden save bonus based on level (awarded at stage completion)
            maiden_save_points = jax.lax.switch(
                (state.level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
                [lambda: self.consts.POINTS_DRAGON_SAVE_L2, 
                 lambda: self.consts.POINTS_DRAGON_SAVE_L4,
                 lambda: self.consts.POINTS_DRAGON_SAVE_L6,
                 lambda: self.consts.POINTS_DRAGON_SAVE_L8]
            )
            
            # Award points for defeating dragon
            final_score = jnp.where(
                dragon_defeated,
                score_with_survival + dragon_kill_points + maiden_save_points,
                score_with_survival
            )
            
            # No enemies in stage 2
            return (lava_player, state.enemies, final_dragon, updated_fireballs, 
                    jnp.where(player_in_lava, lava_score, final_score), 
                    state.last_kill_time, state.consecutive_quick_kills, updated_lava_y, lava_lives, final_survival_timer)
        
        # Choose update based on stage
        stage_results = jax.lax.cond(
            state.stage == 2,
            update_stage_2,
            update_stage_1
        )
        
        new_player, new_enemies, new_dragon, new_fireballs, new_score, new_last_kill_time, new_consecutive_kills, new_lava_y, new_lives_from_stage, new_dragon_survival_timer = stage_results
        
        # Check for extra life bonus
        extra_life_earned = (new_score // self.consts.EXTRA_LIFE_POINTS) > (state.last_extra_life_score // self.consts.EXTRA_LIFE_POINTS)
        new_lives_with_bonus = jnp.where(
            extra_life_earned,
            jnp.minimum(new_lives_from_stage + 1, self.consts.MAX_LIVES),
            new_lives_from_stage
        )
        new_last_extra_life_score = jnp.where(
            extra_life_earned,
            (new_score // self.consts.EXTRA_LIFE_POINTS) * self.consts.EXTRA_LIFE_POINTS,
            state.last_extra_life_score
        )
        
        # Handle player death and respawn
        new_player, new_lives, new_game_over = self._handle_player_death(
            new_player, new_lives_with_bonus, state.stage
        )

        # Cancel enemy bounces if player just died (prevents visual bounce on death)
        died_this_frame = jnp.logical_and(
            new_player.state == self.consts.PLAYER_STATE_DEATH,
            new_player.death_timer == 1  # Death timer is set to 1 on the first death frame
        )

        # Cancel any in-progress bounce animations and rewind Y position to hide bounce
        new_enemies = jax.lax.cond(
            died_this_frame,
            lambda e: e._replace(
                # Rewind just Y to last frame's Y so any in-progress bounce this frame is invisible
                positions=e.positions.at[:, 1].set(state.enemies.positions[:, 1]),
                bounce_timer=jnp.zeros_like(e.bounce_timer),  # Stop bounce animation
                bounce_target_y=state.enemies.positions[:, 1],  # Reset target to original position
                vel_y=jnp.zeros_like(e.vel_y)  # Freeze vertical motion
            ),
            lambda e: e,
            new_enemies
        )
        
        # Check stage completion
        # Stage 1: all enemies defeated
        # Stage 2: dragon defeated
        stage_1_complete = jnp.logical_and(
            state.stage == 1,
            jnp.logical_not(jnp.any(new_enemies.active))
        )
        stage_2_complete = jnp.logical_and(
            state.stage == 2,
            jnp.logical_not(new_dragon.is_active)
        )
        new_stage_complete = jnp.logical_or(
            state.stage_complete,
            jnp.logical_or(stage_1_complete, stage_2_complete)
        )
        
        # Create new state
        new_state = SirLancelotState(
            player=new_player,
            enemies=new_enemies,
            dragon=new_dragon,
            fireballs=new_fireballs,
            score=new_score,
            lives=new_lives,
            quest=state.quest,
            level=state.level,
            stage=state.stage,
            time=new_time,
            game_over=new_game_over,
            stage_complete=new_stage_complete,
            last_kill_time=new_last_kill_time,
            consecutive_quick_kills=new_consecutive_kills,
            last_extra_life_score=new_last_extra_life_score,
            key=state.key,
            lava_y=new_lava_y,
            dragon_survival_timer=new_dragon_survival_timer
        )
        
        # Handle level progression when stage complete
        # Stage 1: check for enemy defeat
        all_enemies_defeated = jnp.logical_not(jnp.any(new_enemies.active))
        stage_1_complete = jnp.logical_and(state.stage == 1, all_enemies_defeated)
        
        # Stage 2: check for dragon defeat
        dragon_defeated = jnp.logical_not(new_dragon.is_active)
        stage_2_complete = jnp.logical_and(state.stage == 2, dragon_defeated)
        
        # Should advance if either stage is complete (and not already marked complete)
        should_advance = jnp.logical_and(
            jnp.logical_or(stage_1_complete, stage_2_complete),
            jnp.logical_not(state.stage_complete)
        )
        
        new_state = jax.lax.cond(
            should_advance,
            self._advance_to_next_level,
            lambda s: s,
            new_state
        )
        
        reward = new_state.score - state.score
        # Game ends on game over only (levels loop back after 8)
        done = new_state.game_over

        return self._get_observation(new_state), new_state, reward, done, self._get_info(new_state)
    
    def _advance_to_next_level(self, state: SirLancelotState) -> SirLancelotState:
        """Advance to the next level or loop back after level 8.
        
        Handles level progression, stage switching, and state resets.
        
        Args:
            state: Current game state
            
        Returns:
            Updated game state for the new level
        """
        # Advance to next level
        next_level = state.level + 1
        
        # After level 8, loop back to level 1 with increased quest
        new_level = jnp.where(
            next_level > self.consts.MAX_LEVEL,
            1,
            next_level
        )
        new_quest = jnp.where(
            next_level > self.consts.MAX_LEVEL,
            jnp.minimum(state.quest + 1, self.consts.MAX_QUEST),
            state.quest
        )
        
        # Determine stage type (odd = beast, even = dragon)
        new_stage = jnp.where(
            new_level % 2 == 0,
            2,  # Dragon stage
            1   # Beast stage
        )
        
        # Reset player position based on stage type
        spawn_x = jnp.where(
            new_stage == 2,
            self.consts.PLAYER_START_X_CASTLE,
            self.consts.PLAYER_START_X
        ).astype(jnp.float32)
        spawn_y = jnp.where(
            new_stage == 2,
            self.consts.PLAYER_START_Y_CASTLE,
            self.consts.PLAYER_START_Y
        ).astype(jnp.float32)
        
        new_player = PlayerState(
            x=spawn_x,
            y=spawn_y,
            vel_x=jnp.array(0.0, dtype=jnp.float32),
            vel_y=jnp.array(0.0, dtype=jnp.float32),
            facing_left=jnp.array(True),
            state=jnp.array(self.consts.PLAYER_STATE_IDLE, dtype=jnp.int32),
            state_timer=jnp.array(0, dtype=jnp.int32),
            is_flapping=jnp.array(False),
            fire_was_pressed=jnp.array(False),
            flap_cooldown=jnp.array(0, dtype=jnp.int32),
            flap_anim_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            tie_cooldown=jnp.array(0, dtype=jnp.int32)
        )
        
        # Reset or activate dragon based on stage
        new_dragon = state.dragon._replace(
            x=jnp.array(self.consts.DRAGON_START_X, dtype=jnp.float32),
            vel_x=jnp.array(self.consts.DRAGON_SPEED, dtype=jnp.float32),
            facing_left=jnp.array(False),
            wing_frame=jnp.array(0, dtype=jnp.int32),
            animation_timer=jnp.array(0, dtype=jnp.int32),
            shoot_cooldown=jnp.array(self.consts.FIREBALL_SHOOT_COOLDOWN, dtype=jnp.int32),
            is_active=jnp.array(new_stage == 2)  # Only active on dragon stages
        )
        
        # Reset enemies for beast stages
        def reset_enemies():
            enemy_x = jnp.full(self.consts.NUM_ENEMIES, self.consts.SCREEN_WIDTH, dtype=jnp.float32)
            enemy_positions = jnp.stack([enemy_x, self.consts.ENEMY_SPAWN_Y], axis=1)
            
            # Initialize enemy velocities based on level
            # For level 1, generate random speeds for each beast
            key = state.key
            keys = jax.random.split(key, 2)
            random_speeds = jax.random.uniform(keys[0], shape=(self.consts.NUM_ENEMIES,), 
                                              minval=self.consts.ENEMY_BASE_SPEED - self.consts.ENEMY_SPEED_VARIATION,
                                              maxval=self.consts.ENEMY_BASE_SPEED + self.consts.ENEMY_SPEED_VARIATION)
            # All beast levels (1, 3, 5, 7) should have horizontal movement
            is_beast_level = (new_level % 2) == 1
            enemy_vel_x = jnp.where(
                is_beast_level,
                -random_speeds,  # Beast levels: move left with random speeds
                jnp.zeros(self.consts.NUM_ENEMIES)
            )
            
            # Generate random animation speeds for each enemy
            anim_keys = jax.random.split(keys[1], self.consts.NUM_ENEMIES)
            random_anim_speeds = jax.vmap(lambda k: jax.random.randint(
                k, shape=(), 
                minval=self.consts.ENEMY_ANIMATION_SPEED - self.consts.ENEMY_ANIMATION_VARIATION,
                maxval=self.consts.ENEMY_ANIMATION_SPEED + self.consts.ENEMY_ANIMATION_VARIATION + 1
            ))(anim_keys)
            
            return EnemyState(
                positions=enemy_positions,
                vel_x=enemy_vel_x,
                vel_y=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.float32),
                active=jnp.ones(self.consts.NUM_ENEMIES, dtype=bool),
                animation_frame=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32),
                facing_left=jnp.ones(self.consts.NUM_ENEMIES, dtype=bool),
                invisibility_timer=jnp.where(
                    new_level == 7,
                    jax.random.randint(state.key, (self.consts.NUM_ENEMIES,), 0, 200),  # Random start times
                    jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32)
                ),
                is_invisible=jnp.zeros(self.consts.NUM_ENEMIES, dtype=bool),
                animation_timer=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32),
                animation_speed=random_anim_speeds,
                bounce_target_y=enemy_positions[:, 1],  # Initialize with current positions
                bounce_timer=jnp.zeros(self.consts.NUM_ENEMIES, dtype=jnp.int32)
            )
        
        new_enemies = jax.lax.cond(
            new_stage == 1,
            reset_enemies,
            lambda: state.enemies
        )
        
        # Award completion bonus based on level
        completion_bonus = jax.lax.switch(
            state.level - 1,  # Current level (before advancing)
            [lambda: 5000, lambda: 0, lambda: 7500, lambda: 0, lambda: 10000, lambda: 0, lambda: 15000, lambda: 0]  # Bonuses for completing levels 1,3,5,7 (0 for dragon levels 2,4,6,8)
        )
        
        return state._replace(
            player=new_player,
            enemies=new_enemies,
            dragon=new_dragon,
            stage=new_stage,
            level=new_level,
            quest=new_quest,
            stage_complete=False,  # Reset stage complete flag
            score=state.score + completion_bonus,
            lava_y=jnp.array(self.consts.LAVA_START_Y, dtype=jnp.float32),
            dragon_survival_timer=jnp.array(0, dtype=jnp.int32)  # Reset survival timer
        )
    
    # =========================================================================
    # FLYING PHYSICS IMPLEMENTATION - ALL FLYING LOGIC IN ONE PLACE
    # =========================================================================
    
    def _update_player_state_machine(self, player: PlayerState, action: int) -> PlayerState:
        """Update player state based on current state and action.
        
        Routes to appropriate state handler (idle, flying, stunned, death).
        
        Args:
            player: Current player state
            action: Player input action
            
        Returns:
            Updated player state
        """
        # Route to appropriate state handler
        return jax.lax.switch(
            player.state,
            [
                self._update_player_idle,
                self._update_player_flying,
                self._update_player_stunned,
                self._update_player_death
            ],
            player, action
        )
    
    # -------------------------------------------------------------------------
    # MAIN FLYING PHYSICS FUNCTION - EDIT THIS TO CHANGE FLYING BEHAVIOR
    # -------------------------------------------------------------------------
    def _apply_physics(self, player: PlayerState, action: int) -> PlayerState:
        """Core physics engine for player movement.
        
        Implements moon-like gravity with flapping mechanics.
        Handles horizontal momentum, vertical thrust, and collisions.
        
        Args:
            player: Current player state
            action: Player input action
            
        Returns:
            Updated player state with physics applied
        """
        # Extract action components
        direction_left = action == Action.LEFT
        direction_right = action == Action.RIGHT
        fire_pressed = action == Action.FIRE
        left_fire = action == Action.LEFTFIRE
        right_fire = action == Action.RIGHTFIRE
        
        any_fire = jnp.logical_or(jnp.logical_or(fire_pressed, left_fire), right_fire)
        fire_just_pressed = jnp.logical_and(any_fire, jnp.logical_not(player.fire_was_pressed))
        
        # 2.6 FACING DIRECTION
        # If you pressed ← this frame you face left; if → you face right
        new_facing_left = jnp.where(
            jnp.logical_or(direction_left, left_fire),
            True,
            jnp.where(
                jnp.logical_or(direction_right, right_fire),
                False,
                player.facing_left
            )
        )
        
        # 2.1 GRAVITY
        # vel_y ← vel_y + 0.5 (pixel per frame, downward)
        new_vel_y = player.vel_y + self.consts.GRAVITY
        
        # Update flap cooldown
        new_flap_cooldown = jnp.maximum(player.flap_cooldown - 1, 0)
        
        # 2.2 WING-FLAP IMPULSE
        # If fire pressed AND cooldown == 0
        can_flap = jnp.logical_and(fire_just_pressed, new_flap_cooldown == 0)
        
        # Apply flap impulse - STACKING MOMENTUM (adds to current velocity)
        new_vel_y = jnp.where(
            can_flap,
            player.vel_y + self.consts.FLAP_IMPULSE,  # ADD to current velocity
            new_vel_y
        )
        
        # Set cooldown
        new_flap_cooldown = jnp.where(
            can_flap,
            self.consts.FLAP_COOLDOWN,  # 5 frames
            new_flap_cooldown
        )
        
        # Flap animation timer
        new_flap_anim_timer = jnp.where(
            can_flap,
            self.consts.FLAP_ANIM_FRAMES,              # Reset timer when you flap
            jnp.maximum(player.flap_anim_timer - 1, 0) # Otherwise count down
        )
        new_is_flapping = new_flap_anim_timer > 0
        
        # 2.3 VERTICAL SPEED LIMITS - moved after ceiling bounce to preserve bounce velocity
        
        # ------------------------------------------------------------------
        # 2.4 ONE-SHOT SIDE IMPULSE
        # ------------------------------------------------------------------
        side_impulse = jnp.where(
            jnp.logical_and(fire_just_pressed, left_fire), -self.consts.HORIZONTAL_FLAP_IMPULSE,
            jnp.where(
                jnp.logical_and(fire_just_pressed, right_fire), self.consts.HORIZONTAL_FLAP_IMPULSE,
                0.0
            )
        )
        
        # Apply decay first, then add impulse (better momentum feel)
        new_vel_x = player.vel_x * self.consts.HORIZONTAL_DECAY + side_impulse
        new_vel_x = jnp.clip(new_vel_x, -self.consts.MAX_VEL_X, self.consts.MAX_VEL_X)

        # 2.5 VELOCITY INTEGRATION BEFORE POSITION UPDATE
        # --- Compute tentative velocities ---
        # (new_vel_y already computed above with gravity and flapping)

        # Predict next Y just to detect floor/ceiling using the tentative velocity
        predicted_y = player.y + new_vel_y
        hitting_ceiling = predicted_y <= self.consts.PLAY_AREA_TOP
        hitting_ground = predicted_y >= (self.consts.PLAY_AREA_BOTTOM - self.consts.PLAYER_HEIGHT)

        # Apply reactions to the VELOCITY first
        vel_y_after = jnp.where(hitting_ceiling, jnp.abs(new_vel_y), new_vel_y)  # Bounce off ceiling
        vel_y_after = jnp.where(hitting_ground, 0.0, vel_y_after)  # Stop at ground

        # Now clamp vertical speed (prevents any single-frame jump)
        vel_y_after = jnp.clip(vel_y_after, self.consts.MAX_VEL_Y_UP, self.consts.MAX_VEL_Y_DOWN)

        # Ground friction
        new_vel_x = jnp.where(hitting_ground, new_vel_x * 0.7, new_vel_x)

        # Finally move using the post-reaction, clamped velocity
        new_x = player.x + new_vel_x
        new_y = jnp.clip(player.y + vel_y_after,
                        self.consts.PLAY_AREA_TOP,
                        self.consts.PLAY_AREA_BOTTOM - self.consts.PLAYER_HEIGHT)

        # Horizontal wrapping (immediate for Level 1)
        new_x = jnp.where(
            new_x < 0,
            self.consts.SCREEN_WIDTH - self.consts.PLAYER_WIDTH,
            jnp.where(
                new_x > self.consts.SCREEN_WIDTH - self.consts.PLAYER_WIDTH,
                0,
                new_x
            )
        )

        # Final velocity is vel_y_after
        new_vel_y = vel_y_after
        
        # State management
        new_state = jnp.where(
            hitting_ground,
            self.consts.PLAYER_STATE_IDLE,
            self.consts.PLAYER_STATE_FLYING
        )
        
        return player._replace(
            x=new_x,
            y=new_y,
            vel_x=new_vel_x,
            vel_y=new_vel_y,
            facing_left=new_facing_left,
            state=new_state,
            state_timer=jnp.where(new_state != player.state, 0, player.state_timer + 1),
            is_flapping=new_is_flapping,
            fire_was_pressed=any_fire,
            flap_cooldown=new_flap_cooldown,
            flap_anim_timer=new_flap_anim_timer
        )
    
    # -------------------------------------------------------------------------
    # PLAYER STATE HANDLERS - These delegate to _apply_physics for flying
    # -------------------------------------------------------------------------
    def _update_player_idle(self, player: PlayerState, action: int) -> PlayerState:
        """Update player in idle/ground state.
        
        Uses same physics as flying state for consistency.
        """
        return self._apply_physics(player, action)
    
    def _update_player_flying(self, player: PlayerState, action: int) -> PlayerState:
        """Update player in flying state.
        
        Uses same physics as ground state for consistency.
        """
        return self._apply_physics(player, action)
    
    # =========================================================================
    # END OF FLYING PHYSICS IMPLEMENTATION
    # =========================================================================
    
    def _update_player_stunned(self, player: PlayerState, action: int) -> PlayerState:
        """Update stunned player state.
        
        Player falls with gravity but cannot control movement.
        Stun timer counts down until recovery.
        
        Args:
            player: Current player state
            action: Player input (ignored while stunned)
            
        Returns:
            Updated player state
        """
        # Player is stunned - falls and can't move
        new_vel_y = jnp.minimum(
            player.vel_y + self.consts.GRAVITY,
            self.consts.MAX_VEL_Y_DOWN
        )
        
        # Update position with bounds
        new_y = jnp.clip(
            player.y + new_vel_y,
            self.consts.PLAY_AREA_TOP,
            self.consts.PLAY_AREA_BOTTOM - self.consts.PLAYER_HEIGHT
        )
        
        new_state_timer = player.state_timer + 1
        
        # Return to flying after stun duration
        new_state = jnp.where(
            new_state_timer >= self.consts.STUN_DURATION_FRAMES,
            self.consts.PLAYER_STATE_FLYING,
            self.consts.PLAYER_STATE_STUNNED
        )
        
        return player._replace(
            x=player.x,  # Keep x as float32
            y=new_y,
            vel_x=jnp.array(0.0, dtype=jnp.float32),
            vel_y=new_vel_y,
            facing_left=player.facing_left,  # Keep facing direction
            state=new_state,
            state_timer=jnp.where(new_state != self.consts.PLAYER_STATE_STUNNED, 0, new_state_timer),
            is_flapping=False,
            fire_was_pressed=False,
            flap_cooldown=jnp.array(0, dtype=jnp.int32),
            flap_anim_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=player.death_timer  # Keep death timer
        )
    
    def _update_player_death(self, player: PlayerState, action: int) -> PlayerState:
        """Update dying player state.
        
        Player falls during death animation.
        Death timer counts up until respawn.
        
        Args:
            player: Current player state
            action: Player input (ignored while dying)
            
        Returns:
            Updated player state
        """
        # Player falls during death animation
        new_vel_y = jnp.minimum(
            player.vel_y + self.consts.GRAVITY,
            self.consts.MAX_VEL_Y_DOWN
        )
        
        new_y = jnp.minimum(
            player.y + new_vel_y,
            self.consts.PLAY_AREA_BOTTOM - self.consts.PLAYER_HEIGHT
        )
        
        new_death_timer = player.death_timer + 1
        
        return player._replace(
            x=player.x,  # Keep x as float32
            y=new_y,
            vel_x=jnp.array(0.0, dtype=jnp.float32),
            vel_y=new_vel_y,
            facing_left=player.facing_left,  # Keep facing direction
            state=player.state,  # Keep state as DEATH
            state_timer=player.state_timer + 1,
            is_flapping=False,
            fire_was_pressed=False,
            flap_cooldown=jnp.array(0, dtype=jnp.int32),
            flap_anim_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=new_death_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_beasts(self, enemies: EnemyState, time: chex.Array, level: chex.Array, quest: chex.Array) -> EnemyState:
        """Update all enemies based on level-specific behaviors.

        Each level has unique enemy movement patterns:
        - Level 1: Simple horizontal with vertical drift
        - Level 3: Horizontal only with speed changes
        - Level 5/7: Border bouncing with random movement

        Args:
            enemies: Current enemy states
            time: Current game time
            level: Current level (1,3,5,7)
            quest: Current quest/difficulty
            
        Returns:
            Updated enemy states
        """
        # Update beasts based on level (1, 3, 5, or 7) and quest difficulty
        
        # Level 1: Horizontal movement only (speeds are set at initialization)
        def update_level_1():
            # Keep existing velocities (they were randomized at initialization)
            new_vel_x = enemies.vel_x
            # Preserve vertical velocity if it's non-zero (from collision bounce)
            # Apply damping to vertical velocity to make it decay over time
            new_vel_y = enemies.vel_y * 0.85  # Decay bounce velocity
            return new_vel_x, new_vel_y
        
        # Level 3: Horizontal movement only with random speed changes
        def update_level_3():
            # Random speed changes for horizontal movement
            key = jax.random.PRNGKey(time.astype(jnp.int32))
            keys = jax.random.split(key, self.consts.NUM_ENEMIES * 3)
            
            # Apply quest difficulty multiplier
            quest_multiplier = 1.0 + (quest - 1) * self.consts.ENEMY_SPEED_QUEST_MULTIPLIER
            
            # Random deceleration/acceleration factor for each enemy
            speed_change_factors = jax.vmap(lambda k: jax.random.uniform(k, minval=0.85, maxval=1.15))(keys[:self.consts.NUM_ENEMIES])
            new_vel_x = enemies.vel_x * speed_change_factors
            
            # When velocity is near 0, reverse direction with new random speed
            near_zero = jnp.abs(new_vel_x) < 0.05
            new_directions = jax.vmap(lambda k: jax.random.choice(k, jnp.array([-1.0, 1.0])))(keys[self.consts.NUM_ENEMIES:2*self.consts.NUM_ENEMIES])
            new_speeds = jax.vmap(lambda k: jax.random.uniform(k, minval=0.3, maxval=0.7))(keys[2*self.consts.NUM_ENEMIES:]) * quest_multiplier
            
            # Apply direction reversal when near zero
            new_vel_x = jnp.where(
                near_zero,
                new_directions * new_speeds,
                new_vel_x
            )
            
            # Clamp horizontal speed
            max_speed = 0.8 * quest_multiplier
            new_vel_x = jnp.clip(new_vel_x, -max_speed, max_speed)
            
            # Preserve vertical velocity if it's non-zero (from collision bounce)
            # Apply damping to vertical velocity to make it decay over time
            new_vel_y = enemies.vel_y * 0.85  # Decay bounce velocity
            return new_vel_x, new_vel_y
        
        # Level 5: Full random movement
        def update_level_5():
            # Random acceleration
            key = jax.random.PRNGKey(time.astype(jnp.int32))
            keys = jax.random.split(key, self.consts.NUM_ENEMIES * 2)
            
            # Apply quest difficulty multiplier
            quest_multiplier = 1.0 + (quest - 1) * self.consts.ENEMY_SPEED_QUEST_MULTIPLIER
            speed_limit = self.consts.ENEMY_RANDOM_SPEED_RANGE * quest_multiplier
            
            accel_x = jax.vmap(lambda k: jax.random.uniform(k, minval=-0.02, maxval=0.02))(keys[:self.consts.NUM_ENEMIES])
            accel_y = jax.vmap(lambda k: jax.random.uniform(k, minval=-0.02, maxval=0.02))(keys[self.consts.NUM_ENEMIES:])
            
            new_vel_x = jnp.clip(enemies.vel_x + accel_x, -speed_limit, speed_limit)
            new_vel_y = jnp.clip(enemies.vel_y + accel_y, -speed_limit, speed_limit)
            return new_vel_x, new_vel_y
        
        # Choose update function based on level
        level_index = jnp.where(
            level <= 1, 0,
            jnp.where(level <= 3, 1, 2)
        )  # 1->0, 3->1, 5/7->2
        
        new_vel_x, new_vel_y = jax.lax.switch(
            level_index,
            [update_level_1, update_level_3, update_level_5]
        )
        
        # Handle smooth bounce animation
        # Decrement bounce timer
        new_bounce_timer = jnp.maximum(0, enemies.bounce_timer - 1)

        # Calculate interpolation factor for smooth movement (0 to 1 over 24 frames)
        BOUNCE_DURATION = 24.0
        interpolation_factor = jnp.where(
            enemies.bounce_timer > 0,
            1.0 - (new_bounce_timer / BOUNCE_DURATION),  # 0 to 1 as timer goes down
            0.0
        )

        # Smooth Y position interpolation towards target
        interpolated_y = jnp.where(
            enemies.bounce_timer > 0,
            enemies.positions[:, 1] + (enemies.bounce_target_y - enemies.positions[:, 1]) * (1.0 / BOUNCE_DURATION),
            enemies.positions[:, 1] + new_vel_y  # Normal velocity-based movement
        )

        # Update positions
        new_x = enemies.positions[:, 0] + new_vel_x
        new_y = interpolated_y
        
        # Screen wrapping for enemies (immediate)
        new_x = jnp.where(
            new_x < -self.consts.ENEMY_WIDTH,
            self.consts.SCREEN_WIDTH,
            jnp.where(
                new_x > self.consts.SCREEN_WIDTH,
                -self.consts.ENEMY_WIDTH,
                new_x
            )
        )
        
        # Border bouncing for levels 5 and 7
        min_y = self.consts.PLAY_AREA_TOP
        max_y = jnp.where(
            jnp.logical_or(level == 5, level == 7),
            self.consts.ENEMY_MIN_Y,  # Ground safety for levels 5 and 7
            self.consts.PLAY_AREA_BOTTOM - self.consts.ENEMY_HEIGHT
        )
        
        # Check if beasts hit top or bottom borders
        hit_top = new_y <= min_y
        hit_bottom = new_y >= max_y
        
        # Reverse Y velocity when hitting borders (for levels 5 and 7)
        should_bounce = jnp.logical_or(level == 5, level == 7)
        new_vel_y = jnp.where(
            jnp.logical_and(should_bounce, hit_top),
            jnp.abs(new_vel_y),  # Bounce down (positive velocity)
            jnp.where(
                jnp.logical_and(should_bounce, hit_bottom),
                -jnp.abs(new_vel_y),  # Bounce up (negative velocity)
                new_vel_y
            )
        )
        
        # Clamp positions
        new_y = jnp.clip(new_y, min_y, max_y)

        new_positions = jnp.stack([new_x, new_y], axis=1)
        
        # Update animation with individual timers
        new_animation_timer = enemies.animation_timer + 1
        # Check if each enemy should change animation
        should_change = new_animation_timer >= enemies.animation_speed
        new_animation = jnp.where(
            should_change,
            (enemies.animation_frame + 1) % 16,
            enemies.animation_frame
        )
        # Reset timer when animation changes
        new_animation_timer = jnp.where(should_change, 0, new_animation_timer)
        
        # Update facing direction based on velocity
        new_facing_left = jnp.where(
            new_vel_x < 0, True,
            jnp.where(new_vel_x > 0, False, enemies.facing_left)
        )
        
        # Level 7: Handle invisibility
        def update_invisibility():
            # Decrement timers
            new_timers = jnp.maximum(0, enemies.invisibility_timer - 1)
            
            # Switch states when timer reaches 0
            switching = new_timers == 0
            new_invisible = jnp.where(
                switching,
                jnp.logical_not(enemies.is_invisible),
                enemies.is_invisible
            )
            
            # Reset timers when switching (scale invisibility by quest)
            quest_multiplier = 1.0 + (quest - 1) * self.consts.INVISIBILITY_QUEST_MULTIPLIER
            base_invis_duration = (self.consts.INVISIBILITY_DURATION * quest_multiplier).astype(jnp.int32)
            base_vis_duration = self.consts.VISIBILITY_DURATION
            
            # Add random variation to each enemy's timing (±30 frames)
            key = jax.random.PRNGKey(time.astype(jnp.int32))
            keys = jax.random.split(key, self.consts.NUM_ENEMIES)
            random_variation = jax.vmap(lambda k: jax.random.randint(k, shape=(), minval=-30, maxval=31))(keys)
            
            # Calculate new duration with variation
            new_duration = jnp.where(
                new_invisible,
                base_invis_duration + random_variation,
                base_vis_duration + random_variation
            )
            new_duration = jnp.maximum(30, new_duration)  # Ensure minimum duration
            
            new_timers = jnp.where(
                switching,
                new_duration,
                new_timers
            )
            
            return new_timers, new_invisible
        
        # Only update invisibility for level 7
        def no_invisibility():
            return enemies.invisibility_timer, enemies.is_invisible
            
        invisibility_timer, is_invisible = jax.lax.cond(
            level == 7,
            update_invisibility,
            no_invisibility
        )
        
        return EnemyState(
            positions=new_positions,
            vel_x=new_vel_x,
            vel_y=new_vel_y,
            active=enemies.active,
            animation_frame=new_animation,
            facing_left=new_facing_left,
            invisibility_timer=invisibility_timer,
            is_invisible=is_invisible,
            animation_timer=new_animation_timer,
            animation_speed=enemies.animation_speed,
            bounce_target_y=enemies.bounce_target_y,  # Keep existing targets
            bounce_timer=new_bounce_timer  # Update timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_dragon(self, dragon: DragonState, player_x: chex.Array, level: chex.Array, time: chex.Array) -> DragonState:
        """Update dragon movement and animation.
        
        Dragon AI follows player within distance thresholds that vary by level.
        Always keeps moving, reversing at boundaries.
        
        Args:
            dragon: Current dragon state
            player_x: Player X position
            level: Current level (2,4,6,8)
            time: Current game time
            
        Returns:
            Updated dragon state
        """
        # Dragon only active in stage 2
        active = dragon.is_active
        
        # Get speed based on level (2, 4, 6, or 8)
        dragon_speed = jax.lax.switch(
            (level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
            [lambda: self.consts.DRAGON_SPEED, lambda: self.consts.DRAGON_SPEED_LEVEL_4, 
             lambda: self.consts.DRAGON_SPEED_LEVEL_6, lambda: self.consts.DRAGON_SPEED_LEVEL_8]
        )
        
        # Dragon follows player within allowed distance threshold
        # Higher levels have smaller allowed distance (dragon stays closer)
        allowed_distance = jax.lax.switch(
            (level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
            [lambda: 80.0,   # Level 2: can be 80 pixels away
             lambda: 60.0,   # Level 4: 60 pixels away
             lambda: 40.0,   # Level 6: 40 pixels away
             lambda: 25.0]   # Level 8: 25 pixels away (very aggressive)
        )
        
        # Calculate distance to player
        dragon_center_x = dragon.x + self.consts.DRAGON_WIDTH // 2
        player_center_x = player_x + self.consts.PLAYER_WIDTH // 2
        distance_to_player = jnp.abs(dragon_center_x - player_center_x)
        
        # Determine if dragon should move towards player
        too_far_from_player = distance_to_player > allowed_distance
        player_is_left = player_center_x < dragon_center_x
        
        # Dragon should always be moving, not stationary
        # If within range, continue current direction but reverse at boundaries
        # If too far, move towards player
        
        # First, check if we need to reverse direction at boundaries
        will_hit_left = (dragon.x + dragon.vel_x) <= self.consts.DRAGON_MIN_X
        will_hit_right = (dragon.x + dragon.vel_x) >= self.consts.DRAGON_MAX_X
        should_reverse = jnp.logical_or(will_hit_left, will_hit_right)
        
        # Calculate base velocity (always moving)
        base_vel_x = jnp.where(
            should_reverse,
            -dragon.vel_x,  # Reverse direction at boundaries
            jnp.where(
                jnp.abs(dragon.vel_x) < 0.1,  # If stopped, start moving
                dragon_speed,  # Default to moving right
                dragon.vel_x   # Keep current velocity
            )
        )
        
        # Override with player tracking if too far
        new_vel_x = jnp.where(
            too_far_from_player,
            jnp.where(
                player_is_left,
                -dragon_speed,  # Move left towards player
                dragon_speed    # Move right towards player
            ),
            base_vel_x  # Use base velocity (always moving)
        )
        
        # Update position
        new_x = dragon.x + new_vel_x
        
        # Apply boundary constraints
        new_x = jnp.clip(new_x, self.consts.DRAGON_MIN_X, self.consts.DRAGON_MAX_X)
        
        # Update facing direction based on velocity
        new_facing_left = jnp.where(
            new_vel_x < 0,
            True,  # Moving left, face left
            jnp.where(
                new_vel_x > 0,
                False,  # Moving right, face right
                dragon.facing_left  # Keep current facing if not moving
            )
        )
        
        # Update wing animation
        # Wing cycle: up (0) -> middle_up (1) -> down (2) -> middle_down (3) -> up (0)
        new_animation_timer = dragon.animation_timer + 1
        
        # Change wing frame every DRAGON_WING_CYCLE_FRAMES
        change_frame = new_animation_timer >= self.consts.DRAGON_WING_CYCLE_FRAMES
        new_wing_frame = jnp.where(
            change_frame,
            (dragon.wing_frame + 1) % 4,
            dragon.wing_frame
        )
        new_animation_timer = jnp.where(change_frame, 0, new_animation_timer)
        
        # Update shoot cooldown
        new_shoot_cooldown = jnp.maximum(0, dragon.shoot_cooldown - 1)
        
        # Check if player is within shooting zone (varies by level)
        shoot_zone = jax.lax.switch(
            (level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
            [lambda: self.consts.FIREBALL_SHOOT_ZONE,         # Level 2: 40 pixels
             lambda: self.consts.DRAGON_TARGET_ZONE_LEVEL_4,  # Level 4: 30 pixels
             lambda: self.consts.DRAGON_TARGET_ZONE_LEVEL_6,  # Level 6: 20 pixels
             lambda: self.consts.DRAGON_TARGET_ZONE_LEVEL_8]  # Level 8: 15 pixels
        )
        player_distance = jnp.abs(player_x - (new_x + self.consts.DRAGON_WIDTH // 2))
        player_in_zone = player_distance <= shoot_zone
        
        # All dragon levels (2, 4, 6, 8) can shoot fire
        can_shoot = jnp.logical_or(
            jnp.logical_or(level == 2, level == 4),
            jnp.logical_or(level == 6, level == 8)
        )
        
        # Dragon shoots when cooldown is 0 AND player is in zone AND dragon can shoot
        should_shoot = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(new_shoot_cooldown == 0, player_in_zone),
                can_shoot
            ),
            active
        )
        
        # Reset cooldown if shooting (varies by level)
        cooldown_duration = jax.lax.switch(
            (level - 2) // 2,  # 2->0, 4->1, 6->2, 8->3
            [lambda: self.consts.FIREBALL_SHOOT_COOLDOWN,      # Level 2: 240 frames (4 sec)
             lambda: self.consts.DRAGON_COOLDOWN_LEVEL_4,      # Level 4: 120 frames (2 sec)
             lambda: self.consts.DRAGON_COOLDOWN_LEVEL_6,      # Level 6: 80 frames (1.3 sec)
             lambda: self.consts.DRAGON_COOLDOWN_LEVEL_8]      # Level 8: 60 frames (1 sec)
        )
        new_shoot_cooldown = jnp.where(
            should_shoot,
            cooldown_duration,
            new_shoot_cooldown
        )
        
        return dragon._replace(
            x=new_x,
            vel_x=new_vel_x,
            facing_left=new_facing_left,
            wing_frame=new_wing_frame,
            animation_timer=new_animation_timer,
            shoot_cooldown=new_shoot_cooldown
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_fireballs(self, fireballs: FireballState, dragon: DragonState, 
                          player_x: chex.Array, quest: chex.Array, dragon_shooting: chex.Array) -> FireballState:
        """Update fireball positions and spawn new ones.
        
        Fireballs fall straight down from dragon position.
        Only one fireball spawns at a time.
        
        Args:
            fireballs: Current fireball states
            dragon: Dragon state for spawn position
            player_x: Player X position (unused)
            quest: Current quest (unused)
            dragon_shooting: Whether dragon just shot
            
        Returns:
            Updated fireball states
        """
        # Update existing fireballs - they fall straight down
        new_y = fireballs.positions[:, 1] + self.consts.FIREBALL_SPEED
        
        # Deactivate fireballs that go off screen OR hit the ground
        new_active = jnp.logical_and(
            fireballs.active,
            jnp.logical_and(
                new_y < self.consts.SCREEN_HEIGHT,
                new_y < self.consts.GROUND_Y - self.consts.FIREBALL_HEIGHT
            )
        )
        
        # Update positions
        new_positions = fireballs.positions.at[:, 1].set(new_y)
        
        # Find first inactive fireball slot
        def spawn_fireball(i, state):
            positions, active, anim, shooting = state
            can_spawn = jnp.logical_and(
                jnp.logical_not(active[i]),
                shooting
            )
            
            # Spawn at dragon position
            new_x = dragon.x + self.consts.DRAGON_WIDTH // 2 - self.consts.FIREBALL_WIDTH // 2
            new_y = self.consts.DRAGON_Y + self.consts.DRAGON_HEIGHT
            
            # Fireballs fall straight down from dragon (no aiming)
            
            positions = positions.at[i, 0].set(
                jnp.where(can_spawn, new_x, positions[i, 0])
            )
            positions = positions.at[i, 1].set(
                jnp.where(can_spawn, new_y, positions[i, 1])
            )
            active = active.at[i].set(
                jnp.where(can_spawn, True, active[i])
            )
            # Only consume the shooting flag for first fireball
            shooting = jnp.where(can_spawn, False, shooting)
            
            return positions, active, anim, shooting
        
        # Try to spawn one fireball
        final_positions, final_active, _, _ = jax.lax.fori_loop(
            0, self.consts.MAX_FIREBALLS,
            spawn_fireball,
            (new_positions, new_active, fireballs.animation_frame, dragon_shooting)
        )
        
        # Update animation timer
        new_animation_timer = fireballs.animation_timer + 1
        
        # Switch frame every FIREBALL_ANIMATION_SPEED frames
        should_switch = new_animation_timer >= self.consts.FIREBALL_ANIMATION_SPEED
        new_animation = jnp.where(
            should_switch,
            (fireballs.animation_frame + 1) % 2,
            fireballs.animation_frame
        )
        new_animation_timer = jnp.where(should_switch, 0, new_animation_timer)
        
        return FireballState(
            positions=final_positions,
            active=final_active,
            animation_frame=new_animation,
            animation_timer=new_animation_timer
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_fireball_collision(self, player: PlayerState, fireballs: FireballState) -> Tuple[PlayerState, FireballState]:
        """Check if any fireball hits the player and apply stun.
        
        Collision causes player to be stunned for 3 seconds.
        Fireball is deactivated on hit.
        
        Args:
            player: Current player state
            fireballs: Current fireball states
            
        Returns:
            Tuple of (updated_player, updated_fireballs)
        """
        # Player bounds
        p_left = player.x
        p_right = player.x + self.consts.PLAYER_WIDTH
        p_top = player.y
        p_bottom = player.y + self.consts.PLAYER_HEIGHT
        
        # Fireball bounds (vectorized)
        f_left = fireballs.positions[:, 0]
        f_right = fireballs.positions[:, 0] + self.consts.FIREBALL_WIDTH
        f_top = fireballs.positions[:, 1]
        f_bottom = fireballs.positions[:, 1] + self.consts.FIREBALL_HEIGHT
        
        # Check collisions
        x_overlap = jnp.logical_and(p_left < f_right, p_right > f_left)
        y_overlap = jnp.logical_and(p_top < f_bottom, p_bottom > f_top)
        collision_mask = jnp.logical_and(
            jnp.logical_and(x_overlap, y_overlap),
            fireballs.active
        )
        
        # Any collision stuns the player
        hit_by_fireball = jnp.any(collision_mask)
        
        # Apply stun if hit and not already stunned/dead
        can_be_stunned = jnp.logical_and(
            player.state == self.consts.PLAYER_STATE_FLYING,
            hit_by_fireball
        )
        
        new_state = jnp.where(
            can_be_stunned,
            self.consts.PLAYER_STATE_STUNNED,
            player.state
        )
        
        new_state_timer = jnp.where(
            can_be_stunned,
            0,  # Reset timer when stunned
            player.state_timer
        )
        
        # Despawn fireballs that hit the player
        new_fireballs_active = jnp.logical_and(
            fireballs.active,
            jnp.logical_not(collision_mask)
        )
        
        updated_fireballs = fireballs._replace(
            active=new_fireballs_active
        )
        
        return player._replace(
            state=new_state,
            state_timer=new_state_timer
        ), updated_fireballs
    
    @partial(jax.jit, static_argnums=(0,))
    def _resolve_combat(self, player: PlayerState, enemies: EnemyState, 
                       score: chex.Array, last_kill_time: chex.Array, 
                       consecutive_kills: chex.Array, time: chex.Array, level: chex.Array):
        """Resolve combat between player and enemies.
        
        Combat rules:
        - Player wins if facing enemy and enemy not facing back
        - If both facing: higher entity wins
        - Quick kill bonuses for consecutive rapid defeats
        
        Args:
            player: Current player state
            enemies: Current enemy states
            score: Current game score
            last_kill_time: Time of last enemy defeat
            consecutive_kills: Count of consecutive quick kills
            time: Current game time
            level: Current level for scoring
            
        Returns:
            Tuple of (updated_player, updated_enemies, new_score, 
                     new_last_kill_time, new_consecutive_kills)
        """
        # Collision boxes with inset for better feel
        INSET = 1.0  # Small inset to reduce grazing kills

        # Player bounds
        p_left = player.x + INSET
        p_right = player.x + self.consts.PLAYER_WIDTH - INSET
        p_top = player.y + INSET
        p_bottom = player.y + self.consts.PLAYER_HEIGHT - INSET

        # Enemy bounds (vectorized)
        e_left = enemies.positions[:, 0] + INSET
        e_right = enemies.positions[:, 0] + self.consts.ENEMY_WIDTH - INSET
        e_top = enemies.positions[:, 1] + INSET
        e_bottom = enemies.positions[:, 1] + self.consts.ENEMY_HEIGHT - INSET
        
        # Check collisions (with tie cooldown to prevent double-hits)
        x_overlap = jnp.logical_and(p_left < e_right, p_right > e_left)
        y_overlap = jnp.logical_and(p_top < e_bottom, p_bottom > e_top)
        collision_mask = jnp.logical_and(
            jnp.logical_and(x_overlap, y_overlap),
            jnp.logical_and(enemies.active, player.tie_cooldown == 0)  # Skip collision if in cooldown
        )
        
        # Calculate centers for collision detection first
        player_center_x = player.x + self.consts.PLAYER_WIDTH / 2
        enemy_center_x = enemies.positions[:, 0] + self.consts.ENEMY_WIDTH / 2
        player_center_y = player.y + self.consts.PLAYER_HEIGHT / 2
        enemy_center_y = enemies.positions[:, 1] + self.consts.ENEMY_HEIGHT / 2

        # Combat resolution based on facing directions
        # Player facing enemy check
        player_facing_enemy = jnp.where(
            player.facing_left,
            enemies.positions[:, 0] < player.x,
            enemies.positions[:, 0] > player.x
        )

        # Use actual enemy facing from state (not hardcoded)
        enemy_facing_player = jnp.where(
            enemies.facing_left,
            player_center_x < enemy_center_x,  # Enemy looks left
            player_center_x > enemy_center_x   # Enemy looks right
        )

        # Combat rules
        # both_facing is now checked inline with one_facing logic

        # Height comparison now done in outcome classification section

        # Check horizontal center alignment (tighter to reduce accidental vertical ties)
        x_center_diff = jnp.abs(player_center_x - enemy_center_x)
        is_center_aligned = x_center_diff <= 2.5  # Within 2.5 pixels for vertical hits

        # Check if player is clearly above or below
        is_above = player_center_y < enemy_center_y - 2.0
        is_below = player_center_y > enemy_center_y + 2.0

        # Check if player is moving mostly vertically (not diagonally) and has minimum speed
        VERT_MIN_SPEED = 0.08  # Same scalar as for bounce calculation
        is_vertical_approach = jnp.logical_and(
            jnp.abs(player.vel_y) >= jnp.abs(player.vel_x) * 1.2,  # Moderate threshold
            jnp.abs(player.vel_y) > VERT_MIN_SPEED  # Require minimum speed to count as vertical
        )

        # Vertical hit ONLY if: centered, above/below, AND moving vertically
        vertical_hit = jnp.logical_and(
            collision_mask,
            jnp.logical_and(
                jnp.logical_and(
                    is_center_aligned,  # Must hit near center (±2 pixels)
                    is_vertical_approach  # Must be moving vertically, not diagonally
                ),
                jnp.logical_or(is_above, is_below)
            )
        )

        # Define horizontal hit: similar height AND not coming from above/below
        # Check if player is moving mostly horizontally (not at steep angle)
        is_horizontal_approach = jnp.abs(player.vel_y) < jnp.abs(player.vel_x) * 0.5  # Mostly horizontal movement

        # ==== VERTICAL-PREFERRED OUTCOME CLASSIFICATION ====
        # Y grows downward; "player higher" means smaller Y.

        DEBUG = jnp.bool_(self.consts.DEBUG_EASY_VERTICAL_BOUNCE)

        TIE_TOLERANCE = jnp.where(DEBUG, 999.0, 2.0)       # ±2 px considered "similar height"
        ADVANTAGE_TOLERANCE = jnp.where(DEBUG, 999.0, 3.0) # ≥3 px advantage decides a win

        dy = enemy_center_y - player_center_y  # >0 => player higher
        player_clearly_higher = dy >= ADVANTAGE_TOLERANCE
        enemy_clearly_higher = dy <= -ADVANTAGE_TOLERANCE
        heights_similar = jnp.abs(dy) <= TIE_TOLERANCE

        pf = player_facing_enemy
        ef = enemy_facing_player
        one_facing = jnp.logical_xor(pf, ef)

        # If we are centered and moving mostly vertically, we PREFER a vertical tie
        vertical_preferred = jnp.logical_and(
            collision_mask,
            jnp.logical_and(is_center_aligned, is_vertical_approach)
        )

        # 0=none, 1=player_win, 2=enemy_win, 3=tie
        outcome = jnp.zeros_like(collision_mask, dtype=jnp.int32)

        # 1) Force vertical tie BEFORE any win logic
        outcome = jnp.where(vertical_preferred, 3, outcome)

        not_vert = jnp.logical_not(vertical_preferred)

        # 2) Height wins (only if not vertical-preferred)
        outcome = jnp.where(jnp.logical_and(not_vert, jnp.logical_and(collision_mask, player_clearly_higher)), 1, outcome)
        outcome = jnp.where(jnp.logical_and(not_vert, jnp.logical_and(collision_mask, enemy_clearly_higher)), 2, outcome)

        # 3) Similar height → facing decides (only if not vertical-preferred)
        similar_mask = jnp.logical_and(jnp.logical_and(collision_mask, heights_similar), not_vert)
        outcome = jnp.where(jnp.logical_and(similar_mask, one_facing & pf), 1, outcome)
        outcome = jnp.where(jnp.logical_and(similar_mask, one_facing & ef), 2, outcome)

        # 4) Anything still 0 but colliding becomes a tie
        outcome = jnp.where(jnp.logical_and(collision_mask, outcome == 0), 3, outcome)

        # Horizontal hit: similar height AND moving horizontally
        horizontal_hit = jnp.logical_and(
            collision_mask,
            jnp.logical_and(
                heights_similar,
                is_horizontal_approach
            )
        )

        # Build masks from outcome encoding
        player_wins = outcome == 1
        enemy_wins = outcome == 2
        tie_condition = outcome == 3

        # --- GLOBAL EXCLUSIVITY (frame-wide) ---
        # Priority: enemy_wins > player_wins > tie
        # This prevents "both die" and "die + bounce" bugs

        any_enemy_wins = jnp.any(enemy_wins)

        # If any enemy wins, suppress ALL player wins and ALL ties
        player_wins = jnp.where(any_enemy_wins, jnp.zeros_like(player_wins), player_wins)
        tie_condition = jnp.where(any_enemy_wins, jnp.zeros_like(tie_condition), tie_condition)

        # If no enemy wins but some player wins, suppress ties this frame
        any_player_wins_after = jnp.any(player_wins)
        tie_condition = jnp.where(any_player_wins_after, jnp.zeros_like(tie_condition), tie_condition)

        # Re-evaluate counts after masking
        num_kills = jnp.sum(player_wins)
        player_hit = jnp.any(enemy_wins)
        
        # Classify tie types strictly (no fake "horizontal" from non-vertical)
        vertical_tie_per_enemy = jnp.logical_and(tie_condition, vertical_hit)
        horizontal_tie_per_enemy = jnp.logical_and(tie_condition, horizontal_hit)
        # If neither vertical nor horizontal, treat as diagonal/other
        other_tie_per_enemy = jnp.logical_and(
            tie_condition,
            jnp.logical_not(jnp.logical_or(vertical_hit, horizontal_hit))
        )

        # Check if any ties exist by type
        any_vertical_tie = jnp.any(vertical_tie_per_enemy)
        any_horizontal_tie = jnp.any(horizontal_tie_per_enemy)
        any_tie = jnp.any(tie_condition)

        # Calculate bounce direction based on enemy positions (consistent with other bounces)
        # Find the first enemy that causes a tie
        tie_enemy_x = jnp.where(
            tie_condition,
            enemies.positions[:, 0],
            jnp.inf
        ).min()

        tie_enemy_y = jnp.where(
            tie_condition,
            enemies.positions[:, 1],
            jnp.inf
        ).min()

        # Only apply bounce if it's a tie AND player doesn't die
        should_bounce = jnp.logical_and(any_tie, jnp.logical_not(player_hit))

        # Gentle vertical bounce parameters
        VERT_MIN_SPEED = 0.08            # ignore micro-motions
        COEFF_RESTITUTION = 0.25         # how "bouncy" vertical ties feel
        MIN_V_BOUNCE = 0.12              # tiny bottom to avoid 0
        MAX_V_BOUNCE = self.consts.MAX_VEL_Y_DOWN  # never exceed global cap

        # Calculate vertical bounce magnitude based on speed
        speed_y = jnp.abs(player.vel_y)
        raw_bounce = jnp.where(speed_y > VERT_MIN_SPEED, COEFF_RESTITUTION * speed_y, 0.0)
        bounce_mag_y = jnp.clip(raw_bounce + MIN_V_BOUNCE, 0.0, MAX_V_BOUNCE)

        # Horizontal bounce parameters
        MIN_H_BOUNCE = 0.8  # Minimum horizontal bounce velocity
        MAX_BOUNCE_VEL = 1.5  # Maximum bounce velocity
        BOUNCE_REDUCTION = 0.5  # Only use 50% of player velocity for bounce

        # Player horizontal bounce - ensure minimum separation
        bounce_mag_x = jnp.maximum(MIN_H_BOUNCE, jnp.minimum(jnp.abs(player.vel_x) * BOUNCE_REDUCTION, MAX_BOUNCE_VEL))
        bounce_vel_x = jnp.where(
            should_bounce,
            jnp.where(
                player.x < tie_enemy_x,
                -bounce_mag_x,  # Bounce left with minimum velocity
                bounce_mag_x     # Bounce right with minimum velocity
            ),
            player.vel_x
        )

        # Player vertical bounce - more controlled now
        bounce_vel_y = jnp.where(
            jnp.logical_and(should_bounce, any_horizontal_tie),
            jnp.maximum(0.5, jnp.abs(player.vel_y) * 0.3),  # Gentler downward bounce for horizontal
            jnp.where(
                jnp.logical_and(should_bounce, any_vertical_tie),
                jnp.where(
                    player.y < tie_enemy_y,  # Player above enemy
                    -bounce_mag_y,  # Bounce up with calculated magnitude
                    bounce_mag_y    # Bounce down with calculated magnitude
                ),
                player.vel_y
            )
        )

        # Replace fixed 3px shove with Minimum Translation Vector (MTV)
        # Find an index for a true tie (falls back to 0 if none, but is gated by any_tie)
        tie_idx = jnp.argmin(jnp.where(tie_condition, enemies.positions[:, 0], jnp.inf))

        # Centers for that enemy
        ex = enemy_center_x[tie_idx]
        ey = enemy_center_y[tie_idx]
        px = player_center_x
        py = player_center_y

        # Half sizes
        phx = self.consts.PLAYER_WIDTH * 0.5
        phy = self.consts.PLAYER_HEIGHT * 0.5
        ehx = self.consts.ENEMY_WIDTH * 0.5
        ehy = self.consts.ENEMY_HEIGHT * 0.5

        # Penetration along each axis
        dx = px - ex
        dy = py - ey
        pen_x = (phx + ehx) - jnp.abs(dx)
        pen_y = (phy + ehy) - jnp.abs(dy)

        # Resolve along the shallowest penetration axis only
        resolve_x = pen_x < pen_y
        eps = 0.5  # tiny bias to ensure separation

        sep_x = jnp.where(resolve_x, jnp.sign(dx) * (pen_x + eps), 0.0)
        sep_y = jnp.where(resolve_x, 0.0, jnp.sign(dy) * (pen_y + eps))

        # Apply only on ties (and only if not dying)
        new_player_x = jnp.where(
            any_tie,
            jnp.clip(player.x + sep_x, 0, self.consts.SCREEN_WIDTH - self.consts.PLAYER_WIDTH),
            player.x
        )
        new_player_y = jnp.where(
            any_tie,
            jnp.clip(player.y + sep_y, self.consts.PLAY_AREA_TOP, self.consts.PLAY_AREA_BOTTOM - self.consts.PLAYER_HEIGHT),
            player.y
        )

        # Set tie cooldown to prevent immediate re-collision
        TIE_COOLDOWN_FRAMES = 10  # Brief invulnerability after ties
        new_tie_cooldown = jnp.where(
            any_tie,
            TIE_COOLDOWN_FRAMES,
            jnp.maximum(0, player.tie_cooldown - 1)
        )

        # Apply bounce and separation to player ONLY if not dying
        player = jax.lax.cond(
            should_bounce,
            lambda p: p._replace(
                x=new_player_x,
                y=new_player_y,
                vel_x=bounce_vel_x,
                vel_y=bounce_vel_y,
                tie_cooldown=new_tie_cooldown
            ),
            lambda p: p._replace(tie_cooldown=jnp.maximum(0, p.tie_cooldown - 1)),  # Decrement cooldown even if no bounce
            player
        )

        # CRITICAL: NO enemy bounces if player dies from ANY collision
        # Check if player dies FIRST
        player_dies = jnp.any(enemy_wins)

        # Only allow tie effects if player survives ALL collisions
        true_tie_per_enemy = jnp.where(
            player_dies,
            jnp.zeros_like(tie_condition),  # No ties if player dies
            tie_condition  # Otherwise use actual tie status
        )

        # Apply bounce to enemies ONLY on true ties
        # --- Impact scaling and distances (player gentle, enemy 5..20) ---
        player_speed = jnp.sqrt(player.vel_x**2 + player.vel_y**2)

        # Scale by the max attainable player speed (keeps 0..1 nicely)
        max_player_speed = jnp.sqrt(self.consts.MAX_VEL_X**2 + self.consts.MAX_VEL_Y_DOWN**2)
        impact_factor = jnp.clip(player_speed / jnp.maximum(1e-6, max_player_speed), 0.0, 1.0)

        # Use the strict tie classifications (already accounts for player survival)
        is_vertical_tie = jnp.logical_and(true_tie_per_enemy, vertical_tie_per_enemy)
        is_horizontal_tie = jnp.logical_and(true_tie_per_enemy, horizontal_tie_per_enemy)
        is_other_tie = jnp.logical_and(true_tie_per_enemy, other_tie_per_enemy)

        # Horizontal hits REVERSE direction (flip sign) and modify speed
        # Vertical hits keep both direction AND speed unchanged
        speed_modifier = 0.8 + impact_factor * 0.4  # 80-120% speed based on impact

        # Apply bounce only to enemies that are in ties (per-enemy)
        enemy_bounce_vel_x = jnp.where(
            is_horizontal_tie,
            -enemies.vel_x * speed_modifier,  # REVERSE direction by flipping sign
            enemies.vel_x  # Keep current velocity for non-tie collisions or dead enemies
        )

        # For vertical hits, set target position (5-20 pixels in debug, 6-18 normally)
        # Movement happens over 0.4 seconds (24 frames at 60fps)
        MIN_VERTICAL_BOUNCE = jnp.where(DEBUG, 5.0, 6.0)   # enemy bounce distance min (pixels)
        MAX_VERTICAL_BOUNCE = jnp.where(DEBUG, 20.0, 18.0) # enemy bounce distance max (pixels)

        vertical_bounce_distance = MIN_VERTICAL_BOUNCE + impact_factor * (MAX_VERTICAL_BOUNCE - MIN_VERTICAL_BOUNCE)
        BOUNCE_DURATION = 24  # keep existing animation duration

        # Compute level-specific Y bounds (same ones used in _update_beasts)
        min_y = self.consts.PLAY_AREA_TOP
        max_y = jnp.where(
            jnp.logical_or(level == 5, level == 7),
            self.consts.ENEMY_MIN_Y,  # ground safety for 5 & 7
            self.consts.PLAY_AREA_BOTTOM - self.consts.ENEMY_HEIGHT
        )

        # Set vertical bounce targets only for true vertical ties
        enemy_bounce_target_y = jnp.where(
            is_vertical_tie,
            enemies.positions[:, 1] + jnp.where(
                enemies.positions[:, 1] < player.y,
                -vertical_bounce_distance,  # up
                vertical_bounce_distance    # down
            ),
            enemies.bounce_target_y
        )

        # Clamp target to allowed area so we never "bounce out"
        enemy_bounce_target_y = jnp.clip(enemy_bounce_target_y, min_y, max_y)

        # Start bounce timer ONLY for true vertical ties
        enemy_bounce_timer = jnp.where(
            is_vertical_tie,
            BOUNCE_DURATION,  # Start 24-frame transition
            enemies.bounce_timer  # Keep existing timer
        )

        # For horizontal ties, add tiny vertical nudge (reduced from ±0.1 to ±0.03)
        enemy_bounce_vel_y = jnp.where(
            is_horizontal_tie,
            enemies.vel_y + jnp.where(
                enemies.positions[:, 1] < player.y,
                -0.03,  # Tiny upward nudge
                0.03    # Tiny downward nudge
            ),
            enemies.vel_y  # Keep current velocity for non-tie collisions
        )

        # For diagonal/other ties, nudge apart gently
        enemy_bounce_vel_y = jnp.where(
            is_other_tie,
            enemies.vel_y + jnp.sign(enemies.positions[:, 1] - player.y) * 0.05,
            enemy_bounce_vel_y
        )

        # Update enemies - only apply bounce to those actually in ties
        new_enemy_active = jnp.logical_and(enemies.active, jnp.logical_not(player_wins))

        # Only update velocities for true ties (already checked player doesn't die)
        final_vel_x = jnp.where(
            true_tie_per_enemy,
            enemy_bounce_vel_x,  # Apply bounce
            enemies.vel_x        # Keep original velocity for dead/non-tie enemies
        )

        final_vel_y = jnp.where(
            true_tie_per_enemy,
            enemy_bounce_vel_y,  # Apply bounce
            enemies.vel_y        # Keep original velocity
        )

        # Only apply bounce animation for vertical ties (already checked player survives)
        final_bounce_target_y = jnp.where(
            is_vertical_tie,  # is_vertical_tie already uses true_tie_per_enemy
            enemy_bounce_target_y,  # Apply new target
            enemies.bounce_target_y  # Keep existing
        )

        final_bounce_timer = jnp.where(
            is_vertical_tie,  # is_vertical_tie already uses true_tie_per_enemy
            enemy_bounce_timer,     # Start timer
            enemies.bounce_timer    # Keep existing
        )

        new_enemies = enemies._replace(
            active=new_enemy_active,
            vel_x=final_vel_x,
            vel_y=final_vel_y,
            bounce_target_y=final_bounce_target_y,
            bounce_timer=final_bounce_timer
        )
        
        # Update score with quick kill bonuses
        time_since_last_kill = time - last_kill_time
        
        # Check which quick kill bonus applies
        is_quick_kill_1 = time_since_last_kill <= self.consts.QUICK_KILL_TIME_LIMIT_1
        
        # Update consecutive kills
        new_consecutive_kills = jnp.where(
            jnp.logical_and(num_kills > 0, is_quick_kill_1),
            consecutive_kills + num_kills,
            jnp.where(num_kills > 0, num_kills, consecutive_kills)
        )
        
        # Calculate bonus based on level and consecutive kills
        # Get quick kill bonuses for this level
        level_bonuses = jax.lax.switch(
            (level - 1) // 2,  # 1->0, 3->1, 5->2, 7->3
            [
                lambda: (self.consts.QUICK_KILL_BONUS_1_L1, self.consts.QUICK_KILL_BONUS_2_L1, self.consts.QUICK_KILL_BONUS_3_L1),
                lambda: (self.consts.QUICK_KILL_BONUS_1_L3, self.consts.QUICK_KILL_BONUS_2_L3, self.consts.QUICK_KILL_BONUS_3_L3),
                lambda: (self.consts.QUICK_KILL_BONUS_1_L5, self.consts.QUICK_KILL_BONUS_2_L5, self.consts.QUICK_KILL_BONUS_3_L5),
                lambda: (self.consts.QUICK_KILL_BONUS_1_L7, self.consts.QUICK_KILL_BONUS_2_L7, self.consts.QUICK_KILL_BONUS_3_L7)
            ]
        )
        bonus_1, bonus_2, bonus_3 = level_bonuses
        
        bonus_multiplier = jnp.where(
            new_consecutive_kills >= 3,
            bonus_3,
            jnp.where(
                new_consecutive_kills >= 2,
                bonus_2,
                jnp.where(
                    new_consecutive_kills >= 1,
                    bonus_1,
                    0
                )
            )
        )
        
        # Calculate score based on level
        points_per_kill = jax.lax.switch(
            (level - 1) // 2,  # 1->0, 3->1, 5->2, 7->3
            [lambda: self.consts.POINTS_LEVEL_1_BEAST, lambda: self.consts.POINTS_LEVEL_3_BEAST, 
             lambda: self.consts.POINTS_LEVEL_5_BEAST, lambda: self.consts.POINTS_LEVEL_7_BEAST]
        )
        # Don't award scores when dying (extra safety check)
        base_score = jnp.where(player_hit, 0, num_kills * points_per_kill)
        quick_bonus = jnp.where(
            jnp.logical_and(
                jnp.logical_and(num_kills > 0, is_quick_kill_1),
                jnp.logical_not(player_hit)
            ),
            bonus_multiplier,
            0
        )
        new_score = score + base_score + quick_bonus
        
        # Update last kill time
        new_last_kill_time = jnp.where(num_kills > 0, time, last_kill_time)
        
        # Update player (start death animation if hit)
        new_player = jax.lax.cond(
            player_hit,
            lambda p: p._replace(
                state=self.consts.PLAYER_STATE_DEATH,
                state_timer=0,
                death_timer=1,
                vel_x=0.0,
                vel_y=0.0
            ),
            lambda p: p,
            player
        )
        
        return new_player, new_enemies, new_score, new_last_kill_time, new_consecutive_kills
    
    @partial(jax.jit, static_argnums=(0,))
    def _handle_player_death(self, player: PlayerState, lives: chex.Array, stage: chex.Array):
        """Handle player death and respawn logic.
        
        Manages death animation, life loss, and respawn.
        Player can continue playing with 0 lives but cannot respawn.
        
        Args:
            player: Current player state
            lives: Current life count
            stage: Current stage for respawn position
            
        Returns:
            Tuple of (updated_player, new_lives, game_over)
        """
        # Check if player death animation is complete
        death_complete = jnp.logical_and(
            player.state == self.consts.PLAYER_STATE_DEATH,
            player.death_timer >= self.consts.DEATH_ANIMATION_FRAMES
        )
        
        # Lose a life and reset player if death is complete
        new_lives = jnp.where(death_complete, lives - 1, lives)
        # Game over only when lives are negative (allows playing with 0 lives)
        new_game_over = new_lives < 0
        
        # Reset player to starting position if death complete and still have lives
        # Use stage-specific spawn positions
        spawn_x = jnp.where(
            stage == 2,
            self.consts.PLAYER_START_X_CASTLE,
            self.consts.PLAYER_START_X
        ).astype(jnp.float32)
        spawn_y = jnp.where(
            stage == 2,
            self.consts.PLAYER_START_Y_CASTLE,
            self.consts.PLAYER_START_Y
        ).astype(jnp.float32)
        
        new_player = jax.lax.cond(
            jnp.logical_and(death_complete, new_lives >= 0),
            lambda p: PlayerState(
                x=spawn_x,
                y=spawn_y,
                vel_x=jnp.array(0.0, dtype=jnp.float32),
                vel_y=jnp.array(0.0, dtype=jnp.float32),
                facing_left=jnp.array(True),
                state=jnp.array(self.consts.PLAYER_STATE_IDLE, dtype=jnp.int32),
                state_timer=jnp.array(0, dtype=jnp.int32),
                is_flapping=jnp.array(False),
                fire_was_pressed=jnp.array(False),
                flap_cooldown=jnp.array(0, dtype=jnp.int32),
                flap_anim_timer=jnp.array(0, dtype=jnp.int32),
                death_timer=jnp.array(0, dtype=jnp.int32),
                tie_cooldown=jnp.array(0, dtype=jnp.int32)
            ),
            lambda p: p,
            player
        )
        
        return new_player, new_lives, new_game_over
    
    def _handle_death_complete(self, player: PlayerState, lives: chex.Array):
        """Check if player can respawn after death.
        
        Args:
            player: Current player state
            lives: Current life count
            
        Returns:
            Tuple of (respawned_player, game_over)
        """
        # Check if can respawn
        can_respawn = lives > 0
        
        # Reset player position if respawning
        new_player = jax.lax.cond(
            can_respawn,
            lambda p: PlayerState(
                x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.float32),
                y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.float32),
                vel_x=jnp.array(0.0, dtype=jnp.float32),
                vel_y=jnp.array(0.0, dtype=jnp.float32),
                facing_left=jnp.array(True),  # Respawn facing left
                state=p.state,  # Keep state
                state_timer=p.state_timer,  # Keep timer
                is_flapping=jnp.array(False),
                fire_was_pressed=p.fire_was_pressed,  # Keep fire state
                flap_cooldown=jnp.array(0, dtype=jnp.int32),
                flap_anim_timer=p.flap_anim_timer,  # Keep anim timer
                death_timer=jnp.array(0, dtype=jnp.int32),
                tie_cooldown=jnp.array(0, dtype=jnp.int32)
            ),
            lambda p: p,
            player
        )
        
        # Game over if no lives left
        game_over = jnp.logical_not(can_respawn)
        
        return new_player, game_over
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SirLancelotState) -> SirLancelotObservation:
        # Player observation
        player_obs = EntityPosition(
            x=state.player.x,
            y=state.player.y,
            width=jnp.array(self.consts.PLAYER_WIDTH, dtype=jnp.float32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, dtype=jnp.float32)
        )
        
        # Enemy observations
        enemy_widths = jnp.full(self.consts.NUM_ENEMIES, self.consts.ENEMY_WIDTH, dtype=jnp.float32)
        enemy_heights = jnp.full(self.consts.NUM_ENEMIES, self.consts.ENEMY_HEIGHT, dtype=jnp.float32)
        enemy_active = state.enemies.active.astype(jnp.float32)
        
        enemy_obs = jnp.stack([
            state.enemies.positions[:, 0],
            state.enemies.positions[:, 1],
            enemy_widths,
            enemy_heights,
            enemy_active
        ], axis=-1)
        
        # Dragon observation
        dragon_obs = jnp.array([
            state.dragon.x,
            self.consts.DRAGON_Y,
            self.consts.DRAGON_WIDTH,
            self.consts.DRAGON_HEIGHT,
            state.dragon.is_active.astype(jnp.float32)
        ])
        
        # Fireball observations
        fireball_widths = jnp.full(self.consts.MAX_FIREBALLS, self.consts.FIREBALL_WIDTH, dtype=jnp.float32)
        fireball_heights = jnp.full(self.consts.MAX_FIREBALLS, self.consts.FIREBALL_HEIGHT, dtype=jnp.float32)
        fireball_active = state.fireballs.active.astype(jnp.float32)
        
        fireball_obs = jnp.stack([
            state.fireballs.positions[:, 0],
            state.fireballs.positions[:, 1],
            fireball_widths,
            fireball_heights,
            fireball_active
        ], axis=-1)
        
        return SirLancelotObservation(
            player=player_obs,
            enemies=enemy_obs,
            dragon=dragon_obs,
            fireballs=fireball_obs,
            score=state.score.astype(jnp.float32),
            lives=state.lives.astype(jnp.float32),
            stage=state.stage.astype(jnp.float32)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SirLancelotState) -> SirLancelotInfo:
        return SirLancelotInfo(
            time=state.time,
            level_complete=state.stage_complete,
            consecutive_kills=state.consecutive_quick_kills,
        )
    
    def _get_done(self, state: SirLancelotState) -> bool:
        return jnp.logical_or(state.game_over, state.stage_complete)
    
    def _get_reward(self, previous_state: SirLancelotState, current_state: SirLancelotState) -> float:
        return (current_state.score - previous_state.score).astype(jnp.float32)

    def render(self, state: SirLancelotState):
        """Render the current game state.
        
        Creates visual representation with background, sprites, and HUD.
        
        Args:
            state: Current game state
            
        Returns:
            RGB image array in HWC format
        """
        # Delegate to renderer
        if not hasattr(self, 'renderer'):
            self.renderer = SirLancelotRenderer(self.consts)
        return self.renderer.render(state)
    
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: SirLancelotObservation) -> jnp.ndarray:
        """Convert structured observation to flat 1D array for ML models.

        Flattens all observation components into a single array:
        - Player position (x, y, width, height): 4 values
        - Enemies array (4 enemies × 5 values each): 20 values
        - Dragon (x, y, width, height, active): 5 values
        - Fireballs (3 fireballs × 5 values each): 15 values
        - Score: 1 value
        - Lives: 1 value
        - Stage: 1 value
        Total: 47 values
        """
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.enemies.flatten(),  # Already shape (NUM_ENEMIES, 5), flattens to (20,)
            obs.dragon.flatten(),  # Shape (5,), flattens to (5,)
            obs.fireballs.flatten(),  # Shape (MAX_FIREBALLS, 5), flattens to (15,)
            obs.score.flatten(),
            obs.lives.flatten(),
            obs.stage.flatten()
        ])
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)  # Full Atari action space
    
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.float32),
                "y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.float32),
                "width": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.float32),
                "height": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.float32),
            }),
            "enemies": spaces.Box(
                low=0,
                high=255,
                shape=(self.consts.NUM_ENEMIES, 5),
                dtype=jnp.float32
            ),
            "dragon": spaces.Box(
                low=0,
                high=255,
                shape=(5,),  # x, y, width, height, active
                dtype=jnp.float32
            ),
            "fireballs": spaces.Box(
                low=0,
                high=255,
                shape=(self.consts.MAX_FIREBALLS, 5),  # 3 fireballs × 5 values each
                dtype=jnp.float32
            ),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.float32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.float32),
            "stage": spaces.Box(low=1, high=2, shape=(), dtype=jnp.float32),  # 1=aerial, 2=dragon
        })
    
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0, 
            high=255, 
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3), 
            dtype=jnp.uint8
        )
    
    def flat_observation_space(self) -> spaces.Box:
        """Return the observation space for flattened observations.

        The flat array contains:
        - Player (x, y, width, height): 4 values
        - Enemies (4 enemies × 5 values each): 20 values
        - Dragon (x, y, width, height, active): 5 values
        - Fireballs (3 fireballs × 5 values each): 15 values
        - Score: 1 value
        - Lives: 1 value
        - Stage: 1 value
        Total: 47 values
        """
        return spaces.Box(
            low=0.0,
            high=jnp.inf,  # Score can be arbitrarily high
            shape=(47,),
            dtype=jnp.float32
        )


# -----------------------------------------------------------------------------
# RENDERER
# -----------------------------------------------------------------------------

class SirLancelotRenderer(JAXGameRenderer):
    def __init__(self, consts: SirLancelotConstants = None):
        self.consts = consts or SirLancelotConstants()
        super().__init__(self.consts)
        
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites", "sir_lancelot")
        
        preprocessed_assets = self._load_and_preprocess_assets(sprite_path)
        asset_config = self._get_asset_config(preprocessed_assets)
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        
        (
            self.PRE_RENDERED_OUTDOOR,
            self.PRE_RENDERED_CASTLE,
            self.PRE_RENDERED_WAVE_RGB
        ) = self._precompute_static_frames()
        
        self._cache_sprite_stacks()

    def _load_and_preprocess_assets(self, sprite_path: str) -> dict:
        target_shape = (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 4)
        
        bg_raw = jnp.load(os.path.join(sprite_path, "Background_lvl1.npy"))
        bg_resized = resize(bg_raw, target_shape, method='nearest').astype(jnp.uint8)
        
        bg2_raw = jnp.load(os.path.join(sprite_path, "Background_lvl2.npy"))
        bg2_resized = resize(bg2_raw, target_shape, method='nearest').astype(jnp.uint8)
        
        return {
            'bg_outdoor': bg_resized,
            'bg_castle': bg2_resized,
        }

    def _get_asset_config(self, preprocessed_assets: dict) -> list:
        return [
            {'name': 'bg_outdoor', 'type': 'background', 'data': preprocessed_assets['bg_outdoor']},
            {'name': 'bg_castle', 'type': 'single', 'data': preprocessed_assets['bg_castle']},
            
            {'name': 'player', 'type': 'group', 'files': ['SirLancelot_lvl1_neutral.npy', 'SirLancelot_lvl1_fly.npy']},
            
            {'name': 'beast_1', 'type': 'group', 'files': ['Beast_1_animation_1.npy', 'Beast_1_animation_2.npy']},
            {'name': 'beast_2', 'type': 'group', 'files': ['Beast_2_animation_1.npy', 'Beast_2_animation_2.npy']},
            {'name': 'beast_3', 'type': 'group', 'files': ['Beast_3_animation_1.npy', 'Beast_3_animation_2.npy']},
            {'name': 'beast_4', 'type': 'group', 'files': ['Beast_4_animation_1.npy', 'Beast_4_animation_1.npy']},
            
            {'name': 'dragon', 'type': 'group', 'files': [
                'Dragon_lvl2_Wing_Up.npy', 'Dragon_lvl2_wing_middle.npy',
                'Dragon_lvl2_wing_down.npy', 'Dragon_lvl2_wing_middle.npy'
            ]},
            {'name': 'dragon_fire', 'type': 'group', 'files': [
                'Dragon_lvl2_wing_up_fire.npy', 'Dragon_lvl2_wing_middle_fire.npy',
                'Dragon_lvl2_wing_down_fire.npy', 'Dragon_lvl2_wing_middle_fire.npy'
            ]},
            
            {'name': 'fireball', 'type': 'group', 'files': ['Dragon_lvl2_fire_animation_1.npy', 'Dragon_lvl2_fire_animation_2.npy']},
            
            {'name': 'digits', 'type': 'digits', 'pattern': 'number_{}.npy'},
            {'name': 'life', 'type': 'single', 'file': 'Life.npy'},
        ]

    def _precompute_static_frames(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raster_outdoor = self.BACKGROUND
        pre_rendered_outdoor = raster_outdoor
        
        raster_castle = self.SHAPE_MASKS['bg_castle']
        pre_rendered_castle = raster_castle
        
        wave_raster_base = jnp.full_like(self.BACKGROUND, self.BACKGROUND[0,0])
        pre_rendered_wave_rgb = self.jr.render_from_palette(wave_raster_base, self.PALETTE)
        
        return pre_rendered_outdoor, pre_rendered_castle, pre_rendered_wave_rgb

    def _cache_sprite_stacks(self):
        self.PLAYER_STACK = self.SHAPE_MASKS['player']
        self.PLAYER_OFFSETS = self.FLIP_OFFSETS['player']
        
        beast_masks = [
            self.SHAPE_MASKS['beast_1'],
            self.SHAPE_MASKS['beast_2'],
            self.SHAPE_MASKS['beast_3'],
            self.SHAPE_MASKS['beast_4']
        ]
        
        beast_offsets = [
            self.FLIP_OFFSETS['beast_1'],
            self.FLIP_OFFSETS['beast_2'],
            self.FLIP_OFFSETS['beast_3'],
            self.FLIP_OFFSETS['beast_4']
        ]
        
        max_h = max(m.shape[1] for m in beast_masks)
        max_w = max(m.shape[2] for m in beast_masks)
        
        padded_beast_masks = []
        for mask in beast_masks:
            h, w = mask.shape[1], mask.shape[2]
            padded_mask = jnp.pad(
                mask,
                ((0, 0), (0, max_h - h), (0, max_w - w)),
                mode="constant",
                constant_values=self.jr.TRANSPARENT_ID
            )
            padded_beast_masks.append(padded_mask)
        
        self.BEAST_STACKS = jnp.stack(padded_beast_masks)
        self.BEAST_OFFSETS = jnp.stack(beast_offsets)
        
        self.DRAGON_STACK = self.SHAPE_MASKS['dragon']
        self.DRAGON_OFFSETS_STACK = self.FLIP_OFFSETS['dragon']
        self.DRAGON_FIRE_STACK = self.SHAPE_MASKS['dragon_fire']
        self.DRAGON_FIRE_OFFSETS_STACK = self.FLIP_OFFSETS['dragon_fire']
        
        self.FIREBALL_STACK = self.SHAPE_MASKS['fireball']
        self.FIREBALL_OFFSETS_STACK = self.FLIP_OFFSETS['fireball']
        
        self.DRAGON_Y_ADJUSTMENTS = jnp.array([4, 0, 0, 0])
        
        self.BLACK_ID = self.COLOR_TO_ID.get((0, 0, 0), self.BACKGROUND[0,0])

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemies(self, state: SirLancelotState, raster):
        def render_enemy(i, raster):
            active = state.enemies.active[i]
            visible = active & ~state.enemies.is_invisible[i]
            x = state.enemies.positions[i, 0].astype(jnp.int32)
            y = state.enemies.positions[i, 1].astype(jnp.int32)
            frame_idx = state.enemies.animation_frame[i]
            
            level_idx = (state.level - 1) // 2
            beast_sprites = self.BEAST_STACKS[level_idx]
            beast_offset = self.BEAST_OFFSETS[level_idx]
            
            anim_frame = frame_idx % 2
            enemy_frame = beast_sprites[anim_frame]
            
            is_level_3 = state.level == 3
            flip = jax.lax.select(
                is_level_3,
                state.enemies.facing_left[i],
                ~state.enemies.facing_left[i]
            )
            
            return jax.lax.cond(
                visible,
                lambda r: self.jr.render_at_clipped(r, x, y, enemy_frame, flip_horizontal=flip, flip_offset=beast_offset),
                lambda r: r,
                raster
            )
        
        return jax.lax.fori_loop(0, self.consts.NUM_ENEMIES, render_enemy, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_player(self, state: SirLancelotState, raster):
        player_visible = (state.player.death_timer == 0) | (state.player.death_timer < self.consts.DEATH_ANIMATION_FRAMES)
        
        def draw_player(r):
            sprite_idx = jnp.where(state.player.is_flapping, 1, 0)
            player_frame = self.PLAYER_STACK[sprite_idx]
            flip = ~state.player.facing_left
            
            return self.jr.render_at(
                r,
                state.player.x.astype(jnp.int32),
                state.player.y.astype(jnp.int32),
                player_frame,
                flip_horizontal=flip,
                flip_offset=self.PLAYER_OFFSETS
            )
        
        return jax.lax.cond(player_visible, draw_player, lambda r: r, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_castle_elements(self, state: SirLancelotState, raster):
        def render_dragon(r):
            is_shooting = (state.dragon.shoot_cooldown > (self.consts.FIREBALL_SHOOT_COOLDOWN - 20)) & \
                          jnp.any(state.fireballs.active)
            
            dragon_sprites = jax.lax.select(is_shooting, self.DRAGON_FIRE_STACK, self.DRAGON_STACK)
            dragon_offset = jax.lax.select(is_shooting, self.DRAGON_FIRE_OFFSETS_STACK, self.DRAGON_OFFSETS_STACK)
            
            frame_idx = state.dragon.wing_frame
            dragon_frame = dragon_sprites[frame_idx]
            flip = state.dragon.facing_left
            
            y_adjustment = self.DRAGON_Y_ADJUSTMENTS[frame_idx]
            adjusted_y = self.consts.DRAGON_Y - y_adjustment
            
            return self.jr.render_at_clipped(
                r,
                state.dragon.x.astype(jnp.int32),
                adjusted_y.astype(jnp.int32),
                dragon_frame,
                flip_horizontal=flip,
                flip_offset=dragon_offset
            )
        
        raster_with_dragon = jax.lax.cond(state.dragon.is_active, render_dragon, lambda r: r, raster)
        
        def render_fireball(i, r):
            x = state.fireballs.positions[i, 0].astype(jnp.int32)
            y = state.fireballs.positions[i, 1].astype(jnp.int32)
            frame_idx = state.fireballs.animation_frame[i]
            
            fireball_frame = self.FIREBALL_STACK[frame_idx]
            
            return jax.lax.cond(
                state.fireballs.active[i],
                lambda r: self.jr.render_at_clipped(r, x, y, fireball_frame, flip_offset=self.FIREBALL_OFFSETS_STACK),
                lambda r: r,
                r
            )
        
        return jax.lax.fori_loop(0, self.consts.MAX_FIREBALLS, render_fireball, raster_with_dragon)

    @partial(jax.jit, static_argnums=(0,))
    def _render_hud(self, state: SirLancelotState, raster):
        raster = self.jr.render_bar(
            raster, 0, 0, 
            1.0, 1.0,
            self.consts.SCREEN_WIDTH, self.consts.TOP_BLACK_BAR_HEIGHT, 
            self.BLACK_ID, self.BLACK_ID
        )
    
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        
        def render_digit(i, r):
            digit_x = self.consts.SCORE_X + i * (6 + 2)
            digit_frame = self.SHAPE_MASKS['digits'][score_digits[i]]
            return self.jr.render_at(r, digit_x, self.consts.SCORE_Y, digit_frame)
        
        raster = jax.lax.fori_loop(0, 6, render_digit, raster)
        
        life_frame = self.SHAPE_MASKS['life']
        num_lives = jnp.minimum(state.lives, self.consts.MAX_LIVES)
        
        raster = self.jr.render_indicator(
            raster, 
            self.consts.LIVES_X, self.consts.LIVES_Y,
            num_lives,
            life_frame,
            spacing=(7 + 1),
            max_value=self.consts.MAX_LIVES
        )
        
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_game(self, state: SirLancelotState):
        use_castle = state.level % 2 == 0
        raster = jax.lax.cond(
            use_castle,
            lambda: self.PRE_RENDERED_CASTLE,
            lambda: self.PRE_RENDERED_OUTDOOR
        )
        
        raster = self._render_enemies(state, raster)
        raster = self._render_player(state, raster)
        
        raster = jax.lax.cond(
            state.stage == 2,
            lambda r: self._render_castle_elements(state, r),
            lambda r: r,
            raster
        )
        
        raster = self._render_hud(state, raster)
        
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SirLancelotState):
        return self._render_game(state)
