import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.modification import AutoDerivedConstants

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Procedural assets
    procedural_bg = jnp.zeros((250, 160, 4), dtype=jnp.uint8).at[:,:,3].set(255) # Opaque Black
    
    # Death flash colors (constant array from KingKongConstants)
    DEATH_FLASH_COLORS = jnp.array([
        [82, 82, 82], [151, 151, 151], [210, 210, 210], [0, 0, 0],
        [82, 82, 82], [151, 151, 151], [210, 210, 210], [128, 88, 0],
        [171, 135, 50], [207, 175, 92], [238, 209, 128], [68, 92, 0],
        [118, 147, 50], [160, 194, 92], [196, 234, 128], [112, 52, 0],
        [160, 107, 50], [201, 154, 92], [236, 194, 128], [0, 100, 20],
        [50, 152, 82], [92, 197, 135], [128, 235, 180], [112, 0, 20]
    ], dtype=jnp.uint8)
    
    # Add flash colors and UI colors to palette
    procedural_colors = jnp.array(
        [[201, 92, 135, 255]] + [list(c) + [255] for c in DEATH_FLASH_COLORS], 
        dtype=jnp.uint8
    ).reshape(-1, 1, 1, 4)
    debug_colors = jnp.array([
        [255, 0, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
        [0, 255, 0, 255],
    ], dtype=jnp.uint8).reshape(-1, 1, 1, 4)
    
    return {
        'background': procedural_bg,
        'ui_colors': procedural_colors,
        'debug_colors': debug_colors,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for KingKong.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    
    # Define sprite groups (for auto-padding)
    player_keys = [
        'player_idle.npy', 'player_move1.npy', 'player_move2.npy',
        'player_dead.npy', 'player_jump.npy', 'player_fall.npy',
        'player_climb1.npy', 'player_climb2.npy'
    ]
    
    bomb_keys = ['bomb.npy', 'magic_bomb.npy']
    princess_keys = ['princess_closed.npy', 'princess_open.npy']
    
    return (
        # Procedural assets
        {'name': 'background', 'type': 'background', 'data': static_procedural['background']},
        {'name': 'ui_colors', 'type': 'procedural', 'data': static_procedural['ui_colors']},
        {'name': 'debug_colors', 'type': 'procedural', 'data': static_procedural['debug_colors']},
        # Groups (will be auto-padded)
        {'name': 'player_group', 'type': 'group', 'files': player_keys},
        {'name': 'bomb_group', 'type': 'group', 'files': bomb_keys},
        {'name': 'princess_group', 'type': 'group', 'files': princess_keys},
        # Single sprites
        {'name': 'level', 'type': 'single', 'file': 'level.npy'},
        {'name': 'kong', 'type': 'single', 'file': 'kingkong_idle.npy'},
        {'name': 'life', 'type': 'single', 'file': 'player_move2.npy'}, # Was a copy of player_move2
        # Digits
        {'name': 'digits', 'type': 'digits', 'pattern': '{}.npy'},
    )

class KingKongConstants(AutoDerivedConstants):
	### Screen
	WIDTH: int = struct.field(pytree_node=False, default=160)
	HEIGHT: int = struct.field(pytree_node=False, default=250)
	DEBUG: int = struct.field(pytree_node=False, default=0) # Debug prints
	DEBUG_RENDER: int = struct.field(pytree_node=False, default=0) # Render some debug helpers

	### Sizes
	PLAYER_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([5, 16]))
	KONG_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([14, 34]))
	PRINCESS_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 17]))
	BOMB_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 14]))
	NUMBER_SIZE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([12, 14]))
	
	### Locations & Bounds

	# Player 
	PLAYER_RESPAWN_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([77, 228]))
	PLAYER_SUCCESS_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([87, 37]))
	
	# Level 
	LEVEL_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([8, 39]))
	
	# bounding boxes (x1, y1, x2, y2) - x1,y1 is top left 
	HOLE_LOCATIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
		[52, 83, 55, 84],  
		[104, 83, 107, 84],
		[40, 131, 43, 132],
		[116, 131, 119, 132],
		[56, 179, 59, 180],
		[100, 179, 103, 180]
	]))
	LADDER_LOCATIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
		[76, 41, 84, 60], # Topmost ladder, bombs can't take this one 
		[20, 61, 28, 85],
		[132, 61, 140, 84],
		[76, 85, 84, 108],
		[12, 109, 20, 132],
		[140, 109, 148, 132],
		[76, 133, 84, 156],
		[12, 157, 20, 180],
		[140, 157, 148, 180],
		[76, 181, 84, 204],
		[12, 205, 20, 228],
		[140, 205, 148, 228]
	]))
	FLOOR_BOUNDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
		[12, 150], # Ground floor 
		[12, 150], # First floor
		[12, 150], # Second floor 
		[12, 150], # Third floor 
		[12, 150], # Fourth floor 
		[12, 150], # Fifth floor 
		[16, 150], # Sixth floor
		[20, 142], # Seventh floor 
		[12, 150], # Princess floor - no bounds required bc goal reached 
	])) # floor bounds by floor (min_x, min_y) - y is always the same (see FLOOR_LOCATIONS)
	FLOOR_LOCATIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([228, 204, 180, 156, 132, 108, 84, 60, 40, 0])) # y corrdinate, 0 for topmost floor calculation reuqired

	PRINCESS_MOVEMENT_BOUNDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([77, 113]))

	LIFE_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([31, 228]))
	LIFE_SPACE_BETWEEN: int = struct.field(pytree_node=False, default=11) 

	SCORE_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 35]))
	SCORE_SPACE_BETWEEN: int = struct.field(pytree_node=False, default=4)

	TIMER_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([112, 35]))
	TIMER_SPACE_BETWEEN: int = struct.field(pytree_node=False, default=4) 

	# Entities 
	KONG_START_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([31, 228]))
	KONG_LOWER_LOCATION: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
	KONG_UPPER_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([31, 84]))
	KONG_LOWER_TELEPORT_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([31, 276])) # teleport here for him to end up at KONG_START_LOCATION
	KONG_UPPER_TELEPORT_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([31, 12])) # teleport here for him to end up at KONG_UPPER_LOCATION
	PRINCESS_START_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([93, 37]))
	PRINCESS_RESPAWN_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([77, 37])) # at the start the princess teleports to the left, this al 
	PRINCESS_SUCCESS_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([95, 37])) 
	BOMB_SPAWN_TOP_LOCATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([20, 60])) #top left top floor
	BOMB_SPAWN_BOTTOM_LOCATION: Optional[chex.Array] = struct.field(pytree_node=False, default=None)  # Kong's lower location
	BOMB_OFFSCREEN_BUFFER: int = struct.field(pytree_node=False, default=4) # 4 additional pixels before despawn

	### Gameplay 
	# There's six seperate stages: 
	# 0. Idle stage (Pre-Startup):
	#    - Runs for 130 steps at the very beginning.
	#    - Nothing happens (except King Kong spawns), game is paused before startup animation begins.
	#
	# 1. Startup stage (Animation):
	#    - Plays whenever the game is launched or the player has lost all lives (full relaunch).
	#    - King Kong leaps to the top of the building and places the girl at the highest floor.
	#
	# 2. Respawn stage (Animation):
	#    - Plays whenever a new life begins after the player died.
	#    - The player character reappears at the bottom of the building, ready to start climbing again.
	#
	# 3. Gameplay stage:
	#    - Core loop where the player climbs from the bottom to the top floor to rescue the girl.
	#    - Hazards:
	#        * King Kong throws bombs. Bombs roll down floors and ladders until the player crosses halfway,
	#          then King Kong jumps to the bottom and bombs roll upward instead.
	#        * Regular bombs kill on contact.
	#        * Magic bombs (look like candles) can be jumped over to instantly boost to the next floor.
	#        * Floors sometimes have holes; falling through costs a life.
	#    - Player controls:
	#        * left/right to move.
	#        * up/down to climb ladders.
	#        * jumps: vertical if standing, forward-arc if walking (cannot jump while climbing or on the top floor).
	#    - Scoring:
	#        * Jumping regular bomb: 25 points.
	#        * Jumping magic bomb: 125 points.
	#        * Bonus timer starts at 990, ticks down by 10 every second.
	#          Remaining bonus added to score when reaching the top.
	#          Timer at zero costs a life.
	#    - Loop:
	#        * Rescuing the girl resets the stage with faster bombs.
	#        * Player begins with 3 lives, game ends after final life is lost.
	#
	# 4. Death stage (Animation):
	#	 - Has two paths. Both lead back to RESPAWN stage:
	#		* BOMB_EXPLODE: Here an additional animation plays for the bomb exploding. The fall animation plays after. Also plays if the timer expires. 
	#		* FALL: Here no additional animation plays because the player only falls. 
	#    - Triggered when the player loses a life (bomb hit or fall).
	#    - Character plays a death animation before the respawn stage begins.
	#
	# 5. Success stage (Animation):
	#    - Triggered when the player reaches the girl at the top.
	#    - A short victory animation plays before the next loop starts with increased difficulty.
	#
	# All of these happen at specific steps and take a specific amount of steps. These are defined below. 
	# Naming Scheme: 
	# - SEQ: When (step count) something starts  
	# - DUR: How long (step count) something takes 

	# Define the gamestates/stages
	GAMESTATE_IDLE: int = struct.field(pytree_node=False, default=0) 
	GAMESTATE_STARTUP: int = struct.field(pytree_node=False, default=1)
	GAMESTATE_RESPAWN: int = struct.field(pytree_node=False, default=2) 
	GAMESTATE_GAMEPLAY: int = struct.field(pytree_node=False, default=3)
	GAMESTATE_DEATH: int = struct.field(pytree_node=False, default=4) 
	GAMESTATE_SUCCESS: int = struct.field(pytree_node=False, default=5)  

	# The way the game logic works is it checks has an internal 
	# stage_steps counter, which gets reset to 0 once a new stage is hit 
	# and counts up one per step. 
	# Since the game esentially is a state machine and it is clearly defined
	# which stage comes after another, knowing the duration of a stage is 
	# enough to model it.  
	# The gameplay stage is an execption, because it can end either when
	# the duration limit is hit (timer reaches 0) but also when the player
	# died or reached the goal. 
	# All variables below are relative to their stage. 

	# Define the idle stage
	DUR_IDLE: int = struct.field(pytree_node=False, default=130) 
	SEQ_IDLE_KINGKONG_SPAWN: int = struct.field(pytree_node=False, default=0) # as soon as this stage is reached kong spawns 

	###################################################################
	# Define the startup stage 
	DUR_STARTUP: int = struct.field(pytree_node=False, default=255) 
	SEQ_STARTUP_PRINCESS_SPAWN: int = struct.field(pytree_node=False, default=226) # Here the princess spawns in at PRINCESS_START_LOCATION  
	
	# First do 15 diagonal up jumps, then one to the left/right, then 3 diagonal down 
	KONG_TOTAL_JUMPS: int = struct.field(pytree_node=False, default=6)
	KONG_JUMPS_UP: int = struct.field(pytree_node=False, default=15)
	KONG_JUMPS_SIDE: int = struct.field(pytree_node=False, default=1) 
	KONG_JUMPS_DOWN: int = struct.field(pytree_node=False, default=3) 
	###################################################################

	###################################################################
	# Define the respawn stage 
	# Here the princess teleports to PRINCESS_RESPAWN_LOCATION 
	# The princess moves around at the top seemingly randomly (either wait, left or right) within her bounds
	# but she gets teleported back three times (the thrid time on the first frame of gameplay). 
	DUR_RESPAWN: int = struct.field(pytree_node=False, default=192) 
	SEQ_RESPAWN_KONG_TELEPORT: int = struct.field(pytree_node=False, default=0) # also teleport kong to the top again if it was at the bottom 
	SEQ_RESPAWN_PRINCESS_TELEPORT0: int = struct.field(pytree_node=False, default=0) # every 64 frame she is tp'd back in this stage for some reason 
	SEQ_RESPAWN_PRINCESS_TELEPORT1: int = struct.field(pytree_node=False, default=64) 
	SEQ_RESPAWN_PRINCESS_TELEPORT2: int = struct.field(pytree_node=False, default=128) 
	###################################################################

	###################################################################
	# Define gameplay stage
	# unused bc we have bonus_timer but could technically replace with this 
	# DUR_GAMEPLAY: int = 99 * FPS # 990 / 10 = 99 seconds, in steps 
	SEQ_GAMEPLAY_PRINCESS_TELEPORT: int = struct.field(pytree_node=False, default=0)  
	###################################################################

	###################################################################
	# Define the death stages
	
	# Path 1: Bomb explode  
	DUR_BOMB_EXPLODE: int = struct.field(pytree_node=False, default=96)
	SEQ_BOMB_EXPLODE_DEATH_FLASHES: int = struct.field(pytree_node=False, default=0) 
	SEQ_DEATH_FLASHES: int = struct.field(pytree_node=False, default=0) 
	CNT_DEATH_FLASHES: int = struct.field(pytree_node=False, default=24) 
	DUR_SINGLE_DEATH_FLASH: int = struct.field(pytree_node=False, default=4) # How long a death flash takes 
	DEATH_FLASH_COLORS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
		[82, 82, 82], # frame 0
		[151, 151, 151], # frame 1
		[210, 210, 210], # frame 2
		[0, 0, 0], # frame 3
		[82, 82, 82], # frame 4
		[151, 151, 151], # frame 5
		[210, 210, 210], # frame 6
		[128, 88, 0], # frame 7
		[171, 135, 50], # frame 8
		[207, 175, 92], # frame 9
		[238, 209, 128], # frame 10
		[68, 92, 0], # frame 11
		[118, 147, 50], # frame 12
		[160, 194, 92], # frame 13
		[196, 234, 128], # frame 14
		[112, 52, 0], # frame 15
		[160, 107, 50], # frame 16
		[201, 154, 92], # frame 17
		[236, 194, 128], # frame 18
		[0, 100, 20], # frame 19
		[50, 152, 82], # frame 20
		[92, 197, 135], # frame 21
		[128, 235, 180], # frame 22
		[112, 0, 20]  # frame 23 
	], dtype=jnp.uint8))

	# Death types
	DEATH_TYPE_NONE: int = struct.field(pytree_node=False, default=0)
	DEATH_TYPE_BOMB_EXPLODE: int = struct.field(pytree_node=False, default=1)
	DEATH_TYPE_FALL: int = struct.field(pytree_node=False, default=2) 

	# Path 2: Fall 
	DUR_FALL: int = struct.field(pytree_node=False, default=232) # During this time the player falls to the floor below. First fall, then show the blob, no step restriction  
	###################################################################

	###################################################################
	# Define the success stage q	
	DUR_SUCCESS: int = struct.field(pytree_node=False, default=232) # ~ death time 
	###################################################################
		
	### Game logic constants 
	BONUS_START: int = struct.field(pytree_node=False, default=990)
	BONUS_DECREMENT: int = struct.field(pytree_node=False, default=10) # per second 

	FLOOR_KONG_MOVE_DOWN: int = struct.field(pytree_node=False, default=4) # fifth floor (start count at 0)
	FLOOR_KONG_MOVE_UP: int = struct.field(pytree_node=False, default=2)

	PRINCESS_MOVE_OPTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 0, 0, 3, -3, 6, -6])) 

	REGULAR_BOMB_POINTS: int = struct.field(pytree_node=False, default=25) 
	MAGIC_BOMB_POINTS: int = struct.field(pytree_node=False, default=125) 

	BOMB_SPEED_BASE_INTERVAL: int = struct.field(pytree_node=False, default=2)
	BOMB_SPEED_MIN_INTERVAL: int = struct.field(pytree_node=False, default=1)

	PLAYER_JUMP_HEIGHT: int = struct.field(pytree_node=False, default=12) 
	PLAYER_CATAPULT_HEIGHT: int = struct.field(pytree_node=False, default=40) 

	MAX_BOMBS: int = struct.field(pytree_node=False, default=8) #one for each floor 
	MAX_SPEED: int = struct.field(pytree_node=False, default=1)
	MAX_LIVES: int = struct.field(pytree_node=False, default=3)
	MAX_SCORE: int = struct.field(pytree_node=False, default=999_999) # this is basically unachievable 

	# Player states
	PLAYER_IDLE_LEFT: int = struct.field(pytree_node=False, default=1)
	PLAYER_IDLE_RIGHT: int = struct.field(pytree_node=False, default=2)
	PLAYER_MOVE_LEFT: int = struct.field(pytree_node=False, default=3)
	PLAYER_MOVE_RIGHT: int = struct.field(pytree_node=False, default=4)
	PLAYER_JUMP_LEFT: int = struct.field(pytree_node=False, default=5) 
	PLAYER_JUMP_RIGHT: int = struct.field(pytree_node=False, default=6)
	PLAYER_CLIMB_UP: int = struct.field(pytree_node=False, default=7)
	PLAYER_CLIMB_DOWN: int = struct.field(pytree_node=False, default=8)
	PLAYER_CLIMB_IDLE: int = struct.field(pytree_node=False, default=42) 
	PLAYER_FALL: int = struct.field(pytree_node=False, default=9)
	PLAYER_DEAD: int = struct.field(pytree_node=False, default=10)
	PLAYER_GOAL: int = struct.field(pytree_node=False, default=11)
	PLAYER_CATAPULT_LEFT: int = struct.field(pytree_node=False, default=44)
	PLAYER_CATAPULT_RIGHT: int = struct.field(pytree_node=False, default=45) 

	# Asset config baked into constants (immutable default) for asset overrides
	ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: _get_default_asset_config())

	def compute_derived(self):
		return {
			"KONG_LOWER_LOCATION": self.KONG_START_LOCATION,
			"BOMB_SPAWN_BOTTOM_LOCATION": self.KONG_START_LOCATION
		}

	def __post_init__(self):
		"""Initialize derived constants and validate relationships."""
		super().__post_init__()
		assert self.DUR_SINGLE_DEATH_FLASH * self.CNT_DEATH_FLASHES == self.DUR_BOMB_EXPLODE, \
			f"Death flash duration mismatch: {self.DUR_SINGLE_DEATH_FLASH} * {self.CNT_DEATH_FLASHES} != {self.DUR_BOMB_EXPLODE}"

@struct.dataclass
class KingKongState:
	# Game state management
	gamestate: chex.Array # Current game stage
	stage_steps: chex.Array # Steps within current stage
	step_counter: chex.Array # Global step counter
	rng_key: chex.PRNGKey
	
	# Player state
	player_x: chex.Array
	player_y: chex.Array
	player_state: chex.Array # PLAYER_IDLE_LEFT, PLAYER_MOVE_LEFT, etc.
	player_floor: chex.Array # Current floor (0-8)
	player_highest_floor: chex.Array # highest floor reached
	player_jump_counter: chex.Array  # For jump animation
	player_dir: chex.Array  # (-1, 0, or 1)

	# Kong state
	kong_x: chex.Array
	kong_y: chex.Array
	kong_visible: chex.Array
	kong_on_top: chex.Array 
	kong_jump_counter: chex.Array
	
	# Princess state
	princess_x: chex.Array
	princess_y: chex.Array
	princess_visible: chex.Array
	princess_waving: chex.Array	
	princess_waving_counter: chex.Array
	# tracks current step in a move, either positive or negative and 
	# counts down to zero from either direction where every count is a step 
	princess_movement_step: chex.Array 

	# Bombs (multiple bombs can exist, up to 8)
	# Shape of all (MAX_BOMBS)
	bomb_positions_x: chex.Array 
	bomb_positions_y: chex.Array
	bomb_active: chex.Array
	bomb_is_magic: chex.Array
	bomb_directions_x: chex.Array
	bomb_directions_y: chex.Array
	bomb_floor: chex.Array
	bomb_points_given: chex.Array
	 
	# Game stats
	score: chex.Array
	lives: chex.Array
	bonus_timer: chex.Array
	level: chex.Array # Difficulty level (impacts bomb speed)
	
	# Death state info
	death_type: chex.Array
	death_flash_counter: chex.Array
	death_target_y: chex.Array # when fallling to death 

@struct.dataclass
class EntityPosition:
	x: jnp.ndarray
	y: jnp.ndarray
	width: jnp.ndarray
	height: jnp.ndarray
	visible: jnp.ndarray

@struct.dataclass
class BombInfo:
	positions_x: jnp.ndarray
	positions_y: jnp.ndarray  
	active: jnp.ndarray
	is_magic: jnp.ndarray
	directions_x: jnp.ndarray
	directions_y: jnp.ndarray

@struct.dataclass
class KingKongObservation:
	player: EntityPosition
	kong: EntityPosition
	princess: EntityPosition
	bombs: BombInfo
	score: jnp.ndarray
	lives: jnp.ndarray
	bonus_timer: jnp.ndarray
	level: jnp.ndarray
	gamestate: jnp.ndarray
	player_floor: jnp.ndarray

@struct.dataclass
class KingKongInfo:
	time: chex.Array

class JaxKingKong(JaxEnvironment[KingKongState, KingKongObservation, KingKongInfo, KingKongConstants]):
	# Minimal ALE action set for King Kong (from scripts/action_space_helper.py)
	ACTION_SET: jnp.ndarray = jnp.array(
		[
			Action.NOOP,
			Action.FIRE,
			Action.UP,
			Action.RIGHT,
			Action.LEFT,
			Action.DOWN,
		],
		dtype=jnp.int32,
	)

	def __init__(self, consts: KingKongConstants = None):
		consts = consts or KingKongConstants()
		super().__init__(consts)
		self.renderer = KingKongRenderer(self.consts)
		self.obs_size = 5 + 5 + 5 + (6 * self.consts.MAX_BOMBS) + 6  # player + kong + princess + bombs + game_info

	def reset(self, key=None) -> Tuple[KingKongObservation, KingKongState]:
		if key is None:
			key = jax.random.PRNGKey(42) # some default seed if none given

		state = KingKongState(
			# Game state
			gamestate=jnp.array(self.consts.GAMESTATE_IDLE).astype(jnp.int32),
			stage_steps=jnp.array(0).astype(jnp.int32),
			step_counter=jnp.array(0).astype(jnp.int32),
			rng_key=key,
			
			# Player state
			player_x=self.consts.PLAYER_RESPAWN_LOCATION[0].astype(jnp.int32),
			player_y=self.consts.PLAYER_RESPAWN_LOCATION[1].astype(jnp.int32),
			player_state=jnp.array(self.consts.PLAYER_IDLE_RIGHT).astype(jnp.int32),
			player_floor=jnp.array(0).astype(jnp.int32), # Ground floor
			player_highest_floor=jnp.array(0).astype(jnp.int32),
			player_jump_counter=jnp.array(0).astype(jnp.int32),
			player_dir=jnp.array(0).astype(jnp.int32),
			
			# Kong state
			kong_x=jnp.array(self.consts.KONG_LOWER_LOCATION[0]).astype(jnp.int32),
			kong_y=jnp.array(self.consts.KONG_LOWER_LOCATION[1]).astype(jnp.int32),
			kong_visible=jnp.array(1).astype(jnp.int32),
			kong_on_top=jnp.array(1).astype(jnp.int32),
			kong_jump_counter=jnp.array(0).astype(jnp.int32),
			
			# Princess state
			princess_x=jnp.array(0).astype(jnp.int32),
			princess_y=jnp.array(0).astype(jnp.int32),
			princess_visible=jnp.array(0).astype(jnp.int32),
			princess_waving=jnp.array(0).astype(jnp.int32),
			princess_waving_counter=jnp.array(0).astype(jnp.int32),
			princess_movement_step=jnp.array(0).astype(jnp.int32),

			# Bombs
			bomb_positions_x=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_positions_y=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_active=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_is_magic=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_directions_x=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_directions_y=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_floor=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			bomb_points_given=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
			
			# Game stats
			score=jnp.array(0).astype(jnp.int32),
			lives=jnp.array(self.consts.MAX_LIVES).astype(jnp.int32),
			bonus_timer=jnp.array(self.consts.BONUS_START).astype(jnp.int32),
			level=jnp.array(1).astype(jnp.int32),
			
			# Death state
			death_type=jnp.array(self.consts.DEATH_TYPE_NONE).astype(jnp.int32),
			death_flash_counter=jnp.array(0).astype(jnp.int32),
			death_target_y=jnp.array(-1).astype(jnp.int32)
		)
		
		initial_obs = self._get_observation(state)
		return initial_obs, state
	
	@partial(jax.jit, static_argnums=(0,))
	def step(self, state: KingKongState, action: chex.Array) -> Tuple[KingKongObservation, KingKongState, float, bool, KingKongInfo]:
		# Translate compact agent action index to ALE console action
		atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
		
		# Handle different game states with current stage_steps
		if self.consts.DEBUG: jax.debug.print("gamestate={g} stage_steps={s}", g=state.gamestate, s=state.stage_steps)
		new_state: KingKongState = jax.lax.switch(
			state.gamestate,
			[
				lambda s, a: self._step_idle(s, a),
				lambda s, a: self._step_startup(s, a),
				lambda s, a: self._step_respawn(s, a),
				lambda s, a: self._step_gameplay(s, a),
				lambda s, a: self._step_death(s, a),
				lambda s, a: self._step_success(s, a),
			],
			state, atari_action
		)

		# Update princess movement here bc she always moves (except on success)
		new_state = jax.lax.cond(
			jnp.logical_and(new_state.princess_visible != 0, state.gamestate != self.consts.GAMESTATE_SUCCESS), # dont move on success state 
			self._update_princess_movement,
			lambda s: s,
			new_state
		)
		
		# Update global step counter
		new_state = new_state.replace(step_counter=state.step_counter + 1)
		
		done = self._get_done(new_state)
		env_reward = self._get_reward(state, new_state)
		info = self._get_info(new_state)
		observation = self._get_observation(new_state)
	
		return observation, new_state, env_reward, done, info

	def _update_princess_movement(self, state: KingKongState) -> KingKongState:
		key, subkey1, subkey2, subkey3 = jax.random.split(state.rng_key, 4)

		def do_normal_step(_):
			def pick_or_idle(_):
				# chance to pick a new move when possible
				do_move = jax.random.bernoulli(subkey1, p=0.15)

				def pick_move(_):
					move_idx = jax.random.randint(subkey2, (), 0, len(self.consts.PRINCESS_MOVE_OPTIONS))
					return self.consts.PRINCESS_MOVE_OPTIONS[move_idx], 1

				def idle_move(_):
					return 0, 0

				return jax.lax.cond(do_move, pick_move, idle_move, operand=None)

			def do_princess_move(_):
				dx = jax.lax.cond(state.princess_movement_step > 0, lambda _: -1, lambda _: +1, operand=None)
				return state.princess_movement_step + dx, dx

			# if waving_counter > 0, continue waving
			def continue_waving(_):
				new_waving_counter = jnp.maximum(state.princess_waving_counter - 1, 0)
				return 0, 0, new_waving_counter, 1

			def normal_move(_):
				new_princess_movement_step, dx = jax.lax.cond(
					state.princess_movement_step == 0,
					pick_or_idle,
					do_princess_move,
					operand=None
				)
				# randomly decide if she's starting to wave (only if not currently waving)
				start_waving = jnp.where(
					jax.random.bernoulli(subkey3, p=0.05),
					jax.random.randint(subkey3, (), 6, 23),# inclusive lower, exclusive upper
					0
				)
				waving = jnp.where(start_waving > 0, 1, 0)
				return new_princess_movement_step, dx, start_waving, waving

			new_princess_movement_step, dx, new_waving_counter, waving = jax.lax.cond(
				state.princess_waving_counter > 0,
				continue_waving,
				normal_move,
				operand=None
			)

			new_x = jnp.clip(
				state.princess_x + dx,
				self.consts.PRINCESS_MOVEMENT_BOUNDS[0],
				self.consts.PRINCESS_MOVEMENT_BOUNDS[1] - self.consts.PRINCESS_SIZE[0]
			)
	
			if self.consts.DEBUG: jax.debug.print("princess_move: x={x} dx={dx} waving={w} waving_counter={wc}", x=state.princess_x, dx=dx, w=waving, wc=new_waving_counter)

			return state.replace(
				princess_x=new_x,
				princess_movement_step=new_princess_movement_step,
				princess_waving=waving,
				princess_waving_counter=new_waving_counter,
				rng_key=key
			)
		

		return jax.lax.cond(state.stage_steps % 4 == 0, do_normal_step, lambda _: state, operand=None)

	def _step_idle(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_IDLE

		# Kong visibility
		kong_visible = jax.lax.cond(
			state.stage_steps == self.consts.SEQ_IDLE_KINGKONG_SPAWN,
			lambda: 1,
			lambda: state.kong_visible
		)
		# Kong spawn position
		kong_x = jax.lax.cond(
			state.stage_steps == self.consts.SEQ_IDLE_KINGKONG_SPAWN,
			lambda: self.consts.KONG_START_LOCATION[0],
			lambda: state.kong_x
		)
		kong_y = jax.lax.cond(
			state.stage_steps == self.consts.SEQ_IDLE_KINGKONG_SPAWN,
			lambda: self.consts.KONG_START_LOCATION[1],
			lambda: state.kong_y
		)
		
		new_gamestate = jax.lax.cond(
			should_transition,
			lambda: self.consts.GAMESTATE_STARTUP,
			lambda: state.gamestate
		)
		
		final_stage_steps = jax.lax.cond(
			should_transition,
			lambda: 0,
			lambda: state.stage_steps + 1
		)
		
		return state.replace(
			gamestate=new_gamestate,
			kong_visible=kong_visible,
			kong_x=kong_x,
			kong_y=kong_y,
			stage_steps=final_stage_steps
		)
	
	def _step_startup(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_STARTUP

		def do_transition(_):
			return state.replace(
				gamestate=self.consts.GAMESTATE_RESPAWN,
				stage_steps=0,
				kong_jump_counter=0,#reset the jump counter
				level=1 # reset the level
			)

		def do_normal_step(_):
			kong_x, kong_y, kong_jump_counter = jax.lax.cond(
				state.stage_steps % 2 == 0,
				lambda _: self._move_kong(state),
				lambda _: (state.kong_x, state.kong_y, state.kong_jump_counter),
				operand=None
			)

			# Princess
			princess_visible = jax.lax.cond(
				state.stage_steps == self.consts.SEQ_STARTUP_PRINCESS_SPAWN,
				lambda: 1,
				lambda: state.princess_visible
			)
			princess_x = jax.lax.cond(
				state.stage_steps == self.consts.SEQ_STARTUP_PRINCESS_SPAWN,
				lambda: self.consts.PRINCESS_START_LOCATION[0],
				lambda: state.princess_x
			)
			princess_y = jax.lax.cond(
				state.stage_steps == self.consts.SEQ_STARTUP_PRINCESS_SPAWN,
				lambda: self.consts.PRINCESS_START_LOCATION[1],
				lambda: state.princess_y
			)

			final_stage_steps = state.stage_steps + 1

			return state.replace(
				stage_steps=final_stage_steps,
				kong_x=kong_x,
				kong_y=kong_y,
				kong_jump_counter=kong_jump_counter,
				princess_visible=princess_visible,
				princess_x=princess_x,
				princess_y=princess_y
			)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)
	
	# Zigzag movement
	def _move_kong(self, state: KingKongState):
		# Stop if we've reached the total jump limit
		total_jumps_per_cycle = self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE + self.consts.KONG_JUMPS_DOWN
		max_jump_counter = self.consts.KONG_TOTAL_JUMPS * total_jumps_per_cycle
		
		# If we've exceeded total jumps, don't move
		should_continue = state.kong_jump_counter < max_jump_counter
		
		def do_jump(state: KingKongState):
			# check which full zigzag phase we are in
			phase = state.kong_jump_counter // total_jumps_per_cycle
			dir_lr = jnp.where(phase % 2 == 0, 1, -1) # even phase: right, odd: left
			step_in_phase = state.kong_jump_counter % total_jumps_per_cycle
			
			new_x, new_y = jax.lax.cond(
				step_in_phase < self.consts.KONG_JUMPS_UP,# diagonal up
				lambda _: (state.kong_x + dir_lr, state.kong_y - 2),
				lambda _: jax.lax.cond(
					step_in_phase < self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE,# side step
					lambda _: (state.kong_x + dir_lr, state.kong_y),
					lambda _: (state.kong_x + dir_lr, state.kong_y + 2),# diagonal down
					operand=None
				),
				operand=None
			)

			return new_x, new_y, state.kong_jump_counter + 1
		
		return jax.lax.cond(
			should_continue,
			do_jump,
			lambda _: (state.kong_x, state.kong_y, state.kong_jump_counter),# Don't increment counter when stopped
			operand=state
		)
		
	def _step_respawn(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_RESPAWN

		def do_transition(_):
			return state.replace(
				gamestate=self.consts.GAMESTATE_GAMEPLAY,
				stage_steps=0,
				player_x=self.consts.PLAYER_RESPAWN_LOCATION[0],
				player_y=self.consts.PLAYER_RESPAWN_LOCATION[1],
				player_floor=0,
				player_highest_floor=0,
				kong_x=self.consts.KONG_UPPER_LOCATION[0],
				kong_y=self.consts.KONG_UPPER_LOCATION[1],
				kong_on_top=1,
				# Reset bombs for new level (this done after for some reason)
				bomb_positions_x=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_positions_y=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_active=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_is_magic=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_directions_x=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_directions_y=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_floor=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
				bomb_points_given=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32)
			)

		def do_normal_step(_):
			# Princess teleports at specific intervals
			should_teleport0 = state.stage_steps == self.consts.SEQ_RESPAWN_PRINCESS_TELEPORT0
			should_teleport1 = state.stage_steps == self.consts.SEQ_RESPAWN_PRINCESS_TELEPORT1
			should_teleport2 = state.stage_steps == self.consts.SEQ_RESPAWN_PRINCESS_TELEPORT2
			teleport_princess = should_teleport0 | should_teleport1 | should_teleport2

			princess_x = jax.lax.cond(
				teleport_princess,
				lambda _: self.consts.PRINCESS_RESPAWN_LOCATION[0],
				lambda _: state.princess_x,
				operand=None
			)

			princess_y = jax.lax.cond(
				jnp.logical_or(should_teleport1, should_teleport2),
				lambda _: self.consts.PRINCESS_RESPAWN_LOCATION[1],
				lambda _: state.princess_y,
				operand=None
			)

			should_teleport_kong = state.stage_steps == self.consts.SEQ_RESPAWN_KONG_TELEPORT
			kong_x = jax.lax.cond(
				should_teleport_kong,
				lambda _: self.consts.KONG_UPPER_LOCATION[0],
				lambda _: state.kong_x,
				operand=None
			)
			kong_y = jax.lax.cond(
				should_teleport_kong,
				lambda _: self.consts.KONG_UPPER_LOCATION[1],
				lambda _: state.kong_y,
				operand=None
			)
			
			final_stage_steps = state.stage_steps + 1

			return state.replace(
				stage_steps=final_stage_steps,
				princess_x=princess_x,
				princess_y=princess_y,
				bonus_timer=jnp.array(self.consts.BONUS_START).astype(jnp.int32),
				kong_x=kong_x,
				kong_y=kong_y
			)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)

	
	def _step_gameplay(self, state: KingKongState, action: chex.Array) -> KingKongState:
		player_reached_top = state.player_floor >= 8
		timer_expired = state.bonus_timer <= 0
		should_die = jnp.logical_or(timer_expired, state.death_type != self.consts.DEATH_TYPE_NONE)
		should_transition = jnp.logical_or(player_reached_top, should_die)

		final_stage_steps = state.stage_steps + 1

		def do_transition(_):
			new_gamestate = jax.lax.cond(
				player_reached_top,
				lambda _: self.consts.GAMESTATE_SUCCESS,
				lambda _: self.consts.GAMESTATE_DEATH,
				operand=None
			)

			# for now move to 0,0 in case of success, let the function tp him then where he needs to be 
			new_player_x = jax.lax.cond(player_reached_top, lambda _: 0, lambda _: state.player_x, operand=None)
			new_player_y = jax.lax.cond(player_reached_top, lambda _: 0, lambda _: state.player_y, operand=None)

			return state.replace(
				gamestate=new_gamestate,
				stage_steps=0,
				player_x=new_player_x,
				player_y=new_player_y,
				princess_visible=1,
				death_type=jax.lax.cond(
					jnp.logical_and(timer_expired, state.death_type == self.consts.DEATH_TYPE_NONE),
					lambda _: self.consts.DEATH_TYPE_BOMB_EXPLODE,
					lambda _: state.death_type,
					operand=None
				)
			)


		def do_normal_step(_):
			# princess teleport
			princess_x = jax.lax.cond(
			(state.stage_steps == self.consts.SEQ_GAMEPLAY_PRINCESS_TELEPORT),
				lambda _: self.consts.PRINCESS_RESPAWN_LOCATION[0],
				lambda _: state.princess_x,
				operand=None
			)

			# bonus timer ticks every 63 frames
			new_bonus_timer = jax.lax.cond(
				jnp.logical_and(jnp.logical_and(state.stage_steps != 0, state.stage_steps % 63 == 0), state.bonus_timer > 0),
				lambda _: jnp.maximum(0, state.bonus_timer - self.consts.BONUS_DECREMENT),
				lambda _: state.bonus_timer,
				operand=None
			)

			# player update every 4 steps
			new_state: KingKongState = jax.lax.cond(
				(state.stage_steps % 4) == 0,
				lambda _: self._update_player_gameplay(state, action),
				lambda _: state,
				operand=None
			)

			new_state = new_state.replace(
				bonus_timer=new_bonus_timer,
				princess_x=princess_x,
				stage_steps=state.stage_steps
			)

			new_state = self._update_bombs(new_state)

			# bombs/kong/collisions every 2 steps
			new_state = jax.lax.cond(
				(state.stage_steps % 2) == 0,
				lambda _: self._check_collisions(self._update_kong(new_state)),
				lambda _: new_state,
				operand=None
			)

			return new_state.replace(stage_steps=final_stage_steps)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)

	def _update_player_gameplay(self, state: KingKongState, action: chex.Array) -> KingKongState:
		# Intentions (only check actions in ALE minimal action set)
		move_left = action == Action.LEFT
		move_right = action == Action.RIGHT
		move_up = action == Action.UP
		move_down = action == Action.DOWN
		jump = action == Action.FIRE

		# Get floor bounds
		current_floor_bounds = self.consts.FLOOR_BOUNDS[state.player_floor]
		min_x = current_floor_bounds[0]
		max_x = current_floor_bounds[1] - (self.consts.PLAYER_SIZE[0] - 2) # account for sprite offset

		new_player_x = state.player_x
		new_player_y = state.player_y
		new_player_floor = state.player_floor
		new_player_state = state.player_state
		new_player_jump_counter = state.player_jump_counter
		new_player_dir = state.player_dir

		at_floor_bottom = state.player_y == self.consts.FLOOR_LOCATIONS[state.player_floor]

		# State flags
		is_climbing = (state.player_state == self.consts.PLAYER_CLIMB_UP) | (state.player_state == self.consts.PLAYER_CLIMB_DOWN) | (state.player_state == self.consts.PLAYER_CLIMB_IDLE)
		is_catapult = ((state.player_state == self.consts.PLAYER_CATAPULT_LEFT) | (state.player_state == self.consts.PLAYER_CATAPULT_RIGHT))
		is_jumping = (state.player_state == self.consts.PLAYER_JUMP_LEFT) | (state.player_state == self.consts.PLAYER_JUMP_RIGHT) | is_catapult

		# --- Handle jumping physics first if already jumping ---
		# NOTE: Since running jumps do not work in the reference, I did not know exactly 
		# how far these can go. If you want to change this in the future it can be done below. 
		# I used this for reference: https://www.youtube.com/watch?v=UX6vvr7iplY&t=370s
		def update_jump_physics():
			# Check if this is a catapult jump
			
			# Use different jump height for catapult
			jump_height = jax.lax.cond(
				is_catapult,
				lambda: self.consts.PLAYER_CATAPULT_HEIGHT,
				lambda: self.consts.PLAYER_JUMP_HEIGHT
			)
			
			jump_peak = jump_height // 2 #3
			at_peak = state.player_jump_counter == jump_peak
			ascending = state.player_jump_counter < jump_peak
			
			dy = jax.lax.cond(
				at_peak,
				lambda: 0,  # pause at peak
				lambda: jax.lax.cond(ascending, lambda: -2, lambda: 2)
			)

			dx = state.player_dir
			new_x = jnp.clip(state.player_x + dx, min_x, max_x)
			floor_position = self.consts.FLOOR_LOCATIONS[new_player_floor]
			new_y = state.player_y + dy
			new_y = jax.lax.cond(
				~ascending, 
				lambda: jnp.minimum(new_y, floor_position),
				lambda: new_y
			)
			
			jump_complete = jnp.logical_or(state.player_jump_counter >= jump_height, new_y == floor_position)
									
			final_jump_dx = jax.lax.cond(jump_complete, lambda: 0, lambda: state.player_dir)

			final_jump_counter = jax.lax.cond(
				jump_complete,
				lambda: 0,
				lambda: state.player_jump_counter + 1
			)

			# Set player to idle when landed
			final_state = jax.lax.cond(
				jump_complete,
				lambda: jax.lax.cond(
					jnp.logical_or(state.player_state == self.consts.PLAYER_JUMP_LEFT, state.player_state == self.consts.PLAYER_CATAPULT_LEFT),
					lambda: self.consts.PLAYER_IDLE_LEFT,
					lambda: self.consts.PLAYER_IDLE_RIGHT
				),
				lambda: state.player_state
			)

			return new_x, new_y, final_jump_counter, final_state, final_jump_dx

		
		# Apply jump physics if currently jumping
		new_player_x, new_player_y, new_player_jump_counter, new_player_state, new_player_dir = jax.lax.cond(
			is_jumping,
			update_jump_physics,
			lambda: (new_player_x, new_player_y, new_player_jump_counter, new_player_state, new_player_dir)
		)

		# --- Horizontal movement (allowed only at floor bottom if not jumping) ---
		can_move_horiz = jnp.logical_and(
			at_floor_bottom,
			jnp.logical_not(is_jumping)
		)
		
		new_player_x = jax.lax.cond(
			jnp.logical_and(can_move_horiz, move_left & (new_player_x > min_x)),
			lambda: new_player_x - 1,
			lambda: new_player_x
		)
		new_player_state = jax.lax.cond(
			jnp.logical_and(can_move_horiz, move_left),
			lambda: self.consts.PLAYER_MOVE_LEFT,
			lambda: new_player_state
		)
		new_player_x = jax.lax.cond(
			jnp.logical_and(can_move_horiz, move_right & (new_player_x < max_x)),
			lambda: new_player_x + 1,
			lambda: new_player_x
		)
		new_player_state = jax.lax.cond(
			jnp.logical_and(can_move_horiz, move_right),
			lambda: self.consts.PLAYER_MOVE_RIGHT,
			lambda: new_player_state
		)

		# --- Ladder climbing (disabled if jumping) ---
		can_climb = jnp.logical_not(is_jumping)
		on_ladder = self._check_on_ladder(new_player_x, new_player_y, inset_x=2) #inset 2 seems to be whats used in the original
		ladder_below = self._check_on_ladder(new_player_x, new_player_y + 2, inset_x=2)
		apply_climb = (state.stage_steps % 8) == 0

		# --- Climb Up ---
		at_ladder_top = ladder_below
		at_top_floor = state.player_floor == 8
		can_climb_up = jnp.logical_and(can_climb, jnp.logical_and(on_ladder, jnp.logical_not(at_top_floor)))
		do_climb_up = jnp.logical_and(move_up, can_climb_up) & apply_climb

		new_player_y = jax.lax.cond(do_climb_up, lambda: new_player_y - 2, lambda: new_player_y)
		new_player_state = jax.lax.cond(
			do_climb_up,
			lambda: self.consts.PLAYER_CLIMB_UP,
			lambda: new_player_state
		)

		# --- Climb Down ---
		at_ladder_bottom = at_floor_bottom
		can_climb_down = can_climb & (
			jnp.logical_or(
				jnp.logical_and(on_ladder, jnp.logical_not(at_ladder_bottom)),  # on ladder but not at bottom
				ladder_below  # ladder below allows climb down
			)
		)
		do_climb_down = move_down & can_climb_down & apply_climb

		# Update y-coordinate first
		target_floor = jax.lax.cond(
			new_player_y < self.consts.FLOOR_LOCATIONS[state.player_floor],
			lambda: state.player_floor,
			lambda: jnp.maximum(state.player_floor - 1, 0)
		)

		new_player_y = jax.lax.cond(
			do_climb_down,
			lambda: jnp.minimum(new_player_y + 2, self.consts.FLOOR_LOCATIONS[target_floor]),
			lambda: new_player_y
		)

		# Update state: idle if at bottom, otherwise climbing down
		new_player_state = jax.lax.cond(
			do_climb_down,
			lambda: self.consts.PLAYER_CLIMB_DOWN,
			lambda: new_player_state
		)

		# --- Jumping (disabled if climbing or already jumping) ---
		can_jump = jnp.logical_and(jump, jnp.logical_and(state.player_floor < 8, jnp.logical_and(jnp.logical_not(is_climbing), jnp.logical_not(is_jumping))))
		
		# Initialize jump - use player's current direction (idle state) to determine jump direction
		new_player_state = jax.lax.cond(
			can_jump,
			lambda: jax.lax.cond(
				state.player_state == self.consts.PLAYER_IDLE_LEFT,
				lambda: self.consts.PLAYER_JUMP_LEFT,
				lambda: self.consts.PLAYER_JUMP_RIGHT  # Default to right if idle right or other states
			),
			lambda: new_player_state
		)
		
		# Only update direction when moving horizontally and not jumping or climbing
		can_update_dir = jnp.logical_and(
			is_jumping == False,
			is_climbing == False
		)

		new_player_dir = jax.lax.cond(
			can_update_dir & move_left,
			lambda: -1,
			lambda: jax.lax.cond(
				can_update_dir & move_right,
				lambda: 1,
				lambda: new_player_dir  # keep previous if not moving
			)
		)

		# Set jump direction based on player's current direction when starting a jump
		new_player_dir = jax.lax.cond(
			can_jump,
			lambda: jax.lax.cond(
				state.player_state == self.consts.PLAYER_IDLE_LEFT,
				lambda: -1,  # Jump left if idle left
				lambda: 1    # Jump right if idle right or other states
			),
			lambda: new_player_dir # Keep current direction if not jumping
		)

		new_player_jump_counter = jax.lax.cond(
			can_jump,
			lambda: 0,  # Reset jump counter when starting new jump
			lambda: new_player_jump_counter
		)
		
		# Determine the new floor based on new_player_y
		new_player_floor = jnp.argmin(new_player_y <= self.consts.FLOOR_LOCATIONS) - 1
		
		# Update highest floor reached
		new_highest_floor = jnp.maximum(state.player_highest_floor, new_player_floor)

		# Determine if climbing but not actively moving
		is_climbing_idle = (
			is_climbing &
			(
				jnp.logical_not(jnp.logical_or(move_up, move_down)) |  # no up/down pressed
				(move_down & at_ladder_bottom) | # pressing down but already at bottom
				(move_up & ~on_ladder & at_ladder_top) # same for top
			)
		)

		no_movement = jnp.logical_and(
			jnp.logical_not(move_left | move_right | move_up | move_down | jump),
			jnp.logical_not(is_jumping)
		)

		state_unchanged = new_player_state == state.player_state

		new_player_state = jax.lax.cond(
			is_climbing_idle & state_unchanged,
			lambda: self.consts.PLAYER_CLIMB_IDLE,
			lambda: jax.lax.cond(
				no_movement,
				lambda: jax.lax.cond(
					state.player_state == self.consts.PLAYER_MOVE_LEFT,
					lambda: self.consts.PLAYER_IDLE_LEFT,
					lambda: jax.lax.cond(
						state.player_state == self.consts.PLAYER_MOVE_RIGHT,
						lambda: self.consts.PLAYER_IDLE_RIGHT,
						lambda: jax.lax.cond(
							jnp.logical_or(
								state.player_state == self.consts.PLAYER_IDLE_LEFT,
								state.player_state == self.consts.PLAYER_IDLE_RIGHT
							),
							lambda: state.player_state,
							lambda: self.consts.PLAYER_IDLE_RIGHT
						)
					)
				),
				lambda: new_player_state
			)
		)

		if self.consts.DEBUG: 
			jax.debug.print("player: x={x} y={y} floor={f} state={st} jump_counter={jc} jump_dx={jdx}, at_floor_bottom={fb}, is_climbing_idle={cbmi}",
				fb=at_floor_bottom, cbmi=is_climbing_idle,
				x=new_player_x, y=new_player_y, f=new_player_floor, st=new_player_state, jc=new_player_jump_counter, jdx=new_player_dir)

		return state.replace(
			player_x=new_player_x,
			player_y=new_player_y,
			player_floor=new_player_floor,
			player_highest_floor=new_highest_floor,
			player_state=new_player_state,
			player_jump_counter=new_player_jump_counter,
			player_dir=new_player_dir
		)

	def _check_on_ladder(self, player_x, player_y, inset_x: int = 0, inset_y: int = 0):
		ladder_x1 = self.consts.LADDER_LOCATIONS[:, 0] + inset_x
		ladder_y1 = self.consts.LADDER_LOCATIONS[:, 1] + inset_y
		ladder_x2 = self.consts.LADDER_LOCATIONS[:, 2] - inset_x
		ladder_y2 = self.consts.LADDER_LOCATIONS[:, 3] - inset_y

		player_w, player_h = self.consts.PLAYER_SIZE

		on_ladders = jnp.logical_and(
			jnp.logical_and(player_x >= ladder_x1, player_x <= ladder_x2),
			jnp.logical_and(player_y >= ladder_y1, player_y <= ladder_y2)
		)
		result = jnp.any(on_ladders)

		if self.consts.DEBUG: 
			jax.debug.print(
				"check_on_ladder: player=({x},{y},{w},{h}), ladders=({x1},{y1},{x2},{y2}), result={res}",
				x=player_x, y=player_y, w=player_w, h=player_h,
				x1=ladder_x1, y1=ladder_y1, x2=ladder_x2, y2=ladder_y2,
				res=result
			)

		return result

	def _update_bombs(self, state: KingKongState) -> KingKongState:
		# get spawn location
		spawn_location = jax.lax.cond(
			state.kong_on_top,
			lambda: self.consts.BOMB_SPAWN_TOP_LOCATION,
			lambda: self.consts.BOMB_SPAWN_BOTTOM_LOCATION
		)
		spawn_floor = jax.lax.cond(state.kong_on_top, lambda: 7, lambda: 0)
		opposite_floor = jax.lax.cond(state.kong_on_top, lambda: 0, lambda: 7)
		
		should_spawn = self._can_spawn_bomb(state, spawn_floor)
		
		# Find first inactive bomb slot
		first_inactive = jnp.argmax(state.bomb_active == 0)
		can_spawn = state.bomb_active[first_inactive] == 0
		spawn_mask = jnp.logical_and(should_spawn, can_spawn)
		
		# Initialize bomb properties
		key, bomb_key, magic_key = jax.random.split(state.rng_key, 3)
		
		initial_dir_x = jax.lax.cond(
			jax.random.bernoulli(bomb_key, p=0.5),
			lambda: -1,  # Left
			lambda: 1    # Right
		)
		initial_dir_y = 0  # Start with horizontal movement
		
		# 20% chance for magic bomb
		is_magic_bomb = jax.random.bernoulli(magic_key, p=0.2)
		
		# Spawn bomb if allowed
		new_bomb_active = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_active.at[first_inactive].set(1),
			lambda: state.bomb_active
		)
		new_bomb_x = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_positions_x.at[first_inactive].set(spawn_location[0]),
			lambda: state.bomb_positions_x
		)
		new_bomb_y = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_positions_y.at[first_inactive].set(spawn_location[1]),
			lambda: state.bomb_positions_y
		)
		new_bomb_dir_x = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_directions_x.at[first_inactive].set(initial_dir_x),
			lambda: state.bomb_directions_x
		)
		new_bomb_dir_y = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_directions_y.at[first_inactive].set(initial_dir_y),
			lambda: state.bomb_directions_y
		)
		new_bomb_floor = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_floor.at[first_inactive].set(spawn_floor),
			lambda: state.bomb_floor
		)
		new_bomb_is_magic = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_is_magic.at[first_inactive].set(jnp.where(is_magic_bomb, 1, 0)),
			lambda: state.bomb_is_magic
		)

		# Update bomb movement for all active bombs
		updated_bombs = self._update_all_bomb_movement(
			state, new_bomb_x, new_bomb_y, new_bomb_active, new_bomb_dir_x, new_bomb_dir_y, new_bomb_floor, new_bomb_is_magic, opposite_floor
		)
		
		new_bomb_x, new_bomb_y, new_bomb_active, new_bomb_dir_x, new_bomb_dir_y, new_bomb_floor = updated_bombs
		
		# Handle despawning - bombs despawn when they reach bounds on opposite floor
		should_despawn = self._check_bomb_despawn(
			new_bomb_x, new_bomb_floor, new_bomb_active, opposite_floor
		)
		
		new_bomb_active = jnp.where(should_despawn, 0, new_bomb_active)
		
		return state.replace(
			bomb_positions_x=new_bomb_x,
			bomb_positions_y=new_bomb_y,
			bomb_active=new_bomb_active,
			bomb_is_magic=new_bomb_is_magic,
			bomb_directions_x=new_bomb_dir_x,
			bomb_directions_y=new_bomb_dir_y,
			bomb_floor=new_bomb_floor,
			rng_key=key
		)	
		
	def _can_spawn_bomb(self, state: KingKongState, spawn_floor: int) -> chex.Array:
		def check_bomb_on_floor(bomb_active, bomb_floor):
			# Check if this active bomb is on the spawn floor
			return jnp.logical_and(bomb_active > 0, bomb_floor == spawn_floor)
		
		# Use vmap to check all bombs at once
		bombs_on_spawn_floor = jax.vmap(check_bomb_on_floor)(state.bomb_active, state.bomb_floor)
		
		# Check if any bomb is currently on the spawn floor
		any_bomb_on_spawn_floor = jnp.any(bombs_on_spawn_floor)
		
		# Can spawn if no bombs are on spawn floor
		return ~any_bomb_on_spawn_floor

	def _check_bomb_despawn(self, bomb_x, bomb_floor, bomb_active, opposite_floor):
		def check_single_bomb(x, floor, active):
			is_active = active > 0
			on_opposite_floor = floor == opposite_floor
			
			# Check if bomb went off-screen (beyond bounds + buffer)
			floor_bounds = self.consts.FLOOR_BOUNDS[opposite_floor]
			min_x, max_x = floor_bounds[0], floor_bounds[1]
			bomb_width = self.consts.BOMB_SIZE[0]
			buffer = self.consts.BOMB_OFFSCREEN_BUFFER
			
			# Allow bombs to go further off-screen before despawning
			off_left_edge = x < (min_x - buffer)
			off_right_edge = x > (max_x - bomb_width + buffer)
			offscreen = jnp.logical_or(off_left_edge, off_right_edge)
			
			should_despawn_this = jnp.logical_and(
				jnp.logical_and(is_active, on_opposite_floor),
				offscreen
			)
			
			return should_despawn_this
		
		# Apply the check to all bombs simultaneously
		return jax.vmap(check_single_bomb)(bomb_x, bomb_floor, bomb_active)

	def _get_kong_floor(self, state: KingKongState) -> chex.Array:
		floor_distances = jnp.abs(self.consts.FLOOR_LOCATIONS - state.kong_y)
		return jnp.argmin(floor_distances)

	def _update_all_bomb_movement(self, state, bomb_x, bomb_y, bomb_active, bomb_dir_x, bomb_dir_y, bomb_floor, bomb_magic, opposite_floor):
		def update_single_bomb(current_x, current_y, is_active, current_dir_x, current_dir_y, current_floor, current_magic):
			# Update bomb position and direction only if active
			new_x, new_y, new_dir_x, new_dir_y, new_floor = jax.lax.cond(
				is_active > 0,  # only update if active/spawned 
				lambda: self._move_single_bomb(state, current_x, current_y, current_dir_x, current_dir_y, current_floor, current_magic, opposite_floor),
				lambda: (current_x, current_y, current_dir_x, current_dir_y, current_floor)
			)
			
			return new_x, new_y, new_dir_x, new_dir_y, new_floor
		
		# Apply the update function to all bombs simultaneously
		new_x, new_y, new_dir_x, new_dir_y, new_floor = jax.vmap(update_single_bomb)(
			bomb_x, bomb_y, bomb_active, bomb_dir_x, bomb_dir_y, bomb_floor, bomb_magic
		)
		
		return new_x, new_y, bomb_active, new_dir_x, new_dir_y, new_floor

	def _move_single_bomb(self, state: KingKongState, x, y, dir_x, dir_y, floor, is_magic, opposite_floor):
		key1, key2, key3 = jax.random.split(state.rng_key, 3)
		
		# Calculate movement interval based on difficulty level
		# Higher levels = smaller intervals = faster movement		
		# Reduce interval by 0.5 frames per level and round it 
		# idk how exactly this is done in the original but seems to match 
		level_reduction = (state.level - 1) * 0.5
		movement_interval = jnp.maximum(
			jnp.int32(self.consts.BOMB_SPEED_BASE_INTERVAL - level_reduction),
			self.consts.BOMB_SPEED_MIN_INTERVAL
		)
		
		# Check if bomb should move this frame
		should_move = (state.stage_steps % movement_interval) == 0
		
		def do_movement():
			move_speed = 1

			# If Kong is at the top, go down otherwise go up 
			go_down = state.kong_on_top 
						
			bomb_width = self.consts.BOMB_SIZE[0]
			bomb_height = self.consts.BOMB_SIZE[1]

			# Check if bomb is perfectly on a ladder
			on_ladder_below = go_down & self._check_bomb_on_ladder_strict(x, y + bomb_height + 2)
			on_ladder_above = ~go_down & self._check_bomb_on_ladder_strict(x, y)
			on_ladder = on_ladder_below | on_ladder_above

			# Check if target floor has a bomb (prevents taking ladder/hole)
			target_floor = jax.lax.cond(go_down, lambda: jnp.maximum(floor - 1, 0), lambda: jnp.minimum(floor + 1, 8))
			target_floor_has_bomb = self._floor_has_bomb(state, target_floor)

			# If moving horizontally
			def do_horizontal_movement():
				# Take ladder if perfectly aligned AND target floor is clear
				can_take_ladder = jnp.logical_and(on_ladder, ~target_floor_has_bomb)
				should_take_ladder = jnp.logical_and(
					can_take_ladder,
					jax.random.bernoulli(key3, p=0.80) #80% chance
				)

				# Horizontal movement only if NOT taking ladder
				def horizontal_move():
					move_x = x + dir_x * move_speed

					left_bound = self.consts.FLOOR_BOUNDS[floor, 0] - (bomb_width // 2)
					right_bound = self.consts.FLOOR_BOUNDS[floor, 1] - (bomb_width // 2) - 2

					# Only bounce off walls if NOT on opposite floor
					is_opposite_floor = floor == opposite_floor
					
					# On opposite floor: just move normally (will despawn off-screen)
					# On other floors: bounce off walls
					hit_wall = jnp.logical_and(
						~is_opposite_floor,  # Only check walls if not on opposite floor
						jnp.logical_or(move_x <= left_bound, move_x >= right_bound)
					)
					
					final_dir_x = jax.lax.cond(hit_wall, lambda: -dir_x, lambda: dir_x)
					
					# On opposite floor: don't clip to bounds (allow off-screen movement)
					# On other floors: clip to bounds
					final_x = jax.lax.cond(
						is_opposite_floor,
						lambda: move_x,  # Don't clip on opposite floor
						lambda: jnp.clip(move_x, left_bound, right_bound)  # Clip on other floors
					)

					return final_x, final_dir_x
				
				final_x, final_dir_x = jax.lax.cond(
					should_take_ladder,
					lambda: (x, 0),  # No horizontal movement if taking ladder
					horizontal_move
				)

				# Determine vertical direction if taking ladder
				ladder_dir = jax.lax.cond(
					go_down,  # Upper floors go down
					lambda: 1,   # Positive = down
					lambda: -1   # Lower floors go up
				)
				final_dir_y = jax.lax.cond(should_take_ladder, lambda: ladder_dir, lambda: 0)

				current_floor_y = self.consts.FLOOR_LOCATIONS[floor]
				return final_x, current_floor_y, final_dir_x, final_dir_y, floor

			# If moving vertically (on ladder)
			def do_vertical_movement():
				# Move vertically
				move_y = y + dir_y * move_speed
				
				# Check if reached a floor
				def moving_down():
					target_floor = jnp.maximum(floor - 1, 0)
					target_y = self.consts.FLOOR_LOCATIONS[target_floor]
					reached = move_y >= target_y
					return target_floor, target_y, reached
					
				def moving_up():
					target_floor = jnp.minimum(floor + 1, 8)
					target_y = self.consts.FLOOR_LOCATIONS[target_floor]
					reached = move_y <= target_y
					return target_floor, target_y, reached
				
				target_floor, target_y, reached = jax.lax.cond(
					dir_y > 0,  # Moving down
					moving_down,
					moving_up
				)
				
				# If reached target floor, switch to horizontal
				def switch_to_horizontal():
					new_dir_x = jax.lax.cond(
						jax.random.bernoulli(key2, p=0.5), # random direction 
						lambda: -1,
						lambda: 1
					)
					return x, target_y, new_dir_x, 0, target_floor
				
				def continue_vertical():
					return x, move_y, 0, dir_y, floor
					
				return jax.lax.cond(reached, switch_to_horizontal, continue_vertical)
			
			# If falling through hole (special case)
			def do_falling_movement():
				fall_y = y + 2
				
				# Check if hit floor below
				target_floor = jnp.maximum(floor - 1, 0)
				target_y = self.consts.FLOOR_LOCATIONS[target_floor]
				hit_floor = fall_y >= target_y
				
				def hit_floor_action():
					# Switch to horizontal movement on new floor
					new_dir_x = jax.lax.cond(
						jax.random.bernoulli(key2, p=0.5), # random direction 
						lambda: -1,
						lambda: 1
					)
					return x, target_y, new_dir_x, 0, target_floor
				
				def continue_falling():
					return x, fall_y, dir_x, 1, floor  # Keep falling
				
				return jax.lax.cond(hit_floor, hit_floor_action, continue_falling)
			
			# Main decision logic remains the same
			is_moving_vertically = dir_y != 0
			is_falling = jnp.logical_and(dir_y > 0, ~on_ladder)
			
			# Execute appropriate movement
			new_x, new_y, new_dir_x, new_dir_y, new_floor = jax.lax.cond(
				is_falling,
				do_falling_movement,
				lambda: jax.lax.cond(
					jnp.logical_and(is_moving_vertically, on_ladder),
					do_vertical_movement,
					do_horizontal_movement
				)
			)
			
			# Handle hole falling (only when moving horizontally AND target floor is clear)
			over_hole = self._check_bomb_over_hole(new_x, new_y)
			can_fall_through_hole = jnp.logical_and(~target_floor_has_bomb, go_down)
			
			should_start_falling = jnp.logical_and(
				jnp.logical_and(
					jnp.logical_and(over_hole, new_dir_y == 0),  # Horizontal movement over hole
					can_fall_through_hole  # Can fall and target floor is clear
				),
				jax.random.bernoulli(key1, p=0.25)  # 25% chance
			)
			
			# Start falling through hole
			final_dir_y, final_floor = jax.lax.cond(
				should_start_falling,
				lambda: (1, new_floor),  # Start falling (dont change floor yet)
				lambda: (new_dir_y, new_floor)
			)
			
			return new_x, new_y, new_dir_x, final_dir_y, final_floor
		
		def no_movement():
			return x, y, dir_x, dir_y, floor
		
		# Only move if its the right frame
		return jax.lax.cond(should_move, do_movement, no_movement)
	
	def _floor_has_bomb(self, state: KingKongState, target_floor: int) -> chex.Array:
		def check_bomb_on_target_floor(bomb_active, bomb_floor):
			# Check if this active bomb is on target floor
			return jnp.logical_and(bomb_active > 0, bomb_floor == target_floor)
		
		# Use vmap to check all bombs at once
		bombs_on_target_floor = jax.vmap(check_bomb_on_target_floor)(state.bomb_active, state.bomb_floor)
		
		# Check if any bomb is on the target floor
		return jnp.any(bombs_on_target_floor)
		
	def _check_bomb_on_ladder_strict(self, bomb_x, bomb_y):
		size_x = self.consts.BOMB_SIZE[0]
		size_y = self.consts.BOMB_SIZE[1]

		ladder_x1 = self.consts.LADDER_LOCATIONS[1:, 0] # bombs cant take the ladder with index 0 
		ladder_y1 = self.consts.LADDER_LOCATIONS[1:, 1]
		ladder_x2 = self.consts.LADDER_LOCATIONS[1:, 2]
		ladder_y2 = self.consts.LADDER_LOCATIONS[1:, 3]

		bomb_x1 = bomb_x 
		bomb_x2 = bomb_x + size_x
		bomb_y1 = bomb_y 
		bomb_y2 = bomb_y - size_y

		on_ladders = jnp.logical_and(
			jnp.logical_and(bomb_x1 == ladder_x1, bomb_x2 == ladder_x2),
			jnp.logical_and(bomb_y1 >= ladder_y1, bomb_y2 <= ladder_y2)
		)

		return jnp.any(on_ladders)
	
	def _check_bomb_over_hole(self, bomb_x, bomb_y):
		size_x = self.consts.BOMB_SIZE[0]

		hole_rects = self.consts.HOLE_LOCATIONS
		
		# Use bomb center point for hole detection
		bomb_x1 = bomb_x + 2
		bomb_x2 = bomb_x + size_x - 3
		
		def point_in_hole(hole_rect):
			x1, y1, x2, y2 = hole_rect
			return jnp.logical_and(
				jnp.logical_and(bomb_x1 >= x1, bomb_x2 <= x2),
				jnp.logical_and(bomb_y >= y1, bomb_y <= y2)
			)
		
		collisions = jax.vmap(point_in_hole)(hole_rects)
		return jnp.any(collisions)

	def _rectangles_overlap(self, r1, r2):
		x1, y1, x2, y2 = r1
		a1, b1, a2, b2 = r2

		horizontal_overlap = (x1 <= a2) & (x2 >= a1)
		vertical_overlap = (y1 <= b2) & (y2 >= b1)

		return horizontal_overlap & vertical_overlap

	def _check_hole_collision(self, player_x, player_y):
		hole_rects = self.consts.HOLE_LOCATIONS

		def point_in_bb(hole_rect):
			x1, y1, x2, y2 = hole_rect
			collision = (player_x >= x1) & (player_x <= x2) & (player_y >= y1) & (player_y <= y2)

			return collision

		collisions = jax.vmap(point_in_bb)(hole_rects)
		return jnp.any(collisions)


	def _check_collisions(self, state: KingKongState) -> KingKongState:
		# Full player rectangle for jumping/scoring detection
		player_rect_full = jnp.array([
			state.player_x,
			state.player_y - self.consts.PLAYER_SIZE[1],
			state.player_x + self.consts.PLAYER_SIZE[0],
			state.player_y
		])
		
		# Player middle line for death detection (strict)
		player_rect_middle = jnp.array([
			state.player_x,
			state.player_y - self.consts.PLAYER_SIZE[1],
			state.player_x + 1,
			state.player_y
		])

		# Active bombs (gives bool array)
		bomb_active = state.bomb_active > 0

		def bomb_collision_lenient(bomb_xy, active):
			x, y = bomb_xy
			bomb_rect = jnp.array([x, y - self.consts.BOMB_SIZE[1], x + self.consts.BOMB_SIZE[0], y]) # full bomb
			return self._rectangles_overlap(player_rect_full, bomb_rect) & active

		def bomb_collision_strict(bomb_xy, active):
			# Bombs insets on all sides
			x, y = bomb_xy
			bomb_rect = jnp.array([x + 2, y - self.consts.BOMB_SIZE[1] + 4, x + self.consts.BOMB_SIZE[0] - 2, y]) # 4 for bc of Fuse
			return self._rectangles_overlap(player_rect_middle, bomb_rect) & active

		# Get both collision types
		bombs_xy = jnp.stack([state.bomb_positions_x, state.bomb_positions_y], axis=1)
		collisions_lenient = jax.vmap(bomb_collision_lenient)(bombs_xy, bomb_active) # For jumping/scoring (lenient - full rectangles)
		collisions_strict = jax.vmap(bomb_collision_strict)(bombs_xy, bomb_active) # For death (strict, with insets)

		#jax.debug.print("collisions_lenient={x}", x=collisions_lenient)
		#jax.debug.print("bombs_xy={x}", x=bombs_xy)
		#jax.debug.print("player_rect_full={x}", x=player_rect_full)
		#jax.debug.print("player_rect_middle={x}", x=player_rect_middle)

		is_jumping = ((state.player_state == self.consts.PLAYER_JUMP_LEFT) | 
					(state.player_state == self.consts.PLAYER_JUMP_RIGHT) |
					(state.player_state == self.consts.PLAYER_CATAPULT_LEFT) |
					(state.player_state == self.consts.PLAYER_CATAPULT_RIGHT))

		# Bombs jumped on - use LENIENT collision, only count if not already given points during this jump
		jumped_bombs = collisions_lenient & is_jumping & (state.bomb_points_given == 0)
		
		# Calculate points for bombs jumped over for the first time this jump
		points = jnp.where(state.bomb_is_magic > 0,
						self.consts.MAGIC_BOMB_POINTS,
						self.consts.REGULAR_BOMB_POINTS)
		new_score = state.score + jnp.sum(jumped_bombs * points)

		# Update bomb_points_given for bombs that were just jumped over
		new_bomb_points_given = jnp.where(jumped_bombs, 1, state.bomb_points_given)
		
		# Reset bomb_points_given when not jumping (landed)
		new_bomb_points_given = jnp.where(~is_jumping, 0, new_bomb_points_given)

		# Magic bomb catapult effect - use LENIENT collision for detection
		magic_bombs_jumped = jumped_bombs & (state.bomb_is_magic > 0)
		can_catapult = (state.player_floor < 7) & ((state.player_state == self.consts.PLAYER_JUMP_LEFT) | 
												(state.player_state == self.consts.PLAYER_JUMP_RIGHT))
		should_catapult = jnp.any(magic_bombs_jumped) & can_catapult

		if self.consts.DEBUG: jax.debug.print("should_catapult={x}", x=should_catapult)
		
		# Convert regular jump to catapult jump (extends current jump with different physics)
		new_player_state = jax.lax.cond(
			should_catapult,
			lambda: jax.lax.cond(
				state.player_state == self.consts.PLAYER_JUMP_LEFT,
				lambda: self.consts.PLAYER_CATAPULT_LEFT,
				lambda: self.consts.PLAYER_CATAPULT_RIGHT
			),
			lambda: state.player_state
		)

		# Bombs hit when not jumping - use STRICT collision for death
		hit_bombs = collisions_strict & (~is_jumping)
		death_type = jnp.where(jnp.any(hit_bombs), self.consts.DEATH_TYPE_BOMB_EXPLODE, state.death_type)

		# Check for falling through holes (only if not already dying)
		fell_through_hole = jax.lax.cond(
			state.death_type == self.consts.DEATH_TYPE_NONE,
			lambda: self._check_hole_collision(state.player_x, state.player_y),
			lambda: False
		)

		death_type = jax.lax.cond(
			fell_through_hole,
			lambda: self.consts.DEATH_TYPE_FALL,
			lambda: death_type
		)

		return state.replace(
			death_type=death_type,
			score=new_score,
			bomb_points_given=new_bomb_points_given,
			player_state=new_player_state
		)

		
	def _update_kong(self, state: KingKongState) -> KingKongState:
		at_floor_bottom = state.player_y == self.consts.FLOOR_LOCATIONS[state.player_floor]

		moved_down_previously = state.player_highest_floor >= self.consts.FLOOR_KONG_MOVE_DOWN

		# Only move down if on top, only move up if on bottom
		can_move_down = jnp.logical_and(state.player_floor >= self.consts.FLOOR_KONG_MOVE_DOWN, state.kong_on_top)
		can_move_up = jnp.logical_and(moved_down_previously, jnp.logical_and(state.player_floor <= self.consts.FLOOR_KONG_MOVE_UP, jnp.logical_not(state.kong_on_top)))

		# Determine if animation is in progress and direction
		animation_in_progress_down = state.kong_jump_counter > 0
		animation_in_progress_up = state.kong_jump_counter < 0
		animation_in_progress = animation_in_progress_down | animation_in_progress_up

		# Decide if we should start moving
		should_move = jnp.logical_or(
			jnp.logical_and((can_move_up | can_move_down), at_floor_bottom),
			animation_in_progress
		)

		# Initialize jump counter with direction if starting
		new_jump_counter = jax.lax.cond(
			state.kong_jump_counter == 0,
			lambda: jnp.where(can_move_up, -1, jnp.where(can_move_down, 1, 0)),
			lambda: state.kong_jump_counter
		)

		def do_zigzag_movement():
			total_jumps_per_cycle = self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE + self.consts.KONG_JUMPS_DOWN
			max_jump_counter = 4 * total_jumps_per_cycle
			should_continue = jnp.abs(new_jump_counter) <= max_jump_counter

			def step_jump():
				phase = jnp.abs(new_jump_counter) // total_jumps_per_cycle
				step_in_phase = jnp.abs(new_jump_counter) % total_jumps_per_cycle
				# NOTE I couldn't really reproduce the down moving animation so but its only visual so i dont think it matters that much  
				dir_lr = jnp.where(phase % 2 == 0, 1, -1) 
				dir_lr2 = jnp.where(new_jump_counter < 0, -dir_lr, dir_lr)

				# Vertical delta depends on direction (up = negative, down = positive)
				vert_delta = jnp.where(new_jump_counter < 0, 2, -2)
				vert_delta2 = -2

				def jump_step():
					return jax.lax.cond(
						step_in_phase < self.consts.KONG_JUMPS_UP,
						lambda: (state.kong_x + dir_lr, state.kong_y + vert_delta),
						lambda: jax.lax.cond(
							step_in_phase < self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE,
							lambda: (state.kong_x + dir_lr2, state.kong_y),
							lambda: (state.kong_x + dir_lr2, state.kong_y - vert_delta2)
						)
					)

				def teleport():
					return jax.lax.cond(
						new_jump_counter < 0,
						lambda: (
							self.consts.KONG_UPPER_TELEPORT_LOCATION[0],
							self.consts.KONG_UPPER_TELEPORT_LOCATION[1],
						),
						lambda: (
							self.consts.KONG_LOWER_TELEPORT_LOCATION[0],
							self.consts.KONG_LOWER_TELEPORT_LOCATION[1],
						)
					)

				new_x, new_y = jax.lax.cond(
					jnp.abs(new_jump_counter) == 2 * total_jumps_per_cycle,
					teleport,
					jump_step
				)

				# Increment counter keeping direction
				new_counter = jnp.where(new_jump_counter < 0, new_jump_counter - 1, new_jump_counter + 1)
				return new_x, new_y, new_counter

			new_x, new_y, final_jump_counter = jax.lax.cond(
				should_continue,
				step_jump,
				lambda: (state.kong_x, state.kong_y, 0)
			)

			# Update kong_on_top based on movement direction
			kong_on_top = jax.lax.cond(
				new_jump_counter > 0,  # moving down
				lambda: 0,
				lambda: jax.lax.cond(
					new_jump_counter < 0,  # moving up
					lambda: 1,
					lambda: state.kong_on_top,
				)
			)

			return state.replace(
				kong_x=new_x,
				kong_y=new_y,
				kong_jump_counter=final_jump_counter,
				kong_on_top=kong_on_top
			)

		def no_movement():
			return state.replace(kong_jump_counter=0)

		return jax.lax.cond(should_move, do_zigzag_movement, no_movement)

	def _step_death(self, state: KingKongState, action: chex.Array) -> KingKongState:
		return jax.lax.cond(
			state.death_type == self.consts.DEATH_TYPE_FALL,
			lambda: self._step_fall_death(state, action),
			lambda: self._step_bomb_explode_death(state, action)
		)
	
	def _step_fall_death(self, state: KingKongState, action: chex.Array) -> KingKongState:
		stage_duration = self.consts.DUR_FALL
		should_transition = state.stage_steps >= stage_duration

		def do_transition(_):
			return state.replace(
				gamestate=self.consts.GAMESTATE_RESPAWN,
				stage_steps=0,
				death_type=self.consts.DEATH_TYPE_NONE, #reset death type
				death_target_y=-1, # reset death target
				lives=state.lives - 1,  # Decrease life on death
			)

		def do_normal_step(_):
			# Get target floor or use stored value
			target_floor_y = jax.lax.cond(
				state.death_target_y == -1,
				lambda: self._get_floor_below(state),  # Calculate on first step
				lambda: state.death_target_y  # Use stored value from previous steps
			)
			
			# Check if player has already hit the target floor
			has_hit_floor = state.player_y >= target_floor_y
			
			# Only update fall physics every 4 steps
			should_update_fall = jnp.logical_and(
				state.stage_steps % 4 == 0,  # Every 4 steps
				~has_hit_floor  # And havent hit floor yet
			)
			
			# Calculate falling physics for hole deaths
			new_player_y = jax.lax.cond(
				should_update_fall,
				lambda: self._update_fall_physics(state, target_floor_y),
				lambda: state.player_y  # Dont move if not updating or hit floor
			)
			
			# Update has_hit_floor after potential Y movement
			final_has_hit_floor = new_player_y >= target_floor_y
			
			# Set player state based on fall progress
			new_player_state = jax.lax.cond(
				final_has_hit_floor,
				lambda: self.consts.PLAYER_DEAD,    # Hit floor = dead
				lambda: self.consts.PLAYER_FALL     # Still falling
			)

			if self.consts.DEBUG:
				jax.debug.print("fall_death: step={s}/{dur} player_state={ps} player_y={py} target_y={ty} hit_floor={hf} should_update={su}", 
					s=state.stage_steps, dur=stage_duration,
					ps=new_player_state, py=new_player_y, ty=target_floor_y, 
					hf=final_has_hit_floor, su=should_update_fall)

			return state.replace(
				stage_steps=state.stage_steps + 1,
				player_state=new_player_state,
				player_y=new_player_y,
				death_target_y=target_floor_y  # Store the target floor Y
			)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)

	def _step_bomb_explode_death(self, state: KingKongState, action: chex.Array) -> KingKongState:
		stage_duration = self.consts.DUR_BOMB_EXPLODE
		should_transition = state.stage_steps >= stage_duration

		def do_transition(_):
			# Determine where player should fall to
			current_floor = state.player_floor
			
			# If on floor 0, fall off screen
			# Otherwise, fall to floor below
			should_fall_off_screen = current_floor == 0
			
			def fall_off_screen():
				return state.replace(
					gamestate=self.consts.GAMESTATE_DEATH,
					death_type=self.consts.DEATH_TYPE_FALL,
					stage_steps=0,
					player_state=self.consts.PLAYER_FALL,
					death_target_y=self.consts.HEIGHT + 50, # Fall somehere off screen
				)
			
			def fall_to_floor_below():
				# Move to floor below
				new_floor = jnp.maximum(current_floor - 1, 0)  # Don't go below 0
				target_y = self.consts.FLOOR_LOCATIONS[new_floor]
				
				return state.replace(
					gamestate=self.consts.GAMESTATE_DEATH,
					death_type=self.consts.DEATH_TYPE_FALL, 
					stage_steps=0,
					player_state=self.consts.PLAYER_FALL,
					player_floor=new_floor,
					death_target_y=target_y,
					# Reset bombs
					bomb_positions_x=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_positions_y=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_active=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_is_magic=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_directions_x=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_directions_y=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_floor=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32),
					bomb_points_given=jnp.zeros(self.consts.MAX_BOMBS).astype(jnp.int32)
				)
			
			return jax.lax.cond(
				should_fall_off_screen,
				fall_off_screen,
				fall_to_floor_below
			)

		def do_normal_step(_):
			death_flash_counter = jnp.minimum(state.death_flash_counter + 1, self.consts.CNT_DEATH_FLASHES)
			return state.replace(
				stage_steps=state.stage_steps + 1,
				death_flash_counter=death_flash_counter
			)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)

	def _get_floor_below(self, state: KingKongState) -> chex.Array:
		# Find the first floor that is below (higher Y value) than the current player position
		floor_below_index = jnp.argmax(state.player_y >= self.consts.FLOOR_LOCATIONS) - 1
		
		# Clamp to valid floor range (0 to num_floors-1)
		floor_below_index = jnp.clip(floor_below_index, 0, len(self.consts.FLOOR_LOCATIONS) - 1)
		
		return self.consts.FLOOR_LOCATIONS[floor_below_index]


	def _update_fall_physics(self, state: KingKongState, max_y: int) -> chex.Array:
		fall_speed = 2  # pixels per step
		return jnp.minimum(state.player_y + fall_speed, max_y)

	def _step_success(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_SUCCESS
		
		def do_transition():
			# Increment difficulty level when princess is rescued successfully
			new_level = state.level + 1
			
			# Reset to idle state for next level
			return state.replace(
				gamestate=self.consts.GAMESTATE_RESPAWN,
				stage_steps=0,
				level=new_level
			)
		
		def do_normal_step():
			# Teleport player to success location on first frame
			player_x = jax.lax.cond(
				state.stage_steps == 0,
				lambda: self.consts.PLAYER_SUCCESS_LOCATION[0],
				lambda: state.player_x
			)
			
			player_y = jax.lax.cond(
				state.stage_steps == 0,
				lambda: self.consts.PLAYER_SUCCESS_LOCATION[1],
				lambda: state.player_y
			)
			
			# Teleport princess to success location on first frame
			princess_x = jax.lax.cond(
				state.stage_steps == 0,
				lambda: self.consts.PRINCESS_SUCCESS_LOCATION[0],
				lambda: state.princess_x
			)
			
			princess_y = jax.lax.cond(
				state.stage_steps == 0,
				lambda: self.consts.PRINCESS_SUCCESS_LOCATION[1],
				lambda: state.princess_y
			)
			
			# Set both to idle/waving states
			player_state = jax.lax.cond(
				state.stage_steps == 0,
				lambda: self.consts.PLAYER_IDLE_RIGHT,  # Player faces princess
				lambda: state.player_state
			)
			
			princess_waving = jax.lax.cond(
				state.stage_steps == 0,
				lambda: 0,  # Princess has arms open
				lambda: state.princess_waving
			)
			
			final_stage_steps = state.stage_steps + 1
			
			return state.replace(
				stage_steps=final_stage_steps,
				player_x=player_x,
				player_y=player_y,
				player_state=player_state,
				princess_x=princess_x,
				princess_y=princess_y,
				princess_waving=princess_waving
			)
		
		return jax.lax.cond(should_transition, do_transition, do_normal_step)


	def render(self, state: KingKongState) -> jnp.ndarray:
		return self.renderer.render(state)

	@partial(jax.jit, static_argnums=(0,))
	def _get_observation(self, state: KingKongState):
		player = EntityPosition(
			x=state.player_x,
			y=state.player_y,
			width=jnp.array(self.consts.PLAYER_SIZE[0]),
			height=jnp.array(self.consts.PLAYER_SIZE[1]),
			visible=jnp.array(1)  # Player is always visible
		)
		
		kong = EntityPosition(
			x=state.kong_x,
			y=state.kong_y,
			width=jnp.array(self.consts.KONG_SIZE[0]),
			height=jnp.array(self.consts.KONG_SIZE[1]),
			visible=state.kong_visible
		)
		
		princess = EntityPosition(
			x=state.princess_x,
			y=state.princess_y,
			width=jnp.array(self.consts.PRINCESS_SIZE[0]),
			height=jnp.array(self.consts.PRINCESS_SIZE[1]),
			visible=state.princess_visible
		)
		
		bombs = BombInfo(
			positions_x=state.bomb_positions_x,
			positions_y=state.bomb_positions_y,
			active=state.bomb_active,
			is_magic=state.bomb_is_magic,
			directions_x=state.bomb_directions_x,
			directions_y=state.bomb_directions_y
		)
		
		return KingKongObservation(
			player=player,
			kong=kong,
			princess=princess,
			bombs=bombs,
			score=state.score,
			lives=state.lives,
			bonus_timer=state.bonus_timer,
			level=state.level,
			gamestate=state.gamestate,
			player_floor=state.player_floor
		)

	@partial(jax.jit, static_argnums=(0,))
	def obs_to_flat_array(self, obs: KingKongObservation) -> jnp.ndarray:
		return jnp.concatenate([
			obs.player.x.flatten(),
			obs.player.y.flatten(),
			obs.player.width.flatten(),
			obs.player.height.flatten(),
			obs.player.visible.flatten(),
			obs.kong.x.flatten(),
			obs.kong.y.flatten(),
			obs.kong.width.flatten(),
			obs.kong.height.flatten(),
			obs.kong.visible.flatten(),
			obs.princess.x.flatten(),
			obs.princess.y.flatten(),
			obs.princess.width.flatten(),
			obs.princess.height.flatten(),
			obs.princess.visible.flatten(),
			obs.bombs.positions_x.flatten(),
			obs.bombs.positions_y.flatten(),
			obs.bombs.active.flatten(),
			obs.bombs.is_magic.flatten(),
			obs.bombs.directions_x.flatten(),
			obs.bombs.directions_y.flatten(),
			obs.score.flatten(),
			obs.lives.flatten(),
			obs.bonus_timer.flatten(),
			obs.level.flatten(),
			obs.gamestate.flatten(),
			obs.player_floor.flatten()
		])

	def action_space(self) -> spaces.Discrete:
		return spaces.Discrete(len(self.ACTION_SET))

	def observation_space(self) -> spaces:
		return spaces.Dict({
			"player": spaces.Dict({
				"x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"visible": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
			}),
			"kong": spaces.Dict({
				"x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"visible": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
			}),
			"princess": spaces.Dict({
				"x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"visible": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
			}),
			"bombs": spaces.Dict({
				"positions_x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(self.consts.MAX_BOMBS,), dtype=jnp.int32),
				"positions_y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(self.consts.MAX_BOMBS,), dtype=jnp.int32),
				"active": spaces.Box(low=0, high=1, shape=(self.consts.MAX_BOMBS,), dtype=jnp.int32),
				"is_magic": spaces.Box(low=0, high=1, shape=(self.consts.MAX_BOMBS,), dtype=jnp.int32),
				"directions_x": spaces.Box(low=-1, high=1, shape=(self.consts.MAX_BOMBS,), dtype=jnp.int32),
				"directions_y": spaces.Box(low=-1, high=1, shape=(self.consts.MAX_BOMBS,), dtype=jnp.int32),
			}),
			"score": spaces.Box(low=0, high=self.consts.MAX_SCORE, shape=(), dtype=jnp.int32),
			"lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
			"bonus_timer": spaces.Box(low=0, high=self.consts.BONUS_START, shape=(), dtype=jnp.int32),
			"level": spaces.Box(low=1, high=100, shape=(), dtype=jnp.int32),
			"gamestate": spaces.Box(low=0, high=5, shape=(), dtype=jnp.int32),
			"player_floor": spaces.Box(low=0, high=8, shape=(), dtype=jnp.int32),
		})

	def image_space(self) -> spaces.Box:
		return spaces.Box(
			low=0,
			high=255,
			shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
			dtype=jnp.uint8
		)

	@partial(jax.jit, static_argnums=(0,))
	def _get_info(self, state: KingKongState) -> KingKongInfo:
		return KingKongInfo(time=state.step_counter)

	@partial(jax.jit, static_argnums=(0,))
	def _get_reward(self, previous_state: KingKongState, state: KingKongState):
		return state.score - previous_state.score # could be changed to also account for bonus timer e.g. 

	@partial(jax.jit, static_argnums=(0,))
	def _get_done(self, state: KingKongState) -> bool:
		# If either no lives (or max score reached), it's done 
		return jnp.logical_or(
			jnp.less_equal(state.lives, 0), 
			jnp.greater_equal(state.score, self.consts.MAX_SCORE),
		)

class KingKongRenderer(JAXGameRenderer):
	def __init__(self, consts: KingKongConstants = None, config: render_utils.RendererConfig = None):
		self.consts = consts or KingKongConstants()
		super().__init__(self.consts)
		self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "kingkong")
		
		# Use injected config if provided, else default
		if config is None:
			self.config = render_utils.RendererConfig(
				game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
				channels=3,
				downscale=None
			)
		else:
			self.config = config
		self.jr = render_utils.JaxRenderingUtils(self.config)
		
		# 2. Start from (possibly modded) asset config provided via constants
		final_asset_config = list(self.consts.ASSET_CONFIG)
		
		# 3. Make one call to load and process all assets
		(
			self.PALETTE,
			self.SHAPE_MASKS,
			self.BACKGROUND,
			self.COLOR_TO_ID,
			self.FLIP_OFFSETS
		) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)
		
		# 4. Store key color IDs
		self.BLACK_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)
		self.BAR_COLOR_ID = self.COLOR_TO_ID.get((201, 92, 135), 0)
		
		# Store death flash color IDs in a JAX array
		# Convert JAX array to list of tuples for dictionary lookup
		death_flash_colors_np = np.asarray(self.consts.DEATH_FLASH_COLORS)
		self.DEATH_FLASH_COLOR_IDS = jnp.array([
			self.COLOR_TO_ID.get(tuple(color), 0) for color in death_flash_colors_np
		])
		
		# Store debug color IDs (always create them, even if not used)
		self.DEBUG_RED_ID = self.COLOR_TO_ID.get((255, 0, 0), 0)
		self.DEBUG_BLUE_ID = self.COLOR_TO_ID.get((0, 0, 255), 0)
		self.DEBUG_WHITE_ID = self.COLOR_TO_ID.get((255, 255, 255), 0)
		self.DEBUG_GREEN_ID = self.COLOR_TO_ID.get((0, 255, 0), 0)
		
		# 5. Create helper mappings to map old sprite names to new indices
		self._create_helper_mappings()
		
		# 6. Store Python int values for static arguments (extracted outside JIT)
		self.SCORE_X = int(self.consts.SCORE_LOCATION[0])
		self.SCORE_Y = int(self.consts.SCORE_LOCATION[1] - self.consts.NUMBER_SIZE[1])
		self.SCORE_SPACING = int(self.consts.NUMBER_SIZE[0] + self.consts.SCORE_SPACE_BETWEEN)
		self.TIMER_X = int(self.consts.TIMER_LOCATION[0])
		self.TIMER_Y = int(self.consts.TIMER_LOCATION[1] - self.consts.NUMBER_SIZE[1])
		self.TIMER_SPACING = int(self.consts.NUMBER_SIZE[0] + self.consts.TIMER_SPACE_BETWEEN)
		self.LIFE_X = int(self.consts.LIFE_LOCATION[0])
		self.LIFE_Y = int(self.consts.LIFE_LOCATION[1] - self.consts.PLAYER_SIZE[1])
		self.LIFE_SPACING = int(self.consts.LIFE_SPACE_BETWEEN)

	def _create_helper_mappings(self):
		"""Creates dicts to map old sprite names to new group indices/offsets."""
		self.sprite_indices = {
			'player_idle': 0, 'player_move1': 1, 'player_move2': 2,
			'player_dead': 3, 'player_jump': 4, 'player_fall': 5,
			'player_climb1': 6, 'player_climb2': 7,
			
			'bomb': 0, 'magic_bomb': 1,
			
			'princess_closed': 0, 'princess_open': 1,
		}
		
		self.group_offsets = {
			'player': self.FLIP_OFFSETS['player_group'],
			'bomb': self.FLIP_OFFSETS['bomb_group'],
			'princess': self.FLIP_OFFSETS['princess_group'],
		}

	@partial(jax.jit, static_argnums=(0,))
	def _get_player_sprite(self, state: KingKongState) -> Tuple[chex.Array, bool]:
		"""JIT-compatible helper to select the correct player mask and flip status."""
		
		player_masks = self.SHAPE_MASKS['player_group']
		idx = self.sprite_indices
		
		# --- Define sprite indices and flip booleans separately ---
		# Store as (sprite_idx, flip_h)
		# Convert to JAX integers for use in JAX conditionals
		SPRITE_IDLE = jnp.int32(idx['player_idle'])
		SPRITE_JUMP = jnp.int32(idx['player_jump'])
		SPRITE_FALL = jnp.int32(idx['player_fall'])
		SPRITE_DEAD = jnp.int32(idx['player_dead'])
		SPRITE_CLIMB1 = jnp.int32(idx['player_climb1'])
		SPRITE_CLIMB2 = jnp.int32(idx['player_climb2'])
		SPRITE_MOVE1 = jnp.int32(idx['player_move1'])
		SPRITE_MOVE2 = jnp.int32(idx['player_move2'])
		
		# --- State checks ---
		is_idle = (state.player_state == self.consts.PLAYER_IDLE_LEFT) | (state.player_state == self.consts.PLAYER_IDLE_RIGHT)
		is_jumping = (state.player_state == self.consts.PLAYER_JUMP_LEFT) | \
					 (state.player_state == self.consts.PLAYER_JUMP_RIGHT) | \
					 (state.player_state == self.consts.PLAYER_CATAPULT_LEFT) | \
					 (state.player_state == self.consts.PLAYER_CATAPULT_RIGHT)
		is_falling = state.player_state == self.consts.PLAYER_FALL
		is_dead = state.player_state == self.consts.PLAYER_DEAD
		is_climbing = (state.player_state == self.consts.PLAYER_CLIMB_UP) | (state.player_state == self.consts.PLAYER_CLIMB_DOWN)
		is_climb_idle = state.player_state == self.consts.PLAYER_CLIMB_IDLE
		is_moving = (state.player_state == self.consts.PLAYER_MOVE_LEFT) | (state.player_state == self.consts.PLAYER_MOVE_RIGHT)
		is_left_facing = (state.player_state == self.consts.PLAYER_IDLE_LEFT) | \
						 (state.player_state == self.consts.PLAYER_MOVE_LEFT) | \
						 (state.player_state == self.consts.PLAYER_JUMP_LEFT) | \
						 (state.player_state == self.consts.PLAYER_CATAPULT_LEFT)
		
		freeze = state.death_type != self.consts.DEATH_TYPE_NONE
		
		def walk_cycle():
			frame = ((state.stage_steps - 1) // 4) % 4
			
			# Select sprite index using jnp.take
			frame_indices = jnp.array([SPRITE_MOVE1, SPRITE_MOVE2, SPRITE_MOVE1, SPRITE_IDLE])
			sprite_idx = jnp.take(frame_indices, frame)
			
			# For freeze, always use idle
			final_idx = jax.lax.cond(freeze, lambda: SPRITE_IDLE, lambda: sprite_idx)
			
			return final_idx
		
		def climb_cycle():
			frame = ((state.stage_steps - 1) // 8) % 2
			# Select sprite index using jnp.take
			frame_indices = jnp.array([SPRITE_CLIMB1, SPRITE_CLIMB2])
			anim_idx = jnp.take(frame_indices, frame)
			return jax.lax.cond(freeze, lambda: SPRITE_CLIMB2, lambda: anim_idx)
		
		# Main selection logic - select sprite index
		sprite_idx = jax.lax.cond(
			is_jumping,
			lambda: SPRITE_JUMP,
			lambda: jax.lax.cond(
				is_falling, lambda: SPRITE_FALL,
				lambda: jax.lax.cond(
					is_dead, lambda: SPRITE_DEAD,
					lambda: jax.lax.cond(
						is_idle, lambda: SPRITE_IDLE,
						lambda: jax.lax.cond(
							is_climb_idle, lambda: SPRITE_CLIMB2,
							lambda: jax.lax.cond(
								is_climbing, climb_cycle,
								lambda: jax.lax.cond(
									is_moving, walk_cycle,
									lambda: SPRITE_IDLE # fallback
								)
							)
						)
					)
				)
			)
		)
		
		# Determine flip based on sprite and state
		# Flip for: idle_left, move_left, jump_left, catapult_left
		# Also flip idle sprite when frozen in left-facing movement
		flip_h = jnp.bool_(is_left_facing)
		
		# Get the actual mask using dynamic indexing
		# Use jnp.take or indexing - sprite_idx is a JAX tracerso we can't use Python indexing
		# We need to use jnp.take or a similar approach
		mask = player_masks[sprite_idx]
		
		return mask, flip_h

	@partial(jax.jit, static_argnums=(0,))
	def _render_debug_overlay(self, raster, state):
		
		def render_bbox_points(r, bbox, color_id):
			x1, y1, x2, y2 = bbox
			r = r.at[y1, x1].set(color_id)
			r = r.at[y1, x2].set(color_id)
			r = r.at[y2, x1].set(color_id)
			r = r.at[y2, x2].set(color_id)
			return r
		
		def render_all_bbox(r, locations, color_id):
			def body(i, r_in):
				return render_bbox_points(r_in, locations[i], color_id)
			return jax.lax.fori_loop(0, locations.shape[0], body, r)
		
		raster = render_all_bbox(raster, self.consts.LADDER_LOCATIONS, self.DEBUG_RED_ID)
		
		# Compute player bounding box
		player_width, player_height = self.consts.PLAYER_SIZE
		player_bbox = jnp.array([
			state.player_x,
			state.player_y - player_height,
			state.player_x + player_width,
			state.player_y
		])
		raster = render_bbox_points(raster, player_bbox, self.DEBUG_WHITE_ID)
		
		# Render player bounding box points
		raster = render_bbox_points(raster, player_bbox, self.DEBUG_WHITE_ID)
		
		def render_floor_points(r, floor_locations, color_id):
			def body(i, r_in):
				y = floor_locations[i]
				return r_in.at[y, 0].set(color_id)
			return jax.lax.fori_loop(0, floor_locations.shape[0], body, r)
		
		raster = render_floor_points(raster, self.consts.FLOOR_LOCATIONS, self.DEBUG_BLUE_ID)
		
		raster = render_all_bbox(raster, self.consts.HOLE_LOCATIONS, self.DEBUG_GREEN_ID)
		
		active_mask = state.bomb_active > 0
		def draw_bomb_box(i, r):
			def draw_box(r_in):
				x = state.bomb_positions_x[i]
				y = state.bomb_positions_y[i]
				w, h = self.consts.BOMB_SIZE
				bbox = jnp.array([x, y, x+w, y-h])
				return render_bbox_points(r_in, bbox, self.DEBUG_WHITE_ID)
			
			return jax.lax.cond(active_mask[i], draw_box, lambda r_in: r_in, r)
		
		raster = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, draw_bomb_box, raster)
		
		return raster

	@partial(jax.jit, static_argnums=(0,))
	def render(self, state: KingKongState) -> jnp.ndarray:
		
		# --- 1. Death Flash Base Raster --- 
		flash_idx = (state.stage_steps // self.consts.DUR_SINGLE_DEATH_FLASH) % self.consts.CNT_DEATH_FLASHES
		flash_color_id = self.DEATH_FLASH_COLOR_IDS[flash_idx]
		flash_bg = jnp.full_like(self.BACKGROUND, flash_color_id)
		
		is_bomb_explosion_death = state.death_type == self.consts.DEATH_TYPE_BOMB_EXPLODE
		raster = jax.lax.cond(
			is_bomb_explosion_death,
			lambda: flash_bg,
			lambda: self.BACKGROUND # Black background
		)
		
		# --- 2. Render Level ---
		raster = self.jr.render_at(
			raster, *self.consts.LEVEL_LOCATION, 
			self.SHAPE_MASKS['level'], 
			flip_offset=self.FLIP_OFFSETS['level']
		)
		
		# --- 3. Render Active Bombs ---
		def render_single_bomb(i, raster_in):
			# Get base sprite mask and offset
			bomb_type_idx = jax.lax.select(state.bomb_is_magic[i] > 0, 1, 0)
			base_mask = self.SHAPE_MASKS['bomb_group'][bomb_type_idx]
			base_offset = self.group_offsets['bomb']
			
			# Create deterministic key for this bomb and time step
			base_key = jax.random.key(42)
			bomb_key = jax.random.fold_in(base_key, i)
			period_key = jax.random.fold_in(bomb_key, state.stage_steps // 8)
			should_mirror = jax.random.bernoulli(period_key, p=0.5)
			
			# Calculate offsets
			base_offset_x = jax.lax.select(state.bomb_is_magic[i] > 0, 1, 0)
			mirror_offset_x = jax.lax.select(
				(should_mirror & (state.bomb_is_magic[i] > 0)), -2, 0
			)
			total_offset_x = base_offset_x + mirror_offset_x
			
			draw_fn = lambda r: self.jr.render_at(
				r,
				state.bomb_positions_x[i] + total_offset_x,
				state.bomb_positions_y[i] - self.consts.BOMB_SIZE[1],
				base_mask,
				flip_horizontal=should_mirror,
				flip_offset=base_offset
			)
			
			return jax.lax.cond(state.bomb_active[i] > 0, draw_fn, lambda r: r, raster_in)
		
		raster = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, render_single_bomb, raster)
					
		# --- 4. Render Player ---
		def render_player(raster_in):
			player_mask, flip_h = self._get_player_sprite(state)
			
			is_climbing = (state.player_state == self.consts.PLAYER_CLIMB_IDLE) | \
						  (state.player_state == self.consts.PLAYER_CLIMB_UP) | \
						  (state.player_state == self.consts.PLAYER_CLIMB_DOWN)
			is_fall_or_dead = (state.player_state == self.consts.PLAYER_FALL) | \
							  (state.player_state == self.consts.PLAYER_DEAD)
			is_right_facing = (state.player_state == self.consts.PLAYER_IDLE_RIGHT) | \
							  (state.player_state == self.consts.PLAYER_MOVE_RIGHT) | \
							  (state.player_state == self.consts.PLAYER_JUMP_RIGHT) | \
							  (state.player_state == self.consts.PLAYER_CATAPULT_RIGHT)
			
			x_offset = jnp.where(is_climbing, -3,
					   jnp.where(is_fall_or_dead, -2,
					   jnp.where(is_right_facing, -2, -5)))
			return self.jr.render_at(
				raster_in,
				state.player_x + x_offset,
				state.player_y - self.consts.PLAYER_SIZE[1],
				player_mask,
				flip_horizontal=flip_h,
				flip_offset=self.group_offsets['player']
			)
				
		raster = jax.lax.cond(
			(state.gamestate == self.consts.GAMESTATE_GAMEPLAY) |
			(state.gamestate == self.consts.GAMESTATE_DEATH) |
			(state.gamestate == self.consts.GAMESTATE_SUCCESS),
			render_player,
			lambda r: r,
			raster
		)
		
		# --- 5. Render Kong ---
		def render_kong(raster_in):
			return self.jr.render_at(
				raster_in,
				state.kong_x,
				state.kong_y - self.consts.KONG_SIZE[1],
				self.SHAPE_MASKS['kong'],
				flip_offset=self.FLIP_OFFSETS['kong']
			)
		
		raster = jax.lax.cond(state.kong_visible != 0, render_kong, lambda r: r, raster)
		
		# --- 6. Render UI Overlay ---
		def render_overlay(raster_in):
			width = self.consts.WIDTH - 48
			box_height = self.consts.PLAYER_SIZE[1] * 3
			
			# Draw pink bar
			r_out = self.jr.draw_rects(
				raster_in, 
				jnp.array([[24, 39]]), 
				jnp.array([[width, 1]]), 
				self.BAR_COLOR_ID
			)
			# Draw black box
			r_out = self.jr.draw_rects(
				r_out, 
				jnp.array([[24, 39 - box_height]]), 
				jnp.array([[width, box_height]]), 
				self.BLACK_ID
			)
			return r_out
		
		raster = jax.lax.cond(
			(state.gamestate != self.consts.GAMESTATE_SUCCESS) & (state.death_type != self.consts.DEATH_TYPE_BOMB_EXPLODE),
			render_overlay,
			lambda r: r,
			raster
		)
		
		# --- 7. Render Princess ---
		def render_princess(raster_in):
			princess_idx = jax.lax.select(state.princess_waving, 0, 1) # 0=closed, 1=open
			x_offset = jax.lax.select(state.princess_waving, 0, -1) # original logic had offset of 1, but render_at_left
			
			return self.jr.render_at(
				raster_in,
				state.princess_x + x_offset,
				state.princess_y - self.consts.PRINCESS_SIZE[1],
				self.SHAPE_MASKS['princess_group'][princess_idx],
				flip_offset=self.group_offsets['princess']
			)
		
		raster = jax.lax.cond(state.princess_visible != 0, render_princess, lambda r: r, raster)
		
		# --- 8. Render Lives ---
		def render_lives(raster_in):
			return self.jr.render_indicator(
				raster_in,
				self.LIFE_X,
				self.LIFE_Y,
				state.lives,
				self.SHAPE_MASKS['life'],
				spacing=self.LIFE_SPACING,
				max_value=self.consts.MAX_LIVES
			)
		
		raster = jax.lax.cond(
			state.gamestate == self.consts.GAMESTATE_RESPAWN,
			render_lives,
			lambda r: r,
			raster
		) 
		
		# --- 9. Render Score ---
		def render_score(raster_in):
			score_digits = self.jr.int_to_digits(state.score, max_digits=4)
			return self.jr.render_label(
				raster_in,
				self.SCORE_X,
				self.SCORE_Y,
				score_digits,
				self.SHAPE_MASKS['digits'],
				spacing=self.SCORE_SPACING,
				max_digits=4
			)
		
		raster = render_score(raster)
		
		# --- 10. Render Bonus Timer ---
		def render_bonus_timer(raster_in):
			timer_digits = self.jr.int_to_digits(state.bonus_timer, max_digits=3)
			return self.jr.render_label(
				raster_in,
				self.TIMER_X,
				self.TIMER_Y,
				timer_digits,
				self.SHAPE_MASKS['digits'],
				spacing=self.TIMER_SPACING,
				max_digits=3
			)
		
		raster = render_bonus_timer(raster)
		
		# --- 11. Render Debug Info ---
		raster = jax.lax.cond(
			self.consts.DEBUG_RENDER,
			lambda r: self._render_debug_overlay(r, state),
			lambda r: r,
			raster
		)
				
		# --- 12. Final Palette Lookup ---
		return self.jr.render_from_palette(raster, self.PALETTE)
