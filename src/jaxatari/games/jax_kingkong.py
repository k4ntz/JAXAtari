import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class KingKongConstants(NamedTuple):
	### Screen
	WIDTH: int = 160
	HEIGHT: int = 250
	FPS: int = 30 # Can this be read dynamically? 
	DEBUG_RENDER: int = 1 # Render some debug helpers

	### Sizes
	PLAYER_SIZE: chex.Array = jnp.array([5, 16])
	KONG_SIZE: chex.Array = jnp.array([14, 34])
	PRINCESS_SIZE: chex.Array = jnp.array([8, 17])
	BOMB_SIZE: chex.Array = jnp.array([8, 14])
	NUMBER_SIZE: chex.Array = jnp.array([12, 14])
	
	### Locations & Bounds

	# Player 
	PLAYER_RESPAWN_LOCATION: chex.Array = jnp.array([77, 228])
	PLAYER_SUCCESS_LOCATION: chex.Array = jnp.array([87, 37])
	
	# Level 
	LEVEL_LOCATION: chex.Array = jnp.array([8, 39])
	
	# bounding boxes (x1, y1, x2, y2) - x1,y1 is top left 
	HOLE_LOCATIONS: chex.Array = jnp.array([
		[52, 83, 55, 84],  
		[104, 83, 107, 84],
		[40, 131, 43, 132],
		[116, 131, 119, 132],
		[56, 179, 59, 180],
		[100, 179, 103, 180]
	])
	LADDER_LOCATIONS: chex.Array = jnp.array([
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
	])
	FLOOR_BOUNDS: chex.Array = jnp.array([
		[12, 150], # Ground floor 
		[12, 150], # First floor
		[12, 150], # Second floor 
		[12, 150], # Third floor 
		[12, 150], # Fourth floor 
		[12, 150], # Fifth floor 
		[16, 150], # Sixth floor
		[20, 142], # Seventh floor 
		[12, 150], # Princess floor - no bounds required bc goal reached 
	]) # floor bounds by floor (min_x, min_y) - y is always the same (see FLOOR_LOCATIONS)
	FLOOR_LOCATIONS: chex.Array = jnp.array([228, 204, 180, 156, 132, 108, 84, 60, 40, 0]) # y corrdinate, 0 for topmost floor calculation reuqired

	PRINCESS_MOVEMENT_BOUNDS: chex.Array = jnp.array([77, 113])

	LIFE_LOCATION: chex.Array = jnp.array([31, 228])
	LIFE_SPACE_BETWEEN: int = 11 

	SCORE_LOCATION: chex.Array = jnp.array([16, 35])
	SCORE_SPACE_BETWEEN: int = 4

	TIMER_LOCATION: chex.Array = jnp.array([112, 35])
	TIMER_SPACE_BETWEEN: int = 4 

	# Entities 
	KONG_START_LOCATION: chex.Array = jnp.array([31, 228])
	KONG_LOWER_LOCATION: chex.Array = KONG_START_LOCATION
	KONG_UPPER_LOCATION: chex.Array = jnp.array([31, 84])
	PRINCESS_START_LOCATION: chex.Array = jnp.array([93, 37])
	PRINCESS_RESPAWN_LOCATION: chex.Array = jnp.array([77, 37]) # at the start the princess teleports to the left, this al 
	PRINCESS_SUCCESS_LOCATION: chex.Array = jnp.array([95, 37]) 
	BOMB_SPAWN_TOP_LOCATION: chex.Array = jnp.array([20, 60]) #top left top floor
	BOMB_SPAWN_BOTTOM_LOCATION: chex.Array = KONG_LOWER_LOCATION  # Kong's lower location
	BOMB_OFFSCREEN_BUFFER: int = 4 # 4 additional pixels before despawn

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
	GAMESTATE_IDLE: int = 0 
	GAMESTATE_STARTUP: int = 1
	GAMESTATE_RESPAWN: int = 2 
	GAMESTATE_GAMEPLAY: int = 3
	GAMESTATE_DEATH: int = 4 
	GAMESTATE_SUCCESS: int = 5  

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
	DUR_IDLE: int = 130 
	SEQ_IDLE_KINGKONG_SPAWN: int = 0 # as soon as this stage is reached kong spawns 

	###################################################################
	# Define the startup stage 
	DUR_STARTUP: int = 255 
	SEQ_STARTUP_PRINCESS_SPAWN: int = 226 # Here the princess spawns in at PRINCESS_START_LOCATION  
	
	# First do 15 diagonal up jumps, then one to the left/right, then 3 diagonal down 
	KONG_TOTAL_JUMPS: int = 6
	KONG_JUMPS_UP: int = 15
	KONG_JUMPS_SIDE: int = 1 
	KONG_JUMPS_DOWN: int = 3 
	###################################################################

	###################################################################
	# Define the respawn stage 
	# Here the princess teleports to PRINCESS_RESPAWN_LOCATION 
	# The princess moves around at the top seemingly randomly (either wait, left or right) within her bounds
	# but she gets teleported back three times (the thrid time on the first frame of gameplay). 
	DUR_RESPAWN: int = 192 
	SEQ_RESPAWN_KONG_TELEPORT: int = 0 # also teleport kong to the top again if it was at the bottom 
	SEQ_RESPAWN_PRINCESS_TELEPORT0: int = 0 # every 64 frame she is tp'd back in this stage for some reason 
	SEQ_RESPAWN_PRINCESS_TELEPORT1: int = 64 
	SEQ_RESPAWN_PRINCESS_TELEPORT2: int = 128 
	###################################################################

	###################################################################
	# Define gameplay stage
	# unused bc we have bonus_timer but could technically replace with this 
	# DUR_GAMEPLAY: int = 99 * FPS # 990 / 10 = 99 seconds, in steps 
	SEQ_GAMEPLAY_PRINCESS_TELEPORT: int = 0  
	###################################################################

	###################################################################
	# Define the death stages
	
	# Path 1: Bomb explode  
	DUR_BOMB_EXPLODE: int = 96
	SEQ_BOMB_EXPLODE_DEATH_FLASHES: int = 0 
	SEQ_DEATH_FLASHES: int = 0 
	CNT_DEATH_FLASHES: int = 24 
	DUR_SINGLE_DEATH_FLASH: int = 4 # How long a death flash takes 
	DEATH_FLASH_COLORS = jnp.array([
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
	], dtype=jnp.uint8)

	assert(DUR_SINGLE_DEATH_FLASH * CNT_DEATH_FLASHES == DUR_BOMB_EXPLODE) 

	# Death types
	DEATH_TYPE_NONE = 0
	DEATH_TYPE_BOMB_EXPLODE = 1
	DEATH_TYPE_FALL = 2 

	# Path 2: Fall 
	DUR_FALL: int = 232 # During this time the player falls to the floor below. First fall, then show the blob, no step restriction  
	###################################################################

	###################################################################
	# Define the success stage q	
	DUR_SUCCESS: int = 232 # TODO couldn't test exact frames yet , assuming ~ death time 
	###################################################################
		
	### Game logic constants 
	BONUS_START: int = 990
	BONUS_DECREMENT: int = 10 # per second 

	FLOOR_KONG_MOVE_DOWN: int = 4 # fifth floor (start count at 0)
	FLOOR_KONG_MOVE_UP: int = 2 # TOOD

	PRINCESS_MOVE_OPTIONS: chex.Array = jnp.array([0, 0, 0, 0, 3, -3, 6, -6]) 

	REGULAR_BOMB_POINTS: int = 25 
	MAGIC_BOMB_POINTS: int = 125 

	BOMB_SPEED_BASE_INTERVAL: int = 2
	BOMB_SPEED_MIN_INTERVAL: int = 1

	PLAYER_JUMP_HEIGHT: int = 12 
	PLAYER_CATAPULT_HEIGHT: int = 40 

	MAX_BOMBS: int = 8 #one for each floor 
	MAX_SPEED: int = 1
	MAX_LIVES: int = 3
	MAX_SCORE: int = 999_999 # this is basically unachievable 

	# Player states
	PLAYER_IDLE_LEFT = 1
	PLAYER_IDLE_RIGHT = 2
	PLAYER_MOVE_LEFT = 3
	PLAYER_MOVE_RIGHT = 4
	PLAYER_JUMP_LEFT = 5 
	PLAYER_JUMP_RIGHT = 6
	PLAYER_CLIMB_UP = 7
	PLAYER_CLIMB_DOWN = 8
	PLAYER_CLIMB_IDLE = 42 
	PLAYER_FALL = 9
	PLAYER_DEAD = 10
	PLAYER_GOAL = 11
	PLAYER_CATAPULT_LEFT = 44
	PLAYER_CATAPULT_RIGHT = 45 

class KingKongState(NamedTuple):
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
	player_jump_counter: chex.Array  # For jump animation
	player_dir: chex.Array  # (-1, 0, or 1)

	# Kong state
	kong_x: chex.Array
	kong_y: chex.Array
	kong_visible: chex.Array
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

class KingKongObservation(NamedTuple):
	pass 
class KingKongInfo(NamedTuple):
	time: chex.Array
	all_rewards: chex.Array

class JaxKingKong(JaxEnvironment[KingKongState, KingKongObservation, KingKongInfo, KingKongConstants]):
	def __init__(self, consts: KingKongConstants = None, reward_funcs: list[callable]=None):
		consts = consts or KingKongConstants()
		super().__init__(consts)
		self.renderer = KingKongRenderer(self.consts)
		if reward_funcs is not None:
			reward_funcs = tuple(reward_funcs)
		self.reward_funcs = reward_funcs
		self.action_set = [
			Action.NOOP,
			Action.LEFT,
			Action.RIGHT,
			Action.UP,
			Action.DOWN,
			Action.FIRE, # Fire is jump 
			Action.LEFTFIRE, # running jumps, this is missing in the reference
			Action.RIGHTFIRE
		]
		self.obs_size = 5 + 3 * 5 + 1 + 1 #TODO

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
			player_jump_counter=jnp.array(0).astype(jnp.int32),
			player_dir=jnp.array(0).astype(jnp.int32),
			
			# Kong state
			kong_x=jnp.array(self.consts.KONG_LOWER_LOCATION[0]).astype(jnp.int32),
			kong_y=jnp.array(self.consts.KONG_LOWER_LOCATION[1]).astype(jnp.int32),
			kong_visible=jnp.array(1).astype(jnp.int32),
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
		# Handle different game states with current stage_steps
		jax.debug.print("gamestate={g} stage_steps={s}", g=state.gamestate, s=state.stage_steps)
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
			state, action
		)

		# Update princess movement
		new_state = jax.lax.cond(
			jnp.logical_and(new_state.princess_visible != 0, state.gamestate != self.consts.GAMESTATE_SUCCESS), # dont move on success state 
			self._update_princess_movement,
			lambda s: s,
			new_state
		)
		
		# Update global step counter
		new_state = new_state._replace(step_counter=state.step_counter + 1)
		
		reward = self._get_reward(state, new_state)
		done = self._get_done(new_state)
		info = self._get_info(new_state)
		observation = self._get_observation(new_state)
		
		return observation, new_state, reward, done, info
	
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
	
			jax.debug.print("princess_move: x={x} dx={dx} waving={w} waving_counter={wc}", x=state.princess_x, dx=dx, w=waving, wc=new_waving_counter)

			return state._replace(
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
		
		return state._replace(
			gamestate=new_gamestate,
			kong_visible=kong_visible,
			kong_x=kong_x,
			kong_y=kong_y,
			stage_steps=final_stage_steps
		)
	
	def _step_startup(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_STARTUP

		def do_transition(_):
			return state._replace(
				gamestate=self.consts.GAMESTATE_RESPAWN,
				stage_steps=0,
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

			return state._replace(
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
			# Determine which full zigzag phase we are in
			phase = state.kong_jump_counter // total_jumps_per_cycle
			dir_lr = jnp.where(phase % 2 == 0, 1, -1)  # even phase: right, odd: left
			step_in_phase = state.kong_jump_counter % total_jumps_per_cycle
			
			new_x, new_y = jax.lax.cond(
				step_in_phase < self.consts.KONG_JUMPS_UP,  # diagonal up
				lambda _: (state.kong_x + dir_lr, state.kong_y - 2),
				lambda _: jax.lax.cond(
					step_in_phase < self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE,  # side step
					lambda _: (state.kong_x + dir_lr, state.kong_y),
					lambda _: (state.kong_x + dir_lr, state.kong_y + 2),  # diagonal down
					operand=None
				),
				operand=None
			)

			return new_x, new_y, state.kong_jump_counter + 1
		
		return jax.lax.cond(
			should_continue,
			do_jump,
			lambda _: (state.kong_x, state.kong_y, state.kong_jump_counter),  # Don't increment counter when stopped
			operand=state
		)
		
	def _step_respawn(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_RESPAWN

		def do_transition(_):
			return state._replace(
				gamestate=self.consts.GAMESTATE_GAMEPLAY,
				stage_steps=0,
				player_x=self.consts.PLAYER_RESPAWN_LOCATION[0],
				player_y=self.consts.PLAYER_RESPAWN_LOCATION[1],
				player_floor=0,
				kong_x=self.consts.KONG_UPPER_LOCATION[0],
				kong_y=self.consts.KONG_UPPER_LOCATION[1],
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

			return state._replace(
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

			return state._replace(
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

			# bonus timer ticks once per second
			new_bonus_timer = jax.lax.cond(
				jnp.logical_and(jnp.logical_and(state.stage_steps != 0, state.stage_steps % self.consts.FPS == 0), state.bonus_timer > 0),
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

			new_state = new_state._replace(
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

			return new_state._replace(stage_steps=final_stage_steps)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)

	def _update_player_gameplay(self, state: KingKongState, action: chex.Array) -> KingKongState:
		# Intentions
		move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
		move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
		move_up = action == Action.UP
		move_down = action == Action.DOWN
		jump = (action == Action.FIRE) | (action == Action.LEFTFIRE) | (action == Action.RIGHTFIRE)

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
		
		# Initialize jump - only use direction if LEFTFIRE or RIGHTFIRE is pressed
		new_player_state = jax.lax.cond(
			can_jump,
			lambda: jax.lax.cond(
				action == Action.LEFTFIRE,
				lambda: self.consts.PLAYER_JUMP_LEFT,
				lambda: jax.lax.cond(
					action == Action.RIGHTFIRE,
					lambda: self.consts.PLAYER_JUMP_RIGHT,
					lambda: jax.lax.cond(
						state.player_state == self.consts.PLAYER_IDLE_LEFT,
						lambda: self.consts.PLAYER_JUMP_LEFT,
						lambda: self.consts.PLAYER_JUMP_RIGHT  # Default to right if idle right or other states
					)
				)
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

		# Set jump direction based on action when starting a jump
		new_player_dir = jax.lax.cond(
			can_jump,
			lambda: jax.lax.cond(
				action == Action.LEFTFIRE,
				lambda: -1,
				lambda: jax.lax.cond(
					action == Action.RIGHTFIRE,
					lambda: 1,
					lambda: 0  # Vertical jump for regular FIRE
				)
			),
			lambda: new_player_dir  # Keep current direction if not jumping
		)

		new_player_jump_counter = jax.lax.cond(
			can_jump,
			lambda: 0,  # Reset jump counter when starting new jump
			lambda: new_player_jump_counter
		)
		
		# Determine the new floor based on new_player_y
		new_player_floor = jnp.argmin(new_player_y <= self.consts.FLOOR_LOCATIONS) - 1

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

		jax.debug.print("player: x={x} y={y} floor={f} state={st} jump_counter={jc} jump_dx={jdx}, at_floor_bottom={fb}, is_climbing_idle={cbmi}",
			fb=at_floor_bottom, cbmi=is_climbing_idle,
			x=new_player_x, y=new_player_y, f=new_player_floor, st=new_player_state, jc=new_player_jump_counter, jdx=new_player_dir)

		return state._replace(
			player_x=new_player_x,
			player_y=new_player_y,
			player_floor=new_player_floor,
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

		jax.debug.print(
			"check_on_ladder: player=({x},{y},{w},{h}), ladders=({x1},{y1},{x2},{y2}), result={res}",
			x=player_x, y=player_y, w=player_w, h=player_h,
			x1=ladder_x1, y1=ladder_y1, x2=ladder_x2, y2=ladder_y2,
			res=result
		)

		return result

	def _update_bombs(self, state: KingKongState) -> KingKongState:
		# Determine Kong's current position and spawn location
		kong_floor = self._get_kong_floor(state)
		kong_at_bottom = kong_floor <= self.consts.FLOOR_KONG_MOVE_DOWN
		
		spawn_location = jax.lax.cond(
			kong_at_bottom,
			lambda: self.consts.BOMB_SPAWN_BOTTOM_LOCATION,
			lambda: self.consts.BOMB_SPAWN_TOP_LOCATION
		)
		spawn_floor = jax.lax.cond(kong_at_bottom, lambda: 0, lambda: 7)
		opposite_floor = jax.lax.cond(kong_at_bottom, lambda: 7, lambda: 0)
		
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
		
		return state._replace(
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
		active_mask = state.bomb_active > 0
		
		def check_bomb_on_floor(i, any_on_floor):
			bomb_is_active = active_mask[i]
			bomb_floor = state.bomb_floor[i]
			
			# Check if this active bomb is on the spawn floor
			on_spawn_floor = jnp.logical_and(bomb_is_active, bomb_floor == spawn_floor)
			
			return jnp.logical_or(any_on_floor, on_spawn_floor)
		
		# Check if any bomb is currently on the spawn floor
		any_bomb_on_spawn_floor = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, check_bomb_on_floor, False)
		
		# Can spawn if no bombs are on spawn floor
		return ~any_bomb_on_spawn_floor

	def _check_bomb_despawn(self, bomb_x, bomb_floor, bomb_active, opposite_floor):
		should_despawn = jnp.zeros_like(bomb_active, dtype=bool)
		
		def check_single_bomb(i, despawn_mask):
			is_active = bomb_active[i] > 0
			on_opposite_floor = bomb_floor[i] == opposite_floor
			
			# Check if bomb went off-screen (beyond bounds + buffer)
			def check_offscreen():
				floor_bounds = self.consts.FLOOR_BOUNDS[opposite_floor]
				min_x, max_x = floor_bounds[0], floor_bounds[1]
				bomb_width = self.consts.BOMB_SIZE[0]
				buffer = self.consts.BOMB_OFFSCREEN_BUFFER
				
				# Allow bombs to go further off-screen before despawning
				off_left_edge = bomb_x[i] < (min_x - buffer)
				off_right_edge = bomb_x[i] > (max_x - bomb_width + buffer)
				
				return jnp.logical_or(off_left_edge, off_right_edge)
			
			should_despawn_this = jnp.logical_and(
				jnp.logical_and(is_active, on_opposite_floor),
				check_offscreen()
			)
			
			return despawn_mask.at[i].set(should_despawn_this)
		
		return jax.lax.fori_loop(0, self.consts.MAX_BOMBS, check_single_bomb, should_despawn)

	def _get_kong_floor(self, state: KingKongState) -> chex.Array:
		floor_distances = jnp.abs(self.consts.FLOOR_LOCATIONS - state.kong_y)
		return jnp.argmin(floor_distances)

	def _update_all_bomb_movement(self, state, bomb_x, bomb_y, bomb_active, bomb_dir_x, bomb_dir_y, bomb_floor, bomb_magic, opposite_floor):
		def update_single_bomb(i, carry):
			bomb_x, bomb_y, bomb_active, bomb_dir_x, bomb_dir_y, bomb_floor, bomb_magic, opposite_floor = carry
			
			# Only update if bomb is active
			is_active = bomb_active[i] > 0
			
			# Get current bomb state
			current_x = bomb_x[i]
			current_y = bomb_y[i]
			current_dir_x = bomb_dir_x[i]
			current_dir_y = bomb_dir_y[i]
			current_floor = bomb_floor[i]
			current_maigc = bomb_magic[i]
			
			# Update bomb position and direction
			new_x, new_y, new_dir_x, new_dir_y, new_floor = jax.lax.cond(
				is_active,#only update if active / spawned 
				lambda: self._move_single_bomb(state, current_x, current_y, current_dir_x, current_dir_y, current_floor, current_maigc, opposite_floor),
				lambda: (current_x, current_y, current_dir_x, current_dir_y, current_floor)
			)
			
			# Update arrays
			bomb_x = bomb_x.at[i].set(new_x)
			bomb_y = bomb_y.at[i].set(new_y)
			bomb_dir_x = bomb_dir_x.at[i].set(new_dir_x)
			bomb_dir_y = bomb_dir_y.at[i].set(new_dir_y)
			bomb_floor = bomb_floor.at[i].set(new_floor)
			
			return (bomb_x, bomb_y, bomb_active, bomb_dir_x, bomb_dir_y, bomb_floor, bomb_magic, opposite_floor)
		
		# Process all bombs
		carry = (bomb_x, bomb_y, bomb_active, bomb_dir_x, bomb_dir_y, bomb_floor, bomb_magic, opposite_floor)
		final_carry = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, update_single_bomb, carry)
		
		return final_carry[:6] # only relevant attribs

	def _move_single_bomb(self, state: KingKongState, x, y, dir_x, dir_y, floor, is_magic, opposite_floor):
		key1, key2, key3 = jax.random.split(state.rng_key, 3)
		
		# Calculate movement interval based on difficulty level
		# Higher levels = smaller intervals = faster movement		
		# Reduce interval by 0.5 frames per level (rounded)
		# idk how exactly this is done in the original but seems to match 
		level_reduction = (state.level - 1) * 0.5
		movement_interval = jnp.maximum(
			jnp.int32(self.consts.BOMB_SPEED_BASE_INTERVAL - level_reduction),
			self.consts.BOMB_SPEED_MIN_INTERVAL
		)
		
		# Check if bomb should move this frame
		should_move = (state.stage_steps % movement_interval) == 0
		
		def do_movement():
			# Basic movement speed
			move_speed = 1

			# If PLAYER floor less than 4, go down - otherwise go up 
			go_down = state.player_floor < 4
			
			# Get current floor Y position
			current_floor_y = self.consts.FLOOR_LOCATIONS[floor]
			
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
					jax.random.bernoulli(key3, p=0.75) #75% chance
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
				lambda: (1, new_floor),  # Start falling (don't change floor yet)
				lambda: (new_dir_y, new_floor)
			)
			
			return new_x, new_y, new_dir_x, final_dir_y, final_floor
		
		def no_movement():
			return x, y, dir_x, dir_y, floor
		
		# Only move if it's the right frame
		return jax.lax.cond(should_move, do_movement, no_movement)
	
	def _floor_has_bomb(self, state: KingKongState, target_floor: int) -> chex.Array:
		active_mask = state.bomb_active > 0
		
		def check_bomb_on_target_floor(i, any_on_target):
			bomb_is_active = active_mask[i]
			bomb_floor = state.bomb_floor[i]
			
			# Check if this active bomb is on target floor
			on_target_floor = jnp.logical_and(bomb_is_active, bomb_floor == target_floor)
			
			return jnp.logical_or(any_on_target, on_target_floor)
		
		return jax.lax.fori_loop(0, self.consts.MAX_BOMBS, check_bomb_on_target_floor, False)
		
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
		return jnp.logical_and(x1 < a2, jnp.logical_and(x2 > a1,
			jnp.logical_and(y1 < b2, y2 > b1)))

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
			state.player_y,
			state.player_x + self.consts.PLAYER_SIZE[0],
			state.player_y + self.consts.PLAYER_SIZE[1]
		])
		
		# Player middle line for death detection (strict)
		player_middle_y = state.player_y + (self.consts.PLAYER_SIZE[1] // 2)  # player_y + 8
		player_rect_middle = jnp.array([
			state.player_x,
			player_middle_y,
			state.player_x + self.consts.PLAYER_SIZE[0],
			player_middle_y + 1  # Just 1 pixel height at the middle
		])

		# Active bombs (gives bool array)
		bomb_active = state.bomb_active > 0

		# Collision for jumping/scoring (lenient - full rectangles)
		def bomb_collision_lenient(bomb_xy, active):
			x, y = bomb_xy
			bomb_rect = jnp.array([x, y, x + self.consts.BOMB_SIZE[0], y + self.consts.BOMB_SIZE[1]])
			return self._rectangles_overlap(player_rect_full, bomb_rect) & active

		# Collision for death (strict - middle of player + bomb insets)
		def bomb_collision_strict(bomb_xy, active):
			x, y = bomb_xy
			# Bomb with 2-pixel insets on all sides
			bomb_rect = jnp.array([
				x + 2,  # inset_x = 2
				y + 2,  # inset_y = 2  
				x + self.consts.BOMB_SIZE[0] - 2,  # width - 2*inset_x
				y + self.consts.BOMB_SIZE[1] - 2   # height - 2*inset_y
			])
			return self._rectangles_overlap(player_rect_middle, bomb_rect) & active

		# Get both collision types
		bombs_xy = jnp.stack([state.bomb_positions_x, state.bomb_positions_y], axis=1)
		collisions_lenient = jax.vmap(bomb_collision_lenient)(bombs_xy, bomb_active)  # For jumping/scoring
		collisions_strict = jax.vmap(bomb_collision_strict)(bombs_xy, bomb_active)    # For death

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

		jax.debug.print("should_catapult={x}", x=should_catapult)
		
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

		return state._replace(
			death_type=death_type,
			score=new_score,
			bomb_points_given=new_bomb_points_given,
			player_state=new_player_state
		)

		
	def _update_kong(self, state: KingKongState) -> KingKongState:
		# Move Kong down when player reaches threshold floor
		# TODO animate 
		new_kong_y = jax.lax.cond(
			state.player_floor >= self.consts.FLOOR_KONG_MOVE_DOWN,
			lambda: self.consts.KONG_LOWER_LOCATION[1],
			lambda: state.kong_y
		)

		new_kong_x = jax.lax.cond(
			state.player_floor >= self.consts.FLOOR_KONG_MOVE_DOWN,
			lambda: self.consts.KONG_LOWER_LOCATION[0],
			lambda: state.kong_x
		)

		return state._replace(
			kong_x=new_kong_x,
			kong_y=new_kong_y,
		)

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
			return state._replace(
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
				~has_hit_floor  # And haven't hit floor yet
			)
			
			# Calculate falling physics for hole deaths
			new_player_y = jax.lax.cond(
				should_update_fall,
				lambda: self._update_fall_physics(state, target_floor_y),
				lambda: state.player_y  # Don't move if not updating or hit floor
			)
			
			# Update has_hit_floor after potential Y movement
			final_has_hit_floor = new_player_y >= target_floor_y
			
			# Set player state based on fall progress
			new_player_state = jax.lax.cond(
				final_has_hit_floor,
				lambda: self.consts.PLAYER_DEAD,    # Hit floor = dead
				lambda: self.consts.PLAYER_FALL     # Still falling
			)

			jax.debug.print("fall_death: step={s}/{dur} player_state={ps} player_y={py} target_y={ty} hit_floor={hf} should_update={su}", 
							s=state.stage_steps, dur=stage_duration,
							ps=new_player_state, py=new_player_y, ty=target_floor_y, 
							hf=final_has_hit_floor, su=should_update_fall)

			return state._replace(
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
				return state._replace(
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
				
				return state._replace(
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
			return state._replace(
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
			return state._replace(
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
			
			return state._replace(
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
		return KingKongObservation()


	@partial(jax.jit, static_argnums=(0,))
	def obs_to_flat_array(self, obs: KingKongObservation) -> jnp.ndarray:
		return jnp.array([])

	def action_space(self) -> spaces.Discrete:
		return spaces.Discrete(len(self.action_set))

	def observation_space(self) -> spaces:
		return spaces.Dict({})

	def image_space(self) -> spaces.Box:
		return spaces.Box(
			low=0,
			high=255,
			shape=(self.consts.WIDTH, self.consts.HEIGHT, 3),
			dtype=jnp.uint8
		)

	@partial(jax.jit, static_argnums=(0,))
	def _get_info(self, state: KingKongState, all_rewards: chex.Array = None) -> KingKongInfo:
		return KingKongInfo(time=state.step_counter, all_rewards=all_rewards)

	@partial(jax.jit, static_argnums=(0,))
	def _get_reward(self, previous_state: KingKongState, state: KingKongState):
		return state.score - previous_state.score

	@partial(jax.jit, static_argnums=(0,))
	def _get_all_reward(self, previous_state: KingKongState, state: KingKongState):
		if self.reward_funcs is None:
			return jnp.zeros(1)
		rewards = jnp.array(
			[reward_func(previous_state, state) for reward_func in self.reward_funcs]
		)
		return rewards

	@partial(jax.jit, static_argnums=(0,))
	def _get_done(self, state: KingKongState) -> bool:
		# If either no lives (or max score reached), it's done 
		return jnp.logical_or(
			jnp.less_equal(state.lives, 0), 
			jnp.greater_equal(state.score, self.consts.MAX_SCORE),
		)

class KingKongRenderer(JAXGameRenderer):
	def __init__(self, consts: KingKongConstants = None):
		super().__init__()
		self.consts = consts
		(
			self.SPRITE_LEVEL,
			self.SPRITE_PLAYER_IDLE_RIGHT,
			self.SPRITE_PLAYER_IDLE_LEFT,
			self.SPRITE_PLAYER_MOVE1_RIGHT,
			self.SPRITE_PLAYER_MOVE1_LEFT,
			self.SPRITE_PLAYER_MOVE2_RIGHT,
			self.SPRITE_PLAYER_MOVE2_LEFT,
			self.SPRITE_PLAYER_DEAD,
			self.SPRITE_PLAYER_JUMP_LEFT,
			self.SPRITE_PLAYER_JUMP_RIGHT,
			self.SPRITE_PLAYER_FALL,
			self.SPRITE_PLAYER_CLIMB1,
			self.SPRITE_PLAYER_CLIMB2,
			self.SPRITE_KONG,
			self.SPRITE_LIFE,
			self.SPRITE_BOMB,
			self.SPRITE_MAGIC_BOMB,
			self.SPRITE_PRINCESS_CLOSED,
			self.SPRITE_PRINCESS_OPEN,
			self.SPRITE_NUMBERS
		) = self.load_sprites()

	def load_sprites(self):
		MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

		# Load sprites
		level = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/level.npy"))
		player_idle = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_idle.npy"))
		player_idle_left = player_idle[:, ::-1, :]
		player_idle_right = player_idle

		player_move1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_move1.npy"))
		player_move2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_move2.npy"))
		player_move1_left = player_move1[:, ::-1, :]
		player_move2_left = player_move2[:, ::-1, :]
		player_move1_right = player_move1
		player_move2_right = player_move2

		player_dead = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_dead.npy"))
		player_jump = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_jump.npy"))
		player_jump_left = player_jump[:, ::-1, :]
		player_jump_right = player_jump
		player_fall = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_fall.npy"))
		player_climb1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_climb1.npy"))
		player_climb2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_climb2.npy"))

		kong = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/kingkong_idle.npy"))
		bomb = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/bomb.npy"))
		magic_bomb = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/magic_bomb.npy"))
		princess_closed = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/princess_closed.npy"))
		princess_open = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/princess_open.npy"))

		numbers = {i: jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/kingkong/{i}.npy")) for i in range(10)}

		# Padding function
		def pad(sprite, target_h, target_w, side="right"):
			h, w, c = sprite.shape
			pad_top = pad_bottom = pad_left = pad_right = 0

			if "bottom" in side:
				pad_top = target_h - h
			else:
				pad_bottom = target_h - h

			if "left" in side:
				pad_left = target_w - w
			else:
				pad_right = target_w - w

			return jnp.pad(sprite, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

		# Compute target dimensions for player sprites
		all_player_sprites = [
			player_idle_right, player_idle_left,
			player_move1_right, player_move1_left,
			player_move2_right, player_move2_left,
			player_dead
		]
		player_target_height = max(max(sprite.shape[0] for sprite in all_player_sprites), 16)
		player_target_width  = max(max(sprite.shape[1] for sprite in all_player_sprites), 8)

		# Pad all player sprites
		player_idle_right  = pad(player_idle_right,  player_target_height, player_target_width)
		player_idle_left   = pad(player_idle_left,   player_target_height, player_target_width, side="left")
		player_move1_right = pad(player_move1_right, player_target_height, player_target_width)
		player_move1_left  = pad(player_move1_left,  player_target_height, player_target_width, side="left")
		player_move2_right = pad(player_move2_right, player_target_height, player_target_width)
		player_move2_left  = pad(player_move2_left,  player_target_height, player_target_width, side="left")
		player_dead        = pad(player_dead,        player_target_height, player_target_width, side="bottom-left")
		player_jump        = pad(player_jump,        player_target_height, player_target_width)
		player_fall        = pad(player_fall,        player_target_height, player_target_width)
		player_climb1      = pad(player_climb1,      player_target_height, player_target_width, side="left")
		player_climb2      = pad(player_climb2,      player_target_height, player_target_width, side="right")

		# Pad bomb sprites
		bomb_sprites = [bomb, magic_bomb]
		bomb_target_height = max(max(sprite.shape[0] for sprite in bomb_sprites), 14)
		bomb_target_width  = max(max(sprite.shape[1] for sprite in bomb_sprites), 8)
		bomb = pad(bomb, bomb_target_height, bomb_target_width)
		magic_bomb = pad(magic_bomb, bomb_target_height, bomb_target_width)

		# Pad princess sprites
		princess_sprites = [princess_closed, princess_open]
		princess_target_height = max(max(sprite.shape[0] for sprite in princess_sprites), 17)
		princess_target_width  = max(max(sprite.shape[1] for sprite in princess_sprites), 8)
		princess_closed = pad(princess_closed, princess_target_height, princess_target_width)
		princess_open   = pad(princess_open,   princess_target_height, princess_target_width)

		life = player_move2_right.copy()

		# Expand dimensions
		SPRITE_LEVEL             = jnp.expand_dims(level, axis=0)
		SPRITE_PLAYER_IDLE_RIGHT = jnp.expand_dims(player_idle_right, axis=0)
		SPRITE_PLAYER_IDLE_LEFT  = jnp.expand_dims(player_idle_left, axis=0)
		SPRITE_PLAYER_MOVE1_RIGHT = jnp.expand_dims(player_move1_right, axis=0)
		SPRITE_PLAYER_MOVE1_LEFT  = jnp.expand_dims(player_move1_left, axis=0)
		SPRITE_PLAYER_MOVE2_RIGHT = jnp.expand_dims(player_move2_right, axis=0)
		SPRITE_PLAYER_MOVE2_LEFT  = jnp.expand_dims(player_move2_left, axis=0)
		SPRITE_PLAYER_DEAD        = jnp.expand_dims(player_dead, axis=0)
		SPRITE_PLAYER_JUMP_LEFT   = jnp.expand_dims(player_jump_left, axis=0)
		SPRITE_PLAYER_JUMP_RIGHT  = jnp.expand_dims(player_jump_right, axis=0)
		SPRITE_PLAYER_FALL        = jnp.expand_dims(player_fall, axis=0)
		SPRITE_PLAYER_CLIMB1      = jnp.expand_dims(player_climb1, axis=0)
		SPRITE_PLAYER_CLIMB2      = jnp.expand_dims(player_climb2, axis=0)
		SPRITE_KONG               = jnp.expand_dims(kong, axis=0)
		SPRITE_LIFE               = jnp.expand_dims(life, axis=0)
		SPRITE_BOMB               = jnp.expand_dims(bomb, axis=0)
		SPRITE_MAGIC_BOMB         = jnp.expand_dims(magic_bomb, axis=0)
		SPRITE_PRINCESS_CLOSED    = jnp.expand_dims(princess_closed, axis=0)
		SPRITE_PRINCESS_OPEN      = jnp.expand_dims(princess_open, axis=0)
		SPRITE_NUMBERS            = jnp.stack([jnp.expand_dims(numbers[i], axis=0) for i in range(10)], axis=0)

		return (
			SPRITE_LEVEL, SPRITE_PLAYER_IDLE_RIGHT, SPRITE_PLAYER_IDLE_LEFT,
			SPRITE_PLAYER_MOVE1_RIGHT, SPRITE_PLAYER_MOVE1_LEFT,
			SPRITE_PLAYER_MOVE2_RIGHT, SPRITE_PLAYER_MOVE2_LEFT,
			SPRITE_PLAYER_DEAD, SPRITE_PLAYER_JUMP_LEFT, SPRITE_PLAYER_JUMP_RIGHT, SPRITE_PLAYER_FALL,
			SPRITE_PLAYER_CLIMB1, SPRITE_PLAYER_CLIMB2, SPRITE_KONG,
			SPRITE_LIFE, SPRITE_BOMB, SPRITE_MAGIC_BOMB, SPRITE_PRINCESS_CLOSED,
			SPRITE_PRINCESS_OPEN, SPRITE_NUMBERS
		)

	def render(self, state: KingKongState) -> jnp.ndarray:
		raster = jr.create_initial_frame(self.consts.WIDTH, self.consts.HEIGHT)

		# --- Death Flash --- 
		flash_idx = (state.stage_steps // self.consts.DUR_SINGLE_DEATH_FLASH) % self.consts.CNT_DEATH_FLASHES
		flash_color = self.consts.DEATH_FLASH_COLORS[flash_idx]
		flash_bg = jnp.full((self.consts.HEIGHT, self.consts.WIDTH, 3), flash_color, dtype=jnp.uint8)
		black_bg = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

		is_bomb_explosion_death = state.death_type == self.consts.DEATH_TYPE_BOMB_EXPLODE
		raster = jax.lax.cond(
			is_bomb_explosion_death,
			lambda: flash_bg,
			lambda: black_bg
		)

		# Render level 
		frame_level = jr.get_sprite_frame(self.SPRITE_LEVEL, 0)
		raster = jr.render_at(raster, *self.consts.LEVEL_LOCATION, frame_level)
					
		# Render player based on state
		def get_player_sprite():
			def is_idle_state():
				return jnp.logical_or(
					state.player_state == self.consts.PLAYER_IDLE_LEFT,
					state.player_state == self.consts.PLAYER_IDLE_RIGHT
				)

			def is_jumping_state():
				return (state.player_state == self.consts.PLAYER_JUMP_LEFT) | \
					(state.player_state == self.consts.PLAYER_JUMP_RIGHT) | \
					(state.player_state == self.consts.PLAYER_CATAPULT_LEFT) | \
					(state.player_state == self.consts.PLAYER_CATAPULT_RIGHT)

			def is_falling():
				return state.player_state == self.consts.PLAYER_FALL

			def is_dead():
				return state.player_state == self.consts.PLAYER_DEAD

			def is_climbing():
				return jnp.logical_or(
					state.player_state == self.consts.PLAYER_CLIMB_UP,
					state.player_state == self.consts.PLAYER_CLIMB_DOWN
				)

			def is_climb_idle():
				return state.player_state == self.consts.PLAYER_CLIMB_IDLE

			def is_moving():
				return jnp.logical_or(
					state.player_state == self.consts.PLAYER_MOVE_LEFT,
					state.player_state == self.consts.PLAYER_MOVE_RIGHT
				)

			def walk_cycle(freeze):
				frame = ((state.stage_steps - 1) // 4) % 4
				def right_frames():
					return jax.lax.switch(
						frame,
						[
							lambda: self.SPRITE_PLAYER_MOVE1_RIGHT,
							lambda: self.SPRITE_PLAYER_MOVE2_RIGHT,
							lambda: self.SPRITE_PLAYER_MOVE1_RIGHT,
							lambda: self.SPRITE_PLAYER_IDLE_RIGHT
						]
					)
				def left_frames():
					return jax.lax.switch(
						frame,
						[
							lambda: self.SPRITE_PLAYER_MOVE1_LEFT,
							lambda: self.SPRITE_PLAYER_MOVE2_LEFT,
							lambda: self.SPRITE_PLAYER_MOVE1_LEFT,
							lambda: self.SPRITE_PLAYER_IDLE_LEFT
						]
					)
				def idle_right():
					return self.SPRITE_PLAYER_IDLE_RIGHT
				def idle_left():
					return self.SPRITE_PLAYER_IDLE_LEFT

				return jax.lax.cond(
					freeze,
					lambda: jax.lax.cond(
						jnp.logical_or(
							state.player_state == self.consts.PLAYER_MOVE_LEFT,
							state.player_state == self.consts.PLAYER_IDLE_LEFT
						),
						idle_left,
						idle_right
					),
					lambda: jax.lax.cond(
						jnp.logical_or(
							state.player_state == self.consts.PLAYER_MOVE_LEFT,
							state.player_state == self.consts.PLAYER_IDLE_LEFT
						),
						left_frames,
						right_frames
					)
				)
			
			def climb_cycle(freeze):
				frame = ((state.stage_steps - 1) // 8) % 2
				def idle():
					return self.SPRITE_PLAYER_CLIMB2
				def anim():
					return jax.lax.switch(
						frame,
						[
							lambda: self.SPRITE_PLAYER_CLIMB1,
							lambda: self.SPRITE_PLAYER_CLIMB2
						]
					)
				return jax.lax.cond(freeze, idle, anim)


			freeze = state.death_type != self.consts.DEATH_TYPE_NONE
			# Main sprite selection logic
			sprite = jax.lax.cond(
				is_jumping_state(),
				lambda: jax.lax.cond(
					jnp.logical_or(state.player_state == self.consts.PLAYER_JUMP_LEFT, state.player_state == self.consts.PLAYER_CATAPULT_LEFT),
					lambda: self.SPRITE_PLAYER_JUMP_LEFT,
					lambda: self.SPRITE_PLAYER_JUMP_RIGHT
				),  # Jump sprite takes priority
				lambda: jax.lax.cond(
					is_falling(),
					lambda: self.SPRITE_PLAYER_FALL,  # Show fall animation when falling
					lambda: jax.lax.cond(
						is_dead(),
						lambda: self.SPRITE_PLAYER_DEAD,  # Show dead sprite when dead
						lambda: jax.lax.cond(
							is_idle_state(),
							lambda: jax.lax.cond(
								state.player_state == self.consts.PLAYER_IDLE_LEFT,
								lambda: self.SPRITE_PLAYER_IDLE_LEFT,
								lambda: self.SPRITE_PLAYER_IDLE_RIGHT
							),
							lambda: jax.lax.cond(
								is_climb_idle(),
								lambda: self.SPRITE_PLAYER_CLIMB2, # freeze on last climb frame
								lambda: jax.lax.cond(
									is_climbing(),
									lambda: climb_cycle(freeze),
									lambda: jax.lax.cond(
										is_moving(),
										lambda: walk_cycle(freeze),
										lambda: self.SPRITE_PLAYER_IDLE_RIGHT  # fallback
									)
								)
							)
						)
					)
				)
			)

			return sprite

		
		# Render player
		player_sprite = get_player_sprite()

		def render_player(raster_in):
			def offset():
				is_climbing_state = jnp.logical_or(
					state.player_state == self.consts.PLAYER_CLIMB_IDLE,
					jnp.logical_or(
						state.player_state == self.consts.PLAYER_CLIMB_UP,
						state.player_state == self.consts.PLAYER_CLIMB_DOWN
					)
				)
				return jax.lax.cond(
					is_climbing_state,
					lambda: -3,
					lambda: jax.lax.cond(
						jnp.logical_or(state.player_state == self.consts.PLAYER_FALL, state.player_state == self.consts.PLAYER_DEAD),
						lambda: -2,
						lambda: jax.lax.cond(
							(state.player_state == self.consts.PLAYER_IDLE_RIGHT) |
							(state.player_state == self.consts.PLAYER_MOVE_RIGHT) |
							(state.player_state == self.consts.PLAYER_JUMP_RIGHT) |
							(state.player_state == self.consts.PLAYER_CATAPULT_RIGHT),
							lambda: -2,
							lambda: -5
						)
					)
				)

			return jr.render_at(
				raster_in,
				state.player_x + offset(),
				state.player_y - self.consts.PLAYER_SIZE[1],
				jr.get_sprite_frame(player_sprite, 0)
			)
				
		raster = jax.lax.cond(
			(state.gamestate == self.consts.GAMESTATE_GAMEPLAY) |
			(state.gamestate == self.consts.GAMESTATE_DEATH) |
			(state.gamestate == self.consts.GAMESTATE_SUCCESS),
			render_player,
			lambda r: r,
			operand=raster
		)

		def render_overlay(raster_in, player_height): # This is done so that the player appers to move out of the level when reaching the top 
			bar_color = jnp.array([201, 92, 135, 255], dtype=jnp.uint8)
			bar = jnp.full((1, 16, 4), bar_color, dtype=jnp.uint8)
			
			raster_out = jr.render_at(raster_in, 76, 39, bar)
			
			black_box = jnp.zeros((player_height, 16, 4), dtype=jnp.uint8)
			black_box = black_box.at[:, :, 3].set(255)
			
			raster_out = jr.render_at(raster_out, 74, 39 - player_height, black_box)
			
			return raster_out

		# Place this call after player rendering but before princess rendering
		raster = jax.lax.cond(
			jnp.logical_and(state.gamestate != self.consts.GAMESTATE_SUCCESS, state.death_type != self.consts.DEATH_TYPE_BOMB_EXPLODE),
			lambda r: render_overlay(r, 16), # self.consts.PLAYER_SIZE[1] for some reason cant access that here
			lambda r: r,
			operand=raster
		)
		
		def render_princess(raster_in):
			def closed(): return self.SPRITE_PRINCESS_CLOSED, 0
			def open(): return self.SPRITE_PRINCESS_OPEN, 1 # offset x by 1

			princess_sprite, x_offset = jax.lax.cond(
				state.princess_waving,
				closed,
				open
			)

			return jr.render_at(
				raster_in,
				state.princess_x - x_offset,
				state.princess_y - self.consts.PRINCESS_SIZE[1],
				jr.get_sprite_frame(princess_sprite, 0)
			)

		raster = jax.lax.cond(
			state.princess_visible != 0,
			render_princess,
			lambda r: r,
			operand=raster
		)

		# Render active bombs
		def render_single_bomb(i, raster_in):
			is_active = state.bomb_active[i] > 0

			def draw_bomb(raster_bomb):				
				# Create deterministic seed that only depends on bomb index and time period
				# Use a fixed base key to avoid dependency on changing state.rng_key
				base_key = jax.random.key(42)  # Fixed seed
				bomb_key = jax.random.fold_in(base_key, i)  # unique per bomb
				period_key = jax.random.fold_in(bomb_key, state.stage_steps // 8)
				
				# Generate random boolean for mirroring (10% chance per period)
				should_mirror = jax.random.bernoulli(period_key, p=0.5)
				
				# Get base sprite
				base_sprite = jax.lax.cond(
					state.bomb_is_magic[i] > 0,
					lambda: self.SPRITE_MAGIC_BOMB,
					lambda: self.SPRITE_BOMB
				)
				
				# Apply horizontal mirroring if needed
				bomb_sprite = jax.lax.cond(
					should_mirror,
					lambda: base_sprite[:, :, ::-1, :],  # Flip horizontally
					lambda: base_sprite
				)

				# Calculate offsets - simplified logic
				base_offset_x = jax.lax.cond(
					state.bomb_is_magic[i] > 0,
					lambda: 1,
					lambda: 0
				)
				
				# Add 2-pixel left offset when mirrored (only for magic bombs)
				mirror_offset_x = jax.lax.cond(
					jnp.logical_and(should_mirror, state.bomb_is_magic[i] > 0),
					lambda: -2,
					lambda: 0
				)
				
				total_offset_x = base_offset_x + mirror_offset_x

				return jr.render_at(
					raster_bomb,
					state.bomb_positions_x[i] + total_offset_x,
					state.bomb_positions_y[i] - self.consts.BOMB_SIZE[1],
					jr.get_sprite_frame(bomb_sprite, 0)
				)

			return jax.lax.cond(is_active, draw_bomb, lambda r: r, operand=raster_in)

		# Render all bombs
		for i in range(self.consts.MAX_BOMBS):
			raster = render_single_bomb(i, raster)


		# Render Kong if visible
		def render_kong(raster_in):
			return jr.render_at(
				raster_in,
				state.kong_x,
				state.kong_y - self.consts.KONG_SIZE[1],
				jr.get_sprite_frame(self.SPRITE_KONG, 0)
			)
		
		raster = jax.lax.cond(
			state.kong_visible != 0,
			render_kong,
			lambda r: r,
			operand=raster
		)
		
		# Render lives
		def render_lives(raster_in):
			def draw_life_loop(i, raster_current):
				life_x = self.consts.LIFE_LOCATION[0] + i * self.consts.LIFE_SPACE_BETWEEN
				life_y = self.consts.LIFE_LOCATION[1]
				should_draw = i < state.lives
				return jax.lax.cond(
					should_draw,
					lambda r: jr.render_at(
						r, life_x, life_y - self.consts.PLAYER_SIZE[1],
						jr.get_sprite_frame(self.SPRITE_LIFE, 0)
					),
					lambda r: r,
					operand=raster_current
				)
			return jax.lax.fori_loop(0, self.consts.MAX_LIVES, draw_life_loop, raster_in)
		
		# Because the Respawn Icons are UI-only they are rendered depending on the gamestate 
		raster = jax.lax.cond(
			state.gamestate == self.consts.GAMESTATE_RESPAWN,
			render_lives,
			lambda r: r,
			operand=raster
		) 
		
		# Render score and timer. They are always shown. 
		def render_score(raster_in):
			def extract_score_digits(score):
				d3 = score // 1000 % 10
				d2 = score // 100 % 10
				d1 = score // 10 % 10
				d0 = score % 10
				return jnp.array([d3, d2, d1, d0])
			
			def draw_score_digits(raster_inner):
				digits = extract_score_digits(state.score)
				def draw_digit_loop(i, raster_current):
					digit_x = self.consts.SCORE_LOCATION[0] + i * (self.consts.NUMBER_SIZE[0] + self.consts.SCORE_SPACE_BETWEEN)
					digit_y = self.consts.SCORE_LOCATION[1] - self.consts.NUMBER_SIZE[1]
					digit_value = digits[i]
					return jr.render_at(
						raster_current, digit_x, digit_y,
						jr.get_sprite_frame(self.SPRITE_NUMBERS[digit_value], 0)
					)
				return jax.lax.fori_loop(0, 4, draw_digit_loop, raster_inner)
			return draw_score_digits(raster_in)
		
		raster = render_score(raster)
		
		def render_bonus_timer(raster_in):
			def extract_timer_digits(bonus):
				d2 = bonus // 100 % 10
				d1 = bonus // 10 % 10
				d0 = bonus % 10
				return jnp.array([d2, d1, d0])
			
			def draw_timer_digits(raster_inner):
				digits = extract_timer_digits(state.bonus_timer)
				def draw_timer_digit_loop(i, raster_current):
					digit_x = self.consts.TIMER_LOCATION[0] + i * (self.consts.NUMBER_SIZE[0] + self.consts.TIMER_SPACE_BETWEEN)
					digit_y = self.consts.TIMER_LOCATION[1] - self.consts.NUMBER_SIZE[1]
					digit_value = digits[i]
					return jr.render_at(
						raster_current, digit_x, digit_y,
						jr.get_sprite_frame(self.SPRITE_NUMBERS[digit_value], 0)
					)
				return jax.lax.fori_loop(0, 3, draw_timer_digit_loop, raster_inner)
			return draw_timer_digits(raster_in)
		
		raster = render_bonus_timer(raster)

		# Render debug info
		if self.consts.DEBUG_RENDER:
			red_pixel = jnp.array([[[255, 0, 0, 255]]], dtype=jnp.uint8)
			blue_pixel = jnp.array([[[0, 0, 255, 255]]], dtype=jnp.uint8)
			white_pixel = jnp.array([[[255, 255, 255, 255]]], dtype=jnp.uint8)
			green_pixel = jnp.array([[[0, 255, 0, 255]]], dtype=jnp.uint8)

			def render_bbox_points(raster, bbox, pixel):
				x1, y1, x2, y2 = bbox
				points = jnp.array([
					[x1, y1],
					[x2, y1],
					[x1, y2],
					[x2, y2]
				])
				def body(i, r):
					x, y = points[i]
					return jr.render_at(r, x, y, pixel)
				return jax.lax.fori_loop(0, points.shape[0], body, raster)

			def render_all_bbox(raster, ladder_locations, pixel):
				def body(i, r):
					return render_bbox_points(r, ladder_locations[i], pixel)
				return jax.lax.fori_loop(0, ladder_locations.shape[0], body, raster)

			raster = render_all_bbox(raster, self.consts.LADDER_LOCATIONS, red_pixel)

			# Compute player bounding box
			player_width, player_height = self.consts.PLAYER_SIZE
			player_bbox = jnp.array([
				state.player_x,
				state.player_y - player_height,
				state.player_x + player_width,
				state.player_y
			])
			raster = render_bbox_points(raster, player_bbox, white_pixel)

			# Render player bounding box points
			raster = render_bbox_points(raster, player_bbox, white_pixel)

			def render_floor_points(raster, floor_locations, pixel):
				def body(i, r):
					y = floor_locations[i]
					return jr.render_at(r, 0, y, pixel)
				return jax.lax.fori_loop(0, floor_locations.shape[0], body, raster)

			raster = render_floor_points(raster, self.consts.FLOOR_LOCATIONS, blue_pixel)
			
			raster = render_all_bbox(raster, self.consts.HOLE_LOCATIONS, green_pixel)
			
			active_mask = state.bomb_active > 0
			def draw_bomb_box(i, r):
				is_active = active_mask[i]
				
				def draw_box(r):
					x = state.bomb_positions_x[i]
					y = state.bomb_positions_y[i]
					w, h = self.consts.BOMB_SIZE
					
					r = render_bbox_points(r, [x, y, x+w, y-h], white_pixel)
					
					return r
				
				return jax.lax.cond(is_active, lambda r: draw_box(r), lambda r: r, r)
			
			# Draw all bomb boxes
			raster = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, draw_bomb_box, raster)

			raster = jr.render_at(raster, 0, 0, jr.get_sprite_frame(self.SPRITE_NUMBERS[0], 0))
			raster = render_bbox_points(raster, [0, 0, 0+self.consts.NUMBER_SIZE[0], 0-self.consts.PLAYER_SIZE[1]], white_pixel)

					
		return raster
