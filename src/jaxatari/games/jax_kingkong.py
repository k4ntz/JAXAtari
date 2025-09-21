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

	### Sizes
	PLAYER_SIZE: chex.Array = jnp.array([5, 16])
	KONG_SIZE: chex.Array = jnp.array([14, 34])
	PRINCESS_SIZE: chex.Array = jnp.array([8, 17])
	BOMB_SIZE: chex.Array = jnp.array([8, 14])
	MAGIC_BOMB_SIZE: chex.Array = jnp.array([6, 14])
	NUMBER_SIZE: chex.Array = jnp.array([12, 14])
	
	### Locations & Bounds

	# Player 
	PLAYER_RESPAWN_LOCATION: chex.Array = jnp.array([77, 228])
	PLAYER_SUCCESS_LOCATION: chex.Array = jnp.array([]) # TODO 

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
	# bounding boxes (x1, y1, x2, y2)
	LADDER_LOCATIONS: chex.Array = jnp.array([
		[76, 39, 83, 58],
		[20, 59, 27, 83], 
		[132, 59, 139, 82],
		[76, 83, 83, 106],
		[12, 107, 19, 130],
		[140, 107, 147, 130],
		[76, 131, 83, 154],
		[12, 155, 19, 178],
		[140, 155, 147, 178],
		[76, 179, 83, 202],
		[12, 203, 19, 226],
		[140, 203, 147, 226]
	])
	FLOOR_BOUNDS: chex.Array = jnp.array([
		[12, 147], # Ground floor 
		[12, 147], # First floor
		[12, 147], # Second floor 
		[12, 147], # Third floor 
		[12, 147], # Fourth floor 
		[12, 147], # Fifth floor 
		[16, 147], # Sixth floor
		[20, 139], # Seventh floor 
		[12, 147], # Princess floor - no bounds required bc goal reached 
	]) # floor bounds by floor (min_x, min_y) - y is always the same (see FLOOR_LOCATIONS)
	FLOOR_LOCATIONS: chex.Array = jnp.array([226, 202, 178, 154, 130, 106, 82, 58, 38]) # y corrdinate

	PRINCESS_MOVEMENT_BOUNDS: chex.Array = jnp.array([77, 113])

	LIFE_LOCATION: chex.Array = jnp.array([31, 228])
	LIFE_SPACE_BETWEEN: int = 11 

	SCORE_LOCATION: chex.Array = jnp.array([16, 35])
	SCORE_SPACE_BETWEEN: int = 4

	TIMER_LOCATION: chex.Array = jnp.array([112, 35])
	TIMER_SPACE_BETWEEN: int = 4 

	# Entities 
	KONG_START_LOCATION: chex.Array = jnp.array([31, 228])
	PRINCESS_START_LOCATION: chex.Array = jnp.array([93, 37])
	PRINCESS_RESPAWN_LOCATION: chex.Array = jnp.array([77, 37]) # at the start the princess teleports to the left, this al 
	PRINCESS_SUCCESS_LOCATION: chex.Array = jnp.array([]) # TODO 

	### Gameplay 
	# There's six seperate stages: 
	# 0. Idle stage (Pre-Startup):
	#    - Runs for 130 steps at the very beginning.
	#    - Nothing happens, game is paused before startup animation begins.
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

	# Define the gamestates 
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
	SEQ_RESPAWN_PRINCESS_TELEPORT0: int = 0 # every 64 frame she is tp'd back in this stage for some reason 
	SEQ_RESPAWN_PRINCESS_TELEPORT1: int = 64 
	SEQ_RESPAWN_PRINCESS_TELEPORT2: int = 128 
	###################################################################

	###################################################################
	# Define gameplay stage
	DUR_GAMEPLAY: int = 99 * FPS # 990 / 10 = 99 seconds, in steps 
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

	PRINCESS_MOVE_OPTIONS: chex.Array = jnp.array([0, 0, 0, 0, 3, -3, 6, -6]) 

	REGULAR_BOMB_POINTS: int = 25 
	MAGIC_BOMB_POINTS: int = 125 

	MAX_BOMBS: int = 6 
	MAX_SPEED: int = 1
	MAX_LIVES: int = 3
	MAX_SCORE: int = 999_999 # this is basically unachievable 

	# Player states
	PLAYER_IDLE_LEFT = 1
	PLAYER_IDLE_RIGHT = 2
	PLAYER_MOVE_LEFT = 3
	PLAYER_MOVE_RIGHT = 4
	PLAYER_JUMP = 5
	PLAYER_CLIMB_1 = 6 # for animation
	PLAYER_CLIMB_2 = 7
	PLAYER_FALL = 8
	PLAYER_DEAD = 9
	PLAYER_GOAL = 10

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

	# Bombs (multiple bombs can exist, up to 6)
	# Shape of all (MAX_BOMBS)
	bomb_positions_x: chex.Array 
	bomb_positions_y: chex.Array
	bomb_active: chex.Array
	bomb_is_magic: chex.Array
	bomb_directions_x: chex.Array
	bomb_directions_y: chex.Array
	bomb_floor: chex.Array
	 
	# Game stats
	score: chex.Array
	lives: chex.Array
	bonus_timer: chex.Array
	level: chex.Array # Difficulty level (impacts bomb speed)
	
	# Death state info
	death_type: chex.Array
	death_flash_counter: chex.Array
	
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
			
			# Kong state
			kong_x=jnp.array(self.consts.KONG_START_LOCATION[0]).astype(jnp.int32),
			kong_y=jnp.array(self.consts.KONG_START_LOCATION[1]).astype(jnp.int32),
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
			
			# Game stats
			score=jnp.array(0).astype(jnp.int32),
			lives=jnp.array(self.consts.MAX_LIVES).astype(jnp.int32),
			bonus_timer=jnp.array(self.consts.BONUS_START).astype(jnp.int32),
			level=jnp.array(1).astype(jnp.int32),
			
			# Death state
			death_type=jnp.array(self.consts.DEATH_TYPE_NONE).astype(jnp.int32),
			death_flash_counter=jnp.array(0).astype(jnp.int32),
		)
		
		initial_obs = self._get_observation(state)
		return initial_obs, state
	
	@partial(jax.jit, static_argnums=(0,))
	def step(self, state: KingKongState, action: chex.Array) -> Tuple[KingKongObservation, KingKongState, float, bool, KingKongInfo]:		
		# Handle different game states with current stage_steps
		jax.debug.print("🤯 {x} 🤯", x=state.gamestate)
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
			new_state.princess_visible != 0,
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
				stage_steps=0
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
			)

		def do_normal_step(_):
			# Princess teleports at specific intervals
			should_teleport0 = state.stage_steps == self.consts.SEQ_RESPAWN_PRINCESS_TELEPORT0
			should_teleport1 = state.stage_steps == self.consts.SEQ_RESPAWN_PRINCESS_TELEPORT1
			should_teleport2 = state.stage_steps == self.consts.SEQ_RESPAWN_PRINCESS_TELEPORT2
			teleport = should_teleport0 | should_teleport1 | should_teleport2

			princess_x = jax.lax.cond(
				teleport,
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

			final_stage_steps = state.stage_steps + 1

			return state._replace(
				stage_steps=final_stage_steps,
				princess_x=princess_x,
				princess_y=princess_y
			)

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)
	
	def _step_gameplay(self, state: KingKongState, action: chex.Array) -> KingKongState:
		# Determine if we should transition immediately (success, death or timer expired)
		player_reached_top = state.player_floor >= 8
		timer_expired = state.bonus_timer <= 0
		should_die = jnp.logical_or(timer_expired, state.death_type != self.consts.DEATH_TYPE_NONE)
		should_transition = jnp.logical_or(player_reached_top, should_die)

		def do_transition(_):
			new_gamestate = jax.lax.cond(
				player_reached_top,
				lambda: self.consts.GAMESTATE_SUCCESS,
				lambda: self.consts.GAMESTATE_DEATH
			)
			return state._replace(
				gamestate=new_gamestate,
				stage_steps=0,
				death_type=jax.lax.cond(
					jnp.logical_and(timer_expired, state.death_type == self.consts.DEATH_TYPE_NONE),
					lambda: self.consts.DEATH_TYPE_BOMB_EXPLODE,
					lambda: state.death_type
				)
			)

		def do_normal_step(_):
			# Princess teleport at start
			princess_x = jax.lax.cond(
				state.stage_steps == self.consts.SEQ_GAMEPLAY_PRINCESS_TELEPORT,
				lambda: self.consts.PRINCESS_RESPAWN_LOCATION[0],
				lambda: state.princess_x
			)

			# Update bonus timer
			new_bonus_timer = jax.lax.cond(
				jnp.logical_and(state.stage_steps % self.consts.FPS == 0, state.bonus_timer > 0),
				lambda: jnp.maximum(0, state.bonus_timer - self.consts.BONUS_DECREMENT),
				lambda: state.bonus_timer
			)

			# Handle player movement
			new_state = self._update_player_gameplay(state, action)
			new_state = new_state._replace(
				bonus_timer=new_bonus_timer,
				princess_x=princess_x,
				stage_steps=state.stage_steps
			)

			# Update bombs and collisions
			new_state = self._update_bombs(new_state)
			new_state = self._check_collisions(new_state)

			# Increment stage steps
			new_state = new_state._replace(stage_steps=state.stage_steps + 1)

			return new_state

		return jax.lax.cond(should_transition, do_transition, do_normal_step, operand=None)


	def _update_player_gameplay(self, state: KingKongState, action: chex.Array) -> KingKongState:
		"""Update player position and state based on action"""
		# Determine movement intentions
		move_left = action == Action.LEFT
		move_right = action == Action.RIGHT
		move_up = action == Action.UP
		move_down = action == Action.DOWN
		jump = action == Action.FIRE
		
		# Get current floor bounds
		current_floor_bounds = self.consts.FLOOR_BOUNDS[state.player_floor]
		min_x = current_floor_bounds[0]
		max_x = current_floor_bounds[1]
		
		# Handle horizontal movement
		new_player_x = state.player_x
		new_player_state = state.player_state  # Initialize with current state
		
		# Left movement
		new_player_x = jax.lax.cond(
			jnp.logical_and(move_left, state.player_x > min_x),
			lambda: state.player_x - 1,
			lambda: new_player_x
		)
		
		new_player_state = jax.lax.cond(
			move_left,
			lambda: self.consts.PLAYER_MOVE_LEFT,
			lambda: new_player_state
		)
		
		# Right movement
		new_player_x = jax.lax.cond(
			jnp.logical_and(move_right, state.player_x < max_x),
			lambda: state.player_x + 1,
			lambda: new_player_x
		)
		
		new_player_state = jax.lax.cond(
			move_right,
			lambda: self.consts.PLAYER_MOVE_RIGHT,
			lambda: new_player_state
		)
		
		# Handle ladder climbing
		on_ladder = self._check_on_ladder(new_player_x, state.player_y)
		
		new_player_floor = state.player_floor
		new_player_y = state.player_y
		
		# Climb up
		can_climb_up = jnp.logical_and(on_ladder, state.player_floor < 8)
		new_player_floor = jax.lax.cond(
			jnp.logical_and(move_up, can_climb_up),
			lambda: state.player_floor + 1,
			lambda: new_player_floor
		)
		
		new_player_y = jax.lax.cond(
			jnp.logical_and(move_up, can_climb_up),
			lambda: self.consts.FLOOR_LOCATIONS[new_player_floor],
			lambda: new_player_y
		)
		
		new_player_state = jax.lax.cond(
			jnp.logical_and(move_up, on_ladder),
			lambda: self.consts.PLAYER_CLIMB_1,
			lambda: new_player_state
		)
		
		# Climb down
		can_climb_down = jnp.logical_and(on_ladder, state.player_floor > 0)
		new_player_floor = jax.lax.cond(
			jnp.logical_and(move_down, can_climb_down),
			lambda: state.player_floor - 1,
			lambda: new_player_floor
		)
		
		new_player_y = jax.lax.cond(
			jnp.logical_and(move_down, can_climb_down),
			lambda: self.consts.FLOOR_LOCATIONS[new_player_floor],
			lambda: new_player_y
		)
		
		new_player_state = jax.lax.cond(
			jnp.logical_and(move_down, on_ladder),
			lambda: self.consts.PLAYER_CLIMB_1,
			lambda: new_player_state
		)
		
		# Handle jumping (simplified - just set jump state)
		can_jump = jnp.logical_and(jump, state.player_floor < 8)  # Can't jump on top floor
		new_player_state = jax.lax.cond(
			can_jump,
			lambda: self.consts.PLAYER_JUMP,
			lambda: new_player_state
		)
		
		# Handle idle states - if no movement action, determine idle state based on previous movement
		no_movement = jnp.logical_not(jnp.logical_or(
			jnp.logical_or(move_left, move_right),
			jnp.logical_or(jnp.logical_or(move_up, move_down), jump)
		))
		
		# Set idle state based on last movement direction
		new_player_state = jax.lax.cond(
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
						lambda: state.player_state,  # Keep current idle state
						lambda: self.consts.PLAYER_IDLE_RIGHT  # Default to right-facing idle
					)
				)
			),
			lambda: new_player_state
		)
		
		return state._replace(
			player_x=new_player_x,
			player_y=new_player_y,
			player_floor=new_player_floor,
			player_state=new_player_state
		)

	
	def _check_on_ladder(self, player_x, player_y):
		ladder_x1 = self.consts.LADDER_LOCATIONS[:, 0]
		ladder_y1 = self.consts.LADDER_LOCATIONS[:, 1]
		ladder_x2 = self.consts.LADDER_LOCATIONS[:, 2]
		ladder_y2 = self.consts.LADDER_LOCATIONS[:, 3]

		on_ladders = jnp.logical_and(
			jnp.logical_and(player_x >= ladder_x1, player_x <= ladder_x2),
			jnp.logical_and(player_y >= ladder_y1, player_y <= ladder_y2)
		)
		return jnp.any(on_ladders)

	def _update_bombs(self, state: KingKongState) -> KingKongState:
		should_spawn = jnp.logical_and(
			state.stage_steps % 60 == 0,
			state.kong_visible > 0
		)

		# Find first inactive bomb slot
		first_inactive = jnp.argmax(state.bomb_active == 0)
		can_spawn = state.bomb_active[first_inactive] == 0

		spawn_mask = jnp.logical_and(should_spawn, can_spawn)

		# Spawn bomb if allowed
		new_bomb_active = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_active.at[first_inactive].set(1),
			lambda: state.bomb_active
		)
		new_bomb_x = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_positions_x.at[first_inactive].set(state.kong_x),
			lambda: state.bomb_positions_x
		)
		new_bomb_y = jax.lax.cond(
			spawn_mask,
			lambda: state.bomb_positions_y.at[first_inactive].set(state.kong_y),
			lambda: state.bomb_positions_y
		)

		# Vectorized movement
		is_active = new_bomb_active > 0
		new_bomb_x = jnp.where(is_active, new_bomb_x + state.bomb_directions_x, new_bomb_x)
		new_bomb_y = jnp.where(is_active, new_bomb_y + 1, new_bomb_y)  # fall down

		# Deactivate bombs off-screen
		new_bomb_active = jnp.where(new_bomb_y > self.consts.HEIGHT, 0, new_bomb_active)

		return state._replace(
			bomb_positions_x=new_bomb_x,
			bomb_positions_y=new_bomb_y,
			bomb_active=new_bomb_active
		)
		
	def _rectangles_overlap(self, r1, r2):
		x1, y1, x2, y2 = r1
		a1, b1, a2, b2 = r2
		return jnp.logical_and(x1 < a2, jnp.logical_and(x2 > a1,
			jnp.logical_and(y1 < b2, y2 > b1)))

	def _check_hole_collision(self, player_x, player_y, player_floor):
		hole_rects = self.consts.HOLE_LOCATIONS
		player_rect = jnp.array([
			player_x,
			player_y,
			player_x + self.consts.PLAYER_SIZE[0],
			player_y + self.consts.PLAYER_SIZE[1]
		])

		floor_y = self.consts.FLOOR_LOCATIONS[player_floor]

		# Vectorized check for holes on this floor
		floor_mask = jnp.isclose(hole_rects[:, 1], floor_y)
		x1, y1, x2, y2 = player_rect

		def rect_overlap(hole_rect, mask):
			a1, b1, a2, b2 = hole_rect
			overlap = jnp.logical_and(x1 < a2,
					jnp.logical_and(x2 > a1,
					jnp.logical_and(y1 < b2, y2 > b1)))
			return overlap & mask

		collisions = jax.vmap(rect_overlap)(hole_rects, floor_mask)
		return jnp.any(collisions)

	def _check_collisions(self, state: KingKongState) -> KingKongState:
		"""Check for player-bomb collisions in JAX-friendly vectorized style"""
		player_rect = jnp.array([
			state.player_x,
			state.player_y,
			state.player_x + self.consts.PLAYER_SIZE[0],
			state.player_y + self.consts.PLAYER_SIZE[1]
		])

		# Active bombs (gives bool array)
		bomb_active = state.bomb_active > 0

		def bomb_collision(bomb_xy, active):
			x, y = bomb_xy
			bomb_rect = jnp.array([x, y, x + self.consts.BOMB_SIZE[0], y + self.consts.BOMB_SIZE[1]])
			return self._rectangles_overlap(player_rect, bomb_rect) & active

		# Collision mask: active bombs overlapping player
		bombs_xy = jnp.stack([state.bomb_positions_x, state.bomb_positions_y], axis=1)
		collisions = jax.vmap(bomb_collision)(bombs_xy, bomb_active)

		is_jumping = state.player_state == self.consts.PLAYER_JUMP

		# Bombs jumped on
		jumped_bombs = collisions & is_jumping
		points = jnp.where(state.bomb_is_magic > 0,
						self.consts.MAGIC_BOMB_POINTS,
						self.consts.REGULAR_BOMB_POINTS)
		new_score = state.score + jnp.sum(jumped_bombs * points)

		# Bombs hit when not jumping
		hit_bombs = collisions & (~is_jumping)
		death_type = jnp.where(jnp.any(hit_bombs), 1, state.death_type)

		# Check for falling through holes
		fell_through_hole = self._check_hole_collision(
			state.player_x, state.player_y, state.player_floor
		)
		death_type = jax.lax.cond(
			fell_through_hole,
			lambda: self.consts.DEATH_TYPE_FALL,  # Fall death
			lambda: death_type
		)

		return state._replace(
			death_type=death_type,
			score=new_score
		)

	def _step_death(self, state: KingKongState, action: chex.Array) -> KingKongState:
		# Determine duration based on death type
		stage_duration = jax.lax.cond(
			state.death_type == self.consts.DEATH_TYPE_BOMB_EXPLODE,  # Bomb explode
			lambda: self.consts.DUR_BOMB_EXPLODE,
			lambda: self.consts.DUR_FALL
		)

		# Increment flash counter if bomb explode
		death_flash_counter = jax.lax.cond(
			state.death_type == self.consts.DEATH_TYPE_BOMB_EXPLODE,
			lambda: jnp.minimum(state.death_flash_counter + 1, self.consts.CNT_DEATH_FLASHES),
			lambda: state.death_flash_counter
		)

		should_transition = state.stage_steps >= stage_duration

		new_gamestate = jax.lax.cond(
			should_transition,
			lambda: self.consts.GAMESTATE_RESPAWN,
			lambda: state.gamestate
		)

		final_stage_steps = jax.lax.cond(
			should_transition,
			lambda: 0,
			lambda: state.stage_steps + 1
		)

		return state._replace(
			gamestate=new_gamestate,
			stage_steps=final_stage_steps,
			death_flash_counter=death_flash_counter
		)

	def _step_success(self, state: KingKongState, action: chex.Array) -> KingKongState:
		should_transition = state.stage_steps >= self.consts.DUR_SUCCESS

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

		# Reset player and princess for next loop
		player_x = jax.lax.cond(should_transition, lambda: self.consts.PLAYER_RESPAWN_LOCATION[0], lambda: state.player_x)
		player_y = jax.lax.cond(should_transition, lambda: self.consts.PLAYER_RESPAWN_LOCATION[1], lambda: state.player_y)
		player_floor = jax.lax.cond(should_transition, lambda: 0, lambda: state.player_floor)

		princess_x = jax.lax.cond(should_transition, lambda: self.consts.PRINCESS_RESPAWN_LOCATION[0], lambda: state.princess_x)
		princess_y = jax.lax.cond(should_transition, lambda: self.consts.PRINCESS_RESPAWN_LOCATION[1], lambda: state.princess_y)
		princess_visible = jax.lax.cond(should_transition, lambda: 0, lambda: state.princess_visible)

		return state._replace(
			gamestate=new_gamestate,
			stage_steps=final_stage_steps,
			player_x=player_x,
			player_y=player_y,
			player_floor=player_floor,
			princess_x=princess_x,
			princess_y=princess_y,
			princess_visible=princess_visible
		)


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
			self.SPRITE_PLAYER_IDLE,
			self.SPRITE_PLAYER_MOVE1,
			self.SPRITE_PLAYER_MOVE2,
			self.SPRITE_PLAYER_DEAD,
			self.SPRITE_PLAYER_JUMP,
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
		
		# Load all sprites first
		level = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/level.npy"))
		player_idle = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_idle.npy"))
		player_move1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_move1.npy"))
		player_move2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_move2.npy"))
		player_dead = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_dead.npy"))
		player_jump = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_jump.npy"))
		player_fall = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_fall.npy"))
		player_climb1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_climb1.npy"))
		player_climb2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_climb2.npy"))

		kong = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/kingkong_idle.npy"))
		bomb = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/bomb.npy"))
		magic_bomb = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/magic_bomb.npy"))
		princess_closed = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/princess_closed.npy"))
		princess_open = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/princess_open.npy"))
		
		# Load number sprites (0-9)
		numbers = {}
		for i in range(10):
			numbers[i] = jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/kingkong/{i}.npy"))
		
		# Universal padding function
		def pad_to_exact_size(sprite, target_h, target_w):
			current_h, current_w = sprite.shape[0], sprite.shape[1]
			pad_h = max(0, target_h - current_h)
			pad_w = max(0, target_w - current_w)
			return jnp.pad(sprite, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
		
		# Find maximum dimensions for player sprites (used in conditionals)
		all_player_sprites = [player_idle, player_move1, player_move2, player_dead]
		player_max_height = max(sprite.shape[0] for sprite in all_player_sprites)
		player_max_width = max(sprite.shape[1] for sprite in all_player_sprites)
		
		# Ensure minimum dimensions based on constants
		player_target_height = max(player_max_height, 16)
		player_target_width = max(player_max_width, 8)
		
		# Pad all player sprites to exact same dimensions
		player_idle = pad_to_exact_size(player_idle, player_target_height, player_target_width)
		player_move1 = pad_to_exact_size(player_move1, player_target_height, player_target_width)
		player_move2 = pad_to_exact_size(player_move2, player_target_height, player_target_width)
		player_dead = pad_to_exact_size(player_dead, player_target_height, player_target_width)
		player_jump = pad_to_exact_size(player_jump, player_target_height, player_target_width)
		player_fall = pad_to_exact_size(player_fall, player_target_height, player_target_width)
		player_climb1 = pad_to_exact_size(player_climb1, player_target_height, player_target_width)
		player_climb1 = pad_to_exact_size(player_climb1, player_target_height, player_target_width)

		# Pad bomb sprites to same dimensions 
		bomb_sprites = [bomb, magic_bomb]
		bomb_max_height = max(sprite.shape[0] for sprite in bomb_sprites)
		bomb_max_width = max(sprite.shape[1] for sprite in bomb_sprites)
		
		bomb_target_height = max(bomb_max_height, 14)
		bomb_target_width = max(bomb_max_width, 8)
		
		bomb = pad_to_exact_size(bomb, bomb_target_height, bomb_target_width)
		magic_bomb = pad_to_exact_size(magic_bomb, bomb_target_height, bomb_target_width)
		
		# Pad princess sprites to same dimensions
		princess_sprites = [princess_closed, princess_open]
		princess_max_height = max(sprite.shape[0] for sprite in princess_sprites)
		princess_max_width = max(sprite.shape[1] for sprite in princess_sprites)
		
		princess_target_height = max(princess_max_height, 17)
		princess_target_width = max(princess_max_width, 8)
		
		princess_closed = pad_to_exact_size(princess_closed, princess_target_height, princess_target_width)
		princess_open = pad_to_exact_size(princess_open, princess_target_height, princess_target_width)
		
		# Use padded player sprite for life display
		life = player_move2.copy()
		
		# Expand dimensions for sprites
		SPRITE_LEVEL = jnp.expand_dims(level, axis=0)
		SPRITE_PLAYER_IDLE = jnp.expand_dims(player_idle, axis=0)
		SPRITE_PLAYER_MOVE1 = jnp.expand_dims(player_move1, axis=0)
		SPRITE_PLAYER_MOVE2 = jnp.expand_dims(player_move2, axis=0)
		SPRITE_PLAYER_DEAD = jnp.expand_dims(player_dead, axis=0)
		SPRITE_PLAYER_JUMP = jnp.expand_dims(player_jump, axis=0)
		SPRITE_PLAYER_FALL = jnp.expand_dims(player_fall, axis=0)
		SPRITE_PLAYER_CLIMB1 = jnp.expand_dims(player_climb1, axis=0)
		SPRITE_PLAYER_CLIMB2 = jnp.expand_dims(player_climb2, axis=0)
		SPRITE_KONG = jnp.expand_dims(kong, axis=0)
		SPRITE_LIFE = jnp.expand_dims(life, axis=0)
		SPRITE_BOMB = jnp.expand_dims(bomb, axis=0)
		SPRITE_MAGIC_BOMB = jnp.expand_dims(magic_bomb, axis=0)
		SPRITE_PRINCESS_CLOSED = jnp.expand_dims(princess_closed, axis=0)
		SPRITE_PRINCESS_OPEN = jnp.expand_dims(princess_open, axis=0)
		
		# Create number sprites array
		SPRITE_NUMBERS = jnp.stack([jnp.expand_dims(numbers[i], axis=0) for i in range(10)], axis=0)
		
		return (
			SPRITE_LEVEL, SPRITE_PLAYER_IDLE, SPRITE_PLAYER_MOVE1, SPRITE_PLAYER_MOVE2,
			SPRITE_PLAYER_DEAD, SPRITE_PLAYER_JUMP, SPRITE_PLAYER_FALL,
			SPRITE_PLAYER_CLIMB1, SPRITE_PLAYER_CLIMB2, SPRITE_KONG,
			SPRITE_LIFE, SPRITE_BOMB, SPRITE_MAGIC_BOMB, SPRITE_PRINCESS_CLOSED,
			SPRITE_PRINCESS_OPEN, SPRITE_NUMBERS
		)

	def render(self, state: KingKongState) -> jnp.ndarray:
		raster = jr.create_initial_frame(self.consts.WIDTH, self.consts.HEIGHT)
		
		frame_level = jr.get_sprite_frame(self.SPRITE_LEVEL, 0)
		raster = jr.render_at(raster, *self.consts.LEVEL_LOCATION, frame_level)
		
		# Render player based on state
		def get_player_sprite():
			# Simplified sprite selection using direct conditionals
			sprite = jax.lax.cond(
				jnp.logical_or(
					state.player_state == self.consts.PLAYER_IDLE_LEFT,
					state.player_state == self.consts.PLAYER_IDLE_RIGHT
				),
				lambda: self.SPRITE_PLAYER_IDLE,
				lambda: jax.lax.cond(
					jnp.logical_or(
						state.player_state == self.consts.PLAYER_FALL,
						state.player_state == self.consts.PLAYER_DEAD
					),
					lambda: self.SPRITE_PLAYER_DEAD,
					lambda: jax.lax.cond(
						state.step_counter % 6 < 3,
						lambda: self.SPRITE_PLAYER_MOVE1,
						lambda: self.SPRITE_PLAYER_MOVE2
					)
				)
			)
			
			# Mirror sprite if facing left
			should_mirror = jnp.logical_or(
				state.player_state == self.consts.PLAYER_IDLE_LEFT,
				state.player_state == self.consts.PLAYER_MOVE_LEFT
			)
			
			return jax.lax.cond(
				should_mirror,
				lambda s: s[:, :, ::-1],
				lambda s: s,
				operand=sprite
			)
			
		# Render player
		player_sprite = get_player_sprite()

		def render_player(raster_in):
			return jr.render_at(
				raster_in,
				state.player_x,
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
				bomb_sprite = jax.lax.cond(
					state.bomb_is_magic[i] > 0,
					lambda: self.SPRITE_MAGIC_BOMB,
					lambda: self.SPRITE_BOMB
				)
				return jr.render_at(
					raster_bomb,
					state.bomb_positions_x[i],
					state.bomb_positions_y[i] - self.consts.BOMB_SIZE[1],
					jr.get_sprite_frame(bomb_sprite, 0)
				)
			
			return jax.lax.cond(is_active, draw_bomb, lambda r: r, operand=raster_in)
		
		# Render all bombs
		for i in range(self.consts.MAX_BOMBS):
			raster = render_single_bomb(i, raster)
		
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
		
		return raster
