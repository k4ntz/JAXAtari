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

	### Sizes
	PLAYER_SIZE: chex.Array = jnp.array([5, 16])
	KONG_SIZE: chex.Array = jnp.array([14, 34])
	PRINCESS_SIZE: chex.Array = jnp.array([8, 17])
	LADDER_SIZE: chex.Array = jnp.array([8, 18])
	FINAL_LADDER_SIZE: chex.Array = jnp.array([8, ])
	BOMB_SIZE: chex.Array = jnp.array([8, 14])
	MAGIC_BOMB_SIZE: chex.Array = jnp.array([6, 14])
	NUMBER_SIZE: chex.Array = jnp.array([12, 14])
	
	### Locations & Bounds
	PLAYER_BOUNDS: chex.Array = jnp.array([
		[12, 147], # Ground floor 
		[12, 147], # First floor
		[12, 147], # Second floor 
		[12, 147], # Third floor 
		[12, 147], # Fourth floor 
		[12, 147], # Fifth floor 
		[16, 147], # Sixth floor
		[20, 139],  # Seventh floor 
		[12, 147], # Princess floor - no bounds required bc goal reached 
	]) # player bounds by floor (min_x, min_y) - y is always the same

	LEVEL_LOCATION: chex.Array = jnp.array([8, 39])
	PLAYER_START_LOCATION: chex.Array = jnp.array([77, 228])
	LADDER_LOCATIONS: chex.Array = jnp.array([30, 60, 90, 120])
	FLOOR_LOCATIONS: chex.Array = jnp.array([226, 202, 178, 154, 130, 106, 82, 58, 38]) # y corrdinate
	LIVE_LOCATION: chex.Array = jnp.array([31, 228])
	LIVE_SPACE_BETWEEN: int = 11 

	SCORE_LOCATION: chex.Array = jnp.array([16, 35])
	SCORE_SPACE_BETWEEN: int = 4

	TIMER_LOCATION: chex.Array = jnp.array([112, 35])
	TIMER_SPACE_BETWEEN: int = 4 

	KONG_START_POSITION: chex.Array = jnp.array([31, 228])
	PRINCESS_START_POSITION: chex.Array = jnp.array([93, 37])
	PRINCESS_START_TELEPORT: chex.Array = jnp.array([77, 36]) # at the start the princess teleports to the left 

	### Gameplay 
	# There's three seperate stages: 
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
	# All of these happen at specific steps and take a specific amount of steps. These are defined below. 
	
	# Define the gamestates 
	GAMESTATE_STARTUP: int = 0
	GAMESTATE_RESPAWN: int = 1 
	GAMESTATE_GAMEPLAY: int = 1 

	### Startup sequence 
	
	# First do 15 diagonal up jumps, then one to the left/right, then 3 diagonal down. This takes 
	KONG_TOTAL_JUMPS: int = 6
	KONG_JUMPS_UP: int = 15
	KONG_JUMPS_SIDE: int = 1 
	KONG_JUMPS_DOWN: int = 3 

	# Start sequence/what happens at step  X
	SEQ_KONG_START_JUMP: int = 130
	SEQ_SHOW_LIVES: int = 385
	SEQ_PRINCESS_TELEPORT: int = 385
	SEQ_PRINCESS_MOVE_LEFT: int = 405 # (every 2 steps): 2 steps left, then 6 right, then tps back, then 6 right, then animatin, then 4 right, then animate  
	SEQ_GAME_BEGIN: int = 580
	
	# Game logic constants 
	BONUS_START: int = 990
	BONUS_DECREMENT: int = 10 # per second 

	REGULAR_BOMB_POINTS: int = 25 
	MAGIC_BOMB_POINTS: int = 125 

	MAGIC_BOMB_BOOST: int = 1 # You advance 1 floor for jumping a magic bomb 
	FLOOR_COUNT: int = 8

	MAX_BOMBS: int = 3 
	MAX_SPEED: int = 1
	MAX_LIVES: int = 3
	MAX_SCORE: int = 999_999 # this is basically unachievable 

	# Player states
	PLAYER_IDLE = 1
	PLAYER_MOVE_LEFT = 2
	PLAYER_MOVE_RIGHT = 3
	PLAYER_JUMP = 4
	PLAYER_CLIMB = 5
	PLAYER_FALL = 6
	PLAYER_DYING = 7
	PLAYER_GOAL = 8

	PLAYER_DIR_LEFT = -1  
	PLAYER_DIR_RIGHT = 1 

class KingKongState(NamedTuple):
	step_counter: chex.Array
	player_x: chex.Array
	player_y: chex.Array
	player_state: chex.Array # idle, move1, move2, climb, jump, fall, dying, goal 
	player_last_dir: chex.Array # left, right
	player_active: chex.Array
	lives: chex.Array
	score: chex.Array
	bonus: chex.Array
	# Can be either BOMB or MAGIC_BOMB 
	bombs: chex.Array  # shape (3, 5): (x, y, width, height, active)
	kong_x: chex.Array 
	kong_y: chex.Array 
	kong_jump_counter: chex.Array # From 1 to 18 where the first 15 are diagonal up and the last 3 are diagonal down, or -1 to -18 if moving left 

	princess_x: chex.Array 
	princess_y: chex.Array 
	princess_active: chex.Array 
	
class EntityPosition(NamedTuple):
	x: chex.Array
	y: chex.Array
	width: chex.Array
	height: chex.Array
	active: chex.Array # 0 or 1

class KingKongObservation(NamedTuple):
	player: EntityPosition
	bombs: tuple[EntityPosition, EntityPosition, EntityPosition] # Maximum of three at a time 
	score: chex.Array
	lives: chex.Array

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
		self.obs_size = 5 + 3 * 5 + 1 + 1

	def reset(self, key=None) -> Tuple[KingKongObservation, KingKongState]:
		initial_bombs = jnp.array([
			[-1, -1, 0, 0, 0],
			[-1, -1, 0, 0, 0],
			[-1, -1, 0, 0, 0]
		], dtype=jnp.int32)

		state = KingKongState(
			step_counter=jnp.array(0),
			player_x=jnp.array(self.consts.PLAYER_START_LOCATION[0]),
			player_y=jnp.array(self.consts.PLAYER_START_LOCATION[1]),
			player_state=jnp.array(0),
			player_last_dir=jnp.array(self.consts.PLAYER_DIR_RIGHT),
			player_active=jnp.array(0),
			lives=jnp.array(self.consts.MAX_LIVES),
			score=jnp.array(0),
			bonus=jnp.array(self.consts.BONUS_START),  # Add this
			bombs=initial_bombs,
			kong_x=jnp.array(self.consts.KONG_START_POSITION[0]),
			kong_y=jnp.array(self.consts.KONG_START_POSITION[1]),
			kong_jump_counter=jnp.array(0),
			princess_x=jnp.array(self.consts.PRINCESS_START_POSITION[0]),
			princess_y=jnp.array(self.consts.PRINCESS_START_POSITION[1]),
			princess_active=jnp.array(0),
		)

		return self._get_observation(state), state

		
	def _startup_step(self, step_counter, kong_x, kong_y, kong_jump_counter):
		do_move = step_counter % 2 == 0
		
		def move_kong(_):
			# Stop if we've reached the total jump limit
			total_jumps_per_cycle = self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE + self.consts.KONG_JUMPS_DOWN
			max_jump_counter = self.consts.KONG_TOTAL_JUMPS * total_jumps_per_cycle
			
			# If we've exceeded total jumps, don't move
			should_continue = kong_jump_counter < max_jump_counter
			
			def do_jump(_):
				# Determine which full zigzag phase we are in
				phase = kong_jump_counter // total_jumps_per_cycle
				dir_lr = jnp.where(phase % 2 == 0, 1, -1)  # even phase: right, odd: left
				step_in_phase = kong_jump_counter % total_jumps_per_cycle
				
				new_x, new_y = jax.lax.cond(
					step_in_phase < self.consts.KONG_JUMPS_UP,  # diagonal up
					lambda _: (kong_x + dir_lr, kong_y - 2),
					lambda _: jax.lax.cond(
						step_in_phase < self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE,  # side step
						lambda _: (kong_x + dir_lr, kong_y),
						lambda _: (kong_x + dir_lr, kong_y + 2),  # diagonal down
						operand=None
					),
					operand=None
				)
				return new_x, new_y, kong_jump_counter + 1
			
			return jax.lax.cond(
				should_continue,
				do_jump,
				lambda _: (kong_x, kong_y, kong_jump_counter),  # Don't increment counter when stopped
				operand=None
			)
		
		return jax.lax.cond(do_move, move_kong, lambda _: (kong_x, kong_y, kong_jump_counter), operand=None)

	def _update_bonus(self, step_counter, current_bonus):
		# Update bonus every second (30 steps assuming 30 FPS)
		# NOTE: This goes down slower in the refrence because incorrect
		# FPS. Here it is correct as per the game spec. 
		should_decrement = step_counter % 30 == 0 
		new_bonus = jax.lax.cond(
			should_decrement,
			lambda b: jnp.maximum(0, b - self.consts.BONUS_DECREMENT),
			lambda b: b,
			operand=current_bonus
		)
		return new_bonus

	def _player_step(self, action: chex.Array, step_counter, player_x, player_y, player_state, last_dir):
		# Only update movement every 4 steps
		do_move = step_counter % 4 == 0

		# Horizontal movement
		move_x = jnp.where(action == Action.RIGHT, self.consts.MAX_SPEED,
					jnp.where(action == Action.LEFT, -self.consts.MAX_SPEED, 0))
		new_player_x = jax.lax.cond(do_move,
			lambda _: player_x + move_x,
			lambda _: player_x,
			operand=None
		)

		# Y position stays the same
		new_player_y = player_y

		# Update player state: move left/right or idle
		def next_state():
			return jax.lax.cond(
				move_x > 0,
				lambda _: self.consts.PLAYER_MOVE_RIGHT,
				lambda _: jax.lax.cond(
					move_x < 0,
					lambda _: self.consts.PLAYER_MOVE_LEFT,
					lambda _: self.consts.PLAYER_IDLE,
					operand=None
				),
				operand=None
			)

		new_player_state = jax.lax.cond(do_move, lambda _: next_state(), lambda _: player_state, operand=None)

		# Update last_dir: 1 = right, -1 = left, keep previous if idle
		new_last_dir = jax.lax.cond(
			move_x > 0, lambda _: 1,
			lambda _: jax.lax.cond(move_x < 0, lambda _: -1, lambda _: last_dir, operand=None),
			operand=None
		)

		return new_player_x, new_player_y, new_player_state, new_last_dir

	def step(self, state: KingKongState, action: chex.Array):
		startup = state.step_counter < self.consts.SEQ_GAME_BEGIN

		def startup_step(state: KingKongState):
			do_move = state.step_counter >= self.consts.SEQ_KONG_START_JUMP
			
			new_kong_x, new_kong_y, new_kong_jump_counter = jax.lax.cond(
				do_move,
				lambda _: self._startup_step(state.step_counter, state.kong_x, state.kong_y, state.kong_jump_counter),
				lambda _: (state.kong_x, state.kong_y, state.kong_jump_counter),
				operand=None
			)
			
			# Check if Kong has finished jumping (princess becomes active)
			total_jumps_per_cycle = self.consts.KONG_JUMPS_UP + self.consts.KONG_JUMPS_SIDE + self.consts.KONG_JUMPS_DOWN
			max_jump_counter = self.consts.KONG_TOTAL_JUMPS * total_jumps_per_cycle
			princess_should_activate = new_kong_jump_counter >= max_jump_counter
			
			# Princess teleports at the designated step 
			should_teleport = state.step_counter >= self.consts.SEQ_PRINCESS_TELEPORT
			
			new_princess_x = jax.lax.cond(
				should_teleport,
				lambda _: self.consts.PRINCESS_START_TELEPORT[0],
				lambda _: state.princess_x,
				operand=None
			)
			
			new_princess_active = jax.lax.cond(
				princess_should_activate,
				lambda _: jnp.array(1),
				lambda _: state.princess_active,
				operand=None
			)

			new_state = KingKongState(
				step_counter=state.step_counter + 1,
				player_x=state.player_x,
				player_y=state.player_y,
				player_state=state.player_state,
				player_last_dir=state.player_last_dir,
				player_active=jnp.array(0), # Keep player inactive during startup
				lives=state.lives,
				score=state.score,
				bonus=state.bonus,
				bombs=state.bombs,
				kong_x=new_kong_x,
				kong_y=new_kong_y,
				kong_jump_counter=new_kong_jump_counter,
				princess_x=new_princess_x,
				princess_y=state.princess_y,
				princess_active=new_princess_active
			)

			done = self._get_done(new_state)
			env_reward = self._get_reward(state, new_state)
			all_rewards = self._get_all_reward(state, new_state)
			info = self._get_info(new_state, all_rewards)
			observation = self._get_observation(new_state)

			return observation, new_state, env_reward, done, info


		def game_step(state: KingKongState, action: chex.Array):
			new_player_x, new_player_y, new_player_state, new_player_last_dir = self._player_step(
				action, state.step_counter, state.player_x, state.player_y, state.player_state, state.player_last_dir
			)

			new_bonus = self._update_bonus(state.step_counter, state.bonus)

			player_reset = False
			new_step_counter = jax.lax.cond(
				player_reset,
				lambda _: jnp.array(0),
				lambda s: s + 1,
				operand=state.step_counter
			)

			new_state = KingKongState(
				step_counter=new_step_counter,
				player_x=new_player_x,
				player_y=new_player_y,
				player_state=new_player_state,
				player_last_dir=new_player_last_dir,
				player_active=jnp.array(1), # Player becomes active in game phase
				lives=state.lives,
				score=state.score,
				bonus=new_bonus,
				bombs=state.bombs,
				kong_x=state.kong_x, 
				kong_y=state.kong_y,
				kong_jump_counter=state.kong_jump_counter,
				princess_x=state.princess_x,  # Added
				princess_y=state.princess_y,  # Added
				princess_active=state.princess_active  # Added
			)


			done = self._get_done(new_state)
			env_reward = self._get_reward(state, new_state)
			all_rewards = self._get_all_reward(state, new_state)
			info = self._get_info(new_state, all_rewards)
			observation = self._get_observation(new_state)

			return observation, new_state, env_reward, done, info

		return jax.lax.cond(startup, lambda _: startup_step(state), lambda _: game_step(state, action), operand=None)
	
	def render(self, state: KingKongState) -> jnp.ndarray:
		return self.renderer.render(state)

	@partial(jax.jit, static_argnums=(0,))
	def _get_observation(self, state: KingKongState):
		player = EntityPosition(
			x=state.player_x,
			y=state.player_y,
			width=jnp.array(self.consts.PLAYER_SIZE[0]),
			height=jnp.array(self.consts.PLAYER_SIZE[1]),
			active=state.player_active  # Use the actual player_active state
		)

		def make_bomb_entity(bomb_row):
			return EntityPosition(
				x=bomb_row[0],
				y=bomb_row[1],
				width=bomb_row[2],
				height=bomb_row[3],
				active=bomb_row[4]
			)

		bombs = tuple(jax.vmap(make_bomb_entity)(state.bombs))

		return KingKongObservation(
			player=player,
			bombs=bombs,
			lives=state.lives,
			score=state.score
		)


	@partial(jax.jit, static_argnums=(0,))
	def obs_to_flat_array(self, obs: KingKongObservation) -> jnp.ndarray:
		# Flatten player
		player_flat = jnp.concatenate([
			obs.player.x.flatten(),
			obs.player.y.flatten(),
			obs.player.width.flatten(),
			obs.player.height.flatten(),
			obs.player.active.flatten(),
			obs.player.active.flatten(),
		])

		# Flatten bombs
		bombs_flat = jnp.concatenate([
			jnp.concatenate([
				b.x.flatten(),
				b.y.flatten(),
				b.width.flatten(),
				b.height.flatten(),
				b.active.flatten()
			]) for b in obs.bombs
		])

		return jnp.concatenate([
			player_flat,
			bombs_flat,
			obs.lives.flatten(),
			obs.score.flatten()
		])

	def action_space(self) -> spaces.Discrete:
		return spaces.Discrete(len(self.action_set))

	def observation_space(self) -> spaces:
		return spaces.Dict({
			"player": spaces.Dict({
				"x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
				"height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
				"active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
			}),
			"bombs": spaces.Dict({
				"x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(3,), dtype=jnp.int32),
				"y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(3,), dtype=jnp.int32),
				"width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(3,), dtype=jnp.int32),
				"height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(3,), dtype=jnp.int32),
				"active": spaces.Box(low=0, high=1, shape=(3,), dtype=jnp.int32),
			}),
			"lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
			"score": spaces.Box(low=0, high=self.consts.MAX_SCORE, shape=(), dtype=jnp.int32),
		})

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
			self.SPRITE_KONG,
			self.SPRITE_LIFE,
			self.SPRITE_BOMB,
			self.SPRITE_MAGIC_BOMB,
			self.SPRITE_LADDER,
			self.SPRITE_PRINCESS_CLOSED,
			self.SPRITE_PRINCESS_OPEN,
			self.SPRITE_NUMBERS
		) = self.load_sprites()

	def load_sprites(self):
		MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
		
		# Load level and character sprites
		level = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/level.npy"))
		player_idle = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_idle.npy"))
		player_move1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_move1.npy"))
		player_move2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_move2.npy"))
		player_dead = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/player_dead.npy"))
		kong = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/kingkong_idle.npy"))
		life = player_move2
		
		# Load game objects
		bomb = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/bomb.npy"))
		magic_bomb = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/magic_bomb.npy"))
		ladder = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/ladder.npy"))
		princess_closed = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/princess_closed.npy"))
		princess_open = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/kingkong/princess_open.npy"))
		
		# Load number sprites (0-9)
		numbers = {}
		for i in range(10):
			numbers[i] = jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/kingkong/{i}.npy"))
		
		# Expand PLAYER_IDLE width from 5 to 6
		player_idle = jnp.pad(player_idle, ((0, 0), (0, 1), (0, 0)), mode="constant")
		
		# Expand PRINCESS_OPEN width from 6 to 8
		princess_open = jnp.pad(princess_open, ((0, 0), (0, 2), (0, 0)), mode="constant")

		# Expand dimensions for sprites
		SPRITE_LEVEL = jnp.expand_dims(level, axis=0)
		SPRITE_PLAYER_IDLE = jnp.expand_dims(player_idle, axis=0)
		SPRITE_PLAYER_MOVE1 = jnp.expand_dims(player_move1, axis=0)
		SPRITE_PLAYER_MOVE2 = jnp.expand_dims(player_move2, axis=0)
		SPRITE_PLAYER_DEAD = jnp.expand_dims(player_dead, axis=0)
		SPRITE_KONG = jnp.expand_dims(kong, axis=0)
		SPRITE_LIFE = jnp.expand_dims(life, axis=0)
		SPRITE_BOMB = jnp.expand_dims(bomb, axis=0)
		SPRITE_MAGIC_BOMB = jnp.expand_dims(magic_bomb, axis=0)
		SPRITE_LADDER = jnp.expand_dims(ladder, axis=0)
		SPRITE_PRINCESS_CLOSED = jnp.expand_dims(princess_closed, axis=0)
		SPRITE_PRINCESS_OPEN = jnp.expand_dims(princess_open, axis=0)
		
		# Create number sprites array
		SPRITE_NUMBERS = jnp.stack([jnp.expand_dims(numbers[i], axis=0) for i in range(10)], axis=0)
		
		return (
			SPRITE_LEVEL,
			SPRITE_PLAYER_IDLE,
			SPRITE_PLAYER_MOVE1,
			SPRITE_PLAYER_MOVE2,
			SPRITE_PLAYER_DEAD,
			SPRITE_KONG,
			SPRITE_LIFE,
			SPRITE_BOMB,
			SPRITE_MAGIC_BOMB,
			SPRITE_LADDER,
			SPRITE_PRINCESS_CLOSED,
			SPRITE_PRINCESS_OPEN,
			SPRITE_NUMBERS
		)

	def render(self, state: KingKongState) -> jnp.ndarray:
		raster = jr.create_initial_frame(self.consts.WIDTH, self.consts.HEIGHT)
		frame_level = jr.get_sprite_frame(self.SPRITE_LEVEL, 0)
		raster = jr.render_at(raster, *self.consts.LEVEL_LOCATION, frame_level)

		def get_player_sprite(step_counter):
			return jax.lax.cond(step_counter % 2 == 0, lambda _: self.SPRITE_PLAYER_MOVE1, lambda _: self.SPRITE_PLAYER_MOVE2, operand=None)
		
		frame_player = jax.lax.cond(
			state.player_state == self.consts.PLAYER_IDLE,
			lambda _: self.SPRITE_PLAYER_IDLE,
			lambda _: get_player_sprite(state.step_counter),
			operand=None
		)

		# Mirror if last movement was left
		frame_player = jax.lax.cond(
			state.player_last_dir == -1,
			lambda f: f[:, :, ::-1],
			lambda f: f,
			operand=frame_player
		)

		# Only render player if active
		def render_player(raster_in):
			return jr.render_at(
				raster_in,
				state.player_x,
				state.player_y - self.consts.PLAYER_SIZE[1],
				jr.get_sprite_frame(frame_player, 0)
			)

		raster = jax.lax.cond(
			state.player_active == 1,
			render_player,
			lambda r: r,
			operand=raster
		)

		raster = jr.render_at(
			raster,
			state.kong_x,
			state.kong_y - self.consts.KONG_SIZE[1],
			jr.get_sprite_frame(self.SPRITE_KONG, 0)
		)

		# Render princess if active
		def render_princess(raster_in):
			princess_sprite = self.SPRITE_PRINCESS_CLOSED
			return jr.render_at(
				raster_in,
				state.princess_x,
				state.princess_y - self.consts.PRINCESS_SIZE[1],
				jr.get_sprite_frame(princess_sprite, 0)
			)

		raster = jax.lax.cond(
			state.princess_active == 1,
			render_princess,
			lambda r: r,
			operand=raster
		)

		# Render lives during startup sequence only
		def render_lives(raster_in):
			show_lives = jnp.logical_and(
				state.step_counter >= self.consts.SEQ_SHOW_LIVES,
				state.step_counter < self.consts.SEQ_GAME_BEGIN
			)
			
			def draw_lives(raster_inner):
				def draw_life_loop(i, raster_current):
					life_x = self.consts.LIVE_LOCATION[0] + i * self.consts.LIVE_SPACE_BETWEEN
					life_y = self.consts.LIVE_LOCATION[1]
					should_draw = i < state.lives
					
					return jax.lax.cond(
						should_draw,
						lambda r: jr.render_at(
							r,
							life_x,
							life_y - self.consts.PLAYER_SIZE[1],
							jr.get_sprite_frame(self.SPRITE_LIFE, 0)
						),
						lambda r: r,
						operand=raster_current
					)
				
				return jax.lax.fori_loop(0, self.consts.MAX_LIVES, draw_life_loop, raster_inner)
			
			return jax.lax.cond(
				show_lives,
				draw_lives,
				lambda r: r,
				operand=raster_in
			)

		raster = render_lives(raster)

		# Render 4-digit score
		def render_score(raster_in):
			def extract_digits(number):
				# Extract 4 digits from score (pad with zeros if needed)
				d3 = number // 1000 % 10
				d2 = (number // 100) % 10
				d1 = (number // 10) % 10
				d0 = number % 10
				return jnp.array([d3, d2, d1, d0])
			
			def draw_score_digits(raster_inner):
				digits = extract_digits(state.score)
				
				def draw_digit_loop(i, raster_current):
					digit_x = self.consts.SCORE_LOCATION[0] + i * (self.consts.NUMBER_SIZE[0] + self.consts.SCORE_SPACE_BETWEEN)
					digit_y = self.consts.SCORE_LOCATION[1] - self.consts.NUMBER_SIZE[1]
					digit_value = digits[i]
					
					return jr.render_at(
						raster_current,
						digit_x,
						digit_y,
						jr.get_sprite_frame(self.SPRITE_NUMBERS[digit_value], 0)
					)
				
				return jax.lax.fori_loop(0, 4, draw_digit_loop, raster_inner)
			
			return draw_score_digits(raster_in)

		raster = render_score(raster)

		def render_bonus_timer(raster_in):
			def extract_timer_digits(bonus):
				d2 = bonus // 100 % 10  
				d1 = (bonus // 10) % 10
				d0 = bonus % 10
				return jnp.array([d2, d1, d0])
			
			def draw_timer_digits(raster_inner):
				digits = extract_timer_digits(state.bonus)  # Use state.bonus directly
				
				def draw_timer_digit_loop(i, raster_current):
					digit_x = self.consts.TIMER_LOCATION[0] + i * (self.consts.NUMBER_SIZE[0] + self.consts.TIMER_SPACE_BETWEEN)
					digit_y = self.consts.TIMER_LOCATION[1] - self.consts.NUMBER_SIZE[1]
					digit_value = digits[i]
					
					return jr.render_at(
						raster_current,
						digit_x,
						digit_y,
						jr.get_sprite_frame(self.SPRITE_NUMBERS[digit_value], 0)
					)
				
				return jax.lax.fori_loop(0, 3, draw_timer_digit_loop, raster_inner)
			
			return draw_timer_digits(raster_in)

		raster = render_bonus_timer(raster)

		return raster
