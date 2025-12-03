"""
Group: Sooraj Rathore, Kadir Özen
Edit: Jan Rafflewski
"""

"""
TODO
    1)  [x] Validate ghost behaviour
    2)  [x] Level progression
    3)  [ ] Performance improvements
    4)  [ ] JIT compatibility

    Optional:
    a)  [ ] Correct speed
    b)  [ ] Pacman death animation
    c)  [ ] Bonus Life at 10000 score
    d)  [ ] Fruit scale and animation
    e)  [ ] Fruit movement patterns
    f)  [ ] Ghost starting positions
    g)  [ ] Ghost behavioral quirks
    h)  [ ] Ghost enjailment
    i)  [ ] Ghost timings
    j)  [ ] Correct maze colors
"""


# -------- Imports --------
from enum import IntEnum
import os
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.games.mspacman_mazes import MsPacmanMaze


# -------- Enums -------- 
class FruitType(IntEnum):
    CHERRY = 0
    STRAWBERRY = 1
    ORANGE = 2
    PRETZEL = 3
    APPLE = 4
    PEAR = 5
    BANANA = 6
    NONE = 7

class GhostType(IntEnum):
    BLINKY = 0
    PINKY = 1
    INKY = 2
    SUE = 3

class GhostMode(IntEnum):
    RANDOM = 0
    CHASE = 1
    SCATTER = 2
    FRIGHTENED = 3
    BLINKING = 4
    RETURNING = 5
    ENJAILED = 6


# -------- Constants --------
# GENERAL
RESET_LEVEL = 1 # the starting level, loaded when reset is called
RESET_TIMER = 40  # Timer for resetting the game after death
MAX_SCORE_DIGITS = 6 # Number of digits to display in the score
MAX_LIVE_COUNT = 8
# PELLETS_TO_COLLECT = 154  # Total pellets to collect in the maze (including power pellets)
PELLETS_TO_COLLECT = 4
INITIAL_LIVES = 2 # Number of starting bonus lives
BONUS_LIFE_LIMIT = 10000 # Maximum number of bonus lives
COLLISION_THRESHOLD = 8 # Contacts below this distance count as collision

# GHOST TIMINGS
CHASE_DURATION = 20*4*8 # Estimated for now, should be 20s  TODO: Adjust value
MAX_CHASE_OFFSET = CHASE_DURATION / 10 # Maximum value that can be added to the chase duration
SCATTER_DURATION = 7*4*8 # Estimated for now, should be 7s  TODO: Adjust value
MAX_SCATTER_OFFSET = SCATTER_DURATION / 10 # Maximum value that can be added to the scatter duration
FRIGHTENED_DURATION = 62*8 # Duration of power pellet effect in frames (x8 steps)
BLINKING_DURATION = 10*8
ENJAILED_DURATION = 120 # in steps
RETURN_DURATION = 2*8 # Estimated for now, should be as long as it takes the ghost to return from jail to the path TODO: Adjust value

# FRUITS
FRUIT_SPAWN_THRESHOLDS = jnp.array([50, 100]) # The original was more like ~50, ~100 but this version has a reduced number of pellets
FRUIT_WANDER_DURATION = 20*8 # Chosen randomly for now, should follow a hardcoded path instead

# POSITIONS
PPX0 = 8
PPX1 = 148
PPY0 = 20
PPY1 = 152
POWER_PELLET_POSITIONS = [[PPX0, PPY0], [PPX1, PPY0], [PPX0, PPY1], [PPX1, PPY1]]
INITIAL_GHOSTS_POSITIONS = jnp.array([[73, 54], [49, 78], [41, 78], [121, 78]])
INITIAL_PACMAN_POSITION = jnp.array([75, 102])
JAIL_POSITION = jnp.array([77, 70])
SCATTER_TARGETS = jnp.array([
    [MsPacmanMaze.WIDTH - 1, 0],                        # Upper right corner - Blinky
    [0, 0],                                             # Upper left corner - Pinky
    [MsPacmanMaze.WIDTH - 1, MsPacmanMaze.HEIGHT - 1],  # Lower right corner - Inky
    [0, MsPacmanMaze.HEIGHT - 1]                        # Lower left corner - Sue
])

# ACTIONS
DIRECTIONS = jnp.array([Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN])
ACTIONS = jnp.array([ # Translates generic JAXAtari actions to MsPacman actions
    (0, 0),   # NOOP
    (0, 0),   # FIRE (unused)
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (-1, 0),  # LEFT
    (0, 1)    # DOWN
])
INITIAL_ACTION = Action.LEFT # LEFT
INITIAL_LAST_ACTION = Action.LEFT # LEFT

# POINTS
PELLET_POINTS = 10
POWER_PELLET_POINTS = 50
FRUIT_REWARDS = jnp.array([100, 200, 500, 700, 1000, 2000, 5000]) # cherry, strawberry, orange, pretzel, apple, pear, banana
EAT_GHOSTS_BASE_POINTS = 200

# COLORS
PATH_COLOR = jnp.array([0, 28, 136], dtype=jnp.uint8)
WALL_COLOR = jnp.array([228, 111, 111], dtype=jnp.uint8)
PELLET_COLOR = WALL_COLOR  # Same color as walls for pellets
POWER_PELLET_SPRITE = jnp.tile(jnp.concatenate([PELLET_COLOR, jnp.array([255], dtype=jnp.uint8)]), (4, 7, 1))  # 4x7 sprite
PACMAN_COLOR = jnp.array([210, 164, 74, 255], dtype=jnp.uint8)
TRANSPARENT = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)


# -------- Entity classes --------
class LevelState(NamedTuple):
    index: chex.Array
    dofmaze: chex.Array # Precomputed degree of freedom maze layout
    pellets: chex.Array  # 2D grid of 0 (empty) or 1 (pellet)
    collected_pellets: chex.Array  # the number of pellets collected
    power_pellets: chex.Array
    loaded: chex.Array  # Whether the level is loaded

class GhostState(NamedTuple):
    type: GhostType
    position: chex.Array  # (x, y)
    action: chex.Array  # 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    mode: chex.Array
    timer: chex.Array

class PlayerState(NamedTuple):
    position: chex.Array  # (x, y)
    current_action: chex.Array # 0: NOOP, 1: FURE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    last_action : chex.Array  # 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    has_pellet: chex.Array  # Boolean indicating if pacman just collected a pellet
    eaten_ghosts: chex.Array  # timers indicating which ghosts have been eaten, when set to one, does not go down, indicate respawned ghost
    power_mode_timer: chex.Array # Timer for power mode, decrements every 8 steps
    death_timer: chex.Array  # Frames left in death animation

class FruitState(NamedTuple):
    type: chex.Array # Type of the fruit
    position: chex.Array # (x, y)
    action: chex.Array # 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    timer: chex.Array # Time until leaving through the closest tunnel
    exit: chex.Array # Tunnel number through which it will exit

class PacmanState(NamedTuple):
    level: LevelState
    player: PlayerState
    ghosts: Tuple[GhostState, GhostState, GhostState, GhostState] # 4 ghosts
    fruit: FruitState
    lives: chex.Array # Number of lives left
    score: chex.Array
    score_changed: chex.Array # indicates which score digit changed since the last step
    step_count: chex.Array

class PacmanObservation(NamedTuple):
    grid: chex.Array  # 2D array showing layout of walls, pellets, pacman, ghosts

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array


# -------- Game class --------
class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo]):
    def __init__(self):
        super().__init__()
        self.skipped = False
        self.frame_stack_size = 1
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
        ]

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for MsPacman.
        Actions are:
        0: NOOP
        1: FIRE
        2: UP
        3: RIGHT
        4: LEFT
        5: DOWN
        6: UPRIGHT
        7: UPLEFT
        8: DOWNRIGHT
        9: DOWNLEFT
        """
        return spaces.Discrete(10)

    def reset(self, key=None) -> Tuple[PacmanObservation, PacmanState]:
        return None, reset_game(RESET_LEVEL, INITIAL_LIVES, 0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array, key: chex.PRNGKey) -> tuple[
        PacmanObservation, PacmanState, jax.Array, jax.Array, PacmanInfo]:
        obs = None
        reward = 0.0
        done = jnp.array(False, dtype=jnp.bool_)
        info = PacmanInfo(score=state.score, done=done)

        # Skip current step if in reset state or level not loaded, so the renderer has time to react
        if not state.level.loaded and self.skipped == False:
            self.skipped = True
            return obs, state, reward, done, info
        self.skipped = False

        # If in death animation, decrement timer and freeze everything
        done = jnp.array(False, dtype=jnp.bool_)
        if state.player.death_timer > 0:
            new_death_timer = state.player.death_timer - 1
            new_state = freeze_game(state, new_death_timer)
            if new_death_timer == 0: # When timer reaches 0, reset entities if lives remain or set state to game over
                if state.lives >= 0:
                    new_state = reset_entities(state)
                else:
                    reward = state.score
                    done = jnp.array(True, dtype=jnp.bool_)
            return obs, new_state, reward, done, info

        # Pacman movement
        dofmaze = state.level.dofmaze 
        last_action = state.player.last_action
        action = last_pressed_action(action, state.player.current_action)
        possible_directions = available_directions(state.player.position, state.level.dofmaze)
        if action < 0 or action > len(ACTIONS) - 1: # Ignore illegal actions
            action = Action.NOOP
        if action != Action.NOOP and action != Action.FIRE and possible_directions[action - 2]:
            executed_action = action
            last_action = action
        else:
            # check for wall deadly_collision
            if state.player.current_action > 1 and stop_wall(state.player.position, dofmaze)[state.player.current_action - 2]:
                executed_action = Action.NOOP
            else:
                executed_action = state.player.current_action
        new_pacman_pos = state.player.position + ACTIONS[executed_action]
        new_pacman_pos = new_pacman_pos.at[0].set(new_pacman_pos[0] % 160)

        # Pellet handling
        collected_pellets = state.level.collected_pellets
        power_pellets = state.level.power_pellets
        power_mode_timer = state.player.power_mode_timer
        px, py = new_pacman_pos // 4
        ate_power_pill = False
        if px == 1:
            if (py == 3 or py == 4) and power_pellets[0]:
                power_pellets = state.level.power_pellets.at[0].set(False)
                ate_power_pill = True
            elif (py == 36 or py == 37) and power_pellets[2]:
                power_pellets = state.level.power_pellets.at[2].set(False)
                ate_power_pill = True
        elif px == 36:
            if (py == 3 or py == 4) and power_pellets[1]:
                power_pellets = state.level.power_pellets.at[1].set(False)
                ate_power_pill = True
            elif (py == 36 or py == 37) and power_pellets[3]:
                power_pellets = state.level.power_pellets.at[3].set(False)
                ate_power_pill = True
        if ate_power_pill:
            collected_pellets += 1
        elif state.player.power_mode_timer > 0 and state.step_count % 8 == 0:
            # Decrement power mode timer
            power_mode_timer = state.player.power_mode_timer - 1
        pellets = state.level.pellets
        # Check for pellet consumption
        has_pellet = jnp.array(False)
        x_offset = 5 if new_pacman_pos[0] < 75 else 1
        if new_pacman_pos[0] % 8 == x_offset and new_pacman_pos[1] % 12 == 6:
            x_pellets = (new_pacman_pos[0] - 2) // 8
            y_pellets = (new_pacman_pos[1] + 4) // 12
            if state.level.pellets[x_pellets, y_pellets]:
                has_pellet = jnp.array(True)
                pellets = state.level.pellets.at[x_pellets, y_pellets].set(False)
        score = state.score + jax.lax.select(has_pellet, PELLET_POINTS, 0)
        if has_pellet:
            print(f"Pacman collected a pellet at {new_pacman_pos}, score: {score}")
            collected_pellets += 1

        # Fruit handling
        fruit_type = state.fruit.type
        fruit_position = state.fruit.position
        fruit_action = state.fruit.action
        fruit_timer = state.fruit.timer
        fruit_exit = state.fruit.exit
        for threshold in FRUIT_SPAWN_THRESHOLDS:
            if collected_pellets == threshold and state.fruit.type == FruitType.NONE: # Spawn fruit
                fruit_type = get_level_fruit(state.level.index, key)
                fruit_position, fruit_action = get_random_tunnel(state.level.index, key)
                fruit_exit, _ = get_random_tunnel(state.level.index, key)
        if state.fruit.type != FruitType.NONE:
            if detect_collision(new_pacman_pos, state.fruit.position): # Consume fruit
                score = score + FRUIT_REWARDS[state.fruit.type]
                fruit_type = FruitType.NONE
                fruit_timer = jnp.array(FRUIT_WANDER_DURATION).astype(jnp.uint8)
            if state.fruit.timer == 0 and jnp.all(jnp.array(state.fruit.position) == jnp.array(state.fruit.exit)): # Remove fruit
                fruit_type = FruitType.NONE
                fruit_timer = jnp.array(FRUIT_WANDER_DURATION).astype(jnp.uint8)
            else:
                fruit_position, fruit_action, fruit_timer = fruit_step(state.fruit, dofmaze, key) # Move fruit

        # Execute ghost steps
        ghost_positions, ghosts_actions, ghosts_modes, ghosts_timers = ghosts_step(
            state.ghosts, state.player, ate_power_pill, dofmaze, key
        )

        # Ghost collision detection
        eaten_ghosts = state.player.eaten_ghosts
        deadly_collision = False
        for i in range(4): 
            if detect_collision(new_pacman_pos, ghost_positions[i]):
                if ghosts_modes[i] == GhostMode.FRIGHTENED or ghosts_modes[i] == GhostMode.BLINKING:  # If are frighted
                    # Ghost eaten
                    score += EAT_GHOSTS_BASE_POINTS * (2 ** eaten_ghosts)   # TODO: Reset eaten_ghosts after power mode ends
                    ghost_positions = ghost_positions.at[i].set(JAIL_POSITION)  # Reset eaten ghost position
                    ghosts_actions = ghosts_actions.at[i].set(Action.NOOP)  # Reset eaten ghost action
                    ghosts_modes = ghosts_modes.at[i].set(GhostMode.ENJAILED.value)
                    ghosts_timers = ghosts_timers.at[i].set(ENJAILED_DURATION)
                    eaten_ghosts = eaten_ghosts + 1
                else:
                    deadly_collision = True  # Collision with an already eaten and respawned ghost           
        else:
            deadly_collision = jnp.any(jnp.all(abs(new_pacman_pos - ghost_positions) < 8, axis=1))
        new_lives = state.lives - jnp.where(deadly_collision, 1, 0)
        new_death_timer = jnp.where(deadly_collision, RESET_TIMER, 0)

        # Progress to next level if all pellets collected
        level_idx = state.level.index
        if collected_pellets >= PELLETS_TO_COLLECT:
            level_idx += 1
            score += 500
            new_state = reset_game(level_idx, state.lives, score)
            print(f"Level completed! Starting level {level_idx}...")
            return obs, new_state, reward, done, info

        # Flag score change digit-wise
        if score != state.score:
            score_digits        = aj.int_to_digits(score, max_digits=MAX_SCORE_DIGITS)
            state_score_digits  = aj.int_to_digits(state.score, max_digits=MAX_SCORE_DIGITS)
            score_changed       = score_digits != state_score_digits
        else:
            score_changed       = jnp.array(False, dtype=jnp.bool_)

        # Update state
        new_state = PacmanState(
            level = LevelState(
                index=level_idx,
                dofmaze=dofmaze,
                pellets=pellets,
                collected_pellets=collected_pellets,
                power_pellets=power_pellets,
                loaded=jnp.array(True, dtype=jnp.bool_)
            ),
            player = PlayerState(
                position=new_pacman_pos,
                current_action=executed_action,
                last_action=last_action,
                has_pellet=has_pellet,
                eaten_ghosts=eaten_ghosts,
                power_mode_timer=power_mode_timer,
                death_timer=new_death_timer
            ),
            ghosts = tuple(
                GhostState(
                    type=GhostType(i),
                    position=ghost_positions[i],
                    action=ghosts_actions[i],
                    mode=ghosts_modes[i],
                    timer=ghosts_timers[i],
                ) for i in range(4)
            ),
            fruit = FruitState(
                type=fruit_type,
                position=fruit_position,
                action=fruit_action,
                timer=fruit_timer,
                exit=fruit_exit
            ),
            lives=new_lives,
            score=score,
            score_changed=score_changed,
            step_count=state.step_count + 1,
        )
        return obs, new_state, reward, done, info


# -------- Render class --------
class MsPacmanRenderer(AtraJaxisRenderer):
    """JAX-based MsPacman game renderer, optimized with JIT compilation."""

    def __init__(self):
        super().__init__()
        self.sprites = MsPacmanRenderer.load_sprites()

    def render_background(self, level: chex.Array, lives: chex.Array, score: chex.Array):
        """Reset the background for a new level."""
        self.SPRITE_BG = MsPacmanMaze.load_background(get_level_maze(level))
        self.SPRITE_BG = MsPacmanRenderer.render_lives(self.SPRITE_BG, lives, self.sprites["pacman"][1][1]) # Life sprite (right looking pacman)
        self.SPRITE_BG = MsPacmanRenderer.render_score(self.SPRITE_BG, score, jnp.arange(MAX_SCORE_DIGITS) >= (MAX_SCORE_DIGITS - get_digit_count(score)), self.sprites["score"])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PacmanState):
        # Render background for new game or level
        if not state.level.loaded:
            self.render_background(state.level.index, state.lives, state.score) # Render game over screen
        raster = self.SPRITE_BG

        # De-render pellets when consumed
        if state.player.has_pellet:
            pellet_x = state.player.position[0] + 3
            pellet_y = state.player.position[1] + 4
            for i in range(4):
                for j in range(2):
                    self.SPRITE_BG = self.SPRITE_BG.at[pellet_x+i, pellet_y+j].set(PATH_COLOR)
        # power pellets
        for i in range(2):
            pel_n = 2*i + ((state.step_count & 0b1000) >> 3) # Alternate power pellet rendering
            if state.level.power_pellets[pel_n]:
                pellet_x, pellet_y = POWER_PELLET_POSITIONS[pel_n]
                raster = aj.render_at(raster, pellet_x, pellet_y, POWER_PELLET_SPRITE)
        # Render pacman
        orientation = state.player.last_action - 2 # convert action to direction
        pacman_sprite = self.sprites["pacman"][orientation][((state.step_count & 0b1000) >> 2)]
        raster = aj.render_at(raster, state.player.position[0], state.player.position[1], 
                              pacman_sprite)
        ghosts_orientation = ((state.step_count & 0b10000) >> 4) # (state.step_count % 32) // 16

        for i, ghost in enumerate(state.ghosts):
            # Render frightened ghost
            if not (state.ghosts[i].mode == GhostMode.FRIGHTENED or state.ghosts[i].mode == GhostMode.BLINKING):
                g_sprite = self.sprites["ghost"][ghosts_orientation][i]
            elif state.ghosts[i].mode == GhostMode.BLINKING and ((state.step_count & 0b1000) >> 3):
                g_sprite = self.sprites["ghost"][ghosts_orientation][5] # white blinking effect
            else:
                g_sprite = self.sprites["ghost"][ghosts_orientation][4] # blue ghost
            raster = aj.render_at(raster, ghost.position[0], ghost.position[1], g_sprite)

        # Render fruit if present
        if state.fruit.type != FruitType.NONE:
            raster = MsPacmanRenderer.render_fruit(raster, state.fruit, self.sprites["fruit"])

        # Render score if changed
        if jnp.any(state.score_changed):
            self.SPRITE_BG = MsPacmanRenderer.render_score(self.SPRITE_BG, state.score, state.score_changed, self.sprites["score"])

        # Remove one life if a life is lost
        if state.player.death_timer == RESET_TIMER-1:
            self.SPRITE_BG = MsPacmanRenderer.render_lives(self.SPRITE_BG, state.lives, self.sprites["pacman"][1][1])
        return raster
    
    @staticmethod
    def render_score(raster, score, score_changed, digit_sprites, score_x=60, score_y=190, spacing=1, bg_color=jnp.array([0, 0, 0])):
        """
        Render the score on the raster at a fixed position.
        Only updates digits that have changed.
        """
        digits = aj.int_to_digits(score, max_digits=MAX_SCORE_DIGITS)
        for idx in range(len(digits)):
            if score_changed[idx]: 
                d_sprite    = digit_sprites[digits[idx]]
                bg_sprite   = jnp.full(d_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
                raster      = aj.render_at(raster, score_x + idx * (d_sprite.shape[1] + spacing), score_y, bg_sprite)
                raster      = aj.render_at(raster, score_x + idx * (d_sprite.shape[1] + spacing), score_y, d_sprite)
        return raster

    @staticmethod
    def render_lives(raster, current_lives, life_sprite, initial_lives=INITIAL_LIVES, life_x=12, life_y=182, spacing=4, bg_color=jnp.array([0, 0, 0])):
        """
        Render the lives on the raster at a fixed position.
        """
        bg_sprite = jnp.full(life_sprite.shape, jnp.append(bg_color, 255), dtype=jnp.uint8)
        for i in range(MAX_LIVE_COUNT):
            if i < current_lives:
                raster = aj.render_at(raster, life_x + i * (life_sprite.shape[1] + spacing), life_y, life_sprite)
            else:
                raster = aj.render_at(raster, life_x + current_lives * (life_sprite.shape[1] + spacing), life_y, bg_sprite)            
        return raster
    
    @staticmethod
    def render_fruit(raster, fruit: FruitState, fruit_sprites):
        raster = aj.render_at(raster, fruit.position[0], fruit.position[1], fruit_sprites[fruit.type])
        return raster
    
    @staticmethod
    def load_sprites() -> dict[str, Any]:
        SPRITE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/mspacman"
        sprites: Dict[str, Any] = {}
     
        # Helper function to load a single sprite frame
        def load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(SPRITE_PATH, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)
            return None
    
        # List of alls sprite names
        sprite_names = [ # The order here is important as it determines the sprites index
            'fruit_cherry','fruit_strawberry','fruit_orange',
            'fruit_pretzel','fruit_apple','fruit_pear','fruit_banana',
            'ghost_blinky','ghost_pinky','ghost_inky','ghost_sue','ghost_blue','ghost_white',
            'pacman_0','pacman_1','pacman_2','pacman_3',
            'score_0','score_1','score_2','score_3','score_4',
            'score_5','score_6','score_7','score_8','score_9'
        ]
        
        # Load raw sprites
        fruits = []
        ghosts = []
        pacmans = []
        score = []
        for name in sprite_names:
            loaded_sprite = load_sprite_frame(name)
            if loaded_sprite is not None:
                if "fruit" in name:
                    fruits.append(loaded_sprite)
                if "ghost" in name:
                    ghosts.append(loaded_sprite)
                elif "pacman" in name:
                    pacmans.append(loaded_sprite)
                elif "score" in name:
                    score.append(loaded_sprite)

        # Postprocess fruit sprites
        padded_fruits = aj.pad_to_match(fruits)
        jax_fruits = jnp.stack(padded_fruits)
        # Postprocess ghost sprites
        symmetric_ghosts = [jnp.flipud(ghost) for ghost in ghosts]
        jax_ghosts = [jnp.array(ghosts), jnp.array(symmetric_ghosts)]
        # Postprocess pacman sprites
        pacmans_right = [jnp.flipud(p) for p in pacmans]
        pacmans_up = [jnp.rot90(p, 3) for p in pacmans_right]
        pacmans_down = [jnp.rot90(p) for p in pacmans_right]
        jax_pacmans = [pacmans_up, pacmans_right, pacmans, pacmans_down]
        # Postprocess score sprites
        padded_score = aj.pad_to_match(score)
        jax_score = jnp.stack(padded_score)

        # Save resulting sprites
        sprites["fruit"] = jax_fruits
        sprites["ghost"] = jax_ghosts
        sprites["pacman"] = jax_pacmans
        sprites["score"] = jax_score
        return sprites


# -------- Helper functions --------
def last_pressed_action(action, prev_action):
    """
    Returns the last pressed action in cases where both actions are pressed
    """
    if action == Action.UPRIGHT:
        if prev_action == Action.UP:
            return Action.RIGHT
        else:
            return Action.UP
    elif action == Action.UPLEFT:
        if prev_action == Action.UP:
            return Action.LEFT
        else:
            return Action.UP
    elif action == Action.DOWNRIGHT:
        if prev_action == Action.DOWN:
            return Action.RIGHT
        else:
            return Action.DOWN
    elif action == Action.DOWNLEFT:
        if prev_action == Action.DOWN:
            return Action.LEFT
        else:
            return Action.DOWN
    else:
        return action


def dof(pos: chex.Array, dofmaze: chex.Array):
    """
    Degree of freedom of the object, can it move up, right, left, down
    """
    x, y = pos
    grid_x = (x+5)//4
    grid_y = (y+3)//4
    return dofmaze[grid_x][grid_y]


def available_directions(pos: chex.Array, dofmaze: chex.Array):
    """
    What direction Pacman or the ghosts can take when at an intersection.
    Returns a tuple of booleans (up, right, left, down) indicating if
    the character can move in that direction.
    The character can only change direction if it is on a vertical or horizontal grid.

    Arguments:
    pos -- (x, y) position of the character
    dofmaze -- precomputed degree of freedom for a maze level/layout

    Returns:
    A tuple of booleans (up, right, left, down) indicating if the 
    character can move in that direction.
    """
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    up, right, left, down = dof(pos, dofmaze)
    return up and on_vertical_grid, right and on_horizontal_grid, left and on_horizontal_grid, down and on_vertical_grid


def stop_wall(pos: chex.Array, dofmaze: chex.Array):
    x, y = pos
    on_vertical_grid = x % 4 == 1 # can potentially move up/down
    on_horizontal_grid = y % 12 == 6 # can potentially move left/right
    up, right, left, down = dof(pos, dofmaze)
    return not(up) and on_horizontal_grid, not(right) and on_vertical_grid, not(left) and on_vertical_grid, not(down) and on_horizontal_grid


def get_digit_count(number: chex.Array) -> int:
    """
    Returns the number of digits in a given decimal number.
    """
    return len(str(abs(number)))


def get_allowed_directions(position: chex.Array, direction: chex.Array, dofmaze: chex.Array):
    """
    Returns an array of all directions (indices) in which movement is possible.
    Turning is only allowed at the centre of each tile and reverting is not allowed.
    """
    allowed = []
    if position[0] % 4 == 1 and position[1] % 12 == 6: # on horizontal and vertical grid - tile centre
        possible = available_directions(position, dofmaze) 
        for i, can_go in zip(DIRECTIONS, possible):
            if can_go and (direction == 0 or i != reverse_direction(direction.item())):
                allowed.append(i)
    else:
        allowed.append(direction)
    return allowed


def get_chase_target(ghost: GhostType,
                     ghost_position: chex.Array, blinky_pos: chex.Array,
                     player_pos: chex.Array, player_dir: chex.Array) -> chex.Array:
    """
    Compute the chase-mode target for each ghost:
    0=Red (Blinky), 1=Pink (Pinky), 2=Blue (Inky), 3=Orange (Sue)
    """
    match ghost:
        case GhostType.BLINKY:
            # Target Pac-Man's current tile
            return player_pos
        case GhostType.PINKY:
            # Target 4 tiles ahead of Pac-Man
            return player_pos + 4*MsPacmanMaze.TILE_SCALE * ACTIONS[player_dir]
        case GhostType.INKY:
            # Target the tip of the vector from Blinky to two tiles ahead of Pac-Man, doubled
            two_ahead = player_pos + 2*MsPacmanMaze.TILE_SCALE * ACTIONS[player_dir]
            vect = two_ahead - blinky_pos
            return blinky_pos + 2 * vect
        case GhostType.SUE:
            # Target Pac-Man if >8 tiles away, else target corner
            dist = jnp.linalg.norm(ghost_position - player_pos)
            return jnp.where(dist > 8*MsPacmanMaze.TILE_SCALE, player_pos, SCATTER_TARGETS[GhostType.SUE])


def ghosts_step(ghosts: GhostState[4], player: PlayerState, ate_power_pill: chex.Array, dofmaze: chex.Array, key: chex.Array
                ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Step all ghosts. key can be a PRNGKey or None for deterministic.
    """
    n_ghosts = len(ghosts)
    keys = jax.random.split(key, n_ghosts)
    new_positions = []
    new_dirs = []
    modes = []
    timers = []
    chase_offset = jax.random.randint(keys[0], (), 0, MAX_CHASE_OFFSET)
    scatter_offset = jax.random.randint(keys[0], (), 0, MAX_SCATTER_OFFSET)
    for i in range(n_ghosts):
        chase_target = get_chase_target(ghosts[i].type, ghosts[i].position, ghosts[GhostType.BLINKY].position,
                                        player.position, player.current_action)
        pos, dir, mode, timer = ghost_step(ghosts[i], ate_power_pill, dofmaze, keys[i],
                                           chase_target, chase_offset, scatter_offset)
        new_positions.append(pos)
        new_dirs.append(dir)
        modes.append(mode)
        timers.append(timer)
    return jnp.stack(new_positions), jnp.stack(new_dirs), jnp.array(modes), jnp.array(timers)


def ghost_step(ghost: GhostState, ate_power_pill: chex.Array, dofmaze: chex.Array, key: chex.Array,
               chase_target: chex.Array, chase_offset: chex.Array, scatter_offset: chex.Array
               ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Step function for a single ghost. Never stops, never reverses, can change direction at intersections.
    """
    """
    GHOST BEHAVIOUR
    -   All Ghosts always aim for a specific target
    -   They try to minimize their horizontal and vertical distance to their target
    -   They prioritizes the longer one of those two distances - If equal, they choose randomly
    -   They can only change direction when they are on a vertical or horizontal grid
    -   They cannot reverse direction without a mode change

    MODES
    -   CHASE: Aim for their specific target
    -   SCATTER: Aim for their specific corner target
    -   FRIGHTENED: Move randomly
    -   Every mode change between CHASE and SCATTER and any mode change to FRIGHTENED causes a reversal of direction
    -   Mode changes happen on a timer with some additional randomnes for CHASE and SCATTER

    BLINKY (Red)
    -   Aims for Ms Pacman directly

    PINKY (Pink)
    -   Aims for a spot 4 tiles in front of Ms Pacman
    -   If Ms Pacman is looking up it aims for a spot 4 tiles up and 4 tiles left of Ms Pacman
        (due to a bug in the original game, not implemented here)

    INKY (Teal)
    -   Aims for a spot determined by Blinkys position relative to Ms Pacman
    -   Draw a straight line from Blinkys position to the spot 2 tiles in front of Ms Pacman
        and extend it equally as far onwards to determine the aiming point

    SUE (Orange)
    -   Aims for Ms Pacman directly if further away than 8 tiles (euclidian distance)
    -   When closer than 8 tiles it aims for the lower left corner

    SOURCES:
    https://www.classicarcadegaming.com/forums/index.php?topic=6701.0
    https://en.wikipedia.org/wiki/Ms._Pac-Man
    https://www.youtube.com/watch?v=sQK7PmR8kpQ
    """

    # 0) Initialize
    new_pos = ghost.position
    new_action = ghost.action
    new_mode = ghost.mode
    last_timer = ghost.timer
    new_timer = ghost.timer
    if ghost.timer > 0:
        new_timer = ghost.timer - 1
    skip = False

    # 1) Update ghost mode and timer
    if ate_power_pill and ghost.mode not in [GhostMode.ENJAILED, GhostMode.RETURNING]:
        new_mode = GhostMode.FRIGHTENED
        new_timer = FRIGHTENED_DURATION
        new_action = reverse_direction(ghost.action)
        skip = True
    elif new_timer == 0 and last_timer > 0:
        match ghost.mode:
            case GhostMode.CHASE:
                new_mode = GhostMode.SCATTER
                new_timer = SCATTER_DURATION + chase_offset
                new_action = reverse_direction(ghost.action)
                skip = True
            case GhostMode.SCATTER:
                new_mode = GhostMode.CHASE
                new_timer = CHASE_DURATION + scatter_offset
                new_action = reverse_direction(ghost.action)
                skip = True
            case GhostMode.ENJAILED:
                new_mode = GhostMode.RETURNING
                new_timer = RETURN_DURATION
                new_action = Action.UP
                skip = True
            case GhostMode.FRIGHTENED:
                new_mode = GhostMode.BLINKING
                new_timer = BLINKING_DURATION
            case GhostMode.BLINKING | GhostMode.RETURNING | GhostMode.RANDOM:
                new_mode = GhostMode.CHASE
                new_timer = CHASE_DURATION

    # 2) Update ghost action
    if skip or new_mode == GhostMode.ENJAILED or new_mode == GhostMode.RETURNING:
        pass
    else:
        # 2.2) Choose new direction
        allowed = get_allowed_directions(ghost.position, ghost.action, dofmaze)
        if not allowed: # If no direction is allowed - continue forward
            pass
        elif len(allowed) == 1: # If only one allowed direction - take it
            new_action = allowed[0]
        elif new_mode == GhostMode.FRIGHTENED or new_mode == GhostMode.BLINKING or new_mode == GhostMode.RANDOM or new_mode == GhostMode.RETURNING:
            new_action = jax.random.choice(key, jnp.array(allowed))
        else:
            if new_mode == GhostMode.SCATTER: # If not SCATTER, mode must be CHASE at this point
                chase_target = SCATTER_TARGETS[ghost.type]
            new_action = pathfind(ghost.position, ghost.action, chase_target, allowed, key)

    # 3) Update ghost position
    new_pos = ghost.position + ACTIONS[new_action]
    new_pos = new_pos.at[0].set(new_pos[0] % 160) # wrap horizontally
    return new_pos, new_action, new_mode, new_timer


def fruit_step(fruit: FruitState, dofmaze: chex.Array, key: chex.Array
               ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    allowed = get_allowed_directions(fruit.position, fruit.action, dofmaze)
    if not allowed:
        new_dir = fruit.action
    if len(allowed) == 1:
        new_dir = allowed[0]
    elif fruit.timer == 0:
        new_dir = pathfind(fruit.position, fruit.action, fruit.exit, allowed, key)
    else:
        new_dir = jax.random.choice(key, jnp.array(allowed))

    new_timer = fruit.timer
    if fruit.timer > 0:
        new_timer -= 1
    new_pos = jnp.array(fruit.position) + ACTIONS[new_dir]
    new_pos = new_pos.at[0].set(new_pos[0] % 160) # wrap horizontally
    return new_pos, new_dir, new_timer


def pathfind(position: chex.Array, direction: chex.Array, target: chex.Array, allowed: chex.Array, key: chex.Array):
    """
    Returns the direction which should be taken to approach the target.
    If multiple options exist the direction is chosen that minimizes the distance on the longer axis - horizontal or vertical.
    If both distances are equal or multiple options exist on the same axis, the direction is chosen randomly.
    """
    # Check allowed directions
    if len(allowed) == 0: # If no direction allowed - Continue forward
        return direction
    if len(allowed) == 1: # If one direction allowed - Take it
        return allowed[0]

    # If multiple directions allowed - Get cost of allowed directions
    cost = {}
    for dir in allowed:
        new_pos = position + ACTIONS[dir]
        cost[dir.item()] = jnp.abs(new_pos[0] - target[0]) + jnp.abs(new_pos[1] - target[1])
    min_cost = min(cost.values())
    min_dirs = jnp.array([k for k, v in cost.items() if v == min_cost])
    if len(min_dirs) == 1: # If one direction advantageous - Take it
        return min_dirs[0]
    
    # If multiple directions advantageous - Prioritize the longer axis
    horizontal_distance = jnp.abs(position[0] - target[0])
    vertical_distance = jnp.abs(position[1] - target[1])
    mask = jnp.full(len(min_dirs), False)
    if horizontal_distance >= vertical_distance:
        mask |= jnp.isin(min_dirs, jnp.array([Action.LEFT, Action.RIGHT]))
    if vertical_distance >= horizontal_distance:
        mask |= jnp.isin(min_dirs, jnp.array([Action.DOWN, Action.UP]))

    if jnp.sum(mask) == 0: # If no direction advantageous on longer axis - Choose randomly
        return jax.random.choice(key, min_dirs)
    if jnp.sum(mask) == 1: # If one direction advantageous on longer axis - Take it
        return jnp.squeeze(min_dirs[mask])
    else: # If multiple directions advantageous on longer or equal axis - Choose randomly with mask
        return jax.random.choice(key, min_dirs[mask])


def get_level_maze(level: chex.Array):
    if level < 0:
        raise ValueError("Invalid level!")
    elif level < 3:
        return 0
    elif level < 6:
        return 1
    elif level < 10:
        return 2
    elif level < 14:
        return 3
    else:
        if level % 4 == 0 or level % 4 == 1:
            return 2
        else:
            return 3


def get_level_fruit(level: chex, key: chex.Array):
    match level:
        case 1: return FruitType.CHERRY
        case 2: return FruitType.STRAWBERRY
        case 3: return FruitType.ORANGE
        case 4: return FruitType.PRETZEL
        case 5: return FruitType.APPLE
        case 6: return FruitType.PEAR
        case 7: return FruitType.BANANA
        case _: return jax.random.randint(key, (), 1, 8)


def get_random_tunnel(level: chex.Array, key: chex.Array):
    maze = get_level_maze(level)
    tunnel_heights = MsPacmanMaze.TUNNEL_HEIGHTS[maze]
    tunnels = [[0, tunnel_heights[0], Action.RIGHT], [MsPacmanMaze.WIDTH - 1, tunnel_heights[0], Action.LEFT], 
               [0, tunnel_heights[1], Action.RIGHT], [MsPacmanMaze.WIDTH - 1, tunnel_heights[1], Action.LEFT]]
    if tunnel_heights[1] == 0: # If the second element is 0, there is only one pair of tunnels
        tunnel_idx = jax.random.randint(key, (), 0, 2)
    else:
        tunnel_idx = jax.random.randint(key, (), 0, 4)

    tunnel = tunnels[tunnel_idx]
    tunnel_pos = tunnel[:2]
    tunnel_dir = tunnel[2]
    return tunnel_pos, tunnel_dir


def reverse_direction(dir_idx: chex.Array):
    INV_DIR = {2:5, 3:4, 4:3, 5:2}
    if dir_idx not in INV_DIR:
        return dir_idx
    return INV_DIR[dir_idx]
    

def detect_collision(position_1: chex.Array, position_2: chex.Array):
    if jnp.all(abs(jnp.array(position_1) - jnp.array(position_2)) < COLLISION_THRESHOLD):
        return True
    return False


def reset_level(level: chex.Array):
    maze = MsPacmanMaze.MAZES[get_level_maze(level)]
    return LevelState(
        index               = jnp.array(level, dtype=jnp.uint8),
        dofmaze             = MsPacmanMaze.precompute_dof(maze), # Precompute degree of freedom maze layout
        pellets             = jnp.copy(MsPacmanMaze.BASE_PELLETS),
        collected_pellets   = jnp.array(0).astype(jnp.uint8),
        power_pellets       = jnp.ones(4, dtype=jnp.bool_),
        loaded              = jnp.array(False, dtype=jnp.bool_)
    )

def reset_player():
    return PlayerState(
        position            = INITIAL_PACMAN_POSITION,
        current_action      = INITIAL_ACTION,
        last_action         = INITIAL_LAST_ACTION,
        has_pellet          = jnp.array(False),
        eaten_ghosts        = jnp.array(0).astype(jnp.uint8), # number of eaten ghost since power pellet consumed
        power_mode_timer    = jnp.array(0).astype(jnp.uint8), # Timer for power mode
        death_timer         = jnp.array(0).astype(jnp.uint8)
    )

def reset_ghosts():
    return tuple(
        GhostState(
            type        = jnp.array(GhostType(i)).astype(jnp.uint8),
            position    = INITIAL_GHOSTS_POSITIONS[i],
            action      = Action.NOOP,
            mode        = (jnp.array(GhostMode.RANDOM).astype(jnp.uint8) if i < 2 
                            else jnp.array(GhostMode.SCATTER).astype(jnp.uint8)),
            timer       = jnp.array(SCATTER_DURATION).astype(jnp.uint16),
        ) for i in range(4)
    )

def reset_fruit():
    return FruitState(
        type        = jnp.array(FruitType.NONE).astype(jnp.uint8),
        position    = jnp.zeros(2, dtype=jnp.int8),
        action      = Action.NOOP,
        timer       = jnp.array(FRUIT_WANDER_DURATION).astype(jnp.uint8),
        exit        = jnp.zeros(2, dtype=jnp.int8)
    )

def reset_game(level: chex.Array, lives: chex.Array, score: chex.Array):
    return PacmanState(
        level           = reset_level(level),
        player          = reset_player(),
        ghosts          = reset_ghosts(),
        fruit           = reset_fruit(),
        lives           = jnp.array(lives, dtype=jnp.int8),
        score           = jnp.array(score, dtype=jnp.uint32),
        score_changed   = jnp.arange(MAX_SCORE_DIGITS) >= (MAX_SCORE_DIGITS - get_digit_count(score)),
        step_count      = jnp.array(0, dtype=jnp.uint32),
    )

def reset_entities(state: PacmanState):
    return PacmanState(
        level = LevelState(
            index = state.level.index,
            dofmaze=state.level.dofmaze,
            pellets=state.level.pellets,
            collected_pellets=state.level.collected_pellets,
            power_pellets=state.level.power_pellets,
            loaded=state.level.loaded
        ),
        player          = reset_player(),
        ghosts          = reset_ghosts(),
        fruit           = reset_fruit(),
        lives           = state.lives,
        score           = state.score,
        score_changed   = state.score_changed,
        step_count      = state.step_count,
    )

def freeze_game(state: PacmanState, death_timer: chex.Array):
    return PacmanState(
        level           = state.level,
        player          = PlayerState(
            position            = state.player.position,
            current_action      = state.player.current_action,
            last_action         = state.player.last_action,
            has_pellet          = state.player.has_pellet,
            eaten_ghosts        = state.player.eaten_ghosts,
            power_mode_timer    = state.player.power_mode_timer,
            death_timer         = death_timer
        ),
        ghosts          = state.ghosts,
        fruit           = state.fruit,
        lives           = state.lives,
        score           = state.score,
        score_changed   = state.score_changed,
        step_count      = state.step_count,
    )
