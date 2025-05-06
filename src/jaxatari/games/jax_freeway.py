import os
from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment

# Actions
NOOP = 0
UP = 1
DOWN = 2


@dataclass
class GameConfig:
    """Game configuration parameters"""

    screen_width: int = 160
    screen_height: int = 210
    chicken_width: int = 6
    chicken_height: int = 8
    chicken_x: int = 40  # Fixed x position
    car_width: int = 8
    car_height: int = 10
    num_lanes: int = 10
    lane_spacing: int = 16
    car_speeds: List[float] = None
    lane_borders: List[int] = None
    top_border: int = 15
    top_path: int = 8
    bottom_border: int = 180

    def __post_init__(self):
        if self.car_speeds is None:
            # Upper 5 lanes move left (-), lower 5 lanes move right (+)
            # Value at i is the frequency in which car at lane i moves one pixel
            self.car_update = [
                -5,  # Lane 0
                -4,  # Lane 1
                -3,  # Lane 2
                -2,  # Lane 3
                -1,  # Lane 4
                1,  # Lane 5
                2,  # Lane 6
                3,  # Lane 7
                4,  # Lane 8
                5,  # Lane 9
            ]
        if self.lane_borders is None:
            # Upper 5 lanes move left (-), lower 5 lanes move right (+)
            # Value at i is the frequency in which car at lane i moves one pixel
            self.lane_borders = [
                self.top_border + self.top_path,  # Lane 0
                1 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 1
                2 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 2
                3 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 3
                4 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 4
                5 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 5
                6 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 6
                7 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 7
                8 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 8
                9 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 10
                10 * self.lane_spacing
                + (self.top_border + self.top_path)
                + 2,  # Lane 10
            ]


class GameState(NamedTuple):
    """Represents the current state of the game"""

    chicken_y: chex.Array
    cars: chex.Array  # Shape: (num_lanes, 2) for x,y positions
    score: chex.Array
    time: chex.Array
    cooldown: chex.Array  # Cooldown after collision
    walking_frames: chex.Array
    game_over: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class FreewayObservation(NamedTuple):
    chicken: EntityPosition
    car: EntityPosition
    score: jnp.ndarray


class FreewayInfo(NamedTuple):
    time: jnp.ndarray


class JaxFreeway(JaxEnvironment[GameState, FreewayObservation, FreewayInfo]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[FreewayObservation, GameState]:
        """Initialize a new game state"""
        # Start chicken at bottom
        chicken_y = self.config.bottom_border + self.config.chicken_height - 1
        # Initialize one car per lane
        cars = []
        for lane in range(self.config.num_lanes):
            lane_y = (
                self.config.lane_borders[lane]
                + int(self.config.lane_spacing / 2)
                - int(self.config.car_height / 2)
            )
            # Upper 5 lanes start from right, lower 5 lanes start from left
            if lane < 5:
                x = (
                    self.config.screen_width - self.config.car_width + 0
                )  # Start from right
            else:
                x = 0  # Start from left
            cars.append([x, lane_y])

        state = GameState(
            chicken_y=jnp.array(chicken_y),
            cars=jnp.array(cars),
            score=jnp.array(10),
            time=jnp.array(0),
            cooldown=jnp.array(0),
            walking_frames=jnp.array(0),
            game_over=jnp.array(False),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> tuple[FreewayObservation, GameState, float, bool, FreewayInfo]:
        """Take a step in the game given an action"""
        # Update chicken position if not in cooldown
        dy = jnp.where(
            jnp.logical_and(state.cooldown > 30, state.cooldown < 54), # throw the chicken back for 24 frames
            1.0,
            jnp.where(action == UP, -1.0, jnp.where(action == DOWN, 1.0, 0.0)),
        )

        dy = jnp.where(
            jnp.logical_and(state.cooldown > 0, state.cooldown < 30), # stun the chicken for 30 frames
            0.0,
            dy,
        )

        # add one to the walking frames if dy != 0, if it is 0 reset to 0
        new_walking_frames = jnp.where(dy != 0, state.walking_frames + 1, 0)

        # reset new_walking frames at 8
        new_walking_frames = jnp.where(new_walking_frames >= 8, 0, new_walking_frames)

        new_y = jnp.clip(
            state.chicken_y + dy,
            self.config.top_border,
            self.config.bottom_border + self.config.chicken_height - 1,
        )

        # Update car positions
        new_cars = state.cars
        for lane in range(self.config.num_lanes):
            # Update x position based on lane speed
            dir = (
                self.config.car_update[lane] / jnp.abs(self.config.car_update[lane])
            ).astype(jnp.int32)
            new_x = jax.lax.cond(
                jnp.equal(jnp.mod(state.time, self.config.car_update[lane]), 0),
                lambda: state.cars[lane, 0] + dir,
                lambda: state.cars[lane, 0],
            )

            # Wrap around screen
            new_x = jnp.where(
                self.config.car_update[lane] > 0,
                jnp.where(
                    new_x > self.config.screen_width, -self.config.car_width, new_x
                ),
                jnp.where(
                    new_x < -self.config.car_width, self.config.screen_width, new_x
                ),
            )

            new_cars = new_cars.at[lane, 0].set(new_x)

        # Check for collisions
        def check_collision(car_pos):
            car_x, car_y = car_pos
            return jnp.logical_and(
                self.config.chicken_x < car_x + self.config.car_width,
                jnp.logical_and(
                    self.config.chicken_x + self.config.chicken_width > car_x,
                    jnp.logical_and(
                        state.chicken_y - self.config.chicken_height < car_y,
                        state.chicken_y > car_y - self.config.car_height,
                    ),
                ),
            )

        # Check collisions for all cars
        collisions = jax.vmap(check_collision)(new_cars)
        any_collision = jnp.any(collisions)
        any_collision = jax.lax.cond(
            state.cooldown > 0, lambda _: False, lambda _: any_collision, operand=None
        )

        # Update cooldown
        new_cooldown = jnp.where(
            any_collision,
            24 + 30,  # Set cooldown frames after collision (24 frames of flying backwards, 30 frames of being "stunned")
            jnp.maximum(0, state.cooldown - 1),
        )

        # Update score if chicken reaches top
        new_score = jnp.where(
            new_y <= self.config.top_border, state.score + 1, state.score
        )

        # Reset chicken position if scored
        new_y = jnp.where(
            new_y <= self.config.top_border,
            self.config.bottom_border + self.config.chicken_height - 1,
            new_y,
        )

        # Update time
        new_time = state.time + 1

        # Check game over (optional: could be based on time or score limit)
        game_over = jnp.where(
            new_time >= 255 * 32,  # 2 minute time limit
            jnp.array(True),
            state.game_over,
        )

        new_state = GameState(
            chicken_y=new_y,
            cars=new_cars,
            score=new_score,
            time=new_time,
            cooldown=new_cooldown,
            walking_frames=new_walking_frames,
            game_over=game_over,
        )
        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        # create chicken
        chicken = EntityPosition(
            x=jnp.array(self.config.chicken_x),
            y=state.chicken_y,
            width=jnp.array(self.config.chicken_width),
            height=jnp.array(self.config.chicken_height),
        )

        # create cars
        cars = jnp.zeros((self.config.num_lanes, 4))
        for i in range(self.config.num_lanes):
            car_pos = state.cars.at[i].get()
            cars = cars.at[i].set(
                jnp.array(
                    [
                        car_pos.at[0].get(),  # x position
                        car_pos.at[1].get(),  # y position
                        self.config.car_width,  # width
                        self.config.car_height,  # height
                    ]
                )
            )
        return FreewayObservation(chicken=chicken, car=cars, score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState) -> FreewayInfo:
        return FreewayInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        return state.game_over


from jaxatari.renderers import AtraJaxisRenderer
import jaxatari.rendering.atraJaxis as aj

class FreewayRenderer(AtraJaxisRenderer):

    def __init__(self):
        super().__init__()
        self.sprites = self._load_sprites()
        self.game_config = GameConfig()

    def _load_sprites(self):
        """Load all sprites required for Freeway rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(MODULE_DIR, "sprites/freeway/")

        sprites: Dict[str, Any] = {}

        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            return frame.astype(jnp.uint8)

        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            'background',
            'player_hit', 'player_walk', 'player_idle',
            'car_dark_red', 'car_light_green', 'car_dark_green', 'car_light_red', 'car_blue', 'car_brown',
            'car_light_blue', 'car_red', 'car_green', 'car_yellow',
        ]

        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite

        # pad the player sprites since they are used interchangably
        ape_sprites = aj.pad_to_match([sprites['player_hit'], sprites['player_walk'], sprites['player_idle']])

        sprites['player_hit'] = ape_sprites[0]
        sprites['player_walk'] = ape_sprites[1]
        sprites['player_idle'] = ape_sprites[2]

        # --- Load Digit Sprites ---
        # Score digits
        score_digit_path = os.path.join(sprite_path, 'score_{}.npy')
        digits = aj.load_and_pad_digits(score_digit_path, num_chars=10)
        sprites['score'] = digits

        # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """Render the game state to a raster image."""
        raster = jnp.zeros((160, 210, 3), dtype=jnp.uint8)

        # draw the background
        background = aj.get_sprite_frame(self.sprites['background'], 0)

        raster = aj.render_at(raster,0, 0, background)

        # draw fixed 2nd chicken at x=110 and y=self.config.bottom_border + self.config.chicken_height - 1
        chicken_idle = aj.get_sprite_frame(self.sprites['player_idle'], 0)
        chicken_walk = aj.get_sprite_frame(self.sprites['player_walk'], 0)
        chicken_hit = aj.get_sprite_frame(self.sprites['player_hit'], 0)
        raster = aj.render_at(raster, 110, self.game_config.bottom_border + self.game_config.chicken_height - 1, chicken_idle)

        # select a frame based on the walking frames (0-3 for walk, 4-7 for idle, repeat)
        use_idle = state.walking_frames < 4
        chicken = jax.lax.cond(
            use_idle,
            lambda: chicken_idle,
            lambda: chicken_walk,
        )

        is_hit = state.cooldown > 0

        chicken = jax.lax.cond(
            jnp.logical_and(is_hit, jnp.logical_or((state.cooldown % 8) < 4, state.cooldown < 30)), # check if the cooldown is either in the alternating 4 frame window or in the stun phase
            lambda: chicken_hit,
            lambda: chicken
        )

        raster = aj.render_at(raster, self.game_config.chicken_x, state.chicken_y, chicken)

        # render the cars in the correct color (starting from the top: dark red, light green, dark green, light red, blue, brown, light blue, red, green, yellow)
        dark_red = aj.get_sprite_frame(self.sprites['car_dark_red'], 0)
        raster = aj.render_at(raster, state.cars[0, 0], state.cars[0, 1], dark_red)

        light_green = aj.get_sprite_frame(self.sprites['car_light_green'], 0)
        raster = aj.render_at(raster, state.cars[1, 0], state.cars[1, 1], light_green)

        dark_green = aj.get_sprite_frame(self.sprites['car_dark_green'], 0)
        raster = aj.render_at(raster, state.cars[2, 0], state.cars[2, 1], dark_green)

        light_red = aj.get_sprite_frame(self.sprites['car_light_red'], 0)
        raster = aj.render_at(raster, state.cars[3, 0], state.cars[3, 1], light_red)

        blue = aj.get_sprite_frame(self.sprites['car_blue'], 0)
        raster = aj.render_at(raster, state.cars[4, 0], state.cars[4, 1], blue)

        brown = aj.get_sprite_frame(self.sprites['car_brown'], 0)
        raster = aj.render_at(raster, state.cars[5, 0], state.cars[5, 1], brown)

        light_blue = aj.get_sprite_frame(self.sprites['car_light_blue'], 0)
        raster = aj.render_at(raster, state.cars[6, 0], state.cars[6, 1], light_blue)

        red = aj.get_sprite_frame(self.sprites['car_red'], 0)
        raster = aj.render_at(raster, state.cars[7, 0], state.cars[7, 1], red)

        green = aj.get_sprite_frame(self.sprites['car_green'], 0)
        raster = aj.render_at(raster, state.cars[8, 0], state.cars[8, 1], green)

        yellow = aj.get_sprite_frame(self.sprites['car_yellow'], 0)
        raster = aj.render_at(raster, state.cars[9, 0], state.cars[9, 1], yellow)

        # ----------- SCORE -------------
        # Define score positions and spacing
        player_score_rightmost_digit_x = 49  # X position for the START of the player's rightmost digit (or single digit)
        enemy_score_rightmost_digit_x = 114  # X position for the START of the enemy's single '0' digit
        score_y = 5
        score_spacing = 8  # Spacing between digits (should match digit width ideally)
        max_score_digits = 2

        # Get digit sprites
        digit_sprites = self.sprites.get('score', None)

        # Define the function to render scores if sprites are available
        def render_scores(raster_to_update):
            # --- Player Score (Left) ---
            # Convert score to digit indices (always 2, e.g., 0 -> [0,0], 10 -> [1,0])
            player_score_digits_indices = aj.int_to_digits(state.score, max_digits=max_score_digits)

            # Determine parameters based on score magnitude
            is_player_single_digit = state.score < 10

            # Start index in player_score_digits_indices:
            # - If single digit (e.g., score=5 -> indices=[0,5]), start reading at index 1 to get the '5'.
            # - If double digit (e.g., score=12 -> indices=[1,2]), start reading at index 0 to get '1' then '2'.
            player_start_index = jax.lax.select(is_player_single_digit, 1, 0)

            # Number of digits to render: 1 for single, 2 for double
            player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)

            # Calculate the starting X position for rendering:
            # - If single digit, render starts at player_score_rightmost_digit_x.
            # - If double digit, render starts score_spacing pixels *before* player_score_rightmost_digit_x,
            #   so the second digit ('0' in '10') aligns correctly at player_score_rightmost_digit_x.
            player_render_x = jax.lax.select(is_player_single_digit,
                                             player_score_rightmost_digit_x,
                                             player_score_rightmost_digit_x - score_spacing)

            # Render player score using selective rendering
            raster_updated = aj.render_label_selective(raster_to_update, player_render_x, score_y,
                                                       player_score_digits_indices, digit_sprites[0],
                                                       player_start_index, player_num_to_render,
                                                       spacing=score_spacing)

            # --- Enemy Score (Right - rendering a Dummy '0' since the right player is not playable) ---
            enemy_score = 0
            enemy_score_digits_indices = aj.int_to_digits(enemy_score, max_digits=max_score_digits)  # [0, 0]
            # Parameters for single digit '0'
            enemy_start_index = 1  # Read the second '0' from indices [0, 0]
            enemy_num_to_render = 1  # Render only one digit
            # Render the enemy '0' starting at its designated rightmost position
            enemy_render_x = enemy_score_rightmost_digit_x

            # Render enemy score using selective rendering
            raster_final = aj.render_label_selective(raster_updated, enemy_render_x, score_y,
                                                     enemy_score_digits_indices, digit_sprites[0],
                                                     enemy_start_index, enemy_num_to_render,
                                                     spacing=score_spacing)  # Spacing doesn't matter here
            return raster_final

        # Render scores conditionally
        raster = jax.lax.cond(
            digit_sprites is not None,
            render_scores,
            lambda r: r,
            raster
        )

        # Force the first 8 columns (x=0 to x=7) to be black (KEEP THIS PART)
        bar_width = 8
        raster = raster.at[0:bar_width, :, :].set(0)

        return raster

def main():
    pygame.init()

    # Initialize game and renderer
    game = JaxFreeway()
    renderer = FreewayRenderer()
    scaling = 4

    screen = pygame.display.set_mode((160 * scaling, 210 * scaling))
    pygame.display.set_caption("Freeway")

    jitted_step = jax.jit(game.step)
    jitted_render = jax.jit(renderer.render)
    jitted_reset = jax.jit(game.reset)

    init_obs, state = jitted_reset()

    # Setup game loop
    clock = pygame.time.Clock()
    running = True
    done = False

    while running and not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = UP
        elif keys[pygame.K_s]:
            action = DOWN
        else:
            action = NOOP

        # Update game state
        obs, state, reward, done, info = (jitted_step(state, action))

        # Render
        frame = jitted_render(state)

        aj.update_pygame(screen, frame, scaling, 160, 210)

        # Cap at 60 FPS
        clock.tick(30)

    # If game over, wait before closing
    if done:
        pygame.time.wait(2000)

    pygame.quit()

if __name__ == "__main__":
    main()
