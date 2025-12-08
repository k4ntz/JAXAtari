from typing import NamedTuple, Tuple
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
import jax.numpy as jnp
import chex
import jax.lax


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    SPEED: int = 4
    PIXEL_COLOR: Tuple[int, int, int] = (255, 255, 255)  # white

    PLAYER_SIZE: Tuple[int, int] = (5, 10)

    PLAYER_LIVES: int = 3

    # Missile constants
    BULLET_SIZE: Tuple[int, int] = (1, 2)
    BULLET_SPEED: int = 8
    AMMO_SUPPLY: int = 300 # frames until ammo runs out

    MAX_LASER_FLASHES: int = 3
    LASER_FLASH_COOLDOWN: int = 60  # frames

    # Creature constants

    INACTIVE: int = 0
    ACTIVE: int = 1

    # Creature Types
    SNAKE: int = 0
    SCORPION: int = 1
    BAT: int = 2
    TURTLE: int = 3
    JACKEL: int = 4
    CONDOR: int = 5
    LION: int = 6
    MOTH: int = 7
    VIRUS: int = 8
    MONKEY: int = 9
    MYSTERY: int = 10
    WEAPON: int = 11
    
    CREATURE_SPEED: chex.Array = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])  # speed for each creature type
    CREATE_POINTS: chex.Array = np.array([1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 0, 3])  # points for each creature type

    MAX_CREATURES: int = 3 # max number of creatures on screen at once


# ---------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------
class TutankhamState(NamedTuple):
    player_x: int
    player_y: int
    player_lives: int

    bullet_state: chex.Array #(, 4) array with (x, y, bullet_rotation, bullet_active)
    laser_flash_count: int # number of laser flashes that can be fired
    laser_flash_cooldown: int # cooldown timer for next laser flash
    amonition_timer: int # if timer runs out, player can not fire again

    creature_states: chex.Array # (3, 5) array with (x, y, creature_type, active) for each creature
    last_creature_spawn: int = 0  # time since last creature spawn



# ---------------------------------------------------------------------
# Renderer (No JAX)
# ---------------------------------------------------------------------
class TutankhamRenderer(JAXGameRenderer):
    def __init__(self):
        super().__init__()
        self.consts = TutankhamConstants()

    def render(self, state: TutankhamState) -> np.ndarray:
        frame = np.zeros(
            (self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=np.uint8
        )

        # -------------------------
        # Draw player
        # -------------------------
        x = min(max(state.player_x, 0), self.consts.WIDTH - 1)
        y = min(max(state.player_y, 0), self.consts.HEIGHT - 1)
        frame[y, x] = self.consts.PIXEL_COLOR

        # -------------------------
        # Draw bullets (1×1 pixels)
        # -------------------------
        bx, by, rot, active = state.bullet_state
        if active:
            # Clip
            #if 0 <= bx < self.consts.WIDTH and 0 <= by < self.consts.HEIGHT:
            frame[int(by), int(bx)] = self.consts.PIXEL_COLOR

        # -------------------------
        # Draw creatures (1×1 pixels)
        for i in range(self.consts.MAX_CREATURES):
            cx, cy, creature_type, active = state.creature_states[i]
            if active == self.consts.ACTIVE:
                # Clip
                if 0 <= cx < self.consts.WIDTH and 0 <= cy < self.consts.HEIGHT:
                    frame[int(cy), int(cx)] = self.consts.PIXEL_COLOR

        return frame


# ---------------------------------------------------------------------
# Environment (No JAX)
# ---------------------------------------------------------------------
class JaxTutankham(JaxEnvironment):
    def __init__(self):
        consts = TutankhamConstants()
        super().__init__(consts)
        self.renderer = TutankhamRenderer()
        self.consts = consts

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ]

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2
        player_lives = self.consts.PLAYER_LIVES
        amonition_timer = self.consts.AMMO_SUPPLY
        bullet_state = np.array([0, 0, 0, False])
        creature_states = np.zeros((self.consts.MAX_CREATURES, 4))  # (x, y, creature_type, active)
        last_creature_spawn = 0
        laser_flash_count = self.consts.MAX_LASER_FLASHES
        laser_flash_cooldown = self.consts.LASER_FLASH_COOLDOWN
        

        state = TutankhamState(player_x=start_x, 
                                player_y=start_y,
                                player_lives=player_lives,
                                bullet_state=bullet_state, 
                                amonition_timer=amonition_timer,
                                creature_states=creature_states,
                                last_creature_spawn=last_creature_spawn,
                                laser_flash_count=laser_flash_count,
                                laser_flash_cooldown=laser_flash_cooldown
                               )
        return state, state #TODO: (EnvObs, EnvState)

    # Player Step
    def player_step(
            self,
            player_x,
            player_y,
            action
    ):

        if action == Action.LEFT:
            player_x -= self.consts.SPEED
        elif action == Action.RIGHT:
            player_x += self.consts.SPEED
        elif action == Action.UP:
            player_y -= self.consts.SPEED
        elif action == Action.DOWN:
            player_y += self.consts.SPEED

        # Clip bounds
        player_x = max(0, min(player_x, self.consts.WIDTH - 1))
        player_y = max(0, min(player_y, self.consts.HEIGHT - 1))

        return player_x, player_y
    
    
    #Bullet Step
    def bullet_step(self, bullet_state, player_x, player_y, amonition_timer, action):

        def get_rotation(action):
            if action == Action.RIGHTFIRE: return 1
            if action == Action.LEFTFIRE: return -1
            return 0  # default if firing up/down/etc

        space = (
                (action == Action.LEFTFIRE)
                or (action == Action.RIGHTFIRE)
            )


        new_bullet = bullet_state.copy() #array with (x, y, bullet_rotation, bullet_active)

        
        # --- update existing bullets ---
        if bullet_state[3]:
            bullet_x = bullet_state[0] + self.consts.BULLET_SPEED * bullet_state[2]
            new_bullet[0] = bullet_x

            # Deactivate if out of bounds
            if not (0 <= bullet_x < self.consts.WIDTH):
                new_bullet = [0, 0, 0, False]


        # --- firing logic ---
        bullet_rdy = not bullet_state[3]


        if space and bullet_rdy and amonition_timer > 0:
            new_bullet = np.array([player_x, player_y, get_rotation(action), True])


        amonition_timer -= 1 # TODO: adjust amonition timer
        
        return new_bullet, amonition_timer
    

    def laser_flash_step(self, creature_states, laser_flash_cooldown, laser_flash_count, action):
        
        laser_flash_cooldown = max(laser_flash_cooldown -1 , 0)
        if action == Action.UPFIRE and laser_flash_count > 0 and laser_flash_cooldown == 0:
            new_laser_flash_count = laser_flash_count - 1
            new_laser_flash_cooldown = self.consts.LASER_FLASH_COOLDOWN

            new_creature_states = creature_states.copy()
            new_creature_states[:, -1] = 0  # set all creatures to inactive
            
            return new_creature_states, new_laser_flash_cooldown, new_laser_flash_count

        return creature_states, laser_flash_cooldown, laser_flash_count
    
    # creature step
    def creature_step(self, creature_states, last_creature_spawn):
        
        def spawn_creature(creature_states, last_creature_spawn):
            # last_creature_spawn = vergangene Zeit (in Sekunden) seit letztem Frame

            # Parameter # TODO: REMOVE HARDCODED VALUES
            MAX_ACTIVE = 3
            GROWTH = 0.0003           # Chance steigt pro Sekunde um 5%
            MAX_PROB = 0.8          # Deckelung (optional)

            # 1) aktive Creatures zählen
            active_count = np.sum(creature_states[:, 3] == self.consts.ACTIVE)
            if active_count >= MAX_ACTIVE:
                return creature_states, last_creature_spawn  # nichts tun, Limit erreicht

            # 2) Spawn-Timer erhöhen
            last_creature_spawn += 1

            # 3) Spawn-Chance berechnen
            spawn_chance = last_creature_spawn * GROWTH
            spawn_chance = min(spawn_chance, MAX_PROB)

            # 4) treffen wir den Zufall?
            if np.random.random() > spawn_chance:
                return creature_states, last_creature_spawn  # nein → nichts machen

            # 5) Ja → wir spawnen einen!
            new_creature_states = creature_states.copy()

            for i in range(self.consts.MAX_CREATURES):
                x, y, creature_type, active = creature_states[i]
                if active == self.consts.INACTIVE: # TODO: find correct spawner x,y
                    new_x = 0
                    new_y = np.random.randint(0, self.consts.HEIGHT)
                    new_creature_type = np.random.randint(0, len(self.consts.CREATE_POINTS))
                    new_creature_states[i] = np.array([new_x, new_y, new_creature_type, self.consts.ACTIVE])

                    # Timer zurücksetzen: Start von vorne
                    last_creature_spawn = 0
                    break

            return new_creature_states, last_creature_spawn
        
        def move_creature(creature_state):
            x, y, creature_type, active = creature_state

            if active:
                speed = self.consts.CREATURE_SPEED[int(creature_type)]
                x += speed  # Move right for simplicity

                # Deactivate if out of bounds
                if x >= self.consts.WIDTH:
                   active = self.consts.INACTIVE

            return np.array([x, y, creature_type, active])
        

        
        
        creature_states, last_creature_spawn = spawn_creature(creature_states, last_creature_spawn)
        new_creature_states = creature_states.copy()

        for i in range(self.consts.MAX_CREATURES):
            new_creature_states[i] = move_creature(creature_states[i])

        return new_creature_states, last_creature_spawn
    
    # TODO:
    def detect_creature_deaths(old_states, new_states):
        """Erkennt Todes-Events: active: 1 → 0."""
        old_active = old_states[:, 3] == 1.0
        new_active = new_states[:, 3] == 1.0
        died = jnp.logical_and(old_active, jnp.logical_not(new_active))
        return died  # shape: (MAX_CREATURES,)

    def update_score(score, deaths, new_states, creature_points):
        creature_types = new_states[:, 2].astype(int)
        gained_points = jnp.sum(creature_points[creature_types] * deaths)
        return score + gained_points





    # -----------------------------
    # Step logic (pure Python)
    # -----------------------------
    def step(self, state: TutankhamState, action: int):

        player_x, player_y = state.player_x, state.player_y
        bullet_state = state.bullet_state
        creature_states = state.creature_states
        laser_flash_count = state.laser_flash_count
        last_creature_spawn = state.last_creature_spawn
        laser_flash_cooldown = state.laser_flash_cooldown
        amonition_timer = state.amonition_timer

        player_x, player_y = self.player_step(player_x, player_y, action)

        bullet_state, amonition_timer =self.bullet_step(bullet_state, player_x, player_y, amonition_timer, action)

        creature_states, laser_flash_cooldown, laser_flash_count = self.laser_flash_step(creature_states, laser_flash_cooldown, laser_flash_count, action)

        creature_states, last_creature_spawn = self.creature_step(creature_states, last_creature_spawn)
        
        player_lives = state.player_lives # TODO: implement player lives logic

        state = TutankhamState(player_x=player_x, 
                               player_y=player_y,
                               player_lives=player_lives,
                               bullet_state=bullet_state, 
                               amonition_timer=amonition_timer, 
                               creature_states=creature_states,
                               last_creature_spawn=last_creature_spawn,
                               laser_flash_count=laser_flash_count,
                               laser_flash_cooldown=laser_flash_cooldown
                               )

        reward = 0.0
        
        #done = self._get_done(state) #TODO: uncomment later
        done = False
        info = None

        return state, state, reward, done, info

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self, state: TutankhamState) -> np.ndarray:
        return self.renderer.render(state)

    # -----------------------------
    # Action & Observation Space
    # -----------------------------
    def action_space(self):
        return Discrete(len(self.action_set))

    def observation_space(self):
        return Box(
            low=0,
            high=max(self.consts.WIDTH, self.consts.HEIGHT),
            shape=(2,),
            dtype=np.int32,
        )
    
    def _get_done(self, state: TutankhamState) -> bool:
        if state.player_lives <= 0: # Game Over
            return False
        # TODO: beat final level condition

        return True