from argparse import Action
import os
import jax
import jax.lax
import jax.numpy as jnp
import chex
from typing import NamedTuple, Tuple, List 
from functools import partial
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action



def _get_default_asset_config() -> tuple:
    """
    Use this to register every unique base .npy file/
    Returns the default declarative asset manifest for Gopher.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'group', 'files': ['farmer_default.npy', 'farmer_shovel_middle.npy', 'farmer_shovel_top.npy']},
        {'name': 'gopher', 'type': 'group', 'files': ['gopher_walk_left_legsclose.npy', 'gopher_walk_left_legsopen.npy', 'gopher_walk_right_legsclosed.npy', 'gopher_walk_right_legsopen.npy']},
        {'name': 'carrot', 'type': 'single', 'file': 'carrot.npy'},
        {'name': 'hole_tile_dug', 'type': 'digits', 'pattern': 'hole_tile_dug.npy'},
        {'name': 'tunnel_tile_dug', 'type': 'digits', 'pattern': 'tunnel_tile_dug.npy'}
    )


class GopherConstants(NamedTuple):
    """
    Top screen - top river: 0-145
    Top river - top ground: 146-160
    top ground - top tunnel: 161-182
    Carrots size: (27, 8)
    Hole width: 8
    """


    # Frame size(gamesize)
    BLOCK_SIZE: int = 5
    WIDTH: int = 160                                   
    HEIGHT: int = 210
    

    # Element size
   
    GROUND_HEIGHT: int = 24
    NUM_COLUMNS: int = 42
    NUM_ROWS: int = 30
    MAX_DEPTH: int = 5 
    # Origin at top left
    # Depth meaning:
    # 0 → Carrot Block A
    # 1 → Carrot Block B
    # 2 → Middle Block
    # 3 → Tunnel Top
    # 4 → Tunnel Bottom
    PLAYER_SIZE: Tuple[int, int] = (13, 50)
    GOPHER_SIZE: Tuple[int, int] = (8, 8)
    CARROT_SIZE: Tuple[int, int] = (7, 15)
    SEED_SIZE: Tuple[int, int] = (1, 1)
    
    GROUND_REPAIR_AMOUNT: int = 1
    BLOCK_PIXEL_HEIGHT: int = BLOCK_SIZE                      # each layer is 4px
    COLUMN_HEIGHT_PX: int = (MAX_DEPTH + 1) * BLOCK_SIZE      # 6 x 4 = 24
    
    TUNNEL_TOP_Y: int = 108
    TUNNEL_BOTTOM_Y: int = 116
    TUNNEL_REPAIR_AMOUNT: int = 2
    LEFT_WALL: int = 0
    RIGHT_WALL: int = 168

    # Element colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (107, 154, 98)
    RIVER_COLOR: Tuple[int, int, int] = (184, 249, 214)
    GROUND_COLOR: Tuple[int, int, int] = (179, 157, 97)
    HOLE_COLOR: Tuple[int, int, int] = (213, 181, 115)
    CARROT_COLOR: Tuple[int, int, int] = (161, 99, 61)
    # DUCK_COLOR: Tuple[int, int, int] = ()
    
    # Initial set up
    NUM_CARROT: int = 3
    CARROT_POSITION: Tuple[int, int, int] = (60, 76, 92)       # (60,151), (76, 151), (92,151)
    PLAYER_SPEED: int = 1
    GOPHER_SPEED1: int = 1                                      # smart gopher speed
    GOPHER_SPEED2: int = 2                                      # very smart gopher speed
    PLAYER_START_X: int = 80
    PLAYER_START_Y: int = 96
    GOPHER_START_X: int = 160
    GOPHER_START_Y: int = 182
    # CARROT_DEPTH: int = 4

    # Rules
    TUNNEL_REPAIR_AMOUNT: int = 4
    MAX_GOPHERS: int = 1
    HOLE_POSITION: List = {12, 28, 44, 108, 124, 140}
    TUNNEL_POSITION: List = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 
                             40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 
                             80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 
                             120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 
                             160, 164,}

    ASSET_CONFIG: tuple = _get_default_asset_config()


class GopherState(NamedTuple):
    """
    Game state including player, gopher, duck, seed
    """
    
    # Player position and movement
    player_x: chex.Array                          
    # player_y: chex.Array  
    player_speed:chex.Array 

    bonk_timer: chex.Array                    # Current frame of animation: default->middle: 3 frames, middle -> top: 3 frames, top -> default: 5 frames

    # Gopher position and movement
    gopher_position: chex.Array            
    gopher_direction_x: chex.Array            # -1 left, +1 right
    # gopher_direction_y:chex.Array           # 1 for going down, -1 for going up
    # gopher_up:chex.Array                    # 1 for pop up, 0 for in ground
    # gopher_running:chex.Array               # 1 for running to steal, 0 for normal
    
    # gopher_bonked:chex.Array                # 0 for not bonked, 1 for bonked
    # hole_state: chex.Array                  # shape [6:3] each hole has 3 layers(3 ground layers), one layer is horizontally two blocks, count from bottom to up: 0, 1, 2
    # tunnel_state:chex.Array                 # shape [42:] 42 entries for tunnel, one entry is vertically two blocks
    # #carrots_depth: chex.Array              # depth of carrots at each x, -1 = none
    # gopher_animation_idx: chex.Array        # 0 for running, 1 for pop up face
    gopher_move_x_timer: chex.Array
    #gopher_timer: chex.Array
    #gopher_action: chex.Array
    
    # # Carrots info
    # carrots_left: chex.Array                # number of remaining carrots
    carrots_present: chex.Array             # 1: carrot in hole, 0: carrot not in hole
    
    # # Duck info
    # duck_position: chex.Array
    # seed_position: chex.Array               # three entrys, x, y and active state: 0 for collected/inactive 1 for not active
    # seed_caught: chex.Array
    # timers: chex.Array
    
    # # Score and level
    # player_score: chex.Array 
    # level: chex.Array
    
    
    # shovel_timer: int = 0
    
    key: chex.Array                         # handle randomness
    frame_count: int = 0


    #     carrot_counter: chex.Array              
    #     highscore: chex.Array                   # player score 
    #     # TODO DUCK info 








class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class GopherObservation(NamedTuple):
    player: EntityPosition
    gopher: EntityPosition
    duck: EntityPosition
    holes: jnp.ndarray
    carrots: jnp.ndarray
    seeds: jnp.ndarray  
    score_player: jnp.ndarray
    gopher_state: jnp.ndarray  
       
    
class GopherInfo(NamedTuple):
    difficulty_level: jnp.ndarray
    time:jnp.ndarray






# ==========================================================================================
# MAIN GAME LOGIC
# ==========================================================================================

class JaxGopher(JaxEnvironment[GopherState, GopherObservation, GopherInfo, GopherConstants]):
    def __init__(self, consts: GopherConstants = None):
        consts = consts or GopherConstants()
        super().__init__(consts)     
        self.renderer = GopherRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWNFIRE,
            
        ]
    
    



    def _player_step(self, state: GopherState, action: chex.Array) -> GopherState:
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

        # Set base speed:
        new_speed = jax.lax.select(left, -2.0, jax.lax.select(right, 2.0, 0.0))

        # Check collisions
        touch_left_wall = state.player_x <= self.consts.LEFT_WALL
        touch_right_wall = state.player_x + self.consts.PLAYER_SIZE[0] >= self.consts.RIGHT_WALL

        # Stop if walking into wall
        final_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(left, touch_left_wall), jnp.logical_and(right, touch_right_wall)),
            lambda _: 0.0,
            lambda _: new_speed,
            operand = None
        )
       
        # Clamp the movement in frame
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            self.consts.LEFT_WALL,
            self.consts.RIGHT_WALL - self.consts.PLAYER_SIZE[0],
        )

        
        


        
        
        

        '''
        # Catch seed
        player_left = state.player_x
        player_right = state.player_x + self.consts.PLAYER_SIZE[0]
        player_top = self.consts.PLAYER_START_Y
        player_bottom = self.consts.PLAYER_START_Y + self.consts.PLAYER_SIZE[1]

       
        seed_x_middle = state.seed_position[0] + self.consts.SEED_SIZE[0]/2
        seed_y_middle = state.seed_position[1] + self.consts.SEED_SIZE[1]/2

        overlap_x = jnp.logical_and(player_left < seed_x_middle, player_right > seed_x_middle)
        overlap_y = jnp.logical_and(player_top < seed_y_middle, player_bottom > seed_y_middle)
        collision = jnp.logical_and(overlap_x, overlap_y)

        seed_position = state.seed_position
        can_catch_seed = jnp.logical_and(collision, jnp.logical_and(seed_position[2] == 1, state.seed_caught == 0))
        seed_active, seed_caught = jax.lax.cond(
            can_catch_seed,
            lambda _: (0, 1),
            lambda _: (state.seed_position[2], state.seed_caught),
            operand=None
        )
        seed_position = state.seed_position.at[2].set(seed_active)
        
        
        state = state._replace(
            new_seed_position = seed_position, 
            new_seed_caught = seed_caught
        )
        
        
        
        # Downpress function:
        # 1.Plant carrots
        # 2.Bonk
        # 3.Repair tunnel
        # 4.Repair ground
        

        # 1.Plant carrots
        player_x_middle = state.player_x + self.consts.PLAYER_SIZE[0]/2
        carrots_x_middle = jnp.array(self.consts.CARROT_POSITION)
        matches = carrots_x_middle == player_x_middle
        is_carrots_spot = jnp.any(matches)
        # Get index of the carrot
        carrot_index = jnp.argmax(matches)
        # Check if the spot already exisit carrot
        carrot_exist = state.carrots_present[carrot_index]

        
        can_plant = jnp.all(jnp.array([
            is_carrots_spot,
            carrot_exist == 0, 
            action == Action.DOWNFIRE]))

        carrot_active, new_seed_caught = jax.lax.cond(
            can_plant,
            lambda _: (1, 0),
            lambda _: (carrot_exist, state.seed_caught),
            operand=None,
        )
        new_carrots_present = state.carrots_present.at[carrot_index].set(carrot_active)
        
        
        state = state._replace(
            carrots_present = new_carrots_present , 
            seed_caught = new_seed_caught,
        )
        '''

        # 2.Bonk 

        fire_pressed = jnp.logical_or(action == Action.FIRE, action == Action.DOWNFIRE)
        
        # Start timer if fire is pressed, otherwise if timer > 0, keep incrementing until 9
        is_bonking = state.bonk_timer > 0

        def increment_timer(t):
            '''
            Increse player bonk movement timer
            '''
            return jax.lax.select(t >= 8, 0 , t + 1)
        
        new_bonk_timer = jax.lax.cond(
            jnp.logical_or(fire_pressed, is_bonking),
            increment_timer,
            lambda t: 0,
            operand = state.bonk_timer
        )


        

        # gopher_x_middle = state.gopher_position[0] + self.consts.GOPHER_SIZE[0]/2

        # overlap_bonk_x = jnp.logical_and(
        #     player_left < gopher_x_middle,
        #     player_right > gopher_x_middle,
        # )

        # # bonk half: when gopher is half out
        # bonk_half = jnp.all(jnp.array([
        #    action == Action.DOWNFIRE,
        #    overlap_bonk_x,
        #    state.gopher_state == 0,
        # ]))


        # # Bonk running: when gopher is running to the carrots
        # bonk_running = jnp.all(jnp.array([
        #    action == Action.DOWNFIRE,
        #    overlap_bonk_x,
        #    state.gopher_state == 1,
        # ]))

        # gopher_bonked_cond = jnp.logical_or(bonk_half, bonk_running)

        # gopher_bonk = jax.lax.cond(
        #     gopher_bonked_cond, 
        #     lambda _: 1,
        #     lambda _: 0,
        #     operand = None
        # )

        # state = state._replace(gopher_bonked = gopher_bonk)
        
        '''
        # 3.Repair Tunnel
        # Check if player is standing above hole, and get the hole index
        matches_hole = player_x_middle == jnp.array(self.consts.HOLE_POSITION)
        player_on_hole = jnp.any(matches_hole)
        hole_index = jnp.argmax(matches_hole)
        
        # Calculate the index of the right-hand tunnel block under the hole
        tunnel_idx_right = jnp.floor_divide(self.consts.HOLE_POSITION[hole_index], self.consts.BLOCK_SIZE)
        # The tunnel block indices we need to repair (assuming they span two columns)
        tunnel_idx_left = jnp.int32(tunnel_idx_right - 1)
        tunnel_idx_right = jnp.int32(tunnel_idx_right)
        # Array of the two indices to update
        tunnel_indices_to_repair = jnp.array([tunnel_idx_left, tunnel_idx_right])

        # Check if ground layer is full dug through
        ground_full_dug = jnp.all(state.hole_state[hole_index] == 1)
        tunnel_full_dug = jnp.logical_and(
            state.tunnel_state[tunnel_idx_left] == 1, 
            state.tunnel_state[tunnel_idx_right] == 1   
        )
        
        tunnel_repaired = jnp.logical_and(
            state.tunnel_state[tunnel_idx_left] == 0, 
            state.tunnel_state[tunnel_idx_right] == 0   
        )
        top_dug = state.hole_state[hole_index][2] == 1
        
        # Gopher pop up over hole
        gopher_up_hole = jnp.logical_and((gopher_x_middle == player_x_middle), state.gopher_state == 0)
        
        repair_tunnel = jnp.all(jnp.array([
           action == Action.DOWNFIRE,
           jnp.logical_not(gopher_up_hole),
           player_on_hole,
           tunnel_full_dug,
        ]))

        new_tunnel_state = jax.lax.cond(
            repair_tunnel,
            lambda ts: ts.at[tunnel_indices_to_repair].set(0),
            lambda ts: ts,
            operand = state.tunnel_state,
        )

        # Repair ground
        repair_ground = jnp.all(jnp.array([
            action == Action.DOWNFIRE,
            tunnel_repaired,
            jnp.logical_not(gopher_up_hole),
            player_on_hole,
            top_dug,
        ]))

        ground_layers = state.hole_state[hole_index]
        first_dug_ground = jnp.argmax(ground_layers == 1)

        new_hole_state = jax.lax.cond(
            repair_ground,
            lambda hs: hs.at[hole_index, first_dug_ground].set(0),
            lambda hs: hs,
            operand = state.hole_state,
        )

        state = state._replace(
            hole_state = new_hole_state,
            tunnel_state = new_tunnel_state)
        
        '''
        return state._replace(
            player_x = proposed_player_x, 
            player_speed = final_speed,
            bonk_timer = new_bonk_timer

        )
    





    
    

    
    def _gopher_move_x(self, state: GopherState, consts:GopherConstants) -> GopherState:
        new_x = state.gopher_position[0] + state.gopher_direction_x * consts.GOPHER_SPEED1
        
        # Handle wraparoud 
        play_width = consts.RIGHT_WALL - consts.LEFT_WALL
        final_x = (new_x - consts.LEFT_WALL) % play_width + consts.LEFT_WALL
        new_timer = state.gopher_move_x_timer + 1
        
        return state._replace(
            gopher_position = state.gopher_position.at[0].set(final_x),
            gopher_move_x_timer  = new_timer                  
            )
    
    ''' 
    def _gopher_move_y(self, state: GopherState, consts:GopherConstants) -> GopherState:
        new_y = state.gopher_position[0] + state.gopher_direction_y * consts.GOPHER_SPEED1
        return state._replace(gopher_position = state.gopher_position.at[0].set(new_y))
    
    def _gopher_dig_up(self, state: GopherState, consts:GopherConstants) -> GopherState:
        gopher_x_middle = state.gopher_position[0] + consts.GOPHER_SIZE[0] / 2
        matches_hole = jnp.abs(gopher_x_middle - jnp.array(consts.HOLE_POSITION)) < 1.0
        hole_index = jnp.argmax(matches_hole)
        
        dig_layer = state.hole_state[hole_index]
        first_dug_ground_idx= jnp.argmax(dig_layer == 0)

        hs = state.hole_state.at[hole_index, first_dug_ground_idx].set(1)
        new_y = state.gopher_position[1] - consts.BLOCK_SIZE 
        return state._replace(
            hole_state = hs,
            gopher_position = state.gopher_position.at[1].set(new_y),
        )

    def _gopher_dig_front(self, state: GopherState, consts:GopherConstants) -> GopherState:
        gopher_x = state.gopher_position[0]
        direction = state.gopher_direction_x
        speed = consts.GOPHER_SPEED1

        # Identify the Block Index for Collision Check
        leading_edge_x = jax.lax.cond(
            direction == 1,
            lambda x: gopher_x + consts.GOPHER_SIZE[0],
            lambda x: gopher_x - 1,
            operand = None,
        )
        # Calculate the column index of the block the Gopher is moving INTO
        collision_column_index = jnp.floor_divide(leading_edge_x, consts.BLOCK_SIZE).astype(jnp.int32)
        # Clamp the index
        column_index_clamped = jnp.clip(collision_column_index, 0, 42)
        # Collision check:if the block ahead undug
        is_undug_ahead = state.tunnel_state[column_index_clamped] == 0
        
        # Update front block state
        ts = state.tunnel_state.at[column_index_clamped].set(1)
        # Move to the block 
        proposed_x = gopher_x + direction * speed
        return state._replace(
            gopher_position = state.gopher_position.at[0].set(proposed_x),
            tunnel_state = ts
        )
    
    # Gopher change direction    
    def _gopher_turn_around_x(state: GopherState) -> GopherState:     
        return state._replace(
            gopher_direction_x = - state.gopher_direction_x
        )
    
    def _gopher_pop_up(self, state: GopherState, consts:GopherConstants) -> GopherState:
        return state._replace(gopher_position = state.gopher_position.at[1].set(consts.SKY_HEIGHT - consts.GOPHER_SIZE[1]))

    def _gopher_steal(self, state: GopherState, target_idx: int) -> GopherState:
        return state._replace(
            carrots_present = state.carrots_present.at[target_idx].set(0))
    

    # state 0
    def _handle_tunnel_move(self, state: GopherState, consts: GopherConstants) -> GopherState:
        gopher_x_middle = state.gopher_position[0] + consts.GOPHER_SIZE[0] / 2
        gopher_under_hole = jnp.any(jnp.abs(gopher_x_middle - jnp.array(consts.HOLE_POSITION)) < 1.0)

        gopher_x = state.gopher_position[0]
        direction = state.gopher_direction_x
        speed = consts.GOPHER_SPEED1

        # Identify the Block Index for Collision Check
        leading_edge_x = jax.lax.cond(
            direction == 1,
            lambda x: gopher_x + consts.GOPHER_SIZE[0] + speed,
            lambda x: gopher_x - speed,
            operand = None,
        )
        # Calculate the column index of the block the Gopher is moving INTO
        collision_column_index = jnp.floor_divide(leading_edge_x, consts.BLOCK_SIZE).astype(jnp.int32)
        # Clamp the index
        column_index_clamped = jnp.clip(collision_column_index, 0, 42)
        # Collision check:if the block ahead undug
        is_undug_ahead = state.tunnel_state[column_index_clamped] == 0
        
        #--- Decision Logic ---
        key, state_key = jax.random.split(state.key)
        roll = jax.random.uniform(key)

        def dig_front_or_turn(s):
            state = jax.lax.cond(
                roll < 0.3,
                self._gopher_turn_around_x,
                self._gopher_dig_front,
                operand = None
            )
            return state

        def default_action(s):
            state = jax.lax.cond(
                is_undug_ahead,
                dig_front_or_turn,
                self._gopher_move_x,
                operand= None
            )
            return state
        
        def ignore_or_up(s):
            state = jax.lax.cond(
                roll < 0.3,
                default_action,
                lambda s: s._replace(gopher_state = 1),
                operand = state
            )
            return state
        
        
        state = jax.lax.cond(
            gopher_under_hole,
            ignore_or_up,
            default_action,


        )
        
        

        return

    # State 1
    def _handle_dig_up(self, state: GopherState, consts: GopherConstants) -> GopherState:
        return
    
    # State 2
    def _handle_pop_up(self, state: GopherState, consts: GopherConstants) -> GopherState:
        return
    
    # state 3
    def _handle_stealing(self, state: GopherState, consts: GopherConstants) -> GopherState:
        return
    
    
    
    # State 4
    def _handle_back_tunnel(self, state: GopherState, consts: GopherConstants) -> GopherState:
        return
    
    #***************************************************************************************************************
     # Helper function: Gopher Actions
    
    # State 0: Gopher pops out from a complete hole
    def _gopher_pop_up(self, state: GopherState, consts:GopherConstants) -> GopherState:
        return state._replace(gopher_position = state.gopher_position.at[1].set(consts.SKY_HEIGHT - consts.GOPHER_SIZE[1]))

    # State 1: run to steal carrot
    def _gopher_running_to_steal(self, state: GopherState, consts:GopherConstants) -> GopherState:
        new_x = state.gopher_position[0] + state.gopher_direction_x * consts.GOPHER_SPEED1
        return state._replace(gopher_position = state.gopher_position.at[0].set(new_x))
    
    # State 1: steal carrot
    def _gopher_steal(self, state: GopherState, target_idx: int) -> GopherState:
        return state._replace(
            carrots_present = state.carrots_present.at[target_idx].set(0),
            gopher_state = 2)
    
    
    # State 2: Hidden in tunnel, move left and right or dug in tunnel
    def _gopher_move_dug_tunnel(self, state: GopherState, consts:GopherConstants) -> GopherState:
        gopher_x = state.gopher_position[0]
        direction = state.gopher_direction_x
        speed = consts.GOPHER_SPEED1

        # Identify the Block Index for Collision Check
        leading_edge_x = jax.lax.cond(
            direction == 1,
            lambda x: gopher_x + consts.GOPHER_SIZE[0] + speed,
            lambda x: gopher_x - speed,
            operand = None,
        )
        # Calculate the column index of the block the Gopher is moving INTO
        collision_column_index = jnp.floor_divide(leading_edge_x, consts.BLOCK_SIZE).astype(jnp.int32)
        # Clamp the index
        column_index_clamped = jnp.clip(collision_column_index, 0, 42)
        # Collision check:if the block ahead undug
        is_undug_ahead = state.tunnel_state[column_index_clamped] == 0

        # Move if ahead block is clear
        proposed_x = gopher_x + direction * speed

        # Helper function
        # Dig one block
        def dig_forward(_):
            # Dig one block
            ts = state.tunnel_state.at[column_index_clamped].set(1)
            return gopher_x, ts
        
        # Keep moving(already dug tunnel block)
        def keep_moving(_):
            return proposed_x, state.tunnel_state
        
        x_before_wrap, final_tunnel_state = jax.lax.cond(
            is_undug_ahead,
            dig_forward,
            keep_moving,
            operand=None
        )

        hits_left_wall = x_before_wrap < consts.LEFT_WALL
        hits_right_wall = (x_before_wrap + consts.GOPHER_SIZE[0]) > consts.RIGHT_WALL

        # Handle wraparound situation
        final_x = jnp.select(
            [hits_left_wall, hits_right_wall],
            [consts.RIGHT_WALL - consts.GOPHER_SIZE[0], consts.LEFT_WALL],
            default=x_before_wrap
        )

        return state._replace(
            gopher_position = state.gopher_position.at[0].set(final_x),
            tunnel_state = final_tunnel_state
        )

    # Gopher change direction    
    def _gopher_turn_around(state: GopherState) -> GopherState:     
        return state._replace(
            gopher_direction_x = - state.gopher_direction_x
        )
    

    # State 3: Gopher dig up in hole
    def _gopher_dig_up(state: GopherState, consts: GopherConstants) -> GopherState:
    # Determine which hole the Gopher is currently underneath
        gopher_x_middle = state.gopher_position[0] + consts.GOPHER_SIZE[0] / 2
        matches_hole = jnp.abs(gopher_x_middle - jnp.array(consts.HOLE_POSITION)) < 1.0
        hole_index = jnp.argmax(matches_hole)

    # Find the deepest undug layer in that hole's section of hole_state
        dig_layer = state.hole_state[hole_index]
        first_dug_ground_idx= jnp.argmax(dig_layer == 0)

    # Can dig:check if the hole is already complete
        not_dug_through = jnp.any(dig_layer == 0)

        # If the hole is not dug through, gopher digs one block up
        def dig_one_block(_):
            hs = state.hole_state.at[hole_index, first_dug_ground_idx].set(1)
            new_y = state.gopher_position[1] - consts.BLOCK_SIZE 
            return hs, new_y, 3                                                          # stay in state 3
        # If the hole is dug through, gopher walks up, and switch state to pop up: state 0
        def move_up(_):
            return state.hole_state, state.gopher_position[1] - consts.BLOCK_SIZE * 3, 0 # go to state 0
    
    # Update that layer's state from 0 (undug) to 1 (dug)
        new_hole_state, new_y, new_gopher_state = jax.lax.cond(
            not_dug_through,
            dig_one_block,
            move_up,
            operand=None,
        )
    
    
    # Return the new state
        return state._replace(
            hole_state = new_hole_state,
            gopher_position = state.gopher_position.at[1].set(new_y),
            gopher_state = new_gopher_state
        )
    



    # State Translation
    # State 0: pop up
    def _handle_pop_up(self, state: GopherState, consts: GopherConstants) -> GopherState:
        state_after_pop = self._gopher_pop_up(state, consts)

        # Decide for next state
        key, state_key = jax.random.split(state_after_pop.key)
        roll = jax.random.uniform(key)

        should_steal = roll < 0.8
        next_state_id = jnp.where(should_steal, 1, 2)
        
        return state_after_pop._replace(gopher_state = next_state_id, key = state_key)
    
    # State 1: run to carrot, and steal
    def _handle_steal(self, state: GopherState, consts: GopherConstants) -> GopherState:
        # check if gopher is bonked(both condition: running bonked or pop up bonked)
        if_bonked = (state.gopher_bonked == 1)
        
        def continue_stealing(s:GopherState):
        # Run to left
            run_to_left = (state.gopher_direction_x == -1)

            target_carrot_idx = jax.lax.cond(
                run_to_left,
                lambda _: (len(s.carrots_present) - 1) - jnp.argmax(jnp.flip(s.carrots_present) == 1),
                lambda _: jnp.argmax(s.carrots_present == 1),
                operand=None
            )
            
            gopher_at_carrot = jnp.abs((state.gopher_position[0] + consts.GOPHER_SIZE[0] / 2 - consts.CARROT_POSITION[target_carrot_idx]) < 5.0)
            new_state = jax.lax.cond(
                gopher_at_carrot,
                lambda st: self._gopher_steal(st, target_carrot_idx),
                self._gopher_running_to_steal,
                operand = s
            )
            return new_state

        next_state = jax.lax.cond(
            if_bonked,
            lambda s: s._replace(gopher_state = 2),
            continue_stealing,
            operand = state
        )

        return state._replace(gopher_state = next_state)


    # State 2, Handle hidden state(in tunnel layer)
    def _handle_hidden(self, state: GopherState, consts: GopherConstants) -> GopherState:
        old_x = state.gopher_position[0]
        
        #--- Execute dig and move ---
        state_after_move = self._gopher_move_dug_tunnel(state, consts)
        #--- Check alignment and stop(after dug one undug block)
        current_x = state_after_move.gopher_position[0]
        gopher_was_stopped = (current_x == old_x)



        gopher_x_middle = state_after_move.gopher_position[0] + consts.GOPHER_SIZE[0] / 2
        gopher_under_hole = jnp.any(jnp.abs(gopher_x_middle - jnp.array(consts.HOLE_POSITION)) < 1.0)

        #--- Decision Logic ---
        key, state_key = jax.random.split(state_after_move.key)
        roll = jax.random.uniform(key)

        #--- Set priorities: 40% chance to dig up, 60% chance to turn around

        def handle_stopped(s):
            # Condition 1: Transition to digging state 3
            can_dig_up = jnp.logical_and(gopher_under_hole, roll < 0.40)
            # Condition 2: Transition to turn around state 4
            should_turn_around = jnp.logical_and(
                roll > 0.40,
                roll < 0.90
            )
            new_id = jnp.select(
                [can_dig_up, should_turn_around],[3, 2], default = 2)
            
            s_final = jax.lax.cond(
                should_turn_around,
                lambda st: self._gopher_turn_around(st),
                lambda st: st,
                operand=s
            )
            return s_final._replace(gopher_state = new_id)
        
        # Apply decision strategy if it's stopped(dig once front layer)
        state_after_decision = jax.lax.cond(
            gopher_was_stopped,
            handle_stopped,
            lambda s: s,
            operand = new_state
        )
        
        return state_after_decision._replace(key = state_key)


    # State 3, handle dig up and transition
    def _handle_digging(self, state: GopherState, consts: GopherConstants) -> GopherState:
        #--- Perfome digging and climbing ---
        state_after_dig: GopherState = self._gopher_dig_up(state, consts)

        #--- Check transition condition: if gopher reached top---
        is_hole_finished = (state_after_dig.gopher_state == 0)
        
        #--- Determine next state, if the hole is not through, dig up(state 3) or back to tunnel(state 2)

        key, state_key = jax.random.split(state_after_dig.key)
        roll = jax.random.uniform(key)
        
        #--- Set probability:60 % dig up(keep state 3), 40 % back to tunnel(state 2)
        
        def decide_up_back(s: GopherState):
            should_go_back = jnp.logical_and(is_hole_finished, roll > 0.6)
            next_id = jnp.where(should_go_back, 2, 3)
            final_y = jnp.where(should_go_back, consts.GOPHER_START_Y, s.gopher_position[1])
            
            return s._replace(
                gopher_state = next_id,
                gopher_position = s.gopher_position.at[1].set(final_y)
            )
        
        final_state = jax.lax.cond(
            is_hole_finished,
            lambda s: s,
            decide_up_back,
            operand=state_after_dig
        )

        
        return final_state._replace(key = state_key)
    
    def _gopher_update_logic(self, state: GopherState, consts: GopherConstants) -> GopherState:
        """
        Maps gopher's state with handler
        """
        state_handlers = [
            self._handle_pop_up,     # state 0
            self._handle_steal,      # state 1
            self._handle_hidden,     # state 2
            self._handle_digging     # state 3
        ]

        return jax.lax.switch(
            state.gopher_state,
            state_handlers,
            state,
            consts
        )

    '''   

    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, GopherState]:
        # Create the initial state
        state = GopherState(
            player_x=jnp.array(84, dtype=jnp.int32), # Start in the middle
            player_speed=jnp.array(0.0),
            gopher_position=jnp.array([150.0, 150.0]),
            gopher_direction_x=jnp.array(-1, dtype=jnp.int32),
            # gopher_state=jnp.array(0),
            bonk_timer=jnp.array(0, dtype=jnp.int32),
            gopher_move_x_timer=jnp.array(0, dtype=jnp.int32),
            carrots_present = jnp.array([1, 1, 1]),
            key=key,
            frame_count=jnp.array(0),
            
            
        )
        

        obs = self.render(state)
        return obs, state

    def step(self, state: GopherState, action: chex.Array) -> tuple[chex.Array, GopherState, jnp.ndarray, jnp.ndarray, dict]:
        # 1. Update player and gopher
        state = self._player_step(state, action)
        state = self._gopher_move_x(state, self.consts)
        
        # 2. Update frame count (for timing things later)
        state = state._replace(frame_count=state.frame_count + 1)
        
        # 3. Create observation (the picture)
        obs = self.render(state)
        
        # 4. Return standard JAX environment tuple
        reward = jnp.array(0.0)
        done = jnp.array(False)
        info = self._get_info(state)
        return obs, state, reward, done, info
    
    def render(self, state: GopherState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def action_space(self):
        """Return the action space"""
        return spaces.Discrete(4)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GopherState, ) -> GopherInfo:
        return GopherInfo(time=state.frame_count, difficulty_level=jnp.array(1, dtype=jnp.int32))



class GopherRenderer(JAXGameRenderer):
    def __init__(self, consts: GopherConstants = None):
        super().__init__(consts)
        self.consts = consts or GopherConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # Get file-based assets from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/gopher"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)


    #def _load_and_prepare_assets(self):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state:GopherState):
        
        # *****Background*****
        raster = self.jr.create_object_raster(self.BACKGROUND)
        

        # *****Carrot*****
        carrot_position = jnp.array(self.consts.CARROT_POSITION)
        carrot_mask = self.SHAPE_MASKS["carrot"]
        print(self.SHAPE_MASKS['carrot'].shape)
        def draw_one_carrot(i, current_raster):
            is_active = state.carrots_present[i] == 1
            x = carrot_position[i]
            y = 151
            
            return jax.lax.cond(
                is_active,
                lambda r: self.jr.render_at(r, jnp.int32(x), jnp.int32(y), carrot_mask),
                lambda r: r,
                operand = current_raster
            )
        
        raster = jax.lax.fori_loop(0, 3, draw_one_carrot, raster)
        
        # *****Player*****
        # Determine which sprite to show
        # 0-1: Default(0), 2-3: middle(1), 4-7: top(2), 8: Default(0)
        player_sprite_idx = jax.lax.select(
            state.bonk_timer < 2, 0, 
            jax.lax.select(state.bonk_timer < 4, 1,
            jax.lax.select(state.bonk_timer < 8, 2, 0)
            )
        )
        player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
        print(self.SHAPE_MASKS['player'].shape)
        print(self.BACKGROUND.shape)
        raster = self.jr.render_at(raster, jnp.int32(state.player_x), jnp.int32(self.consts.PLAYER_START_Y), player_mask)

        # *****Tunnel dig block*****
        # def render_tunnels(state: GopherState, raster):
        #     tunnel_mask = 
        # *****Hole dig block***** 
        
        # *****Gopher*****
        # Walk
        gopher_leg_idx = (state.gopher_move_x_timer // 4) % 2
        gopher_mask = self.SHAPE_MASKS["gopher"][gopher_leg_idx]
        raster = self.jr.render_at(raster, jnp.int32(state.gopher_position[0]), jnp.int32(self.consts.GOPHER_START_Y), gopher_mask)
        
        return self.jr.render_from_palette(raster, self.PALETTE)
    
