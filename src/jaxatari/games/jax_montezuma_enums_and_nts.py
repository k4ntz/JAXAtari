from typing import NamedTuple
from enum import Enum
import jax.numpy as jnp
from jax import Array as jArray
import os
from jaxatari.games.jax_montezuma_utils import SANTAH

class GlobalLadderBehavior:
    TELEPORT_ONTO_LADDER_FROM_HORIZONTAL_DISTANCE = jnp.array([5], jnp.uint32) 
        # At which horizontal distance the player starts to teleport ontp the ladder.
    LADDER_START_HEIGHT_BOTTOM = jnp.array([5], jnp.uint32)
    # How many pixels the player gets teleported up once he gets on the ladder at the bottom
    LADDER_START_HEIGHT_TOP = jnp.array([5], jnp.uint32)
    # How many pixels the player gets teleported DOWN once he gets on the top of the ladder
    LADDER_LEAVE_THRESHOLD_BOTTOM = jnp.array([3], jnp.uint32)
    # How many pixels from the bottom ladder point the ladder allows the player to leave the ladder
    LADDER_LEAVE_THRESHOLD_TOP = jnp.array([3], jnp.uint32)
    # How many pixels from the top ladder point the ladder allows the player to leave
    LADDER_LEAVING_HORIZONTAL_BUMP = jnp.array([3], jnp.uint32)
    # when leaving the ladder in a horizontal direction, how many pixels is the player teleported to the side


class ItemObservation(NamedTuple):
    dummy: jArray # Dummy items are used if there is no item present in the room to preserve shape.
    position: jArray
    type: jArray
    collected: jArray # Whether the item was already collected
    
class EnemyObservation(NamedTuple):
    alive: jArray
    dummy: jArray # Again, dummy enemies are used if there is no enemy in this room
    position: jArray
    type: jArray
        
        
# These values are placed on the second channel and indicate, 
# which features are present at which points in the room
class ObservationCollisionMapAnnotationColors(Enum):
    background = 0                                  
    normal_wall = 1
    ladder = 2
    rope = 3
    conveyor = 4
    door = 5
    lazer_barrier = 6
    dropout_floor = 7


class MontezumaObservation(NamedTuple):
    
    annotated_collision_map: jArray
        # Since we have  a lot of possibilities for the room layout, communicating 
        # the layout is done via the annotated collision map. 
        # The annotated collision map has the same size as the level and communicates
        # which pixels are occupied by which in-game objects. 
        # A list of all anotations can be found in the ObservationCollisionMapAnnotationColors
    current_items: jArray
    has_dropout_floors: jArray
    has_pit: jArray
    is_falling: jArray
    is_jumping: jArray
    is_on_ladder: jArray
    is_on_rope: jArray
    nearest_enemy: EnemyObservation
    nearest_item: ItemObservation
    number_of_lives: jArray
    player_position: jArray
    
    
    
    
    
    




class ConstantShapeRoomFields(Enum): 
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    #                                                                                                                                                           #
    #         All fields in here need to have CONSTANT SHAPE ACROSS ALL ROOMS.                                                                                  #
    #                                                                                                                                                           #
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    height = "height" 
    vertical_offset = "vertical_offset" 
    ROOM_ID = "ROOM_ID"
    left_start_position = "left_start_position"
    right_start_position = "right_start_position"
    top_start_position = "top_start_position"
    bottom_start_position = "bottom_start_position"


class MandatoryLazerBarrierFields(NamedTuple):
    barriers: jnp.ndarray
    global_barrier_info: jnp.ndarray
    global_barrier_map: jnp.ndarray # A collision map for all lazer barriers in the room
class MandatoryLazerBarrierFieldsEnum(Enum):
    barriers = "barriers"
    global_barrier_info = "global_barrier_info"
    global_barrier_map = "global_barrier_map"
    
    
    
class MandatoryItemFields(NamedTuple):
    items: jnp.ndarray
class MandatoryItemFieldsEnum(Enum):
    items = "items"
    
# The DarkRoom tag carries no additional information
# In the renderer, the presence of the DarkRoom Tag
# conditions rendering on the presence of a torch.

class MandatoryDarkroomFields(NamedTuple):
    pass
class MandatoryDarkroomFieldsEnum(Enum):
    pass

class MandatoryBonusRoomFields(NamedTuple):
    bouns_cycle_index: jnp.ndarray # How many frames the player has been in the bonus room
    bonus_cycle_lenght: jnp.ndarray # How long the player stays in the bonus room
    bonus_room_floor_collison_map: jnp.ndarray
    reset_state_on_leave: jArray # This is used to reset the room map in the case where leaving the bonus room doesn't cause the player 
        # to enter a new level, but just to loop in the current level.
class MandatoryBonusRoomFieldsEnum(Enum):
    bouns_cycle_index = "bouns_cycle_index"
    bonus_cycle_lenght = "bonus_cycle_lenght"
    bonus_room_floor_collison_map = "bonus_room_floor_collison_map"
    reset_state_on_leave = "reset_state_on_leave"
    
    
class MandatoryConveyorBeltFields(NamedTuple):
    conveyor_belts: jnp.ndarray
    global_conveyor_collision_map: jnp.ndarray
    # Seperate maps for collision & movement detection
    global_conveyor_movement_collision_map: jnp.ndarray
    global_conveyor_render_map: jnp.ndarray
        
        
class MandatoryConveyorBeltFieldsEnum(Enum):
    conveyor_belts = "conveyor_belts"
    global_conveyor_collision_map = "global_conveyor_collision_map"
    global_conveyor_movement_collision_map = "global_conveyor_movement_collision_map"
    global_conveyor_render_map = "global_conveyor_render_map"
    
    
class MandatoryDoorFields(NamedTuple):
    doors: jnp.ndarray
    global_collision_map: jnp.ndarray
class MandatoryDoorFieldsEnum(Enum):
    doors = "doors"
    global_collision_map = "global_collision_map"

    
class MandatoryDropoutFloorFields(NamedTuple):
    dropout_floors: jnp.ndarray
    # The fraction of "on-time" vs "off-time" for Dropout floors
    # can be adjusted on a per-room basis.
    on_time_dropoutfloor: jnp.ndarray
    off_time_dropoutfloor: jnp.ndarray
    dropout_floor_render_maps: jnp.ndarray
    dropout_floor_colision_map: jnp.ndarray
    
    
class MandatoryDropoutFloorFieldsEnum(Enum):
    dropout_floors = "dropout_floors"
    on_time_dropoutfloor = "on_time_dropoutfloor"
    off_time_dropoutfloor = "off_time_dropoutfloor"
    dropout_floor_render_maps = "dropout_floor_render_maps"
    dropout_floor_colision_map = "dropout_floor_colision_map"
    
    
class MandatoryPitFields(NamedTuple):
    starting_pos_y : jnp.ndarray
    pit_color : jnp.ndarray
    pit_render_maps: jnp.ndarray


class MandatoryPitFieldsEnum(Enum):
    starting_pos_y = "starting_pos_y"
    pit_color = "pit_color"
    pit_render_maps = "pit_render_maps"
    
    
class MandatorySidewallsFields(NamedTuple):
    is_left: jnp.ndarray
    is_right: jnp.ndarray
    side_wall_color: jnp.ndarray
    side_walls_render_map: jnp.ndarray
    side_walls_collision_map: jnp.ndarray


class MandatorySidewallsFieldsEnum(Enum):
    is_left = "is_left"
    is_right = "is_right"
    side_wall_color = "side_wall_color"
    side_walls_render_map = "side_walls_render_map"
    side_walls_collision_map = "side_walls_collision_map"

    
class Rope(NamedTuple):
    x_pos: jArray
    top: jArray
    bottom: jArray
    color_index: jArray
    is_climbable: jArray
    accessible_from_top: jArray
    # A int32 flag signalling whether the rope is accessible from the top, i.e. whether we can get on/ off it from the top
    

class RopeEnum(Enum):
    x_pos = "x_pos"
    top = "top"
    bottom = "bottom"
    color_index = "color_index"
    is_climbable = "is_climbable"
    accessible_from_top = "accessible_from_top"
    
    
class MandatoryRopeFields(NamedTuple):
    ropes: jArray # A stack of ropes which are in the room
    rope_index: jArray # The index of the rope the player is currently hanging from.
    # If player isn't on rope, this is -1
    rope_render_map: jArray # A precomputed render map for the ropes
    rope_colision_map: jArray # A precomputed colision map. Includes only the climbeable ropes, 
    # and rope values in the map correspond to the position of the ropes in the array + 1
    last_hanged_on_rope: jArray # The rope the player was climbing on the last time he was on a rope.
    # This is used to prevent the player from repeatedly regrabbing the same rope, which is not intended behavior. 
    # This gets reset to a default value of -1 every time the player touches the floor.
    room_surfaces: jArray
    # A utility hitmap of the room in which only floor/ platform surfaces are highlighted. 
    # A surface is the pixel-layer right above a solid collision map. 
    # Surfaces have values of 1, everything else 0
    # This has the size of the full room collision hitmap with padding.
    # For surfaces, only collision elements defined at room initialization can be considered. 
    room_rope_top_pixels: jArray
    # This is a hitmap for the top pixels of the rope. 
    # Default value is -1, and the top ropes are set with their index in the rope stack.
class MandatoryRopeFieldsEnum(Enum):
    ropes = "ropes"
    rope_index = "rope_index"
    rope_render_map = "rope_render_map"
    rope_colision_map = "rope_colision_map"
    last_hanged_on_rope = "last_hanged_on_rope"
    room_surfaces = "room_surfaces"
    room_rope_top_pixels = "room_rope_top_pixels"



class MandatoryLadderFields(NamedTuple):
    ladders: jArray # A scalar named tuple stack containing the ladders
    ladder_index: jArray # Integer index of the ladder the player is currently on,
    # if not on ladder: ladder_index is -1
        
    ladder_tops: jArray # A hitmap containing all the collision zones for entering a ladder at the TOP.
        # A value of -1 means no ladder can be entered here, 
        # All other values are the indexes of the ladders that live there
    ladder_bottoms: jArray # A hitmap containing all the collision zones for entering a ladder at the BOTTOM
        # A value of -1 means no ladder can be entered here, 
        # All other values are the indexes of the ladders that live there
    ladders_sprite: jArray # A full-size sprite map with containing all the collision zones for leaving a ladder
    ladder_room_surface_pixels: jArray
    # This is a hitmap for the top pixels of the rope. 
    # Default value is -1, and the top ropes are set with their index in the rope stack.


class MandatoryLadderFieldsEnum(Enum):
    ladders = "ladders"
    ladder_index = "ladder_index"
    ladder_tops = "ladder_tops"
    ladder_bottoms = "ladder_bottoms"
    ladders_sprite = "ladders_sprite"
    ladder_room_surface_pixels = "ladder_room_surface_pixels"


class MandatoryEnemyFields(NamedTuple):
    enemies: jArray
    
class MandatoryEnemyFieldsEnum(Enum):
    enemies = "enemies"
    


class RoomTags(Enum):
    LAZER_BARRIER = MandatoryLazerBarrierFields
    ITEMS = MandatoryItemFields
    ROPES = MandatoryRopeFields
    DOORS = MandatoryDoorFields
    DROPOUTFLOORS = MandatoryDropoutFloorFields 
    PIT = MandatoryPitFields
    LADDERS = MandatoryLadderFields
    SIDEWALLS = MandatorySidewallsFields
    ENEMIES = MandatoryEnemyFields
    CONVEYORBELTS = MandatoryConveyorBeltFields
    DARKROOM = MandatoryDarkroomFields
    BONUSROOM = MandatoryBonusRoomFields
    
class RoomTagsNames(Enum):
    LAZER_BARRIER = MandatoryLazerBarrierFieldsEnum
    ITEMS = MandatoryItemFieldsEnum
    ROPES = MandatoryRopeFieldsEnum
    DOORS = MandatoryDoorFieldsEnum 
    DROPOUTFLOORS = MandatoryDropoutFloorFieldsEnum
    PIT = MandatoryPitFieldsEnum
    LADDERS = MandatoryLadderFieldsEnum
    SIDEWALLS = MandatorySidewallsFieldsEnum
    ENEMIES = MandatoryEnemyFieldsEnum
    CONVEYORBELTS = MandatoryConveyorBeltFieldsEnum
    DARKROOM = MandatoryDarkroomFieldsEnum
    BONUSROOM = MandatoryBonusRoomFieldsEnum


class FieldsThatAreSharedByAllRoomsButHaveDifferentShape(Enum):
    sprite = "sprite" 
    room_collision_map = "room_collision_map"
    
    
class Ladder(NamedTuple):
    left_upper_x: jArray # X coord of left, upper corner
    left_upper_y: jArray # y coord of left, upper corner
    right_lower_x: jArray # ...
    right_lower_y: jArray # ...
    has_background: jArray # Whether a background is rendered
    rope_seeking_at_top: jArray # Whether this ladder directly connects to another ladder at the top 
        # in another room. If it is set to 1, upon entering the next room, the game will seek the nearest 
        # ladder top/ bottom and snap to it
    rope_seeking_at_bottom: jArray # Same but for the bottom
    transparent_background: jArray # Binary integer array, says whether background is transparent
    transparent_foreground: jArray # Binary integer array, says whether foreground is transparent
    foreground_color: jArray # Foreground colors in RGB, values [0, 255]
    background_color: jArray# Background color in RGB, values [0, 255]


class LadderFields(Enum):
    left_upper_x = "left_upper_x"
    left_upper_y = "left_upper_y"
    right_lower_x = "right_lower_x"
    right_lower_y = "right_lower_y"
    has_background = "has_background"
    rope_seeking_at_top = "rope_seeking_at_top" 
    rope_seeking_at_bottom = "rope_seeking_at_bottom"
    transparent_background = "transparent_background"
    transparent_foreground = "transparent_foreground"
    
    foreground_color = "foreground_color"
    background_color = "background_color"




    
class LAZER_BARRIER(NamedTuple): # Contains only barrier specific information. Global information about all barriers in a room is stored in another object.
    X: jnp.ndarray # X-coordinate of the barrier
    upper_point: jnp.ndarray # Y-coord of upper barrier edge
    lower_point: jnp.ndarray # Y coord of lower barrier edge


class LAZER_BARRIER_ENUM(Enum):
    X = "X"
    upper_point = "upper_point"
    lower_point = "lower_point"
    
class ConveyorBelt(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    movement_dir: jnp.ndarray
    color: jnp.ndarray

class ConveyorBeltEnum(Enum):
    x = "x"
    y = "y"
    movement_dir = "movement_dir"
    color = "color"
    
class Door(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    on_field: jnp.ndarray
    color: jnp.ndarray
    
class DoorEnum(Enum):
    x = "x"
    y = "y"
    on_field = "on_field"
    color = "color"
    
    
class DropoutFloor(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    sprite_height_amount: jnp.ndarray
    sprite_width_amount: jnp.ndarray
    sprite_index: jnp.ndarray
    collision_padding_top: jnp.ndarray
    color: jnp.ndarray
    
class DropoutFloorEnum(Enum):
    x = "x"
    y = "y"
    sprite_height_amount = "sprite_height_amount"
    sprite_width_amount = "sprite_width_amount"
    sprite_index = "sprite_index"
    collision_padding_top = "collision_padding_top"
    color = "color"
    

class GlobalLazerBarrierInfo(NamedTuple):
    cycle_length: jnp.ndarray # How long the animation cycle for the barriers is 
    cycle_active_frames: jnp.ndarray # How many frames from the cycle the barrier is active
    cycle_offset: jnp.ndarray # Per default, barrier is active right from the start. This offset controls how many frames the active period is offset from the start
    cycle_index: jnp.ndarray # Gives current index in the animation cycle.

class GlobalLazerBarrierInfoEnum(Enum):
    cycle_length = "cycle_length" 
    cycle_active_frames = "cycle_active_frames"
    cycle_offset = "cycle_offset"
    cycle_index = "cycle_index"
    

class Room(NamedTuple): # This class should contain all attributes that are shared by ALL TYPES OF ROOMS
    height: jnp.ndarray
    vertical_offset: jnp.ndarray
    ROOM_ID: jnp.ndarray
    left_start_position: jnp.ndarray
    right_start_position: jnp.ndarray
    top_start_position: jnp.ndarray
    bottom_start_position: jnp.ndarray

    
class VanillaRoom(NamedTuple): # A vanilla room. NO ladders, No moving Platforms, No items, No barriers
    sprite: jnp.ndarray
    height: jnp.ndarray
    vertical_offset: jnp.ndarray
    ROOM_ID: jnp.ndarray
    room_collision_map: jnp.ndarray
    left_start_position: jnp.ndarray
    right_start_position: jnp.ndarray
    top_start_position: jnp.ndarray
    bottom_start_position: jnp.ndarray


class VanillaRoomFields(Enum):
    sprite = "sprite"
    height = "height"
    vertical_offset = "vertical_offset"
    ROOM_ID = "ROOM_ID"
    room_collision_map = "room_collision_map"
    left_start_position = "left_start_position"
    right_start_position = "right_start_position"
    top_start_position = "top_start_position"
    bottom_start_position = "bottom_start_position"
    

class MontezumaState(NamedTuple):
    room_state: Room # At this level we ALWAYS NEED TO USE THE PROTO_ROOM ROOM, not the individual rooms
        # Operations that require room-specific fields/ shapes always need to be performed
        # In seperae function that lower & raise to the specific room types using the persistence state.   
    persistence_state: jnp.ndarray # Storage for all utable fields in the rooms
    player_position: jnp.ndarray # 2D array (x, y) 
    player_collision_map: jnp.ndarray
    player_velocity: jnp.ndarray
    vertical_directional_input: jnp.ndarray
    horizontal_direction: jnp.ndarray
    last_horizontal_direction: jnp.ndarray # Is used for computing changes in direction.
    last_horizontal_orientation: jnp.ndarray # Is used for computing the orientation of the player sprite
    horizontal_falling_velocitiy: jnp.ndarray
    is_standing: jnp.ndarray
    current_directional_input: jnp.ndarray
    is_jumping: jnp.ndarray
    is_falling: jnp.ndarray
    jump_counter: jnp.ndarray # For how many frames the player has already been jumping
    augmented_collision_map: jnp.ndarray # # (An Array of shape (WIDTH, HEIGHT)). The level specific collision map is embedded into this 
                                         # # with a certain vertical offset. This is needed because JAX cannot handle 
                                         # # Arrays of variable dimensionality.    
    canvas: jnp.ndarray
    jump_input: jnp.ndarray # Singleton integer array that says whether the current input is a jump input
    frame_count: jnp.ndarray
    is_dying: jnp.ndarray # Indicates whether death logic should be executed at the end of this step
    last_entrance_direction: jnp.ndarray
    is_climbing: jArray # Is climbing should be used to generally restrict player movements based on the climbing state.
    is_laddering: jArray # Signifies whether the player is currently standing on a ladder
    score: jArray
    lifes: jArray
    itembar_items: jArray # An array settin the items held by the player.
    # Position translates to type, value translates to amount.
    
    is_on_rope: jArray
    is_key_hold: jArray # Says whether the current input is a fresh key press or a key hold
    last_key_press: jArray # What the last key pressed is
    
    
    force_jump: jArray
    # The option to force a jump. If a jump if forced, all other gates/ conditions are ignored & a jump is executed
    disable_directional_input: jArray # Whether the current directional input is disabled
    queue_enable_directional_input: jArray # queued changes are always executed at the end of the step 
    
    # These fields need to be in here, as the implemented behavior (seeking a new ladder when leaving a room through a ladder)
    # Is a cross-room behavior (cause & effect are in different rooms), and thus cannot be handled in the room specific behavior.
    # These fields are set when getting on/ off a ladder. We could also have just decided to set these fields when we are leaving a room 
    # and are still on a ladder, but then we would have had to implement this behavior in the "leave_room" function, which I don't want to do, 
    # as it's ladder specific behavior and should be handled in the ladder routines as much as possible. 
    # Similar getting on the actual ladder is handled in the ladder step as well.
    get_on_ladder_bottom: jArray
    get_on_ladder_top: jArray
    
    darkness: jnp.ndarray #boolean, if true the room is dark
    room_enter_frame: jnp.ndarray
    hammer_time: jArray # The remaining duration of the hammer pickup.
    first_item_pickup_frame: jnp.ndarray
    rng_key: jnp.ndarray
        
    ## All the logic that is responsible for handling game freezes; ALL SINGLETON INT32 fields.
    queue_freeze: jArray # Sets wether a freeze is executed at the end of this step. If a freeze is executed, 
        # The game is rendered using the pre-step state, but resumed from the post-step state.
    frozen: jArray # Whether the game is currently frozen.
    freeze_type: jArray
    freeze_remaining: jArray
    frozen_state: NamedTuple
    # The post step state after the freeze was queued, this state is used to resume 
    # the game after the freeze.
    
    current_velocity: jArray # What the current velocity is
    last_velocity: jArray # What the velocity of the last step is
    velocity_held: jArray # How many frames the current velocity has been held for
    horizontal_direction_held: jArray # How long we have given the same horizontal direction input. 
        # Used for animations
    
    rope_climb_frame: jArray # Which frame pointer we are currently on for the rope climbing animations
    ladder_climb_frame: jArray # Which frame pointer we are currently on for the ladder climb animation
    walk_frame: jArray # Which animation frame we are currently on for the walking animation.
    fall_position: jArray # When a player starts to fall, this attribute is set to the position from which he last fell from.
    # These fields are used to make sure, that if the player entered the room via a ladder, 
    # he also respawns on the ladder.
    top_seeking_on_entrance: jArray
    bottom_seeking_at_entrance: jArray
    # This field is used to account for small inconsistencies in the fall-death animation.
    render_offset_for_fall_dmg: jnp.ndarray
    is_jump_hold: jnp.ndarray # Whether the current jump input is a key-hold; Used to prevent bunny-hopping.
    
    reset_room_state_on_room_change: jnp.ndarray # Whether the global room state should be reset on the next room transition. 
        # This is used to implement looping behavior for the bonus room. 
        # We need to place it in the state in addition to the bonus_room tag, as cross-room behavior cannot be fully 
        # handled within tags.
    observation: MontezumaObservation
    
class Layouts(Enum):
    test_layout = "test_layout"
    demo_layout = "demo_layout"
    difficulty_1 = "difficulty_1"
    difficulty_2 = "difficulty_2"
    
class MontezumaStateFields(Enum):
    room_state = "room_state"   
    persistence_state = "persistence_state"
    player_position = "player_position" 
    player_collision_map = "player_collision_map"
    player_velocity = "player_velocity"
    vertical_directional_input = "vertical_directional_input"
    horizontal_direction = "horizontal_direction" 
    last_horizontal_direction = "last_horizontal_direction"
    last_horizontal_orientation = "last_horizontal_orientation"
    horizontal_falling_velocitiy = "horizontal_falling_velocitiy"
    is_standing = "is_standing"
    current_directional_input = "current_directional_input" 
    is_jumping = "is_jumping" 
    is_falling = "is_falling" 
    jump_counter = "jump_counter" 
    augmented_collision_map = "augmented_collision_map" 
    canvas = "canvas"
    jump_input = "jump_input"
    frame_count = "frame_count"
    is_dying = "is_dying" 
    last_entrance_direction = "last_entrance_direction"
    is_climbing = "is_climbing"
    is_laddering = "is_laddering"
    score = "score"
    lifes = "lifes"
    itembar_items = "itembar_items"
    is_on_rope = "is_on_rope"
    is_key_hold = "is_key_hold"
    last_key_press = "last_key_press"
    force_jump = "force_jump"
    disable_directional_input = "disable_directional_input"
    queue_enable_directional_input = "queue_enable_directional_input"
    get_on_ladder_bottom = "get_on_ladder_bottom"
    get_on_ladder_top = "get_on_ladder_top"
    darkness = "darkness"
    room_enter_frame = "room_enter_frame"
    hammer_time = "hammer_time"
    first_item_pickup_frame = "first_item_pickup_frame"
    rng_key = "rng_key"
    queue_freeze = "queue_freeze"
    frozen = "frozen" 
    freeze_type = "freeze_type"
    freeze_remaining = "freeze_remaining"
    frozen_state = "frozen_state"
    
    current_velocity = "current_velocity"
    last_velocity = "last_velocity"
    velocity_held = "velocity_held"
    horizontal_direction_held = "horizontal_direction_held"
    
    rope_climb_frame = "rope_climb_frame"
    ladder_climb_frame = "ladder_climb_frame"
    walk_frame = "walk_frame"
    fall_position = "fall_position"
    top_seeking_on_entrance = "top_seeking_on_entrance"
    bottom_seeking_at_entrance = "bottom_seeking_at_entrance"
    render_offset_for_fall_dmg = "render_offset_for_fall_dmg"
    is_jump_hold = "is_jump_hold"
    reset_room_state_on_room_change = "reset_room_state_on_room_change"
    observation = "observation"
    
monte_fields_1 = set([f.value for f in MontezumaStateFields])
monte_fields_2 = set(MontezumaState._fields)
if monte_fields_1 != monte_fields_2:
    raise Exception("MontezumeStateFields and MontezumaState need to have the same fields!")

class MontezumaInfo(NamedTuple):
    step_counter: jArray # Step of the game we are currently on
    lives: jArray # How many lives the player has left
    all_rewards: jArray # Rewards accumulated throughout the whole episode
    room_id: jArray # Id of the room the player is currently in.

class RoomColorForDificulty(NamedTuple):
    l1_p: jnp.ndarray
    l1_s: jnp.ndarray
    l2_p: jnp.ndarray
    l2_s: jnp.ndarray
    l3_p: jnp.ndarray
    l3_s: jnp.ndarray
    l4_p: jnp.ndarray
    l4_s: jnp.ndarray
    
class RoomColors:
    dif_1 = RoomColorForDificulty(
        l1_p=jnp.array([66, 158, 130]),
        l1_s=jnp.array([24, 59, 157]),
        l2_p=jnp.array([104, 25, 157]),
        l2_s=jnp.array([204, 216, 110]),
        l3_p=jnp.array([45, 87, 176]),
        l3_s=jnp.array([92, 186, 92]),
        l4_p=jnp.array([24, 26, 167]),
        l4_s=jnp.array([213, 130, 74])
        )
    dif_2 = RoomColorForDificulty(
        l1_p=jnp.array([104, 25, 154]),
        l1_s=jnp.array([204, 216, 110]),
        l2_p=jnp.array([45, 87, 176]),
        l2_s=jnp.array([92, 186, 92]),
        l3_p=jnp.array([24, 26, 167]),
        l3_s=jnp.array([213, 130, 74]),
        l4_p=jnp.array([72, 160, 72]),
        l4_s=jnp.array([24, 26, 167])
        )
    dif_3 = RoomColorForDificulty(
        l1_p=jnp.array([45, 87, 176]),
        l1_s=jnp.array([92, 186, 92]),
        l2_p=jnp.array([24, 26, 167]),
        l2_s=jnp.array([213, 130, 74]),
        l3_p=jnp.array([72, 160, 72]),
        l3_s=jnp.array([24, 26, 167]),
        l4_p=jnp.array([51, 26, 163]),
        l4_s=jnp.array([223, 183, 85])
        )
    
class ObstacleColors(Enum):   
    BLACK = 0
    WHITE = 1
    DOOR_COLOR_NORMAL = 2
    ROPE_COLOR_NORMAL = 3
    ROPE_COLOR_WHITE = 4
    DIF_1_LAYER_1_PRIMARY = 5
    DIF_1_LAYER_2_PRIMARY = 6
    DIF_1_LAYER_3_PRIMARY = 7
    DIF_1_LAYER_4_PRIMARY = 8
    DIF_1_LAYER_1_SECONDARY = 9
    DIF_1_LAYER_2_SECONDARY = 10
    DIF_1_LAYER_3_SECONDARY = 11
    DIF_1_LAYER_4_SECONDARY = 12
    DIF_2_LAYER_1_PRIMARY = 13
    DIF_2_LAYER_2_PRIMARY = 14
    DIF_2_LAYER_3_PRIMARY = 15
    DIF_2_LAYER_4_PRIMARY = 16
    DIF_2_LAYER_1_SECONDARY = 17
    DIF_2_LAYER_2_SECONDARY = 18
    DIF_2_LAYER_3_SECONDARY = 19
    DIF_2_LAYER_4_SECONDARY = 20
    DIF_3_LAYER_1_PRIMARY = 21
    DIF_3_LAYER_2_PRIMARY = 22
    DIF_3_LAYER_3_PRIMARY = 23
    DIF_3_LAYER_4_PRIMARY = 24
    DIF_3_LAYER_1_SECONDARY = 25
    DIF_3_LAYER_2_SECONDARY = 26
    DIF_3_LAYER_3_SECONDARY = 27
    DIF_3_LAYER_4_SECONDARY = 28
    
    
class MovementDirection(Enum):
    LEFT = 0
    LEFT_UP = 1
    UP = 2
    RIGHT_UP = 3
    RIGHT = 4
    RIGHT_DOWN = 5
    DOWN = 6
    LEFT_DOWN = 7
    NO_DIR = 8

class JUMP_INPUT(Enum):
    NO = 0
    YES = 1
    
    
class FreezeType(Enum):
    FALL_DEATH = 0
    KILLED_BY_MONSTER = 1
    KILLED_A_MONSTER = 2
    LAZER_BARRIER_DEATH = 3
    SARLACC_PIT_DEATH = 4
    ITEM_PICKUP = 5
    DOOR_UNLOCK = 6



class Horizontal_Direction(Enum):
    LEFT = 0
    NO_DIR = 1
    RIGHT = 2


class VERTICAL_Direction(Enum):
    UP = 0
    NO_DIR = 1
    DOWN = 2

    
class Item_Sprites(Enum):
    GEM = 0
    HAMMER = 1
    KEY = 2
    SWORD = 3
    TORCH = 4
    TORCH_FRAME_2 = 5
    
class ItemBar_Sprites(Enum):
    HAMMER = Item_Sprites.HAMMER.value
    KEY = Item_Sprites.KEY.value
    SWORD = Item_Sprites.SWORD.value
    TORCH = Item_Sprites.TORCH.value
    


class Item(NamedTuple):
    sprite: Item_Sprites
    x: jnp.ndarray
    y: jnp.ndarray
    on_field: jnp.ndarray
    
class ItemEnum(Enum):
    sprite = "sprite"
    x = "x"
    y = "y"
    on_field = "on_field"

class Dropout_Floor_Sprites(Enum):
    PIT_FLOOR = 0
    LADDER_FLOOR = 1
    
    
class EnemyType(Enum):
    SNAKE = 0
    ROLL_SKULL = 1
    BOUNCE_SKULL = 2
    SPIDER = 3
    
class Enemy(NamedTuple):
    bbox_left_upper_x: jArray # Corner coordinates of the bounding box in which the enemy lives
        # the enemy ignores all level geometry and moves freely inside the bounding box.
    bbox_left_upper_y: jArray
    bbox_right_lower_x: jArray
    bbox_right_lower_y: jArray
    enemy_type: jArray # Enemy type. Needs to have a value in the EnemyType enum
    alive: jArray # Whether the enemy is currently alive
    pos_x: jArray
    pos_y: jArray
    horizontal_direction: jArray # the horizontal direction in which the enemy is currently moving
    last_movement: jArray # How long ago the last movement occured
    sprite_index: jArray # Which sprite is currently to be rendered
    render_in_reverse: jArray # Whether the sprites are currently rendered in reverse order
    initial_x_pos: jArray # Start position of the enemy.
    initial_y_pos: jArray
    initial_horizontal_direction: jArray # Initial movement direction of the enemy.
    initial_render_in_reverse: jArray
    optional_movement_counter: jArray # An optional counter for movement. 
        # For enemies that have complex movement patterns (like bouncing skulls)
        # This field is used to signify the current position in the cycle
        # For other enemies, it is unused.
    last_animation: jArray # How many frames ago the enemy was last animated.
    optional_utility_field: jArray # We use this one to split the bouncing skull enemy into two
    
class BounceSkullAliveState(Enum):
    FULLY_ALIVE = 0
    LEFT_DEAD = 1
    RIGHT_DEAD = 2
    ALL_DEAD = 3    

class EnemyFields(Enum):
    bbox_left_upper_x = "bbox_left_upper_x"
    bbox_left_upper_y = "bbox_left_upper_y"
    bbox_right_lower_x = "bbox_right_lower_x"
    bbox_right_lower_y = "bbox_right_lower_y"
    enemy_type = "enemy_type"
    alive = "alive"
    pos_x = "pos_x"
    pos_y = "pos_y"
    horizontal_direction = "horizontal_direction"
    last_movement = "last_movement"
    sprite_index = "sprite_index"
    render_in_reverse = "render_in_reverse"
    initial_x_pos = "initial_x_pos"
    initial_y_pos = "initial_y_pos"
    initial_horizontal_direction = "initial_horizontal_direction"
    initial_render_in_reverse = "initial_render_in_reverse"
    optional_movement_counter = "optional_movement_counter"
    last_animation = "last_animation"
    optional_utility_field = "optional_utility_field"








#
#
#
# Register all the enums I have made!!!
#
#
#
#



SANTAH.add_new_named_tuple(MandatoryLazerBarrierFields, field_enum=MandatoryLazerBarrierFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryItemFields, field_enum=MandatoryItemFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryRopeFields, field_enum=MandatoryRopeFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryDoorFields, field_enum=MandatoryDoorFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryDropoutFloorFields,field_enum=MandatoryDropoutFloorFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryPitFields,field_enum=MandatoryPitFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryLadderFields, field_enum=MandatoryLadderFieldsEnum)
SANTAH.add_new_named_tuple(MandatorySidewallsFields, field_enum=MandatorySidewallsFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryEnemyFields, field_enum=MandatoryEnemyFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryConveyorBeltFields, field_enum=MandatoryConveyorBeltFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryDarkroomFields,field_enum=MandatoryDarkroomFieldsEnum)
SANTAH.add_new_named_tuple(MandatoryBonusRoomFields,field_enum=MandatoryBonusRoomFieldsEnum,
                           partial_serialisation_fields=[MandatoryBonusRoomFieldsEnum.bouns_cycle_index.value])


SANTAH.add_new_named_tuple(MontezumaState, field_enum=MontezumaStateFields)
SANTAH.add_new_named_tuple(tup=LAZER_BARRIER, field_enum=LAZER_BARRIER_ENUM)
SANTAH.add_new_named_tuple(tup=GlobalLazerBarrierInfo, field_enum=GlobalLazerBarrierInfoEnum,
                                                   partial_serialisation_fields=[GlobalLazerBarrierInfoEnum.cycle_index.value])
SANTAH.add_new_named_tuple(tup=Item, field_enum=ItemEnum,
                                                           partial_serialisation_fields=[ItemEnum.on_field.value,
                                                                                         ItemEnum.x.value])
SANTAH.add_new_named_tuple(Rope, field_enum=RopeEnum)
SANTAH.add_new_named_tuple(Door, field_enum=DoorEnum,
                                                           partial_serialisation_fields=[DoorEnum.on_field.value])
SANTAH.add_new_named_tuple(DropoutFloor,field_enum=DropoutFloorEnum)
SANTAH.add_new_named_tuple(Ladder, field_enum=LadderFields)
SANTAH.add_new_named_tuple(Enemy, field_enum=EnemyFields, 
                           partial_serialisation_fields=[EnemyFields.alive.value, 
                                                         EnemyFields.horizontal_direction.value,
                                                         EnemyFields.last_movement.value, 
                                                         EnemyFields.pos_x.value, 
                                                         EnemyFields.pos_y.value, 
                                                         EnemyFields.render_in_reverse.value, 
                                                         EnemyFields.sprite_index.value, 
                                                         EnemyFields.optional_movement_counter.value, 
                                                         EnemyFields.last_animation.value, 
                                                         EnemyFields.optional_utility_field.value])
SANTAH.add_new_named_tuple(ConveyorBelt, field_enum=ConveyorBeltEnum)