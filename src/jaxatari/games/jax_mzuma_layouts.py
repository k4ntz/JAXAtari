from jaxatari.games.jax_mzuma_utils import *
from jaxatari.games.jax_mzuma_utils import Room as uRoom
from jaxatari.games.jax_mzuma_enums_and_nts import *
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_PATH: str = os.path.join(MODULE_DIR, "sprites", "montezuma")
SPRITE_PATH_BACKGROUND: str = os.path.join(SPRITE_PATH, "backgrounds")

def make_layout_test(LAYOUT: PyramidLayout, consts) -> PyramidLayout:
        ROOM_1_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_1_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32),
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),  
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom, ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([92], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([92], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        return LAYOUT


def make_demo_layout(LAYOUT: PyramidLayout, consts) -> PyramidLayout:
    ROOM_0_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
    ROOM_0_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_0.npy"), transpose=True), 
                     requires_serialisation=False)
    ROOM_0_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_0.npy"), as_bool=False, transpose=True), 
                     requires_serialisation=False)
    ROOM_0_1.set_field(field_name=VanillaRoomFields.height.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([151], jnp.int32))
    ROOM_0_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([47], jnp.int32))
    ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([4, 27], dtype=jnp.uint16))
    ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([148, 27], dtype=jnp.uint16))
    ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([77, 26], dtype=jnp.uint16))
    ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([77, 26], dtype=jnp.uint16))
    
    item_0 = Item(
        sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
        x=jnp.array([12],dtype=jnp.int32),
        y=jnp.array([53],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32)
    )
    
    item_1 = Item(
        sprite=jnp.array([Item_Sprites.SWORD.value],dtype=jnp.int32),
        x=jnp.array([141],dtype=jnp.int32),
        y=jnp.array([53],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32)
    )
    ROOM_0_1.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                     content=[item_0, item_1],
                     named_tuple_type=Item,
                     requires_serialisation=True
                     )
    
    ladder_mid = Ladder(
        left_upper_x=jnp.array([72], jnp.int32), 
        left_upper_y=jnp.array([49], jnp.int32), 
        right_lower_x=jnp.array([88], jnp.int32), 
        right_lower_y=jnp.array([88], jnp.int32), 
        has_background=jnp.array([1], jnp.int32), 
        rope_seeking_at_bottom=jnp.array([0], jnp.int32),
        rope_seeking_at_top=jnp.array([0], jnp.int32),
        foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value],jnp.int32),
        background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
        transparent_background=jnp.array([0], jnp.int32), 
        transparent_foreground=jnp.array([0], jnp.int32)
    )
    
    ladder_right = Ladder(
        left_upper_x=jnp.array([128], jnp.int32), 
        left_upper_y=jnp.array([92], jnp.int32), 
        right_lower_x=jnp.array([144], jnp.int32), 
        right_lower_y=jnp.array([133], jnp.int32), 
        has_background=jnp.array([1], jnp.int32), 
        rope_seeking_at_bottom=jnp.array([0], jnp.int32),
        rope_seeking_at_top=jnp.array([0], jnp.int32), 
        foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value],jnp.int32),
        background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
        transparent_background=jnp.array([1], jnp.int32), 
        transparent_foreground=jnp.array([0], jnp.int32)
    )
    
    ladder_left = Ladder(
        left_upper_x=jnp.array([16], jnp.int32), 
        left_upper_y=jnp.array([92], jnp.int32), 
        right_lower_x=jnp.array([32], jnp.int32), 
        right_lower_y=jnp.array([133], jnp.int32), 
        has_background=jnp.array([1], jnp.int32), 
        rope_seeking_at_bottom=jnp.array([0], jnp.int32),
        rope_seeking_at_top=jnp.array([0], jnp.int32), 
        foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value],jnp.int32),
        background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
        transparent_background=jnp.array([1], jnp.int32), 
        transparent_foreground=jnp.array([0], jnp.int32)
    )
    
    ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[ladder_mid, ladder_right, ladder_left], 
                     requires_serialisation=False, 
                     named_tuple_type=Ladder)
    
    ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], dtype=jnp.int32), 
                     requires_serialisation=True)
    ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    
    door_0 = Door(
        x=jnp.array([16],dtype=jnp.int32),
        y=jnp.array([7],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32),
        color=jnp.array([ObstacleColors.DOOR_COLOR_NORMAL.value], dtype=jnp.int32)
    )

    door_1 = Door(
        x=jnp.array([140],dtype=jnp.int32),
        y=jnp.array([7],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32),
        color=jnp.array([ObstacleColors.DOOR_COLOR_NORMAL.value], dtype=jnp.int32)
    )


    ROOM_0_1.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                    field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                    content=[door_0, door_1],
                    named_tuple_type=Door,
                    requires_serialisation=True
                    )
    
    rope_0 = Rope(
        x_pos=jnp.array([111], dtype=jnp.int32),
        top=jnp.array([49], dtype=jnp.int32),
        bottom=jnp.array([88], dtype=jnp.int32), 
        color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32), 
        is_climbable=jnp.array([1], dtype=jnp.int32), 
        accessible_from_top=jnp.array([0], dtype=jnp.int32)
    )
    
    ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[rope_0], 
                     requires_serialisation=True, 
                     named_tuple_type=Rope
                     )
    
    ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], jnp.int32), 
                     requires_serialisation=True
                     )
    ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], jnp.int32), 
                     requires_serialisation=True
                     )
    
    enemy = Enemy(
        bbox_left_upper_x=jnp.array([56], jnp.int32), 
        bbox_left_upper_y=jnp.array([146], jnp.int32), 
        bbox_right_lower_x=jnp.array([112], jnp.int32), 
        bbox_right_lower_y=jnp.array([132], jnp.int32), 
        enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
        alive=jnp.array([1], jnp.int32), 
        pos_x=jnp.array([93], jnp.int32), 
        pos_y=jnp.array([119], jnp.int32), 
        horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
        last_movement=jnp.array([0], jnp.int32), 
        sprite_index=jnp.array([0], jnp.int32), 
        render_in_reverse=jnp.array([0], jnp.int32), 
        initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
        initial_render_in_reverse=jnp.array([0], jnp.int32), 
        optional_movement_counter=jnp.array([0], jnp.int32), 
        initial_x_pos=jnp.array([93], jnp.int32), 
        initial_y_pos=jnp.array([119], jnp.int32),
        last_animation=jnp.array([0], jnp.int32), 
        optional_utility_field=jnp.array([0], jnp.int32)
    )

    ROOM_0_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[enemy], requires_serialisation=True, 
                     named_tuple_type=Enemy)
    
    conveyor_0 = ConveyorBelt(
        x=jnp.array([60], dtype=jnp.int32),
        y=jnp.array([89], dtype=jnp.int32),
        movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
        color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value], dtype=jnp.int32) 
    )
    ROOM_0_1.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                               field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                               content=[conveyor_0],
                               named_tuple_type=ConveyorBelt)



    ROOM_0_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.LAZER_BARRIER, RoomTags.LADDERS, RoomTags.ENEMIES]))
    ROOM_0_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_0.npy"), transpose=True), 
                     requires_serialisation=False)
    
    ROOM_0_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                     requires_serialisation=False)
    
    ROOM_0_0.set_field(field_name=VanillaRoomFields.height.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([149], jnp.int32))
    
    ROOM_0_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([47], jnp.int32))
    
    ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([4, 27], dtype=jnp.uint16))
    ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([148, 27], dtype=jnp.uint16))
    ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    
    item_0 = Item(
        sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
        x=jnp.array([24],dtype=jnp.int32),
        y=jnp.array([7],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32)
    )
    
    item_1 = Item(
        sprite=jnp.array([Item_Sprites.TORCH.value],dtype=jnp.int32),
        x=jnp.array([129],dtype=jnp.int32),
        y=jnp.array([7],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32)
    )
    ROOM_0_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                     content=[item_0, item_1],
                     named_tuple_type=Item,
                     requires_serialisation=True
                     )
    
    global_barrier_info = GlobalLazerBarrierInfo(
         cycle_length=jnp.array([128], jnp.int32), 
         cycle_active_frames=jnp.array([92], jnp.int32), 
         cycle_offset=jnp.array([0], jnp.int32), 
         cycle_index=jnp.array([0], jnp.int32)
    )
    ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                      field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                      content=[global_barrier_info], 
                      requires_serialisation=True, 
                      named_tuple_type=GlobalLazerBarrierInfo)
    barrier_0 = LAZER_BARRIER(
        X=jnp.array([16], jnp.int32), 
        upper_point=jnp.array([7], jnp.int32),
        lower_point=jnp.array([46], jnp.int32)
    )
    barrier_1 = LAZER_BARRIER(
        X=jnp.array([36], jnp.int32), 
        upper_point=jnp.array([7], jnp.int32),
        lower_point=jnp.array([46], jnp.int32)
    )
    barrier_2 = LAZER_BARRIER(
        X=jnp.array([44], jnp.int32), 
        upper_point=jnp.array([7], jnp.int32),
        lower_point=jnp.array([46], jnp.int32)
    )
    
    barrier_3 = LAZER_BARRIER(
        X=jnp.array([112], jnp.int32),
        upper_point=jnp.array([7], jnp.int32),
        lower_point=jnp.array([46], jnp.int32)
    )
    
    barrier_4 = LAZER_BARRIER(
        X=jnp.array([120], jnp.int32),
        upper_point=jnp.array([7], jnp.int32),
        lower_point=jnp.array([46], jnp.int32)
    )
    
    barrier_5 = LAZER_BARRIER(
        X=jnp.array([140], jnp.int32),
        upper_point=jnp.array([7], jnp.int32),
        lower_point=jnp.array([46], jnp.int32)
    )
    
    ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5], 
                     requires_serialisation=False, 
                     named_tuple_type=LAZER_BARRIER)
    
    ladder_to_bottom = Ladder(
        left_upper_x=jnp.array([72], jnp.int32), 
        left_upper_y=jnp.array([48], jnp.int32), 
        right_lower_x=jnp.array([88], jnp.int32), 
        right_lower_y=jnp.array([149], jnp.int32), 
        has_background=jnp.array([1], jnp.int32), 
        rope_seeking_at_bottom=jnp.array([1], jnp.int32),
        rope_seeking_at_top=jnp.array([0], jnp.int32),
        foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_SECONDARY.value], jnp.int32),
        background_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value], jnp.int32),
        transparent_background=jnp.array([0], jnp.int32), 
        transparent_foreground=jnp.array([0], jnp.int32)
    )
    
    ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[ladder_to_bottom], 
                     requires_serialisation=False, 
                     named_tuple_type=Ladder)
    
    ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], dtype=jnp.int32), 
                     requires_serialisation=True)
    ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    
    enemy = Enemy(
        bbox_left_upper_x=jnp.array([4], jnp.int32),
        bbox_left_upper_y=jnp.array([7], jnp.int32), 
        bbox_right_lower_x=jnp.array([124], jnp.int32),
        bbox_right_lower_y=jnp.array([45], jnp.int32),
        enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
        alive=jnp.array([1], jnp.int32), 
        pos_x=jnp.array([101], jnp.int32), 
        pos_y=jnp.array([33], jnp.int32), 
        horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
        last_movement=jnp.array([0], jnp.int32), 
        sprite_index=jnp.array([0], jnp.int32), 
        render_in_reverse=jnp.array([0], jnp.int32), 
        initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
        initial_render_in_reverse=jnp.array([0], jnp.int32), 
        optional_movement_counter=jnp.array([0], jnp.int32), 
        initial_x_pos=jnp.array([101], jnp.int32), 
        initial_y_pos=jnp.array([33], jnp.int32),
        last_animation=jnp.array([0], jnp.int32), 
        optional_utility_field=jnp.array([0], jnp.int32)
    )
    
    ROOM_0_0.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[enemy], requires_serialisation=True, 
                     named_tuple_type=Enemy)
                     
    ROOM_2_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.DROPOUTFLOORS, RoomTags.ENEMIES]))
    ROOM_2_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_level_2.npy"), transpose=True), 
                     requires_serialisation=False)
    ROOM_2_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_collision_level_2.npy"), as_bool=False, transpose=True), 
                     requires_serialisation=False)
    ROOM_2_0.set_field(field_name=VanillaRoomFields.height.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([151], jnp.int32))
    ROOM_2_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([47], jnp.int32))
    ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([4, 27], dtype=jnp.uint16))
    ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([148, 27], dtype=jnp.uint16))
    ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    
    item_0 = Item(
        sprite=jnp.array([Item_Sprites.HAMMER.value],dtype=jnp.int32),
        x=jnp.array([77],dtype=jnp.int32),
        y=jnp.array([7],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32)
    )
    ROOM_2_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                     content=[item_0],
                     named_tuple_type=Item,
                     requires_serialisation=True
                     )
    
    rope_0 = Rope(
        x_pos=jnp.array([80], dtype=jnp.int32),
        top=jnp.array([49], dtype=jnp.int32),
        bottom=jnp.array([100], dtype=jnp.int32), 
        color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32), 
        is_climbable=jnp.array([1], dtype=jnp.int32), 
        accessible_from_top=jnp.array([1], dtype=jnp.int32)
    )
    
    ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[rope_0], 
                     requires_serialisation=True, 
                     named_tuple_type=Rope
                     )
    
    ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], jnp.int32), 
                     requires_serialisation=True
                     )
    ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], jnp.int32), 
                     requires_serialisation=True
                     )
    
    dfloor_l_0 = DropoutFloor(
         x=jnp.array([4], dtype=jnp.int32),
         y=jnp.array([56], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_l_1 = DropoutFloor(
         x=jnp.array([4], dtype=jnp.int32),
         y=jnp.array([66], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_l_2 = DropoutFloor(
         x=jnp.array([4], dtype=jnp.int32),
         y=jnp.array([76], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_l_3 = DropoutFloor(
         x=jnp.array([4], dtype=jnp.int32),
         y=jnp.array([86], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_l_4 = DropoutFloor(
         x=jnp.array([4], dtype=jnp.int32),
         y=jnp.array([106], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_l_5 = DropoutFloor(
         x=jnp.array([4], dtype=jnp.int32),
         y=jnp.array([116], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_r_0 = DropoutFloor(
         x=jnp.array([144], dtype=jnp.int32),
         y=jnp.array([56], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_r_1 = DropoutFloor(
         x=jnp.array([144], dtype=jnp.int32),
         y=jnp.array([66], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_r_2 = DropoutFloor(
         x=jnp.array([144], dtype=jnp.int32),
         y=jnp.array([76], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_r_3 = DropoutFloor(
         x=jnp.array([144], dtype=jnp.int32),
         y=jnp.array([86], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_r_4 = DropoutFloor(
         x=jnp.array([144], dtype=jnp.int32),
         y=jnp.array([106], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    dfloor_r_5 = DropoutFloor(
         x=jnp.array([144], dtype=jnp.int32),
         y=jnp.array([116], dtype=jnp.int32),
         sprite_height_amount=jnp.array([1], dtype=jnp.int32),
         sprite_width_amount=jnp.array([1], dtype=jnp.int32),
         sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
         collision_padding_top=jnp.array([0], dtype=jnp.int32),
         color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
         )
    
    ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                     content=[dfloor_l_0, dfloor_l_1, dfloor_l_2, dfloor_l_3, dfloor_l_4, dfloor_l_5, dfloor_r_0, dfloor_r_1, dfloor_r_2, dfloor_r_3, dfloor_r_4, dfloor_r_5],
                     named_tuple_type=DropoutFloor)
    ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([120], dtype=jnp.int32))
    ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([30], dtype=jnp.int32))
    
    enemy = Enemy(
        bbox_left_upper_x=jnp.array([24], jnp.int32), 
        bbox_left_upper_y=jnp.array([36], jnp.int32), 
        bbox_right_lower_x=jnp.array([136], jnp.int32), 
        bbox_right_lower_y=jnp.array([126], jnp.int32), 
        enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
        alive=jnp.array([1], jnp.int32), 
        pos_x=jnp.array([92], jnp.int32), 
        pos_y=jnp.array([36], jnp.int32), 
        horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
        last_movement=jnp.array([0], jnp.int32), 
        sprite_index=jnp.array([0], jnp.int32), 
        render_in_reverse=jnp.array([0], jnp.int32), 
        initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
        initial_render_in_reverse=jnp.array([0], jnp.int32), 
        optional_movement_counter=jnp.array([0], jnp.int32), 
        initial_x_pos=jnp.array([92], jnp.int32), 
        initial_y_pos=jnp.array([36], jnp.int32),
        last_animation=jnp.array([0], jnp.int32), 
        optional_utility_field=jnp.array([0], jnp.int32)
    )
    
    ROOM_2_0.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[enemy], requires_serialisation=True, 
                     named_tuple_type=Enemy)
                     
                     
                     
    ROOM_3_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS,RoomTags.BONUSROOM,RoomTags.SIDEWALLS]))
    ROOM_3_0.set_field(field_name=VanillaRoomFields.sprite.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "bonus_room_sprite.npy"), transpose=True),
                     requires_serialisation=False)
    ROOM_3_0.set_field(field_name=VanillaRoomFields.room_collision_map.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND,"bonus_room_collision_map.npy"),as_bool=False, transpose=True),
                     requires_serialisation=False)
    ROOM_3_0.set_field(field_name=VanillaRoomFields.height.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([148], jnp.int32))
    ROOM_3_0.set_field(field_name=VanillaRoomFields.vertical_offset.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([47], jnp.int32))
    ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([5, 26], dtype=jnp.uint16))
    ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([130, 10], dtype=jnp.uint16))
    ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    item_0 = Item(
        sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
        x=jnp.array([consts.DEFAULT_BONUS_ROOM_GEM_X],dtype=jnp.int32),
        y=jnp.array([7],dtype=jnp.int32),
        on_field=jnp.array([1],dtype=jnp.int32)
    )
    ROOM_3_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                     content=[item_0],
                     named_tuple_type=Item,
                     requires_serialisation=True
                     )
    ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bonus_cycle_lenght.value,
                               field_type=NamedTupleFieldType.INTEGER_SCALAR,
                               content=jnp.array([620], jnp.int32))
    ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bouns_cycle_index.value,
                               field_type=NamedTupleFieldType.INTEGER_SCALAR,
                               content=jnp.array([0], jnp.int32))
    ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.reset_state_on_leave.value,
                               field_type=NamedTupleFieldType.INTEGER_SCALAR,
                               content=jnp.array([1], jnp.int32))
    ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([1], dtype=jnp.int32))
    ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([1], dtype=jnp.int32))
    ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=jnp.array([consts.BONUS_ROOM_COLOR], dtype=jnp.int32))
                     
                     
                     
    ROOM_3_7 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
    ROOM_3_7.set_field(field_name=VanillaRoomFields.sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom.npy"), transpose=True), 
                     requires_serialisation=False)
    ROOM_3_7.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                     requires_serialisation=False)
    ROOM_3_7.set_field(field_name=VanillaRoomFields.height.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([149], jnp.int32))
    ROOM_3_7.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([47], jnp.int32))
    ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([4, 27], dtype=jnp.uint16))
    ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([148, 27], dtype=jnp.uint16))
    ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([10, 10], dtype=jnp.uint16))
    
    dfloor_0 = DropoutFloor(
       x=jnp.array([36], dtype=jnp.int32),
       y=jnp.array([48], dtype=jnp.int32),
       sprite_height_amount=jnp.array([1], dtype=jnp.int32),
       sprite_width_amount=jnp.array([11], dtype=jnp.int32),
       sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
       collision_padding_top=jnp.array([1], dtype=jnp.int32),
       color=jnp.array([ObstacleColors.DIF_1_LAYER_4_PRIMARY.value], dtype=jnp.int32)
       )
    
    ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                    field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                    content=[dfloor_0],
                    named_tuple_type=DropoutFloor)
    
    ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                    field_type=NamedTupleFieldType.INTEGER_SCALAR,
                    content=jnp.array([92], dtype=jnp.int32))
    
    ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                    field_type=NamedTupleFieldType.INTEGER_SCALAR,
                    content=jnp.array([36], dtype=jnp.int32))
    
    ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([75], dtype=jnp.int32))
    
    ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
    
    ladder_to_top = Ladder(
        left_upper_x=jnp.array([72], jnp.int32), 
        left_upper_y=jnp.array([6], jnp.int32), 
        right_lower_x=jnp.array([88], jnp.int32), 
        right_lower_y=jnp.array([44], jnp.int32), 
        has_background=jnp.array([1], jnp.int32), 
        rope_seeking_at_bottom=jnp.array([0], jnp.int32),
        rope_seeking_at_top=jnp.array([1], jnp.int32), 
        foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_4_SECONDARY.value], jnp.int32),
        background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32), 
        transparent_background=jnp.array([0], jnp.int32), 
        transparent_foreground=jnp.array([0], jnp.int32)
    )
    ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[ladder_to_top], 
                     requires_serialisation=False, 
                     named_tuple_type=Ladder)
    ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                     field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                     content=jnp.array([-1], dtype=jnp.int32), 
                     requires_serialisation=True)
    ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                     field_type=NamedTupleFieldType.OTHER_ARRAY, 
                     content=jnp.array([0], dtype=jnp.int32), 
                     requires_serialisation=False)
    
    enemy0 = Enemy(
        bbox_left_upper_x=jnp.array([30], jnp.int32), 
        bbox_left_upper_y=jnp.array([34], jnp.int32), 
        bbox_right_lower_x=jnp.array([37], jnp.int32), 
        bbox_right_lower_y=jnp.array([47], jnp.int32), 
        enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
        alive=jnp.array([1], jnp.int32), 
        pos_x=jnp.array([30], jnp.int32), 
        pos_y=jnp.array([38], jnp.int32), 
        horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
        last_movement=jnp.array([0], jnp.int32), 
        sprite_index=jnp.array([0], jnp.int32), 
        render_in_reverse=jnp.array([0], jnp.int32), 
        initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
        initial_render_in_reverse=jnp.array([0], jnp.int32), 
        optional_movement_counter=jnp.array([0], jnp.int32), 
        initial_x_pos=jnp.array([30], jnp.int32), 
        initial_y_pos=jnp.array([38], jnp.int32),
        last_animation=jnp.array([0], jnp.int32), 
        optional_utility_field=jnp.array([0], jnp.int32)   
    )
    
    ROOM_3_7.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                     field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                     content=[enemy0], requires_serialisation=True, 
                     named_tuple_type=Enemy)
                     
                     
                     
    ROOM_0_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                      other_room=ROOM_0_1, other_location=RoomConnectionDirections.LEFT)
    ROOM_0_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                      other_room=ROOM_3_7, other_location=RoomConnectionDirections.UP)
    ROOM_3_7.connect_to(my_location=RoomConnectionDirections.UP,
                        other_room=ROOM_0_0, other_location=RoomConnectionDirections.DOWN)
    ROOM_3_7.connect_to(my_location=RoomConnectionDirections.LEFT,
                        other_room=ROOM_3_0, other_location=RoomConnectionDirections.RIGHT)
    ROOM_0_0.connect_to(my_location=RoomConnectionDirections.LEFT,
                      other_room=ROOM_2_0, other_location=RoomConnectionDirections.RIGHT)
    ROOM_3_0.connect_to(my_location=RoomConnectionDirections.DOWN, other_room=ROOM_0_1, 
                        other_location=RoomConnectionDirections.UP)
    return LAYOUT



def make_difficulty_1(LAYOUT: PyramidLayout, consts, last_room: uRoom = None, self_loop: bool = True) -> PyramidLayout:
        
        ROOM_0_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
        ROOM_0_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_0.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_0_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_0.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_0_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([151], jnp.int32))
        ROOM_0_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([77, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([13],dtype=jnp.int32),
            y=jnp.array([52],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        ladder_mid = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([49], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([88], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32),
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_right = Ladder(
            left_upper_x=jnp.array([128], jnp.int32), 
            left_upper_y=jnp.array([92], jnp.int32), 
            right_lower_x=jnp.array([144], jnp.int32), 
            right_lower_y=jnp.array([133], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([1], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_left = Ladder(
            left_upper_x=jnp.array([16], jnp.int32), 
            left_upper_y=jnp.array([92], jnp.int32), 
            right_lower_x=jnp.array([32], jnp.int32), 
            right_lower_y=jnp.array([133], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([1], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_mid, ladder_right, ladder_left], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        door_0 = Door(
            x=jnp.array([16],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DOOR_COLOR_NORMAL.value], dtype=jnp.int32)
        )
    
        door_1 = Door(
            x=jnp.array([140],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DOOR_COLOR_NORMAL.value], dtype=jnp.int32)
        )
    
    
        ROOM_0_1.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )
        
        rope_0 = Rope(
            x_pos=jnp.array([111], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([88], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32), 
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )
        
        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )
        
        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([56], jnp.int32), 
            bbox_left_upper_y=jnp.array([146], jnp.int32), 
            bbox_right_lower_x=jnp.array([112], jnp.int32), 
            bbox_right_lower_y=jnp.array([132], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([93], jnp.int32), 
            pos_y=jnp.array([119], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([93], jnp.int32), 
            initial_y_pos=jnp.array([119], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_0_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        conveyor_0 = ConveyorBelt(
            x=jnp.array([60], dtype=jnp.int32),
            y=jnp.array([89], dtype=jnp.int32),
            movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value], dtype=jnp.int32) 
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                                   field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                                   content=[conveyor_0],
                                   named_tuple_type=ConveyorBelt)
    
    
    
        ROOM_0_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.SIDEWALLS, RoomTags.LAZER_BARRIER, RoomTags.LADDERS]))
        ROOM_0_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_0.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_0_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_0_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_0_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([24],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_0_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        
        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
        
        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.LASER_BARRIER_COLOR_NO_ALPHA], dtype=jnp.int32))
        
        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)

        barrier_0 = LAZER_BARRIER(
            X=jnp.array([16], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_1 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_2 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_3 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_4 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_5 = LAZER_BARRIER(
            X=jnp.array([140], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32),
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        
        
        ROOM_0_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_0_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_0.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_0_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_0_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_0_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
        
        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        
        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_1_LAYER_1_PRIMARY], dtype=jnp.int32))
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_1_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([124], jnp.int32),
            bbox_right_lower_y=jnp.array([45], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([101], jnp.int32), 
            pos_y=jnp.array([33], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([101], jnp.int32), 
            initial_y_pos=jnp.array([33], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
        
        ROOM_0_2.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
    
        ROOM_1_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.SIDEWALLS, RoomTags.ENEMIES]))
        ROOM_1_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_1_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_1_LAYER_2_PRIMARY], dtype=jnp.int32))
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32),
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([8], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([155], jnp.int32),
            bbox_right_lower_y=jnp.array([44], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([60], jnp.int32), 
            pos_y=jnp.array([33], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([60], jnp.int32), 
            initial_y_pos=jnp.array([33], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_0.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        ROOM_1_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_1_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32),
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),  
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom, ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([92], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([92], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_1_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
        ROOM_1_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_1.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_1_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_1.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_1_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([150], jnp.int32))
        ROOM_1_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([150, 10], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.TORCH.value],dtype=jnp.int32),
            x=jnp.array([77],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([126], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([150], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        door_0 = Door(
            x=jnp.array([56],dtype=jnp.int32),
            y=jnp.array([86],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )
    
        door_1 = Door(
            x=jnp.array([100],dtype=jnp.int32),
            y=jnp.array([86],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )
    
    
        ROOM_1_2.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )
        
        rope_0 = Rope(
            x_pos=jnp.array([41], dtype=jnp.int32),
            top=jnp.array([50], dtype=jnp.int32),
            bottom=jnp.array([75], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_WHITE.value], dtype=jnp.int32), 
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )
        
        rope_1 = Rope(
            x_pos=jnp.array([125], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], dtype=jnp.int32), 
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )
        
        rope_2 = Rope(
            x_pos=jnp.array([126], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], dtype=jnp.int32), 
            is_climbable=jnp.array([0], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )
        
        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0, rope_1, rope_2], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )
        
        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([56], jnp.int32), 
            bbox_left_upper_y=jnp.array([68], jnp.int32), 
            bbox_right_lower_x=jnp.array([112], jnp.int32), 
            bbox_right_lower_y=jnp.array([82], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([93], jnp.int32), 
            pos_y=jnp.array([119], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([93], jnp.int32), 
            initial_y_pos=jnp.array([119], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_2.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        conveyor_0 = ConveyorBelt(
            x=jnp.array([60], dtype=jnp.int32),
            y=jnp.array([46], dtype=jnp.int32),
            movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                                   field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                                   content=[conveyor_0],
                                   named_tuple_type=ConveyorBelt)
        
        
        
        ROOM_1_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.LADDERS]))
        ROOM_1_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)
    
        ROOM_1_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
    
        ROOM_1_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
    
        ROOM_1_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
    
        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.SWORD.value],dtype=jnp.int32),
            x=jnp.array([12],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_3.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
    
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        
        
        ROOM_1_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.SIDEWALLS, RoomTags.LAZER_BARRIER, RoomTags.LADDERS]))
        ROOM_1_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)
    
        ROOM_1_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
    
        ROOM_1_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
    
        ROOM_1_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
    
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([129],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_4.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
    
        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
    
        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.LASER_BARRIER_COLOR_NO_ALPHA], dtype=jnp.int32))
    
        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_1_4.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)

        barrier_0 = LAZER_BARRIER(
            X=jnp.array([16], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_1 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_2 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
    
        barrier_3 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
    
        barrier_4 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
    
        barrier_5 = LAZER_BARRIER(
            X=jnp.array([140], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
    
        ROOM_1_4.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)
    
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_2_PRIMARY.value], jnp.int32),  
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        
        
        ROOM_2_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.DROPOUTFLOORS, RoomTags.SIDEWALLS]))
        ROOM_2_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_collision_level_2.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([151], jnp.int32))
        ROOM_2_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([77],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        rope_0 = Rope(
            x_pos=jnp.array([80], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32), 
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )
        
        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )
        
        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        
        dfloor_l_0 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([56], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_l_1 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([66], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_l_2 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([76], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_l_3 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([86], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_l_4 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([106], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_l_5 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([116], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_r_0 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([56], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_r_1 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([66], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_r_2 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([76], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_r_3 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([86], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_r_4 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([106], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        dfloor_r_5 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([116], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )
        
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[dfloor_l_0, dfloor_l_1, dfloor_l_2, dfloor_l_3, dfloor_l_4, dfloor_l_5, dfloor_r_0, dfloor_r_1, dfloor_r_2, dfloor_r_3, dfloor_r_4, dfloor_r_5],
                         named_tuple_type=DropoutFloor)
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([120], dtype=jnp.int32))
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([30], dtype=jnp.int32))
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
    
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
    
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_1_LAYER_3_PRIMARY], dtype=jnp.int32))
        
        
        
        ROOM_2_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.SIDEWALLS, RoomTags.ENEMIES]))
        ROOM_2_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32),
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([0], dtype=jnp.int32))
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([1], dtype=jnp.int32))
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                        field_type=NamedTupleFieldType.OTHER_ARRAY,
                        content=jnp.array([consts.DIF_1_LAYER_3_SECONDARY], dtype=jnp.int32))
        
        enemy0 = Enemy(
            bbox_left_upper_x=jnp.array([18], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([25], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([18], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([18], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        enemy1 = Enemy(
            bbox_left_upper_x=jnp.array([50], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([57], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([50], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([50], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        ROOM_2_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0, enemy1], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_2_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ITEMS, RoomTags.DROPOUTFLOORS]))
        ROOM_2_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([1], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([0], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                        field_type=NamedTupleFieldType.OTHER_ARRAY,
                        content=jnp.array([consts.DIF_1_LAYER_3_SECONDARY], dtype=jnp.int32))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
           color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)
        
        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))
        
        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))
        
        ROOM_2_2.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([17],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        
        
        ROOM_2_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_2_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([96, 7], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([96, 125], dtype=jnp.uint16))
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top, ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy0 = Enemy(
            bbox_left_upper_x=jnp.array([44], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([51], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([44], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([44], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        enemy1 = Enemy(
            bbox_left_upper_x=jnp.array([108], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([115], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([108], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([108], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        ROOM_2_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0, enemy1], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_2_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.LAZER_BARRIER]))
        ROOM_2_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_2_4.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)

        barrier_0 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_1 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_2 = LAZER_BARRIER(
            X=jnp.array([60], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_3 = LAZER_BARRIER(
            X=jnp.array([68], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_4 = LAZER_BARRIER(
            X=jnp.array([88], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_5 = LAZER_BARRIER(
            X=jnp.array([96], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_6 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_7 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        ROOM_2_4.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5, barrier_6, barrier_7], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)
        
        
        
        ROOM_2_5 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_2_5.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_5.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([96, 8], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([96, 125], dtype=jnp.uint16))
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], jnp.int32),  
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top, ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([100], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([100], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
        
        ROOM_2_5.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_2_6 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.LADDERS]))
        ROOM_2_6.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_6_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_6_collision_level_2.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([147], jnp.int32))
        ROOM_2_6.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([76],dtype=jnp.int32),
            y=jnp.array([64],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_6.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        rope_0 = Rope(
            x_pos=jnp.array([71], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([97], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32), 
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )
        
        rope_1 = Rope(
            x_pos=jnp.array([87], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([81], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32), 
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )
        
        rope_2 = Rope(
            x_pos=jnp.array([88], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([81], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], dtype=jnp.int32), 
            is_climbable=jnp.array([0], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )
        
        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0, rope_1, rope_2], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )
        
        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([123], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([147], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.DIF_1_LAYER_3_PRIMARY.value], jnp.int32),  
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
    
    
        ROOM_3_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS,RoomTags.BONUSROOM,RoomTags.SIDEWALLS]))
        ROOM_3_0.set_field(field_name=VanillaRoomFields.sprite.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "bonus_room_sprite.npy"), transpose=True),
                         requires_serialisation=False)
        ROOM_3_0.set_field(field_name=VanillaRoomFields.room_collision_map.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND,"bonus_room_collision_map.npy"),as_bool=False, transpose=True),
                         requires_serialisation=False)
        ROOM_3_0.set_field(field_name=VanillaRoomFields.height.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([148], jnp.int32))
        ROOM_3_0.set_field(field_name=VanillaRoomFields.vertical_offset.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([47], jnp.int32))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 26], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 26], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([consts.DEFAULT_BONUS_ROOM_GEM_X],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        ROOM_3_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bonus_cycle_lenght.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([620], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bouns_cycle_index.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([0], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.reset_state_on_leave.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([1], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.BONUS_ROOM_COLOR], dtype=jnp.int32))
        
        
        
        
        ROOM_3_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.DARKROOM]))
        ROOM_3_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        
        
        ROOM_3_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.DOORS, RoomTags.DARKROOM]))
        ROOM_3_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        door_0 = Door(
            x=jnp.array([16],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DOOR_COLOR_NORMAL.value], dtype=jnp.int32)
        )

        door_1 = Door(
            x=jnp.array([140],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DOOR_COLOR_NORMAL.value], dtype=jnp.int32)
        )

        ROOM_3_2.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )
        
        
        
        ROOM_3_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
           color=jnp.array([ObstacleColors.DIF_1_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )
        
        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)
        
        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))
        
        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
        
        ROOM_3_3.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))
        
        ROOM_3_3.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([32], jnp.int32), 
            bbox_left_upper_y=jnp.array([33], jnp.int32), 
            bbox_right_lower_x=jnp.array([128], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([45], jnp.int32), 
            pos_y=jnp.array([47], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([1], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([45], jnp.int32), 
            initial_y_pos=jnp.array([47], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_3_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_3_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.LADDERS, RoomTags.DARKROOM]))
        ROOM_3_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.HAMMER.value],dtype=jnp.int32),
            x=jnp.array([17],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_4.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_4_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        
        
        ROOM_3_5 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.SIDEWALLS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_5.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_5.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_5.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_5.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
           color=jnp.array([ObstacleColors.DIF_1_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_1_LAYER_4_PRIMARY], dtype=jnp.int32))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([139],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        
        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([77],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_5.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        
        
        ROOM_3_6 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_6.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_6.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_1_LAYER_4_SECONDARY], dtype=jnp.int32))
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_4_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([60], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([60], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
        
        ROOM_3_6.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_3_7 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_7.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_7.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
           color=jnp.array([ObstacleColors.DIF_1_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )
        
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)
        
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))
        
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
        
        ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))
        
        ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_1_LAYER_4_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy0 = Enemy(
            bbox_left_upper_x=jnp.array([30], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([37], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([30], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([30], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        ROOM_3_7.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        
        ROOM_3_8 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_8.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_8.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_8.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_8.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_1_LAYER_4_PRIMARY], dtype=jnp.int32))
        
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([99],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        
        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([115],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        
        item_2 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([131],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_8.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1, item_2],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        # Optinally conect a room from the last Layout to the start-room!
        if not last_room is None:
            last_room.connect_to(my_location=RoomConnectionDirections.DOWN, 
                                 other_room=ROOM_0_1, other_location=RoomConnectionDirections.DOWN)
            
            
        # connection between layer 1 rooms
        ROOM_0_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_0_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_0_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_0_2, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 1 and layer 2 rooms
        ROOM_0_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_1_1, other_location=RoomConnectionDirections.UP)
        ROOM_0_2.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_1_3, other_location=RoomConnectionDirections.UP)

        # connection between layer 2 rooms
        ROOM_1_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_2, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_4, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 2 and layer 3 rooms
        ROOM_1_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_1, other_location=RoomConnectionDirections.UP)
        ROOM_1_1.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_2, other_location=RoomConnectionDirections.UP)
        ROOM_1_2.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_3, other_location=RoomConnectionDirections.UP)
        ROOM_1_4.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_5, other_location=RoomConnectionDirections.UP)
        
        # connection between layer 3 rooms
        ROOM_2_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_4, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_4.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_5, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_5.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_6, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 3 and layer 4 rooms
        ROOM_2_3.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_4, other_location=RoomConnectionDirections.UP)
        ROOM_2_5.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_6, other_location=RoomConnectionDirections.UP)
        ROOM_2_6.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_7, other_location=RoomConnectionDirections.UP)
        
        # connection between layer 4 rooms
        ROOM_3_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_2, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_4, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_4.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_5, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_6.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_7, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_7.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_8, other_location=RoomConnectionDirections.LEFT)
        if self_loop:
            # bonus room to start room
            ROOM_3_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                              other_room=ROOM_0_1, other_location=RoomConnectionDirections.UP)
        if not self_loop:
            return LAYOUT, ROOM_3_0
        else:
            return LAYOUT
    
    
def make_difficulty_2(LAYOUT: PyramidLayout, consts, last_room: uRoom = None, self_loop: bool = True) -> PyramidLayout:
    
        ROOM_0_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
        ROOM_0_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_0_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_0_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_0.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_0_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([151], jnp.int32))
        ROOM_0_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([77, 26], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([13],dtype=jnp.int32),
            y=jnp.array([52],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ladder_mid = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([49], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([88], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_right = Ladder(
            left_upper_x=jnp.array([128], jnp.int32), 
            left_upper_y=jnp.array([92], jnp.int32), 
            right_lower_x=jnp.array([144], jnp.int32), 
            right_lower_y=jnp.array([133], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([1], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_left = Ladder(
            left_upper_x=jnp.array([16], jnp.int32), 
            left_upper_y=jnp.array([92], jnp.int32), 
            right_lower_x=jnp.array([32], jnp.int32), 
            right_lower_y=jnp.array([133], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([1], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_mid, ladder_right, ladder_left], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        door_0 = Door(
            x=jnp.array([16],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )
    
        door_1 = Door(
            x=jnp.array([140],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )
    
    
        ROOM_0_1.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )

        rope_0 = Rope(
            x_pos=jnp.array([111], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([88], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([56], jnp.int32), 
            bbox_left_upper_y=jnp.array([146], jnp.int32), 
            bbox_right_lower_x=jnp.array([112], jnp.int32), 
            bbox_right_lower_y=jnp.array([132], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([93], jnp.int32), 
            pos_y=jnp.array([119], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([93], jnp.int32), 
            initial_y_pos=jnp.array([119], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_0_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)

        conveyor_0 = ConveyorBelt(
            x=jnp.array([60], dtype=jnp.int32),
            y=jnp.array([89], dtype=jnp.int32),
            movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_2_LAYER_1_PRIMARY.value],jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                                   field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                                   content=[conveyor_0],
                                   named_tuple_type=ConveyorBelt)

        ROOM_0_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.SIDEWALLS, RoomTags.LAZER_BARRIER, RoomTags.LADDERS]))
        ROOM_0_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_0_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.SWORD.value],dtype=jnp.int32),
            x=jnp.array([24],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_0_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.LASER_BARRIER_COLOR_NO_ALPHA], dtype=jnp.int32))

        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)

        barrier_0 = LAZER_BARRIER(
            X=jnp.array([16], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_1 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_2 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_3 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_4 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_5 = LAZER_BARRIER(
            X=jnp.array([140], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_SECONDARY.value],jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_PRIMARY.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)    

        ROOM_0_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ITEMS]))
        ROOM_0_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_1.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_0_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_0_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_0_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_2_LAYER_1_PRIMARY], dtype=jnp.int32))

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_SECONDARY.value],jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_1_PRIMARY.value],jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([131],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_0_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
        ROOM_1_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.SIDEWALLS, RoomTags.ITEMS]))
        ROOM_1_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_1_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_1_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_1_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_2_LAYER_2_PRIMARY], dtype=jnp.int32))

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([15],dtype=jnp.int32),
            y=jnp.array([5],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([31],dtype=jnp.int32),
            y=jnp.array([5],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )  

    
     
        ROOM_1_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_1_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom, ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([0], jnp.int32), 
            bbox_left_upper_y=jnp.array([0], jnp.int32), 
            bbox_right_lower_x=jnp.array([0], jnp.int32), 
            bbox_right_lower_y=jnp.array([0], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([34], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([34], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )

        ROOM_1_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
    
        ROOM_1_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
        ROOM_1_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_1_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_1_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_1.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_1_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([150], jnp.int32))
        ROOM_1_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.TORCH.value],dtype=jnp.int32),
            x=jnp.array([77],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([126], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([150], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        door_0 = Door(
            x=jnp.array([56],dtype=jnp.int32),
            y=jnp.array([86],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )
    
        door_1 = Door(
            x=jnp.array([100],dtype=jnp.int32),
            y=jnp.array([86],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )
    
    
        ROOM_1_2.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )

        rope_0 = Rope(
            x_pos=jnp.array([41], dtype=jnp.int32),
            top=jnp.array([50], dtype=jnp.int32),
            bottom=jnp.array([75], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_WHITE.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        rope_1 = Rope(
            x_pos=jnp.array([125], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
                bottom=jnp.array([100], dtype=jnp.int32), 
                color_index=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )

        rope_2 = Rope(
            x_pos=jnp.array([126], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], dtype=jnp.int32), 
            is_climbable=jnp.array([0], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0, rope_1, rope_2], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([56], jnp.int32), 
            bbox_left_upper_y=jnp.array([68], jnp.int32), 
            bbox_right_lower_x=jnp.array([112], jnp.int32), 
            bbox_right_lower_y=jnp.array([82], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([93], jnp.int32), 
            pos_y=jnp.array([119], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([93], jnp.int32), 
            initial_y_pos=jnp.array([119], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_2.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)

        conveyor_0 = ConveyorBelt(
            x=jnp.array([60], dtype=jnp.int32),
            y=jnp.array([46], dtype=jnp.int32),
            movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                                   field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                                   content=[conveyor_0],
                                   named_tuple_type=ConveyorBelt)
    
    
    
        
        ROOM_1_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.ENEMIES, RoomTags.LADDERS]))
        ROOM_1_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_1_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_1_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_1_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_SECONDARY.value], jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([124], jnp.int32),
            bbox_right_lower_y=jnp.array([45], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([8], jnp.int32), 
            pos_y=jnp.array([33], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([8], jnp.int32), 
            initial_y_pos=jnp.array([33], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([1], jnp.int32)
        )

        ROOM_1_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
          
        ROOM_1_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.SIDEWALLS, RoomTags.LAZER_BARRIER, RoomTags.LADDERS]))
        ROOM_1_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_2.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_1_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_1_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_1_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([129],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        ROOM_1_4.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.LASER_BARRIER_COLOR_NO_ALPHA], dtype=jnp.int32))

        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_1_4.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)
        barrier_0 = LAZER_BARRIER(
            X=jnp.array([16], jnp.int32), 
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_1 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_2 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_3 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_4 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_5 = LAZER_BARRIER(
            X=jnp.array([140], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        ROOM_1_4.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_2_PRIMARY.value], jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
    
        ROOM_2_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.DROPOUTFLOORS, RoomTags.SIDEWALLS]))
        ROOM_2_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_level_2_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_collision_level_2.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([151], jnp.int32))
        ROOM_2_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([60],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([92],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        rope_0 = Rope(
            x_pos=jnp.array([80], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        dfloor_l_0 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([56], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_1 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([66], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_2 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([76], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_3 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([86], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_4 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([106], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_5 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([116], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_0 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([56], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_1 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([66], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_2 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([76], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_3 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([86], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_4 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([106], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_5 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([116], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[dfloor_l_0, dfloor_l_1, dfloor_l_2, dfloor_l_3, dfloor_l_4, dfloor_l_5, dfloor_r_0, dfloor_r_1, dfloor_r_2, dfloor_r_3, dfloor_r_4, dfloor_r_5],
                         named_tuple_type=DropoutFloor)
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([120], dtype=jnp.int32))
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([30], dtype=jnp.int32))
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
    
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
    
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_2_LAYER_3_PRIMARY], dtype=jnp.int32))
    
     
        ROOM_2_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.SIDEWALLS, RoomTags.ENEMIES]))
        ROOM_2_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([0], dtype=jnp.int32))
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([1], dtype=jnp.int32))
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                        field_type=NamedTupleFieldType.OTHER_ARRAY,
                        content=jnp.array([consts.DIF_2_LAYER_3_SECONDARY], dtype=jnp.int32))

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([60], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([60], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )

        ROOM_2_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
    
        ROOM_2_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ITEMS, RoomTags.DROPOUTFLOORS]))
        ROOM_2_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([1], dtype=jnp.int32))
        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([0], dtype=jnp.int32))
        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                        field_type=NamedTupleFieldType.OTHER_ARRAY,
                        content=jnp.array([consts.DIF_2_LAYER_3_SECONDARY], dtype=jnp.int32))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
        ROOM_2_2.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([15],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
    
        ROOM_2_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ITEMS]))
        ROOM_2_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top, ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.HAMMER.value],dtype=jnp.int32),
            x=jnp.array([15],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_3.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
    
        ROOM_2_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.LAZER_BARRIER]))
        ROOM_2_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_2_4.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)
        barrier_0 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_1 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_2 = LAZER_BARRIER(
            X=jnp.array([60], jnp.int32), 
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_3 = LAZER_BARRIER(
            X=jnp.array([68], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_4 = LAZER_BARRIER(
            X=jnp.array([88], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        barrier_5 = LAZER_BARRIER(
            X=jnp.array([96], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_6 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_7 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([8], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        ROOM_2_4.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5, barrier_6, barrier_7], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)
    
    
    
        ROOM_2_5 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_2_5.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_5.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top, ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy0 = Enemy(
            bbox_left_upper_x=jnp.array([44], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([51], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([44], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([44], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )

        enemy1 = Enemy(
            bbox_left_upper_x=jnp.array([108], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([115], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([108], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([108], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )

        ROOM_2_5.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0, enemy1], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
    
        ROOM_2_6 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.LADDERS]))
        ROOM_2_6.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_6_level_2_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_6_collision_level_2.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([147], jnp.int32))
        ROOM_2_6.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([76],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        ROOM_2_6.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        rope_0 = Rope(
            x_pos=jnp.array([71], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([97], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        rope_1 = Rope(
            x_pos=jnp.array([87], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([81], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        rope_2 = Rope(
            x_pos=jnp.array([88], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([81], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([0], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )

        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0, rope_1, rope_2], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([123], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([147], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
    
        ROOM_3_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS,RoomTags.BONUSROOM,RoomTags.SIDEWALLS]))
        ROOM_3_0.set_field(field_name=VanillaRoomFields.sprite.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "bonus_room_sprite.npy"), transpose=True),
                         requires_serialisation=False)
        ROOM_3_0.set_field(field_name=VanillaRoomFields.room_collision_map.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND,"bonus_room_collision_map.npy"),as_bool=False, transpose=True),
                         requires_serialisation=False)
        ROOM_3_0.set_field(field_name=VanillaRoomFields.height.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([148], jnp.int32))
        ROOM_3_0.set_field(field_name=VanillaRoomFields.vertical_offset.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([47], jnp.int32))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 26], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 26], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([consts.DEFAULT_BONUS_ROOM_GEM_X],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        ROOM_3_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bonus_cycle_lenght.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([620], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bouns_cycle_index.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([0], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.reset_state_on_leave.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([1], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.BONUS_ROOM_COLOR], dtype=jnp.int32))
    
    
    
    
        ROOM_3_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.DARKROOM]))
        ROOM_3_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
    
        ROOM_3_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.DOORS, RoomTags.DARKROOM]))
        ROOM_3_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        door_0 = Door(
            x=jnp.array([16],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )

        door_1 = Door(
            x=jnp.array([140],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )

        ROOM_3_2.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )       
    
    
        ROOM_3_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)

        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))

        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))

        ROOM_3_3.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))

        ROOM_3_3.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([32], jnp.int32), 
            bbox_left_upper_y=jnp.array([33], jnp.int32), 
            bbox_right_lower_x=jnp.array([128], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([45], jnp.int32), 
            pos_y=jnp.array([47], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([1], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([1], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([45], jnp.int32), 
            initial_y_pos=jnp.array([47], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_3_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
    
        ROOM_3_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.ENEMIES, RoomTags.LADDERS, RoomTags.DARKROOM]))
        ROOM_3_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_4_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32),
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([8], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([155], jnp.int32),
            bbox_right_lower_y=jnp.array([44], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([60], jnp.int32), 
            pos_y=jnp.array([33], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([60], jnp.int32), 
            initial_y_pos=jnp.array([33], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )

        ROOM_3_4.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
    
        ROOM_3_5 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.SIDEWALLS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_5.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_5.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_5.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_5.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_2_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)

        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_2_LAYER_4_PRIMARY], dtype=jnp.int32))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([60],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([76],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        item_2 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([92],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_5.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1, item_2],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
    
        ROOM_3_6 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_6.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_6.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_2_LAYER_4_SECONDARY], dtype=jnp.int32))
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_4_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([15],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_6.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
    
    
        
        ROOM_3_7 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_7.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_7.set_field(field_name=VanillaRoomFields.vertical_offset.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
           color=jnp.array([ObstacleColors.DIF_2_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_4_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
        enemy0 = Enemy(  
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([22], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([60], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
    
        ROOM_3_8 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_8.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_8.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_8.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_8.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_2_LAYER_4_PRIMARY], dtype=jnp.int32))
    
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([115],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
    
        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([131],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_8.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        
    
        if not last_room is None:
            last_room.connect_to(my_location=RoomConnectionDirections.DOWN, 
                                 other_room=ROOM_0_1, other_location=RoomConnectionDirections.DOWN)

    
        # connection between layer 1 rooms
        ROOM_0_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_0_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_0_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_0_2, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 1 and layer 2 rooms
        ROOM_0_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_1_1, other_location=RoomConnectionDirections.UP)
        ROOM_0_2.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_1_3, other_location=RoomConnectionDirections.UP)

        # connection between layer 2 rooms
        ROOM_1_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_2, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_4, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 2 and layer 3 rooms
        ROOM_1_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_1, other_location=RoomConnectionDirections.UP)
        ROOM_1_1.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_2, other_location=RoomConnectionDirections.UP)
        ROOM_1_2.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_3, other_location=RoomConnectionDirections.UP)
        ROOM_1_4.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_5, other_location=RoomConnectionDirections.UP)
        
        # connection between layer 3 rooms
        ROOM_2_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_4, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_4.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_5, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_5.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_6, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 3 and layer 4 rooms
        ROOM_2_3.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_4, other_location=RoomConnectionDirections.UP)
        ROOM_2_5.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_6, other_location=RoomConnectionDirections.UP)
        ROOM_2_6.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_7, other_location=RoomConnectionDirections.UP)
        
        # connection between layer 4 rooms
        ROOM_3_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_2, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_4, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_4.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_5, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_6.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_7, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_7.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_8, other_location=RoomConnectionDirections.LEFT)

        if self_loop:
            # bonus room to start room
            ROOM_3_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                              other_room=ROOM_0_1, other_location=RoomConnectionDirections.UP)

        if not self_loop:
            return LAYOUT, ROOM_3_0
        else:
            return LAYOUT
    
    
    
def make_difficulty_3(LAYOUT: PyramidLayout, consts, last_room: uRoom = None, self_loop: bool = True) -> PyramidLayout:
    
    
        ROOM_0_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
        ROOM_0_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_0_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_0_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_0.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_0_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([151], jnp.int32))
        ROOM_0_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 26], dtype=jnp.uint16))

        ROOM_0_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([77, 26], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([13],dtype=jnp.int32),
            y=jnp.array([52],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ladder_mid = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([49], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([88], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_right = Ladder(
            left_upper_x=jnp.array([128], jnp.int32), 
            left_upper_y=jnp.array([92], jnp.int32), 
            right_lower_x=jnp.array([144], jnp.int32), 
            right_lower_y=jnp.array([133], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([1], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_left = Ladder(
            left_upper_x=jnp.array([16], jnp.int32), 
            left_upper_y=jnp.array([92], jnp.int32), 
            right_lower_x=jnp.array([32], jnp.int32), 
            right_lower_y=jnp.array([133], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_PRIMARY.value],jnp.int32),
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([1], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_mid, ladder_right, ladder_left], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        door_0 = Door(
            x=jnp.array([16],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )
    
        door_1 = Door(
            x=jnp.array([140],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )
    
    
        ROOM_0_1.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )

        rope_0 = Rope(
            x_pos=jnp.array([111], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([88], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_0_1.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([56], jnp.int32), 
            bbox_left_upper_y=jnp.array([146], jnp.int32), 
            bbox_right_lower_x=jnp.array([112], jnp.int32), 
            bbox_right_lower_y=jnp.array([132], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([93], jnp.int32), 
            pos_y=jnp.array([119], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([93], jnp.int32), 
            initial_y_pos=jnp.array([119], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_0_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)

        conveyor_0 = ConveyorBelt(
            x=jnp.array([60], dtype=jnp.int32),
            y=jnp.array([89], dtype=jnp.int32),
            movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_3_LAYER_1_PRIMARY.value],jnp.int32)
        )

        ROOM_0_1.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                                   field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                                   content=[conveyor_0],
                                   named_tuple_type=ConveyorBelt)
        
        
        ROOM_0_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LAZER_BARRIER, RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_0_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_0_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_0_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_0_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.LASER_BARRIER_COLOR_NO_ALPHA], dtype=jnp.int32))

        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)

        barrier_0 = LAZER_BARRIER(
            X=jnp.array([16], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_1 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_2 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_3 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_4 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_5 = LAZER_BARRIER(
            X=jnp.array([140], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        ROOM_0_0.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_SECONDARY.value],jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_PRIMARY.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_0_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)    
        
        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([154], jnp.int32),
            bbox_right_lower_y=jnp.array([45], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([66], jnp.int32), 
            pos_y=jnp.array([9], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([66], jnp.int32), 
            initial_y_pos=jnp.array([9], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([1], jnp.int32)
        )

        ROOM_0_0.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
 
        ROOM_0_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_0_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_0_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)

        ROOM_0_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)

        ROOM_0_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))

        ROOM_0_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))

        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_0_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_3_LAYER_1_PRIMARY], dtype=jnp.int32))

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_SECONDARY.value],jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_1_PRIMARY.value],jnp.int32), 
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_0_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        enemy0 = Enemy(
            bbox_left_upper_x=jnp.array([18], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([25], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([18], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([20], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        enemy1 = Enemy(
            bbox_left_upper_x=jnp.array([50], jnp.int32), 
            bbox_left_upper_y=jnp.array([34], jnp.int32), 
            bbox_right_lower_x=jnp.array([57], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([50], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([52], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        ROOM_0_2.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0, enemy1], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        ROOM_1_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ITEMS, RoomTags.SIDEWALLS]))
        ROOM_1_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_1_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_1_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_1_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_1_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_0.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([43],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        
        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([107],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([1], dtype=jnp.int32))
        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([0], dtype=jnp.int32))
        ROOM_1_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=jnp.array([consts.DIF_3_LAYER_2_PRIMARY], dtype=jnp.int32))

        
        ROOM_1_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_1_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_1_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom, ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy_0 = Enemy(
            bbox_left_upper_x=jnp.array([0], jnp.int32), 
            bbox_left_upper_y=jnp.array([0], jnp.int32), 
            bbox_right_lower_x=jnp.array([0], jnp.int32), 
            bbox_right_lower_y=jnp.array([0], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([18], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([18], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )
        
        enemy_1 = Enemy(
            bbox_left_upper_x=jnp.array([0], jnp.int32), 
            bbox_left_upper_y=jnp.array([0], jnp.int32), 
            bbox_right_lower_x=jnp.array([0], jnp.int32), 
            bbox_right_lower_y=jnp.array([0], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([50], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([50], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )

        ROOM_1_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy_0, enemy_1], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        ROOM_1_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.DOORS, RoomTags.LADDERS, RoomTags.ROPES, RoomTags.ENEMIES, RoomTags.CONVEYORBELTS]))
        ROOM_1_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_level_1_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_1_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "mid_room_collision_level_1.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_1_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([150], jnp.int32))
        ROOM_1_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.TORCH.value],dtype=jnp.int32),
            x=jnp.array([77],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([126], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([150], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        door_0 = Door(
            x=jnp.array([56],dtype=jnp.int32),
            y=jnp.array([86],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )
    
        door_1 = Door(
            x=jnp.array([100],dtype=jnp.int32),
            y=jnp.array([86],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )
    
    
        ROOM_1_2.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )

        rope_0 = Rope(
            x_pos=jnp.array([41], dtype=jnp.int32),
            top=jnp.array([50], dtype=jnp.int32),
            bottom=jnp.array([75], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_WHITE.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        rope_1 = Rope(
            x_pos=jnp.array([125], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
                bottom=jnp.array([100], dtype=jnp.int32), 
                color_index=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )

        rope_2 = Rope(
            x_pos=jnp.array([126], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], dtype=jnp.int32), 
            is_climbable=jnp.array([0], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0, rope_1, rope_2], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_1_2.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([56], jnp.int32), 
            bbox_left_upper_y=jnp.array([68], jnp.int32), 
            bbox_right_lower_x=jnp.array([112], jnp.int32), 
            bbox_right_lower_y=jnp.array([82], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([93], jnp.int32), 
            pos_y=jnp.array([119], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([93], jnp.int32), 
            initial_y_pos=jnp.array([119], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_2.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)

        conveyor_0 = ConveyorBelt(
            x=jnp.array([60], dtype=jnp.int32),
            y=jnp.array([46], dtype=jnp.int32),
            movement_dir=jnp.array([MovementDirection.LEFT.value], dtype=jnp.int32),
            color=jnp.array([ObstacleColors.DIF_3_LAYER_2_PRIMARY.value], dtype=jnp.int32)
        )

        ROOM_1_2.set_field(field_name=RoomTagsNames.CONVEYORBELTS.value.conveyor_belts.value,
                                   field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                                   content=[conveyor_0],
                                   named_tuple_type=ConveyorBelt)
        
        
        ROOM_1_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES]))
        ROOM_1_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_1_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_1_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_1_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_2_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_1_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy_0 = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([119], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([119], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)  
        )

        ROOM_1_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy_0], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        
        
        ROOM_1_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ITEMS, RoomTags.SIDEWALLS]))
        ROOM_1_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_1_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_1_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_1_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_1_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_2_LAYER_3_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
    
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_1_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([129],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_1_4.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([0], dtype=jnp.int32))
        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([1], dtype=jnp.int32))
        ROOM_1_4.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=jnp.array([consts.LASER_BARRIER_COLOR_NO_ALPHA], dtype=jnp.int32))

        
        ROOM_2_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.DROPOUTFLOORS, RoomTags.SIDEWALLS]))
        ROOM_2_0.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_level_2_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_0.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_0_collision_level_2.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_0.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([151], jnp.int32))
        ROOM_2_0.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([19],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        rope_0 = Rope(
            x_pos=jnp.array([80], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([100], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_2_0.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        dfloor_l_0 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([56], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_1 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([66], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_2 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([76], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_3 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([86], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_4 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([106], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_l_5 = DropoutFloor(
             x=jnp.array([4], dtype=jnp.int32),
             y=jnp.array([116], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_0 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([56], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_1 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([66], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_2 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([76], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_3 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([86], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_4 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([106], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        dfloor_r_5 = DropoutFloor(
             x=jnp.array([144], dtype=jnp.int32),
             y=jnp.array([116], dtype=jnp.int32),
             sprite_height_amount=jnp.array([1], dtype=jnp.int32),
             sprite_width_amount=jnp.array([1], dtype=jnp.int32),
             sprite_index=jnp.array([Dropout_Floor_Sprites.LADDER_FLOOR.value], dtype=jnp.int32),
             collision_padding_top=jnp.array([0], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
             )

        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[dfloor_l_0, dfloor_l_1, dfloor_l_2, dfloor_l_3, dfloor_l_4, dfloor_l_5, dfloor_r_0, dfloor_r_1, dfloor_r_2, dfloor_r_3, dfloor_r_4, dfloor_r_5],
                         named_tuple_type=DropoutFloor)
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([120], dtype=jnp.int32))
        ROOM_2_0.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([30], dtype=jnp.int32))
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
    
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))
    
        ROOM_2_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_3_LAYER_3_PRIMARY], dtype=jnp.int32))
    
        
        ROOM_2_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.SIDEWALLS]))
        ROOM_2_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        
        ROOM_2_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        
        ROOM_2_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        
        ROOM_2_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        
        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )
        
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        
        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)

        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        ROOM_2_1.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        
        enemy_0 = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([44], jnp.int32), 
            pos_y=jnp.array([36], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([44], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)  
        )

        ROOM_2_1.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy_0], requires_serialisation=True, 
                         named_tuple_type=Enemy)
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([0], dtype=jnp.int32))
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                     field_type=NamedTupleFieldType.INTEGER_SCALAR,
                     content=jnp.array([1], dtype=jnp.int32))
        ROOM_2_1.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                     field_type=NamedTupleFieldType.OTHER_ARRAY,
                     content=jnp.array([consts.DIF_3_LAYER_3_SECONDARY], dtype=jnp.int32))
        
        
        ROOM_2_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ITEMS, RoomTags.DROPOUTFLOORS]))
        ROOM_2_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([1], dtype=jnp.int32))
        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([0], dtype=jnp.int32))
        ROOM_2_2.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                        field_type=NamedTupleFieldType.OTHER_ARRAY,
                        content=jnp.array([consts.DIF_3_LAYER_3_SECONDARY], dtype=jnp.int32))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
        ROOM_2_2.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))

        ROOM_2_2.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.SWORD.value],dtype=jnp.int32),
            x=jnp.array([15],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_2_2.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_2.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        
        ROOM_2_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.ENEMIES, RoomTags.LADDERS]))
        ROOM_2_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32),
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value],jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top, ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_3.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([8], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([155], jnp.int32),
            bbox_right_lower_y=jnp.array([44], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([134], jnp.int32), 
            pos_y=jnp.array([7], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([134], jnp.int32), 
            initial_y_pos=jnp.array([7], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )

        ROOM_2_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
        
        ROOM_2_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.LAZER_BARRIER]))
        ROOM_2_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))

        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))

        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))

        ROOM_2_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        
        global_barrier_info = GlobalLazerBarrierInfo(
             cycle_length=jnp.array([128], jnp.int32), 
             cycle_active_frames=jnp.array([92], jnp.int32), 
             cycle_offset=jnp.array([0], jnp.int32), 
             cycle_index=jnp.array([0], jnp.int32)
        )
        ROOM_2_4.set_field(RoomTagsNames.LAZER_BARRIER.value.global_barrier_info.value, 
                          field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                          content=[global_barrier_info], 
                          requires_serialisation=True, 
                          named_tuple_type=GlobalLazerBarrierInfo)

        barrier_0 = LAZER_BARRIER(
            X=jnp.array([36], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_1 = LAZER_BARRIER(
            X=jnp.array([44], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_2 = LAZER_BARRIER(
            X=jnp.array([60], jnp.int32), 
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_3 = LAZER_BARRIER(
            X=jnp.array([68], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_4 = LAZER_BARRIER(
            X=jnp.array([88], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        barrier_5 = LAZER_BARRIER(
            X=jnp.array([96], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_6 = LAZER_BARRIER(
            X=jnp.array([112], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )
        
        barrier_7 = LAZER_BARRIER(
            X=jnp.array([120], jnp.int32),
            upper_point=jnp.array([7], jnp.int32),
            lower_point=jnp.array([46], jnp.int32)
        )

        ROOM_2_4.set_field(RoomTagsNames.LAZER_BARRIER.value.barriers.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[barrier_0, barrier_1, barrier_2, barrier_3, barrier_4, barrier_5, barrier_6, barrier_7], 
                         requires_serialisation=False, 
                         named_tuple_type=LAZER_BARRIER)
        
        ROOM_2_5 = LAYOUT.create_new_room(tags=tuple([RoomTags.ENEMIES, RoomTags.LADDERS]))
        ROOM_2_5.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_2.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_2_5.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_5.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32),
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([48], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([149], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value],jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top, ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_5.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([8], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([155], jnp.int32),
            bbox_right_lower_y=jnp.array([44], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([118], jnp.int32), 
            pos_y=jnp.array([7], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([118], jnp.int32), 
            initial_y_pos=jnp.array([7], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([2], jnp.int32)
        )

        ROOM_2_5.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
        ROOM_2_6 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS, RoomTags.ROPES, RoomTags.LADDERS]))
        ROOM_2_6.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_6_level_2_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "room_6_collision_level_2.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([147], jnp.int32))
        ROOM_2_6.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_2_6.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.HAMMER.value],dtype=jnp.int32),
            x=jnp.array([76],dtype=jnp.int32),
            y=jnp.array([64],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        ROOM_2_6.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )

        rope_0 = Rope(
            x_pos=jnp.array([71], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([97], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        rope_1 = Rope(
            x_pos=jnp.array([87], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([81], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([1], dtype=jnp.int32), 
            accessible_from_top=jnp.array([1], dtype=jnp.int32)
        )

        rope_2 = Rope(
            x_pos=jnp.array([88], dtype=jnp.int32),
            top=jnp.array([49], dtype=jnp.int32),
            bottom=jnp.array([81], dtype=jnp.int32), 
            color_index=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], dtype=jnp.int32),
            is_climbable=jnp.array([0], dtype=jnp.int32), 
            accessible_from_top=jnp.array([0], dtype=jnp.int32)
        )

        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.ropes.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[rope_0, rope_1, rope_2], 
                         requires_serialisation=True, 
                         named_tuple_type=Rope
                         )

        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.rope_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )
        ROOM_2_6.set_field(field_name=RoomTagsNames.ROPES.value.last_hanged_on_rope.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], jnp.int32), 
                         requires_serialisation=True
                         )

        ladder_to_bottom = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([123], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([147], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([1], jnp.int32),
            rope_seeking_at_top=jnp.array([0], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.DIF_3_LAYER_3_PRIMARY.value], jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_bottom], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_2_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        
        ROOM_3_0 = LAYOUT.create_new_room(tags=tuple([RoomTags.ITEMS,RoomTags.BONUSROOM,RoomTags.SIDEWALLS]))
        ROOM_3_0.set_field(field_name=VanillaRoomFields.sprite.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "bonus_room_sprite.npy"), transpose=True),
                         requires_serialisation=False)
        ROOM_3_0.set_field(field_name=VanillaRoomFields.room_collision_map.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND,"bonus_room_collision_map.npy"),as_bool=False, transpose=True),
                         requires_serialisation=False)
        ROOM_3_0.set_field(field_name=VanillaRoomFields.height.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([148], jnp.int32))
        ROOM_3_0.set_field(field_name=VanillaRoomFields.vertical_offset.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([47], jnp.int32))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 26], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 26], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_0.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([consts.DEFAULT_BONUS_ROOM_GEM_X],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        ROOM_3_0.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bonus_cycle_lenght.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([620], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.bouns_cycle_index.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([0], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.BONUSROOM.value.reset_state_on_leave.value,
                                   field_type=NamedTupleFieldType.INTEGER_SCALAR,
                                   content=jnp.array([1], jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))
        ROOM_3_0.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.BONUS_ROOM_COLOR], dtype=jnp.int32))
    
        
        
        ROOM_3_1 = LAYOUT.create_new_room(tags=tuple([RoomTags.DARKROOM]))
        ROOM_3_1.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_1.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_1.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_1.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_1.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
    
        ROOM_3_2 = LAYOUT.create_new_room(tags=tuple([RoomTags.DOORS, RoomTags.DARKROOM]))
        ROOM_3_2.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_2.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_2.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_2.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_2.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        door_0 = Door(
            x=jnp.array([16],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )

        door_1 = Door(
            x=jnp.array([140],dtype=jnp.int32),
            y=jnp.array([8],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32),
            color=jnp.array([ObstacleColors.ROPE_COLOR_NORMAL.value], dtype=jnp.int32)
        )

        ROOM_3_2.set_field(field_name=RoomTagsNames.DOORS.value.doors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[door_0, door_1],
                        named_tuple_type=Door,
                        requires_serialisation=True
                        )       
    
    
        ROOM_3_3 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_3.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_3.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_3.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_3.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_3.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)

        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))

        ROOM_3_3.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))

        ROOM_3_3.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))

        ROOM_3_3.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([32], jnp.int32), 
            bbox_left_upper_y=jnp.array([33], jnp.int32), 
            bbox_right_lower_x=jnp.array([128], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.ROLL_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([45], jnp.int32), 
            pos_y=jnp.array([47], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([1], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([1], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([45], jnp.int32), 
            initial_y_pos=jnp.array([47], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_3_3.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
        ROOM_3_4 = LAYOUT.create_new_room(tags=tuple([RoomTags.ENEMIES, RoomTags.LADDERS, RoomTags.DARKROOM]))
        ROOM_3_4.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_4.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_4.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_4_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32),
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_4.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([4], jnp.int32), 
            bbox_left_upper_y=jnp.array([36], jnp.int32), 
            bbox_right_lower_x=jnp.array([156], jnp.int32), 
            bbox_right_lower_y=jnp.array([47], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SPIDER.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([22], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.LEFT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([60], jnp.int32), 
            initial_y_pos=jnp.array([36], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )

        ROOM_3_4.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
        ROOM_3_5 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.SIDEWALLS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_5.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_5.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_5.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_5.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_5.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))

        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
             color=jnp.array([ObstacleColors.DIF_3_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )

        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)

        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
        
        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_5.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_3_LAYER_4_PRIMARY], dtype=jnp.int32))

        item_0 = Item(
            sprite=jnp.array([Item_Sprites.KEY.value],dtype=jnp.int32),
            x=jnp.array([137],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_5.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        
        
        ROOM_3_6 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_6.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_6.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_6.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_6.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_3_LAYER_4_SECONDARY], dtype=jnp.int32))
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_4_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_6.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)

        enemy = Enemy(
            bbox_left_upper_x=jnp.array([0], jnp.int32), 
            bbox_left_upper_y=jnp.array([0], jnp.int32), 
            bbox_right_lower_x=jnp.array([0], jnp.int32), 
            bbox_right_lower_y=jnp.array([0], jnp.int32), 
            enemy_type=jnp.array([EnemyType.SNAKE.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([100], jnp.int32), 
            pos_y=jnp.array([38], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([100], jnp.int32), 
            initial_y_pos=jnp.array([38], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)   
        )

        ROOM_3_6.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
        
        ROOM_3_7 = LAYOUT.create_new_room(tags=tuple([RoomTags.PIT, RoomTags.DROPOUTFLOORS, RoomTags.LADDERS, RoomTags.ENEMIES, RoomTags.DARKROOM]))
        ROOM_3_7.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "pitroom_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_7.set_field(field_name=VanillaRoomFields.vertical_offset.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_7.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        dfloor_0 = DropoutFloor(
           x=jnp.array([36], dtype=jnp.int32),
           y=jnp.array([48], dtype=jnp.int32),
           sprite_height_amount=jnp.array([1], dtype=jnp.int32),
           sprite_width_amount=jnp.array([11], dtype=jnp.int32),
           sprite_index=jnp.array([Dropout_Floor_Sprites.PIT_FLOOR.value], dtype=jnp.int32),
           collision_padding_top=jnp.array([1], dtype=jnp.int32),
           color=jnp.array([ObstacleColors.DIF_3_LAYER_4_PRIMARY.value], dtype=jnp.int32)
           )
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.dropout_floors.value,
                        field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                        content=[dfloor_0],
                        named_tuple_type=DropoutFloor)
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.on_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([92], dtype=jnp.int32))
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.DROPOUTFLOORS.value.off_time_dropoutfloor.value,
                        field_type=NamedTupleFieldType.INTEGER_SCALAR,
                        content=jnp.array([36], dtype=jnp.int32))
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.starting_pos_y.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([76], dtype=jnp.int32))
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.PIT.value.pit_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.SARLACC_PIT_COLOR], dtype=jnp.int32))
    
        ladder_to_top = Ladder(
            left_upper_x=jnp.array([72], jnp.int32), 
            left_upper_y=jnp.array([6], jnp.int32), 
            right_lower_x=jnp.array([88], jnp.int32), 
            right_lower_y=jnp.array([44], jnp.int32), 
            has_background=jnp.array([1], jnp.int32), 
            rope_seeking_at_bottom=jnp.array([0], jnp.int32),
            rope_seeking_at_top=jnp.array([1], jnp.int32), 
            foreground_color=jnp.array([ObstacleColors.DIF_3_LAYER_4_SECONDARY.value], jnp.int32), 
            background_color=jnp.array([ObstacleColors.BLACK.value],jnp.int32),
            transparent_background=jnp.array([0], jnp.int32), 
            transparent_foreground=jnp.array([0], jnp.int32)
        )

        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[ladder_to_top], 
                         requires_serialisation=False, 
                         named_tuple_type=Ladder)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_index.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([-1], dtype=jnp.int32), 
                         requires_serialisation=True)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_tops.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladder_bottoms.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
        ROOM_3_7.set_field(field_name=RoomTagsNames.LADDERS.value.ladders_sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([0], dtype=jnp.int32), 
                         requires_serialisation=False)
    
        enemy0 = Enemy(  
            bbox_left_upper_x=jnp.array([8], jnp.int32),
            bbox_left_upper_y=jnp.array([7], jnp.int32), 
            bbox_right_lower_x=jnp.array([155], jnp.int32),
            bbox_right_lower_y=jnp.array([44], jnp.int32),
            enemy_type=jnp.array([EnemyType.BOUNCE_SKULL.value], jnp.int32), 
            alive=jnp.array([1], jnp.int32), 
            pos_x=jnp.array([38], jnp.int32), 
            pos_y=jnp.array([7], jnp.int32), 
            horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            last_movement=jnp.array([0], jnp.int32), 
            sprite_index=jnp.array([0], jnp.int32), 
            render_in_reverse=jnp.array([0], jnp.int32), 
            initial_horizontal_direction=jnp.array([Horizontal_Direction.RIGHT.value], jnp.int32), 
            initial_render_in_reverse=jnp.array([0], jnp.int32), 
            optional_movement_counter=jnp.array([0], jnp.int32), 
            initial_x_pos=jnp.array([38], jnp.int32), 
            initial_y_pos=jnp.array([7], jnp.int32),
            last_animation=jnp.array([0], jnp.int32), 
            optional_utility_field=jnp.array([0], jnp.int32)
        )
    
        ROOM_3_7.set_field(field_name=RoomTagsNames.ENEMIES.value.enemies.value, 
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK, 
                         content=[enemy0], requires_serialisation=True, 
                         named_tuple_type=Enemy)
    
        
        
        
        ROOM_3_8 = LAYOUT.create_new_room(tags=tuple([RoomTags.SIDEWALLS, RoomTags.ITEMS, RoomTags.DARKROOM]))
        ROOM_3_8.set_field(field_name=VanillaRoomFields.sprite.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=loadFrameAddAlpha(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_sprite_level_3_diff_3.npy"), transpose=True), 
                         requires_serialisation=False)
        ROOM_3_8.set_field(field_name=VanillaRoomFields.room_collision_map.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=load_collision_map(fileName=os.path.join(SPRITE_PATH_BACKGROUND, "base_collision_map.npy"), as_bool=False, transpose=True), 
                         requires_serialisation=False)
        ROOM_3_8.set_field(field_name=VanillaRoomFields.height.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([149], jnp.int32))
        ROOM_3_8.set_field(field_name=VanillaRoomFields.vertical_offset.value, 
                         field_type=NamedTupleFieldType.INTEGER_SCALAR, 
                         content=jnp.array([47], jnp.int32))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.left_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([4, 27], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.right_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([148, 27], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.top_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 4], dtype=jnp.uint16))
        ROOM_3_8.set_field(field_name=ConstantShapeRoomFields.bottom_start_position.value, 
                         field_type=NamedTupleFieldType.OTHER_ARRAY, 
                         content=jnp.array([76, 125], dtype=jnp.uint16))
    
        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_left.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([0], dtype=jnp.int32))

        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.is_right.value,
                         field_type=NamedTupleFieldType.INTEGER_SCALAR,
                         content=jnp.array([1], dtype=jnp.int32))

        ROOM_3_8.set_field(field_name=RoomTagsNames.SIDEWALLS.value.side_wall_color.value,
                         field_type=NamedTupleFieldType.OTHER_ARRAY,
                         content=jnp.array([consts.DIF_3_LAYER_4_PRIMARY], dtype=jnp.int32))
    
        item_0 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([99],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
    
        item_1 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([115],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )
        
        item_2 = Item(
            sprite=jnp.array([Item_Sprites.GEM.value],dtype=jnp.int32),
            x=jnp.array([131],dtype=jnp.int32),
            y=jnp.array([7],dtype=jnp.int32),
            on_field=jnp.array([1],dtype=jnp.int32)
        )

        ROOM_3_8.set_field(field_name=RoomTagsNames.ITEMS.value.items.value,
                         field_type=NamedTupleFieldType.NAMED_TUPLE_STACK,
                         content=[item_0, item_1, item_2],
                         named_tuple_type=Item,
                         requires_serialisation=True
                         )
        # Optinally conect a room from the last Layout to the start-room!
        if not last_room is None:
            last_room.connect_to(my_location=RoomConnectionDirections.DOWN, 
                                 other_room=ROOM_0_1, other_location=RoomConnectionDirections.DOWN)
      
      
      
        # connection between layer 1 rooms
        ROOM_0_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_0_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_0_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_0_2, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 1 and layer 2 rooms
        ROOM_0_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_1_1, other_location=RoomConnectionDirections.UP)
        ROOM_0_2.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_1_3, other_location=RoomConnectionDirections.UP)

        # connection between layer 2 rooms
        ROOM_1_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_2, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_1_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_1_4, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 2 and layer 3 rooms
        ROOM_1_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_1, other_location=RoomConnectionDirections.UP)
        ROOM_1_1.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_2, other_location=RoomConnectionDirections.UP)
        ROOM_1_2.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_3, other_location=RoomConnectionDirections.UP)
        ROOM_1_4.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_2_5, other_location=RoomConnectionDirections.UP)
        
        # connection between layer 3 rooms
        ROOM_2_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_4, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_4.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_5, other_location=RoomConnectionDirections.LEFT)
        ROOM_2_5.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_2_6, other_location=RoomConnectionDirections.LEFT)
        
        # connection between layer 3 and layer 4 rooms
        ROOM_2_3.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_4, other_location=RoomConnectionDirections.UP)
        ROOM_2_5.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_6, other_location=RoomConnectionDirections.UP)
        ROOM_2_6.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_3_7, other_location=RoomConnectionDirections.UP)
        
        # connection between layer 4 rooms
        ROOM_3_0.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_1, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_1.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_2, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_2.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_3, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_3.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_4, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_4.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_5, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_6.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_7, other_location=RoomConnectionDirections.LEFT)
        ROOM_3_7.connect_to(my_location=RoomConnectionDirections.RIGHT, 
                          other_room=ROOM_3_8, other_location=RoomConnectionDirections.LEFT)

        # bonus room to start room
        if self_loop:
            ROOM_3_0.connect_to(my_location=RoomConnectionDirections.DOWN, 
                          other_room=ROOM_0_1, other_location=RoomConnectionDirections.UP)
        
        
        if not self_loop:
            return LAYOUT, ROOM_3_0
        else:
            return LAYOUT
        
        

def make_difficulty_1_2(LAYOUT: PyramidLayout, consts):
    LAYOUT, last_room_l1 = make_difficulty_1(LAYOUT, consts, 
                                             last_room=None, self_loop=False)
    
    LAYOUT = make_difficulty_2(LAYOUT, consts=consts, 
                               last_room=last_room_l1, self_loop=True)
    return LAYOUT
    
def make_difficulty_1_2_3(LAYOUT: PyramidLayout, consts):
    LAYOUT, last_room_l1 = make_difficulty_1(LAYOUT, consts, 
                                             last_room=None, self_loop=False)
    
    LAYOUT, last_room_l2 = make_difficulty_2(LAYOUT, consts=consts, 
                               last_room=last_room_l1, self_loop=False)
    
    LAYOUT = make_difficulty_3(LAYOUT=LAYOUT, consts=consts, 
                               last_room=last_room_l2, 
                               self_loop=True)
    return LAYOUT