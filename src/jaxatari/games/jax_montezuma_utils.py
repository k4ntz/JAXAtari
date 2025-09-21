from __future__ import annotations
import jax.numpy as jnp
from typing import NamedTuple, Dict, Callable, List, Type, Tuple, Any, NewType, TypeVar
import inspect
from jaxatari.games.jax_montezuma_constants import *
from collections import namedtuple
import json
from enum import Enum
from jax import Array as jArray
import jax
from functools import partial
import copy
import itertools as it
import warnings



class RoomConnectionDirections(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    
    
class NamedTupleFieldType(Enum):
    INTEGER_SCALAR = 0 # Scalar integer array: These can be synchronised to the global state
    OTHER_ARRAY = 1 # All other array types, arbitrary size, datatype and dimenions. These can't be synchronized
    NAMED_TUPLE_STACK = 2 # Named tuples with scalar integer values in all fields. These can be 
        # synchronised to the global state but they are required to be registered to the 
        # ScalarNamedTupleDeserialisationHandler beforehand.
    
class RequiredRoomFields(Enum):
    ROOM_ID = "ROOM_ID"
    SPRITE = "sprite"
    HEIGHT = "height"
    VERTICAL_OFFSET = "vertical_offset"
    

# Declare aliases for specific types in order to make code more readable
TagNamedTuple = NewType('TagNamedTuple', NamedTuple)
RoomNamedTuple = NewType('RoomNamedTuple', NamedTuple)
ProtoRoomNamedTuple = NewType('ProtoRoomNamedTuple', NamedTuple)
MontezumaNamedTuple = NewType('MontezumaNamedTuple', NamedTuple)
TagEnum = NewType('TagEnum', Enum)
VanillaRoom = NewType('VanillaRoom', NamedTuple)
    
    
class SANTAH():
    # Originally "ScalarNamedTupleDeserializationHandler"
    # - game-agnostic tool for constructing multi-room games in jax.
    # - Along with the Pyramid Layout, allows you to write code that is 
    # - able to handle multiple rooms with different feature sets
    # - Without explicitly defining functions for each room.
    #   Internally, rooms are synchronized to a global storage array, 
    #   to preserve dynamic room attributes (States of enemies, doors, Items, ...).
    #   Serialization & deserialization to this Storage array happens automatically, 
    #   and the programmer does not need to handle persistence across rooms manually.
    #
    #
    # Used to automatically serialize & deserialize parts of the state.
    full_deserializations: Dict[Type, Callable[[jArray], NamedTuple]] = {}
    # Functions of the shape: jArray -> NamedIntegerSingletonTuple
    
    full_serialisations: Dict[Type, Callable[[NamedTuple], jArray]] = {}
    # Functions of the shape: NamedIntegerSingletonTuple -> jArray
    
    # Partial serialisation functions: Only serialise/ deserialize the parts of a named tuple
    # that require storage in the persistance array.
    partial_serialisations: Dict[Type, Callable[[NamedTuple], jax.Array]] = {}
    # Function of the shape: NamedIntegerSingletonTuple -> jArray
    partial_deserialisations: Dict[Type, Callable[[jax.Array], NamedTuple]] = {} 
        # Takes an existing named tuple, 
        # and overwrites the fields to be partially deserialized
        # Of the shape (NamedIntegerSingletonTuple, jArray) -> jArray
    attribute_setters: Dict[Type, Dict[str, Callable[[NamedTuple, jArray], NamedTuple]]] = {}
        # All the attribute setter functions. These can be accessed via: 
        # attribute_setters[NamedTuple]["field"]
    fully_serialised_sizes: Dict[Type, int] = {}
    partially_serialised_sizes: Dict[Type, int] = {}
    
    _fields_constructor: Dict[Type[NamedTuple], Dict[str, Callable[[NamedTuple], jArray]]] = {}
    
    
    # Fields related to the proto room:
    # The proto room consists of the parts of a room 
    # that are shared by all rooms (static type & shape)
    # and can thus be handled in un-wrapped functions.
    proto_room_field_enum: Type = None
    proto_room_named_tuple: NamedTuple = None
    room_to_proto_room: Dict[NamedTuple, Callable[[NamedTuple], NamedTuple]] = {}
    proto_room_to_room: Dict[NamedTuple, Callable[[NamedTuple, NamedTuple], NamedTuple]] = {}

    # The vanilla room. Has all the fields of the proto room as well as the shared by all but not the same fields.
    # Fields that are present in all rooms, but don't
    # necessarily have the same shape OR datatype
    
    vanilla_room: Type[NamedTuple] = None
    vanilla_room_enum: Type[Enum] = None
    shared_by_all_but_not_the_same: Enum = None
    
    # A mapper from Named tuples to their corresponding field enums. 
    # This is mostly used for type checking at startup time & to make 
    # sure that all necessary fields are defined for the rooms created via the API.
    named_tuple_field_enum_mapper: Dict[Type[NamedTuple], Type[Enum]] = {}
    
    
    
    # Fields that are related to the room tags. 
    # Tags are used to define specific functionalities (i.e. Items, LazerBarriers, Enemies, ...)
    # All fields related to a functionality need to be stored in the respective Tag.
    # This prevents reuse of fields & co-dependencies between functionality by design.
    #
    #
    #
    my_tags: Enum = None
    my_tag_mapping: Dict[Enum, Type[NamedTuple]] = {}
    tagged_field_field_type: Dict[str, NamedTupleFieldType] = {}
    extract_tag_from_rooms: Dict[Enum, Callable[[RoomNamedTuple], TagNamedTuple]] = {}
    registered_named_tuple: List[Type[NamedTuple]] = []
    registered_room_nt: List[Type[NamedTuple]] = []
    room_tags: Dict[Type[NamedTuple], Tuple[Enum]] = {}
    write_back_tag_information_to_room: Dict[Type[RoomNamedTuple], Dict[TagEnum, Callable[[RoomNamedTuple, TagNamedTuple], RoomNamedTuple]]]= {}
    
    
    # Constructor fields: 
    # The framework allows the developer to declare constructors/ functions to generate contents for specific fields at startup-time
    # - Constructors can be either declared at the Proto-Room, the Tag or the Room level. 

    
    
    _vanilla_room_field_constructors: Dict[str, Callable[[NamedTuple], jArray]] = {}
    tag_based_room_constructor_fields: Dict[Enum, Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]] = {}
    
        
    @classmethod
    def register_proto_room(cls, room_field_enum: Enum, proto_room_nt: NamedTuple, 
                            fields_that_are_shared_but_have_different_shape: Enum, 
                            vanilla_room_type: Type[NamedTuple],
                            vanilla_room_enum: Type[Enum],
                            constructed_fields: Dict[str, Callable[[VanillaRoom], jArray]] = {}):
        """Registers the template for all rooms. 
            All rooms are required to have at least all the fields specified for the Proto Room. 
            
        Args:
            room_field_enum (Type): Enum enumerating all the present fields in the 
                proto_room, i.e. all fields that are shared by all rooms and have the same shape & datatype.
                The values of the enum need to match the fields in the room, this is enforced
            proto_room_nt (NamedTuple): The default room named tuple. This is used to construct default rooms.

            fields_that_are_shared_but_have_different_shape (Enum):  Fields that are present in all rooms, 
                but have DIFFERENT Shapes and/ or data types in some rooms. 
                This is mostly only used for the collision/ render maps.
        

        """
        if cls.proto_room_named_tuple is not None:
            warnings.warn("Registering a second proto room will have no further effects")
            # Registering a proto room a second time does not really do any harm, 
            # it is just quite computationally expensive. 
            # If in the future it is necessary to test different layouts of the game 
            # in the same program execution, this check can be removed.
            return None
        cls.proto_room_field_enum = room_field_enum
        cls.proto_room_named_tuple = proto_room_nt
        cls.shared_by_all_but_not_the_same = fields_that_are_shared_but_have_different_shape
        cls.vanilla_room = vanilla_room_type
        cls.vanilla_room_enum = vanilla_room_enum
        
        # Check if the field_enum matches the fields present in the Vanilla room, if not throw an error.
        nd_fields = set([f.value for f in cls.proto_room_field_enum])
        pr_fields = set(cls.proto_room_named_tuple._fields)
        vanilla_p_fields = set(cls.vanilla_room._fields)
        vanilla_r_fields =  set([f.value for f in cls.vanilla_room_enum])
        if vanilla_p_fields != vanilla_r_fields:
            raise Exception("Mismatch between the actual fields of the vanilla room, and those specified in the enum")
        
        # Check that there is no overlap between the proto room fields & the flexible-shape vanilla room fields.
        shared_but_not_the_same = set([f.value for f in cls.shared_by_all_but_not_the_same])
        if len(pr_fields.intersection(shared_but_not_the_same)) > 0:
            raise Exception("Overlap between Proto-room fields and 'fields_that_are_share_but_have_different_shape' detected.")
        if nd_fields != pr_fields:
            raise Exception("The Fields of the Proto-Room-Field-Enum do not match those of the initial proto room")

        # Now check that all declared constructor fields are either present in the proto room, or are "shared by all but not the same" fields.
        allowed_fields = set(cls.proto_room_named_tuple._fields).union(set([f.value for f in cls.shared_by_all_but_not_the_same]))
        constructed_keys = set(constructed_fields.keys())
        if len(constructed_keys.intersection(allowed_fields)) != len(constructed_keys):
            raise Exception("It seems that you have specified some constructed fields which are neither present in the proto room, nore are 'shared by all but not the same shape.'")
        
        # Now check that the vanilla room has exactly only those fields that are declared in the 
        # proto room or 'shared_by_all_but_not_the_same'. 
        vnailla_room_fields = set(cls.vanilla_room._fields)
        allowed_fields = set(cls.proto_room_named_tuple._fields).union(set([f.value for f in cls.shared_by_all_but_not_the_same]))
        if not vnailla_room_fields == allowed_fields:
            raise Exception("The vanilla room has the wrong fields. It's only allowed to have all fields of the proto room and the shared-by-all-but-not-the-same fields.")
        
        # Make setter functions for all fields in the proto room. 
        # We use setter functions indexed by enum values instead of the usual "_replace" for named tuples, 
        # So that the names of fields can be changed if necessary, without any rewriting of the program code required.
        cls._vanilla_room_field_constructors = constructed_fields
        def make_setter_functions(tuple_class: NamedTuple):
            fields: List[str] = list(tuple_class._fields)
            def single_setter_function(tuple: NamedTuple, overwrite_value: jArray, overwrite_field: str, all_fields: List[str], t_class: NamedTuple):
                field_dict = {}
                for k in all_fields:
                    field_dict[k] = getattr(tuple, k)
                field_dict[overwrite_field] = overwrite_value
                ret = t_class(**field_dict)
                return ret
            setter_functions: Dict[str, Callable[[NamedTuple, jArray], NamedTuple]] = {}
            for f in fields:
                func = partial(single_setter_function, overwrite_field = f, all_fields = fields, t_class = tuple_class)
                func = jax.jit(func)
                setter_functions[f] = func
            return setter_functions
        
        cls.attribute_setters[proto_room_nt] = make_setter_functions(proto_room_nt)
        
    @classmethod
    def register_room(cls, room: Type[NamedTuple], field_enum: Type[Enum],
            constructed_fields: Dict[str, Callable[[NamedTuple], jArray]] = {}, 
            tags: Tuple[Type[Enum]] = ()):
        """Register a new room. This function is only used internally, and should not be touched by the user.
           Registering a room causes SANTAH to generate all necessary utility functions to cast from ProtoRoom -> Registered room 
           and the other way round. 

        Args:
            room (Type[NamedTuple]): Named tuple describing the room we want to register
            field_enum (Type[Enum]): Enum specifying all fields present in the room to be registered
            constructed_fields (Dict[str, Callable[[NamedTuple], jArray]], optional): Optional constructors 
                at the room level: These are used to precompute the contents of certain fields. 
                We have used this quite a lot to precompute hitmaps for items, doors, lazer-barriers, ... . Defaults to {}.
            tags (Tuple[Type[Enum]], optional): Tags implemented by these room. Tags specify distinct functionalities, 
            and a room is allowed to implement arbitrary many distinc functionalities. Defaults to ().
        """
        
        if room in cls.registered_room_nt:
            # Again, re-registering rooms isn't harmful, it's just very inefficient.
            warnings.warn("Attempts to reregister a room will be ignored, in some cases this might have unintended consequences")
            return None
        # The tags defined for this game need to be registered before any specific rooms are registered
        if cls.my_tags is None:
            raise Exception("Cannot register rooms if the tags used by the Handler have not been defined.")
        # Check if only known tags are used.
        tag_vals = [e for e in cls.my_tags]
        for t in tags:
            if t not in tag_vals:
                raise Exception("The Tag '" + str(t) + "' is unknown.")
        
        
        
        # Enforce some conditions on the attributes that are necessary
        # for the framework to work properly.
        if room in cls.registered_room_nt:
            raise Exception("Room has already been registered, register room only once")
        if cls.proto_room_named_tuple is None or cls.proto_room_field_enum is None:
            raise Exception("Register proto room before registering any individual rooms.")
        if not set(cls.proto_room_named_tuple._fields) <= set(room._fields):
            raise Exception("The registered room does not have all attribute required by the proto room.")
        if not set([f.value for f in cls.shared_by_all_but_not_the_same]) <= set(room._fields):
            raise Exception("Room needs to implement all 'shared_by_all_but_not_the_same' fields.")
        
        
        
        
        if field_enum is None:
            raise Exception("Room needs to be accompanied by a fitting field_enum")
        if set([e.name for e in field_enum]) != set(room._fields):
            raise Exception("The registered room does not have all attribute specified by it's field enum.")
        if not set([e.value for e in field_enum]) == set(room._fields):
            raise Exception("The values in the enum need to match the field names.")
        
        
        
        
        
        if not len(set(list(constructed_fields.keys())).intersection(set(cls.proto_room_named_tuple._fields))) == 0:
            raise Exception("Proto room fields may not be constructed.")
        
        
        
        # Check that all fields required by the declared tags are actually present in the room.
        if not isinstance(tags, Tuple):
            raise Exception("The tags that apply to this room need to be specified in a Tuple.")
        room_fields = set(room._fields)
        for t in tags:
            tag_nt: Type[NamedTuple] = cls.my_tag_mapping[t]
            tag_fields = set(tag_nt._fields)
            if len(room_fields.intersection(tag_fields)) < len(tag_fields):
                raise Exception("The room does not have all the fields specified by the Tag '" + str(t) + "'. Room Cannot be accepted")
        cls.room_tags[room] = tags
        
        # Make infrastructure to automatically
        # read & write Tags back to the room
        # We construct it here, so that we can chage the content of tags without 
        # making any adjustments to the code
        def write_tag_to_room(_room: RoomNamedTuple, _tag: TagNamedTuple, 
                    _room_type: Type[RoomNamedTuple], _room_fields: List[str], 
                    _tag_type: Type[TagNamedTuple], _tag_fields: List[str]):
            kwargs = {}
            for f in _room_fields:
                kwargs[f] = getattr(_room, f)
            for f in _tag_fields:
                kwargs[f] = getattr(_tag, f)
            _new_room = _room_type(**kwargs)
            return _new_room
        
        cls.write_back_tag_information_to_room[room] = {}
        for t in tags:
            tag_nt: Type[NamedTuple] = cls.my_tag_mapping[t]
            _get_tag = partial(write_tag_to_room,
                _room_type = room, 
                _room_fields=list(room._fields), 
                _tag_type=tag_nt, 
                _tag_fields=list(tag_nt._fields)
            )
            
            _get_tag = jax.jit(_get_tag)
            cls.write_back_tag_information_to_room[room][t] = _get_tag
        
        # Construct individual setter functions for the fields here.
        # Again, handled via enums to make renaming of fields easier/ possible.
        cls._fields_constructor[room] = constructed_fields
        def make_setter_functions(tuple_class: NamedTuple):
            fields: List[str] = list(tuple_class._fields)
            def single_setter_function(tuple: NamedTuple, overwrite_value: jArray, overwrite_field: str, all_fields: List[str], t_class: NamedTuple):
                field_dict = {}
                for k in all_fields:
                    field_dict[k] = getattr(tuple, k)
                field_dict[overwrite_field] = overwrite_value
                ret = t_class(**field_dict)
                return ret
            setter_functions: Dict[str, Callable[[NamedTuple, jArray], NamedTuple]] = {}
            for f in fields:
                func = partial(single_setter_function, overwrite_field = f, all_fields = fields, t_class = tuple_class)
                func = jax.jit(func)
                setter_functions[f] = func
            return setter_functions
        
        cls.attribute_setters[room] = make_setter_functions(room)
        
        # Function to cast individual rooms to proto rooms. 
        # This is necessary for the wrapper functionality which implicitly handles 
        # all the switch cases necessary to deal with the fact that we have many different rooms with many differet feature sets.
        def _make_to_proto_room_function(not_proto_room: NamedTuple, not_proto_room_type: NamedTuple, proto_room_type: NamedTuple) -> NamedTuple:
            kwargs_dict = {}
            for f in proto_room_type._fields:
                kwargs_dict[f] = getattr(not_proto_room, f)
            new_proto_room = proto_room_type(**kwargs_dict)
            return new_proto_room
        
        
        # Raises a proto room up to the specific room class. 
        # Again, necessary to automatically handle all the case distinctions between different rooms.
        def _transfer_changes_from_proto_room(not_proto_room: NamedTuple, proto_room: NamedTuple, not_proto_room_type: NamedTuple, proto_room_type: NamedTuple) -> NamedTuple:
            """Transfers the field from a proto room to an existing non-proto room. 
               Overwrites all fields of the non-protoroom shared with the proto room

            Args:
                not_proto_room (NamedTuple): The arbitrary non-proto room.
                proto_room (NamedTuple): The proto room
            """
            kwargs_dict = {}
            for f in not_proto_room_type._fields:
                kwargs_dict[f] = getattr(not_proto_room, f)
            for f in proto_room_type._fields:
                kwargs_dict[f] = getattr(proto_room, f)
            new_proto_room = not_proto_room_type(**kwargs_dict)
            return new_proto_room
        
        to_proto_room_lambda = lambda np_room, nprt=room, prt=cls.proto_room_named_tuple : _make_to_proto_room_function(not_proto_room=np_room, 
                                                                                                        not_proto_room_type=nprt, 
                                                                                                        proto_room_type=prt)
        to_proto_room_lambda = jax.jit(to_proto_room_lambda)
        
        back_to_not_proto_room_lambda = lambda np_room, pr_room, npr_type=room, prt=cls.proto_room_named_tuple : _transfer_changes_from_proto_room(
                                        not_proto_room=np_room, proto_room=pr_room, 
                                        not_proto_room_type=npr_type, 
                                        proto_room_type=prt
        )
        back_to_not_proto_room_lambda = jax.jit(back_to_not_proto_room_lambda)
        cls.room_to_proto_room[room] = to_proto_room_lambda
        cls.proto_room_to_room[room] = back_to_not_proto_room_lambda
        cls.registered_room_nt.append(room)
        
        
        
    
    @classmethod
    def create_room_from_tags(cls, room_id: int, tags: Tuple[Type[Enum]] = ()) -> Type[NamedTuple]:
        """Internal function used to create a new room from only the vanilla room & the specified set of tags.
            CAUTION: This function shouldn't be called by the user. Instead, use the appropriate function in the layout manager.
        Args:
            tags (Tuple[Type[Enum]], optional): Tags describing the functionality this 
                room should implement. Defaults to ().

        Returns:
            Type[NamedTuple]: An appropriately integrated room type.
        """
        
        # Handle all checks necessary to ensure that a valid room can be created.
        room_name: str = "room_" + str(room_id)
        if cls.vanilla_room is None:
            raise Exception("Vanilla room needs to be registered first.")
        for t in tags:
            if t not in cls.my_tag_mapping:
                raise Exception("The tag '" + str(t) + "' is not recognized.")
        
        room_nt_fields: List[str] = []
        for f in cls.vanilla_room._fields:
            room_nt_fields.append(f)
        for t in tags:
            tag_nt: Type[NamedTuple] = cls.my_tag_mapping[t]
            for f in tag_nt._fields:
                if f in room_nt_fields:
                    raise Exception("One of the tags shares fields with the proto room. This is forbidden")
                room_nt_fields.append(f)
        
        enum_declr = {}
        for f in room_nt_fields:
            enum_declr[f] = f
        my_room_named_tuple: Type[NamedTuple] = namedtuple(room_name, room_nt_fields)
        my_room_enum: Type[Enum] = Enum(room_name, enum_declr)
        cls.register_room(room=my_room_named_tuple, 
                          field_enum=my_room_enum, 
                          constructed_fields={}, 
                          tags=tags)
        return my_room_named_tuple
    
    
    @classmethod
    def register_room_tags(cls, room_tags: Type[Enum], room_tags_descriptor: Type[Enum], 
                           constructed_fields: Dict[Enum, Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]]] = {}):
        """Gives the option to register a set of room tags. 
            Room tags are used to signify, that a room has certain features, such as ladders/ LAZERWALLS/ or a Bottomless pit. 
            Room tags are always paired with a NamedTuple representing the fields which are needed for the 
                functionality described by the TAG. 
            In the backend, the necessary Infrastructure is generated to allow for wrapped handling of the 
            room behavior solely in terms of the Proto room and all supported Tags

        Args:
            room_tags (Enum): An Enum of all Tags supported by the Montezuma Implementation. 
                Values in the Enum are Named tuples specifying the fields required to support the tag
            constructed_fields:  Dict[Enum, Dict[str, Callable[[NamedTuple], jArray]]] = {} : Enables support for constructed 
                fields at the tag level. 
                The Dict needs to map Keys from the room_tags enum to dictionaries mapping {field_name -> constructors}.
                The constructors receive both a Vanilla Room representing the already set fields (all fields defined via 
                the room API in game & the constructed fields from the Vanilla Room) and the appropriate room tag, 
                with all fields defined via the Room-API and 0 default values for the not already constructed fields. 
                

        """
        if not cls.my_tags is None:
            warnings.warn("roomtags have already been registered; this call will have no further effects")
            return None
        if cls.proto_room_named_tuple is None:
            raise Exception("Need to register the Proto room before registering the Room Tags")
        
        
        for e in room_tags:
            tag_nt: Type[NamedTuple] = e.value
            if tag_nt in cls.registered_room_nt:
                raise Exception("The tag " + str(e) + " has been registered as a Room, but it needs to be registered as a named tuple.")
            if not tag_nt in cls.registered_named_tuple:
                raise Exception("The tag needs to be registered as a named tuple.")
            
            
        # Check if all constructed fields are actually present in their respective tags.
        existing_room_tags = set([e for e in room_tags])
        t_constrs = set(list(constructed_fields.keys()))
        if not t_constrs.issubset(existing_room_tags):
            raise Exception("The keys of the 'constructed_fields' needs to be a subset of the fields in the respecitv tags.")
        
        # Check if fields for which constructors are defined are acually present in the respective tags.
        for k in constructed_fields.keys():
            tag_nt: Type[NamedTuple] = k.value
            if not set(constructed_fields[k].keys()).issubset(set(tag_nt._fields)):
                raise Exception("It seems that you have tried to write a constructor for a field that does not exist in the Room Tag " + str(k))
        cls.tag_based_room_constructor_fields = constructed_fields
        #
        # Check if the all the Named tuples used for the tags are properly registered with their descriptor. 
        # We check if they have already been registered with the ScalarNamedTupleDeserializationHandler, 
        # because the necessary checks are already performed there
        #
        if room_tags_descriptor is None:
            raise Exception("A room_tags_descriptor object needs to be provided")
        t1 = set([e.name for e in room_tags_descriptor])
        t2 = set([e.name for e in room_tags])
        if len(t1.intersection(t2)) != len(t1) or len(t1.intersection(t2)) != len(t2):
            raise Exception("Mismatch between the present tags in the Tag definition vs the Tag annotation.")
        def_dict = {}
        for e in room_tags:
            def_dict[e.name] = e.value
        descr_dict = {}
        for e in room_tags_descriptor:
            descr_dict[e.name] = e.value
        for k in list(def_dict.keys()):
            if not def_dict[k] in cls.named_tuple_field_enum_mapper:
                raise Exception("The Named tuple for the field '" + k + "' needs to be registered with an appropriate Annotation Enum.")
            if cls.named_tuple_field_enum_mapper[def_dict[k]] != descr_dict[k]:
                raise Exception("Mismatch between Named Tuple annotation given when initially registering the Tag Named Tuple and the Annotation given in the 'room_tags_descriptor' object for the tag '" + k + "'.")
        
        
        #
        cls.my_tags = room_tags
        # Make sure that no tags overlap.
        nt_field_set_list = []
        collection_set = None
        for nt in [e.value for e in room_tags]:
            nt: Type[NamedTuple] = nt
            nt_field_set_list.append(set(nt._fields))
            if collection_set is None:
                collection_set = set(nt._fields)
            else:
                collection_set = collection_set.union(set(nt._fields))
        are_disjoint = all((set(p0).isdisjoint(set(p1))) for p0, p1 in it.combinations(nt_field_set_list, 2))
        if not are_disjoint:
            raise Exception("The Field Sets for all tags are required to be pairwise disjoint.")
        if len(collection_set.intersection(set(cls.proto_room_named_tuple._fields))) > 0:
            raise Exception("Fields declared to be parts of tags and fields declared to be part of the proto room may not overlap.")
        
        
        # Make the Infrastructure to extract the Named tuples specified for each TAG from individual rooms
        for e in room_tags:
            def tnt_maker(room: NamedTuple, tag_nt: Type[NamedTuple]):
                kwargs = {}
                for f in list(tag_nt._fields):
                    kwargs[f] = getattr(room, f)
                nt = tag_nt(**kwargs)
                return nt
            tag_extractor: Callable[[NamedTuple], NamedTuple] = partial(tnt_maker, 
                                                                        tag_nt=e.value)
            tag_extractor = jax.jit(tag_extractor)
            cls.extract_tag_from_rooms[e] = tag_extractor
            
        # collect all the tag values in a dict for easier processing
        for e in room_tags:
            cls.my_tag_mapping[e] = e.value
                
                
        
        
    @classmethod
    def add_new_named_tuple(cls, tup: NamedTuple, field_enum: Type[Enum], partial_serialisation_fields: List[str] = []):
        """Generates functionality for serialization & deserialization of the named tuple.

        Args:
            tuple (NamedTuple): Tuple Class for which the jitted serialisation routines are created. 
                This needs to be the Class, i.e. Not an object.
            partial_serialisation_fields (List[str], optional): Attributes which are saved & recovered
                during partial serialisation/ deserialisation. Partial serialisation operations 
                should be used to preserve individual values in the global state. Defaults to [].
        """
        # Full serialisation method. Serialises the whole named tuple.
        if tup in cls.registered_named_tuple:
            warnings.warn("Named tuple has already been registered; this call has no further effects")
            return None
        if field_enum is None:
            raise Exception("A field_enum needs to be provided")
        if set([e.name for e in field_enum]) != set(tup._fields):
            raise Exception("The registered named tuple does not have all attribute specified by the field enum.")
        if not set([e.value for e in field_enum]) == set(tup._fields):
            raise Exception("The enum has the wrong values")
        
        
        
        if tup in cls.registered_room_nt:
            raise Exception("The NamedTuple has already been registered as a Room.")
        def serialize_fully(tuple_class: NamedTuple):
            fields: Tuple[str] = tuple_class._fields
            fields = list(fields)
            fields = sorted(fields)
            def jittable_serializer(named_tup, fields):
                ret_arr: jnp.ndarray = jnp.zeros(shape=(len(fields)), dtype=jnp.int32)
                for i, f in enumerate(fields):
                    ret_arr = ret_arr.at[i].set(getattr(named_tup, f)[0])
                return ret_arr
            return jax.jit(partial(jittable_serializer, fields=fields))
        
        
        # Full deserialisation method. Deserialises the whole named tuple
        def deserialize_fully(tuple_class: NamedTuple):
            fields: Tuple[str] = tuple_class._fields
            fields = list(fields)
            fields = sorted(fields)
            def jittable_deserializer(serialized_array, fields, t_class):
                res = {}
                for i, f in enumerate(fields):
                    res[f] = jnp.array([serialized_array[i]])

                ret = t_class(**res)
                return ret
            return jax.jit(partial(jittable_deserializer, fields=fields, t_class=tuple_class))
        
        
        
        # Partial serialisation method. Serialises the whole named tuple.
        def serialize_partially(tuple_class: NamedTuple, partial_serialisation_fields: List[str]):
            fields: Tuple[str] = partial_serialisation_fields
            
            fields = sorted(fields)
            def jittable_serializer(named_tup, fields):
                ret_arr: jnp.ndarray = jnp.zeros(shape=(len(fields)), dtype=jnp.int32)
                for i, f in enumerate(fields):
                    ret_arr = ret_arr.at[i].set(getattr(named_tup, f)[0])
                return ret_arr
            return jax.jit(partial(jittable_serializer, fields=fields))
        
        
        # Partial deserialisation method. Deserialises the whole named tuple
        def deserialize_partially(tuple_class: NamedTuple, partial_serialisation_fields: List[str]):
            fields: Tuple[str] = tuple_class._fields
            fields = list(fields)
            fields = sorted(fields)
            partial_serialisation_fields = sorted(partial_serialisation_fields)
            def jittable_deserializer(named_tuple, serialised_array, fields, t_class, partial_serialisation_fields):
                res = {}
                for i, f in enumerate(fields):
                    res[f] = getattr(named_tuple, f)
                for i, f in enumerate(partial_serialisation_fields):
                    res[f] = jnp.array([serialised_array[i]])
                ret = t_class(**res)
                return ret
            return jax.jit(partial(jittable_deserializer, fields=fields, t_class=tuple_class, partial_serialisation_fields=partial_serialisation_fields))
        
        
        # Generate setter functions, again for easier access.
        def make_setter_functions(tuple_class: NamedTuple):
            fields: List[str] = list(tuple_class._fields)
            def single_setter_function(tuple: NamedTuple, overwrite_value: jArray, overwrite_field: str, all_fields: List[str], t_class: NamedTuple):
                field_dict = {}
                for k in all_fields:
                    field_dict[k] = getattr(tuple, k)
                field_dict[overwrite_field] = overwrite_value
                ret = t_class(**field_dict)
                return ret
            setter_functions: Dict[str, Callable[[NamedTuple, jArray], NamedTuple]] = {}
            for f in fields:
                func = partial(single_setter_function, overwrite_field = f, all_fields = fields, t_class = tuple_class)
                func = jax.jit(func)
                setter_functions[f] = func
            return setter_functions
        
        # Store all generated functionality.
        cls.registered_named_tuple.append(tup)
        cls.attribute_setters[tup] = make_setter_functions(tup)
        cls.full_serialisations[tup] = serialize_fully(tuple_class=tup)
        cls.full_deserializations[tup] = deserialize_fully(tuple_class=tup)
        cls.partial_serialisations[tup] = serialize_partially(tuple_class=tup, 
                                partial_serialisation_fields=partial_serialisation_fields)
        cls.partial_deserialisations[tup] = deserialize_partially(tuple_class=tup, 
                                                                  partial_serialisation_fields=partial_serialisation_fields)
        cls.fully_serialised_sizes[tup] = len(tup._fields)
        cls.partially_serialised_sizes[tup] = len(partial_serialisation_fields)
        cls.named_tuple_field_enum_mapper[tup] = field_enum
        







class RoomConnectionObject():
    """
    RoomConnectionObjects are used to define the connections between rooms. 
    They essentially are used to form an undirected, fully connected graph. 
    Self-loops, i.e. connecting a room to itself does not work as of yet.
    """
    def __init__(self, room_id: int):
        self.room_id = room_id
        self.left: RoomConnectionObject = None
        self.right: RoomConnectionObject = None
        self.up: RoomConnectionObject = None
        self.down: RoomConnectionObject = None
        self.visited: bool = False
        
        
    def unvisit(self):
        #
        # Not needed anymore.
        #
        if not self.visited:
            return
        else:
            self.visited = False
            if not self.left is None:
                self.left.unvisit()
            if not self.right is None:
                self.right.unvisit()
            if not self.up is None:
                self.up.unvisit()
            if not self.down is None:
                self.down.unvisit()
                
                
    def visit(self, visited_dict: Dict[int, Dict[RoomConnectionDirections, Tuple[int, RoomConnectionDirections]]] = None) -> Dict[int, Dict[RoomConnectionDirections, Tuple[int, RoomConnectionDirections]]]:
        """Traverses the list of rooms and returns a dictionary that maps out the connections between rooms

        Returns:
            Dict[int, Dict[RoomConnectionDirections, int]]: Dictionary mapping RoomID to a dictionary which maps RoomConnectionDirection to the 
                RoomID of the room which is connected in that direction. 
                If no room is connected in that direction, we use a default value of -1.
        """
        # A helper function used to find out, which side of the other room I am connected to.
        def which_direction_am_i_in_(forein_room_connection_object: RoomConnectionObject, my_id: int):
            if (not forein_room_connection_object.left is None) and forein_room_connection_object.left.room_id == my_id:
                return RoomConnectionDirections.LEFT.value 
            elif (not forein_room_connection_object.right is None) and forein_room_connection_object.right.room_id == my_id:
                return RoomConnectionDirections.RIGHT.value
            elif (not forein_room_connection_object.up is None) and forein_room_connection_object.up.room_id == my_id:
                return RoomConnectionDirections.UP.value
            elif (not forein_room_connection_object.down is None) and forein_room_connection_object.down.room_id == my_id:
                return RoomConnectionDirections.DOWN.value
            else:
                raise Exception("Error during room-connection mapping.")
        if visited_dict is None:
            visited_dict = {}
        if self.room_id in visited_dict or self.visited:
            return visited_dict
        self.visited = True
        my_visit: Dict[RoomConnectionDirections, Tuple[int, RoomConnectionObject]] = {}
        # generate my own connection object.
        # Gathers the information which Rooms I am connected to & at which sides. 
        # This setup allows for non-euclidean room layouts (i.e. you can configure the layout so that if you exit a room to the left
        # you enter the other room to the left as well (or to the top/bottom/right))
        if self.left is None:
            my_visit[RoomConnectionDirections.LEFT] = (-1, -1)
        else:
            otherside_dir = which_direction_am_i_in_(forein_room_connection_object=self.left, 
                                                     my_id=self.room_id)
            my_visit[RoomConnectionDirections.LEFT] = (self.left.room_id, otherside_dir)
        
        
        if self.right is None:
            my_visit[RoomConnectionDirections.RIGHT] = (-1, -1)
        else:
            otherside_dir = which_direction_am_i_in_(forein_room_connection_object=self.right, 
                                                     my_id=self.room_id)
            my_visit[RoomConnectionDirections.RIGHT] = (self.right.room_id, otherside_dir)
            
        if self.up is None:
            my_visit[RoomConnectionDirections.UP] = (-1, -1)
        else:
            otherside_dir = which_direction_am_i_in_(forein_room_connection_object=self.up, 
                                                     my_id=self.room_id)
            my_visit[RoomConnectionDirections.UP] = (self.up.room_id, otherside_dir)
            
            
        if self.down is None:
            my_visit[RoomConnectionDirections.DOWN] = (-1, -1)
        else:
            otherside_dir = which_direction_am_i_in_(forein_room_connection_object=self.down, 
                                                     my_id=self.room_id)
            my_visit[RoomConnectionDirections.DOWN] = (self.down.room_id, otherside_dir)
        visited_dict[self.room_id] = my_visit    
            
        # Recursively visit all neighboring rooms, 
        # to fully map out the Room layout.
        if not self.left is None:
            visited_dict = self.left.visit(visited_dict)
        if not self.right is None:
            visited_dict = self.right.visit(visited_dict)
        if not self.up is None:
            visited_dict = self.up.visit(visited_dict)
        if not self.down is None:
            visited_dict = self.down.visit(visited_dict)
        
        return visited_dict
        
class Room:
    def __init__(self, room_id: int, underlyingTupleClass: NamedTuple):
        """Initialization method for the "Room" handler class. 
           At its core, it wraps around a tuple of the underlyingTupleClass
           and provides a convenient wrapper around individual instances of the
           Named Tuple. Rooms are automatically tracked by the layout manager & integrated with the 
           persistence infrastructure. 
           
           Provides the option to set individual fields to Arrays, Lists of Scalar valued Integer Named tuples, 
           and singleton arrays. 
           ScalarValuedIntegerNamedTuples need to always be registered at the ScalarNamedIntegerTupleDeserialisaitonHandler first.

        Args:
            room_id (int): ID of the underlying Room
            underlyingTupleClass (NamedTuple): NamedTuple Class that gives the basis for the room-tuple this object wraps around.
        """
        self.underlyingTupleClass: NamedTuple = underlyingTupleClass
        self.room_id: int = room_id
        self.connection_object: RoomConnectionObject = RoomConnectionObject(room_id=self.room_id)
        self.field_contents: Dict[str, Any] = {}
        self.field_contents[RequiredRoomFields.ROOM_ID.value] = jnp.array([self.room_id], dtype=jnp.uint16)
        self.field_persistent: Dict[str, bool] = {} # Whether a certain field is supposed to be stored in the 
            # persistant global storage. 
        # Room-IDs may never change, so they don't require persistence.
        self.field_persistent[RequiredRoomFields.ROOM_ID.value] = False
        self.field_type: Dict[str, NamedTupleFieldType] = {} # type of content in this field.
        self.field_type[RequiredRoomFields.ROOM_ID.value] = NamedTupleFieldType.INTEGER_SCALAR
        self.field_named_tuples: Dict[str, NamedTuple] = {} # for all fields that are supposed to be synchronized to 
            # the required global storage and have a namedtuple as value, the class of the named_tuple is stored in here. 
        self.present_fields: List[str] = list(self.underlyingTupleClass._fields)
        if not RequiredRoomFields.ROOM_ID.value in self.present_fields:
            raise Exception("All NamedTuples representing individual rooms are required to have a 'ROOM_ID' field")
        
        
        
    def set_field(self, field_name: str, field_type: NamedTupleFieldType, content: jnp.ndarray|List[NamedTuple], requires_serialisation: bool = False, named_tuple_type: NamedTuple=None):
        """The main method through which fields of individual rooms are set.
        Args:
            field_name (str): name of the field for which the value is to be set. 
                This field name is required to actually be a field in the given namedtuple subclass.
            field_type (NamedTupleFieldType): Type of the field to be set. This can either be integer singleton array, 
                arbitrary array, or a list of NamedTuples which are eventually stored as an array. 
                If it is a list of named tuples, all need to have the same type.
            content (jnp.ndarray | List[NamedTuple]): Content of the field as described above.
            requires_serialisation (bool, optional): Whether the field is supposed to be serialised to the global state
                CAUTION: ONLY SINGLETON INTEGER FIELDS OR NAMED TUPLE FIELDS SUPPORT SYNCHRONISATION TO THE GLOBAL STATE. Defaults to False.
        """
        # Check whether valid field content was passed.
        if not field_name in self.present_fields:
            raise Exception("Attempted to add content to a field that is not present in the underlying namedtuple")
        if field_name == RequiredRoomFields.ROOM_ID.value:
            raise Exception("Cannot overwrite the fixed field 'ROOM_ID'")
        if field_type == NamedTupleFieldType.INTEGER_SCALAR:
            if not isinstance(content, jArray):
                raise Exception("Missmatch between stated field_type and given field_type")
            data: jnp.ndarray = content
            if not data.dtype == jnp.int32:
                raise Exception("Cannot support non int32 singleton arrays")
            if data.shape != (1,):
                raise Exception("Only support singleton arrays of shape '(1, )'")
            self.field_contents[field_name] = data
            self.field_persistent[field_name] = requires_serialisation
            self.field_type[field_name] = NamedTupleFieldType.INTEGER_SCALAR
            
        elif field_type == NamedTupleFieldType.NAMED_TUPLE_STACK:
            if named_tuple_type is None:
                raise Exception("If a list of named tuples is provided as field content, a NamedTupleType needs to be given as well")
            if not isinstance(content, List):
                raise Exception("Expected a list of NamedTupleObjects as content")
            for i in content:
                if not isinstance(i, named_tuple_type):
                    raise Exception("All items in the list need to be of the required named_tuple_type")
            if named_tuple_type not in SANTAH.full_serialisations:
                raise Exception("The NamedTupleType need to be registered with the ScalarNamedTupleDeserialisationHandler")
            field_content: jArray = jnp.zeros((len(content), SANTAH.fully_serialised_sizes[named_tuple_type]), dtype=jnp.int32)
            for i, d in enumerate(content):
                serialised = SANTAH.full_serialisations[named_tuple_type](d)
                field_content = field_content.at[i, ...].set(serialised)
            self.field_contents[field_name] = field_content
            self.field_persistent[field_name] = requires_serialisation
            self.field_type[field_name] = NamedTupleFieldType.NAMED_TUPLE_STACK
            self.field_named_tuples[field_name] = named_tuple_type
            
        elif field_type == NamedTupleFieldType.OTHER_ARRAY:
            if requires_serialisation:
                raise Exception("Cannot serialise field of type 'OTHER_ARRAY'")
            self.field_contents[field_name] = content
            self.field_persistent[field_name] = False
            self.field_type[field_name] = NamedTupleFieldType.OTHER_ARRAY
            
            
    def get_jitted_room_constructor(self)->Callable[[], NamedTuple]:
        """Returns a jitted function that constructs the room-namedtuple specified by this object.

        Returns:
            Callable[[], NamedTuple]: _description_
        """
        
        # Collect all the fields that need to be constructed according to the Tags this room implements.
        implemented_tags: Tuple[Enum] = SANTAH.room_tags[self.underlyingTupleClass]
        tag_constructed_fields = set([])
        for tag in implemented_tags:
            if tag in SANTAH.tag_based_room_constructor_fields:
                constructed_fields = SANTAH.tag_based_room_constructor_fields[tag].keys()
                tag_constructed_fields = tag_constructed_fields.union(set(constructed_fields))
                

        if set(self.field_contents.keys()).union(list(SANTAH._fields_constructor[self.underlyingTupleClass].keys())).union(
            
            list(SANTAH._vanilla_room_field_constructors.keys())).union(tag_constructed_fields) != set(self.present_fields):
            # Fields for which an explicit constructor is given are not required to be
            # initialized manually
            # Explicit constructor can either be given at the proto room level, at the tag level or at the per-room level. 
            # Per room level takes precedence over proto room level & the tag level
            raise Exception("Not all fields are set. All fields need to be set before generating infrastructure functions.")
        # For all constructed fields for which an init value is used, the init value is used 
        # to initialize the respective fields of the named tuple used to construct the default fields. 
        # For all remaining fields, a default value is used.
        
        if SANTAH.vanilla_room is None:
            raise Exception("Vanilla Room has not been set.")
        for k in list(SANTAH._fields_constructor[self.underlyingTupleClass].keys()):
            if k in self.field_contents:
                if self.field_type[k] != NamedTupleFieldType.OTHER_ARRAY:
                    raise Exception("Fields for which an explicit constructor is given on a per-room basis may only be declared as type 'OTHER_ARRAY'.")
        #
        # Check that If a constructor initialized field has already been given a default value, 
        # it is only declares as OTHER_ARRAY.
        #
        #
        
        for k in list(SANTAH._vanilla_room_field_constructors.keys()):
            if k in self.field_contents:
                if self.field_type[k] != NamedTupleFieldType.OTHER_ARRAY:
                    raise Exception("Fields for which an explicit constructor is given on the proto-room level may only be declared as type 'OTHER_ARRAY'.")
        
        for k in list(tag_constructed_fields):
            if k in self.field_contents:
                if self.field_type[k] != NamedTupleFieldType.OTHER_ARRAY:
                    raise Exception("Fields for which an explicit constructor is given on the tag level may only be declared as type 'OTHER_ARRAY'.")
        
        
        
        # Get the values for all constructed fields
        #
        constructed_fields: List[str] = list(SANTAH._fields_constructor[self.underlyingTupleClass].keys())
        constructed_fields.extend(list(SANTAH._vanilla_room_field_constructors.keys()))
        constructed_fields.extend(list(tag_constructed_fields))
        init_field_content_underlying_tuple_class: Dict[str, Any] = copy.deepcopy(self.field_contents)
        init_field_content_vanilla_room: Dict[str, Any] = {}
        default_fs: List[str] = list(set(constructed_fields).difference(set(list(self.field_contents.keys()))))
        
        vanilla_room_fields: List[str] = [ e.value for e in SANTAH.vanilla_room_enum]
        for f in default_fs:
            init_field_content_underlying_tuple_class[f] = 0
            
        for c_f in constructed_fields:
            self.field_type[c_f] = NamedTupleFieldType.OTHER_ARRAY
            
        for f in init_field_content_underlying_tuple_class.keys():
            if f in vanilla_room_fields:
                init_field_content_vanilla_room[f] = init_field_content_underlying_tuple_class[f]
        
        content_dict: Dict[str, Any] = self.field_contents
        # Start at the proto room level:
        vanilla_room_constructed_: Dict[str, jArray] = {}
        # Do it this way, so that from that from the constructors point of view the field construction happens concurrently. 
        # This forbids interaction between constructed fields at the vanilla room level. 
        # I may change my mind about this in the future
        for f in list(SANTAH._vanilla_room_field_constructors.keys()):
            initted_vanilla_room = SANTAH.vanilla_room(**copy.deepcopy(init_field_content_vanilla_room))
            content = SANTAH._vanilla_room_field_constructors[f](initted_vanilla_room)
            vanilla_room_constructed_[f] = content
        for f in vanilla_room_constructed_.keys():
            content_dict[f] = vanilla_room_constructed_[f]
            init_field_content_underlying_tuple_class[f] = vanilla_room_constructed_[f]
            init_field_content_vanilla_room[f] = vanilla_room_constructed_[f]
        
        # Now initialize all the fields that have been decalared as requiring initialization at the TAG level.
        constructed_tag_fields: Dict[str, jArray] = {}
        
        for tag in implemented_tags:
            if tag in SANTAH.tag_based_room_constructor_fields:
                tag_field_constructors: Dict[str, Callable[[VanillaRoom, TagNamedTuple], jArray]] = SANTAH.tag_based_room_constructor_fields[tag]
                for f_ in tag_field_constructors.keys():
                    
                    tag_nt: Type[NamedTuple] = SANTAH.my_tag_mapping[tag]
                    tag_constructor_args: Dict[str, Any] = {}
                    for f in tag_nt._fields:
                        tag_constructor_args[f] = init_field_content_underlying_tuple_class[f]
                    tag_ = tag_nt(**copy.deepcopy(tag_constructor_args))
                    vanilla_room = SANTAH.vanilla_room(**copy.deepcopy(init_field_content_vanilla_room))
                    field_content = tag_field_constructors[f_](vanilla_room, tag_)
                    constructed_tag_fields[f_] = field_content
        # Now move all the constructed tag fields into the content dict & make them available for initializing fields 
        # with constructors declared on the underlying tuple class level.
        for _c in constructed_tag_fields.keys():
            content_dict[_c] = constructed_tag_fields[_c]
            init_field_content_underlying_tuple_class[_c] = constructed_tag_fields[_c]
                    
            
        # Now construct the fields declared at the individual room level. This is a legacy feature and should not be used anyomore
        room_indiv_constructed_fields = list(SANTAH._fields_constructor[self.underlyingTupleClass].keys())
        for f in room_indiv_constructed_fields:
            cnt = SANTAH._fields_constructor[self.underlyingTupleClass][f](
                                self.underlyingTupleClass(**copy.deepcopy(init_field_content_underlying_tuple_class)))
            content_dict[f] = cnt
        
        def jittable_initialisation(content: Dict[str, Any], tuple_class: NamedTuple):
            return tuple_class(**content)
        return jax.jit(partial(jittable_initialisation, content=content_dict, tuple_class=self.underlyingTupleClass))
    
    def get_serialisation_function(self) -> Tuple[Callable[[NamedTuple], jArray], Callable[[NamedTuple, jArray], NamedTuple], int]:
        """Generates a function that takes in a NamedTuple representing a room 
           and returns a serialised array version of the persistant fields in the named tuple, 
           i.e. the fields that need to be stored to the global store. 
           Also generates the corresponding deserialisation function which takes in an initialized NamedTuple and 
           the persistence storage associated with this room and loads the values from the persistence storage into the 
           named_tuple
        Returns:
            Tuple[Callable[[NamedTuple], jArray], int]: Returns a serialisation function and and integer that gives 
                the size of all persisting fields.
        """
        # Split all serializable fields according to their type.
        # Also collect partial + full serialisation functions 
        # for all named-tuple fields.
        size: int = 0
        singleton_integer_fields: List[str] = []
        named_tuple_fields: List[str] = []
        named_tuple_partial_deserialise: List[Callable[[NamedTuple, jArray], jArray]] = []
        named_tuple_partial_serialisation: List[Callable[[NamedTuple, jArray], jArray]] = []
        named_tuple_full_serialisation: List[Callable[[NamedTuple], jArray]] = []
        named_tuple_full_deserialise: List[Callable[[NamedTuple, jArray], jArray]] = []
        named_tuple_sizes: List[int] = []
        named_tuple_stack_heights: List[int] = []
        for f in self.present_fields:
            if self.field_type[f] == NamedTupleFieldType.OTHER_ARRAY:
                continue
            elif self.field_type[f] == NamedTupleFieldType.INTEGER_SCALAR:
                if not self.field_persistent[f]:
                    continue
                else:
                    size += 1
                    singleton_integer_fields.append(f)
            elif self.field_type[f] == NamedTupleFieldType.NAMED_TUPLE_STACK:
                named_tuple_fields.append(f)
                named_tuple_partial_deserialise.append(
                    SANTAH.partial_deserialisations[self.field_named_tuples[f]])
                named_tuple_partial_serialisation.append(
                    SANTAH.partial_serialisations[self.field_named_tuples[f]])
                named_tuple_full_serialisation.append(
                    SANTAH.full_serialisations[self.field_named_tuples[f]])
                named_tuple_full_deserialise.append(
                    SANTAH.full_deserializations[self.field_named_tuples[f]])
                named_tuple_stack_heights.append(len(self.field_contents[f]))
                
                tup_size: int = SANTAH.partially_serialised_sizes[self.field_named_tuples[f]]
                named_tuple_sizes.append(tup_size)
                size += tup_size*len(self.field_contents[f])
                
                
        # The actual function used to write a room to persistence. 
        # This function essentially picks out all the attributes of the room that 
        # were declared to be serializable & concatenates them into a long 
        # array which can be written to global storage.
        def _write_self_to_persistence(_tuple: NamedTuple, _int_fields: List[str], _named_tup_fields: List[str], 
                                       _named_tup_deserialisation_function: List[Callable[[jArray], NamedTuple]], 
                                       _named_tup_partial_serialisation_fun: List[Callable[[NamedTuple], jArray]], 
                                       _named_tuple_sizes: List[int], 
                                       _named_tuple_stack_heights: List[int], 
                                       _full_size: int
                                       ):
            full_serialised_array: jArray = jnp.zeros((_full_size, ), dtype=jnp.int32)
            current_offset = 0
            for i, d in enumerate(_int_fields):
                full_serialised_array = full_serialised_array.at[i].set(getattr(_tuple, d)[0])
                current_offset += 1
            for d in zip(_named_tup_fields, 
                         _named_tup_deserialisation_function, 
                         _named_tup_partial_serialisation_fun, 
                         _named_tuple_stack_heights, 
                         _named_tuple_sizes):
                _field, _deserialise, _serialise_part, _height, _size = d
                for i in range(_height):
                    current_arr: jArray = getattr(_tuple, _field)[i, ...]
                    nt_repr: NamedTuple = _deserialise(current_arr)
                    persistence_arr: jArray = _serialise_part(nt_repr)
                    full_serialised_array = full_serialised_array.at[current_offset:current_offset+_size].set(persistence_arr)
                    current_offset += _size
            return full_serialised_array
        
        
        # The Persistence load function.
        # receives a slice of the global storage corresponding to the serialisable fields of this room
        # & reconstructs the room-named tuple from the serialised fields & default values for all other fields.
        #
        #
        def _load_self_from_persistence(_tuple: NamedTuple, _persistence_storage: jArray, _int_fields: List[str], 
                                        _named_tup_fields: List[str], 
                                       _named_tup_serialisation_function: List[Callable[[NamedTuple], jArray]], 
                                       _named_tup_partial_deserialisation_function: List[Callable[[NamedTuple], jArray]],
                                       _named_tup_full_deserialisation_function: List[Callable[[jArray], NamedTuple]], 
                                       _named_tuple_sizes: List[int], 
                                       _named_tuple_stack_heights: List[int], 
                                       _all_fields: List[str], 
                                       _tuple_class: NamedTuple
                                       ):
            field_contents: Dict[str, jArray] = {}
            for f in _all_fields:
                field_contents[f] = getattr(_tuple, f)
            
            current_offset = 0
            for i, d in enumerate(_int_fields):
                field_contents[d] = jnp.array([_persistence_storage[i]])
                current_offset += 1
            for d in zip(_named_tup_fields, 
                         _named_tup_serialisation_function, 
                         _named_tup_partial_deserialisation_function, 
                         _named_tup_full_deserialisation_function, 
                         _named_tuple_sizes, 
                         _named_tuple_stack_heights):
                _field, _serialise_fully, _deserialize_part, _deserializy_fully, _size, _height = d
                tuple_stack: jArray = field_contents[_field]
                for i in range(_height):
                    tmp = tuple_stack[i, ...]
                    curr_tup: NamedTuple = _deserializy_fully(tmp)
                    _arr_persist = _persistence_storage[current_offset: current_offset+_size]
                    curr_tup = _deserialize_part(curr_tup, _arr_persist)
                    serialised_tup: jArray = _serialise_fully(curr_tup)
                    tuple_stack = tuple_stack.at[i, ...].set(serialised_tup)
                    current_offset += _size
                field_contents[_field] = tuple_stack
            ret = _tuple_class(**field_contents)
            return ret
        
        
        wrapped_persistence_writer: Callable[[NamedTuple], jArray] = partial(
            _write_self_to_persistence, 
                _int_fields = singleton_integer_fields, _named_tup_fields = named_tuple_fields, 
                _named_tup_deserialisation_function = named_tuple_full_deserialise, 
                _named_tup_partial_serialisation_fun = named_tuple_partial_serialisation, 
                _named_tuple_sizes = named_tuple_sizes, 
                _named_tuple_stack_heights = named_tuple_stack_heights, 
                _full_size = size
        )
        wrapped_persistence_writer = jax.jit(wrapped_persistence_writer)
        
        jitted_persistence_loader: Callable[[NamedTuple, jArray], NamedTuple] = partial(
            _load_self_from_persistence, 
            _int_fields = singleton_integer_fields,
            _named_tup_fields = named_tuple_fields, 
            _named_tup_serialisation_function = named_tuple_full_serialisation, 
            _named_tup_partial_deserialisation_function = named_tuple_partial_deserialise,
            _named_tup_full_deserialisation_function = named_tuple_full_deserialise, 
            _named_tuple_sizes = named_tuple_sizes, 
            _named_tuple_stack_heights = named_tuple_stack_heights, 
            _all_fields = self.present_fields, 
            _tuple_class = self.underlyingTupleClass
        )
        jitted_persistence_loader = jax.jit(jitted_persistence_loader)
        
        return wrapped_persistence_writer, jitted_persistence_loader, size
        
        
    def connect_to(self, my_location: RoomConnectionDirections, other_room: Room, other_location: RoomConnectionDirections):
        # 
        # Generate the graph structure which describes the room layout.
        #
        if my_location == RoomConnectionDirections.LEFT:
            self.connection_object.left = other_room.connection_object
            
        if my_location == RoomConnectionDirections.RIGHT:
            self.connection_object.right = other_room.connection_object
        if my_location == RoomConnectionDirections.UP:
            self.connection_object.up = other_room.connection_object
        if my_location == RoomConnectionDirections.DOWN:
            self.connection_object.down = other_room.connection_object
        
        # Create symmetric connection
        if other_location == RoomConnectionDirections.UP:
            other_room.connection_object.up = self.connection_object
        
        if other_location == RoomConnectionDirections.DOWN:
            other_room.connection_object.down = self.connection_object
            
        if other_location == RoomConnectionDirections.LEFT:
            other_room.connection_object.left = self.connection_object
        if other_location == RoomConnectionDirections.RIGHT:
            other_room.connection_object.right = self.connection_object    
        
    

class PyramidLayout:
    """
    PyramidLayout is basically the main manager class for the Room infrastructure. 
    It keeps track of all created rooms and is in charge of generating the necessary wrappers 
    for functions that need to handle rooms of different shape & feature set.
    """
    def __init__(self):
        self.running_counter: int = 0
        self.rooms: Dict[int, Room] = {}
        self.persistence_writer: Callable[[NamedTuple, jArray], jArray] = None
        self.persistence_loader: Callable[[NamedTuple, jArray], NamedTuple] = None
        self.raise_proto_room_to_specific: Dict[int, Callable[[NamedTuple, NamedTuple], NamedTuple]]= {}
        self.lower_specific_room_to_proto_room: Dict[int, Callable[[NamedTuple], NamedTuple]] = {}       
        self.initial_persistence_storage: jArray
        
        # This is ~~ mostly ~~ just used for stashing writers and loaders so that 
        # we don't have to recompute them when we generate wrapper functions that handle room-specific functionality
        self.room_persistence_writers: Dict[int, Callable[[NamedTuple], jArray]] = {}
        self.room_persistence_loaders: Dict[int, Callable[[NamedTuple, jArray], NamedTuple]] = {}
        self.static_jitted_raising_function: Callable[[NamedTuple, jArray, int], NamedTuple] = None
        self.single_room_proto_loader: Dict[int, Callable[[jArray, jArray], NamedTuple]] = {}
        self.single_room_proto_writer: Dict[int, Callable[[jArray, NamedTuple, jArray], jArray]] = {}
    
    
    
    def create_new_room(self, tags: Tuple[Type[Enum]] = ()):
        """The main function used to construct new rooms. 
            Automatically generates all persistence infrastructure required by the framework.

        Args:
            tags (Tuple[Type[Enum]], optional): Tags representing the functionality 
                which this room should be implemented. A room can implement arbitrary many tags, 
                all infrastructure that is required to support this is generated automatically on game-startup.. Defaults to ().

        """
        if not isinstance(tags, Tuple):
            raise Exception("Tags need to be specified as a tuple.")
        roomtype: Type[NamedTuple] = SANTAH.create_room_from_tags(room_id=self.running_counter, 
                                tags=tags)
        new_room = Room(room_id=self.running_counter, underlyingTupleClass=roomtype)
        self.running_counter += 1
        self.rooms[new_room.room_id] = new_room
        # Generate lowering & raising functions necessary for the function wrappers.
        self.lower_specific_room_to_proto_room[new_room.room_id] = SANTAH.room_to_proto_room[roomtype]
        self.raise_proto_room_to_specific[new_room.room_id] = SANTAH.proto_room_to_room[roomtype]
        return new_room
    
    def _create_new_room_deprecated(self, roomtype: NamedTuple) -> Room:
        warnings.warn("DEPRECATED. Please use 'create_new_room' instead.")
        new_room = Room(room_id=self.running_counter, underlyingTupleClass=roomtype)
        self.running_counter += 1
        self.rooms[new_room.room_id] = new_room
        self.lower_specific_room_to_proto_room[new_room.room_id] = SANTAH.room_to_proto_room[roomtype]
        self.raise_proto_room_to_specific[new_room.room_id] = SANTAH.proto_room_to_room[roomtype]
        return new_room
    
    def _get_non_jittable_room_ID_type_mapping(self) -> Callable[[int], Type[NamedTuple]]:
        #
        # This is only ever used internally and should not be called from the outside
        #
        mapping: Dict[str, Type[NamedTuple]] = {}
        for k in list(self.rooms.keys()):
            mapping[k] = self.rooms[k].underlyingTupleClass
        mapping_lambda = lambda room_id, map=mapping : map[room_id]
        return mapping_lambda
         
    
    def _create_initial_persistence_state(self) -> jArray:
        # Also only for internal use. This function generates the persistence state (constant & fixed shape)
        # which is used to store all editable room-attributes.
        # The rows in the persistence state represent individual rooms.
        # The per-row layout is specific to each room and depends on the alphabetical order of the fields.
        
        room_ids: List[int] = []
        room_persistence_writers: List[Callable[[NamedTuple], jArray]] = []
        room_constructors: List[Callable[[], jArray]] = []
        persistence_lenghts: List[int] = []
        room_ids = list(self.rooms.keys())
        room_ids = sorted(room_ids)
        
        # Find out the maximum number of persisted fields used by all rooms
        # to determine the shape of the persistence state.
        for id in room_ids:

            my_room: Room = self.rooms[id]
            initial_constructor = my_room.get_jitted_room_constructor()
            room_constructors.append(initial_constructor)
            persistence_writer, persistence_loader, persistence_length = my_room.get_serialisation_function()
            room_persistence_writers.append(persistence_writer)
            persistence_lenghts.append(persistence_length)
        num_rooms: int = len(room_ids)

        max_pers_length: int = max(persistence_lenghts)
        persistence_state: jArray = jnp.zeros((num_rooms, max_pers_length), dtype=jnp.int32)

        
        # piece together the individual rows
        # to generate the final persistence state.
        for d in zip(room_constructors, room_persistence_writers, persistence_lenghts, room_ids):
            _const, _persistence, _len, _id = d
            tmp = _const()
            tmp = _persistence(tmp)
            persistence_state = persistence_state.at[_id, :_len].set(tmp)
        
        return persistence_state
    
    
    def _wrap_lowered_function(self, lowered_function: Callable[[NamedTuple, NamedTuple, Type[NamedTuple]], NamedTuple], 
                                    montezuma_state_type: Type ) -> Callable[[NamedTuple], str]: 
        
        """This function generates wrappers around functions that directly interact with the specific (non-proto) room state. 
            It automatically wraps functions in the raising & lowering infrastructure necessary for this to work in jitted code. 
            This incurs overhead linear in the number of total rooms, so only use sparingly. 

        Args:
            lowered_function (Callable[[NamedTuple, NamedTuple, Type[NamedTuple]], NamedTuple]): A function 
                That receives a MontezumaState (kwarg: montezuma_state)
                                The montezuma state always also contains a generic ProtoRoom -- ALL CHANGES MADE TO THIS ROOM ARE DISCARDED 
                              a Specific room state (kwarg: room_state)
                              and a type representing the class of the underlying room, 
                              as well as a tuple containing all tags implemented by this room.
                              The tags should be used to determine the room-specific behavior of the wrapped function. 
                    and returns a tuple consisting of the MontezumaState, and the UNLOWERED ROOM STATE. 

        Returns:
            Callable[[NamedTuple], NamedTuple]: A function that takes in the montezumastate, applies your lowered function
                and then again returns a Montezuma state including the raised Proto-State. 
        """
        # Check the function signature of the function to be wrapped.
        if set(self.room_persistence_loaders.keys()) != set(self.rooms.keys()) or set(self.room_persistence_writers.keys()) != set(self.rooms.keys()):
            raise Exception("Persistence writers & loaders for all rooms have not been created yet. This is likely because you haven't called the 'create_proto_room_specific_persistence_infrastructure' function yet.")
        arg_names = inspect.getfullargspec(lowered_function)[0]
        if set(arg_names) != set(["montezuma_state", "room_state", "room_type", "tags"]) and set(arg_names) != set(["montezuma_state", "room_state", "room_type", "tags", "self"]):
            raise Exception('The wrapped function may only receive the attributes ["montezuma_state", "room_state", "room_type", "tags"]')

        
        def wrapped_call_function(montezuma_state: NamedTuple, call_function: Callable[[MontezumaNamedTuple, RoomNamedTuple, Type[RoomNamedTuple], Tuple[TagEnum]], Tuple[MontezumaNamedTuple, RoomNamedTuple]],
                                                        room_id: int, raising_function: Callable[[NamedTuple, jArray, int], NamedTuple], 
                                                        _single_room_persistence_writer: Callable[[NamedTuple], jArray], 
                                                        _single_room_proto_loader: Callable[[jArray, jArray], NamedTuple],
                                                        room_named_tuple_retriever: Callable[[int], Type[NamedTuple]],
                                                        montezuma_state_type: Type, 
                                                        tags: Tuple[TagEnum]) -> NamedTuple:
            """Internal function that wraps arount the function to be actually called. Takes in a montezumastate, applies 
                    the call_function operating on the raised RoomState, writes the raised room state back onto the persistence 
                    state, converts back to ProtoRoom and updates the proto_room held by the montezuma_state.

            Args:
                montezuma_state (NamedTuple): Input1: Montezuma state
                call_function (Callable[[NamedTuple, NamedTuple, int], Tuple[NamedTuple, NamedTuple]]):  
                    Call function: Receives & returns a Tuple consisting of (MontezumaState, RaisedRoomState
                room_id (int): (basically static) ID of the room for which this function is called.
                raising_function (Callable[[NamedTuple, jArray, int], NamedTuple]): A general raising function, 
                    that takes in a regular proto_room, the persistenc storage and a (static) room id and returns the raised room.
                _single_room_persistence_writer (Callable[[jArray, NamedTuple, jArray], jArray]): Transforms the raised room into it's 
                    storage row.
                _single_room_proto_loader (Callable[[jArray, jArray], NamedTuple]): Takes in the room_id (non-static), and the persistence state
                    and returns the loaded proto room
                montezuma_state_type (Type): Type of the montezumastate. Basically just used 
                    in order to access some utility functions.

            Returns:
                NamedTuple: Returns the montezumastate with updated proto-room & storage.
            """
            # raise the give proto room to the actual room-state using the persistence state
            proto_room: NamedTuple = getattr(montezuma_state, MONTEZUMA_STATE_ROOM_FIELD)
            persistence_storage: jArray = getattr(montezuma_state, MONTEZUMA_PERSISTENCE_STORAGE)
            raised_room: NamedTuple = raising_function(proto_room, persistence_storage, room_id)
            room_type: Type[NamedTuple] = room_named_tuple_retriever(room_id)
            # Call the wrapped function
            montezuma_state, raised_room = call_function(montezuma_state, raised_room, room_type, tags)
            
            # Now write back the updated roomstate to the persistence state.
            write_data = _single_room_persistence_writer(raised_room)
            write_data = jnp.expand_dims(a=write_data, axis=0)
            persistence_storage = jax.lax.dynamic_update_slice(persistence_storage, write_data, start_indices=(room_id, 0))
            
            # Now reload the proto room from the persistence storage, so the updates to the proto room are included in the 
            # returned montezuma state.
            proto_room =  _single_room_proto_loader(jnp.array([room_id], dtype=jnp.uint16), persistence_storage)
            montezuma_state = SANTAH.attribute_setters[montezuma_state_type][MONTEZUMA_STATE_ROOM_FIELD](montezuma_state, proto_room)
            montezuma_state = SANTAH.attribute_setters[montezuma_state_type][MONTEZUMA_PERSISTENCE_STORAGE](montezuma_state, persistence_storage)
            return montezuma_state
        
        
        # Generate precompiled wrapped functions for each room. 
        # This is strictly necessary to interact with rooms that have different feature sets.
        # If we didn't automatically handle all the case distinctions, 
        # we would have needed to handle them in the code. 
        # There sadly is no way around massive switch/ case statements when dealing 
        # with a large number of rooms with different features.
        # Handling the case distinctions automatically only increases readability & makes the code 
        # significantly less brittle.
        wrapped_functions = []
        room_ids = list(self.rooms.keys())
        room_ids = sorted(room_ids)
        room_type_getter: Callable[[int], Type[NamedTuple]] = self._get_non_jittable_room_ID_type_mapping()
        for id in room_ids:
            room_type: Type[RoomNamedTuple] = self.rooms[id].underlyingTupleClass
            room_tags: Tuple[TagEnum] = SANTAH.room_tags[room_type]
            w_func = partial(wrapped_call_function, 
                    call_function = lowered_function,
                    room_id=id, 
                    raising_function = self.static_jitted_raising_function, 
                    _single_room_persistence_writer = self.room_persistence_writers[id], 
                    _single_room_proto_loader = self.single_room_proto_loader[id],
                    room_named_tuple_retriever = room_type_getter,
                    montezuma_state_type = montezuma_state_type, 
                    tags=room_tags)
            wrapped_functions.append(w_func)
        
        # The final wrapper function. 
        # Handle the case distinction between different rooms via a single, large switch-case statement.
        def wrapper_function(montezuma_state: NamedTuple, wrapped_functions: List[Callable[[NamedTuple], NamedTuple]]) -> NamedTuple:
            room_state = getattr(montezuma_state, MONTEZUMA_STATE_ROOM_FIELD)
            room_id: jnp.ndarray = getattr(room_state, ROOM_ID_FIELD)
            montezuma_state = jax.lax.switch(room_id[0], wrapped_functions, montezuma_state)
            return montezuma_state
        
        final_ret_function = partial(wrapper_function, wrapped_functions=wrapped_functions)
        final_ret_function = jax.jit(final_ret_function)
        return final_ret_function
        
        
        
    def create_proto_room_specific_persistence_infrastructure(self) -> Any:
        #
        # Generates all the persistence infrastructure necessary to write & load proto rooms
        # to/ from storage.
        #
        
        # Again, collect all necessary writers/ readers to genereate proto room infrastructure.
        room_ids: List[int] = []
        room_persistence_writers: List[Callable[[NamedTuple], jArray]] = []
        room_persistence_loaders: List[Callable[[NamedTuple, jArray], NamedTuple]] = []
        raising_function: List[Callable[[NamedTuple, NamedTuple], NamedTuple]] = []
        lowering_functions: List[Callable[[NamedTuple], NamedTuple]] = []
        room_constructors: List[Callable[[], jArray]] = []
        persistence_lenghts: List[int] = []
        room_ids = list(self.rooms.keys())
        room_ids = sorted(room_ids)
        for id in room_ids:
            lowering_functions.append(self.lower_specific_room_to_proto_room[id])
            raising_function.append(self.raise_proto_room_to_specific[id])
            my_room: Room = self.rooms[id]
            initial_constructor = my_room.get_jitted_room_constructor()
            room_constructors.append(initial_constructor)
            persistence_writer, persistence_loader, persistence_length = my_room.get_serialisation_function()
            room_persistence_writers.append(persistence_writer)
            room_persistence_loaders.append(persistence_loader)
            # Stash them as well for later use
            self.room_persistence_loaders[id] = persistence_loader
            self.room_persistence_writers[id] = persistence_writer
            persistence_lenghts.append(persistence_length)
        num_rooms: int = len(room_ids)

        max_pers_length: int = max(persistence_lenghts)
        persistence_state: jArray = jnp.zeros((num_rooms, max_pers_length), dtype=jnp.int32)

        
        # At this point create the initial global state:
        for d in zip(room_constructors, room_persistence_writers, persistence_lenghts, room_ids):
            _const, _persistence, _len, _id = d
            tmp = _const()
            tmp = _persistence(tmp)
            persistence_state = persistence_state.at[_id, :_len].set(tmp)
        


        # Function that loads the PROTO_ROOM version of a given room from storage.
        # This is mainly needed in the game_reset.        
        def _jittable_singleton_load_proto_from_persistence(room_id: jArray, storage: jArray,  room_creation_function: Callable[[], NamedTuple],  
                                             room_reload_function: Callable[[NamedTuple, jArray], NamedTuple], 
                                             proto_lowering_function: Callable[[NamedTuple], NamedTuple]) -> NamedTuple:
            initial_room: NamedTuple = room_creation_function()
            correct_row: jArray = jax.lax.dynamic_index_in_dim(operand=storage, index=room_id[0], axis=0, keepdims=False)
            reloaded_room = room_reload_function(initial_room, correct_row)
            # Lower to proto room and only return proto room.
            reloaded_room = proto_lowering_function(reloaded_room)
            return reloaded_room
        
        def _jittable_univ_proto_loader(room_id: jArray, storage: jArray, singleton_proto_loaders: List[Callable[[jArray, jArray], jArray]]) -> NamedTuple:
            proto_room: NamedTuple = jax.lax.switch(room_id[0], singleton_proto_loaders, room_id, storage)
            return proto_room
        
        
        # Only for internal use, best not touch.
        #
        def _jittable_raise_proto_with_persistance(proto_room: NamedTuple, storage: jArray, 
                                                
                                            static_room_id: int,
                                            room_creation_functions: List[Callable[[], NamedTuple]],  
                                            room_reload_functions: List[Callable[[NamedTuple, jArray], NamedTuple]], 
                                            proto_raising_functions: Callable[[jArray], jArray] 
                                ):
            initial_room: NamedTuple = room_creation_functions[static_room_id]()
            correct_row: jArray = jax.lax.dynamic_index_in_dim(operand=storage, index=static_room_id, axis=0, keepdims=False)
            reloaded_room = room_reload_functions[static_room_id](initial_room, correct_row)
            raised_room = proto_raising_functions[static_room_id](reloaded_room, proto_room)
            return raised_room
        
        raise_proto_function_with_persistence = partial(_jittable_raise_proto_with_persistance, 
                                                        room_creation_functions = room_constructors, 
                                            room_reload_functions = room_persistence_loaders, 
                                            proto_raising_functions = raising_function 
                                            )
        raise_proto_function_with_persistence = jax.jit(raise_proto_function_with_persistence, static_argnames=["static_room_id"])
        self.static_jitted_raising_function = raise_proto_function_with_persistence
        
        
        def _jittable_singleton_write_proto_to_storage(room_id: jArray, proto_room: NamedTuple, _storage: jArray, 
                        _room_persistance_writer: Callable[[NamedTuple], jArray], 
                        raise_proto_with_pers: Callable[[NamedTuple, jArray, int], NamedTuple], 
                        static_room_id: int) -> jArray:
            
            raised_room = raise_proto_with_pers(proto_room, _storage, static_room_id)
            
            write_data = _room_persistance_writer(raised_room)
            write_data = jnp.expand_dims(a=write_data, axis=0)
            _storage = jax.lax.dynamic_update_slice(_storage, write_data, start_indices=(room_id[0].astype(jnp.int32), 0))
            return _storage
        
        def _jittable_univ_proto_writer(room_id: jArray, proto_room: NamedTuple, storage: jArray, 
                                        singleton_proto_writers: List[Callable[[jArray, NamedTuple, jArray], jArray]]) -> jArray:
            storage = jax.lax.switch(room_id[0], singleton_proto_writers, room_id, proto_room, storage)
            return storage
        
        persistence_lenghts = jnp.array(persistence_lenghts)   
        
        # Construct the Proto Loaders:
        singleton_loaders = []
        for count, r_id in enumerate(room_ids):
            _loader = partial(_jittable_singleton_load_proto_from_persistence, 
                            room_creation_function = room_constructors[count],  
                            room_reload_function = room_persistence_loaders[count], 
                            proto_lowering_function=lowering_functions[count])
            jitted_loader = jax.jit(_loader)
            self.single_room_proto_loader[r_id] = jitted_loader
            singleton_loaders.append(_loader)
        load_proto_from_persistence = partial(_jittable_univ_proto_loader, 
                                singleton_proto_loaders = singleton_loaders
                                              )
        load_proto_from_persistence = jax.jit(load_proto_from_persistence)
        
        
        
        
        singleton_proto_writers = []
        for count, r_id in enumerate(room_ids):
            _proto_writer = partial(_jittable_singleton_write_proto_to_storage, 
                        _room_persistance_writer = room_persistence_writers[count], 
                        raise_proto_with_pers = raise_proto_function_with_persistence, 
                        static_room_id = count
                                    )
            jtted_proto_writer = jax.jit(_proto_writer)
            self.single_room_proto_writer[r_id] = jtted_proto_writer
            singleton_proto_writers.append(_proto_writer)
        write_proto_to_storage = partial(_jittable_univ_proto_writer, 
                                         singleton_proto_writers = singleton_proto_writers
                                         )
        write_proto_to_storage = jax.jit(write_proto_to_storage)
        
               
        return load_proto_from_persistence, write_proto_to_storage
        

    
    
    def _get_connection_report(self) -> Dict[int, Dict[RoomConnectionDirections, Tuple[int, RoomConnectionDirections]]]:
        #
        # Recursively construct a connection map of all rooms.
        #
        connection_dir: Dict[int, Dict[RoomConnectionDirections, Tuple[int, RoomConnectionDirections]]] = {}
        for k in self.rooms.keys():
            connection_dir = self.rooms[k].connection_object.visit(connection_dir)
        
        return connection_dir
    
    
    def get_jitted_room_connection_map(self) -> Callable[[jArray, jArray], jArray]:
        """Generates a jitted connection map between all rooms.

        Returns:
            Callable[[jArray, jArray], jArray]: returns the following function: 
                def fun(room_id: jArray, direction: jArray) -> neighboring_room_id:
                    with:
                    room_id: the room_id that corresponds to the room for which we are looking for neighbors
                    direction: the direction in which we want to querry the neighbors. Singleton integer array with value 
                        equal to the Enum, i.e. RoomConnectionDirections.LEFT.value
                    returns:
                        the integer ID of the room which is connected at that direction. If no room is connected at that direction: 
                            returns -1 as a default value
        """
        # Find out how many directions there are (in theory this would support connectin rooms at the corners as well.)
        direction_values: List[int] = list(map(lambda x: x.value, RoomConnectionDirections._member_map_.values()))
        room_connection_arr_size: int = max(direction_values)
        
        connection_report: Dict[int, Dict[RoomConnectionDirections, Tuple[int, RoomConnectionDirections]]] = self._get_connection_report()

        # Determine the size of our connection map.
        # To support non-euclidean geometry in level layout, 
        # we compute seperate maps for connected rooms & connection direction.
        num_rooms: int = len(list(connection_report.keys()))
        connection_map: jnp.ndarray = jnp.zeros((num_rooms, room_connection_arr_size+1), dtype=jnp.int32)
        connection_map = connection_map - 1
        
        # Direction map
        direction_map: jnp.ndarray = jnp.zeros((num_rooms, room_connection_arr_size+1), dtype=jnp.int32)
        direction_map = direction_map - 1
        
        # Look at the generated connection report & set the appropriate entries in the connection_map & direction_map array.
        for room_ in connection_report.keys():
            for direction_ in connection_report[room_].keys():
                connection_map = connection_map.at[room_, direction_.value].set(connection_report[room_][direction_][0])
                direction_map = direction_map.at[room_, direction_.value].set(connection_report[room_][direction_][1])
              
        def get_connection_map(room_id: jnp.ndarray, direction_id: jnp.ndarray, connection_map: jnp.ndarray, direction_connection_map: jnp.ndarray) -> jnp.ndarray:
            """Jittable connection map. Gives ID of the room as well as the Direction at which the room is supposed to be entered. 
            
            Args:
                room_id (jnp.ndarray): The room in which we are currently in, and might want to leave
                direction_id (jnp.ndarray): The direction we are interested in leaving the room in
                connection_map (jnp.ndarray): The connection map array.
                direction_connection_map (jnp.ndarray): The Direction connection map. Same shape as the connection map array.
                
            Returns:
                jnp.ndarray, jnp.ndarray: Returns the Room connected room in the given direction, and the direction we would be entering the room in.
                In both cases returns -1 if there is no room connected. 
            """
            new_room = connection_map[room_id, direction_id]
            entering_direction = direction_connection_map[room_id, direction_id]
            return new_room, entering_direction
        
        ret_func = partial(get_connection_map, connection_map=connection_map, direction_connection_map=direction_map)
        ret_func = jax.jit(ret_func)
        return ret_func
        



def loadFrameAddAlpha(fileName, transpose=True, add_alpha: bool = False, add_black_as_transparent: bool = False):
    # Custom loading function which turns black background transparent.
    # This is simply to make editing sprites a bit more convenient.
    frame = jnp.load(fileName)
    if frame.shape[-1] != 4 and add_alpha:
        alphas = jnp.ones((*frame.shape[:-1], 1))
        alphas = alphas*255
        frame = jnp.concatenate([frame, alphas], axis=-1)
        if add_black_as_transparent:
            arr_black = jnp.sum(frame[..., :-1], axis=-1)
            alpha_channel = frame[..., -1]
            alpha_channel = alpha_channel.at[arr_black == 0].set(0)
            frame = frame.at[..., -1].set(alpha_channel)
    # Check if the frame's shape is [[[r, g, b, a], ...], ...]
    frame = jnp.astype(frame, jnp.uint8)
    if frame.ndim != 3:
        raise ValueError(
            "Invalid frame format. The frame must have a shape of (height, width, 4)."
        )
    return jnp.transpose(frame, (1, 0, 2)) if transpose else frame


def load_collision_map(fileName, transpose=True, as_bool: bool = True):
    # Returns a boolean array representing the collision map
    # Load frame (np array) from a .npy file and convert to jnp array
    frame = jnp.load(fileName)
    frame = frame[..., 0].squeeze()
    boolean_frame = jnp.zeros(shape=frame.shape, dtype=jnp.bool)
    boolean_frame = boolean_frame.at[frame==0].set(False)
    boolean_frame = boolean_frame.at[frame > 0].set(True)
    frame = boolean_frame
    if not as_bool:
        boolean_frame = jnp.astype(boolean_frame, jnp.uint8)
    return jnp.transpose(frame, (1, 0)) if transpose else frame