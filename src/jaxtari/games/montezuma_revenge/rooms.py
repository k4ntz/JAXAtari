import jax
import jax.numpy as jnp
from flax import struct

from jaxtari.games.montezuma_revenge.core import MontezumaRevengeConstants, MontezumaRevengeState, get_room_idx

_DEFAULT_CONSTS = MontezumaRevengeConstants()


@struct.dataclass
class RoomConfig:
    enemies_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    )
    enemies_y: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    )
    enemies_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    )
    enemies_direction: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    )
    enemies_min_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    )
    enemies_max_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.full(
            _DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM,
            _DEFAULT_CONSTS.WIDTH - 8,
            dtype=jnp.int32,
        )
    )
    enemies_bouncing: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    )
    ladders_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    )
    ladders_top: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    )
    ladders_bottom: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    )
    ladders_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    )
    ropes_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    )
    ropes_top: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    )
    ropes_bottom: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    )
    ropes_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    )
    items_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
    )
    items_y: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
    )
    items_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
    )
    doors_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
    )
    doors_y: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
    )
    doors_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
    )
    conveyors_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    )
    conveyors_y: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    )
    conveyors_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    )
    conveyors_direction: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    )
    lasers_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_LASERS_PER_ROOM, dtype=jnp.int32)
    )
    lasers_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_LASERS_PER_ROOM, dtype=jnp.int32)
    )
    platforms_x: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32)
    )
    platforms_y: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32)
    )
    platforms_width: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32)
    )
    platforms_active: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(_DEFAULT_CONSTS.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32)
    )

# Native room metadata sampled from the ROM's three temple layouts.
# Columns: native entity type, horizontal anchor, vertical anchor, duplicate bits.
# Room architecture is shared; contents and collectible histories reset per temple.
LEVEL_CONTENTS = jnp.array([
    [  # Temple 1; native room order 0..23.
        [1, 25, 254, 0], [4, 14, 254, 0], [5, 116, 254, 1], [5, 60, 254, 1],
        [10, 90, 226, 0], [6, 78, 254, 0], [2, 12, 254, 0], [4, 130, 254, 0],
        [4, 78, 254, 0], [8, 18, 228, 2], [1, 16, 254, 0], [8, 44, 228, 4],
        [0, 80, 254, 0], [10, 100, 226, 0], [4, 77, 201, 0], [1, 20, 254, 0],
        [0, 80, 254, 0], [0, 80, 254, 0], [0, 60, 228, 0], [3, 16, 254, 0],
        [1, 76, 254, 4], [10, 60, 226, 0], [8, 30, 228, 0], [1, 100, 254, 3],
    ],
    [  # Temple 2; native room order 0..23.
        [2, 25, 254, 0], [4, 14, 254, 0], [1, 132, 254, 0], [1, 12, 254, 1],
        [8, 30, 228, 0], [6, 78, 254, 0], [5, 12, 254, 0], [1, 130, 254, 0],
        [1, 61, 254, 2], [10, 40, 226, 0], [4, 16, 254, 0], [3, 16, 254, 0],
        [0, 80, 254, 0], [8, 44, 228, 4], [4, 77, 254, 0], [1, 20, 254, 0],
        [0, 80, 254, 0], [0, 80, 254, 0], [0, 60, 228, 0], [5, 46, 254, 1],
        [1, 61, 254, 3], [4, 16, 254, 0], [10, 30, 226, 0], [1, 116, 254, 1],
    ],
    [  # Temple 3; native room order 0..23.
        [5, 70, 254, 0], [4, 14, 254, 0], [8, 18, 228, 2], [1, 44, 254, 4],
        [8, 18, 228, 2], [6, 78, 254, 0], [10, 130, 226, 0], [4, 130, 254, 0],
        [4, 20, 254, 0], [10, 50, 226, 0], [2, 16, 254, 0], [5, 136, 254, 1],
        [0, 80, 254, 0], [5, 120, 254, 0], [3, 77, 201, 0], [1, 20, 254, 0],
        [0, 80, 254, 0], [0, 80, 254, 0], [0, 60, 228, 0], [10, 30, 226, 0],
        [4, 138, 254, 0], [8, 100, 228, 0], [5, 40, 254, 1], [1, 100, 254, 3],
    ],
], dtype=jnp.int32)
ALE_ROOM_IDS = jnp.array([3,4,5,10,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], dtype=jnp.int32)
# Native exit bits: up, down, left, right (ROM BBCA).
NATIVE_EXIT_MASKS = jnp.array([10,12,6,10,15,14,13,6,8,5,9,15,12,15,6,0,12,12,12,13,4,9,13,4], dtype=jnp.int32)
_PACKED_ROOM_HISTORY = jnp.array([
    [31,51,31,17,21,25,1,17,12,33,145,23],
    [31,19,31,17,81,17,9,17,12,35,113,19],
    [31,89,95,17,17,19,1,17,12,33,17,55],
], dtype=jnp.int32)
NATIVE_ROOM_HISTORY = jnp.stack([_PACKED_ROOM_HISTORY >> 4,
    _PACKED_ROOM_HISTORY & 15], axis=2).reshape(3,24)

# ROM metadata addresses the remaining entities through a four-bit room
# history. Removing one member of a group changes its anchor and spacing.
_DUPLICATES_BY_MASK = jnp.array([0,0,0,1,0,2,1,3,0,4,1,4,5,5,10,6], dtype=jnp.int32)
_ANCHOR_INDEX_BY_MASK = jnp.array([0,0,1,0,2,0,1,0,3,0,0,0,0,0,0,0], dtype=jnp.int32)
_THIRD_PRIMARY_X = jnp.array([70,14,18,44,18,78,130,130,20,50,16,120,80,120,77,20,80,80,60,30,138,100,24,100], dtype=jnp.int32)
_THIRD_GROUP_ROOMS = jnp.array([2,3,4,11,22,23], dtype=jnp.int32)
_THIRD_GROUP_X = jnp.array([[18,0,50,0],[44,0,0,108],[18,0,50,0],
    [120,136,0,0],[24,40,0,0],[100,116,132,0]], dtype=jnp.int32)


def initialize_level(state, consts):
    contents = LEVEL_CONTENTS[jnp.minimum(state.level, 2)]
    native_type, _, _, duplicates = contents.T
    active = jnp.stack([jnp.ones(24, dtype=jnp.int32),
        (duplicates & 5 != 0).astype(jnp.int32),
        (duplicates & 2 != 0).astype(jnp.int32)], axis=1)
    item = jnp.isin(native_type, jnp.array([1,2,3,4,6]))
    enemy = jnp.isin(native_type, jnp.array([5,8,10]))
    item_types = jnp.array([0,1,3,2,0,0,4,0,0,0,0])[native_type]
    enemy_types = jnp.where(native_type == 8, 4, jnp.where(native_type == 10, 3, 1))
    items = jnp.zeros_like(state.global_items_active).at[ALE_ROOM_IDS].set(active * item[:,None])
    enemies = jnp.zeros_like(state.global_enemies_active).at[ALE_ROOM_IDS].set(active * enemy[:,None])
    # Both barrier chambers have a rolling skull alongside their collectible.
    enemies = enemies.at[jnp.array([4,12]),0].set(1)
    enemies = enemies.at[27,0].set(jnp.where(state.level == 2, 1, enemies[27,0]))
    doors = jnp.zeros_like(state.global_doors_active).at[jnp.array([4,12,26])].set(1)
    return state.replace(native_room_history=NATIVE_ROOM_HISTORY[jnp.minimum(state.level, 2)],
        global_items_active=items,
        global_items_type=jnp.zeros_like(state.global_items_type).at[ALE_ROOM_IDS].set(jnp.broadcast_to(item_types[:,None], active.shape)),
        global_enemies_active=enemies,
        global_enemies_type=jnp.zeros_like(state.global_enemies_type).at[ALE_ROOM_IDS].set(jnp.broadcast_to(enemy_types[:,None], active.shape)).at[jnp.array([4,12]),0].set(1),
        global_doors_active=doors)


def load_room(room_id: jnp.ndarray, state: MontezumaRevengeState, consts: MontezumaRevengeConstants) -> MontezumaRevengeState:
    config = RoomConfig()

    def load_room_0_3(config: RoomConfig) -> RoomConfig:
        # New 3 was Old 4
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active
        lax = config.lasers_x
        laa = config.lasers_active

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(50)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)
        
        ix = ix.at[0].set(24)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)
        
        lax = lax.at[0].set(16)
        lax = lax.at[1].set(36)
        lax = lax.at[2].set(44)
        lax = lax.at[3].set(112)
        lax = lax.at[4].set(120)
        lax = lax.at[5].set(140)
        laa = laa.at[0:6].set(1)

        return config.replace(
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            items_x=ix,
            items_y=iy,
            items_active=ia,
            lasers_x=lax,
            lasers_active=laa,
        )

    def load_room_0_4(config: RoomConfig) -> RoomConfig:
        # New 4 was Old 5
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active
        ex = config.enemies_x.at[0].set(90)
        ey = config.enemies_y.at[0].set(119)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        # Reversal attempts hold the last legal pixel for one movement tick.
        # ALE's first-room skull visits x=54..104, including both endpoints.
        eminx = config.enemies_min_x.at[0].set(53)
        emaxx = config.enemies_max_x.at[0].set(105)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(50)
        lb = lb.at[0].set(88)
        la = la.at[0].set(1)
        lx = lx.at[1].set(128)
        lt = lt.at[1].set(93)
        lb = lb.at[1].set(130)
        la = la.at[1].set(1)
        lx = lx.at[2].set(16)
        lt = lt.at[2].set(93)
        lb = lb.at[2].set(130)
        la = la.at[2].set(1)

        rx = config.ropes_x.at[0].set(112)
        rt = config.ropes_top.at[0].set(49)
        rb = config.ropes_bottom.at[0].set(88)
        ra = config.ropes_active.at[0].set(1)

        ix = ix.at[0].set(13)
        iy = iy.at[0].set(52)
        ia = ia.at[0].set(1)

        cx = config.conveyors_x.at[0].set(60)
        cy = config.conveyors_y.at[0].set(89)
        ca = config.conveyors_active.at[0].set(1)
        cd = config.conveyors_direction.at[0].set(-1)
        
        dx = config.doors_x.at[0].set(16)
        dy = config.doors_y.at[0].set(7)
        da = config.doors_active.at[0].set(1)
        dx = dx.at[1].set(140)
        dy = dy.at[1].set(7)
        da = da.at[1].set(1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            ropes_x=rx,
            ropes_top=rt,
            ropes_bottom=rb,
            ropes_active=ra,
            items_x=ix,
            items_y=iy,
            items_active=ia,
            doors_x=dx,
            doors_y=dy,
            doors_active=da,
            conveyors_x=cx,
            conveyors_y=cy,
            conveyors_active=ca,
            conveyors_direction=cd,
        )

    def load_room_0_5(config: RoomConfig) -> RoomConfig:
        # New 5 was Old 3
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        # Native pair: fixed +16 duplicate; base top79, entry apex54.
        ex = config.enemies_x.at[0].set(115)
        ey = config.enemies_y.at[0].set(32)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(4)
        emaxx = config.enemies_max_x.at[0].set(132)

        ex = ex.at[1].set(131)
        ey = ey.at[1].set(32)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(-1)
        eminx = eminx.at[1].set(20)
        emaxx = emaxx.at[1].set(148)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(50)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)

        eb = config.enemies_bouncing.at[0].set(1)
        eb = eb.at[1].set(1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            enemies_bouncing=eb,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )
    
    def load_room_1_3(config: RoomConfig) -> RoomConfig:
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        ex = config.enemies_x.at[0].set(89)
        ey = config.enemies_y.at[0].set(35)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(4)
        emaxx = config.enemies_max_x.at[0].set(156)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(50)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)
        lx = lx.at[1].set(72)
        lt = lt.at[1].set(6)
        lb = lb.at[1].set(46)
        la = la.at[1].set(1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )

    def load_room_1_2(config: RoomConfig) -> RoomConfig:
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        ex = config.enemies_x.at[0].set(59)
        ey = config.enemies_y.at[0].set(32)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(1)
        eminx = config.enemies_min_x.at[0].set(8)
        emaxx = config.enemies_max_x.at[0].set(155)

        ex = ex.at[1].set(75)
        ey = ey.at[1].set(32)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(1)
        eminx = eminx.at[1].set(8)
        emaxx = emaxx.at[1].set(155)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(50)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)

        eb = config.enemies_bouncing.at[0].set(1)
        eb = eb.at[1].set(1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            enemies_bouncing=eb,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )

    def load_room_1_4(config: RoomConfig) -> RoomConfig:
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active

        ex = config.enemies_x.at[0].set(91)
        ey = config.enemies_y.at[0].set(71)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(54)
        emaxx = config.enemies_max_x.at[0].set(106)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(128)
        lb = lb.at[0].set(150)
        la = la.at[0].set(1)

        rx = config.ropes_x.at[0].set(41)
        rt = config.ropes_top.at[0].set(50)
        rb = config.ropes_bottom.at[0].set(75)
        ra = config.ropes_active.at[0].set(1)

        rx = rx.at[1].set(125)
        rt = rt.at[1].set(49)
        rb = rb.at[1].set(100)
        ra = ra.at[1].set(1)

        # item: torch
        ix = ix.at[0].set(77)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        # doors
        dx = config.doors_x.at[0].set(56)
        dy = config.doors_y.at[0].set(86)
        da = config.doors_active.at[0].set(1)

        dx = dx.at[1].set(100)
        dy = dy.at[1].set(86)
        da = da.at[1].set(1)

        # conveyor
        cx = config.conveyors_x.at[0].set(60)
        cy = config.conveyors_y.at[0].set(46)
        ca = config.conveyors_active.at[0].set(1)
        cd = config.conveyors_direction.at[0].set(-1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            ropes_x=rx,
            ropes_top=rt,
            ropes_bottom=rb,
            ropes_active=ra,
            items_x=ix,
            items_y=iy,
            items_active=ia,
            doors_x=dx,
            doors_y=dy,
            doors_active=da,
            conveyors_x=cx,
            conveyors_y=cy,
            conveyors_active=ca,
            conveyors_direction=cd,
        )

    def load_room_1_5(config: RoomConfig) -> RoomConfig:
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active

        # Sword room from Montezuma1 (ROOM_1_3)
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(47)
        la = la.at[0].set(1)

        # Native RAM44 is one past the visible object anchor.
        # Keep the sword's narrow collision bitmap at that same position.
        ix = ix.at[0].set(11)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        return config.replace(
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            items_x=ix,
            items_y=iy,
            items_active=ia,
        )

    def load_room_1_6(config: RoomConfig) -> RoomConfig:
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active
        lax = config.lasers_x
        laa = config.lasers_active

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(50)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)

        ix = ix.at[0].set(128)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        lax = lax.at[0].set(16)
        lax = lax.at[1].set(36)
        lax = lax.at[2].set(44)
        lax = lax.at[3].set(112)
        lax = lax.at[4].set(120)
        lax = lax.at[5].set(140)
        laa = laa.at[0:6].set(1)

        return config.replace(
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            items_x=ix,
            items_y=iy,
            items_active=ia,
            lasers_x=lax,
            lasers_active=laa,
        )

    def load_room_2_2(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_1 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(47)
        la = la.at[0].set(1)

        ex = config.enemies_x.at[0].set(18)
        ey = config.enemies_y.at[0].set(33) # Floor is at 48, snake height is 13. 48-13=35
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(0) # Static snake
        eminx = config.enemies_min_x.at[0].set(18)
        emaxx = config.enemies_max_x.at[0].set(47)

        ex = ex.at[1].set(50)
        ey = ey.at[1].set(33)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(0) # Static snake
        eminx = eminx.at[1].set(50)
        emaxx = emaxx.at[1].set(100) # Assuming some right boundary

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )

    def load_room_2_1(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_0 in M1
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active
        px = config.platforms_x
        py = config.platforms_y
        pw = config.platforms_width
        pa = config.platforms_active

        # item: key
        ix = ix.at[0].set(77)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        # rope
        rx = config.ropes_x.at[0].set(80)
        rt = config.ropes_top.at[0].set(49)
        rb = config.ropes_bottom.at[0].set(100)
        ra = config.ropes_active.at[0].set(1)

        # left platforms
        px = px.at[0:6].set(4)
        py = py.at[0].set(56)
        py = py.at[1].set(66)
        py = py.at[2].set(76)
        py = py.at[3].set(86)
        py = py.at[4].set(106)
        py = py.at[5].set(116)
        pa = pa.at[0:6].set(1)

        # right platforms
        px = px.at[6:12].set(144)
        py = py.at[6].set(56)
        py = py.at[7].set(66)
        py = py.at[8].set(76)
        py = py.at[9].set(86)
        py = py.at[10].set(106)
        py = py.at[11].set(116)
        pa = pa.at[6:12].set(1)

        pw = pw.at[0:12].set(12)

        return config.replace(
            ropes_x=rx,
            ropes_top=rt,
            ropes_bottom=rb,
            ropes_active=ra,
            items_x=ix,
            items_y=iy,
            items_active=ia,
            platforms_x=px,
            platforms_y=py,
            platforms_width=pw,
            platforms_active=pa,
        )

    def load_room_2_3(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_2 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active
        px = config.platforms_x
        py = config.platforms_y
        pw = config.platforms_width
        pa = config.platforms_active

        # Item: Gem
        ix = ix.at[0].set(17)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        # Ladder
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)

        # Dropout floor (using platform)
        px = px.at[0].set(36)
        py = py.at[0].set(46)
        pw = pw.at[0].set(88)
        pa = pa.at[0].set(1)

        return config.replace(
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            items_x=ix,
            items_y=iy,
            items_active=ia,
            platforms_x=px,
            platforms_y=py,
            platforms_width=pw,
            platforms_active=pa,
        )

    def load_room_2_4(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_3 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        # Two ladders: one to top, one to bottom
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(46)
        la = la.at[0].set(1)

        lx = lx.at[1].set(72)
        lt = lt.at[1].set(50)
        lb = lb.at[1].set(149)
        la = la.at[1].set(1)

        # Two snakes
        ex = config.enemies_x.at[0].set(44)
        ey = config.enemies_y.at[0].set(33) # ALE visible snake top is screen80
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(0) # Static snake
        eminx = config.enemies_min_x.at[0].set(44)
        emaxx = config.enemies_max_x.at[0].set(51)

        ex = ex.at[1].set(108)
        ey = ey.at[1].set(33)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(0)
        eminx = eminx.at[1].set(108)
        emaxx = emaxx.at[1].set(115)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )

    def load_room_2_5(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_4 in M1
        lax = config.lasers_x
        laa = config.lasers_active

        # 8 Lasers
        lax = lax.at[0].set(36)
        lax = lax.at[1].set(44)
        lax = lax.at[2].set(60)
        lax = lax.at[3].set(68)
        lax = lax.at[4].set(88)
        lax = lax.at[5].set(96)
        lax = lax.at[6].set(112)
        lax = lax.at[7].set(120)
        laa = laa.at[0:8].set(1)

        return config.replace(
            lasers_x=lax,
            lasers_active=laa,
        )

    def load_room_2_6(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_5 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        # 2 Ladders
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(46)
        la = la.at[0].set(1)
        lx = lx.at[1].set(72)
        lt = lt.at[1].set(50)
        lb = lb.at[1].set(149)
        la = la.at[1].set(1)

        ex = config.enemies_x.at[0].set(99)
        ey = config.enemies_y.at[0].set(35)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(4)
        emaxx = config.enemies_max_x.at[0].set(156)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )

    def load_room_2_7(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_2_6 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active

        # 1 Ladder to bottom
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(125)
        lb = lb.at[0].set(150)
        la = la.at[0].set(1)

        # 2 Ropes
        rx = config.ropes_x.at[0].set(71)
        rt = config.ropes_top.at[0].set(49)
        rb = config.ropes_bottom.at[0].set(97)
        ra = config.ropes_active.at[0].set(1)

        rx = rx.at[1].set(87)
        rt = rt.at[1].set(49)
        rb = rb.at[1].set(81)
        ra = ra.at[1].set(1)

        # 1 Item (Key)
        ix = ix.at[0].set(76)
        iy = iy.at[0].set(64)
        ia = ia.at[0].set(1)

        return config.replace(
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            ropes_x=rx,
            ropes_top=rt,
            ropes_bottom=rb,
            ropes_active=ra,
            items_x=ix,
            items_y=iy,
            items_active=ia,
        )

    def load_room_3_7(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_7 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        px = config.platforms_x
        py = config.platforms_y
        pw = config.platforms_width
        pa = config.platforms_active

        # Ladder to top
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(47)
        la = la.at[0].set(1)

        # Snake enemy
        ex = config.enemies_x.at[0].set(30)
        ey = config.enemies_y.at[0].set(34) # Floor at 47, snake height 13 -> 34
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(0) # Static snake
        eminx = config.enemies_min_x.at[0].set(30)
        emaxx = config.enemies_max_x.at[0].set(37)

        # Dropout floor (using platform)
        px = px.at[0].set(36)
        py = py.at[0].set(46)
        pw = pw.at[0].set(88) # Multiple of 12 (7 * 12)
        pa = pa.at[0].set(1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            platforms_x=px,
            platforms_y=py,
            platforms_width=pw,
            platforms_active=pa,
        )

    def load_room_3_8(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_8 in M1 (rightmost room level 3)
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active

        # 3 Gems on the top right
        ix = ix.at[0].set(99)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)
        
        ix = ix.at[1].set(115)
        iy = iy.at[1].set(7)
        ia = ia.at[1].set(1)
        
        ix = ix.at[2].set(131)
        iy = iy.at[2].set(7)
        ia = ia.at[2].set(1)

        return config.replace(
            items_x=ix,
            items_y=iy,
            items_active=ia,
        )

    def load_room_3_3(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_3 in M1
        px = config.platforms_x
        py = config.platforms_y
        pw = config.platforms_width
        pa = config.platforms_active

        # Dropout floor (using platform)
        px = px.at[0].set(36)
        py = py.at[0].set(46)
        pw = pw.at[0].set(88) # Multiple of 12 (7 * 12)
        pa = pa.at[0].set(1)

        ex = config.enemies_x.at[0].set(45)
        ey = config.enemies_y.at[0].set(33)
        ea = config.enemies_active.at[0].set(1) # This will be overwritten by state.global_enemies_active but good for documentation/completeness
        ed = config.enemies_direction.at[0].set(1)
        eminx = config.enemies_min_x.at[0].set(32)
        emaxx = config.enemies_max_x.at[0].set(121)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            platforms_x=px,
            platforms_y=py,
            platforms_width=pw,
            platforms_active=pa,
        )

    def load_room_3_5(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_5 in M1
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active
        px = config.platforms_x
        py = config.platforms_y
        pw = config.platforms_width
        pa = config.platforms_active

        # Dropout floor (using platform)
        px = px.at[0].set(36)
        py = py.at[0].set(46)
        pw = pw.at[0].set(88) # Multiple of 12 (7 * 12)
        pa = pa.at[0].set(1)
        
        # 2 Gems (coins) on the platform
        ix = ix.at[0].set(139)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        ix = ix.at[1].set(75)
        iy = iy.at[1].set(7)
        ia = ia.at[1].set(1)

        return config.replace(
            items_x=ix,
            items_y=iy,
            items_active=ia,
            platforms_x=px,
            platforms_y=py,
            platforms_width=pw,
            platforms_active=pa,
        )

    def load_room_3_4(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_4 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active

        # Amulet position
        ix = ix.at[0].set(15)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(47)
        la = la.at[0].set(1)

        return config.replace(
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
            items_x=ix,
            items_y=iy,
            items_active=ia,
        )

    def load_room_3_6(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_6 in M1
        lx = config.ladders_x
        lt = config.ladders_top
        lb = config.ladders_bottom
        la = config.ladders_active

        ex = config.enemies_x.at[0].set(59)
        ey = config.enemies_y.at[0].set(35)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(1)
        eminx = config.enemies_min_x.at[0].set(4)
        emaxx = config.enemies_max_x.at[0].set(156)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(47)
        la = la.at[0].set(1)

        return config.replace(
            enemies_x=ex,
            enemies_y=ey,
            enemies_active=ea,
            enemies_direction=ed,
            enemies_min_x=eminx,
            enemies_max_x=emaxx,
            ladders_x=lx,
            ladders_top=lt,
            ladders_bottom=lb,
            ladders_active=la,
        )

    def load_room_3_1(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_1 in M1
        return config

    def load_room_3_2(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_2 in M1
        dx = config.doors_x.at[0].set(20)
        dy = config.doors_y.at[0].set(7)
        da = config.doors_active.at[0].set(1)
        dx = dx.at[1].set(136)
        dy = dy.at[1].set(7)
        da = da.at[1].set(1)

        return config.replace(
            doors_x=dx,
            doors_y=dy,
            doors_active=da,
        )

    def load_room_3_0(config: RoomConfig) -> RoomConfig:
        # Corresponds to ROOM_3_0 in M1 (Bonus Room)
        ix = config.items_x
        iy = config.items_y
        ia = config.items_active

        # Gem position
        ix = ix.at[0].set(19)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        return config.replace(
            items_x=ix,
            items_y=iy,
            items_active=ia,
        )

    config = jax.lax.switch(
        get_room_idx(room_id),
        [
            load_room_0_3,
            load_room_0_4,
            load_room_0_5,
            load_room_1_3,
            load_room_1_2,
            load_room_1_4,
            load_room_1_5,
            load_room_1_6,
            load_room_2_2,
            load_room_2_1,
            load_room_2_3,
            load_room_2_4,
            load_room_2_5,
            load_room_2_6,
            load_room_2_7,
            load_room_3_7,
            load_room_3_8,
            load_room_3_6,
            load_room_3_4,
            load_room_3_3,
            load_room_3_5,
            load_room_3_1,
            load_room_3_2,
            load_room_3_0,
        ],
        config,
    )

    native_room = jnp.argmax(ALE_ROOM_IDS == room_id)
    typ, x, y, duplicates = LEVEL_CONTENTS[jnp.minimum(state.level, 2), native_room]
    entity_active = jnp.where(jnp.isin(typ, jnp.array([5,8,10])),
        state.global_enemies_active[room_id], state.global_items_active[room_id])
    history_weights = jnp.array([1,0,4], dtype=jnp.int32).at[1].set(jnp.where(duplicates & 1 != 0, 2, 8))
    remaining_mask = jnp.sum(entity_active * history_weights)
    group = jnp.argmax(_THIRD_GROUP_ROOMS == native_room)
    grouped = jnp.any(_THIRD_GROUP_ROOMS == native_room)
    restored_x = jnp.where(grouped,
        _THIRD_GROUP_X[group, _ANCHOR_INDEX_BY_MASK[remaining_mask]], _THIRD_PRIMARY_X[native_room])
    restored_x = jnp.where(jnp.logical_and(grouped, typ == 5),
        _THIRD_PRIMARY_X[native_room] + jnp.where(remaining_mask == 1, 0, 16), restored_x)
    x = jnp.where(state.level == 2, restored_x, x)
    duplicates = jnp.where(state.level == 2, _DUPLICATES_BY_MASK[remaining_mask], duplicates)
    offsets = jnp.array([0, 0, 32], dtype=jnp.int32).at[1].set(jnp.where(duplicates & 1 != 0, 16, 64))
    # Keep original history slots stable while drawing the surviving group
    # at the compressed native anchors on the next room initialization.
    rank = jnp.clip(jnp.cumsum(entity_active) - 1, 0, 2)
    compact_offsets = offsets.at[1].set(jnp.where(duplicates & 1 != 0, 16,
        jnp.where(duplicates & 4 != 0, 64, 32)))
    offsets = jnp.where(state.level == 2, compact_offsets[rank], offsets)
    entity_x = x - 1 + offsets + jnp.where(typ == 8, 1, 0)
    base_y = jnp.where(native_room == 0, 55, jnp.where(native_room == 1, 98, jnp.where(native_room == 7, 54, 53))) + 255 - y - 47
    item_y = base_y + jnp.where(y < 216, 4, 0)
    enemy_y = base_y + jnp.where(y < 216, 8, 0) + jnp.where(typ == 5, 25, 0)
    later = state.level > 0
    is_start = jnp.logical_or(jnp.isin(room_id, jnp.array([4,12])),
        jnp.logical_and(state.level == 2, room_id == 27))
    config = config.replace(
        items_x=jnp.where(later, entity_x, config.items_x),
        items_y=jnp.where(later, jnp.full_like(config.items_y, item_y), config.items_y),
        enemies_x=jnp.where(jnp.logical_and(later, ~is_start), entity_x, config.enemies_x),
        enemies_y=jnp.where(jnp.logical_and(later, ~is_start), jnp.full_like(config.enemies_y, enemy_y), config.enemies_y),
        enemies_bouncing=jnp.where(later, jnp.full_like(config.enemies_bouncing, (typ == 5).astype(jnp.int32)), config.enemies_bouncing),
        enemies_direction=jnp.where(jnp.logical_and(later, ~is_start), jnp.full_like(config.enemies_direction, jnp.where(typ == 8, 0, -1)), config.enemies_direction),
        enemies_min_x=jnp.where(jnp.logical_and(later, ~is_start), 4 + offsets, config.enemies_min_x),
        enemies_max_x=jnp.where(jnp.logical_and(later, ~is_start), 148 - jnp.where(duplicates & 1 != 0, 16, jnp.where(duplicates & 4 != 0, 64, jnp.where(duplicates & 2 != 0, 32, 0))) + offsets, config.enemies_max_x))
    pit_skull = jnp.logical_and(state.level == 2, room_id == 27)
    config = config.replace(
        enemies_x=config.enemies_x.at[0].set(jnp.where(pit_skull, 57, config.enemies_x[0])),
        enemies_y=config.enemies_y.at[0].set(jnp.where(pit_skull, 34, config.enemies_y[0])),
        enemies_direction=config.enemies_direction.at[0].set(jnp.where(pit_skull, 1, config.enemies_direction[0])),
        enemies_min_x=config.enemies_min_x.at[0].set(jnp.where(pit_skull, 32, config.enemies_min_x[0])),
        enemies_max_x=config.enemies_max_x.at[0].set(jnp.where(pit_skull, 122, config.enemies_max_x[0])))

    return state.replace(
        room_id=room_id,
        native_rolling_x=jnp.int32(58),
        room5_skull_phase=jnp.int32(0),
        room5_skull_spawn_pending=jnp.logical_or(room_id == 5, room_id == 10),
        enemies_x=config.enemies_x,
        enemies_y=config.enemies_y,
        # Enemy position resets on entry, but the rolling direction survives.
        enemies_direction=jnp.where(room_id == 4,
            config.enemies_direction.at[0].set(state.rolling_skull_direction), config.enemies_direction),
        enemies_min_x=config.enemies_min_x,
        enemies_max_x=config.enemies_max_x,
        enemies_bouncing=config.enemies_bouncing,
        ladders_x=config.ladders_x,
        ladders_top=config.ladders_top,
        ladders_bottom=config.ladders_bottom,
        ladders_active=config.ladders_active,
        ropes_x=config.ropes_x,
        ropes_top=config.ropes_top,
        ropes_bottom=config.ropes_bottom,
        ropes_active=config.ropes_active,
        items_x=config.items_x,
        items_y=config.items_y,
        doors_x=config.doors_x,
        doors_y=config.doors_y,
        conveyors_x=config.conveyors_x,
        conveyors_y=config.conveyors_y,
        conveyors_active=config.conveyors_active,
        conveyors_direction=config.conveyors_direction,
        lasers_x=config.lasers_x,
        lasers_active=config.lasers_active,
        platforms_x=config.platforms_x,
        platforms_y=config.platforms_y,
        platforms_width=config.platforms_width,
        platforms_active=config.platforms_active,
        enemies_active=state.global_enemies_active[room_id],
        enemies_type=state.global_enemies_type[room_id],
        items_active=state.global_items_active[room_id],
        items_type=state.global_items_type[room_id],
        doors_active=state.global_doors_active[room_id]
    )
