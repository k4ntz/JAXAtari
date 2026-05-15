import jax
import jax.numpy as jnp
from flax import struct

from jaxatari.games.montezuma_revenge.core import MontezumaRevengeConstants, MontezumaRevengeState, get_room_idx

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
        lt = lt.at[0].set(48)
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
        ex = config.enemies_x.at[0].set(93)
        ey = config.enemies_y.at[0].set(119)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(1)
        eminx = config.enemies_min_x.at[0].set(55)
        emaxx = config.enemies_max_x.at[0].set(100)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(49)
        lb = lb.at[0].set(88)
        la = la.at[0].set(1)
        lx = lx.at[1].set(128)
        lt = lt.at[1].set(92)
        lb = lb.at[1].set(130)
        la = la.at[1].set(1)
        lx = lx.at[2].set(16)
        lt = lt.at[2].set(92)
        lb = lb.at[2].set(130)
        la = la.at[2].set(1)

        rx = config.ropes_x.at[0].set(111)
        rt = config.ropes_top.at[0].set(49)
        rb = config.ropes_bottom.at[0].set(88)
        ra = config.ropes_active.at[0].set(1)

        ix = ix.at[0].set(13)
        iy = iy.at[0].set(52)
        ia = ia.at[0].set(1)

        cx = config.conveyors_x.at[0].set(60)
        cy = config.conveyors_y.at[0].set(88)
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

        ex = config.enemies_x.at[0].set(112)
        ey = config.enemies_y.at[0].set(33)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(10)
        emaxx = config.enemies_max_x.at[0].set(124)

        ex = ex.at[1].set(95)
        ey = ey.at[1].set(33)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(-1)
        eminx = eminx.at[1].set(4)
        emaxx = emaxx.at[1].set(118)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(48)
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

        ex = config.enemies_x.at[0].set(92)
        ey = config.enemies_y.at[0].set(36)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(4)
        emaxx = config.enemies_max_x.at[0].set(156)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(48)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)
        lx = lx.at[1].set(72)
        lt = lt.at[1].set(6)
        lb = lb.at[1].set(44)
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

        ex = config.enemies_x.at[0].set(30)
        ey = config.enemies_y.at[0].set(33)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(1)
        eminx = config.enemies_min_x.at[0].set(8)
        emaxx = config.enemies_max_x.at[0].set(155)

        ex = ex.at[1].set(47)
        ey = ey.at[1].set(33)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(1)
        eminx = eminx.at[1].set(8)
        emaxx = emaxx.at[1].set(155)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(48)
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

        ex = config.enemies_x.at[0].set(93)
        ey = config.enemies_y.at[0].set(69)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(-1)
        eminx = config.enemies_min_x.at[0].set(48)
        emaxx = config.enemies_max_x.at[0].set(105)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(126)
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
        lb = lb.at[0].set(48)
        la = la.at[0].set(1)

        # item: sword
        ix = ix.at[0].set(12)
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
        lt = lt.at[0].set(48)
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
        lb = lb.at[0].set(48)
        la = la.at[0].set(1)

        ex = config.enemies_x.at[0].set(18)
        ey = config.enemies_y.at[0].set(35) # Floor is at 48, snake height is 13. 48-13=35
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(0) # Static snake
        eminx = config.enemies_min_x.at[0].set(18)
        emaxx = config.enemies_max_x.at[0].set(47)

        ex = ex.at[1].set(50)
        ey = ey.at[1].set(35)
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
        lb = lb.at[0].set(47)
        la = la.at[0].set(1)

        # Dropout floor (using platform)
        px = px.at[0].set(32)
        py = py.at[0].set(47)
        pw = pw.at[0].set(96)
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
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)

        lx = lx.at[1].set(72)
        lt = lt.at[1].set(48)
        lb = lb.at[1].set(149)
        la = la.at[1].set(1)

        # Two snakes
        ex = config.enemies_x.at[0].set(44)
        ey = config.enemies_y.at[0].set(35) # Floor at 48, snake height 13 -> 35
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(0) # Static snake
        eminx = config.enemies_min_x.at[0].set(44)
        emaxx = config.enemies_max_x.at[0].set(51)

        ex = ex.at[1].set(108)
        ey = ey.at[1].set(35)
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
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)
        lx = lx.at[1].set(72)
        lt = lt.at[1].set(48)
        lb = lb.at[1].set(149)
        la = la.at[1].set(1)

        ex = config.enemies_x.at[0].set(100)
        ey = config.enemies_y.at[0].set(36)
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
        lt = lt.at[0].set(123)
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
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)

        # Snake enemy
        ex = config.enemies_x.at[0].set(30)
        ey = config.enemies_y.at[0].set(34) # Floor at 47, snake height 13 -> 34
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(0) # Static snake
        eminx = config.enemies_min_x.at[0].set(30)
        emaxx = config.enemies_max_x.at[0].set(37)

        # Dropout floor (using platform)
        px = px.at[0].set(32)
        py = py.at[0].set(47)
        pw = pw.at[0].set(96) # Multiple of 12 (7 * 12)
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
        px = px.at[0].set(32)
        py = py.at[0].set(47)
        pw = pw.at[0].set(96) # Multiple of 12 (7 * 12)
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
        px = px.at[0].set(32)
        py = py.at[0].set(47)
        pw = pw.at[0].set(96) # Multiple of 12 (7 * 12)
        pa = pa.at[0].set(1)
        
        # 2 Gems (coins) on the platform
        ix = ix.at[0].set(139)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        ix = ix.at[1].set(77)
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
        ix = ix.at[0].set(17)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(44)
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

        ex = config.enemies_x.at[0].set(60)
        ey = config.enemies_y.at[0].set(36)
        ea = config.enemies_active.at[0].set(1)
        ed = config.enemies_direction.at[0].set(1)
        eminx = config.enemies_min_x.at[0].set(4)
        emaxx = config.enemies_max_x.at[0].set(156)

        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(44)
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

    return state.replace(
        room_id=room_id,
        enemies_x=config.enemies_x,
        enemies_y=config.enemies_y,
        enemies_direction=config.enemies_direction,
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
