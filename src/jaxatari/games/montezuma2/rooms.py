import jax
import jax.numpy as jnp
from .core import Montezuma2Constants, Montezuma2State

def load_room(room_id: jnp.ndarray, state: Montezuma2State, consts: Montezuma2Constants) -> Montezuma2State:
    enemies_x = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    enemies_y = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    enemies_active = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    enemies_direction = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    enemies_type = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    enemies_min_x = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    enemies_max_x = jnp.full(consts.MAX_ENEMIES_PER_ROOM, consts.WIDTH - 8, dtype=jnp.int32)
    enemies_bouncing = jnp.zeros(consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
    
    ladders_x = jnp.zeros(consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    ladders_top = jnp.zeros(consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    ladders_bottom = jnp.zeros(consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    ladders_active = jnp.zeros(consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
    
    ropes_x = jnp.zeros(consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    ropes_top = jnp.zeros(consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    ropes_bottom = jnp.zeros(consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
    ropes_active = jnp.zeros(consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)

    items_x = jnp.zeros(consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
    items_y = jnp.zeros(consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
    items_active = jnp.zeros(consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)

    doors_x = jnp.zeros(consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
    doors_y = jnp.zeros(consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
    doors_active = jnp.zeros(consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32)

    conveyors_x = jnp.zeros(consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    conveyors_y = jnp.zeros(consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    conveyors_active = jnp.zeros(consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    conveyors_direction = jnp.zeros(consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
    
    lasers_x = jnp.zeros(consts.MAX_LASERS_PER_ROOM, dtype=jnp.int32)
    lasers_active = jnp.zeros(consts.MAX_LASERS_PER_ROOM, dtype=jnp.int32)

    def load_room_0(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa = args
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
        
        return (enemies_x, enemies_y, enemies_active, enemies_direction, enemies_min_x, enemies_max_x, enemies_bouncing,
                lx, lt, lb, la,
                ropes_x, ropes_top, ropes_bottom, ropes_active,
                ix, iy, ia,
                doors_x, doors_y, doors_active,
                conveyors_x, conveyors_y, conveyors_active, conveyors_direction,
                lax, laa)

    def load_room_1(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa = args
        ex = enemies_x.at[0].set(93)
        ey = enemies_y.at[0].set(119)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(1)
        eminx = enemies_min_x.at[0].set(45)
        emaxx = enemies_max_x.at[0].set(110)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(49)
        lb = lb.at[0].set(88)
        la = la.at[0].set(1)
        lx = lx.at[1].set(128)
        lt = lt.at[1].set(92)
        lb = lb.at[1].set(133)
        la = la.at[1].set(1)
        lx = lx.at[2].set(16)
        lt = lt.at[2].set(92)
        lb = lb.at[2].set(133)
        la = la.at[2].set(1)

        rx = ropes_x.at[0].set(111)
        rt = ropes_top.at[0].set(49)
        rb = ropes_bottom.at[0].set(88)
        ra = ropes_active.at[0].set(1)

        ix = ix.at[0].set(13)
        iy = iy.at[0].set(52)
        ia = ia.at[0].set(1)

        cx = conveyors_x.at[0].set(60)
        cy = conveyors_y.at[0].set(88)
        ca = conveyors_active.at[0].set(1)
        cd = conveyors_direction.at[0].set(1)
        
        dx = doors_x.at[0].set(16)
        dy = doors_y.at[0].set(7)
        da = doors_active.at[0].set(1)
        dx = dx.at[1].set(140)
        dy = dy.at[1].set(7)
        da = da.at[1].set(1)
        
        eb = enemies_bouncing

        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lasers_x, lasers_active)

    def load_room_2(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa = args
        
        ex = enemies_x.at[0].set(112)
        ey = enemies_y.at[0].set(33)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(-1)
        eminx = enemies_min_x.at[0].set(10)
        emaxx = enemies_max_x.at[0].set(124)
        
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

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active

        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        eb = enemies_bouncing.at[0].set(1)
        eb = eb.at[1].set(1)

        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lasers_x, lasers_active)

    args = (ladders_x, ladders_top, ladders_bottom, ladders_active, items_x, items_y, items_active, lasers_x, lasers_active)
    
    def load_room_3(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa = args
        
        ex = enemies_x.at[0].set(92)
        ey = enemies_y.at[0].set(36)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(-1)
        eminx = enemies_min_x.at[0].set(4)
        emaxx = enemies_max_x.at[0].set(156)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(48)
        lb = lb.at[0].set(149)
        la = la.at[0].set(1)
        lx = lx.at[1].set(72)
        lt = lt.at[1].set(6)
        lb = lb.at[1].set(44)
        la = la.at[1].set(1)

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        ix = items_x
        iy = items_y
        ia = items_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        eb = enemies_bouncing

        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lasers_x, lasers_active)

    ex, ey, ea, ed, eminx, emaxx, eb, lx, lt, lb, la, rx, rt, rb, ra, ix, iy, ia, dx, dy, da, cx, cy, ca, cd, lax, laa = jax.lax.switch(room_id, [load_room_0, load_room_1, load_room_2, load_room_3], args)

    return state.replace(
        room_id=room_id,
        enemies_x=ex, enemies_y=ey, enemies_direction=ed, enemies_min_x=eminx, enemies_max_x=emaxx, enemies_bouncing=eb,
        ladders_x=lx, ladders_top=lt, ladders_bottom=lb, ladders_active=la,
        ropes_x=rx, ropes_top=rt, ropes_bottom=rb, ropes_active=ra,
        items_x=ix, items_y=iy,
        doors_x=dx, doors_y=dy,
        conveyors_x=cx, conveyors_y=cy, conveyors_active=ca, conveyors_direction=cd,
        lasers_x=lax, lasers_active=laa,
        enemies_active=state.global_enemies_active[room_id],
        enemies_type=state.global_enemies_type[room_id],
        items_active=state.global_items_active[room_id],
        items_type=state.global_items_type[room_id],
        doors_active=state.global_doors_active[room_id]
    )
