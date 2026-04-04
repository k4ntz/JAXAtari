import jax
import jax.numpy as jnp
from .core import Montezuma2Constants, Montezuma2State, get_room_idx

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

    def load_room_0_3(args):
        # New 3 was Old 4
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
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
                lax, laa, px, py, pw, pa)

    def load_room_0_4(args):
        # New 4 was Old 5
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
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
        lb = lb.at[1].set(130)
        la = la.at[1].set(1)
        lx = lx.at[2].set(16)
        lt = lt.at[2].set(92)
        lb = lb.at[2].set(130)
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
                lasers_x, lasers_active, px, py, pw, pa)

    def load_room_0_5(args):
        # New 5 was Old 3
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
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
                lasers_x, lasers_active, px, py, pw, pa)

    args = (ladders_x, ladders_top, ladders_bottom, ladders_active, items_x, items_y, items_active, lasers_x, lasers_active, jnp.zeros(consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32), jnp.zeros(consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32), jnp.zeros(consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32), jnp.zeros(consts.MAX_PLATFORMS_PER_ROOM, dtype=jnp.int32))
    
    def load_room_1_3(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
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
                lasers_x, lasers_active, px, py, pw, pa)

    def load_room_1_2(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        ex = enemies_x.at[0].set(30)
        ey = enemies_y.at[0].set(33)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(1)
        eminx = enemies_min_x.at[0].set(8)
        emaxx = enemies_max_x.at[0].set(155)
        
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
        
        eb = enemies_bouncing.at[0].set(1)
        eb = eb.at[1].set(1)

        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lasers_x, lasers_active, px, py, pw, pa)

    def load_room_1_4(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        ex = enemies_x.at[0].set(93)
        ey = enemies_y.at[0].set(69)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(-1)
        eminx = enemies_min_x.at[0].set(48)
        emaxx = enemies_max_x.at[0].set(105)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(126)
        lb = lb.at[0].set(150)
        la = la.at[0].set(1)
        
        rx = ropes_x.at[0].set(41)
        rt = ropes_top.at[0].set(50)
        rb = ropes_bottom.at[0].set(75)
        ra = ropes_active.at[0].set(1)
        
        rx = rx.at[1].set(125)
        rt = rt.at[1].set(49)
        rb = rb.at[1].set(100)
        ra = ra.at[1].set(1)
        
        # item: torch
        ix = ix.at[0].set(77)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)
        
        # doors
        dx = doors_x.at[0].set(56)
        dy = doors_y.at[0].set(86)
        da = doors_active.at[0].set(1)
        
        dx = dx.at[1].set(100)
        dy = dy.at[1].set(86)
        da = da.at[1].set(1)
        
        # conveyor
        cx = conveyors_x.at[0].set(60)
        cy = conveyors_y.at[0].set(46)
        ca = conveyors_active.at[0].set(1)
        cd = conveyors_direction.at[0].set(-1)
        
        eb = enemies_bouncing
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_1_5(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        # Sword room from Montezuma1 (ROOM_1_3)
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(48)
        la = la.at[0].set(1)

        # item: sword
        ix = ix.at[0].set(12)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        ex = enemies_x
        ey = enemies_y
        ea = enemies_active
        ed = enemies_direction
        eminx = enemies_min_x
        emaxx = enemies_max_x
        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_1_6(args):
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
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

        ex = enemies_x
        ey = enemies_y
        ea = enemies_active
        ed = enemies_direction
        eminx = enemies_min_x
        emaxx = enemies_max_x
        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_2_2(args):
        # Corresponds to ROOM_2_1 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(48)
        la = la.at[0].set(1)

        ex = enemies_x.at[0].set(18)
        ey = enemies_y.at[0].set(35) # Floor is at 48, snake height is 13. 48-13=35
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(0) # Static snake
        eminx = enemies_min_x.at[0].set(18)
        emaxx = enemies_max_x.at[0].set(47)

        ex = ex.at[1].set(50)
        ey = ey.at[1].set(35)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(0) # Static snake
        eminx = eminx.at[1].set(50)
        emaxx = emaxx.at[1].set(100) # Assuming some right boundary

        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_2_1(args):
        # Corresponds to ROOM_2_0 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        # item: key
        ix = ix.at[0].set(77)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        # rope
        rx = ropes_x.at[0].set(80)
        rt = ropes_top.at[0].set(49)
        rb = ropes_bottom.at[0].set(100)
        ra = ropes_active.at[0].set(1)

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

        ex = enemies_x
        ey = enemies_y
        ea = enemies_active
        ed = enemies_direction
        eminx = enemies_min_x
        emaxx = enemies_max_x
        eb = enemies_bouncing

        lx = ladders_x
        lt = ladders_top
        lb = ladders_bottom
        la = ladders_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa,
                px, py, pw, pa)

    def load_room_2_3(args):
        # Corresponds to ROOM_2_2 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        # Item: Gem
        ix = ix.at[0].set(17)
        iy = iy.at[0].set(7)
        ia = ia.at[0].set(1)

        # Ladder
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(48)
        la = la.at[0].set(1)

        # Dropout floor (using platform)
        px = px.at[0].set(32)
        py = py.at[0].set(48)
        pw = pw.at[0].set(96)
        pa = pa.at[0].set(1)

        ex = enemies_x
        ey = enemies_y
        ea = enemies_active
        ed = enemies_direction
        eminx = enemies_min_x
        emaxx = enemies_max_x
        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa,
                px, py, pw, pa)

    def load_room_2_4(args):
        # Corresponds to ROOM_2_3 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
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
        ex = enemies_x.at[0].set(44)
        ey = enemies_y.at[0].set(35) # Floor at 48, snake height 13 -> 35
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(0) # Static snake
        eminx = enemies_min_x.at[0].set(44)
        emaxx = enemies_max_x.at[0].set(51)

        ex = ex.at[1].set(108)
        ey = ey.at[1].set(35)
        ea = ea.at[1].set(1)
        ed = ed.at[1].set(0)
        eminx = eminx.at[1].set(108)
        emaxx = emaxx.at[1].set(115)

        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_2_5(args):
        # Corresponds to ROOM_2_4 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
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

        ex = enemies_x
        ey = enemies_y
        ea = enemies_active
        ed = enemies_direction
        eminx = enemies_min_x
        emaxx = enemies_max_x
        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_2_6(args):
        # Corresponds to ROOM_2_5 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        # 2 Ladders
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)
        lx = lx.at[1].set(72)
        lt = lt.at[1].set(48)
        lb = lb.at[1].set(149)
        la = la.at[1].set(1)

        ex = enemies_x.at[0].set(100)
        ey = enemies_y.at[0].set(36)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(-1)
        eminx = enemies_min_x.at[0].set(4)
        emaxx = enemies_max_x.at[0].set(156)
        eb = enemies_bouncing

        rx = ropes_x
        rt = ropes_top
        rb = ropes_bottom
        ra = ropes_active
        
        dx = doors_x
        dy = doors_y
        da = doors_active
        
        cx = conveyors_x
        cy = conveyors_y
        ca = conveyors_active
        cd = conveyors_direction
        
        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_2_7(args):
        # Corresponds to ROOM_2_6 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args

        # 1 Ladder to bottom
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(123)
        lb = lb.at[0].set(150)
        la = la.at[0].set(1)

        # 2 Ropes
        rx = ropes_x.at[0].set(71)
        rt = ropes_top.at[0].set(49)
        rb = ropes_bottom.at[0].set(97)
        ra = ropes_active.at[0].set(1)

        rx = rx.at[1].set(87)
        rt = rt.at[1].set(49)
        rb = rb.at[1].set(81)
        ra = ra.at[1].set(1)

        # 1 Item (Key)
        ix = ix.at[0].set(76)
        iy = iy.at[0].set(64)
        ia = ia.at[0].set(1)

        return (enemies_x, enemies_y, enemies_active, enemies_direction, enemies_min_x, enemies_max_x, enemies_bouncing,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                doors_x, doors_y, doors_active,
                conveyors_x, conveyors_y, conveyors_active, conveyors_direction,
                lax, laa, px, py, pw, pa)

    def load_room_3_7(args):
        # Corresponds to ROOM_3_7 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args

        # Ladder to top
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)

        # Snake enemy
        ex = enemies_x.at[0].set(30)
        ey = enemies_y.at[0].set(35) # Floor at 48, snake height 13 -> 35
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(0) # Static snake
        eminx = enemies_min_x.at[0].set(30)
        emaxx = enemies_max_x.at[0].set(37)

        # Dropout floor (using platform)
        px = px.at[0].set(32)
        py = py.at[0].set(48)
        pw = pw.at[0].set(96) # Multiple of 12 (7 * 12)
        pa = pa.at[0].set(1)

        eb = enemies_bouncing

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

        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_3_8(args):
        # Corresponds to ROOM_3_8 in M1 (rightmost room level 3)
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
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

        ex = enemies_x
        ey = enemies_y
        ea = enemies_active
        ed = enemies_direction
        eminx = enemies_min_x
        emaxx = enemies_max_x
        eb = enemies_bouncing

        lx = ladders_x
        lt = ladders_top
        lb = ladders_bottom
        la = ladders_active

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

        return (ex, ey, ea, ed, eminx, emaxx, eb,
                lx, lt, lb, la,
                rx, rt, rb, ra,
                ix, iy, ia,
                dx, dy, da,
                cx, cy, ca, cd,
                lax, laa, px, py, pw, pa)

    def load_room_3_6(args):
        # Corresponds to ROOM_3_6 in M1
        lx, lt, lb, la, ix, iy, ia, lax, laa, px, py, pw, pa = args
        
        ex = enemies_x.at[0].set(60)
        ey = enemies_y.at[0].set(36)
        ea = enemies_active.at[0].set(1)
        ed = enemies_direction.at[0].set(1)
        eminx = enemies_min_x.at[0].set(4)
        emaxx = enemies_max_x.at[0].set(156)
        
        lx = lx.at[0].set(72)
        lt = lt.at[0].set(6)
        lb = lb.at[0].set(44)
        la = la.at[0].set(1)

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
                lax, laa, px, py, pw, pa)

    ex, ey, ea, ed, eminx, emaxx, eb, lx, lt, lb, la, rx, rt, rb, ra, ix, iy, ia, dx, dy, da, cx, cy, ca, cd, lax, laa, px, py, pw, pa = jax.lax.switch(
        get_room_idx(room_id),
        [load_room_0_3, load_room_0_4, load_room_0_5, load_room_1_3, load_room_1_2, load_room_1_4, load_room_1_5, load_room_1_6, load_room_2_2, load_room_2_1, load_room_2_3, load_room_2_4, load_room_2_5, load_room_2_6, load_room_2_7, load_room_3_7, load_room_3_8, load_room_3_6], args)

    return state.replace(
        room_id=room_id,
        enemies_x=ex, enemies_y=ey, enemies_direction=ed, enemies_min_x=eminx, enemies_max_x=emaxx, enemies_bouncing=eb,
        ladders_x=lx, ladders_top=lt, ladders_bottom=lb, ladders_active=la,
        ropes_x=rx, ropes_top=rt, ropes_bottom=rb, ropes_active=ra,
        items_x=ix, items_y=iy,
        doors_x=dx, doors_y=dy,
        conveyors_x=cx, conveyors_y=cy, conveyors_active=ca, conveyors_direction=cd,
        lasers_x=lax, lasers_active=laa,
        platforms_x=px, platforms_y=py, platforms_width=pw, platforms_active=pa,
        enemies_active=state.global_enemies_active[room_id],
        enemies_type=state.global_enemies_type[room_id],
        items_active=state.global_items_active[room_id],
        items_type=state.global_items_type[room_id],
        doors_active=state.global_doors_active[room_id]
    )
