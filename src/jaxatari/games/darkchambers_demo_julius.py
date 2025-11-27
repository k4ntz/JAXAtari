import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from gym import spaces
from jaxatari.environment import JAXAtariAction as Action

from jaxatari.rendering.jax_rendering_utils import RendererConfig, JaxRenderingUtils


# Game / world dimensions
GAME_H = 210
GAME_W = 160
WORLD_W = GAME_W * 4
WORLD_H = GAME_H * 4


class DarkChambersState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    step: jnp.ndarray
    enemy_x: jnp.ndarray
    enemy_y: jnp.ndarray
    rng: jnp.ndarray


class DarkChambersRenderer:
    def __init__(self):
        self.config = RendererConfig(game_dimensions=(GAME_H, GAME_W), channels=3)
        self.jr = JaxRenderingUtils(self.config)
        bg = (8, 10, 20)
        player = (200, 80, 60)
        enemy = (80, 200, 120)
        wall = (150, 120, 70)
        self.PALETTE = jnp.array([bg, player, enemy, wall], dtype=jnp.uint8)

        # Walls defined in WORLD coordinates so camera can move
        wall_thick = 8
        self.WALLS = jnp.array([
            [0, 0, WORLD_W, wall_thick],                        # top border
            [0, WORLD_H - wall_thick, WORLD_W, wall_thick],     # bottom border
            [0, 0, wall_thick, WORLD_H],                        # left border
            [WORLD_W - wall_thick, 0, wall_thick, WORLD_H],     # right border
            [200, 150, 80, 16],                                 # interior horizontal
            [300, 250, 16, 120],                                # interior vertical
        ], dtype=jnp.int32)

    def render(self, state: DarkChambersState) -> jnp.ndarray:
        # object raster: H x W of palette ids
        object_raster = jnp.full((self.config.game_dimensions[0], self.config.game_dimensions[1]), 0, dtype=jnp.uint8)

        # camera origin in world coords
        cam_x = jnp.clip(state.x - GAME_W // 2, 0, WORLD_W - GAME_W).astype(jnp.int32)
        cam_y = jnp.clip(state.y - GAME_H // 2, 0, WORLD_H - GAME_H).astype(jnp.int32)

        # player
        screen_px = (state.x - cam_x).astype(jnp.int32)
        screen_py = (state.y - cam_y).astype(jnp.int32)
        pos = jnp.array([[screen_px, screen_py]], dtype=jnp.int32)
        size = jnp.array([[12, 12]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(object_raster, positions=pos, sizes=size, color_id=1)

        # enemy
        e_px = (state.enemy_x - cam_x).astype(jnp.int32)
        e_py = (state.enemy_y - cam_y).astype(jnp.int32)
        epos = jnp.array([[e_px, e_py]], dtype=jnp.int32)
        esize = jnp.array([[10, 10]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(object_raster, positions=epos, sizes=esize, color_id=2)

        # walls: translate world->viewport
        wall_positions = (self.WALLS[:, 0:2] - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        wall_sizes = self.WALLS[:, 2:4]
        object_raster = self.jr.draw_rects(object_raster, positions=wall_positions, sizes=wall_sizes, color_id=3)

        img = self.jr.render_from_palette(object_raster, self.PALETTE)
        return img


class DarkChambersEnv:
    def __init__(self):
        self.renderer = DarkChambersRenderer()

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "x": spaces.Box(low=0, high=WORLD_W - 1, shape=(), dtype=jnp.int32),
            "y": spaces.Box(low=0, high=WORLD_H - 1, shape=(), dtype=jnp.int32),
            "step": spaces.Box(low=0, high=10**9, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(GAME_H, GAME_W, 3), dtype=jnp.uint8)

    def reset(self, key=None) -> DarkChambersState:
        if key is None:
            key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        ex = jax.random.randint(subkey, (), 0, WORLD_W, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        ey = jax.random.randint(subkey, (), 0, WORLD_H, dtype=jnp.int32)
        return DarkChambersState(
            x=jnp.array(24, dtype=jnp.int32),
            y=jnp.array(24, dtype=jnp.int32),
            step=jnp.array(0, dtype=jnp.int32),
            enemy_x=ex,
            enemy_y=ey,
            rng=key,
        )

    def step(self, state: DarkChambersState, action: int) -> DarkChambersState:
        a = jnp.asarray(action)
        dx = jnp.where(a == Action.LEFT, -1, jnp.where(a == Action.RIGHT, 1, 0))
        dy = jnp.where(a == Action.UP, -1, jnp.where(a == Action.DOWN, 1, 0))
        prop_x = state.x + dx
        prop_y = state.y + dy

        # clamp
        prop_x = jnp.clip(prop_x, 0, WORLD_W - 1)
        prop_y = jnp.clip(prop_y, 0, WORLD_H - 1)

        # collision in WORLD space
        WALLS = self.renderer.WALLS
        pw, ph = 12, 12

        def collides(px, py):
            wx = WALLS[:, 0]
            wy = WALLS[:, 1]
            ww = WALLS[:, 2]
            wh = WALLS[:, 3]
            overlap_x = (px <= (wx + ww - 1)) & ((px + pw - 1) >= wx)
            overlap_y = (py <= (wy + wh - 1)) & ((py + ph - 1) >= wy)
            return jnp.any(overlap_x & overlap_y)

        # sequential axis resolution
        try_x = prop_x
        collide_x = collides(try_x, state.y)
        new_x = jnp.where(~collide_x, try_x, state.x)

        try_y = prop_y
        collide_y = collides(new_x, try_y)
        new_y = jnp.where(~collide_y, try_y, state.y)

        # update enemy random walk
        rng, subkey = jax.random.split(state.rng)
        d = jax.random.randint(subkey, (2,), -1, 2, dtype=jnp.int32)
        edx, edy = d[0], d[1]
        prop_ex = jnp.clip(state.enemy_x + edx, 0, WORLD_W - 1)
        prop_ey = jnp.clip(state.enemy_y + edy, 0, WORLD_H - 1)

        # enemy collision
        e_pw, e_ph = 10, 10
        def enemy_collides(epx, epy):
            e_overlap_x = (epx <= (WALLS[:,0] + WALLS[:,2] - 1)) & ((epx + e_pw - 1) >= WALLS[:,0])
            e_overlap_y = (epy <= (WALLS[:,1] + WALLS[:,3] - 1)) & ((epy + e_ph - 1) >= WALLS[:,1])
            return jnp.any(e_overlap_x & e_overlap_y)

        any_enemy_collide = enemy_collides(prop_ex, prop_ey)
        new_ex = jnp.where(~any_enemy_collide, prop_ex, state.enemy_x)
        new_ey = jnp.where(~any_enemy_collide, prop_ey, state.enemy_y)

        new_step = state.step + 1

        return DarkChambersState(x=new_x, y=new_y, step=new_step, enemy_x=new_ex, enemy_y=new_ey, rng=rng)

    def step_with_info(self, state: DarkChambersState, action: int):
        """Non-JIT wrapper returning (obs, state, reward, done, info) with readable debug prints."""
        # Friendly debug + non-JAX checks
        action_name = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN"}.get(int(action), str(int(action)))

        prop_x = int(state.x)
        prop_y = int(state.y)
        if action == Action.LEFT:
            prop_x -= 1
        elif action == Action.RIGHT:
            prop_x += 1
        elif action == Action.UP:
            prop_y -= 1
        elif action == Action.DOWN:
            prop_y += 1

        prop_x = max(0, min(prop_x, WORLD_W - 1))
        prop_y = max(0, min(prop_y, WORLD_H - 1))

        # quick world-space collision debug
        WALLS = np.array(self.renderer.WALLS)
        pw, ph = 12, 12
        collide_x = False
        for i, w in enumerate(WALLS):
            wx, wy, ww, wh = int(w[0]), int(w[1]), int(w[2]), int(w[3])
            ox = (prop_x <= (wx + ww - 1)) and ((prop_x + pw - 1) >= wx)
            oy = (int(state.y) <= (wy + wh - 1)) and ((int(state.y) + ph - 1) >= wy)
            if ox and oy:
                print(f"  X-COLLISION with wall {i}: [{wx}, {wy}, {ww}, {wh}]")
                collide_x = True
                break

        final_x = int(state.x) if collide_x else prop_x

        collide_y = False
        for i, w in enumerate(WALLS):
            wx, wy, ww, wh = int(w[0]), int(w[1]), int(w[2]), int(w[3])
            ox = (final_x <= (wx + ww - 1)) and ((final_x + pw - 1) >= wx)
            oy = (prop_y <= (wy + wh - 1)) and ((prop_y + ph - 1) >= wy)
            if ox and oy:
                print(f"  Y-COLLISION with wall {i}: [{wx}, {wy}, {ww}, {wh}]")
                collide_y = True
                break

        print(f"[Step {int(state.step):3d}] Action={action_name:5s} | World: ({int(state.x):3d}, {int(state.y):3d}) -> ({prop_x:3d}, {prop_y:3d}) | CollideX={collide_x} CollideY={collide_y}")

        # call JAX step to keep state updates consistent
        new_state = self.step(state, action)
        obs = {"x": new_state.x, "y": new_state.y, "step": new_state.step, "enemy_x": new_state.enemy_x, "enemy_y": new_state.enemy_y}
        reward = 0.0
        done = False
        info = {}
        return obs, new_state, reward, done, info

    def _get_observation(self, state: DarkChambersState):
        return {"x": state.x, "y": state.y, "step": state.step, "enemy_x": state.enemy_x, "enemy_y": state.enemy_y}

    def render(self, state: DarkChambersState) -> np.ndarray:
        img = self.renderer.render(state)
        return np.array(img)
