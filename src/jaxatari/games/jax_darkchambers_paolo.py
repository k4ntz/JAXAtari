import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces
from jaxatari.rendering.jax_rendering_utils import RendererConfig, JaxRenderingUtils


# Minimal game dimensions to match project convention
GAME_H = 210
GAME_W = 160
# Expanded world so camera can follow the player (zoomed out 6x)
WORLD_W = GAME_W * 2
WORLD_H = GAME_H * 2

# Entity sizes (in world pixels)
PLAYER_W = 12
PLAYER_H = 12
ENEMY_W = 14
ENEMY_H = 14

# Item types
ITEM_HEART = 1
ITEM_POISON = 2

# visual sizes for items
ITEM_W = 6
ITEM_H = 6

# Spawner definitions (world coords). simple fixed spawners
SPAWNER_POSITIONS = jnp.array([
    [120, 100],
    [140, 200],
    [300, 300],
], dtype=jnp.int32)
SPAWN_COOLDOWN = 120  # frames between spawn attempts


# -- Paolo-style nested state types (Player/Enemies/Spawners/Level)
class PlayerState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    hp: jnp.ndarray
    facing: jnp.ndarray
    fire_cooldown: jnp.ndarray
    proj_x: jnp.ndarray
    proj_y: jnp.ndarray
    proj_active: jnp.ndarray


class EnemyState(NamedTuple):
    positions: jnp.ndarray
    hp: jnp.ndarray
    active: jnp.ndarray
    enemy_type: jnp.ndarray


class SpawnerState(NamedTuple):
    positions: jnp.ndarray
    hp: jnp.ndarray
    active: jnp.ndarray
    cooldown: jnp.ndarray


class LevelState(NamedTuple):
    tile_map: jnp.ndarray
    room_index: jnp.ndarray
    step_counter: jnp.ndarray
    score: jnp.ndarray
    items_positions: jnp.ndarray
    items_types: jnp.ndarray
    items_active: jnp.ndarray


class DarkChambersState(NamedTuple):
    player: PlayerState
    enemies: EnemyState
    spawners: SpawnerState
    level: LevelState
    rng_key: jnp.ndarray

# Keep a backwards-compatible name used elsewhere in this file
DarkState = DarkChambersState


class DarkChambersRenderer(JAXGameRenderer):
    """Simple renderer that draws a background and a single player rectangle.
    This is intentionally minimal so it integrates with the project's loader.
    """

    def __init__(self, consts=None):
        super().__init__(consts)
        self.config = RendererConfig(game_dimensions=(GAME_H, GAME_W), channels=3)
        self.jr = JaxRenderingUtils(self.config)

        bg = (8, 10, 20)
        player = (200, 80, 60)
        enemy = (255, 0, 255)  # bright magenta
        wall = (150, 120, 70)
        heart = (220, 30, 30)
        poison = (50, 200, 50)
        # palette: bg, player, enemy, wall, heart, poison
        self.PALETTE = jnp.array([bg, player, enemy, wall, heart, poison], dtype=jnp.uint8)

        # Hardcoded walls in WORLD coordinates: (x, y, w, h)
        # Walls move with the world; camera follows player
        wall_thick = 8
        self.WALLS = jnp.array([
            [0, 0, WORLD_W, wall_thick],                          # top border
            [0, WORLD_H - wall_thick, WORLD_W, wall_thick],       # bottom border
            [0, 0, wall_thick, WORLD_H],                          # left border
            [WORLD_W - wall_thick, 0, wall_thick, WORLD_H],       # right border
            [50, 80, 60, 16],                                     # horizontal wall 1
            [150, 120, 16, 80],                                   # vertical wall 1
            [220, 180, 70, 16],                                   # horizontal wall 2
            [280, 60, 16, 100],                                   # vertical wall 2
            [100, 240, 80, 16],                                   # horizontal wall 3
        ], dtype=jnp.int32)

    def render(self, state: DarkState) -> jnp.ndarray:
        # object raster: H x W of palette IDs
        object_raster = jnp.full((self.config.game_dimensions[0], self.config.game_dimensions[1]), 0, dtype=jnp.uint8)

        # Compute camera origin (top-left) in world coordinates (player is in world coords)
        cam_x = jnp.clip(state.player.x - GAME_W // 2, 0, WORLD_W - GAME_W).astype(jnp.int32)
        cam_y = jnp.clip(state.player.y - GAME_H // 2, 0, WORLD_H - GAME_H).astype(jnp.int32)

        # Draw player rectangle (translated to screen coords)
        screen_px = (state.player.x - cam_x).astype(jnp.int32)
        screen_py = (state.player.y - cam_y).astype(jnp.int32)
        pos = jnp.array([[screen_px, screen_py]], dtype=jnp.int32)
        size = jnp.array([[PLAYER_W, PLAYER_H]], dtype=jnp.int32)
        object_raster = self.jr.draw_rects(object_raster, positions=pos, sizes=size, color_id=1)

        # (Health bar will be drawn last so it is on top of world objects)

        # Draw all enemies (translate world -> viewport)
        try:
            enemy_world_pos = state.enemies.positions.astype(jnp.int32)
            enemy_screen_pos = (enemy_world_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32)
            num_en = enemy_screen_pos.shape[0]
            esize = jnp.tile(jnp.array([ENEMY_W, ENEMY_H], dtype=jnp.int32)[None, :], (num_en, 1))
            object_raster = self.jr.draw_rects(object_raster, positions=enemy_screen_pos, sizes=esize, color_id=2)
        except Exception:
            pass

        # Draw items (hearts/poison)
        try:
            items_pos = state.level.items_positions.astype(jnp.int32)
            items_active = state.level.items_active.astype(jnp.int32)
            items_type = state.level.items_types.astype(jnp.int32)
            # only draw active items
            if items_pos.shape[0] > 0:
                active_mask = items_active == 1
                active_pos = items_pos[active_mask]
                active_type = items_type[active_mask]
                if active_pos.shape[0] > 0:
                    # draw hearts and poison separately
                    heart_mask = active_type == ITEM_HEART
                    if jnp.any(heart_mask):
                        heart_pos = active_pos[heart_mask]
                        hsize = jnp.tile(jnp.array([ITEM_W, ITEM_H], dtype=jnp.int32)[None, :], (heart_pos.shape[0], 1))
                        object_raster = self.jr.draw_rects(object_raster, positions=(heart_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32), sizes=hsize, color_id=4)
                    poison_mask = active_type == ITEM_POISON
                    if jnp.any(poison_mask):
                        poison_pos = active_pos[poison_mask]
                        psize = jnp.tile(jnp.array([ITEM_W, ITEM_H], dtype=jnp.int32)[None, :], (poison_pos.shape[0], 1))
                        object_raster = self.jr.draw_rects(object_raster, positions=(poison_pos - jnp.array([cam_x, cam_y])).astype(jnp.int32), sizes=psize, color_id=5)
        except Exception:
            pass

        # Draw walls: translate from world -> viewport coordinates
        wall_positions = (self.WALLS[:, 0:2] - jnp.array([cam_x, cam_y])).astype(jnp.int32)
        wall_sizes = self.WALLS[:, 2:4]
        object_raster = self.jr.draw_rects(object_raster, positions=wall_positions, sizes=wall_sizes, color_id=3)

        # Draw health bar last (screen-space) so it's always on top of world objects
        try:
            hp_n = int(jnp.clip(state.player.hp, 0, 10))
            if hp_n > 0:
                # compute screen-space positions and clamp to renderer dimensions
                max_w = int(self.config.game_dimensions[1])
                max_h = int(self.config.game_dimensions[0])
                xs = (jnp.arange(hp_n) * (ITEM_W + 4) + 4).astype(jnp.int32)
                ys = jnp.zeros((hp_n,), dtype=jnp.int32) + 4
                # clip x positions so hearts never overflow the screen
                xs = jnp.clip(xs, 0, max_w - ITEM_W)
                ys = jnp.clip(ys, 0, max_h - ITEM_H)
                health_pos = jnp.stack([xs, ys], axis=1)
                health_size = jnp.tile(jnp.array([ITEM_W, ITEM_H], dtype=jnp.int32)[None, :], (hp_n, 1))
                object_raster = self.jr.draw_rects(object_raster, positions=health_pos, sizes=health_size, color_id=4)
        except Exception:
            pass

        # Convert to RGB using palette
        img = self.jr.render_from_palette(object_raster, self.PALETTE)
        return img


class JaxDarkChambers(JaxEnvironment[DarkState, DarkState, None, None]):
    """Tiny JaxEnvironment so `scripts/play.py` can load and run it via the existing loader."""

    def __init__(self, consts=None):
        super().__init__(consts)
        self.renderer = DarkChambersRenderer()

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Tuple[DarkState, DarkState]:
        if key is None:
            key = jax.random.PRNGKey(0)

        # Build state while keeping semantics (player at ~100,100, one enemy nearby)
        player = PlayerState(
            x=jnp.array(100, dtype=jnp.int32),
            y=jnp.array(100, dtype=jnp.int32),
            hp=jnp.array(3, dtype=jnp.int32),
            facing=jnp.array(1, dtype=jnp.int32),
            fire_cooldown=jnp.array(0, dtype=jnp.int32),
            proj_x=jnp.array(-1, dtype=jnp.int32),
            proj_y=jnp.array(-1, dtype=jnp.int32),
            proj_active=jnp.array(0, dtype=jnp.int32),
        )

        # Spawn 3 enemies at random world positions
        key, ek1 = jax.random.split(key)
        xs = jax.random.randint(ek1, (3,), 0, WORLD_W, dtype=jnp.int32)
        key, ek2 = jax.random.split(key)
        ys = jax.random.randint(ek2, (3,), 0, WORLD_H, dtype=jnp.int32)
        positions = jnp.stack([xs, ys], axis=1)
        enemies = EnemyState(
            positions=positions,
            hp=jnp.zeros((3,), dtype=jnp.int32),
            active=jnp.ones((3,), dtype=jnp.int32),
            enemy_type=jnp.zeros((3,), dtype=jnp.int32),
        )

        # Spawn some items (hearts and poison) randomly
        key, ik1 = jax.random.split(key)
        n_items = 6
        xs_i = jax.random.randint(ik1, (n_items,), 0, WORLD_W, dtype=jnp.int32)
        key, ik2 = jax.random.split(key)
        ys_i = jax.random.randint(ik2, (n_items,), 0, WORLD_H, dtype=jnp.int32)
        key, ik3 = jax.random.split(key)
        types = jax.random.randint(ik3, (n_items,), 0, 2, dtype=jnp.int32)
        # map 0->HEART, 1->POISON
        types = jnp.where(types == 0, ITEM_HEART, ITEM_POISON)
        items_positions = jnp.stack([xs_i, ys_i], axis=1)
        items_types = types
        items_active = jnp.ones((n_items,), dtype=jnp.int32)

        spawners = SpawnerState(
            positions=SPAWNER_POSITIONS,
            hp=jnp.zeros((SPAWNER_POSITIONS.shape[0],), dtype=jnp.int32),
            active=jnp.zeros((SPAWNER_POSITIONS.shape[0],), dtype=jnp.int32),
            cooldown=jnp.zeros((SPAWNER_POSITIONS.shape[0],), dtype=jnp.int32),
        )

        level = LevelState(
            tile_map=jnp.zeros((1, 1), dtype=jnp.int32),
            room_index=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            items_positions=items_positions,
            items_types=items_types,
            items_active=items_active,
        )

        state = DarkState(player=player, enemies=enemies, spawners=spawners, level=level, rng_key=key)
        obs = state
        return obs, state

    def step(self, state: DarkState, action: jnp.ndarray) -> Tuple[DarkState, DarkState, float, bool, None]:
        return self._step_impl(state, action)
    
    def _step_impl(self, state: DarkState, action: jnp.ndarray) -> Tuple[DarkState, DarkState, float, bool, None]:
        # Map Atari-style actions to simple movement using project constants (speed = 2 pixels/step)
        a = jnp.asarray(action)
        dx = jnp.where(a == Action.LEFT, -2, jnp.where(a == Action.RIGHT, 2, 0)).astype(jnp.int32)
        dy = jnp.where(a == Action.UP, -2, jnp.where(a == Action.DOWN, 2, 0)).astype(jnp.int32)

        prop_x = state.player.x + dx
        prop_y = state.player.y + dy
        
        # Clamp to world bounds
        prop_x = jnp.clip(prop_x, 0, WORLD_W - 1)
        prop_y = jnp.clip(prop_y, 0, WORLD_H - 1)
        
        # Get viewport camera origin (centered on player)
        cam_x = jnp.clip(state.player.x - GAME_W // 2, 0, WORLD_W - GAME_W).astype(jnp.int32)
        cam_y = jnp.clip(state.player.y - GAME_H // 2, 0, WORLD_H - GAME_H).astype(jnp.int32)
        
        # Collision detection in WORLD coordinates (walls are defined in world space)
        WALLS = self.renderer.WALLS
        pw, ph = 12, 12
        
        def collides(px, py):
            # Collision in world space
            wx = WALLS[:, 0]
            wy = WALLS[:, 1]
            ww = WALLS[:, 2]
            wh = WALLS[:, 3]
            overlap_x = (px <= (wx + ww - 1)) & ((px + pw - 1) >= wx)
            overlap_y = (py <= (wy + wh - 1)) & ((py + ph - 1) >= wy)
            return jnp.any(overlap_x & overlap_y)
        
        # Sequential axis collision in world space
        try_x = prop_x
        collide_x = collides(try_x, state.player.y)
        new_x = jnp.where(~collide_x, try_x, state.player.x)
        
        try_y = prop_y
        collide_y = collides(new_x, try_y)
        new_y = jnp.where(~collide_y, try_y, state.player.y)
        
        new_step = state.level.step_counter + 1
        
        # update enemy with simple random walk
        # Move all enemies with per-enemy random walk
        rng, subkey = jax.random.split(state.rng_key)
        n_en = state.enemies.positions.shape[0]
        # per-enemy dx,dy in {-1,0,1}
        d = jax.random.randint(subkey, (n_en, 2), -1, 2, dtype=jnp.int32)
        edx = d[:, 0]
        edy = d[:, 1]

        cur_ex = state.enemies.positions[:, 0]
        cur_ey = state.enemies.positions[:, 1]
        prop_ex = jnp.clip(cur_ex + edx, 0, WORLD_W - 1)
        prop_ey = jnp.clip(cur_ey + edy, 0, WORLD_H - 1)

        # enemy should not pass walls (also in world space) -- vectorized across enemies and walls
        e_pw, e_ph = ENEMY_W, ENEMY_H
        # WALLS shape (W,4); make comparisons (n_en, W)
        wx = WALLS[:, 0][None, :]
        wy = WALLS[:, 1][None, :]
        ww = WALLS[:, 2][None, :]
        wh = WALLS[:, 3][None, :]
        epx = prop_ex[:, None]
        epy = prop_ey[:, None]
        e_overlap_x = (epx <= (wx + ww - 1)) & ((epx + e_pw - 1) >= wx)
        e_overlap_y = (epy <= (wy + wh - 1)) & ((epy + e_ph - 1) >= wy)
        # any wall collision per enemy
        any_enemy_collide = jnp.any(e_overlap_x & e_overlap_y, axis=1)

        new_ex = jnp.where(~any_enemy_collide, prop_ex, cur_ex)
        new_ey = jnp.where(~any_enemy_collide, prop_ey, cur_ey)

        # update enemy positions array (n_en x 2)
        new_positions = jnp.stack([new_ex, new_ey], axis=1).astype(jnp.int32)
        new_enemies = EnemyState(positions=new_positions, hp=state.enemies.hp, active=state.enemies.active, enemy_type=state.enemies.enemy_type)

        # simple spawner behaviour: occasionally (every SPAWN_COOLDOWN frames) place enemy at a spawner
        rng, spawn_key = jax.random.split(rng)
        should_spawn = (state.level.step_counter % SPAWN_COOLDOWN) == 0
        # if spawning, pick a random spawner for each enemy and place them there
        spawner_idx = jax.random.randint(spawn_key, (n_en,), 0, SPAWNER_POSITIONS.shape[0], dtype=jnp.int32)
        spawn_positions = SPAWNER_POSITIONS[spawner_idx]
        # conditionally override all enemy positions with spawn positions
        new_positions = jnp.where(should_spawn, spawn_positions, new_positions)
        new_enemies = EnemyState(positions=new_positions, hp=new_enemies.hp, active=new_enemies.active, enemy_type=new_enemies.enemy_type)
        
        # Handle player-item pickups: items affect only player HP
        try:
            items_pos = state.level.items_positions
            items_ty = state.level.items_types
            items_act = state.level.items_active
            n_items = items_pos.shape[0]
            if n_items > 0:
                # compute overlap between player rect (new_x,new_y) and item rects
                ipx = items_pos[:, 0]
                ipy = items_pos[:, 1]
                overlap_x = (ipx <= (new_x + PLAYER_W - 1)) & ((ipx + ITEM_W - 1) >= new_x)
                overlap_y = (ipy <= (new_y + PLAYER_H - 1)) & ((ipy + ITEM_H - 1) >= new_y)
                picked = (items_act == 1) & overlap_x & overlap_y
                # count hearts and poisons picked
                hearts_picked = jnp.sum(picked & (items_ty == ITEM_HEART)).astype(jnp.int32)
                poisons_picked = jnp.sum(picked & (items_ty == ITEM_POISON)).astype(jnp.int32)
                # update player HP (clamp between 0 and 9)
                new_hp = jnp.clip(state.player.hp + hearts_picked - poisons_picked, 0, 9)
                # mark picked items inactive
                new_items_active = jnp.where(picked, jnp.zeros_like(items_act), items_act)
            else:
                new_hp = state.player.hp
                new_items_active = items_act
        except Exception:
            new_hp = state.player.hp
            new_items_active = state.level.items_active
        
        # build new nested state
        new_player = PlayerState(x=new_x, y=new_y, hp=new_hp, facing=state.player.facing, fire_cooldown=state.player.fire_cooldown, proj_x=state.player.proj_x, proj_y=state.player.proj_y, proj_active=state.player.proj_active)
        new_level = LevelState(tile_map=state.level.tile_map, room_index=state.level.room_index, step_counter=new_step, score=state.level.score, items_positions=state.level.items_positions, items_types=state.level.items_types, items_active=new_items_active)
        new_state = DarkState(player=new_player, enemies=new_enemies, spawners=state.spawners, level=new_level, rng_key=rng)
        obs = new_state
        reward = 0.0
        done = False
        info = None
        return obs, new_state, reward, done, info

    def render(self, state: DarkState) -> jnp.ndarray:
        return self.renderer.render(state)

    def step_with_info(self, state: DarkState, action: jnp.ndarray):
        """Non-JIT wrapper that calls step and prints detailed debug info."""
        # Convert action to string
        action_name = {
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "LEFT",
            4: "RIGHT",
            5: "DOWN",
        }.get(int(action), f"UNKNOWN({int(action)})")
        
        # Calculate proposed position based on action (use nested state)
        prop_x = int(state.player.x)
        prop_y = int(state.player.y)
        if action == 4:  # LEFT
            prop_x -= 1
        elif action == 3:  # RIGHT
            prop_x += 1
        elif action == 2:  # UP
            prop_y -= 1
        elif action == 5:  # DOWN
            prop_y += 1
        
        # Clamp to world bounds
        prop_x = max(0, min(prop_x, WORLD_W - 1))
        prop_y = max(0, min(prop_y, WORLD_H - 1))

        print(f"\n[Step {int(state.level.step_counter):3d}] Action={action_name:5s}")
        print(f"  World: ({int(state.player.x):3d}, {int(state.player.y):3d}) -> ({prop_x:3d}, {prop_y:3d})")
        
        # Check X collision (in world space)
        WALLS = self.renderer.WALLS
        pw, ph = 12, 12
        
        collide_x = False
        for i, wall in enumerate(WALLS):
            wx, wy, ww, wh = int(wall[0]), int(wall[1]), int(wall[2]), int(wall[3])
            ox = (prop_x <= (wx + ww - 1)) and ((prop_x + pw - 1) >= wx)
            oy = (int(state.player.y) <= (wy + wh - 1)) and ((int(state.player.y) + ph - 1) >= wy)
            if ox and oy:
                print(f"  X-COLLISION with wall {i}: [{wx}, {wy}, {ww}, {wh}]")
                collide_x = True
                break
        
        # If X collides, don't move X
        final_x = int(state.player.x) if collide_x else prop_x
        
        # Check Y collision with final X (in world space)
        collide_y = False
        for i, wall in enumerate(WALLS):
            wx, wy, ww, wh = int(wall[0]), int(wall[1]), int(wall[2]), int(wall[3])
            ox = (final_x <= (wx + ww - 1)) and ((final_x + pw - 1) >= wx)
            oy = (prop_y <= (wy + wh - 1)) and ((prop_y + ph - 1) >= wy)
            if ox and oy:
                print(f"  Y-COLLISION with wall {i}: [{wx}, {wy}, {ww}, {wh}]")
                collide_y = True
                break
        
        print(f"  Collide X={collide_x}, Collide Y={collide_y}")
        
        # Call the JAX step (which does its own collision detection)
        obs, new_state, reward, done, info = self.step(state, action)
        
        # Report resulting nested-state player coords
        print(f"  JAX Result: ({int(new_state.player.x):3d}, {int(new_state.player.y):3d})")
        
        return obs, new_state, reward, done, info

    def action_space(self) -> spaces.Discrete:
        # Provide the full Atari action set so all directional actions are supported
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "x": spaces.Box(low=0, high=GAME_W - 1, shape=(), dtype=jnp.int32),
            "y": spaces.Box(low=0, high=GAME_H - 1, shape=(), dtype=jnp.int32),
            "step": spaces.Box(low=0, high=10**9, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(GAME_H, GAME_W, 3), dtype=jnp.uint8)

    def _get_observation(self, state: DarkState) -> DarkState:
        return state

    def obs_to_flat_array(self, obs: DarkState) -> jnp.ndarray:
        return jnp.array([obs.player.x, obs.player.y, obs.level.step_counter], dtype=jnp.int32)

    def _get_info(self, state: DarkState) -> None:
        return None

    def _get_reward(self, previous_state: DarkState, state: DarkState) -> float:
        return 0.0

    def _get_done(self, state: DarkState) -> bool:
        return False



