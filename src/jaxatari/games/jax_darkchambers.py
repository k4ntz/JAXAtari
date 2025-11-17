from turtle import width
from typing import NamedTuple, Tuple
import chex
import jax, jax.numpy as jnp
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer

class DarkChambersConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210 
    tile_size: int = 8  
    map_cols: int = 20
    map_rows: int = 22 

    player_width: int = 6
    player_height: int = 8
    player_speed: float = 1.0
    player_proj_speed: float = 2.0
    player_fire_cooldown: int = 12
    player_invuln_frames: int = 20
    player_hit_inset_x: int = 1
    player_hit_inset_y: int = 1

    enemy_width: int = 6
    enemy_height: int = 8
    enemy_speed: float = 0.75
    max_enemies_on_screen: int = 8
    enemy_spawn_cooldown: int = 60
    enemy_hp_grunt: int = 1
    enemy_hp_spawner: int = 3

# -------- Entity Classes --------

class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array
class PlayerState(NamedTuple):
    x: chex.Array
    y: chex.Array
    hp: chex.Array
    facing: chex.Array               # -1 = left, +1 = right
    fire_cooldown: chex.Array        # frames until next shot allowed

    # player projectile
    proj_x: chex.Array
    proj_y: chex.Array
    proj_active: chex.Array

class EnemyState(NamedTuple):
    positions: chex.Array          # shape (MAX_ENEMIES, 2)
    hp: chex.Array                 # shape (MAX_ENEMIES,)
    active: chex.Array             # shape (MAX_ENEMIES,)
    enemy_type: chex.Array         # shape (MAX_ENEMIES,) 

class SpawnerState(NamedTuple):
    positions: chex.Array          # shape (MAX_SPAWNERS, 2)
    hp: chex.Array                 # shape (MAX_SPAWNERS,)
    active: chex.Array             # shape (MAX_SPAWNERS,)
    cooldown: chex.Array           # shape (MAX_SPAWNERS,)

class LevelState(NamedTuple):
    tile_map: chex.Array           # shape (map_rows, map_cols)
    room_index: chex.Array         # which dungeon room
    step_counter: chex.Array
    score: chex.Array

class DarkChambersState(NamedTuple):
    player: PlayerState
    enemies: EnemyState
    spawners: SpawnerState
    level: LevelState
    rng_key: chex.PRNGKey


