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
