"""
authors: Paula Troszt, Ernst Christian Böhringer, Aiman Sammy Rahlf
"""

import jax.numpy as jnp
from typing import NamedTuple
import chex

class LevelConstants(NamedTuple):
    enemy_size_per_rotation: chex.Array
    level_boss_size: chex.Array
    enemy_speed_per_rotation: chex.Array
    initial_enemies: chex.Array
    enemy_rotation_probability: chex.Array
    min_attack_delay: chex.Array # identical for all levels
    max_attack_delay: chex.Array # identical for all levels

TimePilot_Level_1 = LevelConstants( # 1910
    enemy_size_per_rotation = jnp.array([
        (8, 9), # up
        (8, 10),
        (8, 7), # left
        (8, 10),
        (8, 9), # down
        (8, 10),
        (8, 7), # right
        (8, 10)
    ]),
    level_boss_size = jnp.array([8, 12]), # size does not change with rotation
    enemy_speed_per_rotation = jnp.array([
        (0, -2), # up
        (-2, -2),
        (-2, 0), # left
        (-2, 2),
        (0, 2), # down
        (2, 2),
        (2, 0), # right
        (2, -2)
    ]),
    initial_enemies = jnp.array([
        (134, -13, 0, 1),
        (-17, 199, 0, 1),
        (49, 214, 0, 1),
        (-27, 23, 0, 1)
    ]),
    enemy_rotation_probability = 60,
    min_attack_delay = 0,
    max_attack_delay = 300
)

TimePilot_Level_2 = LevelConstants( # 1940
    enemy_size_per_rotation = jnp.array([
        (7, 9), # up
        (8, 7),
        (8, 6), # left
        (8, 7),
        (7, 9), # down
        (8, 7),
        (8, 6), # right
        (8, 7)
    ]),
    level_boss_size = jnp.array([8, 12]), # size does not change with rotation
    enemy_speed_per_rotation = jnp.array([
        (0, -3), # up
        (-3, -3),
        (-3, 0), # left
        (-3, 3),
        (0, 3), # down
        (3, 3),
        (3, 0), # right
        (3, -3)
    ]),
    initial_enemies = jnp.array([
        (176, -27, 0, 1),
        (-36, 130, 0, 1),
        (125, 220, 0, 1),
        (47, -13, 0, 1)
    ]),
    enemy_rotation_probability = 60,
    min_attack_delay = 0,
    max_attack_delay = 300
)

TimePilot_Level_3 = LevelConstants( # 1970
    enemy_size_per_rotation = jnp.array([
        (7, 8), # up
        (8, 9),
        (8, 7), # left
        (8, 9),
        (7, 8), # down
        (8, 9),
        (8, 7), # right
        (8, 9)
    ]),
    level_boss_size = jnp.array([8, 10]), # size does not change with rotation
    enemy_speed_per_rotation = jnp.array([
        (0, -2), # uo
        (-2, -2),
        (-2, 0), # left
        (-2, 2),
        (0, 2), # down
        (2, 2),
        (2, 0), # right
        (2, -2)
    ]),
    initial_enemies = jnp.array([
        (157, 7, 0, 1),
        (202, 143, 0, 1),
        (207, 182, 0, 1),
        (113, 1, 0, 1)
    ]),
    enemy_rotation_probability = 25,
    min_attack_delay = 0,
    max_attack_delay = 300
)

TimePilot_Level_4 = LevelConstants( # 1983
    enemy_size_per_rotation = jnp.array([
        (7, 9), # up
        (8, 7),
        (8, 6), # left
        (8, 7),
        (7, 9), # down
        (8, 7),
        (8, 6), # right
        (8, 7)
    ]),
    level_boss_size = jnp.array([8, 10]), # size does not change with rotation
    enemy_speed_per_rotation = jnp.array([
        (0, -5), # up
        (-5, -5),
        (-5, 0), # left
        (-5, 5),
        (0, 5), # down
        (5, 5),
        (5, 0), # right
        (5, -5)
    ]),
    initial_enemies = jnp.array([
        (-39, 77, 0, 1),
        (-5, 226, 0, 1),
        (-28, 188, 0, 1),
        (169, 223, 0, 1)
    ]),
    enemy_rotation_probability = 60,
    min_attack_delay = 0,
    max_attack_delay = 300
)

TimePilot_Level_5 = LevelConstants( # 2001
    enemy_size_per_rotation = jnp.array([
        (8, 7), # up
        (8, 7),
        (8, 7), # left
        (8, 7),
        (8, 7), # down
        (8, 7),
        (8, 7), # right
        (8, 7)
    ]),
    level_boss_size = jnp.array([8, 12]), # size does not change with rotation
    enemy_speed_per_rotation = jnp.array([
        (0, -5), # up
        (-5, -5),
        (-5, 0), # left
        (-5, 5),
        (0, 5), # down
        (5, 5),
        (5, 0), # right
        (5, -5)
    ]),
    initial_enemies = jnp.array([
        (129, 1, 0, 1),
        (46, 202, 0, 1),
        (-32, 52, 0, 1),
        (-22, 91, 0, 1)
    ]),
    enemy_rotation_probability = 25,
    min_attack_delay = 0,
    max_attack_delay = 300
)
