import os
from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

@dataclass
class GameConfig:
    """Game configuration parameters"""

    screen_width: int = 160
    screen_height: int = 210
    shark_speeds: List[float] = None
    fish_speed: float = 1
    fish_x: List[int] = None
    fish_y: List[int] = None
    top_border: int = 30
    bottom_border: int = 180
    num_rows: int = 6
    row_spacing: int = 16
    row_borders: List[int] = None


    def __post_init__(self):
        if self.row_borders is None:
            self.row_borders = [
                self.top_border + i * self.row_spacing for i in range(self.num_rows)
            ]


class GameState(NamedTuple):
        """Game state representation"""
        player1_rod_x: chex.Array
        player1_rod_y: chex.Array
        player2_rod_x: chex.Array
        player2_rod_y: chex.Array
        score: chex.Array
        shark_x: chex.Array
        fish_x: chex.Array
        time = chex.Array

class EntityPositions(NamedTuple):
    """Positions of entities in the game"""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class FishingDerbyObservation(NamedTuple):
    """Observation structure for Fishing Derby"""
    player1_rod: EntityPositions
    player2_rod: EntityPositions
    shark: EntityPositions
    fish: EntityPositions
    score: jnp.ndarray

class FishingDerbyInfo(NamedTuple):
    time: jnp.ndarray

class FishingDerby(JaxEnvironment[GameState, FishingDerbyObservation, FishingDerbyInfo]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()

    def reset(self) -> Tuple[FishingDerbyObservation, GameState]:

        for row in range(self.config.num_rows):
            row_y = self.config.fish_y[row]

        state = GameState(
            player1_rod_x=jnp.array(0.0),
            player1_rod_y=jnp.array(0.0),
            player2_rod_x=jnp.array(0.0),
            player2_rod_y=jnp.array(0.0),
            score=jnp.array([0, 0]),
            shark_x=jnp.array([self.config.screen_width // 2]),
            fish_x=jnp.array(self.config.fish_x or [self.config.screen_width // 2]),
        )
        return self._get_observation(state), state

    # test