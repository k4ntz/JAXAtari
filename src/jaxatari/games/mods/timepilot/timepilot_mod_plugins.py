import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.environment import JAXAtariAction as Action

from jaxatari.games.timepilot_levels import (
    TimePilot_Level_1,
    TimePilot_Level_2,
    TimePilot_Level_3,
    TimePilot_Level_4,
    TimePilot_Level_5
)

def get_ordered_asset_config(order):
    """
    Returns a declarative asset manifest with levels in a custom order.
    """
    all_player_sprites_files = []
    for i in order:
        all_player_sprites_files.extend([
            *(f'L{i}/L{i}_Player_Pos{j}.npy' for j in range(8)),
            f'L{i}/L{i}_Player_Death1.npy', f'L{i}/L{i}_Player_Death2.npy',
        ])
    # Add transition sprites
    all_player_sprites_files.extend([
        *(f'L-All/TP_Player_Pos{i}.npy' for i in range(8)),
        'L-All/TP_Player_Death.npy',
    ])

    all_enemy_sprites_files = []
    for i in order:
        if i == 3:
             all_enemy_sprites_files.extend([
                *(f'L3/L3_Enemy_Pos{j}.npy' for j in ["01", "02", "11", "12", "21", "22", "31", "32", "41", "42"]),
                'L3/L3_Enemy_Death.npy',
                *(f'L3/L3_Boss_Pos{j}.npy' for j in ["01", "02", "11", "12"]),
             ])
        elif i == 5:
            all_enemy_sprites_files.extend([
                *(f'L5/L5_Enemy_Pos{j}.npy' for k in range(5) for j in range(2)),
                'L5/L5_Enemy_Death.npy',
                'L5/L5_Boss_Pos0.npy', 'L5/L5_Boss_Pos0.npy',
                'L5/L5_Boss_Pos1.npy', 'L5/L5_Boss_Pos1.npy',
            ])
        else:
            all_enemy_sprites_files.extend([
                *(f'L{i}/L{i}_Enemy_Pos{j}.npy' for j in range(8)),
                f'L{i}/L{i}_Enemy_Pos0.npy', f'L{i}/L{i}_Enemy_Pos1.npy',
                f'L{i}/L{i}_Enemy_Death.npy',
                f'L{i}/L{i}_Boss_Pos0.npy', f'L{i}/L{i}_Boss_Pos0.npy',
                f'L{i}/L{i}_Boss_Pos1.npy', f'L{i}/L{i}_Boss_Pos1.npy',
            ])

    return (
        # Procedural background (empty black screen)
        {'name': 'background', 'type': 'background', 'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)},
        # Procedural pixel to ensure white is in the palette
        {'name': 'white_pixel', 'type': 'procedural', 'data': jnp.array([[[255,255,255,255]]], dtype=jnp.uint8)},
        # General Sprites (Single)
        {'name': 'top_wall', 'type': 'single', 'file': 'L-All/Top.npy'},
        {'name': 'bottom_wall', 'type': 'single', 'file': 'L-All/Bottom.npy'},
        {'name': 'respawn_bottom_wall', 'type': 'single', 'file': 'L-All/Respawn_Bottom.npy'},
        {'name': 'start_screen', 'type': 'single', 'file': 'L-All/First.npy'},
        {'name': 'player_life', 'type': 'single', 'file': 'L-All/Player_Life.npy'},
        {'name': 'black_line', 'type': 'single', 'file': 'L-All/BlackLine.npy'},
        # General Sprites (Group)
        {'name': 'transition_bar', 'type': 'group', 'files': ['L-All/TeleportBar.npy', 'L-All/TeleportBar2.npy']},
        # General Sprites (Digits)
        {'name': 'digits', 'type': 'digits', 'pattern': 'L-All/Digit{}.npy'},
        # --- Level-Dependent Groups ---
        {'name': 'all_clouds', 'type': 'group', 'files': [f'L{i}/L{i}_Cloud.npy' for i in order]},
        {'name': 'all_backgrounds', 'type': 'group', 'files': [f'L{i}/L{i}_Background.npy' for i in order]},
        {'name': 'all_respawn_top_walls', 'type': 'group', 'files': [f'L{i}/L{i}_Top.npy' for i in order]},
        {'name': 'all_player_missiles', 'type': 'group', 'files': [f'L{i}/L{i}_Player_Bullet.npy' for i in order]},
        {'name': 'all_enemy_missiles', 'type': 'group', 'files': [f'L{i}/L{i}_Enemy_Bullet.npy' for i in order]},
        {'name': 'all_enemy_remaining', 'type': 'group', 'files': [
            item for i in order for item in (f'L{i}/L{i}_Enemy_Life.npy', f'L{i}/L{i}_Enemy_Death_Life.npy')
        ]},
        # Massive groups
        {'name': 'all_player_sprites', 'type': 'group', 'files': all_player_sprites_files},
        {'name': 'all_enemy_sprites', 'type': 'group', 'files': all_enemy_sprites_files},
    )

class ReverseChronologyMod(JaxAtariInternalModPlugin):
    """Changes the order of appearance of the enemies by reversing the level sequence (reverse chronology)."""
    name = "reverse_chronology"
    
    _levels = [
        TimePilot_Level_1,
        TimePilot_Level_2,
        TimePilot_Level_3,
        TimePilot_Level_4,
        TimePilot_Level_5
    ]
    
    _order = [5, 4, 3, 2, 1]
    
    constants_overrides = {
        "LEVEL_1": _levels[_order[0]-1],
        "LEVEL_2": _levels[_order[1]-1],
        "LEVEL_3": _levels[_order[2]-1],
        "LEVEL_4": _levels[_order[3]-1],
        "LEVEL_5": _levels[_order[4]-1],
        "ASSET_CONFIG": get_ordered_asset_config(_order)
    }

class InstantTurnMod(JaxAtariInternalModPlugin):
    """Directly places the plane in the direction given by the action instead of progressively turning it, including diagonals."""
    name = "instant_turn"

    # Override the game's ACTION_SET to include diagonal actions
    attribute_overrides = {
        "ACTION_SET": jnp.array(
            [
                Action.NOOP,
                Action.FIRE,
                Action.UP,
                Action.RIGHT,
                Action.LEFT,
                Action.DOWN,
                Action.UPRIGHT,
                Action.UPLEFT,
                Action.DOWNRIGHT,
                Action.DOWNLEFT,
                Action.UPFIRE,
                Action.RIGHTFIRE,
                Action.LEFTFIRE,
                Action.DOWNFIRE,
                Action.UPRIGHTFIRE,
                Action.UPLEFTFIRE,
                Action.DOWNRIGHTFIRE,
                Action.DOWNLEFTFIRE,
            ],
            dtype=jnp.int32,
        )
    }

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state_player_rotation,
        action
    ):
        # get pressed buttons
        left = jnp.logical_or(jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE),
                              jnp.logical_or(jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE),
                                             jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE)))
        right = jnp.logical_or(jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE),
                               jnp.logical_or(jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE),
                                              jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)))
        up = jnp.logical_or(jnp.logical_or(action == Action.UP, action == Action.UPFIRE),
                            jnp.logical_or(jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE),
                                           jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE)))
        down = jnp.logical_or(jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE),
                              jnp.logical_or(jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE),
                                             jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)))

        # determine new rotation according to action (up=0, up-left=1, left=2, down-left=3, down=4, down-right=5, right=6, up-right=7)
        new_rotation = jax.lax.cond(
            up,
            lambda: jax.lax.cond(left, lambda: 1, lambda: jax.lax.cond(right, lambda: 7, lambda: 0)),
            lambda: jax.lax.cond(
                down,
                lambda: jax.lax.cond(left, lambda: 3, lambda: jax.lax.cond(right, lambda: 5, lambda: 4)),
                lambda: jax.lax.cond(
                    left, lambda: 2, lambda: jax.lax.cond(right, lambda: 6, lambda: state_player_rotation)
                )
            )
        )

        return jax.lax.cond(
            jnp.logical_or(jnp.logical_or(up, down), jnp.logical_or(right, left)),
            lambda: new_rotation,
            lambda: state_player_rotation
        )

class DontKillMod(JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin):
    """Punishes for killing and shooting enemies."""
    name = "dont_kill"
    constants_overrides = {
        "POINTS_PER_ENEMY": -1000,
        "POINTS_PER_BOSS": -5000
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Punish for shooting
        # In TimePilot, player_missile_state[3] is the step counter for the missile.
        # If it's > 0, the missile is active and flying.
        is_shooting = new_state.player_missile_state[3] > 0
        shooting_punishment = jnp.where(is_shooting, 2, 0).astype(jnp.int32)
        
        # Reward 10 points every 100 frames
        survival_reward = jnp.where(
            jnp.logical_and(new_state.step_counter > 0, new_state.step_counter % 100 == 0), 
            10, 
            0
        ).astype(jnp.int32)
        
        new_score = new_state.score - shooting_punishment + survival_reward
        
        return new_state.replace(score=new_score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
        # Calculate signed difference directly as score is int32 and won't wrap around
        diff = state.score - previous_state.score
        return diff.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state, state):
        return self._get_reward(previous_state, state)

class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod for TimePilot: black background and green everything."""
    name = "matrix_theme"
    constants_overrides = {
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_PLAYER': (255, 255, 255), # Keeping player white for visibility
        'RGB_ENEMY': (0, 200, 0),
        'RGB_BOSS': (0, 255, 0),
        'RGB_CLOUD': (0, 100, 0),
        'RGB_MISSILE': (0, 255, 100),
        'RGB_UI': (0, 255, 0),
        'RGB_SCORE': (0, 0, 0),
        'RGB_LIVES': (0, 0, 0),
        'RGB_BOTTOM_WALL': (0, 0, 0),
    }

class ExtraLivesMod(JaxAtariPostStepModPlugin):
    """Start with many extra lives."""
    name = "extra_lives"
    
    constants_overrides = {
        "INITIAL_LIVES": 9,
        "MAX_LIVES": 9
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        return obs, state.replace(lives=jnp.array(9, dtype=jnp.int32))

