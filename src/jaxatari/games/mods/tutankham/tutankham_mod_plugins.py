import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_tutankham import TutankhamState
from jaxatari.games.jax_tutankham import can_walk_to



class NightModeMod(JaxAtariPostStepModPlugin):
    pass


class MimicModeMod(JaxAtariPostStepModPlugin):
    pass


class UpsideDownMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "MAP_CHECKPOINTS": jnp.array([
            # MAP 1
            [
                [588, 800, 18, 684],
                [405, 587, 80, 586],
                [201, 404, 12, 403],
                [0,   200, 78, 199],
            ],
            # MAP 2
            [
                [573, 800, 19, 634],
                [425, 572, 24, 572],
                [261, 426, 78, 426],
                [0,   260, 78, 259],
            ],
            # MAP 3
            [
                [553, 800,  107, 715],
                [401, 552,  98,  550],
                [269, 400,  78,  396],
                [0,   268,  39,  248],
            ],
            # MAP 4
            [
                [531, 800, 77,  719],
                [391, 532, 119, 531],
                [204, 392, 18,  391],
                [0 ,  203, 30,  203],
            ],
        ], dtype=jnp.int32),

        "MAP_GOAL_POSITIONS": jnp.array([
            [[134, 61]],  # MAP 1
            [[136, 60]],  # MAP 2
            [[16,  93]],  # MAP 3
            [[82,  95]]   # MAP 4
        ], dtype=jnp.int32)
    }


class MovingItemsMod(JaxAtariInternalModPlugin):
    
    # add item direction to the initial item states (to calculate movement)
    constants_overrides = {
        "MAP_ITEMS": jnp.array([
            # Level 1 (MAP 1)
            [
                [51, 87, 0, 1, 0],   # KEY_MAP1=0       [x, y, item_type, active, direction]
                [99, 183, 5, 1, 0],  # CROWN_02_MAP1=5
                [68, 262, 2, 1, 0],  # RING_MAP1=2
                [7, 311, 3, 1, 0],   # RUBY_MAP1=3
                [93, 382, 4, 1, 0],  # CHALICE_MAP1=4
                [18, 494, 1, 1, 0],  # CROWN_01_MAP1=1
                [0, 0, 0, 0, 0],     # Padding
            ],
            # Level 2 (MAP 2)
            [
                [21, 272, 6, 1, 0],  # KEY_MAP2=6
                [44, 155, 8, 1, 0],  # CROWN_MAP2=8
                [128, 98, 7, 1, 0],  # RING_MAP2=7
                [37, 406, 9, 1, 0],  # EMERALD_MAP2=9
                [91, 482, 10, 1, 0], # GOBLET_MAP2=10
                [23, 547, 11, 1, 0], # BUST_MAP2=11
                [0, 0, 0, 0, 0],     # Padding
            ],
            # Level 3 (MAP 3)
            [
                [22, 411, 12, 1, 0], # KEY_MAP3=12
                [15, 173, 14, 1, 0], # RING_MAP3=14
                [128, 98, 13, 1, 0], # TRIDENT_MAP3=13
                [17, 278, 15, 1, 0], # HERB_MAP3=15
                [108, 323, 16, 1, 0],# DIAMOND_MAP3=16
                [27, 656, 17, 1, 0], # CANDELABRA_MAP3=17
                [0, 0, 0, 0, 0],     # Padding
            ],
            # Level 4 (MAP 4)
            [
                [144, 110, 18, 1, 0], # KEY_MAP4=18
                [125, 221, 19, 1, 0], # RING_MAP4=19
                [117, 269, 20, 1, 0], # AMULET_MAP4=20
                [19, 326, 21, 1, 0],  # FAN_MAP4=21
                [55, 510, 23, 1, 0],  # ZIRCON_MAP4=23
                [110, 401, 22, 1, 0], # CRYSTAL_MAP4=22
                [66, 607, 24, 1, 0],  # DAGGER_MAP4=24
            ],
        ], dtype=jnp.int32),
    }

    @partial(jax.jit, static_argnums=(0,))
    def item_step(self, item_states, level, rng_key):
        """Moves active items using random-walk pathing"""

        def move_item(item, rng_key):
            item_x, item_y, item_type, active, direction = item
            
            # Natural patrol: randomly change direction --------------------------------
            change_probability = 0.08
            possible_directions = jnp.array([-1, -2, 1, 2]) # right, left, down, up

            rng_key, subkey_01, subkey_02 = jax.random.split(rng_key, 3)
            random_dir = jax.random.choice(subkey_01, possible_directions)
            should_change = jax.random.bernoulli(subkey_02, p=change_probability)

            new_direction = jnp.where(should_change, random_dir, direction)
            new_direction = jnp.where(direction == 0, random_dir, new_direction)

            # Indices: 0, 1(Down), 2(Up), -1(Right), -2(Left)
            # mapping array where the index matches the direction value
            lookup_x = jnp.array([0, 0,  0, -1, 1])
            lookup_y = jnp.array([0, 1, -1, 0,  0])
            x_direction = lookup_x[new_direction]
            y_direction = lookup_y[new_direction]

            # move creature
            new_x = item_x +  x_direction * active
            new_y = item_y +  y_direction * active           
            item_x, item_y, is_walkable = can_walk_to(self._env.consts.ITEM_SIZES[item_type], new_x, new_y, item_x, item_y, self._env.consts.VALID_POS_MAPS[level%4])
    
            #--------------------------------------------------------------------------

            return jnp.array([item_x, item_y, item_type, active, new_direction], dtype=jnp.int32), rng_key

        keys = jax.random.split(rng_key, len(item_states) + 1)
        rng_key = keys[0]
        new_item_states, _ = jax.vmap(move_item)(item_states, keys[1:])

        
        return new_item_states, rng_key

    