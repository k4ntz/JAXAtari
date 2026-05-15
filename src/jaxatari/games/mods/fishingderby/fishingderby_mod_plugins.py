import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariPostStepModPlugin

class SharkNoMovementEasyMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(shark_x=jnp.array(105.0, dtype=jnp.float32))

class SharkNoMovementMiddleMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(shark_x=jnp.array(75.0, dtype=jnp.float32))

class SharkTeleportMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        new_x = jnp.where(
            new_state.shark_x == 100.0,
            25.0,
            jnp.where(
                new_state.shark_x == 30.0,
                105.0,
                new_state.shark_x
            )
        )
        return new_state.replace(shark_x=new_x.astype(jnp.float32))

class FishOnPlayerSideMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        fish_x = new_state.fish_positions[:, 0]
        new_fish_x = jnp.where(fish_x > 86.0, 44.0, fish_x)
        return new_state.replace(
            fish_positions=new_state.fish_positions.at[:, 0].set(new_fish_x.astype(jnp.float32))
        )

class FishOnDifferentSidesMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        fish_x = new_state.fish_positions[:, 0]
        indices = jnp.arange(6)
        
        # i % 2 == 1 -> odd. fish > 86 -> 44
        # i % 2 == 0 -> even. fish < 70 -> 116
        new_fish_x = jnp.where(
            (indices % 2 == 1) & (fish_x > 86.0),
            44.0,
            jnp.where(
                (indices % 2 == 0) & (fish_x < 70.0),
                116.0,
                fish_x
            )
        )
        return new_state.replace(
            fish_positions=new_state.fish_positions.at[:, 0].set(new_fish_x.astype(jnp.float32))
        )

class FishInMiddleMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        fish_x = new_state.fish_positions[:, 0]
        p1_hooked = new_state.p1.hooked_fish_idx
        p2_hooked = new_state.p2.hooked_fish_idx
        indices = jnp.arange(6)
        
        # not hooked means indices != p1_hooked and indices != p2_hooked
        not_hooked = (indices != p1_hooked) & (indices != p2_hooked)
        
        new_fish_x = jnp.where(
            not_hooked & (fish_x < 70.0),
            86.0,
            jnp.where(
                not_hooked & (fish_x > 86.0),
                70.0,
                fish_x
            )
        )
        return new_state.replace(
            fish_positions=new_state.fish_positions.at[:, 0].set(new_fish_x.astype(jnp.float32))
        )
