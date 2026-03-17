import jax
import jax.numpy as jnp

# This is an example of how you can replace the hardcoded treasure rendering
# with a clean, compiled JAX loop inside your TutankhamRenderer.render method.

def example_render_snippet(self, state, raster, camera_offset, ZERO_FLIP, is_onscreen):
    
    # 5. Render Treasures
    def render_single_item(i, current_raster):
        # Extract properties for the i-th item
        item_x = state.item_states[i][0]
        item_y = state.item_states[i][1]
        item_type = state.item_states[i][2]
        is_active = state.item_states[i][3] == 1
        
        # Dynamically select the correct mask from your newly created "treasure" group
        item_mask = self.SHAPE_MASKS["treasure"][item_type.astype(jnp.int32)]
        
        # Conditionally render if the item is active AND on screen
        return jax.lax.cond(
            is_active & is_onscreen(item_y, 8, camera_offset),
            lambda r: self.jr.render_at_clipped(
                r,
                item_x,
                item_y - camera_offset,
                item_mask,
                flip_offset=ZERO_FLIP
            ),
            lambda r: r,
            current_raster
        )

    # jax.lax.fori_loop works like: for i in range(lower, upper): carry = body_fun(i, carry)
    # We loop from index 0 up to 7 (exclusive), passing the 'raster' through each iteration.
    raster = jax.lax.fori_loop(0, 7, render_single_item, raster)
    
    return raster
