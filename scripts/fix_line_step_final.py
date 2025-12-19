
import os
import re

file_path = "src/jaxatari/games/jax_beamrider.py"
with open(file_path, "r") as f:
    text = f.read()

# Robustly replace _line_step
pattern = r"    def _line_step\(self, state: BeamriderState\):.*?return positions, counter"
replacement = """    def _line_step(self, state: BeamriderState):
        counter = state.level.blue_line_counter + 1
        
        # Determine current table and index
        # Transition from INIT to LOOP
        # Use length-1 to avoid out of bounds during the very last frame of init
        is_init = counter < len(BLUE_LINE_INIT_TABLE)
        
        def get_init_pos(c):
            return BLUE_LINE_INIT_TABLE[jnp.minimum(c, len(BLUE_LINE_INIT_TABLE) - 1)]
            
        def get_loop_pos(c):
            loop_idx = (c - len(BLUE_LINE_INIT_TABLE)) % len(BLUE_LINE_LOOP_TABLE)
            return BLUE_LINE_LOOP_TABLE[loop_idx]
            
        positions = jax.lax.cond(is_init, get_init_pos, get_loop_pos, counter)
        
        return positions, counter"""

new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(new_text)
