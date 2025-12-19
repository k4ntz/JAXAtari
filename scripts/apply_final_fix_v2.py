
import os

file_path = "src/jaxatari/games/jax_beamrider.py"
with open(file_path, "r") as f:
    lines = f.readlines()

# Read the new tables
with open("tables_done.txt", "r") as f:
    new_tables_code = f.read()

new_lines = []
skip_old_tables = False
for line in lines:
    if line.startswith("BLUE_LINE_INIT_TABLE = "):
        skip_old_tables = True
        new_lines.append(new_tables_code + "\n")
        continue
    if line.startswith("BLUE_LINE_LOOP_TABLE = "):
        skip_old_tables = True # Still skipping
        continue
    if skip_old_tables and line.startswith("class "):
        skip_old_tables = False
    
    if skip_old_tables:
        continue

    if "def _line_step(self, state: BeamriderState):" in line:
        new_lines.append(line)
        new_lines.append("        counter = state.level.blue_line_counter + 1\n")
        new_lines.append("        is_init = counter < len(BLUE_LINE_INIT_TABLE)\n")
        new_lines.append("        def get_init_pos(c):\n")
        new_lines.append("            return BLUE_LINE_INIT_TABLE[c]\n")
        new_lines.append("        def get_loop_pos(c):\n")
        new_lines.append("            loop_idx = (c - len(BLUE_LINE_INIT_TABLE)) % len(BLUE_LINE_LOOP_TABLE)\n")
        new_lines.append("            return BLUE_LINE_LOOP_TABLE[loop_idx]\n")
        new_lines.append("        positions = jax.lax.cond(is_init, get_init_pos, get_loop_pos, counter)\n")
        new_lines.append("        return positions, counter\n")
        # Need to skip the old _line_step body
        continue # I will use a different approach to skip
    
    new_lines.append(line)

# Wait, the previous logic for skipping _line_step body is flawed.
# Let's do a more robust string replacement.
full_text = "".join(new_lines)

# Robustly find _line_step and replace its body
import re
pattern = r"    def _line_step\(self, state: BeamriderState\):.*?return positions, counter"
replacement = """    def _line_step(self, state: BeamriderState):
        counter = state.level.blue_line_counter + 1
        is_init = counter < len(BLUE_LINE_INIT_TABLE)
        def get_init_pos(c):
            return BLUE_LINE_INIT_TABLE[c]
        def get_loop_pos(c):
            loop_idx = (c - len(BLUE_LINE_INIT_TABLE)) % len(BLUE_LINE_LOOP_TABLE)
            return BLUE_LINE_LOOP_TABLE[loop_idx]
        positions = jax.lax.cond(is_init, get_init_pos, get_loop_pos, counter)
        return positions, counter"""

full_text = re.sub(pattern, replacement, full_text, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(full_text)

