
import os

file_path = "src/jaxatari/games/jax_beamrider.py"
with open(file_path, "r") as f:
    lines = f.readlines()

# Read the new tables
with open("tables_final.txt", "r") as f:
    new_tables_code = f.read()

new_lines = []
skip = False
for line in lines:
    if line.startswith("BLUE_LINE_TABLE = "):
        new_lines.append(new_tables_code + "\n")
        skip = False # It was a single line
        continue
    
    if "positions = BLUE_LINE_TABLE[counter]" in line:
        # Replace the _line_step logic
        new_lines.append("        # Determine current table and index\n")
        new_lines.append("        # Transition from INIT to LOOP\n")
        new_lines.append("        is_init = counter < len(BLUE_LINE_INIT_TABLE)\n")
        new_lines.append("        \n")
        new_lines.append("        def get_init_pos():\n")
        new_lines.append("            return BLUE_LINE_INIT_TABLE[counter]\n")
        new_lines.append("            \n")
        new_lines.append("        def get_loop_pos():\n")
        new_lines.append("            loop_idx = (counter - len(BLUE_LINE_INIT_TABLE)) % len(BLUE_LINE_LOOP_TABLE)\n")
        new_lines.append("            return BLUE_LINE_LOOP_TABLE[loop_idx]\n")
        new_lines.append("            \n")
        new_lines.append("        positions = jax.lax.cond(is_init, get_init_pos, get_loop_pos)\n")
        continue
        
    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)
