
import os
import re

file_path = "src/jaxatari/games/jax_beamrider.py"
with open(file_path, "r") as f:
    text = f.read()

# Read the new tables (240 versions)
with open("tables_240.txt", "r") as f:
    new_tables_code = f.read()

# 1. Replace the tables
table_pattern = r"BLUE_LINE_INIT_TABLE = .*?\nBLUE_LINE_LOOP_TABLE = .*?\]\)\n"
text = re.sub(table_pattern, new_tables_code + "\n", text, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(text)

