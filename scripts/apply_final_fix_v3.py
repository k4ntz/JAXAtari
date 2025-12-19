
import os
import re

file_path = "src/jaxatari/games/jax_beamrider.py"
with open(file_path, "r") as f:
    text = f.read()

# Read the new tables (short versions)
with open("tables_short.txt", "r") as f:
    new_tables_code = f.read()

# 1. Replace the tables
# Find everything from BLUE_LINE_INIT_TABLE to before class WhiteUFOPattern
table_pattern = r"BLUE_LINE_INIT_TABLE = .*?\n\n"
text = re.sub(table_pattern, new_tables_code + "\n", text, flags=re.DOTALL)

# 2. Update is_init logic in step()
# Change len(BLUE_LINE_INIT_TABLE) to 128 (or keep it dynamic if I update the tables)
# Actually, the tables ARE updated, so len(BLUE_LINE_INIT_TABLE) will be 128.

with open(file_path, "w") as f:
    f.write(text)

