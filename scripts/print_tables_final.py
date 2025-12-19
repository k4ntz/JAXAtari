
import numpy as np

init = np.load("beamrider_init_final.npy")
loop = np.load("beamrider_loop_final.npy")

print(f"BLUE_LINE_INIT_TABLE = jnp.array({init.tolist()})")
print(f"BLUE_LINE_LOOP_TABLE = jnp.array({loop.tolist()})")
