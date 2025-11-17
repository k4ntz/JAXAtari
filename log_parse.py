# Re-parse with rollover-aware logic:
# - Discard entries when the frame number repeats *consecutively*
# - Use a unique sequential entry_id for each kept entry (frame included as a field)
# - Plot as before

import re, json
from math import atan2, pi
from collections import OrderedDict
from statistics import mean

from numpy import std
import plotly.graph_objects as go
from plotly.subplots import make_subplots


with open("ram_log.txt", "r") as f:
    log_text = f.read()


def to_int16_signed(hi: int, lo: int) -> int:
    value = (hi << 8) | lo
    if value & 0x8000:
        value = value - 0x10000
    return value

# Split into blocks
raw_entries = [blk.strip() for blk in re.split(r'(?m)^\s*---\s*$', log_text) if blk.strip()]
entries_blocks = [e for e in raw_entries if not e.startswith('--- LOG START')]

pattern = re.compile(r'^(.*?):\s*(-?\d+)\s*$', re.M)

kept = []
last_frame = object()  # sentinel distinct from ints
for entry in entries_blocks:
    kv = dict((k.strip(), int(v)) for k, v in pattern.findall(entry))
    frame_id = kv.get('frame (0x80)')
    if frame_id == last_frame:
        # discard consecutive duplicate frame number
        continue
    last_frame = frame_id
    kept.append(kv)

# Build mapping keyed by unique entry_id
entries = OrderedDict()
for uid, kv in enumerate(kept):
    # print(uid, kv)
    a_x_hi = kv.get('enemy_a_X_hi (0xC3)')
    a_x_lo = kv.get('enemy_a_X_lo (0xC4)')
    a_z_hi = kv.get('enemy_a_Z_hi (0xC5)')
    a_z_lo = kv.get('enemy_a_Z_lo (0xC6)')
    b_x_hi = kv.get('enemy_b_X_hi (0xCB)')
    b_x_lo = kv.get('enemy_b_X_lo (0xCC)')
    b_z_hi = kv.get('enemy_b_Z_hi (0xCD)')
    b_z_lo = kv.get('enemy_b_Z_lo (0xCE)')
    AX = to_int16_signed(a_x_hi, a_x_lo)
    AZ = to_int16_signed(a_z_hi, a_z_lo)
    BX = to_int16_signed(b_x_hi, b_x_lo)
    BZ = to_int16_signed(b_z_hi, b_z_lo)
    entries[uid] = {"frame": kv.get('frame (0x80)'), "AX": AX, "AZ": AZ, "BX": BX, "BZ": BZ}


# Prepare series in chronological order of kept entries
uids = list(entries.keys())
ax = [entries[u]["AX"] for u in uids]
az = [entries[u]["AZ"] for u in uids]
bx = [entries[u]["BX"] for u in uids]
bz = [entries[u]["BZ"] for u in uids]

rel_frames = [entries[u] for u in uids]
# print(f"{len(rel_frames)=}")
print([rel_frames[u]["AX"] for u in range(len(rel_frames))])

# deltas = [rel_frames[i+1]["AZ"] - rel_frames[i]["AZ"] for i in range(len(rel_frames)-1)]
# print(f"{mean(deltas)=}, {std(deltas)=}")
# print(deltas)

def wrap_signed_pi(x):
    while x <= -pi: x += 2*pi
    while x >  pi:  x -= 2*pi
    return x

# angle of position vector from origin at each kept time step
def angles_from_origin(xs, zs):
    # optional: skip undefined angles exactly at origin
    return [atan2(z, x) for x, z in zip(xs, zs)]

# change in that angle between consecutive time steps
def angle_deltas_from_angles(angles):
    return [wrap_signed_pi(angles[i+1] - angles[i]) for i in range(len(angles)-1)]

# usage with your arrays ax, az, bx, bz:
a_pos_angles = angles_from_origin(ax, az)
b_pos_angles = angles_from_origin(bx, bz)

a_deltas = angle_deltas_from_angles(a_pos_angles)
b_deltas = angle_deltas_from_angles(b_pos_angles)

angle_changes = a_deltas + b_deltas  # for the histogram
angle_changes = [el for el in b_deltas if el > -0.1 and el != 0] # filter

# Plot
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
    subplot_titles=("XZ positions over time", "Distribution of change in angle between steps (radians)"),
    row_heights=[0.6, 0.4]
)

fig.add_trace(go.Scatter(x=ax, y=az, mode="lines+markers",
                         name="Enemy A", marker=dict(symbol="x", size=8)), row=1, col=1)
fig.add_trace(go.Scatter(x=bx, y=bz, mode="lines+markers",
                         name="Enemy B", marker=dict(symbol="x", size=8)), row=1, col=1)
fig.add_trace(go.Histogram(x=angle_changes, nbinsx=140, name="Δangle per step"), row=2, col=1)

fig.update_xaxes(title_text="X", row=1, col=1)
fig.update_yaxes(title_text="Z", row=1, col=1)
fig.update_xaxes(title_text="Δangle (radians)", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_layout(height=700, showlegend=True)

fig.show()
# print(mean(angle_changes), "radians per step")