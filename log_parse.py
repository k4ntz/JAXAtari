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
from math import sqrt


with open("ram_log.txt", "r") as f:
    log_text = f.read()


def hilo_to_unit(hi: int, lo: int) -> int:
    value = (hi << 8) | lo
    if value & 0x8000:
        value = value - 0x10000
    return value / 256.0

# Split into blocks
raw_entries = [blk.strip() for blk in re.split(r'(?m)^\s*---\s*$', log_text) if blk.strip()]
entries_blocks = [e for e in raw_entries if not e.startswith('--- LOG START')]

pattern = re.compile(r'^(.*?):\s*(-?\d+)\s*$', re.M)
FIRE_RESET_THRESHOLD = 5

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
frame_offset = 0
prev_frame = None
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
    fire0_x = kv.get('fire0_X (0xDC)')
    fire1_x = kv.get('fire1_X (0xD6)')
    AX = hilo_to_unit(a_x_hi, a_x_lo)
    AZ = hilo_to_unit(a_z_hi, a_z_lo)
    BX = hilo_to_unit(b_x_hi, b_x_lo)
    BZ = hilo_to_unit(b_z_hi, b_z_lo)
    frame_value = kv.get('frame (0x80)')
    if prev_frame is not None and frame_value is not None and frame_value == 0 and prev_frame == 255:
        # frame counter wrapped back to 0, so bump the offset for subsequent frames
        frame_offset += 256
    prev_frame = frame_value if frame_value is not None else prev_frame
    normalized_frame = (frame_value + frame_offset) if frame_value is not None else None
    entries[uid] = {
        "frame": normalized_frame,
        "AX": AX,
        "AZ": AZ,
        "BX": BX,
        "BZ": BZ,
        "fire0_X": fire0_x,
        "fire1_X": fire1_x,
    }


# Prepare series in chronological order of kept entries
uids = list(entries.keys())

def log_player_shoot_events(entries, uids):
    trackers = {
        "fire0_X": {"last": None, "armed": False, "reset_frame": None, "reset_val": None},
        "fire1_X": {"last": None, "armed": False, "reset_frame": None, "reset_val": None},
    }
    for uid in uids:
        entry = entries[uid]
        frame = entry["frame"]
        for name, tracker in trackers.items():
            value = entry.get(name)
            if value is None:
                continue
            last_val = tracker["last"]
            if last_val is None:
                tracker["last"] = value
                continue
            if not tracker["armed"]:
                if value <= FIRE_RESET_THRESHOLD and last_val > FIRE_RESET_THRESHOLD:
                    tracker["armed"] = True
                    tracker["reset_frame"] = frame
                    tracker["reset_val"] = value
            else:
                if value > last_val:
                    print(
                        f"player_shoot {name}: fr: {tracker['reset_frame']}, "
                        f"x: {tracker['reset_val']} --> fr: {frame}, x: {value}"
                    )
                    tracker["armed"] = False
                    tracker["reset_frame"] = None
                    tracker["reset_val"] = None
                elif value > FIRE_RESET_THRESHOLD and value <= last_val:
                    tracker["armed"] = False
                    tracker["reset_frame"] = None
                    tracker["reset_val"] = None
            tracker["last"] = value

log_player_shoot_events(entries, uids)

def log_position_change(label, prev_entry, curr_entry, x_key, z_key):
    prev_x, prev_z = prev_entry[x_key], prev_entry[z_key]
    curr_x, curr_z = curr_entry[x_key], curr_entry[z_key]
    delta = sqrt((curr_x - prev_x) ** 2 + (curr_z - prev_z) ** 2)
    if delta <= 2:
        return
    print(
        f"{label}:    fr: {prev_entry['frame']}, "
        f"x: {prev_x:.3f}, z: {prev_z:.3f} --> "
        f"fr: {curr_entry['frame']}, x: {curr_x:.3f}, z: {curr_z:.3f}"
    )

for i in range(len(uids) - 1):
    prev_entry = entries[uids[i]]
    curr_entry = entries[uids[i + 1]]
    if prev_entry["AX"] != curr_entry["AX"] or prev_entry["AZ"] != curr_entry["AZ"]:
        log_position_change("enemyA", prev_entry, curr_entry, "AX", "AZ")
    if prev_entry["BX"] != curr_entry["BX"] or prev_entry["BZ"] != curr_entry["BZ"]:
        log_position_change("enemyB", prev_entry, curr_entry, "BX", "BZ")

ax = [entries[u]["AX"] for u in uids]
az = [entries[u]["AZ"] for u in uids]
bx = [entries[u]["BX"] for u in uids]
bz = [entries[u]["BZ"] for u in uids]
frames = [entries[u]["frame"] for u in uids]

rel_frames = [entries[u] for u in uids]
# print(f"{len(rel_frames)=}")
# print([rel_frames[u]["AZ"] for u in range(len(rel_frames))])

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

def step_distances(xs, zs):
    return [sqrt((xs[i+1] - xs[i])**2 + (zs[i+1] - zs[i])**2) for i in range(len(xs) - 1)]

# usage with your arrays ax, az, bx, bz:
a_pos_angles = angles_from_origin(ax, az)
b_pos_angles = angles_from_origin(bx, bz)

a_deltas = angle_deltas_from_angles(a_pos_angles)
b_deltas = angle_deltas_from_angles(b_pos_angles)

angle_changes = a_deltas + b_deltas  # for the histogram
angle_changes = [el for el in b_deltas if el > -0.1 and el != 0] # filter

distance_changes = step_distances(ax, az) + step_distances(bx, bz)
distance_changes = [el for el in distance_changes if 0 < el < 100]  # filter

# Plot
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.12,
    subplot_titles=(
        "XZ positions over time",
        "Distribution of change in angle between steps (radians)",
        "Distribution of Euclidean distance traveled per step"
    ),
    row_heights=[0.55, 0.25, 0.2]
)

fig.add_trace(go.Scatter(
    x=ax,
    y=az,
    mode="lines+markers",
    name="Enemy A",
    marker=dict(symbol="x", size=8),
    customdata=frames,
    hovertemplate="Enemy A<br>X: %{x}<br>Z: %{y}<br>Frame: %{customdata}<extra></extra>"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=bx,
    y=bz,
    mode="lines+markers",
    name="Enemy B",
    marker=dict(symbol="x", size=8),
    customdata=frames,
    hovertemplate="Enemy B<br>X: %{x}<br>Z: %{y}<br>Frame: %{customdata}<extra></extra>"
), row=1, col=1)
fig.add_trace(go.Histogram(x=angle_changes, nbinsx=140, name="Δangle per step"), row=2, col=1)
fig.add_trace(go.Histogram(x=distance_changes, nbinsx=140, name="Distance per step"), row=3, col=1)

fig.update_xaxes(title_text="X", row=1, col=1)
fig.update_yaxes(title_text="Z", row=1, col=1)
fig.update_xaxes(title_text="Δangle (radians)", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_xaxes(title_text="Distance", row=3, col=1)
fig.update_yaxes(title_text="Count", row=3, col=1)
fig.update_layout(height=900, showlegend=True)

fig.show()
# print(mean(angle_changes), "radians per step")


# TODO: find second enemy respawn delay