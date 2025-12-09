# Re-parse with rollover-aware logic:
# - Discard entries when the frame number repeats *consecutively*
# - Use a unique sequential entry_id for each kept entry (frame included as a field)
# - Plot as before

import re, json
from math import atan2, pi
from collections import OrderedDict
from statistics import mean
import numpy as np

from numpy import std
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import sqrt, cos, sin

# Configuration: if STATIC_COORDS is True, transform enemy positions by undoing
# the player's movement so displayed coordinates are in a static/world frame.
# Tweak `MOVE_SPEED` and `ANGULAR_SPEED` to match the player's in-game movement.
STATIC_COORDS = True
# linear movement per frame when holding UP/DOWN (units matching parsed X/Z)
MOVE_SPEED = 0.25 - 0.00195308333
# angular speed in radians per frame when holding LEFT/RIGHT
ANGULAR_SPEED = np.pi/270 #+0.00005 #0.0115 #2*np.pi / 270.0 - 0.0115 # 0.023271

# TODO: the enemy position is being counter-rotated too much? or maybe around the wrong point? fix this

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
    # Parse optional INPUTS line (format added by battlezone_dev.py):
    # Example: "INPUTS: UP=True DOWN=False LEFT=False RIGHT=True FIRE=False"
    m_inputs = re.search(r'(?m)^INPUTS:\s*(.*)$', entry)
    if m_inputs:
        inputs_str = m_inputs.group(1).strip()
        inputs = {}
        for tok in inputs_str.split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                v_bool = v.strip().lower() in ('true', '1', 't', 'yes', 'y')
                inputs[k.strip()] = v_bool
        # Attach parsed inputs under the 'INPUTS' key so downstream code can use it
        kv['INPUTS'] = inputs
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
events = []

def add_event(frame, message):
    sort_key = frame if frame is not None else float("inf")
    events.append((sort_key, len(events), message))

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
    fire0_z = kv.get('fire0_Z (0xDE)')
    fire1_z = kv.get('fire1_Z (0xD8)')
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
        "fire0_Z": fire0_z,
        "fire1_Z": fire1_z,
        # attach parsed inputs (if any) as a dict of booleans under 'inputs'
        "inputs": kv.get('INPUTS', {}),
    }


# Integrate player inputs to compute a running player world position + heading
# and optionally transform enemy positions into a static/world coordinate frame
# by undoing the player's movement. We iterate in chronological order of
# collected entries and update each `entries[uid]` in-place with a new
# `player` dict and (when STATIC_COORDS) with transformed AX/AZ/BX/BZ values.
px = 0.0
pz = 0.0
theta = np.pi / 2  # facing "up" along positive Z axis initially
prev_frame_for_player = None

# only consider every other frame since enemy position is only updated every 2 frames
skip_frame = True
prev_entry = None
rotation_frames = 0

test_x=[]
test_z=[]

for uid, entry in entries.items():
    skip_frame = not skip_frame
    if skip_frame:
        entries[uid] = prev_entry
        continue
    frame = entry.get("frame")
    # delta frames since last sample (fallback to 1 when unknown)
    if prev_frame_for_player is None or frame is None:
        dt = 1
    else:
        dt = int(frame - prev_frame_for_player)
        if dt <= 0:
            dt = 1
    prev_frame_for_player = frame if frame is not None else prev_frame_for_player
    prev_entry = entry

    inputs = entry.get("inputs", {}) or {}
    left = bool(inputs.get("LEFT", False))
    right = bool(inputs.get("RIGHT", False))
    up = bool(inputs.get("UP", False))
    down = bool(inputs.get("DOWN", False))

    if left or right:
        rotation_frames += dt

    # Apply rotation (LEFT/RIGHT) then forward movement (UP/DOWN)
    if left:
        theta += ANGULAR_SPEED * dt
    if right:
        theta -= ANGULAR_SPEED * dt

    theta = theta % (2 * np.pi)

    forward = 0.0
    if up:
        forward += MOVE_SPEED * dt
    if down:
        forward -= MOVE_SPEED * dt

    # Move in the current heading
    px += forward * cos(theta)
    pz += forward * sin(theta)

    # Attach player state to the entry
    entry["player"] = {"x": px, "z": pz, "theta": theta}

    print(np.sqrt((entry["AX"]-0.150)**2 + (entry["AZ"]-0.398)**2)) # center: 0.150, 0.398

    if 50 < frame < 500:
        test_x.append(entry["AX"])
        test_z.append(entry["AZ"])

    # If requested, transform enemy coordinates from player-relative to world coords
    if STATIC_COORDS:
        def _to_world(lx, lz):
            # Treat logged coords as (lx, lz) where lz is forward (positive Z is
            # the player's facing direction) and lx is lateral (positive to the
            # player's right). World coords = player_pos + lx*right_vec + lz*forward_vec
            if lx is None or lz is None:
                return lx, lz
            
            # lx = lx - 0.15018681344924312
            # lz = lz - 0.3978339525029426
            # print(0.99994619679746**rotation_frames)
            # lx = lx * 0.99994619679746**rotation_frames # correct for drift
            # lz = lz * 0.99994619679746**rotation_frames

            wx = px + sin(theta) * lx + cos(theta) * lz
            wz = pz - cos(theta) * lx + sin(theta) * lz

            return wx, wz

        if entry.get("AX") is not None and entry.get("AZ") is not None:
            wx, wz = _to_world(entry["AX"], entry["AZ"])
            entry["AX"], entry["AZ"] = wx, wz
        if entry.get("BX") is not None and entry.get("BZ") is not None:
            wx, wz = _to_world(entry["BX"], entry["BZ"])
            entry["BX"], entry["BZ"] = wx, wz


def fit_circle(xs, zs):
    """
    Fit a circle to points (xs, zs) using linear least squares.
    Returns:
        cx, cz  - center of circle
        r       - radius
        rms_err - root-mean-square distance error to the fitted circle
    """
    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)

    if xs.shape != zs.shape:
        raise ValueError("xs and zs must have the same shape")
    if xs.size < 3:
        raise ValueError("At least 3 points are required to fit a circle")

    # Build linear system: x^2 + z^2 + D x + E z + F = 0
    A = np.column_stack([xs, zs, np.ones_like(xs)])
    b = -(xs**2 + zs**2)

    # Solve for D, E, F in least squares sense
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]

    # Circle parameters
    cx = -D / 2.0
    cz = -E / 2.0
    r = np.sqrt((D**2 + E**2) / 4.0 - F)

    # Compute RMS error: distances to center minus radius
    dists = np.sqrt((xs - cx)**2 + (zs - cz)**2)
    rms_err = np.sqrt(np.mean((dists - r)**2))

    return cx, cz, r, rms_err


cx, cz, r, err = fit_circle(test_x, test_z)
print(f"Fitted circle: center=({cx:.3f}, {cz:.3f}), radius={r:.3f}, RMS error={err:.3f}")

import numpy as np

def analyze_shape(xs, zs):
    xs = np.asarray(xs, float)
    zs = np.asarray(zs, float)

    # 1) Fit circle to get approximate center (from previous function)
    cx, cz, r, _ = fit_circle(xs, zs)

    ux = xs - cx
    uz = zs - cz
    r_i = np.sqrt(ux**2 + uz**2)
    phi_i = np.arctan2(uz, ux)

    # 2) Fit r^2 = a + b*cos(phi) + c*sin(phi)
    y = r_i**2
    A = np.column_stack([np.ones_like(phi_i), np.cos(phi_i), np.sin(phi_i)])
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]

    y_hat = A @ np.array([a, b, c])
    residual = np.sqrt(np.mean((y - y_hat)**2))

    return {
        "center": (cx, cz),
        "mean_radius": np.mean(r_i),
        "r_var": np.var(r_i),
        "sinusoid_fit_coeffs": (a, b, c),
        "sinusoid_r2_residual": residual,
    }

dict_ = analyze_shape(test_x, test_z)
print("Shape analysis:")
for k, v in dict_.items():
    print(f"  {k}: {v}")

# Prepare series in chronological order of kept entries
uids = list(entries.keys())

def log_player_shoot_events(entries, uids):
    """Detect when missile Z positions drop to a low value and then start rising again."""
    # Each tracker remembers the last observed value, and arms itself once it sees a drop
    # into the low window (≤ threshold). When the next frame shows an increase we emit
    # a player_shoot event keyed to the reset frame that started the rise.
    trackers = {
        "fire0_Z": {"last": None, "armed": False, "reset_frame": None, "reset_val": None},
        "fire1_Z": {"last": None, "armed": False, "reset_frame": None, "reset_val": None},
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
                    add_event(
                        tracker["reset_frame"],
                        f"player_shoot {name}: fr: {tracker['reset_frame']}, "
                        f"z: {tracker['reset_val']} --> fr: {frame}, z: {value}"
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
    if delta <= 5:
        return
    add_event(
        curr_entry["frame"],
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

for _, _, message in sorted(events, key=lambda e: (e[0], e[1])):
    print(message)

ax = [entries[u]["AX"] for u in uids]
az = [entries[u]["AZ"] for u in uids]
bx = [entries[u]["BX"] for u in uids]
bz = [entries[u]["BZ"] for u in uids]
frames = [entries[u]["frame"] for u in uids]

# player trajectory (aggregated from inputs)
pxs = [entries[u].get("player", {}).get("x") for u in uids]
pzs = [entries[u].get("player", {}).get("z") for u in uids]

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
    row_heights=[0.7, 0.18, 0.12]
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
if STATIC_COORDS:
    fig.add_trace(go.Scatter(
        x=pxs,
        y=pzs,
        mode="lines+markers",
        name="Player",
        marker=dict(color="black", symbol="circle", size=6),
        customdata=frames,
        hovertemplate="Player<br>X: %{x}<br>Z: %{y}<br>Frame: %{customdata}<extra></extra>"
    ), row=1, col=1)
# fig.add_trace(go.Histogram(x=angle_changes, nbinsx=140, name="Δangle per step"), row=2, col=1)
# fig.add_trace(go.Histogram(x=distance_changes, nbinsx=140, name="Distance per step"), row=3, col=1)

fig.update_xaxes(title_text="X", row=1, col=1)
fig.update_yaxes(title_text="Z", row=1, col=1)
fig.update_xaxes(title_text="Δangle (radians)", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_xaxes(title_text="Distance", row=3, col=1)
fig.update_yaxes(title_text="Count", row=3, col=1)

# Enforce a 1:1 data-to-pixel aspect ratio for the first subplot's axes.
# Use Plotly's `scaleanchor` mechanism and compute an appropriate figure
# height so the first subplot's pixels reflect a unit X == unit Z relationship.
try:
    x_min, x_max = (min(ax), max(ax)) if ax else (0.0, 1.0)
    y_min, y_max = (min(az), max(az)) if az else (0.0, 1.0)
    x_span = x_max - x_min
    y_span = y_max - y_min
    print(f"x_span: {x_span}, y_span: {y_span}, x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
except Exception:
    x_span = y_span = 1.0

# Choose a target plot width (pixels). This can be increased if you want a larger
# image that may extend beyond the page; user allowed that.
target_width = 1000

# Avoid blowing up when x_span is effectively zero. If x_span is tiny we expand
# the X range to match the Y span so we can still enforce a 1:1 ratio without
# producing extremely tall figures. If both spans are tiny, fall back to a
# sensible default figure size.
_EPS = 1e-6
if y_span <= _EPS:
    # nothing meaningful to show, use default layout
    fig_height = 900
else:
    # If x_span is effectively zero, expand it to match the y_span centered on
    # the current x center so that the aspect ratio remains 1:1.
    if x_span <= _EPS:
        x_center = 0.5 * (x_min + x_max)
        x_min = x_center - 0.5 * y_span
        x_max = x_center + 0.5 * y_span
        x_span = x_max - x_min

    # The pixel height required for the first subplot to have 1:1 is:
    subplot_px_height = int(target_width * (y_span / x_span))
    # Clamp the computed subplot pixel height to avoid extreme sizes
    subplot_px_height = max(200, min(subplot_px_height, 3000))
    # Since the first subplot takes ~70% of the figure height (row_heights[0]),
    # compute the figure height so that that fraction equals the needed subplot px.
    fraction_first = 0.7
    computed_fig_height = int(subplot_px_height / fraction_first)
    # Add some padding for the other subplots and margins
    fig_height = max(900, computed_fig_height + 200)
    # Now it's safe to enforce 1:1 data-to-pixel aspect
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

fig.update_layout(width=target_width, height=fig_height, showlegend=True)

fig.show()
# print(mean(angle_changes), "radians per step")


# TODO: find second enemy respawn delay
