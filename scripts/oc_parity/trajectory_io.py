"""Save/load OCAtari trajectory NPZ files (objects + RAM + frames + actions)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Dense object fields stored as parallel arrays of shape (T, N).
OBJECT_INT_FIELDS = (
    "x",
    "y",
    "w",
    "h",
    "prev_x",
    "prev_y",
    "dx",
    "dy",
    "visible",
    "value",
    "ten",
    "rgb_r",
    "rgb_g",
    "rgb_b",
)


def serialize_oc_object(obj: Any) -> Dict[str, Any]:
    """Extract every useful field from an OCAtari GameObject (or NoObject)."""
    category = getattr(obj, "category", "NoObject")
    rgb = getattr(obj, "rgb", (0, 0, 0))
    if rgb is None:
        rgb = (0, 0, 0)
    entry: Dict[str, Any] = {
        "category": str(category),
        "x": int(getattr(obj, "x", 0)),
        "y": int(getattr(obj, "y", 0)),
        "w": int(getattr(obj, "w", 0)),
        "h": int(getattr(obj, "h", 0)),
        "prev_x": int(getattr(obj, "prev_x", getattr(obj, "x", 0))),
        "prev_y": int(getattr(obj, "prev_y", getattr(obj, "y", 0))),
        "dx": int(getattr(obj, "dx", 0)),
        "dy": int(getattr(obj, "dy", 0)),
        "visible": bool(getattr(obj, "visible", True)),
        "rgb": (int(rgb[0]), int(rgb[1]), int(rgb[2])),
        "orientation": getattr(obj, "orientation", None),
    }
    if hasattr(obj, "value"):
        entry["value"] = int(getattr(obj, "value"))
    else:
        entry["value"] = -1
    if hasattr(obj, "ten"):
        entry["ten"] = int(getattr(obj, "ten"))
    else:
        entry["ten"] = -1
    return entry


def serialize_oc_objects(objects: Sequence[Any]) -> List[Dict[str, Any]]:
    return [serialize_oc_object(o) for o in objects]


def _objects_to_arrays(
    frames_objects: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, np.ndarray]:
    if not frames_objects:
        raise ValueError("Cannot serialize empty object trajectory")

    n_frames = len(frames_objects)
    n_slots = max(len(frame) for frame in frames_objects)
    categories = np.empty((n_frames, n_slots), dtype=object)
    orientations = np.empty((n_frames, n_slots), dtype=object)
    arrays: Dict[str, np.ndarray] = {
        field: np.zeros((n_frames, n_slots), dtype=np.int32)
        for field in OBJECT_INT_FIELDS
    }

    for t, frame_objs in enumerate(frames_objects):
        for i in range(n_slots):
            if i < len(frame_objs):
                obj = frame_objs[i]
            else:
                obj = {
                    "category": "NoObject",
                    "x": 0,
                    "y": 0,
                    "w": 0,
                    "h": 0,
                    "prev_x": 0,
                    "prev_y": 0,
                    "dx": 0,
                    "dy": 0,
                    "visible": False,
                    "rgb": (0, 0, 0),
                    "orientation": None,
                    "value": -1,
                    "ten": -1,
                }
            categories[t, i] = str(obj.get("category", "NoObject"))
            orientations[t, i] = obj.get("orientation", None)
            rgb = obj.get("rgb", (0, 0, 0)) or (0, 0, 0)
            arrays["x"][t, i] = int(obj.get("x", 0))
            arrays["y"][t, i] = int(obj.get("y", 0))
            arrays["w"][t, i] = int(obj.get("w", 0))
            arrays["h"][t, i] = int(obj.get("h", 0))
            arrays["prev_x"][t, i] = int(obj.get("prev_x", obj.get("x", 0)))
            arrays["prev_y"][t, i] = int(obj.get("prev_y", obj.get("y", 0)))
            arrays["dx"][t, i] = int(obj.get("dx", 0))
            arrays["dy"][t, i] = int(obj.get("dy", 0))
            arrays["visible"][t, i] = int(bool(obj.get("visible", True)))
            arrays["value"][t, i] = int(obj.get("value", -1))
            arrays["ten"][t, i] = int(obj.get("ten", -1))
            arrays["rgb_r"][t, i] = int(rgb[0])
            arrays["rgb_g"][t, i] = int(rgb[1])
            arrays["rgb_b"][t, i] = int(rgb[2])

    out = {f"obj_{k}": v for k, v in arrays.items()}
    out["obj_category"] = categories
    out["obj_orientation"] = orientations
    return out


def _arrays_to_objects(data: Mapping[str, np.ndarray]) -> List[List[Dict[str, Any]]]:
    categories = data["obj_category"]
    n_frames, n_slots = categories.shape
    orientations = data.get("obj_orientation")
    frames: List[List[Dict[str, Any]]] = []
    for t in range(n_frames):
        frame: List[Dict[str, Any]] = []
        for i in range(n_slots):
            orient = None
            if orientations is not None:
                orient = orientations[t, i]
                # numpy object arrays may store None as None, or as float nan-like.
                if isinstance(orient, float) and np.isnan(orient):
                    orient = None
            frame.append(
                {
                    "category": str(categories[t, i]),
                    "x": int(data["obj_x"][t, i]),
                    "y": int(data["obj_y"][t, i]),
                    "w": int(data["obj_w"][t, i]),
                    "h": int(data["obj_h"][t, i]),
                    "prev_x": int(data["obj_prev_x"][t, i]),
                    "prev_y": int(data["obj_prev_y"][t, i]),
                    "dx": int(data["obj_dx"][t, i]),
                    "dy": int(data["obj_dy"][t, i]),
                    "visible": bool(data["obj_visible"][t, i]),
                    "value": int(data["obj_value"][t, i]),
                    "ten": int(data["obj_ten"][t, i]),
                    "rgb": (
                        int(data["obj_rgb_r"][t, i]),
                        int(data["obj_rgb_g"][t, i]),
                        int(data["obj_rgb_b"][t, i]),
                    ),
                    "orientation": orient,
                }
            )
        frames.append(frame)
    return frames


def save_trajectory(
    path: str,
    *,
    meta: Mapping[str, Any],
    actions: Sequence[int],
    rewards: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    ram: Sequence[Sequence[int]],
    objects: Sequence[Sequence[Mapping[str, Any]]],
    frames: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """Write a trajectory NPZ with meta_json + dense arrays."""
    n = len(actions)
    if not (len(rewards) == len(terminated) == len(truncated) == len(ram) == len(objects) == n):
        raise ValueError(
            "actions/rewards/terminated/truncated/ram/objects must share the same length"
        )

    payload: Dict[str, Any] = {
        "meta_json": np.asarray(json.dumps(dict(meta), separators=(",", ":"))),
        "actions": np.asarray(actions, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "ram": np.asarray(ram, dtype=np.uint8),
    }
    payload.update(_objects_to_arrays(objects))
    if frames is not None:
        if len(frames) != n:
            raise ValueError("frames length must match actions length")
        payload["frames"] = np.asarray(frames, dtype=np.uint8)

    meta_out = dict(meta)
    meta_out["n_frames"] = n
    meta_out["has_frames"] = frames is not None
    payload["meta_json"] = np.asarray(json.dumps(meta_out, separators=(",", ":")))

    np.savez_compressed(path, **payload)


def load_trajectory(path: str) -> Dict[str, Any]:
    """Load a trajectory NPZ into a dict with decoded objects and meta."""
    with np.load(path, allow_pickle=True) as raw:
        data = {k: raw[k] for k in raw.files}

    meta = json.loads(str(data["meta_json"]))
    objects = _arrays_to_objects(data)
    traj: Dict[str, Any] = {
        "meta": meta,
        "actions": np.asarray(data["actions"], dtype=np.int32),
        "rewards": np.asarray(data["rewards"], dtype=np.float32),
        "terminated": np.asarray(data["terminated"], dtype=np.bool_),
        "truncated": np.asarray(data["truncated"], dtype=np.bool_),
        "ram": np.asarray(data["ram"], dtype=np.uint8),
        "objects": objects,
        "frames": np.asarray(data["frames"], dtype=np.uint8) if "frames" in data else None,
    }
    return traj


def get_frame_objects(traj: Mapping[str, Any], t: int) -> List[Dict[str, Any]]:
    return list(traj["objects"][t])


def find_first_ball_frame(traj: Mapping[str, Any]) -> Optional[int]:
    for t, frame_objs in enumerate(traj["objects"]):
        for obj in frame_objs:
            if obj.get("category") == "Ball" and int(obj.get("w", 0)) > 0:
                return t
    return None


def summarize_trajectory(traj: Mapping[str, Any]) -> str:
    meta = traj["meta"]
    n = int(meta.get("n_frames", len(traj["actions"])))
    ball_t = find_first_ball_frame(traj)
    lines = [
        f"game={meta.get('game')} seed={meta.get('seed')} frames={n}",
        f"frameskip={meta.get('frameskip')} oc_mode={meta.get('oc_mode')}",
        f"has_frames={traj['frames'] is not None} first_ball_frame={ball_t}",
    ]
    sample_ts = [0]
    if ball_t is not None:
        sample_ts.append(ball_t)
    mid = n // 2
    if mid not in sample_ts:
        sample_ts.append(mid)
    for t in sample_ts:
        if t >= n:
            continue
        cats = [
            f"{o['category']}@({o['x']},{o['y']}) dy={o['dy']}"
            for o in traj["objects"][t]
            if o["category"] != "NoObject"
        ]
        lines.append(f"  t={t}: " + ", ".join(cats))
    return "\n".join(lines)
