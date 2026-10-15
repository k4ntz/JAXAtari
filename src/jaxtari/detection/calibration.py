"""One-off calibration that maps a detector's classes and boxes to the fields of a game's observation.

The detector checkpoints do not always match the current registry (several were trained on older class
ids / label geometry), so the wrapper calibrates itself once at construction: it runs the game for a few
hundred random steps, runs the detector on the rendered frames and matches the detections to the
ground-truth label boxes. Per observation field it learns

* the class id the detector actually predicts for it (majority vote over matched detections; fields that
  are not reliably matched fall back to the detector's class names, then the registry class_id),
* where its objects appear on screen (position prior used to route detections when several fields share a
  class, together with the label box size),
* the detector's systematic centre offset w.r.t. the label box (removed before mapping boxes back),
* the true sprite size (recovers x/y/width/height of objects whose labels were padded to the 8 px floor).

Ground truth is only used here, never at inference.
"""
import jax
import numpy as np

from jaxtari.environment import ObjectObservation

from . import labels

MIN_MATCH_FRACTION = 0.2  # calibrated class accepted if its matches cover >= 20% of the field's label boxes


def field_names(obs):
    return list(obs.__dataclass_fields__) if hasattr(obs, "__dataclass_fields__") else list(obs._fields)


def _fallback_class(name, entry, model_names):
    by_name = {v: k for k, v in model_names.items()}
    if name in by_name:
        return by_name[name], "name"
    cls = entry.get("class_id")
    return (cls, "registry") if isinstance(cls, int) and cls in model_names else (None, None)


def _iou(d, box):
    px, py, pw, ph = box
    ix = max(0.0, min(d["x2"], px + pw) - max(d["x1"], px))
    iy = max(0.0, min(d["y2"], py + ph) - max(d["y1"], py))
    inter = ix * iy
    return inter / ((d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) + pw * ph - inter + 1e-9)


def calibrate(base_env, objects_cfg, model_names, detect_fn, steps=300, seed=0, max_pool=False):
    """Returns (fields, template_obs). ``detect_fn(frame) -> [dict(cls, conf, x1, y1, x2, y2)]`` in game px.

    ``fields`` maps observation field name -> dict with the routing / inversion parameters used by the
    wrapper (kind: "oo" ObjectObservation, "xy" (N, 2) positions, "grid" tile grid, "unsupported").
    """
    reset, step, render = jax.jit(base_env.reset), jax.jit(base_env.step), jax.jit(base_env.render)
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs, st = reset(k)
    template = obs
    n_actions = base_env.action_space().n

    fields = {}
    for name in field_names(template):
        entry = objects_cfg.get(name)
        if entry is None or entry.get("detected_type") not in labels.SUPPORTED_TYPES:
            continue
        value = getattr(template, name)
        kind = "oo" if isinstance(value, ObjectObservation) else \
               "xy" if entry["detected_type"] == "xy_pairs" else \
               "grid" if entry["detected_type"] in ("true_2d_grid", "grid") else "unsupported"
        shape = np.shape(value.x) if kind == "oo" else np.shape(value)
        cap = int(np.prod(shape)) if kind == "oo" else shape[0] if kind == "xy" else 10 ** 6
        fields[name] = dict(entry=entry, kind=kind, shape=shape, cap=cap, true_sizes=[], label_sizes=[],
                            label_centres=[], votes={})

    prev_frame = None
    for _ in range(steps):
        raw = frame = np.asarray(render(st))
        if max_pool and prev_frame is not None:  # same input as at inference
            frame = np.maximum(frame, prev_frame)
        gt_boxes = []  # (field, unpadded label box, padded label box)
        for name, f in fields.items():
            value = getattr(obs, name)
            if f["kind"] == "oo":
                for x, y, w, h, a in zip(*(np.atleast_1d(np.asarray(getattr(value, p))).ravel()
                                           for p in ("x", "y", "width", "height", "active"))):
                    if a and x > 1 and y > 1 and w > 0 and h > 0:
                        f["true_sizes"].append((float(w), float(h)))
            for box in labels.label_boxes(value, f["entry"]):
                f["label_sizes"].append(box[2:])
                f["label_centres"].append((box[0] + box[2] / 2, box[1] + box[3] / 2))
                gt_boxes.append((name, box, labels.pad_label(f["entry"], *box)))
        if gt_boxes:
            for d in detect_fn(frame):
                dcx, dcy = (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2
                best, best_dist = None, None
                for name, (lx, ly, lw, lh), padded in gt_boxes:
                    dist = np.hypot(dcx - (lx + lw / 2), dcy - (ly + lh / 2))
                    if (_iou(d, padded) >= 0.3 or dist <= 3) and (best_dist is None or dist < best_dist):
                        best, best_dist = (name, dcx - (lx + lw / 2), dcy - (ly + lh / 2)), dist
                if best is not None:
                    name, bx, by = best
                    fields[name]["votes"].setdefault(d["cls"], []).append((bx, by))
        prev_frame = raw
        key, ka, ks = jax.random.split(key, 3)
        obs, st, _, done, _ = step(st, jax.random.randint(ka, (), 0, n_actions))
        if bool(done):
            obs, st = reset(ks)
            prev_frame = None

    for name in list(fields):
        f = fields[name]
        e = f["entry"]
        true = np.array(f.pop("true_sizes")) if f["true_sizes"] else None
        lab = np.array(f["label_sizes"]) if f["label_sizes"] else None
        cen = np.array(f.pop("label_centres")) if f["label_centres"] else None
        f["true_w"] = float(np.median(true[:, 0])) if true is not None else None
        f["true_h"] = float(np.median(true[:, 1])) if true is not None else None
        f["const_w"] = bool(true is not None and np.ptp(true[:, 0]) == 0)
        f["const_h"] = bool(true is not None and np.ptp(true[:, 1]) == 0)
        # label size before / after the 8 px floor; the padded size is what the detector learned to predict
        f["lab_w"] = float(np.median(lab[:, 0])) if lab is not None else (f["true_w"] or 8.0)
        f["lab_h"] = float(np.median(lab[:, 1])) if lab is not None else (f["true_h"] or 8.0)
        _, _, f["det_w"], f["det_h"] = labels.pad_label(e, 0, 0, f["lab_w"], f["lab_h"])
        f["pad_w"] = e.get("cell_w") is None and f["lab_w"] < labels.MIN_BOX
        f["pad_h"] = e.get("cell_h") is None and f["lab_h"] < labels.MIN_BOX
        f["mu_x"], f["mu_y"] = (float(cen[:, 0].mean()), float(cen[:, 1].mean())) if cen is not None else (None, None)
        f["sd_x"], f["sd_y"] = (float(cen[:, 0].std()), float(cen[:, 1].std())) if cen is not None else (None, None)

        votes = f.pop("votes")
        top = max(votes, key=lambda c: len(votes[c])) if votes else None
        if top is not None and len(votes[top]) >= max(3, MIN_MATCH_FRACTION * len(f["label_sizes"])):
            v = np.array(votes[top])
            f["cls"], f["cls_source"], f["n_matched"] = top, "calibrated", len(v)
            f["bias_x"], f["bias_y"] = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
        else:  # not reliably matched: fall back to class names / registry, no offset
            f["cls"], f["cls_source"] = _fallback_class(name, e, model_names)
            f["n_matched"], f["bias_x"], f["bias_y"] = 0, 0.0, 0.0
        f.pop("label_sizes")
        if f["cls"] is None:
            del fields[name]
    return fields, template
