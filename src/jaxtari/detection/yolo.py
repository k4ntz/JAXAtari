"""Pure-JAX YOLOv8 detector: pre-processing, network, DFL box decoding and class-aware NMS.

Everything is jittable and vmappable. Checkpoints are torch-free ``.npz`` files (raw layer weights plus a
JSON architecture/inference spec); ``train_detection/convert_ultralytics.py`` creates them from ultralytics
``.pt`` files. Supported layers are those of YOLOv8: Conv, C2f (+Bottleneck), SPPF, Upsample, Concat and
the (anchor-free, DFL) Detect head.

    det = load_detector("gray96/pong")            # or a path to a .npz checkpoint
    out = jax.jit(det.detect)(det.params, frame)   # frame: (H, W, C) uint8 as rendered -> Detections
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

FORMAT = "jaxtari-yolov8"
LEGACY_FORMATS = ("jaxatari-yolov8",)  # checkpoints converted before the jaxtari rename
MAX_WH = 7680.0  # class offset for class-aware NMS (as ultralytics)


class Detections(NamedTuple):
    """Fixed-size detections (max_det rows, sorted by confidence); rows with valid=False are padding."""
    boxes: jax.Array   # (max_det, 4) x1, y1, x2, y2 in detector frame pixels (frame_size)
    conf: jax.Array    # (max_det,)
    cls: jax.Array     # (max_det,) int32
    valid: jax.Array   # (max_det,) bool


# ------------------------------------------------------------------------------------ checkpoint loading
def _load_conv(spec, arrays):
    """Raw conv (+ BatchNorm) weights -> fused HWIO weight and bias."""
    k = spec["key"]
    w = arrays[f"{k}.weight"].astype(np.float64)                      # (O, I/g, kh, kw)
    if spec["bn"]:
        scale = arrays[f"{k}.bn.weight"].astype(np.float64) / np.sqrt(arrays[f"{k}.bn.var"].astype(np.float64)
                                                                      + spec["eps"])
        w = w * scale[:, None, None, None]
        b = arrays[f"{k}.bn.bias"].astype(np.float64) - arrays[f"{k}.bn.mean"].astype(np.float64) * scale
    else:
        b = arrays[f"{k}.bias"].astype(np.float64) if f"{k}.bias" in arrays else np.zeros(w.shape[0])
    return dict(w=jnp.asarray(w.transpose(2, 3, 1, 0), jnp.float32), b=jnp.asarray(b, jnp.float32))


def _load_layer(spec, arrays):
    t = spec["type"]
    if t == "Conv":
        return _load_conv(spec["conv"], arrays)
    if t == "C2f":
        return dict(cv1=_load_conv(spec["cv1"], arrays), cv2=_load_conv(spec["cv2"], arrays),
                    m=[dict(cv1=_load_conv(b["cv1"], arrays), cv2=_load_conv(b["cv2"], arrays)) for b in spec["m"]])
    if t == "SPPF":
        return dict(cv1=_load_conv(spec["cv1"], arrays), cv2=_load_conv(spec["cv2"], arrays))
    if t == "Detect":
        return dict(heads=[{br: [_load_conv(c, arrays) for c in h[br]] for br in ("box", "cls")}
                           for h in spec["heads"]])
    return {}


# ----------------------------------------------------------------------------------------------- network
def _conv(p, s, x):
    pad = s["p"]
    y = jax.lax.conv_general_dilated(x, p["w"], (s["s"], s["s"]), [(pad, pad), (pad, pad)],
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=s["g"])
    y = y + p["b"]
    return jax.nn.silu(y) if s["act"] else y


def _run_layer(p, s, x):
    t = s["type"]
    if t == "Conv":
        return _conv(p, s["conv"], x)
    if t == "C2f":
        y = _conv(p["cv1"], s["cv1"], x)
        ys = [y[..., :s["c"]], y[..., s["c"]:]]
        for bp, bs in zip(p["m"], s["m"]):
            h = _conv(bp["cv2"], bs["cv2"], _conv(bp["cv1"], bs["cv1"], ys[-1]))
            ys.append(ys[-1] + h if bs["add"] else h)
        return _conv(p["cv2"], s["cv2"], jnp.concatenate(ys, -1))
    if t == "SPPF":
        k = s["k"]
        y = [_conv(p["cv1"], s["cv1"], x)]
        for _ in range(s["n"]):
            y.append(jax.lax.reduce_window(y[-1], -jnp.inf, jax.lax.max, (1, k, k, 1), (1, 1, 1, 1),
                                           [(0, 0), (k // 2, k // 2), (k // 2, k // 2), (0, 0)]))
        out = _conv(p["cv2"], s["cv2"], jnp.concatenate(y, -1))
        return out + x if s["add"] else out
    if t == "Upsample":
        return jnp.repeat(jnp.repeat(x, s["scale"], axis=1), s["scale"], axis=2)
    if t == "Concat":
        return jnp.concatenate(x, -1)
    raise NotImplementedError(t)


def _detect_head(p, s, feats):
    """-> (B, A, 4 + nc): xyxy boxes in network-input pixels + class probabilities."""
    boxes, scores, anchors, strides = [], [], [], []
    reg = s["reg_max"]
    for hp, hs, f, st in zip(p["heads"], s["heads"], feats, s["strides"]):
        b, c = f, f
        for lp, ls in zip(hp["box"], hs["box"]):
            b = _conv(lp, ls, b)
        for lp, ls in zip(hp["cls"], hs["cls"]):
            c = _conv(lp, ls, c)
        B, H, W, _ = f.shape
        boxes.append(b.reshape(B, H * W, 4 * reg))
        scores.append(c.reshape(B, H * W, s["nc"]))
        gy, gx = jnp.meshgrid(jnp.arange(H) + 0.5, jnp.arange(W) + 0.5, indexing="ij")
        anchors.append(jnp.stack([gx.ravel(), gy.ravel()], -1))
        strides.append(jnp.full((H * W, 1), st))
    box, score = jnp.concatenate(boxes, 1), jnp.concatenate(scores, 1)
    anchor, stride = jnp.concatenate(anchors, 0), jnp.concatenate(strides, 0)
    # DFL: expected distance over reg_max bins for each side (channel = side * reg_max + bin)
    dist = jax.nn.softmax(box.reshape(*box.shape[:2], 4, reg), -1) @ jnp.arange(reg, dtype=box.dtype)
    xy1 = (anchor - dist[..., :2]) * stride
    xy2 = (anchor + dist[..., 2:]) * stride
    return jnp.concatenate([xy1, xy2, jax.nn.sigmoid(score)], -1)


# ------------------------------------------------------------------------------------------------ model
@dataclass(frozen=True, eq=False)
class YOLODetector:
    params: Any
    specs: tuple
    froms: tuple
    save: frozenset
    names: dict
    nc: int
    input_mode: str      # "native": frame is fed as rendered (e.g. JAXtari natively downscaled 96x96x1)
                         # "rgb": native 210x160 RGB frame, nearest-upscaled to frame_size (original checkpoints)
    frame_size: tuple    # (W, H) of the detector frame; boxes are returned in these pixels
    net_shape: tuple     # (H, W) network input (letterboxed)
    resized: tuple       # (H, W) of the resized image inside the letterbox
    pad: tuple           # (top, left)
    gain: tuple          # (gain_y, gain_x) frame px -> network px
    conf: float = 0.40
    iou: float = 0.50
    max_det: int = 300
    max_cand: int = 2048  # candidates entering NMS
    game: str = ""

    def preprocess(self, frame):
        """(H, W, C) uint8 frame -> letterboxed (net_H, net_W, C) float in [0, 1]."""
        fw, fh = self.frame_size
        rh, rw = self.resized
        x = frame.astype(jnp.float32)
        if self.input_mode == "rgb":  # nearest upscale of the native frame to frame_size
            H0, W0 = frame.shape[:2]
            x = x[(jnp.arange(fh) * H0) // fh][:, (jnp.arange(fw) * W0) // fw]
        if (rh, rw) != (fh, fw):  # bilinear, half-pixel centres, no antialias (= cv2.INTER_LINEAR)
            x = jnp.round(jax.image.resize(x, (rh, rw, x.shape[-1]), "linear", antialias=False))
        top, left = self.pad
        nh, nw = self.net_shape
        x = jnp.pad(x, ((top, nh - rh - top), (left, nw - rw - left), (0, 0)), constant_values=114.0)
        return x / 255.0

    def forward(self, params, x):
        """(B, H, W, C) in [0, 1] -> (B, A, 4 + nc)."""
        outs, y = [], x
        for i, (p, s, f) in enumerate(zip(params, self.specs, self.froms)):
            if s["type"] == "Detect":
                return _detect_head(p, s, [outs[j] for j in f])
            y = _run_layer(p, s, y if f == -1 else [y if j == -1 else outs[j] for j in f])
            outs.append(y if i in self.save else None)
        raise RuntimeError("checkpoint has no Detect layer")

    def nms(self, pred):
        """(A, 4 + nc) -> Detections (class-aware greedy NMS, boxes in frame pixels)."""
        scores, cls = pred[:, 4:].max(-1), pred[:, 4:].argmax(-1)
        k = min(self.max_cand, pred.shape[0])
        top_s, idx = jax.lax.top_k(jnp.where(scores > self.conf, scores, -1.0), k)
        valid = top_s > self.conf
        boxes, c = pred[idx, :4], cls[idx]
        ob = boxes + (c * MAX_WH)[:, None]
        area = (ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1])

        def body(carry):  # one candidate per iteration; O(k) memory (vmap friendly)
            i, keep = carry
            inter = jnp.prod(jnp.clip(jnp.minimum(ob[i, 2:], ob[:, 2:]) - jnp.maximum(ob[i, :2], ob[:, :2]), 0), -1)
            iou = inter / (area[i] + area - inter + 1e-9)
            suppressed = jnp.any(keep & (iou > self.iou) & (jnp.arange(k) < i))
            return i + 1, keep.at[i].set(valid[i] & ~suppressed)

        _, keep = jax.lax.while_loop(lambda c: c[0] < valid.sum(), body, (0, jnp.zeros(k, bool)))
        order = jnp.argsort(~keep, stable=True)[: self.max_det]  # kept first, score order preserved
        boxes, conf, c, keep = boxes[order], top_s[order], c[order], keep[order]
        gy, gx = self.gain
        top, left = self.pad
        fw, fh = self.frame_size
        xy = jnp.stack([jnp.clip((boxes[:, 0] - left) / gx, 0, fw), jnp.clip((boxes[:, 1] - top) / gy, 0, fh),
                        jnp.clip((boxes[:, 2] - left) / gx, 0, fw), jnp.clip((boxes[:, 3] - top) / gy, 0, fh)], -1)
        return Detections(xy, conf, c.astype(jnp.int32), keep)

    def detect(self, params, frame):
        """One rendered frame -> Detections. jit/vmap me."""
        return self.nms(self.forward(params, self.preprocess(frame)[None])[0])


def letterbox_geometry(frame_size, imgsz, stride=32):
    """Minimum-rectangle letterbox (ultralytics LetterBox(auto=True)) of a (W, H) frame for imgsz."""
    fw, fh = frame_size
    r = min(imgsz / fh, imgsz / fw)
    new_w, new_h = round(fw * r), round(fh * r)
    pw, ph = np.mod(imgsz - new_w, stride) / 2, np.mod(imgsz - new_h, stride) / 2
    top, bottom, left, right = round(ph - 0.1), round(ph + 0.1), round(pw - 0.1), round(pw + 0.1)
    return dict(net_shape=(new_h + top + bottom, new_w + left + right), resized=(new_h, new_w), pad=(top, left),
                gain=(new_h / fh, new_w / fw))


def resolve_checkpoint(name_or_path):
    """A checkpoint file path, or a name ``"<variant>/<game>"`` (e.g. ``"gray96/pong"``) looked up in the
    detector directory next to the sprites (``jaxtari.paths.get_base_detector_dir()``)."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    from jaxtari.paths import get_base_detector_dir
    q = Path(get_base_detector_dir()) / p.with_suffix(".npz")
    if q.is_file():
        return q
    raise FileNotFoundError(f"detector checkpoint '{name_or_path}' not found (neither a file nor {q}); "
                            f"install the detector weights (python3 -m jaxtari.install_detection) or pass the path "
                            f"of a .npz checkpoint")


def load_detector(path, conf=None, iou=None, max_det=300, max_cand=2048):
    """Load a ``.npz`` detector checkpoint by path or name (see resolve_checkpoint). Arguments override the
    inference defaults stored in the checkpoint."""
    with np.load(resolve_checkpoint(path), allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    spec = json.loads(str(arrays.pop("__spec__")))
    if spec.get("format") not in (FORMAT, *LEGACY_FORMATS):
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    layers = spec["layers"]
    return YOLODetector(
        params=[_load_layer(s, arrays) for s in layers], specs=tuple(layers),
        froms=tuple(s["from"] for s in layers), save=frozenset(spec["save"]),
        names={int(k): v for k, v in spec["names"].items()}, nc=spec["nc"], input_mode=spec["input_mode"],
        frame_size=tuple(spec["frame_size"]), **letterbox_geometry(spec["frame_size"], spec["imgsz"]),
        conf=spec.get("conf", 0.40) if conf is None else conf, iou=spec.get("iou", 0.50) if iou is None else iou,
        max_det=max_det, max_cand=max_cand, game=spec.get("game", ""))
