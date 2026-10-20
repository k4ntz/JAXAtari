"""YOLOObjectCentricWrapper: object-centric observations from a YOLO detector instead of the game state.

    from jaxtari.detection import YOLOObjectCentricWrapper
    env = YOLOObjectCentricWrapper(AtariWrapper(jaxtari.make("pong")), "gray96/pong")  # name or .npz path
    obs, state = env.reset(key)                       # jittable and vmappable
    obs, state, reward, terminated, truncated, info = env.step(state, action)

Observations have exactly the shape and layout of ObjectCentricWrapper (frame_stack_size, num_features).
Everything the detector cannot provide is MISSING (-1): scores, lives, timers, visual_id / state /
orientation, unsupported field types, and width/height of padded objects whose size is not constant.
Object slots and grid cells without a detection are MISSING in all properties; detected objects have
active = 1. Slot identity is not tracked: detections fill an object array left-to-right, top-to-bottom.

The whole step (env frame skip, rendering, optional max-pooling of the last two frames, detector, NMS,
conversion, frame stack) runs inside jax.jit. The calibration (see calibration.py) runs once at
construction. Checkpoints trained on natively downscaled frames switch the environment's renderer to
JAXtari native downscaling automatically.
"""
import dataclasses
import functools
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

import jaxtari.spaces as spaces
from jaxtari.wrappers import AtariWrapper, JaxtariWrapper, ObjectCentricWrapper

from . import labels
from .calibration import calibrate, field_names
from .yolo import Detections, YOLODetector, load_detector

MISSING = -1.0
_KIND = {"oo": 0, "xy": 1, "grid": 2, "unsupported": 3}


@struct.dataclass
class YOLOObjectCentricState:
    atari_state: Any
    obs_stack: chex.Array
    detections: Detections  # detections behind the latest observation (detector frame pixels)


class YOLOObjectCentricWrapper(JaxtariWrapper):
    """Detector-based drop-in replacement for ObjectCentricWrapper. Apply directly on top of AtariWrapper.

    Args:
        env: AtariWrapper.
        detector: YOLODetector, path to a ``.npz`` checkpoint, or checkpoint name such as "gray96/pong".
        game_name: registry key (defaults to the game stored in the checkpoint).
        registry: quirks registry dict (defaults to the one shipped with jaxtari).
        max_pool: run the detector on the max over the last two frames of each frame skip, for games whose
            sprites flicker (drawn on alternating frames, e.g. Asteroids). Inference only: detectors are
            trained on single frames; the calibration uses the same pooled input.
        calibration_steps / calibration_seed: random steps used for the one-off calibration.
        max_det: detections kept per frame. Defaults to twice the object slots of the observation (grid
            fields: their cells), capped at the detector's max_det; detections beyond the slots cannot be
            stored anyway, and the conversion's cost (compile time and runtime) grows with max_det^2.
    """

    def __init__(self, env, detector: Union[YOLODetector, str], game_name=None, registry=None,
                 frame_stack_size=4, frame_skip=4, clip_reward=True, max_pool=False, calibration_steps=300,
                 calibration_seed=0, max_det=None):
        assert isinstance(env, AtariWrapper), "YOLOObjectCentricWrapper must be applied on top of AtariWrapper"
        super().__init__(env)
        self.detector = detector if isinstance(detector, YOLODetector) else load_detector(detector)
        self.game_name = game_name or self.detector.game
        registry = registry or labels.load_registry()
        if self.game_name not in registry:
            raise KeyError(f"game '{self.game_name}' not in the quirks registry")
        self.frame_stack_size, self.frame_skip, self.clip_reward = frame_stack_size, frame_skip, clip_reward
        self.max_pool = max_pool

        core = env
        while hasattr(core, "_env"):
            core = core._env
        game_h, game_w = core.renderer.config.game_dimensions  # object coordinates live in this space
        fw, fh = self.detector.frame_size
        if self.detector.input_mode == "native":  # detector trained on JAXtari natively downscaled frames
            from jaxtari.modification import apply_native_downscaling
            channels = int(self.detector.params[0]["w"].shape[2])
            apply_native_downscaling(env, (fh, fw), channels == 1)
        self.scale = np.array([fw / game_w, fh / game_h, fw / game_w, fh / game_h], np.float32)  # game -> frame

        detect = jax.jit(self.detector.detect)

        def detect_game_px(frame):
            d = detect(self.detector.params, jnp.asarray(frame))
            v = np.asarray(d.valid)
            boxes = np.asarray(d.boxes)[v] / self.scale
            return [dict(cls=int(c), conf=float(s), x1=b[0], y1=b[1], x2=b[2], y2=b[3])
                    for b, c, s in zip(boxes, np.asarray(d.cls)[v], np.asarray(d.conf)[v])]

        self.fields, self._template = calibrate(env._env, registry[self.game_name]["objects"], self.detector.names,
                                                detect_game_px, calibration_steps, calibration_seed, max_pool)
        self.num_features = int(sum(np.size(l) for l in jax.tree.leaves(self._template)))
        if max_det is None:
            slots = sum(int(np.prod(f["shape"])) if f["kind"] == "grid" else f["cap"]
                        for f in self.fields.values() if f["kind"] in ("oo", "xy", "grid"))
            max_det = min(self.detector.max_det, max(2 * slots, 1))
        self.detector = dataclasses.replace(self.detector, max_det=int(max_det))
        oc_space = ObjectCentricWrapper(env, frame_stack_size=frame_stack_size).observation_space()
        self._observation_space = spaces.Box(low=np.minimum(np.asarray(oc_space.low), MISSING),
                                             high=np.asarray(oc_space.high), shape=oc_space.shape,
                                             dtype=jnp.float32)
        self._build_tables()

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    # ------------------------------------------------------------------------- static lookup tables
    def _build_tables(self):
        offsets, off = {}, 0  # flat offset and shape of every observation leaf ("player.x", "blocks", ...)
        for path, leaf in jax.tree_util.tree_flatten_with_path(self._template)[0]:
            offsets[jax.tree_util.keystr(path).lstrip(".")] = (off, np.shape(leaf))
            off += int(np.size(leaf))
        int_fields = {n for n in field_names(self._template)
                      if np.issubdtype(np.asarray(jax.tree.leaves(getattr(self._template, n))[0]).dtype, np.integer)}
        names = list(self.fields)
        F = len(names)
        maxcap = max([f["cap"] for f in self.fields.values() if f["kind"] in ("oo", "xy")] + [1])
        keys = ("cls", "cap", "kind", "det_w", "det_h", "has_pos", "mu_x", "mu_y", "sd_x", "sd_y", "bias_x",
                "bias_y", "is_entity", "off_x", "off_y", "pad_w", "pad_h", "const_w", "const_h", "true_w", "true_h",
                "padded_w", "padded_h", "lab_w", "lab_h", "int", "g_off", "g_rows", "g_cols", "g_c0", "g_tr",
                "g_cw", "g_ch", "g_sx", "g_sy", "g_ox", "g_oy", "g_on")
        T = {k: np.zeros(F, np.float32) for k in keys}
        oo_idx = np.full((F, maxcap, 5), self.num_features, np.int32)  # out of range = dropped write
        xy_idx = np.full((F, maxcap, 2), self.num_features, np.int32)
        for i, n in enumerate(names):
            f, e = self.fields[n], self.fields[n]["entry"]
            T["cls"][i], T["cap"][i], T["kind"][i] = f["cls"], f["cap"], _KIND[f["kind"]]
            T["det_w"][i], T["det_h"][i] = f["det_w"], f["det_h"]
            T["has_pos"][i] = f["mu_x"] is not None
            T["mu_x"][i], T["mu_y"][i] = f["mu_x"] or 0, f["mu_y"] or 0
            T["sd_x"][i], T["sd_y"][i] = f["sd_x"] or 0, f["sd_y"] or 0
            T["bias_x"][i], T["bias_y"][i] = f["bias_x"], f["bias_y"]
            T["is_entity"][i] = e.get("detected_type") == "entity"
            T["off_x"][i], T["off_y"][i] = e.get("x_offset") or 0, e.get("y_offset") or 0
            T["pad_w"][i], T["pad_h"][i] = e.get("w_padding") or 0, e.get("h_padding") or 0
            T["const_w"][i], T["const_h"][i] = f["const_w"], f["const_h"]
            T["true_w"][i], T["true_h"][i] = f["true_w"] or 0, f["true_h"] or 0
            T["padded_w"][i], T["padded_h"][i] = f["pad_w"], f["pad_h"]
            T["lab_w"][i], T["lab_h"][i] = f["lab_w"], f["lab_h"]
            T["int"][i] = n in int_fields
            if f["kind"] == "oo":
                for j, prop in enumerate(("x", "y", "width", "height", "active")):
                    oo_idx[i, :f["cap"], j] = offsets[f"{n}.{prop}"][0] + np.arange(f["cap"])
            elif f["kind"] == "xy":
                base = offsets[n][0]
                xy_idx[i, :f["cap"], 0] = base + 2 * np.arange(f["cap"])
                xy_idx[i, :f["cap"], 1] = base + 2 * np.arange(f["cap"]) + 1
            elif f["kind"] == "grid":
                base, shape = offsets[n]
                rows, cols, cw, ch, sx, sy, ox, oy = labels.grid_geometry(shape, e)
                T["g_off"][i], T["g_rows"][i], T["g_cols"][i], T["g_c0"][i] = base, rows, cols, shape[1]
                T["g_tr"][i] = bool(e.get("transpose_grid"))
                T["g_cw"][i], T["g_ch"][i], T["g_sx"][i], T["g_sy"][i] = cw, ch, sx, sy
                T["g_ox"][i], T["g_oy"][i], T["g_on"][i] = ox, oy, labels.grid_on_value(e)
        self._T = {k: jnp.asarray(v) for k, v in T.items()}
        self._oo_idx, self._xy_idx = jnp.asarray(oo_idx), jnp.asarray(xy_idx)
        self._F, self._maxcap = F, maxcap

    # ------------------------------------------------------------------------- detections -> observation
    def observation_from_detections(self, det: Detections) -> chex.Array:
        """Flat ObjectCentricWrapper-layout observation (num_features,) from one frame's detections."""
        T, F, N = self._T, self._F, self.num_features
        if F == 0:
            return jnp.full(N, MISSING, jnp.float32)
        b = det.boxes / self.scale  # frame px -> game px
        K = b.shape[0]
        w, h = jnp.maximum(b[:, 2] - b[:, 0], 1e-3), jnp.maximum(b[:, 3] - b[:, 1], 1e-3)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2

        # 1) route detections to fields, greedily in confidence order (box size + screen position prior)
        cost = jnp.abs(jnp.log(w[:, None] / T["det_w"])) + jnp.abs(jnp.log(h[:, None] / T["det_h"]))
        pos = 0.5 * jnp.minimum(jnp.abs(cx[:, None] - T["mu_x"]) / (T["sd_x"] + 4), 3) + \
              0.5 * jnp.minimum(jnp.abs(cy[:, None] - T["mu_y"]) / (T["sd_y"] + 4), 3)
        cost = cost + jnp.where(T["has_pos"] > 0, pos, 3.0)  # never-seen fields: last resort
        match = det.valid[:, None] & (det.cls[:, None] == T["cls"].astype(jnp.int32)[None])

        def route(counts, k):
            cand = match[k] & (counts < T["cap"])
            f = jnp.argmin(jnp.where(cand, cost[k], jnp.inf))
            ok = cand.any()
            return counts.at[f].add(ok.astype(counts.dtype)), jnp.where(ok, f, -1)

        _, field = jax.lax.scan(route, jnp.zeros(F, jnp.float32), jnp.arange(K))
        fi = jnp.maximum(field, 0)
        g = {k: v[fi] for k, v in T.items()}

        # 2) slot = rank inside the field by (x1, y1), ties by confidence order
        x1, y1, idx = b[:, 0], b[:, 1], jnp.arange(K)
        same = (field[:, None] == field[None, :]) & (field[None, :] >= 0)
        before = (x1[None] < x1[:, None]) | ((x1[None] == x1[:, None]) & (y1[None] < y1[:, None])) | \
                 ((x1[None] == x1[:, None]) & (y1[None] == y1[:, None]) & (idx[None] < idx[:, None]))
        slot = jnp.sum(same & before, axis=1)

        # 3) remove the calibrated centre offset, then invert the label geometry
        bx1, bx2, by1, by2 = b[:, 0] - g["bias_x"], b[:, 2] - g["bias_x"], b[:, 1] - g["bias_y"], b[:, 3] - g["bias_y"]

        def invert(lo, hi, const, true, padded, lab_size, off, pad):
            off, pad = off * g["is_entity"], pad * g["is_entity"]
            lab = jnp.where(const > 0, true + pad, jnp.where(padded > 0, lab_size, hi - lo))
            size = jnp.where(const > 0, true, jnp.where(padded > 0, MISSING, hi - lo - pad))
            return (lo + hi) / 2 - lab / 2 - off, size

        x, wo = invert(bx1, bx2, g["const_w"], g["true_w"], g["padded_w"], g["lab_w"], g["off_x"], g["pad_w"])
        y, ho = invert(by1, by2, g["const_h"], g["true_h"], g["padded_h"], g["lab_h"], g["off_y"], g["pad_h"])
        rnd = lambda v: jnp.where(g["int"] > 0, jnp.round(v), v)
        keep_missing = lambda s: jnp.where(s == MISSING, MISSING, rnd(s))
        x, y, wo, ho = rnd(x), rnd(y), keep_missing(wo), keep_missing(ho)

        kind = jnp.where(field >= 0, g["kind"], -1)
        in_slot = slot < self._maxcap
        sl = jnp.minimum(slot, self._maxcap - 1)
        oo_i = jnp.where(((kind == _KIND["oo"]) & in_slot)[:, None], self._oo_idx[fi, sl], N)
        oo_v = jnp.stack([x, y, wo, ho, jnp.ones_like(x)], -1)
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        xy_i = jnp.where(((kind == _KIND["xy"]) & in_slot)[:, None], self._xy_idx[fi, sl], N)
        xy_v = jnp.stack([rnd(bcx - g["lab_w"] / 2 - g["off_x"]), rnd(bcy - g["lab_h"] / 2 - g["off_y"])], -1)
        c = jnp.round((bcx - g["g_cw"] / 2 - g["g_ox"]) / g["g_sx"]).astype(jnp.int32)
        r = jnp.round((bcy - g["g_ch"] / 2 - g["g_oy"]) / g["g_sy"]).astype(jnp.int32)
        inside = (r >= 0) & (r < g["g_rows"]) & (c >= 0) & (c < g["g_cols"])
        cell = jnp.where(g["g_tr"] > 0, c * g["g_c0"] + r, r * g["g_c0"] + c).astype(jnp.int32)
        grid_i = jnp.where((kind == _KIND["grid"]) & inside, g["g_off"].astype(jnp.int32) + cell, N)

        out = jnp.full(N, MISSING, jnp.float32)
        out = out.at[oo_i.ravel()].set(oo_v.ravel(), mode="drop")
        out = out.at[xy_i.ravel()].set(xy_v.ravel(), mode="drop")
        return out.at[grid_i].set(g["g_on"], mode="drop")

    # ------------------------------------------------------------------------- env interface
    def render_frame(self, atari_state):
        """Frame as seen by the detector (natively downscaled if the checkpoint requires it)."""
        return self._env.render(atari_state.env_state)

    def _observe(self, atari_state, prev_frame=None):
        frame = self.render_frame(atari_state)
        if prev_frame is not None:
            frame = jnp.maximum(frame, prev_frame)
        det = self.detector.detect(self.detector.params, frame)
        return self.observation_from_detections(det), det

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        _, atari_state = self._env.reset(key)
        flat, det = self._observe(atari_state)
        obs_stack = jnp.tile(flat[None], (self.frame_stack_size, 1))
        return obs_stack, YOLOObjectCentricState(atari_state, obs_stack, det)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: YOLOObjectCentricState, action):
        def body(atari_state, _):
            _, atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return atari_state, (reward, terminated, truncated, info)

        if self.max_pool:  # frame_skip - 1 steps, keep the frame before the last one, then the last step
            atari_state, outs = jax.lax.scan(body, state.atari_state, None, length=self.frame_skip - 1)
            prev_frame = self.render_frame(atari_state)
            atari_state, last = body(atari_state, None)
            rewards, terms, truncs, infos = jax.tree.map(lambda a, b: jnp.concatenate([a, b[None]], 0),
                                                         outs, last)
        else:
            atari_state, (rewards, terms, truncs, infos) = jax.lax.scan(body, state.atari_state, None,
                                                                        length=self.frame_skip)
            prev_frame = None

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated, truncated = terms.any(), truncs.any()
        done = jnp.logical_or(infos["env_done"].any(), truncated)
        # auto-reset by selection (not lax.cond): under vmap the detector still runs once per step
        _, reset_state = self._env.reset(atari_state.key)
        atari_state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, atari_state)
        if prev_frame is not None:  # no previous frame after a reset (max with zeros = no pooling)
            prev_frame = jnp.where(done, jnp.zeros_like(prev_frame), prev_frame)
        flat, det = self._observe(atari_state, prev_frame)
        obs_stack = jnp.where(done, jnp.tile(flat[None], (self.frame_stack_size, 1)),
                              jnp.concatenate([state.obs_stack[1:], flat[None]], 0))

        def reduce_info(k, v):
            if k in ("env_reward", "all_rewards"):
                return jnp.sum(v, axis=0)
            return jnp.any(v) if k == "env_done" else v[-1]

        info = {k: reduce_info(k, v) for k, v in infos.items()}
        return obs_stack, YOLOObjectCentricState(atari_state, obs_stack, det), reward, terminated, truncated, info
