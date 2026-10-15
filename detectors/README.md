# Object detection for JAXtari

JAXtari exposes ground-truth object-centric observations straight from the simulator. This folder
provides the alternative: **object-centric observations obtained by object detection** on the rendered
frames, with exactly the same interface.

```python
import jax, jaxtari
from jaxtari.wrappers import AtariWrapper
from jaxtari.detection import YOLOObjectCentricWrapper

env = YOLOObjectCentricWrapper(AtariWrapper(jaxtari.make("pong")), "gray96/pong")
obs, state = env.reset(jax.random.PRNGKey(0))          # obs.shape == ObjectCentricWrapper's (4, 26)
obs, state, reward, terminated, truncated, info = env.step(state, action)
```

The whole pipeline (frame skip, rendering, YOLOv8 network, NMS, conversion to observations, frame
stack) is pure JAX: it runs inside `jax.jit`, under `jax.vmap` and inside fully jitted training loops
such as `scripts/benchmarks/pqn_agent.py` (`+YOLO_WEIGHTS=<checkpoint>`). No torch is needed at runtime.

## Observations

* Same pytree, flattened order and shape as `ObjectCentricWrapper`.
* Detected objects: `x, y, width, height` in game pixels (top-left), `active = 1`.
* Everything the detector cannot provide is **-1**: scores, lives, timers, `visual_id` / `state` /
  `orientation`, unsupported field types (e.g. per-row river banks), and width/height of objects whose
  label was padded and whose size is not constant. Object slots / grid cells without a detection are -1
  in all properties.
* Slot identity is not tracked: detections fill an object array left-to-right, top-to-bottom.
* `state.detections` holds the raw detections of the latest observation (for visualisation).

At construction the wrapper calibrates itself once (~10 s): it plays a few hundred random steps,
runs the detector and matches its boxes to the ground-truth labels to learn which detector class
belongs to which observation field, where those objects appear, and the true sprite sizes. Ground
truth is only used for this calibration. See `src/jaxtari/detection/calibration.py`.

## Checkpoints

Checkpoints are not part of the repository. Like the sprites, they live in the jaxtari data directory:
`<storage>/detectors/<variant>/<game>.npz` (`jaxtari.paths.get_base_detector_dir()`, e.g.
`~/.local/share/jaxtari/detectors/gray96/pong.npz`). The wrapper and all tools accept such a name
(`"gray96/pong"`) or the path of any `.npz` checkpoint.

| Variant | Input | Games | Notes |
|---|---|---|---|
| `rgb640/` | native RGB frame, nearest-upscaled to 640×840 | 41 | original checkpoints of the detection project |
| `gray96/` | 96×96 grayscale from JAXtari native downscaling | pong, asteroids | ~10× (1 env) to ~30× (vmapped) faster than `rgb640` |

Checkpoints are `.npz` files (raw weights + a JSON spec with architecture and inference defaults: input
mode, frame size, confidence threshold, max-pooling). `gray96` checkpoints switch the environment's
renderer to native downscaling automatically.

**Flicker:** some games draw sprites on alternating frames (e.g. the ship and the asteroids in
Asteroids). Their checkpoints default to `max_pool=True` (max over the last two frames of each frame
skip). Do not enable it for games without flicker: moving objects get smeared (Pong player recall drops
from ~99% to ~92%).

## Tools

| Script | Purpose | Needs |
|---|---|---|
| `generate_dataset.py` | YOLO datasets from random play (labels from the registry + `game_patches.py`), `--preview` to check labels | `[train_detection]` |
| `train.py` | fine-tune YOLOv8n with ultralytics and export a `.npz` checkpoint | `[train_detection]` |
| `evaluate.py` | wrapper observations vs. game state: recall / precision / position error per field, throughput | – |
| `render_episode.py` | video of one episode with the detections drawn on the frames | `[train_detection]` |
| `convert_ultralytics.py` | ultralytics `.pt` → `.npz` | `[train_detection]` |
| `tools/build_registry.py` | add registry entries for new games / inspect observation structure | – |
| `tools/verify_conversion.py` | check a converted `.npz` against ultralytics' own predictions | `[train_detection]` |

`pip install -e ".[train_detection]"` installs ultralytics (torch), OpenCV and imageio for the tooling.

### Adding or improving a detector

```bash
python detectors/tools/build_registry.py <game>                   # 1. registry entries (fix REVIEW classes)
python detectors/generate_dataset.py <game> --preview             # 2. check/tune label geometry in the registry
python detectors/generate_dataset.py <game> --size 96 96          # 3. dataset (add --pool for flickering games)
python detectors/train.py <game> --data detectors/datasets/<game> \
       --out <game>.npz                                           # 4. fine-tune (+ --max-pool if pooled)
python detectors/evaluate.py <game>.npz                           # 5. accuracy vs. ground truth + speed
```

The **quirks registry** (`src/jaxtari/detection/quirks_registry.json`) is the single source of truth for
the labels: per game it maps observation fields to the 7-class taxonomy (player, enemy, projectile,
collectible, structure, neutral, enemy_projectile) and stores the label geometry (offsets, padding,
grid layout). Label rules: `src/jaxtari/detection/labels.py`.

## Results (Apple M5 CPU, JAX CPU backend)

| Checkpoint | Recall | Precision | Pos. error | env steps/s (1 env / vmap 128) |
|---|---|---|---|---|
| `rgb640/pong` | 99.3% | 100% | 0.10 px | 55 / 63 |
| `gray96/pong` | 95.3% | 99.0% | 0.59 px | 585 / 2058 |
| `rgb640/asteroids` (+max-pool) | 86.1% | 89.1% | 0.65 px | 37 / 38 |
| `gray96/asteroids` (+max-pool) | 83.7% | 89.4% | 1.02 px | 274 / 1086 |

PQN (object-centric MLP, 1024 envs, 5M steps) learns from the `gray96` observations almost as well as
from ground truth: Pong −6.8 vs −4.8, Asteroids 893 vs 957 episode return.

## Known limitations

* Accuracy varies a lot per game for the `rgb640` checkpoints (e.g. Chopper Command, Wordzapper,
  Montezuma's Revenge and Haunted House are weak; see `evaluate.py`).
* Objects of ~1 px (missiles, bullets) are hard to detect at 96×96.
* Several `rgb640` checkpoints were trained with class ids that differ from the current registry; the
  calibration handles this, but retraining them with `train.py` would make the mapping explicit.
