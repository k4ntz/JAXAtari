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

The detector keeps `max_det` detections per frame, by default twice the object slots of the observation
(e.g. 6 for Pong, 70 for Seaquest, capped at 300). The conversion to observations grows quadratically
with it, and so do compile time and runtime under large `vmap`s: with Pong at 8192 envs, PQN compiles in
~40 s (did not finish within 30 min with 300). Games with many slots (Ms. Pac-Man pellets, Tetris board,
Amidar paths) stay at 300; pass a smaller `max_det` to trade those dense fields for speed.

## Checkpoints

Checkpoints are not part of the repository. Like the sprites, they live in the jaxtari data directory:
`<storage>/detectors/<variant>/<game>.npz` (`jaxtari.paths.get_base_detector_dir()`, e.g.
`~/.local/share/jaxtari/detectors/gray96/pong.npz`). The wrapper and all tools accept such a name
(`"gray96/pong"`) or the path of any `.npz` checkpoint.

Install the `gray96` checkpoints (all games in [supported_games.md](../src/jaxtari/detection/supported_games.md)) with:

```bash
python3 -m jaxtari.install_detection
# or
.venv/bin/install-detection
```

| Variant | Input | Games | Notes |
|---|---|---|---|
| `rgb640/` | native RGB frame, nearest-upscaled to 640×840 | 41 | original checkpoints of the detection project (not part of the download) |
| `gray96/` | 96×96 grayscale from JAXtari native downscaling | see [supported_games.md](../src/jaxtari/detection/supported_games.md) | ~10× (1 env) to ~30× (vmapped) faster than `rgb640` |

Checkpoints are `.npz` files (raw weights + a JSON spec with architecture and inference defaults: input
mode, frame size, confidence threshold). `gray96` checkpoints switch the environment's
renderer to native downscaling automatically.

**Flicker:** some games draw sprites on alternating frames (e.g. the ship and the asteroids in
Asteroids). `YOLOObjectCentricWrapper(..., max_pool=True)` runs the detector on the max over the last
two frames of each frame skip (`evaluate.py --max-pool`, `pqn_agent.py +YOLO_MAX_POOL=True`). This is
inference only: the detectors are trained on single frames. It is off by default since moving objects
get smeared.

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
python train_detection/tools/build_registry.py <game>                   # 1. registry entries (fix REVIEW classes)
python train_detection/generate_dataset.py <game> --preview             # 2. check/tune label geometry in the registry
python train_detection/generate_dataset.py <game> --size 96 96          # 3. dataset
python train_detection/train.py <game> --data train_detection/datasets/<game> \
       --out <game>.npz                                           # 4. fine-tune
python train_detection/evaluate.py <game>.npz                           # 5. accuracy vs. ground truth + speed
```

The **quirks registry** (`src/jaxtari/detection/quirks_registry.json`) is the single source of truth for
the labels: per game it maps observation fields to the 7-class taxonomy (player, enemy, projectile,
collectible, structure, neutral, enemy_projectile) and stores the label geometry (offsets, padding,
grid layout). Label rules: `src/jaxtari/detection/labels.py`.

## Results (Apple M5 CPU, JAX CPU backend)

Measured with the original `pong` checkpoints; accuracy of the current `gray96` checkpoints per game:
[supported_games.md](../src/jaxtari/detection/supported_games.md).

| Checkpoint | Recall | Precision | Pos. error | env steps/s (1 env / vmap 128) |
|---|---|---|---|---|
| `rgb640/pong` | 99.3% | 100% | 0.10 px | 55 / 63 |
| `gray96/pong` | 95.3% | 99.0% | 0.59 px | 585 / 2058 |

PQN (object-centric MLP, 1024 envs, 5M steps) learns from the `gray96` observations almost as well as
from ground truth: Pong −6.8 vs −4.8 episode return.

## Known limitations

* Accuracy varies a lot per game for the `rgb640` checkpoints (e.g. Chopper Command, Wordzapper,
  Montezuma's Revenge and Haunted House are weak; see `evaluate.py`).
* Objects of ~1 px (missiles, bullets) are hard to detect at 96×96.
* Sprites drawn on alternating frames (e.g. Asteroids) are missed in the frames where they are not drawn
  unless `max_pool=True` is used.
* Several `rgb640` checkpoints were trained with class ids that differ from the current registry; the
  calibration handles this, but retraining them with `train.py` would make the mapping explicit.
