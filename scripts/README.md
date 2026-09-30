# Scripts

Utility and development scripts for JAXAtari. Most scripts accept a `--help` flag for full usage.

---

## Interactive

| Script | Description |
|--------|-------------|
| `play.py` | Play any JAXAtari environment with keyboard input. Requires `pip install -e ".[dev]"` (pygame). `python scripts/play.py -g Pong` |

---

## Development & debugging

These scripts help during active environment development — comparing JAXAtari output against ALE, recording baselines, and inspecting object state.

| Script | Description |
|--------|-------------|
| `gameplay_comparison.py` | Play JAXAtari and ALE side-by-side with mirrored input. Useful for spotting visual or behavioural divergences. Supports `parallel` and `record_replay` modes. |
| `compare_renders.py` | Compare the first rendered frame of a JAXAtari implementation against the ALE equivalent. Reports shape and pixel-level differences. |
| `trajectory_regression.py` | Record a baseline trajectory (states, pixel obs, OC obs) and replay it to verify a refactor did not change behaviour. Run with `--record` to create a baseline, then without to check. |
| `RAMStateDeltas.py` | Play an OCAtari environment and print object state changes between frames. Useful for figuring out which RAM addresses encode which game objects (currently the visuals are bugged and RAM states are not rendered correctly). |
| `ALE_RAMStateDeltas.py` | Same as `RAMStateDeltas.py` but uses the ALE directly (no OCAtari dependency). |
| `get_objects_patches.py` | Automatically extracts per-object pixel patches from an ALE game for sprite analysis. |
| `frame_extractor.py` | Extracts individual frames from a gameplay recording for manual inspection (save frames by pressing 's'). |
| `spriteEditor/spriteEditor.py` | Interactive sprite editor to load frames saved by frame_extractor.py and extract out sprites in the correct format from them. |

---

## Benchmarks (`benchmarks/`)

Training and performance measurement scripts. Require `pip install -e ".[training]"`.

TODO

See [train_agents.md](../train_agents.md) at the project root for step-by-step training instructions.

---

## Asset generation

| Script | Description |
|--------|-------------|
| `generate_gif.py` | Record a gameplay session and export it as a GIF. Used to generate the GIFs in the README. |
| `generate_test_report.py` | Renders a Jinja2 HTML test report from CI environment variables. Used in the CI pipeline. |

---