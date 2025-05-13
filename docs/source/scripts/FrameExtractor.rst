Frame Extractor
===============

The `frame_extractor.py` script is a standalone tool to **play Atari games interactively** using Gymnasium + ALE, while optionally **extracting raw game frames** into `.npy` files for later usage.

----

Purpose
-------

This tool is mainly used to:

- **Collect game frames** during human gameplay.
- Save each frame as a `.npy` file for offline inspection.

It displays the current game using Pygame, supports keyboard-based control, and allows saving snapshots frame-by-frame.

----

Usage
-----

.. code-block:: bash

   python scripts/frame_extractor.py --game Seaquest --screenshot-dir extracted_frames

Available arguments:

- `--game`: Name of the Atari game (e.g., `Pong`, `Seaquest`)
- `--scale`: Integer scaling factor for the display (default: 4)
- `--fps`: Target frame rate for the game loop (default: 30)
- `--screenshot-dir`: Where to save the extracted `.npy` frames (default: `{game}_screenshots/`)

----

Controls
--------

- **Arrow Keys** + `Space`: move/shoot/etc. (key-to-action mapping is dynamically generated based on the game)
- `P`: Pause/unpause the game
- `F`: Enable frame-by-frame stepping
- `N`: Advance one frame (in frame-by-frame mode)
- `R`: Reset the environment
- `S`: Save current frame to `.npy`

