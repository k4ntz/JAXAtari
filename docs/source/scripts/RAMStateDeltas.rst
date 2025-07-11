RAMStateDeltas
==============

The `RAMStateDeltas.py` script is an adaptation of OCAtari's original REMGui tool, redesigned for easier tracking of changes in the Atari game's RAM state.

Unlike the original REMGui, which allowed manual editing and pixel-wise analysis of RAM cells, **RAMStateDeltas** focuses on **RAM change detection**.

----

What It Does
------------

After each environment step, the script displays the **RAM state** with **highlighted differences** compared to the previous frame. This is especially helpful when reverse-engineering the game logic, for example to understand score updates, position changes, timers, or sprite state changes.

The interface visualizes:

- **Current RAM values** for each of the 128 RAM cells
- **Delta (change) values** for each cell (e.g., +3, -1, 0)
- Optional highlighting for significant or frequent changes

----

Usage
-----

Run the script like this:

.. code-block:: bash

   python scripts/RAMStateDeltas.py --game Pong


This will open a GUI where the game screen is shown on the left and RAM values with the corresponding deltas are shown on the right. You interact with the environment using standard control keys (WASD + Space).

----

Controls
--------

- `W`, `A`, `S`, `D`: move the agent
- `Space`: perform action / shoot / jump (depends on game)
- `R`: reset the environment
- `Esc`: close the window

----

Background
----------

This tool is a modified version of the original REMGui from `OCAtari`, which also supported **causative RAM analysis** by altering RAM values and measuring pixel output differences.

While RAMStateDeltas doesn't support RAM editing or pixel-based analysis, it is ideal for:

- Detecting which RAM cells change during gameplay
- Associating RAM patterns with on-screen effects
- Building hypotheses for object detection or scoring

----

Credits
-------

Original design adapted from `rem_gui.py` in `OCAtari`.  
Modified and extended by the JAXAtari team for step-based RAM comparison.
