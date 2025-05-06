Sprite Editor
=============

The Sprite Editor is a specialized tool for editing `.npy`-based RGBA image files extracted from Atari game frames.  
It is a key component in the **sprite creation and rendering pipeline** for JAXAtari.

----

Purpose
-------

Once you've collected raw game frames using the :doc:`FrameExtractor <FrameExtractor>`, the Sprite Editor allows you to:

- **Open `.npy` frames** and extract specific sprites
- **Edit and annotate sprites** using visual tools
- **Save sprites** as RGBA `.npy` images, ready to be used in the rendering system

----

Typical Workflow
----------------

1. Use `frame_extractor.py` to record frames of gameplay:

   .. code-block:: bash

      python scripts/frame_extractor.py --game Seaquest

2. Open a saved frame in the Sprite Editor:

   .. code-block:: bash

      python scripts/spriteEditor/spriteEditor.py

3. Use the editor tools to select, clean, or modify a sprite.
4. Save your selection using **File → Save Selection**, which outputs an `.npy` RGBA sprite file.
5. Include the resulting sprite in the appropriate folder under `src/jaxatari/games/sprites/...`.

----

Editor Features
---------------

- **Pencil Tool** — draw using the current color
- **Magic Wand** — select regions by color
- **Rectangular Selection** — box-select areas
- **Dropper** — pick color from image
- Zoom, pan, transparency slider, ...

----

Shortcuts
---------

- `Ctrl+S`: Save selection
- `Ctrl+Z` / `Ctrl+Y`: Undo / Redo
- `Delete`: Clear selection (set alpha=0)
- `Ctrl+A` / `Ctrl+D`: Select all / deselect all

