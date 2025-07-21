Surround Environment
====================

.. automodule:: jaxatari.games.jax_surround
   :members:
   :undoc-members:
   :show-inheritance:

Visual Design Notes
-------------------

- **Playfield size:** logical grid of 40×24 cells.
- **Cell dimensions:** each cell is 4×8 pixels, giving a 160×192 gameplay area on a 210×160 screen.
- **Background color:** solid color that can vary per mode.
- **Trail colors:** each player uses a distinct trail color (green for player 1, purple for player 2 by default).
- **Moving block:** head uses the same size as a trail segment and can highlight the current position.
- **Visual simplicity:** empty background that gradually fills with colored trail squares.
 - **State representation:** single grid of 40×24 with values ``0`` for empty,
   ``1`` for player 0 trails and ``2`` for player 1 trails. The moving heads are
   tracked separately via ``pos0``/``pos1`` and ``dir0``/``dir1``.
