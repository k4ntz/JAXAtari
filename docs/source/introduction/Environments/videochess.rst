Video Chess
===========

.. raw:: html

       <img src="../../_static/gifs/videochess.gif" alt="Video Chess" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``videochess``. Parity status in ``games_covered.md``: 🥈.

Description
-----------

Atari 2600–style chess: move a cursor with the joystick, press FIRE to select a piece
and again to move to a target square. By default you play **white** and black replies
with a depth-2 (top-k pruned) **minimax** bot after each of your moves.

Actions
-------

Observations
------------

* ``board`` — ``(8, 8)`` piece codes
* ``cursor`` / ``selected`` — ``ObjectObservation`` pointers on the board grid (``active`` when valid)
* ``to_move`` — side to move (0=white, 1=black)

Reward
------

Material change (in pawn units / 1000-style scaling via piece values) plus terminal
outcomes from checkmate / king capture / stalemate.

Modifications
-------------

Registered via ``VideochessEnvMod``:

**Opponent / control**

* ``play_both_sides`` — human controls both colours (disables the black bot)
* ``random_bot_black`` — black replies with a random legal move
* ``greedy_bot_black`` — black replies with a greedy capture heuristic
* ``minimax_bot_black`` — black replies with depth-2 minimax (unmodded default; useful to re-enable after another bot mod)

**Input / display**

* ``instant_movement`` — held direction inputs repeat every frame (FIRE stays edge-triggered)
* ``legal_moves_display`` — show legal-move indicator dots for the selected piece (off by default)

**Board setups**

* ``pawns_only`` / ``queens_only`` / ``rooks_only`` / ``knights_only`` / ``bishops_only``
* ``checkmate_test`` — sparse mate-in-one style position

Recommended medal
-----------------

**🥈 Silver** — playable chess rules with castling, en passant, promotions, check
filtering, and a working black opponent, but not full ALE Video Chess parity
(no cartridge AI levels / opening book / timing / audio), and the minimax bot is
a JAXAtari addition rather than a faithful port of the 2600 engine.

Known issues
------------

* **Default bot cost** — depth-2 minimax is expensive; after each white move the step
  can stall while black thinks. Use ``play_both_sides`` or ``random_bot_black`` for
  fast interaction / training loops.
* **Legal-move dots are a mod** — unmodded play does not highlight targets (ALE UX
  differed; enable ``legal_moves_display`` when useful).
* **Edge-triggered cursor** — by default held sticks do not auto-repeat; enable
  ``instant_movement`` for held-key cursor motion.
* **AI ≠ cartridge** — black is a simple pruned minimax / greedy / random helper,
  not the original Video Chess difficulty levels.
* **No underpromotion** — promotions always become queens.
* **Draw rules incomplete** — stalemate is detected; threefold repetition and
  fifty-move rule are not.
* **Performance** — even without the bot, FIRE-to-move still runs a full legal-move
  + check filter for the selected piece, and every board change runs a game-over /
  checkmate scan. Cursor-only steps and ``play_both_sides`` avoid compiling the
  minimax graph.
* **Two-player ALE modes / select screen** — not modeled; game starts mid-board.
