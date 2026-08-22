Mario Bros
==========

.. raw:: html

       <img src="../../_static/gifs/mariobros.gif" alt="Mario Bros" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``mariobros``. Parity status in ``games_covered.md``: ❌
(**not registered** in ``core.py``; draft code may still exist under
``jaxatari.games.jax_mariobros``).

Description
-----------

Platform pest-control: punch floors to flip enemies, then kick them off for
points. Corner pipes spawn pests; a POW block can flip all grounded enemies a
limited number of times. ALE also runs timed coin phases and numbered stages.

Actions
-------

Observations
------------

Reward
------

Modifications
-------------

None registered.

Known issues
------------

Living ALE-parity checklist. Gaps below are why this env is marked ❌ and
kept out of ``core.py`` until a fuller pass.

* **Sprites missing / incomplete** — no on-disk sprite pack under
  ``games/sprites/mariobros/``. Rendering is mostly procedural (palette-index
  arrays + solid rects). Incomplete vs ALE:

  - No authored assets for later pest types (Sidesteppers, Fighterflies,
    Slipice) — only a shellcreeper-like pest is drawn.
  - Bonus wafers / distinct reward pickup art are approximate or missing.
  - Corner pipes are simple green rectangles (geometry from ALE), not TIA
    playfield striping.
  - Platforms are flat color bars (no ALE brick / ice patterning).
  - HUD differs: lives are red squares; ALE life / score icons and the
    bottom-center level digits (``01``, ``02``, …) are not rendered.
  - Re-check visuals with ``scripts/gameplay_comparison.py -g mariobros``
    (requires loading the draft module directly; not via ``core``).

* **Level / stage progression (not ALE-faithful)** — ALE advances a numeric
  **level** (RAM ``5``, HUD at bottom center) when the current round’s pests
  are cleared, then introduces harder pest types and more fireballs. The draft
  has **no level counter** and does not run discrete clear-the-wave rounds:

  - **Stages** — ALE: numeric level; next round after all pests clear.
    Draft: no level counter; continuous respawn into free slots.
  - **Enemies** — ALE later stages add Sidesteppers, Fighterflies, Slipice.
    Draft: one shellcreeper-like type only.
  - **Coin phase** — ALE: stage-tied bonus rounds. Draft: score multiples of
    4000 only (timer ~15 s at 60 Hz is in the right ballpark; entry differs).
  - **Lives** — ALE / manual typically start at **5** (+ extra every 20 000).
    Draft starts at **4**; extra-life rule unverified.
  - **Fireballs** — ALE introduces them after early stages. Draft has them
    from frame 0. Top fireball lane (``y=30``) is intentionally omitted in
    the draft; early ALE uses mid lanes (``108`` / ``68``) first.
