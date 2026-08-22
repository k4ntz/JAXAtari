Assault
=======

.. raw:: html

       <img src="../../_static/gifs/assault.gif" alt="Assault" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``assault``. Parity status in ``games_covered.md``: 🥉.

Description
-----------

Side-scrolling shooter: the player moves along the bottom, fires vertically or
sideways, and must destroy waves of crabs dropped from a mothership while
managing an overheat meter.

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

Living ALE-parity checklist (update as items are fixed or newly found).

* **Enemy sprites** — only the first enemy type has proper sprites
  (``enemy_0`` / ``enemy_1`` / tiny). Later stage enemy types appear to be
  simple palette recolors of that same shape rather than distinct ALE artwork.
* **Level / page progression** — wave length matches ALE (10 aliens per page;
  advance when those are cleared). Per-page *content* is only partly modelled:
  approximate 4-way color cycle and splitters after page 4 exist, but ALE's
  later vertical bobbing, distinct weapon types (missile groups / laser /
  fireballs), and random-appearance teleports are incomplete or missing.
* **Later-stage movement** — early game is fixed-lane horizontal with
  survivors shuffling down into empty lower lanes and new crabs entering only
  at the top. ALE later pages add vertical bobbing and short random-appearance
  teleports; those modes are not implemented.
* **Enemy spawn density** — three fixed Y slots (53 / 78 / 103) match early ALE;
  denser multi-per-row packing on later pages is still incomplete.
* **Starting lives** — JAX uses ``MAX_LIVES = 3``; ALE typically starts with 4.
* **Enemy fire timing** — ``ENEMY_FIRE_INTERVAL`` is approximate; may still
  diverge from ALE cadence under longer play.
* **Visual / UI leftovers** — residual sprite, heat-bar, or HUD mismatches vs
  ALE may remain after the recent parity pass; re-check with
  ``scripts/gameplay_comparison.py -g assault``.
