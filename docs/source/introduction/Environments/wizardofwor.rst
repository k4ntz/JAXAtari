Wizard of Wor
=============

.. raw:: html

       <img src="../../_static/gifs/wizardofwor.gif" alt="Wizard of Wor" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``wizardofwor``. Parity status in ``games_covered.md``: 🥉.

Description
-----------

Maze shooter: the player fights Burwors (and later Garwors / Thorwors / Worluk /
Wizard) in a shared dungeon, using side teleporters and a bottom radar to track
enemies. JAXAtari currently focuses on the first dungeon layout.

Actions
-------

Observations
------------

Reward
------

Modifications
-------------

Registered via ``WizardOfWorEnvMod``:

* ``skip_startup_wait`` — set ``STARTUP_FREEZE_FRAMES=0`` (skip the ~210-frame ALE intro freeze).
* ``auto_spawn`` — set ``REQUIRE_SPAWN_INPUT=False`` (player starts already in the maze).
* ``skip_intro`` — modpack combining both of the above.

Known issues
------------

Living ALE-parity checklist (update as items are fixed or newly found).

* **Sprite / entity flickering** — ALE draws several objects on alternating
  frames (player, enemies, radar blips). JAX renders them solid every frame;
  flicker is not implemented yet.
* **Radar** — box and blip layout were retuned toward ALE, but blip timing,
  flicker, and occasional alignment vs the maze grid may still diverge.
* **Level progression (not ALE-faithful)** — dungeon 1 Burwor→Garwor→Thorwor
  promotion is only approximate; full multi-dungeon progression needs a pass.
  Known gaps:
  - Enemy **speed** scales with time-in-level (``SPEED_TIMER_*``), not dungeon
    number as in ALE.
  - Clearing a dungeon respawns the **fixed level-1 Burwor layout**, not
    ALE later-dungeon spawns.
  - Only **two** maze layouts are authored; further ``gameboard`` values are
    empty placeholders. ALE has more / different dungeons.
  - ``MAX_LEVEL = 5`` ends the game; ALE continues indefinitely.
  - **Worluk escape** does not end the dungeon; the level advances only when
    every enemy slot is empty.
  - Killing the Worluk never sets ``doubled``, so next-dungeon **double score**
    does not work.
  - Worluk pathfinding toward teleporters and Wizard teleport/combat are
    stubs / out of original scope.
  - Extra life on level clear (``lives + 1``) is unverified vs ALE.
* **Dungeon layouts** — only the first maze topology is fully authored; other
  boards are incomplete / placeholder relative to ALE.
* **Enemy AI** — movement, firing, and invisibility (Garwor / Thorwor) are
  approximate; they will not match ALE decision-for-decision. Early-game
  Burwor step cadence is tuned to ALE; later speed-up tiers are scaled
  proportionally and not measured against ALE.
* **Visual appearance (maze & enemies)** — the maze walls and enemy sprites
  look slightly different from ALE (authored / scaled assets vs the original
  playfield and TIA graphics), even when layout and colors are close. Render
  also uses a non-uniform upscale of half-resolution sprites plus a procedural
  radar; small color, thickness, or HUD mismatches can remain. Re-check with
  ``scripts/gameplay_comparison.py -g WizardOfWor``.
* **Teleporter duty cycle** — open/closed timing uses a simple
  ``frame_counter % 360 < 180`` window; may not match ALE exactly.
