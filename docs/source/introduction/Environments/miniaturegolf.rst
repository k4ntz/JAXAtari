Miniature Golf
==============

.. raw:: html

       <img src="../../_static/gifs/miniaturegolf.gif" alt="Miniature Golf" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``miniaturegolf``. Parity status in ``games_covered.md``: 🥇.

Description
-----------

Put across miniature-golf holes with obstacles.

Actions
-------

Full ALE action set (18 actions, including diagonals such as ``UPLEFT``).
Like ALE, diagonal inputs move on the **horizontal** axis only (left/right);
they do not produce true diagonal motion. Use ``mods=["diagonal_movement"]``
for actual diagonal play.

Starting hole
-------------

The initial hole is controlled by ``MiniatureGolfConstants.START_LEVEL``
(0-indexed; ``0`` = hole 1, …, ``8`` = hole 9). Override it when constructing
constants, or use a mod such as ``mods=["start_level_5"]``.

Observations
------------

Reward
------

Modifications
-------------

Registered via ``MiniatureGolfEnvMod``:

* ``large_hole``
* ``moving_hole``
* ``permeable_obstacle``
* ``permeable_wall``
* ``second_hole``
* ``soft_shot_required``
* ``stationary_obstacle``
* ``zero_shots``
* ``manhattan_reward``
* ``diagonal_movement``
* ``start_level_1`` … ``start_level_9``


Known issues
------------

Living ALE-parity checklist (update as items are fixed or newly found).

* **Diagonal inputs** — ALE exposes the full joystick set, but diagonal
  presses resolve to left/right only. JAXtari matches that by default;
  ``diagonal_movement`` enables true diagonals as an intentional divergence.
