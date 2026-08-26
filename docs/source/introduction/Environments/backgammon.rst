Backgammon
==========

.. raw:: html

       <img src="../../_static/gifs/backgammon.gif" alt="Backgammon" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``backgammon``. Parity status in ``games_covered.md``: 🥈.

Description
-----------

Board game recreation of Backgammon for the Atari 2600.

Actions
-------

Default controls are instant-select (move the cursor with LEFT/RIGHT, confirm
with FIRE). This differs from ALE Backgammon, which uses hold-to-scroll and
release-to-drop. Use ``mods=["ale_controls"]`` for the original ALE scheme.

Observations
------------

Reward
------

Modifications
-------------

Registered via ``BackgammonEnvMod``:

* ``brown_theme``
* ``blue_theme``
* ``classic_theme``
* ``short_game``
* ``simplify``
* ``highlight_legal_moves``
* ``setup_mode``
* ``no_hits``
* ``reward_shaping``
* ``ale_controls``

Known issues
------------

