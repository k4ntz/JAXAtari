Yar's Revenge
=============

.. raw:: html

       <img src="../../_static/gifs/yarsrevenge.gif" alt="Yar's Revenge" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``yarsrevenge``. Parity status in ``games_covered.md``: 🥇.

Description
-----------

The player controls Yar, flying around the Qotile's energy shield: nibble cells
from the shield, fire the Zorlon Cannon, and avoid the Destroyer missile and
Swirl while scoring against the Qotile.

Actions
-------

Observations
------------

Reward
------

Modifications
-------------

Registered via ``YarsRevengeEnvMod``:

* ``no_animations``
* ``speed_up``
* ``more_swirls``
* ``static_energy_shield``
* ``one_shield_shape``
* ``reversed_snake``
* ``visual_noise``
* ``fire_speed``

Known issues
------------

* **Sprite colors** — some sprite colors differ in part from ALE (authored /
  remapped palette vs the original TIA graphics), even when shapes and layout
  are close. Compare with ``scripts/gameplay_comparison.py -g yarsrevenge``.
