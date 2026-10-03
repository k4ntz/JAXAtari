Basic Math
==========

.. raw:: html

       <img src="../../_static/gifs/basicmath.gif" alt="Basic Math" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``basicmath``. Parity status in ``games_covered.md``: 🥈.

Description
-----------

Educational cartridge: solve arithmetic problems by selecting the correct answer.

Actions
-------

Observations
------------

Reward
------

Modifications
-------------

Registered via ``BasicmathEnvMod``:

* ``background_black``
* ``background_random``
* ``number_random``
* ``bigger_numbers``

Known issues
------------

* **Number selection timing** — answer / number selection reacts instantly.
  ALE has selection timing / debounce that this port does not reproduce.
* **Problem RNG** — posed arithmetic problems are drawn from JAXAtari's own
  RNG stream. With the same seed, problem sequences are **not** identical to
  ALE.
