Gopher
======

.. raw:: html

       <img src="../../_static/gifs/gopher.gif" alt="Gopher" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``gopher``. Parity status in ``games_covered.md``: 🥇.

Description
-----------

Protect carrot crops from a tunneling gopher. Move left/right and fill holes before
the gopher reaches the surface; score by blocking digs and clearing threats.

Actions
-------

Observations
------------

Object-centric ``ObjectObservation`` fields plus HUD score:

* ``player`` — farmer (``visual_id`` = has seed, ``state`` = bonk timer, ``orientation`` = facing)
* ``gopher`` — gopher (``state`` / ``visual_id`` = action, ``orientation`` = facing; inactive when hidden underground)
* ``duck``, ``seed``, ``carrots`` (n=3)
* ``holes`` (n=18 dug hole tiles), ``tunnels`` (n=40 dug tunnel tiles)
* ``score_player`` — on-screen score digits

Reward
------

Modifications
-------------

Registered via ``GopherEnvMod``:

* ``greedy_gopher``
* ``mirror_button``
* ``energy_drain``
* ``heavy_shovel``
* ``fast_seed``
* ``invisible_gopher``
* ``wind_gopher``
* ``dizzy_farmer``

Recommended medal
-----------------

**🥇 Gold** — single-screen ALE parity checklist addressed (player side borders,
dig/hole cadence, mod + pixel-obs downscale, energy-drain score, invisible gopher
on pixels, ``AutoDerivedConstants``, faster block rendering). Intentional omission:
ALE start-of-episode delay.

Known issues
------------

* No ALE-style start delay (accepted divergence).
* Dig / hole spawn rates tuned for score feel vs ALE, not claimed bit-exact.
