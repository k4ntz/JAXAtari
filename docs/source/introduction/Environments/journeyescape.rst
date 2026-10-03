Journey Escape
==============

.. raw:: html

       <img src="../../_static/gifs/journeyescape.gif" alt="Journey Escape" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``journeyescape``. Parity status in ``games_covered.md``: 🥈.

Description
-----------

After a concert, guide each of the five Journey band members past Love-Crazed
Groupies, Shifty-Eyed Promoters, Sneaky Photographers, and Stage Barriers to the
Scarab Escape Vehicle before time runs out, while protecting the band's concert
cash. Loyal Roadies grant short invulnerability; the Mighty Manager awards a
large cash bonus and lets you run unhindered until the Scarab. Leftover time
carries to the next band member; clearing all five adds a round bonus and the
obstacle patterns get harder.

Based on the Data Age Atari 2600 cartridge / ALE ``JourneyEscape`` environment
(see the `AtariAge manual <https://atariage.com/manual_html_page.php?SoftwareLabelID=252>`_).

Actions
-------

Observations
------------

* ``player`` — ``ObjectObservation`` (``visual_id`` = band member, ``state`` = invincibility / manager shield, ``orientation`` = facing)
* ``obstacles`` — ``ObjectObservation`` pool (n=``MAX_OBS``; ``visual_id`` = type; photographers inactive while blinked off)
* ``score``, ``countdown`` — HUD dollar score and M:SS timer
* ``band_member`` — current band-member index

Reward
------

Modifications
-------------

Registered via ``JourneyEscapeEnvMod``:

* ``background_static``
* ``speed_up_player``
* ``speed_up_obstacles``
* ``reduce_player_size``
* ``restrict_player_movement``
* ``obstacle_diagonal_movement`` (skip to level-2 diagonal pattern)
* ``obstacle_steep_diagonal_movement`` (skip to level 3)
* ``obstacle_accelerating_bounce`` (skip to level 4)
* ``obstacle_random_direction`` (skip to level 5)
* ``obstacle_chaotic_movement``

The base environment progresses difficulty concurrently via the Scarab; the
level-pattern mods are skip-ahead shortcuts.

Known issues
------------

* Later-level bounce / chaos motion and exact spawn mixes are approximate
  relative to ALE.
* No OCAtari RAM object module, so parity is trajectory / visual rather than
  lockstep object matching.
* Band-member initials HUD and the cartridge intro scene are not rendered.
