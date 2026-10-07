Road Runner
===========

.. raw:: html

       <img src="../../_static/gifs/roadrunner.gif" alt="Road Runner" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``roadrunner``. Parity status in ``games_covered.md``: 🥉.

Description
-----------

Outrun Wile E. Coyote along a scrolling desert highway while collecting birdseed and dodging trucks, mines, cliffs, cannons, and rockets. Jump with the fire button. The Road Runner loses a life when caught by the coyote, hit by a truck or cannonball, landing on a mine, falling into a ravine, or (in later cartridge levels) hit by a falling rock.

This Jaxtari port currently implements **levels 1–4**. The Atari 2600 cartridge has **8 levels** that escalate in difficulty; after level 8 the layouts loop while difficulty continues to rise. Features required for levels 5–8 are summarized below (from the official Atari manual and observed ALE / TAS level structure).

Actions
-------

Observations
------------

* ``player`` / ``enemy`` — ``ObjectObservation`` with facing (``orientation``) and pose/mode (``state``: jump/fall/death; flattened/burnt/rocket)
* ``ravines`` (n=3), ``seeds`` (n=4, ``visual_id`` = pickup type), ``truck``, ``landmine``, ``bullet``, ``cannon``
* ``score``, ``lives``, ``current_level``, ``score_popup`` (bottom-of-screen gain while timer active)

Reward
------

Modifications
-------------

Registered via ``RoadRunnerEnvMod``:

* ``invert_colors``
* ``hue_shift``
* ``invisible_enemy``
* ``harmless_ravines``
* ``no_road_stripes``
* ``invisible_trucks``

Implemented levels (1–4)
------------------------

* **Level 1** — Open highway with trucks and birdseed; coyote chase tutorial.
* **Level 2** — Narrower road with ravines (cliffs) and ravine-linked mines/birdseed.
* **Level 3** — Dynamic road width, trucks, mines, steel-shot signage, coyote roller-skate / speed-phase chase.
* **Level 4** — Offramps with bridges, fixed cannons and bullets, coyote rocket patrol.

Missing levels (5–8) and required features
------------------------------------------

The cartridge ships eight distinct levels. Levels 5–8 are not present in ``DEFAULT_LEVELS``. Reaching ALE/manual parity needs the following.

**Shared systems still incomplete or unused for later stages**

* **Falling rocks** — Manual hazard: deadly boulders that block the road and kill on contact. Not implemented as entities, collision, or sprites. Required especially for **level 6**; also denser in later loops.
* **Steel shot pickup + magnet** — Manual: eating steel shot scores points but Wile E.'s magnet makes escape temporarily harder. Level 3 has steel-shot *signage* and mixed pickup weights, but there is no dedicated steel-shot pickup type or magnet pull / temporary chase boost.
* **Rocket skates (hoverboard charge)** — Manual / TAS: when the coyote scrolls off-screen he charges back on roller/rocket skates. Level 3 uses a speed-phase multiplier and hoverboard sprites exist, but a full off-screen → skate-charge state machine matching ALE is still incomplete for later levels.
* **Level configs 5–8** — New ``LevelConfig`` entries (scroll length, road geometry, spawn tables, coyote modes) wired into ``DEFAULT_LEVELS``, plus end-of-level banners if assets are added.
* **Post–level-8 loop** — Cartridge loops layouts with higher hazard density / coyote aggression; optional for RL use.

**Per-level content to add**

* **Level 5** — Mix of earlier hazards (mines, trucks, birdseed/steel shot) with denser coyote trapping; no unique hazard type beyond combinations already sketched in levels 1–3.
* **Level 6** — Falling-rock segments as the signature hazard (dodge or jump; coyote remains free to chase while rocks block the path).
* **Level 7** — Escalation of level-5 style (more seeds/shot, more luring coyote into traps); typically combines mines/trucks with aggressive skate chase.
* **Level 8** — Level-4-like finale: cannons firing along the road plus coyote rocket capture; completing it finishes one full cartridge loop.

**Assets / constants likely needed**

* Falling-rock sprites and spawn/collision logic.
* Steel-shot sprite as a real pickup (distinct from birdseed / puddle / quad seed) and magnet chase modifiers.
* Optional end-of-level artwork for levels 5–8.
* Balanced spawn intervals and coyote phase tables so difficulty ramps across 5–8 without the level-4 coyote-ahead softlock class of bugs.

Reference: `AtariAge Road Runner manual <https://atariage.com/manual_html_page.php?SoftwareLabelID=412>`_.

Known issues
------------

* Only four of eight cartridge levels are playable.
* Falling rocks and magnet / steel-shot chase interactions are not implemented.
* Coyote skate-charge when fully off-screen may still diverge from ALE timing.
