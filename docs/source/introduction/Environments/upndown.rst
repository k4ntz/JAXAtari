Up N Down
=========

.. raw:: html

       <img src="../../_static/gifs/upndown.gif" alt="Up N Down" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>

Registry name: ``upndown``. Parity status in ``games_covered.md``: 🥉.

Description
-----------

Drive a Baja Bugger along branching hillside roads, collect eight colored
flags, pick up prizes, and avoid or jump onto other cars. Jump with FIRE;
UP/DOWN change speed (no LEFT/RIGHT — lane changes are jumps between roads).

This JAXAtari port implements **three looping track layouts** (``LEVEL_COUNT=3``)
with hazard water on layout 3. The Atari 2600 cartridge progresses through
**rounds 1–9** with escalating difficulty; rounds beyond the three baked
geometries and remaining gameplay gaps are summarized below (from the
`Sega / Bally Midway 2600 manual <https://atariage.com/manual_html_page.php?SoftwareLabelID=574>`_
and ALE / OCAtari play).

Actions
-------

Discrete set: ``NOOP``, ``FIRE``, ``UP``, ``DOWN``, ``UPFIRE``, ``DOWNFIRE``.
Unsupported stick directions map to ``NOOP`` in the comparison / record scripts.

Observations
------------

Screen-space ``ObjectObservation`` entities plus HUD:

* ``player`` — car (``state`` = jumping, ``orientation`` = lean/facing)
* ``enemies`` (n=8; ``visual_id`` = type, ``state`` = speed)
* ``flags`` (n=8; ``visual_id`` = color; active only while on-screen and uncollected)
* ``collectibles`` (n=4; ``visual_id`` = type, ``state`` = color)
* ``flags_collected_mask`` — top HUD blackouts
* ``score``, ``lives``, ``level``

Reward
------

Scoring aims at **ALE RAM behavior**, which often differs from the printed
manual table (see gaps below). Flags currently award ``FLAG_COLLECTION_SCORE``
(100). Collectibles use ~10× the manual table. Late-jump enemy destroys use a
flat ``LATE_JUMP_ENEMY_SCORE`` (400), not per-type manual values.

Modifications
-------------

Registered via ``UpNDownEnvMod``:

* ``allow_jump_backwards``
* ``remove_step_roads``
* ``higher_player_speed``
* ``spawn_more_collectibles``
* ``minimum_car_spawn_gap``
* ``single_lane_car_spawn``
* ``progressive_car_spawn_rate``
* ``collectible_value_time_decay``

Implemented (core loop)
-----------------------

* Two-lane zig-zag tracks with intersections; jump between roads or along a road.
* Eight colored flags per round; HUD check-off; all-flags bonus then level advance.
* Enemy pool (pickup / flag carrier / Camaro / truck types) with spawn, despawn,
  rare direction reverses, ground crash, and late-jump destroy + flash.
* Collectibles: cherry, balloon, lollypop, ice cream.
* Lives HUD, round digit, crash / destroy blink, auto-start after ~64 frozen frames.
* Layout 3 water hazard zones (grounded contact loses a life).

Missing / incomplete vs cartridge manual
----------------------------------------

**Rounds**

* **Rounds 4–9** — Manual: courses escalate through road nine. JAX cycles three
  corner tables via ``level % 3``. Distinct round-4+ layouts, denser opponents,
  and “flag carrier you must jump” chase rounds are not fully modeled.

**Enemies and scoring**

* **Per-type jump-on scores** — Manual: pickup 100, flag carrier 125, Camaro 150,
  truck 175. JAX awards a flat 400 on late-jump destroy (closer to some ALE
  observations than the manual, but not type-split).
* **Manual collectible / flag table** — Printed: flag 75, cherry 50, balloon 65,
  lollypop 70, ice cream 75. JAX/ALE path uses flag 100 and collectibles
  600/600/700/750 (~10× manual). Documented intentionally for ALE parity; not
  manual-table parity.
* **Flag-carrying opponents** — Manual / AtariProtos: later rounds require
  jumping cars that carry needed flags. Types exist in constants, but dedicated
  “must destroy to clear flag” chase logic is incomplete.
* **Enemy density / AI** — Tuned toward ALE’s ~2–3 on-screen trucks
  (``ENEMY_MAX_VISIBLE_COUNT=3``, rare ``ENEMY_DIRECTION_SWITCH_PROB``). Still a
  procedural approximation of ROM spawn tables, not a cycle-accurate AI dump.

**Controls**

* **Jump while reversing** — Manual: jump only while moving forward; pull-back
  mid-jump reverses after the jump completes. Mid-air UP/DOWN accel matches
  ground (ALE); reverse-jump edge cases may still diverge.

**Presentation**

* **Audio / music** — Not in scope for JAXAtari RGB envs.
* **Exact sprite / color / scroll parity** — Ongoing; see
  ``docs/issue-reports/upndown-parity/``.

Reference: `AtariAge Up n' Down manual <https://atariage.com/manual_html_page.php?SoftwareLabelID=574>`_.

Known issues
------------

* Only three track geometries; cartridge rounds 4–9 not distinct.
* Jump-on-enemy and collectible scores follow ALE-oriented constants, not the
  printed manual table; per-type jump scores missing.
* Full ALE-diff writeup still in progress; record with
  ``scripts/oc_parity/record_upndown.sh``.
