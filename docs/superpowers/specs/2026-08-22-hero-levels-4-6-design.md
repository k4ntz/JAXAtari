# H.E.R.O. Levels 4–6 — Design

**Date:** 2026-08-22
**Status:** Approved by user (brainstorming session)
**Prior art:** levels 1–3 in `src/jaxatari/games/jax_hero.py` + `hero_levels.py`,
built from direct ROM measurement (see the provenance notes in `jax_hero.py`).

## Goal

Extend `jax_hero` from levels 1–3 to levels 1–6. Levels 4–6 are measured from
the real Activision H.E.R.O. ROM running under ALE, with the same provenance
standard as levels 1–3. Every mechanic the ROM actually shows in levels 4–6 is
implemented.

**Success criteria**

1. All six levels are completable end-to-end by a scripted autoplayer
   (fresh-start per level and the full 1→6 run).
2. The full test suite passes; the existing 26 tests pass unmodified.
3. Levels 1–3 behave identically to today (new mechanics are guarded by
   empty-by-default data tables).
4. The HUD/UI is unchanged.

## Decisions (user-approved)

- **Ground truth: the real ROM.** The design docs in `b:\TUD\2026_Summer\AI
  project\` describe L4–6 (magma at L4, water at L5, rafts at L6) but were
  unreliable for L1–3; they are used only as hints for what to probe.
- **Scope: everything the ROM shows** in L4–6 — deadly walls, lanterns /
  darkness, water/rafts, new creature kinds — whatever probing confirms.
- **Playability-first tuning is allowed and documented** (precedent: dynamite
  fuse 60 instead of the measured ~26; dynamite breaks walls instead of the
  measured laser-melt).
- **Wall-breaking convention carries over:** any wall blocking the descent
  path is dynamite-breakable (`dynamite_ok=1`); the laser never harms walls.
- **Capture tooling is committed** (`scripts/hero_capture_auto.py`) so it
  survives for future levels 7+, unlike the lost scratchpad scripts.

## Phase 1 — Capture tooling (`scripts/hero_capture_auto.py`)

Rebuild the automated pipeline from the documented method:

- RAM map: `player_x=27`, `player_y=31` (y = 145 − screen_y, wraps 137→1 per
  room), `room=28`, `power=43`, `lives=51`, `level=117`, `dynamite=50`.
- **Teleport-descend**: write Y RAM as `max(0, y−2)`; the game must see `y==0`
  itself to flip rooms. Teleport a few px ABOVE the floor and let the player
  land naturally (writing Y at exact standing height embeds him 1 px into the
  floor and he tunnels through).
- **Median background capture**: hold X/Y constant mid-room; per-pixel median
  over ~60 frames removes moving sprites.
- **Level advance**: teleport to the miner (pink head (228,111,111) with green
  (92,186,92) below), then go hands-off — writing POWER during the end
  sequence blocks the power→score drain and the level never advances.
- **Rescue detection**: player Y frozen ≥14 frames.
- Lives RAM top-up prevents game over; deaths respawn at the top of the
  CURRENT room.
- `ale.cloneState()/restoreState()` for branching probes.
- Self-validation: before capturing L4–6, the tool re-derives L1 room 0 and
  checks it against the known `hero_levels.py` data.
- Shortcut: ALE `mode=1` (game 2) starts at level 5 if supported; otherwise
  advance by rescuing miners.

## Phase 2 — Measurement & probing of levels 4–6

Per room:

- Background → per-level palette → RLE+base64 blob (existing format,
  `decode_bg` compatible).
- Wall rects decomposed from background pixels, spot-verified by collision
  probes (teleport adjacent, walk into, observe blocking).
- **Hazard probes**: walk into each distinct wall/floor type and watch the
  lives byte — classifies solid vs deadly (magma/lava).
- **Creature census**: multi-frame sprite tracking per room → kind
  (appearance), position, patrol range; clone/restore probes for laser and
  dynamite kill-ability. Near-static creatures found via background scans
  (L1–3 precedent: grey-thread column scans).
- **Lantern probe**: if a lantern-like object exists, shoot and touch it,
  observe whether the room blacks out.
- **Water/raft detection**: river rows + moving-platform tracking if present.
- **Destructibility probe**: both weapons against every interior wall on the
  descent path (full sweep where cheap).

## Phase 3 — Data model (`hero_levels.py`)

- `NUM_LEVELS = 6`; `ROOMS_PER_LEVEL` gains the measured counts.
- New per-level tables: `PALETTE_L4..6`, `BG_RLE_L4..6`, `WALL_RECTS_L4..6`,
  `MINER_POS`, `DESTRUCTIBLE`, `SPIDERS` entries.
- New object tables, **empty lists for L1–3**:
  - `DEADLY`: (room, x, y, w, h) — touch kills.
  - `LANTERNS`: (room, x, y) — shot/touch darkens that room.
  - Water/raft tables only if measured to exist (shape decided then).
- Creature entries gain a `kind` field (0=spider, 1=bat, 2=snake, …) used for
  sprite + motion variant. The table and observation key stay named `spiders`
  for backward compatibility (documented as "creatures").

## Phase 4 — Engine (`jax_hero.py`)

- `_build_level_arrays` packs the new tables into padded arrays
  (`DEADLY_*`, `LANTERN_*`, creature `kind`).
- `step()` gains guarded blocks (no-ops on empty tables):
  - deadly-rect AABB touch → death (same priority as creature touch);
  - laser AABB or player touch on a live lantern → that room's dark flag set
    (state: `room_dark` bool array sized `max_rooms`, reset on level advance);
  - raft: moving platform; standing player inherits raft vx; water touch →
    death (only if water/rafts are measured to exist).
- New state fields reset correctly on respawn (keep) and level advance
  (reset), mirroring `wall_stage` handling.
- Renderer: background stack grows to 6 levels; new creature sprite art per
  kind; darkness renders the cave background black while player, creatures,
  dynamite/explosion and the HUD stay visible; raft sprite if applicable.

## Phase 5 — Tests & verification

- New unit tests per confirmed mechanic: deadly wall kills; lantern darkens
  the room and resets next level; raft carries the player; new creature kinds
  die to laser/blast and kill on touch; level-advance chain 1→6;
  reset-inventory invariants at L4+.
- The existing render-all-rooms test loops `num_levels`, so it covers L4–6
  automatically.
- Scripted autoplayer beats each of L4–6 from a fresh start and the full 1→6
  run.
- Existing 26 tests pass unmodified.

## Risks

- Deeper-level RAM semantics may differ (y wrap, room byte) → the tool
  self-validates against L1 before trusting L4–6 output.
- A mechanic may not fit the rect model (e.g., moving walls) → special-cased
  in state, decided when measured.
- Probing every wall of ~20+ new rooms is slow → full rigor on the descent
  path, cheaper sweeps elsewhere.
- ALE `mode` support for level starts is unverified → fallback is advancing
  by scripted rescues.
