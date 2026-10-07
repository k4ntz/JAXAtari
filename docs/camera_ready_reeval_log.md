# Camera-ready re-evaluation log

**Purpose:** Environments whose **observations** (OC and/or pixel) or **environment logic** (dynamics, rewards, termination, action mapping, difficulty) changed on `dev` *after* the April/May 2026 paper runs — so curves need re-running before the early-November camera-ready deadline.

**Assumptions**

| Item | Value |
|---|---|
| Original training runs | ~April / May 2026 |
| Change cutoff used here | **2026-04-30** (end of April). If a run finished in early May *after* a listed May commit, that env may still be valid — check your run dates. |
| Baseline for “already on destin” | `origin/dev` as of 2026-09-29 |
| Post-merge target | `dev` after: ObjectObs PRs **#337–#362** (except #354), **#321/#331/#332**, **#363**, **#334**, **#366**, then **PDF-reviewed** `new_envs` content → destin |
| Lab eval PDF gate | WiSe 25/26 eval PDF — only envs **explicitly reviewed there** may merge into `new_envs`/`dev` for now. Newer lab-batch PRs **not in the PDF** stay open / HOLD until reviewed (includes James Bond **#364** retry). |
| Generalized `ObjectObservation` rollout | **2026-02-08** (`7bb5e6bf`) — already in place before April runs; does *not* by itself force re-runs |

**Legend**

| Tag | Meaning |
|---|---|
| `LOGIC` | Step dynamics / reward / termination / action mapping / difficulty — invalidates **pixel and OC** runs |
| `OBS-OC` | Object-centric observation layout/boxes/entities — invalidates **OC** runs (pixel may still be OK) |
| `OBS-PIXEL` | Rendering / layout that changes the RGB frame — invalidates **pixel** runs (OC may still be OK if boxes unchanged) |
| `NEW` | PDF-reviewed env newly landing on `dev`/`new_envs` — needs **first** paper eval, not a re-run |
| `HOLD` | Not in the WiSe 25/26 eval PDF (new student lab batch / unreviewed retry) — **do not merge yet**; no paper eval until reviewed |
| `SKIP` | Touched on `dev` after cutoff, but change is anonymization / test harness / mod-only hooks — **no** base-env re-run |

---

## 1. Must re-run (already on `origin/dev` after 2026-04-30)

Substantive logic and/or observation changes already merged. Ordered by urgency (clearest training impact first).

| Env | Kind | Why (short) | Key dates / commits |
|---|---|---|---|
| **frostbite** | `LOGIC` + `OBS-PIXEL` | ALE constant retune, temperature timing, exit hitbox, rendering fixes | 2026-05-01 … 05-06 |
| **mspacman** | `LOGIC` + `OBS-OC` | Logic optimize; level-complete score fix; Aug ALE alignment + proper OC/actions (#322) | 2026-05-03, 07-26, **08-20** `a169d791` |
| **pacman** | `LOGIC` + `OBS-PIXEL` + `OBS-OC` | Maze/layout/flicker/gameplay + OC | **08-20** `a169d791` |
| **montezumarevenge** | `LOGIC` + `OBS-PIXEL` | Ladder exploit fix; jumping bug; room/lava/renderer work; **plus pending #366** | 04-30 … 07-28 (+ #366) |
| **airraid** | `LOGIC` + `OBS-PIXEL` | Spawn/layout/hitbox/timing closer to ALE | **08-20** `a169d791` |
| **klax** | `LOGIC` + `OBS-PIXEL` | Screen height, spawn timing, well/paddle layout, visuals | **08-20** `a169d791` |
| **surround** | `LOGIC` + `OBS-PIXEL` | Grid/playfield sizes, spawn positions, trail visuals (#235) | **08-20** `a169d791` |
| **tron** | `LOGIC` + `OBS-PIXEL` | Structure/speed/asset_config (#236) | **08-20** `a169d791` |
| **asterix** | `LOGIC` | Action mapping fix (Discrete index vs Atari ids) | **08-20** `a169d791` |
| **beamrider** | `LOGIC` | Parity / ALE complexity tune (#324) | **07-28** `e42c8d7a` |
| **phoenix** | `LOGIC` | Parity / ALE complexity tune (#324) | **07-28** `e42c8d7a` |
| **skiing** | `LOGIC` + `OBS-OC` + `OBS-PIXEL` | Black buffers / solvability; 5‑min reset; top-left OC coords | 05-01 … 06-05 |
| **gravitar** | `LOGIC` + `OBS-OC` | Progression bugfix (May 3); terrain/top-left OC (June) | 05-03, 06-07 |
| **enduro** | `LOGIC` | Spawning bugfix | **05-06** `8bbdb36d` |
| **alien** | `LOGIC` + `OBS-PIXEL` | Reward tweak; sprite fix | 05-02 … 05-03 |
| **asteroids** | `LOGIC` | Perf rewrite + scoring inconsistencies fixed | **05-04** `35f630c2` |
| **spaceinvaders** | `LOGIC` | Gameplay improvements | **05-04** `cd65f1de` |
| **qbert** | `LOGIC` | Speed improvements | **04-30** |
| **venture** | `LOGIC` | Speed improvements | **04-30** |
| **kangaroo** | `LOGIC` | Level-wrap score bonus removed (+1400) | **07-26** `6cb20b7f` |
| **hangman** | `OBS-OC` + `OBS-PIXEL` | Layout constants + OC entities | **08-20** `a169d791` |
| **videocheckers** | `OBS-PIXEL` (+ layout) | Board offsets/colors closer to ALE; controls accessibility notes | **08-20** `a169d791` |
| **centipede** | `OBS-PIXEL` | Renderer optimization (May 4) — pixel frames may shift | **05-04** `ac909db4` |

### Casino split (Aug 20)

| Env | Kind | Why |
|---|---|---|
| **casino_blackjack** | `LOGIC`? / structural | Split from monolithic casino + PyTree/API refactor — **verify** whether reward/dynamics changed; if only packaging, OC/pixel may be OK |
| **casino_five_stud_poker** | same | same |
| **casino_poker_solitaire** | same | same |

Treat as **re-run unless** you confirm dynamics/obs identical to the April/May casino runs under the old names.

---

## 2. Pending merges into `dev` (will add / deepen re-run needs)

Assume these land as planned before camera-ready.

### 2a. ObjectObservation / bbox PRs — `#337–#362` except `#354`

All are **`OBS-OC` only** (unless noted). Pixel runs stay valid if no parallel logic PR.

| Env | PR(s) | Notes |
|---|---|---|
| tetris | #337 | |
| kingkong | #338 | |
| frostbite | #339 | **also** already `LOGIC` above → full re-run |
| centipede | #340 | **also** pixel renderer change above |
| breakout | #341 | |
| asterix | #342 | **also** action-mapping `LOGIC` → full re-run |
| amidar | #343 | OC boxes; scoring constant may land via separate Amidar port → then `LOGIC` too |
| atlantis | #344 | |
| pacman | #345 | **also** heavy Aug logic → full re-run |
| mspacman | #346 | **also** heavy Aug logic → full re-run |
| alien | #347 | **also** May logic → full re-run |
| venture | #348 | **also** Apr speed → full re-run |
| donkeykong | #349 | Overlaps **#363** — prefer #363 as source of truth |
| flagcapture | #350 | |
| berzerk | #351 | |
| spaceinvaders | #352 | **also** May logic → full re-run |
| enduro | #353 | **also** spawn fix → full re-run |
| surround | #355 | **also** Aug logic → full re-run |
| qbert | #356 | **also** Apr speed → full re-run |
| riverraid | #357 | |
| fishingderby | #358 | |
| phoenix | #359 | **also** #324 logic → full re-run |
| sirlancelot | #360 | |
| montezumarevenge | #361 | **also** MZR logic + #366 |
| hauntedhouse | #362 | |

### 2b. Other planned `dev` merges

| PR | Env(s) | Kind | Re-run? |
|---|---|---|---|
| **#363** DK done | donkeykong | `LOGIC` + `OBS-*` + mods | **Yes — full** |
| **#366** MZR fixes | montezumarevenge | `LOGIC` (+ mods); scrub unrelated churn | **Yes — full** (already required) |
| **#334** Boxing | boxing | `NEW` | **First eval** |
| **#321** MsPacMan mods | mspacman | mods only (no `jax_mspacman.py` diff) | Base eval: no *extra* beyond §1 |
| **#331** Centipede mods | centipede | touches lives constants in `jax_centipede.py` | **Yes if** default lives/max-lives affect base episodes (check); else mods-only |
| **#332** ChopperCommand mods | choppercommand | small consts refactor in game file | Likely **SKIP** for base if behavior identical — quick parity check |

### 2c. Separate ports (your call: Amidar / TimePilot / Gopher)

| Work | Env | Kind | Re-run? |
|---|---|---|---|
| Amidar scoring / no_enemies (e.g. `port/amidar-timepilot-preserve`) | amidar | `LOGIC` if `PIXELS_PER_POINT_HORIZONTAL` 3→4 lands | **Yes** for Amidar if that constant merges |
| TimePilot mod hooks / recolor consts | timepilot | mostly mods | Base **SKIP** unless default colors/paths change without mods |
| Gopher | gopher | already on `new_envs` via #224 | See §3 `NEW` — not a re-run of an old paper env unless Gopher was already evaluated from elsewhere |

---

## 3. `new_envs` / student PRs — PDF gate

**Rule:** Merge / first-eval only if the env appears in the WiSe 25/26 eval PDF. Everything else from the newer lab batch is **`HOLD`** until reviewed.

### 3a. HOLD — not in PDF (do not merge yet)

| Env | PR / location | Notes |
|---|---|---|
| **jamesbond** | **#364** (retry) | Separate from failed PDF entry **#229** (grade 5.0). Still **unreviewed** → HOLD |
| **stargunner** | #354 | New lab batch |
| **kungfumaster** | #336 | New lab batch |
| **demonattack** | #316 | New lab batch |
| **bowling** | #299 | New lab batch |
| **assault** | on `origin/new_envs` | Not in PDF — strip/hold out of merge until reviewed |
| **wizardofwor** | on `origin/new_envs` | Not in PDF — strip/hold out of merge until reviewed |

*(Mario Bros was already being moved off `new_envs` in recent commits; keep it out of this merge wave.)*

### 3b. OK to merge into `new_envs` / then `dev` (in PDF) → first eval (`NEW`)

**Already on `origin/new_envs` (PDF-reviewed ports):**

| Env | PDF PR | Notes |
|---|---|---|
| backgammon | #226 | |
| basicmath | #174 | |
| gopher | #224 | |
| journeyescape | #178 | |
| kaboom | #173 | |
| miniature_golf | #186 | |
| othello | #203 | |
| roadrunner | #175 | |
| upndown | #210 | |
| videochess | #225 | |
| yarsrevenge | #202 | |

**Open student PRs (in PDF) — merge into `new_envs` when ready:**

| Env | PR | PDF grade / env rec (short) |
|---|---|---|
| darkchambers | #205 | ~1.0; feature branch (perf) |
| tictactoe3d | #207 | ~1.3; feature branch (use #207, not #269) |
| adventure | #208 | ~1.7; merge-with-issues |
| crossbow | #221 | ~1.7–2.0; feature branch |
| tutankham | #222 | ~1.3; feature branch |
| lostluggage | #223 | ~2.0–2.3; feature branch |
| pitfall | #254 | ~2.3; feature branch |
| defender | #268 | ~1.7 |

**Also `NEW` first eval (maintainer rewrite; PDF had Boxing #241):** boxing via **#334** → `dev`.

**Still close / do not merge (PDF says kill):** Carnival #212, James Bond #229 (old), etc. — see close list.

**Common games** where `new_envs` and `dev` both have `jax_*.py` but differ slightly:  
alien, amidar, asterix, beamrider, donkeykong, enduro, pacman, sirlancelot, tron, venture — prefer post-§1 / `dev` versions for paper evals.

---

## 4. Likely no base-env re-run (`SKIP`)

Touched after cutoff, but not meaningful for April/May **base** curves:

| Env / area | Reason |
|---|---|
| berzerk, blackjack, choppercommand, flagcapture, galaxian, lasergates, spacewar, timepilot (May 3) | Author-name anonymization only (`7cb11b5c`) |
| riverraid, slotmachine, timepilot, enduro (sprite assert) | Datatype assertions for downscaled sprites (`09bfef56`) — only re-check pixel if you used downscaling in eval |
| sirlancelot (2026-09-29) | Downscale param casting |
| videopinball | Removed pygame human-action helper / test crash — not training dynamics |
| tennis, bankheist, phoenix, … via #298 mods merge | Mods packaging; base step unchanged unless you trained **with** those mods |
| Framework: wrappers JIT / NoiseWrapper / PQN baseline | Does not change env step; only re-run if your **training stack** depended on old wrapper semantics |

**Bankheist** “Optimizing” (May 1): large rewrite — **spot-check** reward/episode length vs old run; if parity holds, can `SKIP`, else treat as `LOGIC`.

---

## 5. Compact checklist — “do I need a new training curve?”

### Full re-run (pixel **and** OC) — already on destin + pending

```
airraid, alien, asterix, asteroids, beamrider, enduro, frostbite,
gravitar, kangaroo, klax, montezumarevenge (+#366), mspacman, pacman,
phoenix, qbert, skiing, spaceinvaders, surround, tron, venture,
donkeykong (+#363),
amidar (if scoring port lands)
```

### OC-only re-run (pixel OK if no other row lists them)

```
atlantis, breakout, fishingderby, flagcapture, hauntedhouse, kingkong,
riverraid, sirlancelot, tetris, berzerk (#351),
+ any env in §2a not already in the full-re-run list
```

### Pixel-only extra attention

```
centipede (renderer), videocheckers, hangman (also OC),
casino_* (verify)
```

### First eval (`NEW`) — PDF-reviewed only

```
boxing (#334),
backgammon, basicmath, gopher, journeyescape, kaboom, miniature_golf,
othello, roadrunner, upndown, videochess, yarsrevenge,
darkchambers (#205), tictactoe3d (#207), adventure (#208),
crossbow (#221), tutankham (#222), lostluggage (#223),
pitfall (#254), defender (#268)
```

### HOLD — not in PDF (no merge / no paper eval yet)

```
jamesbond (#364 retry), stargunner (#354), kungfumaster (#336),
demonattack (#316), bowling (#299),
assault (on new_envs), wizardofwor (on new_envs)
```

### No re-run for base paper curves

```
Most May-3 anonymization-only games; pure mod PRs #321;
TimePilot if only mod/recolor hooks
```

---

## 3c. Updated merge-into-`new_envs` order (PDF-only)

```
205 Dark Chambers → 207 TicTacToe3D → 222 Tutankham → 208 Adventure
→ 268 Defender → 221 Crossbow → 223 Lost Luggage → 254 Pitfall
```

**Do not merge yet:** #364, #354, #336, #316, #299 (+ assault / wizardofwor if present on the branch).

---

## 6. Method / how to refresh this log

```bash
# Substantive game-file history since cutoff on destin
git log --since=2026-04-30 --oneline origin/dev -- 'src/jaxtari/games/jax_*.py' \
  'src/jaxtari/games/montezuma_revenge/'

# After merging a PR, append the env to the right section above.
```

**Maintainer note:** Prefer one row per env; if both `LOGIC` and `OBS-OC` apply, always full re-run. When in doubt for camera-ready, re-run — November is close and April/May curves are stale for anything in §1–§2.

*Generated 2026-09-29 from `origin/dev` / `origin/new_envs` / open PR refs / WiSe 25/26 eval PDF. Updated same day: HOLD non-PDF lab-batch envs (incl. James Bond #364). Refresh after the merge wave completes.*
