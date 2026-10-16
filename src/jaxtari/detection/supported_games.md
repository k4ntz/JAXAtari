# Detector support (`gray96`)

Games with a `gray96` detector checkpoint (YOLOv8n on 96×96 grayscale frames from JAXtari native downscaling). Checkpoints live in `<storage>/detectors/gray96/<game>.npz` (`jaxtari.paths.get_base_detector_dir()`) and are used as `YOLOObjectCentricWrapper(env, "gray96/<game>")`.

Metrics: `train_detection/evaluate.py --steps 500` (random policy, wrapper observations vs. game state, match within 3 px). *Tier*: good = recall and precision ≥ 90%, usable ≥ 70%, weak below. *Weak fields*: fields with recall < 50% (≥ 20 visible ground-truth objects).

To add or retrain a game see [README.md](../../../train_detection/README.md#adding-or-improving-a-detector).

## Supported (59 games)

| Game | Tier | Recall | Precision | Pos. err (px) | Weak fields |
|---|---|---|---|---|---|
| airraid | usable | 88.4% | 92.3% | 0.70 | enemy_missiles (23.7%) |
| alien | usable | 85.2% | 97.3% | 0.82 | – |
| amidar | weak | 44.5% | 57.1% | 0.59 | player_gorilla (0.0%), enemy (6.6%), paths (46.2%) |
| assault | good | 97.0% | 99.4% | 0.47 | – |
| asterix | good | 96.1% | 97.5% | 0.53 | – |
| asteroids | weak | 64.0% | 74.6% | 0.95 | player (0.0%), missiles (0.0%) |
| atlantis | good | 98.7% | 99.0% | 1.07 | – |
| bankheist | weak | 68.0% | 96.7% | 0.95 | player (36.6%), enemies (0.0%), dynamite (6.9%) |
| beamrider | usable | 73.1% | 89.3% | 0.89 | enemy_shots (26.7%), rejuvenator (0.0%) |
| berzerk | good | 94.1% | 95.4% | 1.18 | enemy_bullets (0.0%) |
| boxing | good | 100.0% | 100.0% | 0.44 | – |
| breakout | weak | 41.5% | 100.0% | 0.39 | blocks_bot (49.9%), blocks_mid (33.3%), blocks_top (38.9%) |
| centipede | usable | 83.5% | 97.5% | 0.79 | centipede (41.3%), spider (33.2%) |
| choppercommand | weak | 61.9% | 73.7% | 1.40 | jets (2.3%), enemy_missiles (46.3%), player_missiles (19.4%) |
| donkeykong | good | 94.2% | 97.3% | 0.12 | – |
| enduro | usable | 78.8% | 86.0% | 0.87 | – |
| fishingderby | good | 92.7% | 98.7% | 0.74 | – |
| flagcapture | good | 98.4% | 100.0% | 0.00 | player (0.0%) |
| freeway | good | 96.2% | 98.8% | 0.69 | – |
| frostbite | usable | 84.5% | 89.1% | 1.03 | obstacles (22.8%) |
| galaxian | usable | 87.2% | 98.3% | 0.46 | – |
| gopher | weak | 50.3% | 71.0% | 0.34 | gopher (0.0%), holes (20.0%) |
| gravitar | usable | 75.4% | 86.1% | 0.74 | ship (36.6%), saucer (40.8%) |
| hauntedhouse | good | 100.0% | 100.0% | 0.39 | – |
| humancannonball | usable | 89.6% | 93.2% | 0.13 | human (40.0%) |
| journeyescape | good | 93.7% | 95.9% | 1.21 | – |
| kaboom | good | 97.9% | 99.5% | 0.61 | – |
| kangaroo | good | 94.1% | 98.8% | 0.39 | falling_coconut (22.0%), thrown_coconuts (0.0%) |
| kingkong | usable | 86.2% | 91.0% | 1.07 | – |
| klax | usable | 72.6% | 96.8% | 0.76 | board (0.0%) |
| lasergates | weak | 57.2% | 68.6% | 0.63 | player (28.0%), homing_missile (4.2%), densepack (0.0%) |
| miniaturegolf | usable | 79.5% | 91.3% | 0.64 | ball (33.6%) |
| montezumarevenge | usable | 84.4% | 85.2% | 0.19 | enemies (0.0%) |
| mspacman | usable | 71.4% | 96.7% | 0.50 | ghosts (16.9%) |
| namethisgame | weak | 49.2% | 52.8% | 0.61 | tentacles (29.8%), oxygen_line (0.0%) |
| pacman | weak | 64.4% | 92.9% | 0.91 | player (14.1%), ghosts (16.7%), power_pellets (0.0%) |
| phoenix | usable | 88.6% | 98.0% | 0.81 | – |
| pong | good | 97.2% | 99.3% | 0.57 | – |
| qbert | usable | 89.8% | 92.2% | 0.99 | purple_ball (0.0%) |
| riverraid | weak | 83.8% | 49.0% | 0.92 | – |
| roadrunner | usable | 88.7% | 94.3% | 1.09 | seeds (0.0%) |
| seaquest | good | 91.7% | 96.3% | 0.64 | – |
| sirlancelot | usable | 86.8% | 92.1% | 0.69 | – |
| skiing | usable | 74.9% | 91.8% | 0.71 | skier (24.6%) |
| slotmachine | weak | 0.0% | 0.0% | – | reel_symbols (0.0%) |
| spaceinvaders | good | 94.0% | 98.9% | 1.05 | player_bullet (0.0%), enemy_bullets (0.0%) |
| spacewar | usable | 72.0% | 78.6% | 0.23 | player (20.4%), player_missile (35.2%) |
| surround | weak | 49.2% | 90.5% | 1.17 | trail (42.1%) |
| tennis | usable | 85.4% | 93.7% | 0.64 | – |
| tetris | weak | 33.7% | 64.0% | 0.28 | board (31.8%), active_piece (42.4%) |
| timepilot | usable | 71.9% | 80.7% | 0.98 | player (38.0%), player_missile (34.5%), enemy_missiles (0.0%) |
| tron | usable | 80.5% | 96.8% | 0.69 | discs (16.4%) |
| turmoil | good | 94.0% | 96.1% | 0.65 | – |
| upndown | good | 95.0% | 94.6% | 0.84 | flags (40.4%) |
| venture | good | 93.1% | 96.6% | 0.97 | – |
| videocube | weak | 28.6% | 89.1% | 0.28 | cube_current_view (0.0%) |
| videopinball | usable | 89.3% | 94.1% | 0.12 | ball (16.6%) |
| wordzapper | usable | 76.0% | 83.4% | 0.54 | player_missile (38.2%), player_zapper (5.7%) |
| yarsrevenge | weak | 58.3% | 93.9% | 1.00 | destroyer (12.4%), swirl (0.0%), energy_missile (18.5%) |

Notes on weak results:

* Sprites drawn on alternating frames (e.g. Asteroids) are missed in the frames where they are not drawn (try the wrapper's `max_pool=True`).
* Objects of ~1 px (bullets, missiles) are largely lost at 96×96.
* Fields sharing a class and size (e.g. Chopper Command jets / choppers) are routed by position only and can end up in the wrong field.
* Dense structure fields (Amidar paths, Tetris board, Gopher tunnels) overlap heavily after the 8 px label floor; the Breakout wall has more blocks than the 12×12 stride-8 grid of a 96×96 YOLOv8 can separate.

## Not supported (11 games)

| Game | Reason |
|---|---|
| backgammon | board / card / text game, no registry entry (box detection is a poor fit) |
| basicmath | board / card / text game, no registry entry (box detection is a poor fit) |
| blackjack | no spatial objects in the registry |
| casinoblackjack | card game, no spatial objects |
| casinofivestudpoker | board / card / text game, no registry entry (box detection is a poor fit) |
| casinopokersolitaire | board / card / text game, no registry entry (box detection is a poor fit) |
| hangman | board / card / text game, no registry entry (box detection is a poor fit) |
| othello | board / card / text game, no registry entry (box detection is a poor fit) |
| videocheckers | board / card / text game, no registry entry (box detection is a poor fit) |
| videochess | board / card / text game, no registry entry (box detection is a poor fit) |
| wizardofwor | observation is in logical maze coordinates (renderer scales x by 19/16, y by 2); labels would need a coordinate transform the registry does not support |
