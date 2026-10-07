"""Runnable JamesBond skeleton environment.

This file intentionally defines only the shared environment contract and minimal
placeholder behavior. Gameplay systems such as object spawning, collisions,
scoring, lives, and sprite-accurate rendering are left for follow-up work.
"""
import os

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

## Sprites live in the repo (src/jaxatari/jb_sprites), not in the downloaded
## sprite pack, so the renderer must load them from here.
## NOTE: background.npy, bullet.npy and score_6..9.npy are placeholder
## sprites so the environment can run; the sprite task owner should replace
## them with real extractions.
JB_SPRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jb_sprites")


def get_default_asset_config() -> tuple:
        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'black_border', 'type': 'single', 'file': 'black_border.npy'}, ## For not showing sprites at ends (x < 3, x > 207) of the screen.
            {'name': 'ground', 'type': 'single', 'file': 'ground.npy'},
            {
                'name': 'car', 'type': 'group',
                ## All three recolors feed the death color-cycle
                'files': ['car.npy', 'car_dead_1.npy', 'car_dead_2.npy', 'car_dead_3.npy']
            },
            {'name': 'satellite', 'type': 'single', 'file': 'satellite.npy'},
            {
                'name': 'helicopter', 'type': 'group',
                'files': ['helicopter_1.npy', 'helicopter_2.npy']
            },
            {
                'name': 'helicopter_melee', 'type': 'group',
                ## The searchlight sweep table indexes sprites 0..14, so the
                ## whole extracted sequence has to be here (the group loader
                ## pads the differing widths).
                'files': [f'helicopter_shot_{i}.npy' for i in range(1, 17)]
            },
            {
                'name': 'pit', 'type': 'group',
                'files': ['fire_pit_1.npy', 'fire_pit_2.npy']
            },
            {
                'name': 'diamond', 'type': 'group',
                'files': ['diamond_1.npy', 'diamond_2.npy']
            },
            {
                'name': 'stars', 'type': 'group',
                'files': ['stars_1.npy', 'stars_2.npy']
            },
            ## Water scene terrain and actors
            {'name': 'water', 'type': 'single', 'file': 'water.npy'},
            ## seabed_full is the complete 160px repeating strip including
            ## the 14-column valley gap; the old seabed.npy had the gap
            ## columns deleted, which tiled a valley-less seabed.
            {'name': 'seabed', 'type': 'single', 'file': 'seabed_full.npy'},
            {'name': 'water_sky', 'type': 'single', 'file': 'water_sky.npy'}, ## solid 74,74,74 measured in ALE
            ## The splash frogman's two poses, both pixel-exact extractions
            {
                'name': 'splash', 'type': 'group', 
                'files': ['explosion_1_(small).npy', 'explosion_2.npy']
            },
            ## A bolt sinking past a living frogman is drawn in his colors
            {'name': 'laser_green', 'type': 'single', 'file': 'laser_green.npy'},
            ## Second water scene: darker water and its roster, all cropped
            ## from real ALE frames of that scene
            {'name': 'water_b', 'type': 'single', 'file': 'water_b.npy'},
            {'name': 'sky_flash', 'type': 'single', 'file': 'sky_flash.npy'}, ## whole sky flashes gray when the rocket bursts
            {'name': 'death_flash', 'type': 'single', 'file': 'death_flash.npy'}, ## the sky's one-frame flash on a water death
            {
                'name': 'rocket', 'type': 'group', 
                'files': ['rocket_w_ignition.npy', 'rocket_wo_ignition.npy']
            },
            {
                'name': 'submarine', 'type': 'group', 
                'files': ['boat_left.npy', 'boat_right.npy']
            },
            ## Water B's pink ball (the old "pink helicopter"): a 9x11
            ## sphere block-sampled from the longplay, solid and striped
            ## poses alternating in flight.
            {
                'name': 'wb_ball', 'type': 'group',
                'files': ['ball_sideways.npy', 'ball_frontal.npy']
            },
            {
                'name': 'rocket_ball', 'type': 'group', 
                'files': ['rocket_ball.npy', 'rocket_ball_narrow.npy']
            },
            {
                'name': 'scuba', 'type': 'group',
                'files': ['scuba_1.npy', 'scuba_2.npy']
            },

            {'name': 'life', 'type': 'single', 'file': 'car_life.npy'},
            {'name': 'oil_rig', 'type': 'group', 'files': ['oil_rig.npy', 'oil_rig.npy']},

            {
                'name': 'score_digits', 'type': 'digits',
                'pattern': 'score_{}.npy'
            },
            {
                'name': 'bullet', 'type': 'group',
                'files': ['bullet.npy', 'w_bullet.npy'] ## w_bullet is the player's water bullet
            },
            ## The submarine's shot: two yellow 2x2 dots stacked with a gap
            ## (block-sampled from the longplay)
            {
                'name': 'sub_shot', 'type': 'group', 
                'files': ['boat_bullet_wide.npy', 'boat_bullet_sideways.npy']
            },
            ## Rocket debris that reached the waterline: a red / pink
            ## sparkle alternating between two dot patterns (video)
            {
                'name': 'debris_splash', 'type': 'group',
                'files': ['rocket_splash_1.npy', 'rocket_splash_2.npy']
            },
        ]
        return asset_config

def _aabb_overlap(
    ax: chex.Array,
    ay: chex.Array,
    aw: chex.Array,
    ah: chex.Array,
    bx: chex.Array,
    by: chex.Array,
    bw: chex.Array,
    bh: chex.Array,
) -> chex.Array:
    """Return whether two top-left anchored AABB rectangles overlap."""

    x_overlap = jnp.logical_and(ax < bx + bw, ax + aw > bx)
    y_overlap = jnp.logical_and(ay < by + bh, ay + ah > by)
    return jnp.logical_and(x_overlap, y_overlap)


class JamesBondConstants(struct.PyTreeNode):
    """Static JamesBond placeholder constants shared by state, spaces, and render."""

    # Atari-style frame dimensions and initial play-area bounds.
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    ## Already has player sprite width/height respected
    GAME_AREA_MIN_X: int = struct.field(pytree_node=False, default=4) ## Playable Area: 5 (Coordinate system starting with 1)
    GAME_AREA_MAX_X: int = struct.field(pytree_node=False, default=73) ## Playable Area: 81 (Coordinate system starting with 1)
    GAME_AREA_MIN_Y: int = struct.field(pytree_node=False, default=0) ## Playable Area: 123 (Top-left coordinate system); 87 (Bottom-right co-sys)
    GAME_AREA_MAX_Y: int = struct.field(pytree_node=False, default=119)
    ## GAME_AREA_MAX_X above is the PLAYER's hard stop (measured: the hull
    ## stops exactly at columns 73-80). World objects use the real screen
    ## edges: they enter on the right around column 150-158 and leave on
    ## the left, exactly like in ALE.
    OBJECT_SPAWN_X: int = struct.field(pytree_node=False, default=150) ## helicopter entry column
    OBJECT_SPAWN_X_FAR: int = struct.field(pytree_node=False, default=158) ## diamond / pit entry column
    OBJECT_EXIT_X: int = struct.field(pytree_node=False, default=159) ## rightward movers leave here

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=4)
    PLAYER_INIT_X: int = struct.field(pytree_node=False, default=29) ## 30 if starting with 1
    PLAYER_INIT_Y: int = struct.field(pytree_node=False, default=119) ## 120 if starting with 1 (Not 122?)
    PLAYER_IN_Y_STEPS = jnp.array([ ## For the gravity feel of jumps. Each jump is 71 frames, 72nd frame is the start of the fall
        0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 
        0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1 ## Water matrix: 111010101110011010101010010010100100010000100000001000000000-1; Nearly the same as air. Also, can jump higher and faster from water to air
    ], dtype=jnp.int32)
    PLAYER_WATER_BULLET_STEPS = jnp.array([
        (2, -1), (2, -1), (2, -1), (2, -1),
        (1, 1),  (1, 2),  (1, 1),  (1, 2)  ## And then (1, 0), (0, 2) until 60 frames
    ], dtype=jnp.int32)

    MAX_LIVES: int = struct.field(pytree_node=False, default=6) ## the real game starts with 6 (both ALE agents measured it)
    MAX_DIAMONDS: int = struct.field(pytree_node=False, default=8)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_HELICOPTERS: int = struct.field(pytree_node=False, default=4)
    MAX_SATELLITES: int = struct.field(pytree_node=False, default=4)
    ## Reaching the second water scene takes ~9000 clean frames (land
    ## 4454 + water 4435 + death freezes), so the old 5000 cap ended every
    ## episode before the dock bonus could ever pay out.
    MAX_EPISODE_STEPS: int = struct.field(pytree_node=False, default=20000)

    DIAMOND_WIDTH: int = struct.field(pytree_node=False, default=8)
    DIAMOND_HEIGHT: int = struct.field(pytree_node=False, default=11)
    HELICOPTER_ENEMY_WIDTH: int = struct.field(pytree_node=False, default=8)
    HELICOPTER_ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=6)
    HELICOPTER_MELEE_SPRITE_STEPS = jnp.array([ ## 2nd elements are x positions of sprites. Follows the sequence: Sprite 1 -> Nothing -> Sprite 1 -> Nothing -> Sprite 2 -> ...
        (-1,-1), (0,5), (-1,-1), (1,4), (-1,-1), (1,4), (-1,-1), (2,3), (-1,-1), (2,3), 
        (-1,-1), (3,0), (-1,-1), (3,0), (-1,-1), (4,-8), (-1,-1), (4,-8), (-1,-1), (5,-15), 
        (-1,-1), (5,-15), (-1,-1), (6,-22), (-1,-1), (6,-22), (-1,-1), (7,-30), (-1,-1), (7,-38),
        (-1,-1), (8,-46), (-1,-1), (8,-46), (-1,-1), (9,-54), (-1,-1), (9,-54), (-1,-1), (10,9),
        (-1,-1), (10,9), (-1,-1), (11,6), (-1,-1), (11,6), (-1,-1), (12,5), (-1,-1), (12,5),
        (-1,-1), (13,4), (-1,-1), (13,4), (-1,-1), (14,3), (-1,-1), (14,3), (-1,-1), (1,4),
        (-1,-1), (1,4), (-1,-1), (2,3), (-1,-1), (2,3), (-1,-1), (3,0), (-1,-1), (3,0),
        (-1,-1), (4,-8), (-1,-1), (4,-8), (-1,-1), (5,-15), (-1,-1), (5,-15), (-1,-1), (6,-22), 
        (-1,-1), (6,-22), (-1,-1), (7,-30), (-1,-1), (7,-30), (-1,-1), (8,-38), (-1,-1), (8,-38)
    ], dtype=jnp.int32)
    SATELLITE_ENEMY_WIDTH: int = struct.field(pytree_node=False, default=8)
    SATELLITE_ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=14)
    BULLET_WIDTH: int = struct.field(pytree_node=False, default=1)
    BULLET_HEIGHT: int = struct.field(pytree_node=False, default=4)
    ## Fire pit sprite size, re-extracted from real frames: the crater top
    ## pokes 2px above the road and the ember tail reaches ~38 rows down.
    PIT_WIDTH: int = struct.field(pytree_node=False, default=16)
    PIT_HEIGHT: int = struct.field(pytree_node=False, default=40)

    # Collision boxes are kept a little smaller than the real sprite sizes so
    # near-misses do not register, matching how the original game feels.
    PLAYER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)   ## < PLAYER_WIDTH 8
    PLAYER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=3)  ## < PLAYER_HEIGHT 4
    DIAMOND_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=5)  ## < DIAMOND_WIDTH 8
    DIAMOND_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=6) ## < DIAMOND_HEIGHT 11
    HELICOPTER_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)  ## < HELICOPTER_ENEMY_WIDTH 8
    HELICOPTER_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=4) ## < HELICOPTER_ENEMY_HEIGHT 6
    SATELLITE_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=6)   ## < SATELLITE_ENEMY_WIDTH 8
    SATELLITE_COLLISION_HEIGHT: int = struct.field(pytree_node=False, default=12) ## < SATELLITE_ENEMY_HEIGHT 14
    PIT_COLLISION_WIDTH: int = struct.field(pytree_node=False, default=12)  ## < PIT_WIDTH 16, edge taps survivable

    SCORE_DIAMOND: int = struct.field(pytree_node=False, default=50) ## Manual scoring table says diamond = 50
    SCORE_ENEMY: int = struct.field(pytree_node=False, default=250)
    HIT_COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=60)
    ## Losing a life freezes the whole scene for exactly this many frames
    ## (measured: sprite color-cycles, scroll and every actor stand still),
    ## then the player respawns and the sky actors are cleared.
    DEATH_ANIMATION_FRAMES: int = struct.field(pytree_node=False, default=59)

    ## Enemy fire: one helicopter bomb and one satellite laser, single objects
    ## like everything else. Numbers checked against the real ROM in ALE frame
    ## by frame, not guessed.
    ## The bomb picks its direction once when it drops: 1px per frame towards
    ## the side the player is on, and keeps it for the whole fall. Found while
    ## playing (bombs fall left AND right diagonal) and confirmed in ALE by
    ## teleporting the player around via RAM.
    HELICOPTER_BOMB_SPEED_X: int = struct.field(pytree_node=False, default=1)
    HELICOPTER_BOMB_VY: int = struct.field(pytree_node=False, default=2) ## falls 2px per frame
    ## Drop trigger is player relative, not screen relative. Measured in ALE:
    ## first bomb releases when the heli closes to ~66-70 real px of the player
    ## (that is ~31 in our half width coordinates), later bombs at ~30 real px
    ## (~14 here). The old fixed searchlight zone only matched because the test
    ## player never moved.
    ## treatment, talk to Indi before touching it.
    HELICOPTER_BOMB_RANGE: int = struct.field(pytree_node=False, default=75) ## drops observed at 46-75px on the approach side
    HELICOPTER_BOMB_RETRY_FRAMES: int = struct.field(pytree_node=False, default=33) ## same-pass re-drops measured 29-37 frames apart
    HELICOPTER_BOMB_MAX_PER_PASS: int = struct.field(pytree_node=False, default=4)
    ## The real picker looks like the ROM's internal random generator (it's
    ## not position, speed or the missile slot, we tested all three). So:
    ## while the heli is in range it gets a chance every RETRY_FRAMES, each
    ## one rarer than the last (chance / (1 + drops so far)), which lands at
    ## roughly: one bomb common, two rarer, three much rarer, four rare.
    HELICOPTER_BOMB_DROP_CHANCE: float = struct.field(pytree_node=False, default=0.5)
    ## Land cadence: the pass is ~204 frames and the real game usually
    ## drops 2 lasers per pass at 30-90 frame spacings; a 75-frame timer
    ## lands on 2 drops per pass (52 was giving 3-4).
    SATELLITE_LASER_DROP_PERIOD: int = struct.field(pytree_node=False, default=75)
    SATELLITE_LASER_FALL_SPEED: int = struct.field(pytree_node=False, default=1) ## laser falls straight down, no sideways drift

    ## Stage progression, measured clean (death freezes subtracted) in ALE:
    ## the land scene runs ~4454 frames before the terrain turns to water;
    ## the first water scene runs ~4435 frames and ends at a dock that pays
    ## a 5000 point bonus; after it comes the second water scene (darker
    ## water, new enemy set) which never ended in a 22000 frame probe, so
    ## it is treated as endless here.
    ## START_STAGE jumps a fresh game straight into a later scene for
    ## playtesting (0 land, 1 first water, 2 second water). Settable
    ## without code changes via the JB_START_STAGE environment variable:
    ##   JB_START_STAGE=1 python scripts/play.py -g jamesbond
    START_STAGE: int = struct.field(pytree_node=False, default=0)
    STAGE_ONE_LENGTH: int = struct.field(pytree_node=False, default=4454)
    STAGE_TWO_LENGTH: int = struct.field(pytree_node=False, default=4435)
    SCORE_STAGE_BONUS: int = struct.field(pytree_node=False, default=5000)

    ## Water scene: scuba diver. Measured in ALE frame by frame (two
    ## agents cross-checked): a vertical swimmer, 7 wide x 20 tall
    ## (scuba_1/2.npy are pixel exact), who enters from the right screen
    ## edge at a fixed depth below the surface, swims left 1px every 4th
    ## frame, never chases, alternates his two animation frames every 15
    ## frames, and vanishes mid-screen on an age clock rather than at the
    ## left edge. He cannot be shot (the player's only round is the
    ## up-forward anti-air shot); running into him under water costs a
    ## life. A boat on the surface floats above his body.
    SCUBA_WIDTH: int = struct.field(pytree_node=False, default=7)
    SCUBA_HEIGHT: int = struct.field(pytree_node=False, default=20)
    SCUBA_SPAWN_X: int = struct.field(pytree_node=False, default=155) ## right screen edge
    SCUBA_SPAWN_Y: int = struct.field(pytree_node=False, default=129) ## body below the surface row (ALE 131, our rows sit 2 higher)
    SCORE_SCUBA: int = struct.field(pytree_node=False, default=200)
    ## The deep swimmer's own clock (both clean tracked episodes of the
    ## 7x20 vertical diver ran 333 frames); the ~120 frame figure floating
    ## around belongs to the surface splash creature, not to him.
    SCUBA_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=333)
    SCUBA_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=5) ## breather between divers
    ## Radioactivity in the first water scene has two independent rules:
    ##   no diver on screen             -> a spent satellite bolt may create
    ##                                     the radioactive surface splash
    ##   diver anywhere on screen       -> the bolt disappears normally
    ##   diver near the player's boat   -> the DIVER becomes radioactive
    ## Proximity is checked every frame; it does not wait for a satellite
    ## bolt to land. Distance is horizontal because the diver remains below
    ## the surface while the boat can jump or dive.
    SCUBA_RADIOACTIVE_RANGE: int = struct.field(pytree_node=False, default=40)
    SCUBA_RADIOACTIVE_FRAMES: int = struct.field(pytree_node=False, default=220) ## how long he stays radioactive
    ## The laser bolt splashes THROUGH the surface: it keeps falling under
    ## water and detonates into a static green surface explosion that
    ## rides the world scroll, blocks the lane for a while, and kills the
    ## boat on near-contact (measured: adjacency within ~1px kills, even
    ## submerged; only a clearly airborne boat passes safely).
    SPLASH_WIDTH: int = struct.field(pytree_node=False, default=20)  ## explosion_1_(small).npy is 20x7
    SPLASH_HEIGHT: int = struct.field(pytree_node=False, default=7)
    SPLASH_Y: int = struct.field(pytree_node=False, default=123) ## straddles the surface row
    SPLASH_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=120)
    SPLASH_SAFE_PLAYER_Y: int = struct.field(pytree_node=False, default=119) ## airborne above this is safe
    WATER_LASER_FLOOR: int = struct.field(pytree_node=False, default=132) ## bolt sinks this deep before detonating
    ## Oil rig, sprite is a static 16x22
    OIL_RIG_WIDTH: int = struct.field(pytree_node=False, default=16)
    OIL_RIG_HEIGHT: int = struct.field(pytree_node=False, default=22)
    OIL_RIG_SEQ_TOTAL: int = struct.field(pytree_node=False, default=60)      ## visible for two seconds at 30 fps; scroll continues while hidden
    OIL_RIG_RIGHT_X: int = struct.field(pytree_node=False, default=144)
    OIL_RIG_Y: int = struct.field(pytree_node=False, default=100)             ## top-left y: deck at waterline, legs in water
    OIL_RIG_MIN_STAGE1_STEPS: int = struct.field(pytree_node=False, default=4000) ## rig can't appear until 4000+ steps into the water scene
    DIAMOND_FLASH_FRAMES: int = struct.field(pytree_node=False, default=60)
    STAGE_TRANSITION_FRAMES: int = struct.field(pytree_node=False, default=30) ## one-second completion flash before water B

    ## Second water scene (after the dock bonus): darker water and a fresh
    ## enemy roster, all sprites cropped from real ALE frames. The floating
    ## rocket is the only scoring object (+200 for ramming it, though the
    ## ram usually costs a life too); after idling on the surface a while
    ## it ignites and launches skyward. The submarine cruises underwater
    ## and only threatens a diving boat; the two flyers cross the sky.
    SCORE_ROCKET: int = struct.field(pytree_node=False, default=200)
    ## The rocket's full launch cycle was measured frame by frame: it
    ## appears mid-screen already submerged (tip row ~140), floats a few
    ## frames, climbs at exactly 1px/frame straight up (a single-frame
    ## splash blip marks the waterline crossing), and EXPLODES with its
    ## tip at row 61 in a brief gray flash, dropping a red bomb that falls
    ## to the waterline and sparkles there. One rocket every 256 frames.
    ROCKET_WIDTH: int = struct.field(pytree_node=False, default=8)
    ROCKET_HEIGHT: int = struct.field(pytree_node=False, default=11)
    ROCKET_Y: int = struct.field(pytree_node=False, default=142) ## rests low in the water; a diving boat can ram it
    ROCKET_SPAWN_X: int = struct.field(pytree_node=False, default=66) ## appears at x = 66
    ROCKET_IGNITE_AGE: int = struct.field(pytree_node=False, default=6) ## floats briefly, then climbs
    ROCKET_EXPLODE_Y: int = struct.field(pytree_node=False, default=61) ## tip row where it bursts
    ROCKET_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=171) ## 256 frame cycle minus ~85 frames of life
    DEBRIS_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=120)
    ROCKET_INITIAL_SPAWN_DELAY: int = struct.field(pytree_node=False, default=300) ## First rocket pass is delayed 300 frames after entering stage 2
    SKY_FLASH_FRAMES: int = struct.field(pytree_node=False, default=30)
    SUBMARINE_WIDTH: int = struct.field(pytree_node=False, default=16)
    SUBMARINE_HEIGHT: int = struct.field(pytree_node=False, default=11)
    SUBMARINE_Y: int = struct.field(pytree_node=False, default=135) ## deep under the surface
    SUBMARINE_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=260)
    SUBMARINE_INITIAL_SPAWN_DELAY: int = struct.field(pytree_node=False, default=150) ## First submarine pass is delayed 150 frames after entering stage 2
    ## The pink ball (fields still called pinkball_*): enters at the LEFT
    ## edge at row 57 and crosses to the right at 1.75 px/f (7 px every 4
    ## frames, read off the longplay at 60 fps). The anti-air shot pops
    ## it for 500; the pop that reaches WB_BALL_HITS_TO_EXIT pays the
    ## 5000 scene bonus and ends the scene (the footage shows three
    ## 500-point hits before the exit: 6000->6500, 6700->7200,
    ## 7600->8100; the team's count was two, so this is one constant to
    ## flip). In the real game a 59-frame freeze with a flickering sky
    ## follows and the daylight scene starts; that scene is out of scope.
    PINKBALL_Y: int = struct.field(pytree_node=False, default=57)
    PINKBALL_WIDTH: int = struct.field(pytree_node=False, default=9)
    PINKBALL_HEIGHT: int = struct.field(pytree_node=False, default=11)
    PINKBALL_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=200)
    SCORE_BALL: int = struct.field(pytree_node=False, default=500)
    WB_BALL_HITS_TO_EXIT: int = struct.field(pytree_node=False, default=3)
    PINKBALL_INITIAL_SPAWN_DELAY: int = struct.field(pytree_node=False, default=200) ## First pinkball pass is delayed 200 frames after entering stage 2
    ## Rocket debris (the wb_flyer slot): born where the rocket bursts,
    ## falls 1px/frame to the waterline, sparkles there red/pink for a
    ## moment, then vanishes. Lethal on contact; the anti-air shot pops it.
    WB_FLYER_Y: int = struct.field(pytree_node=False, default=61)
    WB_FLYER_WIDTH: int = struct.field(pytree_node=False, default=4)
    WB_FLYER_HEIGHT: int = struct.field(pytree_node=False, default=5)
    DEBRIS_REST_Y: int = struct.field(pytree_node=False, default=119)
    DEBRIS_REST_FRAMES: int = struct.field(pytree_node=False, default=40)
    DEBRIS_SPLASH_Y: int = struct.field(pytree_node=False, default=118)
    DEBRIS_SPLASH_FLIP_FRAMES: int = struct.field(pytree_node=False, default=6) ## pattern swap cadence
    SCORE_DEBRIS_SHOT: int = struct.field(pytree_node=False, default=100)
    ## Submarine, water B and C alike (longplay, 60 fps): it enters from
    ## the LEFT edge and cruises right, 2 px every 3 frames. Once per pass,
    ## as it crosses SUB_FIRE_X, it fires a double-dot shot from its bow
    ## that runs diagonally back and up ((-2,-1) px/frame, the "back-
    ## upward" leg) until it reaches SUB_SHOT_LEVEL_Y just under the
    ## surface, then straight left along that row at 4 px every 3 frames
    ## until it leaves the screen. It costs a life on contact: it passes
    ## under a surfaced hull but crosses a diving boat's path.
    SUB_SPAWN_X: int = struct.field(pytree_node=False, default=-12)
    SUB_FIRE_X: int = struct.field(pytree_node=False, default=70)
    SUB_SHOT_LEVEL_Y: int = struct.field(pytree_node=False, default=124)
    SUB_SHOT_WIDTH: int = struct.field(pytree_node=False, default=2)
    SUB_SHOT_HEIGHT: int = struct.field(pytree_node=False, default=8)

    ## The depth charge sinks the submarine (longplay: +200 each time)
    SCORE_SUBMARINE: int = struct.field(pytree_node=False, default=200)
    SCORE_TORPEDO_SHOT: int = struct.field(pytree_node=False, default=100)

    ## Water scene: the satellite stops using the kitchen timer and instead
    ## releases its laser when it passes directly above the player (measured:
    ## the drop column always matched the player column). Aim once, straight
    ## down, no homing -- same as the stage one laser fall.
    ## Nailed with RAM-injection scans: at discrete check moments the
    ## satellite drops iff the drop column is 1..95px to the RIGHT of the
    ## player's hull -- it never fires while still left of the player, and
    ## drops opportunistically any time its belly is ahead of the boat.
    SATELLITE_DROP_AHEAD_MIN: int = struct.field(pytree_node=False, default=1)
    SATELLITE_DROP_AHEAD_MAX: int = struct.field(pytree_node=False, default=95)
    SATELLITE_CHECK_PERIOD: int = struct.field(pytree_node=False, default=30) ## check moments ~10-60f apart in ALE
    SATELLITE_WATER_MAX_DROPS: int = struct.field(pytree_node=False, default=2) ## 1-2 per pass observed
    SATELLITE_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=46) ## measured 44-48 frame gap
    SATELLITE_INITIAL_SPAWN_DELAY: int = struct.field(pytree_node=False, default=180) ## First satellite pass is delayed 180 frames after game start

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=get_default_asset_config)


@struct.dataclass
class JamesBondState:
    """Full internal state with fixed-size object arrays and active masks."""

    player_x: chex.Array
    player_y: chex.Array
    player_vy: chex.Array
    player_vx: chex.Array
    player_jumping: chex.Array
    player_falling: chex.Array
    player_fast_falling: chex.Array
    player_in_air_step: chex.Array
    player_diving: chex.Array
    player_floating: chex.Array
    player_fast_floating: chex.Array
    player_in_water_step: chex.Array
    player_bullet_active: chex.Array
    player_bullet_step: chex.Array
    player_bullet_x: chex.Array
    player_bullet_y: chex.Array
    player_wbullet_active: chex.Array ## Water bullet
    player_wbullet_step: chex.Array
    player_wbullet_x: chex.Array
    player_wbullet_y: chex.Array
    lives: chex.Array
    score: chex.Array
    step_count: chex.Array
    stage: chex.Array
    hit_cooldown: chex.Array
    death_timer: chex.Array ## frames left in the freeze-everything death animation
    sky_flash_timer: chex.Array ## diamond hit or rocket burst, independent of rig visibility
    stage_transition_timer: chex.Array ## freezes the completed scene before the hand-off
    diamond_x: chex.Array
    diamond_y: chex.Array
    diamond_active: chex.Array
    spawn_diamond_next: chex.Array
    pit_x: chex.Array
    pit_y: chex.Array
    pit_active: chex.Array
    helicopter_x: chex.Array
    helicopter_y: chex.Array
    helicopter_active: chex.Array
    helicopter_melee_step: chex.Array
    satellite_x: chex.Array
    satellite_y: chex.Array
    satellite_active: chex.Array
    ## Enemy fire, single objects (the 2600 also only had one missile per object)
    helicopter_bomb_x: chex.Array
    helicopter_bomb_y: chex.Array
    helicopter_bomb_vx: chex.Array ## +-1, aimed at the player once on release
    helicopter_bomb_active: chex.Array
    helicopter_bombs_dropped: chex.Array ## chances used this pass, resets with the heli
    helicopter_bomb_timer: chex.Array ## frames until the next chance
    satellite_laser_x: chex.Array
    satellite_laser_y: chex.Array
    satellite_laser_active: chex.Array
    satellite_laser_timer: chex.Array ## counts down to the next laser drop
    satellite_lasers_dropped: chex.Array ## water scene: lasers used this pass
    satellite_respawn_timer: chex.Array ## gap between satellite passes
    ## Water scene scuba diver, one at a time like every other object
    scuba_x: chex.Array
    scuba_y: chex.Array
    scuba_active: chex.Array
    scuba_respawn_timer: chex.Array ## breather before the next diver enters
    scuba_radioactive: chex.Array ## the diver currently holds the radioactive state
    scuba_radioactive_age: chex.Array ## how long he has been glowing
    scuba_seen: chex.Array ## a diver has appeared this scene: the satellite never splashes again
    ## Laser splash explosion: static in world space, rides the scroll
    splash_x: chex.Array
    splash_active: chex.Array
    splash_age: chex.Array
    ## Oil rig
    oil_rig_x: chex.Array
    oil_rig_y: chex.Array
    oil_rig_active: chex.Array
    oil_rig_visible: chex.Array  ## visibility expires; the hidden rig still scrolls with the seabed
    oil_rig_done: chex.Array     ## mirrors an ongoing attempt; clears when the rig scrolls off-screen
    oil_rig_landing_timer: chex.Array  ## legacy state slot, kept at zero; leaving the screen ends an attempt
    oil_rig_obstacles_remaining: chex.Array  ## legacy state slot, kept at zero; next diamond may retry
    stage1_start_step: chex.Array   ## step_count at the moment the water scene (stage 1) began
    oil_rig_seq: chex.Array      ## visibility countdown; does not control movement or landing lifetime
    diamond_shot: chex.Array     ## True the frame a diamond is shot; triggers the rig next frame
    ## Second water scene roster
    rocket_x: chex.Array
    rocket_y: chex.Array
    rocket_active: chex.Array
    rocket_age: chex.Array ## ignites and launches after idling
    rocket_timer: chex.Array
    submarine_x: chex.Array
    submarine_active: chex.Array
    submarine_timer: chex.Array
    pinkball_x: chex.Array
    pinkball_active: chex.Array
    pinkball_timer: chex.Array
    wb_ball_hits: chex.Array ## pink balls shot this scene (water B exit counter)
    ## Rocket explosion debris (the falling red bomb); timer is its age
    wb_flyer_x: chex.Array
    wb_flyer_y: chex.Array ## falls from the burst row to the waterline
    wb_flyer_active: chex.Array
    wb_flyer_timer: chex.Array
    ## Submarine's double-dot shot
    sub_torp_x: chex.Array
    sub_torp_y: chex.Array
    sub_torp_active: chex.Array
    sub_fired: chex.Array ## this pass already fired its one shot
    key: chex.PRNGKey


@struct.dataclass
class JamesBondObservation:
    """Object-centric observation matching observation_space()."""

    player: ObjectObservation
    diamonds: ObjectObservation
    helicopters: ObjectObservation
    satellites: ObjectObservation
    scubas: ObjectObservation
    ## rocket, submarine, pink ball, red debris (second water scene)
    waterb_enemies: ObjectObservation
    bullets: ObjectObservation
    lives: jnp.ndarray
    score: jnp.ndarray
    stage: jnp.ndarray


@struct.dataclass
class JamesBondInfo:
    """Debug/event info for smoke tests and future gameplay systems."""

    score: jnp.ndarray
    lives: jnp.ndarray
    stage: jnp.ndarray
    step_count: jnp.ndarray


class JaxJamesBond(
    JaxEnvironment[JamesBondState, JamesBondObservation, JamesBondInfo, JamesBondConstants]
):
    """Minimal runnable JamesBond environment following the JAXAtari API."""

    # Compact agent action indices map to these ALE-style actions.
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP, 
            Action.FIRE, 
            Action.UP, 
            Action.RIGHT, 
            Action.LEFT, 
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: JamesBondConstants = None):
        if consts is None:
            ## JB_START_STAGE lets playtesters jump straight into a later
            ## scene through scripts/play.py without touching code
            start_stage = int(os.environ.get("JB_START_STAGE", "0"))
            consts = JamesBondConstants(START_STAGE=min(max(start_stage, 0), 2))
        super().__init__(consts)
        self.renderer = JamesBondRenderer(self.consts)

    def reset(
        self, key: chex.PRNGKey = jax.random.PRNGKey(0)
    ) -> Tuple[JamesBondObservation, JamesBondState]:
        """Create an empty level state with inactive object slots."""

        if key is None:
            key = jax.random.PRNGKey(0)
        state_key, _ = jax.random.split(key)

        state = JamesBondState(
            player_x=jnp.array(self.consts.PLAYER_INIT_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_INIT_Y, dtype=jnp.int32),
            player_vx=jnp.array(0, dtype=jnp.int32),
            player_vy=jnp.array(0, dtype=jnp.int32),
            player_jumping=jnp.array(False, dtype=jnp.bool_),
            player_falling=jnp.array(False, dtype=jnp.bool_),
            player_fast_falling=jnp.array(False, dtype=jnp.bool_),
            player_in_air_step=jnp.array(0, dtype=jnp.int32),
            player_diving=jnp.array(False, dtype=jnp.bool_),
            player_floating=jnp.array(False, dtype=jnp.bool_),
            player_fast_floating=jnp.array(False, dtype=jnp.bool_),
            player_in_water_step=jnp.array(0, dtype=jnp.int32),
            player_bullet_active=jnp.array(False, dtype=jnp.bool_),
            player_bullet_step=jnp.array(-1, dtype=jnp.int32),
            player_bullet_x=jnp.array(-1, dtype=jnp.int32),
            player_bullet_y=jnp.array(-1, dtype=jnp.int32),
            player_wbullet_active=jnp.array(False, dtype=jnp.bool_),
            player_wbullet_step=jnp.array(-1, dtype=jnp.int32),
            player_wbullet_x=jnp.array(-1, dtype=jnp.int32),
            player_wbullet_y=jnp.array(-1, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            stage=jnp.array(self.consts.START_STAGE, dtype=jnp.int32),
            hit_cooldown=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            sky_flash_timer=jnp.array(0, dtype=jnp.int32),
            stage_transition_timer=jnp.array(0, dtype=jnp.int32),
            diamond_x=jnp.array(0, dtype=jnp.int32),
            diamond_y=jnp.array(0, dtype=jnp.int32),
            diamond_active=jnp.array(False, dtype=jnp.bool_),
            pit_x=jnp.array(0, dtype=jnp.int32),
            pit_y=jnp.array(0, dtype=jnp.int32),
            pit_active=jnp.array(False, dtype=jnp.bool_),
            spawn_diamond_next=jnp.array(False, dtype=jnp.bool_),
            helicopter_x=jnp.array(0, dtype=jnp.int32),
            helicopter_y=jnp.array(0, dtype=jnp.int32),
            helicopter_active=jnp.array(False, dtype=jnp.bool_),
            helicopter_melee_step=jnp.array(0, dtype=jnp.int32),
            satellite_x=jnp.array(0, dtype=jnp.int32),
            satellite_y=jnp.array(0, dtype=jnp.int32),
            satellite_active=jnp.array(False, dtype=jnp.bool_),
            helicopter_bomb_x=jnp.array(-1, dtype=jnp.int32),
            helicopter_bomb_y=jnp.array(-1, dtype=jnp.int32),
            helicopter_bomb_vx=jnp.array(0, dtype=jnp.int32),
            helicopter_bomb_active=jnp.array(False, dtype=jnp.bool_),
            helicopter_bombs_dropped=jnp.array(0, dtype=jnp.int32),
            helicopter_bomb_timer=jnp.array(0, dtype=jnp.int32),
            satellite_laser_x=jnp.array(-1, dtype=jnp.int32),
            satellite_laser_y=jnp.array(-1, dtype=jnp.int32),
            satellite_laser_active=jnp.array(False, dtype=jnp.bool_),
            ## Start full so the first laser comes one full period after the satellite shows up
            satellite_laser_timer=jnp.array(
                self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32
            ),
            satellite_lasers_dropped=jnp.array(0, dtype=jnp.int32),
            satellite_respawn_timer=jnp.array(self.consts.SATELLITE_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            scuba_x=jnp.array(-1, dtype=jnp.int32),
            scuba_y=jnp.array(-1, dtype=jnp.int32),
            scuba_active=jnp.array(False, dtype=jnp.bool_),
            scuba_respawn_timer=jnp.array(0, dtype=jnp.int32),
            scuba_radioactive=jnp.array(False, dtype=jnp.bool_),
            scuba_radioactive_age=jnp.array(0, dtype=jnp.int32),
            scuba_seen=jnp.array(False, dtype=jnp.bool_),
            splash_x=jnp.array(-1, dtype=jnp.int32),
            splash_active=jnp.array(False, dtype=jnp.bool_),
            splash_age=jnp.array(0, dtype=jnp.int32),
            oil_rig_x=jnp.array(-1, dtype=jnp.int32),
            oil_rig_y=jnp.array(-1, dtype=jnp.int32),
            oil_rig_active=jnp.array(False, dtype=jnp.bool_),
            oil_rig_visible=jnp.array(False, dtype=jnp.bool_),
            oil_rig_done=jnp.array(False, dtype=jnp.bool_),
            oil_rig_landing_timer=jnp.array(0, dtype=jnp.int32),
            oil_rig_obstacles_remaining=jnp.array(0, dtype=jnp.int32),
            stage1_start_step=jnp.array(0, dtype=jnp.int32),
            oil_rig_seq=jnp.array(0, dtype=jnp.int32),
            diamond_shot=jnp.array(False, dtype=jnp.bool_),
            #oil_rig_visible_timer=jnp.array(0, dtype=jnp.int32),
            rocket_x=jnp.array(-1, dtype=jnp.int32),
            rocket_y=jnp.array(-1, dtype=jnp.int32),
            rocket_active=jnp.array(False, dtype=jnp.bool_),
            rocket_age=jnp.array(0, dtype=jnp.int32),
            rocket_timer=jnp.array(self.consts.ROCKET_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            submarine_x=jnp.array(-1, dtype=jnp.int32),
            submarine_active=jnp.array(False, dtype=jnp.bool_),
            submarine_timer=jnp.array(self.consts.SUBMARINE_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            pinkball_x=jnp.array(-1, dtype=jnp.int32),
            pinkball_active=jnp.array(False, dtype=jnp.bool_),
            pinkball_timer=jnp.array(self.consts.PINKBALL_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            wb_ball_hits=jnp.array(0, dtype=jnp.int32),
            wb_flyer_x=jnp.array(-1, dtype=jnp.int32),
            wb_flyer_y=jnp.array(self.consts.WB_FLYER_Y, dtype=jnp.int32),
            wb_flyer_active=jnp.array(False, dtype=jnp.bool_),
            wb_flyer_timer=jnp.array(0, dtype=jnp.int32),
            sub_torp_x=jnp.array(-1, dtype=jnp.int32),
            sub_torp_y=jnp.array(-1, dtype=jnp.int32),
            sub_torp_active=jnp.array(False, dtype=jnp.bool_),
            sub_fired=jnp.array(False, dtype=jnp.bool_),
            key=state_key,
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: JamesBondState, action: chex.Array
    ) -> Tuple[JamesBondObservation, JamesBondState, chex.Array, chex.Array, JamesBondInfo]:
        """Advance one placeholder frame and return the repo-standard tuple."""

        atari_action = self._decode_action(action)
        previous_state = state

        def live_step(state: JamesBondState) -> JamesBondState:
            state = state.replace(
                ## The frame counter only ticks while the scene is alive:
                ## the real game freezes its clock during the death
                ## animation too, and every render animation and scroll
                ## offset derives from this counter.
                step_count=state.step_count + 1,
                hit_cooldown=jnp.maximum(state.hit_cooldown - 1, 0),
                sky_flash_timer=jnp.maximum(state.sky_flash_timer - 1, 0),
            )
            state = self._update_stage(state)

            def advance_scene(state):
                before_movement = state
                state = self._step_player(state, atari_action)
                state = self._update_objects(state)
                state = self._resolve_oil_rig_landing(before_movement, state)

                def resolve_hazards(state):
                    state = self._update_enemy_bombs(state)
                    return self._resolve_collisions(state)

                ## Reveal on the exact landing frame, before any damage.
                return jax.lax.cond(
                    state.stage_transition_timer > 0, lambda s: s, resolve_hazards, state
                )

            ## A successful landing freezes immediately, before any hazards move.
            return jax.lax.cond(state.stage_transition_timer > 0, lambda s: s, advance_scene, state)

        def transition_step(state: JamesBondState) -> JamesBondState:
            ## Keep the scene still for the completion flash. _update_stage
            ## performs the hand-off on the final tick, clearing the rig.
            state = self._update_stage(state)
            return state.replace(stage_transition_timer=jnp.maximum(state.stage_transition_timer - 1, 0))

        def frozen_step(state: JamesBondState) -> JamesBondState:
            """The death animation: the whole scene stands still while the
            sprite color-cycles; when the timer runs out the player
            respawns and the scene is swept clean, exactly like the
            measured 59-frame freeze in the real game."""

            death_timer = state.death_timer - 1
            respawn = death_timer <= 0

            def sweep(v, park):
                return jnp.where(respawn, jnp.array(park, dtype=v.dtype), v)

            return state.replace(
                death_timer=death_timer,
                player_x=sweep(state.player_x, self.consts.PLAYER_INIT_X),
                player_y=sweep(state.player_y, self.consts.PLAYER_INIT_Y),
                player_jumping=sweep(state.player_jumping, False),
                player_falling=sweep(state.player_falling, False),
                player_fast_falling=sweep(state.player_fast_falling, False),
                player_in_air_step=sweep(state.player_in_air_step, 0),
                player_diving=sweep(state.player_diving, False),
                player_floating=sweep(state.player_floating, False),
                player_fast_floating=sweep(state.player_fast_floating, False),
                player_in_water_step=sweep(state.player_in_water_step, 0),
                ## Every actor and projectile leaves with the fallen agent.
                ## The shots also park their coordinates: their logic does
                ## not run while inactive, so a stale position would
                ## resurrect the old shot on the first post-respawn FIRE.
                player_bullet_active=sweep(state.player_bullet_active, False),
                player_bullet_x=sweep(state.player_bullet_x, -1),
                player_bullet_y=sweep(state.player_bullet_y, -1),
                player_bullet_step=sweep(state.player_bullet_step, -1),
                player_wbullet_active=sweep(state.player_wbullet_active, False),
                player_wbullet_x=sweep(state.player_wbullet_x, -1),
                player_wbullet_y=sweep(state.player_wbullet_y, -1),
                player_wbullet_step=sweep(state.player_wbullet_step, -1),
                helicopter_active=sweep(state.helicopter_active, False),
                helicopter_melee_step=sweep(state.helicopter_melee_step, 0),
                helicopter_bomb_active=sweep(state.helicopter_bomb_active, False),
                satellite_active=sweep(state.satellite_active, False),
                satellite_laser_active=sweep(state.satellite_laser_active, False),
                diamond_active=sweep(state.diamond_active, False),
                diamond_shot=sweep(state.diamond_shot, False),
                ## Hit flashes finish even when a death freezes the scene.
                sky_flash_timer=jnp.maximum(state.sky_flash_timer - 1, 0),
                scuba_active=sweep(state.scuba_active, False),
                splash_active=sweep(state.splash_active, False),
                rocket_active=sweep(state.rocket_active, False),
                submarine_active=sweep(state.submarine_active, False),
                pinkball_active=sweep(state.pinkball_active, False),
                wb_flyer_active=sweep(state.wb_flyer_active, False),
                sub_torp_active=sweep(state.sub_torp_active, False),
                ## A new life gets a fresh shot at the oil rig: clear the
                ## current attempt and its hidden landing target on respawn.
                oil_rig_done=sweep(state.oil_rig_done, False),
                oil_rig_seq=sweep(state.oil_rig_seq, 0),
                oil_rig_landing_timer=sweep(state.oil_rig_landing_timer, 0),
                oil_rig_obstacles_remaining=sweep(state.oil_rig_obstacles_remaining, 0),
                oil_rig_active=sweep(state.oil_rig_active, False),
                oil_rig_visible=sweep(state.oil_rig_visible, False),
                oil_rig_x=sweep(state.oil_rig_x, -1),
                oil_rig_y=sweep(state.oil_rig_y, -1),
                ## The pit resets to its measured post-death position
                pit_x=sweep(state.pit_x, 124),
                hit_cooldown=sweep(state.hit_cooldown, 0),
            )

        state = jax.lax.cond(
            state.death_timer > 0, frozen_step,
            lambda s: jax.lax.cond(s.stage_transition_timer > 0, transition_step, live_step, s),
            state,
        )

        _, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key)

        observation = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._is_done(state)
        info = self._get_info(state)

        return observation, state, reward, done, info

    def render(self, state: JamesBondState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        screen_size = (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)
        return spaces.Dict(
            {
                "player": spaces.get_object_space(n=None, screen_size=screen_size),
                ## Diamond is a single object now like the enemies
                "diamonds": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                "helicopters": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                "satellites": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                ## Water scene scuba diver, single object like the enemies
                "scubas": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                ## rocket, submarine, pink ball, red debris
                "waterb_enemies": spaces.get_object_space(
                    n=4, screen_size=screen_size
                ),
                ## player air bullet, player water bullet, helicopter bomb,
                ## satellite laser, submarine shot
                "bullets": spaces.get_object_space(
                    n=5, screen_size=screen_size
                ),
                "lives": spaces.Box(
                    low=0,
                    high=self.consts.MAX_LIVES,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "score": spaces.Box(
                    low=0,
                    high=1_000_000,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "stage": spaces.Box(
                    low=0,
                    high=self.consts.MAX_EPISODE_STEPS,
                    shape=(),
                    dtype=jnp.int32,
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: JamesBondState) -> JamesBondObservation:
        """Build the structured object observation from internal state."""

        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
            active=jnp.array(True, dtype=jnp.bool_),
            orientation=jnp.array(0.0, dtype=jnp.float32),
            state=jnp.array(0, dtype=jnp.int32),
            visual_id=jnp.array(0, dtype=jnp.int32),
        )
        diamonds = self._object_group_observation(
            state.diamond_x,
            state.diamond_y,
            state.diamond_active,
            self.consts.DIAMOND_WIDTH,
            self.consts.DIAMOND_HEIGHT,
        )
        helicopters = self._object_group_observation(
            state.helicopter_x,
            state.helicopter_y,
            state.helicopter_active,
            self.consts.HELICOPTER_ENEMY_WIDTH,
            self.consts.HELICOPTER_ENEMY_HEIGHT,
        )
        satellites = self._object_group_observation(
            state.satellite_x,
            state.satellite_y,
            state.satellite_active,
            self.consts.SATELLITE_ENEMY_WIDTH,
            self.consts.SATELLITE_ENEMY_HEIGHT,
        )
        scubas = self._object_group_observation(
            state.scuba_x,
            jnp.where(
                state.scuba_radioactive,
                jnp.array(self.consts.SPLASH_Y, dtype=jnp.int32),
                state.scuba_y,
            ),
            state.scuba_active,
            self.consts.SCUBA_WIDTH,
            self.consts.SCUBA_HEIGHT,
        )
        ## Second water scene roster in one fixed-slot group:
        ## rocket, submarine, pink ball, red debris. The shared box is
        ## the largest of the four sprites.
        waterb_enemies = self._object_group_observation(
            jnp.stack([
                state.rocket_x,
                state.submarine_x,
                state.pinkball_x,
                state.wb_flyer_x,
            ]),
            jnp.stack([
                state.rocket_y,
                jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32),
                jnp.array(self.consts.PINKBALL_Y, dtype=jnp.int32),
                state.wb_flyer_y,
            ]),
            jnp.stack([
                state.rocket_active,
                state.submarine_active,
                state.pinkball_active,
                state.wb_flyer_active,
            ]),
            self.consts.SUBMARINE_WIDTH,
            self.consts.SUBMARINE_HEIGHT,
        )
        ## All projectiles in one group (the 1x4 box is close enough for
        ## the submarine's 2x5 shot too): player air bullet, player water
        ## bullet, helicopter bomb, satellite laser, submarine shot
        bullets = self._object_group_observation(
            jnp.stack([
                state.player_bullet_x,
                state.player_wbullet_x,
                state.helicopter_bomb_x,
                state.satellite_laser_x,
                state.sub_torp_x,
            ]),
            jnp.stack([
                state.player_bullet_y,
                state.player_wbullet_y,
                state.helicopter_bomb_y,
                state.satellite_laser_y,
                state.sub_torp_y,
            ]),
            jnp.stack([
                state.player_bullet_active,
                state.player_wbullet_active,
                state.helicopter_bomb_active,
                state.satellite_laser_active,
                state.sub_torp_active,
            ]),
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        return JamesBondObservation(
            player=player,
            diamonds=diamonds,
            helicopters=helicopters,
            satellites=satellites,
            scubas=scubas,
            waterb_enemies=waterb_enemies,
            bullets=bullets,
            lives=state.lives,
            score=state.score,
            stage=state.stage,
        )

    def _object_group_observation(
        self,
        x: chex.Array,
        y: chex.Array,
        active: chex.Array,
        width: int,
        height: int,
        orientation: chex.Array = None,
    ) -> ObjectObservation:
        """Convert fixed-size object arrays plus masks into ObjectObservation."""

        if orientation is None:
            orientation = jnp.zeros_like(x, dtype=jnp.float32)

        ## Inactive objects keep drifting in _update_objects, so
        ## their stale coordinates can leave the screen bounds. Zero them out
        ## and clamp active ones so the observation stays inside its space.
        safe_x = jnp.where(active, jnp.clip(x, 0, self.consts.SCREEN_WIDTH), 0.0)
        safe_y = jnp.where(active, jnp.clip(y, 0, self.consts.SCREEN_HEIGHT), 0.0)

        return ObjectObservation.create(
            x=safe_x,
            y=safe_y,
            width=jnp.full(x.shape, width, dtype=jnp.int32),
            height=jnp.full(y.shape, height, dtype=jnp.int32),
            active=active,
            orientation=orientation,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: JamesBondState) -> JamesBondInfo:
        return JamesBondInfo(
            score=state.score,
            lives=state.lives,
            stage=state.stage,
            step_count=state.step_count,
        )

    def _decode_action(self, action: chex.Array) -> chex.Array:
        """Translate compact action-space indices to JAXAtariAction values."""

        return jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

    def step_player_stage_one( ## Tip: Use air logic function for better memory performance?
        self, args
    ) -> JamesBondState:
        state, atari_action = args
        
        player_x = state.player_x
        player_y = state.player_y
        player_jumping = state.player_jumping
        player_falling = state.player_falling
        player_fast_falling = state.player_fast_falling
        player_in_air_step = state.player_in_air_step

        player_bullet_active = state.player_bullet_active
        player_bullet_step = state.player_bullet_step
        player_bullet_x = state.player_bullet_x
        player_bullet_y = state.player_bullet_y

        up_pressed = jnp.any(
            jnp.array([
                atari_action == Action.UP,
                atari_action == Action.UPRIGHT,
                atari_action == Action.UPLEFT,
                atari_action == Action.UPFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.UPLEFTFIRE,
            ])
        )

        right_pressed = jnp.any(
            jnp.array([
                atari_action == Action.RIGHT,
                atari_action == Action.UPRIGHT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.DOWNRIGHTFIRE
            ])
        )

        left_pressed = jnp.any(
            jnp.array([
                atari_action == Action.LEFT,
                atari_action == Action.UPLEFT,
                atari_action == Action.DOWNLEFT,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.DOWNLEFTFIRE
            ])
        )

        down_pressed = jnp.any(
            jnp.array([
                atari_action == Action.DOWN,
                atari_action == Action.DOWNLEFT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
            ])
        )

        fire_pressed = jnp.any(
            jnp.array([
                atari_action == Action.FIRE,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.UPFIRE,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
            ])
        )


        ###
        ### Player Movement Controller
        ###

        player_x = jnp.where(
            right_pressed, 
            jnp.where(
                state.step_count % 2 == 0,
                jnp.clip(player_x + 1, self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MAX_X), 
                player_x
            ),
            jnp.where(
                left_pressed, 
                jnp.where(
                    state.step_count % 4 == 0, 
                    jnp.clip(player_x - 1, self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MAX_X), 
                    player_x
                ), 
                player_x
            )
        )

        up_pressed = jnp.where(player_jumping, False, up_pressed)
        down_pressed = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, down_pressed)
        
        player_jumping = jnp.where(
            player_jumping,
            player_jumping, 
            jnp.where(
                jnp.logical_and(up_pressed, player_in_air_step < 71), 
                True, 
                False
            )
        )
        
        player_falling = jnp.where(
            jnp.logical_and(
                jnp.logical_or(player_falling, player_in_air_step >= 71), 
                player_y != self.consts.PLAYER_INIT_Y
            ), 
            True, 
            player_falling
        )
        
        player_fast_falling = jnp.where(
            player_fast_falling,
            player_fast_falling,
            jnp.where(
                jnp.logical_and(
                    down_pressed,
                    jnp.logical_or(player_jumping, player_falling)
                ),
                True,
                False
            )
        )

        player_falling = jnp.where(
            player_fast_falling,
            False,
            player_falling,
        )

        player_jumping = jnp.where(
            jnp.logical_or(player_falling, player_fast_falling),
            False,
            player_jumping
        )
        
        player_in_air_step = jnp.where( ## Start immediately falling when reaching the peak of the jump
            player_in_air_step >= 71, 
            63, 
            player_in_air_step
        )
        
        player_y = jnp.where(
            player_fast_falling,
            jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step] + 1, self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), ## Tip: Copy player_int_y_steps for performance?
            jnp.where(
                player_jumping, 
                player_y - self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], 
                jnp.where(
                    player_falling, 
                    jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
                    player_y
                )
            )
        )

        player_falling = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_falling)
        player_fast_falling = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_fast_falling)

        player_in_air_step = jnp.where(
            player_jumping, 
            player_in_air_step + 1, 
            jnp.where(
                player_y == self.consts.PLAYER_INIT_Y,
                0,
                jnp.where(
                    jnp.logical_or(player_falling, player_fast_falling), 
                    player_in_air_step - 1, 
                    0,
                )
            )
        )

        ###
        ### Player Bullet controller
        ###

        fire_pressed = jnp.where(
            jnp.logical_or(player_bullet_active, player_bullet_step >= 30),
            False,
            fire_pressed
        )

        player_bullet_active = jnp.where( ## 1st frame is creation, 31st is deactivation, 30th is the last active
            player_bullet_step < 30, 
            jnp.where(
                player_bullet_active,
                player_bullet_active,
                jnp.where(
                    fire_pressed,
                    True,
                    False
                )
            ),
            False
        )

        player_bullet_x = jnp.where(
            jnp.logical_and(player_bullet_active, player_bullet_x == -1), 
            player_x + self.consts.PLAYER_WIDTH + 2,
            jnp.where(
                player_bullet_active,
                player_bullet_x + 2,
                -1
            )
        )

        player_bullet_y = jnp.where(
            jnp.logical_and(player_bullet_active, player_bullet_y == -1), 
            player_y - 4,
            jnp.where(
                player_bullet_active,
                player_bullet_y - 2,
                -1
            )
        )

        player_bullet_step = jnp.where(
            player_bullet_active,
            player_bullet_step + 1,
            -1
        )

        player_bullet_active = jnp.where(
            player_bullet_step >= 30,
            False,
            player_bullet_active
        )

        return state.replace(
            player_x = player_x,
            player_y = player_y,

            player_jumping = player_jumping,
            player_falling = player_falling,
            player_fast_falling = player_fast_falling,
            player_in_air_step = player_in_air_step,

            player_bullet_active = player_bullet_active,
            player_bullet_step = player_bullet_step,
            player_bullet_x = player_bullet_x,
            player_bullet_y = player_bullet_y,
        )

    def air_movement_logic(
        self, args
    ) -> JamesBondState:
        state, up_pressed, down_pressed = args

        player_y = state.player_y

        player_jumping = state.player_jumping
        player_falling = state.player_falling
        player_fast_falling = state.player_fast_falling
        player_in_air_step = state.player_in_air_step

        up_pressed = jnp.where(player_jumping, False, up_pressed)
        
        player_jumping = jnp.where(
            player_jumping,
            player_jumping, 
            jnp.where(
                jnp.logical_and(
                    jnp.logical_and(up_pressed, player_in_air_step < 71),
                    player_y == self.consts.PLAYER_INIT_Y
                ), 
                True, 
                False
            )
        )
        
        player_falling = jnp.where(
            jnp.logical_and(
                jnp.logical_or(player_falling, player_in_air_step >= 71), 
                player_y != self.consts.PLAYER_INIT_Y
            ), 
            True, 
            player_falling
        )
        
        player_fast_falling = jnp.where(
            player_fast_falling,
            True,
            jnp.where(
                jnp.logical_and(
                    down_pressed,
                    jnp.logical_or(player_jumping, player_falling)
                ),
                True,
                False
            )
        )

        player_falling = jnp.where(
            player_fast_falling,
            False,
            player_falling,
        )

        player_jumping = jnp.where(
            jnp.logical_or(player_falling, player_fast_falling),
            False,
            player_jumping
        )
        
        player_in_air_step = jnp.where( ## Start immediately falling when reaching the peak of the jump
            player_in_air_step >= 71, 
            63, 
            player_in_air_step
        )
        
        player_y = jnp.where(
            player_fast_falling,
            jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step] + 1, self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
            jnp.where(
                player_jumping, 
                player_y - self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], 
                jnp.where(
                    player_falling, 
                    jnp.clip(player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_air_step], self.consts.GAME_AREA_MIN_Y, self.consts.GAME_AREA_MAX_Y), 
                    player_y
                )
            )
        )

        player_falling = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_falling)
        player_fast_falling = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_fast_falling)

        player_in_air_step = jnp.where(
            player_jumping, 
            player_in_air_step + 1, 
            jnp.where(
                player_y == self.consts.PLAYER_INIT_Y,
                0,
                jnp.where(
                    jnp.logical_or(player_falling, player_fast_falling), 
                    player_in_air_step - 1, 
                    0,
                )
            )
        )

        return state.replace(
            player_y = player_y,

            player_jumping = player_jumping,
            player_falling = player_falling,
            player_fast_falling = player_fast_falling,
            player_in_air_step = player_in_air_step,
        )

    def water_movement_logic(
        self, args
    ) -> JamesBondState:
        state, up_pressed, down_pressed = args

        player_y = state.player_y

        player_diving = state.player_diving
        player_floating = state.player_floating
        player_fast_floating = state.player_fast_floating
        player_in_water_step = state.player_in_water_step
        
        down_pressed = jnp.where(player_diving, False, down_pressed)
        
        ## Diving starts on DOWN (the mirror of the air logic, where UP
        ## starts the jump). The old copy used up_pressed here, but the
        ## stage router only sends DOWN presses into the water branch, so
        ## the dive could never begin.
        player_diving = jnp.where(
            player_diving,
            player_diving,
            jnp.where(
                jnp.logical_and(
                    jnp.logical_and(down_pressed, player_in_water_step < 71),
                    player_y == self.consts.PLAYER_INIT_Y
                ),
                True,
                False
            )
        )
        
        player_floating = jnp.where(
            jnp.logical_and(
                jnp.logical_or(player_floating, player_in_water_step >= 71), 
                player_y != self.consts.PLAYER_INIT_Y
            ), 
            True, 
            player_floating
        )
        
        ## UP while under water rushes the boat back to the surface, the
        ## mirror of DOWN fast-falling a jump in the air logic.
        player_fast_floating = jnp.where(
            player_fast_floating,
            True,
            jnp.where(
                jnp.logical_and(
                    up_pressed,
                    jnp.logical_or(player_diving, player_floating)
                ),
                True,
                False
            )
        )

        player_floating = jnp.where(
            player_fast_floating,
            False,
            player_floating,
        )

        player_diving = jnp.where(
            jnp.logical_or(player_floating, player_fast_floating),
            False,
            player_diving
        )
        
        player_in_water_step = jnp.where( ## Start immediately floating up when reaching the bottom of the dive
            player_in_water_step >= 71, 
            63, 
            player_in_water_step
        )

        player_y = jnp.where(
            player_fast_floating,
            jnp.clip(player_y - (self.consts.PLAYER_IN_Y_STEPS[player_in_water_step] + 1), self.consts.GAME_AREA_MAX_Y, 210), ## 210 is arbitrary
            jnp.where(
                player_diving, 
                player_y + self.consts.PLAYER_IN_Y_STEPS[player_in_water_step], 
                jnp.where(
                    player_floating, 
                    jnp.clip(player_y - self.consts.PLAYER_IN_Y_STEPS[player_in_water_step], self.consts.GAME_AREA_MAX_Y, 210), ## 210 is arbitrary
                    player_y
                )
            )
        )

        player_floating = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_floating)
        player_fast_floating = jnp.where(player_y == self.consts.PLAYER_INIT_Y, False, player_fast_floating)

        player_in_water_step = jnp.where(
            player_diving, 
            player_in_water_step + 1, 
            jnp.where(
                player_y == self.consts.PLAYER_INIT_Y,
                0,
                jnp.where(
                    jnp.logical_or(player_floating, player_fast_floating), 
                    player_in_water_step - 1, 
                    0,
                )
            )
        )

        return state.replace(
            player_y = player_y,

            player_diving = player_diving,
            player_floating = player_floating,
            player_fast_floating = player_fast_floating,
            player_in_water_step = player_in_water_step,
        )
    
    def step_player_stage_two(
        self, args
    ) -> JamesBondState:
        state, atari_action = args

        player_x = state.player_x
        player_y = state.player_y

        up_pressed = jnp.any(
            jnp.array([
                atari_action == Action.UP,
                atari_action == Action.UPRIGHT,
                atari_action == Action.UPLEFT,
                atari_action == Action.UPFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.UPLEFTFIRE,
            ])
        )

        right_pressed = jnp.any(
            jnp.array([
                atari_action == Action.RIGHT,
                atari_action == Action.UPRIGHT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.DOWNRIGHTFIRE
            ])
        )

        left_pressed = jnp.any(
            jnp.array([
                atari_action == Action.LEFT,
                atari_action == Action.UPLEFT,
                atari_action == Action.DOWNLEFT,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.DOWNLEFTFIRE
            ])
        )

        down_pressed = jnp.any(
            jnp.array([
                atari_action == Action.DOWN,
                atari_action == Action.DOWNLEFT,
                atari_action == Action.DOWNRIGHT,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
            ])
        )

        fire_pressed = jnp.any(
            jnp.array([
                atari_action == Action.FIRE,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.UPFIRE,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.DOWNLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
            ])
        )

        ###
        ### Player Movement Controller
        ###

        player_x = jnp.where(
            right_pressed, 
            jnp.where(
                state.step_count % 2 == 0,
                jnp.clip(player_x + 1, self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MAX_X), 
                player_x
            ),
            jnp.where(
                left_pressed, 
                jnp.where(
                    state.step_count % 4 == 0, 
                    jnp.clip(player_x - 1, self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MAX_X), 
                    player_x
                ), 
                player_x
            )
        )

        y_function = jnp.where(
            jnp.logical_or(
                state.player_in_air_step > 0,
                jnp.logical_and(up_pressed, player_y == self.consts.PLAYER_INIT_Y),
            ),
            0, ## Air
            jnp.where(
                jnp.logical_or(
                    state.player_in_water_step > 0,
                    jnp.logical_and(down_pressed, player_y == self.consts.PLAYER_INIT_Y),
                ),
                1, ## Water
                2  ## No change
            )
        )

        y_state = jax.lax.switch(
            y_function,
            [
                self.air_movement_logic,
                self.water_movement_logic,
                lambda args: args[0],
            ],
            (state, up_pressed, down_pressed)
        )

        ###
        ### Player Bullet controller
        ###

        def air_bullet_logic(
            state
        ) -> JamesBondState:

            player_bullet_active = state.player_bullet_active
            player_bullet_step = state.player_bullet_step
            player_bullet_x = state.player_bullet_x
            player_bullet_y = state.player_bullet_y
            
            fire_pressed = True
            
            fire_pressed = jnp.where(
                jnp.logical_or(player_bullet_active, player_bullet_step >= 30),
                False,
                fire_pressed
            )

            player_bullet_active = jnp.where( ## 1st frame is creation, 31st is deactivation, 30th is the last active
                player_bullet_step < 30, 
                jnp.where(
                    player_bullet_active,
                    player_bullet_active,
                    jnp.where(
                        fire_pressed,
                        True,
                        False
                    )
                ),
                False
            )

            player_bullet_x = jnp.where(
                jnp.logical_and(player_bullet_active, player_bullet_x == -1), 
                player_x + self.consts.PLAYER_WIDTH + 2,
                jnp.where(
                    player_bullet_active,
                    player_bullet_x + 2,
                    -1
                )
            )

            player_bullet_y = jnp.where(
                jnp.logical_and(player_bullet_active, player_bullet_y == -1), 
                player_y - 4,
                jnp.where(
                    player_bullet_active,
                    player_bullet_y - 2,
                    -1
                )
            )

            player_bullet_step = jnp.where(
                player_bullet_active,
                player_bullet_step + 1,
                -1
            )

            player_bullet_active = jnp.where(
                player_bullet_step >= 30,
                False,
                player_bullet_active
            )

            return state.replace(
                player_bullet_active = player_bullet_active,
                player_bullet_step = player_bullet_step,
                player_bullet_x = player_bullet_x,
                player_bullet_y = player_bullet_y,
            )

        def water_bullet_logic(
            state
        ) -> JamesBondState:
    
            player_wbullet_active = state.player_wbullet_active
            player_wbullet_step = state.player_wbullet_step
            player_wbullet_x = state.player_wbullet_x
            player_wbullet_y = state.player_wbullet_y
            
            fire_pressed = True
            
            fire_pressed = jnp.where(
                jnp.logical_or(player_wbullet_active, player_wbullet_step >= 60),
                False,
                fire_pressed
            )

            player_wbullet_active = jnp.where( ## 1st frame is creation, 61st is deactivation, 60th is the last active
                player_wbullet_step < 60, 
                jnp.where(
                    player_wbullet_active,
                    player_wbullet_active,
                    jnp.where(
                        fire_pressed,
                        True,
                        False
                    )
                ),
                False
            )

            player_wbullet_x = jnp.where(
                jnp.logical_and(player_wbullet_active, player_wbullet_x == -1), 
                player_x + 7,
                jnp.where(
                    player_wbullet_active,
                    jnp.where(player_wbullet_step < 8,
                        player_wbullet_x + self.consts.PLAYER_WATER_BULLET_STEPS[player_wbullet_step][0],
                        jnp.where(
                            player_wbullet_step % 2 == 1,
                            player_wbullet_x + 1,
                            player_wbullet_x
                        )
                    ),
                    -1
                )
            )

            player_wbullet_y = jnp.where(
                jnp.logical_and(player_wbullet_active, player_wbullet_y == -1), 
                player_y - 1,
                jnp.where(
                    player_wbullet_active,
                    jnp.where(player_wbullet_step < 8,
                        player_wbullet_y + self.consts.PLAYER_WATER_BULLET_STEPS[player_wbullet_step][1],
                        jnp.where(
                            player_wbullet_step % 2 == 1,
                            player_wbullet_y + 1,
                            player_wbullet_y
                        )
                    ),
                    -1
                )
            )

            player_wbullet_step = jnp.where(
                player_wbullet_active,
                player_wbullet_step + 1,
                -1
            )

            player_wbullet_active = jnp.where(
                player_wbullet_step >= 60,
                False,
                player_wbullet_active
            )

            return state.replace(
                player_wbullet_active = player_wbullet_active,
                player_wbullet_step = player_wbullet_step,
                player_wbullet_x = player_wbullet_x,
                player_wbullet_y = player_wbullet_y,
            )

        """
        bullet_function = jnp.where( ## Water bullet is always the first one shot
            fire_pressed,
            jnp.where(
                ~state.player_wbullet_active,
                1, ## Only run water_bullet_logic
                2  ## Run both bullet logics
            ),
            jnp.where(
                state.player_wbullet_active,
                jnp.where(
                    state.player_bullet_active,
                    2,
                    1
                ),
                jnp.where(
                    state.player_bullet_active,
                    3, ## Only run air_bullet_logic
                    0
                )
            )
            0 ## Don't run anything
        )
        """

        bullet_function = (state.player_bullet_active.astype(jnp.int32) << 1) | state.player_wbullet_active.astype(jnp.int32)

        bullet_function = jnp.where(
            bullet_function == 0,
            jnp.where(
                fire_pressed,
                1, ## Water bullet is always the first one shot
                0
            ),
            bullet_function
        )

        bullet_function = jnp.where(
            jnp.logical_and(
                fire_pressed,
                jnp.logical_and(
                    state.player_wbullet_step >= 8, ## Tip: Maybe more? ALE is too buggy to check well
                    ~state.player_bullet_active,
                )
            ),
            3,
            bullet_function
        )

        bullet_state = jax.lax.switch(
            bullet_function,
            [
                lambda r: r,
                water_bullet_logic,
                air_bullet_logic,
                lambda r: air_bullet_logic(water_bullet_logic(r))
            ],
            state
        )


        return state.replace(
            player_x = player_x,
            player_y = y_state.player_y,

            player_jumping = y_state.player_jumping,
            player_falling = y_state.player_falling,
            player_fast_falling = y_state.player_fast_falling,
            player_in_air_step = y_state.player_in_air_step,

            player_diving = y_state.player_diving,
            player_floating = y_state.player_floating,
            player_fast_floating = y_state.player_fast_floating,
            player_in_water_step = y_state.player_in_water_step,

            player_wbullet_active = bullet_state.player_wbullet_active,
            player_wbullet_step = bullet_state.player_wbullet_step,
            player_wbullet_x = bullet_state.player_wbullet_x,
            player_wbullet_y = bullet_state.player_wbullet_y,

            player_bullet_active = bullet_state.player_bullet_active,
            player_bullet_step = bullet_state.player_bullet_step,
            player_bullet_x = bullet_state.player_bullet_x,
            player_bullet_y = bullet_state.player_bullet_y,
        )

    def step_player_stage_three_placeholder(
        self, args
    ):
        state, atari_action = args
        
        return state

    def _step_player(
        self, state: JamesBondState, atari_action: chex.Array
    ) -> JamesBondState:

        return jax.lax.switch( ## Stage indexing starts with 0
            state.stage,
            [
                self.step_player_stage_one,
                self.step_player_stage_two,
                ## The second water scene drives the same boat physics
                self.step_player_stage_two,
            ],
            (state, atari_action)
        )

    def _update_stage(self, state: JamesBondState) -> JamesBondState:

        lives_lost = self.consts.MAX_LIVES - state.lives

        new_stage = jnp.where(
            (state.stage == 0) & (state.step_count > 3000 + lives_lost * 1000),
            1,
            state.stage
        )

        ## A downward landing starts the completion flash in
        ## _resolve_oil_rig_landing; hand off only when that flash finishes.
        new_stage = jnp.where(
            (state.stage == 1) & (state.stage_transition_timer == 1),
            2,
            new_stage
        )

        switch = state.stage != new_stage

        def clear(v, park):
            return jnp.where(switch, jnp.array(park, dtype=v.dtype), v)

        ## Remember when the water scene (stage 1) begins, so the oil rig
        ## retains the team's OIL_RIG_MIN_STAGE1_STEPS eligibility delay.
        entering_stage1 = jnp.logical_and(switch, new_stage == 1)
        stage1_start_step = jnp.where(
            entering_stage1, state.step_count, state.stage1_start_step
        )

        entering_stage2 = jnp.logical_and(switch, new_stage == 2)
        next_rocket_timer = jnp.where(
            entering_stage2,
            jnp.array(self.consts.ROCKET_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            state.rocket_timer
        )
        next_submarine_timer = jnp.where(
            entering_stage2,
            jnp.array(self.consts.SUBMARINE_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            state.submarine_timer
        )
        next_pinkball_timer = jnp.where(
            entering_stage2,
            jnp.array(self.consts.PINKBALL_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            state.pinkball_timer
        )
        
        oil_rig_done_reset = jnp.where(entering_stage1, jnp.array(False, dtype=jnp.bool_), state.oil_rig_done)

        ## Transition to end
        new_stage = jnp.where(
            state.wb_ball_hits >= self.consts.WB_BALL_HITS_TO_EXIT,
            3,
            new_stage
        )

        return state.replace(
            stage=new_stage,
            stage1_start_step=stage1_start_step,
            rocket_timer=next_rocket_timer,
            submarine_timer=next_submarine_timer,
            pinkball_timer=next_pinkball_timer,
            oil_rig_done=oil_rig_done_reset,
            stage_transition_timer=clear(state.stage_transition_timer, 0),
            oil_rig_active=clear(state.oil_rig_active, False),
            oil_rig_visible=clear(state.oil_rig_visible, False),
            oil_rig_seq=clear(state.oil_rig_seq, 0),
            oil_rig_landing_timer=clear(state.oil_rig_landing_timer, 0),
            oil_rig_obstacles_remaining=clear(state.oil_rig_obstacles_remaining, 0),
            oil_rig_x=clear(state.oil_rig_x, -1),
            oil_rig_y=clear(state.oil_rig_y, -1),
            sky_flash_timer=clear(state.sky_flash_timer, 0),
            diamond_shot=clear(state.diamond_shot, False),
            player_bullet_active=clear(state.player_bullet_active, False),
            player_bullet_x=clear(state.player_bullet_x, -1),
            player_bullet_y=clear(state.player_bullet_y, -1),
            player_bullet_step=clear(state.player_bullet_step, -1),
            player_wbullet_active=clear(state.player_wbullet_active, False),
            player_wbullet_x=clear(state.player_wbullet_x, -1),
            player_wbullet_y=clear(state.player_wbullet_y, -1),
            player_wbullet_step=clear(state.player_wbullet_step, -1),
            ## Land objects vanish at the shoreline
            helicopter_active=clear(state.helicopter_active, False),
            helicopter_melee_step=clear(state.helicopter_melee_step, 0),
            diamond_active=clear(state.diamond_active, False),
            pit_active=clear(state.pit_active, False),
            helicopter_bomb_active=clear(state.helicopter_bomb_active, False),
            helicopter_bombs_dropped=clear(state.helicopter_bombs_dropped, 0),
            helicopter_bomb_timer=clear(state.helicopter_bomb_timer, 0),
            ## The satellite and its laser reset for the new scene
            satellite_active=clear(state.satellite_active, False),
            satellite_laser_active=clear(state.satellite_laser_active, False),
            satellite_laser_timer=clear(
                state.satellite_laser_timer, self.consts.SATELLITE_LASER_DROP_PERIOD
            ),
            satellite_lasers_dropped=clear(state.satellite_lasers_dropped, 0),
            ## Water objects vanish when the water ends
            scuba_active=clear(state.scuba_active, False),
            scuba_radioactive=clear(state.scuba_radioactive, False),
            scuba_radioactive_age=clear(state.scuba_radioactive_age, 0),
            scuba_respawn_timer=clear(
                state.scuba_respawn_timer, self.consts.SCUBA_RESPAWN_FRAMES
            ),
            splash_active=clear(state.splash_active, False),
            splash_age=clear(state.splash_age, 0),
        )

    def _update_objects(self, state: JamesBondState) -> JamesBondState:
        # Future object lifecycle logic belongs here.

        # === 1. Movement and off-screen cleanup ===

        # Diamonds (Scroll left)
        ## Diamond speed, here is 0.5 pixels per frame
        next_diamond_x = jnp.where(
            state.step_count % 2 == 0,
            state.diamond_x - 1,
            state.diamond_x
        )
        next_diamond_y = state.diamond_y
        diamond_on_screen = next_diamond_x >= (self.consts.GAME_AREA_MIN_X - self.consts.DIAMOND_WIDTH)
        next_diamond_active = state.diamond_active & diamond_on_screen & (state.stage < 2)

        # Scuba (Scroll left)
        ## Measured in ALE: the diver swims left 1px every 4th frame (0.25
        ## px per frame) at a fixed depth, never chases, and vanishes
        ## mid-screen on an age clock -- he does not swim off the left
        ## edge (two clean episodes lasted exactly 333 frames each).
        next_scuba_x = jnp.where(
            state.step_count % 4 == 0,
            state.scuba_x - 1,
            state.scuba_x
        )
        next_scuba_y = state.scuba_y
        ## An existing glow runs on its own clock. Once its fixed lifetime
        ## expires it may be activated again only by player proximity.
        scuba_rad_age = jnp.where(
            state.scuba_radioactive, state.scuba_radioactive_age + 1, 0
        )
        still_glowing = state.scuba_radioactive & (
            scuba_rad_age < self.consts.SCUBA_RADIOACTIVE_FRAMES
        )
        ## The diver becomes radioactive as soon as the boat is horizontally
        ## close enough. This is independent of every satellite projectile.
        ## Measure the visible edge-to-edge gap rather than the two sprites'
        ## left origins, so a diver that looks close really is in range.
        player_right = state.player_x + self.consts.PLAYER_WIDTH
        scuba_right = next_scuba_x + self.consts.SCUBA_WIDTH
        scuba_horizontal_gap = jnp.maximum(
            jnp.maximum(
                next_scuba_x - player_right,
                state.player_x - scuba_right,
            ),
            0,
        )
        scuba_in_range = (
            (state.stage == 1)
            & state.scuba_active
            & (scuba_horizontal_gap <= self.consts.SCUBA_RADIOACTIVE_RANGE)
        )
        newly_radioactive = scuba_in_range & (~state.scuba_radioactive)
        next_scuba_radioactive = (
            (still_glowing | scuba_in_range) & state.scuba_active
        )
        scuba_rad_age = jnp.where(newly_radioactive, 0, scuba_rad_age)

        # Laser splash explosion (rides the world scroll)
        ## Static in world space: it drifts left with the terrain, 1px
        ## every 4th frame, and burns out on its own clock.
        next_splash_x = jnp.where(
            state.step_count % 4 == 0,
            state.splash_x - 1,
            state.splash_x
        )
        splash_age = jnp.where(state.splash_active, state.splash_age + 1, 0)
        next_splash_active = state.splash_active & (
            splash_age < self.consts.SPLASH_LIFETIME_FRAMES
        )

        ## The rig is fixed in world space, like the seabed. A diamond shot
        ## reveals it briefly; after that only its drawing is hidden. The
        ## landing target continues scrolling until it leaves the screen.
        ## The 4000-step gate enables future diamond hits; reaching the gate
        ## alone never spawns a rig or reuses an earlier diamond hit.
        in_water = state.stage == 1
        steps_into_stage1 = state.step_count - state.stage1_start_step
        past_delay = steps_into_stage1 >= self.consts.OIL_RIG_MIN_STAGE1_STEPS
        start_seq = (
            in_water & past_delay & ~state.oil_rig_active
        )
        ## Match _render_water's step_count // 4 offset exactly: one pixel
        ## left every fourth live frame, whether visible or hidden. The
        ## 60-frame reveal timer must never set the rig's horizontal speed.
        scroll_tick = state.step_count % 4 == 0
        scrolled_rig_x = state.oil_rig_x - scroll_tick.astype(jnp.int32)
        ## A missed attempt ends only after the entire rig passes the left
        ## playfield edge; there is no separate hidden-position timeout.
        rig_on_screen = scrolled_rig_x + self.consts.OIL_RIG_WIDTH > self.consts.GAME_AREA_MIN_X
        next_oil_rig_active = in_water & (start_seq | (state.oil_rig_active & rig_on_screen)) & past_delay
        next_oil_rig_x = jnp.where(
            next_oil_rig_active,
            jnp.where(start_seq, self.consts.OIL_RIG_RIGHT_X, scrolled_rig_x),
            -1,
        )
        oil_rig_seq = jnp.where(
            start_seq,
            jnp.array(self.consts.OIL_RIG_SEQ_TOTAL, dtype=jnp.int32),
            jnp.maximum(state.oil_rig_seq - 1, 0),
        )
        oil_rig_seq = jnp.where(next_oil_rig_active, oil_rig_seq, 0)
        next_oil_rig_visible = next_oil_rig_active & (state.sky_flash_timer > 0)
        ## A missed pass needs a fresh diamond hit, with no obstacle-count
        ## delay. Hits during an active pass cannot teleport or restart it.
        oil_rig_done = next_oil_rig_active

        # Second water scene roster (stage 2)
        in_water_b = state.stage == 2
        ## Rocket, measured cycle: floats submerged for a few frames, then
        ## climbs straight up 1px/frame and bursts with its tip at the
        ## measured explosion row. Only the flash remains; no falling bomb.
        rocket_age = jnp.where(state.rocket_active, state.rocket_age + 1, 0)
        rocket_flying = rocket_age >= self.consts.ROCKET_IGNITE_AGE
        next_rocket_x = jnp.where(
            jnp.logical_and(state.rocket_active, jnp.logical_and(rocket_flying, (state.step_count % 5 == 0))),
            state.rocket_x - 1,
            state.rocket_x
        )
        next_rocket_y = jnp.where(
            state.rocket_active & rocket_flying,
            state.rocket_y - 1,
            state.rocket_y
        )
        rocket_explodes = in_water_b & state.rocket_active & (
            next_rocket_y <= self.consts.ROCKET_EXPLODE_Y
        )
        next_rocket_active = state.rocket_active & (~rocket_explodes) & (
            next_rocket_x > self.consts.GAME_AREA_MIN_X - self.consts.ROCKET_WIDTH
        )
        sky_flash_timer = jnp.where(rocket_explodes, self.consts.SKY_FLASH_FRAMES, state.sky_flash_timer)

        player_hit_from_explosion = ((state.stage == 2) & (sky_flash_timer > 0)) & (state.player_y <= self.consts.PLAYER_INIT_Y)

        ## Submarine (longplay): enters from the LEFT and cruises right,
        ## 2px every 3 frames
        sub_dx = jnp.where(state.step_count % 3 != 0, 1, 0)
        next_submarine_x = jnp.where(
            state.submarine_active,
            state.submarine_x + sub_dx,
            state.submarine_x
        )
        next_submarine_active = state.submarine_active & (
            next_submarine_x < self.consts.OBJECT_EXIT_X
        )
        ## Submarine shot (longplay): two stacked dots leave the bow as the
        ## submarine crosses SUB_FIRE_X, run diagonally back and up at
        ## (-2,-1)/frame until they sit just under the surface, then
        ## straight left along that row at 4px every 3 frames until they
        ## leave the screen.
        at_level = state.sub_torp_y <= self.consts.SUB_SHOT_LEVEL_Y
        run_dx = jnp.where(state.step_count % 3 == 0, -2, -1)
        torp_dx = jnp.where(at_level, run_dx, -2)
        torp_dy = jnp.where(at_level, 0, -1)
        next_torp_x = jnp.where(state.sub_torp_active, state.sub_torp_x + torp_dx, -1)
        next_torp_y = jnp.where(
            state.sub_torp_active,
            jnp.maximum(state.sub_torp_y + torp_dy, self.consts.SUB_SHOT_LEVEL_Y),
            -1,
        )
        next_torp_active = state.sub_torp_active & (
            next_torp_x > self.consts.GAME_AREA_MIN_X - self.consts.SUB_SHOT_WIDTH
        )
        crossed_fire_x = jnp.logical_and(
            state.submarine_x < self.consts.SUB_FIRE_X,
            next_submarine_x >= self.consts.SUB_FIRE_X,
        )
        fire_torp = (
            in_water_b & next_submarine_active & (~state.sub_fired)
            & (~next_torp_active) & crossed_fire_x
        )
        next_torp_x = jnp.where(
            fire_torp, next_submarine_x + self.consts.SUBMARINE_WIDTH - self.consts.SUB_SHOT_WIDTH, next_torp_x
        )
        next_torp_y = jnp.where(fire_torp, jnp.array(self.consts.SUBMARINE_Y - 3, dtype=jnp.int32), next_torp_y)
        next_torp_active = next_torp_active | fire_torp
        ## One shot per pass: the flag lifts when the submarine is gone
        next_sub_fired = (state.sub_fired | fire_torp) & next_submarine_active
        ## Pink ball: crosses the sky left to right, 7px every 4 frames
        ## (+2,+2,+2,+1), and leaves at the right edge if nobody pops it
        next_pinkball_x = jnp.where(
            state.pinkball_active,
            state.pinkball_x + jnp.where(state.step_count % 4 == 3, 1, 2),
            state.pinkball_x
        )
        next_pinkball_active = state.pinkball_active & (
            next_pinkball_x < self.consts.OBJECT_EXIT_X
        )
        ## Rocket debris: drifts with the world and falls 1px/frame to the
        ## waterline (its clock is held while falling), sparkles there, then goes.
        debris_age = jnp.where(state.wb_flyer_active, state.wb_flyer_timer + 1, 0)
        next_wb_flyer_x = jnp.where(
            state.wb_flyer_active & scroll_tick,
            state.wb_flyer_x - 1,
            state.wb_flyer_x
        )
        debris_falling = state.wb_flyer_y < self.consts.DEBRIS_REST_Y
        next_wb_flyer_y = jnp.where(
            state.wb_flyer_active & debris_falling,
            state.wb_flyer_y + 1,
            state.wb_flyer_y
        )
        debris_age = jnp.where(debris_falling, 0, debris_age)
        next_wb_flyer_active = in_water_b & state.wb_flyer_active & (
            debris_age < self.consts.DEBRIS_REST_FRAMES
        )
        next_wb_flyer_active = next_wb_flyer_active | rocket_explodes
        next_wb_flyer_x = jnp.where(
            rocket_explodes,
            next_rocket_x + (self.consts.ROCKET_WIDTH - self.consts.WB_FLYER_WIDTH) // 2,
            next_wb_flyer_x,
        )
        next_wb_flyer_y = jnp.where(
            rocket_explodes,
            jnp.array(self.consts.WB_FLYER_Y, dtype=jnp.int32),
            next_wb_flyer_y,
        )
        debris_age = jnp.where(rocket_explodes, 0, debris_age)

        # Enemies
        ## Helicopter enemy (Scroll left)
        ## 1. Determine which speed zone the helicopter is currently in, also affecting melee behavior of helicopter
        in_slow_mode = (state.helicopter_x <= 96) & (state.helicopter_x > 63)
        ## 2. The speed of helicopter is 0.6 pixels per frame in normal mode, and 0.375 pixels per frame in slow mode
        move_normal = (state.step_count % 5 == 0) | (state.step_count % 5 == 2) | (state.step_count % 5 == 4)
        move_slow = (state.step_count % 8 == 1) | (state.step_count % 8 == 4) | (state.step_count % 8 == 7)
        move_helicopter = jnp.where(
            in_slow_mode,
            move_slow,
            move_normal
        )
        next_helicopter_x = jnp.where(
            move_helicopter,
            state.helicopter_x - 1,
            state.helicopter_x
        )
        next_helicopter_y = state.helicopter_y
        helicopter_on_screen = next_helicopter_x >= (self.consts.GAME_AREA_MIN_X - self.consts.HELICOPTER_ENEMY_WIDTH)
        next_helicopter_active = state.helicopter_active & helicopter_on_screen
        ## 3. The melee step of helicopter is only incremented when the helicopter is in slow mode
        ## Update the step counter for the next frame, reset to 0 if the helicopter is not in slow mode or not active
        next_helicopter_melee_step = jnp.where(
            next_helicopter_active & in_slow_mode,
            state.helicopter_melee_step + 1,
            0
        )
        
        ## Satellite enemy (Scroll right)
        ## Measured exactly: +1,+1,+1,+0 repeating = 3px every 4 frames
        next_satellite_x = jnp.where(
            state.step_count % 4 != 3,
            state.satellite_x + 1,
            state.satellite_x
        )
        next_satellite_y = state.satellite_y
        ## The satellite crosses the whole screen (measured pass ~209
        ## frames, entry at the left edge, exit past the right edge)
        satellite_on_screen = next_satellite_x <= (self.consts.OBJECT_EXIT_X)
        next_satellite_active = state.satellite_active & satellite_on_screen

        # Fire pit (Scroll left)
        ## Fire pit speed, here is 0.25 pixels per frame, as far as i checked, fire pit will move if step count % 4 == 3
        next_pit_x = jnp.where(
            state.step_count % 4 == 3,
            state.pit_x -1,
            state.pit_x
        )
        next_pit_y = state.pit_y
        pit_on_screen = next_pit_x >= (self.consts.GAME_AREA_MIN_X - self.consts.PIT_WIDTH)
        next_pit_active = state.pit_active & pit_on_screen

        # === 2. Spawning logic ===
        ## Stage 1 (water) delay timing logic for scuba and oil rig
        ## Rule: Alternative spawning only when the entire row is empty
        on_land = state.stage == 0
        in_water = state.stage == 1
        row_57_empty = (~jnp.any(next_helicopter_active)) & (~jnp.any(next_diamond_active))
        # Check whose turn it is to spawn. The pit is land-only; the
        # diamond floats through the sky of every scene (measured at
        # y60-64 on land and y62 over the water), and the red helicopter
        # also patrols the first water scene, dropping the same bombs
        # (measured: enters right, ~0.58 px/f slowing mid-screen, ~2 bombs
        # per crossing). Only the second water scene retires it.
        spawn_diamond = row_57_empty & state.spawn_diamond_next & (~in_water_b)
        ## The helicopter spawning will be delayed by 75 frames from initial state
        helicopter_delay_passed = state.step_count >= 75
        spawn_helicopter = row_57_empty & (~state.spawn_diamond_next) & (~in_water_b) & (~state.oil_rig_active) & (helicopter_delay_passed)
        ## The satellite takes a measured ~65 frame breather between passes;
        ## the gap also lets its per-pass laser counter reset.
        satellite_respawn_timer = jnp.where(
            next_satellite_active,
            jnp.array(self.consts.SATELLITE_RESPAWN_FRAMES, dtype=jnp.int32),
            jnp.maximum(state.satellite_respawn_timer - 1, 0),
        )
        ## The satellite patrols the land and the first water scene; the
        ## second water scene never shows it (22k frame scan).
        ## The first satellite is delayed by 180 frames
        satellite_delay_passed = state.step_count > self.consts.SATELLITE_INITIAL_SPAWN_DELAY
        can_spawn_satellite = (
            (~next_satellite_active) & (satellite_respawn_timer == 0) & (satellite_delay_passed) & (state.stage <= 1) & (~state.oil_rig_active) & (~next_oil_rig_active)
        )
        can_spawn_pit = (~next_pit_active) & on_land ## Only spawn when the previous pit left the screen

        ## Second water scene spawners: each object enters on its own
        ## staggered breather so the roster stays mixed.
        def waterb_spawner(active, timer, gap):
            next_timer = jnp.where(
                active | (~in_water_b),
                jnp.array(gap, dtype=jnp.int32),
                jnp.maximum(timer - 1, 0),
            )
            spawn = in_water_b & (~active) & (next_timer == 0)
            return spawn, next_timer

        ## The rocket appears mid-screen already submerged (measured x~66),
        ## on its 256-frame cycle: ~85 frames of life + this breather.
        spawn_rocket, rocket_timer = waterb_spawner(
            next_rocket_active, state.rocket_timer, self.consts.ROCKET_RESPAWN_FRAMES)
        next_rocket_active = next_rocket_active | spawn_rocket
        next_rocket_x = jnp.where(
            spawn_rocket,
            jnp.maximum(self.consts.ROCKET_SPAWN_X, state.player_x + 30),
            next_rocket_x
        )
        next_rocket_y = jnp.where(spawn_rocket, self.consts.ROCKET_Y, next_rocket_y)
        rocket_age = jnp.where(spawn_rocket, 0, rocket_age)

        ## The submarine enters from the left (longplay)
        spawn_submarine, submarine_timer = waterb_spawner(
            next_submarine_active, state.submarine_timer, self.consts.SUBMARINE_RESPAWN_FRAMES)
        next_submarine_active = next_submarine_active | spawn_submarine
        next_submarine_x = jnp.where(spawn_submarine, self.consts.SUB_SPAWN_X, next_submarine_x)
        next_sub_fired = next_sub_fired & (~spawn_submarine)

        ## The pink ball enters at the LEFT edge (longplay)
        spawn_pinkball, pinkball_timer = waterb_spawner(
            next_pinkball_active, state.pinkball_timer, self.consts.PINKBALL_RESPAWN_FRAMES)
        next_pinkball_active = next_pinkball_active | spawn_pinkball
        next_pinkball_x = jnp.where(
            spawn_pinkball,
            self.consts.GAME_AREA_MIN_X - self.consts.PINKBALL_WIDTH,
            next_pinkball_x,
        )

        ## Scuba diver: water only, one at a time, entering from the right
        ## edge with a breather between divers.
        scuba_respawn_timer = jnp.where(
            state.scuba_active | on_land,
            jnp.array(self.consts.SCUBA_RESPAWN_FRAMES, dtype=jnp.int32),
            jnp.maximum(state.scuba_respawn_timer - 1, 0),
        )
        scuba_delay_passed = jnp.logical_and(
            in_water,
            steps_into_stage1 >= 1500
        )
        spawn_scuba = in_water & (~state.scuba_active) & (scuba_respawn_timer == 0) & (scuba_delay_passed) & (~next_oil_rig_active)
        scuba_on_screen = next_scuba_x >= (self.consts.GAME_AREA_MIN_X - self.consts.SCUBA_WIDTH)
        next_scuba_active = ((state.scuba_active & scuba_on_screen) | spawn_scuba) & (state.scuba_radioactive_age != self.consts.SCUBA_RADIOACTIVE_FRAMES)
        next_scuba_x = jnp.where(
            spawn_scuba,
            jnp.array(self.consts.SCUBA_SPAWN_X, dtype=jnp.int32),
            next_scuba_x
        )
        next_scuba_y = jnp.where(
            spawn_scuba,
            jnp.array(self.consts.SCUBA_SPAWN_Y, dtype=jnp.int32),
            next_scuba_y
        )
        ## Once the first diver has shown up, radioactivity belongs to the
        ## divers for the rest of the scene (team rule: only one thing is
        ## ever radioactive, and the satellite must not splash after the
        ## divers start spawning)
        next_scuba_seen = state.scuba_seen | spawn_scuba
        # Check if either helicopter or diamond is spawning on this frame
        spawn_occurred = jnp.logical_or(spawn_diamond, spawn_helicopter)
        # Flip the turn flag ONLY if a spawn is happening on this frame
        next_spawn_diamond_next = jnp.where(
            spawn_occurred,
            ~state.spawn_diamond_next, ## Swap to the other object for next time
            state.spawn_diamond_next ## Keep it the same while they are flying
        )
        # Diamonds
        # Apply new active status, position coordinates for spawned diamonds
        next_diamond_active = next_diamond_active | spawn_diamond
        next_diamond_x = jnp.where(
            spawn_diamond,
            self.consts.OBJECT_SPAWN_X_FAR, ## enters at the right screen edge like in ALE
            next_diamond_x
        )
        next_diamond_y = jnp.where(
            spawn_diamond,
            ## Measured heights: gem core rows y60-64 on land, y62 over water
            jnp.where(on_land, 57, 62),
            next_diamond_y
        )
        ## The rig stays at the waterline throughout its visible and hidden pass.
        next_oil_rig_y = jnp.where(
            next_oil_rig_active,
            self.consts.OIL_RIG_Y,
            -1,
        )
        # Enemies
        ## Helicopter
        # Apply new active status, position coordinates for spawned helicopter enemies
        next_helicopter_active = next_helicopter_active | spawn_helicopter
        next_helicopter_x = jnp.where(
            spawn_helicopter,
            self.consts.OBJECT_SPAWN_X, ## measured entry around column 150
            next_helicopter_x
        )
        next_helicopter_y = jnp.where(
            spawn_helicopter,
            57,
            next_helicopter_y
        )
        ## Satellite
        # Apply new active status, position coordinates for spawned satellite enemies
        next_satellite_active = next_satellite_active | can_spawn_satellite
        next_satellite_x = jnp.where(
            can_spawn_satellite,
            self.consts.GAME_AREA_MIN_X - self.consts.SATELLITE_ENEMY_WIDTH,
            next_satellite_x
        )
        next_satellite_y = jnp.where(
            can_spawn_satellite,
            75,
            next_satellite_y
        )
        ## Fire pit
        ## The pit is a single scalar object (see reset and _render_pit), so
        ## spawn with plain jnp.where instead of array indexing.
        next_pit_active = next_pit_active | can_spawn_pit
        next_pit_x = jnp.where(
            can_spawn_pit,
            self.consts.OBJECT_SPAWN_X_FAR, ## next pit enters right as the last leaves: ~160px world spacing
            next_pit_x
        )
        next_pit_y = jnp.where(
            can_spawn_pit,
            119, ## sprite row 0 = two rows above the crater rim; crater top lands just over the road
            next_pit_y
        )

        ## Clear the existing roster when the rig arrives, as well as
        ## suppressing new spawns during its visible and hidden windows.
        next_helicopter_active = next_helicopter_active & ~start_seq
        next_satellite_active = next_satellite_active & ~start_seq
        next_scuba_active = next_scuba_active & ~start_seq
        next_splash_active = next_splash_active & ~start_seq
        next_helicopter_melee_step = jnp.where(start_seq, 0, next_helicopter_melee_step)

        return state.replace(
            diamond_x=next_diamond_x,
            ## Reaching burst height means the player missed the rocket.
            ## Charge one life on that event, if player is not underwater.
            ## and cooldown so neither the flash nor a simultaneous hit can
            ## charge another life. Shooting it earlier prevents the burst.
            lives=jnp.maximum(state.lives - player_hit_from_explosion.astype(jnp.int32), 0),
            hit_cooldown=jnp.where(player_hit_from_explosion, self.consts.HIT_COOLDOWN_STEPS, state.hit_cooldown),
            death_timer=jnp.where(player_hit_from_explosion, self.consts.DEATH_ANIMATION_FRAMES, state.death_timer),
            diamond_y=next_diamond_y,
            diamond_active=next_diamond_active,
            helicopter_x=next_helicopter_x,
            helicopter_y=next_helicopter_y,
            helicopter_active=next_helicopter_active,
            helicopter_melee_step=next_helicopter_melee_step,
            helicopter_bomb_active=state.helicopter_bomb_active & ~start_seq,
            satellite_x=next_satellite_x,
            satellite_y=next_satellite_y,
            satellite_active=next_satellite_active,
            satellite_laser_active=state.satellite_laser_active & ~start_seq,
            satellite_respawn_timer=satellite_respawn_timer,
            spawn_diamond_next=next_spawn_diamond_next,
            pit_x=next_pit_x,
            pit_y=next_pit_y,
            pit_active=next_pit_active,
            scuba_x=next_scuba_x,
            scuba_y=next_scuba_y,
            scuba_active=next_scuba_active,
            scuba_radioactive=next_scuba_radioactive,
            scuba_radioactive_age=scuba_rad_age,
            scuba_respawn_timer=scuba_respawn_timer,
            scuba_seen=next_scuba_seen,
            splash_x=next_splash_x,
            splash_active=next_splash_active,
            splash_age=splash_age,
            oil_rig_x=next_oil_rig_x,
            oil_rig_y=next_oil_rig_y,
            oil_rig_active=next_oil_rig_active,
            oil_rig_visible=next_oil_rig_visible,
            oil_rig_done=oil_rig_done,
            oil_rig_landing_timer=jnp.zeros_like(state.oil_rig_landing_timer),
            oil_rig_obstacles_remaining=jnp.zeros_like(state.oil_rig_obstacles_remaining),
            oil_rig_seq=oil_rig_seq,
            stage1_start_step=state.stage1_start_step,
            rocket_x=next_rocket_x,
            rocket_y=next_rocket_y,
            rocket_active=next_rocket_active,
            rocket_age=rocket_age,
            rocket_timer=rocket_timer,
            sky_flash_timer=sky_flash_timer,
            submarine_x=next_submarine_x,
            submarine_active=next_submarine_active,
            submarine_timer=submarine_timer,
            pinkball_x=next_pinkball_x,
            pinkball_active=next_pinkball_active,
            pinkball_timer=pinkball_timer,
            wb_flyer_x=next_wb_flyer_x,
            wb_flyer_y=next_wb_flyer_y.astype(jnp.int32),
            wb_flyer_active=next_wb_flyer_active,
            wb_flyer_timer=debris_age,
            sub_torp_x=next_torp_x,
            sub_torp_y=next_torp_y,
            sub_torp_active=next_torp_active,
            sub_fired=next_sub_fired,
        )

    def _update_enemy_bombs(self, state: JamesBondState) -> JamesBondState:
        """Move and spawn the helicopter bomb and the satellite laser."""

        ## The old version still indexed helicopter_x like an array and crashed
        ## every step after the single object change. Rewritten for scalars.
        ground = self.consts.GAME_AREA_MAX_Y

        ## 1. Move whatever is flying, remove it once it reaches the ground
        ## (checked in the real game: they just disappear there, no explosion)
        heli_bomb_x = jnp.where(
            state.helicopter_bomb_active,
            state.helicopter_bomb_x + state.helicopter_bomb_vx,
            -1,
        )
        heli_bomb_y = jnp.where(
            state.helicopter_bomb_active,
            state.helicopter_bomb_y + self.consts.HELICOPTER_BOMB_VY,
            -1,
        )
        heli_bomb_active = jnp.logical_and(
            state.helicopter_bomb_active, heli_bomb_y < ground
        )

        laser_x = jnp.where(state.satellite_laser_active, state.satellite_laser_x, -1)
        laser_y = jnp.where(
            state.satellite_laser_active,
            state.satellite_laser_y + self.consts.SATELLITE_LASER_FALL_SPEED,
            -1,
        )
        ## On land the bolt vanishes at the ground line. In the water it
        ## keeps sinking below the surface and only detonates deeper down
        ## (measured: bolt visible under water to ~y134-137 before the
        ## explosion appears at the surface).
        laser_floor = jnp.where(
            state.stage == 1,
            jnp.array(self.consts.WATER_LASER_FLOOR, dtype=jnp.int32),
            jnp.array(ground, dtype=jnp.int32),
        )
        laser_active = jnp.logical_and(state.satellite_laser_active, laser_y < laser_floor)

        ## Splash detonation, measured frame-exact in ALE: the spent bolt
        ## becomes the green frogman at the impact column (narrow pose the
        ## first frame), riding the world scroll for exactly 120 frames.
        ## A scuba diver anywhere on screen suppresses this conversion, and
        ## so does the memory of one: once the first diver has appeared in
        ## the scene the bolt never splashes again, it simply disappears at
        ## its normal water floor. Only one thing is radioactive at a time.
        detonate = jnp.logical_and(
            jnp.logical_and(state.satellite_laser_active, laser_y >= laser_floor),
            state.stage == 1,
        )
        divers_own_it = jnp.logical_or(state.scuba_active, state.scuba_seen)
        spawn_splash = jnp.logical_and(
            detonate,
            jnp.logical_not(jnp.logical_or(divers_own_it, state.splash_active)),
        )
        refresh_splash = jnp.logical_and(
            jnp.logical_and(detonate, state.splash_active),
            jnp.logical_not(divers_own_it),
        )
        splash_x = jnp.where(
            spawn_splash,
            laser_x, ## the frogman surfaces at [laser_x, laser_x+19], not centered
            state.splash_x,
        )
        ## If a diver enters while an older radioactive splash is alive, the
        ## diver rule wins immediately; both radioactive forms never coexist.
        splash_active = jnp.logical_and(
            jnp.logical_or(state.splash_active, spawn_splash),
            jnp.logical_not(state.scuba_active),
        )
        splash_age = jnp.where(
            state.scuba_active,
            0,
            jnp.where(
                jnp.logical_or(spawn_splash, refresh_splash),
                0,
                state.splash_age,
            ),
        )

        ## 2. Helicopter drop. Measured against the ROM: the trigger is the
        ## distance to the player, not the searchlight. First bomb when the
        ## heli closes to RANGE_FAR, one more at RANGE_NEAR. The near one can
        ## happen while the heli is basically on top of the player or already
        ## past, which is why bombs also fall right diagonal in the real game.
        distance = state.helicopter_x - state.player_x
        in_range = jnp.logical_and(
            state.helicopter_active,
            distance <= self.consts.HELICOPTER_BOMB_RANGE,
        )
        ## The timer sits at 0 outside the range, so entering it gives the
        ## first chance right away, then one more every RETRY_FRAMES. Each
        ## chance is consumed whether or not the coin flip succeeds, and
        ## every next chance is rarer than the one before.
        bomb_timer = jnp.where(
            in_range, jnp.maximum(state.helicopter_bomb_timer - 1, 0), 0
        )
        chance = jnp.logical_and(
            jnp.logical_and(in_range, bomb_timer == 0),
            state.helicopter_bombs_dropped < self.consts.HELICOPTER_BOMB_MAX_PER_PASS,
        )
        roll = jax.random.uniform(
            jax.random.fold_in(state.key, state.helicopter_bombs_dropped)
        )
        lucky = roll < self.consts.HELICOPTER_BOMB_DROP_CHANCE / (
            1 + state.helicopter_bombs_dropped
        )
        ## The bomb and the satellite laser share one hardware sprite slot
        ## in the real game: they were never airborne together in 3k+
        ## measured frames, naturally or forced. Enforce the exclusivity.
        drop_bomb = jnp.logical_and(
            jnp.logical_and(chance, lucky),
            jnp.logical_not(
                jnp.logical_or(heli_bomb_active, state.satellite_laser_active)
            ),
        )
        bomb_timer = jnp.where(
            chance,
            jnp.array(self.consts.HELICOPTER_BOMB_RETRY_FRAMES, dtype=jnp.int32),
            bomb_timer,
        )
        ## Count used chances (not drops), forget once the heli is gone
        bombs_dropped = jnp.where(
            state.helicopter_active,
            state.helicopter_bombs_dropped + chance.astype(jnp.int32),
            0,
        )
        heli_bomb_x = jnp.where(
            drop_bomb,
            state.helicopter_x + self.consts.HELICOPTER_ENEMY_WIDTH // 2,
            heli_bomb_x,
        )
        heli_bomb_y = jnp.where(
            drop_bomb,
            state.helicopter_y + self.consts.HELICOPTER_ENEMY_HEIGHT,
            heli_bomb_y,
        )
        ## Direction picked once on release, but NOT aimed at the player:
        ## measured across every recorded drop, the bomb trails the flying
        ## helicopter 1px left per frame, and falls straight down only when
        ## the helicopter releases near the left edge. A rightward bomb was
        ## never observed, player position notwithstanding.
        release_vx = jnp.where(
            state.helicopter_x > 31,
            -self.consts.HELICOPTER_BOMB_SPEED_X,
            0,
        ).astype(jnp.int32)
        heli_bomb_vx = jnp.where(drop_bomb, release_vx, state.helicopter_bomb_vx)
        heli_bomb_active = jnp.logical_or(heli_bomb_active, drop_bomb)

        ## 3. Satellite laser. Two triggers, one per scene, both measured:
        ## - Land: a simple kitchen timer. Counts down while a satellite is
        ##   on screen, drops from its belly at zero, rewinds. Parked at
        ##   full while no satellite is around, so every new pass starts a
        ##   fresh countdown.
        ## - Water: the timer is ignored. The satellite releases the laser
        ##   the moment its belly passes over the player's column (the drop
        ##   column always matched the player column in ALE), a couple of
        ##   times per pass at most.
        laser_timer = jnp.where(
            state.satellite_active,
            jnp.maximum(state.satellite_laser_timer - 1, 0),
            jnp.array(self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32),
        )
        timer_drop = jnp.logical_and(state.satellite_active, laser_timer == 0)

        ## Water rule, nailed with RAM-injection scans in ALE: at discrete
        ## check moments the satellite fires iff the drop column sits 1..95
        ## px AHEAD (right) of the player's hull -- never while still
        ## behind it. With a parked player this looks like "drops when
        ## overhead", but the window is wide open to the right.
        sat_belly = state.satellite_x + self.consts.SATELLITE_ENEMY_WIDTH // 2
        ahead = sat_belly - state.player_x
        in_drop_window = jnp.logical_and(
            ahead >= self.consts.SATELLITE_DROP_AHEAD_MIN,
            ahead <= self.consts.SATELLITE_DROP_AHEAD_MAX,
        )
        check_moment = (state.step_count % self.consts.SATELLITE_CHECK_PERIOD) == 0
        window_drop = jnp.logical_and(
            jnp.logical_and(state.satellite_active, check_moment),
            jnp.logical_and(
                in_drop_window,
                state.satellite_lasers_dropped < self.consts.SATELLITE_WATER_MAX_DROPS,
            ),
        )

        in_water = state.stage == 1
        drop_laser = jnp.logical_and(
            jnp.where(in_water, window_drop, timer_drop),
            ## One laser at a time, and never while the helicopter bomb is
            ## airborne -- they share the single projectile slot.
            jnp.logical_not(jnp.logical_or(laser_active, heli_bomb_active)),
        )
        laser_x = jnp.where(drop_laser, sat_belly, laser_x)
        laser_y = jnp.where(
            drop_laser,
            state.satellite_y + self.consts.SATELLITE_ENEMY_HEIGHT,
            laser_y,
        )
        laser_active = jnp.logical_or(laser_active, drop_laser)
        laser_timer = jnp.where(
            drop_laser,
            jnp.array(self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32),
            laser_timer,
        )
        ## Count drops per pass, forget once the satellite is gone
        lasers_dropped = jnp.where(
            state.satellite_active,
            state.satellite_lasers_dropped + drop_laser.astype(jnp.int32),
            0,
        )

        return state.replace(
            satellite_lasers_dropped=lasers_dropped,
            helicopter_bomb_x=heli_bomb_x.astype(jnp.int32),
            helicopter_bomb_y=heli_bomb_y.astype(jnp.int32),
            helicopter_bomb_vx=heli_bomb_vx,
            helicopter_bomb_active=heli_bomb_active,
            helicopter_bombs_dropped=bombs_dropped,
            helicopter_bomb_timer=bomb_timer,
            satellite_laser_x=laser_x.astype(jnp.int32),
            satellite_laser_y=laser_y.astype(jnp.int32),
            satellite_laser_active=laser_active,
            satellite_laser_timer=laser_timer,
            splash_x=splash_x.astype(jnp.int32),
            splash_active=splash_active,
            splash_age=splash_age,
        )

    def _is_done(self, state: JamesBondState) -> chex.Array:
        return self._get_done(state)

    def _resolve_collisions(self, state: JamesBondState) -> JamesBondState:
        """Run all collision systems after movement and object updates."""

        stage = state.stage

        new_state = lax.switch(
            stage,
            (
                self._resolve_stage_one_collisions,
                self._resolve_stage_two_collisions,
                self._resolve_stage_three_collisions,
            ),
            state,
        )

        return new_state

    def _resolve_stage_one_collisions(self, state: JamesBondState) -> JamesBondState:
        new_state = self._resolve_bullet_diamond_collisions(state)
        new_state = self._resolve_pit_player_collisions(new_state)
        new_state = self._resolve_bullet_player_collisions(new_state)

        return new_state

    def _resolve_stage_two_collisions(self, state: JamesBondState) -> JamesBondState:
        new_state = self._resolve_bullet_diamond_collisions(state)
        new_state = self._resolve_wbullet_scuba_collisions(new_state)
        new_state = self._resolve_bullet_oil_rig_collisions(new_state)
        new_state = self._resolve_bullet_player_collisions(new_state)
        new_state = self._resolve_splash_player_collisions(new_state)
        new_state = self._resolve_oil_rig_collision(new_state)

        return new_state

    def _resolve_stage_three_collisions(self, state: JamesBondState) -> JamesBondState:
        new_state = self._resolve_waterb_ball_shot(state)
        new_state = self._resolve_water_shots(new_state)
        new_state = self._resolve_waterb_collisions(new_state)
        new_state = self._resolve_debris_contacts(new_state)

        return new_state


    def _resolve_debris_contacts(self, state: JamesBondState) -> JamesBondState:
        """Water B: the falling / sparkling red rocket debris costs a life."""

        debris_hit = jnp.logical_and(
            jnp.logical_and(state.stage == 2, state.wb_flyer_active),
            _aabb_overlap(
                state.player_x, state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH, self.consts.PLAYER_COLLISION_HEIGHT,
                state.wb_flyer_x - 1, state.wb_flyer_y,
                self.consts.WB_FLYER_WIDTH + 2, self.consts.WB_FLYER_HEIGHT,
            ),
        )
        took_damage = jnp.logical_and(debris_hit, state.hit_cooldown <= 0)

        return state.replace(
            lives=jnp.maximum(0, state.lives - took_damage.astype(jnp.int32)).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def _resolve_waterb_ball_shot(self, state: JamesBondState) -> JamesBondState:
        """Water B: the anti-air shot pops the pink ball for 500.

        The pop that completes WB_BALL_HITS_TO_EXIT also pays the scene
        bonus and ends the scene (stage 3 is the end marker, see
        _get_done; the scene after it is out of scope).
        """

        ball_hit = jnp.logical_and(
            state.stage == 2,
            jnp.logical_and(
                jnp.logical_and(state.player_bullet_active, state.pinkball_active),
                _aabb_overlap(
                    state.player_bullet_x, state.player_bullet_y,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.pinkball_x, jnp.array(self.consts.PINKBALL_Y, dtype=jnp.int32),
                    self.consts.PINKBALL_WIDTH, self.consts.PINKBALL_HEIGHT,
                ),
            ),
        )
        hits = state.wb_ball_hits + ball_hit.astype(jnp.int32)
        final_hit = jnp.logical_and(ball_hit, hits >= self.consts.WB_BALL_HITS_TO_EXIT)
        bullet_keep = state.player_bullet_active & (~ball_hit)
        gained = (
            ball_hit.astype(jnp.int32) * self.consts.SCORE_BALL
            + final_hit.astype(jnp.int32) * self.consts.SCORE_STAGE_BONUS
        )

        def park(keep, v):
            return jnp.where(keep, v, jnp.array(-1, dtype=v.dtype))

        return state.replace(
            score=(state.score + gained).astype(jnp.int32),
            wb_ball_hits=hits,
            pinkball_active=state.pinkball_active & (~ball_hit),
            player_bullet_active=bullet_keep,
            player_bullet_step=park(bullet_keep, state.player_bullet_step),
            player_bullet_x=park(bullet_keep, state.player_bullet_x),
            player_bullet_y=park(bullet_keep, state.player_bullet_y),
        )

    def _resolve_water_shots(self, state: JamesBondState) -> JamesBondState:
        """Water B: the player's rounds hit the rocket, its debris and the
        submarine.

        Read off the longplay: the anti-air shot destroys a climbing
        rocket and the depth charge a submerged or surfacing one (+200,
        6500 -> 6700 the moment the shot touched it); the depth charge
        sinks the submarine for 200; the anti-air shot pops the falling
        debris for 100. The used round is consumed on impact.
        """

        in_water_b = state.stage == 2

        def rocket_hits(bx, by, active):
            return jnp.logical_and(
                jnp.logical_and(active, state.rocket_active),
                _aabb_overlap(
                    bx, by,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.rocket_x, state.rocket_y,
                    self.consts.ROCKET_WIDTH, self.consts.ROCKET_HEIGHT,
                ),
            )

        air_hit = jnp.logical_and(
            in_water_b,
            rocket_hits(state.player_bullet_x, state.player_bullet_y, state.player_bullet_active),
        )
        water_hit = jnp.logical_and(
            in_water_b,
            rocket_hits(state.player_wbullet_x, state.player_wbullet_y, state.player_wbullet_active),
        )
        rocket_hit = air_hit | water_hit
        debris_hit = jnp.logical_and(
            jnp.logical_and(in_water_b, state.player_bullet_active),
            jnp.logical_and(
                state.wb_flyer_active,
                _aabb_overlap(
                    state.player_bullet_x, state.player_bullet_y,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.wb_flyer_x, state.wb_flyer_y,
                    self.consts.WB_FLYER_WIDTH, self.consts.WB_FLYER_HEIGHT,
                ),
            ),
        )
        sub_hit = jnp.logical_and(
            jnp.logical_and(in_water_b, state.player_wbullet_active),
            jnp.logical_and(
                state.submarine_active,
                _aabb_overlap(
                    state.player_wbullet_x, state.player_wbullet_y,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.submarine_x, jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32),
                    self.consts.SUBMARINE_WIDTH, self.consts.SUBMARINE_HEIGHT,
                ),
            ),
        )
        torp_hit = jnp.logical_and(
            in_water_b & (state.player_bullet_active | state.player_wbullet_active),
            jnp.logical_and(
                state.sub_torp_active,
                jnp.logical_or(
                    _aabb_overlap(
                        state.player_wbullet_x, state.player_wbullet_y,
                        self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                        state.sub_torp_x, state.sub_torp_y,
                        self.consts.SUB_SHOT_WIDTH, self.consts.SUB_SHOT_HEIGHT,
                    ),
                    _aabb_overlap(
                        state.player_bullet_x, state.player_bullet_y,
                        self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                        state.sub_torp_x, state.sub_torp_y,
                        self.consts.SUB_SHOT_WIDTH, self.consts.SUB_SHOT_HEIGHT,
                    )
                )
            )
        )

        air_used = air_hit | debris_hit | torp_hit
        water_used = water_hit | sub_hit | torp_hit
        gained = (
            rocket_hit.astype(jnp.int32) * self.consts.SCORE_ROCKET
            + debris_hit.astype(jnp.int32) * self.consts.SCORE_DEBRIS_SHOT
            + sub_hit.astype(jnp.int32) * self.consts.SCORE_SUBMARINE
            + torp_hit.astype(jnp.int32) * self.consts.SCORE_TORPEDO_SHOT
        )

        def park(keep, v):
            return jnp.where(keep, v, jnp.array(-1, dtype=v.dtype))

        air_keep = state.player_bullet_active & (~air_used)
        water_keep = state.player_wbullet_active & (~water_used)
        return state.replace(
            score=(state.score + gained).astype(jnp.int32),
            rocket_active=state.rocket_active & (~rocket_hit),
            wb_flyer_active=state.wb_flyer_active & (~debris_hit),
            submarine_active=state.submarine_active & (~sub_hit),
            sub_torp_active=state.sub_torp_active & (~torp_hit),
            player_bullet_active=air_keep,
            player_bullet_step=park(air_keep, state.player_bullet_step),
            player_bullet_x=park(air_keep, state.player_bullet_x),
            player_bullet_y=park(air_keep, state.player_bullet_y),
            player_wbullet_active=water_keep,
            player_wbullet_step=park(water_keep, state.player_wbullet_step),
            player_wbullet_x=park(water_keep, state.player_wbullet_x),
            player_wbullet_y=park(water_keep, state.player_wbullet_y),
        )

    def _resolve_oil_rig_landing(
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> JamesBondState:
        ## Only the player's downward crossing of the top counts as landing.
        ## Merely rising beside the rig or having it scroll under the boat
        ## must not complete the scene. Never move the player onto the rig.
        ## Compare the player's feet before/after movement against the rig's
        ## current scrolled position, not where it was last drawn. This also
        ## catches a fast descent that crosses the top between two frames.
        over_deck = (
            (state.player_x + self.consts.PLAYER_COLLISION_WIDTH > (state.oil_rig_x + self.consts.OIL_RIG_WIDTH / 2)) ## Has to land on the top-right of the oil rig (the platform)
            & (state.player_x < state.oil_rig_x + self.consts.OIL_RIG_WIDTH)
        )
        crossing_top = (
            (previous_state.player_y + self.consts.PLAYER_COLLISION_HEIGHT <= state.oil_rig_y)
            & (state.player_y + self.consts.PLAYER_COLLISION_HEIGHT >= state.oil_rig_y)
            & (state.player_y > previous_state.player_y)
            & (state.player_falling | state.player_fast_falling)
        )
        landed = (
            (state.stage == 1) & state.oil_rig_active & over_deck & crossing_top
            & (state.death_timer == 0) & (state.stage_transition_timer == 0)
        )
        return state.replace(
            oil_rig_visible=state.oil_rig_visible | landed,
            stage_transition_timer=jnp.where(
                landed, self.consts.STAGE_TRANSITION_FRAMES, state.stage_transition_timer
            ),
            score=jnp.where(
                landed, 
                state.score + self.consts.SCORE_STAGE_BONUS, 
                state.score
            ),
        )

    def _resolve_oil_rig_collision(self, state: JamesBondState) -> JamesBondState:
        ## Successful top landings already start the completion freeze before
        ## this collision pass. Any remaining body overlap is a crash.
        side_hit = jnp.logical_and(
            (state.stage == 1) & state.oil_rig_active,
            _aabb_overlap(
                state.player_x, state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH, self.consts.PLAYER_COLLISION_HEIGHT,
                state.oil_rig_x, state.oil_rig_y,
                self.consts.OIL_RIG_WIDTH, self.consts.OIL_RIG_HEIGHT,
            ),
        )
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(side_hit, can_take_damage)
        return state.replace(
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            stage1_start_step=jnp.where( ## For proper oil rig spawning
                took_damage,
                state.step_count, ## Tip: Maybe change to step_count + 2000 if you want to start from scuba
                state.stage1_start_step,
            ),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
            ## Crashing into the rig decommissions it -- it disappears on
            ## contact like other objects, while the player also dies.
            oil_rig_seq=jnp.where(
                side_hit, jnp.array(0, dtype=jnp.int32), state.oil_rig_seq
            ),
            oil_rig_landing_timer=jnp.where(side_hit, 0, state.oil_rig_landing_timer),
            oil_rig_active=state.oil_rig_active & ~side_hit,
            oil_rig_visible=state.oil_rig_visible | side_hit,
            scuba_seen=jnp.where(
                took_damage,
                False,
                state.scuba_seen
            ),
        )

    def _resolve_waterb_collisions(self, state: JamesBondState) -> JamesBondState:
        """Second water scene contacts.

        Ramming the floating rocket is the scene's only score (+200), and
        in the real game the ram usually costs a life as well -- both
        effects fire here, the damage under the usual cooldown. The
        submarine and the two flyers just hurt: the submarine can only
        reach a diving boat, the flyers only a jumping one.
        """

        def touch(ox, oy, ow, oh):
            return _aabb_overlap(
                state.player_x,
                state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH,
                self.consts.PLAYER_COLLISION_HEIGHT,
                ox, oy, ow, oh,
            )

        rocket_hit = jnp.logical_and(
            state.rocket_active,
            touch(state.rocket_x, state.rocket_y,
                  self.consts.ROCKET_WIDTH, self.consts.ROCKET_HEIGHT),
        )
        submarine_hit = jnp.logical_and(
            state.submarine_active,
            touch(state.submarine_x, jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32),
                  self.consts.SUBMARINE_WIDTH, self.consts.SUBMARINE_HEIGHT),
        )
        heli_hit = jnp.logical_and(
            state.pinkball_active,
            touch(state.pinkball_x, jnp.array(self.consts.PINKBALL_Y, dtype=jnp.int32),
                  self.consts.PINKBALL_WIDTH, self.consts.PINKBALL_HEIGHT),
        )
        ## The submarine's double-dot shot, on its diagonal or its run
        ## along the surface, costs a life too
        shot_hit = jnp.logical_and(
            state.sub_torp_active,
            touch(state.sub_torp_x, state.sub_torp_y - 3, ## Hits higher than the sprite size
                  self.consts.SUB_SHOT_WIDTH, self.consts.SUB_SHOT_HEIGHT),
        )
        any_hit = rocket_hit | submarine_hit | heli_hit | shot_hit
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(any_hit, can_take_damage)

        return state.replace(
            score=state.score + rocket_hit.astype(jnp.int32) * self.consts.SCORE_ROCKET + submarine_hit.astype(jnp.int32) * self.consts.SCORE_SUBMARINE,
            rocket_active=jnp.logical_and(state.rocket_active, ~rocket_hit),
            sub_torp_active=jnp.logical_and(state.sub_torp_active, ~shot_hit),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            rocket_y=jnp.where(
                took_damage,
                self.consts.ROCKET_Y,
                state.rocket_y,
            ),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def _resolve_splash_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """One life of damage from the radioactive splash hazard.

        The laser splash explosion straddles the surface and kills on
        near-contact (measured: within ~1px, submerged or afloat); only
        a clearly airborne boat passes over it safely.
        """

        ## Splash: the measured kill window is boat_x in
        ## [splash_x - 9, splash_x + 19], plus an altitude gate
        x_touch = jnp.logical_and(
            state.player_x >= state.splash_x - 9,
            state.player_x <= state.splash_x + 19,
        )
        low_enough = state.player_y >= self.consts.SPLASH_SAFE_PLAYER_Y
        splash_hit = jnp.logical_and(
            jnp.logical_and(
                state.splash_active,
                jnp.logical_not(state.scuba_active),
            ),
            jnp.logical_and(x_touch, low_enough),
        )

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - splash_hit.astype(jnp.int32)
            ).astype(jnp.int32),
            stage1_start_step=jnp.where( ## For proper oil rig spawning
                splash_hit,
                state.step_count,
                state.stage1_start_step,
            ),
            hit_cooldown=jnp.where(
                splash_hit,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                splash_hit,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
            scuba_seen=jnp.where(
                splash_hit,
                False,
                state.scuba_seen
            ),
        )

    def _resolve_bullet_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """One life of damage when the bomb or the laser hits the player.

        The projectile always disappears on contact, the life is only lost
        when the hit cooldown ran out, same rule as the pit. Manual says the
        laser destroys on impact and can't be shot down in stage 1.
        """

        bomb_overlap = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.helicopter_bomb_x,
            state.helicopter_bomb_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        laser_overlap = _aabb_overlap(
            state.player_x,
            state.player_y,
            self.consts.PLAYER_COLLISION_WIDTH,
            self.consts.PLAYER_COLLISION_HEIGHT,
            state.satellite_laser_x,
            state.satellite_laser_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        bomb_hit = jnp.logical_and(state.helicopter_bomb_active, bomb_overlap)
        laser_hit = jnp.logical_and(state.satellite_laser_active, laser_overlap)
        hit_any = jnp.logical_or(bomb_hit, laser_hit)
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(hit_any, can_take_damage)

        return state.replace(
            helicopter_bomb_active=jnp.logical_and(
                state.helicopter_bomb_active, jnp.logical_not(bomb_hit)
            ),
            satellite_laser_active=jnp.logical_and(
                state.satellite_laser_active, jnp.logical_not(laser_hit)
            ),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            stage1_start_step=jnp.where( ## For proper oil rig spawning
                took_damage,
                state.step_count,
                state.stage1_start_step,
            ),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
            scuba_seen=jnp.where(
                took_damage,
                False,
                state.scuba_seen
            ),
        )

    def _resolve_pit_player_collisions(self, state: JamesBondState) -> JamesBondState:
        """Apply one life of damage when the player drives into the fire pit.

        Only ground contact is deadly: a jumping player clears the pit. The
        player bullet and helicopter bombs pass over pits without responding,
        matching the original game, so no projectile checks happen here. The
        deadly zone is centered inside the wider pit sprite so an edge tap is
        survivable.
        """

        pit_inset = (self.consts.PIT_WIDTH - self.consts.PIT_COLLISION_WIDTH) / 2
        pit_left = state.pit_x + pit_inset
        pit_right = pit_left + self.consts.PIT_COLLISION_WIDTH
        x_overlap = jnp.logical_and(
            state.player_x < pit_right,
            state.player_x + self.consts.PLAYER_COLLISION_WIDTH > pit_left - 3,
        )
        on_ground = (state.player_y == self.consts.PLAYER_INIT_Y)
        pit_collision = jnp.logical_and(
            state.pit_active, jnp.logical_and(on_ground, x_overlap)
        )
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(pit_collision, can_take_damage)

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def scuba_collisions_logic(self, state: JamesBondState) -> JamesBondState:
        """The depth charge removes the diver and pays SCORE_SCUBA.

        The hit box follows the figure on screen: the swimmer's own 7x20
        body at his depth, or -- once he is radioactive -- the 20x7 splash
        figure straddling the waterline.
        """
        diver_active = jnp.logical_and(state.scuba_active, ~state.scuba_radioactive)

        bullet_overlap = _aabb_overlap(
            state.player_wbullet_x,
            state.player_wbullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            state.scuba_x,
            state.scuba_y,
            self.consts.SCUBA_WIDTH,
            self.consts.SCUBA_HEIGHT
        )

        bullet_hit = jnp.logical_and(
            jnp.logical_and(diver_active, state.player_wbullet_active),
            bullet_overlap,
        )

        player_wbullet_active = jnp.logical_and(
            state.player_wbullet_active, ~bullet_hit
        )

        new_score = state.score + bullet_hit * self.consts.SCORE_SCUBA

        ## Scuba splash collision logic
        x_touch = jnp.logical_and(
            state.player_x >= state.scuba_x - 9,
            state.player_x <= state.scuba_x + 19,
        )
        low_enough = state.player_y >= self.consts.SPLASH_SAFE_PLAYER_Y
        splash_hit = jnp.logical_and(
            state.scuba_radioactive,
            jnp.logical_and(x_touch, low_enough),
        )

        def park(active, v):
            return jnp.where(active, v, -1)

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - splash_hit.astype(jnp.int32)
            ).astype(jnp.int32),
            stage1_start_step=jnp.where( ## For proper oil rig spawning
                splash_hit,
                state.step_count,
                state.stage1_start_step,
            ),
            hit_cooldown=jnp.where(
                splash_hit,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                splash_hit,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
            scuba_seen=jnp.where(
                splash_hit,
                False,
                state.scuba_seen
            ),
            scuba_active = jnp.logical_and(
                state.scuba_active, ~bullet_hit
            ),
            scuba_radioactive=jnp.logical_and(state.scuba_radioactive, ~bullet_hit),
            player_wbullet_active=player_wbullet_active,
            player_wbullet_step=park(player_wbullet_active, state.player_wbullet_step),
            player_wbullet_x=park(player_wbullet_active, state.player_wbullet_x),
            player_wbullet_y=park(player_wbullet_active, state.player_wbullet_y),
            score=new_score
        )

    def oil_rig_collisions_logic(self, state: JamesBondState) -> JamesBondState:
        """
        Player bullets hitting the oil rig should disappear. Might be useful to indicate that the invisible oil rig is there.
        """
        bullet_overlap = _aabb_overlap(
            state.player_bullet_x,
            state.player_bullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            state.oil_rig_x,
            state.oil_rig_y,
            self.consts.OIL_RIG_WIDTH,
            self.consts.OIL_RIG_HEIGHT
        )

        wbullet_overlap = _aabb_overlap(
            state.player_wbullet_x,
            state.player_wbullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            state.oil_rig_x,
            state.oil_rig_y,
            self.consts.OIL_RIG_WIDTH,
            self.consts.OIL_RIG_HEIGHT
        )

        hit = (
            state.oil_rig_active
            & state.player_bullet_active
            & bullet_overlap
        )

        w_hit = (
            state.oil_rig_active
            & state.player_wbullet_active
            & wbullet_overlap
        )

        player_bullet_active = jnp.logical_and(
            state.player_bullet_active, ~hit
        )

        player_wbullet_active = jnp.logical_and(
            state.player_wbullet_active, ~w_hit
        )

        def park(active, v):
            return jnp.where(active, v, -1)

        return state.replace(
            player_bullet_active=player_bullet_active,
            player_bullet_step=park(player_bullet_active, state.player_bullet_step),
            player_bullet_x=park(player_bullet_active, state.player_bullet_x),
            player_bullet_y=park(player_bullet_active, state.player_bullet_y),

            player_wbullet_active=player_wbullet_active,
            player_wbullet_step=park(player_wbullet_active, state.player_wbullet_step),
            player_wbullet_x=park(player_wbullet_active, state.player_wbullet_x),
            player_wbullet_y=park(player_wbullet_active, state.player_wbullet_y),
        )

    def collectible_collisions_logic(self, state: JamesBondState) -> JamesBondState:
        """Collect active diamonds that overlap a player shot.

        Both shots count: the land round and the water anti-air round fly
        the same up-forward path, and shooting the floating gem is worth
        +50 in every scene (verified in ALE on land and over the water).
        """

        ## Both animation poses have their solid gem at x+1..5, y+3..8.
        ## The sprite's first rows contain sparkles; anchoring the hitbox
        ## there excluded the bottom tip and let visible hits pass through.
        overlap = _aabb_overlap(
            state.player_bullet_x,
            state.player_bullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            state.diamond_x + 1, ## For better hit boxes
            state.diamond_y + 3,
            self.consts.DIAMOND_COLLISION_WIDTH,
            self.consts.DIAMOND_COLLISION_HEIGHT,
        )

        collected = jnp.logical_and(
            jnp.logical_and(state.diamond_active, state.player_bullet_active),
            overlap,
        )

        player_bullet_active = jnp.logical_and(
            state.player_bullet_active, ~collected
        )

        next_sky_flash_timer = jnp.where(
            collected & (state.stage == 1),
            self.consts.DIAMOND_FLASH_FRAMES,
            state.sky_flash_timer,
        )

        new_score = state.score + collected * self.consts.SCORE_DIAMOND

        def park(active, v):
            return jnp.where(active, v, -1)

        return state.replace(
            diamond_shot=collected,
            sky_flash_timer=next_sky_flash_timer,
            diamond_active = jnp.logical_and( ## Tip: reset diamond x and y?
                state.diamond_active, ~collected
            ),
            player_bullet_active=player_bullet_active,
            player_bullet_step=park(player_bullet_active, state.player_bullet_step),
            player_bullet_x=park(player_bullet_active, state.player_bullet_x),
            player_bullet_y=park(player_bullet_active, state.player_bullet_y),
            score=new_score
        )
    
    def _resolve_bullet_diamond_collisions(self, state: JamesBondState) -> JamesBondState:
        ## Consume last frame's hit even if it already consumed the bullet.
        state = state.replace(diamond_shot=jnp.array(False, dtype=jnp.bool_))
        return lax.cond(
            jnp.logical_and(state.player_bullet_active, state.diamond_active),
            self.collectible_collisions_logic,
            lambda s: s,
            state
        )

    def _resolve_wbullet_scuba_collisions(self, state: JamesBondState) -> JamesBondState:
        return lax.cond(
            state.scuba_active | state.scuba_radioactive,
            self.scuba_collisions_logic,
            lambda s: s,
            state
        )

    def _resolve_bullet_oil_rig_collisions(self, state: JamesBondState) -> JamesBondState:
        return lax.cond(
            jnp.logical_and(
                jnp.logical_or(state.player_bullet_active, state.player_wbullet_active),
                state.oil_rig_active
            ),
            self.oil_rig_collisions_logic,
            lambda s: s,
            state
        )

    def _get_reward(
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        """Calculate reward from collision-driven state transitions."""

        return state.score - previous_state.score

    def _get_done(self, state: JamesBondState) -> chex.Array:
        return jnp.logical_or(
            state.lives <= 0,
            state.stage >= 3,
        )


class JamesBondRenderer(JAXGameRenderer):
    """Procedural rectangle renderer for the skeleton environment."""

    def __init__(
        self,
        consts: JamesBondConstants = None,
        config: render_utils.RendererConfig = None,
    ):
        self.consts = consts or JamesBondConstants()
        super().__init__(self.consts)

        if config is None:
            config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None,
            )
        self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        ## The jamesbond sprites are committed in the repo, not part of the
        ## downloadable sprite pack, so load them from jb_sprites directly.
        sprite_path = JB_SPRITE_DIR

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: JamesBondState) -> jnp.ndarray:
        """Render a simple background, inactive object slots, and player box."""

        raster = self.jr.create_object_raster(self.BACKGROUND)

        ## Both water scenes have a gray sky covering the measured rows
        ## 29-120, meeting the water at 121; it goes under the stars so
        ## the stars still twinkle on it like in the real scenes
        raster = jax.lax.cond(
            state.stage >= 1,
            lambda r: self.jr.render_at_clipped(r, 4, 29, self.SHAPE_MASKS['water_sky']),
            lambda r: r,
            raster,
        )
        ## Measured: on the very first frame of a water-scene death the
        ## whole sky flashes #6f6f6f (the death_timer sits at its full
        ## value for exactly that one frame)
        raster = jax.lax.cond(
            jnp.logical_and(
                state.stage >= 1,
                state.death_timer >= self.consts.DEATH_ANIMATION_FRAMES - 2,
            ),
            lambda r: self.jr.render_at_clipped(r, 4, 29, self.SHAPE_MASKS['death_flash']),
            lambda r: r,
            raster,
        )
        raster = self._render_oil_rig_flash(raster, state)
        raster = self._render_stars(raster, state)
        ## Terrain follows the scene: dry land in stage 0, water in stage 1
        raster = jax.lax.cond(
            state.stage == 0,
            lambda r: self._render_ground(r, state),
            lambda r: self._render_water(r, state),
            raster,
        )

        raster = self._render_car(raster, state)
        raster = self._render_diamond(raster, state)
        raster = self._render_pit(raster, state)
        raster = self._render_helicopter(raster, state)
        raster = self._render_helicopter_melee(raster, state)
        raster = self._render_satellite(raster, state)
        raster = self._render_scuba(raster, state)
        raster = self._render_splash(raster, state)
        raster = self._render_waterb(raster, state)
        raster = self._render_oil_rig(raster, state)

        raster = self._render_bullets(raster, state)
        raster = self._render_sinking_bolt(raster, state)

        ## Render life counter: the HUD shows RESERVE lives (max 3 icons),
        ## not the life currently being played, matching the real game
        raster = self.jr.render_indicator(
            raster, 9, 184,
            jnp.maximum(state.lives - 1, 0),
            self.SHAPE_MASKS['life'], 16, 3,
        )

        ## Render black borders on the sides of the screen
        raster = self.jr.render_at(
            raster,
            0,
            2,
            self.SHAPE_MASKS['black_border']
        )
        raster = self.jr.render_at(
            raster,
            210,
            2,
            self.SHAPE_MASKS['black_border']
        )
        
        ## Render Score counter (From Seaquest)
        max_score_digits = 5
        score_digits = self.jr.int_to_digits(state.score, max_digits=max_score_digits)
        clamped_score = jnp.minimum(jnp.maximum(state.score, 0), 10**max_score_digits - 1)
        score_digit_thresholds = jnp.array([1, 10, 100, 1000, 10000, 100000], dtype=clamped_score.dtype)
        num_score_digits = jnp.maximum(1, jnp.sum(clamped_score >= score_digit_thresholds))
        score_start_index = max_score_digits - num_score_digits
        score_x = 59 + score_start_index * 8

        raster = self.jr.render_label_selective(
            raster,
            score_x,
            9,
            score_digits,
            self.SHAPE_MASKS['score_digits'],
            score_start_index,
            num_score_digits,
            spacing=8,
            max_digits_to_render=max_score_digits,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_ground(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the static ground strip using the existing ground sprite.

        The sprite's first row is the gray road top, which sits at row 124
        in the real game -- BELOW the player, whose hull rides rows
        119-122 with the wheels touching the road. Drawing it at 119 put
        the road at roof height and sank the car into the ground.
        """

        return self.jr.render_at_clipped(
            raster,
            4,    # x - real playfield left edge (columns 0-7 stay black like ALE)
            123,  # y - road top just under the wheels, like the real rows
            self.SHAPE_MASKS['ground'],
        )

    def _render_water(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the water scene terrain: the water body and the seabed.

        Both sprites were extracted from the real water scene. The water
        band starts at the same row the land ground uses, so the boat rides
        the surface exactly where the car drives; the seabed silhouette
        sits at the bottom of the water body.
        """

        ## The second water scene runs the same layout with darker water.
        ## The water surface is row 121 in the real game: the boat's hull
        ## (119-122) rides half above, half below the waterline.
        raster = jax.lax.cond(
            state.stage == 2,
            lambda r: self.jr.render_at_clipped(r, 4, 121, self.SHAPE_MASKS['water_b']),
            lambda r: self.jr.render_at_clipped(r, 4, 121, self.SHAPE_MASKS['water']),
            raster,
        )
        ## The seabed is a 160px repeating strip that scrolls left with the
        ## world, 1px every 4th frame. Drawing the pattern twice, one strip
        ## width apart, keeps the wrap seamless.
        scroll = (state.step_count // 4) % 160
        raster = self.jr.render_at_clipped(
            raster,
            4 - scroll,
            158,  # y - measured seabed top row
            self.SHAPE_MASKS['seabed'],
        )
        return self.jr.render_at_clipped(
            raster,
            4 - scroll + 160,
            158,
            self.SHAPE_MASKS['seabed'],
        )

    def _render_scuba(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw normal scuba or the satellite-drop radioactive figure.

        Normal scuba switches swim frames every 15 frames. When proximity
        activates radioactivity, replace the swimmer with the exact same
        narrow/wide two-pose figure used by a radioactive satellite drop.
        """

        normal_idx = (state.step_count // 15) % 2
        radioactive_narrow = ((state.scuba_radioactive_age + 6) // 7) % 2 == 0

        def draw_normal(r):
            return self.jr.render_at_clipped(
                r,
                state.scuba_x,
                state.scuba_y,
                self.SHAPE_MASKS['scuba'][normal_idx],
            )

        ## The radioactive figure surfaces: it is drawn straddling the
        ## waterline like the satellite splash, not at the diver's depth
        def draw_radioactive(r):
            return jax.lax.cond(
                radioactive_narrow,
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    state.scuba_x,
                    jnp.array(self.consts.SPLASH_Y, dtype=jnp.int32),
                    self.SHAPE_MASKS['splash'][0],
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    state.scuba_x - 4,
                    jnp.array(self.consts.SPLASH_Y, dtype=jnp.int32),
                    self.SHAPE_MASKS['splash'][1],
                ),
                r,
            )

        def draw_active(r):
            return jax.lax.cond(
                state.scuba_radioactive,
                draw_radioactive,
                draw_normal,
                r,
            )

        return jax.lax.cond(state.scuba_active, draw_active, lambda r: r, raster)

    def _render_splash(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the splash only when no scuba diver is present.

        Frame-exact from ALE: the narrow pose shows for exactly 1 frame at
        spawn, then strict 7-frame phases alternate starting with the wide
        pose (which reaches 4px further left and carries the yellow
        under-glow rows). Green is visible every single frame -- only the
        under-glow toggles with the wide phase.
        """

        narrow = ((state.splash_age + 6) // 7) % 2 == 0

        def draw_fn(r):
            return jax.lax.cond(
                narrow,
                lambda rr: self.jr.render_at_clipped(
                    rr, state.splash_x, self.consts.SPLASH_Y,
                    self.SHAPE_MASKS['splash'][0],
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr, state.splash_x - 4, self.consts.SPLASH_Y,
                    self.SHAPE_MASKS['splash'][1],
                ),
                r,
            )

        visible = jnp.logical_and(
            state.splash_active,
            jnp.logical_not(state.scuba_active),
        )
        return jax.lax.cond(visible, draw_fn, lambda r: r, raster)

    ##def _render_oil_rig(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
    ##    """ Draw the oil rig only during active flase frames. """
    ##    is_visible = state.oil_rig_active & (state.oil_rig_visible_timer > 0)
    ##
    ##    draw_fn = lambda r: self.jr.render_at_clipped(
    ##        r,
    ##        state.oil_rig_x,
    ##        state.oil_rig_y,
    ##        self.SHAPE_MASKS['oil_rig'][0],
    ##    )
    ##
    ##    return jax.law.cond(is_visible, draw_fn, lambda r: r, raster)

    def _render_sinking_bolt(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """A bolt landing while the frogman lives sinks radioactive-green.

        Measured: no second frogman spawns; the bar keeps falling under the
        waterline drawn in the frogman's green instead of yellow.
        """

        submerged = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(state.scuba_active),
                jnp.logical_and(state.stage == 1, state.splash_active),
            ),
            jnp.logical_and(
                state.satellite_laser_active,
                state.satellite_laser_y > 119,
            ),
        )

        def draw_fn(r):
            return self.jr.render_at_clipped(
                r,
                state.satellite_laser_x,
                state.satellite_laser_y,
                self.SHAPE_MASKS['laser_green'],
            )

        return jax.lax.cond(submerged, draw_fn, lambda r: r, raster)

    def _render_oil_rig_flash(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray: ###########
        """Timed medium-gray flashes for hits, rocket bursts and completion."""
        rig_flash = (state.stage >= 1) & (
            (state.sky_flash_timer > 0) | (state.stage_transition_timer > 0)
        )
        def _rig_flash(r):
            pos = jnp.array([[0, 29]], dtype=jnp.int32)   # sky band only, starts below the HUD
            size = jnp.array([[self.consts.SCREEN_WIDTH, 92]], dtype=jnp.int32)  # rows 29..120
            new_raster = jax.lax.cond(
                state.step_count % 2,
                lambda r: self.jr.draw_rects(r, pos, size, self.COLOR_TO_ID[(142, 142, 142)]),
                lambda r: r,
                raster,
            )
            return new_raster

        return jax.lax.cond(rig_flash, _rig_flash, lambda r: r, raster)

    def _render_oil_rig(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the reveal or landing confirmation, only in water A."""
        def draw_fn(r):
            return self.jr.render_at_clipped(
                r, state.oil_rig_x, state.oil_rig_y, self.SHAPE_MASKS['oil_rig'][0]
            )
        return jax.lax.cond(state.oil_rig_visible, draw_fn, lambda r: r, raster)

    def _render_waterb(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the second water scene roster."""

        index_switch = (state.step_count // 8) % 2

        def render_with_switch(raster, active, x, y, mask):
            return jax.lax.cond(
                active,
                lambda r: self.jr.render_at_clipped(r, x, y, mask),
                lambda r: r,
                raster,
            )

        raster = render_with_switch(
            raster,
            state.submarine_active,
            state.submarine_x,
            self.consts.SUBMARINE_Y,
            self.SHAPE_MASKS["submarine"][index_switch],
        )

        raster = render_with_switch(
            raster,
            state.rocket_active,
            state.rocket_x,
            state.rocket_y,
            self.SHAPE_MASKS["rocket"][index_switch],
        )

        raster = render_with_switch(
            raster,
            state.pinkball_active,
            state.pinkball_x,
            self.consts.PINKBALL_Y,
            self.SHAPE_MASKS["wb_ball"][index_switch],
        )

        raster = render_with_switch(
            raster,
            state.sub_torp_active,
            state.sub_torp_x,
            state.sub_torp_y,
            self.SHAPE_MASKS["sub_shot"][index_switch],
        )

        ## Rocket debris: the red bomb while falling, then the red / pink
        ## sparkle (two dot patterns swapping) once it rests at the waterline
        debris_resting = state.wb_flyer_y >= self.consts.DEBRIS_REST_Y
        splash_pose = (state.step_count // self.consts.DEBRIS_SPLASH_FLIP_FRAMES) % 2
        raster = render_with_switch(
            raster,
            state.wb_flyer_active & (~debris_resting),
            state.wb_flyer_x,
            state.wb_flyer_y,
            self.SHAPE_MASKS["rocket_ball"][index_switch],
        )
        raster = render_with_switch(
            raster,
            state.wb_flyer_active & debris_resting,
            state.wb_flyer_x - 1,
            self.consts.DEBRIS_SPLASH_Y,
            self.SHAPE_MASKS["debris_splash"][splash_pose],
        )

        return raster

    def _render_stars(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Scattered star field across the whole sky; two frames alternate
        slowly for a gentle twinkle."""
        idx = jnp.where((state.step_count // 24) % 2 == 0, 0, 1)
        return self.jr.render_at_clipped(raster, 4, 34, self.SHAPE_MASKS['stars'][idx])

    def _render_background(self, raster: jnp.ndarray) -> jnp.ndarray:
        """Draw the placeholder play area."""

        position = jnp.array(
            [[self.consts.GAME_AREA_MIN_X, self.consts.GAME_AREA_MIN_Y]],
            dtype=jnp.int32,
        )
        size = jnp.array(
            [
                [
                    self.consts.GAME_AREA_MAX_X - self.consts.GAME_AREA_MIN_X,
                    self.consts.GAME_AREA_MAX_Y - self.consts.GAME_AREA_MIN_Y,
                ]
            ],
            dtype=jnp.int32,
        )
        return self.jr.draw_rects(raster, position, size, self.PLAY_AREA_ID)

    def _render_car(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the player, color-cycling through the recolors while dying.

        The world clock freezes during the 59-frame death animation, so
        the cycle is driven by death_timer (which keeps counting down);
        the multiplier makes the recolor order look random like the real
        sprite's per-frame color roll.
        """

        sprite_idx = jnp.where(
            state.death_timer > 0,
            1 + (state.death_timer * 5) % 3,
            jnp.where(
                state.hit_cooldown > 0,
                jnp.where(state.step_count % 2 == 0, 1, 2),
                0
            )
        )

        return self.jr.render_at_clipped(
            raster,
            state.player_x,
            state.player_y,
            self.SHAPE_MASKS['car'][sprite_idx]
        )
    
    def _render_diamond(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        ## Sparkle alternation is a few frames per phase in the real game;
        ## flipping every single frame made the whole gem vibrate.
        sprite_idx = jnp.where(
            (state.step_count // 4) % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.diamond_x,
            state.diamond_y,
            self.SHAPE_MASKS['diamond'][sprite_idx],
        )

        return jax.lax.cond((state.stage < 2) & state.diamond_active, draw_fn, lambda r: r, raster)

    def _render_pit(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the fire pit."""

        ## The flame blinks in irregular multi-frame bursts in the real
        ## game; a few frames per phase reads right without the strobe.
        sprite_idx = jnp.where(
            (state.step_count // 3) % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r, 
            state.pit_x, 
            state.pit_y, 
            self.SHAPE_MASKS['pit'][sprite_idx],
        )

        return jax.lax.cond(state.pit_active, draw_fn, lambda r: r, raster)
    
    def _render_helicopter(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        ## Rotor frames alternate every 2-3 frames in the real game
        sprite_idx = jnp.where(
            (state.step_count // 2) % 2 == 0,
            0,
            1
        )

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.helicopter_x,
            state.helicopter_y,
            self.SHAPE_MASKS['helicopter'][sprite_idx]
        )

        return jax.lax.cond(state.helicopter_active, draw_fn, lambda r: r, raster)

    def _render_helicopter_melee(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the helicopter melee animation."""

        """
        ## UNFINISHED
        helicopter_melee_step = jnp.where( ## Only first 14 needed; ## 39 -> last visible left, 40 -> invisible, 41 -> second right
                    state.helicopter_melee_step > 40,
                    jnp.ceil(state.helicopter_melee_step / 2),
                    state.helicopter_melee_step
                )
                
                melee_idx = jnp.where( ## Create -> gone -> same -> gone -> new
                    helicopter_melee_step % 2 == 0,
                    0,
                    jnp.where(
                        state.step_count % 2 == 0,
                        (helicopter_melee_step - 1) / 2,
                        jnp.maximum(
                            (helicopter_melee_step - 3) / 4,
                            (helicopter_melee_step - 1) / 2,
                        )
                    )
                )
        """

        step = jnp.clip(
            state.helicopter_melee_step,
            0,
            self.consts.HELICOPTER_MELEE_SPRITE_STEPS.shape[0] - 1,
        )
        melee_idx = self.consts.HELICOPTER_MELEE_SPRITE_STEPS[step][0]

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.helicopter_x + self.consts.HELICOPTER_MELEE_SPRITE_STEPS[step][1],
            state.helicopter_y + 7,
            self.SHAPE_MASKS['helicopter_melee'][jnp.maximum(melee_idx, 0)]
        )

        visible = jnp.logical_and(state.helicopter_active, melee_idx != -1)
        return jax.lax.cond(visible, draw_fn, lambda r: r, raster)

    def _render_satellite(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.satellite_x,
            state.satellite_y,
            self.SHAPE_MASKS['satellite'],
        )

        return jax.lax.cond(state.satellite_active, draw_fn, lambda r: r, raster)
    
    def _render_bullets(self, raster: jnp.ndarray, state: JamesBondState,) -> jnp.ndarray:
        """Draw all projectiles, they share the same 1x4 bullet sprite."""

        ## player air bullet, player water bullet, helicopter bomb, satellite laser
        active_bullets = jnp.stack([
            state.player_bullet_active,
            state.player_wbullet_active,
            state.helicopter_bomb_active,
            state.satellite_laser_active,
        ])

        bullet_positions = jnp.stack([
            jnp.stack([state.player_bullet_x, state.player_bullet_y]),
            jnp.stack([state.player_wbullet_x, state.player_wbullet_y]),
            jnp.stack([state.helicopter_bomb_x, state.helicopter_bomb_y]),
            jnp.stack([state.satellite_laser_x, state.satellite_laser_y]),
        ])

        def render_single_bullet(i, current_raster):
            should_draw = (active_bullets[i] == 1)

            sprite_idx = jnp.where( ## Different sprite for water bullet
                i == 1,
                1,
                0
            )

            draw_fn = lambda r: self.jr.render_at_clipped(
                r,
                bullet_positions[i][0],
                bullet_positions[i][1],
                self.SHAPE_MASKS['bullet'][sprite_idx],
            )

            return jax.lax.cond(should_draw, draw_fn, lambda r: r, current_raster)

        return jax.lax.fori_loop(0, jnp.size(active_bullets), render_single_bullet, raster)
