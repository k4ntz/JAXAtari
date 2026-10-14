import jax
import jax.numpy as jnp
import chex
from functools import partial
from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin
from jaxtari.games.jax_seaquest import SeaquestState, SpawnState


class DisableEnemiesMod(JaxtariPostStepModPlugin):
    """Disable enemies in the environment."""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxtariModWrapper).
        """
        # Zero out all enemy positions
        return new_state.replace(
            shark_positions=jnp.zeros_like(new_state.shark_positions),
            sub_positions=jnp.zeros_like(new_state.sub_positions),
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions),
            surface_sub_position=jnp.zeros_like(new_state.surface_sub_position)
        )


class NoDiversMod(JaxtariInternalModPlugin):
    """
    Internal mod to remove Divers from the game.
    It suppresses the logic that updates/spawns divers and disables their rendering.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step_diver_movement(self,
            diver_positions: chex.Array,
            shark_positions: chex.Array,
            sub_positions: chex.Array,
            state_player_x: chex.Array,
            state_player_y: chex.Array,
            state_divers_collected: chex.Array,
            spawn_state: SpawnState,
            step_counter: chex.Array,
            rng: chex.PRNGKey
        ):
        """
        Override for step_diver_movement: keep divers off-screen / inactive.
        Signature matches JaxSeaquest.step_diver_movement (incl. sub_positions).
        """
        _ = (shark_positions, sub_positions, state_player_x, state_player_y, step_counter)
        return (
            jnp.full_like(diver_positions, -1),
            state_divers_collected,
            spawn_state,
            rng
        )

    @partial(jax.jit, static_argnums=(0,))
    def _draw_divers(self, raster: jnp.ndarray, state: SeaquestState):
        """
        Override for the renderer to skip drawing divers.
        """
        # Simply return the raster without drawing the sprite
        return raster


class EnemyMinesMod(JaxtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

    asset_overrides = {
        "shark_base": {
            'name': 'shark_base',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        },
        "enemy_sub": {
            'name': 'enemy_sub',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        }
    }

    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([[128, 128, 128]] * 5),
    }


class FireBallsMod(JaxtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

class UnlimitedOxygenMod(JaxtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        return new_state.replace(oxygen=jnp.array(64, dtype=jnp.int32))

class GravityMod(JaxtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        new_player_y = jnp.where(
            new_state.step_counter % 4 == 0,
            jnp.minimum(new_state.player_y + 1, self._env.consts.PLAYER_BOUNDS[1, 1]),
            new_state.player_y
        )
        return new_state.replace(player_y=new_player_y)

class RandomColorEnemiesMod(JaxtariInternalModPlugin):
    pass


class PenalizeDiverShootingMod(JaxtariPostStepModPlugin):
    """
    Penalizes the player for shooting divers with the torpedo.

    When the player's missile hits a diver:
    - The diver is killed (removed from play)
    - The missile is consumed (like hitting any other entity)
    - Score is penalized proportionally to difficulty:
        penalty = min(PENALTY_BASE + PENALTY_STEP * successful_rescues, PENALTY_MAX)
    """

    PENALTY_BASE = 50
    PENALTY_STEP = 25  # per successful rescue
    PENALTY_MAX = 500

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        missile_pos = new_state.player_missile_position
        missile_active = missile_pos[2] != 0

        # Calculate penalty based on successful rescues (difficulty proxy)
        penalty = jnp.minimum(
            self.PENALTY_BASE + self.PENALTY_STEP * new_state.successful_rescues,
            self.PENALTY_MAX
        )

        missile_xy = missile_pos[:2]
        missile_size = self._env.consts.MISSILE_SIZE
        diver_size = self._env.consts.DIVER_SIZE

        def check_diver(i, carry):
            state, missile_gone = carry
            diver_pos = state.diver_positions[i]

            should_check = jnp.logical_and(
                diver_pos[2] != 0,
                jnp.logical_not(missile_gone)
            )

            collision = self._env.check_collision_single(
                missile_xy, missile_size,
                jnp.array([diver_pos[0], diver_pos[1]]), diver_size
            )
            hit = jnp.logical_and(should_check, collision)

            state = state.replace(
                diver_positions=state.diver_positions.at[i].set(
                    jnp.where(hit, jnp.zeros(3, dtype=jnp.int32), diver_pos)
                ),
                score=jnp.where(hit, state.score - penalty, state.score),
                player_missile_position=jnp.where(
                    hit, jnp.zeros(3, dtype=jnp.int32), state.player_missile_position
                ),
            )

            return state, jnp.logical_or(missile_gone, hit)

        final_state = jax.lax.cond(
            missile_active,
            lambda s: jax.lax.fori_loop(0, 4, check_diver, (s, jnp.array(False)))[0],
            lambda s: s,
            new_state,
        )

        return final_state

# ---------------------------------------------------------------------------
# Upstreamed from seaquest-robustness (manual-verify set)
# ---------------------------------------------------------------------------

def _rgba_swatch(rgb: tuple[int, int, int]):
    """1×1 procedural RGBA pixel so `rgb` is present in the renderer palette."""
    import numpy as np

    return np.array([[[rgb[0], rgb[1], rgb[2], 255]]], dtype=np.uint8)

class ChangeBackgroundColorMod(JaxtariInternalModPlugin):
    """Magenta playfield OOD (visual-only).

    Do **not** replace the packed background asset: that drops the water blues /
    black from ``COLOR_TO_ID`` and makes ``_bake_surface_wave_backgrounds``
    KeyError when it falls back to ``bg/2.npy``. Instead we only seed the new
    color into the palette; ``SeaquestEnvMod`` remaps ``BACKGROUND_FRAMES``
    after a normal bake (left black pillar kept).
    """

    _NEW_BG = (180, 0, 180)

    constants_overrides = {
        "BACKGROUND_COLOR": jnp.array(list(_NEW_BG)),
    }
    asset_overrides = {
        "change_bg_swatch": {
            "name": "change_bg_swatch",
            "type": "procedural",
            "data": _rgba_swatch(_NEW_BG),
        }
    }

def _boost_enemy_x(positions: chex.Array, extra: int) -> chex.Array:
    """Apply extra horizontal movement to active enemy slots (shape N×3)."""

    def _boost_one(pos):
        active = pos[2] != 0
        new_x = pos[0] + pos[2] * extra
        return jnp.where(active, pos.at[0].set(new_x), pos)

    return jax.vmap(_boost_one)(positions)

def _oscillate_enemy_y(positions: chex.Array, offset: chex.Array, y_min: int, y_max: int) -> chex.Array:
    def _osc_one(pos):
        active = pos[2] != 0
        new_y = jnp.clip(pos[1] + offset, y_min, y_max)
        return jnp.where(active, pos.at[1].set(new_y), pos)

    return jax.vmap(_osc_one)(positions)

class PeacefulEnemiesMod(JaxtariInternalModPlugin):
    """Enemies remain but player collisions no longer cause death or bonus points."""

    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision(
        self,
        player_x,
        player_y,
        submarine_list,
        shark_list,
        surface_sub_pos,
        enemy_projectile_list,
        score,
        successful_rescues,
    ):
        del player_x, player_y, submarine_list, shark_list, surface_sub_pos
        del enemy_projectile_list, score, successful_rescues
        return jnp.array(False, dtype=jnp.bool_), jnp.array(0, dtype=jnp.int32)

class NoEnemyTorpedoesMod(JaxtariPostStepModPlugin):
    """Remove enemy torpedoes after each step (collision-only threats remain)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        return new_state.replace(
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions)
        )

class FasterEnemiesMod(JaxtariPostStepModPlugin):
    """Slightly faster escorts + their torpedoes: +1px on even frames.

    Vanilla torpedoes already scale with difficulty on their own schedule; this
    mod must boost missiles too or sped-up subs appear to escort their shot.
    Even-frame boost (~+0.5 px/frame) still stacks with the difficulty ramp.
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        extra = jnp.where(new_state.step_counter % 2 == 0, jnp.int32(1), jnp.int32(0))

        def _boost(positions):
            def _one(pos):
                active = pos[2] != 0
                new_x = pos[0] + pos[2] * extra
                return jnp.where(active, pos.at[0].set(new_x), pos)

            return jax.vmap(_one)(positions)

        def _exit_mask(pos):
            active = pos[2] != 0
            exited = jnp.where(pos[2] > 0, pos[0] > 160, pos[0] < 0)
            return jnp.logical_and(active, exited)

        def _clear_exited(pos):
            return jnp.where(_exit_mask(pos), jnp.zeros_like(pos), pos)

        sharks_b = _boost(new_state.shark_positions)
        subs_b = _boost(new_state.sub_positions)
        shark_exit = jax.vmap(_exit_mask)(sharks_b)
        sub_exit = jax.vmap(_exit_mask)(subs_b)
        sharks = jax.vmap(_clear_exited)(sharks_b)
        subs = jax.vmap(_clear_exited)(subs_b)

        spawn = new_state.spawn_state
        # ``survived`` is 12 slots for the *current* escort type (shark or sub).
        is_sub_lane = spawn.prev_sub == 1  # (4,)
        exit_12 = jnp.where(jnp.repeat(is_sub_lane, 3), sub_exit, shark_exit)
        dir_12 = jnp.where(
            jnp.repeat(is_sub_lane, 3),
            subs_b[:, 2].astype(jnp.int32),
            sharks_b[:, 2].astype(jnp.int32),
        )
        new_survived = jnp.where(exit_12, dir_12, spawn.survived)

        sharks_live = jnp.any(sharks.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        subs_live = jnp.any(subs.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        live = jnp.where(is_sub_lane, subs_live, sharks_live)
        reload = self._env.spawn_timer_reload(spawn.diver_array, after_survive=True)
        timers = jnp.where(
            jnp.logical_and(spawn.spawn_timers <= 0, jnp.logical_not(live)),
            reload,
            spawn.spawn_timers,
        )
        # Torpedoes use a separate difficulty schedule and were left at stock
        # speed, so boosted subs looked like they were escorting their own shot.
        # Match the same even-frame boost; drop missiles that leave the playfield.
        missiles_b = _boost(new_state.enemy_missile_positions)
        x_lo = self._env.consts.X_BORDERS[0]
        x_hi = self._env.consts.X_BORDERS[1]

        def _clear_missile(pos):
            active = pos[2] != 0
            oob = jnp.logical_or(pos[0] < x_lo, pos[0] > x_hi)
            return jnp.where(jnp.logical_and(active, oob), jnp.zeros_like(pos), pos)

        missiles = jax.vmap(_clear_missile)(missiles_b)

        return new_state.replace(
            shark_positions=sharks,
            sub_positions=subs,
            enemy_missile_positions=missiles,
            spawn_state=spawn.replace(survived=new_survived, spawn_timers=timers),
        )

class SlowerEnemiesMod(JaxtariPostStepModPlugin):
    """Halve enemy horizontal speed by skipping X updates on even frames.

    The old "undo 1px" implementation reversed slow enemies (diff-0 often moves
    0–1px/frame), freezing escorts on-screen and parking spawn timers at 0 so
    no divers appeared after the opening pair.
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        hold = new_state.step_counter % 2 == 0

        def _hold_x(new_pos, old_pos):
            active = new_pos[2] != 0
            return jnp.where(
                jnp.logical_and(active, hold),
                new_pos.at[0].set(old_pos[0]),
                new_pos,
            )

        return new_state.replace(
            shark_positions=jax.vmap(_hold_x)(
                new_state.shark_positions, prev_state.shark_positions
            ),
            sub_positions=jax.vmap(_hold_x)(
                new_state.sub_positions, prev_state.sub_positions
            ),
        )

class ShiftLanesMod(JaxtariInternalModPlugin):
    """Shift enemy lane Y positions downward by 10 pixels."""

    constants_overrides = {
        # Vanilla SPAWN_POSITIONS_Y [69, 93, 117, 141] + 10
        "SPAWN_POSITIONS_Y": jnp.array([79, 103, 127, 151], dtype=jnp.int32),
        # Vanilla ENEMY_MISSILE_Y [73, 97, 121, 145] + 10 (keep +4 lane offset)
        "ENEMY_MISSILE_Y": jnp.array([83, 107, 131, 155], dtype=jnp.int32),
    }


class OnlySubmarinesMod(JaxtariPostStepModPlugin):
    """Force enemy escorts to be submarines; keep diver spawn timers cycling.

    Shark→sub conversion happens after the shark+diver co-spawn path so divers
    still appear. Zeroing sharks without a timer reload used to park timers at 0
    forever (same failure mode as ``no_enemies``).
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        sharks = new_state.shark_positions
        subs = new_state.sub_positions
        # Prefer live sharks (just co-spawned) remapped into sub slots.
        converted_subs = jnp.where(sharks[:, 2:3] != 0, sharks, subs)
        spawn = new_state.spawn_state
        live = jnp.any(converted_subs.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        reload = self._env.spawn_timer_reload(spawn.diver_array)
        timers = jnp.where(
            jnp.logical_and(spawn.spawn_timers <= 0, jnp.logical_not(live)),
            reload,
            spawn.spawn_timers,
        )
        spawn = spawn.replace(
            prev_sub=jnp.where(
                spawn.prev_sub < 0, spawn.prev_sub, jnp.ones(4, dtype=jnp.int32)
            ),
            spawn_timers=timers,
            survived=jnp.zeros_like(spawn.survived),
            to_be_spawned=jnp.zeros_like(spawn.to_be_spawned),
        )
        return new_state.replace(
            shark_positions=jnp.zeros_like(sharks),
            sub_positions=converted_subs,
            spawn_state=spawn,
        )

class OnlySharksMod(JaxtariPostStepModPlugin):
    """Force all lanes to spawn sharks instead of enemy submarines.

    Dropping sub follow-up waves without reloading timers parks the countdown
    at 0 and stops further diver co-spawns after the opening pair.
    """

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        # Keep prev_sub < 0 so opening still uses FIRST_WAVE_DIVER_LANES.
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        sharks = new_state.shark_positions
        spawn = new_state.spawn_state
        live = jnp.any(sharks.reshape(4, 3, 3)[:, :, 2] != 0, axis=1)
        reload = self._env.spawn_timer_reload(spawn.diver_array)
        timers = jnp.where(
            jnp.logical_and(spawn.spawn_timers <= 0, jnp.logical_not(live)),
            reload,
            spawn.spawn_timers,
        )
        # Force shark waves going forward without erasing the opening sentinel
        # until the first wave has actually started (prev_sub already >= 0).
        prev_sub = jnp.where(spawn.prev_sub < 0, spawn.prev_sub, jnp.zeros_like(spawn.prev_sub))
        spawn = spawn.replace(
            prev_sub=prev_sub,
            spawn_timers=timers,
            survived=jnp.zeros_like(spawn.survived),
            to_be_spawned=jnp.zeros_like(spawn.to_be_spawned),
        )
        return new_state.replace(
            sub_positions=jnp.zeros_like(new_state.sub_positions),
            spawn_state=spawn,
        )

class VerticalOscillationMod(JaxtariPostStepModPlugin):
    """Sharks, subs, and divers oscillate vertically (±3 px)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        offset = (3.0 * jnp.sin(new_state.step_counter.astype(jnp.float32) * 0.15)).astype(
            jnp.int32
        )
        y_min = jnp.int32(46)
        y_max = jnp.int32(self._env.consts.PLAYER_BOUNDS[1, 1])
        return new_state.replace(
            shark_positions=_oscillate_enemy_y(new_state.shark_positions, offset, y_min, y_max),
            sub_positions=_oscillate_enemy_y(new_state.sub_positions, offset, y_min, y_max),
            diver_positions=_oscillate_enemy_y(new_state.diver_positions, offset, y_min, y_max),
        )

class FastOxygenDrainMod(JaxtariPostStepModPlugin):
    """Additional oxygen drain underwater every 16 frames (on top of base drain)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        underwater = new_state.player_y > jnp.int32(52)
        extra_tick = jnp.logical_and(
            underwater,
            jnp.logical_and(
                new_state.step_counter % 16 == 0,
                new_state.step_counter % 32 != 0,
            ),
        )
        new_oxygen = jnp.where(
            extra_tick,
            jnp.maximum(new_state.oxygen - 1, jnp.int32(0)),
            new_state.oxygen,
        )
        return new_state.replace(oxygen=new_oxygen)

class SlowOxygenDrainMod(JaxtariPostStepModPlugin):
    """Refill one oxygen unit on half of the normal underwater drain ticks."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        underwater = new_state.player_y > jnp.int32(52)
        would_drain = jnp.logical_and(underwater, new_state.step_counter % 32 == 0)
        skip_drain = jnp.logical_and(would_drain, new_state.step_counter % 64 != 0)
        new_oxygen = jnp.where(
            skip_drain,
            jnp.minimum(new_state.oxygen + 1, jnp.int32(64)),
            new_state.oxygen,
        )
        return new_state.replace(oxygen=new_oxygen)

class SurfaceSubAlwaysMod(JaxtariPostStepModPlugin):
    """Keep the surface patrol boat looping: respawn on the right when absent.

    Pinning X every frame left it glued to the right edge. Let ``surface_sub_step``
    move it; only re-arm a spawn when the slot is empty.
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        dtype = new_state.surface_sub_position.dtype
        exists = new_state.surface_sub_position[2] != 0
        spawned = jnp.array([159, 45, -1], dtype=dtype)
        return new_state.replace(
            surface_sub_position=jnp.where(exists, new_state.surface_sub_position, spawned)
        )

class InvertedLanesMod(JaxtariPostStepModPlugin):
    """Flip default lane travel directions."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        flipped = 1 - state.spawn_state.lane_directions
        spawn = state.spawn_state.replace(lane_directions=flipped.astype(jnp.int32))
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        flipped = 1 - new_state.spawn_state.lane_directions
        spawn = new_state.spawn_state.replace(lane_directions=flipped.astype(jnp.int32))
        return new_state.replace(spawn_state=spawn)

class LethalDiversMod(JaxtariPostStepModPlugin):
    """Diver contact triggers the death animation (game mod only; CBL labels still COLLECT)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state

        def _diver_hits(diver_pos):
            active = diver_pos[2] != 0
            hit = self._env.check_collision_single(
                jnp.array([new_state.player_x, new_state.player_y]),
                self._env.consts.PLAYER_SIZE,
                diver_pos,
                self._env.consts.DIVER_SIZE,
            )
            return jnp.logical_and(active, hit)

        diver_hit = jnp.any(jax.vmap(_diver_hits)(new_state.diver_positions))
        already_dying = new_state.death_counter > 0
        should_die = jnp.logical_and(diver_hit, jnp.logical_not(already_dying))
        new_spawn = jax.lax.cond(
            should_die,
            lambda s: self._env.soft_reset_spawn_state(
                s,
                rearm_divers=True,
                rng=jax.random.fold_in(new_state.rng_key, new_state.step_counter),
            ),
            lambda s: s,
            new_state.spawn_state,
        )
        return new_state.replace(
            death_counter=jnp.where(should_die, jnp.int32(90), new_state.death_counter),
            spawn_state=new_spawn,
        )

class ShootableDiversMod(JaxtariPostStepModPlugin):
    """Torpedoes destroy divers and award kill points (stock Seaquest ignores them).

    Used by ``swap_diver_shark_roles`` so DESTROY skills can clear lethal divers.
    Distinct from ``penalize_diver_shooting`` (removes but subtracts score).
    """

    conflicts_with = ["penalize_diver_shooting"]

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        missile_pos = new_state.player_missile_position
        missile_active = missile_pos[2] != 0
        kill_pts = self._env.calculate_kill_points(new_state.successful_rescues)

        missile_xy = missile_pos[:2]
        missile_size = self._env.consts.MISSILE_SIZE
        diver_size = self._env.consts.DIVER_SIZE

        def check_diver(i, carry):
            state, missile_gone = carry
            diver_pos = state.diver_positions[i]

            should_check = jnp.logical_and(diver_pos[2] != 0, jnp.logical_not(missile_gone))
            collision = self._env.check_collision_single(
                missile_xy,
                missile_size,
                jnp.array([diver_pos[0], diver_pos[1]]),
                diver_size,
            )
            hit = jnp.logical_and(should_check, collision)

            state = state.replace(
                diver_positions=state.diver_positions.at[i].set(
                    jnp.where(hit, jnp.zeros(3, dtype=diver_pos.dtype), diver_pos)
                ),
                score=jnp.where(hit, state.score + kill_pts, state.score),
                player_missile_position=jnp.where(
                    hit,
                    jnp.zeros(3, dtype=state.player_missile_position.dtype),
                    state.player_missile_position,
                ),
            )
            return state, jnp.logical_or(missile_gone, hit)

        return jax.lax.cond(
            missile_active,
            lambda s: jax.lax.fori_loop(0, 4, check_diver, (s, jnp.array(False)))[0],
            lambda s: s,
            new_state,
        )

class PeacefulSharksOnlyMod(JaxtariInternalModPlugin):
    """Internal helper: shark contact no longer kills (subs/missiles/surface still do)."""

    conflicts_with = ["peaceful_enemies"]

    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision(
        self,
        player_x,
        player_y,
        submarine_list,
        shark_list,
        surface_sub_pos,
        enemy_projectile_list,
        score,
        successful_rescues,
    ):
        del shark_list, score
        submarine_collisions = jnp.any(
            self._env.check_collision_batch(
                jnp.array([player_x, player_y]),
                self._env.consts.PLAYER_SIZE,
                submarine_list,
                self._env.consts.ENEMY_SUB_SIZE,
            )
        )
        surface_collision = self._env.check_collision_single(
            jnp.array([player_x, player_y]),
            self._env.consts.PLAYER_SIZE,
            surface_sub_pos,
            self._env.consts.ENEMY_SUB_SIZE,
        )
        missile_collisions = jnp.any(
            self._env.check_collision_batch(
                jnp.array([player_x, player_y]),
                self._env.consts.PLAYER_SIZE,
                enemy_projectile_list,
                self._env.consts.MISSILE_SIZE,
            )
        )
        collision_points = jnp.where(
            submarine_collisions,
            self._env.calculate_kill_points(successful_rescues),
            jnp.where(
                surface_collision,
                self._env.calculate_kill_points(successful_rescues),
                0,
            ),
        )
        died = jnp.any(
            jnp.array([submarine_collisions, missile_collisions, surface_collision])
        )
        return died, collision_points

class CollectSharksOnContactMod(JaxtariPostStepModPlugin):
    """Post-step helper: touching a shark fills a diver slot (reverse affordance)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        already_dying = new_state.death_counter > 0

        def _collect_one(i, state):
            shark = state.shark_positions[i]
            active = shark[2] != 0
            can_collect = state.divers_collected < 6
            hit = self._env.check_collision_single(
                jnp.array([state.player_x, state.player_y]),
                self._env.consts.PLAYER_SIZE,
                shark,
                self._env.consts.SHARK_SIZE,
            )
            should = jnp.logical_and(
                jnp.logical_and(jnp.logical_and(active, hit), can_collect),
                jnp.logical_not(already_dying),
            )
            return state.replace(
                shark_positions=state.shark_positions.at[i].set(
                    jnp.where(should, jnp.zeros_like(shark), shark)
                ),
                divers_collected=jnp.where(
                    should, state.divers_collected + 1, state.divers_collected
                ),
            )

        return jax.lax.fori_loop(0, new_state.shark_positions.shape[0], _collect_one, new_state)

class DisableDiverCollectMod(JaxtariInternalModPlugin):
    """Divers keep moving/spawning, but player contact no longer fills a rescue slot.

    Used by ``swap_diver_shark_roles`` so divers are threats only (see LethalDiversMod)
    while sharks become the collectable resource (CollectSharksOnContactMod).
    """

    conflicts_with = ["no_divers"]

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step_diver_movement(
        self,
        diver_positions: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        state_player_x: chex.Array,
        state_player_y: chex.Array,
        state_divers_collected: chex.Array,
        spawn_state: SpawnState,
        step_counter: chex.Array,
        rng: chex.PRNGKey,
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        # Force can_collect=False inside the base loop (collected < 6 gate), then
        # restore the real rescue count so shark-collection still surfaces.
        new_pos, _ignored_collected, new_spawn, new_rng = JaxSeaquest.step_diver_movement(
            self._env,
            diver_positions,
            shark_positions,
            sub_positions,
            state_player_x,
            state_player_y,
            jnp.int32(6),
            spawn_state,
            step_counter,
            rng,
        )
        return new_pos, state_divers_collected, new_spawn, new_rng

class NoSurfacingDeathMod(JaxtariInternalModPlugin):
    """Surfacing with 0 divers: no death, no O₂ refill, no surface lock.

    Vanilla refills whenever ``divers_collected >= 0`` (always true for an empty
    bag), so gating only on the single ``lose_life`` frame still let later
    surface frames top up oxygen. Keep withholding refill for the whole empty
    surface stay (non-init). Deposit tax / collisions / oxygen-out unchanged.
    """

    @partial(jax.jit, static_argnums=(0,))
    def update_oxygen(self, state, player_x, player_y, player_missile_position):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        in_x, in_y, in_missile = player_x, player_y, player_missile_position
        (
            new_oxygen,
            out_x,
            out_y,
            out_missile,
            oxygen_depleted,
            _lose_life,
            new_divers_collected,
            should_reset,
            new_just_surfaced,
            new_difficulty,
        ) = JaxSeaquest.update_oxygen(
            self._env, state, player_x, player_y, player_missile_position
        )
        in_init = state.just_surfaced == -1
        at_surface = jnp.logical_or(in_y == 46, out_y == 46)
        empty_surface = jnp.logical_and(
            jnp.logical_and(state.divers_collected == 0, at_surface),
            jnp.logical_not(in_init),
        )
        return (
            jnp.where(empty_surface, state.oxygen, new_oxygen),
            jnp.where(empty_surface, in_x, out_x),
            jnp.where(empty_surface, in_y, out_y),
            jnp.where(empty_surface, in_missile, out_missile),
            oxygen_depleted,
            jnp.array(False),
            jnp.maximum(new_divers_collected, jnp.int32(0)),
            should_reset,
            new_just_surfaced,
            new_difficulty,
        )


class UnshootableSharksMod(JaxtariInternalModPlugin):
    """Player torpedoes pass through sharks (subs still die). Used by role-swap."""

    conflicts_with = ["peaceful_enemies"]

    @partial(jax.jit, static_argnums=(0,))
    def check_missile_collisions(
        self,
        missile_pos,
        shark_positions,
        sub_positions,
        diver_positions,
        score,
        successful_rescues,
        spawn_state,
        rng_key,
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        (
            new_missile,
            _ignored_sharks,
            new_subs,
            new_divers,
            new_score,
            new_spawn_state,
            new_rng,
        ) = JaxSeaquest.check_missile_collisions(
            self._env,
            missile_pos,
            jnp.zeros_like(shark_positions),
            sub_positions,
            diver_positions,
            score,
            successful_rescues,
            spawn_state,
            rng_key,
        )
        return (
            new_missile,
            shark_positions,
            new_subs,
            new_divers,
            new_score,
            new_spawn_state,
            new_rng,
        )


# Full axis invert: left↔right and up↔down (plus diagonals / fire variants).
_LR_UD_SWAP = jnp.array(
    [
        0,   # NOOP
        1,   # FIRE
        5,   # UP -> DOWN
        4,   # RIGHT -> LEFT
        3,   # LEFT -> RIGHT
        2,   # DOWN -> UP
        9,   # UPRIGHT -> DOWNLEFT
        8,   # UPLEFT -> DOWNRIGHT
        7,   # DOWNRIGHT -> UPLEFT
        6,   # DOWNLEFT -> UPRIGHT
        13,  # UPFIRE -> DOWNFIRE
        12,  # RIGHTFIRE -> LEFTFIRE
        11,  # LEFTFIRE -> RIGHTFIRE
        10,  # DOWNFIRE -> UPFIRE
        17,  # UPRIGHTFIRE -> DOWNLEFTFIRE
        16,  # UPLEFTFIRE -> DOWNRIGHTFIRE
        15,  # DOWNRIGHTFIRE -> UPLEFTFIRE
        14,  # DOWNLEFTFIRE -> UPRIGHTFIRE
    ],
    dtype=jnp.int32,
)

def _action_has_horizontal(action: chex.Array) -> tuple[chex.Array, chex.Array]:
    from jaxtari.environment import JaxtariAction as Action

    left = jnp.any(
        jnp.array(
            [
                action == Action.LEFT,
                action == Action.UPLEFT,
                action == Action.DOWNLEFT,
                action == Action.LEFTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    right = jnp.any(
        jnp.array(
            [
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.DOWNRIGHT,
                action == Action.RIGHTFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.DOWNRIGHTFIRE,
            ]
        )
    )
    return left, right

class InvertedControlsMod(JaxtariInternalModPlugin):
    """Permute the action table: left ↔ right and up ↔ down."""

    conflicts_with = ["mirror_world", "momentum"]

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        remapped = _LR_UD_SWAP[action.astype(jnp.int32)]
        return JaxSeaquest.player_step(self._env, state, remapped)

class MomentumMod(JaxtariInternalModPlugin):
    """Player inertia: coast horizontally in facing direction when no L/R input."""

    conflicts_with = ["inverted_controls", "mirror_world"]

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        player_x, player_y, player_direction = JaxSeaquest.player_step(
            self._env, state, action
        )
        left, right = _action_has_horizontal(action)
        has_lr = jnp.logical_or(left, right)

        # Coast when idle horizontally; facing (player_direction) is ±1 once set.
        facing = jnp.where(state.player_direction == 0, jnp.int32(1), state.player_direction)
        coast_x = state.player_x + facing
        player_x = jnp.where(has_lr, player_x, coast_x)

        bounds = self._env.consts.PLAYER_BOUNDS
        player_x = jnp.clip(player_x, bounds[0, 0], bounds[0, 1])
        return player_x, player_y, player_direction

class SparseWorldMod(JaxtariPostStepModPlugin):
    """DEPRECATED — ego exclusion bubble; looks badly broken in play. Not registered."""

    MIN_DIST = 51

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        px = new_state.player_x.astype(jnp.float32)
        py = new_state.player_y.astype(jnp.float32)
        min_dist = jnp.float32(self.MIN_DIST)
        y_min = jnp.float32(46)
        y_max = jnp.float32(self._env.consts.PLAYER_BOUNDS[1, 1])

        def _push(pos):
            active = pos[2] != 0
            dx = pos[0].astype(jnp.float32) - px
            dy = pos[1].astype(jnp.float32) - py
            dist = jnp.sqrt(dx * dx + dy * dy)
            # If coincident, push along travel direction (or +x).
            fallback_dx = jnp.where(pos[2] == 0, jnp.float32(1.0), pos[2].astype(jnp.float32))
            dx = jnp.where(dist < 1e-3, fallback_dx, dx)
            dy = jnp.where(dist < 1e-3, jnp.float32(0.0), dy)
            dist = jnp.sqrt(dx * dx + dy * dy)
            scale = min_dist / jnp.maximum(dist, jnp.float32(1e-3))
            too_close = jnp.logical_and(active, dist < min_dist)
            new_x = jnp.where(too_close, (px + dx * scale).astype(pos.dtype), pos[0])
            new_y = jnp.where(
                too_close,
                jnp.clip((py + dy * scale), y_min, y_max).astype(pos.dtype),
                pos[1],
            )
            new_x = jnp.clip(new_x, jnp.int32(-8), jnp.int32(170)).astype(pos.dtype)
            return jnp.where(active, pos.at[0].set(new_x).at[1].set(new_y), pos)

        return new_state.replace(
            shark_positions=jax.vmap(_push)(new_state.shark_positions),
            sub_positions=jax.vmap(_push)(new_state.sub_positions),
            diver_positions=jax.vmap(_push)(new_state.diver_positions),
        )

class TinyDiversMod(JaxtariInternalModPlugin):
    """Shrink diver hitboxes and sprites.

    Sprite override lives in ``ASSET_CONFIG`` via ``asset_overrides``, so a
    native-downscale renderer rebuild (fresh bake from consts) keeps the tiny
    divers without any core ``apply_native_downscaling`` hook.
    """

    constants_overrides = {
        "DIVER_SIZE": jnp.array([4, 5], dtype=jnp.int32),
    }

    asset_overrides = {
        "diver": {
            "name": "diver",
            "type": "group",
            "files": ["tiny_diver_1.npy", "tiny_diver_2.npy"],
        }
    }

class DynamicLaneDriftMod(JaxtariInternalModPlugin):
    """Bucket B: lanes slowly oscillate vertically with per-lane phase.

    Unlike ``vertical_oscillation`` (fast shared wobble), each lane drifts on a
    slow sinusoid so absolute "safe bands" keep moving while relative geometry
    stays coherent for ego.

    Applied inside ``spawn_step`` / diver movement so collision uses the drifted
    Y (a post-step-only remap left lethal hitboxes on the stock lanes).
    """

    conflicts_with = [
        "vertical_oscillation",
        "continuous_random_spawns",
        "random_spawns",
        "lane_scramble",
        "shift_lanes",
    ]

    AMPLITUDE = 10.0
    FREQ = 0.035  # ~180 frames per cycle
    PHASES = (0.0, 1.7, 3.4, 5.1)

    @partial(jax.jit, static_argnums=(0,))
    def spawn_step(
        self,
        state,
        spawn_state,
        shark_positions,
        sub_positions,
        diver_positions,
        rng_key,
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        (
            new_spawn_state,
            new_sharks,
            new_subs,
            new_divers,
            new_key,
        ) = JaxSeaquest.spawn_step(
            self._env,
            state,
            spawn_state,
            shark_positions,
            sub_positions,
            diver_positions,
            rng_key,
        )
        t = state.step_counter.astype(jnp.float32)
        phases = jnp.array(self.PHASES, dtype=jnp.float32)
        lane_offsets = (self.AMPLITUDE * jnp.sin(t * self.FREQ + phases)).astype(
            jnp.int32
        )
        y_min = jnp.int32(46)
        y_max = jnp.int32(self._env.consts.PLAYER_BOUNDS[1, 1])

        def _drift_enemies(positions):
            reshaped = positions.reshape(4, 3, 3)

            def _lane(lane_i, lane_pos):
                off = lane_offsets[lane_i]

                def _one(pos):
                    active = pos[2] != 0
                    new_y = jnp.clip(pos[1] + off, y_min, y_max)
                    return jnp.where(active, pos.at[1].set(new_y), pos)

                return jax.vmap(_one)(lane_pos)

            return jax.vmap(_lane)(jnp.arange(4), reshaped).reshape(positions.shape)

        def _drift_divers(positions):
            def _one(lane_i, pos):
                active = pos[2] != 0
                new_y = jnp.clip(pos[1] + lane_offsets[lane_i], y_min, y_max)
                return jnp.where(active, pos.at[1].set(new_y), pos)

            return jax.vmap(_one)(jnp.arange(4), positions)

        return (
            new_spawn_state,
            _drift_enemies(new_sharks),
            _drift_enemies(new_subs),
            _drift_divers(new_divers),
            new_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_diver_movement(
        self,
        diver_positions,
        shark_positions,
        sub_positions,
        state_player_x,
        state_player_y,
        state_divers_collected,
        spawn_state,
        step_counter,
        rng,
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        final_pos, final_collected, new_spawn_state, new_rng = (
            JaxSeaquest.step_diver_movement(
                self._env,
                diver_positions,
                shark_positions,
                sub_positions,
                state_player_x,
                state_player_y,
                state_divers_collected,
                spawn_state,
                step_counter,
                rng,
            )
        )
        t = step_counter.astype(jnp.float32)
        phases = jnp.array(self.PHASES, dtype=jnp.float32)
        lane_offsets = (self.AMPLITUDE * jnp.sin(t * self.FREQ + phases)).astype(
            jnp.int32
        )
        y_min = jnp.int32(46)
        y_max = jnp.int32(self._env.consts.PLAYER_BOUNDS[1, 1])

        def _one(lane_i, pos):
            active = pos[2] != 0
            new_y = jnp.clip(pos[1] + lane_offsets[lane_i], y_min, y_max)
            return jnp.where(active, pos.at[1].set(new_y), pos)

        final_pos = jax.vmap(_one)(jnp.arange(4), final_pos)
        return final_pos, final_collected, new_spawn_state, new_rng

    @partial(jax.jit, static_argnums=(0,))
    def enemy_missiles_step(
        self,
        curr_sub_positions,
        curr_enemy_missile_positions,
        step_counter,
        difficulty,
    ):
        """Missiles follow drifted sub Y."""
        from jaxtari.games.jax_seaquest import JaxSeaquest

        lanes = self._env.consts.SPAWN_POSITIONS_Y.astype(jnp.int32)
        stock_missile_y = self._env.consts.ENEMY_MISSILE_Y.astype(jnp.int32)
        y_off = stock_missile_y - lanes
        all_lane_subs = curr_sub_positions.reshape(4, 3, 3)

        def lane_missile_y(lane_i, lane_subs):
            front = self._env.get_front_entity(0, lane_subs)
            return jnp.where(
                front[2] != 0,
                front[1] + y_off[lane_i],
                stock_missile_y[lane_i],
            )

        missile_ys = jax.vmap(lane_missile_y)(jnp.arange(4), all_lane_subs)
        stock = JaxSeaquest.enemy_missiles_step(
            self._env,
            curr_sub_positions,
            curr_enemy_missile_positions,
            step_counter,
            difficulty,
        )

        def _fix(missile, lane_y):
            active = missile[2] != 0
            return jnp.where(
                active, missile.at[1].set(lane_y.astype(missile.dtype)), missile
            )

        return jax.vmap(_fix)(stock, missile_ys)

class ContinuousRandomSpawnsMod(JaxtariInternalModPlugin):
    """Bucket B: rip the lane grid — random Y on spawn for sharks, subs, divers.

    Spawn still goes through Seaquest's lane slots (engine requirement), but on
    inactive→active we sample Y within the stock lane band and **keep that Y
    through movement + collision** (base ``step_enemy_movement`` would otherwise
    snap back to ``SPAWN_POSITIONS_Y`` every frame).

    Important: this cannot be a post-step-only remapping. Collision runs *inside*
    ``step`` before post-step mods, so a post-step Y rewrite left lethal hitboxes
    on the original lane while sprites floated elsewhere — remote phantom deaths
    when opposite-lane sharks visually crossed away from the player.
    """

    conflicts_with = [
        "dynamic_lane_drift",
        "shift_lanes",
        "vertical_oscillation",
        "lane_scramble",
    ]

    @staticmethod
    def _keep_y(prev_pos, new_pos):
        """Preserve Y across a move for slots that stay alive."""
        still_active = jnp.logical_and(prev_pos[2] != 0, new_pos[2] != 0)
        return jnp.where(still_active, new_pos.at[1].set(prev_pos[1]), new_pos)

    @staticmethod
    def _sample_spawn_y(prev_pos, new_pos, slot_rng, y_min, y_max):
        was_inactive = prev_pos[2] == 0
        now_active = new_pos[2] != 0
        just_spawned = jnp.logical_and(was_inactive, now_active)
        rand_y = jax.random.randint(slot_rng, (), y_min, y_max + 1)
        return jnp.where(
            just_spawned,
            new_pos.at[1].set(rand_y.astype(new_pos.dtype)),
            new_pos,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_enemy_movement(
        self, spawn_state, shark_positions, sub_positions, step_counter, rng
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        new_sharks, new_subs, new_spawn_state, new_rng = JaxSeaquest.step_enemy_movement(
            self._env, spawn_state, shark_positions, sub_positions, step_counter, rng
        )
        new_sharks = jax.vmap(self._keep_y)(shark_positions, new_sharks)
        new_subs = jax.vmap(self._keep_y)(sub_positions, new_subs)
        return new_sharks, new_subs, new_spawn_state, new_rng

    @partial(jax.jit, static_argnums=(0,))
    def spawn_step(
        self,
        state,
        spawn_state,
        shark_positions,
        sub_positions,
        diver_positions,
        rng_key,
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        (
            new_spawn_state,
            new_sharks,
            new_subs,
            new_divers,
            new_key,
        ) = JaxSeaquest.spawn_step(
            self._env,
            state,
            spawn_state,
            shark_positions,
            sub_positions,
            diver_positions,
            rng_key,
        )
        lanes = self._env.consts.SPAWN_POSITIONS_Y.astype(jnp.int32)
        y_min = lanes[0]
        y_max = lanes[-1]

        n_shark = new_sharks.shape[0]
        n_sub = new_subs.shape[0]
        n_diver = new_divers.shape[0]
        keys = jax.random.split(new_key, n_shark + n_sub + n_diver + 1)
        shark_keys = keys[:n_shark]
        sub_keys = keys[n_shark : n_shark + n_sub]
        diver_keys = keys[n_shark + n_sub : n_shark + n_sub + n_diver]
        new_key = keys[-1]

        new_sharks = jax.vmap(
            lambda p, n, k: self._sample_spawn_y(p, n, k, y_min, y_max)
        )(shark_positions, new_sharks, shark_keys)
        new_subs = jax.vmap(
            lambda p, n, k: self._sample_spawn_y(p, n, k, y_min, y_max)
        )(sub_positions, new_subs, sub_keys)
        new_divers = jax.vmap(
            lambda p, n, k: self._sample_spawn_y(p, n, k, y_min, y_max)
        )(diver_positions, new_divers, diver_keys)

        return new_spawn_state, new_sharks, new_subs, new_divers, new_key

    @partial(jax.jit, static_argnums=(0,))
    def step_diver_movement(
        self,
        diver_positions,
        shark_positions,
        sub_positions,
        state_player_x,
        state_player_y,
        state_divers_collected,
        spawn_state,
        step_counter,
        rng,
    ):
        from jaxtari.games.jax_seaquest import JaxSeaquest

        final_pos, final_collected, new_spawn_state, new_rng = (
            JaxSeaquest.step_diver_movement(
                self._env,
                diver_positions,
                shark_positions,
                sub_positions,
                state_player_x,
                state_player_y,
                state_divers_collected,
                spawn_state,
                step_counter,
                rng,
            )
        )
        final_pos = jax.vmap(self._keep_y)(diver_positions, final_pos)
        return final_pos, final_collected, new_spawn_state, new_rng

    @partial(jax.jit, static_argnums=(0,))
    def enemy_missiles_step(
        self,
        curr_sub_positions,
        curr_enemy_missile_positions,
        step_counter,
        difficulty,
    ):
        """Missiles track the live sub Y (not the stock lane ENEMY_MISSILE_Y)."""
        from jaxtari.games.jax_seaquest import JaxSeaquest

        # Stock missile Y is a fixed offset above SPAWN_POSITIONS_Y; keep that
        # offset relative to the (possibly random) sub Y.
        lanes = self._env.consts.SPAWN_POSITIONS_Y.astype(jnp.int32)
        stock_missile_y = self._env.consts.ENEMY_MISSILE_Y.astype(jnp.int32)
        y_off = stock_missile_y - lanes

        all_lane_subs = curr_sub_positions.reshape(4, 3, 3)

        def lane_missile_y(lane_i, lane_subs):
            front = self._env.get_front_entity(0, lane_subs)
            return jnp.where(
                front[2] != 0,
                front[1] + y_off[lane_i],
                stock_missile_y[lane_i],
            )

        missile_ys = jax.vmap(lane_missile_y)(jnp.arange(4), all_lane_subs)

        # Stock updater spawns/moves using lane Y; snap active missiles onto
        # the live sub-relative Y so hitboxes match sprites.
        stock = JaxSeaquest.enemy_missiles_step(
            self._env,
            curr_sub_positions,
            curr_enemy_missile_positions,
            step_counter,
            difficulty,
        )

        def _fix(missile, lane_y):
            active = missile[2] != 0
            return jnp.where(
                active, missile.at[1].set(lane_y.astype(missile.dtype)), missile
            )

        return jax.vmap(_fix)(stock, missile_ys)

