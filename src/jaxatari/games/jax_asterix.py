import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


def min_delay(level, base_min=30, spawn_accel=2, min_delay_clamp=20, max_delay_clamp=120):
    return jnp.clip(base_min - level * spawn_accel, min_delay_clamp, max_delay_clamp)


def max_delay(level, base_max=60, spawn_accel=2, min_delay_clamp=20, max_delay_clamp=120):
    return jnp.clip(base_max - level * spawn_accel, min_delay_clamp, max_delay_clamp)


class AsterixConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    player_width: int = 8
    player_height: int = 8
    num_stages: int = 8
    stage_spacing: int = 16 # ursprünglich 16
    stage_positions: List[int] = None
    top_border: int = 23 # oberer Rand des Spielfelds
    bottom_border: int = 8 * stage_spacing + top_border
    cooldown_frames: int = 8 # Cooldown frames for lane changes
    hit_frames: int = 100 # Anzahl Frames, die das Hit-Sprite angezeigt wird
    respawn_frames: int = 240 # Anzahl Frames, bis der Spieler nach einem hit respawned wird.
    num_lives: int = 3 # Anzahl der Leben
    max_digits_score: int = 6 # Maximal anzuzeigende Ziffern im Score
    entity_base_speed : float = 0.5 # Base Speed der Gegner und Collectibles
    entity_character_speed_factor : float = 0.7 # Speed-Faktor pro Charakterstufe (Asterix=0, Obelix=1)
    ASTERIX_ITEM_POINTS = jnp.array([50, 100, 200, 300, 0], dtype=jnp.int32)  # Cauldron, Helmet, Shield, Lamp
    OBELIX_ITEM_POINTS = jnp.array([400, 500, 500, 500, 500], dtype=jnp.int32)  # Apple, Fish, Wild Boar Leg, Mug, Cauldron

    stage_positions = [
        top_border, # TOP
        1 * stage_spacing + top_border,  # Stage 1
        2 * stage_spacing + top_border,  # Stage 2
        3 * stage_spacing + top_border,  # Stage 3
        4 * stage_spacing + top_border,  # Stage 4
        5 * stage_spacing + top_border,  # Stage 5
        6 * stage_spacing + top_border,  # Stage 6
        7 * stage_spacing + top_border,  # Stage 7
        8 * stage_spacing + top_border,  # BOTTOM
    ]

class CollectibleEnt(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    alive: jnp.ndarray
    type_index: jnp.ndarray

class Enemy(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    alive: jnp.ndarray

class ScorePopup(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    value: jnp.ndarray
    timer: jnp.ndarray
    active: jnp.ndarray


class AsterixState(NamedTuple):
    """Represents the current state of the game"""
    player_x: chex.Array # X-Position des Spielers
    player_y: chex.Array # Y-Position des Spielers
    score: chex.Array # Punktestand
    lives: chex.Array # Anzahl der Leben
    game_over: chex.Array # True, wenn keine Leben mehr übrig sind
    stage_cooldown: chex.Array # Cooldown für Lane-Wechsel
    bonus_life_stage: chex.Array # Stage für das nächste Bonusleben
    player_direction: chex.Array # 1 = left, 2 = right
    enemies: Enemy # Enemy Entities
    spawn_timer: jnp.ndarray # Timer für das Spawnen von Enemies
    rng: jax.random.PRNGKey # Random number generator state
    #wave_id: chex.Array
    character_id: chex.Array # 0 = Asterix, 1 = Obelix
    collect_type_index: chex.Array # Index im aktuellen Set
    collect_type_count: chex.Array # Anzahl eingesammelt vom aktuellen Typ (0..49)
    collectibles: CollectibleEnt # Collectible Entities
    collect_spawn_timer: jnp.ndarray # Timer für das Spawnen von Collectibles
    hit_timer: chex.Array # Zählt Frames herunter, in denen Hit-Sprite angezeigt wird
    respawn_timer: chex.Array # Zählt Frames herunter, bis Respawn nach Hit erfolgt
    score_popups: ScorePopup # Score Popups nach einsammeln eines Collectibles


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AsterixObservation(NamedTuple):
    player: EntityPosition


class AsterixInfo(NamedTuple):
    all_rewards: jnp.ndarray




class JaxAsterix(JaxEnvironment[AsterixState, AsterixObservation, AsterixInfo, AsterixConstants]):
    def __init__(self, consts: AsterixConstants = None, reward_funcs: list[callable] = None):
        if consts is None:
            consts = AsterixConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.state = self.reset()
        self.renderer = AsterixRenderer()
        self.obs_type = "rgb"  # "rgb", "ram", "grayscale"


    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[AsterixObservation, AsterixState]:
        """Initialize a new game state"""
        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        player_x = self.consts.screen_width // 2
        player_y = (stage_borders[3] + stage_borders[4]) // 2 - (self.consts.player_height // 2)

        if key is None:
            key = jax.random.PRNGKey(0)
        max_enemies = 32
        max_collectibles = 32
        spawn_rng, timer_rng, state_rng = jax.random.split(key, 3)
        spawn_timer = jax.random.randint(timer_rng, (), min_delay(1), max_delay(1) + 1)
        enemies = Enemy(
            x=jnp.full((max_enemies,), -9999.0),
            y=jnp.full((max_enemies,), -9999.0),
            vx=jnp.zeros((max_enemies,)),
            alive=jnp.zeros((max_enemies,), dtype=bool)
        )
        collectibles = CollectibleEnt(
            x=jnp.full((max_collectibles,), -9999.0),
            y=jnp.full((max_collectibles,), -9999.0),
            vx=jnp.zeros((max_collectibles,)),
            alive=jnp.zeros((max_collectibles,), dtype=bool),
            type_index = jnp.zeros((max_collectibles,), dtype=jnp.int32)

        )
        collect_spawn_timer = jax.random.randint(timer_rng, (), min_delay(1), max_delay(1) + 1)

        score_popups = ScorePopup(
            x=jnp.full((max_collectibles,), -9999.0),
            y=jnp.full((max_collectibles,), -9999.0),
            value=jnp.zeros((max_collectibles,), dtype=jnp.int32),
            timer=jnp.zeros((max_collectibles,), dtype=jnp.int32),
            active=jnp.zeros((max_collectibles,), dtype=bool)
        )

        state = AsterixState(
            player_x =jnp.array(player_x, dtype=jnp.int32),
            player_y=jnp.array(player_y, dtype=jnp.int32),
            score=jnp.array(29900, dtype=jnp.int32), # Start with 0 points
            lives=jnp.array(self.consts.num_lives, dtype=jnp.int32),  # 3 Leben
            game_over=jnp.array(False, dtype=jnp.bool_),
            stage_cooldown = jnp.array(self.consts.cooldown_frames, dtype=jnp.int32), # Cooldown initial 0
            bonus_life_stage=jnp.array(0, dtype=jnp.int32),  # Stage for bonus life
            player_direction=jnp.array(1, dtype=jnp.int32),  # Initial direction (1=links)
            enemies=enemies,
            spawn_timer=spawn_timer,
            rng=state_rng,
            # wave_id = jnp.array(0, dtype=jnp.int32),
            character_id=jnp.array(0, dtype=jnp.int32),  # Asterix
            collect_type_index=jnp.array(0, dtype=jnp.int32),  # erster collectable Typ
            collect_type_count=jnp.array(0, dtype=jnp.int32),
            collectibles=collectibles,
            collect_spawn_timer=collect_spawn_timer,
            hit_timer=jnp.array(0, dtype=jnp.int32),
            respawn_timer = jnp.array(0, dtype=jnp.int32),
            score_popups = score_popups
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AsterixState, action: int) -> tuple[
        AsterixObservation, AsterixState, float, bool, AsterixInfo]:
        player_height = self.consts.player_height
        cooldown_frames = self.consts.cooldown_frames
        can_switch_stage = state.stage_cooldown <= 0

        stage_borders = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        num_stage = stage_borders.shape[0]

        # Aktuelle Stage bestimmen
        stage_diffs = jnp.abs(stage_borders - state.player_y)
        current_stage = jnp.argmin(stage_diffs)

        # Action mapping
        action = jnp.asarray(action, dtype=jnp.int32)
        remap = jnp.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        mapped = remap[action]
        dx_table = jnp.array([0, 0, 1, -1, 0, 1, -1, 1, -1], dtype=jnp.int32)
        dy_table = jnp.array([0, -1, 0, 0, 1, -1, -1, 1, 1], dtype=jnp.int32)
        dx = dx_table[mapped]
        dy = dy_table[mapped]
        action = mapped

        # Pause-Status
        paused = state.respawn_timer > 0

        # Lane-Wechsel nur wenn nicht pausiert
        stage_move = jnp.where(dy < 0, -1, jnp.where(dy > 0, 1, 0))
        tentative_stage = current_stage + jnp.where(~paused & can_switch_stage, stage_move, 0)
        new_stage_idx = jnp.clip(tentative_stage, 0, num_stage - 2)
        computed_y = ((stage_borders[new_stage_idx] + stage_borders[new_stage_idx + 1]) // 2) - (player_height // 2)

        # Seitliche Begrenzung
        stage_left_x = (self.consts.screen_width - self.renderer.sprites['STAGE'][0].shape[1]) // 2
        stage_right_x = stage_left_x + self.renderer.sprites['STAGE'][0].shape[1]
        computed_player_x = jnp.clip(
            state.player_x + jnp.where(paused, 0, dx),
            stage_left_x,
            stage_right_x - self.consts.player_width,
        ).astype(jnp.int32)

        # Blickrichtung
        computed_direction = jnp.where(
            dx < 0, 1,
            jnp.where(dx > 0, 2, state.player_direction)
        )

        changed_stage = (stage_move != 0) & can_switch_stage & (~paused)
        computed_cooldown = jnp.where(
            changed_stage,
            cooldown_frames,
            jnp.maximum(state.stage_cooldown - jnp.where(paused, 0, 1), 0)
        )

        new_player_x = jnp.where(paused, state.player_x, computed_player_x)
        new_y = jnp.where(paused, state.player_y, computed_y)
        new_player_direction = jnp.where(paused, state.player_direction, computed_direction)
        new_cooldown = jnp.where(paused, state.stage_cooldown, computed_cooldown)

        # --- Spawns / Bewegung von Gegnern & Collectibles ---
        platformY = jnp.array(self.consts.stage_positions, dtype=jnp.int32)
        item_w = 8
        item_h = 8
        enemy_width = 8
        enemy_w = 8
        enemy_h = 8
        screen_width = self.consts.screen_width
        level = 1

        rng_enemy_spawn, rng_enemy_delay, rng_col_spawn, rng_col_delay, rng_next = jax.random.split(state.rng, 5)

        spawn_timer = jnp.where(paused, state.spawn_timer, state.spawn_timer - 1)
        collect_spawn_timer = jnp.where(paused, state.collect_spawn_timer, state.collect_spawn_timer - 1)

        def spawn_enemy(rng, level, platformY, screen_width, enemy_width, lyre_height=8):
            rng_side, rng_platform = jax.random.split(rng)
            num_platforms = len(platformY) - 1
            platform = jax.random.randint(rng_platform, (), 0, num_platforms)
            y_center = (platformY[platform] + platformY[platform + 1]) // 2
            y = y_center - (lyre_height // 2)
            x = jax.lax.select(jax.random.bernoulli(rng_side), screen_width + enemy_width, -enemy_width)
            speed = self.consts.entity_base_speed + state.character_id * self.consts.entity_character_speed_factor
            vx = speed * jax.lax.select(x > 0, -1.0, 1.0)
            return Enemy(x, y, vx, True)

        def spawn_collectible(rng, level, platformY, screen_width, item_width):
            rng_side, rng_platform = jax.random.split(rng)
            num_platforms = len(platformY) - 1
            platform = jax.random.randint(rng_platform, (), 0, num_platforms)
            y_center = (platformY[platform] + platformY[platform + 1]) // 2
            y = y_center - (item_width // 2)
            x = jax.lax.select(jax.random.bernoulli(rng_side), screen_width + item_width, -item_width)
            speed = self.consts.entity_base_speed + state.character_id * self.consts.entity_character_speed_factor
            vx = speed * jax.lax.select(x > 0, -1.0, 1.0)
            return CollectibleEnt(x, y, vx, True, jnp.int32(0))

        def spawn_fn(args):
            enemies, collectibles, rng_enemy_spawn, level = args
            new_enemy = spawn_enemy(rng_enemy_spawn, level, platformY, screen_width, enemy_width)
            occupied = jnp.any(((enemies.y == new_enemy.y) & enemies.alive) |
                               ((collectibles.y == new_enemy.y) & collectibles.alive))

            def do_spawn():
                idx = jnp.argmax(~enemies.alive)
                return enemies._replace(
                    x=enemies.x.at[idx].set(new_enemy.x),
                    y=enemies.y.at[idx].set(new_enemy.y),
                    vx=enemies.vx.at[idx].set(new_enemy.vx),
                    alive=enemies.alive.at[idx].set(True)
                )

            return jax.lax.cond(occupied, lambda: enemies, do_spawn)

        def spawn_collectibles_fn(args):
            enemies, collectibles, rng_col, level = args
            new_item = spawn_collectible(rng_col, level, platformY, self.consts.screen_width, item_w)
            occupied = jnp.any(((collectibles.y == new_item.y) & collectibles.alive) |
                               ((enemies.y == new_item.y) & enemies.alive))

            def do_spawn():
                idx = jnp.argmax(~collectibles.alive)
                return collectibles._replace(
                    x=collectibles.x.at[idx].set(new_item.x),
                    y=collectibles.y.at[idx].set(new_item.y),
                    vx=collectibles.vx.at[idx].set(new_item.vx),
                    alive=collectibles.alive.at[idx].set(True),
                    type_index=collectibles.type_index.at[idx].set(state.collect_type_index)
                )

            return jax.lax.cond(occupied, lambda: collectibles, do_spawn)

        should_spawn = (~paused) & (spawn_timer <= 0)
        should_spawn_col = (~paused) & (collect_spawn_timer <= 0)

        enemies = jax.lax.cond(
            should_spawn,
            spawn_fn,
            lambda args: args[0],
            (state.enemies, state.collectibles, rng_enemy_spawn, level)
        )

        collectibles = jax.lax.cond(
            should_spawn_col,
            spawn_collectibles_fn,
            lambda args: args[1],
            (state.enemies, state.collectibles, rng_col_spawn, level)
        )

        def new_timer_fn(_):
            minD = min_delay(level)
            maxD = max_delay(level)
            return jax.random.randint(rng_enemy_delay, (), minD, maxD + 1)

        spawn_timer = jax.lax.cond(
            should_spawn,
            new_timer_fn,
            lambda _: spawn_timer,
            operand=None
        )

        def new_collect_timer_fn(_):
            minD = min_delay(level)
            maxD = max_delay(level)
            return jax.random.randint(rng_col_delay, (), minD, maxD + 1)

        collect_spawn_timer = jax.lax.cond(
            should_spawn_col,
            new_collect_timer_fn,
            lambda _: collect_spawn_timer,
            operand=None
        )

        # Bewegung/Bounds nur wenn nicht pausiert
        new_enemy_x = jnp.where(paused, enemies.x, enemies.x + enemies.vx)
        alive_enemies = jnp.where(
            paused,
            enemies.alive,
            (new_enemy_x >= -enemy_width) & (new_enemy_x <= screen_width + enemy_width) & enemies.alive
        )
        enemies = enemies._replace(x=new_enemy_x, alive=alive_enemies)

        new_item_x = jnp.where(paused, collectibles.x, collectibles.x + collectibles.vx)
        alive_items = jnp.where(
            paused,
            collectibles.alive,
            (new_item_x >= -item_w) & (new_item_x <= self.consts.screen_width + item_w) & collectibles.alive
        )
        collectibles = collectibles._replace(x=new_item_x, alive=alive_items)

        # --- Kollisionen ---
        def check_collision(px, py, pw, ph, ex, ey, ew, eh):
            return ((px < ex + ew) & (px + pw > ex) & (py < ey + eh) & (py + ph > ey))

        collisions_enemy = jnp.where(
            paused,
            False,
            check_collision(new_player_x, new_y, self.consts.player_width, self.consts.player_height,
                            enemies.x, enemies.y, enemy_w, enemy_h) & enemies.alive
        )
        any_collision_enemy = jnp.where(paused, False, jnp.any(collisions_enemy))
        enemies = enemies._replace(alive=jnp.where(collisions_enemy, False, enemies.alive))

        collisions_item = jnp.where(
            paused,
            False,
            check_collision(new_player_x, new_y,
                            self.consts.player_width, self.consts.player_height,
                            collectibles.x, collectibles.y, item_w, item_h) & collectibles.alive
        )
        hit_items_count = jnp.where(paused, jnp.int32(0), jnp.sum(collisions_item).astype(jnp.int32))
        collectibles = collectibles._replace(alive=jnp.where(collisions_item & (~paused), False, collectibles.alive))

        # Punktevergabe (nur wenn nicht pausiert)
        char_id = state.character_id
        start_type_idx = state.collect_type_index
        start_type_count = state.collect_type_count
        types_count = jnp.where(char_id == 0, jnp.int32(4), jnp.int32(5))
        points_array = jnp.where(char_id == 0, self.consts.ASTERIX_ITEM_POINTS, self.consts.OBELIX_ITEM_POINTS)

        def per_item_body(i, carry):
            total_points, type_idx, in_type_count = carry
            total_points = total_points + points_array[type_idx]
            in_type_count = in_type_count + 1
            reached = (in_type_count == 50)
            type_idx = jnp.where(reached, (type_idx + 1) % types_count, type_idx)
            in_type_count = jnp.where(reached, 0, in_type_count)
            return total_points, type_idx, in_type_count

        init = (jnp.int32(0), start_type_idx, start_type_count)
        total_points, end_type_idx, end_type_count = jax.lax.fori_loop(0, hit_items_count, per_item_body, init)
        computed_score = state.score + total_points
        new_score = jnp.where(paused, state.score, computed_score)

        # Score-Popups
        def spawn_score_popup(score_popups, collisions_item, collectibles, points_array):
            def body(i, popup):
                should_spawn = collisions_item[i]
                idx = jnp.argmax(~popup.active)
                value = points_array[collectibles.type_index[i]]  # <-- Typ pro Collectible
                popup = popup._replace(
                    x=popup.x.at[idx].set(jnp.where(should_spawn, collectibles.x[i], popup.x[idx])),
                    y=popup.y.at[idx].set(jnp.where(should_spawn, collectibles.y[i], popup.y[idx])),
                    value=popup.value.at[idx].set(jnp.where(should_spawn, value, popup.value[idx])),
                    timer=popup.timer.at[idx].set(jnp.where(should_spawn, 480, popup.timer[idx])),
                    active=popup.active.at[idx].set(jnp.where(should_spawn, True, popup.active[idx]))
                )
                return popup

            popup = score_popups
            popup = jax.lax.fori_loop(0, collectibles.x.shape[0], body, popup)
            return popup

        score_popups = spawn_score_popup(state.score_popups, collisions_item, collectibles, points_array)

        # Charakterwechsel
        switch_to_obelix = (state.character_id == 0) & (new_score >= jnp.int32(32500)) & (
                    state.score < jnp.int32(32500))
        new_character_id = jnp.where(switch_to_obelix, jnp.int32(1), state.character_id)
        new_collect_type_index = jnp.where(paused, state.collect_type_index, jnp.where(switch_to_obelix, jnp.int32(0), end_type_idx))
        new_collect_type_count = jnp.where(paused, state.collect_type_count, jnp.where(switch_to_obelix, jnp.int32(0), end_type_count))

        # Bonus/Leben
        bonus_thresholds = jnp.array([10_000, 30_000, 50_000, 80_000, 110_000], dtype=jnp.int32)
        bonus_interval = 40_000

        def calc_bonus_stage(score):
            below = jnp.sum(score >= bonus_thresholds)
            above = jnp.maximum(score - 110_000, 0) // bonus_interval
            return below + above

        new_bonus_stage = jnp.where(paused, state.bonus_life_stage, calc_bonus_stage(new_score))
        bonus_lives_gained = jnp.maximum(new_bonus_stage - state.bonus_life_stage, 0)
        new_lives = jnp.where(any_collision_enemy, state.lives - 1, state.lives + bonus_lives_gained)

        # Respawn-/Hit-Timer
        rt_prev = state.respawn_timer
        rt_after_decr = jnp.maximum(rt_prev - 1, 0)
        new_respawn_timer = jnp.where(any_collision_enemy & (~paused),
                                      jnp.int32(self.consts.respawn_frames),
                                      rt_after_decr)
        new_hit_timer = jnp.where(
            any_collision_enemy & (~paused),
            jnp.array(self.consts.respawn_frames, dtype=jnp.int32),
            jnp.maximum(state.hit_timer - 1, 0),
        )

        # Wenn Respawn-Timer gerade abgelaufen ist: alles löschen und Spieler auf Stage 4 setzen
        just_finished_respawn = (rt_prev > 0) & (new_respawn_timer == 0)

        # Stage 4 Mitte berechnen (zwischen Border 3 und 4)
        target_stage_idx = 3
        respawn_y = ((stage_borders[target_stage_idx] + stage_borders[target_stage_idx + 1]) // 2) - (
                    player_height // 2)
        respawn_x = self.consts.screen_width // 2

        enemies = jax.lax.cond(
            just_finished_respawn,
            lambda _: enemies._replace(
                x=jnp.full_like(enemies.x, -9999.0),
                y=jnp.full_like(enemies.y, -9999.0),
                vx=jnp.zeros_like(enemies.vx),
                alive=jnp.zeros_like(enemies.alive)
            ),
            lambda _: enemies,
            operand=None
        )
        collectibles = jax.lax.cond(
            just_finished_respawn,
            lambda _: collectibles._replace(
                x=jnp.full_like(collectibles.x, -9999.0),
                y=jnp.full_like(collectibles.y, -9999.0),
                vx=jnp.zeros_like(collectibles.vx),
                alive=jnp.zeros_like(collectibles.alive)
            ),
            lambda _: collectibles,
            operand=None
        )

        # Player neu setzen und Cooldown + Spawn-Timer neu würfeln
        new_player_x = jnp.where(just_finished_respawn, jnp.int32(respawn_x), new_player_x)
        new_y = jnp.where(just_finished_respawn, jnp.int32(respawn_y), new_y)
        new_cooldown = jnp.where(just_finished_respawn, jnp.int32(self.consts.cooldown_frames), new_cooldown)

        spawn_timer = jax.lax.cond(just_finished_respawn, new_timer_fn, lambda _: spawn_timer, operand=None)
        collect_spawn_timer = jax.lax.cond(just_finished_respawn, new_collect_timer_fn, lambda _: collect_spawn_timer, operand=None)

        # Timer runterzählen und Popups deaktivieren, wenn timer==0
        def _update_popups(_):
            new_timer = jnp.maximum(score_popups.timer - 1, 0)
            new_active = score_popups.active & (new_timer > 0)
            return score_popups._replace(timer=new_timer, active=new_active)

        def _clear_popups(_):
            return score_popups._replace(
                x=jnp.full_like(score_popups.x, -9999.0),
                y=jnp.full_like(score_popups.y, -9999.0),
                value=jnp.zeros_like(score_popups.value),
                timer=jnp.zeros_like(score_popups.timer),
                active=jnp.zeros_like(score_popups.active),
            )

        score_popups = jax.lax.cond(just_finished_respawn, _clear_popups, _update_popups, operand=None)

        game_over = jnp.where(new_lives <= 0, jnp.array(True), state.game_over)

        new_state = AsterixState(
            player_x=new_player_x,
            player_y=new_y,
            lives=new_lives,
            score=new_score,
            game_over=game_over,
            stage_cooldown=new_cooldown,
            bonus_life_stage=new_bonus_stage,
            player_direction=new_player_direction,
            enemies=enemies,
            spawn_timer=spawn_timer,
            rng=rng_next,
            character_id=new_character_id,
            collect_type_index=new_collect_type_index,
            collect_type_count=new_collect_type_count,
            collectibles=collectibles,
            collect_spawn_timer=collect_spawn_timer,
            hit_timer=new_hit_timer,
            respawn_timer=new_respawn_timer,
            score_popups=score_popups,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, all_rewards)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AsterixState):
        # create player
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.player_width, dtype=jnp.int32),
            height=jnp.array(self.consts.player_height, dtype=jnp.int32),
        )

        return AsterixObservation(player=player)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AsterixState, all_rewards: chex.Array = None) -> AsterixInfo:
        return AsterixInfo(all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsterixState, state: AsterixState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: AsterixState, state: AsterixState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AsterixState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Asterix.
        Actions are:
        0: NOOP
        1: UP
        2: RIGHTS
        3: LEFT
        4: DOWN
        5: UPRIGHT
        6: UPLEFT
        7: DOWNRIGHT
        8: DOWNLEFT
        """
        return spaces.Discrete(9)

    def observation_space(self) -> spaces.Dict: # TODO kann entfernt werden? wird nicht verwendet / benötigt
        #Returns the observation space for Asterix.
        if self.obs_type == "rgb":
            return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)
        elif self.obs_type == "ram":
            return spaces.Box(low=0, high=255, shape=(128,), dtype=jnp.uint8)
        elif self.obs_type == "grayscale":
            return spaces.Box(low=0, high=255, shape=(210, 160), dtype=jnp.uint8)
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    """
        def observation_space(self) -> spaces.Dict: # TODO kann entfernt werden? wird nicht verwendet / benötigt
        #Returns the observation space for Asterix.
        #The observation contains:
        #- player: EntityPosition (x, y, width, height)
        #- score: int (0-99)
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            #"car": spaces.Box(low=0, high=160, shape=(10, 4), dtype=jnp.int32),
            #"score": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
        })
    """

    def image_space(self) -> spaces.Box: # TODO kann entfernt werden? wird nicht verwendet / benötigt
        """Returns the image space for Asterix.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def render(self, state: AsterixState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: AsterixObservation) -> jnp.ndarray: # TODO kann entfernt werden? wird nicht verwendet / benötigt
        """Convert the observation to a flat array."""
        return jnp.array([
            obs.player.x,
            obs.player.y,
            obs.player.width,
            obs.player.height,
        ], dtype=jnp.int32)

class AsterixRenderer(JAXGameRenderer):
    def __init__(self, consts: AsterixConstants = None):
        super().__init__()
        self.consts = consts or AsterixConstants()
        self.sprites, self.offsets = self._load_sprites()


    def _load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(MODULE_DIR, "sprites/asterix/")

        sprites: Dict[str, Any] = {}
        offsets: Dict[str, Any] = {}

        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            return frame.astype(jnp.uint8)

        sprite_names = [
            'ASTERIX_LEFT', 'ASTERIX_RIGHT', 'ASTERIX_LEFT_HIT', 'ASTERIX_RIGHT_HIT',
            'STAGE', 'TOP', 'BOTTOM', 'LYRE_LEFT', 'LYRE_RIGHT',
            'OBELIX_LEFT', 'OBELIX_RIGHT', 'OBELIX_LEFT_HIT',
        ]
        asterix_item_names = ['CAULDRON', 'HELMET', 'SHIELD', 'LAMP']
        obelix_item_names = ['APPLE', 'FISH', 'WILD_BOAR_LEG', 'MUG', 'CAULDRON']
        blank8 = jnp.zeros((8, 8, 3), dtype=jnp.uint8)

        for name in sprite_names + asterix_item_names + obelix_item_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                if loaded_sprite.ndim == 2:
                    loaded_sprite = loaded_sprite[..., None]
                    loaded_sprite = jnp.repeat(loaded_sprite, 3, axis=2)
                sprites[name] = loaded_sprite

        for name in asterix_item_names + obelix_item_names:
            if name not in sprites:
                sprites[name] = blank8

        # Player-Sprites padden und als Array speichern
        asterix_sprites_list = [
            sprites['ASTERIX_LEFT'], sprites['ASTERIX_RIGHT'],
            sprites['ASTERIX_LEFT_HIT'], sprites['ASTERIX_RIGHT_HIT']
        ]
        obelix_sprites_list = [
            sprites['OBELIX_LEFT'], sprites['OBELIX_RIGHT'],
            sprites['OBELIX_LEFT_HIT'], sprites['OBELIX_LEFT']
        ]

        # Gemeinsames Padding für beide Sprite-Listen
        all_player_sprites = asterix_sprites_list + obelix_sprites_list
        all_padded, _ = jr.pad_to_match(all_player_sprites)

        # Aufteilen in die jeweiligen Gruppen
        sprites['ASTERIX_SPRITES'] = all_padded[:4]
        sprites['OBELIX_SPRITES'] = all_padded[4:]

        # Digit-Sprites laden
        digit_path = os.path.join(sprite_path, 'DIGIT_{}.npy')
        digits = jr.load_and_pad_digits(digit_path, num_chars=10)
        # Fälle abdecken: (10,H,W) -> (10,H,W,3); (10,H,W,1) -> (10,H,W,3)
        if digits.ndim == 3:  # (10, H, W)
            digits = digits[..., None]
        if digits.shape[-1] == 1:  # (10, H, W, 1)
            digits = jnp.repeat(digits, 3, axis=3)
        sprites['digit'] = digits

        # Punktesprites laden und padden
        points_names = ['POINTS50', 'POINTS100', 'POINTS200', 'POINTS300', 'POINTS400', 'POINTS500']
        point_frames = []
        for name in points_names:
            frame = _load_sprite_frame(name)
            if frame is None:
                frame = point_frames[0] if point_frames else jnp.zeros((8, 8, 4), dtype=jnp.uint8)
            point_frames.append(frame)
        point_frames_padded, _ = jr.pad_to_match(point_frames)
        for name, frame in zip(points_names, point_frames_padded):
            sprites[name] = jnp.expand_dims(frame, axis=0)

        # Alle Sprites auf 4D bringen (Batch-Dim)
        import numpy as np
        for key in sprites.keys():
            value = sprites[key]
            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.ndim == 3:
                sprites[key] = jnp.expand_dims(value, axis=0)

        return sprites, offsets

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """Render the game state to a raster image."""
        # ----------- RASTER INITIALIZATION -------------
        raster = jnp.zeros((self.consts.screen_height, self.consts.screen_width, 3), dtype=jnp.uint8)

        # ----------- STAGE -------------
        stage_sprite = jr.get_sprite_frame(self.sprites['STAGE'], 0)
        stage_height = stage_sprite.shape[0]
        stage_x = (self.consts.screen_width - stage_sprite.shape[1]) // 2 # Center the stage horizontally

        for stage_y in self.consts.stage_positions:
            # oberste und unterste stage nicht rendern
            if stage_y == self.consts.stage_positions[0] or stage_y == self.consts.stage_positions[-1]:
                continue
            raster = jr.render_at(
                raster,
                stage_x,
                stage_y,  # Y-Position: Lane-Grenze
                stage_sprite
            )

        # ----------- TOP AND BOTTOM -------------
        top_sprite = jr.get_sprite_frame(self.sprites['TOP'], 0)
        bottom_sprite = jr.get_sprite_frame(self.sprites['BOTTOM'], 0)
        top_x = (self.consts.screen_width - top_sprite.shape[1]) // 2  # Center the top sprite horizontally
        # top_y = top_sprite.shape[0] // 2
        top_y = self.consts.top_border - self.consts.stage_spacing + stage_height
        bottom_x = (self.consts.screen_width - bottom_sprite.shape[1]) // 2  # Center the bottom sprite horizontally
        bottom_y = self.consts.stage_positions[-1]
        raster = jr.render_at(
            raster,
            top_x,
            top_y,
            top_sprite
        )
        raster = jr.render_at(
            raster,
            bottom_x,
            bottom_y,
            bottom_sprite
        )


        # ----------- LYRES -------------
        lyre_left_sprite = jr.get_sprite_frame(self.sprites['LYRE_LEFT'], 0)
        lyre_right_sprite = jr.get_sprite_frame(self.sprites['LYRE_RIGHT'], 0)

        # ----------- LYRES (Feinde) -------------
        def render_lyres(raster_to_update):
            def render_single_lyre(i, raster_inner):
                is_alive = state.enemies.alive[i]
                x = state.enemies.x[i]
                y = state.enemies.y[i]
                vx = state.enemies.vx[i]
                lyre_sprite = jax.lax.select(
                    vx < 0,
                    lyre_left_sprite,
                    lyre_right_sprite
                )
                # Nur rendern, wenn alive
                raster_inner = jax.lax.cond(
                    is_alive,
                    lambda r: jr.render_at(r, x.astype(jnp.int32), y.astype(jnp.int32), lyre_sprite),
                    lambda r: r,
                    raster_inner
                )
                return raster_inner

            raster_out = raster_to_update
            for i in range(state.enemies.x.shape[0]):
                raster_out = render_single_lyre(i, raster_out)
            return raster_out

        raster = render_lyres(raster)

        # ----------- COLLECTIBLES -------------
        asterix_item_names = ['CAULDRON', 'HELMET', 'SHIELD', 'LAMP']
        obelix_item_names = ['APPLE', 'FISH', 'WILD_BOAR_LEG', 'MUG', 'CAULDRON']

        ax_sprites = [jr.get_sprite_frame(self.sprites[name], 0) for name in asterix_item_names]
        ob_sprites = [jr.get_sprite_frame(self.sprites[name], 0) for name in obelix_item_names]


        def render_collectibles(raster_to_update):
            def render_one(i, r_in):
                is_alive = state.collectibles.alive[i]
                x = state.collectibles.x[i].astype(jnp.int32)
                y = state.collectibles.y[i].astype(jnp.int32)
                idx_item = state.collectibles.type_index[i]  # per-Item-Typ verwenden

                def render_for_char(_):
                    # Asterix: 4 Typen (0..3)
                    def render_ax(_2):
                        idx_ax = jnp.minimum(idx_item, jnp.int32(3))
                        return jax.lax.switch(
                            idx_ax,
                            [
                                lambda __: jr.render_at(r_in, x, y, ax_sprites[0]),
                                lambda __: jr.render_at(r_in, x, y, ax_sprites[1]),
                                lambda __: jr.render_at(r_in, x, y, ax_sprites[2]),
                                lambda __: jr.render_at(r_in, x, y, ax_sprites[3]),
                            ],
                            operand=None
                        )

                    # Obelix: 5 Typen (0..4)
                    def render_ob(_2):
                        idx_ob = jnp.minimum(idx_item, jnp.int32(4))
                        return jax.lax.switch(
                            idx_ob,
                            [
                                lambda __: jr.render_at(r_in, x, y, ob_sprites[0]),
                                lambda __: jr.render_at(r_in, x, y, ob_sprites[1]),
                                lambda __: jr.render_at(r_in, x, y, ob_sprites[2]),
                                lambda __: jr.render_at(r_in, x, y, ob_sprites[3]),
                                lambda __: jr.render_at(r_in, x, y, ob_sprites[4]),
                            ],
                            operand=None
                        )

                    return jax.lax.switch(state.character_id, [render_ax, render_ob], operand=None)

                return jax.lax.cond(is_alive, render_for_char, lambda _: r_in, operand=None)

            r_out = raster_to_update
            for i in range(state.collectibles.x.shape[0]):
                r_out = render_one(i, r_out)
            return r_out

        raster = render_collectibles(raster)

        # ----------- SCORE POPUPS -------------
        def render_score_popups(raster_to_update):
            def body(i, r_in):
                def render_active(r):
                    value = state.score_popups.value[i]  # int32 Tracer

                    # Wert -> Index [0..5] (inkl. 500)
                    idx = jnp.where(value == 50, 0,
                          jnp.where(value == 100, 1,
                          jnp.where(value == 200, 2,
                          jnp.where(value == 300, 3,
                          jnp.where(value == 400, 4,
                          jnp.where(value == 500, 5, jnp.int32(0)))))))

                    # Sprite per jax.lax.switch auswählen (alle Frames gleich gepaddet)
                    sprite = jax.lax.switch(
                        idx,
                        [
                            lambda _: jr.get_sprite_frame(self.sprites['POINTS50'], 0),
                            lambda _: jr.get_sprite_frame(self.sprites['POINTS100'], 0),
                            lambda _: jr.get_sprite_frame(self.sprites['POINTS200'], 0),
                            lambda _: jr.get_sprite_frame(self.sprites['POINTS300'], 0),
                            lambda _: jr.get_sprite_frame(self.sprites['POINTS400'], 0),
                            lambda _: jr.get_sprite_frame(self.sprites['POINTS500'], 0),
                        ],
                        operand=None
                    )

                    x = state.score_popups.x[i].astype(jnp.int32)
                    y = state.score_popups.y[i].astype(jnp.int32)
                    return jr.render_at(r, x, y, sprite)

                return jax.lax.cond(
                    state.score_popups.active[i],
                    render_active,
                    lambda r: r,
                    r_in
                )

            return jax.lax.fori_loop(0, state.score_popups.x.shape[0], body, raster_to_update)

        raster = render_score_popups(raster)

        # ----------- PLAYER -------------
        player_sprites = jax.lax.switch(
            state.character_id,
            [
                lambda _: self.sprites['ASTERIX_SPRITES'],
                lambda _: self.sprites['OBELIX_SPRITES'],
            ],
            None
        )
        direction = state.player_direction

        def pick_normal(_):
            return jax.lax.switch(
                direction - 1,
                [
                    lambda _: player_sprites[0],
                    lambda _: player_sprites[1],
                ],
                None
            )


        direction = state.player_direction

        def pick_hit(_):
            return jax.lax.switch(
                direction - 1,
                [
                    lambda _: player_sprites[2],
                    lambda _: player_sprites[3],
                ],
                None
            )

        player_sprite = jax.lax.cond(state.hit_timer > 0, pick_hit, pick_normal, operand=None)

        offset = self.offsets.get('ASTERIX', (0, 0))

        raster = jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            player_sprite,
            flip_offset=offset
        )

        # ----------- SCORE -------------
        # Define score positions and spacing
        player_score_rightmost_digit_x = 49  # X position for the START of the player's rightmost digit (or single digit)
        max_score_digits = 6

        # Get digit sprites
        digit_sprites = self.sprites.get('digit', None)
        max_digits = self.consts.max_digits_score

        # Define the function to render scores if sprites are available
        def render_scores(raster_to_update):
            score = state.score.astype(jnp.int32)

            # Stellenanzahl ohne Floating-Point (vermeidet log10-Grenzfall bei 100, 1000, ...)
            def count_digits(n):
                return jnp.where(n < 10, 1,
                                 jnp.where(n < 100, 2,
                                           jnp.where(n < 1000, 3,
                                                     jnp.where(n < 10_000, 4,
                                                               jnp.where(n < 100_000, 5, 6)))))

            num_digits = count_digits(score)
            #num_digits = jnp.minimum(num_digits, max_digits)

            # Digits von rechts nach links einfüllen
            def fill_body(i, carry):
                n, digits = carry
                digit = n % 10
                pos = max_digits - 1 - i
                digits = digits.at[pos].set(digit)
                n = n // 10
                return (n, digits)

            init = (score, jnp.zeros((max_digits,), dtype=jnp.int32))
            _, digits_full = jax.lax.fori_loop(0, max_digits, fill_body, init)

            start_idx = max_digits - num_digits

            score_spacing = 8
            score_x = ((self.consts.screen_width - (num_digits * score_spacing)) // 2) + 20
            score_y = bottom_y + bottom_sprite.shape[0] + jr.get_sprite_frame(self.sprites['ASTERIX_LEFT'], 0).shape[0] + 6

            return jr.render_label_selective(
                raster_to_update,
                score_x,
                score_y,
                digits_full,
                digit_sprites,
                start_idx,
                num_digits,
                spacing=score_spacing
            )

        # Render scores conditionally
        raster = jax.lax.cond(
            digit_sprites is not None,
            render_scores,
            lambda r: r,
            raster
        )


        # ----------- LIVES -------------
        num_lives = jnp.maximum(state.lives, 0).astype(jnp.int32) - 1
        life_sprite = jax.lax.switch(
            state.player_direction - 1,
            [
                lambda _: jr.get_sprite_frame(self.sprites['ASTERIX_LEFT'], 0),  # links
                lambda _: jr.get_sprite_frame(self.sprites['ASTERIX_RIGHT'], 0),  # rechts
            ],
            None
        )
        life_width = life_sprite.shape[1]
        life_height = life_sprite.shape[0]
        lives_spacing = 8  # Abstand zwischen den Leben
        total_lives_width = num_lives * life_width + (num_lives - 1) * lives_spacing
        lives_start_x = (self.consts.screen_width - total_lives_width) // 2
        lives_y = bottom_y + bottom_sprite.shape[0] + 3  # 3 Pixel unter Bottom

        def render_life(i, raster_to_update):
            x = lives_start_x + i * (life_width + lives_spacing)
            return jr.render_at(
                raster_to_update,
                x,
                lives_y,
                life_sprite
            )

        def render_lives(raster_to_update):
            def body_fun(i, r):
                return render_life(i, r)

            return jax.lax.fori_loop(0, num_lives, body_fun, raster_to_update)

        raster = jax.lax.cond(
            num_lives > 0,
            render_lives,
            lambda r: r,
            raster
        )


        return raster