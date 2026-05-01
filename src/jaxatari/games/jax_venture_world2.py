import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
from typing import Tuple, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
import jaxatari.spaces as spaces
from jaxatari.games.venture.core import (
    VentureConstants, GameState, PlayerState, MonsterState, ChaserState, LaserState, ProjectileState,
    VentureObservation, VentureInfo, _build_venture_static_data, FIREACTIONS
)
from jaxatari.games.venture.logic import (
    _update_player, _update_monsters_optimized, _update_dead_for, _update_chaser,
    _check_player_hazard_collision, _handle_level_transitions
)
from jaxatari.games.venture.renderer import VentureRenderer

class JaxVentureWorld2(JaxEnvironment[GameState, VentureObservation, VentureInfo, VentureConstants]):
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
    ], dtype=jnp.int32)

    def __init__(self, consts: VentureConstants = None):
        base_consts = consts or VentureConstants()
        static_data = _build_venture_static_data()
        initialized_consts = base_consts.replace(
            ALL_WALL_MAPS_PER_WORLD=static_data["all_wall_maps_per_world"],
            LEVEL_MONSTER_CONFIGS=static_data["level_monster_configs"],
            TOTAL_MONSTERS=static_data["total_monsters"],
            LEVEL_OFFSETS=static_data["level_offsets"],
            ALL_MONSTER_SPAWNS=static_data["all_monster_spawns"],
            ALL_MONSTER_IMMORTAL_FLAGS=static_data["all_monster_immortal_flags"],
            JAX_TRANSITIONS=static_data["jax_transitions"],
            MAIN_MAP_PORTAL_MASKS=static_data["main_map_portal_masks"],
            MAIN_MAP_PORTAL_TO_LEVELS=static_data["main_map_portal_to_levels"],
        )
        super().__init__(initialized_consts)
        self.renderer = VentureRenderer(self.consts)

    def reset(self, key: jrandom.PRNGKey = None) -> tuple[VentureObservation, GameState]:
        key, monster_key = jrandom.split(key, 2)
        player_state = PlayerState(
            x=jnp.array(self.consts.PLAYER_INITIAL_X, dtype=jnp.float32),
            y=jnp.array(self.consts.PLAYER_INITIAL_Y, dtype=jnp.float32),
            last_valid_x=jnp.array(self.consts.PLAYER_INITIAL_X, dtype=jnp.float32),
            last_valid_y=jnp.array(self.consts.PLAYER_INITIAL_Y, dtype=jnp.float32),
            last_dx=jnp.array(1.0, dtype=jnp.float32), last_dy=jnp.array(0.0, dtype=jnp.float32)
        )
        projectile_state = ProjectileState(
            x=jnp.array(0.0, dtype=jnp.float32), y=jnp.array(0.0, dtype=jnp.float32),
            dx=jnp.array(0.0, dtype=jnp.float32), dy=jnp.array(0.0, dtype=jnp.float32),
            active=jnp.array(False, dtype=jnp.bool_), lifetime=jnp.array(0, dtype=jnp.int32)
        )
        angles = jrandom.uniform(monster_key, shape=(self.consts.TOTAL_MONSTERS,), minval=0, maxval=2 * jnp.pi, dtype=jnp.float32)
        indices = jnp.arange(self.consts.TOTAL_MONSTERS)
        active_monsters = (indices >= self.consts.LEVEL_OFFSETS[5]) & (indices < self.consts.LEVEL_OFFSETS[6])
        monster_state = MonsterState(
            x=self.consts.ALL_MONSTER_SPAWNS[:, 0].astype(jnp.float32),
            y=self.consts.ALL_MONSTER_SPAWNS[:, 1].astype(jnp.float32),
            dx=jnp.cos(angles), dy=jnp.sin(angles), active=active_monsters,
            is_immortal=self.consts.ALL_MONSTER_IMMORTAL_FLAGS,
            dead_for=jnp.full((self.consts.TOTAL_MONSTERS,), -1, dtype=jnp.int32)
        )
        state = GameState(
            player=player_state, monsters=monster_state, projectile=projectile_state,
            chaser=ChaserState(x=jnp.array(0.0), y=jnp.array(0.0), active=jnp.array(False)), chests_active=jnp.ones(4, dtype=jnp.bool_),
            lasers=LaserState(positions=self.consts.LASER_INITIAL_POSITIONS, directions=self.consts.LASER_INITIAL_DIRECTIONS),
            kill_bonus_active=jnp.zeros(4, dtype=jnp.bool_), key=key, game_over_timer=jnp.array(0, dtype=jnp.int32),
            life_lost_timer=jnp.array(0, dtype=jnp.int32), level_timer=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32), score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.LIVES, dtype=jnp.int32), is_in_collision=jnp.array(False, dtype=jnp.bool_),
            current_level=jnp.array(0, dtype=jnp.int32), world_level=jnp.array(2, dtype=jnp.int32),
            monster_speed_index=jnp.array(0, dtype=jnp.int32), world_transition_timer=jnp.array(0, dtype=jnp.int32),
            last_level=jnp.array(0, dtype=jnp.int32), collected_chest_in_current_visit=jnp.array(-1, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: jnp.ndarray) -> tuple[VentureObservation, GameState, jnp.ndarray, bool, VentureInfo]:
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        state = state.replace(last_level=state.current_level)

        def handle_life_lost_delay(s: GameState) -> GameState:
            return jax.lax.cond(s.life_lost_timer == 1, self._respawn_entities, lambda _s: _s, s).replace(life_lost_timer=s.life_lost_timer - 1)

        def handle_normal_gameplay(s: GameState) -> GameState:
            key, m_key = jrandom.split(s.key, 2)
            p_post_trans = _handle_level_transitions(s, self.consts)
            trans_occ = (p_post_trans.current_level != s.current_level)
            
            def normal_move():
                wall_map = self.consts.ALL_WALL_MAPS_PER_WORLD[1, s.current_level]
                t = self.consts.JAX_TRANSITIONS[1, 0]
                to_lvls = t[:, 4].astype(jnp.int32)
                l_portals = (to_lvls > 0) & (~s.chests_active[jnp.clip(to_lvls-1, 0, 3)])
                n_p, n_coll = jax.lax.cond(jnp.mod(s.step_counter, 2) == 0, lambda: _update_player(s.player, action, s.is_in_collision, wall_map, s.current_level != 0, s.current_level == 0, t[:, 0:4], l_portals, self.consts), lambda: (s.player, s.is_in_collision))
                n_m = jax.lax.cond(jnp.mod(s.step_counter, 3) == 0, lambda: _update_monsters_optimized(s.monsters, m_key, wall_map, self.consts.MONSTER_SPEEDS[s.monster_speed_index], s.current_level, 1, s.current_level == 0, t[:, 0:4], l_portals, self.consts).replace(dead_for=_update_dead_for(s.monsters.dead_for)), lambda: s.monsters)
                return s.replace(player=n_p, monsters=n_m, is_in_collision=n_coll)

            s = jax.lax.cond(trans_occ, lambda: p_post_trans, normal_move)
            
            should_fire = (s.current_level != 0) & ~s.projectile.active & jnp.isin(action, FIREACTIONS)
            s = jax.lax.cond(should_fire, lambda _s: _s.replace(projectile=ProjectileState(x=_s.player.x, y=_s.player.y, dx=_s.player.last_dx, dy=_s.player.last_dy, active=True, lifetime=self.consts.PROJECTILE_LIFETIME_FRAMES)), lambda _s: _s, s)
            
            def update_projectile(ps: GameState):
                proj = ps.projectile
                nx, ny = proj.x + proj.dx * self.consts.PROJECTILE_SPEED, proj.y + proj.dy * self.consts.PROJECTILE_SPEED
                wall_map = self.consts.ALL_WALL_MAPS_PER_WORLD[1, ps.current_level]
                hit_wall = wall_map[jnp.clip(ny.astype(jnp.int32), 0, 209), jnp.clip(nx.astype(jnp.int32), 0, 159)] == 1
                mon_hw, mon_hh = self.consts.MONSTER_RENDER_WIDTH / 2.0, self.consts.MONSTER_RENDER_HEIGHT / 2.0
                dist_sq = (nx - jnp.clip(nx, ps.monsters.x - mon_hw, ps.monsters.x + mon_hw))**2 + (ny - jnp.clip(ny, ps.monsters.y - mon_hh, ps.monsters.y + mon_hh))**2
                hit_mask = (dist_sq < self.consts.PROJECTILE_RADIUS**2) & ps.monsters.active & ~ps.monsters.is_immortal
                return ps.replace(projectile=proj.replace(x=nx, y=ny, lifetime=proj.lifetime - 1, active=~(hit_wall | jnp.any(hit_mask) | (proj.lifetime <= 1))), monsters=ps.monsters.replace(active=jnp.where(hit_mask, False, ps.monsters.active), dead_for=jnp.where(hit_mask, self.consts.DEAD_MONSTER_LIFETIME_FRAMES, ps.monsters.dead_for)), score=ps.score + jax.lax.select(jnp.any(hit_mask) & ps.kill_bonus_active[jnp.clip(ps.current_level-1, 0, 3)], jnp.sum(hit_mask)*100, 0))
            s = jax.lax.cond(s.projectile.active, update_projectile, lambda _s: _s, s)

            def collect_chest(cs: GameState):
                c_idx = cs.current_level - 1
                pos = self.consts.CHEST_POSITIONS[5 + cs.current_level]
                dist_sq = (cs.player.x - jnp.clip(cs.player.x, pos[0]-self.consts.CHEST_WIDTH/2, pos[0]+self.consts.CHEST_WIDTH/2))**2 + (cs.player.y - jnp.clip(cs.player.y, pos[1]-self.consts.CHEST_HEIGHT/2, pos[1]+self.consts.CHEST_HEIGHT/2))**2
                coll = dist_sq < self.consts.PLAYER_ROOM_RADIUS**2
                return jax.lax.cond(coll & cs.chests_active[c_idx] & (cs.collected_chest_in_current_visit != c_idx), lambda: cs.replace(score=cs.score + self.consts.CHEST_SCORE, collected_chest_in_current_visit=c_idx, kill_bonus_active=cs.kill_bonus_active.at[c_idx].set(True)), lambda: cs)
            s = jax.lax.cond(s.current_level > 0, collect_chest, lambda _s: _s, s)

            new_lvl_timer = jnp.where(s.current_level > 0, s.level_timer + 1, 0)
            s = jax.lax.cond((s.current_level > 0) & (new_lvl_timer == self.consts.CHASER_SPAWN_FRAMES) & ~s.chaser.active, lambda _s: _s.replace(chaser=_s.chaser.replace(x=self.consts.CHASER_SPAWN_POS[0], y=self.consts.CHASER_SPAWN_POS[1], active=True)), lambda _s: _s, s)
            s = jax.lax.cond(s.chaser.active, lambda _s: _s.replace(chaser=_update_chaser(_s.chaser, _s.player, self.consts.CHASER_SPEED)), lambda _s: _s, s)

            if_coll = _check_player_hazard_collision(s.player, s.monsters, s.chaser, s.lasers, s.current_level, 2, self.consts)
            s = jax.lax.cond(if_coll, lambda _s: _s.replace(lives=_s.lives - 1, life_lost_timer=self.consts.LIFE_LOST_DELAY_FRAMES, game_over_timer=jax.lax.select(_s.lives == 1, self.consts.FINAL_GAME_OVER_DELAY_FRAMES, 0)), lambda _s: _s, s)
            return s.replace(key=key, step_counter=s.step_counter + 1, level_timer=new_lvl_timer)

        new_state = jax.lax.cond(state.game_over_timer > 0, lambda s: s.replace(game_over_timer=s.game_over_timer - 1), lambda s: jax.lax.cond(s.life_lost_timer > 0, handle_life_lost_delay, handle_normal_gameplay, s), state)
        return self._get_observation(new_state), new_state, (new_state.score - state.score).astype(jnp.float32), (new_state.game_over_timer == 1), VentureInfo(time=new_state.step_counter, score=new_state.score, lives=new_state.lives)

    def _respawn_entities(self, state: GameState) -> GameState:
        k, mk = jrandom.split(state.key)
        indices = jnp.arange(self.consts.TOTAL_MONSTERS)
        active = (indices >= self.consts.LEVEL_OFFSETS[5]) & (indices < self.consts.LEVEL_OFFSETS[6])
        return state.replace(player=PlayerState(x=self.consts.PLAYER_INITIAL_X, y=self.consts.PLAYER_INITIAL_Y, last_valid_x=self.consts.PLAYER_INITIAL_X, last_valid_y=self.consts.PLAYER_INITIAL_Y, last_dx=1.0, last_dy=0.0), monsters=MonsterState(x=self.consts.ALL_MONSTER_SPAWNS[:, 0], y=self.consts.ALL_MONSTER_SPAWNS[:, 1], dx=jnp.cos(jrandom.uniform(mk, (self.consts.TOTAL_MONSTERS,))), dy=jnp.sin(jrandom.uniform(mk, (self.consts.TOTAL_MONSTERS,))), active=active, is_immortal=self.consts.ALL_MONSTER_IMMORTAL_FLAGS, dead_for=jnp.full((self.consts.TOTAL_MONSTERS,), -1, dtype=jnp.int32)), chaser=ChaserState(x=0.0, y=0.0, active=False), key=k, current_level=0, is_in_collision=False, level_timer=0, kill_bonus_active=jnp.zeros(4, dtype=jnp.bool_), collected_chest_in_current_visit=-1)

    def render(self, state: GameState) -> jnp.ndarray: return self.renderer.render(state)
    def action_space(self) -> spaces.Discrete: return spaces.Discrete(len(self.ACTION_SET))
    def observation_space(self) -> spaces.Dict: return spaces.Dict({"player": spaces.get_object_space(n=None, screen_size=(210, 160), xy_low=-1.0), "monsters": spaces.get_object_space(n=self.consts.TOTAL_MONSTERS, screen_size=(210, 160), xy_low=-1.0), "portals": spaces.get_object_space(n=self.consts.JAX_TRANSITIONS.shape[2], screen_size=(210, 160), xy_low=-1.0), "chest": spaces.get_object_space(n=None, screen_size=(210, 160), xy_low=-1.0), "lasers": spaces.get_object_space(n=4, screen_size=(210, 160), xy_low=-1.0), "chaser": spaces.get_object_space(n=None, screen_size=(210, 160), xy_low=-1.0)})

    def _get_observation(self, state: GameState) -> VentureObservation:
        w, h = self.consts.SCREEN_WIDTH, self.consts.SCREEN_HEIGHT
        def clip_x(t): return jnp.clip(jnp.round(t), -1, w).astype(jnp.int16)
        def clip_y(t): return jnp.clip(jnp.round(t), -1, h).astype(jnp.int16)
        is_room = state.current_level != 0
        p_w = jnp.where(is_room, self.consts.PLAYER_DETAILED_RENDER_WIDTH, self.consts.PLAYER_DOT_RENDER_WIDTH)
        p_h = jnp.where(is_room, self.consts.PLAYER_DETAILED_RENDER_HEIGHT, self.consts.PLAYER_DOT_RENDER_HEIGHT)
        player = ObjectObservation.create(x=clip_x(state.player.x), y=clip_y(state.player.y), width=clip_x(p_w), height=clip_y(p_h), active=jnp.array(1, dtype=jnp.int8), orientation=0.0)
        monsters = ObjectObservation.create(x=clip_x(jnp.where(state.monsters.active, state.monsters.x, -1)), y=clip_y(jnp.where(state.monsters.active, state.monsters.y, -1)), width=clip_x(jnp.full((self.consts.TOTAL_MONSTERS,), self.consts.MONSTER_RENDER_WIDTH)), height=clip_y(jnp.full((self.consts.TOTAL_MONSTERS,), self.consts.MONSTER_RENDER_HEIGHT)), active=state.monsters.active.astype(jnp.int8), orientation=jnp.zeros(self.consts.TOTAL_MONSTERS))
        t = self.consts.JAX_TRANSITIONS[1, state.current_level]
        portals = ObjectObservation.create(x=clip_x(jnp.where(t[:, 2] > 0, t[:, 0] + t[:, 2] / 2, -1.0)), y=clip_y(jnp.where(t[:, 2] > 0, t[:, 1] + t[:, 3] / 2, -1.0)), width=clip_x(t[:, 2]), height=clip_y(t[:, 3]), active=(t[:, 2] > 0).astype(jnp.int8), orientation=jnp.zeros(t.shape[0]))
        c_pos = self.consts.CHEST_POSITIONS[5 + state.current_level]
        c_active = (state.current_level > 0) & state.chests_active[state.current_level - 1] & (state.collected_chest_in_current_visit != state.current_level - 1)
        chest = ObjectObservation.create(x=clip_x(jnp.where(c_active, c_pos[0], -1.0)), y=clip_y(jnp.where(c_active, c_pos[1], -1.0)), width=clip_x(self.consts.CHEST_WIDTH), height=clip_y(self.consts.CHEST_HEIGHT), active=c_active.astype(jnp.int8), orientation=0.0)
        lasers = ObjectObservation.create(x=jnp.full(4, -1.0), y=jnp.full(4, -1.0), width=jnp.zeros(4), height=jnp.zeros(4), active=jnp.zeros(4, dtype=jnp.int8), orientation=jnp.zeros(4))
        chaser = ObjectObservation.create(x=clip_x(jnp.where(state.chaser.active, state.chaser.x, -1.0)), y=clip_y(jnp.where(state.chaser.active, state.chaser.y, -1.0)), width=clip_x(self.consts.CHASER_RENDER_WIDTH), height=clip_y(self.consts.CHASER_RENDER_HEIGHT), active=state.chaser.active.astype(jnp.int8), orientation=0.0)
        return VentureObservation(player=player, monsters=monsters, portals=portals, chest=chest, lasers=lasers, chaser=chaser)
