import chex
import jax
import jax.numpy as jnp
from functools import partial
from flax import struct

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_bankheist import JaxBankHeist, BankHeistState, Entity


class RandomBankSpawnsMod(JaxAtariInternalModPlugin):
    """
    Restores the procedural, fully random bank spawns over all valid map tiles
    instead of using the 16-step deterministic ALE loop.
    """

    @partial(jax.jit, static_argnums=(0,))
    def spawn_banks_fn(
        self, state: BankHeistState, step_random_key: jax.Array
    ) -> BankHeistState:
        # We override the base function and inject the procedural logic using state.spawn_points
        new_bank_spawns = jax.random.randint(
            step_random_key,
            shape=(state.bank_positions.position.shape[0],),
            minval=0,
            maxval=state.spawn_points.shape[0],
        )
        chosen_points = state.spawn_points[new_bank_spawns]

        spawning_mask = state.bank_spawn_timers == 0
        new_bank_positions = jnp.where(
            spawning_mask[:, None], chosen_points, state.bank_positions.position
        )
        new_visibility = jnp.where(
            spawning_mask,
            jnp.array([1, 1, 1]),
            state.bank_positions.visibility,
        )

        new_banks = state.bank_positions.replace(
            position=new_bank_positions, visibility=new_visibility
        )
        return state.replace(bank_positions=new_banks)


class UnlimitedGasMod(JaxAtariInternalModPlugin):
    """
    Mod that prevents gas consumption. Only dropping a dynamite consumes gaz.
    """
    @partial(jax.jit, static_argnums=(0,))
    def fuel_step(self, state: BankHeistState) -> BankHeistState:
        return state


class NoPoliceMod(JaxAtariInternalModPlugin):
    """
    Mod that removes all police cars from the game and automatically respawns banks.
    """
    @partial(jax.jit, static_argnums=(0,))
    def spawn_police_car(self, state: BankHeistState, bank_index: chex.Array) -> BankHeistState:
        # Instead of spawning a police car, trigger a bank respawn for the next frame
        new_bank_timers = state.bank_spawn_timers.at[bank_index].set(1)
        return state.replace(bank_spawn_timers=new_bank_timers)

class TwoPoliceCarsMod(JaxAtariInternalModPlugin):
    """
    Replaces 2 banks with police cars. Each bank robbed gives 50 points.
    """
    constants_overrides = {
        "BASE_BANK_ROBBERY_REWARD": 50
    }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, state = JaxBankHeist.reset(self._env, key)
        
        key1, key2 = jax.random.split(state.random_key)
        pos1 = state.spawn_points[1]
        pos2 = state.spawn_points[2]
        
        new_police_pos = state.enemy_positions.position.at[1].set(pos1).at[2].set(pos2)
        new_police_vis = state.enemy_positions.visibility.at[1].set(1).at[2].set(1)
        new_police_dir = state.enemy_positions.direction.at[1].set(self._env.consts.DIR_UP).at[2].set(self._env.consts.DIR_DOWN)
        
        new_enemy = state.enemy_positions.replace(position=new_police_pos, visibility=new_police_vis, direction=new_police_dir)
        new_bank_timers = state.bank_spawn_timers.at[1].set(-1).at[2].set(-1)
        
        state = state.replace(
            enemy_positions=new_enemy,
            bank_spawn_timers=new_bank_timers,
            random_key=key2
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: BankHeistState) -> BankHeistState:
        # Multiply bank_heists by 3 to compensate for fewer banks for gas refill and bonus
        modified_state = state.replace(bank_heists=state.bank_heists * 3)
        state = JaxBankHeist.map_transition(self._env, modified_state)
        
        key1, key2 = jax.random.split(state.random_key)
        pos1 = state.spawn_points[1]
        pos2 = state.spawn_points[2]
        
        new_police_pos = state.enemy_positions.position.at[1].set(pos1).at[2].set(pos2)
        new_police_vis = state.enemy_positions.visibility.at[1].set(1).at[2].set(1)
        new_police_dir = state.enemy_positions.direction.at[1].set(self._env.consts.DIR_UP).at[2].set(self._env.consts.DIR_DOWN)
        
        new_enemy = state.enemy_positions.replace(position=new_police_pos, visibility=new_police_vis, direction=new_police_dir)
        new_bank_timers = state.bank_spawn_timers.at[1].set(-1).at[2].set(-1)
        
        state = state.replace(
            enemy_positions=new_enemy,
            bank_spawn_timers=new_bank_timers,
            random_key=key2
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def handle_bank_robbery(self, state: BankHeistState, bank_hit_index: chex.Array) -> BankHeistState:
        new_bank_visibility = state.bank_positions.visibility.at[bank_hit_index].set(0)
        new_banks = state.bank_positions.replace(visibility=new_bank_visibility)
        
        new_bank_heists = state.bank_heists + 1
        
        # Display 50 points (index 5)
        new_pending_scores = state.pending_police_scores.at[bank_hit_index].set(5)
        new_pending_spawns = state.pending_police_spawns.at[bank_hit_index].set(120)
        new_pending_bank_indices = state.pending_police_bank_indices.at[bank_hit_index].set(bank_hit_index)
        
        # Capture robbery position for score display
        robbery_position = state.bank_positions.position[bank_hit_index]
        new_pending_spawn_positions = state.pending_police_spawn_positions.at[bank_hit_index].set(robbery_position)

        new_money = state.money + self._env.consts.BASE_BANK_ROBBERY_REWARD
        
        return state.replace(
            bank_positions=new_banks,
            pending_police_spawns=new_pending_spawns,
            pending_police_bank_indices=new_pending_bank_indices,
            pending_police_spawn_positions=new_pending_spawn_positions,
            pending_police_scores=new_pending_scores,
            bank_heists=new_bank_heists,
            money=new_money
        )

    @partial(jax.jit, static_argnums=(0,))
    def process_pending_police_spawns(self, state: BankHeistState) -> BankHeistState:
        # We spawn a bank instead of a police car for slot 0
        def process_single_spawn(i, current_state):
            def spawn_bank_instead(state_inner):
                key, subkey = jax.random.split(state_inner.random_key)
                new_pos_idx = jax.random.randint(subkey, (), minval=0, maxval=state_inner.spawn_points.shape[0])
                new_pos = state_inner.spawn_points[new_pos_idx]
                
                new_bank_pos = state_inner.bank_positions.position.at[i].set(new_pos)
                new_bank_vis = state_inner.bank_positions.visibility.at[i].set(1)
                new_banks = state_inner.bank_positions.replace(position=new_bank_pos, visibility=new_bank_vis)
                
                new_pending_spawns = state_inner.pending_police_spawns.at[i].set(-1)
                new_pending_indices = state_inner.pending_police_bank_indices.at[i].set(-1)
                new_pending_scores = state_inner.pending_police_scores.at[i].set(-1)
                new_pending_positions = state_inner.pending_police_spawn_positions.at[i].set(jnp.array([-1, -1]))

                return state_inner.replace(
                    bank_positions=new_banks,
                    pending_police_spawns=new_pending_spawns,
                    pending_police_bank_indices=new_pending_indices,
                    pending_police_scores=new_pending_scores,
                    pending_police_spawn_positions=new_pending_positions,
                    random_key=key
                )            
            ready_to_spawn = current_state.pending_police_spawns[i] == 0
            return jax.lax.cond(ready_to_spawn, spawn_bank_instead, lambda s: s, current_state)
            
        return jax.lax.fori_loop(0, len(state.pending_police_spawns), process_single_spawn, state)

    @partial(jax.jit, static_argnums=(0,))
    def timer_step(self, state: BankHeistState, step_random_key: chex.PRNGKey) -> BankHeistState:
        just_hit_0 = (state.bank_spawn_timers == 1)
        
        new_state = JaxBankHeist.timer_step(self._env, state, step_random_key)
        
        # Revert bank spawn for slots 1 and 2 and spawn police car instead
        mask_police = just_hit_0 & jnp.array([False, True, True])
        
        new_bank_vis = new_state.bank_positions.visibility
        new_bank_vis = jnp.where(mask_police, 0, new_bank_vis)
        new_banks = new_state.bank_positions.replace(visibility=new_bank_vis)
        
        new_pol_pos = new_state.enemy_positions.position
        new_pol_pos = jnp.where(mask_police[:, None], new_state.bank_positions.position, new_pol_pos)
        
        new_pol_vis = new_state.enemy_positions.visibility
        new_pol_vis = jnp.where(mask_police, 1, new_pol_vis)
        
        new_pol_dir = new_state.enemy_positions.direction
        new_pol_dir = jnp.where(mask_police, self._env.consts.DIR_UP, new_pol_dir)
        
        new_enemy = new_state.enemy_positions.replace(position=new_pol_pos, visibility=new_pol_vis, direction=new_pol_dir)
        
        return new_state.replace(bank_positions=new_banks, enemy_positions=new_enemy)


class RandomCityMod(JaxAtariInternalModPlugin):
    """
    Randomizes which city is entered next.
    """
    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: BankHeistState) -> BankHeistState:        
        # Call the original map_transition to handle level progression and difficulty
        new_state = JaxBankHeist.map_transition(self._env, state)
        
        key, subkey = jax.random.split(new_state.random_key)
        num_maps = len(self._env.city_collision_maps)
        random_map_id = jax.random.randint(subkey, (), minval=0, maxval=num_maps)
        
        new_map_collision = jax.lax.dynamic_index_in_dim(self._env.city_collision_maps, random_map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(self._env.city_spawns, random_map_id, axis=0, keepdims=False)
        
        # Load banks for the random map
        new_state = JaxBankHeist.load_city_state(self._env, new_state, random_map_id)
        
        return new_state.replace(
            map_collision=new_map_collision,
            spawn_points=new_spawn_points,
            map_id=random_map_id,
            random_key=key
        )


class RevisitCityMod(JaxAtariInternalModPlugin):
    """
    Allows the player to go back to the previous city by going through the left edge portal.
    Now also persists bank and police car positions when switching cities.
    """

    class ExtendedCityPersistentState(struct.PyTreeNode):
        bank_positions: Entity
        bank_spawn_timers: jnp.ndarray
        bank_spawn_indices: jnp.ndarray
        bank_heists: jnp.ndarray
        enemy_positions: Entity

    @partial(jax.jit, static_argnums=(0,))
    def _init_city_states(self) -> ExtendedCityPersistentState:
        num_maps = len(self._env.city_collision_maps)
        positions = jnp.zeros((num_maps, 3, 2), dtype=jnp.int32)
        directions = jnp.full((num_maps, 3), 4, dtype=jnp.int32)
        visibilities = jnp.zeros((num_maps, 3), dtype=jnp.int32)
        
        bank_positions = Entity(position=positions, direction=directions, visibility=visibilities)
        enemy_positions = Entity(position=positions, direction=directions, visibility=visibilities)
        
        bank_spawn_timers = jnp.ones((num_maps, 3), dtype=jnp.int32)
        
        base_indices = jnp.array([0, 5, 10], dtype=jnp.int32)
        map_ids = jnp.arange(num_maps)[:, None]
        bank_spawn_indices = (base_indices + map_ids * 3) % 16
        
        bank_heists = jnp.zeros((num_maps,), dtype=jnp.int32)
        
        return self.ExtendedCityPersistentState(
            bank_positions=bank_positions,
            bank_spawn_timers=bank_spawn_timers,
            bank_spawn_indices=bank_spawn_indices,
            bank_heists=bank_heists,
            enemy_positions=enemy_positions
        )

    @partial(jax.jit, static_argnums=(0,))
    def save_city_state(self, city_states: ExtendedCityPersistentState, map_id: int, 
                        bank_positions: Entity, bank_spawn_timers: jnp.ndarray, 
                        bank_spawn_indices: jnp.ndarray, bank_heists: jnp.ndarray,
                        enemy_positions: Entity = None) -> ExtendedCityPersistentState:
        new_city_states = city_states.replace(
            bank_positions=city_states.bank_positions.replace(
                position=city_states.bank_positions.position.at[map_id].set(bank_positions.position),
                direction=city_states.bank_positions.direction.at[map_id].set(bank_positions.direction),
                visibility=city_states.bank_positions.visibility.at[map_id].set(bank_positions.visibility),
            ),
            bank_spawn_timers=city_states.bank_spawn_timers.at[map_id].set(bank_spawn_timers),
            bank_spawn_indices=city_states.bank_spawn_indices.at[map_id].set(bank_spawn_indices),
            bank_heists=city_states.bank_heists.at[map_id].set(bank_heists),
        )
        
        def update_enemies(cs):
            return cs.replace(
                enemy_positions=cs.enemy_positions.replace(
                    position=cs.enemy_positions.position.at[map_id].set(enemy_positions.position),
                    direction=cs.enemy_positions.direction.at[map_id].set(enemy_positions.direction),
                    visibility=cs.enemy_positions.visibility.at[map_id].set(enemy_positions.visibility),
                )
            )
        
        return jax.lax.cond(enemy_positions is not None, update_enemies, lambda cs: cs, new_city_states)

    @partial(jax.jit, static_argnums=(0,))
    def load_city_state(self, state: BankHeistState, map_id: int) -> BankHeistState:
        city_states = state.city_states
        return state.replace(
            bank_positions=Entity(
                position=city_states.bank_positions.position[map_id],
                direction=city_states.bank_positions.direction[map_id],
                visibility=city_states.bank_positions.visibility[map_id],
            ),
            enemy_positions=Entity(
                position=city_states.enemy_positions.position[map_id],
                direction=city_states.enemy_positions.direction[map_id],
                visibility=city_states.enemy_positions.visibility[map_id],
            ),
            bank_spawn_timers=city_states.bank_spawn_timers[map_id],
            bank_spawn_indices=city_states.bank_spawn_indices[map_id],
            bank_heists=city_states.bank_heists[map_id],
        )

    @partial(jax.jit, static_argnums=(0,))
    def map_transition(self, state: BankHeistState) -> BankHeistState:
        # 1. Save current city state
        saved_city_states = self.save_city_state(
            state.city_states, state.map_id, 
            state.bank_positions, state.bank_spawn_timers, 
            state.bank_spawn_indices, state.bank_heists,
            state.enemy_positions
        )

        new_level = state.level + 1
        new_difficulty_level = jnp.minimum(new_level // self._env.consts.CITIES_PER_LEVEL, self._env.consts.MAX_LEVEL - 1)
        
        # 2. Reset city states if starting a new loop
        num_maps = len(self._env.city_collision_maps)
        is_new_loop = (new_level % num_maps == 0) & (new_level > state.level)
        saved_city_states = jax.lax.cond(
            is_new_loop,
            lambda cs: self._init_city_states(),
            lambda cs: cs,
            saved_city_states
        )

        new_map_id = new_level % num_maps
        
        # 3. Load next city state
        temp_state = state.replace(city_states=saved_city_states)
        loaded_state = self.load_city_state(temp_state, new_map_id)

        default_player_position = jnp.array(self._env.consts.LEVEL_TRANSITION_SPAWN).astype(jnp.int32)
        new_player = state.player.replace(position=default_player_position)
        
        new_speed = self._env.consts.BASE_SPEED * jnp.power(self._env.consts.SPEED_INCREASE_PER_LEVEL, new_difficulty_level)
        new_fuel_consumption = jnp.power(self._env.consts.FUEL_CONSUMPTION_INCREASE_PER_LEVEL, new_difficulty_level)
        
        new_fuel = jnp.maximum(state.fuel, jax.lax.dynamic_index_in_dim(self._env.consts.REFILL_TABLE, state.bank_heists, axis=0, keepdims=False))
        new_fuel_refill = jnp.array(0).astype(jnp.int32)
        
        new_map_collision = jax.lax.dynamic_index_in_dim(self._env.city_collision_maps, new_map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(self._env.city_spawns, new_map_id, axis=0, keepdims=False)
        new_dynamite_position = jnp.array([-1, -1]).astype(jnp.int32)  
        new_police_spawn_timers = jnp.array([-1, -1, -1]).astype(jnp.int32)
        new_dynamite_timer = jnp.array([-1]).astype(jnp.int32)
        
        new_player_lives = jax.lax.cond(
            state.bank_heists >= 9, 
            lambda: state.player_lives + 1, 
            lambda: state.player_lives
        )
        
        city_reward_index = state.level % 4
        city_reward = jnp.where(city_reward_index == 0, self._env.consts.CITY_REWARD[0],
                      jnp.where(city_reward_index == 1, self._env.consts.CITY_REWARD[1],
                      jnp.where(city_reward_index == 2, self._env.consts.CITY_REWARD[2], self._env.consts.CITY_REWARD[3])))
        
        total_bonus = jax.lax.cond(
            state.bank_heists >= 9,
            lambda: city_reward + (state.difficulty_level * self._env.consts.BONUS_REWARD),
            lambda: jnp.array(0).astype(jnp.int32)
        )
        new_money = state.money + total_bonus
        
        return state.replace(
            level=new_level,
            map_id=new_map_id,
            difficulty_level=new_difficulty_level,
            player=new_player,
            player_move_direction=new_player.direction,
            portal_pending=jnp.array(False).astype(jnp.bool_),
            portal_pending_side=jnp.array(-1).astype(jnp.int32),
            player_lives=new_player_lives,
            money=new_money,
            enemy_positions=loaded_state.enemy_positions,
            bank_positions=loaded_state.bank_positions,
            speed=new_speed,
            fuel_consumption=new_fuel_consumption,
            fuel=new_fuel,
            fuel_refill=new_fuel_refill,
            map_collision=new_map_collision,
            spawn_points=new_spawn_points,
            dynamite_position=new_dynamite_position,
            bank_spawn_timers=loaded_state.bank_spawn_timers,
            police_spawn_timers=new_police_spawn_timers,
            dynamite_timer=new_dynamite_timer,
            pending_police_spawns=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_bank_indices=jnp.array([-1, -1, -1]).astype(jnp.int32),
            pending_police_spawn_positions=jnp.full((3, 2), -1).astype(jnp.int32),
            bank_heists=loaded_state.bank_heists,
            bank_spawn_indices=loaded_state.bank_spawn_indices,
            city_states=saved_city_states,
            pending_exit=jnp.array(False).astype(jnp.bool_),
            random_key=jax.random.fold_in(state.random_key, new_level + 100)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # Call the actual modded step from the environment (respects other patches)
        obs, next_state, reward, done, info = JaxBankHeist.step(self._env, state, action)
        
        # Revisit logic
        went_left_portal = (state.player.position[0] <= 30) & (next_state.player.position[0] >= 120) & (state.level > 0)
        
        def transition_back(s):
            # 1. Save current city state
            saved_city_states = self.save_city_state(
                s.city_states, s.map_id, 
                s.bank_positions, s.bank_spawn_timers, 
                s.bank_spawn_indices, s.bank_heists,
                s.enemy_positions
            )

            # 2. To go back a level
            temp_state = s.replace(level=s.level - 2)
            state_after = self.map_transition(temp_state)
            
            # 3. Restore
            new_map_id = (s.level - 1) % len(self._env.city_collision_maps)
            preservation_state = state_after.replace(city_states=saved_city_states)
            loaded_state = self.load_city_state(preservation_state, new_map_id)
            
            return state_after.replace(
                bank_positions=loaded_state.bank_positions,
                enemy_positions=loaded_state.enemy_positions,
                bank_spawn_timers=loaded_state.bank_spawn_timers,
                bank_heists=loaded_state.bank_heists,
                bank_spawn_indices=loaded_state.bank_spawn_indices,
                city_states=saved_city_states,
                player=next_state.player,
                player_move_direction=next_state.player_move_direction
            )
            
        next_state = jax.lax.cond(went_left_portal, transition_back, lambda s: s, next_state)
        # Update observation if state changed
        obs = jax.lax.cond(went_left_portal, lambda: self._env._get_observation(next_state), lambda: obs)
        
        return obs, next_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        return JaxBankHeist.reset(self._env, key)


@partial(jax.jit, static_argnums=(0,))
def _internal_move_banks(env, state: BankHeistState, random_key: jax.Array) -> BankHeistState:
    def move_single_bank(i, current_state):
        def move_bank(state_inner):
            bank_position = state_inner.bank_positions.position[i]
            current_direction = state_inner.bank_positions.direction[i]
            
            # Use a unique salt for each bank's random decision
            bank_key = jax.random.fold_in(random_key, i + 100)
            
            # Reuse the police direction logic
            new_direction = JaxBankHeist.choose_police_direction(
                env, state_inner, bank_position, current_direction, bank_key
            )
            
            temp_entity = Entity(
                position=bank_position,
                direction=jnp.array(new_direction),
                visibility=jnp.array(1)
            )
            moved_entity = JaxBankHeist.move(env, temp_entity, new_direction)
            collision = JaxBankHeist.check_background_collision(env, state_inner, moved_entity)
            
            moved_entity = jax.lax.cond(collision >= 255,
                lambda: temp_entity, 
                lambda: moved_entity 
            )
            
            # Banks use the same portal logic as police
            moved_entity = JaxBankHeist.portal_handler(env, moved_entity, collision, False)
            
            new_positions = state_inner.bank_positions.position.at[i].set(moved_entity.position)
            new_directions = state_inner.bank_positions.direction.at[i].set(new_direction)
            
            new_banks = state_inner.bank_positions.replace(
                position=new_positions,
                direction=new_directions
            )
            
            return state_inner.replace(bank_positions=new_banks)
        
        is_visible = current_state.bank_positions.visibility[i] > 0
        return jax.lax.cond(is_visible, move_bank, lambda s: s, current_state)
    
    return jax.lax.fori_loop(0, len(state.bank_positions.visibility), move_single_bank, state)


class MovingBanksMod(JaxAtariInternalModPlugin):
    """
    Mod that makes banks move like police cars.
    """

    @partial(jax.jit, static_argnums=(0,))
    def move_police_cars(self, state: BankHeistState, random_key: jax.Array) -> BankHeistState:
        # Move police cars using original logic
        state = JaxBankHeist.move_police_cars(self._env, state, random_key)
        # Move banks using the same logic (this is called from the speed loop in step)
        state = _internal_move_banks(self._env, state, random_key)
        return state

class DoubleSpeedMod(JaxAtariInternalModPlugin):
    """
    Mod that doubles the player's speed.
    It does so by calling the player_move_step twice, which guarantees
    per-pixel collision checks are maintained (preventing tunneling through walls).
    """
    @partial(jax.jit, static_argnums=(0,))
    def player_move_step(self, state: BankHeistState) -> BankHeistState:
        state = JaxBankHeist.player_move_step(self._env, state)
        state = JaxBankHeist.player_move_step(self._env, state)
        return state
