import chex
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_bankheist import BankHeistState


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
        from jaxatari.games.jax_bankheist import JaxBankHeist
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
        from jaxatari.games.jax_bankheist import JaxBankHeist
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
        
        new_money = state.money + self._env.consts.BASE_BANK_ROBBERY_REWARD
        
        return state.replace(
            bank_positions=new_banks,
            pending_police_spawns=new_pending_spawns,
            pending_police_bank_indices=new_pending_bank_indices,
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
                
                return state_inner.replace(
                    bank_positions=new_banks,
                    pending_police_spawns=new_pending_spawns,
                    pending_police_bank_indices=new_pending_indices,
                    pending_police_scores=new_pending_scores,
                    random_key=key
                )
            
            ready_to_spawn = current_state.pending_police_spawns[i] == 0
            return jax.lax.cond(ready_to_spawn, spawn_bank_instead, lambda s: s, current_state)
            
        return jax.lax.fori_loop(0, len(state.pending_police_spawns), process_single_spawn, state)

    @partial(jax.jit, static_argnums=(0,))
    def timer_step(self, state: BankHeistState, step_random_key: chex.PRNGKey) -> BankHeistState:
        from jaxatari.games.jax_bankheist import JaxBankHeist
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
        from jaxatari.games.jax_bankheist import JaxBankHeist
        
        # Call the original map_transition to handle level progression and difficulty
        new_state = JaxBankHeist.map_transition(self._env, state)
        
        key, subkey = jax.random.split(new_state.random_key)
        num_maps = len(self._env.city_collision_maps)
        random_map_id = jax.random.randint(subkey, (), minval=0, maxval=num_maps)
        
        new_map_collision = jax.lax.dynamic_index_in_dim(self._env.city_collision_maps, random_map_id, axis=0, keepdims=False)
        new_spawn_points = jax.lax.dynamic_index_in_dim(self._env.city_spawns, random_map_id, axis=0, keepdims=False)
        
        return new_state.replace(
            map_collision=new_map_collision,
            spawn_points=new_spawn_points,
            random_key=key
        )


class RevisitCityMod(JaxAtariPostStepModPlugin):
    """
    Allows the player to go back to the previous city by going through the left edge portal.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BankHeistState, new_state: BankHeistState) -> BankHeistState:
        from jaxatari.games.jax_bankheist import JaxBankHeist
        
        # Detect if the player wrapped around from the left edge to the right edge
        went_left_portal = (prev_state.player.position[0] <= 30) & (new_state.player.position[0] >= 120) & (prev_state.level > 0)
        
        def transition_back(s):
            # To go back a level, we temporarily subtract 2 from the level before triggering map_transition
            # Since map_transition adds 1, it results in level - 1
            temp_state = s.replace(level=s.level - 2)
            return JaxBankHeist.map_transition(self._env, temp_state)
            
        return jax.lax.cond(went_left_portal, transition_back, lambda s: s, new_state)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: BankHeistState):
        return obs, state
