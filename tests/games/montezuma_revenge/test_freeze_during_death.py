import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge

def test_enemies_freeze_during_death():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Place an enemy and player at the same position to trigger death
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100),
        enemies_active=state.enemies_active.at[0].set(1),
        enemies_direction=state.enemies_direction.at[0].set(1),
        enemies_min_x=state.enemies_min_x.at[0].set(0),
        enemies_max_x=state.enemies_max_x.at[0].set(160)
    )
    
    # First step triggers death
    obs, state, reward, done, info = env.step(state, 0)
    assert state.death_timer > 0
    
    enemy_x_at_death = state.enemies_x[0]
    
    # Step again while death_timer is active
    for _ in range(5):
        obs, state, reward, done, info = env.step(state, 0)
        assert state.death_timer > 0
    
    # Currently it might still move because I haven't implemented the fix yet
    # I'll check if it moved
    assert state.enemies_x[0] == enemy_x_at_death, f"Enemy moved during death! {enemy_x_at_death} -> {state.enemies_x[0]}"

def test_lasers_freeze_during_death():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Trigger death by laser
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(14, dtype=jnp.int32), state, env.consts)
    state = state.replace(
        player_x=jnp.array(40, dtype=jnp.int32),
        player_y=jnp.array(20, dtype=jnp.int32),
        lasers_x=state.lasers_x.at[0].set(40),
        lasers_active=state.lasers_active.at[0].set(1),
        laser_cycle=jnp.array(0, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0)
    assert state.death_timer > 0
    
    laser_cycle_at_death = state.laser_cycle
    
    # Step again
    obs, state, reward, done, info = env.step(state, 0)
    assert state.death_timer > 0
    
    assert state.laser_cycle == laser_cycle_at_death, f"Laser cycle incremented during death! {laser_cycle_at_death} -> {state.laser_cycle}"

if __name__ == "__main__":
    test_enemies_freeze_during_death()
    test_lasers_freeze_during_death()
