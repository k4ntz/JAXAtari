import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_darkchambers import (
    DarkChambersState,
    ITEM_SPEED_POTION,
    ITEM_HEAL_POTION,
    ITEM_POISON_POTION,
    ITEM_HAMMER,
    NUM_ITEMS,
)
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


EASY_MODE_RAPID_FIRE_DELTA = 10_000
EASY_MODE_REGULAR_ITEM_START = 5
EASY_MODE_EXTRA_REGULAR_SLOTS = 16
POTION_CLUSTER_START_SLOT = 6


def _spawn_potion_cluster_near_player(env, state: DarkChambersState, potion_item_type: int) -> DarkChambersState:
    """Fill regular item slots with one potion type near player spawn in the first chamber."""
    idx = jnp.arange(NUM_ITEMS, dtype=jnp.int32)
    cluster_mask = idx >= POTION_CLUSTER_START_SLOT

    # Build a repeatable local pattern around the player for dense pickup access.
    grid_x = ((idx - POTION_CLUSTER_START_SLOT) % 4) * 10
    grid_y = ((idx - POTION_CLUSTER_START_SLOT) // 4) * 10
    base_x = state.player_x + jnp.array(env.consts.PLAYER_WIDTH // 2 - 12, dtype=jnp.int32)
    base_y = state.player_y + jnp.array(env.consts.PLAYER_HEIGHT // 2 + 58, dtype=jnp.int32)

    potion_x = jnp.clip(base_x + grid_x, 0, env.consts.WORLD_WIDTH - 8)
    potion_y = jnp.clip(base_y + grid_y, 0, env.consts.WORLD_HEIGHT - 8)
    potion_positions = jnp.stack([potion_x, potion_y], axis=1).astype(jnp.int32)

    clustered_positions = jnp.where(cluster_mask[:, None], potion_positions, state.item_positions)
    clustered_types = jnp.where(cluster_mask, jnp.array(potion_item_type, dtype=jnp.int32), state.item_types)
    clustered_active = jnp.where(cluster_mask, jnp.array(1, dtype=jnp.int32), state.item_active)

    is_first_chamber = (state.map_index == 0) & (state.current_level == 0)
    return jax.lax.cond(
        is_first_chamber,
        lambda _: state.replace(
            item_positions=clustered_positions,
            item_types=clustered_types,
            item_active=clustered_active,
        ),
        lambda _: state,
        operand=None,
    )


# ============================================================================
# MOD 1: SPEED POTION
# ============================================================================
class SpeedPotionMod(JaxAtariPostStepModPlugin):
    """
    Speed Potion Mod - Proper Integration (Option B)
    
    Visual: Orange 8×8 pixel square (RGB: 255, 100, 0)
    
    When player collects a speed potion item:
    - Speed boost timer is activated (SPEED_POTION_DURATION steps)
    - Player movement speed is doubled during active period
    - Effect automatically expires after duration
    
    Gameplay: Collect to temporarily move 2x faster, useful for dodging/escaping.
    """
    
    # Note: ITEM_SPEED_POTION is a module-level constant (imported above), not a NamedTuple field
    # Enable speed potion item spawning when this mod is active.
    constants_overrides = {
        "ENABLE_SPEED_POTION_SPAWN": True,
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        """
        Post-step logic:
        1. Check for speed potion pickup (item collision)
        2. Activate timer if collected
        3. Decrement active timer
        4. Return updated state
        """
        # Detect collision: player position vs item positions
        player_x = new_state.player_x
        player_y = new_state.player_y
        player_w = jnp.array(self._env.consts.PLAYER_WIDTH, dtype=jnp.int32)
        player_h = jnp.array(self._env.consts.PLAYER_HEIGHT, dtype=jnp.int32)
        
        item_positions = new_state.item_positions
        item_types = new_state.item_types
        item_active = new_state.item_active
        
        # Get actual item size (8×8 for potions)
        item_size = jnp.array(8, dtype=jnp.int32)
        
        # Check each item for collision with speed potion
        def check_speed_potion_pickup(carry, i):
            new_state_carry, timer_set = carry
            item_x = item_positions[i, 0]
            item_y = item_positions[i, 1]
            item_type = item_types[i]
            # Check if item was active in PREV state (before base game removed it)
            was_active = prev_state.item_active[i] == 1
            
            # Simple AABB collision check using actual item size
            collision = (
                (player_x < item_x + item_size) & (player_x + player_w > item_x) &
                (player_y < item_y + item_size) & (player_y + player_h > item_y) &
                (item_type == ITEM_SPEED_POTION) &
                (was_active == 1)
            )
            
            # If collision and not already set, activate timer (item already removed by base game)
            new_state_after = jax.lax.cond(
                collision & (~timer_set),
                lambda s: s.replace(
                    speed_boost_timer=jnp.array(self._env.consts.SPEED_POTION_DURATION, dtype=jnp.int32)
                ),
                lambda s: s,
                operand=new_state_carry
            )
            
            new_timer_set = timer_set | collision
            return (new_state_after, new_timer_set), None
        
        # Scan through all items to detect pickups
        (updated_state, _), _ = jax.lax.scan(
            check_speed_potion_pickup,
            (new_state, False),
            jnp.arange(NUM_ITEMS)
        )
        
        # Decrement active timer
        new_timer = jnp.maximum(
            0,
            updated_state.speed_boost_timer - 1
        )
        
        final_state = updated_state.replace(speed_boost_timer=new_timer)
        return final_state
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        """Initialize timer and spawn many speed potions near player in chamber 0-0."""
        seeded_state = _spawn_potion_cluster_near_player(self._env, state, ITEM_SPEED_POTION)
        return obs, seeded_state.replace(speed_boost_timer=jnp.array(0, dtype=jnp.int32))


# ============================================================================
# MOD 2: HEAL POTION
# ============================================================================
class HealPotionMod(JaxAtariPostStepModPlugin):
    """
    Heal Potion Mod - Proper Integration (Option B)
    
    Visual: Magenta 8×8 pixel square (RGB: 255, 0, 255)
    
    When player collects a heal potion item:
    - Player health is increased by +1000 (clamped to MAX_HEALTH)
    - Works analogously to HEART items with much larger heal amount
    - Item is marked as collected/inactive
    - Effect applies immediately on pickup
    
    Gameplay: Collect to fully heal, strategic timing before dangerous areas.
    """
    
    # Note: ITEM_HEAL_POTION is a module-level constant (imported above), not a NamedTuple field
    # Enable heal potion item spawning when this mod is active.
    constants_overrides = {
        "ENABLE_HEAL_POTION_SPAWN": True,
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        """
        Detect heal potion pickup and add +1000 health (clipped to MAX_HEALTH) immediately.
        """
        player_x = new_state.player_x
        player_y = new_state.player_y
        player_w = jnp.array(self._env.consts.PLAYER_WIDTH, dtype=jnp.int32)
        player_h = jnp.array(self._env.consts.PLAYER_HEIGHT, dtype=jnp.int32)
        
        item_positions = new_state.item_positions
        item_types = new_state.item_types
        item_active = new_state.item_active
        
        # Get actual item size (8×8 for potions)
        item_size = jnp.array(8, dtype=jnp.int32)
        
        # Check each item for heal potion pickup
        def check_heal_potion_pickup(carry, i):
            state_carry, healed = carry
            item_x = item_positions[i, 0]
            item_y = item_positions[i, 1]
            item_type = item_types[i]
            # Check if item was active in PREV state (before base game removed it)
            was_active = prev_state.item_active[i] == 1
            
            # Collision check using actual item size
            collision = (
                (player_x < item_x + item_size) & (player_x + player_w > item_x) &
                (player_y < item_y + item_size) & (player_y + player_h > item_y) &
                (item_type == ITEM_HEAL_POTION) &
                (was_active == 1)
            )
            
            # If collision and not already healed, add +1000 health (clipped to MAX_HEALTH)
            # Follow the exact same pattern as HEART items in the main game
            # Note: item_active is already set to 0 by base game, so we don't modify it
            health_gain = jnp.array(1000, dtype=jnp.int32)
            
            def apply_heal(s):
                new_health = jnp.clip(s.health + health_gain, 0, self._env.consts.MAX_HEALTH)
                return s.replace(health=new_health)
            
            new_state_after = jax.lax.cond(
                collision & (~healed),
                apply_heal,
                lambda s: s,
                operand=state_carry
            )
            
            new_healed = healed | collision
            return (new_state_after, new_healed), None
        
        # Scan through all items
        (final_state, _), _ = jax.lax.scan(
            check_heal_potion_pickup,
            (new_state, False),
            jnp.arange(NUM_ITEMS)
        )
        
        return final_state
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        """Spawn many heal potions near player in chamber 0-0."""
        seeded_state = _spawn_potion_cluster_near_player(self._env, state, ITEM_HEAL_POTION)
        return obs, seeded_state


# ============================================================================
# MOD 3: POISON POTION
# ============================================================================
class PoisonPotionMod(JaxAtariPostStepModPlugin):
    """
    Poison Potion Mod - Proper Integration (Option B)
    
    Visual: Bright Green 8×8 pixel square (RGB: 0, 255, 0)
    
    When player collects a poison potion item:
    - Poison cloud is activated at player's current position
    - Enemies within POISON_RADIUS (60px) take gradual damage
    - Damage applies every 30 steps (once per second) to prevent instant kills
    - Effect lasts for POISON_DURATION (360 steps = ~12 seconds)
    - Cloud position remains fixed at pickup location
    
    Gameplay: Collect to create AoE damage zone, useful for area denial and crowd control.
    Enemies take 1 hit per second while in the cloud (decrements enemy type by 1).
    """
    
    # Note: ITEM_POISON_POTION is a module-level constant (imported above), not a NamedTuple field
    # POISON_RADIUS, POISON_DURATION, POISON_DAMAGE_INTERVAL are already in DarkChambersConstants
    # Enable poison potion item spawning when this mod is active.
    constants_overrides = {
        "ENABLE_POISON_POTION_SPAWN": True,
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        """
        Detect poison potion pickup and manage poison cloud effect.
        """
        player_x = new_state.player_x
        player_y = new_state.player_y
        player_w = jnp.array(self._env.consts.PLAYER_WIDTH, dtype=jnp.int32)
        player_h = jnp.array(self._env.consts.PLAYER_HEIGHT, dtype=jnp.int32)
        
        item_positions = new_state.item_positions
        item_types = new_state.item_types
        item_active = new_state.item_active
        
        # Get actual item size (8×8 for potions)
        item_size = jnp.array(8, dtype=jnp.int32)
        
        # Check for poison potion pickup
        def check_poison_potion_pickup(carry, i):
            state_carry, timer_set = carry
            item_x = item_positions[i, 0]
            item_y = item_positions[i, 1]
            item_type = item_types[i]
            # Check if item was active in PREV state (before base game removed it)
            was_active = prev_state.item_active[i] == 1
            
            # Collision check using actual item size
            collision = (
                (player_x < item_x + item_size) & (player_x + player_w > item_x) &
                (player_y < item_y + item_size) & (player_y + player_h > item_y) &
                (item_type == ITEM_POISON_POTION) &
                (was_active == 1)
            )
            
            # Activate poison cloud at player position (item already removed by base game)
            new_state_after = jax.lax.cond(
                collision & (~timer_set),
                lambda s: s.replace(
                    poison_cloud_x=s.player_x,
                    poison_cloud_y=s.player_y,
                    poison_cloud_timer=jnp.array(self._env.consts.POISON_DURATION, dtype=jnp.int32)
                ),
                lambda s: s,
                operand=state_carry
            )
            
            new_timer_set = timer_set | collision
            return (new_state_after, new_timer_set), None
        
        # Detect pickups
        (state_after_pickup, _), _ = jax.lax.scan(
            check_poison_potion_pickup,
            (new_state, False),
            jnp.arange(NUM_ITEMS)
        )
        
        # Decrement poison timer
        new_poison_timer = jnp.maximum(
            0,
            state_after_pickup.poison_cloud_timer - 1
        )
        
        final_state = state_after_pickup.replace(poison_cloud_timer=new_poison_timer)
        return final_state
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        """Initialize poison cloud state and spawn many poison potions near player in chamber 0-0."""
        seeded_state = _spawn_potion_cluster_near_player(self._env, state, ITEM_POISON_POTION)
        return obs, seeded_state.replace(
            poison_cloud_x=jnp.array(0, dtype=jnp.int32),
            poison_cloud_y=jnp.array(0, dtype=jnp.int32),
            poison_cloud_timer=jnp.array(0, dtype=jnp.int32)
        )


# ============================================================================
# MOD 4: GRIM REAPER ENEMIES
# ============================================================================
class GrimReaperEnemiesMod(JaxAtariPostStepModPlugin):
    """
    Enables spawning of the highest enemy tier (Grim Reaper).

    This unlocks enemy type 5 in regular enemy and spawner-based spawn pools.
    """

    constants_overrides = {
        "ENABLE_GRIM_REAPER_ENEMIES": True,
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        return obs, state


# ============================================================================
# MOD 5: WIZARD BULLET SHOOTING
# ============================================================================
class WizardBulletShootingMod(JaxAtariPostStepModPlugin):
    """
    Enables wizard projectile attacks.

    Wizards fire tracked enemy bullets at the player based on their shoot timers.
    """

    constants_overrides = {
        "ENABLE_WIZARD_BULLETS": True,
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        return obs, state


# ============================================================================
# MOD 6: HAMMER ITEM SPAWNING
# ============================================================================
class HammerMod(JaxAtariPostStepModPlugin):
    """
    Enables hammer item spawning.

    When active, ITEM_HAMMER enters level spawn pools according to
    probabilities defined in the base environment.
    """

    constants_overrides = {
        "ENABLE_HAMMER_SPAWN": True,
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        """Spawn many hammers near player in chamber 0-0 at the same cluster position."""
        seeded_state = _spawn_potion_cluster_near_player(self._env, state, ITEM_HAMMER)
        return obs, seeded_state


# ============================================================================
# MOD 7: EASY MODE (RAPID FIRE + DENSE ITEMS)
# ============================================================================
class EasyModeMod(JaxAtariPostStepModPlugin):
    """
    Easy mode preset for DarkChambers.

    Effects:
    - Rapid-fire feel: cooldown is reset each step and fire edge is re-armed.
    - Gun is always active.
    - Most regular item slots are kept active to maintain high item density.

    Note:
    - Item count is bounded by NUM_ITEMS (array shape in base environment).
    """

    # Easy mode should not implicitly enable special item families.
    # Those are controlled by their dedicated mods only.
    # Only boost baseline movement speed for a more forgiving run.
    constants_overrides = {
        "PLAYER_SPEED": 2,
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        # Do not force item_active every step; base game must control pickup/removal.
        return new_state.replace(
            gun_active=jnp.array(1, dtype=jnp.int32),
            fire_was_pressed=jnp.array(0, dtype=jnp.int32),
            last_shot_step=new_state.step_counter - jnp.array(EASY_MODE_RAPID_FIRE_DELTA, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        # One-time density boost at reset only (keeps pickups working during play).
        idx = jnp.arange(NUM_ITEMS)
        regular_slots = idx >= EASY_MODE_REGULAR_ITEM_START
        force_active_mask = regular_slots & (idx < (EASY_MODE_REGULAR_ITEM_START + EASY_MODE_EXTRA_REGULAR_SLOTS))

        dense_item_active = jnp.where(
            force_active_mask,
            jnp.ones_like(state.item_active),
            state.item_active,
        )

        return obs, state.replace(
            gun_active=jnp.array(1, dtype=jnp.int32),
            fire_was_pressed=jnp.array(0, dtype=jnp.int32),
            last_shot_step=state.step_counter - jnp.array(EASY_MODE_RAPID_FIRE_DELTA, dtype=jnp.int32),
            item_active=dense_item_active,
        )


# ============================================================================
# MOD 8: CHECKPOINTS
# ============================================================================
class CheckpointsMod(JaxAtariPostStepModPlugin):
    """
    Enables chamber-based checkpoint respawn with multiple lives.

    Behavior:
    - Each chamber acts as a checkpoint (middle/left/right).
    - On death with lives remaining, player respawns at the current chamber checkpoint.
    - Score is preserved (base game already keeps score through death-freeze respawn).
    - Episode starts with 5 lives.
    """

    constants_overrides = {
        "ENABLE_CHECKPOINT_RESPAWN": True,
        "ENABLE_INTRA_LEVEL_CHECKPOINTS": True,
        "CHECKPOINT_SPAWN_X_BY_MAP": (50, 20, 260),
        "CHECKPOINT_SPAWN_Y_BY_MAP": (50, 70, 70),
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DarkChambersState, new_state: DarkChambersState) -> DarkChambersState:
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        return obs, state.replace(lives=jnp.array(5, dtype=jnp.int32))
