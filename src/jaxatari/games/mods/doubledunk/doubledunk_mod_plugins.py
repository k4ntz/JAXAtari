import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_doubledunk import DunkGameState, PlayerID
from jaxatari.environment import JAXAtariAction as Action
import jax.random as random
import chex
from typing import Tuple

class TimerMod(JaxAtariInternalModPlugin):
    """
    Ends the game after 3600 frames
    instead of the default score limit (24).
    """
    def _get_done(self, state: DunkGameState) -> bool:
        return state.step_counter >= 3600

class SuperDunkMod(JaxAtariPostStepModPlugin):
    """
    Makes dunks (close range shots) worth 5 points.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Detect score change
        p_score_diff = new_state.scores.player - prev_state.scores.player
        e_score_diff = new_state.scores.enemy - prev_state.scores.enemy
        
        # Check distance
        # We need to access basket pos from consts
        basket_x, basket_y = self._env.consts.BASKET_POSITION
        
        sx = prev_state.ball.shooter_pos_x
        sy = prev_state.ball.shooter_pos_y
        
        dist = jnp.sqrt((sx - basket_x)**2 + (sy - basket_y)**2)
        
        # Check if it was a close shot (Dunk range)
        is_close = dist < self._env.consts.DUNK_RADIUS
        
        # Player
        new_p_score = jax.lax.select(
            jnp.logical_and(p_score_diff > 0, is_close),
            prev_state.scores.player + 5,
            new_state.scores.player
        )
        
        # Enemy
        new_e_score = jax.lax.select(
            jnp.logical_and(e_score_diff > 0, is_close),
            prev_state.scores.enemy + 5,
            new_state.scores.enemy
        )
        
        new_scores = new_state.scores.replace(player=new_p_score, enemy=new_e_score)
        return new_state.replace(scores=new_scores)

class TenSecondViolationInternalMod(JaxAtariInternalModPlugin):
    """
    Enforces a 10-second possession limit (CPU Logic Override).
    If CPU holds ball for 150 frames -> Forced to Pass/Shoot.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _handle_player_actions(self, state, action, key):
        from jaxatari.games.jax_doubledunk import DoubleDunk, PlayerID
        from jaxatari.environment import JAXAtariAction as Action
        
        # 1. Run the base game logic first
        actions, new_key, new_timer, new_last_actions = DoubleDunk._handle_player_actions(self._env, state, action, key)
        
        # Extract the actions decided by the base game
        p1_in_act, p1_out_act, p2_in_act, p2_out_act = actions
        
        # 2. Check if the CPU has been holding the ball too long
        force_cpu_action = state.timers.possession > 150
        
        p2_in_holding = (state.ball.holder == PlayerID.PLAYER2_INSIDE)
        p2_out_holding = (state.ball.holder == PlayerID.PLAYER2_OUTSIDE)
        
        # 3. Override the specific CPU player's action with FIRE if penalty is active
        final_p2_in_act = jax.lax.select(
            jnp.logical_and(p2_in_holding, force_cpu_action),
            Action.FIRE,
            p2_in_act
        )
        
        final_p2_out_act = jax.lax.select(
            jnp.logical_and(p2_out_holding, force_cpu_action),
            Action.FIRE,
            p2_out_act
        )
        
        # 4. Return the newly modified actions
        final_actions = (p1_in_act, p1_out_act, final_p2_in_act, final_p2_out_act)
        
        return final_actions, new_key, new_timer, new_last_actions

class TenSecondViolationPostStepMod(JaxAtariPostStepModPlugin):
    """
    Enforces a 10-second possession limit (Player Foul).
    If Player holds ball > 10s -> Turnover (Foul).
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Check if Player has held ball too long
        holder = new_state.ball.holder
        is_player_holding = jnp.logical_or(holder == PlayerID.PLAYER1_INSIDE, holder == PlayerID.PLAYER1_OUTSIDE)
        
        # 10s = 600 frames
        penalty = jnp.logical_and(is_player_holding, new_state.timers.possession > 200)
        
        def apply_turnover(s):
            # Reuse travel timer field to trigger penalty mode
            s = s.replace(
                game_mode=2, # TRAVEL_PENALTY (2)
                timers=s.timers.replace(travel=60, possession=0) 
            )
            # Mark P1 as triggering travel to switch possession
            p1_in = s.player1_inside.replace(triggered_travel=True)
            return s.replace(player1_inside=p1_in)

        final_state = jax.lax.cond(penalty, apply_turnover, lambda s: s, new_state)
        return final_state

class SingleMode(JaxAtariPostStepModPlugin):
    """
    Removes CPU players (opponent team) entirely by placing them permanently off-screen.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        from jaxatari.games.jax_doubledunk import PlayerID

        # Move CPU opponents off-screen
        p2_in = new_state.player2_inside.replace(x=-50, y=0, vel_x=0, vel_y=0, z=0, is_out_of_bounds=True)
        p2_out = new_state.player2_outside.replace(x=-50, y=0, vel_x=0, vel_y=0, z=0, is_out_of_bounds=True)
        
        holder = new_state.ball.holder
        is_p2_holding = jnp.logical_or(holder == PlayerID.PLAYER2_INSIDE, holder == PlayerID.PLAYER2_OUTSIDE)
        
        # If the game tries to give the ball to P2 (e.g. after P1 scores), 
        # instantly give it back to P1 Outside.
        new_ball = jax.lax.cond(
            is_p2_holding,
            lambda b: b.replace(holder=jnp.array(PlayerID.PLAYER1_OUTSIDE, dtype=jnp.int32)),
            lambda b: b,
            new_state.ball
        )
        
        return new_state.replace(
            player2_inside=p2_in,
            player2_outside=p2_out,
            ball=new_ball
        )

class OneVsOneInternalMod(JaxAtariInternalModPlugin):
    """
    Internal mod for 1v1 mode. Overrides passing logic and fixes AI defense calculations.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _handle_passing(self, state, actions):
        from jaxatari.games.jax_doubledunk import OffensiveAction
        
        current_step = state.strategy.offense_step
        pattern = state.strategy.offense_pattern
        is_pass_step = jnp.logical_and(
            current_step < 4, 
            pattern[current_step] == OffensiveAction.PASS
        )
        
        # Skip the pass step
        step_increment = jax.lax.select(is_pass_step, 1, 0)
        return state.ball, step_increment, False

    @partial(jax.jit, static_argnums=(0,))
    def _handle_player_actions(self, state, action, key):
        from jaxatari.games.jax_doubledunk import PlayerID, DoubleDunk
        from jaxatari.environment import JAXAtariAction as Action
        
        # 1. TRICK THE AI: Make the INSIDE ghosts mirror the OUTSIDE active players
        fake_p1_in = state.player1_outside.replace(id=jnp.array(PlayerID.PLAYER1_INSIDE, dtype=jnp.int32))
        fake_p2_in = state.player2_outside.replace(id=jnp.array(PlayerID.PLAYER2_INSIDE, dtype=jnp.int32))
        
        fake_state = state.replace(
            player1_inside=fake_p1_in,
            player2_inside=fake_p2_in
        )
        
        # 2. Run the ORIGINAL AI logic from the base class, bypassing the mod completely
        actions, new_key, new_timer, new_last_actions = DoubleDunk._handle_player_actions(self._env, fake_state, action, key)
        
        # 3. Extract the chosen actions
        p1_in_act, p1_out_act, p2_in_act, p2_out_act = actions
        
        # 4. Force the INSIDE ghost players to NOOP and let the OUTSIDE active players move
        final_actions = (Action.NOOP, p1_out_act, Action.NOOP, p2_out_act)
        
        return final_actions, new_key, new_timer, new_last_actions

class OneVsOnePostMod(JaxAtariPostStepModPlugin):
    """
    Post step mod for 1v1 mode. Keeps INSIDE players off-screen.
    Uses the shorter OUTSIDE Guards to fix the 5-pixel sprite rendering offset.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        from jaxatari.games.jax_doubledunk import PlayerID
        
        # Move INSIDE players permanently off-screen
        p1_in_ghost = new_state.player1_inside.replace(x=-50, y=0, vel_x=0, vel_y=0, z=0, is_out_of_bounds=True)
        p2_in_ghost = new_state.player2_inside.replace(x=-50, y=0, vel_x=0, vel_y=0, z=0, is_out_of_bounds=True)
        
        # Turn over the ball if an inside ghost somehow gets it (Safety check)
        holder = new_state.ball.holder
        is_in_holding = jnp.logical_or(holder == PlayerID.PLAYER1_INSIDE, holder == PlayerID.PLAYER2_INSIDE)
        
        new_ball = jax.lax.cond(
            is_in_holding,
            lambda b: b.replace(holder=PlayerID.NONE, x=80.0, y=100.0, vel_x=0.0, vel_y=0.0),
            lambda b: b,
            new_state.ball
        )
        
        # Ensure human is always controlling the OUTSIDE player
        ctrl_id = new_state.controlled_player_id
        new_ctrl_id = jax.lax.select(
            ctrl_id == PlayerID.PLAYER1_INSIDE,
            jnp.array(PlayerID.PLAYER1_OUTSIDE, dtype=jnp.int32),
            ctrl_id
        )
        
        return new_state.replace(
            player1_inside=p1_in_ghost,
            player2_inside=p2_in_ghost,
            ball=new_ball,
            controlled_player_id=new_ctrl_id
        )

class HalfCourtMod(JaxAtariInternalModPlugin):
    """
    Halves the playable width of the court, making it a much tighter space to play.
    """
    constants_overrides = {
        "PLAYER_Y_MIN": 80,
        "PLAYER_Y_MAX": 160,
        "BASKET_POSITION": (80, 60),
        "AREA_3_POINT": (45, 160, 90),
        "P1_INSIDE_START": (100, 100),
        "P2_INSIDE_START": (50, 110),
    }

    # Override the background to visually match the new boundaries
    asset_overrides = {
        'background': {'name': 'background', 'type': 'background', 'file': 'background_mod.npy'}
    }

class CollisionMod(JaxAtariPostStepModPlugin):
    """
    Offensive foul mod: If the user (player) has the ball and collides with a CPU player,
    it results in an immediate turnover.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        from jaxatari.games.jax_doubledunk import PlayerID

        # Collision distance threshold (squared). 
        # 16 = 4 pixels of distance. 
        COLLISION_DIST_SQ = 16
        
        p1_in = new_state.player1_inside
        p1_out = new_state.player1_outside
        p2_in = new_state.player2_inside
        p2_out = new_state.player2_outside

        # Helper to calculate squared distance
        def get_dist_sq(pA, pB):
            return (pA.x - pB.x)**2 + (pA.y - pB.y)**2

        # Check if P1 Inside holds the ball and is colliding with P2
        p1_in_holding = (new_state.ball.holder == PlayerID.PLAYER1_INSIDE)
        p1_in_collided = jnp.logical_and(
            p1_in_holding,
            jnp.logical_or(
                get_dist_sq(p1_in, p2_in) < COLLISION_DIST_SQ,
                get_dist_sq(p1_in, p2_out) < COLLISION_DIST_SQ
            )
        )

        # Check if P1 Outside holds the ball and is colliding with P2
        p1_out_holding = (new_state.ball.holder == PlayerID.PLAYER1_OUTSIDE)
        p1_out_collided = jnp.logical_and(
            p1_out_holding,
            jnp.logical_or(
                get_dist_sq(p1_out, p2_in) < COLLISION_DIST_SQ,
                get_dist_sq(p1_out, p2_out) < COLLISION_DIST_SQ
            )
        )

        is_offensive_foul = jnp.logical_or(p1_in_collided, p1_out_collided)

        def apply_turnover(s):
            # Repurpose the TRAVEL_PENALTY (GameMode 2) to freeze the game
            # and automatically switch possession to the CPU
            s = s.replace(
                game_mode=2,
                timers=s.timers.replace(travel=60)
            )
            
            # Flag the specific player so the reset logic switches possession properly
            new_p1_in = s.player1_inside.replace(
                triggered_travel=jax.lax.select(p1_in_collided, True, s.player1_inside.triggered_travel)
            )
            new_p1_out = s.player1_outside.replace(
                triggered_travel=jax.lax.select(p1_out_collided, True, s.player1_outside.triggered_travel)
            )
            
            return s.replace(player1_inside=new_p1_in, player1_outside=new_p1_out)

        # Only apply the penalty if the game is currently IN_PLAY (GameMode 1)
        should_apply = jnp.logical_and(is_offensive_foul, new_state.game_mode == 1)

        return jax.lax.cond(should_apply, apply_turnover, lambda s: s, new_state)

class PenaltyMod(JaxAtariPostStepModPlugin):
    """
    Hardcore Mod: Awards 1 point to the enemy team instantly 
    whenever the player triggers a turnover penalty (Travel, Out of Bounds, Clearance).
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Check if the game was just in standard play on the last frame
        was_in_play = (prev_state.game_mode == 1) # GameMode.IN_PLAY
        
        # Check if the game has transitioned into any penalty freeze mode this frame
        is_now_penalty = jnp.logical_or(
            new_state.game_mode == 2, # TRAVEL_PENALTY
            jnp.logical_or(
                new_state.game_mode == 3, # OUT_OF_BOUNDS_PENALTY
                new_state.game_mode == 4  # CLEARANCE_PENALTY
            )
        )
        
        # Trigger is true ONLY on the exact frame the penalty is called
        just_penalized = jnp.logical_and(was_in_play, is_now_penalty)
        
        # Award a 1-point goal to the enemy
        new_enemy_score = jax.lax.select(
            just_penalized,
            new_state.scores.enemy + 1, 
            new_state.scores.enemy
        )
        
        # Update the state with the new score
        new_scores = new_state.scores.replace(enemy=new_enemy_score)
        return new_state.replace(scores=new_scores)