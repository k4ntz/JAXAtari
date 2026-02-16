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
    Ends the game after 1 minute (3600 frames at 60fps)
    instead of the default score limit (24).
    """
    def _get_done(self, state: DunkGameState) -> bool:
        # 1 minute = 60 seconds * 60 frames = 3600 frames
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
    If CPU holds ball > 8.3s -> Forced to Pass/Shoot.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _handle_player_actions(self, state: DunkGameState, action: int, key: chex.PRNGKey) -> Tuple[Tuple[int, ...], chex.PRNGKey, chex.Array, chex.Array]:
        # --- RE-IMPLEMENTATION OF BASE LOGIC + MODIFICATION ---
        
        def get_move_to_target(current_x, current_y, target_x, target_y, threshold=10):
            dx = target_x - current_x
            dy = target_y - current_y
            want_right = dx > threshold
            want_left  = dx < -threshold
            want_down  = dy > threshold
            want_up    = dy < -threshold
            action = Action.NOOP
            action = jax.lax.select(want_up, Action.UP, action)
            action = jax.lax.select(want_down, Action.DOWN, action)
            action = jax.lax.select(want_left, Action.LEFT, action)
            action = jax.lax.select(want_right, Action.RIGHT, action)
            action = jax.lax.select(jnp.logical_and(want_up, want_left), Action.UPLEFT, action)
            action = jax.lax.select(jnp.logical_and(want_up, want_right), Action.UPRIGHT, action)
            action = jax.lax.select(jnp.logical_and(want_down, want_left), Action.DOWNLEFT, action)
            action = jax.lax.select(jnp.logical_and(want_down, want_right), Action.DOWNRIGHT, action)
            return action

        # --- Human Control ---
        is_p1_inside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_INSIDE)
        is_p1_outside_controlled = (state.controlled_player_id == PlayerID.PLAYER1_OUTSIDE)
        p1_inside_action = jax.lax.select(is_p1_inside_controlled, action, Action.NOOP)
        p1_outside_action = jax.lax.select(is_p1_outside_controlled, action, Action.NOOP)

        # --- AI Logic Setup ---
        key, p2_action_key, teammate_action_key = random.split(key, 3)
        p2_inside_prob_key, p2_outside_prob_key, p2_inside_move_key, p2_outside_move_key = random.split(p2_action_key, 4)
        movement_actions = jnp.array([
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.UPLEFT, Action.UPRIGHT, Action.DOWNLEFT, Action.DOWNRIGHT
        ])

        # --- Strategy & Possession ---
        p1_has_ball = jnp.logical_or((state.ball.holder == PlayerID.PLAYER1_INSIDE), (state.ball.holder == PlayerID.PLAYER1_OUTSIDE))
        p2_has_ball = jnp.logical_or((state.ball.holder == PlayerID.PLAYER2_INSIDE), (state.ball.holder == PlayerID.PLAYER2_OUTSIDE))
        
        defensive_strat = state.strategy.defense_pattern
        basket_x, basket_y = self._env.consts.BASKET_POSITION

        # Helper to select target based on strategy
        def select_def_target(strat, t_lane, t_tight, t_pass, t_pick, t_reb):
            from jaxatari.games.jax_doubledunk import DefensiveStrategy
            return jax.lax.select(
                strat == DefensiveStrategy.LANE_DEFENSE, t_lane,
                jax.lax.select(
                    strat == DefensiveStrategy.TIGHT_DEFENSE, t_tight,
                    jax.lax.select(
                        strat == DefensiveStrategy.PASS_DEFENSE, t_pass,
                        jax.lax.select(
                            strat == DefensiveStrategy.PICK_DEFENSE, t_pick,
                            t_reb 
                        )
                    )
                )
            )

        # P1 Defensive Targets
        p1_in_t_tight_x, p1_in_t_tight_y = state.player2_inside.x, state.player2_inside.y
        p1_out_t_tight_x, p1_out_t_tight_y = state.player2_outside.x, state.player2_outside.y
        
        p1_in_t_lane_x = (state.player2_inside.x + state.player2_outside.x) // 2
        p1_in_t_lane_y = (state.player2_inside.y + state.player2_outside.y) // 2
        p1_out_t_lane_x, p1_out_t_lane_y = p1_out_t_tight_x, p1_out_t_tight_y
        
        mid_p2_x = (state.player2_inside.x + state.player2_outside.x) // 2
        mid_p2_y = (state.player2_inside.y + state.player2_outside.y) // 2
        p1_in_t_pass_x, p1_in_t_pass_y = mid_p2_x, mid_p2_y
        p1_out_t_pass_x, p1_out_t_pass_y = mid_p2_x, mid_p2_y
        
        dist_p2 = jnp.sqrt((state.player2_inside.x - state.player2_outside.x)**2 + (state.player2_inside.y - state.player2_outside.y)**2)
        switch_p1 = dist_p2 < 15
        p1_in_t_pick_x = jax.lax.select(switch_p1, state.player2_outside.x, state.player2_inside.x)
        p1_in_t_pick_y = jax.lax.select(switch_p1, state.player2_outside.y, state.player2_inside.y)
        p1_out_t_pick_x = jax.lax.select(switch_p1, state.player2_inside.x, state.player2_outside.x)
        p1_out_t_pick_y = jax.lax.select(switch_p1, state.player2_inside.y, state.player2_outside.y)
        
        p1_in_t_reb_x, p1_in_t_reb_y = basket_x, basket_y + 10
        p1_out_t_reb_x, p1_out_t_reb_y = p1_out_t_tight_x, p1_out_t_tight_y

        p1_in_def_x = select_def_target(defensive_strat, p1_in_t_lane_x, p1_in_t_tight_x, p1_in_t_pass_x, p1_in_t_pick_x, p1_in_t_reb_x)
        p1_in_def_y = select_def_target(defensive_strat, p1_in_t_lane_y, p1_in_t_tight_y, p1_in_t_pass_y, p1_in_t_pick_y, p1_in_t_reb_y)
        p1_out_def_x = select_def_target(defensive_strat, p1_out_t_lane_x, p1_out_t_tight_x, p1_out_t_pass_x, p1_out_t_pick_x, p1_out_t_reb_x)
        p1_out_def_y = select_def_target(defensive_strat, p1_out_t_lane_y, p1_out_t_tight_y, p1_out_t_pass_y, p1_out_t_pick_y, p1_out_t_reb_y)
        
        p1_in_def_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, p1_in_def_x, p1_in_def_y)
        p1_out_def_action = get_move_to_target(state.player1_outside.x, state.player1_outside.y, p1_out_def_x, p1_out_def_y)

        # P2 Defensive Targets
        p2_in_t_tight_x, p2_in_t_tight_y = state.player1_inside.x, state.player1_inside.y
        p2_out_t_tight_x, p2_out_t_tight_y = state.player1_outside.x, state.player1_outside.y

        p2_in_t_lane_x = (state.player1_inside.x + state.player1_outside.x) // 2
        p2_in_t_lane_y = (state.player1_inside.y + state.player1_outside.y) // 2
        p2_out_t_lane_x, p2_out_t_lane_y = p2_out_t_tight_x, p2_out_t_tight_y

        mid_p1_x = (state.player1_inside.x + state.player1_outside.x) // 2
        mid_p1_y = (state.player1_inside.y + state.player1_outside.y) // 2
        p2_in_t_pass_x, p2_in_t_pass_y = mid_p1_x, mid_p1_y
        p2_out_t_pass_x, p2_out_t_pass_y = mid_p1_x, mid_p1_y

        dist_p1 = jnp.sqrt((state.player1_inside.x - state.player1_outside.x)**2 + (state.player1_inside.y - state.player1_outside.y)**2)
        switch_p2 = dist_p1 < 15
        p2_in_t_pick_x = jax.lax.select(switch_p2, state.player1_outside.x, state.player1_inside.x)
        p2_in_t_pick_y = jax.lax.select(switch_p2, state.player1_outside.y, state.player1_inside.y)
        p2_out_t_pick_x = jax.lax.select(switch_p2, state.player1_inside.x, state.player1_outside.x)
        p2_out_t_pick_y = jax.lax.select(switch_p2, state.player1_inside.y, state.player1_outside.y)

        p2_in_t_reb_x, p2_in_t_reb_y = basket_x, basket_y + 10
        p2_out_t_reb_x, p2_out_t_reb_y = p2_out_t_tight_x, p2_out_t_tight_y

        p2_in_def_x = select_def_target(defensive_strat, p2_in_t_lane_x, p2_in_t_tight_x, p2_in_t_pass_x, p2_in_t_pick_x, p2_in_t_reb_x)
        p2_in_def_y = select_def_target(defensive_strat, p2_in_t_lane_y, p2_in_t_tight_y, p2_in_t_pass_y, p2_in_t_pick_y, p2_in_t_reb_y)
        p2_out_def_x = select_def_target(defensive_strat, p2_out_t_lane_x, p2_out_t_tight_x, p2_out_t_pass_x, p2_out_t_pick_x, p2_out_t_reb_x)
        p2_out_def_y = select_def_target(defensive_strat, p2_out_t_lane_y, p2_out_t_tight_y, p2_out_t_pass_y, p2_out_t_pick_y, p2_out_t_reb_y)

        p2_in_def_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, p2_in_def_x, p2_in_def_y)
        p2_out_def_action = get_move_to_target(state.player2_outside.x, state.player2_outside.y, p2_out_def_x, p2_out_def_y)

        # --- Teammate & AI Movement ---
        from jaxatari.games.jax_doubledunk import OffensiveAction
        
        is_p1_inside_teammate_ai = jnp.logical_not(is_p1_inside_controlled)
        is_p1_outside_teammate_ai = jnp.logical_not(is_p1_outside_controlled)

        prev_step = state.strategy.offense_step - 1
        prev_action_was_move = jnp.logical_and((prev_step >= 0), (state.strategy.offense_pattern[prev_step] == OffensiveAction.MOVE_TO_POST))
        
        play_dir = state.strategy.play_direction
        post_x = jax.lax.select(play_dir == -1, 40, jax.lax.select(play_dir == 1, 120, 80))
        post_y = 60
        
        p1_inside_dist_to_post = jnp.sqrt((state.player1_inside.x - post_x)**2 + (state.player1_inside.y - post_y)**2)
        p1_at_post = p1_inside_dist_to_post < 5
        
        should_move_to_post = jnp.logical_and(prev_action_was_move, jnp.logical_not(p1_at_post))
        p1_move_to_post_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, post_x, post_y)

        random_teammate_move_idx = random.randint(teammate_action_key, shape=(), minval=0, maxval=8)
        random_teammate_move_action = movement_actions[random_teammate_move_idx]
        rand_teammate = random.uniform(teammate_action_key)
        
        p1_off_action = jax.lax.select(rand_teammate < 0.5, Action.NOOP, random_teammate_move_action)

        ball_is_free = (state.ball.holder == PlayerID.NONE)
        p1_in_chase = get_move_to_target(state.player1_inside.x, state.player1_inside.y, state.ball.x, state.ball.y, 2)
        p1_out_chase = get_move_to_target(state.player1_outside.x, state.player1_outside.y, state.ball.x, state.ball.y, 2)
        p2_in_chase = get_move_to_target(state.player2_inside.x, state.player2_inside.y, state.ball.x, state.ball.y, 2)
        p2_out_chase = get_move_to_target(state.player2_outside.x, state.player2_outside.y, state.ball.x, state.ball.y, 2)

        p1_dist_to_basket_x = jnp.abs(state.player1_inside.x - basket_x)
        p1_dist_to_basket_y = jnp.abs(state.player1_inside.y - basket_y)
        p1_is_far = jnp.logical_or((p1_dist_to_basket_x > 20), (p1_dist_to_basket_y > 80)) # 40x80 area
        p1_return_to_basket_action = get_move_to_target(state.player1_inside.x, state.player1_inside.y, basket_x, basket_y)
        
        p1_inside_action = jax.lax.select(
            is_p1_inside_teammate_ai,
            jax.lax.select(
                ball_is_free,
                p1_in_chase,
                jax.lax.select(
                    p2_has_ball, 
                    p1_in_def_action, 
                    jax.lax.select(
                        should_move_to_post,
                        p1_move_to_post_action,
                        jax.lax.select(p1_is_far, p1_return_to_basket_action, p1_off_action)
                    )
                )
            ),
            p1_inside_action
        )
        p1_outside_action = jax.lax.select(
            is_p1_outside_teammate_ai,
            jax.lax.select(
                ball_is_free,
                p1_out_chase,
                jax.lax.select(p2_has_ball, p1_out_def_action, p1_off_action)
            ),
            p1_outside_action
        )

        # --- P2 AI Logic ---
        p2_inside_has_ball = (state.ball.holder == PlayerID.PLAYER2_INSIDE)
        rand_inside = random.uniform(p2_inside_prob_key)
        random_inside_move_idx = random.randint(p2_inside_move_key, shape=(), minval=0, maxval=8)
        random_inside_move_action = movement_actions[random_inside_move_idx]
        action_if_ball_inside = jax.lax.select(rand_inside < 0.2, Action.FIRE, random_inside_move_action)
        
        p2_dist_to_basket_x = jnp.abs(state.player2_inside.x - basket_x)
        p2_dist_to_basket_y = jnp.abs(state.player2_inside.y - basket_y)
        p2_is_far = jnp.logical_or((p2_dist_to_basket_x > 20), (p2_dist_to_basket_y > 40)) 
        p2_return_to_basket_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, basket_x, basket_y)

        p2_in_off_action = jax.lax.select(
            p2_inside_has_ball, 
            action_if_ball_inside, 
            jax.lax.select(
                p2_is_far, 
                p2_return_to_basket_action, 
                jax.lax.select(rand_inside < 0.5, Action.NOOP, random_inside_move_action)
            )
        )

        p2_outside_has_ball = (state.ball.holder == PlayerID.PLAYER2_OUTSIDE)
        rand_outside = random.uniform(p2_outside_prob_key)
        random_outside_move_idx = random.randint(p2_outside_move_key, shape=(), minval=0, maxval=8)
        random_outside_move_action = movement_actions[random_outside_move_idx]
        action_if_ball_outside = jax.lax.select(rand_outside < 0.2, Action.FIRE, random_outside_move_action)
        p2_out_off_action = jax.lax.select(p2_outside_has_ball, action_if_ball_outside, jax.lax.select(rand_outside < 0.5, Action.NOOP, random_outside_move_action))

        # --- MODIFICATION: FORCE CPU TO SHOOT/PASS IF TIMER IS HIGH ---
        force_cpu_action = state.timers.possession > 150
        
        p2_in_off_action = jax.lax.select(
            jnp.logical_and(p2_inside_has_ball, force_cpu_action),
            Action.FIRE,
            p2_in_off_action
        )
        
        p2_out_off_action = jax.lax.select(
            jnp.logical_and(p2_outside_has_ball, force_cpu_action),
            Action.FIRE,
            p2_out_off_action
        )
        # -------------------------------------------------------------

        # P2 Clearance Override
        def get_smart_clearance_target(px, py):
            px_f = px.astype(jnp.float32)
            py_f = py.astype(jnp.float32)
            dist_left = px_f - 25.0
            dist_right = 135.0 - px_f
            dx = px_f - 80.0
            term = 1.0 - (dx / 55.0)**2
            valid_term = jnp.maximum(0.0, term)
            boundary_y = 80.0 + 45.0 * jnp.sqrt(valid_term)
            dist_down = boundary_y - py_f
            go_left = jnp.logical_and((dist_left < dist_right), (dist_left < dist_down))
            go_right = jnp.logical_and((dist_right <= dist_left), (dist_right < dist_down))
            tx = jax.lax.select(go_left, 20.0, jax.lax.select(go_right, 140.0, px_f))
            ty = jax.lax.select(jnp.logical_or(go_left, go_right), py_f, boundary_y + 10.0)
            return tx.astype(jnp.int32), ty.astype(jnp.int32)

        p2_in_tx, p2_in_ty = get_smart_clearance_target(state.player2_inside.x, state.player2_inside.y)
        p2_in_clearance_action = get_move_to_target(state.player2_inside.x, state.player2_inside.y, p2_in_tx, p2_in_ty)
        p2_in_final_off = jax.lax.select(state.player2_inside.clearance_needed, p2_in_clearance_action, p2_in_off_action)

        p2_out_tx, p2_out_ty = get_smart_clearance_target(state.player2_outside.x, state.player2_outside.y)
        p2_out_clearance_action = get_move_to_target(state.player2_outside.x, state.player2_outside.y, p2_out_tx, p2_out_ty)
        p2_out_final_off = jax.lax.select(state.player2_outside.clearance_needed, p2_out_clearance_action, p2_out_off_action)

        p2_inside_action = jax.lax.select(
            ball_is_free,
            p2_in_chase,
            jax.lax.select(p1_has_ball, p2_in_def_action, p2_in_final_off)
        )
        
        p2_outside_action = jax.lax.select(
            ball_is_free,
            p2_out_chase,
            jax.lax.select(p1_has_ball, p2_out_def_action, p2_out_final_off)
        )

        use_last_action = state.timers.enemy_reaction > 0
        final_p2_inside_action = jax.lax.select(use_last_action, state.strategy.last_enemy_actions[0], p2_inside_action)
        final_p2_outside_action = jax.lax.select(use_last_action, state.strategy.last_enemy_actions[1], p2_outside_action)
        new_timer = jax.lax.select(use_last_action, state.timers.enemy_reaction - 1, 6)
        new_last_actions = jax.lax.select(use_last_action, state.strategy.last_enemy_actions, jnp.array([p2_inside_action, p2_outside_action]))

        def check_p1_jump_condition(pid, p_state, p_act):
            has_ball = (state.ball.holder == pid)
            is_in_air = (p_state.z > 0)
            is_fire_variant = jnp.logical_or(p_act == Action.FIRE, jnp.logical_and(p_act >= Action.UPFIRE, p_act <= Action.DOWNLEFTFIRE))
            is_starting_jump = jnp.logical_and((p_state.z == 0), is_fire_variant)
            return jnp.logical_and(has_ball, jnp.logical_or(is_in_air, is_starting_jump))

        p1_in_jump = check_p1_jump_condition(PlayerID.PLAYER1_INSIDE, state.player1_inside, p1_inside_action)
        p1_out_jump = check_p1_jump_condition(PlayerID.PLAYER1_OUTSIDE, state.player1_outside, p1_outside_action)

        final_p2_inside_action = jax.lax.select(p1_in_jump, Action.FIRE, final_p2_inside_action)
        final_p2_outside_action = jax.lax.select(p1_out_jump, Action.FIRE, final_p2_outside_action)

        def check_p2_shoot(pid, p_state):
             return jnp.logical_and((state.ball.holder == pid), (p_state.z > 0))

        p2_in_shoot = check_p2_shoot(PlayerID.PLAYER2_INSIDE, state.player2_inside)
        p2_out_shoot = check_p2_shoot(PlayerID.PLAYER2_OUTSIDE, state.player2_outside)

        final_p2_inside_action = jax.lax.select(p2_in_shoot, Action.FIRE, final_p2_inside_action)
        final_p2_outside_action = jax.lax.select(p2_out_shoot, Action.FIRE, final_p2_outside_action)

        actions = (p1_inside_action, p1_outside_action, final_p2_inside_action, final_p2_outside_action)
        return actions, key, new_timer, new_last_actions

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