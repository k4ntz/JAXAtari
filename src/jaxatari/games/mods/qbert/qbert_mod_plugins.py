import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariPostStepModPlugin

class NoRedBallsMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(red_ball_positions=jnp.full((3, 2), -1, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoPurpleBallMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(purple_ball_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoCoilyMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(snake_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoGreenBallMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(green_ball_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoSamMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(sam_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoEnemiesMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            red_ball_positions=jnp.full((3, 2), -1, dtype=jnp.int32),
            purple_ball_position=jnp.array([-1, -1], dtype=jnp.int32),
            snake_position=jnp.array([-1, -1], dtype=jnp.int32),
            green_ball_position=jnp.array([-1, -1], dtype=jnp.int32),
            sam_position=jnp.array([-1, -1], dtype=jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state
from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.environment import JAXAtariAction as Action
from typing import Tuple
import chex
from jaxatari.games.jax_qbert import QbertState, QbertObservation, QbertInfo

class DiagonalControlMod(JaxAtariInternalModPlugin):
    attribute_overrides = {
        "ACTION_SET": jnp.array([
            Action.NOOP,
            Action.FIRE,
            Action.UPRIGHT,
            Action.DOWNRIGHT,
            Action.UPLEFT,
            Action.DOWNLEFT,
        ], dtype=jnp.int32),
        "action_mapping": jnp.array([
            [0, 0],   # 0
            [0, 0],   # 1
            [0, 0],   # 2 (not used)
            [0, 0],   # 3 (not used)
            [0, 0],   # 4 (not used)
            [0, 0],   # 5 (not used)
            [0, -1],  # 6: UPRIGHT -> was UP
            [-1, -1], # 7: UPLEFT -> was LEFT
            [1, 1],   # 8: DOWNRIGHT -> was RIGHT
            [0, 1],   # 9: DOWNLEFT -> was DOWN
        ], dtype=jnp.int32)
    }

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: QbertState, action: chex.Array) -> Tuple[QbertObservation, QbertState, float, bool, QbertInfo]:
        action = jnp.take(self._env.ACTION_SET, action.astype(jnp.int32))

        # Handle player movement
        tick_counter_reset = jnp.array([31, 227, 144, 124, 110, 95, 81, 66, 52]).astype(jnp.int32)
        is_player_moving = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_or(jnp.logical_and(state.player_moving_counter == 0, action != Action.NOOP), (state.player_moving_counter + 1) % tick_counter_reset[state.player_position_category] > 1))), 1, 0)
        player_moving_counter = jnp.where(state.is_player_moving == 1, (state.player_moving_counter + 1) % tick_counter_reset[state.player_position_category], state.player_moving_counter)
        player_last_position = jnp.where(jnp.logical_or(state.dead_animation_counter != 0, jnp.logical_or(state.next_round_animation_counter != 0, player_moving_counter != 0)), state.player_last_position, state.player_position)
        player_position = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action != Action.NOOP))), self._env.move(state, action, state.player_position), state.player_position)
        player_direction = jnp.select(
            condlist=[
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.DOWNRIGHT))),
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.UPLEFT))),
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.UPRIGHT))),
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.DOWNLEFT)))
            ],
            choicelist=[2, 0, 3, 1],
            default=state.player_direction
        ).astype(jnp.int32)

        player_position_category = jnp.where(player_moving_counter == 0, jnp.select(
            condlist=[
                state.pyramid[player_position[1]][player_position[0]] >= 0,
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -2, player_position[1] == 4),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -2, player_position[1] == 2),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 0),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 1),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 2),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 3),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 4),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 5),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 7),
            ],
            choicelist=[0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            default=0,
        ), state.player_position_category)

        pyramid,player_position,lives, trash=self._env.checkField(state,0,0,player_position)

        spawned=jax.lax.cond(
            pred=jnp.logical_or(player_position[0] != 1, player_position[1] != 1),
            true_fun=lambda i: 0,
            false_fun=lambda i: i,
            operand=state.just_spawned
        )

        # Increase enemy moving counter depending on current level
        enemy_moving_counter = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, state.step_counter == state.green_ball_freeze_step)), (state.enemy_moving_counter + 1) % self._env._enemy_move_tick[state.level_number], state.enemy_moving_counter)

        # Handle red ball movement
        red_pos = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)), self._env.move_red_balls(state), state.red_ball_positions)
        trash1, red0_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self._env.checkField(s, 1, 0, red_pos[0]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[0], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        trash1, red1_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self._env.checkField(s, 1, 1, red_pos[1]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[1], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        trash1, red2_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self._env.checkField(s, 1, 2, red_pos[2]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[2], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        # Handle purple ball movement
        trash1, purple_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self._env.checkField(s, 2, 0, self._env.move_purple_ball(s)),
            false_fun=lambda s: (state.pyramid, state.purple_ball_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        temp = jnp.abs(player_position[0] - player_last_position[0]) + jnp.abs(player_position[1] - player_last_position[1])

        snake_lock=jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(jnp.logical_and(temp > 2, jnp.abs(state.snake_position[0] - player_last_position[0]) + jnp.abs(state.snake_position[1] - player_last_position[1]) < 4) , state.snake_lock[0] == -1), state.snake_position[0] != -1),
            true_fun=lambda i: jax.lax.cond(
                pred=i.player_last_position[0] > 1,
                true_fun=lambda j: jnp.array([j.player_last_position[0], j.player_last_position[1] - 1]),
                false_fun=lambda j: jnp.array([j.player_last_position[0] - 1, j.player_last_position[1] - 1]),
                operand=i
            ),
            false_fun=lambda i: i.snake_lock,
            operand=state
        )

        # When player is on a disk (category 1 or 2), Coily should target the disk cell so he jumps off.
        disk_target = jnp.where(
            player_position_category == 1,
            jnp.where(player_last_position[0] < 3, jnp.array([0, 4], dtype=jnp.int32), jnp.array([5, 4], dtype=jnp.int32)),
            jnp.where(
                player_position_category == 2,
                jnp.where(player_last_position[0] < 2, jnp.array([0, 2], dtype=jnp.int32), jnp.array([3, 2], dtype=jnp.int32)),
                player_position,
            ),
        )
        target = jnp.where(
            jnp.logical_or(player_position_category == 1, player_position_category == 2),
            disk_target,
            jax.lax.cond(
                pred=snake_lock[0] == -1,
                true_fun=lambda i: i[0],
                false_fun=lambda i: i[1],
                operand=(player_position, snake_lock),
            ),
        )

        # Handle snake movement
        trash1, snake_pos, trash2, points4 = jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)), state.snake_position[0] != -1),
            true_fun=lambda s: self._env.checkField(s[0], 3, 0, self._env.move_snake(s[0], s[1])),
            false_fun=lambda s: ((s[0].pyramid, s[0].snake_position, s[0].lives, jnp.array(0).astype(jnp.int32))),
            operand=(state, target),
        )


        # Handle green ball movement
        trash1, green_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self._env.checkField(s, 4, 0, self._env.move_green_ball(s)),
            false_fun=lambda s: (state.pyramid, state.green_ball_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        # Handle sam movement
        trash1, sam_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self._env.checkField(s, 5, 0, self._env.move_sam(s)),
            false_fun=lambda s: (state.pyramid, state.sam_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        snake_lock=jax.lax.cond(
            pred=snake_pos[0]==-1,
            true_fun=lambda i: jnp.array([-1,-1]),
            false_fun=lambda i: i,
            operand=snake_lock
        )

        tmp_lives = lives

        # Checks for collisions of the player with the enemies
        lives, red_pos, purple_pos, green_pos, snake_pos, sam_pos, points1, green_ball_freeze_step, same_cell_frames = self._env.checkCollisions(
            lives,
            jnp.where(jnp.logical_and(player_position_category != 1, player_position_category != 2), player_last_position, player_position),
            jnp.array([red0_pos, red1_pos, red2_pos]), purple_pos, green_pos, snake_pos, sam_pos,
            state.green_ball_freeze_step,
            state.same_cell_frames
        )

        # Increase dead_animation_counter if player lost a live
        dead_animation_counter = jnp.where(jnp.logical_or(state.dead_animation_counter != 0, tmp_lives != lives), (state.dead_animation_counter + 1) % 128, state.dead_animation_counter)

        # Changes the colors of the pyramid depending on the player position and the position of sam
        pyramid, points2 = self._env.changeColors(state, player_position, sam_pos, pyramid, state.level_number, spawned)

        # Spawn new enemies: only after FIRST_SPAWN_DELAY and when SPAWN_INTERVAL has elapsed (or first spawn).
        spawn_allowed = (
            jnp.logical_and(state.dead_animation_counter == 0,
            jnp.logical_and(state.next_round_animation_counter == 0,
            jnp.logical_and(enemy_moving_counter == 0,
            jnp.logical_and(state.step_counter >= self._env.consts.FIRST_SPAWN_DELAY,
            jnp.logical_or(state.step_counter - state.last_spawn_step >= self._env.consts.SPAWN_INTERVAL, state.last_spawn_step == 0)))))
        )
        before_red = jnp.array(red_pos)
        before_purple = jnp.array(purple_pos)
        before_green = jnp.array(green_pos)
        before_sam = jnp.array(sam_pos)
        red_pos, purple_pos, green_pos, sam_pos = jax.lax.cond(
            pred=spawn_allowed,
            true_fun=lambda op: self._env.spawnCreatures(op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7], op[8], op[9]),
            false_fun=lambda op: (op[1], op[2], op[3], op[4]),
            operand=(
                jnp.array([state.prng_state[7], state.prng_state[8]]),
                jnp.array(red_pos),
                jnp.array(purple_pos),
                jnp.array(green_pos),
                jnp.array(sam_pos),
                jnp.array(snake_pos),
                self._env.consts.DIFFICULTY,
                state.step_counter,
                state.enemy_spawn_count,
                state.level_number,
            ),
        )
        red_pos_arr = jnp.array(red_pos)
        spawned_this_tick = (jnp.any(red_pos_arr != before_red) | jnp.any(jnp.array(purple_pos) != before_purple) | jnp.any(jnp.array(green_pos) != before_green) | jnp.any(jnp.array(sam_pos) != before_sam))
        last_spawn_step = jnp.where(spawned_this_tick, state.step_counter, state.last_spawn_step)

        # Calculates values at the end of the round
        pyramid, round, level, points3, player_position, spawned, green_ball_freeze_step = jax.lax.cond(
            pred=jnp.logical_and(is_player_moving == 0, player_moving_counter == 0),
            true_fun=lambda op: self._env.nextRound(op[0], op[1], op[2], op[3], op[4], op[5], op[6]),
            false_fun=lambda op: (op[1], op[2], op[3], 0, op[4], op[5], op[6]),
            operand=(state, pyramid, state.round_number, state.level_number, player_position, spawned, green_ball_freeze_step)
        )

        # Increases next_round_animation_counter if end of round is reached
        next_round_animation_counter = jnp.where(jnp.logical_or(state.next_round_animation_counter != 0, round != state.round_number), (state.next_round_animation_counter + 1) % 160, state.next_round_animation_counter)

        # Gains an extra live at the end of a round
        lives=jnp.where(jnp.logical_and(jnp.logical_and(is_player_moving == 0, player_moving_counter == 0), round != state.round_number), self._env.extraLives(round,level,lives), lives)

        # Updates last pyramid
        last_pyramid = jnp.where(jnp.logical_or(state.next_round_animation_counter != 0, jnp.logical_or(jnp.logical_and(jnp.logical_or(player_position_category == 1, player_position_category == 2), player_moving_counter == 45), jnp.logical_and(jnp.logical_and(player_position_category != 1, player_position_category != 2), player_moving_counter == 27))), state.pyramid, state.last_pyramid)

        red_pos, purple_pos, green_pos, snake_pos, sam_pos = jax.lax.cond(
            pred=jnp.logical_or(round != state.round_number, lives < state.lives),
            true_fun=lambda c: (jnp.array([[-1, -1], [-1, -1], [-1, -1]]).astype(jnp.int32), jnp.array([-1, -1]), jnp.array([-1, -1]), jnp.array([-1, -1]), jnp.array([-1, -1])),
            false_fun=lambda c: (c[0], c[1], c[2], c[3], c[4]),
            operand=(red_pos, purple_pos, green_pos, snake_pos, sam_pos)
        )
        same_cell_frames = jnp.where(jnp.logical_or(round != state.round_number, lives < state.lives), 0, same_cell_frames)
        last_spawn_step = jnp.where(round != state.round_number, 0, last_spawn_step)

        # Track how many enemies have spawned in the current level:
        # increment when something spawned this tick, reset when level changes.
        enemy_spawn_count = jnp.where(spawned_this_tick, state.enemy_spawn_count + 1, state.enemy_spawn_count)
        enemy_spawn_count = jnp.where(level != state.level_number, jnp.array(0, dtype=jnp.int32), enemy_spawn_count)

        new_state = state.replace(
            player_score=state.player_score + points1 + points2 + points3 + points4,
            lives=lives,
            pyramid=pyramid,
            last_pyramid=last_pyramid,
            player_position=player_position,
            player_last_position=player_last_position,
            player_direction=player_direction,
            player_position_category=player_position_category,
            is_player_moving=is_player_moving,
            player_moving_counter=player_moving_counter,
            level_number=level,
            round_number=round,
            green_ball_freeze_step=jnp.where(state.step_counter == green_ball_freeze_step, green_ball_freeze_step + 1, green_ball_freeze_step),
            red_ball_positions=red_pos,
            enemy_moving_counter=enemy_moving_counter,
            purple_ball_position=purple_pos,
            snake_position=snake_pos,
            green_ball_position=green_pos,
            sam_position=sam_pos,
            step_counter=state.step_counter + 1,
            prng_state=jax.random.split(state.prng_state[9], 10),
            just_spawned=spawned,
            snake_lock=jnp.array([snake_lock[0], snake_lock[1]]),
            same_cell_frames=same_cell_frames,
            last_spawn_step=last_spawn_step,
            enemy_spawn_count=enemy_spawn_count,
            next_round_animation_counter=next_round_animation_counter,
            dead_animation_counter=dead_animation_counter,
        )

        done = self._env._get_done(new_state)
        env_reward = self._env._get_reward(state, new_state)
        info = self._env._get_info(new_state)
        observation = self._env._get_observation(new_state)

        return observation, new_state, env_reward, done, info

class AlternatingColorsMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: QbertState) -> jnp.ndarray:
        jr = self._env.renderer.jr
        M = self._env.renderer.SHAPE_MASKS

        raster = jnp.where(state.step_counter == state.green_ball_freeze_step, self._env.renderer.BACKGROUND, self._env.renderer.FREEZE_BG)

        round_idx = jnp.where(state.next_round_animation_counter != 0, state.round_number - 2, state.round_number - 1)
        raster = jr.render_at(raster, 68, 40, M['cube_shadow_right'][round_idx])

        raster = jr.render_at(raster, 56, 69, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 80, 69, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 44, 98, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 68, 98, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 92, 98, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 32, 127, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 56, 127, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 80, 127, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 104, 127, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 20, 156, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 44, 156, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 68, 156, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 92, 156, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 116, 156, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 8, 185, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 32, 185, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 56, 185, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 80, 185, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 104, 185, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 128, 185, M['cube_shadow_left'][round_idx])

        raster = jnp.where(state.last_pyramid[2][0] == -2, jr.render_at(raster, 38, 84, M['disc']), raster)
        raster = jnp.where(state.last_pyramid[2][3] == -2, jr.render_at(raster, 110, 84, M['disc']), raster)
        raster = jnp.where(state.last_pyramid[4][0] == -2, jr.render_at(raster, 14, 142, M['disc']), raster)
        raster = jnp.where(state.last_pyramid[4][5] == -2, jr.render_at(raster, 134, 142, M['disc']), raster)

        pyra = state.pyramid.at[state.player_position[1], state.player_position[0]].set(state.last_pyramid[state.player_position[1]][state.player_position[0]])
        raster = jax.lax.fori_loop(
            lower=1,
            upper=7,
            body_fun=lambda i, val: jax.lax.fori_loop(
                lower=1,
                upper=i + 1,
                body_fun=lambda j, val2: jnp.select(
                    condlist=[
                        state.next_round_animation_counter != 0,
                        pyra[i, j] == 0,
                        pyra[i, j] == 1,
                        pyra[i, j] == 2,
                    ],
                    choicelist=[
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['win_animation'][jnp.floor(state.next_round_animation_counter / 32).astype(jnp.int32)]),
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, jnp.where(state.level_number % 2 != 0, M['color_start'], M['color_destination'])),
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_intermediate']),
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, jnp.where(state.level_number % 2 != 0, M['color_destination'], M['color_start'])),
                    ],
                    default=val2,
                ),
                init_val=val,
            ),
            init_val=raster,
        )

        player_score_digits = jr.int_to_digits(state.player_score, max_digits=5)
        raster = jnp.where(state.player_position[1] >= 3, jr.render_label_selective(raster, 34, 6, player_score_digits, M['score_digits'], 0, 5, spacing=8, max_digits_to_render=5), raster)

        raster = jax.lax.fori_loop(
            lower=0,
            upper=state.lives,
            body_fun=lambda i, val: jnp.where(state.player_position[1] >= 3, jr.render_at(val, self._env.renderer.LIVE_POSITIONS[i][0], self._env.renderer.LIVE_POSITIONS[i][1], M['qbert_live']), val),
            init_val=raster,
        )

        qbert_j = state.player_last_position[0]
        qbert_i = state.player_last_position[1]
        move_positions = jnp.array([self._env.renderer.QBERT_MOVE_LEFT_UP, self._env.renderer.QBERT_MOVE_LEFT_DOWN, self._env.renderer.QBERT_MOVE_RIGHT_DOWN, self._env.renderer.QBERT_MOVE_RIGHT_UP]).astype(jnp.int32)
        qbert_sprites = M['qbert_sprites']
        raster = jax.lax.switch(
            index=state.player_position_category,
            branches=[
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + move_positions[state.player_direction][state.player_moving_counter][0],
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + move_positions[state.player_direction][state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_DISC_LEFT_BOTTOM[state.player_moving_counter][0] * (state.player_direction == 0) + self._env.renderer.QBERT_MOVE_DISC_RIGHT_BOTTOM[state.player_moving_counter][0] * (state.player_direction == 3),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_DISC_LEFT_BOTTOM[state.player_moving_counter][1] * (state.player_direction == 0) + self._env.renderer.QBERT_MOVE_DISC_RIGHT_BOTTOM[state.player_moving_counter][1] * (state.player_direction == 3),
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_DISC_LEFT_TOP[state.player_moving_counter][0] * (state.player_direction == 0) + self._env.renderer.QBERT_MOVE_DISC_RIGHT_TOP[state.player_moving_counter][0] * (state.player_direction == 3),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_DISC_LEFT_TOP[state.player_moving_counter][1] * (state.player_direction == 0) + self._env.renderer.QBERT_MOVE_DISC_RIGHT_TOP[state.player_moving_counter][1] * (state.player_direction == 3),
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_1[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_1[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_2[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_2[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_3[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_3[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_4[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_4[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_5[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_5[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self._env.renderer.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
            ],
            operand=state,
        )

        sam_j = state.sam_position[0]
        sam_i = state.sam_position[1]
        raster = jnp.where(jnp.logical_and(state.sam_position[0] == -1, state.sam_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                           self._env.renderer.QBERT_POSITIONS[jnp.array(sam_i * (sam_i - 1) / 2 + (sam_j - 1)).astype(jnp.int32)][0],
                                                                                                                           self._env.renderer.QBERT_POSITIONS[jnp.array(sam_i * (sam_i - 1) / 2 + (sam_j - 1)).astype(jnp.int32)][1] + 1,
                                                                                                                           M['sam']))

        green_ball_j = state.green_ball_position[0]
        green_ball_i = state.green_ball_position[1]
        green_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._env.renderer._enemy_move_tick[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.green_ball_position[0] == -1, state.green_ball_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                                         self._env.renderer.QBERT_POSITIONS[jnp.array(green_ball_i * (green_ball_i - 1) / 2 + (green_ball_j - 1)).astype(jnp.int32)][0] + self._env.renderer.BALL_MOVE[green_ball_index][0],
                                                                                                                                         self._env.renderer.QBERT_POSITIONS[jnp.array(green_ball_i * (green_ball_i - 1) / 2 + (green_ball_j - 1)).astype(jnp.int32)][1] + self._env.renderer.BALL_MOVE[green_ball_index][1],
                                                                                                                                         M['green_ball'][green_ball_index]))

        purple_ball_j = state.purple_ball_position[0]
        purple_ball_i = state.purple_ball_position[1]
        purple_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._env.renderer._enemy_move_tick[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.purple_ball_position[0] == -1, state.purple_ball_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                                           self._env.renderer.QBERT_POSITIONS[jnp.array(purple_ball_i * (purple_ball_i - 1) / 2 + (purple_ball_j - 1)).astype(jnp.int32)][0] + self._env.renderer.BALL_MOVE[purple_ball_index][0],
                                                                                                                                           self._env.renderer.QBERT_POSITIONS[jnp.array(purple_ball_i * (purple_ball_i - 1) / 2 + (purple_ball_j - 1)).astype(jnp.int32)][1] + self._env.renderer.BALL_MOVE[purple_ball_index][1],
                                                                                                                                           M['purple_ball'][purple_ball_index]))

        snake_j = state.snake_position[0]
        snake_i = state.snake_position[1]
        snake_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._env.renderer._enemy_move_tick[state.level_number] / 5).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.snake_position[0] == -1, state.snake_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                               self._env.renderer.QBERT_POSITIONS[jnp.array(snake_i * (snake_i - 1) / 2 + (snake_j - 1)).astype(jnp.int32)][0] + self._env.renderer.SNAKE_MOVE[snake_index][0],
                                                                                                                               self._env.renderer.QBERT_POSITIONS[jnp.array(snake_i * (snake_i - 1) / 2 + (snake_j - 1)).astype(jnp.int32)][1] + self._env.renderer.SNAKE_MOVE[snake_index][1],
                                                                                                                               M['snake'][snake_index]))

        def render_red_ball(i, r):
            red_ball_j = state.red_ball_positions[i][0]
            red_ball_i = state.red_ball_positions[i][1]
            red_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._env.renderer._enemy_move_tick[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
            r = jnp.where(jnp.logical_and(state.red_ball_positions[i][0] == -1, state.red_ball_positions[i][1] == -1), r, jr.render_at(r,
                                                                                                                                            self._env.renderer.QBERT_POSITIONS[jnp.array(red_ball_i * (red_ball_i - 1) / 2 + (red_ball_j - 1)).astype(jnp.int32)][0] + self._env.renderer.BALL_MOVE[red_ball_index][0],
                                                                                                                                            self._env.renderer.QBERT_POSITIONS[jnp.array(red_ball_i * (red_ball_i - 1) / 2 + (red_ball_j - 1)).astype(jnp.int32)][1] + self._env.renderer.BALL_MOVE[red_ball_index][1],
                                                                                                                                            M['red_ball'][red_ball_index]))
            return r

        raster = jax.lax.fori_loop(
            lower=0,
            upper=3,
            body_fun=render_red_ball,
            init_val=raster,
        )

        raster = jnp.where(state.dead_animation_counter != 0, jr.render_at(raster,
                                                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + 15,
                                                                           self._env.renderer.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] - 9,
                                                                           M['dead']), raster)

        return jr.render_from_palette(raster, self._env.renderer.PALETTE)
from jaxatari.modification import JaxAtariInternalModPlugin
import jax.numpy as jnp
import jax
from functools import partial
from jaxatari.games.jax_qbert import QbertState

class SwapColorsMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _draw_colors(self, raster: jnp.ndarray, state: QbertState, pyra: jnp.ndarray) -> jnp.ndarray:
        jr = self._env.renderer.jr
        M = self._env.renderer.SHAPE_MASKS
        return jax.lax.fori_loop(
            lower=1,
            upper=7,
            body_fun=lambda i, val: jax.lax.fori_loop(
                lower=1,
                upper=i + 1,
                body_fun=lambda j, val2: jnp.select(
                    condlist=[
                        state.next_round_animation_counter != 0,
                        pyra[i, j] == 0,
                        pyra[i, j] == 1,
                        pyra[i, j] == 2,
                    ],
                    choicelist=[
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['win_animation'][jnp.floor(state.next_round_animation_counter / 32).astype(jnp.int32)]),
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_destination']),
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_intermediate']),
                        jr.render_at(val2, self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self._env.renderer.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_start']),
                    ],
                    default=val2,
                ),
                init_val=val,
            ),
            init_val=raster,
        )

