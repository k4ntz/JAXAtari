import os
import chex
import jax.lax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs, EnvState
from jaxatari.rendering import jax_rendering_utils_legacy as jr
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces

class BoxingConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    initial_x_white: int = 30
    initial_y_white: int = 35
    initial_x_black: int = 115
    initial_y_black: int = 130
    x_boundaries: Tuple[int,int] = (28, 120)
    y_boundaries: Tuple[int,int] = (30, 135)

class BoxingState(NamedTuple):
    player_score: chex.Array
    enemy_score: chex.Array
    player_x: chex.Array
    player_y: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    time: chex.Array
    punch_timer: chex.Array
    enemy_punch_timer: chex.Array
    hit_left: chex.Array
    enemy_hit_left: chex.Array
    enemy_target_x: chex.Array
    enemy_target_y: chex.Array
    enemy_target_timer: chex.Array
    step_counter: chex.Array
    enemy_dir_x: chex.Array
    enemy_dir_y: chex.Array
    enemy_knockback_timer: chex.Array
    player_knockback_timer: chex.Array
    player_knockback_tick: chex.Array
    enemy_knockback_tick: chex.Array
    enemy_mode: chex.Array
    enemy_direction_timer: chex.Array
    enemy_punch_cooldown: chex.Array


class EntityPositions(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray

class BoxingObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_score: chex.Array
class BoxingInfo(NamedTuple):
    time: jnp.ndarray
    step_counter: jnp.ndarray
    all_rewards: chex.Array

class carryState(NamedTuple):
    player_score:chex.Array
class JaxBoxing(JaxEnvironment[BoxingState, BoxingObservation, BoxingInfo, BoxingConstants]):
    def __init__(self, consts: BoxingConstants = None, reward_funcs: list[callable] = None):
        consts = consts or BoxingConstants()
        super().__init__(consts)
        self.renderer = BoxingRenderer(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.step_counter = 0
        self.action_set = [
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UP,
            Action.UPRIGHT,
            Action.DOWNRIGHT,
            Action.UPLEFT,
            Action.DOWNLEFT,
        ]

    def reset(self, key=None) -> Tuple[BoxingObservation, BoxingState]:
        state = BoxingState(
            player_score=jnp.array(0),
            enemy_score=jnp.array(0),
            player_x=jnp.array(self.consts.initial_x_white),
            player_y=jnp.array(self.consts.initial_y_white),
            enemy_x=jnp.array(self.consts.initial_x_black),
            enemy_y=jnp.array(self.consts.initial_y_black),
            time= jnp.array(7200),
            punch_timer=jnp.array(0),
            enemy_punch_timer=jnp.array(0),
            hit_left = jnp.array(False),
            enemy_hit_left=jnp.array(True),
            enemy_target_x = jnp.array(0),
            enemy_target_y = jnp.array(0),
            enemy_target_timer = jnp.array(0),
            step_counter=jnp.array(0),
            enemy_knockback_timer=jnp.array(0),
            enemy_mode=jnp.array(0),
            enemy_direction_timer=jnp.array(0),
            enemy_dir_x=jnp.array(0),
            enemy_dir_y=jnp.array(0),
            enemy_punch_cooldown=jnp.array(0),
            enemy_knockback_tick=jnp.array(0),
            player_knockback_timer=jnp.array(0),
            player_knockback_tick=jnp.array(0),
        )

        initial_obs = self._get_observation(state)
        return initial_obs, state

    def player_step(self, state, action: chex.Array):
        left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)
        up = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT)
        down = (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT)
        fire = (action == Action.FIRE)
        def apply_knockback(state):
            knockback_active = jnp.where(state.player_knockback_timer > 0, True, False)
            player_at_boundary = state.player_x - 2 <= self.consts.x_boundaries[0]
            dx_knockback = jnp.where(knockback_active & player_at_boundary, True, False)
            enemy_hit_left = dx_knockback & state.enemy_hit_left
            new_knockback_tick = (state.player_knockback_tick + 1) % 2
            knockback_dy = jnp.where(enemy_hit_left, -1, jnp.where(dx_knockback, 1, 0))
            knockback_dx = jnp.where(knockback_active & (new_knockback_tick == 0),
                                 -1, 0)
            new_player_y = jnp.clip(state.player_y + knockback_dy,
                                    self.consts.y_boundaries[0],
                                    self.consts.y_boundaries[1])
            new_player_x = jnp.clip(state.player_x + knockback_dx,
                               self.consts.x_boundaries[0],
                               self.consts.x_boundaries[1])
            new_knockback_timer = jnp.maximum(state.player_knockback_timer - 1, 0)
            return state._replace(player_x=new_player_x,player_y=new_player_y,player_knockback_timer=new_knockback_timer,player_knockback_tick=new_knockback_tick)

        state = jax.lax.cond(
            state.player_knockback_timer > 0,
            apply_knockback,
            lambda s: s,
            operand=state
        )

        punch_time = jnp.maximum(state.punch_timer - 1, 0)
        punch_time = jnp.where(fire & (punch_time == 0), 20, punch_time)
        left_hit_y = state.player_y - 15
        right_hit_y = state.player_y + 15
        dist_left = jnp.abs(left_hit_y - state.enemy_y)
        dist_right = jnp.abs(right_hit_y - state.enemy_y)
        chosen_hit_left = dist_left < dist_right
        hit_left = jnp.where(punch_time == 20, chosen_hit_left, state.hit_left)

        # Apply movement
        delta_x = jnp.where(right, 2, 0) - jnp.where(left, 2, 0)
        delta_y = jnp.where(down, 2, 0) - jnp.where(up, 2, 0)

        player_x = jnp.clip(state.player_x + delta_x, self.consts.x_boundaries[0], self.consts.x_boundaries[1])
        player_y = jnp.clip(state.player_y + delta_y, self.consts.y_boundaries[0], self.consts.y_boundaries[1])

        check_at_correct_punch_point = punch_time == 10
        # check if punch hits enemy
        def check_collision(hit_direction):
            def hit_collision_left_swing():
                hit_x = player_x + 18
                hit_y = player_y - 15
                hit = (jnp.abs(hit_x - state.enemy_x) < 10) & (jnp.abs(hit_y - state.enemy_y) < 8)
                return hit
            def hit_collision_right_swing():
                hit_x = player_x + 18
                hit_y = player_y + 15
                hit = (jnp.abs(hit_x - state.enemy_x) < 10) & (jnp.abs(hit_y - state.enemy_y) < 8)
                return hit

            return jax.lax.cond(hit_direction, hit_collision_left_swing, hit_collision_right_swing)

        hit = jnp.where(check_at_correct_punch_point,check_collision(hit_left), False)
        new_enemy_knockback_timer = jnp.where(hit, 32, state.enemy_knockback_timer)


        player_score = jnp.where(hit, state.player_score + 1, state.player_score)

        return state._replace(
            player_x=player_x,
            player_y=player_y,
            punch_timer=punch_time,
            hit_left=hit_left,
            player_score=player_score,
            enemy_knockback_timer=new_enemy_knockback_timer,
        )

    def enemy_step(self, state, rng_key=None):
        step_size = 1
        def knockback_mode(state):
            knockback_active = state.enemy_knockback_timer > 0
            enemy_at_boundary = state.enemy_x + 2 >= self.consts.x_boundaries[1]
            dy_knockback = jnp.where(knockback_active & enemy_at_boundary, True, False)
            new_knockback_tick = (state.enemy_knockback_tick + 1) % 2
            hit_left = dy_knockback & state.hit_left
            knockback_dx = jnp.where(knockback_active & (new_knockback_tick == 0),
                                     2, 0)
            knockback_dy = jnp.where(hit_left, 1, jnp.where(dy_knockback, -1, 0))
            new_enemy_y = jnp.clip(state.enemy_y + knockback_dy,
                                   self.consts.y_boundaries[0],
                                   self.consts.y_boundaries[1])
            new_enemy_x = jnp.clip(state.enemy_x + knockback_dx,
                               self.consts.x_boundaries[0],
                               self.consts.x_boundaries[1])
            new_knockback_timer = jnp.maximum(state.enemy_knockback_timer - 1, 0)
            return state._replace(
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_knockback_timer=new_knockback_timer,
                enemy_knockback_tick=new_knockback_tick,
                enemy_mode=0,
            )
        def chase_mode(state):
            # Move toward player to align
            dy = state.player_y - state.enemy_y
            step_y = jnp.clip(jnp.sign(dy) * step_size, -step_size, step_size)
            new_enemy_y = jnp.clip(state.enemy_y + step_y,
                                   self.consts.y_boundaries[0],
                                   self.consts.y_boundaries[1])

            dx = state.player_x - state.enemy_x
            step_x = jnp.clip(jnp.sign(dx) * step_size, -step_size, step_size)
            new_enemy_x = jnp.clip(state.enemy_x + step_x,
                                   self.consts.x_boundaries[0],
                                   self.consts.x_boundaries[1])

            # Condition to switch into attack mode
            switch_to_attack = (dx < 0) & (jnp.abs(dy) < 22) & (jnp.abs(dx) <= 30)
            new_mode = jnp.where(switch_to_attack, 1, state.enemy_mode)

            return state._replace(enemy_x=new_enemy_x,
                                  enemy_y=new_enemy_y,
                                  enemy_mode=new_mode)

        def attack_mode(state, rng_key):
            dy = state.player_y - state.enemy_y
            dx = state.player_x - state.enemy_x
            box_y = (jnp.maximum(state.player_y - 20,self.consts.y_boundaries[0]), jnp.minimum(state.player_y + 25, self.consts.y_boundaries[1]))
            box_x = (jnp.maximum(state.player_x + 15,self.consts.x_boundaries[0]), jnp.minimum(state.player_x + 30, self.consts.x_boundaries[1]))
            enemy_jitter_timer = jnp.maximum(state.enemy_direction_timer - 1, 0)
            direction_choice_x = jax.random.choice(rng_key, jnp.array([-1,0,1]), p=jnp.array([0.4, 0.2, 0.4]))
            direction_choice_y = jax.random.choice(rng_key, jnp.array([-1,0,1]), p=jnp.array([0.4, 0.2, 0.4]))
            do_direction_change = (enemy_jitter_timer == 0)
            new_enemy_dir_x = jnp.where(do_direction_change, direction_choice_x, state.enemy_dir_x)
            new_enemy_dir_y = jnp.where(do_direction_change, direction_choice_y, state.enemy_dir_y)
            enemy_jitter_timer = jnp.where(do_direction_change, jax.random.randint(rng_key, (), 5, 20), enemy_jitter_timer)
            step_x = new_enemy_dir_x * step_size
            step_y = new_enemy_dir_y * step_size
            new_enemy_x = jnp.clip(state.enemy_x + step_x,box_x[0], box_x[1])
            new_enemy_y = jnp.clip(state.enemy_y + step_y, box_y[0], box_y[1])
            hit_alignment_x = (dx >= -20) & (dx <= -16)
            hit_alignment_left = hit_alignment_x & ((state.enemy_y - 17 <= state.player_y) & (state.player_y <= state.enemy_y - 13))
            hit_alignment_right = hit_alignment_x & ((state.enemy_y + 13 <= state.player_y) & (state.player_y <= state.enemy_y + 17))

            fire = hit_alignment_left | hit_alignment_right

            enemy_punch_timer = jnp.maximum(state.enemy_punch_timer - 1, 0)
            cooldown_timer = jnp.maximum(state.enemy_punch_cooldown - 1, 0)

            can_start_punch = fire & (cooldown_timer == 0) & (enemy_punch_timer == 0)

            enemy_punch_timer = jnp.where(can_start_punch, 20, enemy_punch_timer)
            cooldown_timer = jnp.where(can_start_punch, 75, cooldown_timer)

            enemy_hit_left = jnp.where(hit_alignment_left, False,
                  jnp.where(hit_alignment_right, True, state.enemy_hit_left))

            def check_collision(hit_direction):
                def hit_collision_left_swing():
                    hit_x = state.enemy_x - 18
                    hit_y = state.enemy_y + 15
                    hit = (jnp.abs(hit_x - state.player_x) < 10) & (jnp.abs(hit_y - state.player_y) < 8)
                    return hit

                def hit_collision_right_swing():
                    hit_x = state.enemy_x - 18
                    hit_y = state.enemy_y - 15
                    hit = (jnp.abs(hit_x - state.player_x) < 10) & (jnp.abs(hit_y - state.player_y) < 8)
                    return hit

                return jax.lax.cond(hit_direction, hit_collision_left_swing, hit_collision_right_swing)

            check_at_correct_punch_point = enemy_punch_timer == 10
            hit = jnp.where(check_at_correct_punch_point, check_collision(enemy_hit_left), False)

            new_player_knockback_timer = jnp.where(hit, 32, state.player_knockback_timer)

            new_enemy_score = jnp.where(hit, state.enemy_score + 1, state.enemy_score)



            # Condition to return to align mode
            back_to_align = ((jnp.abs(dy) > 20) & (dx > 20)) | dx > 0
            new_mode = jnp.where(back_to_align, 0, state.enemy_mode)
            return state._replace(enemy_x=new_enemy_x,
                                  enemy_y=new_enemy_y,
                                  enemy_mode=new_mode,
                                  enemy_direction_timer = enemy_jitter_timer,
                                  enemy_dir_x=new_enemy_dir_x,
                                  enemy_dir_y=new_enemy_dir_y,
                                  enemy_hit_left=enemy_hit_left,
                                  enemy_punch_timer=enemy_punch_timer,
                                  enemy_punch_cooldown=cooldown_timer,
                                  enemy_score=new_enemy_score,
                                  player_knockback_timer=new_player_knockback_timer,
                                  )
        rng_key = rng_key or jax.random.PRNGKey(state.step_counter)
        state = jax.lax.cond(
            state.enemy_knockback_timer > 0,
            lambda _: knockback_mode(state),
            lambda _: jax.lax.cond(
                state.enemy_mode == 0,
                lambda _: chase_mode(state),
                lambda _: attack_mode(state, rng_key),
                operand=None
            ),
            operand=None
        )
        return state



    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, float, bool, BoxingInfo]:
        state = self.player_step(state, action)
        state = self.enemy_step(state)
        new_time = jnp.maximum(0, state.time - 1)
        new_state = BoxingState(player_score=state.player_score,
                                enemy_score=state.enemy_score,
                                player_x=state.player_x,
                                player_y=state.player_y,
                                enemy_x=state.enemy_x,
                                enemy_y=state.enemy_y,
                                time = new_time,
                                punch_timer=state.punch_timer,
                                enemy_punch_timer=state.enemy_punch_timer,
                                hit_left=state.hit_left,
                                enemy_hit_left=state.enemy_hit_left,
                                enemy_target_x=state.enemy_target_x,
                                enemy_target_y=state.enemy_target_y,
                                enemy_target_timer=state.enemy_target_timer,
                                step_counter=state.step_counter + 1,
                                enemy_knockback_timer=state.enemy_knockback_timer,
                                enemy_mode=state.enemy_mode,
                                enemy_direction_timer=state.enemy_direction_timer,
                                enemy_dir_y=state.enemy_dir_y,
                                enemy_dir_x=state.enemy_dir_x,
                                enemy_punch_cooldown=state.enemy_punch_cooldown,
                                enemy_knockback_tick=state.enemy_knockback_tick,
                                player_knockback_timer=state.player_knockback_timer,
                                player_knockback_tick=state.player_knockback_tick,
                                )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(new_state, state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)
        return observation, new_state, env_reward, done, info

    def render(self, state: BoxingState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_done(self, state: EnvState) -> bool:
        return state.time == 0
    def _get_reward(self, previous_state: EnvState, state: EnvState) -> float:
        return (state.player_score - previous_state.player_score) - (state.enemy_score - previous_state.enemy_score)
    def _get_observation(self, state: BoxingState):
        return BoxingObservation(player_x=state.player_x, player_y=state.player_y, player_score=state.player_score)
    def _get_info(self, state: BoxingState, all_rewards: chex.Array = None) -> BoxingInfo:
        return BoxingInfo(
            time=state.time,
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )
    def _get_all_rewards(self, state: BoxingState, previous_state: BoxingState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([
            reward_func(state, previous_state) for reward_func in self.reward_funcs
        ])
        return rewards

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210,160,3),
            dtype=jnp.uint8
        )
    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=self.consts.WIDTH - 1, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=self.consts.HEIGHT - 1, shape=(), dtype=jnp.int32),
            "player_score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),
        })

    def obs_to_flat_array(self, obs: BoxingObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_x.flatten(),
            obs.player_y.flatten(),
            obs.player_score.flatten(),
        ]
        )

class BoxingRenderer(JAXGameRenderer):
    def __init__(self, consts: BoxingConstants = None):
        self.consts = consts or BoxingConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            self.DIGITS,
            self.TIME_SEPERATION,
            self.ENEMY_SCORE_DIGITS,
            self.PLAYER_SCORE_DIGITS,
            self.PLAYER_PUNCH_ANIMATION,
            self.PLAYER_PUNCH_ANIMATION_LEFT,
            self.ENEMY_PUNCH_RIGHT,
            self.ENEMY_PUNCH_LEFT,
        ) = self.load_sprites()
    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        background = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/background.npy"))
        player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/white_player.npy"))
        enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/black_player.npy"))

        DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/boxing/digits_time/{}.npy"))
        ENEMY_SCORE_DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/boxing/digits_black/{}.npy"))
        PLAYER_SCORE_DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/boxing/digits_white/{}.npy"))
        TIME_SEPERATION = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/boxing/digits_time_seperation.npy"))


        punch_paths_right = [
            os.path.join(MODULE_DIR, f"sprites/boxing/player_boxing_animation_right/{i}.npy")
            for i in range(4)
        ]
        punch_frames = [jr.loadFrame(p) for p in punch_paths_right]
        padded_punch_frames, punch_offsets = jr.pad_to_match(punch_frames)
        punch_offsets = jnp.array(punch_offsets)  #

        punch_paths_left = [
            os.path.join(MODULE_DIR, f"sprites/boxing/player_boxing_animation_left/{i}.npy")
            for i in range(4)
        ]
        punch_frames_left = [jr.loadFrame(p) for p in punch_paths_left]
        padded_punch_frames_left, punch_offsets_left = jr.pad_to_match(punch_frames_left)

        enemy_punch_paths_right = [
            os.path.join(MODULE_DIR, f"sprites/boxing/enemy_boxing_animation_right/{i}.npy")
            for i in range(4)
        ]
        enemy_punch_frames_right = [jr.loadFrame(p) for p in enemy_punch_paths_right]
        padded_punch_frames_enemy_right , enemy_punch_offsets_right = jr.pad_to_match(enemy_punch_frames_right)

        enemy_punch_paths_left = [
            os.path.join(MODULE_DIR, f"sprites/boxing/enemy_boxing_animation_left/{i}.npy")
            for i in range(4)
        ]
        enemy_punch_frames_left = [jr.loadFrame(p) for p in enemy_punch_paths_left]
        padded_punch_frames_enemy_left , enemy_punch_offsets_left = jr.pad_to_match(enemy_punch_frames_left)


        SPRITE_BG = jnp.expand_dims(background, axis=0)

        return (SPRITE_BG, player, enemy, DIGITS, TIME_SEPERATION, ENEMY_SCORE_DIGITS, PLAYER_SCORE_DIGITS, padded_punch_frames, padded_punch_frames_left, padded_punch_frames_enemy_right, padded_punch_frames_enemy_left)
    def render(self, state):
        def render_time(self, raster, time):
            total_seconds = time // 60
            minutes = (total_seconds // 60).astype(int)
            seconds = (total_seconds % 60).astype(int)

            # Get digits
            min_tens = minutes // 10
            min_ones = minutes % 10
            sec_tens = seconds // 10
            sec_ones = seconds % 10

            # Render digits using your digit sprites at the desired location
            raster = jr.render_at(raster, 60, 15, self.DIGITS[min_tens])
            raster = jr.render_at(raster, 68, 15, self.DIGITS[min_ones])
            raster = jr.render_at(raster, 88, 15, self.DIGITS[sec_tens])
            raster = jr.render_at(raster, 96, 15, self.DIGITS[sec_ones])
            return raster

        def render_score(self, raster, score, x, y, digits):
            tens = score // 10
            ones = score % 10
            raster = jnp.where(score >= 10,jr.render_at(raster, x, y, digits[tens]), raster)
            raster = jr.render_at(raster, x+5, y, digits[ones])
            return raster

        def render_player(self,raster, state):
            def render_punch(frame_index):
                punch_animation = jax.lax.cond(
                    state.hit_left,
                    lambda: self.PLAYER_PUNCH_ANIMATION_LEFT,
                    lambda: self.PLAYER_PUNCH_ANIMATION
                )
                sprite = jax.lax.switch(
                    frame_index,
                    [
                        lambda: punch_animation[0],
                        lambda: punch_animation[1],
                        lambda: punch_animation[2],
                        lambda: punch_animation[3],
                    ]
                )
                return jr.render_at(raster, state.player_x, state.player_y, sprite)

            def render_idle():
                return jr.render_at(raster, state.player_x, state.player_y, self.SPRITE_PLAYER)

            frame_index = (state.punch_timer // 5) % 4
            raster_out = jax.lax.cond(
                state.punch_timer > 0,
                lambda _: render_punch(frame_index),
                lambda _: render_idle(),
                operand=None,
            )
            return raster_out
        def render_enemy(self,raster, state):
            def render_punch_enemy(frame_index):
                punch_animation_enemy = jax.lax.cond(
                    #state.hit_left,
                    state.enemy_hit_left,
                    lambda: self.ENEMY_PUNCH_LEFT,
                    lambda: self.ENEMY_PUNCH_RIGHT
                )
                offset = jnp.array([0, 7, 16, 6])
                sprite = jax.lax.switch(
                    frame_index,
                    [
                        lambda: punch_animation_enemy[0],
                        lambda: punch_animation_enemy[1],
                        lambda: punch_animation_enemy[2],
                        lambda: punch_animation_enemy[3],
                    ]
                )
                return jr.render_at(raster, state.enemy_x - offset[frame_index], state.enemy_y, sprite)

            def render_idle_enemy():
                return jr.render_at(raster, state.enemy_x, state.enemy_y, self.SPRITE_ENEMY)

            frame_index = (state.enemy_punch_timer // 5) % 4
            raster_out = jax.lax.cond(
                state.enemy_punch_timer > 0,
                lambda _: render_punch_enemy(frame_index),
                lambda _: render_idle_enemy(),
                operand=None,
            )
            return raster_out
        raster = jr.create_initial_frame(width=160, height=210)
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0,0, frame_bg)
        raster = jr.render_at(raster, 80, 15, self.TIME_SEPERATION)
        raster = render_player(self,raster, state)
        raster = render_enemy(self,raster, state)
        raster = render_time(self, raster, state.time)
        raster = render_score(self,raster, state.player_score, 45, 7, self.PLAYER_SCORE_DIGITS)
        raster = render_score(self, raster, state.enemy_score, 104, 7, self.ENEMY_SCORE_DIGITS)
        return raster

