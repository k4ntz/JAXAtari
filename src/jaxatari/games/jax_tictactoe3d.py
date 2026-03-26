"""
3D Tic-Tac-Toe for JAXAtari
"""

from typing import NamedTuple, Tuple
import os
from functools import partial
from flax.nnx import state
import jax
import jax.numpy as jnp
import chex
import jax.lax 
import jaxatari.spaces as spaces
import numpy as np
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# --- Helper to generate masks ---
def _generate_win_masks():
    masks = []
    def get_mask(coords):
        m = np.zeros((4, 4, 4), dtype=np.int32)
        for z, y, x in coords:
            m[z, y, x] = 1
        return m

    # 1. Rows
    for z in range(4):
        for y in range(4): masks.append(get_mask([(z, y, x) for x in range(4)]))
    # 2. Cols
    for z in range(4):
        for x in range(4): masks.append(get_mask([(z, y, x) for y in range(4)]))
    # 3. Pillars
    for y in range(4):
        for x in range(4): masks.append(get_mask([(z, y, x) for z in range(4)]))
    # 4. Plane Diagonals (z-fixed)
    for z in range(4):
        masks.append(get_mask([(z, i, i) for i in range(4)]))
        masks.append(get_mask([(z, i, 3-i) for i in range(4)]))
    # 5. Plane Diagonals (y-fixed)
    for y in range(4):
        masks.append(get_mask([(i, y, i) for i in range(4)]))
        masks.append(get_mask([(3-i, y, i) for i in range(4)]))
    # 6. Plane Diagonals (x-fixed)
    for x in range(4):
        masks.append(get_mask([(i, i, x) for i in range(4)]))
        masks.append(get_mask([(3-i, i, x) for i in range(4)]))
    # 7. Space Diagonals
    masks.append(get_mask([(i, i, i) for i in range(4)]))
    masks.append(get_mask([(i, i, 3-i) for i in range(4)]))
    masks.append(get_mask([(i, 3-i, i) for i in range(4)]))
    masks.append(get_mask([(i, 3-i, 3-i) for i in range(4)]))
    
    return jnp.array(masks, dtype=jnp.int32)

WIN_MASKS_ARRAY = _generate_win_masks()

# --- CALIBRATED COORDINATE GENERATOR ---
def _generate_pixel_coords():
    coords = np.zeros((4, 4, 4, 2), dtype=np.int32)
    start_x = 58
    start_y = 18
    level_step_y = 46
    level_step_x = 0
    cell_step_x = 8
    cell_step_y = 10
    row_skew_x = 5
    
    for z in range(4):
        level_base_x = start_x + (z * level_step_x)
        level_base_y = start_y + (z * level_step_y)
        for y in range(4):
            for x in range(4):
                px = level_base_x + (x * cell_step_x) + (y * row_skew_x)
                py = level_base_y + (y * cell_step_y)
                coords[z, y, x] = [px, py]
    return jnp.array(coords, dtype=jnp.int32)

PIXEL_COORDS_GENERATED = _generate_pixel_coords()


def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'x', 'type': 'single', 'file': 'X.npy'},
        {'name': 'o', 'type': 'single', 'file': 'O.npy'},
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
    )
    
class TicTacToe3DConstants(NamedTuple):
    BOARD_SIZE: int = 4
    EMPTY: int = 0
    PLAYER_X: int = 1
    PLAYER_O: int = 2
    FIRST_PLAYER: int = 1
    WIN_MASKS: chex.Array = WIN_MASKS_ARRAY
    HEIGHT: int = 210
    WIDTH: int = 160
    NUM_ACTIONS: int = 8
    BLINK_PERIOD: int = 8
    MOVE_COOLDOWN: int = 6
    CELL_CENTER_X: int = 0
    CELL_CENTER_Y: int = 0
    BLACKOUT_FRAMES: int = 50
    PIXEL_COORDS: chex.Array = PIXEL_COORDS_GENERATED
    ASSET_CONFIG: tuple = _get_default_asset_config()

class TicTacToe3DState(NamedTuple):
    board: jnp.ndarray          
    current_player: jnp.ndarray 
    game_over: jnp.ndarray      
    winner: jnp.ndarray         
    move_count: jnp.ndarray     
    cursor_x: jnp.ndarray
    cursor_y: jnp.ndarray
    cursor_z: jnp.ndarray
    last_cpu_x: jnp.ndarray
    last_cpu_y: jnp.ndarray
    last_cpu_z: jnp.ndarray
    blackout_active: jnp.ndarray
    blackout_timer: jnp.ndarray
    pending_cpu_x: jnp.ndarray
    pending_cpu_y: jnp.ndarray
    pending_cpu_z: jnp.ndarray
    pending_cpu_valid: jnp.ndarray
    frame: jnp.ndarray
    key: chex.PRNGKey           

class TicTacToe3DObservation(NamedTuple):
    board: jnp.ndarray          
    current_player: jnp.ndarray 
    valid_moves: jnp.ndarray    
    game_over: jnp.ndarray      
    winner: jnp.ndarray         

class TicTacToe3DInfo(NamedTuple):
    move_count: jnp.ndarray
    game_phase: jnp.ndarray     
    last_move_player: jnp.ndarray  
    last_move_action: jnp.ndarray  

class JaxTicTacToe3DEnvironment(JaxEnvironment):
    """3D Tic-Tac-Toe environment for JAXAtari (Vectorized)."""
    def __init__(self, consts: TicTacToe3DConstants | None = None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)
        self.renderer = TicTacToe3DRenderer(self.consts)

        self.action_set = [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN
        ]
        self.ACTION_MAP = jnp.array([int(a) for a in self.action_set], dtype=jnp.int32)
        h, w = self.renderer.BACKGROUND.shape[:2]
        self._image_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=jnp.uint8)

    def action_space(self):
        return spaces.Discrete(len(self.action_set)) 

    def observation_space(self) -> spaces.Space:
        """
        Object-centric observation space. 
        KEYS MUST BE ALPHABETICAL to match JAX sorting!
        Order: b, c, g, v, w.
        """
        return spaces.Dict({
            "board": spaces.Box(0, 2, shape=(4, 4, 4), dtype=jnp.int32),
            "current_player": spaces.Discrete(3),  
            "game_over": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
            "valid_moves": spaces.Box(0, 1, shape=(64,), dtype=jnp.int32),
            "winner": spaces.Box(0, 2, shape=(), dtype=jnp.int32),
        })
    
    def image_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def render(self, state: TicTacToe3DState) -> jnp.ndarray:
        return self.renderer.render(state)

    def reset(self, key: chex.PRNGKey):
        key, subkey = jax.random.split(key)
        state = TicTacToe3DState(
            board=jnp.zeros((4, 4, 4), dtype=jnp.uint8),
            current_player=jnp.int32(self.consts.FIRST_PLAYER),
            game_over=jnp.bool_(False),
            winner=jnp.int32(0),
            move_count=jnp.int32(0),
            cursor_x=jnp.int32(0),
            cursor_y=jnp.int32(0),
            cursor_z=jnp.int32(0),
            last_cpu_x=jnp.int32(-1),
            last_cpu_y=jnp.int32(-1),
            last_cpu_z=jnp.int32(-1),
            blackout_active=jnp.bool_(False),
            blackout_timer=jnp.int32(0),
            pending_cpu_x=jnp.int32(-1),
            pending_cpu_y=jnp.int32(-1),
            pending_cpu_z=jnp.int32(-1),
            pending_cpu_valid=jnp.bool_(False),
            frame=jnp.int32(0),
            key=subkey
        )
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TicTacToe3DState, action: jnp.ndarray):
        ale_action = self.ACTION_MAP[action]
        key, subkey = jax.random.split(state.key)

        # ---------------------------------------------------------
        # 0. Handle blackout first
        # ---------------------------------------------------------
        def handle_blackout(s):
            new_timer = s.blackout_timer - 1
            blackout_finished = new_timer <= 0

            def apply_pending_cpu(ss):
                new_board = ss.board.at[ss.pending_cpu_z, ss.pending_cpu_y, ss.pending_cpu_x].set(self.consts.PLAYER_O)
                new_winner = self._check_winner(new_board)
                new_move_count = ss.move_count + jnp.where(ss.pending_cpu_valid, 1, 0)
                new_game_over = jnp.logical_or(new_winner != self.consts.EMPTY, new_move_count >= 64)

                new_state = ss._replace(
                    board=new_board,
                    move_count=new_move_count,
                    winner=new_winner,
                    game_over=new_game_over,
                    last_cpu_x=ss.pending_cpu_x,
                    last_cpu_y=ss.pending_cpu_y,
                    last_cpu_z=ss.pending_cpu_z,
                    blackout_active=jnp.bool_(False),
                    blackout_timer=jnp.int32(0),
                    pending_cpu_x=jnp.int32(-1),
                    pending_cpu_y=jnp.int32(-1),
                    pending_cpu_z=jnp.int32(-1),
                    pending_cpu_valid=jnp.bool_(False),
                    frame=ss.frame + 1,
                    key=key
                )
                obs = self._get_observation(new_state)
                reward = self._get_reward(s, new_state)
                done = self._get_done(new_state)
                info = self._get_info(new_state, ale_action)
                return obs, new_state, reward, done, info

            def continue_blackout(ss):
                new_state = ss._replace(
                    blackout_active=jnp.bool_(True),
                    blackout_timer=new_timer,
                    frame=ss.frame + 1,
                    key=key
                )
                obs = self._get_observation(new_state)
                reward = self._get_reward(s, new_state)
                done = self._get_done(new_state)
                info = self._get_info(new_state, ale_action)
                return obs, new_state, reward, done, info

            return jax.lax.cond(
                jnp.logical_and(blackout_finished, s.pending_cpu_valid),
                apply_pending_cpu,
                continue_blackout,
                s
            )

        def handle_normal_play(s):
            board = s.board
            is_first_turn = jnp.logical_and(s.move_count == 0, jnp.logical_not(s.game_over))

            # ---------------------------------------------------------
            # 1. CPU fixed opening move first (ALE-like opening)
            # layer 3, row 2, col 2 -> zero-based (2, 1, 1)
            # ---------------------------------------------------------
            opening_z = jnp.int32(2)
            opening_y = jnp.int32(1)
            opening_x = jnp.int32(1)

            def cpu_opening_move(current_board):
                return current_board.at[opening_z, opening_y, opening_x].set(self.consts.PLAYER_O)

            board_after_opening = jax.lax.cond(
                is_first_turn,
                cpu_opening_move,
                lambda b: b,
                board
            )

            move_count_after_opening = s.move_count + jnp.where(is_first_turn, 1, 0)
            winner_after_opening = self._check_winner(board_after_opening)
            game_over_after_opening = jnp.logical_or(
                winner_after_opening != self.consts.EMPTY,
                move_count_after_opening >= 64
            )

            last_cpu_x_after_opening = jnp.where(is_first_turn, opening_x, s.last_cpu_x)
            last_cpu_y_after_opening = jnp.where(is_first_turn, opening_y, s.last_cpu_y)
            last_cpu_z_after_opening = jnp.where(is_first_turn, opening_z, s.last_cpu_z)

            # ---------------------------------------------------------
            # 2. USER TURN
            # ---------------------------------------------------------
            cursor_array = jnp.array([s.cursor_x, s.cursor_y, s.cursor_z], dtype=jnp.int32)
            new_cursor = self._move_cursor(cursor_array, ale_action, s.frame)
            cx, cy, cz = new_cursor[0], new_cursor[1], new_cursor[2]

            is_place = ale_action == int(Action.FIRE)
            is_empty = board_after_opening[cz, cy, cx] == self.consts.EMPTY
            can_place = jnp.logical_and(is_place, is_empty)
            can_place = jnp.logical_and(can_place, jnp.logical_not(game_over_after_opening))

            def place_mark_user(b):
                return b.at[cz, cy, cx].set(self.consts.PLAYER_X)

            board_after_user = jax.lax.cond(
                can_place,
                place_mark_user,
                lambda b: b,
                board_after_opening
            )

            winner_after_user = self._check_winner(board_after_user)
            move_count_after_user = move_count_after_opening + jnp.where(can_place, 1, 0)
            game_over_after_user = jnp.logical_or(
                winner_after_user != self.consts.EMPTY,
                move_count_after_user >= 64
            )

            # ---------------------------------------------------------
            # 3. CPU RESPONSE PREPARATION
            # Skip response on very first step, because CPU already opened
            # After user move, blackout starts, CPU move is delayed
            # ---------------------------------------------------------
            cpu_should_prepare = jnp.logical_and(
                can_place,
                jnp.logical_not(game_over_after_user)
            )
            cpu_should_prepare = jnp.logical_and(
                cpu_should_prepare,
                jnp.logical_not(is_first_turn)
            )

            best_move_flat = self._compute_cpu_move(board_after_user, subkey)
            cpu_bz = (best_move_flat // 16).astype(jnp.int32)
            cpu_by = ((best_move_flat % 16) // 4).astype(jnp.int32)
            cpu_bx = (best_move_flat % 4).astype(jnp.int32)

            new_state = s._replace(
                board=board_after_user,
                current_player=jnp.int32(self.consts.PLAYER_X),
                move_count=move_count_after_user,
                cursor_x=cx,
                cursor_y=cy,
                cursor_z=cz,
                last_cpu_x=last_cpu_x_after_opening,
                last_cpu_y=last_cpu_y_after_opening,
                last_cpu_z=last_cpu_z_after_opening,
                blackout_active=cpu_should_prepare,
                blackout_timer=jnp.where(cpu_should_prepare, jnp.int32(self.consts.BLACKOUT_FRAMES), jnp.int32(0)),
                pending_cpu_x=jnp.where(cpu_should_prepare, cpu_bx, jnp.int32(-1)),
                pending_cpu_y=jnp.where(cpu_should_prepare, cpu_by, jnp.int32(-1)),
                pending_cpu_z=jnp.where(cpu_should_prepare, cpu_bz, jnp.int32(-1)),
                pending_cpu_valid=cpu_should_prepare,
                frame=s.frame + 1,
                winner=winner_after_user,
                game_over=game_over_after_user,
                key=key
            )

            obs = self._get_observation(new_state)
            reward = self._get_reward(s, new_state)
            done = self._get_done(new_state)
            info = self._get_info(new_state, ale_action)
            return obs, new_state, reward, done, info

        return jax.lax.cond(
            state.blackout_active,
            handle_blackout,
            handle_normal_play,
            state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TicTacToe3DState) -> spaces.Dict:
        """
        Extracts observation dictionary. 
        KEYS MUST BE ALPHABETICAL: b, c, g, v, w.
        DTYPES MUST BE INT32.
        """
        return {
            "board": state.board.astype(jnp.int32),
            "current_player": state.current_player.astype(jnp.int32),
            "game_over": state.game_over.astype(jnp.int32),
            "valid_moves": (state.board == self.consts.EMPTY).reshape(64).astype(jnp.int32),
            "winner": state.winner.astype(jnp.int32),
        }

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs) -> jnp.ndarray:
        """
        Flattens observation. 
        MUST RETURN INT32 to match space definition.
        ORDER MUST BE ALPHABETICAL BY KEY:
        1. board
        2. current_player
        3. game_over
        4. valid_moves
        5. winner
        """
        board_flat = obs["board"].reshape(-1).astype(jnp.int32)
        current_player = obs["current_player"].reshape(1).astype(jnp.int32)
        game_over = obs["game_over"].reshape(1).astype(jnp.int32)
        valid_moves_flat = obs["valid_moves"].astype(jnp.int32)
        winner = obs["winner"].reshape(1).astype(jnp.int32)
        
        return jnp.concatenate([
            board_flat, 
            current_player, 
            game_over,
            valid_moves_flat, 
            winner
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _compute_cpu_move(self, board, key):
        """Vectorized Heuristic AI."""
        is_x = (board == self.consts.PLAYER_X).astype(jnp.int32)
        is_o = (board == self.consts.PLAYER_O).astype(jnp.int32)
        is_empty = (board == self.consts.EMPTY).astype(jnp.int32)
        
        x_counts = jnp.tensordot(self.consts.WIN_MASKS, is_x, axes=((1, 2, 3), (0, 1, 2)))
        o_counts = jnp.tensordot(self.consts.WIN_MASKS, is_o, axes=((1, 2, 3), (0, 1, 2)))
        
        winnable_lines = jnp.logical_and(o_counts == 3, x_counts == 0)
        blockable_lines = jnp.logical_and(x_counts == 3, o_counts == 0)
        
        win_heatmap = jnp.sum(self.consts.WIN_MASKS * winnable_lines[:, None, None, None], axis=0)
        block_heatmap = jnp.sum(self.consts.WIN_MASKS * blockable_lines[:, None, None, None], axis=0)
        random_heatmap = jax.random.uniform(key, shape=(4, 4, 4))
        
        final_scores = (win_heatmap * 1000.0) + (block_heatmap * 100.0) + random_heatmap
        final_scores = jnp.where(is_empty, final_scores, -1e9)
        return jnp.argmax(final_scores.ravel())

    @partial(jax.jit, static_argnums=(0,))
    def _move_cursor(self, cursor, ale_action, step_count):
        x, y, z = cursor[0], cursor[1], cursor[2]
        can_move = (step_count % self.consts.MOVE_COOLDOWN) == 0

        # ---------------------------------------------------------
        # X movement (wraparound)
        # ---------------------------------------------------------
        dx = jnp.where(
            ale_action == int(Action.LEFT), -1,
            jnp.where(ale_action == int(Action.RIGHT), 1, 0)
        )
        new_x = jnp.where(can_move, (x + dx) % 4, x)

        # ---------------------------------------------------------
        # Y movement (wraparound WITH layer transitions)
        # ---------------------------------------------------------
        dy = jnp.where(
            ale_action == int(Action.UP), -1,
            jnp.where(ale_action == int(Action.DOWN), 1, 0)
        )

        raw_y = y + dy

        # Wrap y between 0..3
        wrapped_y = raw_y % 4

        # Detect wrap events
        moved_down_layer = raw_y > 3   # from 3 → 4
        moved_up_layer = raw_y < 0     # from 0 → -1

        # Z wraparound
        new_z = jnp.where(
            moved_down_layer,
            (z + 1) % 4,
            jnp.where(
                moved_up_layer,
                (z - 1) % 4,
                z
            )
        )

        new_y = jnp.where(can_move, wrapped_y, y)
        new_z = jnp.where(can_move, new_z, z)

        return jnp.array([new_x, new_y, new_z], dtype=jnp.int32)
    @partial(jax.jit, static_argnums=(0,))
    def _check_winner(self, board: jnp.ndarray) -> jnp.ndarray:
        is_x = (board == self.consts.PLAYER_X).astype(jnp.int32)
        is_o = (board == self.consts.PLAYER_O).astype(jnp.int32)
        scores_x = jnp.tensordot(self.consts.WIN_MASKS, is_x, axes=((1, 2, 3), (0, 1, 2)))
        scores_o = jnp.tensordot(self.consts.WIN_MASKS, is_o, axes=((1, 2, 3), (0, 1, 2)))
        x_wins = jnp.any(scores_x == 4)
        o_wins = jnp.any(scores_o == 4)
        return jnp.where(x_wins, self.consts.PLAYER_X, jnp.where(o_wins, self.consts.PLAYER_O, self.consts.EMPTY))
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
        player_won = state.winner == self.consts.PLAYER_X
        opponent_won = state.winner == self.consts.PLAYER_O
        return jnp.where(player_won, 1.0, jnp.where(opponent_won, -1.0, 0.0))
    
    # Make action optional to prevent wrappers from crashing
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TicTacToe3DState, action: jnp.ndarray = None) -> TicTacToe3DInfo:
        """Returns info dict (Compatible with Gym wrappers)."""
        act = jax.lax.select(action is None, jnp.int32(0), action) if action is not None else jnp.int32(0)
        game_phase = jnp.where(
            state.winner == self.consts.PLAYER_X, 1,
            jnp.where(state.winner == self.consts.PLAYER_O, 2, 
            jnp.where(state.move_count >= 64, 3, 0))
        )
        return TicTacToe3DInfo(
            move_count=state.move_count, 
            game_phase=jnp.int32(game_phase),
            last_move_player=state.current_player,
            last_move_action=act
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state):
        return jnp.logical_or(state.game_over, state.move_count >= 64)


class TicTacToe3DRenderer(JAXGameRenderer):
    def __init__(self, consts: TicTacToe3DConstants | None = None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        final_asset_config = list(self.consts.ASSET_CONFIG)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tictactoe3d"
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, self.COLOR_TO_ID, self.FLIP_OFFSETS) = \
            self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        self.x_mask = self.SHAPE_MASKS["x"]
        self.o_mask = self.SHAPE_MASKS["o"]
        self.cursor_mask = self.SHAPE_MASKS["cursor"]
        

    def cell_to_pixel(self, x, y, z):
        px = self.consts.PIXEL_COORDS[z, y, x, 0]
        py = self.consts.PIXEL_COORDS[z, y, x, 1]
        return px, py

    def cell_center_to_pixel(self, x, y, z):
        px, py = self.cell_to_pixel(x, y, z) 
        cpx = px + self.consts.CELL_CENTER_X
        cpy = py + self.consts.CELL_CENTER_Y
        return cpx, cpy

    def render(self, state):
        def render_black(_):
            return jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

        def render_normal(_):
            raster = self.jr.create_object_raster(self.BACKGROUND)
            board = state.board
            blink_on = (state.frame // self.consts.BLINK_PERIOD) % 2 == 0

            last_cpu_x = state.last_cpu_x
            last_cpu_y = state.last_cpu_y
            last_cpu_z = state.last_cpu_z

            for idx in range(64):
                z = idx // 16
                y = (idx % 16) // 4
                x = idx % 4
                cell_val = board[z, y, x]
                px, py = self.cell_to_pixel(x, y, z)

                is_latest_cpu_move = (
                    (x == last_cpu_x) &
                    (y == last_cpu_y) &
                    (z == last_cpu_z)
                )

                should_draw_o = jnp.logical_or(
                    jnp.logical_not(is_latest_cpu_move),
                    blink_on
                )

                raster = jax.lax.cond(
                    cell_val == self.consts.PLAYER_X,
                    lambda r: self.jr.render_at(r, px, py, self.x_mask),
                    lambda r: r,
                    raster
                )

                raster = jax.lax.cond(
                    jnp.logical_and(cell_val == self.consts.PLAYER_O, should_draw_o),
                    lambda r: self.jr.render_at(r, px, py, self.o_mask),
                    lambda r: r,
                    raster
                )

            cpx, cpy = self.cell_center_to_pixel(state.cursor_x, state.cursor_y, state.cursor_z)
            raster = jax.lax.cond(
                blink_on,
                lambda r: self.jr.render_at(r, cpx, cpy, self.cursor_mask),
                lambda r: r,
                raster
            )

            return self.jr.render_from_palette(raster, self.PALETTE)

        return jax.lax.cond(
            state.blackout_active,
            render_black,
            render_normal,
            operand=None
        )