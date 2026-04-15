import os
from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


def _casino_asset_config() -> tuple:
    labels = [
        "bet",
        "bj",
        "bust",
        "dble",
        "fold",
        "hit",
        "insr",
        "lose",
        "pass",
        "push",
        "split",
        "stay",
        "win",
        "cut",
    ]
    label_files = [f"labels/{label}.npy" for label in labels]

    numbers = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "k", "q", "a"]
    card_files = ["cards/turned.npy", "cards/turned.npy"]
    card_files.extend([f"cards/diamond/{n}.npy" for n in numbers])
    card_files.extend([f"cards/club/{n}.npy" for n in numbers])
    card_files.extend([f"cards/heart/{n}.npy" for n in numbers])
    card_files.extend([f"cards/spade/{n}.npy" for n in numbers])

    return (
        {"name": "background", "type": "background", "file": "background.npy"},
        {"name": "isymbol", "type": "single", "file": "i.npy"},
        {"name": "question", "type": "single", "file": "question.npy"},
        {"name": "labels", "type": "group", "files": label_files},
        {"name": "cards", "type": "group", "files": card_files},
        {"name": "digits", "type": "digits", "pattern": "digits/{}.npy"},
        {"name": "cursor", "type": "single", "file": "cursor.npy"},
    )


class CasinoRenderer(JAXGameRenderer):
    def __init__(self, consts=None, config=None):
        super().__init__(consts)
        self.consts = consts
        self.config = config or render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "casino")
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(list(_casino_asset_config()), sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def _render_frame(
        self,
        card_matrix: jnp.ndarray,
        player_score: jnp.ndarray,
        char: jnp.ndarray,
        player_main_bet: jnp.ndarray,
        player_split_bet: jnp.ndarray,
        label_main: jnp.ndarray,
        label_split: jnp.ndarray,
        blinking_card: jnp.ndarray,
    ) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        y_positions = jnp.array([8, 30, 68, 106, 144, 182], dtype=jnp.int32)

        # Card IDs: -2 empty, -1 face-down, 0 empty, 1..52 real cards.
        safe_cards = jnp.where(card_matrix < -1, 0, card_matrix).astype(jnp.int32)

        def draw_row(i, acc):
            def draw_col(j, row_acc):
                card = safe_cards[i, j]
                return jax.lax.cond(
                    card != 0,
                    lambda v: self.jr.render_at(v, 12 + j * 32, y_positions[i], self.SHAPE_MASKS["cards"][card + 1]),
                    lambda v: v,
                    row_acc,
                )

            return jax.lax.fori_loop(0, safe_cards.shape[1], draw_col, acc)

        raster = jax.lax.fori_loop(0, safe_cards.shape[0], draw_row, raster)

        score_digits = self.jr.int_to_digits(player_score.astype(jnp.int32), max_digits=4)
        raster = self.jr.render_label_selective(
            raster,
            44,
            53,
            score_digits,
            self.SHAPE_MASKS["digits"],
            start_index=0,
            num_to_render=4,
            spacing=4,
            max_digits_to_render=4,
        )

        raster = jax.lax.cond(
            char == 0,
            lambda r: self.jr.render_at(r, 90, 53, self.SHAPE_MASKS["isymbol"]),
            lambda r: jax.lax.cond(
                char == 1,
                lambda r2: self.jr.render_at(r2, 89, 53, self.SHAPE_MASKS["question"]),
                lambda r2: r2,
                r,
            ),
            raster,
        )

        raster = jax.lax.cond(
            player_main_bet != -1,
            lambda r: self.jr.render_label_selective(
                r,
                77,
                53,
                self.jr.int_to_digits(player_main_bet.astype(jnp.int32), max_digits=3),
                self.SHAPE_MASKS["digits"],
                start_index=0,
                num_to_render=3,
                spacing=4,
                max_digits_to_render=3,
            ),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            player_split_bet != -1,
            lambda r: self.jr.render_label_selective(
                r,
                77,
                126,
                self.jr.int_to_digits(player_split_bet.astype(jnp.int32), max_digits=3),
                self.SHAPE_MASKS["digits"],
                start_index=0,
                num_to_render=3,
                spacing=4,
                max_digits_to_render=3,
            ),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            label_main != -1,
            lambda r: self.jr.render_at(r, 104, 48, self.SHAPE_MASKS["labels"][label_main]),
            lambda r: r,
            raster,
        )

        raster = jax.lax.cond(
            label_split != -1,
            lambda r: self.jr.render_at(r, 104, 121, self.SHAPE_MASKS["labels"][label_split]),
            lambda r: r,
            raster,
        )

        cursor_col = blinking_card[0]
        cursor_row = blinking_card[1]
        raster = jax.lax.cond(
            jnp.all(blinking_card != jnp.array([-1, -1], dtype=jnp.int32)),
            lambda r: self.jr.render_at(r, 10 + cursor_col * 32, y_positions[cursor_row], self.SHAPE_MASKS["cursor"]),
            lambda r: r,
            raster,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_blackjack(self, state) -> jnp.ndarray:
        card_matrix = jnp.zeros((6, 5), dtype=jnp.int32)
        card_matrix = card_matrix.at[0, :].set(state.cards_dealer[:5])
        card_matrix = card_matrix.at[2, :].set(state.cards_player_main[:5])
        card_matrix = card_matrix.at[4, :].set(state.cards_player_split[:5])

        char = jnp.select(
            [state.state_counter == 3, state.state_counter == 1],
            [0, 1],
            default=-1,
        ).astype(jnp.int32)

        player_split_bet = jnp.where(state.player_split_bet != 0, state.player_split_bet, -1).astype(jnp.int32)

        label_main = jnp.select(
            [
                jnp.isin(state.last_round_main, jnp.array([4, 5, 2, 3, 1], dtype=jnp.int32)) & jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)),
                jnp.isin(state.player_main_action, jnp.array([1, 2, 3, 0], dtype=jnp.int32)) & jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)),
            ],
            [state.last_round_main + 8, state.player_main_action + 3],
            default=-1,
        ).astype(jnp.int32)

        label_split = jnp.select(
            [
                jnp.isin(state.last_round_split, jnp.array([4, 5, 2, 3, 1], dtype=jnp.int32)) & jnp.isin(state.state_counter, jnp.array([1, 12], dtype=jnp.int32)),
                jnp.isin(state.player_split_action, jnp.array([1, 2, 0], dtype=jnp.int32)) & (state.is_splitting_selected == 1) & jnp.isin(state.state_counter, jnp.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)),
            ],
            [state.last_round_split + 8, state.player_split_action + 3],
            default=-1,
        ).astype(jnp.int32)

        return self._render_frame(
            card_matrix=card_matrix,
            player_score=state.player_score,
            char=char,
            player_main_bet=state.player_main_bet.astype(jnp.int32),
            player_split_bet=player_split_bet,
            label_main=label_main,
            label_split=label_split,
            blinking_card=jnp.array([-1, -1], dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_five_stud(self, state) -> jnp.ndarray:
        card_matrix = jnp.zeros((6, 5), dtype=jnp.int32)
        card_matrix = card_matrix.at[0, :].set(state.dealer_cards[:5])
        card_matrix = card_matrix.at[2, :].set(state.player_cards[:5])

        result_label = jnp.select(
            [state.last_round_result == 1, state.last_round_result == 2, state.last_round_result == 3],
            [12, 7, 9],
            default=-1,
        ).astype(jnp.int32)

        char = jnp.where((state.state_counter % 2 == 0) & (state.state_counter < 9), 1, -1).astype(jnp.int32)

        return self._render_frame(
            card_matrix=card_matrix,
            player_score=state.player_score,
            char=char,
            player_main_bet=state.round_bet.astype(jnp.int32),
            player_split_bet=state.total_bet.astype(jnp.int32),
            label_main=result_label,
            label_split=-1,
            blinking_card=jnp.array([-1, -1], dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_poker_solitaire(self, state) -> jnp.ndarray:
        board = state.board.reshape(5, 5).astype(jnp.int32)
        card_matrix = jnp.zeros((6, 5), dtype=jnp.int32)
        card_matrix = card_matrix.at[1:, :].set(board)

        top_row = jnp.array([0, 0, state.current_card, 0, 0], dtype=jnp.int32)
        card_matrix = card_matrix.at[0, :].set(top_row)

        cursor = jax.lax.cond(
            jnp.logical_and(state.cursor_pos_x >= 0, state.cursor_pos_y >= 0),
            lambda _: jnp.array([state.cursor_pos_x, state.cursor_pos_y + 1], dtype=jnp.int32),
            lambda _: jnp.array([-1, -1], dtype=jnp.int32),
            operand=None,
        )

        label_main = jnp.where(state.state_counter == 2, 13, -1).astype(jnp.int32)  # cut

        return self._render_frame(
            card_matrix=card_matrix,
            player_score=state.player_score,
            char=-1,
            player_main_bet=state.player_round_score.astype(jnp.int32),
            player_split_bet=-1,
            label_main=label_main,
            label_split=-1,
            blinking_card=cursor,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        if hasattr(state, "cards_player_main"):
            return self._render_blackjack(state)
        if hasattr(state, "player_cards"):
            return self._render_five_stud(state)
        if hasattr(state, "board"):
            return self._render_poker_solitaire(state)

        # Fallback to table background if an unknown state object is provided.
        return self.jr.render_from_palette(self.BACKGROUND, self.PALETTE)


class JaxCasino(JaxEnvironment):
    def __init__(self, consts=None, mode: str = "blackjack"):
        mode = (mode or "blackjack").lower()

        if mode == "blackjack":
            from jaxatari.games.jax_casino_blackjack import JaxCasinoBlackjack

            self.env = JaxCasinoBlackjack() if consts is None else JaxCasinoBlackjack(consts=consts)
        elif mode in ("five_stud", "five_stud_poker", "five-stud", "5stud"):
            from jaxatari.games.jax_casino_five_stud_poker import JaxCasinoFiveStudPoker

            self.env = JaxCasinoFiveStudPoker() if consts is None else JaxCasinoFiveStudPoker(consts=consts)
        elif mode in ("poker_solitaire", "poker_solitair", "solitaire", "poker-solitaire"):
            from jaxatari.games.jax_casino_poker_solitaire import JaxCasinoPokerSolitaire

            self.env = JaxCasinoPokerSolitaire() if consts is None else JaxCasinoPokerSolitaire(consts=consts)
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: blackjack, five_stud, poker_solitaire"
            )

        super().__init__(self.env.consts)
        # Expose renderer/action set like concrete game envs so external tools
        # (e.g. scripts/play.py) can treat this wrapper identically.
        self.renderer = self.env.renderer
        if hasattr(self.env, "ACTION_SET"):
            self.ACTION_SET = self.env.ACTION_SET

    def reset(self, key=None):
        if key is None:
            return self.env.reset()
        return self.env.reset(key)

    def step(self, state, action):
        return self.env.step(state, action)

    def render(self, state):
        # Use the wrapper-level renderer so renderer hot-swaps (e.g. native
        # downscaling) applied to this environment are respected.
        return self.renderer.render(state)

    def action_space(self):
        return self.env.action_space()

    def observation_space(self):
        return self.env.observation_space()

    def image_space(self):
        return self.env.image_space()

    def _get_observation(self, state):
        return self.env._get_observation(state)

    def _get_info(self, state, *args, **kwargs):
        return self.env._get_info(state)

    def _get_reward(self, previous_state, state):
        return self.env._get_reward(previous_state, state)

    def _get_done(self, state):
        return self.env._get_done(state)
