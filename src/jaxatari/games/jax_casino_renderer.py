import os
#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX Casino Renderer
#
# Renderer for the Atari Casino Games Blackjack, Five Stud Poker and Poker Solitaire
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember

from typing import TypeVar
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/casino/background.npy"), transpose=False)
    SPRITE_BACKGROUND = aj.get_sprite_frame(
        jnp.expand_dims(background, axis=0), 0)

    # Load i sprite
    isymbol = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/casino/i.npy"), transpose=False)
    SPRITE_ISYMBOL = aj.get_sprite_frame(
        jnp.expand_dims(isymbol, axis=0), 0)
    # Load questionmark sprite
    questionmark = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/casino/question.npy"), transpose=False)
    SPRITE_QUESTIONMARK = aj.get_sprite_frame(
        jnp.expand_dims(questionmark, axis=0), 0)
    # Load all label sprites
    labels = ["bet", "bj", "bust", "dble", "fold", "hit", "insr",
              "lose", "pass", "push", "split", "stay", "win", "cut"]

    def pad(a):
        a = jnp.asarray(a)
        if a.ndim == 2:
            a = a[..., None]
        elif a.ndim != 3:
            raise ValueError("Wrong dimension count")
        h, w, c = a.shape
        if c < 4:
            a = jnp.pad(a, ((0,0), (0,0), (0, 4 - c)), mode="constant", constant_values=0)
        elif c > 4:
            a = a[..., :4]
        dh, dw = max(0, 20 - h), max(0, 20 - w)
        t, b = dh // 2, dh - dh // 2
        l, r = dw // 2, dw - dw // 2
        a = jnp.pad(a, ((t,b), (l,r), (0,0)), mode="constant", constant_values=0)
        return a[:20, :20, :4]


    labels_array = []
    for label in labels:
        path = os.path.join(
            MODULE_DIR, "sprites/casino/labels/" + label + ".npy")
        frame = aj.loadFrame(path, transpose=False)
        labels_array.append(aj.get_sprite_frame(
            jnp.expand_dims(pad(frame), axis=0), 0))
    SPRITES_LABELS = jnp.array(labels_array)

    # Load all black card sprites
    numbers = ["2", "3", "4", "5", "6", "7",
               "8", "9", "10", "j", "k", "q", "a"]
    cards = []
    card_empty = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/casino/cards/turned.npy"), transpose=False)
    cards.append(aj.get_sprite_frame(
        jnp.expand_dims(card_empty, axis=0), 0))
    cards.append(aj.get_sprite_frame(
        jnp.expand_dims(card_empty, axis=0), 0))
    
    for number in numbers:
        path = os.path.join(
            MODULE_DIR, "sprites/casino/cards/diamond/" + number + ".npy")
        frame = aj.loadFrame(path)
        cards.append(aj.get_sprite_frame(
            jnp.expand_dims(frame, axis=0), 0))

    for number in numbers:
        path = os.path.join(
            MODULE_DIR, "sprites/casino/cards/club/" + number + ".npy")
        frame = aj.loadFrame(path, transpose=False)
        cards.append(aj.get_sprite_frame(
            jnp.expand_dims(frame, axis=0), 0))

    for number in numbers:
        path = os.path.join(
            MODULE_DIR, "sprites/casino/cards/heart/" + number + ".npy")
        frame = aj.loadFrame(path)
        cards.append(aj.get_sprite_frame(
            jnp.expand_dims(frame, axis=0), 0))
    for number in numbers:
        path = os.path.join(
            MODULE_DIR, "sprites/casino/cards/spade/" + number + ".npy")
        frame = aj.loadFrame(path)
        cards.append(aj.get_sprite_frame(
            jnp.expand_dims(frame, axis=0), 0))
    SPRITES_CARDS = jnp.array(cards)

    
    # Load cursor sprite
    cursor = aj.loadFrame(os.path.join(
        MODULE_DIR, "sprites/casino/cursor.npy"), transpose=False)
    SPRITE_CURSOR = aj.get_sprite_frame(
        jnp.expand_dims(cursor, axis=0), 0)


    # Load digits for player score
    SPRITES_DIGITS = aj.load_and_pad_digits(os.path.join(
        MODULE_DIR, "sprites/casino/digits/{}.npy"), num_chars=10)

    return (
        SPRITE_BACKGROUND,
        SPRITE_ISYMBOL,
        SPRITE_QUESTIONMARK,
        SPRITES_LABELS,
        SPRITES_CARDS,
        SPRITES_DIGITS,
        SPRITE_CURSOR
    )


class CasinoRenderer(JAXGameRenderer):
    def __init__(self):
        super().__init__()
        (
            self.SPRITE_BACKGROUND,
            self.SPRITE_ISYMBOL,
            self.SPRITE_QUESTIONMARK,
            self.SPRITES_LABELS,
            self.SPRITES_CARDS,
            self.SPRITES_DIGITS,
            self.SPRITE_CURSOR
        ) = load_sprites()

        self.counter = 0

    @partial(jax.jit, static_argnums=(0,))
    def render(self, card_matrix, player_score, width, height, char=-1, player_main_bet=-1, player_split_bet=-1, label_main=-1, label_split=-1, blinking_card=jnp.array([-1, -1]).astype(jnp.int32)):
        """ Responsible for the graphical representation of the game

        :param card_matrix: A 6x5 matrix showing which card is at which position
        :param player_score: The score of the player
        :param width: The width of the game ui
        :param height: The height of the game ui
        :param char: Indicates whether an i should be displayed or a question mark. -1 = None, 0 = i, 1 = question mark
        :param player_main_bet: The main bet of the player. -1 = None
        :param player_split_bet: The split bet of the player. -1 = None
        :param label_main: A label showing different actions: -1 = None, 0 = Bet, 1 = BJ, 2 = Bust, 3 = Double, 4 = Fold, 5 = Hit, 6 = Insr, 7 = Lose, 8 = Pass, 9 = Push, 10 = Split, 11 = Stay, 12 = Win
        :param label_split: A label showing different actions: -1 = None, 0 = Bet, 1 = BJ, 2 = Bust, 3 = Double, 4 = Fold, 5 = Hit, 6 = Insr, 7 = Lose, 8 = Pass, 9 = Push, 10 = Split, 11 = Stay, 12 = Win
        :param blinking_card: The blinking card coordinate. -1,-1 if not existent
        """
        raster = jnp.zeros((210, 160, 3))

        # render background
        raster = aj.render_at(raster, 0, 0, self.SPRITE_BACKGROUND)

        # render cards
        raster = jax.lax.fori_loop(
            lower=0,
            upper=len(card_matrix),
            body_fun=lambda i, val: jax.lax.fori_loop(
                lower=0,
                upper=len(card_matrix[i]),
                body_fun=lambda j, val2: jax.lax.cond(
                    pred=card_matrix[i][j] != 0,
                    true_fun=lambda v: aj.render_at(
                        v, 12 + j * 32, jax.lax.switch(i, [lambda: 8, lambda: 30, lambda: 68, lambda: 106, lambda: 144, lambda: 182]), self.SPRITES_CARDS[card_matrix[i][j] + 1]
                    ),
                    false_fun=lambda v: v,
                    operand=val2,
                ),
                init_val=val,
            ),
            init_val=raster,
        )

        # render player score
        raster = aj.render_label_selective(raster, 44, 53, aj.int_to_digits(
            player_score, max_digits=4), self.SPRITES_DIGITS, 0, 4, spacing=4)

        # render questionmark and i
        raster = jax.lax.cond(
            pred=char == 0,
            true_fun=lambda r: aj.render_at(r, 90, 53, self.SPRITE_ISYMBOL),
            false_fun=lambda r: jax.lax.cond(
                pred=char == 1,
                true_fun=lambda r2: aj.render_at(
                    r2, 89, 53, self.SPRITE_QUESTIONMARK),
                false_fun=lambda r2: r2,
                operand=r
            ),
            operand=raster
        )

        # render player main bet
        raster = jax.lax.cond(
            pred=player_main_bet == -1,
            true_fun=lambda r: r[0],
            false_fun=lambda r: aj.render_label_selective(r[0], 77, 53, aj.int_to_digits(
                r[1], max_digits=3), self.SPRITES_DIGITS, 0, 3, spacing=4),
            operand=(raster, player_main_bet)
        )

        # render player split bet
        raster = jax.lax.cond(
            pred=player_split_bet == -1,
            true_fun=lambda r: r[0],
            false_fun=lambda r: aj.render_label_selective(r[0], 77, 126, aj.int_to_digits(
                r[1], max_digits=3), self.SPRITES_DIGITS, 0, 3, spacing=4),
            operand=(raster, player_split_bet)
        )

        # render main label
        raster = jax.lax.cond(
            pred=label_main == -1,
            true_fun=lambda r: r[0],
            false_fun=lambda r:  aj.render_at(
                r[0], 104, 48, self.SPRITES_LABELS[r[1]],),
            operand=(raster, label_main)
        )

        # render split label
        raster = jax.lax.cond(
            pred=label_split == -1,
            true_fun=lambda r: r[0],
            false_fun=lambda r:  aj.render_at(
                r[0], 104, 121, self.SPRITES_LABELS[r[1]],),
            operand=(raster, label_split)
        )

        (bl, br) = blinking_card
        # render cursor
        raster = jax.lax.cond(
            pred=jnp.all(blinking_card != jnp.array([-1, -1])),
            true_fun=lambda r2:  aj.render_at(r2[0], 10 + r2[1] * 32, jax.lax.switch(r2[2], [lambda: 6, lambda: 28, lambda: 66, lambda: 104, lambda: 142, lambda: 180]), self.SPRITE_CURSOR
                ),
            false_fun=lambda r: r[0],
            operand=(raster, bl, br)
        )

        self.counter = self.counter + 1

        return raster
