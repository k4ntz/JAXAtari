# Blackjack

## Running the Game
The game can be run with two parameters used for setting the shuffle behavior and and the rule set of the game.

### Setting the rule set 

```bash
python3 jax_blackjack.py --difficulty <value>
```
Value must be between 0 and 1, 0 for casino rules (default if not specified) and 1 for private rules.

#### CASINO BLACK JACK RULES

Computer dealer must hit a soft 17 or less. (Soft means, that an Ace is used as 11 points).
Computer dealer must stay on a hard 17. (Hard means, that any combination of cards is used except an Ace worth 11 points).
The player gets no points but he looses also no points if there is a tie.
The player is only allowed to hit double before the first hit and he needs to have 10 or 11 points.
A player is allowed four hits.

#### PRIVATE BLACK JACK RULES

Computer dealer must stay on 17 or more points.
The dealer wins all tie games.
The player is only allowed to hit double before the first hit but with any combination of cards.
A player wins the game when he hits four times without busting. 

### Setting the shuffle behavior

```bash
python3 jax_blackjack.py --shuffle <value>
```
Value must be between 0 and 1, 0 for shuffling at the end of every round (default) and 1 for shuffling after drawing 34 cards.

### Controls

The Game is controlled using 'W', 'S' and 'Space'. 'W' usually indicates up, 'S' usually indicates down and 'Space' usually indicates confirming the input.
