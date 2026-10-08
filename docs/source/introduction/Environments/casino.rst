Casino
======

The original Atari 2600 *Casino* cartridge contains three distinct games. Jaxtari
does not expose a combined ``casino`` environment and does not use ALE-style
``mode`` selection. Create each game separately:

.. code-block:: python

   import jaxtari

   blackjack = jaxtari.make("casinoblackjack")
   five_stud = jaxtari.make("casinofivestudpoker")
   solitaire = jaxtari.make("casinopokersolitaire")

These IDs are not the same as ``blackjack``, which is the separate Atari
*Blackjack* cartridge.

Casino Blackjack
----------------

``jaxtari.make("casinoblackjack")``

Casino Five Stud Poker
----------------------

``jaxtari.make("casinofivestudpoker")``

Casino Poker Solitaire
----------------------

``jaxtari.make("casinopokersolitaire")``

Known issues
------------
