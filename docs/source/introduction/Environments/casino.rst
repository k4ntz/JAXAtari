Casino
======

The original Atari 2600 *Casino* cartridge contains three distinct games. JAXAtari
does not expose a combined ``casino`` environment and does not use ALE-style
``mode`` selection. Create each game separately:

.. code-block:: python

   import jaxatari

   blackjack = jaxatari.make("casinoblackjack")
   five_stud = jaxatari.make("casinofivestudpoker")
   solitaire = jaxatari.make("casinopokersolitaire")

These IDs are not the same as ``blackjack``, which is the separate Atari
*Blackjack* cartridge.

Casino Blackjack
----------------

``jaxatari.make("casinoblackjack")``

Casino Five Stud Poker
----------------------

``jaxatari.make("casinofivestudpoker")``

Casino Poker Solitaire
----------------------

``jaxatari.make("casinopokersolitaire")``
