Environments
============

JAXAtari provides GPU-accelerated, object-centric implementations of classic Atari games.
Each environment exposes the same JAX-native interface, compatible with the Arcade Learning Environment (ALE).

.. code-block:: python

   from jaxatari import JAXtari

   env = JAXtari("seaquest")
   state = env.get_init_state()
   state, reward, done, info = env.step(state, action=0)

----

.. raw:: html

   <div class="env-grid">

     <a class="env-card" href="pong.html">
       <img src="../../_static/gifs/pong.gif" alt="Pong" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Pong</div>
     </a>

     <a class="env-card" href="kangaroo.html">
       <img src="../../_static/gifs/kangaroo.gif" alt="Kangaroo" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Kangaroo</div>
     </a>

     <a class="env-card" href="seaquest.html">
       <img src="../../_static/gifs/seaquest.gif" alt="Seaquest" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Seaquest</div>
     </a>

     <a class="env-card" href="freeway.html">
       <img src="../../_static/gifs/freeway.gif" alt="Freeway" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Freeway</div>
     </a>

     <a class="env-card" href="frostbite.html">
       <img src="../../_static/gifs/frostbite.gif" alt="Frostbite" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Frostbite</div>
     </a>

     <a class="env-card" href="beamrider.html">
       <img src="../../_static/gifs/beamrider.gif" alt="Beamrider" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Beamrider</div>
     </a>

     <a class="env-card" href="skiing.html">
       <img src="../../_static/gifs/skiing.gif" alt="Skiing" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Skiing</div>
     </a>

     <a class="env-card" href="montezumarevenge.html">
       <img src="../../_static/gifs/montezumarevenge.gif" alt="Montezuma's Revenge" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Montezuma's Revenge</div>
     </a>

     <a class="env-card" href="phoenix.html">
       <img src="../../_static/gifs/phoenix.gif" alt="Phoenix" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Phoenix</div>
     </a>

     <a class="env-card" href="venture.html">
       <img src="../../_static/gifs/venture.gif" alt="Venture" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Venture</div>
     </a>

     <a class="env-card" href="qbert.html">
       <img src="../../_static/gifs/qbert.gif" alt="Q*bert" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Q*bert</div>
     </a>

     <a class="env-card" href="battlezone.html">
       <img src="../../_static/gifs/battlezone.gif" alt="Battlezone" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Battlezone</div>
     </a>

   </div>
   <div class="env-load-more" id="env-load-more">
     <button onclick="var g=document.getElementById('env-grid-extra');g.querySelectorAll('img[data-src]').forEach(function(i){i.src=i.dataset.src});g.classList.remove('env-grid-extra');this.parentElement.style.display='none'">Show all environments</button>
   </div>
   <div class="env-grid env-grid-extra" id="env-grid-extra">

     <a class="env-card" href="airraid.html">
       <img data-src="../../_static/gifs/airraid.gif" alt="Air Raid" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Air Raid</div>
     </a>

     <a class="env-card" href="alien.html">
       <img data-src="../../_static/gifs/alien.gif" alt="Alien" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Alien</div>
     </a>

     <a class="env-card" href="amidar.html">
       <img data-src="../../_static/gifs/amidar.gif" alt="Amidar" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Amidar</div>
     </a>

     <a class="env-card" href="asterix.html">
       <img data-src="../../_static/gifs/asterix.gif" alt="Asterix" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Asterix</div>
     </a>

     <a class="env-card" href="asteroids.html">
       <img data-src="../../_static/gifs/asteroids.gif" alt="Asteroids" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Asteroids</div>
     </a>

     <a class="env-card" href="atlantis.html">
       <img data-src="../../_static/gifs/atlantis.gif" alt="Atlantis" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Atlantis</div>
     </a>

     <a class="env-card" href="bankheist.html">
       <img data-src="../../_static/gifs/bankheist.gif" alt="Bank Heist" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Bank Heist</div>
     </a>

     <a class="env-card" href="berzerk.html">
       <img data-src="../../_static/gifs/berzerk.gif" alt="Berzerk" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Berzerk</div>
     </a>

     <a class="env-card" href="blackjack.html">
       <img data-src="../../_static/gifs/blackjack.gif" alt="Blackjack" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Blackjack</div>
     </a>

     <a class="env-card" href="breakout.html">
       <img data-src="../../_static/gifs/breakout.gif" alt="Breakout" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Breakout</div>
     </a>

     <a class="env-card" href="casino.html">
       <img data-src="../../_static/gifs/casino.gif" alt="Casino" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Casino</div>
     </a>

     <a class="env-card" href="centipede.html">
       <img data-src="../../_static/gifs/centipede.gif" alt="Centipede" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Centipede</div>
     </a>

     <a class="env-card" href="choppercommand.html">
       <img data-src="../../_static/gifs/choppercommand.gif" alt="Chopper Command" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Chopper Command</div>
     </a>

     <a class="env-card" href="enduro.html">
       <img data-src="../../_static/gifs/enduro.gif" alt="Enduro" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Enduro</div>
     </a>

     <a class="env-card" href="fishingderby.html">
       <img data-src="../../_static/gifs/fishingderby.gif" alt="Fishing Derby" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Fishing Derby</div>
     </a>

     <a class="env-card" href="flagcapture.html">
       <img data-src="../../_static/gifs/flagcapture.gif" alt="Flag Capture" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Flag Capture</div>
     </a>

     <a class="env-card" href="galaxian.html">
       <img data-src="../../_static/gifs/galaxian.gif" alt="Galaxian" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Galaxian</div>
     </a>

     <a class="env-card" href="hangman.html">
       <img data-src="../../_static/gifs/hangman.gif" alt="Hangman" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Hangman</div>
     </a>

     <a class="env-card" href="hauntedhouse.html">
       <img data-src="../../_static/gifs/hauntedhouse.gif" alt="Haunted House" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Haunted House</div>
     </a>

     <a class="env-card" href="humancannonball.html">
       <img data-src="../../_static/gifs/humancannonball.gif" alt="Human Cannonball" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Human Cannonball</div>
     </a>

     <a class="env-card" href="kingkong.html">
       <img data-src="../../_static/gifs/kingkong.gif" alt="King Kong" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">King Kong</div>
     </a>

     <a class="env-card" href="klax.html">
       <img data-src="../../_static/gifs/klax.gif" alt="Klax" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Klax</div>
     </a>

     <a class="env-card" href="lasergates.html">
       <img data-src="../../_static/gifs/lasergates.gif" alt="Laser Gates" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Laser Gates</div>
     </a>

     <a class="env-card" href="namethisgame.html">
       <img data-src="../../_static/gifs/namethisgame.gif" alt="Name This Game" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Name This Game</div>
     </a>

     <a class="env-card" href="riverraid.html">
       <img data-src="../../_static/gifs/riverraid.gif" alt="River Raid" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">River Raid</div>
     </a>

     <a class="env-card" href="sirlancelot.html">
       <img data-src="../../_static/gifs/sirlancelot.gif" alt="Sir Lancelot" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Sir Lancelot</div>
     </a>

     <a class="env-card" href="slotmachine.html">
       <img data-src="../../_static/gifs/slotmachine.gif" alt="Slot Machine" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Slot Machine</div>
     </a>

     <a class="env-card" href="spaceinvaders.html">
       <img data-src="../../_static/gifs/spaceinvaders.gif" alt="Space Invaders" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Space Invaders</div>
     </a>

     <a class="env-card" href="spacewar.html">
       <img data-src="../../_static/gifs/spacewar.gif" alt="Spacewar" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Spacewar</div>
     </a>

     <a class="env-card" href="surround.html">
       <img data-src="../../_static/gifs/surround.gif" alt="Surround" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Surround</div>
     </a>

     <a class="env-card" href="tennis.html">
       <img data-src="../../_static/gifs/tennis.gif" alt="Tennis" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Tennis</div>
     </a>

     <a class="env-card" href="tetris.html">
       <img data-src="../../_static/gifs/tetris.gif" alt="Tetris" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Tetris</div>
     </a>

     <a class="env-card" href="timepilot.html">
       <img data-src="../../_static/gifs/timepilot.gif" alt="Time Pilot" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Time Pilot</div>
     </a>

     <a class="env-card" href="tron.html">
       <img data-src="../../_static/gifs/tron.gif" alt="Tron" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Tron</div>
     </a>

     <a class="env-card" href="turmoil.html">
       <img data-src="../../_static/gifs/turmoil.gif" alt="Turmoil" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Turmoil</div>
     </a>

     <a class="env-card" href="videocheckers.html">
       <img data-src="../../_static/gifs/videocheckers.gif" alt="Video Checkers" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Video Checkers</div>
     </a>

     <a class="env-card" href="videocube.html">
       <img data-src="../../_static/gifs/videocube.gif" alt="Video Cube" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Video Cube</div>
     </a>

     <a class="env-card" href="videopinball.html">
       <img data-src="../../_static/gifs/videopinball.gif" alt="Video Pinball" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Video Pinball</div>
     </a>

     <a class="env-card" href="wordzapper.html">
       <img data-src="../../_static/gifs/wordzapper.gif" alt="Word Zapper" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
       <div class="env-placeholder" style="display:none">🕹</div>
       <div class="env-name">Word Zapper</div>
     </a>

   </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   seaquest
