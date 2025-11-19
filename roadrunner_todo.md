- [ ] Check the actual reference implementation of the game, not just the atari emulator online
- [ ] Find out exactly what each level looks like and needs

## Rendering
- [ ] Proper color hues for player and enemy sprites
- [ ] Better matching color for the desert background
- [ ] Counter for lives
- [ ] Level transition screens

### Missing Sprites
- [ ] Seeds
- [ ] Background bushes
- [ ] Background mountains
- [ ] Cars
- [ ] Burnt Roadrunner
- [ ] Jumping Roadrunner
- [ ] Run over enemy
- [ ] Ravine
- [ ] Background Signs
- [ ] Score (total at the top and picked up at the bottom)
- [ ] Lives icon

## Scoring system
- [ ] Make seeds pickupable
- [ ] Seeds score 100 points, each subsequent seed increases seed score by 100 up to 1000
    - When a seed is missed, the points per seed get reset to 100
- [ ] 1000 Points when the enemy gets run over by a car
- [ ] Seeds are spawned in semi random patterns (find out what the patterns are)

## Life system
- [ ] The player has 3 lives
- [ ] lose a life on every death
    - Enemy catches the player
    - Player gets hit by car
    - Player steps on landmine
    - Player falls in ravine

## Level system
- [ ] After a certain time (or distance covered) the current level ends and the next one starts
- [ ] Each level looks different or has differen mechanics
    - [ ] Cars
    - [ ] Ravines
    - [ ] Landmines
    - [ ] Narrowing and widening road
    - [ ] Not every level spawns seeds
    - [ ] more...

## Enemy Behavior
- [x] Follows player somewhat delayed
- [ ] Changes speed semi randomly (find out what the pattern is)
- [ ] Can go off screen if the player is fast enough
- [ ] Can be run over by cars
- [ ] Backs of when the player moves directly in the direction of the enemy

## Player Movement
- [ ] Jump
