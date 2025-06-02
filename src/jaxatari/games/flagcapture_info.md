# Links
## Wikipedia
https://en.wikipedia.org/wiki/Flag_Capture

## Spielbare Version
https://atarionline.org/atari-2600/flag-capture

## ALE Documentation
https://ale.farama.org/environments/flag_capture/

## Handbuch
https://www.gamesdatabase.org/Media/SYSTEM/Atari_2600/Manual/formated/Flag_Capture_-_1978_-_Atari.pdf

# Wie das Spiel funktioniert
## Hinweis zu den Modi
Wir implementieren Modus 8. Modi 1-7 sind Mehrspieler-Modi, die nicht unterstützt werden.

| Game Number | Number of Players | Stationary Flags | Moving Flag with Wall | Moving Flag with Wraparound |
|-------------|-------------------|------------------|-----------------------|-----------------------------|
| 1           | 2☠️               | X                |                       |                             |
| 2           | 2☠️               | X                |                       |                             |
| 3           | 2☠️               |                  | X                     |                             |
| 4           | 2☠️               |                  |                       | X                           |
| 5           | 2☠️               | X                |                       |                             |
| 6           | 2☠️               |                  | X                     |                             |
| 7           | 2☠️               |                  |                       | X                           |
| _**8**_           | _**1**_                 | _**X**_                |                       |                             |
| 9           | 1                 |                  | X                     |                             |
| 10          | 1                 |                  |                       | X                           |

## Spielregeln
- Auf einem 9 x 7 Raster ist eine Flagge versteckt.
- Der Spieler kann sich in 8 Richtungen bewegen.
- Eine Bewegungseingabe während dem Aufdecken eines Feldes wird ignoriert.
- Eine Physisch unmögliche Bewegung führt im originalen Spiel zu einem Teleport nach unten rechts.
- Der Spieler kann Felder aufdecken.
  - Deckt er die Flagge auf bekommt, er einen Punkt und das Feld wird neu generiert.
  - Deckt er eine Bombe auf, wird das Feld neu generiert.
  - Alle anderen Felder haben einen Richtungs- oder Entfernungs-Hinweis die aufgedeckt werden können.
    - Richtungshinweise sind Pfeile die in eine von 8 Richtungen zeigen.
    - Horizontale und vertikale Pfeile zeigen direkt in die Richtung der Flagge.
    - Diagonale Pfeile grenzen die Flagge auf den quadranten vom Hinweis aus ein.
    - Entfernungs-Hinweise sind Zahlen von 1 bis 8.
- Der Spieler hat 75 Sekunden um möglichst viele Flaggen aufzudecken.
- Es gibt keine Aktion, die zu einer Zeitstrafe führt oder die Zeit verlängert.