import numpy as np

# Erzeuge ein Dummy-Hintergrundbild (z. B. braun mit voller Deckkraft)
height, width = 210, 160
green_rgba = [61, 137, 37, 255]
black_rgba = [0, 0, 0, 255]

# Erstelle das Bild als Array
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)

# Rows
frame[16:19, 12:143] = black_rgba
frame[38:41, 12:143] = black_rgba
frame[60:63, 12:143] = black_rgba
frame[82:85, 12:143] = black_rgba
frame[104:107, 12:143] = black_rgba
frame[126:129, 12:143] = black_rgba
frame[148:151, 12:143] = black_rgba
frame[170:173, 12:143] = black_rgba
frame[192:195, 12:143] = black_rgba

# Columns
frame[16:195, 12:15] = black_rgba
frame[16:195, 28:31] = black_rgba
frame[16:195, 44:47] = black_rgba
frame[16:195, 60:63] = black_rgba
frame[16:195, 76:79] = black_rgba
frame[16:195, 92:95] = black_rgba
frame[16:195, 108:111] = black_rgba
frame[16:195, 124:127] = black_rgba
frame[16:195, 140:143] = black_rgba

# Speichere als .npy-Datei
np.save("othello_background.npy", frame)


player_white_disc_height, player_white_disc_width = 14, 8
enemy_black_disc_height, enemy_black_disc_width = 14, 8

white_turquoise_rgba = [132, 252, 212, 255]

player_frame = np.full((player_white_disc_height, player_white_disc_width, 4), white_turquoise_rgba, dtype=np.uint8)
np.save("player_white_disc.npy", player_frame)

enemy_frame = np.full((enemy_black_disc_height, enemy_black_disc_width, 4), black_rgba, dtype=np.uint8)
np.save("enemy_black_disc.npy", enemy_frame)


# Numbers
height=10
width=12

#1 
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:4, 4:7] = green_rgba
frame[6:11, 0:7] = green_rgba
np.save("number_1_player.npy", frame)
frame[0:4, 4:7] = black_rgba
frame[6:11, 0:7] = black_rgba
np.save("number_1_enemy.npy", frame)

#2

#3

#4
frame = np.full((height, width, 4), white_turquoise_rgba, dtype=np.uint8)
frame[0:4, 4:7] = green_rgba
frame[6:11, 0:7] = green_rgba
np.save("number_4_player.npy", frame)
frame[0:4, 4:7] = black_rgba
frame[6:11, 0:7] = black_rgba
np.save("number_4_enemy.npy", frame)