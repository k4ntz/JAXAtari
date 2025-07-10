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
height=12
width=10

#0 we dont need numer zero
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:12, 0:2] = white_turquoise_rgba
frame[0:12, 8:10] = white_turquoise_rgba
frame[0:2, 2:8] = white_turquoise_rgba
frame[10:12, 2:8] = white_turquoise_rgba
np.save("number_0_player.npy", frame)
frame[0:12, 0:2] = black_rgba
frame[0:12, 8:10] = black_rgba
frame[0:2, 2:8] = black_rgba
frame[10:12, 2:8] = black_rgba
np.save("number_0_enemy.npy", frame)

#1 
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[2:4, 0:4] = white_turquoise_rgba
frame[0:10, 4:6] = white_turquoise_rgba
frame[10:12, 0:10] = white_turquoise_rgba
np.save("number_1_player.npy", frame)
frame[2:4, 0:4] = black_rgba
frame[0:10, 4:6] = black_rgba
frame[10:12, 0:10] = black_rgba
np.save("number_1_enemy.npy", frame)

#2
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:1, 0:10] = white_turquoise_rgba
frame[1:5, 7:10] = white_turquoise_rgba
frame[5:6, 0:10] = white_turquoise_rgba
frame[6:10, 0:2] = white_turquoise_rgba
frame[10:12, 0:10] = white_turquoise_rgba
np.save("number_2_player.npy", frame)
frame[0:1, 0:10] = black_rgba
frame[1:5, 7:10] = black_rgba
frame[5:6, 0:10] = black_rgba
frame[6:10, 0:2] = black_rgba
frame[10:12, 0:10] = black_rgba
np.save("number_2_enemy.npy", frame)

#3
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:1, 0:10] = white_turquoise_rgba
frame[5:6, 4:10] = white_turquoise_rgba
frame[10:12, 0:10] = white_turquoise_rgba
frame[0:12, 7:10] = white_turquoise_rgba
np.save("number_3_player.npy", frame)
frame[0:1, 0:10] = black_rgba
frame[5:6, 4:10] = black_rgba
frame[10:12, 0:10] = black_rgba
frame[0:12, 7:10] = black_rgba
np.save("number_3_enemy.npy", frame)

#4
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:7, 0:3] = white_turquoise_rgba
frame[6:7, 3:10] = white_turquoise_rgba
frame[0:12, 7:10] = white_turquoise_rgba
np.save("number_4_player.npy", frame)
frame[0:7, 0:3] = black_rgba
frame[6:7, 3:10] = black_rgba
frame[0:12, 7:10] = black_rgba
np.save("number_4_enemy.npy", frame)

#5
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:2, 0:10] = white_turquoise_rgba
frame[2:5, 0:2] = white_turquoise_rgba
frame[5:6, 0:10] = white_turquoise_rgba
frame[6:9, 7:10] = white_turquoise_rgba
frame[9:12, 0:10] = white_turquoise_rgba
np.save("number_5_player.npy", frame)
frame[0:2, 0:10] = black_rgba
frame[2:5, 0:2] = black_rgba
frame[5:6, 0:10] = black_rgba
frame[6:9, 7:10] = black_rgba
frame[9:12, 0:10] = black_rgba
np.save("number_5_enemy.npy", frame)

#6
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:1, 0:10] = white_turquoise_rgba
frame[1:5, 0:3] = white_turquoise_rgba
frame[5:6, 0:10] = white_turquoise_rgba
frame[6:10, 0:3] = white_turquoise_rgba
frame[6:10, 7:10] = white_turquoise_rgba
frame[10:12, 0:10] = white_turquoise_rgba
np.save("number_6_player.npy", frame)
frame[0:1, 0:10] = black_rgba
frame[1:5, 0:3] = black_rgba
frame[5:6, 0:10] = black_rgba
frame[6:10, 0:3] = black_rgba
frame[6:10, 7:10] = black_rgba
frame[10:12, 0:10] = black_rgba
np.save("number_6_enemy.npy", frame)

#7
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:3, 0:10] = white_turquoise_rgba
frame[3:11, 6:10] = white_turquoise_rgba
np.save("number_7_player.npy", frame)
frame[0:3, 0:10] = black_rgba
frame[3:11, 6:10] = black_rgba
np.save("number_7_enemy.npy", frame)


#8
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:1, 0:10] = white_turquoise_rgba
frame[1:5, 0:3] = white_turquoise_rgba
frame[1:5, 7:10] = white_turquoise_rgba
frame[5:6, 0:10] = white_turquoise_rgba
frame[6:10, 0:3] = white_turquoise_rgba
frame[6:10, 7:10] = white_turquoise_rgba
frame[10:12, 0:10] = white_turquoise_rgba
np.save("number_8_player.npy", frame)
frame[0:1, 0:10] = black_rgba
frame[1:5, 0:3] = black_rgba
frame[1:5, 7:10] = black_rgba
frame[5:6, 0:10] = black_rgba
frame[6:10, 0:3] = black_rgba
frame[6:10, 7:10] = black_rgba
frame[10:12, 0:10] = black_rgba
np.save("number_8_enemy.npy", frame)


#9
frame = np.full((height, width, 4), green_rgba, dtype=np.uint8)
frame[0:2, 0:10] = white_turquoise_rgba
frame[2:5, 0:2] = white_turquoise_rgba
frame[2:5, 7:10] = white_turquoise_rgba
frame[5:6, 0:10] = white_turquoise_rgba
frame[6:10, 7:10] = white_turquoise_rgba
frame[10:12, 0:10] = white_turquoise_rgba
np.save("number_9_player.npy", frame)
frame[0:2, 0:10] = black_rgba
frame[2:5, 0:2] = black_rgba
frame[2:5, 7:10] = black_rgba
frame[5:6, 0:10] = black_rgba
frame[6:10, 7:10] = black_rgba
frame[10:12, 0:10] = black_rgba
np.save("number_9_enemy.npy", frame)
