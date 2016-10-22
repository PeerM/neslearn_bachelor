from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface
import numpy as np
import matplotlib.pyplot as plt
import imageio
from hsa.nes_python_input import *

mario_rom_path = "/home/peer/mario.nes"
nes = NESInterface(mario_rom_path)
movie_writer = imageio.get_writer("~/mario_video.mp4", fps=60)

nes.reset_game()
nes.reset_game()
for i in range(10000):
    if i % 70 <= 20:
        print("jump")
        nes.act(A)
    else:
        print("run")
        nes.act(Right)
    movie_writer.append_data(nes.getScreenRGB())
    if nes.game_over():
        print("game over")
        nes.reset_game()
        break

# ram_for_ref = np.array(2048, dtype=np.uint8)
ram = nes.getRAM()

print(ram)
movie_writer.close()
