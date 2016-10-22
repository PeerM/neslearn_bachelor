import datetime

from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface
from hsa.nes_python_input import *

# mario_rom_path = "/home/peer/mario.nes"
mario_rom_path = "/home/mathia/fceux/mario.nes"
nes = NESInterface(mario_rom_path)

nes.reset_game()
for i in range(200):
    start_time = datetime.datetime.now()
    for j in range(240):
        nes.act(A)
    end_time = datetime.datetime.now()
    print(end_time - start_time)