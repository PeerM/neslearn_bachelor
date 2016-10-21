from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface
import numpy as np
import matplotlib.pyplot as plt

mario_rom_path="/home/peer/playground/marionn/Super Mario Bros. (JU) [!].nes"
nes = NESInterface(mario_rom_path)

# ram_for_ref = np.array(2048, dtype=np.uint8)
ram = nes.getRAM()

print(ram)