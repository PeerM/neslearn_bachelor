import socket

from hsa import emu_connect
from hsa.nes_python_input import py_to_nes_wrapper
from hsa.reward_evaluation import mario_x_speed
from hsa.visualization.liveplot2 import Plotter
from nes_python_interface.nes_python_interface import NESInterface
import hsa.machine_constants
from hsa.visualization.liveplot import LivePlotter
import pandas
inputs = pandas.read_hdf("../mario_1_1_first.hdf", key="inputs")
movie = "../bin_deps/fceux/movies/happylee-supermariobros,warped.fm2"
nes = NESInterface(hsa.machine_constants.mario_rom_location,eb_compatible=False,auto_render_period=1)


plotter = Plotter(interval=100, history_length=2000)

plotter.start()
print(inputs.iloc[60])
print(py_to_nes_wrapper(inputs.iloc[60]))

def visualize_ram(ram):
    plotter.plot(mario_x_speed(ram))

try:
    nes.reset_game()
    for row in inputs.iterrows():
        reward = nes.act(py_to_nes_wrapper(row[1]))
        # plotter.plot(mario_x_speed(nes.getRAM()))
        plotter.plot(reward)
    plotter.stop()
except KeyboardInterrupt:
    pass
    #emu.close()