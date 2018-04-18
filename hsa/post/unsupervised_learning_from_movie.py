from PIL import Image

from extern.fceux_learningenv.nes_python_interface import NESInterface
from hsa import machine_constants
from hsa.nes_python_input import py_to_nes_wrapper
from hsa.visualization.parse_fm2 import parse_fm2

import numpy as np

def movie_to_train(movie_path):
    nes = NESInterface(machine_constants.mario_rom_location, eb_compatible=False, auto_render_period=1)
    with open(movie_path) as movie_file:
        for combi in parse_fm2(movie_file):
            reward = nes.act(py_to_nes_wrapper(combi))
            screen = nes.getScreenRGB()
            yield screen

def scale_down(frame, size):
    im = Image.fromarray(frame)
    small = im.transform(size)
    return small

movie = "../movies/5_1-1_without-shortcut.fm2"
video = np.stack(movie_to_train(movie))
def main():
    pass
    #movie = "../bin_deps/fceux/movies/happylee-supermariobros,warped.fm2"

if __name__ == '__main__':
    main()