from hsa.nes_python_input import py_to_nes_wrapper
from hsa.visualization.parse_fm2 import parse_fm2
from nes_python_interface.nes_python_interface import NESInterface
import hsa.machine_constants
import pandas



def ram_from_movie(movie_path):
    nes = NESInterface(hsa.machine_constants.mario_rom_location, eb_compatible=False, auto_render_period=1)
    with open(movie_path) as movie_file:
        for combi in parse_fm2(movie_file):
            reward = nes.act(py_to_nes_wrapper(combi))
            yield nes.getRAM()

def main():
    #movie = "../bin_deps/fceux/movies/happylee-supermariobros,warped.fm2"
    movie = "../movies/5_1-1_without-shortcut.fm2"
    list(ram_from_movie(movie))


if __name__ == '__main__':
    main()