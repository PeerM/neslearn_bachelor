from nes_python_interface import NESInterface

from hsa.gen3.nes_env import NesEnv
from hsa.gen3.process import DynamicProxyProcess


def fceux_process_factory(rom_location, **kwargs):
    def nes_factory():
        return NESInterface(rom=rom_location)

    return NesEnv(game_path=None, nes=DynamicProxyProcess(nes_factory), **kwargs)
