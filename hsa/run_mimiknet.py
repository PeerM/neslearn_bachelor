from neon.models import Model
from neon.data import ArrayIterator
from neon.backends import gen_backend
from hsa import emu_connect
import importlib
import socket
import pandas
import numpy as np


def run_mimiknet(network_filename, speed, weighted_random, run_from_savestate):
    prim_soc = socket.create_connection(("localhost", 9090))

    emu = emu_connect.Emu2(prim_soc)

    gen_backend(backend='cpu', batch_size=1)
    nn = Model(network_filename)

    input_transform_keys = ["A", "B", "down", "left", "right", "select", "start",
                            "up"]
    weighted_random = False
    threshold = np.zeros(8, dtype=np.float32) + 0.5

    def neon_input_to_emu(neon_input: np.ndarray):
        # not sure if weighted random is a good joice
        if weighted_random:
            input_booleans = neon_input > np.random.normal(0.3, 0.2, 8)
        else:
            input_booleans = neon_input > threshold
        numpy_values = [r[0] for r in input_booleans.T]
        return dict(zip(input_transform_keys, numpy_values))

    def run_simulation(emu):
        if run_from_savestate:
            emu.load_slot(10)
        else:
            emu.softreset()
        emu.speed_mode(speed)
        emu.step()
        ram = emu.get_ram()
        for i in range(60 * 60 * 5):
            ram_iter = ArrayIterator(np.frombuffer(ram, dtype=np.uint8).reshape((1, 2048)), make_onehot=False)
            neon_input = nn.get_outputs(ram_iter)
            nex_input = neon_input_to_emu(neon_input)
            ram = emu.full_step(nex_input)

    try:
        emu.unpause()
        run_simulation(emu)
    except KeyboardInterrupt:
        # emu.pause()
        emu.close()


if __name__ == "__main__":
    run_mimiknet("mimik_4_mario_1_1_first.prm", "turbo", weighted_random=False, run_from_savestate=True)
