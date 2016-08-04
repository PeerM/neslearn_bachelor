import socket
from hsa import emu_connect
import math


def _unsigned_to_singed(byte):
    if byte > 127:
        return (256 - byte) * (-1)
    else:
        return byte


class MarioXAcceleration(object):
    def __init__(self):
        self.last = 0

    def reward(self, ram: bytes):
        raw = _unsigned_to_singed(ram[0x0057])
        current = math.floor(raw / 5)
        delta = current - self.last
        self.last = current
        # return current
        return delta
        # return math.tanh(delta)


class MarioScore(object):
    def __init__(self):
        self.last = 0

    def reward(self, ram: bytes):
        # read the successive ram addresses join it as as string and read it as a int. I know there are better ways with 2 powers
        current = int("".join([str(ram[0x07DE + i]) for i in range(6)]))
        delta = current - self.last
        self.last = current
        # return current
        return min(delta / 1000, 10)
        # return math.tanh(delta)


class MarioDeath(object):
    def __init__(self):
        self.dying = False

    def reward(self, ram: bytes):
        # return -10 if ram[0x000E] == 0x0B else 0
        if self.dying:
            if ram[0x000E] != 0x0B:
                self.dying = False
                return 0
            else:
                return 0
        else:
            if ram[0x000E] == 0x0B:
                self.dying = True
                return -30
            else:
                return 0


class MultiReward(object):
    def __init__(self, *rewarders):
        self.rewarders = rewarders

    def reward(self, ram):
        return sum([r.reward(ram) for r in self.rewarders])


def evaluate(run_from_savestate, rewarder):
    prim_soc = socket.create_connection(("localhost", 9090))

    emu = emu_connect.Emu2(prim_soc)

    def run_simulation(emu):
        if run_from_savestate:
            emu.load_slot(10)
        emu.speed_mode("normal")
        for i in range(60 * 60 * 1):
            emu.step()
            ram = emu.get_ram()
            print(rewarder.reward(ram))

    try:
        emu.unpause()
        run_simulation(emu)
    except KeyboardInterrupt:
        # emu.pause()
        emu.close()


if __name__ == "__main__":
    evaluate(True, MultiReward(MarioScore(), MarioXAcceleration(), MarioDeath()))
