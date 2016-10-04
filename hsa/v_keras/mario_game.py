import numpy as np
from hsa.v_keras.qlearning4k.games.game import Game
from hsa import emu_connect
from hsa.dqn_mario.dqn_input import numpy_to_rdqn, rdqn_to_py
from hsa.reward_evaluation import mario_x_speed


def process_raw_frame(last_frame):
    return np.frombuffer(last_frame, dtype=np.uint8)  # .reshape((1, 2048))


class MarioEmuGame(Game):
    def __init__(self, emu: emu_connect.Emu2, nb_actions=37):
        self.emu = emu
        self.last_frame_ram = process_raw_frame(emu.get_ram())
        super().__init__()
        self._nb_actions = nb_actions

    def get_state(self):
        return self.last_frame_ram

    def play(self, action, repeat_actions=1):
        step = self.emu.step_repeat_actions(rdqn_to_py(action), repeat_actions)
        if len(step) != 2048:
            raise AssertionError()
        self.last_frame_ram = process_raw_frame(step)

    def reset(self):
        self.emu.load_slot_async(10)

    def is_over(self):
        isover = self.last_frame_ram[0x075A] <= 1
        return isover

    def get_score(self):
        return mario_x_speed(self.last_frame_ram)

    def draw(self):
        return self.last_frame_ram.reshape((32, 64))

    @property
    def nb_actions(self):
        return self._nb_actions

    @property
    def name(self):
        return "Nes Mario"
