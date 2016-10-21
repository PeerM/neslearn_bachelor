import numpy as np

from hsa.nes_python_input import py_to_nes_wrapper
from hsa.v_keras.qlearning4k.games.game import Game, PlayResult
from nes_python_interface import NESInterface
from hsa.dqn_mario.dqn_input import numpy_to_rdqn, rdqn_to_py
from hsa.reward_evaluation import mario_x_speed


def process_raw_frame(last_frame):
    return last_frame.reshape((1, 2048))


class FceuxMarioEmuGame(Game):
    def __init__(self, nes: NESInterface, nb_actions=37):
        self.nes = nes
        self.last_frame_ram = nes.getRAM()
        super().__init__("super mario bros.", nb_actions)
        self._nb_actions = nb_actions

    def get_current_state(self):
        return self.nes.getRAM()

    def play(self, action, repeat_actions=1):
        py_action = rdqn_to_py(action)
        for i in range(repeat_actions):
            self.nes.act(py_to_nes_wrapper(py_action))
        last_frame = self.nes.getRAM()
        return PlayResult(last_frame, mario_x_speed(last_frame), action, self.nes.lives() <= 1, False)

    def reset(self):
        self.nes.reset_game()

    def draw(self):
        return self.nes.getScreenRGB()
