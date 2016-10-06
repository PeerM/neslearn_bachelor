import numpy as np
from hsa.v_keras.qlearning4k.games.game import Game, PlayResult
from hsa import emu_connect
from hsa.dqn_mario.dqn_input import numpy_to_rdqn, rdqn_to_py
from hsa.reward_evaluation import mario_x_speed


def process_raw_frame(last_frame):
    return np.frombuffer(last_frame, dtype=np.uint8)  # .reshape((1, 2048))


# TODO write action encoders construct

class MarioEmuGame(Game):
    def __init__(self, emu: emu_connect.Emu2, nb_actions=37):
        self.emu = emu
        self.last_frame_ram = process_raw_frame(emu.get_ram())
        super().__init__("super mario bros.", nb_actions)
        self._nb_actions = nb_actions

    def get_current_state(self):
        return process_raw_frame(self.emu.get_ram())

    def play(self, action, repeat_actions=1):
        step = self.emu.step_repeat_actions(rdqn_to_py(action), repeat_actions)
        if len(step) != 2048:
            raise AssertionError()
        last_frame = process_raw_frame(step)
        return PlayResult(last_frame, mario_x_speed(last_frame), action, last_frame[0x075A] <= 1, False)

    def reset(self):
        self.emu.load_slot_async(10)

    def draw(self):
        return self.last_frame_ram.reshape((32, 64))


class MarioReplay(Game):
    def __init__(self, actions, states, nb_actions=37):
        super().__init__("super mario bros.", nb_actions)
        self.states = states
        self.actions = actions
        self.index = 0
        self.nb_frames = self.states.shape[0]

    def get_current_state(self):
        return self.states.iloc[self.index]

    def play(self, action, repeat_actions=1):
        index_now = self.index
        last_frame = self.states.iloc[index_now]
        self.index += 1
        return PlayResult(last_frame,
                          mario_x_speed(last_frame),
                          numpy_to_rdqn(self.actions.iloc[index_now]),
                          self.index >= self.nb_frames, False)

    def reset(self):
        self.index = 0
