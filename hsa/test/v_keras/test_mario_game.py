from unittest import TestCase
import numpy as np
import pandas

from hsa.dqn_mario.dqn_input import rdqn_to_py
from hsa.v_keras.mario_game import MarioReplay


class TestMarioReplay(TestCase):
    def test_play(self):
        nr = 200
        replay = MarioReplay(pandas.DataFrame.from_dict([rdqn_to_py(0)]*nr), pandas.DataFrame([[0]*2048]*nr))
        stop = False
        while not stop :
            stop = replay.play(0).is_over
