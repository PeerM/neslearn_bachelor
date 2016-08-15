import numpy as np
import pandas

from hsa.dqn_mario.dqn_input import numpy_to_rdqn
from hsa.reward_evaluation import mario_x_speed


def load_memories(_memory):
    inputs = pandas.read_hdf("../mario_1_1_first.hdf", key="inputs")
    rams = pandas.read_hdf("../mario_1_1_first.hdf", key="rams")
    dqn_inputs = np.array([numpy_to_rdqn(row) for row in inputs.values])
    for i, s_prime in enumerate(rams.values[1:]):
        s_bloated = np.expand_dims(np.expand_dims(rams.values[i - 1], 0), 0)
        s_prime_bloated = np.expand_dims(np.expand_dims(s_prime, 0), 0)
        _memory.remember(s_bloated, dqn_inputs[i], mario_x_speed(s_prime), s_prime_bloated, False)