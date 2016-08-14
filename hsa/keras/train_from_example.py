import pandas
import numpy as np
from hsa.dqn_mario.dqn_input import numpy_to_rdqn, rdqn_to_py

inputs = pandas.read_hdf("mario_1_1_first.hdf", key="inputs")
rams = pandas.read_hdf("mario_1_1_first.hdf", key="rams")
dqn_inputs = np.array([numpy_to_rdqn(row) for row in inputs.values])
