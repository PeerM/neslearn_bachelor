import pandas
import hsa
import numpy as np
from hsa.dqn_mario.dqn_input import pandas_to_dqn, numpy_to_dqn, dqn_to_py
import hsa.reward_evaluation as re
from hsa.simple_dqn.deepqnetwork import DeepQNetwork
from hsa.simple_dqn.replay_memory import ReplayMemory
from hsa.dqn_mario.dqn_argparse import parse_args

args = parse_args("")

inputs = pandas.read_hdf("mario_1_1_first.hdf", key="inputs")
rams = pandas.read_hdf("mario_1_1_first.hdf", key="rams")

dqn_inputs = np.array([numpy_to_dqn(row) for row in inputs.values])

rewarder = re.MultiReward(re.MarioDeath(), re.MarioScore(), re.MarioXAcceleration())

rewards = np.array([rewarder.reward(row) for row in rams.values])

replay_memory = ReplayMemory(8000, args)

ram_np = rams.values
for i in range(rewards.shape[0]):
    replay_memory.add(dqn_inputs[i], rewards[i], ram_np[i], False)

dqn = DeepQNetwork(255, args)

dqn.train(replay_memory.getMinibatch(), 0)

prediction = dqn.predict(np.array([[x] for x in rams.values[2548:2548 + 8]]))
print(prediction)
for batch in prediction:
     print(dqn_to_py(batch.argmax()))
