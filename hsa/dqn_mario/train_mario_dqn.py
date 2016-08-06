import pandas
import hsa
import numpy as np

from hsa.dqn_mario.dqn_configurations import Deep3QNetwork
from hsa.dqn_mario.dqn_input import pandas_to_dqn, numpy_to_dqn, dqn_to_py, numpy_to_rdqn, rdqn_to_py
import hsa.reward_evaluation as re
from hsa.simple_dqn.deepqnetwork import DeepQNetwork
from hsa.simple_dqn.replay_memory import ReplayMemory
from hsa.dqn_mario.dqn_argparse import parse_args

args = parse_args("")
nr_actions = 37

inputs = pandas.read_hdf("mario_1_1_first.hdf", key="inputs")
rams = pandas.read_hdf("mario_1_1_first.hdf", key="rams")

dqn_inputs = np.array([numpy_to_rdqn(row) for row in inputs.values])

rewarder = re.MultiReward(re.MarioDeath(), re.MarioScore(), re.MarioXAcceleration())

rewards = np.array([rewarder.reward(row) for row in rams.values])

replay_memory = ReplayMemory(10000, args)

ram_np = rams.values
for i in range(rewards.shape[0]):
    replay_memory.add(dqn_inputs[i], rewards[i], ram_np[i], False)

dqn = Deep3QNetwork(nr_actions, args)

for i in range(500):
    dqn.train(replay_memory.getMinibatch(), 0)
    print("period {} over".format(i))

dqn.save_weights("dqn_weights/3LayerNorm/3")

prediction = dqn.predict(np.array(rams.values[1337:1337 + 8]))
# prediction = dqn.predict(np.array([[x] for x in rams.values[1337:1337 + 8]]))
# print(prediction[0].shape)
# print(prediction[0].argsort()[4])
for batch in prediction:
    dqn_input = batch.argmax()
    print(dqn_input)
    print(rdqn_to_py(dqn_input))
