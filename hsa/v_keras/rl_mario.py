import socket
import uuid

import pandas

from hsa import emu_connect
from hsa.v_keras.mario_game import MarioEmuGame
from hsa.v_keras.memories import load_memories
from hsa.v_keras.qlearning4k import Agent
from hsa.v_keras.qlearning4k import ExperienceReplay
import hsa.v_keras.model_zoo as zoo

# parameters
nb_frames = 1
ram_size = 2048
nr_actions = 36
play_period = 20
batch_size = play_period * 32
nr_epoch = 500
save_every_n_epochs = 50

model_filename = None
# model_filename = "./M1"
# model_filename = "../dqn_weights/keras/P3_1Layer"
# model_filename = "../dqn_weights/keras/tmp/5_layer_firsttime"
memories_filename = "../mario_1_1_third.hdf"

model_name = str(uuid.uuid1())
# Model hyper parameters
model = zoo.make_8layer_unstable(nb_frames, ram_size, nr_actions)

# second*minute*hour*n
memory = ExperienceReplay(memory_size=60 * 60 * 60 * 2)
if model_filename:
    model.load_weights(model_filename + ".hdf5")
    load_memories(memory, memories_filename)
    epsilon = (0.1, 0.01)
    epsilon_rate = 0.4
else:
    epsilon = (0.4, 0)
    epsilon_rate = 0.7
    load_memories(memory, memories_filename)

prim_soc = socket.create_connection(("localhost", 9090))
prim_soc.setblocking(True)
print("connecting")
emu = emu_connect.Emu2(prim_soc)
emu.speed_mode("turbo")
emu.load_slot(10)
print("connected")

game = MarioEmuGame(emu, nr_actions)
agent = Agent(model, memory=memory, nb_frames=nb_frames)
print("starting")
epoch_results = list()
try:
    training_generator = agent.train(game, nb_epoch=nr_epoch, epsilon=epsilon, play_period=play_period, batch_size=batch_size,
                                     action_repeat=4, epsilon_rate=epsilon_rate)
    for epoch, epoch_result in enumerate(training_generator):
        print(epoch_result)
        epoch_results.append(epoch_result)
        if epoch % save_every_n_epochs == 0:
            model.save_weights("../dqn_weights/keras/tmp/{}_{}.hdf5".format(model_name, epoch))
            # agent.play(game, nb_epoch=4)
finally:
    print("stopping")
    model.save_weights("../dqn_weights/keras/tmp/{}_{}.hdf5".format(model_name, "final"))
    # noinspection PyTypeChecker
    pandas.DataFrame.from_dict(epoch_results).to_csv("../dqn_weights/keras/tmp/{}.csv".format(model_name))
    emu.close_immediately()
