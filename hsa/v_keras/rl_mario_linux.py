import socket
import uuid

import itertools
import pandas

from hsa import emu_connect
from hsa.v_keras.functional import epsilon_schedule_gen
from hsa.v_keras.mario_game import MarioEmuGame
from hsa.v_keras.mario_game_direct import FceuxMarioEmuGame
from hsa.v_keras.memories import load_memories
from hsa.v_keras.qlearning4k import Agent
from hsa.v_keras.qlearning4k import ExperienceReplay
import hsa.v_keras.model_zoo as zoo
from nes_python_interface import NESInterface


def save_epoch_result_csv(model_name, epoch):
    # noinspection PyTypeChecker
    pandas.DataFrame.from_dict(epoch_results).to_csv("../dqn_weights/keras/tmp/{}_{}.csv".format(model_name, epoch))


# parameters
nb_frames = 1
ram_size = 2048
nr_actions = 36
play_period = 20
batch_size = play_period * 32
nr_epoch = None
save_every_n_epochs = 50

model_filename = None
# model_filename = "./M1"
# model_filename = "../dqn_weights/keras/P3_1Layer"
# model_filename = "../dqn_weights/keras/tmp/5_layer_firsttime"
memories_filename = "../mario_1_1_third.hdf"
mario_rom_path = "/home/peer/playground/mario/Super Mario Bros. (JU) [!].nes"

model_name = str(uuid.uuid1())
# Model hyper parameters
model = zoo.make_8layer_unstable(nb_frames, ram_size, nr_actions)

# second*minute*hour*n
memory = ExperienceReplay(memory_size=60 * 60 * 60 * 5)
if model_filename:
    model.load_weights(model_filename + ".hdf5")
    load_memories(memory, memories_filename)
    epsilon_schedule = itertools.repeat(0.05)
else:
    # if continuous mode slope until epoch 800
    epsilon_schedule = epsilon_schedule_gen(0.5, 0.05, ((nr_epoch or 0) * 0.5 or 800))
    load_memories(memory, memories_filename)

nes = NESInterface(mario_rom_path)
game = FceuxMarioEmuGame(nes, nb_actions=nr_actions)
game.reset()
print("fceux started")
agent = Agent(model, memory=memory, nb_frames=nb_frames)
print("starting")
epoch_results = list()
try:
    training_generator = agent.train(game, nb_epoch=nr_epoch, epsilon_schedule=epsilon_schedule, play_period=play_period, batch_size=batch_size,
                                     action_repeat=4)
    for epoch, epoch_result in enumerate(training_generator):
        print(epoch_result)
        epoch_results.append(epoch_result)
        if epoch % save_every_n_epochs == 0:
            model.save_weights("../dqn_weights/keras/tmp/{}_{}.hdf5".format(model_name, epoch))
            save_epoch_result_csv(model_name, epoch)
            # agent.play(game, nb_epoch=4)
finally:
    print("stopping")
    model.save_weights("../dqn_weights/keras/tmp/{}_{}.hdf5".format(model_name, "final"))
    save_epoch_result_csv(model_name, "final")
    del nes
