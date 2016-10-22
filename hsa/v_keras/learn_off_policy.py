import socket
import uuid

import pandas
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras.optimizers import sgd
from keras.regularizers import l1

from hsa.v_keras.functional import epsilon_schedule_gen
from hsa.v_keras.qlearning4k import ExperienceReplay
from hsa import emu_connect
from hsa.v_keras.mario_game import MarioEmuGame, MarioReplay
from hsa.v_keras.memories import load_memories
from hsa.v_keras.qlearning4k import Agent

# parameters

nb_frames = 1
ram_size = 2048
nr_actions = 36
play_period = 20
batch_size = 2
nr_epoch = 30

model_filename = None
# model_filename = "./M1"
# model_filename = "../dqn_weights/keras/P3_1Layer"
memories_filename = "../mario_1_1_third.hdf"

# Model hyper parameters
# TODO IDEA model factory methods
model = Sequential()
model.add(Flatten(input_shape=(nb_frames, ram_size)))
model.add(Dense(1024, init="glorot_uniform", activation='relu', W_regularizer=l1()))
model.add(Dense(128, init="glorot_uniform", activation='relu'))
model.add(Dense(nr_actions))
model.compile(sgd(lr=10 ** -6, momentum=0.9), "mse")

memory = ExperienceReplay(memory_size=50000,fast=False)
epsilon_schedule = epsilon_schedule_gen(0.5, 0.05, ((nr_epoch or 0) * 0.5 or 800))

inputs = pandas.read_hdf(memories_filename, key="inputs")
rams = pandas.read_hdf(memories_filename, key="rams")

game = MarioReplay(inputs, rams, nr_actions)
agent = Agent(model, memory=memory, nb_frames=nb_frames)
try:
    training_iter = agent.train(game, nb_epoch=nr_epoch, epsilon_schedule=epsilon_schedule, play_period=play_period, batch_size=batch_size, action_repeat=4)
    for epoch_result in training_iter:
        print(epoch_result)
    # agent.play(game, nb_epoch=4)
finally:
    print("stopping")
    #model.save_weights("../dqn_weights/keras/tmp/" + str(uuid.uuid1()) + ".hdf5")
