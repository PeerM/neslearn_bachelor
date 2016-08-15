import socket
import uuid

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras.optimizers import sgd
from keras.regularizers import l1

from hsa import emu_connect
from hsa.keras.mario_game import MarioEmuGame
from hsa.keras.qlearning4k import Agent

nb_frames = 1
ram_size = 2048
nr_actions = 36
play_period = 30
batch_size = play_period * 32
nr_epoch=15

# model_filename = None
# model_filename = "./1_15_1"
model_filename = "../dqn_weights/keras/P3_1Layer"

# Model
model = Sequential()
model.add(Flatten(input_shape=(nb_frames, ram_size)))
model.add(Dense(512, init="glorot_uniform", activation='relu'))
model.add(Dense(nr_actions))
model.compile(sgd(lr=10 ** -6, momentum=0.9), "mse")

if model_filename:
    model.load_weights(model_filename + ".hdf5")
    epsilon = (0.5, 0.05)
else:
    epsilon = (1.0, 0.1)
prim_soc = socket.create_connection(("localhost", 9090))
prim_soc.setblocking(True)
emu = emu_connect.Emu2(prim_soc)
emu.speed_mode("turbo")
emu.load_slot(10)

game = MarioEmuGame(emu, nr_actions)
agent = Agent(model, nb_frames=nb_frames)
try:
    agent.train(game, nb_epoch=nr_epoch, epsilon=epsilon, play_period=play_period, batch_size=batch_size)
    # agent.play(game, nb_epoch=4)
finally:
    print("stopping")
    model.save_weights(str(uuid.uuid1()) + ".hdf5")
    emu.close()
