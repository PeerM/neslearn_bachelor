import socket
import uuid

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizers import sgd

from hsa import emu_connect
from hsa.keras.mario_game import MarioEmuGame
from hsa.keras.qlearning4k import Agent

nb_frames = 1
ram_size = 2048
nr_actions = 36

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, ram_size)))
# model.add(Input((ram_size,)))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nr_actions))
model.compile(sgd(lr=.2), "mse")

prim_soc = socket.create_connection(("localhost", 9090))
prim_soc.setblocking(True)
emu = emu_connect.Emu2(prim_soc)
emu.speed_mode("turbo")
emu.load_slot(10)

game = MarioEmuGame(emu, nr_actions)
agent = Agent(model, nb_frames=nb_frames)
try:
    agent.train(game, nb_epoch=10)
    # agent.play(game, nb_epoch=2)
finally:
    print("stopping")
    model.save_weights(str(uuid.uuid1()) + ".hdf5")
    emu.close()
