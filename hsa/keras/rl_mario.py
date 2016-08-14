import socket

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
model.add(Dense(2, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(nr_actions))
model.compile(sgd(lr=.2), "mse")

prim_soc = socket.create_connection(("localhost", 9090))
prim_soc.setblocking(True)
emu = emu_connect.Emu2(prim_soc)
emu.speed_mode("turbo")
emu.load_slot(10)

game = MarioEmuGame(emu, nr_actions)
agent = Agent(model, memory_size=100000, nb_frames=nb_frames)
agent.train(game)
agent.play(game)
