import socket
import uuid

from hsa import emu_connect
from hsa.v_keras.mario_game import MarioEmuGame
from hsa.v_keras.memories import load_memories
from hsa.v_keras.qlearning4k import Agent
from hsa.v_keras.qlearning4k import ExperienceReplay
from hsa.v_keras.model_zoo import make_2_hidden_wide_beginning_stable

# parameters
nb_frames = 1
ram_size = 2048
nr_actions = 36
play_period = 20
batch_size = play_period * 32
nr_epoch = 30

model_filename = None
# model_filename = "./M1"
# model_filename = "../dqn_weights/keras/P3_1Layer"
memories_filename = "../mario_1_1_third.hdf"

# Model hyper parameters
# TODO IDEA model factory methods
model = make_2_hidden_wide_beginning_stable(nb_frames, ram_size, nr_actions)

memory = ExperienceReplay(memory_size=50000)
if model_filename:
    model.load_weights(model_filename + ".hdf5")
    epsilon = (0.5, 0.05)
    epsilon_rate = 0.5
else:
    epsilon = (0.6, 0.1)
    epsilon_rate = 0.3
    load_memories(memory, memories_filename)

prim_soc = socket.create_connection(("localhost", 9090))
prim_soc.setblocking(True)
emu = emu_connect.Emu2(prim_soc)
emu.speed_mode("turbo")
emu.load_slot(10)

game = MarioEmuGame(emu, nr_actions)
agent = Agent(model, memory=memory, nb_frames=nb_frames)
try:
    agent.train(game, nb_epoch=nr_epoch, epsilon=epsilon, play_period=play_period, batch_size=batch_size,
                action_repeat=4, epsilon_rate=epsilon_rate)
    # agent.play(game, nb_epoch=4)
finally:
    print("stopping")
    model.save_weights("../dqn_weights/keras/tmp/" + str(uuid.uuid1()) + ".hdf5")
    emu.close_immediately()
