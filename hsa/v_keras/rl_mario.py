import socket
import uuid

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
nr_epoch = 60

model_filename = None
# model_filename = "./M1"
# model_filename = "../dqn_weights/keras/P3_1Layer"
model_filename = "../dqn_weights/keras/tmp/5_layer_firsttime"
memories_filename = "../mario_1_1_third.hdf"

# Model hyper parameters
# TODO IDEA model factory methods
model = zoo.make_5layer_unstable(nb_frames, ram_size, nr_actions)

memory = ExperienceReplay(memory_size=100000)
if model_filename:
    model.load_weights(model_filename + ".hdf5")
    load_memories(memory, memories_filename)
    epsilon = (0.1, 0.01)
    epsilon_rate = 0.4
else:
    epsilon = (0.1, 0)
    epsilon_rate = 0.6
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
try:
    agent.train(game, nb_epoch=nr_epoch, epsilon=epsilon, play_period=play_period, batch_size=batch_size,
                action_repeat=4, epsilon_rate=epsilon_rate)
    # agent.play(game, nb_epoch=4)
finally:
    print("stopping")
    model.save_weights("../dqn_weights/keras/tmp/" + str(uuid.uuid1()) + ".hdf5")
    emu.close_immediately()
