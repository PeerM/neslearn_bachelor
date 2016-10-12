import uuid

import pandas
import numpy as np

from hsa.v_keras.functional import epsilon_schedule_gen
from hsa.v_keras.qlearning4k import Agent
from hsa.v_keras.qlearning4k import ExperienceReplay
import hsa.v_keras.model_zoo as zoo

# parameters
from hsa.v_keras.qlearning4k.games.game import Game, PlayResult

import gym

# TODO IDEA unify these launch scripts
env = gym.make('Breakout-ram-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

nb_frames = 1
ram_size = input_size
nr_actions = output_size
play_period = 5
batch_size = play_period * 32
nr_epoch = None
save_every_n_epochs = 1000

render_game = False

model_filename = None
# memories_filename = "../mario_1_1_third.hdf"

model_name = str(uuid.uuid1())
# Model hyper parameters
model = zoo.make_3_hidden_stable(nb_frames, ram_size, nr_actions, decay=0.01)

# second*minute*hour*n
memory = ExperienceReplay(memory_size=60 * 60 * 60 * 2)
if model_filename:
    model.load_weights(model_filename + ".hdf5")
    # load_memories(memory, memories_filename)
    epsilon_schedule = epsilon_schedule_gen(0.2, 0.02, ((nr_epoch or 0) * 0.4 or 1500))
else:
    epsilon_schedule = epsilon_schedule_gen(0.9, 0.05, ((nr_epoch or 0) * 0.5 or 1500))


class BreakoutGame(Game):
    def __init__(self):
        super().__init__("breakout", nr_actions)
        self.latest = np.zeros(env.observation_space.shape)

    def play(self, action, nr_repeat_actions) -> PlayResult:
        if render_game:
            env.render()
        observation, reward, done, info = env.step(action)
        return PlayResult(observation, reward, action, done, False)

    def reset(self):
        self.latest = env.reset()

    def get_current_state(self):
        return np.array(self.latest)


print("input: ", input_size)
print("output: ", output_size)

game = BreakoutGame()
game.reset()
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
finally:
    print("stopping")
    model.save_weights("../dqn_weights/keras/tmp/{}_{}.hdf5".format(model_name, "final"))
    # noinspection PyTypeChecker
    pandas.DataFrame.from_dict(epoch_results).to_csv("../dqn_weights/keras/tmp/breakout_{}.csv".format(model_name))
