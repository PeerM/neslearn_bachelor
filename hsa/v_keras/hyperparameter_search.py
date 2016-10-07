import datetime

import numpy as np
import pandas

from hsa.v_keras.memories import load_memories
from hsa.v_keras.model_zoo import first_unstable
from hsa.v_keras.qlearning4k import ExperienceReplay

nb_frames = 1
ram_size = 2048
nr_actions = 36
batch_size = 256
nr_epoch = 30
memories_filename = "../mario_1_1_third.hdf"

nr_attempts = 10
nb_epochs = 100

# including high
learning_rate_low = -6
learning_rate_high = -5
decay_low = 0
decay_high = 0.5
decay_steps = 0.01

memory = ExperienceReplay(memory_size=50000)
load_memories(memory, memories_filename)

example_model = first_unstable(nb_frames=nb_frames, ram_size=ram_size, nr_actions=nr_actions)
test_data = memory.get_batch(model=example_model, batch_size=batch_size, gamma=0.9)


def test_hyperparameters(learning_rate, learning_rate_decay):
    model = first_unstable(learning_rate, learning_rate_decay, nb_frames, ram_size, nr_actions)
    for i in range(nr_epoch):
        training_data = memory.get_batch(model=example_model, batch_size=batch_size, gamma=0.9)
        inputs, targets = training_data
        loss = model.train_on_batch(inputs, targets)
        # print("loss: {}".format(loss))
    score = model.evaluate(test_data[0], test_data[1])
    print("\n{:1e},{:.2f},{:7.3f}\n".format(learning_rate, learning_rate_decay, score))
    return score


learning_rates = (10 ** random_int for random_int in np.random.randint(learning_rate_low, learning_rate_high + 1, nr_attempts))
decays = (random_int * decay_steps for random_int in np.random.randint(decay_low * 1 / decay_steps, (decay_high * 1 / decay_steps) + 1, nr_attempts))
# learning_rates = [10 ** i for i in range(3, 11)]
# decays = [i * 0.1 for i in range(0, 11)]
variable_space = list(zip(learning_rates, decays))
print("variable_space {};".format(variable_space))

score_df = pandas.DataFrame(
    data=(test_hyperparameters(rate, decay) for (rate, decay) in variable_space),
    index=pandas.MultiIndex.from_tuples(variable_space))
print(score_df)
score_df.to_csv("scores{}.csv".format(datetime.datetime.now().timestamp()))
