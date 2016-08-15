import uuid

import pandas
import numpy as np
from keras.layers import Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import sgd
from keras.regularizers import l1
import matplotlib.pyplot as plt

from hsa.dqn_mario.dqn_input import numpy_to_rdqn, rdqn_to_py
from hsa.keras.qlearning4k import ExperienceReplay
from hsa.reward_evaluation import mario_x_speed

inputs = pandas.read_hdf("../mario_1_1_first.hdf", key="inputs")
rams = pandas.read_hdf("../mario_1_1_first.hdf", key="rams")
dqn_inputs = np.array([numpy_to_rdqn(row) for row in inputs.values])

nr_frames = rams.shape[0]
nr_actions = 36
# This does not work in this file right now
nb_frames = 1
ram_size = 2048
nr_trainings = 3000
batch_size = 64

memory = ExperienceReplay(nr_frames, fast=False)

# Model
model = Sequential()
model.add(Flatten(input_shape=(nb_frames, ram_size)))
# model.add(Input((ram_size,)))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(256, init="glorot_uniform", activation='relu', W_regularizer=l1(0.001)))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu', W_regularizer=l1()))
# model.add(Dense(64, activation='relu', W_regularizer=l1()))
# model.add(BatchNormalization())
model.add(Dense(nr_actions))
# 7 seems ok 512
# 6 127
model.compile(sgd(lr=10 ** -6, momentum=0.9), "mse")

for i, s_prime in enumerate(rams.values[1:]):
    s_bloated = np.expand_dims(np.expand_dims(rams.values[i - 1], 0), 0)
    s_prime_bloated = np.expand_dims(np.expand_dims(s_prime, 0), 0)
    memory.remember(s_bloated, dqn_inputs[i], mario_x_speed(s_prime), s_prime_bloated, False)

batch = memory.get_batch(model=model, batch_size=batch_size)
training_loss = list()
for i in range(nr_trainings):
    if batch:
        inputs, targets = batch
        loss = model.train_on_batch(inputs, targets)
        training_loss.append(loss)
        print(loss)

pandas.DataFrame(training_loss).plot()

# inputs, targets = batch
# history = model.fit(inputs,targets,batch_size=batch_size,nb_epoch=nr_trainings)
# history_df = pandas.DataFrame.from_dict(history.history)
# history_df["loss"].plot()
plt.show()

model_id = "P" + str(uuid.uuid1())

print(model_id)
model_json = model.to_json()
with open(model_id + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_id + ".hdf5")
