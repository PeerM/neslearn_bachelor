import uuid

import pandas
import numpy as np
from keras.layers import Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import sgd
from keras.regularizers import l1
import matplotlib.pyplot as plt

from hsa.keras.memories import load_memories
from hsa.keras.qlearning4k import ExperienceReplay

nr_actions = 36
# This does not work in this file right now
nb_frames = 1
ram_size = 2048
nr_trainings = 200
batch_size = 64
memory = ExperienceReplay(4000, fast=False)
load_memories(memory)

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, ram_size)))
model.add(Dense(512, init="glorot_uniform", activation='relu'))
model.add(Dense(nr_actions))
model.compile(sgd(lr=10 ** -6, momentum=0.9), "mse")

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
