from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras.optimizers import sgd
from keras.regularizers import l1


def make_2_hidden_wide_beginning_stable(nb_frames, ram_size, nr_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(nb_frames, ram_size)))
    model.add(Dense(1024, init="glorot_uniform", activation='relu', W_regularizer=l1()))
    model.add(Dense(128, init="glorot_uniform", activation='relu'))
    model.add(Dense(nr_actions))
    model.compile(sgd(lr=10 ** -6, momentum=0.9), "mse")
    return model


def make_3_hidden_stable(nb_frames=1, ram_size=2048, nr_actions=36, learning_rate=1e-6, decay=0.0, ):
    model = Sequential()
    model.add(Flatten(input_shape=(nb_frames, ram_size)))
    model.add(Dense(2 ** 9 + 2 ** 8, init="glorot_uniform", activation='relu', W_regularizer=l1()))
    model.add(Dense(512, init="glorot_uniform", activation='relu'))
    model.add(Dense(256, init="glorot_uniform", activation='relu'))
    model.add(Dense(nr_actions))
    model.compile(sgd(learning_rate, momentum=0.9, decay=decay), "mse")
    return model


def first_unstable(learning_rate=1e-6, decay=0.0, nb_frames=1, ram_size=2048, nr_actions=36):
    model = Sequential()
    model.add(Flatten(input_shape=(nb_frames, ram_size)))
    model.add(Dense(2 ** 9 + 2 ** 8, init="glorot_uniform", activation='relu', W_regularizer=l1()))
    model.add(Dense(512, init="glorot_uniform", activation='relu'))
    model.add(Dense(256, init="glorot_uniform", activation='relu'))
    model.add(Dense(nr_actions))
    model.compile(sgd(learning_rate, momentum=0.9, decay=decay), "mse")
    return model
