from v_keras.memories import load_memories
from v_keras.model_zoo import make_2_hidden_wide_beginning_stable, first_unstable

# parameters
from v_keras.qlearning4k import ExperienceReplay

nb_frames = 1
ram_size = 2048
nr_actions = 36
batch_size = 128
nr_epoch = 30
memories_filename = "../mario_1_1_third.hdf"

memory = ExperienceReplay(memory_size=50000)
load_memories(memory, memories_filename)

example_model = first_unstable(nb_frames=nb_frames, ram_size=ram_size, nr_actions=nr_actions)
batch = memory.get_batch(model=example_model, batch_size=batch_size, gamma=0.9)


def test_hyperparameters(learning_rate, learning_rate_decay):
    model = first_unstable(learning_rate, learning_rate_decay, nb_frames, ram_size, nr_actions)

    inputs, targets = batch
    return [float(model.train_on_batch(inputs, targets)) for i in range(10)]


learning_rates = [10 ** i for i in range(3, 11)]
decays = [i * 0.1 for i in range(0, 11)]

print("rates {}; decays {}".format(learning_rates, decays))
for learning_rate in learning_rates:
    for decay in decays:
        print("{:.1e};\t{:.2f}\t;{}".format(learning_rate, decay, test_hyperparameters(learning_rate, decay)))
