import itertools
from keras.models import Sequential
from scipy.interpolate import interp1d


def copy_model(model: Sequential):
    config = model.get_config()
    the_copy = Sequential.from_config(config)
    the_copy.set_weights(model.get_weights())
    return the_copy

# epsilon_schedule(1,0.1,15) has 0.1 at 16 and 0.999 at 15
def epsilon_schedule_gen(max_value, min_value, min_at):
    interpolating_func = interp1d((0, min_at), (max_value, min_value), bounds_error=False, fill_value=min_value)
    for i in itertools.count():
        yield float(interpolating_func(i))
