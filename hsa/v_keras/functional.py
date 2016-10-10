from keras.models import Sequential

def copy_model(model: Sequential):
    config = model.get_config()
    the_copy = Sequential.from_config(config)
    the_copy.set_weights(model.get_weights())
    return the_copy