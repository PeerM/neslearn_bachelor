input_transform_keys = ["A", "B", "down", "left", "right", "select", "start", "up"]


# input_transform_keys.reverse()


def dqn_to_py(dqn_input):
    booleans = [bool(1 << i & dqn_input) for i in range(8)]
    return dict(zip(input_transform_keys, booleans))


def py_to_dqn(py_input):
    return sum([int(py_input[key]) * (1 << i) for i, key in enumerate(input_transform_keys)])


def numpy_to_dqn(row):
    return sum([int(is_button_pressed) * (1 << i) for i, is_button_pressed in enumerate(row)])


def pandas_to_dqn(row):
    return sum([int(is_button_pressed) * (1 << i) for i, is_button_pressed in enumerate(row[1])])


for i in range(256):
    assert py_to_dqn(dqn_to_py(i)) == i
