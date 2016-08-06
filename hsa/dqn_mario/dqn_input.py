import itertools

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

# Reduced input set

_rdqn_directions = [
    (False, False, False, False),
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (False, False, False, True),
    (True, True, False, False),
    (True, False, False, True),
    (False, True, True, False),
    (False, False, True, True)]
_rdqn_ab = [(False, False), (True, False), (False, True), (True, True)]
_start = 36
_select = 37
_rdqn_trunk = list(itertools.product(_rdqn_ab, _rdqn_directions))
_rdqn_trunk_lookup = {v: i for i, v in enumerate(_rdqn_trunk)}


def rdqn_to_py(rdqn_input):
    if rdqn_input > 37:
        raise ValueError("rdqn input {} larger than 37".format(rdqn_input))
    a, b, select, start, up, right, down, left = 8 * [False]
    if rdqn_input == _start:
        start = True
    elif rdqn_input == _select:
        select = True
    else:
        (a, b), (up, right, down, left) = _rdqn_trunk[rdqn_input]
    return {"A": a, "B": b, "down": down, "left": left, "right": right, "select": select, "start": start, "up": up}


def py_to_rdqn(py_input: dict):
    if py_input["start"]:
        return _start
    elif py_input["select"]:
        return _select
    else:
        return _rdqn_trunk_lookup[
            ((py_input["A"], py_input["B"]),
             (py_input["up"], py_input["right"], py_input["down"], py_input["left"]))]


for i in range(37):
    assert py_to_rdqn(rdqn_to_py(i)) == i
