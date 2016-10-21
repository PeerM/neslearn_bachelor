A = 1
B = 2
Select = 4
Start = 8
Up = 16
Down = 32
Left = 64
Right = 128

key_names = ["A", "B", "select", "start", "up", "down", "left", "right"]
key_dict = dict(zip(key_names, (1 << i for i in range(8))))


def py_to_nes_wrapper(py_input):
    return \
        sum(
            map(
                lambda item: key_dict[item[0]],
                filter(
                    lambda item: item[1]
                    , py_input.items())))
