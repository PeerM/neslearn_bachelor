import logging


def make_delta_points():
    last = 0

    def delta_points(ram: bytes):
        nonlocal last
        current = int("".join([str(ram[0x07DE + i]) for i in range(6)]))
        delta = current - last
        last = current
        return delta / 10

    return delta_points


def unsigned_to_singed(byte):
    if byte > 127:
        return (256 - byte) * (-1)
    else:
        return byte


def mario_x_speed(ram: bytes):
    raw = unsigned_to_singed(ram[0x0057])
    return round(raw / 50, 1)


# TODO: fix. currently incorrect
def make_delta_potential():
    logger = logging.getLogger(__name__)
    last_potential = 0

    def delta_potential(ram):
        # only applicable to level 1-1
        if not (ram[0x0760] == 0 and ram[0x075F] == 0):
            return 0
        current_screen = ram[0x071A]
        next_screen = ram[0x071B]
        if current_screen == 0:
            if next_screen == 4:
                current_potential = 9
            elif next_screen == 0:
                current_potential = 0
            else:
                logger.warn("unknown current/ next screen combination")
                current_potential = 0
        else:
            current_potential = current_screen
        nonlocal last_potential
        delta = current_potential - last_potential
        last_potential = current_potential
        return delta

    return delta_potential


def time_left(ram):
    return int("".join([str(ram[0x07F8 + i]) for i in range(3)]))

def reward_for_time_left(ram):
    # 0713 - Used during flag contact
    if ram[0x0713] != 0:
        return time_left(ram)