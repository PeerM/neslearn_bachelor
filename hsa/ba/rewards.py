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


class PlayerState:
    Leftmost_of_screen = 0x00
    entering_reversed_L_pipe = 0x02
    going_down_a_pipe = 0x03
    player_dies = 0x06
    dying = 0x0B


def make_delta_potential():
    # logger = logging.getLogger(__name__)
    last_potential = 0
    state = "initial"

    def delta_potential(ram):
        # only applicable to level 1-1
        if not (ram[0x0760] == 0 and ram[0x075F] == 0):
            return 0
        nonlocal state
        nonlocal last_potential

        screen = ram[0x071A]
        player_state = ram[0x000E]
        if player_state == PlayerState.going_down_a_pipe:
            state = "going down"
        # if state == "going down" and screen == 0:
        if state == "going down" and player_state == PlayerState.Leftmost_of_screen:
            state = "in shortcut"
        if state == "in shortcut" and player_state == PlayerState.entering_reversed_L_pipe:
            state = "going up"
        # if state == "going up" and screen == 0:
        if state == "going up" and player_state == PlayerState.Leftmost_of_screen:
            state = "initial"
        if state == "in shortcut" and (player_state == PlayerState.dying or player_state == PlayerState.player_dies):
            # player might die in shortcut, for example times up
            state = "initial"

        potential_offset = 0
        if state == "in shortcut" or state == "going up":
            potential_offset = 6

        current_potential = screen + potential_offset

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
