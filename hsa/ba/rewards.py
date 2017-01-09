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


def potential(ram, state):
    # only applicable to level 1-1
    if not (ram[0x0760] == 0 and ram[0x075F] == 0):
        return 0, state, True

    screen = ram[0x071A]
    player_state = ram[0x000E]
    if player_state == PlayerState.going_down_a_pipe:
        state = "going down"
    if state == "going down" and screen == 0:
        # if state == "going down" and player_state == PlayerState.Leftmost_of_screen:
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
        potential_offset = 9

    current_potential = screen + potential_offset

    return current_potential, state, False


def make_delta(func, initial_value, initial_residual):
    last_value = initial_value
    last_residual = initial_residual

    def state_func(ram):
        nonlocal last_value
        nonlocal last_residual
        current_value, current_residual, reset = func(ram, last_residual)
        if reset:
            last_value = 0
            return 0
        delta = current_value - last_value
        last_value = current_value
        last_residual = current_residual
        return delta

    return state_func


def make_delta_potential():
    return make_delta(potential, 0, "initial")


def scale_by(func, factor):
    return lambda ram: factor * func(ram)


def time_left(ram):
    return int("".join([str(ram[0x07F8 + i]) for i in range(3)]))


def reward_for_time_left(ram):
    # 0713 - Used during flag contact
    if ram[0x0713] != 0:
        return time_left(ram)
    else:
        return 0


scaled_for_time_left = scale_by(reward_for_time_left, 0.1)


def sum_of_rewards(rewards):
    return lambda ram: sum((reward(ram) for reward in rewards))


def make_main_reward():
    delta_potential = make_delta_potential()
    return sum_of_rewards([delta_potential, scaled_for_time_left])
