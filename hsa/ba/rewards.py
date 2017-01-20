import logging

import math


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


def make_fine_potential(kind="delta", subsections_selector=2 ** 5):
    """kind can be delta, all or normal"""
    state = "initial"
    last_sum = 0
    last_fine = 0

    # this function returns not stellar values before the start of 1-1
    def inner_fine_potential(ram):
        nonlocal state
        nonlocal last_sum
        nonlocal last_fine
        # only applicable to level 1-1
        if not (ram[0x0760] == 0 and ram[0x075F] == 0):
            if kind == "all":
                return {"current_sum": 0}
            return 0

        coarse_x = ram[0x006D]
        player_state = ram[0x000E]
        current_fine = ram[0x0086]
        if player_state == PlayerState.going_down_a_pipe:
            state = "going down"
        if state == "going down" and coarse_x == 0:
            state = "in shortcut"
        if state == "in shortcut" and player_state == PlayerState.entering_reversed_L_pipe:
            state = "going up"
            # TODO fix going up the pipe
        if state == "going up" and player_state == 7:
            state = "initial"
        if state == "in shortcut" and (player_state == PlayerState.dying or player_state == PlayerState.player_dies):
            # player might die in shortcut, for example times up
            state = "initial"

        coarse_potential_offset = 0
        if state == "in shortcut" or state == "going up":
            coarse_potential_offset = 9

        current_coarse_potential = coarse_x + coarse_potential_offset

        # going up is kinda strange, make it that we have a delta of 0
        if state == "going up":
            current_fine = last_fine
        last_fine = current_fine

        adjusted_fine = math.floor(current_fine / subsections_selector) / (256 / subsections_selector)
        # adjusted_fine = round(current_fine/subsections_selector,0) /subsections_selector

        if kind == "delta":
            current_sum = current_coarse_potential + adjusted_fine
            delta_sum = current_sum - last_sum
            last_sum = current_sum
            return delta_sum
        elif kind == "all":
            return {"state": state,
                    "player_state": player_state,
                    "coarse": coarse_x,
                    "current_sum": current_coarse_potential + adjusted_fine,
                    "current_coarse_potential": current_coarse_potential}
        else:
            return current_coarse_potential + adjusted_fine

    return inner_fine_potential


def make_finer_potential():
    return make_fine_potential(subsections_selector=1)


def fine_x_pos(ram, last_coarse_pos):
    # only applicable to level 1-1
    if not (ram[0x0760] == 0 and ram[0x075F] == 0):
        return 0, None, True

    coarse_pos_in_level = ram[0x006D]  # check for lexical ordering
    fine_pos = ram[0x0086]
    player_state = ram[0x000E]

    # fine_pos = lowest value and coarse changed
    if coarse_pos_in_level != last_coarse_pos:
        return fine_pos, coarse_pos_in_level, True
    if player_state == PlayerState.dying or player_state == PlayerState.player_dies:
        return 0, None, True  # TODO None or coarse_pos?
    return fine_pos, coarse_pos_in_level, False


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


# unused for now
def make_delta_without_residual(func):
    last_value = 0

    def state_func(ram):
        nonlocal last_value
        current_value, reset = func(ram)
        if reset:
            last_value = 0
            return 0
        delta = current_value - last_value
        last_value = current_value
        return delta

    return state_func


def make_delta_potential():
    return make_delta(potential, 0, "initial")


def make_delta_fine_x_pos():
    return make_delta(fine_x_pos, 1, None)


def scale_by(func, factor):
    return lambda ram: factor * func(ram)


def time_left(ram):
    return int("".join([str(ram[0x07F8 + i]) for i in range(3)]))


def with_residual(func, inital_residual):
    last_residual = inital_residual

    def inner_func(ram):
        nonlocal last_residual
        reward, current_residual, reset = func(ram, last_residual)
        last_residual = current_residual
        return reward, reset

    return inner_func


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


def make_fine_main_reward():
    delta_potential = make_fine_potential(kind="delta")
    return sum_of_rewards([delta_potential, scaled_for_time_left])


def make_finer_main_reward():
    delta_potential = make_finer_potential()
    return sum_of_rewards([delta_potential, scaled_for_time_left])


def make_scaled_finer_main_reward():
    delta_potential = scale_by(make_finer_potential(), 10)
    return sum_of_rewards([delta_potential, scaled_for_time_left])


def make_finer_main_reward_with_points():
    delta_potential = make_finer_potential()
    # 50 points should be 0.5
    delta_points = scale_by(make_delta_points(), 0.1)
    return sum_of_rewards([delta_potential, scaled_for_time_left, delta_points])


def make_scaled_finer_main_reward_with_points():
    return scale_by(make_finer_main_reward_with_points(), 10)
