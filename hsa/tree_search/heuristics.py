from hsa.ba.rewards import unsigned_to_singed
import math


def x_pos(ram):
    coarse_x = ram[0x006D]
    current_fine = ram[0x0086]
    return coarse_x * 256 + current_fine


# screewed up in the live displaying screen, is it still?

def shortcut_bonus(ram):
    coarse_x = ram[0x006D]
    whatever = ram[0x071B]  # Next screen (in level)
    level_layout = ram[0x072C]
    if whatever == 0 and coarse_x == 0 and not level_layout == 10:
        return 9 * 256
    if whatever == 10 and coarse_x == 0:
        return 10 * 256
    return 0

    # bonus for speed, mario_x_speed


def x_pos_with_bonus(ram):
    return x_pos(ram) + shortcut_bonus(ram)


def x_speed(ram):
    return unsigned_to_singed(unsigned_to_singed(ram[0x0057]))


def height(ram):
    return ram[0x00CE]


def combined(ram, x_pos_resolution=1, x_pos_factor=1, shortcut_factor=1, height_factor=0, punishment_death=True,
             speed_factor=0):
    """
    sped_factor: sensible values are 0, 1(results in range 0-40) and 0.1
    height_factor: sensible values are 0 and 0.01
    """
    x = math.floor(x_pos(ram) / x_pos_resolution) * x_pos_factor
    bonus = shortcut_bonus(ram) * shortcut_factor
    speed = x_speed(ram) * speed_factor
    height_bonus = height(ram) * height_factor

    reward_sum = x + bonus + speed + height_bonus
    if punishment_death:
        return dont_die(ram, reward_sum)
    else:
        return reward_sum


# punishment for dying
def dont_die(ram, other_value):
    player_state = ram[0x000E]
    if player_state == 0x0B:
        return -1
    return other_value

    # TODO consider bonus for height

    # TODO consider punishment for beeing close to enemies
    # TODO punishment for being below the screen

    #    # sanity check, this is a strong inidcation that somethin is wrong
    #    if reward > 1500:
    #        return 0
