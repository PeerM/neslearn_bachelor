def x_pos(ram):
    coarse_x = ram[0x006D]
    current_fine = ram[0x0086]
    return coarse_x * 256 + current_fine

#screewed up in the live displaying screen, is it still?

def shortcut_bonus(ram):
    coarse_x = ram[0x006D]
    whatever = ram[0x071B] #Next screen (in level) 
    level_layout = ram[0x072C]
    if whatever == 0 and coarse_x == 0 and not level_layout == 10:
        return 9*256
    if whatever == 10 and coarse_x == 0:
        return 10*256
    return 0    


def combined(ram):
    reward = x_pos(ram) + shortcut_bonus(ram)
    return reward


#punishment for dying
def dont_die(ram):
    player_state = ram[0x000E]
    if player_state == 0x0B:
        return -1
    return combined(ram)

#TODO consider bonus for height
#TODO consider bonus for speed, mario_x_speed
#TODO consider punishment for beeing close to enemies
#TODO punishment for being below the screen

#    # sanity check, this is a strong inidcation that somethin is wrong
#    if reward > 1500:
#        return 0