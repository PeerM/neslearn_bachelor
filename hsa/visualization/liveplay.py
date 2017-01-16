from hsa.nes_python_input import py_to_nes_wrapper
from hsa.reward_evaluation import mario_x_speed
import hsa.ba.rewards as rewards
from hsa.visualization.liveplot2 import Plotter
from hsa.visualization.parse_fm2 import parse_fm2
from nes_python_interface.nes_python_interface import NESInterface, RewardTypes
import hsa.machine_constants

# movie = "../bin_deps/fceux/movies/happylee-supermariobros,warped.fm2"
# movie = "../movies/12_1-1_dieing_to_first_gumba.fm2"
movie = "../movies/13_1-1_dieing_to_gumbas.fm2"
discount_factor_gamma = 0.9

plotter = Plotter(interval=300, history_length=60*60)

plotter.start()


def visualize_ram(ram):
    plotter.plot(mario_x_speed(ram))


try:
    # nes.reset_game()
    with open(movie) as movie_file:
        nes = NESInterface(hsa.machine_constants.mario_rom_location,
                           eb_compatible=False,
                           auto_render_period=1,
                           reward_type= RewardTypes.factory,
                           reward_function_factory= rewards.make_delta_potential)
        cumulative_reward = 0
        discounted_cumulative_reward = 0
        for combi in parse_fm2(movie_file):
            reward = nes.act(py_to_nes_wrapper(combi))
            plotter.plot(reward)
            cumulative_reward += reward
            # discounted_cumulative_reward = discounted_cumulative_reward * discount_factor_gamma + reward
            # plotter.plot(reward)
    plotter.stop()
    print("sum(r):", cumulative_reward)
    # print("sum(gama*r):",discounted_cumulative_reward)
except KeyboardInterrupt:
    pass
    # emu.close()
