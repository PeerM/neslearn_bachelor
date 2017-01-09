# coding: utf-8
import pandas
from hsa.ba.analyse_ram import ram_from_movie
# from hsa.ba.rewards import time_left, make_delta_potential
import hsa.ba.rewards as rewards
import matplotlib.pyplot as plt

movies = {}

# movies[5] = pandas.DataFrame(ram_from_movie("../../movies/5_1-1_without-shortcut.fm2"))
movies[6] = pandas.DataFrame(ram_from_movie("../../movies/6_1-1_with_shortcut.fm2"))
# movies[7] = pandas.DataFrame(ram_from_movie("../../movies/7_1-1_death_and_checkpoint_no_shortcut.fm2"))
# movies[8] = pandas.DataFrame(ram_from_movie("../../movies/8_1-1_death_and_shortcut.fm2"))
# movies[9] = pandas.DataFrame(ram_from_movie("../../movies/9_1-1_death_in_shortcut.fm2"))
delta_potential = rewards.make_delta_potential()


def reward_func(ram):
    return {"delta_potential": delta_potential(ram),
            "time_left": rewards.reward_for_time_left(ram) / 10,
            "current_screen": ram[0x071A],
            "playerstate": ram[0x000E]
            }


# noinspection PyTypeChecker
def reward_series_from_frame(df):
    return pandas.DataFrame.from_dict((reward_func(ram) for index, ram in df.iterrows()))


reward_series_from_frame(movies[6]).plot()
plt.show()
