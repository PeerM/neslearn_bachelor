

import matplotlib.pyplot as plt
import numpy as np
from hsa.tree_search import fceux_a_star
from tqdm import tqdm
from functools import partial


# In[4]:

from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface
from hsa.machine_constants import mario_rom_location
from hsa.tree_search import heuristics

nes = NESInterface(mario_rom_location,eb_compatible=False)


# In[8]:

def w_to_config(w):
    action_repeat = int(w[0])
    #ram, x_pos_resolution=1, x_pos_factor=1, shortcut_factor=1, height_factor=0, punishment_death=True,speed_factor=0
    heuristic_params = {"x_pos_resolution": w[1],
                        "x_pos_factor": w[2],
                        "shortcut_factor": w[2],
                        "height_factor": w[3],
                        "speed_factor": w[4]}
    heuristic = partial(heuristics.combined, **heuristic_params)
    return fceux_a_star.ConfigPack(nes, heuristic=heuristic, action_repeat=action_repeat, nr_nodes_to_expand=35)

default_w = np.array([6, 5, 1, 0, 0])


def w_to_r(w):
    config = w_to_config(w)
    start = fceux_a_star.make_start(nes, config.heuristic, True)
    graph = fceux_a_star.AStarGraph(config)
    best = None
    for best in fceux_a_star.best_first(start, graph, config):
        pass
    #print(best.potential)
    nes.restoreState(best.state)
    ram  = nes.getRAM()
    nes.render()
    progress = heuristics.combined(ram)
    return progress


np.random.seed(3)
nn = 1  # number of steps to take (and plot horizontally)
alpha = 0.1  # learning rate
sigma = 3  # standard deviation of the samples around current parameter vector

w = np.array(default_w)  # start point

for q in tqdm(range(nn)):
    # draw a population of samples in black
    noise = np.random.randn(40, w.shape[0])  # second value should be w.shape[0]
    wp = np.expand_dims(w, 0) + sigma * noise

    # draw estimated gradient as white arrow
    R = np.array([w_to_r(wi) for wi in wp])  # do the experiments here
    if np.all(R[0] == R):
        print("useless iteration all of R is {}".format(R[0]))
        continue
    R -= R.mean()
    R /= R.std()  # standardize the rewards to be N(0,1) gaussian
    g = np.dot(R, noise)
    u = alpha * g

    w += u

print(w)