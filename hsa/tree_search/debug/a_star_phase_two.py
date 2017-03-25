from hsa.tree_search import fceux_a_star

from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface
from hsa.machine_constants import mario_rom_location

nes = NESInterface(mario_rom_location, eb_compatible=False)

config = fceux_a_star.ConfigPack(nes, render_search=False, nr_nodes_to_expand=50)

start = fceux_a_star.make_start(config, reset=True)

graph = fceux_a_star.AStarGraph(config)

best = None
for i, current_best in enumerate(fceux_a_star.best_first(start, graph, config)):
    best = current_best
    # nes.render()
    # ipywidgets.IntProgress(value=i, max=100)
    fceux_a_star.render_node(current_best,nes)
    print(i)

# ## Debug analysis

# In[7]:

graph.get_vertex_neighbours(start)

# In[12]:

import ipywidgets


# In[ ]:
