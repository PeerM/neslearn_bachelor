import math
from collections import namedtuple

from hsa.tree_search import heuristics
from hsa.tree_search.heap import Heap

MarioNode = namedtuple("MarioNode", ["id", "potential", "recv_reward", "state"])


class ConfigPack(object):
    def __init__(self, nes, render_search=False, heuristic=heuristics.combined, action_repeat=40, nr_nodes_to_expand=100):
        """
        """
        super().__init__()
        self.nr_nodes_to_expand = nr_nodes_to_expand
        self.actions = nes.getMinimalActionSet()
        self.render_search = render_search
        self.heuristic = heuristic
        self.action_repeat = action_repeat
        self.nes = nes


class AStarGraph(object):
    # Define a class board like grid with two barriers

    def __init__(self, config):
        """

        :type config: ConfigPack
        """
        self.config = config

    def get_vertex_neighbours(self, pos):
        n = []
        for action in self.config.actions:
            self.config.nes.restoreState(pos.state)
            reward_for_action = sum([self.config.nes.act(action) for i in range(self.config.action_repeat)])
            if self.config.render_search:
                self.config.nes.render()
            new_id = hash((pos.id, action))
            new_potential = self.config.heuristic(self.config.nes.getRAM())
            new_state = self.config.nes.cloneState()
            n.append(MarioNode(id=new_id, potential=new_potential,
                               recv_reward=reward_for_action, state=new_state))
        return n


def make_start(config, reset=False):
    if reset:
        config.nes.reset_game()
        for i in range(160):
            config.nes.act(0)
    reward = config.nes.act(0)
    state = config.nes.cloneState()
    potential = config.heuristic(config.nes.getRAM())
    return MarioNode(id="astart", potential=potential, recv_reward=reward, state=state)


def render_node(node, nes):
    nes.restoreState(node.state)
    nes.render()


def best_first(start, graph, config):
    seen_nodes = []
    candidates = Heap([start], lambda node: -node.potential)
    best_seen = start
    for i in range(config.nr_nodes_to_expand):
        best_candidate = candidates.pop()
        for child in graph.get_vertex_neighbours(best_candidate):
            candidates.push(child)
            seen_nodes.append(child)
            if child.potential > best_seen.potential:
                best_seen = child
        yield best_seen


def best_first_full(start, graph, config):
    item = None
    for item in best_first(start, graph, config):
        pass
    return item
