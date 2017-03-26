import math
from collections import namedtuple

from hsa.tree_search import heuristics
from hsa.tree_search.heap import Heap


# PlanNode = namedtuple("PlanNode", ["parent", "action", "action_repeat"])
# ExpandedNode = namedtuple("ExpandedNode", ["plan", "state", "potential", "recv_reward"])


class MarioNode(namedtuple("MarioNode", ["potential", "recv_reward", "state", "parent", "action", "action_repeat"])):
    __slots__ = ()

    def __str__(self):
        return "MarioNode(potential:{s.potential}, action:{s.action})".format(s=self)

    def __repr__(self):
        return "MarioNode(potential:{s.potential}, action:{s.action})".format(s=self)

    def __eq__(self, other):
        return self.state == other.state
    #
    # def __gt__(self, other):
    #     return self.potential.__gt__(other.potential)
    #
    # def __lt__(self, other):
    #     return self.potential.__lt__(other.potential)
    #
    # def __ge__(self, other):
    #     return self.potential.__ge__(other.potential)
    #
    # def __le__(self, other):
    #     return self.potential.__le__(other.potential)


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

    def __str__(self):
        return str(self.__dict__)


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
            new_potential = self.config.heuristic(self.config.nes.getRAM())
            new_state = self.config.nes.cloneState()
            n.append(MarioNode(potential=new_potential,
                               recv_reward=reward_for_action, state=new_state,
                               parent=pos, action=action, action_repeat=self.config.action_repeat))
        return n


def make_start(nes, heuristic, reset=True):
    if reset:
        nes.reset_game()
        for i in range(160):
            nes.act(0)
    reward = nes.act(0)
    state = nes.cloneState()
    potential = heuristic(nes.getRAM())
    return MarioNode(potential=potential, recv_reward=reward, state=state, parent=None, action=None, action_repeat=None)


def render_node(node, nes):
    nes.restoreState(node.state)
    nes.render()


def walk_nodes(node: MarioNode):
    # would be a nice place to do recursion with yield from
    current_pos = node
    while current_pos.parent is not None:
        yield (current_pos.action, current_pos.action_repeat)
        current_pos = current_pos.parent


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
