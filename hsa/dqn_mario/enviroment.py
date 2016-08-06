from hsa.emu_connect import Emu2
from hsa.simple_dqn.environment import Environment


def nn_action_to_lua(nn_action):
    return {"A": True}


class NesEnvironment(Environment):
    def __init__(self, emu: Emu2, rewarder):
        super().__init__()
        self.rewarder = rewarder
        self.emu = emu
        self.last_ram = None

    def getScreen(self):
        return self.last_ram

    def isTerminal(self):
        return False

    def numActions(self):
        return 255

    def act(self, action):
        self.last_ram = self.emu.full_step(nn_action_to_lua(action))
        return self.rewarder.reward(self.last_ram)

    def restart(self):
        self.emu.softreset()
