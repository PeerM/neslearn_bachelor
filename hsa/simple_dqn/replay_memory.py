import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


class ReplayMemory:
    def __init__(self, size, args):
        self.size = size
        # preallocate memory
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.int32)
        self.screens = np.empty((self.size, args.screen_height), dtype=np.uint8)
        self.terminals = np.empty(self.size, dtype=np.bool)
        self.dims = (args.screen_height,)
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size,) + self.dims, dtype=np.uint8)
        self.poststates = np.empty((self.batch_size,) + self.dims, dtype=np.uint8)

        logger.info("Replay memory size: %d" % self.size)

    def add(self, action, reward, screen, terminal):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size
        # logger.debug("Memory count %d" % self.count)

    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        return self.screens[index]

    def getMinibatch(self):
        # memory must include poststate, prestate and history
        # PEER: maybe find a proper equivalent for this
        # assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            index = random.randint(0, self.count - 1)
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        return self.prestates, actions, rewards, self.poststates, terminals
