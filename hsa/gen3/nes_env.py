import os

from gym.core import Env

from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np
import logging

from hsa.dqn_mario.dqn_input import rdqn_to_py
from hsa.nes_python_input import py_to_nes_wrapper
from hsa.reward_evaluation import mario_x_speed

logger = logging.getLogger(__name__)

# from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface

try:
    from nes_python_interface import NESInterface
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: build and install fceux interface)".format(e))


class NesEnv(Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}
    nr_actions = 36

    def __init__(self, game_path="~/mario.nes", frameskip=1, obs_type="ram", score_compatibility="v_keras"):
        """
            :arg score_compatibility witch scoring system to be compatible to
        """

        # noinspection PyCallByClass,PyTypeChecker
        utils.EzPickle.__init__(self, game_path, frameskip)
        self.score_compatibility = score_compatibility
        assert obs_type in ('ram', 'image')

        self.game_path = game_path
        if not os.path.exists(self.game_path):
            raise IOError('path %s does not exist' % self.game_path)
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.nes = NESInterface(game_path)
        self.viewer = None

        self._seed()

        (screen_width, screen_height) = self.nes.getScreenDims()
        self._buffer = np.empty((screen_height, screen_width, 4), dtype=np.uint8)

        self.action_space = spaces.Discrete(NesEnv.nr_actions)

        (screen_width, screen_height) = self.nes.getScreenDims()
        if self._obs_type == 'ram':
            # nes has 2KB ram
            self.observation_space = spaces.Box(low=np.zeros(2048), high=np.zeros(2048) + 255)
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _seed(self, seed=None):
        logger.warning("NES/FCEUX does not support seeding")
        return []

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self.nes.getRAM()
        elif self._obs_type == 'image':
            return self.nes.getScreenRGB()

    def _step(self, action):
        encoded_action = py_to_nes_wrapper(rdqn_to_py(action))
        reward = 0.0
        num_steps = self.frameskip
        for _ in range(num_steps):
            if self.score_compatibility != "v_keras":
                self.nes.act(encoded_action)
                reward += mario_x_speed(self.nes.getRAM())
            else:
                self.nes.act(encoded_action)
        if self.score_compatibility == "v_keras":
            reward = mario_x_speed(self.nes.getRAM())
        ob = self._get_obs()

        return ob, reward, self.nes.lives() <= 1, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.nes.getScreenRGB()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _close(self):
        del self.nes

    def _reset(self):
        self.nes.reset_game()
        return self._get_obs()

    def _configure(self):
        pass

    def get_action_meanings(self):
        return [rdqn_to_py(i) for i in NesEnv.nr_actions]
