import collections
import os
import sys

import numpy as np
from nes_python_interface import NESInterface
import scipy.misc

import environment

# ideally this would be gotten from the c++ code but because of shitty globals and fork we have to have it redundantly in python
n_actions = 15

class NES(environment.EpisodicEnvironment):
    """Arcade Learning Environment.
    """

    def __init__(self, rom_filename, use_sdl=False, n_last_screens=4,
                 frame_skip=4, treat_life_lost_as_terminal=False,
                 crop_or_scale='scale', max_start_nullops=0,
                 record_screen_path=None, outside_nes_interface=None
                 , every_action_callback=None, scale_size=(64,64)):
        self.every_action_callback = every_action_callback
        self.n_last_screens = n_last_screens
        self.treat_life_lost_as_terminal = treat_life_lost_as_terminal
        self.crop_or_scale = crop_or_scale
        self.max_start_nullops = max_start_nullops
        self.scale_size = scale_size

        if outside_nes_interface is not None:
            nes = outside_nes_interface
        else:
            nes = NESInterface(rom_filename)
#        nes.setInt(b'random_seed', seed)
#        nes.setFloat(b'repeat_action_probability', 0.0)
#        nes.setBool(b'color_averaging', False)
        self.frame_skip = frame_skip
#        if use_sdl:
#            if 'DISPLAY' not in os.environ:
#                raise RuntimeError(
#                    'Please set DISPLAY environment variable for use_sdl=True')
            # SDL settings below are from the ALE python example
#            if sys.platform == 'darwin':
#                import pygame
#                pygame.init()
#                ale.setBool(b'sound', False)  # Sound doesn't work on OSX
#            elif sys.platform.startswith('linux'):
#                ale.setBool(b'sound', True)
#            ale.setBool(b'display_screen', True)
#        ale.loadROM(str.encode(rom_filename))

        self.nes = nes
        self.legal_actions = nes.getMinimalActionSet()
        self.initialize()

        # maybe initialize recording
        if record_screen_path is not None:
            import imageio
            self.movie_writer = imageio.get_writer(record_screen_path, fps=60, quality=9,
                                                   ffmpeg_params=["-vf", "scale=iw*2:ih*2:flags=neighbor"])
        else:
            self.movie_writer = None

    def current_screen(self):
        rgb_img = self.nes.getScreenRGB()
        img = scipy.misc.imresize(rgb_img, self.scale_size)
        return img

    @property
    def state(self):
        assert len(self.last_screens) == 4
        return list(self.last_screens)

    @property
    def is_terminal(self):
        if self.treat_life_lost_as_terminal:
            return self.lives_lost or self.nes.game_over()
        else:
            return self.nes.game_over()

    @property
    def reward(self):
        return self._reward

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

    def receive_action(self, action):
        assert not self.is_terminal

        rewards = []
        for i in range(4):

            # Last screeen must be stored before executing the 4th action
            if i == 3:
                self.last_raw_screen = self.nes.getScreenRGB()

            rewards.append(self.nes.act(self.legal_actions[action]))
            if self.every_action_callback is not None:
                self.every_action_callback(i,self.nes.getRAM())
            # TODO append frame to video
            if self.movie_writer is not None:
                # method does not return rgb #blame ale
                bgr = self.nes.getScreenRGB()
                rgb = bgr[:,:,[2,1,0]]
                self.movie_writer.append_data(rgb)

            # Check if lives are lost
            if self.lives > self.nes.lives():
                self.lives_lost = True
            else:
                self.lives_lost = False
            self.lives = self.nes.lives()

            if self.is_terminal:
                break

        # We must have last screen here unless it's terminal
        if not self.is_terminal:
            self.last_screens.append(self.current_screen())

        self._reward = sum(rewards)

        return self._reward

    def initialize(self):

        self.nes.reset_game()

        if self.max_start_nullops > 0:
            n_nullops = np.random.randint(0, self.max_start_nullops + 1)
            for _ in range(n_nullops):
                self.nes.act(0)

        self._reward = 0

        self.last_raw_screen = self.nes.getScreenRGB()

        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * 3 +
            [self.current_screen()],
            maxlen=self.n_last_screens)

        self.lives_lost = False
        self.lives = self.nes.lives()


