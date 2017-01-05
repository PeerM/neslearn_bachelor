from unittest import TestCase

from hsa.ba.rewards import make_delta_points, make_delta_potential
import numpy as np


class Test_rewards(TestCase):
    def test_delta_points(self):
        delta_points = make_delta_points()
        ram = np.zeros(2048, dtype=np.uint8)
        ram[0x07E1] = 3
        self.assertEqual(delta_points(ram), 30)
        self.assertEqual(delta_points(ram), 0)
        ram[0x07E1] = 2
        self.assertEqual(delta_points(ram), -10)
        ram[0x07E0] = 1
        self.assertEqual(delta_points(ram), 100)

    def test_delta_potential_without_shortcut(self):
        delta_potential = make_delta_potential()
        ram = np.zeros(2048, dtype=np.uint8)
        screens = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12)]
        deltas = [0] + [1] * 12
        for (current_s, next_s), right_delta in zip(screens, deltas):
            ram[0x071A] = current_s
            ram[0x071B] = next_s
            delta = delta_potential(ram)
            self.assertEqual(delta, right_delta)

    def test_delta_potential_standing(self):
        delta_potential = make_delta_potential()
        ram = np.zeros(2048, dtype=np.uint8)
        screens = [(0, 0), (0, 0), (0, 0), (1, 1), (1, 1)]
        deltas = [0, 0, 0, 1, 0]
        for (current_s, next_s), right_delta in zip(screens, deltas):
            ram[0x071A] = current_s
            ram[0x071B] = next_s
            subject = delta_potential(ram)
            self.assertEqual(subject, right_delta)

    def test_delta_potential_with_shortcut(self):
        delta_potential = make_delta_potential()
        ram = np.zeros(2048, dtype=np.uint8)
        screens = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 4), (10, 0), (11, 10), (12, 11)]
        deltas = [0, 1, 1, 1, 6, 1, 1, 1]
        for (current_s, next_s), right_delta in zip(screens, deltas):
            ram[0x071A] = current_s
            ram[0x071B] = next_s
            delta = delta_potential(ram)
            self.assertEqual(delta, right_delta)
