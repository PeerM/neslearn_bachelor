from unittest import TestCase

from hsa.v_keras.qlearning4k.agent import make_epsilon_schedule


class TestMakeEpsilonSchedule(TestCase):
    def test_make_epsilon_schedule(self):
        self.assertSequenceEqual([0.5] * 20, list(make_epsilon_schedule(0.5, 0.5, 20, 0.4)))

    def test_epsilonSchedule_oneEpoch_oneEpsilon(self):
        self.assertSequenceEqual([0.5], list(make_epsilon_schedule(0.5, 0.5, 1, 1)))

    def test_epsilonSchedule_twoEpoch_twoEpsilon(self):
        self.assertSequenceEqual([0.5, 0.5], list(make_epsilon_schedule(0.5, 0.5, 2, 1)))

    def test_epsilonSchedule_realisticSchedule(self):
        self.assertSequenceEqual([0.9, 0.5, 0.1, 0.1], list(make_epsilon_schedule(0.9, 0.1, 4, 0.5)))

    def test_epsilonSchedule_epsilonRateHigh(self):
        subject = list(make_epsilon_schedule(0.9, 0.1, 5, 0.8))
        self.assertEqual(subject[0],0.9)
        self.assertEqual(subject[4],0.1)