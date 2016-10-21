from unittest import TestCase

from hsa.nes_python_input import py_to_nes_wrapper


class TestPy_to_nes_wrapper(TestCase):
    def test_py_to_nes_wrapper(self):
        self.assertEqual(17, py_to_nes_wrapper({"A": True, "B": False, "up": True}))
        self.assertEqual(0, py_to_nes_wrapper({"A": False, "B": False, "up": False}))
