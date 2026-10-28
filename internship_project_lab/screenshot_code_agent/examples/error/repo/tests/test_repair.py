import unittest
import miniapp as m

class TestRepair(unittest.TestCase):
    def test_target(self):
        self.assertEqual(m.mean([2, 4, 6]), 4)
        self.assertEqual(m.mean([9]), 9)

    def test_regression(self):
        self.assertTrue(callable(m.mean))
