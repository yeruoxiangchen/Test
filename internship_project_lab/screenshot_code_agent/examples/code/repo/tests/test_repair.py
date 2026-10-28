import unittest
import miniapp as m

class TestRepair(unittest.TestCase):
    def test_target(self):
        self.assertEqual(m.add(7, 2), 9)
        self.assertEqual(m.add(-2, 2), 0)

    def test_regression(self):
        self.assertTrue(callable(m.add))
