"""Check that diagnostic interventions cannot leak into subsequent variants."""
import unittest
import torch
from .forward_probe import condition


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.view_gate = torch.nn.Identity()
        self.match_absolute_tolerance = .02
        self.fail = False
        self.seen_gate = None
        self.seen_tolerance = None

    def build_conditions(self, sample, **kwargs):
        self.seen_gate = self.view_gate(torch.ones(2))
        self.seen_tolerance = self.match_absolute_tolerance
        if self.fail:
            raise RuntimeError('simulated forward failure')
        return self.seen_gate, torch.ones(3), {'marker': torch.tensor(7.)}


class InterventionIsolationTests(unittest.TestCase):
    def test_gate_is_removed_only_within_injection_probe(self):
        model = TinyModel()
        result = condition(model, {}, None, None, 'injection_only', 'cpu')
        self.assertTrue(torch.equal(result[0], torch.zeros(2)))
        self.assertTrue(torch.equal(model.view_gate(torch.ones(2)), torch.ones(2)))
        self.assertEqual(len(model.view_gate._forward_hooks), 0)

    def test_gate_hook_removed_after_failure(self):
        model = TinyModel()
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            condition(model, {}, None, None, 'injection_only', 'cpu')
        self.assertEqual(len(model.view_gate._forward_hooks), 0)
        self.assertTrue(torch.equal(model.view_gate(torch.ones(2)), torch.ones(2)))

    def test_tolerance_restored_after_failure(self):
        model = TinyModel()
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            condition(model, {}, None, None, 'tolerance_010', 'cpu')
        self.assertEqual(model.seen_tolerance, .1)
        self.assertEqual(model.match_absolute_tolerance, .02)

    def test_gate_only_keeps_appearance_and_removes_entire_injection(self):
        result = condition(TinyModel(), {}, None, None, 'view_gate_only', 'cpu')
        self.assertTrue(torch.equal(result[0], torch.ones(2)))
        self.assertIsNone(result[1])
        self.assertEqual(float(result[2]['marker']), 7.)


if __name__ == '__main__':
    unittest.main()
