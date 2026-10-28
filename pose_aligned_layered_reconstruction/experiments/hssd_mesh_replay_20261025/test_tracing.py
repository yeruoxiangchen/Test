import tempfile
import unittest
from pathlib import Path

import torch

from .common import binding
from .tracing import Trace, flatten, compare_traces


class TraceTests(unittest.TestCase):
    def test_detach_copy_prevents_later_inplace_mutation(self):
        x = torch.arange(4., requires_grad=True)
        snapshot = flatten(x)['tensor']
        with torch.no_grad():
            x.add_(10)
        self.assertTrue(torch.equal(snapshot, torch.arange(4.)))
        self.assertFalse(snapshot.requires_grad)

    def test_ordered_first_difference_and_bfloat16(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Trace(Path(d)/'a'), Trace(Path(d)/'b')
            for t in [a, b]:
                t.add('initial', torch.ones(3, dtype=torch.bfloat16))
            a.add('stage', torch.tensor([1., 2.]))
            b.add('stage', torch.tensor([1., 2.25]))
            result = compare_traces(a.finish(), b.finish())
            self.assertFalse(result['all_exact'])
            self.assertEqual(result['first_difference']['name'], 'stage')
            self.assertEqual(result['first_difference']['differences']['tensor']['max_abs'], .25)

    def test_missing_event_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Trace(Path(d)/'a'), Trace(Path(d)/'b')
            a.add('initial', torch.ones(1))
            b.add('wrong_stage', torch.ones(1))
            with self.assertRaises(ValueError):
                compare_traces(a.finish(), b.finish())

    def test_trace_never_changes_forward_output(self):
        with tempfile.TemporaryDirectory() as d:
            model = torch.nn.Linear(3, 4)
            x = torch.randn(2, 3)
            expected = model(x)
            trace = Trace(Path(d)/'trace')
            def hook(m, args, value):
                trace.add('output', value)
            handle = model.register_forward_hook(hook)
            actual = model(x)
            handle.remove()
            self.assertTrue(torch.equal(actual, expected))

    def test_storage_bound_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            t = Trace(Path(d)/'small', limit=4)
            with self.assertRaises(RuntimeError):
                t.add('too_big', torch.zeros(2))
            t = Trace(Path(d)/'finite')
            with self.assertRaises(ValueError):
                t.add('bad', torch.tensor([float('nan')]))


if __name__ == '__main__':
    unittest.main()
