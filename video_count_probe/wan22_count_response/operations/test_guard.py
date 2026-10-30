"""Regression checks for the launcher that interrupted VAE decoding on 298 MiB."""
import unittest
from unittest.mock import Mock, patch
from video_count_probe.wan22_count_response.operations import guarded_retry as guard


class GuardTests(unittest.TestCase):
    def test_observed_298_mib_incident_keeps_capacity(self):
        info = {'used_mib': 15193, 'free_mib': 8934}
        self.assertEqual(guard.capacity_for_our_process(info, [{'pid': 3861817, 'memory_mib': '298'}]), 23829)
        self.assertGreater(guard.capacity_for_our_process(info, []), guard.MIN_CAPACITY_MIB)

    def test_our_own_allocations_are_not_external_pressure(self):
        self.assertEqual(guard.capacity_for_our_process({'used_mib': 23000, 'free_mib': 1127}, []), 24127)

    def test_startup_allows_small_idle_context_but_rejects_active_or_large(self):
        info = {'used_mib': 300, 'free_mib': 23827, 'utilization': 0,
                'processes': [{'pid': 2, 'memory_mib': '290'}]}
        self.assertTrue(guard.startup_ready(info))
        self.assertFalse(guard.startup_ready({**info, 'utilization': 51}))
        self.assertFalse(guard.startup_ready({**info, 'used_mib': 3556, 'free_mib': 20572}))

    def test_no_memory_value_is_not_assumed_excessive_usage(self):
        self.assertIsNone(guard.capacity_for_our_process({'used_mib': 100, 'free_mib': 24027}, [{'memory_mib': 'N/A'}]))

    def test_tiny_competitor_does_not_interrupt_stage(self):
        child = Mock(pid=123)
        child.poll.side_effect = [None, None, 0]
        child.wait.return_value = 0
        info = {'used_mib': 15193, 'free_mib': 8934, 'processes': [{'pid': 2, 'memory_mib': '298'}]}
        with patch.object(guard, 'wait_idle', return_value={}), \
             patch.object(guard.subprocess, 'Popen', return_value=child), \
             patch.object(guard, 'snapshot', return_value=info), \
             patch.object(guard, 'foreign_processes', side_effect=lambda info, pid: info['processes']), \
             patch.object(guard.time, 'sleep'), patch.object(guard, 'stop_owned') as stop, \
             patch.object(guard, 'event') as event:
            self.assertEqual(guard.run_stage('/data/zjr/fixture', 'smoke', '3'), 0)
            stop.assert_called_once_with(child)  # finally cleanup, no early termination
            self.assertNotIn('gpu_headroom_insufficient', [c.args[0] for c in event.call_args_list])

    def test_only_consecutive_pressure_terminates_own_child(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        bad = {'used_mib': 20000, 'free_mib': 4127, 'processes': [{'pid': 2, 'memory_mib': '4000'}]}
        good = {**bad, 'processes': [{'pid': 2, 'memory_mib': '298'}]}
        with patch.object(guard, 'wait_idle', return_value={}), \
             patch.object(guard.subprocess, 'Popen', return_value=child), \
             patch.object(guard, 'snapshot', side_effect=[bad, good, bad, bad, bad]) as sample, \
             patch.object(guard, 'foreign_processes', side_effect=lambda info, pid: info['processes']), \
             patch.object(guard.time, 'sleep'), patch.object(guard, 'stop_owned') as stop, \
             patch.object(guard, 'event'):
            self.assertEqual(guard.run_stage('/data/zjr/fixture', 'smoke', '3'), 75)
            self.assertEqual(sample.call_count, 5)
            self.assertTrue(all(c.args == (child,) for c in stop.call_args_list))


if __name__ == '__main__':
    unittest.main()
