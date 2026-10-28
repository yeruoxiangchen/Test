import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_count_probe.core import atomic_json, build_plan as old_plan, file_sha, source_hash, writer_lock
from video_count_probe.action_calibration.experiment import ability_score, build_plan, generate, prepare, report, verify_reference


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / 'reference'
        self.plan = old_plan('pilot')
        self.identity = {'plan_hash': self.plan['plan_hash'], 'source_hash': source_hash(),
                         'model_receipt_hash': 'fixture', 'packages': {}}
        atomic_json(self.reference / 'plan.json', self.plan)
        atomic_json(self.reference / 'identity.json', self.identity)
        annotations = []
        for job in self.plan['jobs']:
            video = self.reference / 'videos' / (job['id'] + '.mp4')
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'fixture output')
            record = {'status': 'ok', 'job': job, 'identity': self.identity, 'video': str(video.relative_to(self.reference)),
                      'video_sha256': file_sha(video), 'initial_latent_sha256': str(job['seed'])}
            atomic_json(self.reference / 'records' / (job['id'] + '.json'), record)
            annotations.append({'job_id': job['id'], 'video_sha256': record['video_sha256'], 'rater': 'AI fixture',
                                'complete': True, 'visually_usable': True, 'action_present': True,
                                'ambiguous': False, 'completion_times': [1.0]})
        self.annotation_path = self.reference / 'ai_review_20261027/annotations_ai.json'
        atomic_json(self.annotation_path, {'plan_hash': self.plan['plan_hash'], 'annotation_source': 'ai_visual_frame_review',
                                         'annotations': annotations})

    def test_only_twelve_jobs_with_matching_config_and_numeric_ablation(self):
        plan = build_plan(self.reference, 'pilot')
        self.assertEqual(len(plan['jobs']), 12)
        self.assertEqual(len({j['id'] for j in plan['jobs']}), 12)
        self.assertEqual(plan['config'], self.plan['config'])
        self.assertEqual(len(plan['reference']['counted_records']), 18)
        for case in plan['cases']:
            for old in self.plan['cases']:
                if old['action'] == case['id'] and old['schedule'] == 'regular':
                    self.assertEqual(case['prompts']['uncounted_matched'], old['count_prompt'].replace(f"exactly {old['target_count']} ", '', 1))
        units = {(j['case_id'], j['seed']) for j in plan['jobs']}
        self.assertEqual(len(units), 6)
        self.assertTrue(all(sum((j['case_id'], j['seed']) == u for j in plan['jobs']) == 2 for u in units))

    def test_smoke_is_separate_and_excluded_from_scientific_analysis(self):
        smoke = build_plan(self.reference, 'smoke')
        pilot = build_plan(self.reference, 'pilot')
        self.assertEqual(len(smoke['jobs']), 1)
        self.assertEqual(smoke['config']['steps'], 4)
        self.assertFalse(smoke['scientific_evaluation'])
        self.assertFalse({j['id'] for j in smoke['jobs']} & {j['id'] for j in pilot['jobs']})

    def test_corrupt_reference_video_rejected(self):
        plan = build_plan(self.reference, 'pilot')
        next((self.reference / 'videos').glob('*.mp4')).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            verify_reference(plan)

    def test_changed_reference_annotations_rejected(self):
        plan = build_plan(self.reference, 'pilot')
        self.annotation_path.write_text(self.annotation_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'annotations changed'):
            verify_reference(plan)

    def test_prepare_is_idempotent_and_preserves_reference(self):
        destination = self.root / 'new'
        before = file_sha(self.reference / 'plan.json')
        prepare(destination, self.reference)
        path = destination / 'pilot/plan.json'
        timestamp, sha = path.stat().st_mtime_ns, file_sha(path)
        prepare(destination, self.reference)
        self.assertEqual((path.stat().st_mtime_ns, file_sha(path)), (timestamp, sha))
        self.assertEqual(file_sha(self.reference / 'plan.json'), before)

    def test_completed_resume_does_not_access_gpu_or_model(self):
        plan = build_plan(self.reference, 'pilot')
        run = self.root / 'new/pilot'
        identity = {'plan_hash': plan['plan_hash'], 'test': True}
        atomic_json(run / 'plan.json', plan)
        atomic_json(run / 'identity.json', identity)
        for job in plan['jobs']:
            video = run / 'videos' / (job['id'] + '.mp4')
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'complete')
            atomic_json(run / 'records' / (job['id'] + '.json'),
                        {'status': 'ok', 'job': job, 'identity': identity,
                         'video': str(video.relative_to(run)), 'video_sha256': file_sha(video)})
        with patch('video_count_probe.action_calibration.experiment.current_identity', return_value=identity), \
             patch('video_count_probe.runtime.check_gpu', side_effect=AssertionError('GPU must not be used')):
            generate(run.parent, 'pilot', 'unused')

    def test_duplicate_writer_is_rejected_before_gpu_access(self):
        plan = build_plan(self.reference, 'pilot')
        run = self.root / 'new/pilot'
        atomic_json(run / 'plan.json', plan)
        with writer_lock(run / '.writer.lock'), \
             patch('video_count_probe.action_calibration.experiment.current_identity', return_value={}), \
             patch('video_count_probe.runtime.check_gpu', side_effect=AssertionError('GPU must not be used')):
            with self.assertRaisesRegex(RuntimeError, 'Another writer'):
                generate(run.parent, 'pilot', 'unused')

    def test_readonly_status_works_with_active_writer_and_no_annotations(self):
        plan = build_plan(self.reference, 'pilot')
        run = self.root / 'new/pilot'
        atomic_json(run / 'plan.json', plan)
        with writer_lock(run / '.writer.lock'):
            result = report(run.parent, 'pilot', write=False)
        self.assertFalse((run / 'summary.json').exists())
        self.assertEqual(result['generated'], 0)
        self.assertFalse(result['assessment_complete'])
        self.assertFalse(result['human_validated'])
        self.assertEqual(result['methods']['short_action']['n'], 0)
        self.assertEqual(result['methods']['count_n1']['n'], 6)


class AbilityTests(unittest.TestCase):
    def label(self, **kw):
        return {'visually_usable': True, 'action_present': True, 'ambiguous': False,
                'completion_times': [1.0], **kw}

    def test_half_cycle_does_not_establish_complete_action(self):
        self.assertEqual(ability_score(self.label(completion_times=[])), 'no_complete_cycle')

    def test_uncertain_exact_count_can_confirm_action_ability(self):
        self.assertEqual(ability_score(self.label(ambiguous=True, count_range=[10, 15])), 'complete_cycle')
        self.assertEqual(ability_score(self.label(ambiguous=True, count_range=[0, 1])), 'uncertain')

    def test_unusable_video_is_not_rescued_by_candidate_times(self):
        self.assertEqual(ability_score(self.label(visually_usable=False)), 'unusable')

    def test_explicit_cycle_evidence_can_resolve_total_count_ambiguity(self):
        self.assertEqual(ability_score(self.label(ambiguous=True, complete_cycle_observed=True)), 'complete_cycle')

    def test_inconsistent_annotations_rejected(self):
        for change in ({'complete_cycle_observed': False}, {'complete_cycle_observed': 'true'},
                       {'complete_cycle_observed': True, 'action_present': False},
                       {'ambiguous': True, 'count_range': [2, 1]},
                       {'complete_cycle_observed': True, 'count_range': [0, 0]}):
            with self.assertRaises(ValueError):
                ability_score(self.label(**change))


if __name__ == '__main__':
    unittest.main()
