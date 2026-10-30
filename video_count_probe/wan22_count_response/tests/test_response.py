import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from video_count_probe.core import atomic_json, digest, file_sha, read_json
from video_count_probe.wan22_count_response import protocol, review, runtime


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.reference, self.root = self.base / 'reference/pilot', self.base / 'output'
        config = {'frames': 193, 'fps': 24, 'height': 704, 'width': 1280, 'steps': 50,
                  'guidance_scale': 5.0, 'flow_shift': 5.0, 'input_image': None,
                  'prompt_rewrite': False, 'negative_prompt': 'fixture negative'}
        old = {'experiment': 'wan22_duration_v1', 'mode': 'pilot',
               'conditions': {'c8': {'config': config, 'target_count': 2, 'prompt': protocol.count_prompt(2)}},
               'jobs': [{'condition': 'c8', 'seed': s, 'id': f'ref{s}'} for s in protocol.SEEDS]}
        old['plan_hash'] = digest(old)
        self.ref_identity = {'plan_hash': old['plan_hash'], 'source_hash': protocol.duration_code_hash(),
                             'model_receipt_hash': 'fixture', 'packages': {'diffusers': '0.35.1'}}
        atomic_json(self.reference / 'plan.json', old)
        atomic_json(self.reference / 'identity.json', self.ref_identity)
        labels = {'plan_hash': old['plan_hash'], 'rater': 'fixture', 'rater_type': 'ai', 'items': []}
        for job in old['jobs']:
            video = self.reference / 'videos' / (job['id'] + '.mp4')
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(job['id'].encode())
            record = {'status': 'ok', 'job': job, 'identity': self.ref_identity, 'config': config,
                      'prompt': protocol.count_prompt(2), 'video': str(video.relative_to(self.reference)),
                      'video_sha256': file_sha(video), 'initial_latent_sha256': f'noise{job["seed"]}',
                      'initial_latent_shape': [1, 48, 49, 44, 80]}
            atomic_json(self.reference / 'records' / (job['id'] + '.json'), record)
            labels['items'].append({'job_id': job['id'], 'video_sha256': record['video_sha256'],
                                   'reviewed': True, 'count_judgeable': True, 'complete_cycles': [],
                                   'partial_attempts': [], 'initial_upright': True,
                                   'final_second_upright_still': True, 'notes': 'fixture, zero complete actions'})
        atomic_json(self.reference / 'annotations.json', labels)
        self.original_labels = (self.reference / 'annotations.json').read_bytes()
        self.plan = protocol.build_plan('pilot', self.reference)
        atomic_json(self.root / 'pilot/plan.json', self.plan)
        self.identity = {**self.ref_identity, 'plan_hash': self.plan['plan_hash'], 'source_hash': 'new'}

    def commit(self, job, plan=None):
        plan = plan or self.plan
        run = self.root / plan['mode']
        identity = {**self.identity, 'plan_hash': plan['plan_hash']}
        atomic_json(run / 'identity.json', identity)
        video = run / 'videos' / (job['id'] + '.mp4')
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(job['id'].encode())
        spec = plan['conditions'][job['condition']]
        record = {'status': 'ok', 'job': job, 'identity': identity, 'config': spec['config'], 'prompt': spec['prompt'],
                  'video': str(video.relative_to(run)), 'video_sha256': file_sha(video),
                  'initial_latent_sha256': f'noise{job["seed"]}', 'initial_latent_shape': [1, 48, 49, 44, 80],
                  'generation_seconds': 1, 'peak_allocated_mib': 18000, 'peak_reserved_mib': 21902,
                  **{k: spec['config'][k] for k in ('frames', 'fps', 'height', 'width')}}
        atomic_json(run / 'records' / (job['id'] + '.json'), record)
        return record

    def test_plan_only_adds_six_and_preserves_n2_prompt(self):
        self.assertEqual(len(self.plan['jobs']), 6)
        self.assertEqual({j['condition'] for j in self.plan['jobs']}, {'n1', 'n3'})
        self.assertEqual(self.plan['conditions']['n2']['prompt'], read_json(self.reference / 'plan.json')['conditions']['c8']['prompt'])
        self.assertNotIn('second', protocol.count_prompt(1))
        self.assertIn('exactly three squats', protocol.count_prompt(3))
        self.assertEqual(protocol.count_prompt(3).count('completing the third squat'), 2)
        self.assertTrue(all(c['config'] == self.plan['conditions']['n2']['config'] for c in self.plan['conditions'].values()))
        self.assertEqual((self.reference / 'annotations.json').read_bytes(), self.original_labels)

    def test_changed_reference_labels_or_video_rejected(self):
        path = self.reference / 'annotations.json'
        labels = read_json(path)
        labels['rater'] = 'different reviewer'
        atomic_json(path, labels)
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            protocol.verify_reference(self.plan)
        path.write_bytes(self.original_labels)
        (self.reference / 'videos/ref2026.mp4').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'missing/corrupt'):
            protocol.verify_reference(self.plan)

    def test_noise_must_match_for_both_new_targets(self):
        for condition in ('n1', 'n3'):
            job = next(j for j in self.plan['jobs'] if j['condition'] == condition)
            record = self.commit(job)
            record['initial_latent_sha256'] = 'wrong noise'
            path = self.root / 'pilot/records' / (job['id'] + '.json')
            atomic_json(path, record)
            with self.assertRaisesRegex(ValueError, 'latent mismatch'):
                review.report(self.root, write=False)
            path.unlink()

    def test_unreviewed_is_unknown_and_reference_not_new(self):
        self.commit(self.plan['jobs'][0])
        result = review.report(self.root)
        self.assertEqual((result['generated_new'], result['reused_n2'], result['reviewed_new']), (1, 3, 0))
        self.assertFalse(result['assessment_complete'])
        self.assertEqual(result['conditions']['n1']['exact_count']['false'], 0)
        self.assertTrue(all(x['flat_response'] is None for x in result['paired_count_response']))
        self.assertEqual((self.reference / 'annotations.json').read_bytes(), self.original_labels)

    def test_target_dependent_scoring_and_stop_without_deadline(self):
        label = {'reviewed': True, 'count_judgeable': True,
                 'complete_cycles': [{'start_frame': 0, 'bottom_frame': 90, 'end_frame': 190}],
                 'partial_attempts': [], 'initial_upright': True, 'final_second_upright_still': False}
        self.assertTrue(review.score(label, self.plan['conditions']['n1'])['count_and_stop'])
        self.assertFalse(review.score(label, self.plan['conditions']['n3'])['count_and_stop'])
        label['partial_attempts'] = [{'start_frame': 191, 'end_frame': 192}]
        self.assertFalse(review.score(label, self.plan['conditions']['n1'])['count_and_stop'])

    def test_matching_full_geometry_smoke_required(self):
        smoke = protocol.build_plan('smoke', self.reference)
        atomic_json(self.root / 'smoke/plan.json', smoke)
        with self.assertRaisesRegex(RuntimeError, 'n3 smoke'):
            runtime.require_smoke(self.root, self.plan, self.identity, {'n1', 'n3'}, None)
        self.commit(smoke['jobs'][0], smoke)
        runtime.require_smoke(self.root, self.plan, self.identity, {'n1', 'n3'}, None)
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            runtime.require_smoke(self.root, self.plan, {**self.identity, 'source_hash': 'changed'}, {'n1'}, None)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            runtime.require_smoke(self.root, self.plan, self.identity, {'n1'}, 17408)

    def test_resume_failure_and_condition_filter(self):
        job = self.plan['jobs'][0]
        self.commit(job)
        run = self.root / 'pilot'
        complete, pending = runtime.pending_jobs(run, self.plan, self.identity, 'n1', None, False)
        self.assertEqual((len(complete), len(pending)), (1, 2))
        error = run / 'errors' / (pending[0]['id'] + '.json')
        atomic_json(error, {'state': 'interrupted'})
        self.assertEqual(len(runtime.pending_jobs(run, self.plan, self.identity, 'n1', 1, False)[1]), 1)
        atomic_json(error, {'state': 'failed'})
        with self.assertRaisesRegex(RuntimeError, 'Prior failure'):
            runtime.pending_jobs(run, self.plan, self.identity, 'n1', None, False)

    def test_review_hides_target_and_preserves_existing_annotations(self):
        self.commit(self.plan['jobs'][0])
        def sheet(_video, dest, _cfg, _title):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b'fixture')
        with patch.object(review, 'make_sheet', side_effect=sheet) as writer:
            review.review(self.root)
            annotations = self.root / 'pilot/annotations.json'
            annotations.write_text('{"reviewer_draft":true}')
            review.review(self.root)
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(read_json(annotations), {'reviewer_draft': True})
        public = read_json(self.root / 'pilot/review_frames/blind_manifest.json')['items'][0]
        self.assertNotIn('condition', public)
        self.assertNotIn('target_count', public)


if __name__ == '__main__':
    unittest.main()
