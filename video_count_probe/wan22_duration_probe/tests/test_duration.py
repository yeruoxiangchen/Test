import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from video_count_probe.core import atomic_json, digest, file_sha, read_json
from video_count_probe.wan22_squat_count.protocol import prompts, SEEDS
from video_count_probe.wan22_duration_probe import protocol, review, runtime


def seal(value):
    value.pop('plan_hash', None)
    value['plan_hash'] = digest(value)
    return value


class DurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.reference = self.base / 'previous/pilot'
        self.root = self.base / 'new'
        cfg = {'height': 704, 'width': 1280, 'frames': 121, 'fps': 24, 'steps': 50,
               'guidance_scale': 5.0, 'flow_shift': 5.0, 'input_image': None,
               'prompt_rewrite': False, 'negative_prompt': 'bad anatomy'}
        jobs = [{'id': f'ref{seed}', 'case_id': 'squat_n2', 'seed': seed, 'method': 'phase_prompt'} for seed in SEEDS]
        old = seal({'experiment': 'wan22_squat_count_v1', 'mode': 'pilot', 'config': cfg, 'jobs': jobs,
                    'scoring': {'last_frame': 120, 'rest_start_frame': 96}})
        identity = {'plan_hash': old['plan_hash'], 'source_hash': 'old-source',
                    'model_receipt_hash': 'model', 'packages': {'diffusers': '0.35.1'}}
        atomic_json(self.reference / 'plan.json', old)
        atomic_json(self.reference / 'identity.json', identity)
        labels = []
        for job in jobs:
            video = self.reference / 'videos' / (job['id'] + '.mp4')
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(job['id'].encode())
            record = {'status': 'ok', 'job': job, 'identity': identity, 'prompt': prompts(2)['phase_prompt'],
                      'video': str(video.relative_to(self.reference)), 'video_sha256': file_sha(video),
                      'initial_latent_sha256': 'noise' + str(job['seed'])}
            atomic_json(self.reference / 'records' / (job['id'] + '.json'), record)
            labels.append({'job_id': job['id'], 'video_sha256': record['video_sha256'], 'reviewed': True,
                           'count_judgeable': True, 'complete_cycles': [{'start_frame': 0, 'bottom_frame': 48, 'end_frame': 90}],
                           'partial_attempts': [], 'initial_upright': True, 'final_second_upright_still': True, 'notes': 'test fixture'})
        atomic_json(self.reference / 'annotations.json', {'plan_hash': old['plan_hash'], 'rater': 'fixture', 'rater_type': 'ai', 'items': labels})
        self.old_label_bytes = (self.reference / 'annotations.json').read_bytes()
        self.plan = protocol.build_plan('pilot', self.reference)
        self.identity = {'plan_hash': self.plan['plan_hash'], 'source_hash': 'new-source',
                         'model_receipt_hash': 'model', 'packages': {'diffusers': '0.35.1'}}
        atomic_json(self.root / 'pilot/plan.json', self.plan)

    def commit_record(self, job, plan=None, identity=None):
        plan = plan or self.plan
        run = self.root / plan['mode']
        identity = identity or {**self.identity, 'plan_hash': plan['plan_hash']}
        spec = plan['conditions'][job['condition']]
        atomic_json(run / 'identity.json', identity)
        video = run / 'videos' / (job['id'] + '.mp4')
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(job['id'].encode())
        record = {'status': 'ok', 'job': job, 'identity': identity, 'config': spec['config'], 'prompt': spec['prompt'],
                  'video': str(video.relative_to(run)), 'video_sha256': file_sha(video),
                  'initial_latent_sha256': ('noise' if job['condition'] == 'b5' else 'long-noise') + str(job['seed']),
                  'peak_allocated_mib': 14000, 'peak_reserved_mib': 18000, 'generation_seconds': 1.0,
                  **{k: spec['config'][k] for k in ('frames', 'fps', 'height', 'width')}}
        atomic_json(run / 'records' / (job['id'] + '.json'), record)
        return record

    def test_reference_and_factor_isolation(self):
        p = self.plan
        self.assertEqual(len(p['jobs']), 6)
        self.assertEqual({j['condition'] for j in p['jobs']}, {'b5', 'c8'})
        b, c = p['conditions']['b5'], p['conditions']['c8']
        self.assertEqual(b['prompt'], c['prompt'])
        self.assertNotIn('seconds', b['prompt'])
        self.assertIn('completing the second squat', b['prompt'])
        self.assertEqual(c['config'], {**b['config'], 'frames': 193})
        self.assertEqual(protocol.scoring_plan(c)['scoring'], {'last_frame': 192, 'rest_start_frame': 168})
        self.assertEqual(len(protocol.all_prompts(p)), 2)
        self.assertEqual((self.reference / 'annotations.json').read_bytes(), self.old_label_bytes)

    def test_reference_corruption_and_label_change_rejected(self):
        target = self.reference / 'videos' / f'ref{SEEDS[0]}.mp4'
        old = target.read_bytes()
        target.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'incomplete/corrupt'):
            protocol.verify_reference(self.plan)
        target.write_bytes(old)
        annotations = read_json(self.reference / 'annotations.json')
        annotations['rater'] = 'another reviewer'
        atomic_json(self.reference / 'annotations.json', annotations)
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            protocol.verify_reference(self.plan)

    def test_unknown_is_not_failure_and_no_new_fixed_deadline(self):
        row = {'reviewed': True, 'count_judgeable': False, 'complete_cycles': None, 'partial_attempts': None,
               'initial_upright': None, 'final_second_upright_still': None}
        self.assertIsNone(review.score(row, self.plan['conditions']['c8'])['exact_count'])
        row.update(count_judgeable=True, complete_cycles=[{'start_frame': 0, 'bottom_frame': 50, 'end_frame': 100},
                                                        {'start_frame': 101, 'bottom_frame': 150, 'end_frame': 190}],
                   partial_attempts=[], final_second_upright_still=False)
        result = review.score(row, self.plan['conditions']['c8'])
        self.assertTrue(result['exact_count'])
        self.assertTrue(result['count_and_stop'])
        self.assertFalse(result['completed_before_final_second'])
        self.assertAlmostEqual(result['target_completion_seconds'], 190 / 24)
        row['partial_attempts'] = [{'start_frame': 191, 'end_frame': 192}]
        self.assertFalse(review.score(row, self.plan['conditions']['c8'])['count_and_stop'])

    def test_long_annotation_bounds_and_overlaps(self):
        job = next(j for j in self.plan['jobs'] if j['condition'] == 'c8')
        record = self.commit_record(job)
        row = {'job_id': job['id'], 'video_sha256': record['video_sha256'], 'reviewed': True,
               'count_judgeable': True, 'complete_cycles': [{'start_frame': 125, 'bottom_frame': 150, 'end_frame': 190}],
               'partial_attempts': [], 'initial_upright': True, 'final_second_upright_still': False, 'notes': 'fixture'}
        review.validate_annotation(row, record, protocol.scoring_plan(self.plan['conditions']['c8']))
        with self.assertRaises(ValueError):
            review.validate_annotation(row, record, protocol.scoring_plan(self.plan['conditions']['b5']))
        row['partial_attempts'] = [{'start_frame': 160, 'end_frame': 192}]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            review.validate_annotation(row, record, protocol.scoring_plan(self.plan['conditions']['c8']))

    def test_report_does_not_count_reused_A_as_new_success(self):
        job = self.plan['jobs'][0]
        self.commit_record(job)
        result = review.report(self.root, 'pilot')
        self.assertEqual(result['generated_new'], 1)
        self.assertEqual(result['reused_reference'], 3)
        self.assertEqual(result['reviewed_new'], 0)
        self.assertFalse(result['assessment_complete'])
        self.assertEqual(result['paired']['a5_vs_b5']['exact_count']['decided_pairs'], 0)
        self.assertEqual(result['verified_a_b_noise_pairs'], 1)
        self.assertEqual((self.reference / 'annotations.json').read_bytes(), self.old_label_bytes)

    def test_record_prompt_and_B_noise_corruption_rejected(self):
        job = self.plan['jobs'][0]
        rec = self.commit_record(job)
        path = self.root / 'pilot/records' / (job['id'] + '.json')
        original = copy.deepcopy(rec)
        rec['prompt'] = 'wrong prompt'
        atomic_json(path, rec)
        with self.assertRaisesRegex(ValueError, 'configuration/prompt'):
            review.report(self.root, 'pilot', write=False)
        original['initial_latent_sha256'] = 'wrong noise'
        atomic_json(path, original)
        with self.assertRaisesRegex(ValueError, 'latent mismatch'):
            review.report(self.root, 'pilot', write=False)

    def test_resume_limit_condition_and_failure_policy(self):
        job = self.plan['jobs'][0]
        self.commit_record(job)
        run = self.root / 'pilot'
        complete, pending = runtime.pending_jobs(run, self.plan, self.identity, 'b5', None, False)
        self.assertEqual(len(complete), 1)
        self.assertEqual(len(pending), 2)
        path = run / 'errors' / (pending[0]['id'] + '.json')
        atomic_json(path, {'state': 'interrupted'})
        self.assertEqual(len(runtime.pending_jobs(run, self.plan, self.identity, 'b5', 1, False)[1]), 1)
        atomic_json(path, {'state': 'failed', 'exception': 'OOM'})
        with self.assertRaisesRegex(RuntimeError, 'Prior failure'):
            runtime.pending_jobs(run, self.plan, self.identity, 'b5', 1, False)
        self.assertEqual(len(runtime.pending_jobs(run, self.plan, self.identity, 'b5', 1, True)[1]), 1)

    def test_each_geometry_requires_matching_smoke_and_budget(self):
        smoke = protocol.build_plan('smoke', self.reference)
        atomic_json(self.root / 'smoke/plan.json', smoke)
        self.commit_record(smoke['jobs'][0], smoke)
        with self.assertRaisesRegex(RuntimeError, 'smoke --condition c8'):
            runtime.require_smoke(self.root, self.plan, self.identity, {'c8'}, 20000)
        self.commit_record(smoke['jobs'][1], smoke)
        runtime.require_smoke(self.root, self.plan, self.identity, {'b5', 'c8'}, 20000)
        with self.assertRaisesRegex(RuntimeError, 'budget .* smaller'):
            runtime.require_smoke(self.root, self.plan, self.identity, {'c8'}, 17408)
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            runtime.require_smoke(self.root, self.plan, {**self.identity, 'source_hash': 'changed'}, {'c8'}, None)
        record_path = self.root / 'smoke/records' / (smoke['jobs'][1]['id'] + '.json')
        changed = read_json(record_path)
        changed['config']['frames'] = 121
        atomic_json(record_path, changed)
        with self.assertRaisesRegex(RuntimeError, 'configuration mismatch'):
            runtime.require_smoke(self.root, self.plan, self.identity, {'c8'}, None)

    def test_review_stable_ids_and_existing_labels_preserved(self):
        job = self.plan['jobs'][0]
        self.commit_record(job)
        def fake_sheet(video, dest, cfg, title):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b'fixture sheet')
        with patch.object(review, 'make_sheet', side_effect=fake_sheet) as writer:
            review.review(self.root, 'pilot')
            run = self.root / 'pilot'
            mapping = (run / 'review_frames/private_mapping.json').read_bytes()
            (run / 'annotations.json').write_text('{"preserve":"reviewer draft"}')
            review.review(self.root, 'pilot')
            self.assertEqual(writer.call_count, 1)
            self.assertEqual((run / 'review_frames/private_mapping.json').read_bytes(), mapping)
            self.assertEqual(read_json(run / 'annotations.json'), {'preserve': 'reviewer draft'})
            public = read_json(run / 'review_frames/blind_manifest.json')['items'][0]
            self.assertNotIn('condition', public)
            self.assertNotIn('seed', public)


class RealCpuPipelineTests(unittest.TestCase):
    def test_same_pipeline_accepts_two_lengths_and_reproducible_noise(self):
        import numpy as np
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
        from video_count_probe.wan22_action_probe.vae_compat import install
        torch.set_num_threads(2)
        transformer = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=12, in_channels=4,
                                            out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
                                            num_layers=1, rope_max_seq_len=64).eval()
        vae = AutoencoderKLWan(base_dim=4, decoder_base_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1],
                               num_res_blocks=1, patch_size=2, in_channels=12, out_channels=12, is_residual=True,
                               scale_factor_temporal=4, scale_factor_spatial=16,
                               latents_mean=[0.] * 4, latents_std=[1.] * 4).eval()
        install(vae)
        pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, transformer=transformer,
                           scheduler=UniPCMultistepScheduler(flow_shift=5, prediction_type='flow_prediction',
                                                           use_flow_sigmas=True), expand_timesteps=True)
        pipe.set_progress_bar_config(disable=True)
        positive, negative = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
        short = {'height': 64, 'width': 64, 'frames': 5, 'steps': 2, 'guidance_scale': 5.0}
        hashes, shapes = [], []
        for frames in (5, 9, 5):
            cfg = {**short, 'frames': frames}
            generator, latents, sha = runtime.initial_latents(pipe, cfg, 2026)
            output = runtime.infer(pipe, cfg, positive, negative, latents, generator, 'cpu')
            self.assertEqual(output.shape, (frames, 64, 64, 3))
            self.assertTrue(np.isfinite(output).all())
            hashes.append(sha)
            shapes.append(tuple(latents.shape))
        self.assertEqual(hashes[0], hashes[2])
        self.assertNotEqual(hashes[0], hashes[1])
        self.assertNotEqual(shapes[0], shapes[1])
        runtime.verify_noise({'condition': 'b5', 'seed': 2026}, hashes[0], {2026: {'initial_latent_sha256': hashes[2]}})
        runtime.verify_noise({'condition': 'c8', 'seed': 2026}, hashes[1], {2026: {'initial_latent_sha256': hashes[0]}})
        self.assertTrue(all(p.grad is None for p in transformer.parameters()))


if __name__ == '__main__':
    unittest.main()
