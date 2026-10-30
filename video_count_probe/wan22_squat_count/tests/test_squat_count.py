import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, writer_lock
from video_count_probe.wan22_action_probe.protocol import MODEL_ID, NEGATIVE
from video_count_probe.wan22_squat_count import protocol, review, runtime


def reference_fixture(root):
    config = {'model_id': MODEL_ID, 'height': 704, 'width': 1280, 'frames': 121, 'fps': 24,
              'steps': 50, 'guidance_scale': 5.0, 'flow_shift': 5.0, 'input_image': None,
              'prompt_rewrite': False, 'negative_prompt': NEGATIVE}
    plan = {'config': config}
    plan['plan_hash'] = digest(plan)
    atomic_json(root / 'plan.json', plan)
    atomic_json(root / 'ai_review_20261028/summary_ai.json', {
        'plan_hash': plan['plan_hash'], 'ai_review_complete': True,
        'per_video': [{'job_id': f'squat-{i}', 'action': 'squat', 'complete_cycle': True} for i in (1, 2)]})
    return root


def run_fixture(root, completed_count=2, mode='pilot'):
    ref = reference_fixture(root / 'reference')
    plan = protocol.build_plan(mode, ref)
    run = root / mode
    atomic_json(run / 'plan.json', plan)
    identity = {'plan_hash': plan['plan_hash'], 'packages': {'diffusers': '0.35.1'}}
    atomic_json(run / 'identity.json', identity)
    records = {}
    for job in plan['jobs'][:completed_count]:
        video = run / 'videos' / (job['id'] + '.mp4')
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b'test-content-' + job['id'].encode())
        record = {'status': 'ok', 'identity': identity, 'job': job,
                  'video': str(video.relative_to(run)), 'video_sha256': file_sha(video),
                  'initial_latent_sha256': 'same-initial-noise', 'peak_allocated_mib': 10, 'peak_reserved_mib': 12}
        atomic_json(run / 'records' / (job['id'] + '.json'), record)
        records[job['id']] = record
    return run, plan, identity, records


def label(record):
    return {'job_id': record['job']['id'], 'video_sha256': record['video_sha256'], 'reviewed': True,
            'count_judgeable': True, 'complete_cycles': [{'start_frame': 0, 'bottom_frame': 24, 'end_frame': 48}],
            'partial_attempts': [], 'initial_upright': True, 'final_second_upright_still': True,
            'notes': 'Standing at frame 0, lowered hips at 24, upright at 48; rests thereafter.'}


class ProtocolTests(unittest.TestCase):
    def test_twelve_paired_jobs_fresh_seeds_and_balanced_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = reference_fixture(Path(tmp) / 'ref')
            pilot, smoke = (protocol.build_plan(mode, ref) for mode in ('pilot', 'smoke'))
        self.assertEqual(len(pilot['jobs']), 12)
        self.assertEqual(len({j['id'] for j in pilot['jobs']}), 12)
        self.assertFalse({j['seed'] for j in pilot['jobs']} & {42, 123})
        self.assertEqual(len(smoke['jobs']), 2)
        first_methods = []
        for start in range(0, 12, 2):
            a, b = pilot['jobs'][start:start + 2]
            self.assertEqual((a['seed'], a['case_id'], a['pair_id']), (b['seed'], b['case_id'], b['pair_id']))
            self.assertNotEqual(a['method'], b['method'])
            first_methods.append(a['method'])
        self.assertEqual(first_methods.count('count_prompt'), 3)
        self.assertEqual(first_methods.count('phase_prompt'), 3)
        for key in ('height', 'width', 'frames', 'fps'):
            self.assertEqual(pilot['config'][key], smoke['config'][key])
        self.assertEqual((pilot['config']['steps'], smoke['config']['steps']), (50, 2))
        self.assertEqual(len(protocol.all_prompts(pilot)), 5)
        for case in pilot['cases']:
            for prompt in case['prompts'].values():
                self.assertTrue(prompt.startswith(protocol.COMMON))
                self.assertTrue(prompt.endswith(protocol.REST))

    def test_frozen_plan_and_changed_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = reference_fixture(root / 'ref')
            protocol.prepare(root / 'run', ref)
            p = root / 'run/pilot/plan.json'
            plan = checked_plan(p)
            plan['jobs'][0]['seed'] = 999
            atomic_json(p, plan)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                checked_plan(p)
            with self.assertRaisesRegex(RuntimeError, 'Frozen plan'):
                protocol.prepare(root / 'run', ref)
            old = protocol.build_plan('pilot', ref)
            review_path = ref / 'ai_review_20261028/summary_ai.json'
            data = json.loads(review_path.read_text())
            data['revised_notes'] = 'Different review revision'
            atomic_json(review_path, data)
            with self.assertRaisesRegex(ValueError, 'reference changed'):
                protocol.verify_reference(old)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'scoring': {'rest_start_frame': 96, 'last_frame': 120}}
        self.record = {'job': {'id': 'test'}, 'video_sha256': 'digest'}
        self.row = label(self.record)

    def test_correct_count_with_extra_half_squat_fails_stopping(self):
        self.row['partial_attempts'] = [{'start_frame': 60, 'end_frame': 84}]
        review.validate_annotation(self.row, self.record, self.plan)
        scores = review.score(self.row, 1, self.plan)
        self.assertTrue(scores['exact_count'])
        self.assertTrue(scores['final_rest'])
        self.assertTrue(scores['extra_partial_after_target'])
        self.assertFalse(scores['stopped_after_target'])
        self.assertFalse(scores['joint_count_stop'])

    def test_two_full_cycles_are_not_one_and_late_completion_fails_rest_deadline(self):
        self.row['complete_cycles'].append({'start_frame': 48, 'bottom_frame': 80, 'end_frame': 100})
        review.validate_annotation(self.row, self.record, self.plan)
        one = review.score(self.row, 1, self.plan)
        two = review.score(self.row, 2, self.plan)
        self.assertFalse(one['exact_count'])
        self.assertFalse(one['stopped_after_target'])
        self.assertTrue(two['exact_count'])
        self.assertTrue(two['stopped_after_target'])
        self.assertFalse(two['target_completed_by_rest'])
        self.assertFalse(two['joint_count_stop'])

    def test_unknown_is_not_zero_and_missing_partial_review_is_not_a_pass(self):
        self.row.update(count_judgeable=False, complete_cycles=None, partial_attempts=None,
                        final_second_upright_still=None)
        review.validate_annotation(self.row, self.record, self.plan)
        scores = review.score(self.row, 1, self.plan)
        self.assertIsNone(scores['observed_complete_count'])
        self.assertIsNone(scores['exact_count'])
        self.assertIsNone(scores['joint_count_stop'])
        self.row = label(self.record)
        self.row['partial_attempts'] = None
        self.assertIsNone(review.score(self.row, 1, self.plan)['joint_count_stop'])

    def test_phase_order_overlap_and_hash_are_validated(self):
        bad = copy.deepcopy(self.row)
        bad['complete_cycles'][0]['bottom_frame'] = 49
        with self.assertRaisesRegex(ValueError, 'ordered'):
            review.validate_annotation(bad, self.record, self.plan)
        bad = copy.deepcopy(self.row)
        bad['partial_attempts'] = [{'start_frame': 20, 'end_frame': 30}]
        with self.assertRaisesRegex(ValueError, 'cannot overlap'):
            review.validate_annotation(bad, self.record, self.plan)
        bad = {**self.row, 'video_sha256': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            review.validate_annotation(bad, self.record, self.plan)

    def test_valid_stop_and_boundary_frame(self):
        self.row['complete_cycles'][0]['end_frame'] = 96
        review.validate_annotation(self.row, self.record, self.plan)
        self.assertTrue(review.score(self.row, 1, self.plan)['joint_count_stop'])
        self.row['complete_cycles'] = []
        scores = review.score(self.row, 1, self.plan)
        self.assertEqual(scores['observed_complete_count'], 0)
        self.assertFalse(scores['joint_count_stop'])


class ReportingAndResumeTests(unittest.TestCase):
    def test_partial_pairs_unknowns_and_noise_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, plan, _, records = run_fixture(root, completed_count=2)
            jobs = plan['jobs'][:2]
            positive = label(records[jobs[0]['id']])
            uncertain = label(records[jobs[1]['id']])
            uncertain.update(count_judgeable=False, complete_cycles=None, partial_attempts=None,
                             final_second_upright_still=None, notes='Occluded knee; cannot count reliably.')
            data = {'plan_hash': plan['plan_hash'], 'rater': 'AI test reviewer', 'rater_type': 'ai',
                    'items': [positive, uncertain]}
            atomic_json(run / 'annotations.json', data)
            result = review.report(root, 'pilot', write=False)
            self.assertEqual((result['reviewed'], result['decided'], result['reviewed_uncertain']), (2, 1, 1))
            self.assertEqual(result['paired']['joint_count_stop']['decided_pairs'], 0)
            self.assertFalse(result['assessment_complete'])
            self.assertEqual(result['review_provenance']['rater_type'], 'ai')
            self.assertFalse((run / 'summary.json').exists())
            record = records[jobs[1]['id']]
            record['initial_latent_sha256'] = 'different'
            atomic_json(run / 'records' / (jobs[1]['id'] + '.json'), record)
            with self.assertRaisesRegex(ValueError, 'Paired initial latent'):
                review.report(root, 'pilot', write=False)

    def test_complete_resume_loads_no_gpu_or_embeddings_and_corruption_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _, identity, records = run_fixture(root, completed_count=2, mode='smoke')
            with patch.object(runtime, 'current_identity', return_value=identity), \
                 patch.object(runtime, 'gpu_preflight') as gpu, patch.object(runtime, 'text_identity') as text, \
                 patch.object(runtime, 'load_pipe') as load:
                runtime.generate(root, 'smoke', '/unused')
                gpu.assert_not_called()
                text.assert_not_called()
                load.assert_not_called()
            first = next(iter(records.values()))
            (run / first['video']).write_bytes(b'corrupt')
            # The damaged job becomes pending and reaches cache preflight, instead of silently skipping.
            with patch.object(runtime, 'current_identity', return_value=identity), \
                 patch.object(runtime, 'text_identity', side_effect=RuntimeError('pending requires cache')):
                with self.assertRaisesRegex(RuntimeError, 'pending requires cache'):
                    runtime.generate(root, 'smoke', '/unused')

    def test_duplicate_writer_and_failure_require_explicit_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, plan, identity, _ = run_fixture(root, completed_count=0, mode='smoke')
            with patch.object(runtime, 'current_identity', return_value=identity):
                with writer_lock(run / '.writer.lock'):
                    with self.assertRaisesRegex(RuntimeError, 'Another writer'):
                        runtime.generate(root, 'smoke', '/unused')
                error = run / 'errors' / (plan['jobs'][0]['id'] + '.json')
                atomic_json(error, {'state': 'failed'})
                with self.assertRaisesRegex(RuntimeError, 'Prior failure'):
                    runtime.generate(root, 'smoke', '/unused')
                atomic_json(error, {'state': 'interrupted'})
                with patch.object(runtime, 'text_identity', side_effect=RuntimeError('resuming interrupted')):
                    with self.assertRaisesRegex(RuntimeError, 'resuming interrupted'):
                        runtime.generate(root, 'smoke', '/unused')

    def test_review_keeps_existing_labels_and_stable_blind_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _, _, _ = run_fixture(root, completed_count=2)
            labels = run / 'annotations.json'
            labels.write_text('{"do_not_replace": true}\n')
            before = labels.read_bytes()

            def fake_sheet(_video, destination, _config, _title):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b'evidence')

            with patch.object(review, 'make_sheet', side_effect=fake_sheet) as sheet:
                review.review(root, 'pilot')
                mapping = (run / 'review_frames/private_mapping.json').read_bytes()
                review.review(root, 'pilot')
                self.assertEqual(sheet.call_count, 2)  # Cached overview pages are not decoded again.
            self.assertEqual(labels.read_bytes(), before)
            self.assertEqual((run / 'review_frames/private_mapping.json').read_bytes(), mapping)
            public = json.loads((run / 'review_frames/blind_manifest.json').read_text())
            self.assertTrue(all('target_count' not in r and 'method' not in r and 'seed' not in r for r in public['items']))


class RealCpuPipelineTests(unittest.TestCase):
    def test_real_wan_paired_noise_and_inference_path(self):
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
        config = {'height': 64, 'width': 64, 'frames': 5, 'steps': 2, 'guidance_scale': 5.0}
        a_gen, a_latent, a_hash = runtime.initial_latents(pipe, config, 2026)
        b_gen, b_latent, b_hash = runtime.initial_latents(pipe, config, 2026)
        _, _, other_hash = runtime.initial_latents(pipe, config, 3407)
        self.assertEqual(a_hash, b_hash)
        self.assertNotEqual(a_hash, other_hash)
        torch.testing.assert_close(a_latent, b_latent, rtol=0, atol=0)
        positive, negative = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
        first = runtime.infer(pipe, config, positive, negative, a_latent, a_gen, 'cpu')
        second = runtime.infer(pipe, config, positive, negative, b_latent, b_gen, 'cpu')
        self.assertEqual(first.shape, (5, 64, 64, 3))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(p.grad is None for p in transformer.parameters()))
        runtime.verify_paired_noise({'pair_id': 'pair'}, a_hash,
                                    {'a': {'job': {'pair_id': 'pair'}, 'initial_latent_sha256': b_hash}})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            runtime.verify_paired_noise({'pair_id': 'pair'}, other_hash,
                                        {'a': {'job': {'pair_id': 'pair'}, 'initial_latent_sha256': a_hash}})


if __name__ == '__main__':
    unittest.main()
