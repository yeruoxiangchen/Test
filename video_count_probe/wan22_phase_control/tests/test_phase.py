import unittest
from unittest.mock import patch

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

from video_count_probe.core import atomic_json, token_groups
from video_count_probe.wan22_count_response.tests import test_response as fixtures
from video_count_probe.wan22_phase_control import protocol, review, runtime
from video_count_probe.wan22_phase_control.routing import PhaseRouting, phase_bias


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(817)
        return WanTransformer3DModel(num_attention_heads=2, attention_head_dim=12, in_channels=4,
                                    out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
                                    num_layers=1, rope_max_seq_len=64).eval()

    def test_seconds_short_phase_boundary_and_global_tokens(self):
        bias = phase_bias(49, 8, [{'start': 0., 'end': .5, 'tokens': [2]},
                                {'start': .5, 'end': 1.75, 'tokens': [3]}])
        self.assertTrue(torch.equal(bias[:, 0], torch.zeros(49)))
        self.assertEqual(bias[3, 2].item(), 0)
        self.assertEqual(bias[3, 3].item(), 0)
        self.assertAlmostEqual(bias[6, 2].item(), -2.)
        self.assertEqual(bias[-1, 2].item(), -8.)
        with self.assertRaises(ValueError):
            phase_bias(49, 8, [{'start': 0, 'end': 1, 'tokens': [2]}, {'start': 1, 'end': 2, 'tokens': [2]}])

    def test_repeated_text_offsets_are_distinct(self):
        offsets = [(0, 0), (0, 4), (5, 9), (10, 14), (0, 0)]
        groups = token_groups(offsets, [{'start': 0, 'end': 1, 'char_start': 0, 'char_end': 4},
                                       {'start': 1, 'end': 2, 'char_start': 10, 'char_end': 14}])
        self.assertEqual([g['tokens'] for g in groups], [[1], [3]])

    def test_dense_equivalence_and_conditional_only(self):
        model = self.model()
        attn = model.blocks[0].attn2
        old = attn.processor
        x, text = torch.randn(1, 20, 24), torch.randn(1, 8, 24)
        bias = phase_bias(5, 8, [{'start': 0, 'end': .2, 'tokens': [2, 3]}])
        with torch.no_grad():
            native = attn(x, text)
            q = attn.norm_q(attn.to_q(x)).reshape(1, 20, 2, 12).transpose(1, 2)
            k = attn.norm_k(attn.to_k(text)).reshape(1, 8, 2, 12).transpose(1, 2)
            v = attn.to_v(text).reshape(1, 8, 2, 12).transpose(1, 2)
            logits = q @ k.transpose(-1, -2) / 12**.5 + bias.repeat_interleave(4, 0)[None, None]
            expected = (logits.softmax(-1) @ v).transpose(1, 2).flatten(2)
            expected = attn.to_out[1](attn.to_out[0](expected))
            for enabled in (False, True):
                with PhaseRouting(model, bias, 4, enabled, chunk_size=7) as route:
                    with model.cache_context('uncond'):
                        torch.testing.assert_close(attn(x, text), native, atol=0, rtol=0)
                    with model.cache_context('cond'):
                        actual = attn(x, text)
                    torch.testing.assert_close(actual, expected if enabled else native, atol=1e-6, rtol=1e-5)
                    route.validate_calls(1)
                if enabled:
                    self.assertGreater((actual - native).abs().max().item(), 1e-5)
        self.assertIs(attn.processor, old)
        self.assertNotIn('cache_context', model.__dict__)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with PhaseRouting(model, bias, 4, True):
                raise RuntimeError('intentional')
        self.assertIs(attn.processor, old)

    def test_real_wan22_pipeline_expanded_timesteps_and_cfg(self):
        vae = AutoencoderKLWan(base_dim=4, decoder_base_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1],
                              num_res_blocks=1, patch_size=2, in_channels=12, out_channels=12, is_residual=True,
                              scale_factor_temporal=4, scale_factor_spatial=16,
                              latents_mean=[0.] * 4, latents_std=[1.] * 4).eval()
        pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, transformer=self.model(),
                           scheduler=UniPCMultistepScheduler(flow_shift=5, prediction_type='flow_prediction',
                                                           use_flow_sigmas=True), expand_timesteps=True)
        pipe.set_progress_bar_config(disable=True)
        bias = phase_bias(2, 8, [{'start': 0., 'end': .08, 'tokens': [2]}])
        positive, negative = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
        outputs = []
        for enabled in (False, True):
            with PhaseRouting(pipe.transformer, bias, 4, enabled, chunk_size=3) as route:
                result = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                              height=64, width=64, num_frames=5, num_inference_steps=2, guidance_scale=5,
                              generator=torch.Generator().manual_seed(42), output_type='latent').frames
            route.validate_calls(2)
            self.assertEqual(tuple(result.shape), (1, 4, 2, 4, 4))
            self.assertTrue(torch.isfinite(result).all())
            outputs.append(result)
        self.assertGreater((outputs[0] - outputs[1]).abs().max().item(), 1e-6)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ResponseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan = protocol.build_plan('pilot', self.fixture.reference)
        atomic_json(self.root / 'pilot/plan.json', self.plan)
        self.fixture.plan = self.plan
        self.fixture.identity['plan_hash'] = self.plan['plan_hash']

    def commit(self, job, plan=None):
        record = self.fixture.commit(job, plan)
        record['routing_stats'] = {'enabled': job['condition'] == 'routed',
                                   'conditional_calls': record['config']['steps'] * 30,
                                   'unconditional_calls': record['config']['steps'] * 30}
        record.update(positive_cache_key='same positive', negative_cache_key='same negative', routing_audit_sha256='same bias')
        atomic_json(self.root / (plan or self.plan)['mode'] / 'records' / (job['id'] + '.json'), record)
        return record

    def test_identical_conditions_six_pilot_two_smoke(self):
        a, b = self.plan['conditions'].values()
        self.assertEqual(a['prompt'], b['prompt'])
        self.assertEqual(a['config'], b['config'])
        self.assertEqual(len(self.plan['jobs']), 6)
        self.assertEqual(len(protocol.build_plan('smoke', self.fixture.reference)['jobs']), 2)
        self.assertFalse(self.plan['reused_conditions'])

    def test_pair_noise_cache_binding_and_missing_labels_unknown(self):
        for job in self.plan['jobs'][:2]:
            record = self.commit(job)
        result = review.report(self.root, write=False)
        self.assertEqual(result['generated_new'], 2)
        self.assertEqual(result['reviewed_new'], 0)
        self.assertFalse(result['assessment_complete'])
        self.assertEqual(result['conditions']['routed']['exact_count']['false'], 0)
        record['positive_cache_key'] = 'different'
        atomic_json(self.root / 'pilot/records' / (record['job']['id'] + '.json'), record)
        with self.assertRaisesRegex(ValueError, 'Paired input mismatch'):
            review.report(self.root, write=False)

    def test_both_smoke_arms_required(self):
        smoke = protocol.build_plan('smoke', self.fixture.reference)
        atomic_json(self.root / 'smoke/plan.json', smoke)
        self.commit(smoke['jobs'][0], smoke)
        with self.assertRaisesRegex(RuntimeError, 'Both'):
            runtime.require_smoke(self.root, self.plan, self.fixture.identity, set(protocol.CONDITIONS), None)
        self.commit(smoke['jobs'][1], smoke)
        runtime.require_smoke(self.root, self.plan, self.fixture.identity, set(protocol.CONDITIONS), None)


if __name__ == '__main__':
    unittest.main()
