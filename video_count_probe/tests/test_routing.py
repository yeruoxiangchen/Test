"""Uses the actual Diffusers Wan modules with tiny random weights, no checkpoint/GPU."""
import unittest

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

from video_count_probe.routing import RelayRouting, temporal_bias


class WanRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(817)
        return WanTransformer3DModel(num_attention_heads=2, attention_head_dim=12, in_channels=4,
                                     out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
                                     num_layers=1, rope_max_seq_len=64).eval()

    def test_penalty_matches_published_equation(self):
        groups = [{"start": 0, "end": 7, "tokens": [2, 3]}, {"start": 7, "end": 14, "tokens": [5]}]
        b = temporal_bias(21, 8, groups)
        self.assertTrue(torch.equal(b[:, 0], torch.zeros(21)))  # global text stays visible
        self.assertEqual(b[3, 2].item(), 0)
        self.assertAlmostEqual(b[0, 2].item(), -(2**2)/(2*0.1448**2), places=4)
        self.assertLess(b[20, 5].item(), -100)

    def test_actual_wan_forward_conditional_only_and_restore(self):
        model = self.model()
        x, text, t = torch.randn(1, 4, 21, 4, 4), torch.randn(1, 8, 16), torch.tensor([400.])
        old = model.blocks[0].attn2.processor
        bias = temporal_bias(21, 8, [{"start": 0, "end": 7, "tokens": [2, 3]}])
        with torch.no_grad():
            baseline = model(x, t, text).sample
            with RelayRouting(model, bias, 4) as route:
                with model.cache_context("uncond"):
                    uncond = model(x, t, text).sample
                self.assertTrue(torch.equal(baseline, uncond))
                self.assertEqual(route.routed_calls, 0)
                with model.cache_context("cond"):
                    routed = model(x, t, text).sample
                self.assertEqual(route.routed_calls, 1)
                self.assertTrue(torch.isfinite(routed).all())
                self.assertGreater((routed-baseline).abs().max().item(), 1e-5)
            self.assertIs(model.blocks[0].attn2.processor, old)
            self.assertNotIn("cache_context", model.__dict__)

    def test_exception_restores_and_zero_strength_is_exact_identity(self):
        model = self.model()
        old = model.blocks[0].attn2.processor
        x, text, t = torch.randn(1, 4, 21, 4, 4), torch.randn(1, 8, 16), torch.tensor([400.])
        bias = temporal_bias(21, 8, [{"start": 0, "end": 7, "tokens": [2]}])
        with torch.no_grad():
            baseline = model(x, t, text).sample
            with RelayRouting(model, bias, 4, strength=0):
                with model.cache_context("cond"):
                    zero = model(x, t, text).sample
            self.assertTrue(torch.equal(baseline, zero))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with RelayRouting(model, bias, 4):
                raise RuntimeError("intentional")
        self.assertIs(model.blocks[0].attn2.processor, old)

    def test_chunked_attention_matches_independent_dense_reference(self):
        model = self.model()
        attn = model.blocks[0].attn2
        x, text = torch.randn(1, 84, 24), torch.randn(1, 8, 24)
        bias = temporal_bias(21, 8, [{"start": 0, "end": 7, "tokens": [2, 3]}])
        with torch.no_grad():
            q = attn.norm_q(attn.to_q(x)).reshape(1, 84, 2, 12).transpose(1, 2)
            k = attn.norm_k(attn.to_k(text)).reshape(1, 8, 2, 12).transpose(1, 2)
            v = attn.to_v(text).reshape(1, 8, 2, 12).transpose(1, 2)
            logits = q @ k.transpose(-1, -2) / (12**0.5) + bias.repeat_interleave(4, dim=0)[None, None]
            expected = (logits.softmax(-1) @ v).transpose(1, 2).flatten(2)
            expected = attn.to_out[1](attn.to_out[0](expected))
            with RelayRouting(model, bias, 4, chunk_size=7):
                with model.cache_context("cond"):
                    actual = attn(x, text)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_real_pipeline_cfg_and_denoising_loop(self):
        # Tiny real VAE supplies shape configuration. Decode is not exercised here.
        vae = AutoencoderKLWan(base_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1)
        pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, transformer=self.model(),
                           scheduler=UniPCMultistepScheduler(flow_shift=8.0, prediction_type="flow_prediction",
                                                           use_flow_sigmas=True, num_train_timesteps=1000))
        pipe.set_progress_bar_config(disable=True)
        bias = temporal_bias(21, 8, [{"start": 0, "end": 7, "tokens": [2]}])
        with RelayRouting(pipe.transformer, bias, 4) as route:
            result = pipe(prompt_embeds=torch.randn(1, 8, 16), negative_prompt_embeds=torch.randn(1, 8, 16),
                          height=32, width=32, num_frames=81, num_inference_steps=2,
                          guidance_scale=6, generator=torch.Generator().manual_seed(42), output_type="latent").frames
        self.assertEqual(route.routed_calls, 2)
        self.assertEqual(tuple(result.shape), (1, 4, 21, 4, 4))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
