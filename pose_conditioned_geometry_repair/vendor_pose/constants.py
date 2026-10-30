"""Frozen TRELLIS.2 geometry-stage contracts used by this project."""

from __future__ import annotations

TRELLIS2_REPOSITORY = "microsoft/TRELLIS.2-4B"
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

SS_FLOW_MODEL = f"{TRELLIS2_REPOSITORY}/ckpts/ss_flow_img_dit_1_3B_64_bf16"
SS_DECODER_MODEL = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
SHAPE_FLOW_MODEL = (
    f"{TRELLIS2_REPOSITORY}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
)
SHAPE_ENCODER_MODEL = (
    f"{TRELLIS2_REPOSITORY}/ckpts/shape_enc_next_dc_f16c32_fp16"
)
SHAPE_DECODER_MODEL = (
    f"{TRELLIS2_REPOSITORY}/ckpts/shape_dec_next_dc_f16c32_fp16"
)

DINO_IMAGE_SIZE = 512
DINO_PATCH_SIZE = 16
DINO_PATCH_SIDE = DINO_IMAGE_SIZE // DINO_PATCH_SIZE
DINO_PATCH_TOKENS = DINO_PATCH_SIDE**2
DINO_PREFIX_TOKENS = 5
DINO_FULL_TOKENS = DINO_PREFIX_TOKENS + DINO_PATCH_TOKENS
DINO_FEATURE_DIM = 1024

SS_LATENT_RESOLUTION = 16
SS_LATENT_CHANNELS = 8
SS_DECODED_RESOLUTION = 64

SHAPE_LATENT_RESOLUTION = 32
SHAPE_LATENT_CHANNELS = 32
SHAPE_OUTPUT_RESOLUTION = 512
# Released Shape-512 training configuration filters (rather than truncates)
# objects above this active-token count.
SHAPE_MAX_TOKENS = 8192

FLOW_MODEL_CHANNELS = 1536
FLOW_BLOCKS = 30

SIGMA_MIN = 1.0e-5

# Official TRELLIS.2 Shape-SLat 512 normalization from pipeline.json.  Keeping
# it here makes training/cache checks independent of a pipeline object that
# would otherwise try to load the unavailable optional 1024-stage weights.
SHAPE_LATENT_MEAN = (
    0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252,
    1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944,
    -3.186688, 1.578665,
)
SHAPE_LATENT_STD = (
    5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237,
    5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
    5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
    5.347008, 5.405691,
)

SS_SAMPLER_DEFAULTS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.7,
    "guidance_interval": [0.6, 1.0],
    "rescale_t": 5.0,
}
SHAPE_SAMPLER_DEFAULTS = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.5,
    "guidance_interval": [0.6, 1.0],
    "rescale_t": 3.0,
}

CACHE_FORMAT = "pose_constrained_trellis2.cache.v1"
SAMPLE_FORMAT = "pose_constrained_trellis2.sample.v1"
CHECKPOINT_FORMAT = "pose_constrained_trellis2.adapter_checkpoint.v1"
