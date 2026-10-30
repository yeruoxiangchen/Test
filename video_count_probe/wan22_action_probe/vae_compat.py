"""Local fix for Diffusers 0.35.1 Wan2.2 tiled decode; never edits site-packages.

The bundled tiled_decode misses first_chunk for residual upsampling and returns
12 patch channels without unpatchify. Follow its existing tiling/blending, adding
the same first-chunk, unpatchify and clamp operations used by its untiled _decode.
"""
from types import MethodType


def tiled_decode(self, z, return_dict=True):
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
    from diffusers.models.autoencoders.vae import DecoderOutput
    _, _, frames, height, width = z.shape
    scale = self.spatial_compression_ratio  # Internal patch-space scale (8), not RGB scale (16).
    tile_h, tile_w = self.tile_sample_min_height // scale, self.tile_sample_min_width // scale
    stride_h, stride_w = self.tile_sample_stride_height // scale, self.tile_sample_stride_width // scale
    blend_h = self.tile_sample_min_height - self.tile_sample_stride_height
    blend_w = self.tile_sample_min_width - self.tile_sample_stride_width
    rows = []
    try:
        for i in range(0, height, stride_h):
            row = []
            for j in range(0, width, stride_w):
                self.clear_cache()
                temporal = []
                for k in range(frames):
                    self._conv_idx = [0]
                    tile = self.post_quant_conv(z[:, :, k:k + 1, i:i + tile_h, j:j + tile_w])
                    temporal.append(self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx,
                                                 first_chunk=k == 0))
                row.append(torch.cat(temporal, dim=2))
            rows.append(row)
    finally:
        self.clear_cache()
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_h)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_w)
            result_row.append(tile[:, :, :, :self.tile_sample_stride_height, :self.tile_sample_stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))
    output = torch.cat(result_rows, dim=3)[:, :, :, :height * scale, :width * scale]
    output = unpatchify(output, self.config.patch_size).clamp(-1., 1.)
    return DecoderOutput(sample=output) if return_dict else (output,)


def install(vae):
    if vae.config.patch_size != 2 or not vae.config.is_residual:
        raise ValueError('This compatibility fix is only for the Wan2.2 residual patch VAE')
    vae.tiled_decode = MethodType(tiled_decode, vae)
    return vae
