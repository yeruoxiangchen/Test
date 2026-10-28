"""Native TRELLIS.2 loading and a parity-checked piecewise-smooth mesh path."""
import gc
from contextlib import contextmanager

import torch

from .config import MODELS, OUT, PARAMS, read, write, setup

NAMES=dict(ss_flow='ss_flow_img_dit_1_3B_64_bf16',ss_decoder='ss_dec_conv3d_16l8_fp16',
           shape_flow='slat_flow_img2shape_dit_1_3B_512_bf16',shape_decoder='shape_dec_next_dc_f16c32_fp16')


def load_model(role, device='cuda:0', checkpoint=False):
    setup()
    from trellis2 import models
    from safetensors.torch import load_file
    prefix=MODELS/'ckpts'/NAMES[role]
    config=read(str(prefix)+'.json')
    model=getattr(models,config['name'])(**config['args'])
    incompatible=model.load_state_dict(load_file(str(prefix)+'.safetensors'),strict=False)
    # Upstream creates RoPE phase buffers analytically, and its released
    # checkpoint omits them. Every learned parameter must still be present.
    buffers=dict(model.named_buffers())
    permitted={key for key in incompatible.missing_keys if key in buffers and key.endswith('rope_phases')}
    if set(incompatible.missing_keys)-permitted or incompatible.unexpected_keys:
        raise RuntimeError(f'checkpoint mismatch: {incompatible}')
    if role=='shape_decoder':
        model.set_resolution(PARAMS['resolution'])
    write(OUT/'bindings'/f'{role}.json',dict(model=str(prefix),derived_buffers=sorted(permitted),
                                          missing_learned_parameters=[],unexpected_keys=[],
                                          output_resolution=getattr(model,'resolution',None)))
    model.eval().requires_grad_(False)
    if checkpoint:
        from .compat import install_frozen_weight_backward
        install_frozen_weight_backward()
        for module in model.modules():
            if hasattr(module,'use_checkpoint'):module.use_checkpoint=True
    return model.to(device)


def release():
    gc.collect();torch.cuda.empty_cache()


def sparse(features,coords):
    from trellis2.modules.sparse import SparseTensor
    # A fresh cache for every call is essential: channel2spatial caches can
    # otherwise override newly predicted subdivisions or leak encoder support.
    return SparseTensor(feats=features,coords=coords.to(dtype=torch.int32))


def normalization(device):
    values=read(MODELS/'pipeline.json')['args']['shape_slat_normalization']
    return (torch.tensor(values['mean'],device=device)[None],torch.tensor(values['std'],device=device)[None])


def decode_native(decoder,features,coords):
    if decoder.resolution!=PARAMS['resolution']:raise RuntimeError('wrong Shape decoder resolution')
    if bool(((coords[:,1:]<0)|(coords[:,1:]>=PARAMS['resolution']//16)).any()):
        raise RuntimeError('Shape coordinates and output resolution differ')
    mesh=decoder(sparse(features.float(),coords))[0]
    if not len(mesh.faces) or not bool(torch.isfinite(mesh.vertices).all()):
        raise RuntimeError('invalid native mesh')
    return mesh.vertices,mesh.faces


def save_mesh(path,vertices,faces):
    import trimesh
    path.parent.mkdir(parents=True,exist_ok=True)
    trimesh.Trimesh(vertices.detach().float().cpu().numpy(),faces.detach().long().cpu().numpy(),process=False).export(path,include_texture=False)


class FixedBranch:
    """Freeze native discrete subdivisions/faces; retain exact vertex gradients.

    Topology decisions are sampled from this same generated latent, never GT.
    Native decoded proposals must be rechecked before delivery.
    """
    def __init__(self,decoder,features,coords):
        self.decoder,self.coords=decoder,coords
        self.guides={};self.handles=[]
        def capture(name):
            def hook(module,args,output):
                self.guides[name]=(output.coords.detach().clone(),output.feats.detach().clone())
            return hook
        for name,module in decoder.named_modules():
            if name.endswith('to_subdiv'):self.handles.append(module.register_forward_hook(capture(name)))
        try:
            with torch.no_grad():self.base_vertices,self.faces=decode_native(decoder,features,coords)
        finally:
            for handle in self.handles:handle.remove()
            self.handles=[]
        self.base_vertices=self.base_vertices.detach().clone();self.faces=self.faces.detach().clone()
        self.raw_coords=None
        self.enabled=True
        def replace(name):
            def hook(module,args,output):
                if not self.enabled:return output
                coords,features=self.guides[name]
                if not torch.equal(output.coords,coords):raise RuntimeError('frozen branch coordinates changed')
                return output.replace(features.to(dtype=output.feats.dtype))
            return hook
        # Keep hooks alive through checkpoint backward recomputation.
        for name,module in decoder.named_modules():
            if name in self.guides:self.handles.append(module.register_forward_hook(replace(name)))

    @contextmanager
    def native_mode(self):
        previous=self.enabled;self.enabled=False
        try:yield
        finally:self.enabled=previous

    def native(self,features):
        with self.native_mode():return decode_native(self.decoder,features,self.coords)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def __call__(self,features):
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
        self.enabled=True
        h=SparseUnetVaeDecoder.forward(self.decoder,sparse(features.float(),self.coords))
        if self.raw_coords is None:self.raw_coords=h.coords.detach().clone()
        elif not torch.equal(self.raw_coords,h.coords):raise RuntimeError('frozen decoded support drift')
        margin=self.decoder.voxel_margin
        offsets=(1+2*margin)*torch.sigmoid(h.feats[:,:3])-margin
        vertices=(h.coords[:,1:].float()+offsets)*(1./self.decoder.resolution)-.5
        if len(vertices)!=len(self.base_vertices):raise RuntimeError('frozen native vertex count changed')
        return vertices,self.faces


class NativePath:
    """Actual native mesh at every forward, with piecewise vertex gradients."""
    def __init__(self,decoder,features,coords):
        self.decoder,self.coords=decoder,coords
        with torch.no_grad():self.base_vertices,self.faces=decode_native(decoder,features,coords)
    def __call__(self,features):return decode_native(self.decoder,features,self.coords)
    def native(self,features):return self(features)
    def close(self):pass
