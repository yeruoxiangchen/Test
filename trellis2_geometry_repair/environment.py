import importlib
import importlib.metadata
from .config import OUT,setup,write,now


def main():
    setup();modules={}
    for name in ('torch','transformers','nvdiffrast.torch','o_voxel','cumesh','flex_gemm','flash_attn','trimesh','scipy'):
        module=importlib.import_module(name);modules[name]=getattr(module,'__version__','imported')
    import torch
    x=torch.ones(1,device='cuda:0');x.add_(1);torch.cuda.synchronize()
    value=dict(time=now(),modules=modules,gpu=torch.cuda.get_device_name(),cuda_available=torch.cuda.is_available(),
               torch_cuda=torch.version.cuda,device_probe=float(x))
    write(OUT/'validation/environment.json',value);print(value,flush=True)


if __name__=='__main__':main()
