"""Reuse /data model weights; copy only missing home weights once, low priority."""
import os
from pathlib import Path
import shutil
import subprocess
from . import common as c

def main():
    os.environ.update(c.environment())
    repair=Path('/data/zjr/trellis2_geometry_repair/models/ckpts')
    vae=Path('/data/zjr/models/microsoft_TRELLIS.2-4B_shape_vae/ckpts')
    hub=Path('/home/zjr/.cache/huggingface/hub')
    ss=hub/'models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts'
    mapping={'ss_flow':repair/'ss_flow_img_dit_1_3B_64_bf16','shape_flow':repair/'slat_flow_img2shape_dit_1_3B_512_bf16',
      'shape_encoder':vae/'shape_enc_next_dc_f16c32_fp16','shape_decoder':vae/'shape_dec_next_dc_f16c32_fp16',
      'ss_decoder':repair/'ss_dec_conv3d_16l8_fp16','ss_encoder':ss/'ss_enc_conv3d_16l8_fp16'}
    receipts=[]
    for key,base in list(mapping.items()):
        if str(base).startswith('/data/'):
            for ext in ['.json','.safetensors']:
                p=Path(str(base)+ext)
                if not p.is_file():raise FileNotFoundError(p)
            continue
        dest=c.ROOT/'models'/base.name
        for ext in ['.json','.safetensors']:
            src=Path(str(base)+ext);dst=Path(str(dest)+ext);dst.parent.mkdir(parents=True,exist_ok=True)
            receipt=dst.with_suffix(dst.suffix+'.storage.json')
            if not receipt.exists():
                tmp=dst.with_suffix(dst.suffix+'.part')
                subprocess.run(['ionice','-c','3','cp',str(src),str(tmp)],check=True)
                source_hash=c.sha(src);target_hash=c.sha(tmp)
                if source_hash!=target_hash:raise ValueError('Copy mismatch')
                tmp.replace(dst);c.write(receipt,{'source':str(src),'target':str(dst),'sha256':target_hash,'bytes':dst.stat().st_size})
            receipts.append(c.read(receipt))
        mapping[key]=dest
    dino_src=hub/'models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881'
    dino=c.ROOT/'models/dinov3'
    dino.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','model.safetensors']:
        src=dino_src/name;dst=dino/name;receipt=dino/(name+'.storage.json')
        if not receipt.exists():
            temp=dst.with_suffix('.part');subprocess.run(['ionice','-c','3','cp',str(src),str(temp)],check=True)
            h=c.sha(src)
            if c.sha(temp)!=h:raise ValueError('DINO copy mismatch')
            temp.replace(dst);c.write(receipt,{'source':str(src),'target':str(dst),'sha256':h,'bytes':dst.stat().st_size})
    mapping['dino']=dino
    c.immutable(c.OUT/'models.json',{k:str(v) for k,v in mapping.items()})
    # Isolate dependencies instead of modifying either existing conda environment.
    if not (c.ROOT/'packages/peft').exists():
        subprocess.run([c.PYTHON,'-m','pip','install','--no-deps','--target',str(c.ROOT/'packages'),
                        'peft==0.18.1','accelerate==1.12.0'],check=True)
    import peft,accelerate,torch,o_voxel,cumesh,flex_gemm
    c.write(c.OUT/'environment.json',{'time':c.now(),'peft':peft.__version__,
        'accelerate':accelerate.__version__,'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'model_paths':{k:str(v) for k,v in mapping.items()},'storage_receipts':receipts})

if __name__=='__main__': main()
