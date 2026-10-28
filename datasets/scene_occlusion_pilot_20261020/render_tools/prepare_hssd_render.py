"""Preserve GLB geometry and replace Basis KTX2 images with decoded PNGs."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HSSD=ROOT/'HSSD-hab'
OUT=ROOT/'renders'/'hssd_102344049'

def read_glb(path):
    raw=path.read_bytes()
    assert raw[:4]==b'glTF'
    n,t=struct.unpack_from('<II',raw,12)
    assert t==0x4e4f534a
    data=json.loads(raw[20:20+n])
    nb,tb=struct.unpack_from('<II',raw,20+n)
    assert tb==0x004e4942
    return data,raw[28+n:28+n+nb]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'download_manifest.json').read_text())
    scene=next(s for s in manifest['hssd_scenes'] if '102344049.' in s['scene'])
    assets=[]
    for cfgpath in scene['object_configs']+[scene['stage_config']]:
        cfgfile=HSSD/cfgpath
        cfg=json.loads(cfgfile.read_text())
        assets.append((cfgfile.parent/cfg['render_asset']).resolve())
    assets=sorted(set(assets))
    cache=OUT/'texture_cache';cache.mkdir(exist_ok=True)
    jobs={}
    for asset in assets:
        data,binary=read_glb(asset)
        for image in data.get('images',[]):
            if image.get('mimeType')!='image/ktx2':continue
            view=data['bufferViews'][image['bufferView']]
            start=view.get('byteOffset',0);payload=binary[start:start+view['byteLength']]
            digest=hashlib.sha256(payload).hexdigest()
            src=cache/(digest+'.ktx2');dst=cache/(digest+'.png')
            if not src.exists():src.write_bytes(payload)
            jobs[digest]={'input':str(src),'output':str(dst)}
    job_path=OUT/'texture_jobs.json';job_path.write_text(json.dumps(list(jobs.values()),indent=2))
    print(json.dumps({'assets':len(assets),'compressed_textures':len(jobs)}),flush=True)
    subprocess.run(['node',str(ROOT/'render_tools/decode_ktx2.cjs'),str(job_path)],check=True)
    records=[]
    for asset in assets:
        data,binary=read_glb(asset);blob=bytearray(binary);converted=0
        for image in data.get('images',[]):
            if image.get('mimeType')!='image/ktx2':continue
            view=data['bufferViews'][image['bufferView']];start=view.get('byteOffset',0)
            digest=hashlib.sha256(binary[start:start+view['byteLength']]).hexdigest()
            payload=(cache/(digest+'.png')).read_bytes()
            blob.extend(b'\0'*((-len(blob))%4))
            image['bufferView']=len(data['bufferViews']);image['mimeType']='image/png'
            data['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(payload)})
            blob.extend(payload);converted+=1
        for tex in data.get('textures',[]):
            ext=tex.get('extensions',{})
            if 'KHR_texture_basisu' in ext:
                tex['source']=ext.pop('KHR_texture_basisu')['source']
            if not ext:tex.pop('extensions',None)
        for key in ['extensionsUsed','extensionsRequired']:
            if key in data:
                data[key]=[x for x in data[key] if x!='KHR_texture_basisu']
                if not data[key]:data.pop(key)
        blob.extend(b'\0'*((-len(blob))%4));data['buffers'][0]['byteLength']=len(blob)
        js=json.dumps(data,separators=(',',':')).encode();js+=b' '*((-len(js))%4)
        packed=struct.pack('<4sII',b'glTF',2,28+len(js)+len(blob))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(blob),0x004e4942)+blob
        destination=OUT/'converted_glb'/asset.relative_to(HSSD)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(packed)
        records.append({'original':str(asset),'converted':str(destination),'png_images':converted,
                        'original_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
                        'converted_sha256':hashlib.sha256(packed).hexdigest()})
    (OUT/'conversion_manifest.json').write_text(json.dumps({'scene':'102344049','geometry_modified':False,'mipmap':'Decoded original level 0 to RGBA32 PNG','assets':records},indent=2)+'\n')
    print(json.dumps({'converted_assets':len(records),'output':str(OUT)}),flush=True)

if __name__=='__main__':main()
