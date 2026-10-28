"""Blender CPU assembly and shared indoor photograph plan, with no target search."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET = Path('/data/zjr/pose_aligned_layered_reconstruction/datasets/hssd_generalization_v1')
OUT = Path('/data/zjr/pose_aligned_layered_reconstruction/outputs/hssd_generalization_20261027')
spec = importlib.util.spec_from_file_location('_existing_scene_helpers', ROOT / 'datasets/scene_occlusion_pilot_20261020/render_tools/render_hssd_blender.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
C, CI = h.C, h.CI


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def binding(path): return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def transform(row):
    return (Matrix.Translation(Vector(row.get('translation',[0,0,0])))
            @ Quaternion(row.get('rotation',[1,0,0,0])).to_matrix().to_4x4()
            @ Matrix.Diagonal(Vector([*row.get('non_uniform_scale',[1,1,1]),1])))


def main(sid):
    doc = json.loads((OUT / 'candidate_plan.json').read_text()); p = doc['plan']
    dep = next(r for r in p['dependencies']['scenes'] if r['scene'] == sid)
    folder = DATASET / 'scenes' / sid; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'plan.json').exists():
        old = json.loads((folder / 'plan.json').read_text())
        if old['plan']['candidate_plan_identity'] != doc['identity']: raise ValueError('Changed candidate plan')
        print('SCENE_CAPTURE_REUSED ' + sid, flush=True); return
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine='CYCLES'; scene.cycles.device='CPU'
    scene.world = bpy.data.worlds.new('Shared capture daylight'); scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.78,.86,1,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.45
    original = json.loads((DATASET / 'raw' / dep['scene_file']).read_text())
    prototype = {}; cfgs = {Path(s).name.removesuffix('.object_config.json'): s for s in dep['configs']}
    stage = DATASET / 'converted_glb' / dep['render_assets'][dep['stage_config']]
    h.instance(h.load_prototype(stage),'Architecture',C @ transform(original['stage_instance']) @ CI,1)
    for i,row in enumerate(original['object_instances']):
        asset = row['template_name']; path = DATASET / 'converted_glb' / dep['render_assets'][cfgs[asset]]
        if asset not in prototype: prototype[asset] = h.load_prototype(path)
        h.instance(prototype[asset], f'Instance_{i:03d}_{asset[:8]}', C @ transform(row) @ CI, i+2)
        if (i+1)%50==0: print('SCENE_ASSEMBLY '+json.dumps({'scene':sid,'instances':i+1}),flush=True)
    for region in dep['capture_regions']:
        lo,hi = region['min_bounds'],region['max_bounds']; area=(hi[0]-lo[0])*(hi[2]-lo[2])
        light=bpy.data.lights.new('Room fill '+str(region.get('name','region')),'AREA')
        light.energy=max(150,area*22); light.shape='DISK'; light.size=min(3,max(.8,math.sqrt(area)*.5))
        lamp=bpy.data.objects.new(light.name,light); scene.collection.objects.link(lamp)
        lamp.location=C.to_3x3() @ Vector([(lo[0]+hi[0])/2,lo[1]+2.6,(lo[2]+hi[2])/2])
    sun=bpy.data.lights.new('Exterior daylight','SUN'); sun.energy=1.3; sun.angle=math.radians(15)
    lamp=bpy.data.objects.new(sun.name,sun); scene.collection.objects.link(lamp)
    lamp.rotation_euler=(math.radians(20),math.radians(-35),math.radians(25))
    bpy.context.view_layer.update(); depsgraph=bpy.context.evaluated_depsgraph_get()
    policy=p['capture_policy']; eyes=[]; matrices=[]; camera_receipts=[]
    for ri,region in enumerate(dep['capture_regions']):
        lo,hi=region['min_bounds'],region['max_bounds']; positions=[]
        for a,b in [(.2,.2),(.8,.8),(.2,.8),(.8,.2),(.5,.5),(.2,.5),(.8,.5),(.5,.2),(.5,.8)]:
            origin=Vector([lo[0]+a*(hi[0]-lo[0]),lo[1]+.3,lo[2]+b*(hi[2]-lo[2])])
            hit,location,_,_,obj,_=scene.ray_cast(depsgraph,C.to_3x3()@origin,C.to_3x3()@Vector([0,-1,0]),distance=.8)
            if not hit or obj.pass_index != 1: continue
            floor=CI.to_3x3()@location; eye=Vector([origin.x,floor.y+policy['height_m'],origin.z])
            clear=True
            for level in (.5,policy['height_m']):
                point=Vector([eye.x,floor.y+level,eye.z])
                for angle in range(0,360,45):
                    direction=Vector([math.cos(math.radians(angle)),0,math.sin(math.radians(angle))])
                    if scene.ray_cast(depsgraph,C.to_3x3()@point,C.to_3x3()@direction,distance=policy['clearance_m'])[0]:
                        clear=False;break
                if not clear:break
            if clear and all((eye-other[0]).length>=.2 for other in positions): positions.append((eye,floor.y))
            if len(positions)==policy['camera_positions_per_room']:break
        for pi,(eye,floor_y) in enumerate(positions):
            for yaw in policy['yaw_degrees']:
                angle=math.radians(yaw); direction=Vector([math.cos(angle),-.18,math.sin(angle)])
                d=C.to_3x3()@direction; pose=d.to_track_quat('-Z','Y').to_matrix().to_4x4();pose.translation=C.to_3x3()@eye
                world_cv = CI @ pose @ Matrix.Diagonal(Vector([1,-1,-1,1]))
                eyes.append(list(eye));matrices.append([list(r) for r in world_cv.inverted()])
                camera_receipts.append({'room_index':ri,'position_index':pi,'yaw':yaw,'floor_y':floor_y})
    if len(eyes)<8: raise ValueError('Too few collision-free scene cameras; no per-target fallback search')
    side=policy['side'];focal=.5*side/math.tan(math.radians(policy['fov_degrees'])/2)
    K=[[focal,0,(side-1)/2],[0,focal,(side-1)/2],[0,0,1]]
    blend=folder/'complete_scene.blend';bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    frozen={'candidate_plan_identity':doc['identity'],'scene':sid,'blend':binding(blend),
            'objects':[r for r in p['candidates'] if r['scene']==sid],
            'image_side':side,'cycles_samples':policy['cycles_samples'],'frame_count':len(eyes),
            'world_eye':eyes,'world_w2c':matrices,'intrinsics':[K]*len(eyes),'camera_receipts':camera_receipts,
            'capture':'fixed room-grid photographs with full original architecture/instances, no target camera search'}
    payload={'identity':digest(frozen),'plan':frozen};(folder/'plan.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('SCENE_CAPTURE_READY '+json.dumps({'scene':sid,'frames':len(eyes),'candidates':len(frozen['objects'])}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);main(args.scene)
