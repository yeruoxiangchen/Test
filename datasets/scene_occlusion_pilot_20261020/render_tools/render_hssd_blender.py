"""Render original HSSD placement in Blender. Run with blender -b -P this_file."""
import json
import math
from pathlib import Path
import sys
import time

import bpy
import bmesh
from mathutils import Matrix, Quaternion, Vector

ROOT=Path(__file__).resolve().parents[1]
HSSD=ROOT/'HSSD-hab'
OUT=ROOT/'renders'/'hssd_102344049'
C=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
CI=C.inverted()

def log(**kwargs):print('HSSD_RENDER '+json.dumps(kwargs),flush=True)

def point(xyz):return C.to_3x3()@Vector(xyz)

def load_prototype(path):
    before=set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path),merge_vertices=False)
    objects=list(set(bpy.data.objects)-before)
    for obj in objects:
        for collection in list(obj.users_collection):collection.objects.unlink(obj)
    return objects

def instance(prototypes,name,transform,instance_id):
    empty=bpy.data.objects.new(name,None);bpy.context.scene.collection.objects.link(empty)
    empty.matrix_world=transform
    mapping={old:old.copy() for old in prototypes}
    for old,new in mapping.items():
        bpy.context.scene.collection.objects.link(new)
        new.name=name+'__'+old.name
        new.parent=mapping.get(old.parent,empty)
        new.matrix_parent_inverse=Matrix.Identity(4)
        new.matrix_basis=old.matrix_basis.copy()
        new.pass_index=instance_id
    return empty,list(mapping.values())

def camera(name,eye,target,lens=23,ortho=None):
    data=bpy.data.cameras.new(name)
    obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj)
    obj.location=point(eye);direction=point(target)-obj.location
    obj.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
    data.lens=lens;data.sensor_width=36;data.clip_start=.05;data.clip_end=200
    if ortho is not None:data.type='ORTHO';data.ortho_scale=ortho
    return obj

def fit_overview(cam,width,height):
    bpy.context.view_layer.update()
    inverse=cam.matrix_world.inverted()
    points=[inverse@(o.matrix_world@Vector(corner)) for o in bpy.context.scene.objects if o.type=='MESH' for corner in o.bound_box]
    lo=[min(p[i] for p in points) for i in range(2)]
    hi=[max(p[i] for p in points) for i in range(2)]
    cam.location+=cam.matrix_world.to_3x3()@Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,0))
    cam.data.ortho_scale=max(hi[0]-lo[0],(hi[1]-lo[1])*width/height)*1.08
    bpy.context.view_layer.update()
    return {'eye_blender':list(cam.location),'ortho_scale':cam.data.ortho_scale,'margin':1.08}

def main():
    started=time.monotonic()
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine='CYCLES';scene.cycles.device='CPU'
    scene.cycles.samples=24;scene.cycles.use_denoising=True
    scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=3;scene.cycles.glossy_bounces=3
    scene.cycles.transparent_max_bounces=8
    scene.render.threads_mode='FIXED';scene.render.threads=8
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.resolution_percentage=100
    scene.view_settings.view_transform='Filmic';scene.view_settings.look='Medium High Contrast'
    scene.view_settings.exposure=.3
    world=bpy.data.worlds.new('Preview daylight') if scene.world is None else scene.world
    scene.world=world;world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.78,.86,1,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.45
    original=json.loads((HSSD/'scenes/102344049.scene_instance.json').read_text())
    configs={p.name.removesuffix('.object_config.json'):p for p in (HSSD/'objects').rglob('*.object_config.json')}
    stage_path=OUT/'converted_glb/stages/102344049.glb'
    stage_proto=load_prototype(stage_path)
    stage_empty,stage_objs=instance(stage_proto,'Architecture',Matrix.Identity(4),1)
    cache={};receipts=[]
    for i,obj in enumerate(original['object_instances']):
        template=obj['template_name'];cfgpath=configs[template]
        cfg=json.loads(cfgpath.read_text());src=cfgpath.parent/cfg['render_asset']
        if template not in cache:cache[template]=load_prototype(OUT/'converted_glb'/src.relative_to(HSSD))
        q=Quaternion(obj.get('rotation',[1,0,0,0]));s=obj.get('non_uniform_scale',[1,1,1])
        transform=Matrix.Translation(Vector(obj.get('translation',[0,0,0])))@q.to_matrix().to_4x4()@Matrix.Diagonal(Vector([*s,1]))
        root,parts=instance(cache[template],f'Instance_{i:03d}_{template[:8]}',C@transform@CI,i+2)
        receipts.append({'index':i,'template':template,'source':str(src),'matrix_world_habitat':[list(r) for r in transform],
                         'matrix_world_blender':[list(r) for r in root.matrix_world],'mesh_parts':sum(o.type=='MESH' for o in parts)})
        if (i+1)%25==0:log(phase='assembly',instances=i+1,total=len(original['object_instances']))
    bpy.context.view_layer.update()
    regions=json.loads((HSSD/'semantics/scenes/102344049.semantic_config.json').read_text())['region_annotations']
    for i,region in enumerate(regions):
        lo,hi=region['min_bounds'],region['max_bounds']
        area=(hi[0]-lo[0])*(hi[2]-lo[2])
        light=bpy.data.lights.new('Ceiling fill '+region['name'],'AREA')
        light.energy=max(150,area*22);light.shape='DISK';light.size=min(3,max(.8,math.sqrt(area)*.5))
        obj=bpy.data.objects.new(light.name,light);scene.collection.objects.link(obj)
        obj.location=point([(lo[0]+hi[0])/2,2.6,(lo[2]+hi[2])/2])
    sun=bpy.data.lights.new('Exterior daylight','SUN');sun.energy=1.3;sun.angle=math.radians(15)
    obj=bpy.data.objects.new(sun.name,sun);scene.collection.objects.link(obj)
    obj.rotation_euler=(math.radians(20),math.radians(-35),math.radians(25))
    views=[
        {'name':'00_overview_cutaway','eye':[24,32,15],'target':[2,0,-7.5],'ortho':33,'size':[1500,1100],
         'description':'Complete scene overview; architecture faces entirely above 2.55 m removed for this view only; all 157 object instances retained.'},
        {'name':'01_living_room','eye':[-4.7,1.55,-.75],'target':[-7.7,.8,-3.2],'lens':21,'size':[1440,960],
         'description':'Interior living room, original full architecture and object placement.'},
        {'name':'02_dining_occlusion','eye':[-3.25,1.45,-4.3],'target':[-1.6,.75,-1.3],'lens':22,'size':[1440,960],
         'description':'Interior dining area: foreground seating and table occlude other surfaces.'},
        {'name':'03_kitchen_occlusion','eye':[.6,1.55,-.5],'target':[3.4,.85,-3.5],'lens':20,'size':[1440,960],
         'description':'Interior kitchen/dining area, original full scene.'},
    ]
    cameras={v['name']:camera(v['name'],v['eye'],v['target'],v.get('lens',23),v.get('ortho')) for v in views}
    views[0]['auto_fit']=fit_overview(cameras['00_overview_cutaway'],*views[0]['size'])
    scene.camera=cameras['01_living_room']
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'complete_scene.blend'))
    metadata={'source_scene':'102344049','instances':len(receipts),'unique_assets':len(cache),
              'basis_change':'(x,y,z) Habitat -> (x,-z,y) Blender; all instance rotations/scales preserved',
              'renderer':'Blender 3.0.1 Cycles CPU','lighting':'Preview daylight plus ceiling area fill lights; not dataset-captured illumination',
              'views':views,'instances_receipt':receipts}
    (OUT/'render_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for v in views:
        backups=[]
        if v['name']=='00_overview_cutaway':
            for obj in stage_objs:
                if obj.type!='MESH':continue
                original_mesh=obj.data;obj.data=original_mesh.copy();backups.append((obj,original_mesh))
                bm=bmesh.new();bm.from_mesh(obj.data)
                faces=[f for f in bm.faces if min((obj.matrix_world@p.co).z for p in f.verts)>2.55]
                bmesh.ops.delete(bm,geom=faces,context='FACES');bm.to_mesh(obj.data);bm.free()
        scene.camera=cameras[v['name']];scene.render.resolution_x,scene.render.resolution_y=v['size']
        scene.render.filepath=str(OUT/(v['name']+'.png'))
        log(phase='render',view=v['name'],seconds=round(time.monotonic()-started,1))
        bpy.ops.render.render(write_still=True)
        for obj,mesh in backups:
            temporary=obj.data;obj.data=mesh;bpy.data.meshes.remove(temporary)
        log(phase='render_complete',view=v['name'],seconds=round(time.monotonic()-started,1))
    log(phase='complete',seconds=round(time.monotonic()-started,1))

if __name__=='__main__':main()
