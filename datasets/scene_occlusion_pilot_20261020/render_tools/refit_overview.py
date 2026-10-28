"""Reframe the existing complete scene; change no interior renders."""
import json
import sys
from pathlib import Path
import bpy
import bmesh
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_hssd_blender import OUT,fit_overview

scene=bpy.context.scene
cam=bpy.data.objects['00_overview_cutaway']
fit=fit_overview(cam,1500,1100)
meta=json.loads((OUT/'render_manifest.json').read_text());meta['views'][0]['auto_fit']=fit
(OUT/'render_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'complete_scene.blend'))
for obj in list(scene.objects):
    if obj.type!='MESH' or not obj.name.startswith('Architecture__'):continue
    obj.data=obj.data.copy();bm=bmesh.new();bm.from_mesh(obj.data)
    faces=[f for f in bm.faces if min((obj.matrix_world@p.co).z for p in f.verts)>2.55]
    bmesh.ops.delete(bm,geom=faces,context='FACES');bm.to_mesh(obj.data);bm.free()
scene.camera=cam;scene.render.resolution_x=1500;scene.render.resolution_y=1100
scene.render.filepath=str(OUT/'00_overview_cutaway.png')
bpy.ops.render.render(write_still=True)
print('REFIT_COMPLETE '+json.dumps(fit),flush=True)
